"""Bounded saved-image probe: production observation/binding, no device execution."""
import argparse
from copy import deepcopy
from dataclasses import replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

POC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POC))
from PIL import Image
from agent.application.universal_agent_orchestrator import ObservationBridge
from agent.application.qwen_visual_decision import QwenVisualDecisionObserver
from agent.domain.universal_action_controller import UniversalActionController
from agent.infrastructure.atomic_files import atomic_replace_bytes, json_bytes
from agent.infrastructure.dashscope_vision_provider import DashScopeVisionProvider
from agent.infrastructure.generic_scene_observer import SingleStepGenericSceneObserver
from agent.infrastructure.generic_scene_observer import (
    INPUT_STRUCTURE_AUDIT_VERSION, SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
)
from agent.infrastructure.trusted_observation_frames import (
    build_trusted_observation, validate_trusted_observation_against_frames,
)
from test_canonical_action_protocol import ime_profile
from contract_tests.input.test_simple_input_completion_contract import saved_response, input_graph
from test_universal_agent_orchestrator import _clear_graph


class ProbeProvider(DashScopeVisionProvider):
    def __init__(self, directory, case, online, fixture):
        super().__init__(timeout=60, max_attempts=1)
        self.directory, self.case, self.online, self.fixture = directory, case, online, fixture
        self.calls = 0

    def _chat(self, messages, max_tokens, **kwargs):
        self.calls += 1
        if self.calls != 1:
            raise RuntimeError('Probe forbids repeat requests per case')
        recorded = deepcopy(messages)
        for message in recorded:
            content = message.get('content')
            for part in content if isinstance(content, list) else []:
                if part.get('type') == 'image_url':
                    value = part['image_url']['url']
                    part['image_url']['url'] = 'sha256:' + hashlib.sha256(value.encode()).hexdigest()
        atomic_replace_bytes(self.directory / f'{self.case}_request.json', json_bytes({
            'messages_with_image_hashes': recorded, 'response_format': kwargs.get('response_format'),
            'model': self.model, 'base_url': self.base_url, 'max_attempts': 1,
        }))
        if not self.online:
            if isinstance(self.fixture, int):
                payload = saved_response(self.fixture)
            else:
                evidence = POC / 'output/explicit_focus_probe/20260904_173525' / self.fixture
                record = json.loads(evidence.read_text(encoding='utf-8'))
                payload = json.loads(record['redacted_raw_response'])
                payload['protocol_version'] = SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION
                payload['input_structure']['protocol_version'] = INPUT_STRUCTURE_AUDIT_VERSION
            return json.dumps(payload, ensure_ascii=False)
        if self.base_url != 'https://dashscope.aliyuncs.com/compatible-mode/v1':
            raise RuntimeError('Probe only permits the approved Aliyun destination')
        kwargs['max_attempts'] = 1
        return super()._chat(messages, max_tokens, **kwargs)


def cases():
    clear = _clear_graph()
    typing = input_graph()
    focus = replace(clear, raw_user_goal='让当前消息编辑栏获得输入焦点，但不输入文字',
        goal=replace(clear.goal, objective='让当前消息编辑栏获得输入焦点'),
        subgoals=(replace(clear.subgoals[0], objective='让当前消息编辑栏获得输入焦点',
            input_operation='focus', completion_conditions=('当前输入框已聚焦',)),))
    return [
        ('empty_clear_finish', 'generic_supervised_20260904_194443_a14b08e9', 1, clear, 0,
            'finish', None, 'any'),
        ('visible_caret_clear', 'generic_supervised_20260904_170753_06971f2d', 4, clear,
            'visible_caret_clear_75f44c76b1ab45928d0cd57da1fe3b30_model_response.json',
            'action', 'clear_verified_text', 'true'),
        ('ambiguous_focus_tap', 'generic_supervised_20260904_200708_9854a016', 2, focus,
            'empty_field_without_focus_evidence_b97665266a514ef6aeba5413a5b402db_model_response.json',
            'action', 'tap_semantic', 'not_true'),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--online', action='store_true')
    args = parser.parse_args()
    directory = POC / 'output/simple_input_model_probe' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    directory.mkdir(parents=True)
    results = []
    for name, source, step, graph, fixture, expected_status, expected_action, expected_focus in cases():
        graph.validate()
        provider = ProbeProvider(directory, name, args.online, fixture)
        observer = SingleStepGenericSceneObserver(provider)
        goal = ObservationBridge().goal_draft(graph)
        allowed = frozenset({'tap_semantic', 'clear_verified_text', 'input_verified_text', 'back', 'home', 'scroll'})
        if expected_status == 'finish':
            allowed = allowed - {'input_verified_text'}
        result = {'case': name, 'source': source, 'expected_status': expected_status,
            'expected_action': expected_action, 'passed': False}
        try:
            frames = []
            for i in range(1, 5):
                with Image.open(POC / 'output/web' / source / f'before_step_{step}_frame_attempt_1_{i}.jpg') as frame:
                    frames.append(frame.convert('RGB'))
            scene, decision = observer.observe_with_decision(frames=frames,
                goal_context={'entities': goal.entities}, device_id=graph.device_id,
                available_action_kinds=allowed, response_evidence_dir=directory,
                response_evidence_prefix=name)
            trusted = build_trusted_observation(frames=frames, device_id=graph.device_id, scene=scene)
            binding = QwenVisualDecisionObserver(provider,
                trusted_observation_frame_validator=validate_trusted_observation_against_frames).decide(
                    frames=frames, task_context=graph.to_qwen_context(), trusted_observation=trusted,
                    model_decision=decision, available_action_kinds=allowed,
                    text_transport_profile=ime_profile(device_id=graph.device_id))
            inputs = [item for item in scene.elements if item.role == 'input']
            action = binding.proposal.action
            if action:
                UniversalActionController().resolve_one(action, scene)  # Resolve only; never execute.
            unique_empty = len(inputs) == 1 and inputs[0].states.get('value') == ''
            focus = inputs[0].states.get('focused') if len(inputs) == 1 else None
            focus_ok = expected_focus == 'any' or (focus is True if expected_focus == 'true' else focus is not True)
            input_value_ok = (len(inputs) == 1 and
                (not unique_empty if expected_action == 'clear_verified_text' else unique_empty))
            result.update(status=binding.proposal.status, action=action.action if action else None,
                focused=focus, input_count=len(inputs), input_empty=unique_empty,
                passed=(binding.proposal.status == expected_status and
                    (action.action if action else None) == expected_action and input_value_ok and
                    focus_ok))
        except Exception as exc:
            result.update(error_type=type(exc).__name__, error=str(exc))
        result.update(response_evidence=observer.last_response_evidence_path,
            model_calls=provider.calls if args.online else 0,
            network_attempts=provider.last_network_attempts if args.online else 0,
            usage=provider.last_usage)
        results.append(result)
        report = {'mode': 'online_saved_images' if args.online else 'offline_replay_preflight',
            'hardware_actions': 0, 'project_api_used': False, 'planned_cases': 3,
            'passed': sum(r['passed'] for r in results), 'results': results}
        atomic_replace_bytes(directory / 'report.json', json_bytes(report))
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if not result['passed']:
            break
    print(str(directory / 'report.json'), flush=True)
    return 0 if len(results) == 3 and all(r['passed'] for r in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
