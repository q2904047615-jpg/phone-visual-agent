"""One-shot saved-image focus validation; no project API or hardware executor."""
from dataclasses import replace
from datetime import datetime
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
from agent.infrastructure.trusted_observation_frames import (
    build_trusted_observation, validate_trusted_observation_against_frames,
)
from test_canonical_action_protocol import ime_profile
from test_universal_agent_orchestrator import _clear_graph


def main():
    directory = POC / 'output/explicit_focus_probe' / datetime.now().strftime('%Y%m%d_%H%M%S')
    directory.mkdir(parents=True)
    cases = [
        ('visible_caret_clear', 'generic_supervised_20260904_170753_06971f2d', 4,
         '清空当前输入框的全部内容', 'clear_verified_text', True, 'clear_verified_text'),
        ('visible_caret_paraphrase', 'generic_supervised_20260904_170753_06971f2d', 4,
         '把编辑栏里尚未发送的草稿删干净，保留聊天记录', 'clear_verified_text', True, 'clear_verified_text'),
        ('empty_field_without_focus_evidence', 'generic_supervised_20260902_210410_454a6a7f', 2,
         '让当前消息输入框获得输入焦点，但不输入文字', 'focus', False, 'tap_semantic'),
    ]
    results = []
    for name, source, step, objective, operation, expected_focus, expected_action in cases:
        provider = DashScopeVisionProvider(timeout=60, max_attempts=1)
        observer = SingleStepGenericSceneObserver(provider)
        base = _clear_graph()
        graph = replace(base, raw_user_goal=objective, goal=replace(base.goal, objective=objective),
            subgoals=(replace(base.subgoals[0], objective=objective, input_operation=operation,
                completion_conditions=('输入框已聚焦' if operation == 'focus' else '当前输入框正文为空',)),))
        goal = ObservationBridge().goal_draft(graph)
        frames = [Image.open(POC / 'output/web' / source / f'before_step_{step}_frame_attempt_1_{i}.jpg').convert('RGB')
            for i in range(1,5)]
        allowed = frozenset({'tap_semantic', 'clear_verified_text', 'back', 'home', 'scroll'})
        result = {'case': name, 'source': source, 'expected_focus': expected_focus,
            'expected_action': expected_action, 'physical_actions': 0, 'passed': False}
        try:
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
            inputs = [item for item in scene.elements if item.meaning == 'application_text_input']
            action = binding.proposal.action
            if action:
                UniversalActionController().resolve_one(action, scene)
            focus = inputs[0].states.get('focused') if len(inputs) == 1 else None
            result.update(focus=focus, action=action.action if action else None,
                scene=scene.to_dict(), decision=decision,
                passed=(len(inputs) == 1 and (focus is True if expected_focus else focus is not True) and action is not None
                    and action.action == expected_action))
        except Exception as exc:
            result.update(error_type=type(exc).__name__, error=str(exc))
        result['response_evidence'] = observer.last_response_evidence_path
        results.append(result)
        print(json.dumps({key: value for key, value in result.items() if key not in ('scene','decision')}, ensure_ascii=False), flush=True)
        atomic_replace_bytes(directory / 'report.json', json_bytes({
            'mode': 'online_model_saved_images_only', 'hardware_actions': 0, 'project_api_used': False,
            'results': results, 'passed': sum(item['passed'] for item in results), 'planned_cases': len(cases)}))
        if not result['passed']:
            break  # No sampling or prompt changes to make a failed probe pass.
    print(str(directory / 'report.json'), flush=True)
    return 0 if len(results) == len(cases) and all(item['passed'] for item in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
