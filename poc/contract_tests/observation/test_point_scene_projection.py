"""Raw-response/observer/binder/controller composition. No model or device I/O."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import unittest
from PIL import Image
from agent.application.qwen_visual_decision import QwenVisualDecisionObserver
from agent.domain.canonical_action_protocol import MODEL_STEP_DECISION_FIELDS
from agent.domain.qwen_task_context import QwenTaskContext
from agent.domain.universal_action_controller import UniversalActionController
from agent.domain.vision_model import VisionAgentError
from agent.infrastructure.generic_scene_observer import (
    SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION, SingleStepGenericSceneObserver,
    _parse_single_step_observation_envelope,
)
from agent.infrastructure.trusted_observation_frames import (
    build_trusted_observation, validate_trusted_observation_against_frames,
)
from test_canonical_action_protocol import ime_profile
from test_support.generic_action_adapter import (
    RawSceneProvider,
)
from test_support.generic_scene_observer import (
    audited_application_input,
    input_audit_payload,
    scene_payload,
)
from test_qwen_visual_decision import patterned_frames, StatusOnlyProvider

def task_context(raw_goal='执行当前任务', exact_input_text=None):
    return QwenTaskContext(task_id='raw-task', device_id='device-1', revision=1,
        raw_goal=raw_goal, exact_input_text=exact_input_text)


def decision(kind='tap_semantic', **parts):
    value = dict.fromkeys(MODEL_STEP_DECISION_FIELDS)
    value.update(status='action', action=kind, evidence_refs=[], confidence=1, reason='当前截图证据')
    if kind in {'tap_semantic', 'dismiss_overlay', 'long_press', 'double_tap'}:
        value.update(target={'element_id': 'e1', 'role': 'button', 'meaning': 'open_entry',
            'label': '入口', 'evidence': ['入口可见']}, tap_point=[420, 220])
    value.update(parts)
    return value


def wire(choice=None, audit=None):
    scene = scene_payload()
    scene['elements'][0]['bounds'] = [100, 280, 400, 350]
    return {'protocol_version': SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
        'coordinate_space': {'kind': 'axis_grid', 'width': 1000, 'height': 1000},
        'scene': scene, 'input_structure': audit, 'decision': choice or decision()}


def observe(payload, graph=None, *, allowed=None):
    graph = graph or task_context()
    goal_context = {'objective': graph.raw_goal, 'entities': {'history': list(graph.history), 'exact_input_text': graph.exact_input_text}}
    height = payload['coordinate_space']['height']
    frames = [frame.resize((720, 1280), Image.Resampling.NEAREST)
        for frame in patterned_frames()]
    provider = RawSceneProvider([deepcopy(payload)])  # Does not repair/upgrade model fields.
    observer = SingleStepGenericSceneObserver(provider)
    allowed = frozenset(allowed or {'tap_semantic', 'dismiss_overlay', 'long_press', 'double_tap',
        'press_enter', 'clear_verified_text', 'input_verified_text', 'scroll', 'home', 'swipe_element'})
    scene, choice = observer.observe_with_decision(frames=frames,
        goal_context=goal_context, device_id=graph.device_id, available_action_kinds=allowed)
    trusted = build_trusted_observation(frames=frames, device_id=graph.device_id, scene=scene)
    binder = QwenVisualDecisionObserver(StatusOnlyProvider(),
        trusted_observation_frame_validator=validate_trusted_observation_against_frames)
    result = binder.decide(frames=frames, task_context=graph,
        trusted_observation=trusted, model_decision=choice, available_action_kinds=allowed,
        text_transport_profile=ime_profile(device_id=graph.device_id))
    resolved = (UniversalActionController().resolve_one(result.proposal.action, scene)
        if result.proposal.status == 'action' else None)
    assert provider.calls == 1
    return scene, result, resolved, observer


class PointSceneProjectionTests(unittest.TestCase):
    def test_saved_response_replays_through_observer_binder_controller(self):
        archived = json.loads((Path(__file__).resolve().parents[2] / 'test_fixtures/point_scene_saved_response.json').read_text(encoding='utf-8'))
        self.assertEqual('2026-09-04-single-step-flat-target-point-v13', archived['protocol_version'])
        current = deepcopy(archived)
        # Explicit offline version rebind only; production does not accept historical versions.
        current['protocol_version'] = SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION
        scene, result, resolved, observer = observe(current)
        self.assertEqual((), scene.elements)
        self.assertEqual('e1', result.proposal.action.params['element_id'])
        self.assertEqual((.45, .648), resolved.normalized_point)
        self.assertEqual(current, json.loads(observer.last_raw_response))
        with self.assertRaisesRegex(VisionAgentError, '协议版本'):
            _parse_single_step_observation_envelope(json.dumps(archived),
                input_structure_required=False, request_image_size=(720, 1280))

    def test_unconsumed_scene_variations_cannot_change_any_noninput_point_action(self):
        for kind in ('tap_semantic', 'dismiss_overlay', 'double_tap', 'long_press'):
            baseline = wire(decision(kind, evidence_refs=['element:e1']))
            baseline['scene']['elements'] = []
            expected = observe(baseline)[2]
            for extra in (None, 'irrelevant', {}, [42],
                    [{'element_id': 'e1', 'role': 'input', 'bounds': [-1, 0, 9000, 9000],
                      'states': {'focused': False}, 'action': 'home'}, {'element_id': 'e1'}]):
                with self.subTest(kind=kind, extra=extra):
                    payload = deepcopy(baseline)
                    payload['scene']['elements'] = extra
                    scene, _, actual, observer = observe(payload)
                    self.assertEqual((), scene.elements)
                    self.assertEqual(expected, actual)
                    self.assertEqual(extra, json.loads(observer.last_raw_response)['scene']['elements'])

    def test_missing_target_point_and_second_target_still_reject(self):
        for extra in ({'target': None}, {'tap_point': None}, {'tap_point': [1001, 220]},
                {'element_id': 'second'}, {'target': {**decision()['target'], 'bounds': [1, 2, 3, 4]}}):
            with self.subTest(extra=extra), self.assertRaises(VisionAgentError):
                observe(wire(decision(**extra)))

    def test_issued_action_scope_and_action_like_scene_fields_still_reject(self):
        with self.assertRaisesRegex(VisionAgentError, '动作集合'):
            observe(wire(), allowed={'home'})
        bad = wire()
        bad['scene']['action'] = 'home'
        with self.assertRaisesRegex(VisionAgentError, 'scene包含动作'):
            observe(bad)

    def test_input_focus_uses_only_audit_and_keeps_direct_point(self):
        item = audited_application_input(bounds=[80, 320, 850, 410], text='旧草稿', focused=False,
            visible_editable_cues=['完整输入边框'])
        item['element_id'] = 'field'
        target = {'element_id': 'field', 'role': 'input', 'meaning': 'application_text_input',
            'label': '输入框', 'evidence': ['完整输入边框']}
        payload = wire(decision(target=target, tap_point=[610, 370], evidence_refs=['element:field']),
            input_audit_payload(application_inputs=[item]))
        payload['scene']['elements'] = [{'element_id': 'field', 'role': 'input',
            'bounds': [0, 0, 0, 0], 'states': {'value': '冲突文字', 'focused': True}}]
        scene, result, resolved, _ = observe(payload, task_context('清空当前输入框'))
        bound_id = result.proposal.action.params['element_id']
        self.assertEqual('旧草稿', scene.get_element(bound_id).states['value'])
        self.assertIsNot(scene.get_element(bound_id).states.get('focused'), True)
        self.assertEqual(1, len([item for item in scene.elements if item.role == 'input']))
        self.assertEqual((.61, .37), resolved.normalized_point)
        broken = deepcopy(payload)
        broken['input_structure']['application_inputs'][0]['bounds'] = [0, 0, 0, 0]
        with self.assertRaises(VisionAgentError):
            observe(broken, task_context('清空当前输入框'))

    def test_press_enter_keeps_same_frame_typed_projection(self):
        graph = task_context('换行', exact_input_text='draft\n')
        item = audited_application_input(bounds=[80, 170, 850, 260], text='draft',
            visible_editable_cues=['插入光标'])
        item['element_id'] = 'field'
        item['multiline'] = True
        audit = input_audit_payload(application_inputs=[item])
        payload = wire(decision('press_enter'), audit)
        scene, _, resolved, _ = observe(payload, graph)
        self.assertEqual('draft\n', resolved.expected_input_value)
        self.assertEqual('local_audited_input_1', resolved.target_element_id)
        self.assertIsNone(resolved.normalized_point)
        self.assertEqual('adb_keyboard', resolved.text_transport)
        self.assertEqual('\n', resolved.input_fragment)
        bad = deepcopy(payload)
        bad['input_structure']['application_inputs'][0]['focused'] = None
        with self.assertRaises(VisionAgentError):
            observe(bad, graph)

    def test_nonpoint_actions_still_bind_referenced_scene_elements(self):
        for choice in (decision('scroll', direction='up', element_id='e1'),):
            payload = wire(choice)
            scene, _, _, _ = observe(payload)
            self.assertEqual('e1', scene.elements[0].element_id)
            bad = deepcopy(payload)
            bad['scene']['elements'].append(deepcopy(bad['scene']['elements'][0]))
            with self.assertRaisesRegex(VisionAgentError, '不唯一'):
                observe(bad)


    def test_finish_does_not_make_optional_scene_ids_authoritative(self):
        payload = wire(decision(None, status='finish', evidence_refs=['element:missing']))
        payload['scene']['elements'][0]['bounds'] = [0, 0, 0, 0]
        self.assertEqual('finish', observe(payload)[1].proposal.status)


if __name__ == '__main__':
    unittest.main()
