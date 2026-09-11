from __future__ import annotations
import json
import unittest
from pathlib import Path
from agent.application.qwen_visual_decision import QWEN_VISUAL_DECISION_PROTOCOL_VERSION
from agent.domain.canonical_action_kinds import CANONICAL_ACTION_KINDS
from agent.domain.canonical_action_protocol import (
    CanonicalActionProtocolError, MODEL_STEP_DECISION_FIELDS, MODEL_STEP_DIRECT_POINT_ACTIONS,
    normalize_model_step_decision,
)
from agent.domain.vision_model import VisionAgentError
from agent.infrastructure.generic_scene_observer import (
    SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION, _parse_single_step_observation_envelope,
    _single_step_observation_prompt, _single_step_response_format,
)


def decision(**values):
    result = dict.fromkeys(MODEL_STEP_DECISION_FIELDS)
    result.update(status='action', evidence_refs=[], confidence=1.0, reason='当前画面依据')
    result.update(values)
    return result


TARGET = {'element_id': 'entry', 'role': 'list_item', 'meaning': 'open_entry',
    'label': '目标', 'evidence': ['当前条目可见']}


class FlatObservationContractTests(unittest.TestCase):
    def test_frontend_contract_cannot_drift_from_backend_version(self):
        root = Path(__file__).resolve().parents[2]
        adapter = (root / 'static/protocol_adapter.js').read_text(encoding='utf-8')
        fixture = json.loads((root / 'frontend_contract_fixtures/qwen_whole_task_decision.json').read_text(encoding='utf-8'))
        self.assertIn(f'const formalQwenProtocol = "{QWEN_VISUAL_DECISION_PROTOCOL_VERSION}";', adapter)
        self.assertEqual(QWEN_VISUAL_DECISION_PROTOCOL_VERSION, fixture['decision']['protocol_version'])
        self.assertNotIn('target_region', fixture['decision'])
        self.assertEqual([0.77, 0.275], fixture['decision']['next_action']['params']['tap_point'])
        self.assertNotIn('trusted_observation', fixture['decision'])

    def test_every_action_scope_has_one_fixed_nullable_schema(self):
        for kind in sorted(CANONICAL_ACTION_KINDS):
            with self.subTest(action=kind):
                schema = _single_step_response_format({}, input_structure_required=False,
                    request_height=1280, available_action_kinds=(kind,))['json_schema']['schema']
                wire = json.dumps(schema)
                for retired in ('oneOf', 'anyOf', 'allOf'):
                    self.assertNotIn(retired, wire)
                properties = schema['properties']['decision']['properties']
                self.assertEqual(MODEL_STEP_DECISION_FIELDS, set(properties))
                self.assertEqual(set(properties), set(schema['properties']['decision']['required']))
                self.assertEqual([kind, None], properties['action']['enum'])
                self.assertEqual(['object', 'null'] if kind in MODEL_STEP_DIRECT_POINT_ACTIONS else 'null',
                    properties['target']['type'])
                self.assertEqual({'type': 'string'}, schema['properties']['scene']['properties']['overlays']['items'])

    def test_all_canonical_actions_keep_the_existing_parser_authority(self):
        for kind in sorted(CANONICAL_ACTION_KINDS):
            parts = {'action': kind}
            if kind == "input_verified_text":
                parts["text"] = "原文？你好"
            if kind == "launch_app":
                parts["app"] = "系统设置"
            if kind in MODEL_STEP_DIRECT_POINT_ACTIONS:
                parts.update(target=TARGET, tap_point=[400, 250])
            elif kind in {'input_verified_text', 'clear_verified_text'}:
                parts['element_id'] = 'field'
            elif kind == 'scroll':
                parts['direction'] = 'up'
            elif kind == 'swipe_element':
                parts.update(element_id='item', start=[500, 500], end=[100, 500])
            elif kind == 'drag':
                parts.update(source_element_id='item', destination_element_id='slot')
            with self.subTest(action=kind):
                normalized = normalize_model_step_decision(decision(**parts))
                self.assertEqual(kind, normalized['action'])

    def test_finish_null_fields_do_not_create_an_action(self):
        result = normalize_model_step_decision(decision(status='finish', evidence_refs=['scene.summary']))
        self.assertEqual('finish', result['status'])
        self.assertIsNone(result['target'])
        self.assertIsNone(result['tap_point'])

    def test_finish_with_nonnull_action_or_target_is_rejected(self):
        for extra in ({'action': 'home'}, {'target': TARGET}, {'tap_point': [500, 500]}):
            with self.subTest(extra=extra), self.assertRaises(CanonicalActionProtocolError):
                normalize_model_step_decision(decision(status='finish', evidence_refs=['scene.summary'], **extra))

    def test_wrong_action_field_combinations_are_not_repaired(self):
        for parts in (
            {'action': 'home', 'tap_point': [500, 500]},
            {'action': 'tap_semantic', 'target': TARGET, 'tap_point': [500, 500], 'element_id': 'entry'},
            {'action': 'tap_semantic', 'target': {**TARGET, 'bounds': [0, 0, 1000, 1000]}, 'tap_point': [500, 500]},
            {'action': 'scroll', 'direction': 'up', 'target': TARGET},
        ):
            with self.subTest(parts=parts), self.assertRaises(CanonicalActionProtocolError):
                normalize_model_step_decision(decision(**parts))

    def test_direct_point_parses_without_scene_geometry_on_two_canvases(self):
        for height, y in ((1280, 340), (960, 300)):
            payload = {'protocol_version': SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
                'coordinate_space': {'kind': 'axis_grid', 'width': 1000, 'height': 1000},
                'scene': {'elements': [], 'summary': '当前目标可见'}, 'input_structure': None,
                'decision': decision(action='tap_semantic', target=TARGET, tap_point=[500, y])}
            for wrapped in (False, True):
                with self.subTest(height=height, wrapped=wrapped):
                    parsed = _parse_single_step_observation_envelope(
                        json.dumps([payload] if wrapped else payload, ensure_ascii=False),
                        input_structure_required=False, request_image_size=(720, height))
                    self.assertEqual([500, y], parsed['decision']['tap_point'])
                    self.assertEqual([], parsed['scene']['elements'])
            payload['scene']['elements'] = [{'element_id': 'entry', 'bounds': [10, 10, 900, 900]}]
            extra = _parse_single_step_observation_envelope(json.dumps(payload),
                input_structure_required=False, request_image_size=(720, height))
            self.assertEqual([], extra['scene']['elements'])
            self.assertEqual(parsed['decision'], extra['decision'])

    def test_full_prompt_scopes_old_element_contract_to_nonpoint_actions(self):
        prompt = _single_step_observation_prompt({}, include_input_structure=False,
            image_count=1, request_image_size=(720, 1280), available_action_kinds=('tap_semantic', 'home'))
        self.assertIn('SCENE CONTRACT的元素规则只适用于非点按动作或finish', prompt)
        self.assertIn('跳过下面SCENE CONTRACT全部元素枚举、bounds和states要求', prompt)
        self.assertIn('DECISION_OBJECT使用固定字段', prompt)
        self.assertIn('decision.element_id必须为null', prompt)
        self.assertNotIn('不属于当前动作变体的null字段无需输出', prompt)
        self.assertNotIn('必须在当前scene中发布并选中唯一可见的系统', prompt)


if __name__ == '__main__':
    unittest.main()
