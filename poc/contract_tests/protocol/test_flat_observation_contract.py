from __future__ import annotations
import json
import unittest
from pathlib import Path
from agent.application.qwen_visual_decision import QWEN_VISUAL_DECISION_PROTOCOL_VERSION
from agent.domain.action_catalog import CANONICAL_ACTION_KINDS
from agent.domain.canonical_action_protocol import (
    CanonicalActionProtocolError, MODEL_STEP_DECISION_FIELDS, MODEL_STEP_DIRECT_POINT_ACTIONS,
    normalize_model_step_decision,
)
from agent.infrastructure.generic_scene_observer import (
    SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION, _parse_single_step_observation_envelope,
    _single_step_response_format,
)
from agent.domain.vision_model import VisionAgentError


def decision(**values):
    result = dict.fromkeys(MODEL_STEP_DECISION_FIELDS)
    result.update(status='action', confidence=1.0, reason='当前画面依据')
    result.update(values)
    return result


WIRE_FOR_CANONICAL = {
    'tap_semantic': 'tap',
    'dismiss_overlay': 'dismiss',
    'swipe_element': 'swipe',
    'input_verified_text': 'input',
    'clear_verified_text': 'clear_input',
}


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
                self.assertEqual([WIRE_FOR_CANONICAL.get(kind, kind), None], properties['action']['enum'])
                self.assertNotIn('target', properties)
                for field in ('point', 'direction', 'start', 'end'):
                    self.assertIn('null', properties[field]['type'])
                self.assertEqual({'type': 'string'}, schema['properties']['scene']['properties']['overlays']['items'])

    def test_all_canonical_actions_keep_the_existing_parser_authority(self):
        for kind in sorted(CANONICAL_ACTION_KINDS):
            parts = {'action': WIRE_FOR_CANONICAL.get(kind, kind)}
            if kind == "input_verified_text":
                parts["text"] = "原文？你好"
            if kind == "launch_app":
                parts["app"] = "系统设置"
            if kind in MODEL_STEP_DIRECT_POINT_ACTIONS:
                parts.update(point=[400, 250])
            elif kind == 'scroll':
                parts['direction'] = 'up'
            elif kind == 'swipe_element':
                parts.update(start=[500, 500], end=[100, 500])
            elif kind == 'drag':
                parts.update(start=[500, 500], end=[100, 500])
            with self.subTest(action=kind):
                normalized = normalize_model_step_decision(decision(**parts))
                self.assertEqual(kind, normalized['action'])

    def test_finish_null_fields_do_not_create_an_action(self):
        result = normalize_model_step_decision(decision(status='finish'))
        self.assertEqual('finish', result['status'])
        self.assertIsNone(result['target'])
        self.assertIsNone(result['tap_point'])

    def test_wait_for_change_accepts_a_finite_wait_duration_only_for_wait(self):
        normalized = normalize_model_step_decision(decision(
            action='wait_for_change', wait_seconds=300, reason='等待开奖后重新观察'))
        self.assertEqual(300, normalized['wait_seconds'])
        with self.assertRaises(CanonicalActionProtocolError):
            normalize_model_step_decision(decision(action='home', wait_seconds=300))
        with self.assertRaises(CanonicalActionProtocolError):
            normalize_model_step_decision(decision(action='wait_for_change', wait_seconds=-1))

    def test_scroll_rejects_legacy_free_trajectory_fields(self):
        with self.assertRaises(CanonicalActionProtocolError):
            normalize_model_step_decision(decision(
                action='scroll', direction='up', start=[500, 800], end=[500, 400]))

    def test_observation_parser_does_not_repair_legacy_scroll_trajectory(self):
        payload = {
            'protocol_version': SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
            'coordinate_space': {'kind': 'axis_grid', 'width': 1000, 'height': 1000},
            'scene': {'elements': [], 'summary': '可滚动页面'},
            'input_structure': None,
            'decision': decision(action='scroll', direction='up', start=[500, 800], end=[500, 400]),
        }
        with self.assertRaisesRegex(VisionAgentError, '协议'):
            _parse_single_step_observation_envelope(
                json.dumps(payload, ensure_ascii=False),
                input_structure_required=False,
                request_image_size=(720, 1280),
            )

    def test_finish_with_nonnull_action_or_target_is_rejected(self):
        for extra in ({'action': 'home'}, {'target': TARGET}, {'tap_point': [500, 500]}):
            with self.subTest(extra=extra), self.assertRaises(CanonicalActionProtocolError):
                normalize_model_step_decision(decision(status='finish', **extra))

    def test_wrong_action_field_combinations_are_not_repaired(self):
        for parts in (
            {'action': 'home', 'point': [500, 500]},
            {'action': 'tap', 'target': TARGET, 'point': [500, 500]},
            {'action': 'tap', 'point': [500, 500], 'start': [0, 0]},
            {'action': 'scroll', 'direction': 'up', 'target': TARGET},
        ):
            with self.subTest(parts=parts), self.assertRaises(CanonicalActionProtocolError):
                normalize_model_step_decision(decision(**parts))

    def test_direct_point_parses_without_scene_geometry_on_two_canvases(self):
        for height, y in ((1280, 340), (960, 300)):
            payload = {'protocol_version': SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
                'coordinate_space': {'kind': 'axis_grid', 'width': 1000, 'height': 1000},
                'scene': {'elements': [], 'summary': '当前目标可见'}, 'input_structure': None,
                'decision': decision(action='tap', point=[500, y])}
            for wrapped in (False, True):
                with self.subTest(height=height, wrapped=wrapped):
                    parsed = _parse_single_step_observation_envelope(
                        json.dumps([payload] if wrapped else payload, ensure_ascii=False),
                        input_structure_required=False, request_image_size=(720, height))
                    self.assertEqual([500, y], parsed['decision']['point'])
                    self.assertEqual([], parsed['scene']['elements'])
            payload['scene']['elements'] = [{'element_id': 'entry', 'bounds': [10, 10, 900, 900]}]
            extra = _parse_single_step_observation_envelope(json.dumps(payload),
                input_structure_required=False, request_image_size=(720, height))
            self.assertEqual([{'element_id': 'entry', 'bounds': [10, 10, 900, 900]}], extra['scene']['elements'])
            self.assertEqual(parsed['decision'], extra['decision'])



if __name__ == '__main__':
    unittest.main()
