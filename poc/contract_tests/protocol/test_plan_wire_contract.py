"""Model wire fields for plan proposals and per-condition finish reports."""

from __future__ import annotations

import json
import unittest

from agent.domain.canonical_action_protocol import MODEL_STEP_DECISION_FIELDS, normalize_model_step_decision
from agent.infrastructure.generic_scene_observer import (
    SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION, TASK_PLAN_RULE, _parse_single_step_observation_envelope,
    _single_step_observation_prompt, _single_step_response_format,
)


def wire_decision(**values):
    result = dict.fromkeys(MODEL_STEP_DECISION_FIELDS)
    result.update(status='finish', confidence=1.0, reason='当前画面满足任务')
    result.update(values)
    return result


def parse(decision):
    payload = {'protocol_version': SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
        'coordinate_space': {'kind': 'axis_grid', 'width': 1000, 'height': 1000},
        'scene': {'elements': [], 'summary': '蓝牙已开启'}, 'input_structure': None, 'decision': decision}
    return _parse_single_step_observation_envelope(json.dumps(payload, ensure_ascii=False),
        input_structure_required=False, request_image_size=(720, 1280))['decision']


class PlanWireContractTests(unittest.TestCase):
    def test_schema_exposes_nullable_plan_fields_on_every_branch(self):
        schema = _single_step_response_format({}, input_structure_required=False, request_height=1280,
            available_action_kinds=('tap_semantic', 'home'))['json_schema']['schema']
        properties = schema['properties']['decision']['properties']
        self.assertIn('null', properties['success_conditions']['type'])
        self.assertIn('null', properties['condition_results']['type'])
        item = properties['condition_results']['items']
        self.assertEqual(['met', 'not_met', 'unknown'], item['properties']['status']['enum'])
        self.assertIn('success_conditions', schema['properties']['decision']['required'])

    def test_parser_preserves_proposals_and_condition_reports(self):
        parsed = parse(wire_decision(success_conditions=[' 蓝牙开关为开启 ', ''],
            condition_results=[{'id': 'c1', 'status': 'met', 'evidence': '开关为蓝色'}]))
        self.assertEqual(['蓝牙开关为开启'], parsed['success_conditions'])
        self.assertEqual([{'id': 'c1', 'status': 'met', 'evidence': '开关为蓝色'}], parsed['condition_results'])

    def test_malformed_plan_fields_degrade_to_null_instead_of_vetoing_the_step(self):
        for bad in ({'success_conditions': '一句话'}, {'condition_results': [{'id': 'c1', 'status': 'done'}]},
                    {'condition_results': 'all met'}):
            with self.subTest(bad=bad):
                parsed = parse(wire_decision(**bad))
                self.assertEqual('finish', parsed['status'])
                key = next(iter(bad))
                self.assertIsNone(parsed[key])

    def test_action_decisions_still_parse_without_plan_fields(self):
        normalized = normalize_model_step_decision({'status': 'action', 'action': 'home', 'reason': '回桌面'})
        self.assertIsNone(normalized['success_conditions'])
        self.assertIsNone(normalized['condition_results'])

    def test_prompt_carries_the_plan_rule_and_committed_plan(self):
        context = {'objective': '打开蓝牙', 'entities': {'task_plan': {'version': 1, 'source': 'model_gateway_proposal',
            'success_conditions': [{'id': 'c1', 'description': '蓝牙开关为开启'}]}}}
        prompt = _single_step_observation_prompt(context, include_input_structure=False, image_count=1,
            request_image_size=(720, 1280), available_action_kinds=('home',))
        self.assertIn(TASK_PLAN_RULE, prompt)
        self.assertIn('蓝牙开关为开启', prompt)


if __name__ == '__main__':
    unittest.main()
