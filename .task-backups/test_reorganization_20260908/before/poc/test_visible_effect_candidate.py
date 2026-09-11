from copy import deepcopy
import json
import unittest

from experiments.visible_effect_candidate import prepare, project_for_evaluation, FIELD
from experiments.bound_action_result_candidate import PREFIX
from experiments.probe_bound_action_result import fixtures
import test_bound_action_result_candidate as bound_fixtures


class VisibleEffectTests(unittest.TestCase):
    def test_only_result_contract_changes_and_literal_task_is_preserved(self):
        fixture = bound_fixtures.BoundActionResultTests()
        context = fixture.context()
        baseline = fixture.prepare(context, bind_result=False)
        candidate = prepare(context, request_width=720, request_height=1280, image_count=3,
            available_action_kinds=('tap_semantic', 'home', 'clear_verified_text'))
        sent, _ = json.JSONDecoder().raw_decode(candidate['prompt'].split(PREFIX, 1)[1])
        self.assertEqual(context, sent)
        self.assertIn('不判断动作选得对不对', candidate['prompt'])
        self.assertIn('CURRENT组每张JPEG为720×1280', candidate['prompt'])
        self.assertNotIn('LAST_EXECUTED_ACTION_ID', candidate['prompt'])
        self.assertIn(FIELD, candidate['response_format']['json_schema']['schema']['properties']['decision']['properties'])
        for field in ('scene', 'input_structure', 'coordinate_space'):
            self.assertEqual(baseline['response_format']['json_schema']['schema']['properties'][field],
                candidate['response_format']['json_schema']['schema']['properties'][field])

    def test_mapping_is_literal_not_a_second_visual_judge(self):
        for visible, expected in ((True, 'matched'), (False, 'unmatched'), (None, 'uncertain')):
            payload = {'decision': {FIELD: visible, 'status': 'action', 'action': 'home',
                'reason': '可能仍有模型语义错误', 'text': '原文\n  e\u0301 '}}
            original = deepcopy(payload)
            result = project_for_evaluation(payload, has_history=True)
            self.assertEqual(expected, result['decision']['previous_action_outcome'])
            self.assertEqual(original, payload)
            self.assertEqual(payload['decision']['reason'], result['decision']['reason'])
            self.assertEqual(payload['decision']['text'], result['decision']['text'])

    def test_invalid_type_legacy_field_and_initial_claim_rejected(self):
        for value in (1, 0, 'true', 'matched', [], {}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                project_for_evaluation({'decision': {FIELD: value}}, has_history=True)
        with self.assertRaises(ValueError):
            project_for_evaluation({'decision': {FIELD: True}}, has_history=False)
        with self.assertRaises(ValueError):
            project_for_evaluation({'decision': {FIELD: None, 'previous_action_outcome': None}}, has_history=True)
        self.assertIsNone(project_for_evaluation({'decision': {FIELD: None}},
            has_history=False)['decision']['previous_action_outcome'])

    def test_all_saved_fixture_contexts_remain_unchanged(self):
        for name, case in fixtures().items():
            with self.subTest(name=name):
                candidate = prepare(case['context'], request_width=720, request_height=1280,
                    image_count=3, available_action_kinds=tuple(case['allowed']))
                sent, _ = json.JSONDecoder().raw_decode(candidate['prompt'].split(PREFIX, 1)[1])
                self.assertEqual(case['context'], sent)
                self.assertIsNone(sent['entities']['history'][-1]['visual_outcome'])


if __name__ == '__main__':
    unittest.main()
