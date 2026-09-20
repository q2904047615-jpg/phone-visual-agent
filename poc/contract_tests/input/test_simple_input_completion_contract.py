"""Saved model failures through the real observer/binder/Controller; no live I/O."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from agent.infrastructure.generic_scene_observer import (
    SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION, INPUT_STRUCTURE_AUDIT_VERSION,
)
from contract_tests.observation.test_point_scene_projection import observe, task_context


def saved_response(index):
    records = json.loads((Path(__file__).resolve().parents[2] / 'test_fixtures' /
        'input_reference_failures_20260904.json').read_text(encoding='utf-8'))
    payload = deepcopy(records[index]['response'])
    # Only upgrade explicit version identifiers. Preserve all model facts,
    # identifiers, action choices and reference strings from the failed response.
    payload['protocol_version'] = SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION
    payload['input_structure']['protocol_version'] = INPUT_STRUCTURE_AUDIT_VERSION
    if payload['decision'].get('action') == 'input_verified_text':
        payload['decision']['text'] = 'aaazjie？你好'  # New wire field, exact legacy task body.
    return payload


def input_graph():
    return task_context('输入本次文字', exact_input_text='aaazjie？你好')


class SimpleInputCompletionTests(unittest.TestCase):
    def test_saved_empty_finish_uses_current_fact_not_reference_syntax(self):
        scene, result, resolved, _ = observe(saved_response(0), task_context('清空输入框'))
        self.assertEqual('finish', result.proposal.status)
        self.assertIsNone(resolved)
        self.assertEqual('', next(e for e in scene.elements if e.role == 'input').states['value'])

    def test_saved_input_choice_binds_without_model_knowing_local_id(self):
        _, result, resolved, _ = observe(saved_response(1), input_graph())
        self.assertEqual('input_verified_text', result.proposal.action.action)
        self.assertIsNotNone(resolved)

    def test_input_and_clear_do_not_require_any_wire_element_id(self):
        for kind, graph in (('input_verified_text', input_graph()), ('clear_verified_text', task_context('清空输入框'))):
            payload = saved_response(1)
            payload['decision'].update(action=kind, element_id=None, text='aaazjie？你好' if kind == 'input_verified_text' else None)
            payload['decision'].pop('evidence_refs')
            payload['input_structure']['application_inputs'][0].pop('element_id')
            payload['input_structure']['application_inputs'][0]['text'] = '草稿' if kind.startswith('clear') else ''
            with self.subTest(kind=kind):
                _, result, _, _ = observe(payload, graph)
                self.assertEqual(kind, result.proposal.action.action)

    def test_finish_is_whole_task_model_judgment_not_local_subgoal_checklist(self):
        payload = saved_response(0)
        payload['decision'].pop('evidence_refs')
        self.assertEqual('finish', observe(payload, task_context('检查当前输入框'))[1].proposal.status)
        self.assertNotIn('transition_receipt', task_context().to_dict())

    def test_unknown_focus_never_authorizes_text_even_with_correct_id(self):
        for focus in (False, None):
            payload = saved_response(1)
            payload['input_structure']['application_inputs'][0]['focused'] = focus
            with self.subTest(focus=focus), self.assertRaisesRegex(RuntimeError, '聚焦'):
                observe(payload, input_graph())

    def test_ambiguous_fields_never_choose_first(self):
        payload = saved_response(1)
        other = deepcopy(payload['input_structure']['application_inputs'][0])
        other.update(bounds=[100,200,800,300], element_id='different-field')
        payload['input_structure']['application_inputs'].append(other)
        with self.assertRaises(RuntimeError):
            observe(payload, input_graph())

    def test_optional_label_cannot_select_one_of_two_input_candidates(self):
        payload = saved_response(1)
        first = payload['input_structure']['application_inputs'][0]
        first['field_labels'] = ['聊天输入框']
        second = deepcopy(first)
        second.update(bounds=[100,200,800,300], field_labels=['其他输入框'])
        payload['input_structure']['application_inputs'].append(second)
        with self.assertRaises(RuntimeError):
            observe(payload, task_context('清空输入框'))

    def test_focus_tap_does_not_require_internal_id_or_diagnostic_prose(self):
        payload = saved_response(1)
        payload['decision'].update(action='tap_semantic', element_id=None, text=None,
            target={'role': 'input', 'meaning': 'application_text_input'}, tap_point=[400,906])
        payload['input_structure']['application_inputs'][0].update(focused=None)
        _, result, resolved, _ = observe(payload, input_graph())
        self.assertEqual('tap_semantic', result.proposal.action.action)
        self.assertEqual((.4, .906), resolved.normalized_point)

    def test_focus_tap_can_create_input_fact_when_audit_is_empty(self):
        payload = saved_response(1)
        payload['input_structure']['application_inputs'] = []
        payload['decision'].update(action='tap_semantic', element_id=None, text=None,
            target={'role': 'input', 'meaning': 'application_text_input'}, tap_point=[400,906])
        _, result, resolved, _ = observe(payload, input_graph())
        self.assertEqual('tap_semantic', result.proposal.action.action)
        self.assertEqual((.4, .906), resolved.normalized_point)

    def test_retired_reference_and_identity_projection_functions_are_absent(self):
        from agent.infrastructure import generic_scene_observer as module
        for name in ('_single_step_input_surface_attestation', '_projected_input_element_id',
                '_focus_only_compact_input_surface'):
            self.assertFalse(hasattr(module, name), name)
        schema = module._single_step_response_format({}, input_structure_required=True,
            request_height=1280, available_action_kinds={'tap_semantic','input_verified_text'})
        choice = schema['json_schema']['schema']['properties']['decision']['properties']
        self.assertNotIn('evidence_refs', choice)
        self.assertNotIn('element_id', choice['target']['properties'])

    def test_other_app_and_field_names_use_same_contract(self):
        payload = saved_response(1)
        payload['scene'].update(foreground_app_id='browser', screen_id='search', summary='搜索框为空且已聚焦')
        payload['decision'].update(element_id='search_query', reason='输入到当前已聚焦搜索框')
        payload['input_structure']['application_inputs'][0]['element_id'] = 'search_query'
        self.assertEqual('input_verified_text', observe(payload, input_graph())[1].proposal.action.action)


if __name__ == '__main__':
    unittest.main()
