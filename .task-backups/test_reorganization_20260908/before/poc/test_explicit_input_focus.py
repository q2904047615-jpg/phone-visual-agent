"""Explicit same-response focus authority: real observer -> binder -> Controller."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from test_point_scene_projection import observe, wire, decision
from agent.application.universal_agent_orchestrator import ObservationBridge
from test_qwen_visual_decision import patterned_frames
from test_generic_action_adapter import RawSceneProvider
from agent.infrastructure.generic_scene_observer import SingleStepGenericSceneObserver
from agent.infrastructure.generic_action_adapter import GenericSingleActionAdapter
from test_generic_scene_observer import audited_application_input, input_audit_payload
from test_point_scene_projection import task_context


class ExplicitInputFocusTests(unittest.TestCase):
    def test_multiframe_input_prompt_has_one_fact_first_authority(self):
        from agent.infrastructure.generic_scene_observer import (
            _input_structure_audit_prompt, _single_step_observation_prompt,
        )
        graph = task_context('清空当前输入框')
        context = {'history': [], 'exact_input_text': None}
        prompt = _single_step_observation_prompt({'entities': context},
            include_input_structure=True, image_count=3, request_image_size=(720, 1280),
            available_action_kinds=('tap_semantic', 'clear_verified_text'))
        audit = _input_structure_audit_prompt({'entities': context}, wire_height=1280)
        self.assertNotIn('only facts visible in Image 1', audit)
        self.assertIn('all supplied images from this observation', audit)
        self.assertIn('先独立报告scene和input_structure，再选择动作', prompt)
        self.assertNotIn('先按当前目标选择动作或finish，再按对应字段合同报告', prompt)

    def test_focus_prompt_requires_positive_and_goal_independent_evidence(self):
        from agent.infrastructure.generic_scene_observer import _input_structure_audit_prompt
        prompt = _input_structure_audit_prompt({'entities':
            {'history': [], 'exact_input_text': None}}, wire_height=1280)
        self.assertIn('Never infer focus from the requested action', prompt)
        self.assertIn('false only when direct visual evidence shows another field is focused', prompt)
        self.assertIn('otherwise report null', prompt)

    def payload(self, focused, *, text='draft', cues=(), keyboard=False):
        item = audited_application_input(bounds=[100, 380, 900, 440], text=text,
            visible_editable_cues=list(cues))
        item['focused'] = focused
        audit = input_audit_payload(application_inputs=[item])
        audit['keyboard']['visible'] = keyboard
        choice = decision('clear_verified_text', element_id='local_audited_input_1')
        result = wire(choice, audit)
        result['scene']['elements'] = []
        return result

    def test_explicit_focus_allows_clear_without_keyboard_or_keyword(self):
        for value in ('draft', '中文草稿'):
            with self.subTest(text=value):
                scene, result, resolved, _ = observe(self.payload(True, text=value), task_context('清空当前输入框'))
                self.assertIs(True, scene.elements[0].states['focused'])
                self.assertEqual('clear_verified_text', result.proposal.action.action)

    def test_old_cues_cannot_override_false_or_unknown_focus(self):
        for focused in (False, None):
            with self.subTest(focused=focused):
                with self.assertRaisesRegex(RuntimeError, '聚焦'):
                    observe(self.payload(focused, cues=['cursor', 'AI输入中']), task_context('清空当前输入框'))

    def test_invalid_or_missing_focus_does_not_authorize_clear(self):
        for focus in ('true', 1, [], {}, None):
            with self.subTest(focus=focus):
                with self.assertRaisesRegex(RuntimeError, '聚焦'):
                    observe(self.payload(focus), task_context('清空当前输入框'))
        payload = self.payload(True)
        del payload['input_structure']['application_inputs'][0]['focused']
        with self.assertRaisesRegex(RuntimeError, '聚焦'):
            observe(payload, task_context('清空当前输入框'))

    def test_keyboard_or_scene_focus_cannot_override_target_field(self):
        payload = self.payload(False, cues=['cursor'], keyboard=True)
        payload['scene']['elements'] = [{'element_id': 'local_audited_input_1',
            'role': 'input', 'meaning': 'application_text_input', 'bounds': [100,380,900,440],
            'label': '输入框', 'states': {'focused': True}, 'confidence': 1, 'evidence': ['cursor']}]
        with self.assertRaisesRegex(RuntimeError, '聚焦'):
            observe(payload, task_context('清空当前输入框'))

    def test_unknown_focus_allows_focus_tap_without_changing_point(self):
        payload = self.payload(None)
        payload['decision'] = decision('tap_semantic', target={
            'element_id': 'local_audited_input_1', 'role': 'input',
            'meaning': 'application_text_input', 'label': '输入框', 'evidence': ['完整可见输入栏']},
            tap_point=[500,410])
        scene, result, resolved, _ = observe(payload, task_context('清空当前输入框'))
        self.assertIsNot(True, scene.elements[0].states.get('focused'))
        self.assertEqual('tap_semantic', result.proposal.action.action)
        self.assertEqual((.5, .854), resolved.normalized_point)

    def test_other_fields_focus_cannot_authorize_current_field_clear(self):
        payload = self.payload(False)
        first = payload['input_structure']['application_inputs'][0]
        first['field_labels'] = ['输入框']
        other = deepcopy(first)
        other.update(bounds=[100,100,900,160], field_labels=['其他字段'], focused=True)
        payload['input_structure']['application_inputs'].append(other)
        with self.assertRaises(RuntimeError):
            observe(payload, task_context('清空当前输入框'))

    def test_old_focus_inference_functions_are_removed(self):
        from agent.infrastructure import generic_scene_observer as source
        for name in ('_audited_input_has_focus_cue', '_locally_verified_blinking_caret',
                     '_locally_detect_caret_visual_row', '_INPUT_FOCUS_CUE_MARKERS'):
            self.assertFalse(hasattr(source, name), name)

    def test_empty_field_can_finish_without_focus(self):
        payload = self.payload(None, text='')
        payload['decision'] = decision(None, status='finish',
            evidence_refs=['element:local_audited_input_1'])
        scene, result, resolved, _ = observe(payload, task_context('清空当前输入框'))
        self.assertEqual('finish', result.proposal.status)
        self.assertIsNone(resolved)
        self.assertEqual('', scene.elements[0].states['value'])

    def test_focus_type_is_visible_in_request_schema(self):
        from agent.infrastructure.generic_scene_observer import _single_step_response_format
        schema = _single_step_response_format({}, input_structure_required=True,
            request_height=480, available_action_kinds={'tap_semantic'})
        item = schema['json_schema']['schema']['properties']['input_structure']['properties'][
            'application_inputs']['items']
        self.assertEqual(['boolean', 'null'], item['properties']['focused']['type'])
        self.assertIn('focused', item['required'])

    def test_preparse_response_saved_even_when_parser_fails_and_never_overwritten(self):
        raw = 'invalid JSON Bearer abcdefghijklmnop ' + 'x' * 18000
        provider = RawSceneProvider([raw, raw])
        observer = SingleStepGenericSceneObserver(provider)
        with tempfile.TemporaryDirectory() as directory:
            for _ in range(2):
                with self.assertRaises(RuntimeError):
                    observer.observe_with_decision(frames=patterned_frames(), device_id='device-1',
                        available_action_kinds=frozenset({'tap_semantic'}),
                        response_evidence_dir=Path(directory), response_evidence_prefix='same-step')
            records = list(Path(directory).glob('*_model_response.json'))
            self.assertEqual(2, len(records))
            for path in records:
                saved = json.loads(path.read_text(encoding='utf-8'))
                self.assertEqual(len(raw), saved['raw_response_length'])
                self.assertFalse(saved['redacted_response_truncated'])
                self.assertGreater(len(saved['redacted_raw_response']), 18000)
                self.assertNotIn('abcdefghijklmnop', saved['redacted_raw_response'])
                self.assertEqual('device-1', saved['device_id'])

    def test_real_adapter_passes_evidence_sink_before_any_later_binding(self):
        payload = wire()
        provider = RawSceneProvider([payload])
        observer = SingleStepGenericSceneObserver(provider)
        adapter = SimpleNamespace(observer=observer, device_id='device-1', 
            supported_action_kinds=lambda: frozenset({'tap_semantic'}))
        with tempfile.TemporaryDirectory() as directory:
            GenericSingleActionAdapter._observe_scene(adapter, patterned_frames(), {},
                available_action_kinds=frozenset({'tap_semantic'}),
                response_evidence_dir=Path(directory), response_evidence_prefix='before_step_4')
            records = list(Path(directory).glob('*_model_response.json'))
            self.assertEqual(1, len(records))
            saved = json.loads(records[0].read_text(encoding='utf-8'))
            self.assertEqual(payload, json.loads(saved['redacted_raw_response']))
            self.assertEqual('device-1', saved['device_id'])
            self.assertTrue(saved['fingerprint'])


if __name__ == '__main__':
    unittest.main()
