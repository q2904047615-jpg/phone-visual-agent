"""Offline validation-design regressions; no model or hardware calls."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from experiments import probe_history_dialogue as probe


class HistoryProbeFixtureTests(unittest.TestCase):
    def receipt(self, meaning='send_message'):
        return {
            'requested_action': {'node_id': 'step-1', 'action': 'tap_semantic', 'params': {
                'target': meaning, 'role': 'button', 'label': '提交',
                'formal_transition': {'expectations': [{'value': True}]},
                'expected_effect': {'goal_complete_on_success': True}}},
            'resolved_action': {'node_id': 'step-1', 'kind': 'tap_semantic',
                'normalized_point': [0.86, 0.55], 'before_fingerprint': 'actual-before',
                'expected_effect': {'goal_complete_on_success': True},
                'formal_transition': {'expectations': [{'value': True}]},
                'formal_candidate_id': 'old', 'input_method': None},
        }

    def test_both_action_structures_exclude_retired_metadata_without_mutation(self):
        for meaning in ['send_message', 'save_note']:
            with self.subTest(meaning=meaning):
                source = self.receipt(meaning)
                before = deepcopy(source)
                request, resolved = probe.project_historical_point_receipt(source)
                probe.assert_clean_history([request, resolved])
                self.assertEqual(meaning, request['params']['target'])
                self.assertEqual([0.86, 0.55], resolved['normalized_point'])
                self.assertEqual('actual-before', resolved['before_fingerprint'])
                self.assertEqual(before, source)

    def test_old_failed_projection_is_detected(self):
        request, _ = probe.project_historical_point_receipt(self.receipt())
        with self.assertRaisesRegex(ValueError, 'Retired'):
            probe.assert_clean_history({'canonical_action': request,
                'action': self.receipt()['resolved_action']})

    def test_unknown_actual_semantics_and_nonnull_legacy_transport_are_not_silently_dropped(self):
        for key, value in [('input_method', 'mechanical'), ('future_action_fact', 'unknown')]:
            with self.subTest(key=key):
                source = self.receipt()
                source['resolved_action'][key] = value
                with self.assertRaisesRegex(ValueError, 'Unaudited'):
                    probe.project_historical_point_receipt(source)
        source = self.receipt()
        source['requested_action']['params']['future_action_fact'] = 1
        with self.assertRaisesRegex(ValueError, 'Unaudited'):
            probe.project_historical_point_receipt(source)

    def test_literal_user_text_is_not_metadata(self):
        value = {'text': '  expected_effect goal_complete_on_success 中文？e\u0301\n '}
        before = deepcopy(value)
        probe.assert_clean_history(value)
        self.assertEqual(before, value)

    def test_unknown_visual_outcome_and_zero_action_facts_stay_unknown(self):
        request, resolved = probe.project_historical_point_receipt(self.receipt())
        history = probe.execution_history_entry(step=1, requested_action=request,
            resolved_action=resolved, physical_actions=0, transport_outcome='executed')
        self.assertIsNone(history['visual_outcome'])
        self.assertEqual('', history['after_scene'])
        self.assertEqual(0, history['physical_actions'])
        probe.assert_clean_history(history)

    def test_budget_counts_original_attempts_and_blocks_ninth_call(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / 'clean_fixture_v2'
            output.mkdir()
            for index in range(4):
                (root / f'{index}_attempt.json').touch()
            with patch.object(probe, 'CAMPAIGN', root), patch.object(probe, 'OUT', output):
                probe.budget_precheck('sent', 'baseline')
                for index in range(4):
                    (output / f'{index}_attempt.json').touch()
                with self.assertRaisesRegex(AssertionError, 'Original authorization budget'):
                    probe.budget_precheck('sent', 'baseline')

    def test_stopped_or_duplicate_case_cannot_reach_provider(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(probe, 'CAMPAIGN', root), patch.object(probe, 'OUT', root), \
                    patch.object(probe, 'DashScopeVisionProvider') as provider:
                (root / 'stopped.json').touch()
                with self.assertRaisesRegex(AssertionError, 'Comparison stopped'):
                    probe.recognize('sent', 'candidate')
                provider.assert_not_called()
            with patch.object(probe, 'CAMPAIGN', root), patch.object(probe, 'OUT', root / 'new'):
                probe.OUT.mkdir()
                (probe.OUT / 'sent_candidate_attempt.json').touch()
                with self.assertRaisesRegex(AssertionError, 'No resampling'):
                    probe.budget_precheck('sent', 'candidate')

    def test_two_extra_calls_are_only_for_empty_clear_and_do_not_reset_old_budget(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / 'clean_fixture_v2'
            output.mkdir()
            for index in range(8):
                (root / f'{index}_attempt.json').touch()
            permission = {'case': 'empty_clear', 'additional_calls': 2,
                'variants': ['baseline', 'candidate'], 'phone_actions': 0}
            with patch.object(probe, 'CAMPAIGN', root), patch.object(probe, 'OUT', output):
                with self.assertRaises(FileNotFoundError):
                    probe.budget_precheck('empty_clear', 'baseline')
                (output / 'empty_clear_authorization.json').write_text(json.dumps(permission), encoding='utf-8')
                probe.budget_precheck('empty_clear', 'baseline')
                with self.assertRaisesRegex(AssertionError, 'Original authorization'):
                    probe.budget_precheck('sent', 'candidate')
                (output / 'empty_clear_baseline_attempt.json').touch()
                probe.budget_precheck('empty_clear', 'candidate')
                with self.assertRaisesRegex(AssertionError, 'No resampling'):
                    probe.budget_precheck('empty_clear', 'baseline')
                (output / 'empty_clear_candidate_attempt.json').touch()
                with self.assertRaisesRegex(AssertionError, 'Extended authorization'):
                    probe.budget_precheck('empty_clear', 'candidate')


if __name__ == '__main__':
    unittest.main()
