import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import probe_gui_owl_native_history as probe


class NativeHistoryTests(unittest.TestCase):
    def test_receipt_projection_does_not_leak_future_success(self):
        receipt = {'physical_actions': 1, 'requested_action': {'action': 'tap_semantic',
            'params': {'label': '发送', 'expected_effect': {'goal_complete_on_success': True}}},
            'resolved_action': {'kind': 'tap_semantic', 'normalized_point': [.4, .8],
                                'formal_transition': {'value': 'LEAK'}},
            'verification': 'LEAK'}
        action = probe.receipt_action(receipt)
        self.assertNotIn('LEAK', action)
        self.assertNotIn('goal_complete', action)
        self.assertEqual(probe.parse_action(action)['arguments']['coordinate'], [400, 800])

    def test_unexecuted_history_is_rejected(self):
        with self.assertRaises(ValueError):
            probe.receipt_action({'physical_actions': 0})

    def test_source_must_match_exact_audited_hash(self):
        with self.assertRaises(ValueError):
            probe.load_pure_upstream('raise RuntimeError("do not run")')

    def test_source_evidence_order(self):
        for name in probe.CASE_NAMES:
            current, history, audit = probe.evidence(name)
            self.assertEqual(current, audit[-1]['after'])
            self.assertEqual([h['image'] for h in history], [a['before'] for a in audit])
            self.assertTrue(all(Path(a['before']).is_file() for a in audit))
            self.assertTrue(all('goal_complete' not in h['output'] for h in history))

    def test_stop_prevents_any_network(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(probe, 'OUT', Path(directory)):
            (Path(directory) / 'stopped.json').touch()
            with self.assertRaises(RuntimeError):
                probe.recognize('sent')

    def test_no_resampling(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(probe, 'OUT', Path(directory)):
            (Path(directory) / 'sent_attempt.json').touch()
            with self.assertRaises(RuntimeError):
                probe.guard('sent')


if __name__ == '__main__':
    unittest.main()
