"""No network in these campaign guards/fixture tests."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from experiments import probe_bound_action_result as probe


class ProbeTests(unittest.TestCase):
    def test_duplicate_reservation_does_not_reset_attempt(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(probe, 'OUT', Path(directory)):
            probe.reserve_attempt('failed_send_baseline')
            before = (probe.OUT / 'failed_send_baseline_attempt.json').read_bytes()
            with self.assertRaises(FileExistsError):
                probe.reserve_attempt('failed_send_baseline')
            self.assertEqual(before, (probe.OUT / 'failed_send_baseline_attempt.json').read_bytes())

    def test_stopped_campaign_and_total_budget_reject_new_call(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(probe, 'OUT', Path(directory)):
            probe.save('stopped.json', {'reason': 'counterexample'})
            with self.assertRaises(AssertionError):
                probe.reserve_attempt('failed_send_baseline')
        with tempfile.TemporaryDirectory() as directory, patch.object(probe, 'OUT', Path(directory)):
            for case in probe.CASE_NAMES:
                for variant in probe.VARIANTS:
                    probe.reserve_attempt(case + '_' + variant)
            with self.assertRaises(AssertionError):
                probe.reserve_attempt('failed_send_baseline')
            self.assertEqual(10, len(list(probe.OUT.glob('*_attempt.json'))))

    def test_failed_send_uses_actual_after_frames_and_no_future_outcome(self):
        case = probe.fixtures()['failed_send']
        self.assertEqual(4, len(case['frames']))
        last = case['context']['entities']['history'][-1]
        self.assertEqual(4, last['step'])
        self.assertEqual('send_message', last['canonical_action']['params']['target'])
        self.assertEqual('executed', last['transport_outcome'])
        self.assertIsNone(last['visual_outcome'])
        self.assertEqual('', last['after_scene'])

    def test_all_pairs_offline_preserve_images_and_authorization(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(probe, 'OUT', Path(directory)), \
                patch('agent.infrastructure.dashscope_vision_provider.httpx.post') as network:
            probe.prepare()
            network.assert_not_called()
            preflight = probe.read(probe.OUT / 'preflight.json')
            self.assertEqual(10, len(preflight['wire_hashes']))
            self.assertFalse(preflight['uniform_xy'])
            self.assertFalse(list(probe.OUT.glob('*_attempt.json')))


if __name__ == '__main__':
    unittest.main()
