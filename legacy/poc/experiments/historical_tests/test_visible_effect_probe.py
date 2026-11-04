from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from experiments import probe_visible_effect as probe


class VisibleEffectProbeTests(unittest.TestCase):
    def test_budget_duplicate_and_stop_cannot_resume_calls(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(probe, 'OUT', Path(directory)):
            probe.reserve('failed_send')
            with self.assertRaises(FileExistsError):
                probe.reserve('failed_send')
            for name in probe.NAMES[1:]:
                probe.reserve(name)
            with self.assertRaises(AssertionError):
                probe.reserve('failed_send')
        with tempfile.TemporaryDirectory() as directory, patch.object(probe, 'OUT', Path(directory)):
            probe.save('stopped.json', {})
            with self.assertRaises(AssertionError):
                probe.reserve('failed_send')

    def test_prepare_is_network_free_and_contains_exactly_six_cases(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(probe, 'OUT', Path(directory)), \
                patch('agent.infrastructure.dashscope_vision_provider.httpx.post') as post:
            probe.prepare()
            post.assert_not_called()
            self.assertEqual(set(probe.NAMES), set(probe.read(probe.OUT / 'preflight.json')['wire_hashes']))
            self.assertFalse(list(probe.OUT.glob('*_attempt.json')))

    def test_variation_changes_only_task_wording(self):
        cases = probe.fixtures()
        first = cases['failed_send']['context']
        second = cases['failed_send_reworded']['context']
        self.assertNotEqual(first['objective'], second['objective'])
        self.assertEqual(first['entities'], second['entities'])
        for a,b in zip(cases['failed_send']['frames'], cases['failed_send_reworded']['frames']):
            self.assertEqual(a.tobytes(), b.tobytes())


if __name__ == '__main__':
    unittest.main()
