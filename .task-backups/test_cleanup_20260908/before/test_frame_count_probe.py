import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from experiments import probe_frame_count as probe


class FrameCountTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.out = Path(temp.name)
        patcher = patch.object(probe, 'OUT', self.out)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_both_pairs_differ_only_in_frame_count_and_description(self):
        provider = probe.base.DashScopeVisionProvider(api_key='offline-dummy', enable_thinking=True)
        for case in ('send', 'recent'):
            three, size, required, original = probe.build(provider, case+'_three')
            single, size2, required2, original2 = probe.build(provider, case+'_single')
            self.assertEqual(probe.single_frame(three), single)
            self.assertEqual((size, required, original), (size2, required2, original2))
            self.assertEqual(three['response_format'], single['response_format'])
            self.assertEqual(single['messages'][1]['content'][-1], three['messages'][1]['content'][-1])
            self.assertEqual(len(single['messages'][1]['content']), 3)

    def test_preflight_freezes_four_requests(self):
        probe.prepare()
        pre = probe.base.read(self.out/'preflight.json')
        self.assertEqual(pre['max_calls'], 4)
        self.assertEqual(len(pre['wire_hashes']), 4)
        with self.assertRaises(RuntimeError):
            probe.prepare()

    def test_duplicate_and_unresolved_attempts_rejected(self):
        probe.reserve('send_three')
        with self.assertRaises(FileExistsError):
            probe.reserve('send_three')
        with self.assertRaises(RuntimeError):
            probe.reserve('send_single')

    def test_failed_single_blocks_next_pair_even_without_stop_file(self):
        probe.save('send_three_result.json', {'transport_succeeded': True})
        probe.save('send_single_result.json', {'transport_succeeded': True, 'inside_target_region': False})
        with self.assertRaises(RuntimeError):
            probe.reserve('recent_three')

    def test_stopped_campaign_cannot_resume(self):
        probe.save('stopped.json', {'resume_allowed': False})
        with self.assertRaises(RuntimeError):
            probe.reserve('send_three')


if __name__ == '__main__':
    unittest.main()
