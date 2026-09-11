"""Offline guards for stopped experiments; historical prompt shapes are not live contracts."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import probe_frame_count as frames
from experiments import probe_prompt_delta as prompts


class HistoricalProbeGuardTests(unittest.TestCase):
    def test_campaign_guards(self):
        for probe, first, second in (
            (frames, 'send_three', 'send_single'), (prompts, 'old', 'current'),
        ):
            for state in ('duplicate', 'unresolved', 'failed', 'stopped', 'frozen'):
                with self.subTest(probe=probe.__name__, state=state), \
                        tempfile.TemporaryDirectory() as directory, \
                        patch.object(probe, 'OUT', Path(directory)), \
                        patch.object(probe, 'build') as build:
                    if state == 'duplicate':
                        probe.reserve(first)
                        with self.assertRaises(FileExistsError):
                            probe.reserve(first)
                    elif state == 'unresolved':
                        probe.reserve(first)
                        with self.assertRaises(RuntimeError):
                            probe.reserve(second)
                    elif state == 'failed':
                        probe.save(first + '_result.json', {'transport_succeeded': False})
                        with self.assertRaises(RuntimeError):
                            probe.reserve(second)
                    elif state == 'stopped':
                        probe.save('stopped.json', {'resume_allowed': False})
                        with self.assertRaises(RuntimeError):
                            probe.reserve(first)
                    else:
                        probe.save('preflight.json', {})
                        with self.assertRaises(RuntimeError):
                            probe.prepare()
                    build.assert_not_called()

    def test_failed_single_frame_blocks_next_pair_without_stop_file(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(frames, 'OUT', Path(directory)):
            frames.save('send_three_result.json', {'transport_succeeded': True})
            frames.save('send_single_result.json', {'transport_succeeded': True, 'inside_target_region': False})
            with self.assertRaises(RuntimeError):
                frames.reserve('recent_three')

    def test_received_malformed_prompt_result_allows_prespecified_comparison(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(prompts, 'OUT', Path(directory)):
            prompts.save('old_result.json', {'transport_succeeded': True, 'parse_succeeded': False})
            prompts.reserve('current')
            self.assertTrue((prompts.OUT / 'current_attempt.json').exists())


if __name__ == '__main__':
    unittest.main()
