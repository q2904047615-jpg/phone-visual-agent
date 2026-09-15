from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from experiments import probe_thinking_mode as probe


class ThinkingModeProbeTests(unittest.TestCase):
    def test_prepare_is_offline_and_changes_only_thinking(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(probe, 'OUT', Path(directory)), \
                patch('agent.infrastructure.dashscope_vision_provider.httpx.post') as post:
            probe.prepare()
            post.assert_not_called()
            self.assertEqual(12, len(probe.read(probe.OUT / 'preflight.json')['wire_hashes']))
            for name in probe.NAMES:
                off = probe.read(probe.OUT / f'{name}_off_wire.json')
                on = probe.read(probe.OUT / f'{name}_on_wire.json')
                self.assertIs(off.pop('enable_thinking'), False)
                self.assertIs(on.pop('enable_thinking'), True)
                self.assertEqual(off, on)
            with self.assertRaises(AssertionError):
                probe.prepare()

    def test_duplicate_budget_stop_and_unknown_case(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(probe, 'OUT', Path(directory)):
            probe.reserve('failed_send', 'on')
            with self.assertRaises(FileExistsError):
                probe.reserve('failed_send', 'on')
            with self.assertRaises(AssertionError):
                probe.reserve('unapproved', 'on')
            for name in probe.NAMES:
                for mode in probe.MODES:
                    if (name, mode) != ('failed_send', 'on'):
                        probe.reserve(name, mode)
            self.assertEqual(12, len(list(probe.OUT.glob('*_attempt.json'))))
            with self.assertRaises(AssertionError):
                probe.reserve('sent', 'on')
        with tempfile.TemporaryDirectory() as directory, patch.object(probe, 'OUT', Path(directory)):
            probe.save('stopped.json', {})
            with self.assertRaises(AssertionError):
                probe.reserve('sent', 'off')

    def test_wording_variation_preserves_history_and_pixels(self):
        cases = probe.fixtures()
        a, b = cases['failed_send'], cases['failed_send_reworded']
        self.assertNotEqual(a['context']['objective'], b['context']['objective'])
        self.assertEqual(a['context']['entities'], b['context']['entities'])
        self.assertEqual([f.tobytes() for f in a['frames']], [f.tobytes() for f in b['frames']])
        self.assertIsNone(a['context']['entities']['history'][-1]['visual_outcome'])

    def test_actual_transport_passes_flag_and_records_reasoning_usage(self):
        response = Mock()
        response.json.return_value = {'choices': [{'message': {'content': '{}',
            'reasoning_content': 'not an action'}, 'finish_reason': 'stop'}],
            'model': 'qwen3-vl-plus', 'usage': {'completion_tokens_details': {'reasoning_tokens': 42}}}
        bodies = []
        for enabled in (False, True):
            provider = probe.DashScopeVisionProvider(api_key='offline-dummy', enable_thinking=enabled)
            with patch('agent.infrastructure.dashscope_vision_provider.httpx.post', return_value=response) as post:
                self.assertEqual('{}', provider._chat([{'role':'user', 'content':'JSON'}],
                    max_tokens=None, max_attempts=1, timeout=probe.OBSERVATION_TIMEOUT_SECONDS,
                    response_format={'type':'json_object'}))
                self.assertEqual(1, post.call_count)
                bodies.append(post.call_args.kwargs['json'])
                self.assertEqual(42, provider.last_usage['completion_tokens_details']['reasoning_tokens'])
                self.assertEqual('stop', provider.last_finish_reason)
        self.assertIs(bodies[0].pop('enable_thinking'), False)
        self.assertIs(bodies[1].pop('enable_thinking'), True)
        self.assertEqual(bodies[0], bodies[1])


if __name__ == '__main__':
    unittest.main()
