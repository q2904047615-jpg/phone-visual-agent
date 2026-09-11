import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import httpx
from experiments import probe_full_response as probe


class FullResponseProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        self.override = patch.object(probe, 'OUT', self.out)
        self.override.start()
        self.addCleanup(self.override.stop)
        self.provider = probe.base.DashScopeVisionProvider(api_key='offline-dummy', enable_thinking=True)
        self.body = dict(model=self.provider.model, temperature=0.0, messages=[],
            response_format={'type': 'json_object'}, **self.provider.model_config.request_options())

    def test_preserves_full_response_without_changing_provider_content(self):
        payload = {'id': 'test', 'model': self.provider.model, 'choices': [{'finish_reason': 'stop',
            'message': {'content': '{"tap_point":[1,2]}', 'reasoning_content': 'diagnostic only'}}]}
        response = httpx.Response(200, json=payload, request=httpx.Request('POST', self.provider.base_url))
        with patch.object(probe.transport.httpx, 'post', return_value=response) as post:
            content = probe.capture_chat(self.provider, self.body)
        self.assertEqual(content, payload['choices'][0]['message']['content'])
        self.assertEqual(probe.base.read(self.out / 'response_body.json'), payload)
        self.assertEqual(post.call_args.kwargs['json'], self.body)
        post.assert_called_once()
        self.assertNotIn('offline-dummy', (self.out / 'response_body.json').read_text())

    def test_timeout_has_no_retry(self):
        with patch.object(probe.transport.httpx, 'post', side_effect=httpx.ReadTimeout('test')) as post:
            with self.assertRaises(Exception):
                probe.capture_chat(self.provider, self.body)
        post.assert_called_once()
        self.assertEqual(self.provider.last_network_attempts, 1)

    def test_bad_content_still_preserves_reasoning(self):
        payload = {'choices': [{'message': {'content': '', 'reasoning_content': 'retained'}}]}
        response = httpx.Response(200, json=payload, request=httpx.Request('POST', self.provider.base_url))
        with patch.object(probe.transport.httpx, 'post', return_value=response) as post:
            with self.assertRaises(Exception):
                probe.capture_chat(self.provider, self.body)
        self.assertEqual(probe.base.read(self.out / 'response_body.json'), payload)
        post.assert_called_once()

    def test_budget_cannot_be_reused(self):
        probe.reserve()
        with self.assertRaises(FileExistsError):
            probe.reserve()

    def test_saved_fixture_has_empty_history_and_full_schema(self):
        body, size, include_input, original = probe.build(self.provider)
        self.assertEqual(size, (720, 1280))
        self.assertEqual(original, (810, 1440))
        self.assertTrue(include_input)
        self.assertEqual(body['response_format']['type'], 'json_schema')
        self.assertTrue(body['enable_thinking'])


if __name__ == '__main__':
    unittest.main()
