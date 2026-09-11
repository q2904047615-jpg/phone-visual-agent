"""Offline experiment-boundary tests, not model/device capability evidence."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
from experiments import probe_response_format as probe


class ResponseFormatProbeTests(unittest.TestCase):
    def test_only_format_changes_and_preparation_cannot_reset(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(probe, 'OUT', Path(directory)), \
                patch('agent.infrastructure.dashscope_vision_provider.httpx.post') as post:
            probe.prepare()
            post.assert_not_called()
            schema = probe.base.read(probe.OUT / 'schema_wire.json')
            omitted = probe.base.read(probe.OUT / 'omitted_wire.json')
            self.assertEqual(schema, probe.base.read(probe.REFERENCE))
            del schema['response_format']
            self.assertEqual(schema, omitted)
            self.assertTrue(omitted['enable_thinking'])
            with self.assertRaises(RuntimeError):
                probe.prepare()

    def test_actual_transport_omits_field_not_null_or_json_object(self):
        provider = probe.base.DashScopeVisionProvider(api_key='offline-dummy', enable_thinking=True, max_attempts=1)
        for variant in probe.ORDER:
            body, _, _, _ = probe.build(provider, variant)
            response = httpx.Response(200, request=httpx.Request('POST', 'https://offline.invalid'),
                json={'id': 'offline-id', 'model': provider.model,
                    'choices': [{'message': {'content': '{}'}, 'finish_reason': 'stop'}], 'usage': {}})
            with patch('agent.infrastructure.dashscope_vision_provider.httpx.post', return_value=response) as post:
                self.assertEqual('{}', probe.chat(provider, body))
                self.assertEqual(body, post.call_args.kwargs['json'])
                self.assertEqual(1, post.call_count)

    def test_parse_failure_keeps_raw_and_allows_second_but_never_third(self):
        raw = json.dumps([probe.base.grounding_schema(1280)['json_schema']['schema']])
        valid = json.dumps({'coordinate_space': {'kind': 'axis_grid', 'width': 1000, 'height': 1280},
            'tap_point': [895, 1125]})
        with tempfile.TemporaryDirectory() as directory, patch.object(probe, 'OUT', Path(directory)):
            probe.prepare()
            with patch.object(probe.base.DashScopeVisionProvider, 'configured', True), \
                    patch.object(probe, 'chat', side_effect=[raw, valid]) as chat:
                probe.recognize('schema')
                self.assertEqual(raw, probe.base.read(probe.OUT / 'schema_raw.json')['raw'])
                self.assertFalse((probe.OUT / 'stopped.json').exists())
                probe.recognize('omitted')
                self.assertEqual(2, chat.call_count)
                self.assertTrue(probe.base.read(probe.OUT / 'omitted_result.json')['inside_target_region'])
                with self.assertRaises(RuntimeError):
                    probe.recognize('omitted')

    def test_transport_failure_or_unresolved_reservation_never_retries(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(probe, 'OUT', Path(directory)):
            probe.prepare()
            with patch.object(probe.base.DashScopeVisionProvider, 'configured', True), \
                    patch.object(probe, 'chat', side_effect=TimeoutError('offline simulated timeout')) as chat:
                probe.recognize('schema')
                with self.assertRaises(RuntimeError):
                    probe.recognize('omitted')
                self.assertEqual(1, chat.call_count)
        with tempfile.TemporaryDirectory() as directory, patch.object(probe, 'OUT', Path(directory)):
            probe.reserve('schema')
            with self.assertRaises(RuntimeError):
                probe.reserve('omitted')
            with self.assertRaises(RuntimeError):
                probe.reserve('schema')


if __name__ == '__main__':
    unittest.main()
