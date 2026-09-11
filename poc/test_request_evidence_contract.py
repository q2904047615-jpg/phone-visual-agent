"""Offline contract and exact request-content evidence; no model/device access."""
import base64
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import httpx
from PIL import Image
from agent.infrastructure.dashscope_vision_provider import DashScopeVisionProvider, _image_data_url
from agent.infrastructure.generic_scene_observer import (
    SingleStepGenericSceneObserver, _single_step_response_format,
    _parse_single_step_observation_envelope, INPUT_STRUCTURE_AUDIT_VERSION,
)
from contract_tests.protocol.test_flat_observation_contract import decision, SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION
import test_vision_model_config as model_tests
from test_qwen_visual_decision import patterned_frames
from test_single_visual_loop import LoopHarness, scene, decision as loop_decision


class RequestEvidenceTests(unittest.TestCase):
    def test_exact_outgoing_body_and_jpeg_are_saved_before_network_without_auth(self):
        for thinking in (False, True):
            with self.subTest(thinking=thinking), tempfile.TemporaryDirectory() as tmp:
                provider = DashScopeVisionProvider(api_key='test-secret-not-in-artifact',
                    enable_thinking=thinking, max_attempts=1)
                path = Path(tmp) / 'nested/request.json'
                image_url = _image_data_url(Image.new('RGB', (810, 1440), 'purple'))
                messages = [{'role': 'user', 'content': [{'type': 'text', 'text': '原文？\n  e\u0301 '},
                    {'type': 'image_url', 'image_url': {'url': image_url}}]}]
                schema = _single_step_response_format({}, input_structure_required=True,
                    request_height=1280, available_action_kinds=('tap_semantic', 'home'))

                def post(*args, **kwargs):
                    saved = json.loads(path.read_text(encoding='utf-8'))
                    self.assertEqual(kwargs['json'], saved)
                    self.assertEqual(thinking, saved['enable_thinking'])
                    input_schema = saved['response_format']['json_schema']['schema']['properties']['input_structure']
                    self.assertEqual([INPUT_STRUCTURE_AUDIT_VERSION],
                        input_schema['properties']['protocol_version']['enum'])
                    self.assertNotIn('max_tokens', saved)
                    self.assertNotIn('test-secret-not-in-artifact', path.read_text(encoding='utf-8'))
                    url = saved['messages'][0]['content'][1]['image_url']['url']
                    self.assertEqual(base64.b64decode(image_url.split(',')[1]),
                        base64.b64decode(url.split(',')[1]))
                    return model_tests.DashScopeVisionModelRequestTests._response()

                with patch('agent.infrastructure.dashscope_vision_provider.httpx.post', side_effect=post) as network:
                    with provider.call_scope(stage='test', fingerprint='frame', request_evidence_path=path):
                        provider._chat(messages, max_tokens=None, response_format=schema)
                self.assertEqual(1, network.call_count)
                self.assertIsNone(provider._active_request_evidence.get())

    def test_network_failure_retains_request_and_resets_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            provider = DashScopeVisionProvider(api_key='test-key', max_attempts=1)
            path = Path(tmp) / 'request.json'
            with patch('agent.infrastructure.dashscope_vision_provider.httpx.post',
                side_effect=httpx.ConnectError('offline')) as network:
                with self.assertRaisesRegex(Exception, '中断'), provider.call_scope(
                    stage='test', request_evidence_path=path):
                    provider._chat([{'role': 'user', 'content': 'json'}], max_tokens=None)
            self.assertTrue(path.is_file())
            self.assertEqual(1, network.call_count)
            self.assertIsNone(provider._active_request_evidence.get())

    def test_evidence_io_failure_does_not_send_unrecorded_request(self):
        provider = DashScopeVisionProvider(api_key='test-key', max_attempts=1)
        with patch('agent.infrastructure.dashscope_vision_provider.atomic_replace_bytes',
            side_effect=OSError('disk full')), patch(
            'agent.infrastructure.dashscope_vision_provider.httpx.post') as network:
            with self.assertRaisesRegex(OSError, 'disk full'), provider.call_scope(
                stage='test', request_evidence_path=Path('unused.json')):
                provider._chat([{'role': 'user', 'content': 'json'}], max_tokens=None)
            network.assert_not_called()

    def test_real_observer_links_request_and_response_even_when_parse_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            provider = DashScopeVisionProvider(api_key='test-key', max_attempts=1)
            observer = SingleStepGenericSceneObserver(provider)
            with patch('agent.infrastructure.dashscope_vision_provider.httpx.post',
                return_value=model_tests.DashScopeVisionModelRequestTests._response()) as network:
                with self.assertRaises(Exception):
                    observer.observe_with_decision(frames=patterned_frames(), response_evidence_dir=Path(tmp))
            requests = list(Path(tmp).glob('*_model_request.json'))
            responses = list(Path(tmp).glob('*_model_response.json'))
            self.assertEqual((1, 1, 1), (len(requests), len(responses), network.call_count))
            saved_response = json.loads(responses[0].read_text(encoding='utf-8'))
            self.assertEqual(str(requests[0]), saved_response['request_evidence_path'])
            self.assertEqual(network.call_args.kwargs['json'], json.loads(requests[0].read_text(encoding='utf-8')))


class OptionalFieldContractTests(unittest.TestCase):
    def test_optional_target_details_on_two_canvases_keep_exact_point(self):
        for height in (960, 1280):
            schema = _single_step_response_format({}, input_structure_required=True,
                request_height=height, available_action_kinds=('tap_semantic',))['json_schema']['schema']
            target = schema['properties']['decision']['properties']['target']
            self.assertEqual(['role', 'meaning'], target['required'])
            fields = schema['properties']['input_structure']['properties']['application_inputs']['items']
            self.assertNotIn('text', fields['required'])
            self.assertIn('focused', fields['required'])
            for details in ({}, {'label': '目标', 'evidence': ['当前可见']}):
                payload = {'protocol_version': SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
                    'coordinate_space': {'kind': 'axis_grid', 'width': 1000, 'height': height},
                    'scene': {'elements': [], 'summary': '当前目标可见'}, 'input_structure': None,
                    'decision': decision(action='tap_semantic',
                        target={'role': 'button', 'meaning': 'open_details', **details}, tap_point=[400, 500])}
                parsed = _parse_single_step_observation_envelope(json.dumps(payload),
                    input_structure_required=False, request_image_size=(720, height))
                self.assertEqual([400, round(500 * 1000 / height)], parsed['decision']['tap_point'])


class ExecutionEvidenceTests(LoopHarness):
    def test_transport_receipt_never_overwrites_visual_outcome(self):
        for outcome in ('matched', 'unmatched', 'uncertain'):
            with self.subTest(outcome=outcome):
                loop, session, adapter = self.start([
                    (scene(0), loop_decision('tap_semantic', meaning='send_message')),
                    (scene(1), loop_decision(outcome=outcome))], goal='发送一条消息')
                loop.run_autonomous_safe_loop(session)
                self.assertEqual('succeeded' if outcome == 'matched' else 'failed', session.status)
                self.assertEqual(1, len(adapter.calls))
                for filename in ('verification_step_1.json', 'post_action_transition_step_1.json'):
                    record = json.loads((session.run_dir / filename).read_text(encoding='utf-8'))
                    self.assertEqual('executed', record['execution_outcome'])
                    self.assertEqual(outcome, record['visual_outcome'])
                    self.assertNotIn('outcome', record)
                self.assertEqual('executed', session.history[-1]['execution']['action_outcome'])
                self.assertEqual(outcome, session.history[-1]['execution']['visual_outcome'])


if __name__ == '__main__':
    unittest.main()
