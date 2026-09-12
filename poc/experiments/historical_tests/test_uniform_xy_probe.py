"""Isolated request/coordinate checks; no model or device calls."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from experiments import probe_uniform_xy as probe
from test_support.generic_scene_observer import current_axis_grid_payload


class UniformXYTests(unittest.TestCase):
    def test_preflight_no_network_four_pairs(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(probe, 'OUT', Path(directory)), \
                patch('agent.infrastructure.dashscope_vision_provider.httpx.post') as post:
            probe.preflight()
            post.assert_not_called()
            pre = probe.read(probe.OUT / 'preflight.json')
            self.assertEqual(8, len(pre['wire_hashes']))
            self.assertTrue(pre['model_config']['request_options']['enable_thinking'])

    def test_no_future_send_fact_in_failure_fixture(self):
        case = probe.fixture('failed_point')
        last = case['context']['entities']['history'][-1]
        self.assertEqual('input_verified_text', last['canonical_action']['action'])
        self.assertIsNone(last['visual_outcome'])
        self.assertEqual('', last['after_scene'])
        self.assertIn('revision_2_', case['frame_paths'][0])
        self.assertEqual(4, len(case['frames']))

    def test_success_and_focus_are_original_empty_histories(self):
        for name in ('successful_point', 'input_focus'):
            self.assertEqual([], probe.fixture(name)['context']['entities']['history'])
            self.assertFalse(probe.fixture(name)['modified_goal'])
        self.assertTrue(probe.fixture('scroll_variation')['modified_goal'])

    def test_reservation_single_use_stop_and_budget(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(probe, 'OUT', Path(directory)):
            probe.reserve('failed_point_baseline')
            with self.assertRaises(FileExistsError):
                probe.reserve('failed_point_baseline')
            probe.save('stopped.json', {'reason': 'counterexample'})
            with self.assertRaises(AssertionError):
                probe.reserve('failed_point_uniform')
        with tempfile.TemporaryDirectory() as directory, patch.object(probe, 'OUT', Path(directory)):
            for case in probe.CASES:
                for variant in probe.VARIANTS:
                    probe.reserve(case + '_' + variant)
            with self.assertRaises(AssertionError):
                probe.reserve('failed_point_uniform')

    def test_pure_candidate_parser_preserves_uniform_geometry(self):
        record = probe.read(probe.CURRENT / '3baacd4002744532ad5b3c8280bf577e_model_response.json')
        reply = current_axis_grid_payload(json.loads(record['redacted_raw_response']), request_height=1280)
        baseline = probe.parse_reply(json.dumps(reply), uniform=False, actual_size=(720, 1280), include_input=True)
        self.assertEqual([915, 906], baseline['decision']['tap_point'])
        reply['coordinate_space']['height'] = 1000
        reply['decision']['postcondition'] = {'status': 'unknown', 'fact': '当前截图无法确认上一步动作结果'}
        reply['decision']['tap_point'] = [850, 870]
        reply['input_structure']['application_inputs'][0]['bounds'] = [130, 850, 830, 900]
        candidate = probe.parse_reply(json.dumps(reply), uniform=True, actual_size=(720, 1280), include_input=True)
        self.assertEqual([850, 870], candidate['decision']['tap_point'])
        self.assertEqual([130, 850, 830, 900], candidate['input_structure']['application_inputs'][0]['bounds'])
        self.assertEqual('aaazjie！你好', candidate['input_structure']['application_inputs'][0]['text'])


if __name__ == '__main__':
    unittest.main()
