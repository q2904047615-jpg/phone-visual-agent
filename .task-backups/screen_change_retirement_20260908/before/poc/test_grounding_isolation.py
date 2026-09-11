"""Offline-only checks for the frozen grounding diagnosis, not model capability tests."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import probe_grounding_isolation as probe


class GroundingIsolationTests(unittest.TestCase):
    def test_four_pairs_no_network_no_answer_coordinates(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(probe, 'OUT', Path(directory)), \
                patch('agent.infrastructure.dashscope_vision_provider.httpx.post') as post:
            probe.preflight()
            post.assert_not_called()
            pre = probe.read(probe.OUT / 'preflight.json')
            self.assertEqual(8, len(pre['wire_hashes']))
            self.assertTrue(pre['model_config']['request_options']['enable_thinking'])
            for name in probe.CASES:
                full = probe.read(probe.OUT / (name + '_full_wire.json'))
                pure = probe.read(probe.OUT / (name + '_grounding_wire.json'))
                self.assertEqual(full['messages'][1]['content'][1:], pure['messages'][1]['content'][1:])
                prompt = pure['messages'][1]['content'][0]['text']
                self.assertNotIn('history', prompt)
                self.assertNotIn('aaazjie', prompt)
                self.assertNotIn('641', prompt)
                self.assertIn(probe.SPECS[name][3], prompt)
            with self.assertRaises(RuntimeError):
                probe.preflight()

    def test_historical_timing_no_future_fact(self):
        last = probe.fixture('failed_send')['context']['entities']['history'][-1]
        self.assertEqual('input_verified_text', last['canonical_action']['action'])
        self.assertIsNone(last['visual_outcome'])
        self.assertEqual('', last['after_scene'])
        self.assertEqual([], probe.fixture('successful_send')['context']['entities']['history'])
        self.assertEqual('open_recent_apps', probe.fixture('recent_clear')['context']['entities']['history'][-1]['canonical_action']['action'])

    def test_parser_same_scale_no_remapping_and_invalid_points(self):
        value = {'coordinate_space': {'kind': 'axis_grid', 'width': 1000, 'height': 1280}, 'tap_point': [915, 1160]}
        point, _ = probe.parse_reply(json.dumps(value), 'grounding', (720, 1280), True)
        self.assertEqual([915, 906], point)
        for bad in ([True, 2], [1001, 900], [900, 1281], [1], [1, 2, 3], ['1', 2]):
            value['tap_point'] = bad
            with self.assertRaises(ValueError):
                probe.parse_reply(json.dumps(value), 'grounding', (720, 1280), True)
        value['tap_point'] = None
        self.assertIsNone(probe.parse_reply(json.dumps(value), 'grounding', (720, 1280), True)[0])

    def test_regions_distinguish_original_miss_and_success(self):
        self.assertFalse(probe.in_region([740, 1304], probe.REGIONS['failed_send']))
        self.assertTrue(probe.in_region([724, 1265], probe.REGIONS['successful_send']))
        self.assertTrue(probe.in_region([411, 1249], probe.REGIONS['recent_clear']))
        self.assertFalse(probe.in_region([455, 1293], probe.REGIONS['recent_clear']))
        self.assertFalse(probe.in_region(None, probe.REGIONS['input_focus']))

    def test_observed_singleton_wrapper_is_offline_recoverable_not_a_point_miss(self):
        # Both branches now use the same existing production normalization.
        from agent.infrastructure.dashscope_vision_provider import _extract_json_object
        raw = '[{"coordinate_space":{"kind":"axis_grid","width":1000,"height":1280},"tap_point":[915,1130]}]'
        value = _extract_json_object(raw, reject_duplicate_keys=True, unwrap_singleton_object_array=True)
        point, _ = probe.parse_reply(raw, 'grounding', (720, 1280), True)
        self.assertEqual(point, probe.parse_reply(json.dumps(value), 'grounding', (720, 1280), True)[0])
        self.assertEqual([915, 883], point)
        frame = [round(point[0]*809/1000), round(point[1]*1439/1000)]
        self.assertEqual([740, 1271], frame)
        self.assertTrue(probe.in_region(frame, probe.REGIONS['failed_send']))

    def test_array_normalization_never_selects_among_multiple_answers(self):
        obj = {'coordinate_space': {'kind': 'axis_grid', 'width': 1000, 'height': 1280}, 'tap_point': [915, 1130]}
        for bad in ([], [obj, obj], [[obj]]):
            with self.assertRaises(Exception):
                probe.parse_reply(json.dumps(bad), 'grounding', (720, 1280), True)

    def test_echoed_schema_is_not_a_coordinate_answer(self):
        # This real failure has no actual point; do not mine numeric schema
        # limits such as 1000/1280 and mistake them for a model click.
        schema = probe.grounding_schema(1280)['json_schema']['schema']
        with self.assertRaises(ValueError):
            probe.parse_reply(json.dumps([schema]), 'grounding', (720, 1280), True)

    def test_new_supplement_excludes_prior_case_and_preserves_frozen_wires(self):
        from experiments import probe_grounding_supplement as supplement
        # Build a same-version parent in a temporary directory. Never update a
        # historical campaign's hashes to accommodate current production code.
        with tempfile.TemporaryDirectory() as parent_directory, tempfile.TemporaryDirectory() as directory, \
                patch.object(supplement, 'OUT', Path(directory)), \
                patch.object(probe, 'OUT', Path(directory)), patch.object(probe, 'CASES', probe.CASES), \
                patch.object(probe, 'ORDER', probe.ORDER), \
                patch('agent.infrastructure.dashscope_vision_provider.httpx.post') as post:
            archived_parent = supplement.PARENT
            with patch.object(probe, 'OUT', Path(parent_directory)):
                probe.preflight()
                probe.save('stopped.json', {'calls_used': 2})
                for variant in probe.VARIANTS:
                    name = f'failed_send_{variant}_raw.json'
                    probe.save(name, probe.read(archived_parent / name))
            with patch.object(supplement, 'PARENT', Path(parent_directory)):
                with patch.object(probe, 'production_hashes', return_value={'changed': 'code'}):
                    with self.assertRaisesRegex(AssertionError, 'Production changed'):
                        supplement.prepare()
                self.assertFalse((Path(directory) / 'preflight.json').exists())
                supplement.prepare()
            post.assert_not_called()
            pre = probe.read(probe.OUT / 'preflight.json')
            self.assertEqual(6, pre['max_calls'])
            self.assertEqual(6, len(pre['wire_hashes']))
            self.assertNotIn('failed_send', probe.CASES)
            with self.assertRaises(RuntimeError):
                probe.reserve('failed_send', 'full')
            for name, variant in probe.ORDER:
                probe.reserve(name, variant)
                probe.save(name + '_' + variant + '_result.json', {})
            with self.assertRaises(RuntimeError):
                probe.reserve('successful_send', 'full')

    def test_no_duplicate_out_of_order_or_unresolved_calls(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(probe, 'OUT', Path(directory)):
            with self.assertRaises(RuntimeError):
                probe.reserve('successful_send', 'full')
            probe.reserve('failed_send', 'full')
            with self.assertRaises(RuntimeError):
                probe.reserve('failed_send', 'full')
            with self.assertRaises(RuntimeError):
                probe.reserve('failed_send', 'grounding')
            probe.save('failed_send_full_result.json', {})
            probe.save('stopped.json', {})
            with self.assertRaises(RuntimeError):
                probe.reserve('failed_send', 'grounding')

    def test_budget_cannot_be_reset(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(probe, 'OUT', Path(directory)):
            for name, variant in probe.ORDER:
                probe.reserve(name, variant)
                probe.save(name + '_' + variant + '_result.json', {})
            with self.assertRaises(RuntimeError):
                probe.reserve('failed_send', 'full')


if __name__ == '__main__':
    unittest.main()
