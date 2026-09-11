"""Offline tests only. Never calls a model or a device."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from experiments import probe_zoom_grounding as probe


class ZoomProbeTests(unittest.TestCase):
    def test_roi_parsing(self):
        self.assertEqual(probe.parse_roi("```json\n[0,1,998,999]\n```"), [0, 1, 998, 999])
        self.assertIsNone(probe.parse_roi("null"))

    def test_invalid_roi(self):
        for raw in ('[true,0,2,3]', '[0,0,1000,999]', '[2,2,1,1]', '[0,0,0,1]',
                    '[0,0,NaN,3]', '[0,1]', '{"box":[0,0,1,1]}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                probe.parse_roi(raw)

    def test_actual_mixed_unit_response_is_not_guessed_or_clamped(self):
        with self.assertRaises(ValueError):
            probe.parse_roi('[97, 1093, 153, 1153]')

    def test_exact_source_identity(self):
        url, im, sha = probe.source_image()
        self.assertEqual(im.size, (720, 1280))
        self.assertEqual(sha, '453cd9b3754b9669867f615a4bff90dc4b9988979199046a53a3dc0bb5f7d324')
        self.assertTrue(url.startswith('data:image/jpeg;base64,'))

    def test_full_image_crop_and_resize(self):
        data, g = probe.make_crop(Image.new('RGB', (10, 20), 'red'), [0, 0, 999, 999])
        self.assertTrue(data.startswith(b'\x89PNG'))
        self.assertEqual(g['crop_box_exclusive'], [0, 0, 10, 20])
        self.assertEqual(g['zoom_size'], [20, 40])
        self.assertEqual(probe.to_original([499.5, 499.5], g), [4.5, 9.5])

    def test_crop_inverse_pixel_centers(self):
        g = {'crop_box_exclusive': [20, 50, 30, 70], 'zoom_size': [20, 40], 'scale': 2}
        self.assertEqual(probe.to_original([499.5, 499.5], g), [24.5, 59.5])
        point = [(2 * 3 + .5) / 19 * 999, (2 * 8 + .5) / 39 * 999]
        xy = probe.to_original(point, g)
        self.assertAlmostEqual(xy[0], 23)
        self.assertAlmostEqual(xy[1], 58)

    def test_no_reference_regions_in_request(self):
        body = probe.request('data:image/png;base64,AA==', probe.POINT_PROMPT.format(w=100, h=100, target='目标'))
        self.assertEqual(len(body['messages']), 1)
        self.assertNotIn('response_format', body)
        self.assertNotIn('regions', json.dumps(body))
        self.assertTrue(body['enable_thinking'])

    def test_original_coordinate_scoring(self):
        for target, xy in (('voice', [95,1116]), ('more', [619,16]), ('send', [616,1117]), ('emoji',[533,1116])):
            self.assertTrue(probe.score(xy, target)['inside_visible_region'])
            self.assertFalse(probe.score([0, 0], target)['inside_visible_region'])

    def test_round_icon_edge_diagnostic(self):
        result = probe.score([512, 1093], 'emoji')
        self.assertTrue(result['inside_visible_region'])
        self.assertFalse(result['inside_reference_ellipse_diagnostic'])

    def test_guard_budget_replay_and_stop(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(probe, 'OUT', Path(tmp)):
            pre = {'frozen_hashes': probe.frozen_hashes(), 'production_hashes': probe.base.hashes()}
            probe.guard(1, pre)
            for idx in (0, 2, 9):
                with self.assertRaises(RuntimeError):
                    probe.guard(idx, pre)
            (Path(tmp)/'1_attempt.json').touch()
            with self.assertRaises(RuntimeError):
                probe.guard(1, pre)
            with self.assertRaises(RuntimeError):
                probe.guard(2, pre)
            (Path(tmp)/'1_result.json').touch()
            probe.guard(2, pre)
            probe.stop('test')
            with self.assertRaises(RuntimeError):
                probe.guard(2, pre)

    def test_guard_drift(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(probe, 'OUT', Path(tmp)):
            with self.assertRaises(RuntimeError):
                probe.guard(1, {'frozen_hashes': {}, 'production_hashes': {}})


if __name__ == '__main__':
    unittest.main()
