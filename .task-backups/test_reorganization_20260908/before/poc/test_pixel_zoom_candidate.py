"""Offline candidate tests do not prove actual Qwen coordinate adherence."""
import base64
import io
import json
import unittest

from PIL import Image
from experiments import pixel_zoom_candidate as candidate


class PixelZoomTests(unittest.TestCase):
    def encode(self, image):
        stream = io.BytesIO()
        image.save(stream, format='PNG')
        return 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode('ascii')

    def test_dimensions_come_from_encoded_image(self):
        for dims in ((720,1280), (1280,720), (192,128)):
            url = self.encode(Image.new('RGB', dims))
            result = candidate.build_request(url, '通用目标', region=True)
            content = result['messages'][0]['content']
            self.assertEqual(content[0]['image_url']['url'], url)
            self.assertIn(f'{dims[0]}×{dims[1]}', content[1]['text'])
            self.assertNotIn('0..999', content[1]['text'])

    def test_prior_failed_array_not_reinterpreted(self):
        with self.assertRaises(ValueError):
            candidate.parse('[97,1093,153,1153]', 720,1280, region=True)

    def test_explicit_wrong_unit_rejected(self):
        for unit in ('normalized', 'axis_grid', 'pixels', None):
            with self.assertRaises(ValueError):
                candidate.parse(json.dumps({'coordinate_space':unit,'bbox':[1,2,3,4]}),720,1280,region=True)

    def test_null_not_invented(self):
        self.assertIsNone(candidate.parse('null',720,1280,region=True))
        self.assertIsNone(candidate.parse('null',720,1280,region=False))

    def test_pixel_roi_valid_for_each_orientation(self):
        for w,h in ((720,1280),(1280,720),(20,10)):
            box=[0,0,w-1,h-1]
            raw=json.dumps({'coordinate_space':'image_pixels','bbox':box})
            self.assertEqual(candidate.parse(raw,w,h,region=True),box)

    def test_invalid_coordinates(self):
        for box in ([0,0,720,1279], [0,0,719,1280], [2,3,1,4], [1,1,1,2],
                    [True,0,1,2], [0,0,float('nan'),2], [0,0,float('inf'),2], [-1,0,1,2]):
            with self.subTest(box=box), self.assertRaises(ValueError):
                candidate.parse(json.dumps({'coordinate_space':'image_pixels','bbox':box}),720,1280,region=True)

    def test_exact_crop_and_inverse(self):
        image=Image.new('RGB',(200,300))
        zoom,geometry=candidate.crop_and_enlarge(image,[20,50,29,69])
        self.assertEqual(zoom.size,(20,40))
        raw=json.dumps({'coordinate_space':'image_pixels','point':[6.5,16.5]})
        self.assertEqual(candidate.to_original(raw,geometry),[23,58])

    def test_crop_rejects_outside_no_padding(self):
        with self.assertRaises(ValueError):
            candidate.crop_and_enlarge(Image.new('RGB',(100,200)),[0,0,100,200])

    def test_point_uses_zoom_dimensions(self):
        geometry={'bounds_exclusive':[20,50,30,70],'zoom_size':[20,40],'scale':2}
        with self.assertRaises(ValueError):
            candidate.to_original('{"coordinate_space":"image_pixels","point":[500,500]}',geometry)

    def test_correct_unit_label_does_not_prove_correct_grounding(self):
        # In-range wrong semantics cannot be detected by numerical validation.
        raw='{"coordinate_space":"image_pixels","point":[1,1]}'
        self.assertEqual(candidate.parse(raw,720,1280,region=False),[1,1])


if __name__ == '__main__':
    unittest.main()
