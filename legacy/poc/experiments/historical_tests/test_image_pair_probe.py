from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from experiments import probe_image_pair as probe


class ImagePairTests(unittest.TestCase):
    def test_only_images_change_and_no_historical_task_leaks(self):
        provider = probe.base.DashScopeVisionProvider(api_key='offline-dummy', enable_thinking=True)
        old, size, required, a = probe.build(provider, 'old')
        current, size2, required2, b = probe.build(provider, 'current')
        self.assertEqual(probe.without_images(old), probe.without_images(current))
        self.assertNotEqual(old, current)
        self.assertEqual((size, required), (size2, required2))
        self.assertEqual(a['context'], b['context'])
        self.assertEqual(a['context']['entities']['history'], [])
        self.assertEqual(a['allowed'], ['tap_semantic'])
        self.assertNotIn('蓝鲸', a['context']['objective'])
        self.assertNotIn('aaazjie', a['context']['objective'])
        self.assertEqual(sum(p['type'] == 'image_url' for p in old['messages'][1]['content']), 3)

    def test_freeze_cannot_reset_and_uses_separate_budget(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(probe, 'OUT', Path(directory)):
            probe.prepare()
            pre = probe.base.read(Path(directory) / 'preflight.json')
            self.assertEqual(pre['max_calls'], 2)
            self.assertEqual(pre['history_count'], 0)
            with self.assertRaises(RuntimeError):
                probe.prepare()

    def test_driver_override_is_scoped_and_restored(self):
        original = probe.driver.OUT, probe.driver.build, probe.driver.hashes
        def check(variant):
            self.assertEqual(variant, 'old')
            self.assertEqual(probe.driver.OUT, probe.OUT)
            self.assertIs(probe.driver.build, probe.build)
            self.assertIs(probe.driver.hashes, probe.hashes)
        with patch.object(probe.driver, 'recognize', side_effect=check):
            probe.recognize('old')
        self.assertEqual((probe.driver.OUT, probe.driver.build, probe.driver.hashes), original)

    def test_unknown_variant_has_no_execution(self):
        with patch.object(probe.driver, 'recognize') as call:
            with self.assertRaises(ValueError):
                probe.recognize('extra')
            call.assert_not_called()


if __name__ == '__main__':
    unittest.main()
