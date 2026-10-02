"""Runner budget and crop-review tests; no model calls."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from experiments import probe_pixel_zoom as runner


class PixelZoomRunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)
        self.patch = patch.object(runner, 'OUT', self.out)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.pre = {'frozen': runner.frozen(), 'production_hashes': runner.base.hashes()}

    def test_initial_budget_order_and_drift(self):
        runner.guard(1, self.pre)
        for index in (0,2,5):
            with self.assertRaises(RuntimeError):
                runner.guard(index, self.pre)
        with self.assertRaises(RuntimeError):
            runner.guard(1, {'frozen': {}, 'production_hashes': {}})

    def test_roi_requires_visual_review(self):
        (self.out/'1_attempt.json').touch()
        runner.save('1_result.json', {'ok':True,'stage':'roi','png_sha256':'fixture'})
        with self.assertRaises(FileNotFoundError):
            runner.guard(2,self.pre)
        runner.review(1,True)
        runner.guard(2,self.pre)
        with self.assertRaises(RuntimeError):
            runner.review(1,True)

    def test_missing_target_stops_without_second_call(self):
        (self.out/'1_attempt.json').touch()
        runner.save('1_result.json', {'ok':True,'stage':'roi','png_sha256':'fixture'})
        runner.review(1,False)
        with self.assertRaises(RuntimeError):
            runner.guard(2,self.pre)
        self.assertEqual(runner.base.read(self.out/'stopped.json')['reserved_calls'],1)

    def test_unresolved_attempt_not_retried(self):
        (self.out/'1_attempt.json').touch()
        for index in (1,2):
            with self.assertRaises(RuntimeError):
                runner.guard(index,self.pre)


if __name__ == '__main__':
    unittest.main()
