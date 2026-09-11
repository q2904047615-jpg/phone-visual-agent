import tempfile
import json
import unittest
from pathlib import Path
from unittest.mock import patch
from experiments import probe_spatial_targets as probe


class SpatialTargetsTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.out = Path(temp.name)
        p = patch.object(probe, 'OUT', self.out)
        p.start()
        self.addCleanup(p.stop)

    def test_only_target_changes_across_same_frames(self):
        provider = probe.base.DashScopeVisionProvider(api_key='offline-dummy', enable_thinking=True)
        bodies = []
        for name in probe.ORDER:
            body,size,required,case = probe.build(provider,name)
            bodies.append(probe.neutral(body,name))
            self.assertEqual(size,(720,1280))
            self.assertTrue(required)
            self.assertEqual(case['context']['entities']['history'],[])
            self.assertEqual(case['allowed'],['tap_semantic'])
            safe_wire = json.dumps(probe.base.redacted_wire(body), ensure_ascii=False)
            self.assertNotIn('reference_center',safe_wire)
            self.assertNotIn(json.dumps(probe.REGIONS[name]),safe_wire)
            self.assertNotIn(json.dumps(probe.REGIONS[name],separators=(',',':')),safe_wire)
            self.assertFalse('regions' in body, 'Offline references must not be request fields')
        self.assertTrue(all(x == bodies[0] for x in bodies))

    def test_freeze_is_four_calls_not_reusable(self):
        probe.prepare()
        self.assertEqual(probe.base.read(self.out/'preflight.json')['max_calls'],4)
        with self.assertRaises(RuntimeError):
            probe.prepare()

    def test_order_duplicate_and_unresolved_block(self):
        with self.assertRaises(RuntimeError):
            probe.reserve('send')
        probe.reserve('upper_avatar')
        with self.assertRaises(RuntimeError):
            probe.reserve('upper_avatar')
        with self.assertRaises(RuntimeError):
            probe.reserve('middle_avatar')

    def test_miss_does_not_destroy_spatial_comparison(self):
        probe.reserve('upper_avatar')
        probe.save('upper_avatar_result.json',{'transport_succeeded':True,'parse_succeeded':True,'inside_visible_region':False})
        probe.reserve('middle_avatar')
        self.assertTrue((self.out/'middle_avatar_attempt.json').exists())

    def test_stop_prevents_continuation(self):
        probe.save('stopped.json',{'resume_allowed':False})
        with self.assertRaises(RuntimeError):
            probe.reserve('upper_avatar')


if __name__ == '__main__':
    unittest.main()
