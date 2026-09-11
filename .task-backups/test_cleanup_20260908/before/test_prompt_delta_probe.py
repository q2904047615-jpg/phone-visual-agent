from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from experiments import probe_prompt_delta as probe


class PromptDeltaTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.out = Path(temp.name)
        patcher = patch.object(probe, 'OUT', self.out)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_only_outer_prompt_changes_and_template_override_is_restored(self):
        provider = probe.base.DashScopeVisionProvider(api_key='offline-dummy', enable_thinking=True)
        original = probe.observer._prompt_template
        old, _, _, case = probe.build(provider, 'old')
        new, _, _, _ = probe.build(provider, 'current')
        self.assertIs(probe.observer._prompt_template, original)
        a,b = deepcopy(old),deepcopy(new)
        x=a['messages'][1]['content'][0].pop('text')
        y=b['messages'][1]['content'][0].pop('text')
        self.assertEqual(a,b)
        self.assertEqual(len(y)-len(x),190)
        self.assertEqual(len(case['context']['entities']['history']),2)
        self.assertNotIn('唯一例外：你选择 open_recent_apps',x)
        self.assertIn('唯一例外：你选择 open_recent_apps',y)

    def test_preflight_and_no_reset(self):
        probe.prepare()
        self.assertEqual(probe.base.read(self.out/'preflight.json')['max_calls'],2)
        self.assertTrue((self.out/'prompt.diff').exists())
        with self.assertRaises(RuntimeError):
            probe.prepare()

    def test_duplicate_and_unresolved_calls_rejected(self):
        probe.reserve('old')
        with self.assertRaises(FileExistsError):
            probe.reserve('old')
        with self.assertRaises(RuntimeError):
            probe.reserve('current')

    def test_malformed_content_does_not_prevent_prespecified_comparison(self):
        probe.save('old_result.json',{'transport_succeeded':True,'parse_succeeded':False})
        probe.reserve('current')
        self.assertTrue((self.out/'current_attempt.json').exists())

    def test_failed_or_stopped_campaign_cannot_continue(self):
        probe.save('old_result.json',{'transport_succeeded':False})
        with self.assertRaises(RuntimeError):
            probe.reserve('current')
        probe.save('stopped.json',{'resume_allowed':False})
        with self.assertRaises(RuntimeError):
            probe.reserve('old')


if __name__ == '__main__':
    unittest.main()
