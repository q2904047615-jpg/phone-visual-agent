"""Offline diagnostic isolation: no remote request is performed by these tests."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import diagnose_visual_result as diagnosis


class VisualResultDiagnosisTests(unittest.TestCase):
    def test_independent_questions_share_only_current_images(self):
        case = diagnosis.fixtures()['failed_send']
        facts = diagnosis.build('facts', case)
        result = diagnosis.build('last_action', case)
        self.assertEqual(facts[1]['content'][1:], result[1]['content'][1:])
        self.assertEqual(3, len(facts[1]['content'][1:]))
        self.assertNotIn(case['context']['objective'], facts[1]['content'][0]['text'])
        question = result[1]['content'][0]['text']
        self.assertIn('send_message', question)
        self.assertNotIn('input_verified_text', question)
        self.assertNotIn('visual_outcome', question)
        self.assertNotIn('previous_action_outcome', question)
        self.assertNotIn('expected_input_value', question)

    def test_offline_prepare_never_calls_network_or_consumes_budget(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(diagnosis, 'OUT', Path(directory)), \
                patch('agent.infrastructure.dashscope_vision_provider.httpx.post') as post:
            diagnosis.prepare()
            post.assert_not_called()
            self.assertFalse(list(diagnosis.OUT.glob('*_attempt.json')))
            self.assertEqual(2, diagnosis.read(diagnosis.OUT / 'preflight.json')['max_calls'])

    def test_stop_and_duplicate_are_rejected_before_provider_creation(self):
        for marker in ('stopped.json', 'facts_attempt.json'):
            with tempfile.TemporaryDirectory() as directory, patch.object(diagnosis, 'OUT', Path(directory)), \
                    patch.object(diagnosis, 'DashScopeVisionProvider') as provider:
                diagnosis.save(marker, {})
                with self.assertRaises(AssertionError):
                    diagnosis.recognize('facts')
                provider.assert_not_called()


if __name__ == '__main__':
    unittest.main()
