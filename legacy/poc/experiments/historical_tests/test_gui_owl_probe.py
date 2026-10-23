import json
import unittest
from copy import deepcopy
from PIL import Image
from experiments.probe_gui_owl import extract_prompt, request_body, redact_images, parse_action


class GuiOwlProbeTests(unittest.TestCase):
    def test_extract_literal_without_executing_source(self):
        self.assertEqual(extract_prompt("raise RuntimeError()\nSYSTEM_PROMPT='native'"), 'native')

    def test_request_has_only_current_images_and_actual_history(self):
        history = [{'visual_outcome': None, 'after_scene': '', 'body': 'aaazjie？你好'}]
        case = {'context': {'objective': '发送', 'entities': {'history': history}},
                'frames': [Image.new('RGB', (40, 80), color) for color in ('red', 'green', 'blue', 'white')],
                'expected_status': 'SECRET_EXPECTED_RESULT'}
        before = deepcopy(case['context'])
        request = request_body(case, 'native')
        self.assertEqual(case['context'], before)
        self.assertEqual(len(request['messages'][1]['content']), 4)
        self.assertNotIn('SECRET_EXPECTED_RESULT', json.dumps(request))
        self.assertIn('aaazjie？你好', request['messages'][1]['content'][0]['text'])
        self.assertNotIn('data:image', json.dumps(redact_images(request)))
        self.assertNotIn('response_format', request)

    def test_latest_future_result_is_rejected(self):
        case = {'context': {'objective': '发送', 'entities': {'history': [
            {'visual_outcome': 'matched', 'after_scene': 'success'}]}}, 'frames': []}
        with self.assertRaises(AssertionError):
            request_body(case, 'native')

    def test_native_finish_and_home(self):
        for args in ({'action': 'terminate', 'status': 'success'},
                     {'action': 'system_button', 'button': 'Home'}):
            text = '<tool_call>' + json.dumps({'name': 'mobile_use', 'arguments': args}) + '</tool_call>'
            self.assertEqual(parse_action(text)['arguments'], args)

    def test_multiple_actions_are_not_executed_or_repaired(self):
        text = '<tool_call>{"name":"mobile_use","arguments":{"action":"wait"}}</tool_call>'
        with self.assertRaises(ValueError):
            parse_action(text + text)

    def test_invalid_point_rejected(self):
        with self.assertRaises(ValueError):
            parse_action('<tool_call>{"name":"mobile_use","arguments":'
                         '{"action":"click","coordinate":[true,12]}}</tool_call>')


if __name__ == '__main__':
    unittest.main()
