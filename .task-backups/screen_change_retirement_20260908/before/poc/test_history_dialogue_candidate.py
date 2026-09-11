"""Request composition tests only; these do not establish model accuracy."""
from copy import deepcopy
import json
import unittest
from unittest.mock import Mock, patch

from experiments.history_dialogue_candidate import DialogueCandidateObserver, reframe_as_dialogue
from agent.infrastructure.generic_scene_observer import SingleStepGenericSceneObserver
import test_action_comparison_context as history_fixtures
from test_generic_scene_observer import SequenceProvider, stable_frames
from test_point_scene_projection import wire, decision


class FactualDialogueCandidateTests(unittest.TestCase):
    def context(self, history):
        return {'objective': '给联系人甲发送原文 aaazjie？你好\n  保留空格 ',
            'entities': {'task_id': 'task-current', 'history': history,
                'launch_app_aliases': ['笔记', '聊天'], 'exact_input_text': 'aaazjie？你好\n  保留空格 '}}

    def history(self, meaning='open_chat_session', step=1):
        return history_fixtures.ActionComparisonContextTests().history(meaning, step)

    def observe(self, cls, context, payload=None):
        reply = payload or wire(decision('home'))
        reply['coordinate_space'] = {'kind': 'normalized_1000', 'width': 1000, 'height': 1000}
        provider = SequenceProvider([reply])
        observer = cls(provider)
        observer.observe_with_decision(frames=stable_frames(), goal_context=context, device_id='device-1')
        return provider, observer

    def test_initial_task_stays_two_messages_with_exact_body(self):
        ctx = self.context([])
        p, _ = self.observe(DialogueCandidateObserver, ctx)
        self.assertEqual(['system', 'user'], [x['role'] for x in p.messages_seen[0]])
        text = p.messages_seen[0][1]['content'][0]['text']
        first = json.loads(text.split('：', 1)[1])
        self.assertEqual(ctx['objective'], first['objective'])
        self.assertEqual(ctx['entities']['exact_input_text'], first['entities']['exact_input_text'])
        self.assertNotIn('history', first['entities'])
        self.assertEqual([], ctx['entities']['history'])

    def test_factual_actions_and_results_are_lossless_ordered_turns(self):
        entries = [self.history('open_chat_session', 1), self.history('send_message', 2)]
        entries[1]['canonical_action']['params']['text'] = '中文？\n  e\u0301 '
        entries[1]['extra_diagnostic'] = {'value': None}
        ctx = self.context(entries)
        original = deepcopy(ctx)
        p, _ = self.observe(DialogueCandidateObserver, ctx)
        messages = p.messages_seen[0]
        self.assertEqual(['system', 'user', 'assistant', 'user', 'assistant', 'user'],
            [x['role'] for x in messages])
        for index, expected in enumerate(entries):
            actual = json.loads(messages[2 + 2 * index]['content'])['已执行动作']
            actual.update(json.loads(messages[3 + 2 * index]['content'][0]['text'].split('：', 1)[1]))
            self.assertEqual(expected, actual)
        self.assertEqual(original, ctx)
        self.assertNotIn('已执行动作', messages[-1]['content'][1]['text'])

    def test_current_images_prompt_rules_and_model_output_unchanged(self):
        ctx = self.context([self.history()])
        for choice in [decision('home'), decision(None, status='finish', reason='当前字段已空')]:
            with self.subTest(choice=choice['status']):
                payload = wire(choice)
                baseline, _ = self.observe(SingleStepGenericSceneObserver, ctx, deepcopy(payload))
                candidate, observer = self.observe(DialogueCandidateObserver, ctx, deepcopy(payload))
                old = baseline.messages_seen[0]
                new = candidate.messages_seen[0]
                self.assertEqual(old[0], new[0])
                images = lambda messages: [part for m in messages if isinstance(m['content'], list)
                    for part in m['content'] if part['type'] == 'image_url']
                self.assertEqual(images(old), images(new))
                self.assertEqual(3, len(images(new)))
                self.assertEqual(1, candidate.calls)
                self.assertEqual(choice['status'], observer.last_diagnostics['decision_status'])
                self.assertEqual(0, len(images(new[:-1])))
                self.assertIn('旧聊天气泡或相同既有内容都不证明', new[-1]['content'][1]['text'])

    def test_reused_observer_does_not_keep_previous_task_or_images(self):
        reply = wire(decision('home'))
        reply['coordinate_space'] = {'kind': 'normalized_1000', 'width': 1000, 'height': 1000}
        p = SequenceProvider([deepcopy(reply), deepcopy(reply)])
        observer = DialogueCandidateObserver(p)
        observer.observe_with_decision(frames=stable_frames((200, 30, 30)),
            goal_context=self.context([self.history('old_task_only')]))
        observer.observe_with_decision(frames=stable_frames((30, 30, 200)),
            goal_context={'objective': '新任务', 'entities': {'task_id': 'new', 'history': []}})
        self.assertEqual(2, len(p.messages_seen[1]))
        self.assertNotIn('old_task_only', str(p.messages_seen[1]))

    def test_wait_zero_actions_and_unknown_result_are_not_dropped_or_completed(self):
        entry = self.history()
        entry.update(physical_actions=0, visual_outcome=None, after_scene='')
        p, _ = self.observe(DialogueCandidateObserver, self.context([entry]))
        result = json.loads(p.messages_seen[0][-1]['content'][0]['text'].split('：', 1)[1])
        self.assertEqual(0, result['physical_actions'])
        self.assertIsNone(result['visual_outcome'])
        self.assertEqual('', result['after_scene'])

    def test_wire_shape_drift_stops_experiment_before_provider(self):
        with self.assertRaises(ValueError):
            reframe_as_dialogue([{'role': 'user', 'content': []}])

    def test_actual_http_wire_changes_only_messages_not_settings_or_schema(self):
        from agent.infrastructure.dashscope_vision_provider import DashScopeVisionProvider, _image_request_size
        from test_generic_scene_observer import current_axis_grid_payload
        frames = stable_frames()
        payload = wire(decision('home'))
        payload['coordinate_space'] = {'kind': 'normalized_1000', 'width': 1000, 'height': 1000}
        payload = current_axis_grid_payload(payload, request_height=_image_request_size(frames[-1])[1])
        response = Mock()
        response.json.return_value = {'choices': [{'message': {'content': json.dumps(payload)},
            'finish_reason': 'stop'}], 'model': 'offline', 'usage': {}}
        bodies = []
        for observer_type in [SingleStepGenericSceneObserver, DialogueCandidateObserver]:
            with patch('agent.infrastructure.dashscope_vision_provider.httpx.post', return_value=response) as post:
                observer_type(DashScopeVisionProvider(api_key='offline-dummy')).observe_with_decision(
                    frames=frames, goal_context=self.context([self.history()]))
                self.assertEqual(1, post.call_count)
                body = deepcopy(post.call_args.kwargs['json'])
                body.pop('messages')
                bodies.append(body)
        self.assertEqual(bodies[0], bodies[1])

    def test_provider_failure_does_not_trigger_another_call_or_change_finish(self):
        provider = Mock()
        provider.status.return_value = {'model': 'offline'}
        provider._chat.side_effect = RuntimeError('isolated connection failure')
        provider.call_scope.side_effect = lambda **_: __import__('contextlib').nullcontext()
        observer = DialogueCandidateObserver(provider)
        with self.assertRaisesRegex(RuntimeError, 'isolated connection failure'):
            observer.observe_with_decision(frames=stable_frames(), goal_context=self.context([self.history()]))
        self.assertEqual(1, provider._chat.call_count)
        self.assertEqual(1, observer.last_diagnostics['model_calls'])


if __name__ == '__main__':
    unittest.main()
