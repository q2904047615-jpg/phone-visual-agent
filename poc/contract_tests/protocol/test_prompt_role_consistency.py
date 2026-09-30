"""Prompt composition regression only; no network or device calls."""
import unittest
from unittest.mock import patch
from agent.infrastructure.generic_scene_observer import (
    SingleStepGenericSceneObserver, _json_only_system_message,
    _single_step_observation_prompt,
)
from test_support.generic_action_adapter import (
    RawSceneProvider,
)
from contract_tests.observation.test_point_scene_projection import wire, decision
from test_qwen_visual_decision import patterned_frames


class PromptRoleConsistencyTests(unittest.TestCase):
    def test_system_role_allows_the_existing_single_action_or_finish(self):
        message = _json_only_system_message()
        self.assertEqual('system', message['role'])
        self.assertIn('通用手机视觉操作Agent', message['content'])
        self.assertIn('一个canonical动作或整任务finish', message['content'])
        self.assertIn('JSON', message['content'])
        self.assertNotIn('只读页面观察器', message['content'])
        self.assertIn('福袋模块', message['content'])
        self.assertIn('没有福袋时等待60秒', message['content'])
        self.assertIn('左滑、右滑、上滑、下滑按钮', message['content'])
        self.assertIn('不创建 Qwen 任务', message['content'])

    def test_empty_history_does_not_block_current_state_verification(self):
        prompt = _single_step_observation_prompt(
            {'objective': '检查当前状态', 'entities': {'history': []}},
            include_input_structure=False, image_count=1,
            request_image_size=(720, 1280), available_action_kinds=('wait_for_change',))
        self.assertIn('CURRENT独立满足该条件即可finish', prompt)
        self.assertIn('不得仅因history为空改为等待', prompt)
        self.assertIn('该组事实就是充分条件', prompt)
        self.assertIn('不得另加历史归因、身份资料或额外稳定等待条件', prompt)


    def test_actual_request_image_labels_and_action_semantics(self):
        provider = RawSceneProvider([wire(decision('home'))])
        observer = SingleStepGenericSceneObserver(provider)
        frames = [frame.resize((240, 480)) for frame in patterned_frames()]
        with patch.object(provider, '_chat', wraps=provider._chat) as chat:
            _, chosen = observer.observe_with_decision(frames=frames,
                goal_context={'objective': '回到主屏幕'}, device_id='device-1',
                available_action_kinds=('home',))
        messages = chat.call_args.args[0]
        text_parts = [part['text'] for part in messages[1]['content'] if part['type'] == 'text']
        self.assertTrue(any('CURRENT PHONE SURFACE' in text for text in text_parts))
        self.assertFalse(any('STABLE PHONE SURFACE' in text for text in text_parts))
        self.assertEqual('home', chosen['action'])
        self.assertEqual(1, provider.calls)


if __name__ == '__main__':
    unittest.main()
