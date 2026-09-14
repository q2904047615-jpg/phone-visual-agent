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

    def test_full_prompt_keeps_fact_sections_and_removes_stability_assumption(self):
        for image_count in (1, 3):
            for include_input in (False, True):
                with self.subTest(images=image_count, input=include_input):
                    prompt = _single_step_observation_prompt(
                        {'objective': '打开当前入口', 'entities': {'history': []}},
                        include_input_structure=include_input, image_count=image_count,
                        request_image_size=(720, 1280), available_action_kinds=('tap_semantic', 'home'))
                    self.assertNotIn('稳定手机画面', prompt)
                    self.assertNotIn('stable image', prompt)
                    self.assertIn('视频、动画或跨帧像素变化本身不是错误', prompt)
                    self.assertIn('SCENE CONTRACT只约束scene字段', prompt)
                    self.assertIn('只允许一个action或finish', prompt)
                    self.assertIn('横坐标和纵坐标都为0..1000', prompt)
                    self.assertIn('旧聊天气泡或相同既有内容都不证明本次', prompt)
                    self.assertIn('系统界面、前景弹窗/抽屉/键盘、当前页面、卡片/缩略图及视频或直播内容', prompt)
                    self.assertIn('常见位置只能作弱提示', prompt)
                    self.assertIn('评论行内的回复控件属于该评论，页面底部独立编辑区才是新评论入口', prompt)
                    self.assertIn('UI惯例只帮助解释CURRENT可见事实', prompt)
                    if include_input:
                        self.assertIn('INPUT CONTRACT applies only to input_structure', prompt)
                        self.assertIn('all supplied CURRENT images from this observation', prompt)
                        self.assertIn('Never infer focus from the requested action', prompt)
                    if image_count == 3:
                        self.assertIn('闪烁光标可从任一帧读取', prompt)

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
