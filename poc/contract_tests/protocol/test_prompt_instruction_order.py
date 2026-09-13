"""Instruction-order prompt transport, not a simulated model-success test."""
import unittest
from unittest.mock import patch
from agent.infrastructure.generic_scene_observer import (
    SingleStepGenericSceneObserver, _single_step_observation_prompt,
)
from test_support.generic_action_adapter import (
    RawSceneProvider,
)
from contract_tests.observation.test_point_scene_projection import wire, decision
from test_qwen_visual_decision import patterned_frames


class PromptInstructionOrderTests(unittest.TestCase):
    def test_order_guidance_preserves_existing_state_and_effect_contracts(self):
        for count in (1, 3):
            with self.subTest(count=count):
                prompt = _single_step_observation_prompt({}, include_input_structure=True,
                    image_count=count, request_image_size=(720, 1280),
                    available_action_kinds=('home', 'tap_semantic'))
                for text in ('严格遵守用户明确指定的顺序、对象、数量和正文',
                    '不得因某个动作更方便而跳过、重排或擅自替换用户要求',
                    '用户未指定顺序时，才自行选择执行路径',
                    '依据本会话实际历史与当前画面判断前项是否满足，再推进后项',
                    '不能反过来要求所有任务必须有动作',
                    '输入、发送等效果不确定时不得自动重做',
                    '不生成固定步骤清单', '本地不按业务子目标清单推进',
                    '通用文字效果闭环', '输入动作后必须以新画面确认正文匹配',
                    '效果动作后再次观察'):
                    self.assertTrue(text in prompt, msg=text)

    def test_input_append_requires_complete_expected_text(self):
        prompt = _single_step_observation_prompt({}, include_input_structure=True,
            image_count=1, request_image_size=(720, 1280),
            available_action_kinds=('input_verified_text',))
        self.assertIn('text仍必须填写追加后的完整正文', prompt)
        self.assertIn('本地会从完整text计算唯一input_fragment', prompt)
    def test_original_goals_and_guidance_reach_actual_request_without_local_planning(self):
        goals = ('先清空卡片，再打开抖音，给十条视频点赞',
            '先打开备忘录，再清空输入框，最后回到主屏幕',
            '打开设置查看当前页面', '给文件传输助手发送aaazjie？你好')
        for goal in goals:
            with self.subTest(goal=goal):
                provider = RawSceneProvider([wire(decision('home'))])
                observer = SingleStepGenericSceneObserver(provider)
                frames = [frame.resize((240, 480)) for frame in patterned_frames()]
                with patch.object(provider, '_chat', wraps=provider._chat) as chat:
                    _, chosen = observer.observe_with_decision(frames=frames,
                        goal_context={'objective': goal, 'entities': {'history': []}},
                        device_id='device-1', available_action_kinds=('home',))
                prompt = chat.call_args.args[0][1]['content'][0]['text']
                self.assertTrue(goal in prompt)
                self.assertTrue('严格遵守用户明确指定的顺序、对象、数量和正文' in prompt)
                self.assertEqual(1, provider.calls)
                # Same legal response still passes: no local task-order veto was added.
                self.assertEqual('home', chosen['action'])


    def test_publish_capability_is_scoped_to_supported_surfaces(self):
        prompt = _single_step_observation_prompt({}, include_input_structure=True,
            image_count=1, request_image_size=(720, 1280),
            available_action_kinds=('tap_semantic',))
        self.assertIn('publish_content仅在当前前台画面明确属于小红书', prompt)
        self.assertIn('微信（com.tencent.mm）及其他应用不选择publish_content', prompt)
        self.assertIn('不改变通用输入、观察、验证闭环', prompt)
if __name__ == '__main__':
    unittest.main()
