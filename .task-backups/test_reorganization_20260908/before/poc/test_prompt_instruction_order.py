"""Instruction-order prompt transport, not a simulated model-success test."""
import unittest
from unittest.mock import patch

from agent.infrastructure.generic_scene_observer import (
    SingleStepGenericSceneObserver, _single_step_observation_prompt,
)
from test_generic_action_adapter import RawSceneProvider
from test_point_scene_projection import wire, decision
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
                    '不生成固定步骤清单', '本地不按业务子目标清单推进'):
                    self.assertTrue(text in prompt, msg=text)

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


if __name__ == '__main__':
    unittest.main()
