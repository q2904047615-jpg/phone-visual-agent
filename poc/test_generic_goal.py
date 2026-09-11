from __future__ import annotations
import unittest
from agent.domain.generic_goal import (
    GenericIntentDraft,
    GenericIntentError,
    safe_goal_context,
)
from agent.domain.vision_model import VisionAgentError


class GenericGoalProjectionTests(unittest.TestCase):
    def test_low_level_words_are_data_not_a_second_action_veto(self) -> None:
        draft = GenericIntentDraft(
            understood=True,
            app_id="example",
            app_name="示例",
            objective="点击按钮后输入内容并滑动",
            entities={
                "action": "用户原文中的动作词",
                "coordinates": "用户描述中的普通字段",
            },
            success_criteria={"result": "输入完成"},
        )
        self.assertEqual("点击按钮后输入内容并滑动", draft.to_dict()["objective"])

    def test_projection_does_not_force_confirmation_for_ordinary_work(self) -> None:
        draft = GenericIntentDraft(
            understood=True,
            app_id="example",
            app_name="示例",
            objective="发送普通消息",
            needs_confirmation=False,
        )
        self.assertFalse(draft.to_dict()["needs_confirmation"])

    def test_current_context_preserves_history_without_aliasing(self) -> None:
        context = {"task": "打开应用", "history": [{"text": "你好"}]}
        projected = safe_goal_context(context)
        self.assertEqual(context, projected)
        projected["history"][0]["text"] = "changed"
        self.assertEqual("你好", context["history"][0]["text"])

    def test_current_context_keeps_json_normalization_and_validation(self) -> None:
        self.assertEqual(
            {"items": ["输入", None, False]},
            safe_goal_context({"items": ("输入", None, False)}),
        )
        with self.assertRaises(VisionAgentError):
            safe_goal_context([])
        with self.assertRaises(GenericIntentError):
            safe_goal_context({"value": {"not-json"}})

    def test_projection_rejects_blank_keys_and_unsupported_values(self) -> None:
        with self.assertRaisesRegex(GenericIntentError, "目标参数字段无效"):
            GenericIntentDraft(
                understood=True,
                app_id="example",
                app_name="示例",
                objective="查看结果",
                entities={"": "value"},
            ).to_dict()
        with self.assertRaisesRegex(GenericIntentError, "目标参数类型不受支持"):
            GenericIntentDraft(
                understood=True,
                app_id="example",
                app_name="示例",
                objective="查看结果",
                entities={"value": {"unsupported"}},
            ).to_dict()


if __name__ == "__main__":
    unittest.main()
