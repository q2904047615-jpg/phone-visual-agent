from __future__ import annotations

import unittest

from generic_goal import GenericIntentDraft


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


if __name__ == "__main__":
    unittest.main()
