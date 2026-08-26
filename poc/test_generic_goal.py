from __future__ import annotations

import unittest

from agent.domain.generic_goal import (
    GenericIntentDraft,
    GenericIntentError,
    _parse_json_object,
)


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

    def test_json_object_parser_accepts_a_fenced_object(self) -> None:
        self.assertEqual(
            {"objective": "点击并输入"},
            _parse_json_object('```json\n{"objective":"点击并输入"}\n```'),
        )

    def test_json_object_parser_rejects_invalid_or_non_object_roots(self) -> None:
        with self.assertRaisesRegex(GenericIntentError, "没有返回有效 JSON"):
            _parse_json_object("not-json")
        with self.assertRaisesRegex(GenericIntentError, "不是 JSON 对象"):
            _parse_json_object("[]")

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
