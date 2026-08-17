from __future__ import annotations

import unittest

from constraint_target_filter import constraint_excludes_candidate


class ConstraintTargetFilterTests(unittest.TestCase):
    def test_state_change_constraint_keeps_settings_launcher(self) -> None:
        self.assertFalse(
            constraint_excludes_candidate(
                ("不得改变任何设置值",),
                ("app_launcher", "设置", "齿轮图标，下方文字设置"),
                candidate_role="icon",
            )
        )

    def test_state_result_constraint_keeps_calculator_launcher_variation(self) -> None:
        self.assertFalse(
            constraint_excludes_candidate(
                ("不得保存任何计算记录",),
                ("app_launcher", "计算器", "本地计算工具入口"),
                candidate_role="icon",
            )
        )

    def test_explicit_open_prohibition_excludes_search_result(self) -> None:
        self.assertTrue(
            constraint_excludes_candidate(
                ("不得打开任何搜索结果",),
                ("search_result", "Wi-Fi 搜索结果"),
                candidate_role="list_item",
            )
        )

    def test_explicit_use_prohibition_excludes_named_entry(self) -> None:
        self.assertTrue(
            constraint_excludes_candidate(
                ("不要再次使用设置入口",),
                ("app_launcher", "设置", "设置入口"),
                candidate_role="icon",
            )
        )

    def test_page_element_scope_still_excludes_page_control(self) -> None:
        self.assertTrue(
            constraint_excludes_candidate(
                ("禁止通过任何页面正文链接、按钮或元素跳转",),
                ("open_details", "查看详情"),
                candidate_role="button",
            )
        )


if __name__ == "__main__":
    unittest.main()
