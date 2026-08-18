from __future__ import annotations

import unittest

from constraint_target_filter import constraint_excludes_candidate


class ConstraintTargetFilterTests(unittest.TestCase):
    def test_blanket_exception_keeps_explicitly_allowed_wechat_entry(self) -> None:
        self.assertFalse(
            constraint_excludes_candidate(
                ("不得执行除使微信首页在前台可见并停在首页之外的任何操作",),
                ("app_launcher_wechat", "微信", "绿色气泡图标"),
                candidate_role="icon",
            )
        )

    def test_blanket_exception_excludes_entry_outside_allowed_scope(self) -> None:
        self.assertTrue(
            constraint_excludes_candidate(
                ("不得执行除使微信首页在前台可见并停在首页之外的任何操作",),
                ("app_launcher_settings", "设置", "齿轮图标"),
                candidate_role="icon",
            )
        )

    def test_english_blanket_exception_keeps_only_named_entry(self) -> None:
        constraint = "Do not perform any action except open Settings"
        self.assertFalse(
            constraint_excludes_candidate(
                (constraint,),
                ("app_launcher_settings", "Settings", "gear icon"),
                candidate_role="icon",
            )
        )
        self.assertTrue(
            constraint_excludes_candidate(
                (constraint,),
                ("app_launcher_gallery", "Gallery", "photo icon"),
                candidate_role="icon",
            )
        )

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

    def test_effect_scoped_operation_keeps_settings_launcher(self) -> None:
        self.assertFalse(
            constraint_excludes_candidate(
                ("不得执行任何可能修改设置的操作",),
                ("app_launcher", "设置", "设置入口"),
                candidate_role="icon",
            )
        )

    def test_effect_scoped_operation_keeps_gallery_launcher_variation(self) -> None:
        self.assertFalse(
            constraint_excludes_candidate(
                ("不得进行任何可能删除相册内容的操作",),
                ("app_launcher", "相册", "相册入口"),
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

    def test_search_result_prohibition_keeps_search_input_with_shared_prefix(self) -> None:
        self.assertFalse(
            constraint_excludes_candidate(
                ("不得提交搜索", "不得选择任何搜索结果"),
                (
                    "application_text_input",
                    "搜索系统设置项",
                    "search icon",
                    "rounded rectangular border",
                ),
                candidate_role="input",
            )
        )

    def test_search_result_prohibition_still_excludes_actual_result(self) -> None:
        self.assertTrue(
            constraint_excludes_candidate(
                ("不得选择任何搜索结果",),
                ("search_result", "Wi-Fi 搜索结果", "搜索结果列表项"),
                candidate_role="list_item",
            )
        )

    def test_english_search_result_prohibition_keeps_search_input(self) -> None:
        self.assertFalse(
            constraint_excludes_candidate(
                ("Do not open any search results",),
                ("search_input", "Search settings", "text input"),
                candidate_role="input",
            )
        )

    def test_allowed_contrast_clause_is_not_treated_as_prohibited(self) -> None:
        constraint = "不要点击广告，但点击确定按钮"
        self.assertTrue(
            constraint_excludes_candidate(
                (constraint,),
                ("advertisement", "广告"),
                candidate_role="button",
            )
        )
        self.assertFalse(
            constraint_excludes_candidate(
                (constraint,),
                ("confirm_action", "确定按钮"),
                candidate_role="button",
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

    def test_explicit_operate_prohibition_excludes_named_entry(self) -> None:
        self.assertTrue(
            constraint_excludes_candidate(
                ("不得操作设置入口",),
                ("app_launcher", "设置", "设置入口"),
                candidate_role="icon",
            )
        )

    def test_direct_target_clause_overrides_effect_scoped_operation(self) -> None:
        self.assertTrue(
            constraint_excludes_candidate(
                ("不得执行可能修改设置的操作，也不得打开设置入口",),
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
