import unittest

from message_intent import (
    subgoal_binds_recipient,
    subgoal_targets_recipient_control,
)


class RecipientBindingTests(unittest.TestCase):
    def test_recipient_becomes_exact_only_for_bound_subgoal(self):
        self.assertFalse(subgoal_binds_recipient("张三", "聊天应用在前台可见"))
        self.assertTrue(subgoal_binds_recipient("张三", "张三的聊天页面可见"))
        self.assertTrue(
            subgoal_binds_recipient(
                "Alice",
                {"completion_conditions": ["当前聊天标题逐字显示 Alice"]},
            )
        )

    def test_recipient_selection_and_page_identity_are_distinct(self):
        self.assertTrue(subgoal_targets_recipient_control("张三", "选择张三"))
        self.assertFalse(
            subgoal_targets_recipient_control(
                "张三",
                "张三的页面已打开，在输入框中编辑正文",
            )
        )


if __name__ == "__main__":
    unittest.main()
