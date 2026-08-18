import unittest

from verified_text_transaction import (
    VerifiedTextTransactionError,
    local_pinyin,
    plan_next_verified_input,
)


class VerifiedTextTransactionTests(unittest.TestCase):
    def test_ascii_progress_uses_exact_remaining_segment(self) -> None:
        step = plan_next_verified_input("meeting at eight", "meeting ")
        self.assertEqual("at", step.segment)
        self.assertEqual("direct_latin", step.kind)
        self.assertEqual("meeting at", step.expected_value)

    def test_chinese_progress_is_capped_and_locally_converted(self) -> None:
        first = plan_next_verified_input("你好世界继续", "")
        second = plan_next_verified_input("你好世界继续", "你好世界")
        self.assertEqual("你好世界", first.segment)
        self.assertEqual("nihaoshijie", first.pinyin)
        self.assertEqual("继续", second.segment)
        self.assertEqual(local_pinyin("继续"), second.pinyin)

    def test_mixed_text_preserves_order(self) -> None:
        first = plan_next_verified_input("你好abc。", "")
        second = plan_next_verified_input("你好abc。", "你好")
        third = plan_next_verified_input("你好abc。", "你好abc")
        self.assertEqual(("你好", "chinese_pinyin"), (first.segment, first.kind))
        self.assertEqual(("abc", "direct_latin"), (second.segment, second.kind))
        self.assertEqual(("。", "symbol"), (third.segment, third.kind))

    def test_finished_returns_none(self) -> None:
        self.assertIsNone(plan_next_verified_input("你好", "你好"))

    def test_non_prefix_fails_closed(self) -> None:
        with self.assertRaisesRegex(VerifiedTextTransactionError, "精确前缀"):
            plan_next_verified_input("你好", "你号")

    def test_unsupported_character_fails_before_planning(self) -> None:
        with self.assertRaisesRegex(ValueError, "暂不支持"):
            plan_next_verified_input("你好🙂", "")


if __name__ == "__main__":
    unittest.main()
