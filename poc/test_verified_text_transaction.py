import unittest

from verified_text_transaction import (
    VerifiedTextTransactionError,
    is_direct_latin_batch_segment,
    local_pinyin,
    plan_next_verified_input,
)


class VerifiedTextTransactionTests(unittest.TestCase):
    def test_ascii_progress_uses_exact_remaining_segment(self) -> None:
        step = plan_next_verified_input("meeting at eight", "meeting ")
        self.assertEqual("at eight", step.segment)
        self.assertEqual("direct_latin", step.kind)
        self.assertEqual("meeting at eight", step.expected_value)

    def test_multiline_phrase_uses_two_text_batches_and_one_newline(self) -> None:
        first = plan_next_verified_input("first line\nsecond line", "")
        newline = plan_next_verified_input("first line\nsecond line", "first line")
        second = plan_next_verified_input("first line\nsecond line", "first line\n")

        self.assertEqual(("first line", "direct_latin"), (first.segment, first.kind))
        self.assertEqual(("\n", "literal_key"), (newline.segment, newline.kind))
        self.assertEqual(("second line", "direct_latin"), (second.segment, second.kind))

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
        self.assertEqual(("。", "literal_key"), (third.segment, third.kind))

    def test_uppercase_uses_exact_uppercase_segment_and_lowercase_key_sequence(self) -> None:
        step = plan_next_verified_input("Meeting", "")
        self.assertEqual("M", step.segment)
        self.assertEqual("direct_latin", step.kind)
        self.assertEqual("upper", step.required_case_mode)
        self.assertEqual("m", step.physical_keys)

    def test_digits_and_punctuation_are_one_visible_key_step(self) -> None:
        for target, current, expected in (
            ("a 8", "a ", "8"),
            ("a。", "a", "。"),
        ):
            with self.subTest(target=target, current=current):
                step = plan_next_verified_input(target, current)
                self.assertEqual(expected, step.segment)
                self.assertEqual("literal_key", step.kind)
                self.assertEqual("visible_key", step.required_mode)

    def test_batch_charset_has_one_authoritative_validator(self) -> None:
        for value in ("first line", " leading", "trailing ", " ", "a" * 20):
            with self.subTest(value=value):
                self.assertTrue(is_direct_latin_batch_segment(value))
        for value in ("", "Agent", "agent1", "agent.com", "a" * 21, "中文"):
            with self.subTest(value=value):
                self.assertFalse(is_direct_latin_batch_segment(value))

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
