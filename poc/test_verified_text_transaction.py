import unittest

from verified_text_transaction import (
    VerifiedTextTransactionError,
    is_direct_latin_segment,
    keyboard_layout_switch_advances,
    local_pinyin,
    next_keyboard_layout_towards,
    plan_next_verified_input,
    preferred_keyboard_layout,
)


class VerifiedTextTransactionTests(unittest.TestCase):
    def test_ascii_progress_uses_exact_remaining_segment(self) -> None:
        step = plan_next_verified_input("meeting at eight", "meeting ")
        self.assertEqual("at", step.segment)
        self.assertEqual("direct_latin", step.kind)
        self.assertEqual("meeting at", step.expected_value)

    def test_multiline_phrase_keeps_spaces_and_newline_as_visible_keys(self) -> None:
        target = "first line\nsecond line"
        current = ""
        actual = []
        while current != target:
            step = plan_next_verified_input(target, current)
            actual.append((step.segment, step.kind))
            current = step.expected_value

        self.assertEqual(
            [
                ("first", "direct_latin"),
                (" ", "literal_key"),
                ("line", "direct_latin"),
                ("\n", "literal_key"),
                ("second", "direct_latin"),
                (" ", "literal_key"),
                ("line", "direct_latin"),
            ],
            actual,
        )

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
            ("a 8", "a", " "),
            ("a 8", "a ", "8"),
            ("a。", "a", "。"),
        ):
            with self.subTest(target=target, current=current):
                step = plan_next_verified_input(target, current)
                self.assertEqual(expected, step.segment)
                self.assertEqual("literal_key", step.kind)
                self.assertEqual("visible_key", step.required_mode)

    def test_keyboard_layout_path_has_one_shared_multihop_authority(self) -> None:
        self.assertEqual("numeric", preferred_keyboard_layout("8"))
        self.assertEqual("qwerty", preferred_keyboard_layout("你"))
        self.assertEqual("symbol", preferred_keyboard_layout("？"))
        self.assertEqual(
            "numeric",
            next_keyboard_layout_towards("qwerty", "symbol"),
        )
        self.assertEqual(
            "numeric",
            next_keyboard_layout_towards("symbol", "qwerty"),
        )
        self.assertTrue(
            keyboard_layout_switch_advances(
                current_layout="qwerty",
                target_layout="numeric",
                desired_layout="symbol",
            )
        )
        self.assertTrue(
            keyboard_layout_switch_advances(
                current_layout="qwerty",
                target_layout="symbol",
                desired_layout="symbol",
            )
        )
        self.assertFalse(
            keyboard_layout_switch_advances(
                current_layout="qwerty",
                target_layout="numeric",
                desired_layout="qwerty",
            )
        )

    def test_direct_latin_charset_has_one_authoritative_validator(self) -> None:
        for value in ("first", "line", "a" * 20):
            with self.subTest(value=value):
                self.assertTrue(is_direct_latin_segment(value))
        for value in ("", "first line", " ", "Agent", "agent1", "agent.com", "a" * 21, "中文"):
            with self.subTest(value=value):
                self.assertFalse(is_direct_latin_segment(value))

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
