from __future__ import annotations

import unittest

from agent.domain.text_input_utils import (
    editable_character_count,
    normalize_user_text,
)


class TextInputUtilsTests(unittest.TestCase):
    def test_editable_count_matches_pinyin_and_unicode_text(self) -> None:
        self.assertEqual(5, editable_character_count("ni' hao"))
        self.assertEqual(4, editable_character_count("你好1，"))

    def test_normalization_accepts_newline_but_rejects_carriage_return(self) -> None:
        self.assertEqual("你好\n世界", normalize_user_text("你好\n世界", field_name="正文"))
        with self.assertRaisesRegex(ValueError, "回车控制符"):
            normalize_user_text("你好\r世界", field_name="正文")

    def test_normalization_preserves_emoji_zwj_and_complex_unicode(self) -> None:
        value = "你好👨‍👩‍👧‍👦🙂𠮷\n第二行"
        self.assertEqual(value, normalize_user_text(value, field_name="正文"))

    def test_normalization_rejects_control_and_unicode_noncharacters(self) -> None:
        for value in ("a\x00b", "a\x1fb", "a\ufdd0b", "a\uffffb"):
            with self.subTest(value=repr(value)):
                with self.assertRaisesRegex(ValueError, "控制字符"):
                    normalize_user_text(value, field_name="正文")


if __name__ == "__main__":
    unittest.main()
