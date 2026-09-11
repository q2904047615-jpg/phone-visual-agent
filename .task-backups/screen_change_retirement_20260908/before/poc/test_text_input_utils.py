from __future__ import annotations

import unittest

from agent.domain.text_input_utils import (
    normalize_user_text,
)


class TextInputUtilsTests(unittest.TestCase):
    def test_validation_preserves_newlines_carriage_returns_and_tabs(self) -> None:
        for value in ("你好\n世界", "你好\r\n世界", "你好\t世界"):
            self.assertEqual(value, normalize_user_text(value, field_name="正文"))

    def test_normalization_preserves_emoji_zwj_and_complex_unicode(self) -> None:
        value = "你好👨‍👩‍👧‍👦🙂𠮷\n第二行"
        self.assertEqual(value, normalize_user_text(value, field_name="正文"))

    def test_valid_utf8_codepoints_are_not_rejected_by_an_extra_blacklist(self) -> None:
        for value in ("a\x00b", "a\x1fb", "a\ufdd0b", "a\uffffb"):
            with self.subTest(value=repr(value)):
                self.assertEqual(value, normalize_user_text(value, field_name="正文"))

    def test_unencodable_surrogate_is_a_real_encoding_error(self) -> None:
        with self.assertRaises(UnicodeEncodeError):
            normalize_user_text("\ud800", field_name="正文")


if __name__ == "__main__":
    unittest.main()
