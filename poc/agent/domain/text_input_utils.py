from __future__ import annotations

from .validation import reject_if
import unicodedata
from typing import Any


MAX_TEXT_LENGTH = 4000

def normalize_user_text(value: Any, *, field_name: str) -> str:
    reject_if(not isinstance(value, str), ValueError(f"{field_name}必须是文字。"))
    text = unicodedata.normalize("NFC", value)
    reject_if(not text, ValueError(f"{field_name}不能为空。"))
    reject_if(len(text) > MAX_TEXT_LENGTH, ValueError(f"{field_name}不能超过{MAX_TEXT_LENGTH}个字符。"))
    reject_if('\r' in text, ValueError("输入文字不允许回车控制符；请使用换行字符。"))
    reject_if(any(_forbidden_unicode_scalar(char) for char in text),
        ValueError(f"{field_name}含不允许的控制字符、代理项或 Unicode 非字符。"))
    return text


def _forbidden_unicode_scalar(character: str) -> bool:
    """Reject transport-unsafe scalars without banning ordinary Unicode or emoji."""

    codepoint = ord(character)
    if character == '\n':
        return False
    if unicodedata.category(character) in {'Cc', 'Cs'}:
        return True
    return 0xFDD0 <= codepoint <= 0xFDEF or (codepoint & 0xFFFF) in {0xFFFE, 0xFFFF}


def editable_character_count(text: str) -> int:
    normalized = unicodedata.normalize("NFC", text)
    if normalized and all(char in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz \n\t'’" for char in normalized):
        return sum(char not in " \n\t'’" for char in normalized)
    return sum((1 for char in normalized if not unicodedata.combining(char) and char not in {'︎', '️', '\u200d'}))
