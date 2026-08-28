from __future__ import annotations

import re
import unicodedata
from typing import Any


MAX_TEXT_LENGTH = 4000

_SUPPORTED_TEXT_RE = re.compile(
    r"^[\u3400-\u4dbf\u4e00-\u9fffA-Za-z0-9 \n"
    r"，。！？、；：,.!?;:'\"（）()《》【】\[\]<>“”‘’…—\-+_@#%&/\\=]+$"
)


def normalize_user_text(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name}必须是文字。")
    text = unicodedata.normalize("NFC", value)
    if not text:
        raise ValueError(f"{field_name}不能为空。")
    if len(text) > MAX_TEXT_LENGTH:
        raise ValueError(f"{field_name}不能超过{MAX_TEXT_LENGTH}个字符。")
    if "\r" in text:
        raise ValueError("输入文字不允许回车控制符；请使用换行字符。")
    if not _SUPPORTED_TEXT_RE.fullmatch(text):
        raise ValueError(f"{field_name}含暂不支持的表情、生僻符号或控制字符。")
    return text


def editable_character_count(text: str) -> int:
    normalized = unicodedata.normalize("NFC", text)
    if re.fullmatch(r"[A-Za-z\s'’]+", normalized):
        return len(re.sub(r"[\s'’]", "", normalized))
    return sum((1 for char in normalized if not unicodedata.combining(char) and char not in {'︎', '️', '\u200d'}))
