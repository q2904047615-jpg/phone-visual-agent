from __future__ import annotations

from .validation import reject_if
from typing import Any


def normalize_user_text(value: Any, *, field_name: str) -> str:
    """Validate a UTF-8 literal without changing codepoints or imposing a business size limit."""
    reject_if(not isinstance(value, str), ValueError(f"{field_name}必须是文字。"))
    reject_if(not value, ValueError(f"{field_name}不能为空。"))
    value.encode("utf-8")  # Invalid surrogate scalars are a real encoding error.
    return value
