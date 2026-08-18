from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

from operation_specs import normalize_user_text


class VerifiedTextTransactionError(ValueError):
    pass


_CHINESE_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+\Z")


def local_pinyin(text: str) -> str:
    if not _CHINESE_RE.fullmatch(text):
        raise VerifiedTextTransactionError("拼音分段必须全部为中文。")
    try:
        from pypinyin import Style, lazy_pinyin
    except ImportError as exc:
        raise VerifiedTextTransactionError("缺少本地拼音组件 pypinyin。") from exc
    value = "".join(lazy_pinyin(text, style=Style.NORMAL, errors="strict"))
    value = re.sub(r"[^a-z]", "", value.casefold())
    if not value or len(value) > 30:
        raise VerifiedTextTransactionError("本地拼音不是1到30个小写字母。")
    return value


@dataclass(frozen=True)
class VerifiedInputStep:
    target_text: str
    current_text: str
    segment: str
    kind: str
    required_mode: str
    expected_value: str
    pinyin: str = ""

    def validate(self) -> None:
        if self.kind not in {"direct_latin", "chinese_pinyin", "symbol"}:
            raise VerifiedTextTransactionError("输入分段类型无效。")
        if not self.target_text.startswith(self.current_text):
            raise VerifiedTextTransactionError("当前输入值不是目标文字的精确前缀。")
        if self.expected_value != self.current_text + self.segment:
            raise VerifiedTextTransactionError("输入分段后置值没有精确拼接当前前缀。")
        if not self.segment:
            raise VerifiedTextTransactionError("输入分段不能为空。")
        if self.kind == "chinese_pinyin":
            if self.required_mode != "chinese_pinyin" or self.pinyin != local_pinyin(self.segment):
                raise VerifiedTextTransactionError("中文分段缺少确定性拼音。")
        elif self.pinyin:
            raise VerifiedTextTransactionError("非中文分段不能携带拼音。")


def plan_next_verified_input(
    target_text: Any,
    current_text: Any,
) -> VerifiedInputStep | None:
    target = normalize_user_text(target_text, field_name="输入文字")
    if not isinstance(current_text, str):
        raise VerifiedTextTransactionError("当前输入框缺少精确文字值。")
    if not target.startswith(current_text):
        raise VerifiedTextTransactionError("当前输入值不是目标文字的精确前缀。")
    if current_text == target:
        return None

    remaining = target[len(current_text) :]
    first = remaining[0]
    if _CHINESE_RE.fullmatch(first):
        segment = ""
        for char in remaining:
            if not _CHINESE_RE.fullmatch(char) or len(segment) >= 4:
                break
            segment += char
        step = VerifiedInputStep(
            target_text=target,
            current_text=current_text,
            segment=segment,
            kind="chinese_pinyin",
            required_mode="chinese_pinyin",
            expected_value=current_text + segment,
            pinyin=local_pinyin(segment),
        )
    elif first in "abcdefghijklmnopqrstuvwxyz":
        segment = ""
        for char in remaining:
            if char not in "abcdefghijklmnopqrstuvwxyz" or len(segment) >= 20:
                break
            segment += char
        step = VerifiedInputStep(
            target_text=target,
            current_text=current_text,
            segment=segment,
            kind="direct_latin",
            required_mode="direct_latin",
            expected_value=current_text + segment,
        )
    else:
        segment = first
        step = VerifiedInputStep(
            target_text=target,
            current_text=current_text,
            segment=segment,
            kind="symbol",
            required_mode="symbol",
            expected_value=current_text + segment,
        )
    step.validate()
    return step


def plan_from_input_states(
    target_text: Any,
    states: Mapping[str, Any],
) -> VerifiedInputStep | None:
    if not isinstance(states, Mapping):
        raise VerifiedTextTransactionError("输入框 states 格式无效。")
    return plan_next_verified_input(target_text, states.get("value"))
