from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

from .text_input_utils import normalize_user_text


class VerifiedTextTransactionError(ValueError):
    pass


_CHINESE_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+\Z")
MAX_DIRECT_LATIN_SEGMENT_CHARS = 20
DIRECT_LATIN_CHARACTERS = frozenset("abcdefghijklmnopqrstuvwxyz")
# The real keyboard exposes numeric and symbol layouts from the alphabetic
# surface. QWERTY is therefore the hub: a symbol must never be approached by
# treating the numeric ``123`` key as an implicit first hop.
KEYBOARD_LAYOUT_PATH = ("numeric", "qwerty", "symbol")


def is_direct_latin_segment(value: Any) -> bool:
    """Return whether one fragment is an audited visible lowercase sequence."""

    return bool(
        isinstance(value, str)
        and 1 <= len(value) <= MAX_DIRECT_LATIN_SEGMENT_CHARS
        and all(char in DIRECT_LATIN_CHARACTERS for char in value)
    )


def preferred_keyboard_layout(character: Any) -> str:
    """Return the canonical visible-key layout for one next character."""

    if not isinstance(character, str) or len(character) != 1:
        raise VerifiedTextTransactionError("下一逐键字符必须恰好一个字符。")
    if character.isdecimal():
        return "numeric"
    if character == " " or character.isalpha():
        return "qwerty"
    return "symbol"


def next_keyboard_layout_towards( current_layout: Any, desired_layout: Any, ) -> str | None:
    """Return the sole adjacent layout that shortens the canonical path."""

    if (
        current_layout not in KEYBOARD_LAYOUT_PATH
        or desired_layout not in KEYBOARD_LAYOUT_PATH
        or current_layout == desired_layout
    ):
        return None
    current_index = KEYBOARD_LAYOUT_PATH.index(current_layout)
    desired_index = KEYBOARD_LAYOUT_PATH.index(desired_layout)
    return KEYBOARD_LAYOUT_PATH[current_index + (1 if desired_index > current_index else -1)]


def keyboard_layout_switch_advances( *, current_layout: Any, target_layout: Any, desired_layout: Any, ) -> bool:
    """Accept a visible direct edge or the sole shortest-path next hop."""

    if (
        current_layout not in KEYBOARD_LAYOUT_PATH
        or target_layout not in KEYBOARD_LAYOUT_PATH
        or desired_layout not in KEYBOARD_LAYOUT_PATH
        or current_layout == target_layout
    ):
        return False
    return bool(
        target_layout == desired_layout
        or target_layout
        == next_keyboard_layout_towards(current_layout, desired_layout)
    )


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
    required_case_mode: str = ""
    physical_keys: str = ""

    def validate(self) -> None:
        if self.kind not in {"direct_latin", "chinese_pinyin", "literal_key"}:
            raise VerifiedTextTransactionError("输入分段类型无效。")
        if not self.target_text.startswith(self.current_text):
            raise VerifiedTextTransactionError("当前输入值不是目标文字的精确前缀。")
        if self.expected_value != self.current_text + self.segment:
            raise VerifiedTextTransactionError("输入分段后置值没有精确拼接当前前缀。")
        if not self.segment:
            raise VerifiedTextTransactionError("输入分段不能为空。")
        if self.kind == "chinese_pinyin":
            if (
                self.required_mode != "chinese_pinyin"
                or self.pinyin != local_pinyin(self.segment)
                or self.required_case_mode
                or self.physical_keys != self.pinyin
            ):
                raise VerifiedTextTransactionError("中文分段缺少确定性拼音。")
        elif self.kind == "direct_latin":
            if (
                self.required_mode != "direct_latin"
                or self.pinyin
                or self.required_case_mode not in {"", "upper"}
                or self.physical_keys != self.segment.casefold()
                or not all(
                    char in DIRECT_LATIN_CHARACTERS
                    for char in self.physical_keys
                )
            ):
                raise VerifiedTextTransactionError("英文分段缺少确定性键序列。")
        elif (
            self.required_mode != "visible_key"
            or self.pinyin
            or self.required_case_mode
            or self.physical_keys
            or len(self.segment) != 1
        ):
            raise VerifiedTextTransactionError("逐键分段合同无效。")


def required_keyboard_input_mode_for_step( step: VerifiedInputStep, ) -> str | None:
    """Return the one input mode required before executing ``step``.

    Letters use direct Latin, Chinese uses pinyin, and complex symbols first
    return to direct Latin before entering the symbol layout. Digits, spaces
    and newline keys do not require a Chinese/English mode transition.
    """

    if step.kind in {"direct_latin", "chinese_pinyin"}:
        return step.required_mode
    if (
        step.kind == "literal_key"
        and step.segment.isprintable()
        and preferred_keyboard_layout(step.segment) == "symbol"
    ):
        return "direct_latin"
    return None


def plan_next_verified_input( target_text: Any, current_text: Any, ) -> VerifiedInputStep | None:
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
            physical_keys=local_pinyin(segment),
        )
    elif first in DIRECT_LATIN_CHARACTERS:
        segment = ""
        for char in remaining:
            if ( char not in DIRECT_LATIN_CHARACTERS or len(segment) >= MAX_DIRECT_LATIN_SEGMENT_CHARS ):
                break
            segment += char
        step = VerifiedInputStep(
            target_text=target,
            current_text=current_text,
            segment=segment,
            kind="direct_latin",
            required_mode="direct_latin",
            expected_value=current_text + segment,
            physical_keys=segment,
        )
    elif first in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        segment = ""
        for char in remaining:
            if ( char not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" or len(segment) >= MAX_DIRECT_LATIN_SEGMENT_CHARS ):
                break
            segment += char
        step = VerifiedInputStep(
            target_text=target,
            current_text=current_text,
            segment=segment,
            kind="direct_latin",
            required_mode="direct_latin",
            required_case_mode="upper",
            expected_value=current_text + segment,
            physical_keys=segment.casefold(),
        )
    else:
        segment = first
        step = VerifiedInputStep(
            target_text=target,
            current_text=current_text,
            segment=segment,
            kind="literal_key",
            required_mode="visible_key",
            expected_value=current_text + segment,
        )
    step.validate()
    return step


def plan_from_input_states( target_text: Any, states: Mapping[str, Any], ) -> VerifiedInputStep | None:
    if not isinstance(states, Mapping):
        raise VerifiedTextTransactionError("输入框 states 格式无效。")
    return plan_next_verified_input(target_text, states.get("value"))
