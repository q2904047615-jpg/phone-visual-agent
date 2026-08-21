from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any


MAX_TEXT_LENGTH = 4000
MAX_FULL_RETYPES = 2

_SUPPORTED_TEXT_RE = re.compile(
    r"^[\u3400-\u4dbf\u4e00-\u9fffA-Za-z0-9 \n"
    r"，。！？、；：,.!?;:'\"（）()《》【】\[\]<>“”‘’…—\-+_@#%&/\\=]+$"
)
_CHINESE_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


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
    return sum(
        1
        for char in normalized
        if not unicodedata.combining(char)
        and char not in {"\ufe0e", "\ufe0f", "\u200d"}
    )


def split_input_segments(text: str) -> list[str]:
    text = normalize_user_text(text, field_name="输入文字")
    segments: list[str] = []
    current = ""
    current_kind: str | None = None

    def flush() -> None:
        nonlocal current, current_kind
        if current:
            segments.append(current)
        current = ""
        current_kind = None

    for char in text:
        if _CHINESE_RE.fullmatch(char):
            kind, limit = "chinese", 4
        elif char.isascii() and (char.isalnum() or char == " "):
            kind, limit = "ascii", 20
        else:
            flush()
            segments.append(char)
            continue
        if current_kind != kind or len(current) >= limit:
            flush()
        current_kind = kind
        current += char
    flush()
    return segments


@dataclass
class InputAttemptState:
    target_text: str
    segments: list[str]
    max_retypes: int = MAX_FULL_RETYPES
    completed_segments: list[str] = field(default_factory=list)
    retypes_used: int = 0
    awaiting_empty_confirmation: bool = False

    @property
    def expected_prefix(self) -> str:
        return "".join(self.completed_segments)

    def accept_segment(self, segment: str, observed_text: str) -> bool:
        expected = self.expected_prefix + segment
        if observed_text != expected:
            return False
        self.completed_segments.append(segment)
        return True

    def begin_full_retype(self, observed_text: str) -> int:
        if self.retypes_used >= self.max_retypes:
            raise ValueError("完整重输已达到2次上限。")
        count = editable_character_count(observed_text)
        if count < 1:
            raise ValueError("无法确认输入框内实际字符数，拒绝猜测退格次数。")
        self.retypes_used += 1
        self.awaiting_empty_confirmation = True
        return count

    def confirm_empty(self, is_empty: bool) -> None:
        if not self.awaiting_empty_confirmation:
            raise ValueError("当前不在清空复核阶段。")
        if not is_empty:
            raise ValueError("输入框尚未确认完全为空。")
        self.completed_segments.clear()
        self.awaiting_empty_confirmation = False


class InputRecoveryCoordinator:
    def __init__(self, sessions: list[dict[str, Any]], plan: list[dict[str, Any]]) -> None:
        self.states: dict[str, InputAttemptState] = {}
        self.step_to_session: dict[str, str] = {}
        self.start_cursor: dict[str, int] = {}
        plan_indexes = {str(step.get("id")): index for index, step in enumerate(plan)}
        for index, raw in enumerate(sessions):
            if not isinstance(raw, dict):
                raise ValueError("input_sessions 每一项必须是对象。")
            session_id = f"input_{index + 1}"
            target_text = normalize_user_text(raw.get("target_text"), field_name="输入会话目标")
            segments = raw.get("segments")
            step_ids = raw.get("step_ids")
            if (
                not isinstance(segments, list)
                or not segments
                or not all(isinstance(value, str) and value for value in segments)
                or "".join(segments) != target_text
                or not isinstance(step_ids, list)
                or len(step_ids) != len(segments)
                or not all(step_id in plan_indexes for step_id in step_ids)
            ):
                raise ValueError("input_sessions 与冻结计划的文字分段不一致。")
            max_retypes = raw.get("max_full_retypes", MAX_FULL_RETYPES)
            if max_retypes != MAX_FULL_RETYPES:
                raise ValueError("每个输入会话必须固定最多完整重输2次。")
            self.states[session_id] = InputAttemptState(target_text, list(segments), max_retypes)
            self.start_cursor[session_id] = plan_indexes[step_ids[0]]
            for step_id in step_ids:
                if step_id in self.step_to_session:
                    raise ValueError("同一冻结步骤不能属于多个输入会话。")
                self.step_to_session[step_id] = session_id

    def state_for_step(self, step_id: str) -> InputAttemptState | None:
        session_id = self.step_to_session.get(step_id)
        return self.states.get(session_id) if session_id else None

    def begin_recovery(self, step_id: str, observed_text: str) -> tuple[int, int]:
        session_id = self.step_to_session.get(step_id)
        if not session_id:
            raise ValueError("当前步骤不属于可恢复的输入会话。")
        return self.states[session_id].begin_full_retype(observed_text), self.start_cursor[session_id]

    def confirm_empty(self, step_id: str) -> int:
        session_id = self.step_to_session.get(step_id)
        if not session_id:
            raise ValueError("当前步骤不属于可恢复的输入会话。")
        self.states[session_id].confirm_empty(True)
        return self.start_cursor[session_id]

    def metrics(self) -> dict[str, Any]:
        sessions = [
            {
                "session_id": session_id,
                "target_length": editable_character_count(state.target_text),
                "segment_count": len(state.segments),
                "completed_segments": len(state.completed_segments),
                "retypes_used": state.retypes_used,
                "max_retypes": state.max_retypes,
                "awaiting_empty_confirmation": state.awaiting_empty_confirmation,
            }
            for session_id, state in self.states.items()
        ]
        return {"total_retypes": sum(item["retypes_used"] for item in sessions), "sessions": sessions}
