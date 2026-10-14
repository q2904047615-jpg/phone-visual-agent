"""Stable Runtime event and failure-taxonomy contracts for read-only replay.

The replay reader is an observability surface.  It validates durable records,
provides a bounded cursor, and classifies failures for reports; it never grants
permission to execute or re-execute a tool.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


RUNTIME_EVENT_REPLAY_PROTOCOL = "2026-10-14-runtime-event-replay-v1"
MAX_REPLAY_PAGE_SIZE = 200

FAILURE_CATEGORIES = frozenset({
    "configuration",
    "device",
    "execution",
    "model",
    "observation",
    "policy",
    "recovery",
    "verification",
    "system",
})


def _text(value: Any) -> str:
    return str(value or "").strip()


@dataclass(frozen=True)
class RuntimeFailure:
    """A report-only primary failure classification."""

    category: str
    code: str
    stage: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "category": self.category,
            "code": self.code,
            "stage": self.stage,
            "message": self.message,
        }


def classify_runtime_failure(
    error: BaseException | str | None,
    *,
    stage: str = "",
    payload: Mapping[str, Any] | None = None,
) -> RuntimeFailure | None:
    """Return one stable primary category for a failed/uncertain event.

    Explicit producer fields win.  The fallback is deliberately conservative
    and diagnostic-only; it cannot turn a failure into a retry or a success.
    """

    fields = payload if isinstance(payload, Mapping) else {}
    message = _text(error) or _text(fields.get("failed_reason")) or _text(fields.get("error"))
    resolved_stage = _text(stage) or _text(fields.get("failure_stage"))
    explicit_category = _text(fields.get("failure_category"))
    explicit_code = _text(fields.get("failure_code"))
    if explicit_category in FAILURE_CATEGORIES:
        return RuntimeFailure(
            category=explicit_category,
            code=explicit_code or "classified_failure",
            stage=resolved_stage or "runtime",
            message=message,
        )
    lowered = f"{resolved_stage} {message}".casefold()
    if any(token in lowered for token in ("recovery", "恢复", "unknown_effect", "not_dispatched")):
        category = "recovery"
    elif any(token in lowered for token in ("policy", "策略", "confirmation", "确认scope")):
        category = "policy"
    elif any(token in lowered for token in ("verif", "验证", "完成事实")):
        category = "verification"
    elif any(token in lowered for token in ("observ", "观察", "截图", "fingerprint")):
        category = "observation"
    elif any(token in lowered for token in ("qwen", "model", "模型", "json", "协议")):
        category = "model"
    elif any(token in lowered for token in ("device", "设备", "controller", "camera", "机械", "adb")):
        category = "device"
    elif any(token in lowered for token in ("config", "配置", "registry", "注册表")):
        category = "configuration"
    elif any(token in lowered for token in ("execute", "execution", "执行", "transport", "动作")):
        category = "execution"
    else:
        category = "system"
    code = explicit_code or resolved_stage or "runtime_failure"
    return RuntimeFailure(category=category, code=code, stage=resolved_stage or "runtime", message=message)


@dataclass(frozen=True)
class RuntimeEvent:
    """Validated durable event record exposed to replay consumers."""

    sequence: int
    session_id: str
    event_type: str
    status: str
    revision: int
    occurred_at: str
    payload: dict[str, Any]
    failure: RuntimeFailure | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": RUNTIME_EVENT_REPLAY_PROTOCOL,
            "sequence": self.sequence,
            "session_id": self.session_id,
            "event_type": self.event_type,
            "status": self.status,
            "revision": self.revision,
            "occurred_at": self.occurred_at,
            "payload": dict(self.payload),
            "failure": self.failure.to_dict() if self.failure else None,
        }

    @classmethod
    def from_row(cls, row: Mapping[str, Any], *, expected_session_id: str) -> "RuntimeEvent":
        sequence = row.get("sequence")
        revision = row.get("revision")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence <= 0:
            raise ValueError("事件sequence必须是正整数。")
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
            raise ValueError("事件revision必须是非负整数。")
        session_id = _text(row.get("session_id"))
        if not session_id or session_id != expected_session_id:
            raise ValueError("事件session_id与回放会话不一致。")
        event_type = _text(row.get("event_type"))
        status = _text(row.get("status"))
        occurred_at = _text(row.get("occurred_at"))
        payload = row.get("payload")
        if not event_type or not status or not occurred_at or not isinstance(payload, dict):
            raise ValueError("事件记录缺少稳定字段或payload不是对象。")
        failure = None
        if status in {"failed", "blocked"} or payload.get("uncertain_effect"):
            failure = classify_runtime_failure(None, payload=payload)
        return cls(sequence=sequence, session_id=session_id, event_type=event_type,
                   status=status, revision=revision, occurred_at=occurred_at,
                   payload=dict(payload), failure=failure)


@dataclass(frozen=True)
class RuntimeEventPage:
    """One bounded, repeatable page in the event stream."""

    session_id: str
    after_sequence: int
    events: tuple[RuntimeEvent, ...]
    has_more: bool

    @property
    def next_sequence(self) -> int:
        return self.events[-1].sequence if self.events else self.after_sequence

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": RUNTIME_EVENT_REPLAY_PROTOCOL,
            "session_id": self.session_id,
            "after_sequence": self.after_sequence,
            "events": [event.to_dict() for event in self.events],
            "next_sequence": self.next_sequence,
            "has_more": self.has_more,
        }


def validate_replay_cursor(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("回放cursor必须是非负整数。")
    return value


def validate_replay_page_size(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_REPLAY_PAGE_SIZE:
        raise ValueError(f"回放limit必须是1～{MAX_REPLAY_PAGE_SIZE}整数。")
    return value


__all__ = [
    "FAILURE_CATEGORIES",
    "MAX_REPLAY_PAGE_SIZE",
    "RUNTIME_EVENT_REPLAY_PROTOCOL",
    "RuntimeEvent",
    "RuntimeEventPage",
    "RuntimeFailure",
    "classify_runtime_failure",
    "validate_replay_cursor",
    "validate_replay_page_size",
]
