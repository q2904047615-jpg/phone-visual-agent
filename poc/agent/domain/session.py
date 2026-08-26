from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Protocol


ACTIVE_SESSION_STATUSES = frozenset(
    {
        "awaiting_effect_confirmation",
        "awaiting_confirmation",
        "paused_after_action",
        "needs_reobservation",
        "needs_effect_verification",
    }
)


class AgentSessionNotFoundError(LookupError):
    """Raised when a requested universal-agent session does not exist."""


class AgentSessionConflictError(RuntimeError):
    """Raised when a device already owns another active session."""


class AgentSessionDeviceMismatchError(ValueError):
    """Raised when a command targets a device other than the session device."""

    def __init__(self, expected_device_id: str, requested_device_id: str) -> None:
        super().__init__("请求device_id与会话锁定设备不一致。")
        self.expected_device_id = expected_device_id
        self.requested_device_id = requested_device_id


class AgentSession(Protocol):
    """Minimum domain-facing shape of a universal-agent session."""

    session_id: str
    device_id: str
    status: str
    physical_actions: int
    run_dir: Path

    def snapshot(self) -> dict[str, Any]: ...


class AgentSessionRepository(Protocol):
    """Single process-local authority for universal-agent session storage."""

    def add(self, session: AgentSession) -> None: ...

    def get(self, session_id: str) -> AgentSession | None: ...

    def require(self, session_id: str) -> AgentSession: ...

    def locked(self, session_id: str) -> AbstractContextManager[AgentSession]: ...

    def clear(self) -> None: ...

    def active_snapshots(self) -> list[dict[str, Any]]: ...


def require_session_device(
    session: AgentSession,
    requested_device_id: str,
) -> None:
    """Enforce the device identity already fixed by the session aggregate."""

    resolved = str(requested_device_id or "")
    if resolved != session.device_id:
        raise AgentSessionDeviceMismatchError(session.device_id, resolved)
