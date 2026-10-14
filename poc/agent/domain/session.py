from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Protocol


ACTIVE_SESSION_STATUSES = frozenset({'created', 'planning', 'observing', 'awaiting_effect_confirmation',
    'awaiting_confirmation', 'executing_one_action', 'needs_reobservation', 'budget_paused', 'paused'})

TERMINAL_SESSION_STATUSES = frozenset({'succeeded', 'blocked', 'failed', 'cancelled'})

# A process restart after an external dispatch cannot prove whether the
# physical effect happened.  Recovery is therefore an explicit non-terminal
# state which blocks replay until a human or a future recovery workflow
# resolves the durable record.
RECOVERY_REQUIRED_STATUS = 'recovery_required'


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


class AgentSessionRecoveryRequiredError(RuntimeError):
    """Raised when a durable session needs post-crash reconciliation."""


class AgentSession(Protocol):
    """Minimum domain-facing shape of a universal-agent session."""

    session_id: str
    device_id: str
    status: str
    physical_actions: int
    run_dir: Path

    def snapshot(self) -> dict[str, Any]: ...


class AgentSessionRepository(Protocol):
    """Durable authority for universal-agent session storage and events."""

    def add(self, session: AgentSession) -> None: ...

    def get(self, session_id: str) -> AgentSession | None: ...

    def require(self, session_id: str) -> AgentSession: ...

    def locked(self, session_id: str) -> AbstractContextManager[AgentSession]: ...

    def clear(self) -> None: ...

    def active_snapshots(self) -> list[dict[str, Any]]: ...

    def save(self, session: AgentSession, *, event_type: str = 'state_snapshot') -> None: ...

    def events(self, session_id: str, *, after_sequence: int = 0) -> list[dict[str, Any]]: ...

    def read_events(self, session_id: str, *, after_sequence: int = 0,
                    limit: int = 200) -> Any: ...

    def reconcile_recovered(
        self,
        session_id: str,
        *,
        requested_device_id: str,
        external_effect: str,
        resolution: str = 'reconciled',
    ) -> AgentSession: ...


def require_session_device(session: AgentSession, requested_device_id: str) -> None:
    """Enforce the device identity already fixed by the session aggregate."""

    resolved = str(requested_device_id or "")
    if resolved != session.device_id:
        raise AgentSessionDeviceMismatchError(session.device_id, resolved)
