from __future__ import annotations

from agent.domain.validation import reject_if
import threading
from contextlib import contextmanager
from typing import Iterator

from agent.domain import ACTIVE_SESSION_STATUSES, AgentSession, AgentSessionNotFoundError, RECOVERY_REQUIRED_STATUS


class InMemoryAgentSessionRepository:
    """Thread-safe process-local implementation of the session repository port."""

    def __init__(self) -> None:
        self._sessions: dict[str, AgentSession] = {}
        self._lock = threading.RLock()

    def add(self, session: AgentSession) -> None:
        with self._lock:
            self._sessions[session.session_id] = session

    def get(self, session_id: str) -> AgentSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def require(self, session_id: str) -> AgentSession:
        session = self.get(session_id)
        reject_if(session is None, AgentSessionNotFoundError("通用单步会话不存在。"))
        return session

    @contextmanager
    def locked(self, session_id: str) -> Iterator[AgentSession]:
        with self._lock:
            session = self._sessions.get(session_id)
            reject_if(session is None, AgentSessionNotFoundError("通用单步会话不存在。"))
            yield session

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()

    def active_snapshots(self) -> list[dict[str, object]]:
        with self._lock:
            return [session.snapshot() for session in self._sessions.values() if session.status
                in ACTIVE_SESSION_STATUSES]

    def reconcile_recovered(
        self,
        session_id: str,
        *,
        requested_device_id: str,
        external_effect: str,
        resolution: str = "reconciled",
    ) -> AgentSession:
        with self._lock:
            session = self.require(session_id)
            reject_if(session.device_id != requested_device_id, ValueError("请求device_id与会话锁定设备不一致。"))
            reject_if(session.status != RECOVERY_REQUIRED_STATUS, ValueError("会话当前不需要恢复核对。"))
            reject_if(external_effect not in {"not_dispatched", "unknown"}, ValueError("恢复核对的外部效果类型不受支持。"))
            session.status = "cancelled"
            if hasattr(session, "recovery_required"):
                session.recovery_required = False
            if hasattr(session, "recovery_resolution"):
                session.recovery_resolution = {
                    "external_effect": external_effect,
                    "resolution": resolution,
                }
            return session
