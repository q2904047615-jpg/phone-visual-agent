from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator

from agent.domain import (
    ACTIVE_SESSION_STATUSES,
    AgentSession,
    AgentSessionNotFoundError,
)


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
        if session is None:
            raise AgentSessionNotFoundError("通用单步会话不存在。")
        return session

    @contextmanager
    def locked(self, session_id: str) -> Iterator[AgentSession]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise AgentSessionNotFoundError("通用单步会话不存在。")
            yield session

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()

    def active_snapshots(self) -> list[dict[str, object]]:
        with self._lock:
            return [
                session.snapshot()
                for session in self._sessions.values()
                if session.status in ACTIVE_SESSION_STATUSES
            ]
