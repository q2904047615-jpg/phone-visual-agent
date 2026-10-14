"""Durable Runtime session snapshots and event ledger.

The physical connector is not part of a SQLite transaction.  The repository
therefore persists a state snapshot and a monotonically ordered event before
dispatch, and marks any in-flight session ``recovery_required`` on the next
process start.  This is deliberately conservative: it prevents a replay of
an effect whose external outcome is unknown.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any, Iterator

from agent.domain import (
    ACTIVE_SESSION_STATUSES,
    AgentSession,
    AgentSessionNotFoundError,
    RECOVERY_REQUIRED_STATUS,
)
from agent.domain.validation import reject_if


SCHEMA_VERSION = "runtime-sqlite-v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


@dataclass
class RecoveredAgentSession:
    """Read-only projection of a session found after a process restart."""

    session_id: str
    device_id: str
    run_dir: Path
    _snapshot: dict[str, Any]
    status: str = RECOVERY_REQUIRED_STATUS
    physical_actions: int = 0
    recovery_required: bool = True
    recovery_resolution: dict[str, Any] | None = None

    def snapshot(self) -> dict[str, Any]:
        result = dict(self._snapshot)
        result.update({
            "session_id": self.session_id,
            "task_id": self.session_id,
            "device_id": self.device_id,
            "status": self.status,
            "recovery_required": self.recovery_required,
            "recovery_reason": (
                "进程重启后仍有未完成运行；外部动作结果必须先核对，不允许自动重放。"
                if self.recovery_required else ""
            ),
        })
        if self.recovery_resolution is not None:
            result["recovery_resolution"] = dict(self.recovery_resolution)
        return result


class SqliteAgentSessionRepository:
    """Thread-safe durable repository used by the production Runtime.

    Live session objects remain in memory so the existing connector and
    orchestrator contracts stay typed.  Each mutation is mirrored to SQLite
    in one transaction together with a ledger event.  A restart never
    reconstructs a live connector object from JSON; it exposes a recovery
    projection and blocks mutation until reconciliation is implemented.
    """

    def __init__(self, database_path: Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._sessions: dict[str, AgentSession] = {}
        self._connection = sqlite3.connect(
            self.database_path,
            timeout=10,
            check_same_thread=False,
            isolation_level=None,
        )
        self._connection.row_factory = sqlite3.Row
        with self._lock:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA foreign_keys=ON")
            self._create_schema()
            self._mark_inflight_for_recovery()

    def _create_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS runtime_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runtime_sessions (
                session_id TEXT PRIMARY KEY,
                device_id TEXT NOT NULL,
                status TEXT NOT NULL,
                revision INTEGER NOT NULL,
                physical_actions INTEGER NOT NULL,
                run_dir TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                snapshot_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runtime_events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                status TEXT NOT NULL,
                revision INTEGER NOT NULL,
                occurred_at TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES runtime_sessions(session_id)
            );
            CREATE INDEX IF NOT EXISTS runtime_events_session_sequence
                ON runtime_events(session_id, sequence);
            INSERT INTO runtime_meta(key, value) VALUES ('schema_version', 'runtime-sqlite-v1')
                ON CONFLICT(key) DO UPDATE SET value=excluded.value;
            """
        )

    def _mark_inflight_for_recovery(self) -> None:
        statuses = tuple(sorted(ACTIVE_SESSION_STATUSES))
        placeholders = ",".join("?" for _ in statuses)
        rows = self._connection.execute(
            f"SELECT session_id, device_id, revision, snapshot_json FROM runtime_sessions "
            f"WHERE status IN ({placeholders})",
            statuses,
        ).fetchall()
        if not rows:
            return
        now = _now()
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            for row in rows:
                snapshot = json.loads(row["snapshot_json"])
                snapshot.update({
                    "status": RECOVERY_REQUIRED_STATUS,
                    "recovery_required": True,
                    "recovery_reason": "进程重启后发现未完成运行。",
                })
                revision = int(row["revision"]) + 1
                self._connection.execute(
                    "UPDATE runtime_sessions SET status=?, revision=?, updated_at=?, snapshot_json=? "
                    "WHERE session_id=?",
                    (RECOVERY_REQUIRED_STATUS, revision, now, _json(snapshot), row["session_id"]),
                )
                self._connection.execute(
                    "INSERT INTO runtime_events(session_id,event_type,status,revision,occurred_at,payload_json) "
                    "VALUES(?,?,?,?,?,?)",
                    (row["session_id"], "recovery_required", RECOVERY_REQUIRED_STATUS, revision, now,
                     _json({"reason": "process_restart", "device_id": row["device_id"]})),
                )
            self._connection.execute("COMMIT")
        except Exception:
            self._connection.execute("ROLLBACK")
            raise

    @staticmethod
    def _snapshot_fields(session: AgentSession, snapshot: dict[str, Any]) -> tuple[str, str, int, str, str]:
        return (
            str(getattr(session, "device_id", snapshot.get("device_id", ""))),
            str(snapshot.get("status", getattr(session, "status", "created"))),
            int(snapshot.get("physical_actions", getattr(session, "physical_actions", 0)) or 0),
            str(getattr(session, "run_dir", snapshot.get("run_dir", ""))),
            str(snapshot.get("created_at", _now())),
        )

    def _persist_locked(self, session: AgentSession, *, event_type: str) -> None:
        snapshot = dict(session.snapshot())
        session_id = str(getattr(session, "session_id", snapshot.get("session_id", "")))
        reject_if(not session_id, ValueError("持久化会话缺少session_id。"))
        device_id, status, physical_actions, run_dir, created_at = self._snapshot_fields(session, snapshot)
        now = _now()
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            current = self._connection.execute(
                "SELECT revision FROM runtime_sessions WHERE session_id=?", (session_id,)
            ).fetchone()
            revision = int(current["revision"]) + 1 if current is not None else 1
            snapshot["revision"] = revision
            snapshot["runtime_event_type"] = event_type
            self._connection.execute(
                "INSERT INTO runtime_sessions(session_id,device_id,status,revision,physical_actions,run_dir,created_at,updated_at,snapshot_json) "
                "VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(session_id) DO UPDATE SET "
                "device_id=excluded.device_id,status=excluded.status,revision=excluded.revision,"
                "physical_actions=excluded.physical_actions,run_dir=excluded.run_dir,updated_at=excluded.updated_at,"
                "snapshot_json=excluded.snapshot_json",
                (session_id, device_id, status, revision, physical_actions, run_dir, created_at, now, _json(snapshot)),
            )
            self._connection.execute(
                "INSERT INTO runtime_events(session_id,event_type,status,revision,occurred_at,payload_json) "
                "VALUES(?,?,?,?,?,?)",
                (session_id, event_type, status, revision, now, _json(snapshot)),
            )
            self._connection.execute("COMMIT")
        except Exception:
            self._connection.execute("ROLLBACK")
            raise

    def add(self, session: AgentSession) -> None:
        with self._lock:
            self._sessions[session.session_id] = session
            self._persist_locked(session, event_type="session_created")

    def save(self, session: AgentSession, *, event_type: str = "state_snapshot") -> None:
        with self._lock:
            self._sessions[session.session_id] = session
            self._persist_locked(session, event_type=event_type)

    def _recovered_from_row(self, row: sqlite3.Row) -> RecoveredAgentSession:
        snapshot = json.loads(row["snapshot_json"])
        stored_status = str(row["status"])
        # A row written by another process after this repository was opened is
        # still a live projection from this process's perspective. Treat every
        # active status as recovery-required before exposing it to orchestration
        # so a missing in-memory session can never receive a live cancel/execute
        # path and crash on absent runtime-only fields.
        recovery_required = stored_status in ACTIVE_SESSION_STATUSES or bool(
            snapshot.get("recovery_required", False)
        ) or stored_status == RECOVERY_REQUIRED_STATUS
        return RecoveredAgentSession(
            session_id=str(row["session_id"]),
            device_id=str(row["device_id"]),
            run_dir=Path(str(row["run_dir"])),
            _snapshot=snapshot,
            status=RECOVERY_REQUIRED_STATUS if recovery_required else stored_status,
            physical_actions=int(row["physical_actions"] or 0),
            recovery_required=recovery_required,
            recovery_resolution=(
                dict(snapshot["recovery_resolution"])
                if isinstance(snapshot.get("recovery_resolution"), dict) else None
            ),
        )

    def get(self, session_id: str) -> AgentSession | None:
        with self._lock:
            live = self._sessions.get(session_id)
            if live is not None:
                return live
            row = self._connection.execute(
                "SELECT * FROM runtime_sessions WHERE session_id=?", (session_id,)
            ).fetchone()
            return self._recovered_from_row(row) if row is not None else None

    def require(self, session_id: str) -> AgentSession:
        session = self.get(session_id)
        reject_if(session is None, AgentSessionNotFoundError("通用单步会话不存在。"))
        return session

    @contextmanager
    def locked(self, session_id: str) -> Iterator[AgentSession]:
        with self._lock:
            session = self.require(session_id)
            yield session
            if not getattr(session, "recovery_required", False):
                self._persist_locked(session, event_type="state_updated")

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                self._connection.execute("DELETE FROM runtime_events")
                self._connection.execute("DELETE FROM runtime_sessions")
                self._connection.execute("COMMIT")
            except Exception:
                self._connection.execute("ROLLBACK")
                raise

    def active_snapshots(self) -> list[dict[str, Any]]:
        with self._lock:
            statuses = tuple(sorted(ACTIVE_SESSION_STATUSES | {RECOVERY_REQUIRED_STATUS}))
            rows = self._connection.execute(
                "SELECT * FROM runtime_sessions WHERE status IN (" + ",".join("?" for _ in statuses) + ") "
                "ORDER BY updated_at DESC",
                statuses,
            ).fetchall()
            live_ids = set(self._sessions)
            # The current process owns live aggregates.  Read their latest
            # projection so an in-process terminal transition is visible even
            # before the next lifecycle snapshot; durable rows cover only
            # sessions recovered from a prior process.
            live = []
            for session in self._sessions.values():
                snapshot = dict(session.snapshot())
                if snapshot.get("status", getattr(session, "status", "")) in statuses:
                    live.append(snapshot)
            recovered = [
                self._recovered_from_row(row).snapshot()
                for row in rows
                if str(row["session_id"]) not in live_ids
            ]
            return sorted([*live, *recovered], key=lambda item: str(item.get("updated_at", "")), reverse=True)

    def events(self, session_id: str, *, after_sequence: int = 0) -> list[dict[str, Any]]:
        with self._lock:
            reject_if(self.get(session_id) is None, AgentSessionNotFoundError("通用单步会话不存在。"))
            rows = self._connection.execute(
                "SELECT sequence,session_id,event_type,status,revision,occurred_at,payload_json "
                "FROM runtime_events WHERE session_id=? AND sequence>? ORDER BY sequence ASC",
                (session_id, int(after_sequence)),
            ).fetchall()
            return [
                {
                    "sequence": int(row["sequence"]),
                    "session_id": row["session_id"],
                    "event_type": row["event_type"],
                    "status": row["status"],
                    "revision": int(row["revision"]),
                    "occurred_at": row["occurred_at"],
                    "payload": json.loads(row["payload_json"]),
                }
                for row in rows
            ]

    def reconcile_recovered(
        self,
        session_id: str,
        *,
        requested_device_id: str,
        external_effect: str,
        resolution: str = "reconciled",
    ) -> RecoveredAgentSession:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM runtime_sessions WHERE session_id=?", (session_id,)
            ).fetchone()
            reject_if(row is None, AgentSessionNotFoundError("通用单步会话不存在。"))
            reject_if(str(row["device_id"]) != str(requested_device_id), ValueError("请求device_id与会话锁定设备不一致。"))
            reject_if(str(row["status"]) != RECOVERY_REQUIRED_STATUS, ValueError("会话当前不需要恢复核对。"))
            reject_if(external_effect not in {"not_dispatched", "unknown"}, ValueError("恢复核对的外部效果类型不受支持。"))
            snapshot = json.loads(row["snapshot_json"])
            resolution_payload = {
                "external_effect": external_effect,
                "resolution": resolution,
                "physical_actions_before": int(row["physical_actions"] or 0),
                "reconciled_at": _now(),
            }
            snapshot.update({
                "status": "cancelled",
                "recovery_required": False,
                "recovery_reason": "",
                "recovery_resolution": resolution_payload,
            })
            revision = int(row["revision"]) + 1
            now = _now()
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                self._connection.execute(
                    "UPDATE runtime_sessions SET status=?, revision=?, updated_at=?, snapshot_json=? "
                    "WHERE session_id=?",
                    ("cancelled", revision, now, _json(snapshot), session_id),
                )
                self._connection.execute(
                    "INSERT INTO runtime_events(session_id,event_type,status,revision,occurred_at,payload_json) "
                    "VALUES(?,?,?,?,?,?)",
                    (session_id, "recovery_reconciled", "cancelled", revision, now, _json(resolution_payload)),
                )
                self._connection.execute("COMMIT")
            except Exception:
                self._connection.execute("ROLLBACK")
                raise
            updated = self._connection.execute(
                "SELECT * FROM runtime_sessions WHERE session_id=?", (session_id,)
            ).fetchone()
            assert updated is not None
            return self._recovered_from_row(updated)

    def close(self) -> None:
        with self._lock:
            self._connection.close()


__all__ = ["SCHEMA_VERSION", "RecoveredAgentSession", "SqliteAgentSessionRepository"]
