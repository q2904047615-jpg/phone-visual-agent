"""Durable, read-only Control Plane projection for the local API.

This store is deliberately separate from Runtime admission and Policy.  It
provides a small SQLite-backed projection for organization, team, principal,
quota and read-audit queries.  The local seed is an installation projection,
not an identity provider or a task authorization decision.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any, Mapping
import uuid

from agent.domain.control_plane import (
    API_CONTRACT_VERSION,
    CONTROL_PLANE_PROTOCOL,
    ControlPlaneOrganization,
    ControlPlanePrincipal,
    ControlPlaneSnapshot,
    ControlPlaneTeam,
    QuotaLimit,
)


CONTROL_PLANE_SCHEMA_VERSION = "control-plane-sqlite-v1"
LOCAL_ORGANIZATION_ID = "local-org"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class SqliteControlPlaneRepository:
    """SQLite projection used by the read-only Control Plane routes."""

    def __init__(self, database_path: Path, *, seed_local: bool = True) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
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
            if seed_local:
                self._seed_local_projection()

    def _create_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS control_plane_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS control_plane_organizations (
                organization_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                owner_user_id TEXT NOT NULL,
                team_ids_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS control_plane_teams (
                team_id TEXT PRIMARY KEY,
                organization_id TEXT NOT NULL,
                name TEXT NOT NULL,
                member_ids_json TEXT NOT NULL,
                FOREIGN KEY(organization_id) REFERENCES control_plane_organizations(organization_id)
            );
            CREATE INDEX IF NOT EXISTS control_plane_teams_org
                ON control_plane_teams(organization_id, team_id);
            CREATE TABLE IF NOT EXISTS control_plane_principals (
                user_id TEXT PRIMARY KEY,
                organization_id TEXT NOT NULL,
                role TEXT NOT NULL,
                team_ids_json TEXT NOT NULL,
                FOREIGN KEY(organization_id) REFERENCES control_plane_organizations(organization_id)
            );
            CREATE INDEX IF NOT EXISTS control_plane_principals_org
                ON control_plane_principals(organization_id, user_id);
            CREATE TABLE IF NOT EXISTS control_plane_quotas (
                organization_id TEXT NOT NULL,
                quota_key TEXT NOT NULL,
                period TEXT NOT NULL,
                max_runs INTEGER,
                max_physical_actions INTEGER,
                max_model_tokens INTEGER,
                PRIMARY KEY(organization_id, quota_key),
                FOREIGN KEY(organization_id) REFERENCES control_plane_organizations(organization_id)
            );
            CREATE TABLE IF NOT EXISTS control_plane_audit (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                occurred_at TEXT NOT NULL,
                request_id TEXT NOT NULL,
                organization_id TEXT NOT NULL,
                resource TEXT NOT NULL,
                outcome TEXT NOT NULL,
                actor TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS control_plane_audit_org_sequence
                ON control_plane_audit(organization_id, sequence);
            INSERT INTO control_plane_meta(key, value)
                VALUES ('schema_version', 'control-plane-sqlite-v1')
                ON CONFLICT(key) DO UPDATE SET value=excluded.value;
            """
        )

    def _seed_local_projection(self) -> None:
        """Create a deterministic local projection without replacing data."""
        organization = ControlPlaneOrganization(
            organization_id=LOCAL_ORGANIZATION_ID,
            name="Local Control Plane",
            owner_user_id="local-owner",
            team_ids=("local-team",),
        )
        team = ControlPlaneTeam(
            team_id="local-team",
            organization_id=organization.organization_id,
            name="Local Team",
            member_ids=(organization.owner_user_id,),
        )
        principal = ControlPlanePrincipal(
            user_id=organization.owner_user_id,
            organization_id=organization.organization_id,
            role="owner",
            team_ids=(team.team_id,),
        )
        snapshot = ControlPlaneSnapshot(
            organization=organization,
            teams=(team,),
            principals=(principal,),
            quotas={
                "daily": QuotaLimit("daily", max_runs=100, max_physical_actions=1000, max_model_tokens=2_000_000),
                "monthly": QuotaLimit("monthly", max_runs=2_000, max_physical_actions=20_000, max_model_tokens=40_000_000),
            },
        )
        snapshot.validate()
        self._connection.execute(
            "INSERT OR IGNORE INTO control_plane_organizations(organization_id,name,owner_user_id,team_ids_json) VALUES(?,?,?,?)",
            (organization.organization_id, organization.name, organization.owner_user_id, _json(list(organization.team_ids))),
        )
        self._connection.execute(
            "INSERT OR IGNORE INTO control_plane_teams(team_id,organization_id,name,member_ids_json) VALUES(?,?,?,?)",
            (team.team_id, team.organization_id, team.name, _json(list(team.member_ids))),
        )
        self._connection.execute(
            "INSERT OR IGNORE INTO control_plane_principals(user_id,organization_id,role,team_ids_json) VALUES(?,?,?,?)",
            (principal.user_id, principal.organization_id, principal.role, _json(list(principal.team_ids))),
        )
        for quota_key, quota in snapshot.quotas.items():
            self._connection.execute(
                "INSERT OR IGNORE INTO control_plane_quotas(organization_id,quota_key,period,max_runs,max_physical_actions,max_model_tokens) VALUES(?,?,?,?,?,?)",
                (organization.organization_id, quota_key, quota.period, quota.max_runs, quota.max_physical_actions, quota.max_model_tokens),
            )

    @staticmethod
    def _organization(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "organization_id": str(row["organization_id"]),
            "name": str(row["name"]),
            "owner_user_id": str(row["owner_user_id"]),
            "team_ids": list(json.loads(row["team_ids_json"])),
        }

    def get_organization(self, organization_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM control_plane_organizations WHERE organization_id=?", (organization_id,)
            ).fetchone()
            return self._organization(row) if row is not None else None

    def list_teams(self, organization_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM control_plane_teams WHERE organization_id=? ORDER BY team_id", (organization_id,)
            ).fetchall()
            return [{
                "team_id": str(row["team_id"]),
                "organization_id": str(row["organization_id"]),
                "name": str(row["name"]),
                "member_ids": list(json.loads(row["member_ids_json"])),
            } for row in rows]

    def list_principals(self, organization_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM control_plane_principals WHERE organization_id=? ORDER BY user_id", (organization_id,)
            ).fetchall()
            return [{
                "user_id": str(row["user_id"]),
                "organization_id": str(row["organization_id"]),
                "role": str(row["role"]),
                "team_ids": list(json.loads(row["team_ids_json"])),
            } for row in rows]

    def list_quotas(self, organization_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM control_plane_quotas WHERE organization_id=? ORDER BY quota_key", (organization_id,)
            ).fetchall()
            return [{
                "quota_key": str(row["quota_key"]),
                "organization_id": str(row["organization_id"]),
                "period": str(row["period"]),
                "max_runs": row["max_runs"],
                "max_physical_actions": row["max_physical_actions"],
                "max_model_tokens": row["max_model_tokens"],
                "usage": {"runs": 0, "physical_actions": 0, "model_tokens": 0},
            } for row in rows]

    def record_audit(self, *, organization_id: str, resource: str, outcome: str,
                     request_id: str | None = None, metadata: Mapping[str, Any] | None = None) -> int:
        safe_metadata = {
            str(key): value for key, value in (metadata or {}).items()
            if str(key).lower() not in {"token", "authorization", "password", "secret", "cookie"}
        }
        with self._lock:
            cursor = self._connection.execute(
                "INSERT INTO control_plane_audit(occurred_at,request_id,organization_id,resource,outcome,actor,metadata_json) VALUES(?,?,?,?,?,?,?)",
                (_now(), request_id or uuid.uuid4().hex, organization_id, resource, outcome, "local-control-token", _json(safe_metadata)),
            )
            return int(cursor.lastrowid)

    def read_audit(self, organization_id: str, *, after_sequence: int = 0, limit: int = 50) -> dict[str, Any]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM control_plane_audit WHERE organization_id=? AND sequence>? ORDER BY sequence ASC LIMIT ?",
                (organization_id, int(after_sequence), int(limit) + 1),
            ).fetchall()
            has_more = len(rows) > limit
            rows = rows[:limit]
            entries = [{
                "sequence": int(row["sequence"]),
                "occurred_at": str(row["occurred_at"]),
                "request_id": str(row["request_id"]),
                "organization_id": str(row["organization_id"]),
                "resource": str(row["resource"]),
                "outcome": str(row["outcome"]),
                "actor": str(row["actor"]),
                "metadata": json.loads(row["metadata_json"]),
            } for row in rows]
            return {"after_sequence": int(after_sequence), "entries": entries, "has_more": has_more}

    def close(self) -> None:
        with self._lock:
            self._connection.close()


__all__ = ["CONTROL_PLANE_SCHEMA_VERSION", "LOCAL_ORGANIZATION_ID", "SqliteControlPlaneRepository"]
