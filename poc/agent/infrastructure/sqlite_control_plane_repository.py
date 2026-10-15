"""Durable, read-only Control Plane projection for the local API.

This store is deliberately separate from Runtime admission and Policy.  It
provides a small SQLite-backed projection for organization, team, principal,
quota and read-audit queries.  The local seed is an installation projection,
not an identity provider or a task authorization decision.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import secrets
import sqlite3
import threading
from typing import Any, Callable, Mapping
import uuid
import uuid

from agent.domain.control_plane import (
    API_CONTRACT_VERSION,
    CONTROL_PLANE_PROTOCOL,
    ControlPlaneOrganization,
    ControlPlanePrincipal,
    ControlPlaneSnapshot,
    ControlPlaneTeam,
    QuotaLimit,
    ROLE_PERMISSIONS,
    manageable_member_roles,
    validate_github_subject,
)


CONTROL_PLANE_SCHEMA_VERSION = "control-plane-sqlite-v1"
LOCAL_ORGANIZATION_ID = "local-org"
IDENTITY_SESSION_TTL_SECONDS = 3600


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
            CREATE TABLE IF NOT EXISTS control_plane_team_versions (
                team_id TEXT PRIMARY KEY REFERENCES control_plane_teams(team_id),
                revision INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS control_plane_projects (
                project_id TEXT PRIMARY KEY,
                organization_id TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('active','archived')),
                member_ids_json TEXT NOT NULL,
                FOREIGN KEY(organization_id) REFERENCES control_plane_organizations(organization_id)
            );
            CREATE TABLE IF NOT EXISTS control_plane_project_versions (
                project_id TEXT PRIMARY KEY REFERENCES control_plane_projects(project_id),
                revision INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS control_plane_projects_org
                ON control_plane_projects(organization_id, project_id);
            CREATE TABLE IF NOT EXISTS control_plane_device_assignment_versions (
                device_id TEXT PRIMARY KEY,
                revision INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS control_plane_member_state (
                user_id TEXT PRIMARY KEY REFERENCES control_plane_principals(user_id),
                status TEXT NOT NULL CHECK(status IN ('active','disabled')),
                revision INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS control_plane_github_identities (
                subject TEXT PRIMARY KEY,
                organization_id TEXT NOT NULL,
                user_id TEXT,
                login TEXT NOT NULL,
                revision INTEGER NOT NULL,
                FOREIGN KEY(user_id) REFERENCES control_plane_principals(user_id)
            );
            CREATE TABLE IF NOT EXISTS control_plane_resource_owners (
                kind TEXT NOT NULL,
                resource_id TEXT NOT NULL,
                organization_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                device_id TEXT NOT NULL,
                PRIMARY KEY(kind, resource_id)
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
            CREATE TABLE IF NOT EXISTS control_plane_notifications (
                notification_id TEXT PRIMARY KEY,
                organization_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                notification_type TEXT NOT NULL,
                title TEXT NOT NULL,
                body TEXT NOT NULL,
                created_at TEXT NOT NULL,
                read_at TEXT,
                FOREIGN KEY(organization_id) REFERENCES control_plane_organizations(organization_id)
            );
            CREATE INDEX IF NOT EXISTS control_plane_notifications_user
                ON control_plane_notifications(organization_id, user_id, created_at DESC);
            CREATE TABLE IF NOT EXISTS control_plane_identity_sessions (
                session_id TEXT PRIMARY KEY,
                token_hash TEXT NOT NULL UNIQUE,
                organization_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                issued_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                revoked_at TEXT,
                FOREIGN KEY(organization_id) REFERENCES control_plane_organizations(organization_id)
            );
            CREATE INDEX IF NOT EXISTS control_plane_identity_sessions_lookup
                ON control_plane_identity_sessions(organization_id, user_id, expires_at);
            CREATE TABLE IF NOT EXISTS control_plane_idempotency (
                organization_id TEXT NOT NULL,
                principal_user_id TEXT NOT NULL,
                idempotency_key TEXT NOT NULL,
                operation TEXT NOT NULL,
                request_hash TEXT NOT NULL,
                status_code INTEGER NOT NULL,
                response_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY(organization_id, principal_user_id, idempotency_key)
            );
            CREATE TABLE IF NOT EXISTS control_plane_oauth_states (
                state_hash TEXT PRIMARY KEY,
                provider_id TEXT NOT NULL,
                redirect_uri TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                consumed_at TEXT
            );
            CREATE INDEX IF NOT EXISTS control_plane_oauth_states_expiry
                ON control_plane_oauth_states(provider_id, expires_at, consumed_at);
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

    def _member_teams(self, organization_id: str, user_id: str) -> list[str]:
        return [row['team_id'] for row in self._connection.execute(
            'SELECT team_id,member_ids_json FROM control_plane_teams WHERE organization_id=? ORDER BY team_id',
            (organization_id,)) if user_id in json.loads(row['member_ids_json'])]

    def _sync_member_team_projection(self, organization_id: str) -> None:
        # Keep the old column for rollback readers; it has no independent writer.
        for row in self._connection.execute('SELECT user_id FROM control_plane_principals WHERE organization_id=?', (organization_id,)).fetchall():
            self._connection.execute('UPDATE control_plane_principals SET team_ids_json=? WHERE user_id=?',
                (_json(self._member_teams(organization_id, row['user_id'])), row['user_id']))

    def _team_revision(self, team_id: str) -> int:
        row = self._connection.execute('SELECT revision FROM control_plane_team_versions WHERE team_id=?', (team_id,)).fetchone()
        return row['revision'] if row else 0

    def _organization(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "organization_id": str(row["organization_id"]),
            "name": str(row["name"]),
            "owner_user_id": str(row["owner_user_id"]),
            "team_ids": [team['team_id'] for team in self.list_teams(row['organization_id'])],
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
                "SELECT t.*,COALESCE(v.revision,0) AS revision FROM control_plane_teams t "
                "LEFT JOIN control_plane_team_versions v ON v.team_id=t.team_id WHERE t.organization_id=? ORDER BY t.team_id", (organization_id,)
            ).fetchall()
            return [{
                "team_id": str(row["team_id"]),
                "organization_id": str(row["organization_id"]),
                "name": str(row["name"]),
                "member_ids": list(json.loads(row["member_ids_json"])),
                "revision": row['revision'],
            } for row in rows]

    def list_projects(self, organization_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT p.*,COALESCE(v.revision,0) AS revision FROM control_plane_projects p "
                "LEFT JOIN control_plane_project_versions v ON v.project_id=p.project_id "
                "WHERE p.organization_id=? ORDER BY p.project_id", (organization_id,)
            ).fetchall()
            return [{"project_id": str(row["project_id"]), "organization_id": str(row["organization_id"]),
                     "name": str(row["name"]), "description": str(row["description"]),
                     "status": str(row["status"]), "member_ids": list(json.loads(row["member_ids_json"])),
                     "revision": int(row["revision"])} for row in rows]

    def list_principals(self, organization_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT p.*,COALESCE(m.status,'active') AS status,COALESCE(m.revision,0) AS revision "
                "FROM control_plane_principals p LEFT JOIN control_plane_member_state m ON m.user_id=p.user_id "
                "WHERE p.organization_id=? ORDER BY p.user_id", (organization_id,)
            ).fetchall()
            return [{
                "user_id": str(row["user_id"]),
                "organization_id": str(row["organization_id"]),
                "role": str(row["role"]),
                "team_ids": self._member_teams(row['organization_id'], row['user_id']),
                "status": row["status"], "revision": row["revision"],
                "permissions": sorted(ROLE_PERMISSIONS[row["role"]]),
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

    @staticmethod
    def request_hash(payload: Mapping[str, Any]) -> str:
        return hashlib.sha256(_json(payload).encode("utf-8")).hexdigest()

    @staticmethod
    def _token_hash(token: str) -> str:
        return hashlib.sha256(str(token).encode("utf-8")).hexdigest()

    def issue_identity_session(self, *, organization_id: str, user_id: str,
                               ttl_seconds: int = IDENTITY_SESSION_TTL_SECONDS) -> dict[str, Any]:
        """Issue an opaque principal session; only its hash is persisted."""
        if ttl_seconds < 60 or ttl_seconds > 86400:
            raise ValueError("身份会话有效期必须在 60-86400 秒。")
        with self._lock:
            row = self._connection.execute(
                "SELECT user_id, organization_id, role, team_ids_json FROM control_plane_principals "
                "WHERE user_id=? AND organization_id=?", (user_id, organization_id),
            ).fetchone()
            if row is None:
                raise KeyError("principal_not_found")
            if not self.member_active(user_id):
                raise KeyError("principal_disabled")
            issued_at = datetime.now(timezone.utc)
            expires_at = issued_at.timestamp() + ttl_seconds
            issued_text = issued_at.isoformat(timespec="milliseconds")
            expires_text = datetime.fromtimestamp(expires_at, timezone.utc).isoformat(timespec="milliseconds")
            token = secrets.token_urlsafe(32)
            session_id = uuid.uuid4().hex
            self._connection.execute(
                "INSERT INTO control_plane_identity_sessions(session_id,token_hash,organization_id,user_id,issued_at,expires_at,revoked_at) "
                "VALUES(?,?,?,?,?,?,NULL)",
                (session_id, self._token_hash(token), organization_id, user_id, issued_text, expires_text),
            )
            return {
                "session_id": session_id,
                "principal_token": token,
                "organization_id": organization_id,
                "user_id": user_id,
                "role": str(row["role"]),
                "permissions": sorted(ROLE_PERMISSIONS[str(row["role"])]),
                "issued_at": issued_text,
                "expires_at": expires_text,
            }

    def issue_external_identity_session(self, identity) -> dict[str, Any]:
        """Final mapping check and issuance share the unlink transaction lock."""
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                if identity.provider_id == "github":
                    binding = self.github_identity(identity.subject)
                    if (not binding or binding["user_id"] != identity.user_id
                            or binding["organization_id"] != identity.organization_id
                            or binding["revision"] != identity.claims.get("binding_revision")):
                        raise KeyError("external_identity_mapping_changed")
                session = self.issue_identity_session(organization_id=identity.organization_id, user_id=identity.user_id)
                self._connection.execute("COMMIT")
                return session
            except Exception:
                self._connection.execute("ROLLBACK")
                raise

    def authenticate_identity_session(self, token: str) -> dict[str, Any] | None:
        token = str(token or "").strip()
        if not token:
            return None
        now = _now()
        with self._lock:
            row = self._connection.execute(
                "SELECT s.session_id,s.organization_id,s.user_id,s.issued_at,s.expires_at,p.role,p.team_ids_json "
                "FROM control_plane_identity_sessions s JOIN control_plane_principals p "
                "ON p.user_id=s.user_id AND p.organization_id=s.organization_id "
                "WHERE s.token_hash=? AND s.revoked_at IS NULL AND s.expires_at>?",
                (self._token_hash(token), now),
            ).fetchone()
            if row is None or not self.member_active(row["user_id"]):
                return None
            return {
                "session_id": str(row["session_id"]),
                "organization_id": str(row["organization_id"]),
                "user_id": str(row["user_id"]),
                "role": str(row["role"]),
                "team_ids": self._member_teams(row['organization_id'], row['user_id']),
                "permissions": sorted(ROLE_PERMISSIONS[str(row["role"])]),
                "issued_at": str(row["issued_at"]),
                "expires_at": str(row["expires_at"]),
            }

    def revoke_identity_session(self, session_id: str) -> bool:
        with self._lock:
            cursor = self._connection.execute(
                "UPDATE control_plane_identity_sessions SET revoked_at=? WHERE session_id=? AND revoked_at IS NULL",
                (_now(), session_id),
            )
            return cursor.rowcount == 1

    def get_identity_session(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT session_id,organization_id,user_id,issued_at,expires_at,revoked_at "
                "FROM control_plane_identity_sessions WHERE session_id=?", (session_id,),
            ).fetchone()
            if row is None:
                return None
            return {
                "session_id": str(row["session_id"]),
                "organization_id": str(row["organization_id"]),
                "user_id": str(row["user_id"]),
                "issued_at": str(row["issued_at"]),
                "expires_at": str(row["expires_at"]),
                "revoked_at": str(row["revoked_at"]) if row["revoked_at"] else None,
            }

    def revoke_identity_session_idempotent(
        self, *, organization_id: str, principal_user_id: str, idempotency_key: str,
        request_hash: str, session_id: str,
    ) -> tuple[int, dict[str, Any], bool]:
        def write() -> tuple[int, dict[str, Any]]:
            row = self._connection.execute(
                "SELECT organization_id,user_id,revoked_at FROM control_plane_identity_sessions WHERE session_id=?",
                (session_id,),
            ).fetchone()
            if row is None:
                raise KeyError("identity_session_not_found")
            if str(row["organization_id"]) != organization_id:
                raise ValueError("organization_forbidden")
            changed = False
            if row["revoked_at"] is None:
                self._connection.execute(
                    "UPDATE control_plane_identity_sessions SET revoked_at=? WHERE session_id=?",
                    (_now(), session_id),
                )
                changed = True
            return 200, {"session_id": session_id, "revoked": changed}
        return self._run_idempotent(
            organization_id=organization_id,
            principal_user_id=principal_user_id,
            idempotency_key=idempotency_key,
            operation="identity.session.revoke",
            request_hash=request_hash,
            write=write,
        )

    def create_oauth_state(self, *, state_hash: str, provider_id: str,
                           redirect_uri: str, expires_at: str) -> None:
        """Persist a one-time OAuth state without storing the browser secret."""
        if not state_hash or not provider_id or not redirect_uri or not expires_at:
            raise ValueError("OAuth state 字段不能为空。")
        with self._lock:
            self._connection.execute(
                "INSERT INTO control_plane_oauth_states(state_hash,provider_id,redirect_uri,expires_at,created_at,consumed_at) "
                "VALUES(?,?,?,?,?,NULL)",
                (state_hash, provider_id, redirect_uri, expires_at, _now()),
            )

    def consume_oauth_state(self, *, state_hash: str, provider_id: str,
                            now: str | None = None) -> dict[str, str] | None:
        """Atomically consume an unexpired state and return its redirect URI."""
        current = now or _now()
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._connection.execute(
                    "SELECT provider_id,redirect_uri,expires_at,consumed_at FROM control_plane_oauth_states "
                    "WHERE state_hash=?", (state_hash,),
                ).fetchone()
                if row is None or str(row["provider_id"]) != str(provider_id) \
                        or row["consumed_at"] is not None or str(row["expires_at"]) <= current:
                    self._connection.execute("ROLLBACK")
                    return None
                consumed_at = _now()
                self._connection.execute(
                    "UPDATE control_plane_oauth_states SET consumed_at=? WHERE state_hash=? AND consumed_at IS NULL",
                    (consumed_at, state_hash),
                )
                self._connection.execute("COMMIT")
                return {
                    "provider_id": str(row["provider_id"]),
                    "redirect_uri": str(row["redirect_uri"]),
                    "expires_at": str(row["expires_at"]),
                    "consumed_at": consumed_at,
                }
            except Exception:
                self._connection.execute("ROLLBACK")
                raise

    def idempotent_result(self, *, organization_id: str, principal_user_id: str,
                          idempotency_key: str, operation: str, request_hash: str):
        with self._lock:
            row = self._connection.execute(
                "SELECT operation,request_hash,status_code,response_json FROM control_plane_idempotency "
                "WHERE organization_id=? AND principal_user_id=? AND idempotency_key=?",
                (organization_id, principal_user_id, idempotency_key)).fetchone()
            if row is None:
                return None
            if row["operation"] != operation or row["request_hash"] != request_hash:
                raise ValueError("idempotency_conflict")
            return int(row["status_code"]), json.loads(row["response_json"]), True

    def _run_idempotent(self, *, organization_id: str, principal_user_id: str,
                        idempotency_key: str, operation: str, request_hash: str,
                        write: Callable[[], tuple[int, dict[str, Any]]]) -> tuple[int, dict[str, Any], bool]:
        if not idempotency_key or len(idempotency_key) > 128:
            raise ValueError("idempotency_key_required")
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                replay = self.idempotent_result(organization_id=organization_id, principal_user_id=principal_user_id,
                    idempotency_key=idempotency_key, operation=operation, request_hash=request_hash)
                if replay is not None:
                    self._connection.execute("COMMIT")
                    return replay
                status_code, response = write()
                self._connection.execute(
                    "INSERT INTO control_plane_idempotency(organization_id,principal_user_id,idempotency_key,operation,request_hash,status_code,response_json,created_at) VALUES(?,?,?,?,?,?,?,?)",
                    (organization_id, principal_user_id, idempotency_key, operation, request_hash,
                     int(status_code), _json(response), _now()),
                )
                self._connection.execute("COMMIT")
                return int(status_code), response, False
            except Exception:
                self._connection.execute("ROLLBACK")
                raise

    def _require_organization_locked(self, organization_id: str) -> None:
        row = self._connection.execute(
            "SELECT 1 FROM control_plane_organizations WHERE organization_id=?", (organization_id,)
        ).fetchone()
        if row is None:
            raise KeyError("organization_not_found")

    def _validate_member_ids_locked(self, organization_id: str, member_ids: tuple[str, ...]) -> None:
        if len(set(member_ids)) != len(member_ids):
            raise ValueError("duplicate_member_ids")
        for member_id in member_ids:
            row = self._connection.execute(
                "SELECT 1 FROM control_plane_principals WHERE organization_id=? AND user_id=?",
                (organization_id, member_id),
            ).fetchone()
            if row is None:
                raise ValueError("member_not_found")

    def create_team(self, *, organization_id: str, principal_user_id: str, idempotency_key: str,
                    request_hash: str, team_id: str, name: str, member_ids: tuple[str, ...]) -> tuple[int, dict[str, Any], bool]:
        def write() -> tuple[int, dict[str, Any]]:
            self._require_organization_locked(organization_id)
            self._authorize_team_write(organization_id, principal_user_id)
            team = ControlPlaneTeam(team_id=team_id, organization_id=organization_id, name=name, member_ids=member_ids)
            self._validate_member_ids_locked(organization_id, team.member_ids)
            exists = self._connection.execute("SELECT 1 FROM control_plane_teams WHERE team_id=?", (team.team_id,)).fetchone()
            if exists is not None:
                raise ValueError("team_already_exists")
            self._connection.execute(
                "INSERT INTO control_plane_teams(team_id,organization_id,name,member_ids_json) VALUES(?,?,?,?)",
                (team.team_id, organization_id, team.name, _json(list(team.member_ids))),
            )
            organization = self._connection.execute(
                "SELECT team_ids_json FROM control_plane_organizations WHERE organization_id=?", (organization_id,)
            ).fetchone()
            team_ids = list(json.loads(organization["team_ids_json"]))
            if team.team_id not in team_ids:
                team_ids.append(team.team_id)
                self._connection.execute(
                    "UPDATE control_plane_organizations SET team_ids_json=? WHERE organization_id=?",
                    (_json(team_ids), organization_id),
                )
            self._sync_member_team_projection(organization_id)
            self.record_audit(organization_id=organization_id, resource='team.create', outcome='created',
                              actor=principal_user_id, metadata={'team_id': team_id, 'member_ids': list(team.member_ids), 'revision': 0})
            return 201, {"team": {"team_id": team.team_id, "organization_id": organization_id,
                                    "name": team.name, "member_ids": list(team.member_ids), "revision": 0}}
        return self._run_idempotent(organization_id=organization_id, principal_user_id=principal_user_id,
                                    idempotency_key=idempotency_key, operation="team.create",
                                    request_hash=request_hash, write=write)

    def update_team(self, *, organization_id: str, principal_user_id: str, idempotency_key: str,
                    request_hash: str, team_id: str, name: str | None,
                    member_ids: tuple[str, ...] | None, expected_revision: int | None = None) -> tuple[int, dict[str, Any], bool]:
        def write() -> tuple[int, dict[str, Any]]:
            self._require_organization_locked(organization_id)
            self._authorize_team_write(organization_id, principal_user_id)
            row = self._connection.execute(
                "SELECT name,member_ids_json FROM control_plane_teams WHERE team_id=? AND organization_id=?",
                (team_id, organization_id),
            ).fetchone()
            if row is None:
                raise KeyError("team_not_found")
            revision = self._team_revision(team_id)
            if expected_revision is not None and revision != expected_revision:
                raise ValueError('team_conflict')
            final_name = name if name is not None else str(row["name"])
            final_members = tuple(member_ids) if member_ids is not None else tuple(json.loads(row["member_ids_json"]))
            team = ControlPlaneTeam(team_id=team_id, organization_id=organization_id, name=final_name, member_ids=final_members)
            self._validate_member_ids_locked(organization_id, team.member_ids)
            self._connection.execute(
                "UPDATE control_plane_teams SET name=?,member_ids_json=? WHERE team_id=? AND organization_id=?",
                (team.name, _json(list(team.member_ids)), team_id, organization_id),
            )
            self._connection.execute('INSERT OR REPLACE INTO control_plane_team_versions VALUES(?,?)', (team_id, revision+1))
            self._sync_member_team_projection(organization_id)
            self.record_audit(organization_id=organization_id, resource='team.update', outcome='updated', actor=principal_user_id,
                metadata={'team_id': team_id, 'old_name': row['name'], 'name': team.name,
                          'old_member_ids': json.loads(row['member_ids_json']), 'member_ids': list(team.member_ids), 'revision': revision+1})
            return 200, {"team": {"team_id": team.team_id, "organization_id": organization_id,
                                    "name": team.name, "member_ids": list(team.member_ids), "revision": revision+1}}
        return self._run_idempotent(organization_id=organization_id, principal_user_id=principal_user_id,
                                    idempotency_key=idempotency_key, operation="team.update",
                                    request_hash=request_hash, write=write)

    def _authorize_team_write(self, organization_id: str, user_id: str) -> None:
        row = self._connection.execute('SELECT role FROM control_plane_principals WHERE organization_id=? AND user_id=?',
                                       (organization_id, user_id)).fetchone()
        if not row or not self.member_active(user_id) or 'team.manage' not in ROLE_PERMISSIONS[row['role']]:
            raise ValueError('team_permission_denied')

    def _validate_project_members_locked(self, organization_id: str, member_ids: tuple[str, ...]) -> None:
        if len(set(member_ids)) != len(member_ids):
            raise ValueError('duplicate_project_member_ids')
        for user_id in member_ids:
            row = self._connection.execute('SELECT 1 FROM control_plane_principals WHERE organization_id=? AND user_id=?',
                                           (organization_id, user_id)).fetchone()
            if row is None:
                raise ValueError('project_member_not_found')

    def create_project(self, *, organization_id: str, principal_user_id: str, idempotency_key: str,
                       request_hash: str, project_id: str, name: str, description: str,
                       member_ids: tuple[str, ...]) -> tuple[int, dict[str, Any], bool]:
        def write() -> tuple[int, dict[str, Any]]:
            self._require_organization_locked(organization_id)
            self._authorize_team_write(organization_id, principal_user_id)
            if self._connection.execute('SELECT 1 FROM control_plane_projects WHERE project_id=?', (project_id,)).fetchone():
                raise ValueError('project_already_exists')
            if not str(name or '').strip():
                raise ValueError('project_name_required')
            self._validate_project_members_locked(organization_id, member_ids)
            self._connection.execute('INSERT INTO control_plane_projects VALUES(?,?,?,?,?,?)',
                (project_id, organization_id, name.strip(), str(description or '').strip(), 'active', _json(list(member_ids))))
            self.record_audit(organization_id=organization_id, resource='project.create', outcome='created', actor=principal_user_id,
                              metadata={'project_id': project_id, 'name': name.strip(), 'member_ids': list(member_ids), 'revision': 0})
            return 201, {'project': {'project_id': project_id, 'organization_id': organization_id, 'name': name.strip(),
                                     'description': str(description or '').strip(), 'status': 'active', 'member_ids': list(member_ids), 'revision': 0}}
        return self._run_idempotent(organization_id=organization_id, principal_user_id=principal_user_id,
            idempotency_key=idempotency_key, operation='project.create', request_hash=request_hash, write=write)

    def update_project(self, *, organization_id: str, principal_user_id: str, idempotency_key: str,
                       request_hash: str, project_id: str, name: str | None, description: str | None,
                       status: str | None, member_ids: tuple[str, ...] | None,
                       expected_revision: int | None = None) -> tuple[int, dict[str, Any], bool]:
        def write() -> tuple[int, dict[str, Any]]:
            self._require_organization_locked(organization_id)
            self._authorize_team_write(organization_id, principal_user_id)
            row = self._connection.execute('SELECT * FROM control_plane_projects WHERE project_id=? AND organization_id=?',
                                           (project_id, organization_id)).fetchone()
            if row is None:
                raise KeyError('project_not_found')
            revision = self._connection.execute('SELECT revision FROM control_plane_project_versions WHERE project_id=?', (project_id,)).fetchone()
            revision = int(revision['revision']) if revision else 0
            if expected_revision is not None and revision != expected_revision:
                raise ValueError('project_conflict')
            final_name = str(name).strip() if name is not None else str(row['name'])
            final_description = str(description).strip() if description is not None else str(row['description'])
            final_status = status if status is not None else str(row['status'])
            final_members = tuple(member_ids) if member_ids is not None else tuple(json.loads(row['member_ids_json']))
            if not final_name:
                raise ValueError('project_name_required')
            if final_status not in {'active', 'archived'}:
                raise ValueError('project_status_invalid')
            self._validate_project_members_locked(organization_id, final_members)
            self._connection.execute('UPDATE control_plane_projects SET name=?,description=?,status=?,member_ids_json=? WHERE project_id=? AND organization_id=?',
                (final_name, final_description, final_status, _json(list(final_members)), project_id, organization_id))
            next_revision = revision + 1
            self._connection.execute('INSERT OR REPLACE INTO control_plane_project_versions VALUES(?,?)', (project_id, next_revision))
            self.record_audit(organization_id=organization_id, resource='project.update', outcome='updated', actor=principal_user_id,
                              metadata={'project_id': project_id, 'old_name': row['name'], 'name': final_name,
                                        'old_status': row['status'], 'status': final_status,
                                        'old_member_ids': json.loads(row['member_ids_json']), 'member_ids': list(final_members), 'revision': next_revision})
            return 200, {'project': {'project_id': project_id, 'organization_id': organization_id, 'name': final_name,
                                     'description': final_description, 'status': final_status, 'member_ids': list(final_members), 'revision': next_revision}}
        return self._run_idempotent(organization_id=organization_id, principal_user_id=principal_user_id,
            idempotency_key=idempotency_key, operation='project.update', request_hash=request_hash, write=write)

    def create_principal(self, *, organization_id: str, principal_user_id: str, idempotency_key: str,
                         request_hash: str, user_id: str, role: str, team_ids: tuple[str, ...]) -> tuple[int, dict[str, Any], bool]:
        def write() -> tuple[int, dict[str, Any]]:
            self._require_organization_locked(organization_id)
            principal = ControlPlanePrincipal(user_id=user_id, organization_id=organization_id, role=role, team_ids=team_ids)
            self._authorize_member_change(organization_id, principal_user_id, None, role)
            if self._connection.execute("SELECT 1 FROM control_plane_principals WHERE user_id=?", (user_id,)).fetchone() is not None:
                raise ValueError("principal_already_exists")
            for team_id in principal.team_ids:
                if self._connection.execute("SELECT 1 FROM control_plane_teams WHERE team_id=? AND organization_id=?", (team_id, organization_id)).fetchone() is None:
                    raise ValueError("team_not_found")
            self._connection.execute(
                "INSERT INTO control_plane_principals(user_id,organization_id,role,team_ids_json) VALUES(?,?,?,?)",
                (user_id, organization_id, role, _json(list(principal.team_ids))),
            )
            for team_id in principal.team_ids:
                members = json.loads(self._connection.execute('SELECT member_ids_json FROM control_plane_teams WHERE team_id=?', (team_id,)).fetchone()[0])
                self._connection.execute('UPDATE control_plane_teams SET member_ids_json=? WHERE team_id=?', (_json(members + [user_id]), team_id))
                self._connection.execute('INSERT OR REPLACE INTO control_plane_team_versions VALUES(?,?)', (team_id, self._team_revision(team_id)+1))
            self.record_audit(organization_id=organization_id, resource="principal.create", outcome="created",
                actor=principal_user_id, metadata={"user_id": user_id, "role": role})
            return 201, {"principal": {"user_id": user_id, "organization_id": organization_id,
                                        "role": role, "team_ids": list(principal.team_ids),
                                        "permissions": sorted(principal.permissions())}}
        return self._run_idempotent(organization_id=organization_id, principal_user_id=principal_user_id,
                                    idempotency_key=idempotency_key, operation="principal.create",
                                    request_hash=request_hash, write=write)

    def member_active(self, user_id: str) -> bool:
        with self._lock:
            row = self._connection.execute("SELECT status FROM control_plane_member_state WHERE user_id=?", (user_id,)).fetchone()
            return row is None or row["status"] == "active"

    def _authorize_member_change(self, organization_id, actor_id, target_role, role):
        actor = self._connection.execute(
            "SELECT role FROM control_plane_principals WHERE organization_id=? AND user_id=?",
            (organization_id, actor_id)).fetchone()
        if (not actor or not self.member_active(actor_id)
                or role not in manageable_member_roles(actor["role"], target_role)):
            raise ValueError("member_permission_denied")

    def update_principal(self, *, organization_id: str, principal_user_id: str, user_id: str,
                         role: str, status: str, expected_revision: int, idempotency_key: str,
                         unavailable_reason: str = "") -> tuple[int, dict[str, Any], bool]:
        payload = dict(user_id=user_id, role=role, status=status, expected_revision=expected_revision)
        def write():
            members = self.list_principals(organization_id)
            old = next((m for m in members if m["user_id"] == user_id), None)
            if old is None:
                raise KeyError("principal_not_found")
            self._authorize_member_change(organization_id, principal_user_id, old["role"], role)
            if status not in {"active", "disabled"}:
                raise ValueError("member_status_invalid")
            if old["revision"] != expected_revision:
                raise ValueError("member_conflict")
            other_owners = [m for m in members if m["user_id"] != user_id and m["role"] == "owner" and m["status"] == "active"]
            if old["role"] == "owner" and old["status"] == "active" and (role != "owner" or status != "active") and not other_owners:
                raise ValueError("last_owner_required")
            if (old["role"] == "owner" and old["status"] == "active" and (role != "owner" or status != "active")
                    and self._linked_owner_exists(organization_id) and not self._linked_owner_exists(organization_id, exclude_user=user_id)):
                raise ValueError("last_linked_owner_required")
            if status == "disabled" and old["status"] == "active" and unavailable_reason:
                raise ValueError("member_has_active_work")
            self._connection.execute("UPDATE control_plane_principals SET role=? WHERE user_id=? AND organization_id=?",
                                     (role, user_id, organization_id))
            self._connection.execute("INSERT INTO control_plane_member_state VALUES(?,?,?) ON CONFLICT(user_id) DO UPDATE SET status=excluded.status,revision=excluded.revision",
                                     (user_id, status, expected_revision + 1))
            if status == "disabled":
                self._connection.execute("UPDATE control_plane_identity_sessions SET revoked_at=? WHERE user_id=? AND organization_id=? AND revoked_at IS NULL",
                                         (_now(), user_id, organization_id))
            # Keep the organization's designated owner consistent with its active owners.
            if (role != "owner" or status != "active") and other_owners:
                self._connection.execute("UPDATE control_plane_organizations SET owner_user_id=? WHERE organization_id=? AND owner_user_id=?",
                                         (other_owners[0]["user_id"], organization_id, user_id))
            self.record_audit(organization_id=organization_id, resource="principal.update", outcome="updated",
                actor=principal_user_id, metadata={"user_id": user_id, "old_role": old["role"], "role": role,
                    "old_status": old["status"], "status": status, "revision": expected_revision + 1})
            updated = next(m for m in self.list_principals(organization_id) if m["user_id"] == user_id)
            return 200, {"principal": updated}
        return self._run_idempotent(organization_id=organization_id, principal_user_id=principal_user_id,
            idempotency_key=idempotency_key, operation="principal.update", request_hash=self.request_hash(payload), write=write)

    def github_mappings_initialized(self) -> bool:
        with self._lock:
            return self._connection.execute("SELECT 1 FROM control_plane_meta WHERE key='github_identity_authority_v1'").fetchone() is not None

    def import_github_mappings(self, mappings: Mapping[str, Mapping[str, str]]) -> bool:
        """One-time, atomic import. A retained legacy file can never rebind a user."""
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                if self.github_mappings_initialized():
                    self._connection.execute("COMMIT")
                    return False
                for subject, target in mappings.items():
                    validate_github_subject(subject)
                    if not self._connection.execute("SELECT 1 FROM control_plane_principals WHERE organization_id=? AND user_id=?",
                            (target["organization_id"], target["user_id"])).fetchone():
                        raise ValueError("github_migration_member_not_found")
                    self._connection.execute("INSERT INTO control_plane_github_identities VALUES(?,?,?,?,0)",
                        (subject, target["organization_id"], target["user_id"], ""))
                self._connection.execute("INSERT INTO control_plane_meta VALUES('github_identity_authority_v1',?)", (_now(),))
                for organization in {m["organization_id"] for m in mappings.values()}:
                    self.record_audit(organization_id=organization, resource="github_identity.migrate", outcome="imported",
                        actor="installation", metadata={"count": sum(m["organization_id"] == organization for m in mappings.values())})
                self._connection.execute("COMMIT")
                return True
            except Exception:
                self._connection.execute("ROLLBACK")
                raise

    def github_identity(self, subject: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute("SELECT * FROM control_plane_github_identities WHERE subject=?", (subject,)).fetchone()
            return dict(row) if row else None

    def list_github_identities(self, organization_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._connection.execute(
                "SELECT g.*,p.role,COALESCE(m.status,'active') AS status FROM control_plane_github_identities g "
                "JOIN control_plane_principals p ON p.user_id=g.user_id AND p.organization_id=g.organization_id "
                "LEFT JOIN control_plane_member_state m ON m.user_id=p.user_id "
                "WHERE g.organization_id=? ORDER BY g.user_id,g.subject", (organization_id,))]

    def _linked_owner_exists(self, organization_id, *, exclude_subject="", exclude_user="") -> bool:
        return self._connection.execute(
            "SELECT 1 FROM control_plane_github_identities g JOIN control_plane_principals p "
            "ON p.user_id=g.user_id AND p.organization_id=g.organization_id "
            "LEFT JOIN control_plane_member_state m ON m.user_id=p.user_id "
            "WHERE p.organization_id=? AND p.role='owner' AND COALESCE(m.status,'active')='active' "
            "AND g.subject<>? AND p.user_id<>? LIMIT 1", (organization_id, exclude_subject, exclude_user)).fetchone() is not None

    def update_github_identity(self, *, organization_id: str, principal_user_id: str, subject: str,
                               user_id: str | None, expected_revision: int, idempotency_key: str,
                               login: str = "") -> tuple[int, dict[str, Any], bool]:
        validate_github_subject(subject)
        payload = dict(subject=subject, user_id=user_id, expected_revision=expected_revision)
        def write():
            old = self.github_identity(subject)
            if old and old["user_id"] and (old["organization_id"] != organization_id or user_id not in {None, old["user_id"]}):
                raise ValueError("github_identity_in_use")
            revision = old["revision"] if old else 0
            if revision != expected_revision:
                raise ValueError("github_identity_conflict")
            target_id = user_id or (old["user_id"] if old and old["organization_id"] == organization_id else None)
            member = next((m for m in self.list_principals(organization_id) if m["user_id"] == target_id), None)
            if not member:
                raise KeyError("principal_not_found")
            self._authorize_member_change(organization_id, principal_user_id, member["role"], member["role"])
            if user_id is None and member["role"] == "owner" and member["status"] == "active" and not self._linked_owner_exists(organization_id, exclude_subject=subject):
                raise ValueError("last_linked_owner_required")
            self._connection.execute("INSERT INTO control_plane_github_identities VALUES(?,?,?,?,?) "
                "ON CONFLICT(subject) DO UPDATE SET organization_id=excluded.organization_id,user_id=excluded.user_id,login=excluded.login,revision=excluded.revision",
                (subject, organization_id, user_id, login if user_id else old["login"], revision+1))
            if user_id is None:
                self._connection.execute("UPDATE control_plane_identity_sessions SET revoked_at=? WHERE organization_id=? AND user_id=? AND revoked_at IS NULL",
                    (_now(), organization_id, target_id))
            self.record_audit(organization_id=organization_id, resource="github_identity.update", outcome="linked" if user_id else "unlinked",
                actor=principal_user_id, metadata={"subject": subject, "user_id": target_id, "revision": revision+1})
            return 200, {"identity": self.github_identity(subject)}
        return self._run_idempotent(organization_id=organization_id, principal_user_id=principal_user_id,
            idempotency_key=idempotency_key, operation="github_identity.update", request_hash=self.request_hash(payload), write=write)

    def upsert_quota(self, *, organization_id: str, principal_user_id: str, idempotency_key: str,
                     request_hash: str, quota_key: str, period: str, max_runs: int | None,
                     max_physical_actions: int | None, max_model_tokens: int | None) -> tuple[int, dict[str, Any], bool]:
        def write() -> tuple[int, dict[str, Any]]:
            self._require_organization_locked(organization_id)
            quota = QuotaLimit(period=period, max_runs=max_runs, max_physical_actions=max_physical_actions, max_model_tokens=max_model_tokens)
            self._connection.execute(
                "INSERT INTO control_plane_quotas(organization_id,quota_key,period,max_runs,max_physical_actions,max_model_tokens) VALUES(?,?,?,?,?,?) "
                "ON CONFLICT(organization_id,quota_key) DO UPDATE SET period=excluded.period,max_runs=excluded.max_runs,max_physical_actions=excluded.max_physical_actions,max_model_tokens=excluded.max_model_tokens",
                (organization_id, quota_key, quota.period, quota.max_runs, quota.max_physical_actions, quota.max_model_tokens),
            )
            return 200, {"quota": {"quota_key": quota_key, "organization_id": organization_id, "period": quota.period,
                                    "max_runs": quota.max_runs, "max_physical_actions": quota.max_physical_actions,
                                    "max_model_tokens": quota.max_model_tokens,
                                    "usage": {"runs": 0, "physical_actions": 0, "model_tokens": 0}}}
        return self._run_idempotent(organization_id=organization_id, principal_user_id=principal_user_id,
                                    idempotency_key=idempotency_key, operation="quota.update",
                                    request_hash=request_hash, write=write)

    def record_audit(self, *, organization_id: str, resource: str, outcome: str,
                     request_id: str | None = None, metadata: Mapping[str, Any] | None = None,
                     actor: str = "local-control-token") -> int:
        safe_metadata = {
            str(key): value for key, value in (metadata or {}).items()
            if str(key).lower() not in {"token", "authorization", "password", "secret", "cookie"}
        }
        with self._lock:
            cursor = self._connection.execute(
                "INSERT INTO control_plane_audit(occurred_at,request_id,organization_id,resource,outcome,actor,metadata_json) VALUES(?,?,?,?,?,?,?)",
                (_now(), request_id or uuid.uuid4().hex, organization_id, resource, outcome, actor[:128], _json(safe_metadata)),
            )
            # Notifications are a small user-scoped projection of material
            # writes. Never copy request metadata or authentication material.
            if actor and actor != "local-control-token" and outcome in {
                "created", "updated", "assigned", "unassigned", "linked", "unlinked",
                "disabled", "enabled", "revoked", "archived", "restored",
            }:
                title = f"{resource} · {outcome}"
                body = "组织资源状态已更新，可在对应管理页面查看详情。"
                self._connection.execute(
                    "INSERT INTO control_plane_notifications(notification_id,organization_id,user_id,notification_type,title,body,created_at) VALUES(?,?,?,?,?,?,?)",
                    (uuid.uuid4().hex, organization_id, actor[:128], "control_plane_change", title[:256], body, _now()),
                )
            return int(cursor.lastrowid)

    def list_notifications(self, organization_id: str, user_id: str, *, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM control_plane_notifications WHERE organization_id=? AND user_id=? "
                "ORDER BY created_at DESC, notification_id DESC LIMIT ?",
                (organization_id, user_id, int(limit)),
            ).fetchall()
            return [{
                "notification_id": str(row["notification_id"]),
                "organization_id": str(row["organization_id"]),
                "user_id": str(row["user_id"]),
                "notification_type": str(row["notification_type"]),
                "title": str(row["title"]),
                "body": str(row["body"]),
                "created_at": str(row["created_at"]),
                "read_at": row["read_at"],
            } for row in rows]

    def mark_notification_read(self, organization_id: str, user_id: str, notification_id: str) -> bool:
        with self._lock:
            cursor = self._connection.execute(
                "UPDATE control_plane_notifications SET read_at=COALESCE(read_at, ?) "
                "WHERE notification_id=? AND organization_id=? AND user_id=?",
                (_now(), notification_id, organization_id, user_id),
            )
            return cursor.rowcount == 1

    def read_audit(self, organization_id: str, *, after_sequence: int = 0, limit: int = 50,
                   actor: str | None = None) -> dict[str, Any]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM control_plane_audit WHERE organization_id=? AND sequence>? "
                + ("AND actor=? " if actor is not None else "") + "ORDER BY sequence ASC LIMIT ?",
                (organization_id, int(after_sequence), *((actor,) if actor is not None else ()), int(limit) + 1),
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

    def device_assignments(self, organization_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT r.resource_id AS device_id,r.organization_id,r.user_id,COALESCE(v.revision,0) AS revision "
                "FROM control_plane_resource_owners r LEFT JOIN control_plane_device_assignment_versions v "
                "ON v.device_id=r.resource_id WHERE r.kind='device' AND r.organization_id=? ORDER BY r.resource_id",
                (organization_id,)).fetchall()
            return [{**dict(row), "user_id": row["user_id"] or None} for row in rows]

    def assign_device(self, *, organization_id: str, principal_user_id: str, device_id: str,
                      user_id: str | None, expected_revision: int, idempotency_key: str,
                      unavailable_reason: str = "") -> tuple[int, dict[str, Any], bool]:
        payload = {"device_id": device_id, "user_id": user_id, "expected_revision": expected_revision}
        def write():
            row = next((x for x in self.device_assignments(organization_id) if x["device_id"] == device_id), None)
            if row is None:
                raise KeyError("device_not_found")
            if user_id is not None and (not self.member_active(user_id) or not self._connection.execute(
                "SELECT 1 FROM control_plane_principals WHERE organization_id=? AND user_id=?", (organization_id, user_id)
            ).fetchone()):
                raise ValueError("assignee_not_found")
            if row["revision"] != expected_revision:
                raise ValueError("assignment_conflict")
            if unavailable_reason:
                raise ValueError(unavailable_reason)
            revision = row["revision"] + 1
            self._connection.execute("UPDATE control_plane_resource_owners SET user_id=? WHERE kind='device' AND resource_id=? AND organization_id=?",
                                     (user_id or "", device_id, organization_id))
            self._connection.execute("INSERT INTO control_plane_device_assignment_versions VALUES(?,?) ON CONFLICT(device_id) DO UPDATE SET revision=excluded.revision", (device_id, revision))
            self.record_audit(organization_id=organization_id, resource="device.assign", outcome="assigned" if user_id else "unassigned",
                              request_id=idempotency_key, actor=principal_user_id,
                              metadata={"device_id": device_id, "previous_user_id": row["user_id"], "user_id": user_id, "revision": revision})
            return 200, {"assignment": {**row, "user_id": user_id, "revision": revision}}
        return self._run_idempotent(organization_id=organization_id, principal_user_id=principal_user_id,
            idempotency_key=idempotency_key, operation="device.assign", request_hash=self.request_hash(payload), write=write)

    def claim_resource(self, kind: str, resource_id: str, principal: Mapping[str, Any], *, device_id: str = "") -> None:
        """Insert immutable ownership before dispatch. Never claim on a read."""
        owner = (principal["organization_id"], principal["user_id"], device_id)
        with self._lock:
            self._connection.execute(
                "INSERT OR IGNORE INTO control_plane_resource_owners VALUES(?,?,?,?,?)",
                (kind, resource_id, *owner))
            row = self._connection.execute(
                "SELECT organization_id,user_id,device_id FROM control_plane_resource_owners WHERE kind=? AND resource_id=?",
                (kind, resource_id)).fetchone()
            if tuple(row) != owner:
                raise ValueError("resource_owner_conflict")

    def owned_resource(self, kind: str, resource_id: str, principal: Mapping[str, Any]) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM control_plane_resource_owners WHERE kind=? AND resource_id=? AND organization_id=? AND user_id=?",
                (kind, resource_id, principal["organization_id"], principal["user_id"])).fetchone()
            return dict(row) if row else None

    def website_ownership_initialized(self) -> bool:
        with self._lock:
            return self._connection.execute("SELECT 1 FROM control_plane_meta WHERE key='website_ownership_v1'").fetchone() is not None

    def initialize_website_ownership(self, default_device_id: str, *, legacy_resources=()) -> None:
        """One-time upgrade: preserve this single-user installation's history.

        No unknown future resource is ever assigned to the owner on access.
        The marker and snapshot migration commit atomically.
        """
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                if not self._connection.execute("SELECT 1 FROM control_plane_meta WHERE key='website_ownership_v1'").fetchone():
                    self._connection.execute("INSERT OR IGNORE INTO control_plane_resource_owners VALUES('device',?,'local-org','local-owner',?)",
                                             (default_device_id, default_device_id))
                    if self._connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='runtime_sessions'").fetchone():
                        self._connection.execute("INSERT OR IGNORE INTO control_plane_resource_owners SELECT 'session',session_id,'local-org','local-owner',device_id FROM runtime_sessions")
                    for kind, resource_id, device_id in legacy_resources:
                        self._connection.execute("INSERT OR IGNORE INTO control_plane_resource_owners VALUES(?,?,'local-org','local-owner',?)",
                                                 (kind, resource_id, device_id))
                    self._connection.execute("INSERT INTO control_plane_meta VALUES('website_ownership_v1',?)", (_now(),))
                self._connection.execute("COMMIT")
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise

    def close(self) -> None:
        with self._lock:
            self._connection.close()


__all__ = ["CONTROL_PLANE_SCHEMA_VERSION", "IDENTITY_SESSION_TTL_SECONDS", "LOCAL_ORGANIZATION_ID", "SqliteControlPlaneRepository"]
