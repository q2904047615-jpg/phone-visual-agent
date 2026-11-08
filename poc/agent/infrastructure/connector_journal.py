"""Windows delivery journal; records transport effects, never Task/Run state."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from .remote_connector import ConnectorProtocolError, _canonical, result_payload

#: Acknowledged deliveries beyond the newest this-many rows are pruned.
#: Replay protection only needs recent command ids (the cloud expires a command
#: within seconds) and a capture is consumed in the same observe/act cycle, so
#: older rows only cost time: every poll used to scan all of them, including
#: every preview image. Unfinished or unacknowledged rows are never pruned.
KEEP_RECENT_DELIVERIES = 5000
PRUNE_EVERY_FINISHES = 500


class ConnectorJournal:
    def __init__(self, path: Path, *, keep_recent: int = KEEP_RECENT_DELIVERIES):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.keep_recent = int(keep_recent)
        self._finished_since_prune = 0
        with self.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS deliveries (
                command_id TEXT PRIMARY KEY, command_json TEXT NOT NULL, digest TEXT NOT NULL,
                result_json TEXT, acknowledged INTEGER NOT NULL DEFAULT 0)""")
            db.execute("CREATE TABLE IF NOT EXISTS used_captures (capture_id TEXT PRIMARY KEY)")
            db.execute("CREATE TABLE IF NOT EXISTS used_tool_scopes (device_id TEXT NOT NULL, nonce TEXT NOT NULL, PRIMARY KEY(device_id,nonce))")
        self.prune(compact=True)
        with self.connect() as db:
            # Partial indexes matching the two per-poll queries in pending(),
            # so each poll reads only the rows it needs.
            db.execute("CREATE INDEX IF NOT EXISTS deliveries_unfinished ON deliveries(command_id) "
                       "WHERE result_json IS NULL")
            db.execute("CREATE INDEX IF NOT EXISTS deliveries_unacknowledged ON deliveries(command_id) "
                       "WHERE acknowledged=0 AND result_json IS NOT NULL")

    def prune(self, *, compact: bool = False) -> int:
        """Delete old acknowledged deliveries; keep the latest machine-position record."""
        with self.connect() as db:
            top = db.execute("SELECT max(rowid) FROM deliveries").fetchone()[0]
            if top is None:
                return 0
            latest_position = db.execute(
                "SELECT max(rowid) FROM deliveries "
                "WHERE json_extract(command_json,'$.operation')='select_machine_position'").fetchone()[0]
            removed = db.execute(
                "DELETE FROM deliveries WHERE rowid<=? AND acknowledged=1 AND result_json IS NOT NULL AND rowid<>?",
                (top - self.keep_recent, latest_position if latest_position is not None else -1)).rowcount
            if removed:
                # A pruned capture cannot be consumed any more anyway ("观察帧在本机不存在").
                db.execute("DELETE FROM used_captures WHERE capture_id NOT IN (SELECT command_id FROM deliveries)")
        if compact and removed:
            db = sqlite3.connect(self.path, timeout=10, isolation_level=None)
            try:
                pages, free = (db.execute(f"PRAGMA {name}").fetchone()[0] for name in ("page_count", "freelist_count"))
                if pages and free * 2 >= pages:
                    db.execute("VACUUM")
            finally:
                db.close()
        return removed

    def consume_tool_scope(self, scope):
        with self.connect() as db:
            try:
                db.execute("INSERT INTO used_tool_scopes VALUES (?,?)", (scope.device_id, scope.nonce))
            except sqlite3.IntegrityError as exc:
                raise ConnectorProtocolError("文字 scope 已消费，不能重放。", attempted=False) from exc

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        try:
            with db:
                yield db
        finally:
            db.close()

    def begin(self, command: dict) -> dict | None:
        encoded = _canonical(command).decode()
        digest = hashlib.sha256(encoded.encode()).hexdigest()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT digest,result_json FROM deliveries WHERE command_id=?", (command["command_id"],)).fetchone()
            if row:
                if row[0] != digest:
                    raise ConnectorProtocolError("命令 ID 与本机记录冲突。")
                if row[1] is not None:
                    return json.loads(row[1])
                payload = result_payload(command_id=command["command_id"], connector_id=command["connector_id"],
                                         device_id=command["device_id"], ok=False, error="execution_unknown_after_restart")
                db.execute("UPDATE deliveries SET result_json=? WHERE command_id=?", (_canonical(payload).decode(), command["command_id"]))
                return payload
            db.execute("INSERT INTO deliveries(command_id,command_json,digest) VALUES (?,?,?)", (command["command_id"], encoded, digest))
        return None

    def finish(self, payload: dict):
        with self.connect() as db:
            db.execute("UPDATE deliveries SET result_json=? WHERE command_id=? AND result_json IS NULL",
                       (_canonical(payload).decode(), payload["command_id"]))
        self._finished_since_prune += 1
        if self._finished_since_prune >= PRUNE_EVERY_FINISHES:
            self._finished_since_prune = 0
            self.prune()

    def pending(self) -> list[dict]:
        # Caller holds the single connector OS lease during crash recovery.
        with self.connect() as db:
            started = db.execute("SELECT command_json FROM deliveries WHERE result_json IS NULL").fetchall()
        for row in started:
            self.begin(json.loads(row[0]))
        with self.connect() as db:
            return [json.loads(row[0]) for row in db.execute(
                "SELECT result_json FROM deliveries WHERE acknowledged=0 AND result_json IS NOT NULL LIMIT 20")]

    def acknowledge(self, command_id: str):
        with self.connect() as db:
            db.execute("UPDATE deliveries SET acknowledged=1 WHERE command_id=?", (command_id,))

    def last_confirmed_machine_position(self, *, connector_id, device_id):
        """Restore state without issuing another selection after a restart."""
        with self.connect() as db:
            rows = db.execute("SELECT command_json,result_json FROM deliveries ORDER BY rowid DESC")
            for raw_command, raw_result in rows:
                command = json.loads(raw_command)
                if (command["operation"] != "select_machine_position" or command["connector_id"] != connector_id
                        or command["device_id"] != device_id):
                    continue
                result = json.loads(raw_result) if raw_result else None
                if result and result.get("ok") is True:
                    return command["args"]["machine_position"]
                return None
        return None

    def consume_capture(self, capture_id: str, *, connector_id: str, device_id: str) -> dict:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT command_json,result_json FROM deliveries WHERE command_id=?", (capture_id,)).fetchone()
            if not row or row[1] is None:
                raise ConnectorProtocolError("观察帧在本机不存在。")
            command, result = json.loads(row[0]), json.loads(row[1])
            if (command["operation"] != "vision_capture" or command["device_id"] != device_id
                    or command["connector_id"] != connector_id or result["ok"] is not True):
                raise ConnectorProtocolError("观察帧归属无效。")
            try:
                db.execute("INSERT INTO used_captures VALUES (?)", (capture_id,))
            except sqlite3.IntegrityError as exc:
                raise ConnectorProtocolError("该观察对应动作已领取，不能重放。") from exc
            return result["value"]
