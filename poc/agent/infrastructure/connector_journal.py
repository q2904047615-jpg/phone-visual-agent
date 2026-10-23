"""Windows delivery journal; records transport effects, never Task/Run state."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from .remote_connector import ConnectorProtocolError, _canonical, result_payload


class ConnectorJournal:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS deliveries (
                command_id TEXT PRIMARY KEY, command_json TEXT NOT NULL, digest TEXT NOT NULL,
                result_json TEXT, acknowledged INTEGER NOT NULL DEFAULT 0)""")
            db.execute("CREATE TABLE IF NOT EXISTS used_captures (capture_id TEXT PRIMARY KEY)")
            db.execute("CREATE TABLE IF NOT EXISTS used_tool_scopes (device_id TEXT NOT NULL, nonce TEXT NOT NULL, PRIMARY KEY(device_id,nonce))")

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
