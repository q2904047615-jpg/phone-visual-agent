"""Authenticated command bridge between the cloud Runtime and a Windows connector.

The bridge is deliberately narrower than an RPC shell: commands name one
registered device operation, carry typed JSON arguments, expire, and can be
claimed only once. Results are idempotent and unknown on disconnect.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import sqlite3
import threading
import time
from io import BytesIO
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from PIL import Image
from .robot_controller import RobotController


PROTOCOL = "2026-10-23-remote-device-connector-v2"
COMMAND_TTL_SECONDS = 30.0
RESULT_WAIT_SECONDS = 35.0
CONNECTOR_OPERATIONS = frozenset({
    "device_status", "capture_preview", "vision_capture", "select_machine_position",
    "begin_new_task", "prepare_machine_position", "request_stop",
    "vision_tap_relative", "vision_dismiss_overlay_relative",
    "vision_double_tap_relative", "vision_long_press_relative",
    "vision_drag_relative", "vision_swipe_relative", "vision_android_home",
    "vision_android_back", "vision_android_recent_apps",
    "vision_swipe_up", "vision_swipe_down", "vision_swipe_left",
    "vision_swipe_right", "vision_reveal_system_navigation",
})
PHYSICAL_ACTIONS = {
    "vision_tap_relative": "tap_semantic", "vision_dismiss_overlay_relative": "dismiss_overlay",
    "vision_double_tap_relative": "double_tap", "vision_long_press_relative": "long_press",
    "vision_drag_relative": "drag", "vision_swipe_relative": "swipe",
    "vision_android_home": "home", "vision_android_back": "back",
    "vision_android_recent_apps": "open_recent_apps", "vision_reveal_system_navigation": "reveal_system_navigation",
    **{f"vision_swipe_{direction}": "swipe" for direction in ("up", "down", "left", "right")},
}


class ConnectorProtocolError(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True, allow_nan=False).encode()


def sign_message(secret: str, method: str, path: str, timestamp: str, body: bytes,
                 *, connector_id: str, nonce: str) -> str:
    material = "\n".join((PROTOCOL, connector_id, nonce, method.upper(), path, timestamp,
                          hashlib.sha256(body).hexdigest())).encode()
    return hmac.new(secret.encode(), material, hashlib.sha256).hexdigest()


def verify_message(secret: str, method: str, path: str, timestamp: str, body: bytes, signature: str,
                   *, connector_id: str, nonce: str) -> None:
    if not secret or not timestamp or not signature:
        raise ConnectorProtocolError("连接器认证材料不完整。")
    try:
        age = abs(time.time() - float(timestamp))
    except ValueError as exc:
        raise ConnectorProtocolError("连接器时间戳无效。") from exc
    if not math.isfinite(age) or age > 90:
        raise ConnectorProtocolError("连接器请求已过期。")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", connector_id) or not re.fullmatch(r"[a-f0-9]{32}", nonce):
        raise ConnectorProtocolError("连接器身份或 nonce 无效。")
    expected = sign_message(secret, method, path, timestamp, body, connector_id=connector_id, nonce=nonce)
    if not hmac.compare_digest(expected, signature):
        raise ConnectorProtocolError("连接器签名无效。")


def validate_args(operation: str, args: Any) -> None:
    if not isinstance(operation, str) or operation not in CONNECTOR_OPERATIONS or not isinstance(args, dict):
        raise ConnectorProtocolError("连接器操作或参数无效。")
    fields = {
        "select_machine_position": {"machine_position"}, "capture_preview": {"quality"},
        "vision_tap_relative": {"x", "y"}, "vision_dismiss_overlay_relative": {"x", "y"},
        "vision_double_tap_relative": {"x", "y"},
        "vision_long_press_relative": {"x", "y", "hold_seconds"},
        "vision_drag_relative": {"start_x", "start_y", "end_x", "end_y"},
        "vision_swipe_relative": {"start_x", "start_y", "end_x", "end_y", "direction"},
    }.get(operation, set())
    if set(args) != fields:
        raise ConnectorProtocolError("连接器参数字段不匹配。")
    for name, value in args.items():
        if name == "direction":
            valid = isinstance(value, str) and value in {"up", "down", "left", "right"}
        elif name == "hold_seconds":
            valid = type(value) in (int, float) and math.isfinite(value) and 0.5 <= value <= 2
        else:
            low, high = (1, 10) if name == "machine_position" else (1, 100) if name == "quality" else (0, 1000)
            valid = type(value) is int and low <= value <= high
        if not valid:
            raise ConnectorProtocolError("连接器参数值无效：" + name)


def validate_command(value: Any, *, connector_id: str | None = None, device_id: str | None = None) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"protocol", "command_id", "connector_id", "device_id", "operation", "args", "authority", "expires_at"}:
        raise ConnectorProtocolError("连接器命令字段无效。")
    if value["protocol"] != PROTOCOL or not isinstance(value["command_id"], str) or not value["command_id"]:
        raise ConnectorProtocolError("连接器命令协议或 ID 无效。")
    if any(not isinstance(value[field], str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", value[field])
           for field in ("device_id", "connector_id", "command_id")):
        raise ConnectorProtocolError("连接器命令缺少设备标识。")
    if (connector_id is not None and value["connector_id"] != connector_id) or (device_id is not None and value["device_id"] != device_id):
        raise ConnectorProtocolError("连接器命令归属不匹配。")
    validate_args(value["operation"], value["args"])
    authority = value["authority"]
    if value["operation"] in PHYSICAL_ACTIONS:
        from .orientation_safety import OrientationCredential
        if not isinstance(authority, dict) or set(authority) != {"capture_id", "credential"} or not isinstance(authority["capture_id"], str):
            raise ConnectorProtocolError("物理命令缺少平台单步凭据。")
        credential = OrientationCredential.from_dict(authority["credential"])
        credential.assert_authorizes(device_id=value["device_id"], scene_fingerprint=credential.scene_fingerprint,
                                     frame_size=credential.frame_size, action=PHYSICAL_ACTIONS[value["operation"]])
    elif authority is not None:
        raise ConnectorProtocolError("非物理命令不应携带动作凭据。")
    if type(value["expires_at"]) not in (int, float) or not math.isfinite(value["expires_at"]) or value["expires_at"] <= time.time():
        raise ConnectorProtocolError("连接器命令已过期。")
    return dict(value)


class ConnectorCommandStore:
    """Persistent queue shared by API workers and the single Runtime owner."""

    def __init__(self, path: Path, *, secret: str = "", connector_id: str = "windows-local-01",
                 keys: dict[str, str] | None = None) -> None:
        self.path = Path(path)
        # Each key names exactly one connector. Never accept an arbitrary identity
        # simply because it knows a fleet-wide shared key.
        self.keys = dict(keys) if keys is not None else {
            os.environ.get("PHONE_AGENT_CONNECTOR_ID", connector_id):
            secret or os.environ.get("PHONE_AGENT_CONNECTOR_SECRET", "")}
        self._condition = threading.Condition()

    @property
    def configured(self) -> bool:
        return any(self.keys.values())

    @contextmanager
    def connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path, timeout=10)
        try:
            with db:
                db.execute("""CREATE TABLE IF NOT EXISTS connector_commands_v2 (
                    command_id TEXT PRIMARY KEY, connector_id TEXT NOT NULL, device_id TEXT NOT NULL,
                    operation TEXT NOT NULL, args_json TEXT NOT NULL, authority_json TEXT NOT NULL,
                    status TEXT NOT NULL, result_json TEXT, created_at REAL NOT NULL,
                    expires_at REAL NOT NULL, claimed_at REAL, updated_at REAL NOT NULL)""")
                db.execute("CREATE TABLE IF NOT EXISTS connector_nonces (connector_id TEXT NOT NULL, nonce TEXT NOT NULL, expires_at REAL NOT NULL, PRIMARY KEY(connector_id,nonce))")
                yield db
        finally:
            db.close()

    def authenticate(self, *, connector_id: str, method: str, target: str, timestamp: str,
                     nonce: str, body: bytes, signature: str) -> None:
        verify_message(self.keys.get(connector_id, ""), method, target, timestamp, body, signature,
                       connector_id=connector_id, nonce=nonce)
        with self.connect() as db:
            db.execute("DELETE FROM connector_nonces WHERE expires_at < ?", (time.time(),))
            try:
                db.execute("INSERT INTO connector_nonces VALUES (?,?,?)", (connector_id, nonce, time.time() + 180))
            except sqlite3.IntegrityError as exc:
                raise ConnectorProtocolError("连接器请求已使用。") from exc

    def enqueue(self, *, connector_id: str, device_id: str, operation: str, args: dict[str, Any], authority: Any = None) -> str:
        command_id = secrets.token_hex(16)
        now = time.time()
        validate_command(dict(protocol=PROTOCOL, command_id=command_id, connector_id=connector_id,
                              device_id=device_id, operation=operation, args=args, authority=authority, expires_at=now + COMMAND_TTL_SECONDS))
        with self.connect() as db:
            db.execute("INSERT INTO connector_commands_v2 VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                       (command_id, connector_id, device_id, operation, _canonical(args).decode(), _canonical(authority).decode(), "queued", None,
                        now, now + COMMAND_TTL_SECONDS, None, now))
        with self._condition:
            self._condition.notify_all()
        return command_id

    def claim(self, *, connector_id: str, device_id: str) -> dict[str, Any] | None:
        now = time.time()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT command_id,operation,args_json,expires_at,authority_json FROM connector_commands_v2 "
                             "WHERE connector_id=? AND device_id=? AND status='queued' AND expires_at>? ORDER BY created_at LIMIT 1",
                             (connector_id, device_id, now)).fetchone()
            if row is None:
                return None
            changed = db.execute("UPDATE connector_commands_v2 SET status='claimed',claimed_at=?,updated_at=? "
                                 "WHERE command_id=? AND status='queued'", (now, now, row[0])).rowcount
            if changed != 1:
                return None
        return validate_command({"protocol": PROTOCOL, "command_id": row[0], "connector_id": connector_id, "device_id": device_id,
                                 "operation": row[1], "args": json.loads(row[2]), "expires_at": row[3], "authority": json.loads(row[4])})

    def complete(self, *, connector_id: str, result: dict[str, Any]) -> bool:
        validate_result(result, connector_id=connector_id)
        command_id = result["command_id"]
        encoded = _canonical(result).decode()
        now = time.time()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT status,result_json,expires_at FROM connector_commands_v2 WHERE command_id=? AND connector_id=? AND device_id=?",
                             (command_id, connector_id, result["device_id"])).fetchone()
            if row is None:
                return False
            if row[0] in {"completed", "unknown", "not_dispatched"} and row[1] is not None:
                return row[1] == encoded
            if row[0] not in {"claimed", "unknown"}:
                return False
            # Late evidence is retained, but cannot turn a timed-out command into
            # a successful Runtime step or cause its physical action to replay.
            status = "completed" if now < row[2] and row[0] == "claimed" else "unknown"
            db.execute("UPDATE connector_commands_v2 SET status=?,result_json=?,updated_at=? WHERE command_id=? AND status IN ('claimed','unknown')",
                       (status, encoded, now, command_id))
        with self._condition:
            self._condition.notify_all()
        return True

    def wait(self, command_id: str, timeout: float = RESULT_WAIT_SECONDS) -> dict[str, Any]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self.connect() as db:
                row = db.execute("SELECT status,result_json,expires_at FROM connector_commands_v2 WHERE command_id=?",
                                 (command_id,)).fetchone()
            if row is None:
                raise ConnectorProtocolError("连接器命令不存在。")
            if row[0] == "completed":
                return json.loads(row[1])
            if row[0] in {"unknown", "not_dispatched"} or row[2] < time.time():
                break
            with self._condition:
                self._condition.wait(timeout=min(0.25, max(0.01, deadline - time.time())))
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("UPDATE connector_commands_v2 SET status=CASE status WHEN 'queued' THEN 'not_dispatched' ELSE 'unknown' END,updated_at=? WHERE command_id=? AND status IN ('queued','claimed')",
                       (time.time(), command_id))
            row = db.execute("SELECT status,result_json FROM connector_commands_v2 WHERE command_id=?", (command_id,)).fetchone()
        if row and row[0] == "completed":
            return json.loads(row[1])
        if row and row[0] == "not_dispatched":
            raise ConnectorProtocolError("连接器未领取命令，未派发。")
        raise ConnectorProtocolError("连接器结果未知，不能自动重放。")

    def last_confirmed_machine_position(self, *, connector_id: str, device_id: str) -> int | None:
        """Restore the latest completed selection; never send a recovery action."""
        if not self.path.exists():
            return None
        with self.connect() as db:
            row = db.execute(
                "SELECT args_json,status,result_json FROM connector_commands_v2 "
                "WHERE connector_id=? AND device_id=? AND operation='select_machine_position' "
                "ORDER BY created_at DESC,rowid DESC LIMIT 1", (connector_id, device_id),
            ).fetchone()
        if not row or row[1] != 'completed' or not row[2]:
            return None
        if json.loads(row[2]).get('ok') is not True:
            return None
        args = json.loads(row[0])
        validate_args('select_machine_position', args)
        return args['machine_position']


class RemoteRobotController(RobotController):
    """RobotController-shaped proxy backed by the authenticated command queue."""

    def __init__(self, *, device_id: str, connector_id: str, store: ConnectorCommandStore,
                 verified_actions: set[str] | frozenset[str] | None = None, **controller_options: Any) -> None:
        super().__init__(device_id=device_id, verified_actions=verified_actions, **controller_options)
        self.connector_id = connector_id
        self.store = store
        selected = store.last_confirmed_machine_position(connector_id=connector_id, device_id=device_id)
        if selected is not None:
            self.machine_position = selected
        self._remote_frame: Image.Image | None = None
        self._remote_capture_id: str | None = None

    def _call(self, operation: str, **args: Any) -> Any:
        validate_args(operation, args)
        authority = None
        if operation in PHYSICAL_ACTIONS:
            if self.stop_event.is_set():
                raise ConnectorProtocolError("用户已请求停止任务。")
            action = PHYSICAL_ACTIONS[operation]
            self._require_verified_action(action, action)
            if self._remote_frame is None:
                raise ConnectorProtocolError("远程动作缺少当前观察。")
            credential = self._consume_physical_execution(action, self._remote_frame)
            authority = {"capture_id": self._remote_capture_id, "credential": credential.to_dict()}
            self._remote_frame = None
            self._remote_capture_id = None
        command_id = self.store.enqueue(connector_id=self.connector_id, device_id=self.device_id, operation=operation, args=args, authority=authority)
        result = self.store.wait(command_id, timeout=3 if operation == "device_status" else RESULT_WAIT_SECONDS)
        if not result.get("ok"):
            raise ConnectorProtocolError(str(result.get("error") or "连接器操作失败。"))
        value = result.get("value")
        if operation in PHYSICAL_ACTIONS:
            if not isinstance(value, dict) or set(value) != {"return_value", "click_receipt", "long_press_receipt", "swipe_receipt"}:
                raise ConnectorProtocolError("远程动作回执无效，效果未知。")
            self._last_click_receipt = value["click_receipt"]
            self._last_long_press_receipt = value["long_press_receipt"]
            self._last_swipe_receipt = value["swipe_receipt"]
            return value["return_value"]
        return value

    def device_status(self) -> dict[str, Any]:
        try:
            value = self._call("device_status")
            if not isinstance(value, dict):
                raise ConnectorProtocolError("连接器状态响应无效。")
            return {**value, "busy": self.operation_lock.locked(), "connector_id": self.connector_id}
        except ConnectorProtocolError as exc:
            return {"controller_online": False, "camera_online": False, "busy": self.operation_lock.locked(),
                    "error": str(exc), "camera_error": str(exc), "execution_mode": "hardware",
                    "physical_execution": True, "connector_id": self.connector_id}

    def capture_preview(self, quality: int = 72) -> bytes:
        value = self._call("capture_preview", quality=quality)
        if not isinstance(value, str):
            raise ConnectorProtocolError("连接器预览响应无效。")
        try:
            return base64.b64decode(value, validate=True)
        except Exception as exc:
            raise ConnectorProtocolError("连接器预览数据无效。") from exc

    def vision_capture(self) -> Image.Image:
        value = self._call("vision_capture")
        with Image.open(BytesIO(base64.b64decode(value["png"], validate=True))) as frame:
            self._remote_frame = frame.convert("RGB")
        self._remote_capture_id = value["capture_id"]
        return self._remote_frame.copy()

    def begin_new_task(self) -> None:
        super().begin_new_task()
        self._remote_frame = None
        self._call("begin_new_task")

    def prepare_machine_position(self) -> None:
        if self.machine_position is not None and not self._machine_position_ready:
            self.select_machine_position(self.machine_position)

    def request_stop(self) -> None:
        super().request_stop()
        # Do not block API cancellation/shutdown on a disconnected device.
        self.store.enqueue(connector_id=self.connector_id, device_id=self.device_id, operation="request_stop", args={})

    def select_machine_position(self, machine_position: int) -> None:
        value = self._call("select_machine_position", machine_position=machine_position)
        self.machine_position = machine_position
        self._machine_position_ready = True
        self._remote_frame = None
        return value

    def vision_tap_relative(self, x: int, y: int):
        return self._call("vision_tap_relative", x=x, y=y)

    def vision_dismiss_overlay_relative(self, x: int, y: int):
        return self._call("vision_dismiss_overlay_relative", x=x, y=y)

    def vision_double_tap_relative(self, x: int, y: int):
        return self._call("vision_double_tap_relative", x=x, y=y)

    def vision_long_press_relative(self, x: int, y: int, hold_seconds: float = 0.8):
        return self._call("vision_long_press_relative", x=x, y=y, hold_seconds=hold_seconds)

    def vision_drag_relative(self, start_x: int, start_y: int, end_x: int, end_y: int):
        return self._call("vision_drag_relative", start_x=start_x, start_y=start_y, end_x=end_x, end_y=end_y)

    def vision_swipe_relative(self, start_x: int, start_y: int, end_x: int, end_y: int, direction: str):
        return self._call("vision_swipe_relative", start_x=start_x, start_y=start_y,
                          end_x=end_x, end_y=end_y, direction=direction)

    def vision_android_home(self):
        return self._call("vision_android_home")

    def vision_android_back(self):
        return self._call("vision_android_back")

    def vision_android_recent_apps(self):
        return self._call("vision_android_recent_apps")

    def vision_reveal_system_navigation(self):
        return self._call("vision_reveal_system_navigation")

    def _vision_swipe(self, direction: str) -> None:
        self._call("vision_swipe_" + direction)


def result_payload(*, command_id: str, connector_id: str, device_id: str, ok: bool, value: Any = None, error: str = "") -> dict[str, Any]:
    return {"protocol": PROTOCOL, "command_id": command_id, "connector_id": connector_id, "device_id": device_id,
            "ok": ok, "value": value, "error": str(error or "")[:300]}


def validate_result(value: Any, *, connector_id: str) -> None:
    if not isinstance(value, dict) or set(value) != {"protocol", "command_id", "connector_id", "device_id", "ok", "value", "error"}:
        raise ConnectorProtocolError("连接器结果字段无效。")
    if value["protocol"] != PROTOCOL or value["connector_id"] != connector_id or type(value["ok"]) is not bool:
        raise ConnectorProtocolError("连接器结果协议或身份无效。")
    if any(not isinstance(value[field], str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", value[field])
           for field in ("command_id", "device_id")) or not isinstance(value["error"], str) or len(value["error"]) > 300:
        raise ConnectorProtocolError("连接器结果标识无效。")
    try:
        _canonical(value)
    except (TypeError, ValueError) as exc:
        raise ConnectorProtocolError("连接器结果不是有限 JSON。") from exc

