"""Authenticated HTTPS device transport on the seller's interactive desktop."""
from __future__ import annotations

import base64
import json
import secrets
import time
from contextlib import ExitStack
from io import BytesIO
from pathlib import Path
from urllib.parse import urlencode, urlsplit

import httpx
from PIL import Image

from agent.infrastructure.connector_journal import ConnectorJournal
from agent.infrastructure.device_controller_registry import DeviceControllerRegistry
from agent.infrastructure.device_exclusivity import InterProcessLease, SHARED_DEVICE_LEASE_DIR
from agent.infrastructure.device_task_registry import DeviceTaskRegistry
from agent.infrastructure.orientation_safety import (
    OrientationCredential, _mint_single_step_scene_credential, frame_fingerprint,
)
from agent.infrastructure.remote_connector import (
    CONNECTOR_OPERATIONS, PHYSICAL_ACTIONS, ConnectorProtocolError,
    result_payload, sign_message, validate_command,
)


class DeviceConnector:
    def __init__(self, *, base_url: str, connector_id: str, secret: str, registry_path: Path,
                 device_id: str, journal_path: Path, controller=None, client=None,
                 lease_directory: Path = SHARED_DEVICE_LEASE_DIR):
        url = urlsplit(base_url)
        if url.scheme != "https" or not url.hostname or url.username or url.password or url.path not in ("", "/") or url.query or url.fragment:
            raise ConnectorProtocolError("连接器必须使用无路径、无凭据的 HTTPS 站点地址。")
        if not secret or not connector_id or not device_id:
            raise ConnectorProtocolError("连接器身份配置不完整。")
        self.base_url = base_url.rstrip("/")
        self.connector_id, self.secret, self.device_id = connector_id, secret, device_id
        if controller is None:
            registry = DeviceControllerRegistry(registry_path, promotable_actions=(), mock=False)
            if any(d.get("connector_id") for d in registry.descriptors()):
                raise ConnectorProtocolError("Windows 连接器注册表必须指向本机控制器，不能递归代理。")
            controller = registry.controller(device_id)
        self.controller = controller
        self.client = client or httpx.Client(timeout=httpx.Timeout(40, connect=8), follow_redirects=False, trust_env=False)
        self.journal = ConnectorJournal(journal_path)
        self.lease_directory = Path(lease_directory)
        self._ownership = None

    def __enter__(self):
        ownership = ExitStack()
        try:
            tasks = DeviceTaskRegistry(lease_directory=self.lease_directory)
            tasks.reserve(self.device_id, "connector-" + self.connector_id)
            ownership.callback(tasks.release, self.device_id, "connector-" + self.connector_id)
            lease = InterProcessLease(self.lease_directory / "physical_hardware_action.lease",
                                      owner_id=self.connector_id, metadata={"device_id": self.device_id, "purpose": "device_connector"})
            if not lease.acquire():
                raise ConnectorProtocolError("本机物理控制权已被其他进程占用。")
            ownership.callback(lease.release)
            self._ownership = ownership
            return self
        except BaseException:
            ownership.close()
            raise

    def __exit__(self, *exc):
        self._ownership.close()
        self._ownership = None
        self.client.close()

    def _headers(self, method: str, target: str, body: bytes = b"") -> dict[str, str]:
        timestamp, nonce = str(time.time()), secrets.token_hex(16)
        return {"X-Connector-Id": self.connector_id, "X-Connector-Timestamp": timestamp, "X-Connector-Nonce": nonce,
                "X-Connector-Signature": sign_message(self.secret, method, target, timestamp, body,
                                                     connector_id=self.connector_id, nonce=nonce),
                "Content-Type": "application/json"}

    def poll(self) -> dict | None:
        target = "/api/device-connector/v1/poll?" + urlencode({"device_id": self.device_id})
        response = self.client.get(self.base_url + target, headers=self._headers("GET", target))
        if response.status_code == 204:
            return None
        response.raise_for_status()
        return validate_command(response.json(), connector_id=self.connector_id, device_id=self.device_id)

    def _execute_operation(self, command):
        operation, args = command["operation"], command["args"]
        if operation == "vision_capture":
            frame = self.controller.vision_capture().convert("RGB")
            buffer = BytesIO()
            frame.save(buffer, format="PNG")
            return {"png": base64.b64encode(buffer.getvalue()).decode(), "capture_id": command["command_id"]}
        if operation == "capture_preview":
            return base64.b64encode(self.controller.capture_preview(**args)).decode()
        if operation == "device_status":
            return self.controller.device_status()
        if operation in PHYSICAL_ACTIONS:
            authority = command["authority"]
            capture = self.journal.consume_capture(authority["capture_id"], connector_id=self.connector_id, device_id=self.device_id)
            with Image.open(BytesIO(base64.b64decode(capture["png"], validate=True))) as image:
                frame = image.convert("RGB")
            credential = OrientationCredential.from_dict(authority["credential"])
            action = PHYSICAL_ACTIONS[operation]
            credential.assert_authorizes(device_id=self.device_id, scene_fingerprint=credential.scene_fingerprint,
                                         frame_size=frame.size, action=action)
            if credential.frame_fingerprint != frame_fingerprint(frame):
                raise ConnectorProtocolError("平台单步凭据未绑定本机原始观察帧。")
            local = _mint_single_step_scene_credential(device_id=self.device_id, scene_fingerprint=credential.scene_fingerprint, frame=frame)
            self.controller.arm_physical_execution(local, action=action, scene_fingerprint=credential.scene_fingerprint)
        methods = {name: getattr(self.controller, name) for name in CONNECTOR_OPERATIONS}
        try:
            value = methods[operation](**args)
            if operation in PHYSICAL_ACTIONS:
                return {"return_value": value, "click_receipt": self.controller.consume_last_click_receipt(),
                        "long_press_receipt": self.controller.consume_last_long_press_receipt(),
                        "swipe_receipt": self.controller.consume_last_swipe_receipt()}
            return value
        finally:
            self.controller.clear_physical_execution_authorization()

    def execute(self, command: dict) -> dict:
        if self._ownership is None:
            raise ConnectorProtocolError("连接器未持有本机控制租约。")
        validate_command(command, connector_id=self.connector_id, device_id=self.device_id)
        cached = self.journal.begin(command)
        if cached is not None:
            return cached
        try:
            value = self._execute_operation(command)
            payload = result_payload(command_id=command["command_id"], connector_id=self.connector_id,
                                     device_id=self.device_id, ok=True, value=value)
        except Exception as exc:
            payload = result_payload(command_id=command["command_id"], connector_id=self.connector_id,
                                     device_id=self.device_id, ok=False,
                                     error="operation_failed_effect_unknown:" + type(exc).__name__)
        self.journal.finish(payload)
        return payload

    def submit(self, payload: dict) -> None:
        target = "/api/device-connector/v1/result"
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
        response = self.client.post(self.base_url + target, content=body, headers=self._headers("POST", target, body))
        response.raise_for_status()
        value = response.json()
        if value.get("accepted") is not True or value.get("command_id") != payload["command_id"]:
            raise ConnectorProtocolError("云端结果回执不匹配。")
        self.journal.acknowledge(payload["command_id"])

    def run_once(self) -> bool:
        if self._ownership is None:
            raise ConnectorProtocolError("连接器未持有本机控制租约。")
        for payload in self.journal.pending():
            self.submit(payload)
        command = self.poll()
        if command is None:
            return False
        self.submit(self.execute(command))
        return True
