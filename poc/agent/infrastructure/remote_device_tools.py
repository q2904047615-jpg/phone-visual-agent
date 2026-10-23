"""Station-bound adapters for the existing text and trusted-launch contracts.

Only the Windows host loads executable paths, serials and package commands.
The cloud forwards one authorized operation; the normal post-action observer
and Verifier still decide whether it achieved its business result.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import re
import secrets
import time

from agent.application.action_adapter import AppLaunchTarget
from agent.domain.text_transport import (
    EMPTY_TEXT_DIGEST, TEXT_TRANSPORT_PROTOCOL, TextTransportActionScope,
    TextTransportProfile, TextTransportResult, text_digest,
)
from agent.domain.validation import canonical_digest
from .adb_keyboard_transport import AdbKeyboardRuntimeRegistry
from .adb_package_launcher import AdbPackageLauncher, AdbPackageLauncherError
from .remote_connector import ConnectorProtocolError


def validate_tool_args(operation, args):
    fields = {"machine_position"}
    if operation != "tool_describe":
        fields.add("binding")
    if operation in {"text_append", "text_clear"}:
        fields.add("scope")
    if operation == "text_append":
        fields.add("text")
    if operation == "app_launch":
        fields.add("launch_ref")
    if set(args) != fields or type(args["machine_position"]) is not int or not 1 <= args["machine_position"] <= 10:
        raise ConnectorProtocolError("远程工具字段或机位无效。", attempted=False)
    if "binding" in fields and (not isinstance(args["binding"], str) or not re.fullmatch(r"[a-f0-9]{64}", args["binding"])):
        raise ConnectorProtocolError("远程工具配置绑定无效。", attempted=False)
    if "scope" in fields:
        scope = TextTransportActionScope.from_dict(args["scope"])
        if operation == "text_append":
            text = args["text"]
            valid = isinstance(text, str) and bool(text) and "\r" not in text and text_digest(text) == scope.fragment_text_digest
        else:
            valid = scope.fragment_text_digest == EMPTY_TEXT_DIGEST and scope.expected_text_digest == EMPTY_TEXT_DIGEST
        if not valid:
            raise ConnectorProtocolError("远程文字与 scope 不匹配。", attempted=False)
    if "launch_ref" in fields and (not isinstance(args["launch_ref"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", args["launch_ref"])):
        raise ConnectorProtocolError("远程 App 引用无效。", attempted=False)


class LocalDeviceTools:
    """Select profiles from the enabled station registry, never fall back to #1."""

    def __init__(self, *, registry_path: Path, device_id: str,
                 text_registry_path: Path, app_registry_path: Path):
        raw = json.loads(Path(registry_path).read_text(encoding="utf-8"))
        devices = [d for d in raw["devices"] if d.get("enabled") is True]
        current = next(d for d in devices if d["device_id"] == device_id)
        window = current.get("window_title") or ""
        self.device_id = device_id
        self.stations = {d["machine_position"]: d["device_id"] for d in devices
                         if d.get("machine_position") is not None and (d.get("window_title") or "") == window}
        self.texts = AdbKeyboardRuntimeRegistry(text_registry_path)
        self.launchers = {device: AdbPackageLauncher(app_registry_path, device)
                          for device in self.stations.values()} if app_registry_path.exists() else {}
        # A restarted/reconfigured connector invalidates all old discovery results.
        self.epoch = secrets.token_hex(32)

    def describe(self, position):
        device = self.stations.get(position)
        text = self.texts.transport_for_device(device) if device else None
        launcher = self.launchers.get(device)
        profile = replace(text.profile, device_id=self.device_id).to_dict() if text else None
        catalog = launcher.catalog() if launcher else []
        binding = canonical_digest({"epoch": self.epoch, "position": position, "device": device,
                                    "profile": profile, "catalog": catalog})
        return {"machine_position": position, "binding": binding,
                "text_profile": profile, "apps": catalog}

    def execute(self, command, *, controller, journal):
        op, args = command["operation"], command["args"]
        position = args["machine_position"]
        if controller.machine_position != position:
            raise ConnectorProtocolError("当前机位与工具绑定不一致。", attempted=False)
        descriptor = self.describe(position)
        if op == "tool_describe":
            return descriptor
        if args["binding"] != descriptor["binding"]:
            raise ConnectorProtocolError("工具配置已变化，需要重新观察。", attempted=False)
        device = self.stations.get(position)
        text = self.texts.transport_for_device(device) if device else None
        if op == "text_status":
            if text is None:
                return {"ready": False, "reason_code": "station_not_configured"}
            return {**text.status(), "device_id": self.device_id, "machine_position": position}
        if controller.stop_event.is_set():
            raise ConnectorProtocolError("用户已请求停止。", attempted=False)
        if op in {"text_append", "text_clear"}:
            if text is None:
                raise ConnectorProtocolError("当前机位没有文字配置。", attempted=False)
            scope = TextTransportActionScope.from_dict(args["scope"])
            operation = "append_text" if op == "text_append" else "clear_text"
            local_scope = replace(scope, device_id=device)
            # Validate before consuming observation/nonce; this never sends ADB.
            text._validate_scope(local_scope, operation=operation, text=args.get("text"))
        else:
            launcher = self.launchers.get(device)
            if launcher is None or not launcher.enabled or args["launch_ref"] not in {a["launch_ref"] for a in descriptor["apps"]}:
                raise ConnectorProtocolError("当前机位没有该可信启动目标。", attempted=False)
        capture = journal.consume_capture(command["authority"]["capture_id"],
            connector_id=command["connector_id"], device_id=command["device_id"])
        if capture.get("machine_position") != position or capture.get("tool_binding") != descriptor["binding"]:
            raise ConnectorProtocolError("观察帧不属于当前机位或配置。", attempted=False)
        if op in {"text_append", "text_clear"}:
            journal.consume_tool_scope(scope)
            result = text.append_text(local_scope, args["text"]) if op == "text_append" else text.clear_text(local_scope)
            return replace(result, device_id=self.device_id).to_dict()
        launcher.launch(args["launch_ref"])
        return {"status": "accepted", "visual_verification_required": True}


def describe_remote_tools(controller):
    position = controller.machine_position
    if type(position) is not int:
        return None
    try:
        value = controller._call("tool_describe", machine_position=position)
        if not isinstance(value, dict) or set(value) != {"machine_position", "binding", "text_profile", "apps"}:
            raise ValueError("Invalid tool discovery")
        validate_tool_args("text_status", {"machine_position": value["machine_position"], "binding": value["binding"]})
        if value["machine_position"] != position or not isinstance(value["apps"], list):
            raise ValueError("Invalid station discovery")
        return value
    except (ConnectorProtocolError, ValueError, TypeError, KeyError):
        return None


class RemoteTextTransport:
    def __init__(self, controller, descriptor):
        self.controller = controller
        self.binding = {key: descriptor[key] for key in ("machine_position", "binding")}
        self.profile = TextTransportProfile.from_dict(descriptor["text_profile"])
        if self.profile.device_id != controller.device_id:
            raise ValueError("Remote text profile belongs to a different device")
        self._consumed = set()

    def mint_action_scope(self, **values):
        scope = TextTransportActionScope(protocol_version=TEXT_TRANSPORT_PROTOCOL,
            device_id=self.profile.device_id, issued_at_epoch=time.time(), nonce=secrets.token_hex(16), **values)
        scope.validate()
        return scope

    def status(self):
        try:
            return self.controller._call("text_status", **self.binding)
        except ConnectorProtocolError:
            return {"ready": False, "reason_code": "connector_unavailable", "device_id": self.profile.device_id}

    def _execute(self, scope, operation, text=None):
        args = {**self.binding, "scope": scope.to_dict()}
        op = "text_append" if operation == "append_text" else "text_clear"
        if text is not None:
            args["text"] = text
        validate_tool_args(op, args)
        if scope.device_id != self.profile.device_id or scope.nonce in self._consumed or operation not in self.profile.capabilities:
            raise ConnectorProtocolError("文字动作 scope 无效或已经消费。", attempted=False)
        self._consumed.add(scope.nonce)
        try:
            value = self.controller._call(op, **args)
            if not isinstance(value, dict) or set(value) != TextTransportResult._FIELDS:
                raise ValueError("Invalid remote text receipt")
            result = TextTransportResult(**value)
            result.validate()
            if (result.device_id, result.action_id, result.nonce, result.operation) != (scope.device_id, scope.action_id, scope.nonce, operation):
                raise ValueError("Mismatched remote text receipt")
            return result
        except (ConnectorProtocolError, ValueError, TypeError) as exc:
            attempted = getattr(exc, "attempted", True)
            return TextTransportResult(protocol_version=TEXT_TRANSPORT_PROTOCOL, device_id=scope.device_id,
                action_id=scope.action_id, nonce=scope.nonce, operation=operation,
                status="unknown" if attempted else "unavailable", attempted=attempted, accepted=False,
                reason_code="connector_result_unknown" if attempted else "connector_not_dispatched",
                command_digest=None, receipt_digest=None)

    def append_text(self, scope, text):
        return self._execute(scope, "append_text", text)

    def clear_text(self, scope):
        return self._execute(scope, "clear_text")


class RemoteAppLauncher:
    def __init__(self, controller, descriptor):
        self.controller = controller
        self.binding = {key: descriptor[key] for key in ("machine_position", "binding")}
        self._aliases = {}
        for item in descriptor["apps"]:
            if set(item) != {"alias", "launch_ref", "expected_app_id"}:
                raise ValueError("Invalid remote launch catalog")
            self._aliases[item["alias"]] = AppLaunchTarget(item["launch_ref"], item["expected_app_id"])
        self.enabled = bool(self._aliases)

    def aliases(self):
        return tuple(sorted(self._aliases))

    def resolve(self, app_id, app_name):
        matches = {self._aliases[value.casefold()] for value in (app_id, app_name) if value and value.casefold() in self._aliases}
        if len(matches) > 1:
            raise AdbPackageLauncherError("App 名称映射不一致。")
        return next(iter(matches), None)

    def launch(self, launch_ref):
        if launch_ref not in {target.launch_ref for target in self._aliases.values()}:
            raise AdbPackageLauncherError("App 引用未登记。")
        try:
            value = self.controller._call("app_launch", **self.binding, launch_ref=launch_ref)
            if value != {"status": "accepted", "visual_verification_required": True}:
                raise ConnectorProtocolError("App 启动回执无效，结果未知。")
            return value
        except ConnectorProtocolError as exc:
            raise AdbPackageLauncherError(str(exc), attempted=exc.attempted) from exc
