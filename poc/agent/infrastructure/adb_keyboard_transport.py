"""Bounded ADB Keyboard transport adapted from Open-AutoGLM's ADB input path."""

from __future__ import annotations

import base64
import json
from pathlib import Path
import secrets
import subprocess
import time
from typing import Any, Callable, Mapping

from agent.domain.validation import canonical_digest
from agent.domain.text_transport import (EMPTY_TEXT_DIGEST, TEXT_TRANSPORT_PROTOCOL,
    TextTransportActionScope, TextTransportContractError, TextTransportProfile,
    TextTransportReplayError, TextTransportResult, TextTransportScopeError, text_digest)


ADB_KEYBOARD_IME_ID = "com.android.adbkeyboard/.AdbIME"
ADB_KEYBOARD_RUNTIME_REGISTRY_VERSION = "2026-09-02-adb-keyboard-runtime-v1"
_REGISTRY_FIELDS = frozenset({"version", "devices"})
_DEVICE_FIELDS = frozenset({"profile", "adb_executable"})


class AdbKeyboardConfigError(RuntimeError):
    pass


Runner = Callable[..., subprocess.CompletedProcess[str]]


class AdbKeyboardTextTransport:
    """Execute only the fixed ADB Keyboard command family for one fixed serial."""

    def __init__(self, profile: TextTransportProfile, adb_executable: Path, *, runner: Runner=subprocess.run,
        clock: Callable[[], float]=time.time, nonce_factory: Callable[[], str] | None=None) -> None:
        profile.validate()
        executable = Path(adb_executable)
        if executable.name.casefold() not in {"adb", "adb.exe"}:
            raise AdbKeyboardConfigError("adb_executable 必须指向 adb 或 adb.exe。")
        self._profile = profile
        self._adb_executable = executable
        self._runner = runner
        self._clock = clock
        self._nonce_factory = nonce_factory or (lambda: secrets.token_hex(16))
        self._consumed_nonces: set[str] = set()

    @property
    def profile(self) -> TextTransportProfile:
        return self._profile

    def status(self) -> dict[str, Any]:
        """Return a secret-free, read-only readiness snapshot."""
        reason = self._preflight_reason()
        return {"protocol_version": TEXT_TRANSPORT_PROTOCOL, "transport": "adb_keyboard",
            "device_id": self._profile.device_id, "adb_serial": self._profile.adb_serial,
            "ime_id": ADB_KEYBOARD_IME_ID, "enabled": self._profile.enabled,
            "ready": reason is None, "reason_code": reason,
            "capabilities": list(self._profile.capabilities), "physical_actions": 0}

    def mint_action_scope(self, *, session_id: str, task_id: str, revision: int, action_id: str,
        input_field_id: str, observation_fingerprint: str, prior_text_digest: str,
        fragment_text_digest: str, expected_text_digest: str) -> TextTransportActionScope:
        if not self._profile.enabled:
            raise TextTransportContractError("ADB Keyboard profile 未启用。")
        scope = TextTransportActionScope(protocol_version=TEXT_TRANSPORT_PROTOCOL,
            device_id=self._profile.device_id, session_id=session_id, task_id=task_id, revision=revision,
            action_id=action_id, input_field_id=input_field_id,
            observation_fingerprint=observation_fingerprint, prior_text_digest=prior_text_digest,
            fragment_text_digest=fragment_text_digest, expected_text_digest=expected_text_digest,
            issued_at_epoch=float(self._clock()), nonce=self._nonce_factory())
        scope.validate()
        return scope

    def _run(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return self._runner([str(self._adb_executable), "-s", self._profile.adb_serial, *arguments],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=float(self._profile.command_timeout_seconds), check=False)

    def _preflight_reason(self) -> str | None:
        if not self._profile.enabled:
            return "profile_disabled"
        if not self._adb_executable.is_file():
            return "adb_missing"
        try:
            state = self._run("get-state")
            if state.returncode != 0 or state.stdout.strip() != "device":
                return "device_offline"
            installed = self._run("shell", "ime", "list", "-s")
            if installed.returncode != 0 or ADB_KEYBOARD_IME_ID not in installed.stdout.splitlines():
                return "ime_not_enabled"
            selected = self._run("shell", "settings", "get", "secure", "default_input_method")
            if selected.returncode != 0 or selected.stdout.strip() != ADB_KEYBOARD_IME_ID:
                return "ime_not_selected"
        except subprocess.TimeoutExpired:
            return "preflight_timeout"
        except (OSError, ValueError):
            return "adb_unavailable"
        return None

    def _validate_scope(self, scope: TextTransportActionScope, *, operation: str, text: str | None) -> None:
        scope.validate()
        if scope.device_id != self._profile.device_id:
            raise TextTransportScopeError("ADB Keyboard scope 不属于当前设备。")
        if operation not in self._profile.capabilities:
            raise TextTransportScopeError("ADB Keyboard profile 未开放当前操作。")
        if scope.nonce in self._consumed_nonces:
            raise TextTransportReplayError("ADB Keyboard 单动作 scope 已经消费。")
        if operation == "append_text":
            if not isinstance(text, str) or not text or "\r" in text:
                raise TextTransportContractError("ADB Keyboard 输入需要非空且不含回车的 Unicode 文字。")
            if text_digest(text) != scope.fragment_text_digest:
                raise TextTransportScopeError("ADB Keyboard scope 与输入正文摘要不一致。")
        elif text is not None or scope.fragment_text_digest != EMPTY_TEXT_DIGEST \
            or scope.expected_text_digest != EMPTY_TEXT_DIGEST:
            raise TextTransportScopeError("ADB Keyboard 清空 scope 必须绑定空正文。")

    def _result(self, scope: TextTransportActionScope, operation: str, *, status: str,
        reason: str | None, command_digest: str | None=None, receipt_digest: str | None=None) -> TextTransportResult:
        result = TextTransportResult(protocol_version=TEXT_TRANSPORT_PROTOCOL, device_id=self._profile.device_id,
            action_id=scope.action_id, nonce=scope.nonce, operation=operation, status=status,
            attempted=status != "unavailable", accepted=status == "accepted", reason_code=reason,
            command_digest=command_digest, receipt_digest=receipt_digest)
        result.validate()
        return result

    def _broadcast(self, scope: TextTransportActionScope, *, operation: str,
        arguments: tuple[str, ...], payload_digest: str) -> TextTransportResult:
        reason = self._preflight_reason()
        if reason is not None:
            return self._result(scope, operation, status="unavailable", reason=reason)
        command_digest = canonical_digest({"protocol_version": TEXT_TRANSPORT_PROTOCOL, "operation": operation,
            "device_id": self._profile.device_id, "adb_serial": self._profile.adb_serial,
            "scope": scope.to_dict(), "payload_digest": payload_digest})
        self._consumed_nonces.add(scope.nonce)
        try:
            completed = self._run("shell", "am", "broadcast", *arguments)
        except subprocess.TimeoutExpired:
            return self._result(scope, operation, status="unknown", reason="broadcast_timeout",
                command_digest=command_digest)
        except (OSError, ValueError):
            return self._result(scope, operation, status="unknown", reason="broadcast_unavailable",
                command_digest=command_digest)
        receipt_digest = canonical_digest({"returncode": completed.returncode, "stdout": completed.stdout,
            "stderr": completed.stderr, "command_digest": command_digest})
        accepted = completed.returncode == 0 and "Broadcast completed" in completed.stdout
        return self._result(scope, operation, status="accepted" if accepted else "rejected",
            reason=None if accepted else "broadcast_rejected", command_digest=command_digest,
            receipt_digest=receipt_digest)

    def append_text(self, scope: TextTransportActionScope, text: str) -> TextTransportResult:
        self._validate_scope(scope, operation="append_text", text=text)
        encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
        return self._broadcast(scope, operation="append_text",
            arguments=("-a", "ADB_INPUT_B64", "--es", "msg", encoded), payload_digest=text_digest(text))

    def clear_text(self, scope: TextTransportActionScope) -> TextTransportResult:
        self._validate_scope(scope, operation="clear_text", text=None)
        return self._broadcast(scope, operation="clear_text", arguments=("-a", "ADB_CLEAR_TEXT"),
            payload_digest=EMPTY_TEXT_DIGEST)

def load_adb_keyboard_transports(registry_path: Path, *, runner: Runner=subprocess.run,
    clock: Callable[[], float]=time.time) -> dict[str, AdbKeyboardTextTransport]:
    path = Path(registry_path)
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AdbKeyboardConfigError("ADB Keyboard 运行配置不可读。") from exc
    if not isinstance(raw, Mapping) or set(raw) != _REGISTRY_FIELDS \
        or raw.get("version") != ADB_KEYBOARD_RUNTIME_REGISTRY_VERSION or not isinstance(raw.get("devices"), list):
        raise AdbKeyboardConfigError("ADB Keyboard 运行配置字段或版本无效。")
    transports: dict[str, AdbKeyboardTextTransport] = {}
    for item in raw["devices"]:
        if not isinstance(item, Mapping) or set(item) != _DEVICE_FIELDS:
            raise AdbKeyboardConfigError("ADB Keyboard 设备配置字段无效。")
        profile = TextTransportProfile.from_dict(item["profile"])
        if profile.device_id in transports:
            raise AdbKeyboardConfigError(f"ADB Keyboard device_id 重复：{profile.device_id!r}。")
        executable = Path(item["adb_executable"])
        if not executable.is_absolute():
            executable = path.parent / executable
        if profile.enabled:
            transports[profile.device_id] = AdbKeyboardTextTransport(profile, executable.resolve(),
                runner=runner, clock=clock)
    return transports


class AdbKeyboardRuntimeRegistry:
    def __init__(self, registry_path: Path, *, runner: Runner=subprocess.run,
        clock: Callable[[], float]=time.time) -> None:
        self._transports = load_adb_keyboard_transports(registry_path, runner=runner, clock=clock)

    @property
    def configured_device_ids(self) -> tuple[str, ...]:
        return tuple(self._transports)

    def transport_for_device(self, device_id: str) -> AdbKeyboardTextTransport | None:
        return self._transports.get(str(device_id or "").strip())


__all__ = ["ADB_KEYBOARD_IME_ID", "ADB_KEYBOARD_RUNTIME_REGISTRY_VERSION", "AdbKeyboardConfigError",
    "AdbKeyboardRuntimeRegistry", "AdbKeyboardTextTransport", "load_adb_keyboard_transports"]
