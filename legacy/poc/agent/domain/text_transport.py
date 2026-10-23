"""Strict contracts for one authorized ADB Keyboard text action."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping


TEXT_TRANSPORT_PROTOCOL = "2026-09-02-adb-keyboard-v1"
TEXT_TRANSPORT_OPERATIONS = frozenset({"append_text", "clear_text"})
EMPTY_TEXT_DIGEST = hashlib.sha256(b"").hexdigest()
_HEX_DIGEST = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}")
_NONCE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{15,127}")


class TextTransportContractError(ValueError):
    """The bounded text-transport contract is malformed or inconsistent."""


class TextTransportReplayError(TextTransportContractError):
    """The same one-action authorization was consumed more than once."""


class TextTransportScopeError(TextTransportContractError):
    """The authorization does not belong to this device/action."""


def text_digest(value: str) -> str:
    if not isinstance(value, str):
        raise TextTransportContractError("文字摘要只能由字符串生成。")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise TextTransportContractError(f"{label} 无效。")
    return value


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _HEX_DIGEST.fullmatch(value):
        raise TextTransportContractError(f"{label} 无效。")
    return value


def _strict_mapping(value: Any, required: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != required:
        raise TextTransportContractError(f"{label}字段不完整或包含额外字段。")
    return value


@dataclass(frozen=True)
class TextTransportActionScope:
    """Exact local authorization for one canonical text action on one typed field."""

    protocol_version: str
    device_id: str
    session_id: str
    task_id: str
    revision: int
    action_id: str
    input_field_id: str
    observation_fingerprint: str
    prior_text_digest: str
    fragment_text_digest: str
    expected_text_digest: str
    issued_at_epoch: float
    nonce: str

    _FIELDS = frozenset({"protocol_version", "device_id", "session_id", "task_id", "revision", "action_id",
        "input_field_id", "observation_fingerprint", "prior_text_digest", "fragment_text_digest",
        "expected_text_digest", "issued_at_epoch", "nonce"})

    def validate(self) -> None:
        if self.protocol_version != TEXT_TRANSPORT_PROTOCOL:
            raise TextTransportContractError("ADB Keyboard 文字协议版本不匹配。")
        for name in ("device_id", "session_id", "task_id", "action_id", "input_field_id",
            "observation_fingerprint"):
            _identifier(getattr(self, name), name)
        if isinstance(self.revision, bool) or not isinstance(self.revision, int) or self.revision < 0:
            raise TextTransportContractError("revision 无效。")
        for name in ("prior_text_digest", "fragment_text_digest", "expected_text_digest"):
            _digest(getattr(self, name), name)
        if (isinstance(self.issued_at_epoch, bool) or not isinstance(self.issued_at_epoch, (int, float))
            or not math.isfinite(float(self.issued_at_epoch))):
            raise TextTransportContractError("issued_at_epoch 无效。")
        if not isinstance(self.nonce, str) or not _NONCE.fullmatch(self.nonce):
            raise TextTransportContractError("nonce 无效。")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {name: getattr(self, name) for name in self._FIELDS}

    @classmethod
    def from_dict(cls, value: Any) -> "TextTransportActionScope":
        item = _strict_mapping(value, cls._FIELDS, "ADB Keyboard action scope")
        scope = cls(**{name: item[name] for name in cls._FIELDS})
        scope.validate()
        return scope


@dataclass(frozen=True)
class TextTransportProfile:
    protocol_version: str
    profile_id: str
    device_id: str
    adb_serial: str
    enabled: bool
    capabilities: tuple[str, ...]
    command_timeout_seconds: float

    _FIELDS = frozenset({"protocol_version", "profile_id", "device_id", "adb_serial", "enabled",
        "capabilities", "command_timeout_seconds"})

    def validate(self) -> None:
        if self.protocol_version != TEXT_TRANSPORT_PROTOCOL:
            raise TextTransportContractError("ADB Keyboard profile 协议版本不匹配。")
        for name in ("profile_id", "device_id", "adb_serial"):
            _identifier(getattr(self, name), name)
        if not isinstance(self.enabled, bool):
            raise TextTransportContractError("ADB Keyboard profile enabled 无效。")
        if (not isinstance(self.capabilities, tuple) or not self.capabilities
            or len(set(self.capabilities)) != len(self.capabilities)
            or any(item not in TEXT_TRANSPORT_OPERATIONS for item in self.capabilities)):
            raise TextTransportContractError("ADB Keyboard capabilities 无效。")
        timeout = self.command_timeout_seconds
        if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout)) or not 0 < float(timeout) <= 30):
            raise TextTransportContractError("ADB Keyboard command timeout 无效。")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {name: (list(self.capabilities) if name == "capabilities" else getattr(self, name))
            for name in self._FIELDS}

    @classmethod
    def from_dict(cls, value: Any) -> "TextTransportProfile":
        item = _strict_mapping(value, cls._FIELDS, "ADB Keyboard profile")
        profile = cls(protocol_version=item["protocol_version"], profile_id=item["profile_id"],
            device_id=item["device_id"], adb_serial=item["adb_serial"], enabled=item["enabled"],
            capabilities=tuple(item["capabilities"]) if isinstance(item["capabilities"], list) else (),
            command_timeout_seconds=item["command_timeout_seconds"])
        profile.validate()
        return profile


@dataclass(frozen=True)
class TextTransportResult:
    """ADB transport receipt only; accepted never means visual text success."""

    protocol_version: str
    device_id: str
    action_id: str
    nonce: str
    operation: str
    status: str
    attempted: bool
    accepted: bool
    reason_code: str | None
    command_digest: str | None
    receipt_digest: str | None
    visual_verification_required: bool = True

    _FIELDS = frozenset({"protocol_version", "device_id", "action_id", "nonce", "operation", "status",
        "attempted", "accepted", "reason_code", "command_digest", "receipt_digest",
        "visual_verification_required"})

    def validate(self) -> None:
        if self.protocol_version != TEXT_TRANSPORT_PROTOCOL or self.operation not in TEXT_TRANSPORT_OPERATIONS:
            raise TextTransportContractError("ADB Keyboard result 协议或 operation 无效。")
        for name in ("device_id", "action_id"):
            _identifier(getattr(self, name), name)
        if not isinstance(self.nonce, str) or not _NONCE.fullmatch(self.nonce):
            raise TextTransportContractError("ADB Keyboard result nonce 无效。")
        if self.status not in {"unavailable", "unknown", "rejected", "accepted"}:
            raise TextTransportContractError("ADB Keyboard result status 无效。")
        valid = {"unavailable": (False, False), "unknown": (True, False),
            "rejected": (True, False), "accepted": (True, True)}[self.status]
        if (self.attempted, self.accepted) != valid:
            raise TextTransportContractError("ADB Keyboard result 状态冲突。")
        if self.reason_code is not None and (not isinstance(self.reason_code, str)
            or not re.fullmatch(r"[a-z][a-z0-9_.-]{0,63}", self.reason_code)):
            raise TextTransportContractError("ADB Keyboard reason_code 无效。")
        for name in ("command_digest", "receipt_digest"):
            value = getattr(self, name)
            if value is not None:
                _digest(value, name)
        if self.status in {"accepted", "rejected"} and (self.command_digest is None or self.receipt_digest is None):
            raise TextTransportContractError("ADB Keyboard 已尝试结果缺少摘要。")
        if self.visual_verification_required is not True:
            raise TextTransportContractError("ADB Keyboard 回执不能替代动作后视觉验证。")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {name: getattr(self, name) for name in self._FIELDS}


__all__ = ["EMPTY_TEXT_DIGEST", "TEXT_TRANSPORT_OPERATIONS", "TEXT_TRANSPORT_PROTOCOL",
    "TextTransportActionScope", "TextTransportContractError", "TextTransportProfile",
    "TextTransportReplayError", "TextTransportResult", "TextTransportScopeError", "text_digest"]
