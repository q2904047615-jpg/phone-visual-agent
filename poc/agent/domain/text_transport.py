"""Transport-neutral contracts for one authorized Companion IME text action."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
from dataclasses import dataclass, field, replace
from typing import Any, Mapping


TEXT_TRANSPORT_PROTOCOL = "2026-08-30-companion-ime-v1"
TEXT_TRANSPORT_MAX_FRAME_BYTES = 1024 * 1024
# The whole signed JSON envelope is capped at 1 MiB. A quarter-frame UTF-8
# fragment remains below that cap even when every byte must be JSON-escaped.
TEXT_TRANSPORT_MAX_FRAGMENT_UTF8_BYTES = TEXT_TRANSPORT_MAX_FRAME_BYTES // 4
TEXT_TRANSPORT_OPERATIONS = frozenset({"append_text", "clear_text"})
TEXT_TRANSPORT_NETWORK_OPERATIONS = {
    "append_text": "commit_text",
    "clear_text": "clear_text",
}
EMPTY_TEXT_DIGEST = hashlib.sha256(b"").hexdigest()
_HEX_DIGEST = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}")
_NONCE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{15,127}")


class TextTransportContractError(ValueError):
    """The typed text-transport contract is malformed or inconsistent."""


class TextTransportFrameSizeError(TextTransportContractError):
    """A complete authenticated wire envelope cannot fit in one bridge frame."""


class TextTransportAuthenticationError(TextTransportContractError):
    """A signed command or acknowledgement cannot be authenticated."""


class TextTransportExpiredError(TextTransportContractError):
    """The single-action authorization is not fresh."""


class TextTransportReplayError(TextTransportContractError):
    """The same action nonce was consumed more than once."""


class TextTransportScopeError(TextTransportContractError):
    """The command scope does not belong to the selected device/action."""


def canonical_text_transport_json(value: Mapping[str, Any]) -> bytes:
    """Return the one byte representation covered by protocol signatures."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def text_digest(value: str) -> str:
    if not isinstance(value, str):
        raise TextTransportContractError("文字摘要只能由字符串生成。")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _pairing_key(value: bytes) -> bytes:
    if not isinstance(value, bytes) or len(value) < 32:
        raise TextTransportAuthenticationError("Companion IME 配对密钥无效。")
    return value


def sign_text_transport_payload(value: Mapping[str, Any], pairing_key: bytes) -> str:
    return hmac.new(_pairing_key(pairing_key), canonical_text_transport_json(value), hashlib.sha256).hexdigest()


def verify_text_transport_signature(value: Mapping[str, Any], signature: str, pairing_key: bytes) -> None:
    if not isinstance(signature, str) or not _HEX_DIGEST.fullmatch(signature):
        raise TextTransportAuthenticationError("Companion IME 签名格式无效。")
    expected = sign_text_transport_payload(value, pairing_key)
    if not hmac.compare_digest(expected, signature):
        raise TextTransportAuthenticationError("Companion IME 签名校验失败。")


def _strict_mapping(value: Any, required: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != required:
        raise TextTransportContractError(f"{label}字段不完整或包含额外字段。")
    return value


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise TextTransportContractError(f"{label} 无效。")
    return value


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _HEX_DIGEST.fullmatch(value):
        raise TextTransportContractError(f"{label} 无效。")
    return value


def _epoch(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise TextTransportContractError(f"{label} 无效。")
    return float(value)


@dataclass(frozen=True)
class TextTransportActionScope:
    """Exact authorization minted for one canonical text action on one typed field."""

    protocol_version: str
    device_id: str
    session_id: str
    task_id: str
    revision: int
    action_id: str
    input_field_id: str
    editor_session_id: str
    observation_fingerprint: str
    prior_text_digest: str
    fragment_text_digest: str
    expected_text_digest: str
    issued_at_epoch: float
    expires_at_epoch: float
    nonce: str

    _FIELDS = frozenset({"protocol_version", "device_id", "session_id", "task_id", "revision", "action_id",
        "input_field_id", "editor_session_id", "observation_fingerprint", "prior_text_digest", "fragment_text_digest",
        "expected_text_digest", "issued_at_epoch", "expires_at_epoch", "nonce"})

    def validate(self) -> None:
        if self.protocol_version != TEXT_TRANSPORT_PROTOCOL:
            raise TextTransportContractError("Companion IME 协议版本不匹配。")
        for name in ("device_id", "session_id", "task_id", "action_id", "input_field_id", "editor_session_id",
            "observation_fingerprint"):
            _identifier(getattr(self, name), name)
        if isinstance(self.revision, bool) or not isinstance(self.revision, int) or self.revision < 0:
            raise TextTransportContractError("revision 无效。")
        for name in ("prior_text_digest", "fragment_text_digest", "expected_text_digest"):
            _digest(getattr(self, name), name)
        issued = _epoch(self.issued_at_epoch, "issued_at_epoch")
        expires = _epoch(self.expires_at_epoch, "expires_at_epoch")
        if expires <= issued:
            raise TextTransportContractError("Companion IME 授权有效期无效。")
        if not isinstance(self.nonce, str) or not _NONCE.fullmatch(self.nonce):
            raise TextTransportContractError("nonce 无效。")

    def require_fresh(self, now_epoch: float) -> None:
        self.validate()
        now = _epoch(now_epoch, "now_epoch")
        if now < float(self.issued_at_epoch) or now > float(self.expires_at_epoch):
            raise TextTransportExpiredError("Companion IME 单动作授权已过期或尚未生效。")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {name: getattr(self, name) for name in self._FIELDS}

    @classmethod
    def from_dict(cls, value: Any) -> "TextTransportActionScope":
        item = _strict_mapping(value, cls._FIELDS, "Companion IME action scope")
        scope = cls(**{name: item[name] for name in cls._FIELDS})
        scope.validate()
        return scope


@dataclass(frozen=True)
class TextTransportCommand:
    """One signed command; plaintext is intentionally absent from safe views."""

    protocol_version: str
    operation: str
    scope: TextTransportActionScope
    _text: str | None = field(default=None, repr=False, compare=False)
    _signature: str = field(default="", repr=False, compare=False)

    _ENVELOPE_FIELDS = frozenset({"command", "signature"})
    _WIRE_FIELDS = frozenset({"protocol_version", "operation", "scope", "text"})

    def __repr__(self) -> str:
        return (f"TextTransportCommand(protocol_version={self.protocol_version!r}, operation={self.operation!r}, "
            f"scope={self.scope!r}, text_digest={self.scope.fragment_text_digest!r}, "
            f"signature_present={bool(self._signature)!r})")

    @property
    def text(self) -> str | None:
        """In-memory plaintext for the trusted IME endpoint only."""

        return self._text

    @property
    def signature_present(self) -> bool:
        return bool(self._signature)

    def validate(self) -> None:
        if self.protocol_version != TEXT_TRANSPORT_PROTOCOL or self.operation not in TEXT_TRANSPORT_OPERATIONS:
            raise TextTransportContractError("Companion IME command 类型无效。")
        self.scope.validate()
        if self.operation == "append_text":
            if not isinstance(self._text, str) or not self._text or "\r" in self._text:
                raise TextTransportContractError("append_text 需要非空且不含回车的 Unicode 文字。")
            if text_digest(self._text) != self.scope.fragment_text_digest:
                raise TextTransportScopeError("授权 fragment 摘要与文字不一致。")
        elif self._text is not None:
            raise TextTransportContractError("clear_text 不得携带文字。")
        if self.operation == "clear_text" and (self.scope.fragment_text_digest != EMPTY_TEXT_DIGEST
            or self.scope.expected_text_digest != EMPTY_TEXT_DIGEST):
            raise TextTransportScopeError("clear_text 必须绑定空 fragment 与空期望值摘要。")
        if self._signature and not _HEX_DIGEST.fullmatch(self._signature):
            raise TextTransportAuthenticationError("Companion IME command 签名格式无效。")

    def _wire_payload(self) -> dict[str, Any]:
        self.validate()
        return {"protocol_version": self.protocol_version,
            "operation": TEXT_TRANSPORT_NETWORK_OPERATIONS[self.operation], "scope": self.scope.to_dict(),
            "text": self._text}

    @property
    def command_digest(self) -> str:
        return hashlib.sha256(canonical_text_transport_json(self._wire_payload())).hexdigest()

    def signed(self, pairing_key: bytes) -> "TextTransportCommand":
        signature = sign_text_transport_payload(self._wire_payload(), pairing_key)
        return replace(self, _signature=signature)

    def to_wire_bytes(self) -> bytes:
        self.validate()
        if not self._signature:
            raise TextTransportAuthenticationError("Companion IME command 尚未签名。")
        value = canonical_text_transport_json({"command": self._wire_payload(), "signature": self._signature})
        if len(value) > TEXT_TRANSPORT_MAX_FRAME_BYTES:
            raise TextTransportFrameSizeError("Companion IME command 超过单帧上限，必须由上层显式分段。")
        return value

    def to_dict(self) -> dict[str, Any]:
        """Safe diagnostic view: never include plaintext or signature material."""

        self.validate()
        return {"protocol_version": self.protocol_version, "operation": self.operation, "scope": self.scope.to_dict(),
            "text_digest": self.scope.fragment_text_digest, "text_length": len(self._text or ""),
            "command_digest": self.command_digest, "signature_present": bool(self._signature)}

    @classmethod
    def create(cls, *, operation: str, scope: TextTransportActionScope, text: str | None,
        pairing_key: bytes) -> "TextTransportCommand":
        command = cls(protocol_version=TEXT_TRANSPORT_PROTOCOL, operation=operation, scope=scope, _text=text)
        command.validate()
        return command.signed(pairing_key)

    @classmethod
    def from_wire_bytes(cls, value: bytes, *, pairing_key: bytes,
        now_epoch: float) -> "TextTransportCommand":
        if not isinstance(value, bytes):
            raise TextTransportContractError("Companion IME wire command 必须是 bytes。")
        if not value or len(value) > TEXT_TRANSPORT_MAX_FRAME_BYTES:
            raise TextTransportFrameSizeError("Companion IME wire command 大小无效。")
        try:
            envelope = json.loads(value.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TextTransportContractError("Companion IME wire command 不是有效 JSON。") from exc
        envelope = _strict_mapping(envelope, cls._ENVELOPE_FIELDS, "Companion IME command envelope")
        payload = _strict_mapping(envelope["command"], cls._WIRE_FIELDS, "Companion IME command")
        verify_text_transport_signature(payload, envelope["signature"], pairing_key)
        network_operation = payload["operation"]
        reverse = {network: operation for operation, network in TEXT_TRANSPORT_NETWORK_OPERATIONS.items()}
        if network_operation not in reverse:
            raise TextTransportContractError("Companion IME wire operation 无效。")
        command = cls(protocol_version=payload["protocol_version"], operation=reverse[network_operation],
            scope=TextTransportActionScope.from_dict(payload["scope"]), _text=payload["text"],
            _signature=envelope["signature"])
        command.validate()
        command.scope.require_fresh(now_epoch)
        return command


@dataclass(frozen=True)
class TextTransportProfile:
    protocol_version: str
    profile_id: str
    device_id: str
    pairing_id: str
    enabled: bool
    capabilities: tuple[str, ...]
    ack_timeout_seconds: float

    _FIELDS = frozenset({"protocol_version", "profile_id", "device_id", "pairing_id", "enabled",
        "capabilities", "ack_timeout_seconds"})

    def validate(self) -> None:
        if self.protocol_version != TEXT_TRANSPORT_PROTOCOL:
            raise TextTransportContractError("Companion IME profile 协议版本不匹配。")
        for name in ("profile_id", "device_id", "pairing_id"):
            _identifier(getattr(self, name), name)
        if not isinstance(self.enabled, bool):
            raise TextTransportContractError("Companion IME profile enabled 无效。")
        if (not isinstance(self.capabilities, tuple) or not self.capabilities
            or len(set(self.capabilities)) != len(self.capabilities)
            or any(item not in TEXT_TRANSPORT_OPERATIONS for item in self.capabilities)):
            raise TextTransportContractError("Companion IME capabilities 无效。")
        timeout = _epoch(self.ack_timeout_seconds, "ack_timeout_seconds")
        if timeout <= 0 or timeout > 30:
            raise TextTransportContractError("Companion IME ACK timeout 无效。")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {name: (list(self.capabilities) if name == "capabilities" else getattr(self, name))
            for name in self._FIELDS}

    @classmethod
    def from_dict(cls, value: Any) -> "TextTransportProfile":
        item = _strict_mapping(value, cls._FIELDS, "Companion IME profile")
        profile = cls(protocol_version=item["protocol_version"], profile_id=item["profile_id"],
            device_id=item["device_id"], pairing_id=item["pairing_id"], enabled=item["enabled"],
            capabilities=tuple(item["capabilities"]) if isinstance(item["capabilities"], list) else (),
            ack_timeout_seconds=item["ack_timeout_seconds"])
        profile.validate()
        return profile


@dataclass(frozen=True)
class TextTransportResult:
    """Transport receipt only. ``accepted`` never means visual text success."""

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
            raise TextTransportContractError("Companion IME result 协议或 operation 无效。")
        for name in ("device_id", "action_id"):
            _identifier(getattr(self, name), name)
        if not isinstance(self.nonce, str) or not _NONCE.fullmatch(self.nonce):
            raise TextTransportContractError("Companion IME result nonce 无效。")
        if self.status not in {"unavailable", "unknown", "rejected", "accepted"}:
            raise TextTransportContractError("Companion IME result status 无效。")
        if not isinstance(self.attempted, bool) or not isinstance(self.accepted, bool):
            raise TextTransportContractError("Companion IME result 布尔字段无效。")
        valid_flags = {"unavailable": (False, False), "unknown": (True, False), "rejected": (True, False),
            "accepted": (True, True)}[self.status]
        if (self.attempted, self.accepted) != valid_flags:
            raise TextTransportContractError("Companion IME result 状态与 attempted/accepted 冲突。")
        if self.reason_code is not None and (not isinstance(self.reason_code, str)
            or not re.fullmatch(r"[a-z][a-z0-9_.-]{0,63}", self.reason_code)):
            raise TextTransportContractError("Companion IME reason_code 无效。")
        for name in ("command_digest", "receipt_digest"):
            value = getattr(self, name)
            if value is not None:
                _digest(value, name)
        if self.status in {"accepted", "rejected"} and (self.command_digest is None or self.receipt_digest is None):
            raise TextTransportContractError("已收到 ACK 的结果缺少摘要。")
        if self.visual_verification_required is not True:
            raise TextTransportContractError("Companion IME ACK 不能替代动作后视觉验证。")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {name: getattr(self, name) for name in self._FIELDS}

    @classmethod
    def from_dict(cls, value: Any) -> "TextTransportResult":
        item = _strict_mapping(value, cls._FIELDS, "Companion IME result")
        result = cls(**{name: item[name] for name in cls._FIELDS})
        result.validate()
        return result


__all__ = ["EMPTY_TEXT_DIGEST", "TEXT_TRANSPORT_MAX_FRAGMENT_UTF8_BYTES",
    "TEXT_TRANSPORT_MAX_FRAME_BYTES", "TEXT_TRANSPORT_NETWORK_OPERATIONS", "TEXT_TRANSPORT_OPERATIONS",
    "TEXT_TRANSPORT_PROTOCOL", "TextTransportActionScope",
    "TextTransportAuthenticationError", "TextTransportCommand", "TextTransportContractError",
    "TextTransportExpiredError", "TextTransportFrameSizeError", "TextTransportProfile",
    "TextTransportReplayError", "TextTransportResult", "TextTransportScopeError",
    "canonical_text_transport_json", "sign_text_transport_payload", "text_digest",
    "verify_text_transport_signature"]
