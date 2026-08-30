"""Authenticated, single-attempt Companion IME bridge infrastructure."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
import hashlib
import json
import math
import re
import secrets
import socket
import ssl
import struct
import threading
import time
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

from agent.domain.text_transport import (
    TEXT_TRANSPORT_MAX_FRAME_BYTES,
    TEXT_TRANSPORT_NETWORK_OPERATIONS,
    TEXT_TRANSPORT_PROTOCOL,
    TextTransportActionScope,
    TextTransportAuthenticationError,
    TextTransportCommand,
    TextTransportContractError,
    TextTransportFrameSizeError,
    TextTransportProfile,
    TextTransportReplayError,
    TextTransportResult,
    TextTransportScopeError,
    canonical_text_transport_json,
    sign_text_transport_payload,
    verify_text_transport_signature,
)
from agent.infrastructure.windows_companion_pairing_store import (
    StoredCompanionPairing,
    WindowsCompanionPairingStore,
)


_PAIRING_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}")
_REASON_CODE = re.compile(r"[a-z][a-z0-9_.-]{0,63}")
_ACK_FIELDS = frozenset({"protocol_version", "device_id", "action_id", "nonce", "operation", "status",
    "reason_code", "command_digest", "acknowledged_at_epoch"})
_HELLO_FIELDS = frozenset({"protocol_version", "type", "device_id", "pairing_id", "issued_at_epoch",
    "expires_at_epoch", "nonce"})
_READY_FIELDS = frozenset({"protocol_version", "type", "device_id", "pairing_id", "editor_session_id",
    "issued_at_epoch", "expires_at_epoch", "nonce"})
_PAIR_REQUEST_FIELDS = frozenset({"protocol_version", "type", "installation_id", "client_nonce",
    "one_time_token"})
_PAIR_CONFIRM_FIELDS = frozenset({"protocol_version", "type", "installation_id", "client_nonce",
    "pairing_id", "device_id"})
_PAIR_CONFIRM_ENVELOPE_FIELDS = frozenset({"confirm", "signature"})
_PAIR_COMMIT_ENVELOPE_FIELDS = frozenset({"commit", "signature"})
_MAX_FRAME_BYTES = TEXT_TRANSPORT_MAX_FRAME_BYTES


class PairingTokenRegistry:
    """Thread-safe in-memory credential store that never exposes key material."""

    def __init__(self) -> None:
        self._keys: dict[str, bytes] = {}
        self._guard = threading.RLock()

    def __repr__(self) -> str:
        with self._guard:
            count = len(self._keys)
        return f"PairingTokenRegistry(token_count={count})"

    def register(self, pairing_id: str, pairing_key: bytes) -> None:
        if not isinstance(pairing_id, str) or not _PAIRING_ID.fullmatch(pairing_id):
            raise TextTransportAuthenticationError("Companion IME pairing_id 无效。")
        if not isinstance(pairing_key, bytes) or len(pairing_key) < 32:
            raise TextTransportAuthenticationError("Companion IME 配对密钥无效。")
        with self._guard:
            existing = self._keys.get(pairing_id)
            if existing is not None and existing != pairing_key:
                raise TextTransportAuthenticationError("Companion IME pairing_id 已绑定其他密钥。")
            self._keys[pairing_id] = bytes(pairing_key)

    def resolve(self, pairing_id: str) -> bytes | None:
        with self._guard:
            value = self._keys.get(pairing_id)
            return bytes(value) if value is not None else None

    def revoke(self, pairing_id: str) -> None:
        with self._guard:
            self._keys.pop(pairing_id, None)

    def replace(self, pairing_id: str, pairing_key: bytes) -> None:
        """Atomically rotate one configured pairing after durable persistence."""

        if not isinstance(pairing_id, str) or not _PAIRING_ID.fullmatch(pairing_id):
            raise TextTransportAuthenticationError("Companion IME pairing_id 无效。")
        if not isinstance(pairing_key, bytes) or len(pairing_key) != 32:
            raise TextTransportAuthenticationError("Companion IME 配对密钥无效。")
        with self._guard:
            self._keys[pairing_id] = bytes(pairing_key)


@dataclass(frozen=True, repr=False)
class OneTimePairingGrant:
    device_id: str
    pairing_id: str
    expires_at_epoch: float
    _token: str = field(repr=False, compare=False)

    def __repr__(self) -> str:
        return (f"OneTimePairingGrant(device_id={self.device_id!r}, pairing_id={self.pairing_id!r}, "
            f"expires_at_epoch={self.expires_at_epoch!r}, token=<redacted>)")

    def reveal_token(self) -> str:
        """Return the one value the local setup UI must show to the user."""

        return self._token

    def to_dict(self) -> dict[str, Any]:
        return {"device_id": self.device_id, "pairing_id": self.pairing_id,
            "expires_at_epoch": self.expires_at_epoch, "token_present": True}


class OneTimePairingTokenRegistry:
    """In-memory, digest-only grants consumed atomically by one TLS request."""

    def __init__(self, *, clock: Callable[[], float]=time.time,
        token_factory: Callable[[int], str]=secrets.token_urlsafe) -> None:
        self._clock = clock
        self._token_factory = token_factory
        self._grants: dict[str, tuple[str, str, float]] = {}
        self._guard = threading.Lock()

    def __repr__(self) -> str:
        with self._guard:
            count = len(self._grants)
        return f"OneTimePairingTokenRegistry(grant_count={count})"

    @staticmethod
    def _digest(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def issue(self, profile: TextTransportProfile, *, ttl_seconds: float=300.0) -> OneTimePairingGrant:
        profile.validate()
        if (isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, (int, float))
            or not math.isfinite(float(ttl_seconds)) or not 10.0 <= float(ttl_seconds) <= 600.0):
            raise TextTransportContractError("Companion IME pairing token 有效期无效。")
        token = self._token_factory(32)
        if not isinstance(token, str) or not token or len(token) > 512:
            raise TextTransportAuthenticationError("Companion IME pairing token 生成失败。")
        expires = float(self._clock()) + float(ttl_seconds)
        digest = self._digest(token)
        with self._guard:
            self._grants[digest] = (profile.device_id, profile.pairing_id, expires)
        return OneTimePairingGrant(device_id=profile.device_id, pairing_id=profile.pairing_id,
            expires_at_epoch=expires, _token=token)

    def consume(self, token: str, profile: TextTransportProfile) -> None:
        profile.validate()
        if not isinstance(token, str) or not token or len(token) > 512:
            raise TextTransportAuthenticationError("Companion IME pairing token 无效。")
        now = float(self._clock())
        digest = self._digest(token)
        with self._guard:
            self._grants = {item: grant for item, grant in self._grants.items() if grant[2] >= now}
            grant = self._grants.pop(digest, None)
        if grant is None or grant[0] != profile.device_id or grant[1] != profile.pairing_id:
            raise TextTransportAuthenticationError("Companion IME pairing token 无效、过期或已使用。")

    def clear(self) -> None:
        with self._guard:
            self._grants.clear()


@dataclass(frozen=True, repr=False)
class _PendingCompanionImePairing:
    credential: StoredCompanionPairing = field(repr=False, compare=False)
    client_nonce: str
    expires_at_epoch: float

    def __repr__(self) -> str:
        return (f"_PendingCompanionImePairing(device_id={self.credential.device_id!r}, "
            f"pairing_id={self.credential.pairing_id!r}, installation_id={self.credential.installation_id!r}, "
            f"client_nonce={self.client_nonce!r}, expires_at_epoch={self.expires_at_epoch!r}, "
            "shared_key=<redacted>)")


@dataclass(frozen=True)
class CompletedCompanionImePairing:
    device_id: str
    pairing_id: str
    installation_id: str
    completed_at_epoch: float

    def to_dict(self) -> dict[str, Any]:
        return {"device_id": self.device_id, "pairing_id": self.pairing_id,
            "installation_id": self.installation_id, "completed_at_epoch": self.completed_at_epoch}


class CompanionImePairingAuthority:
    """Prepare one key, then persist and activate it only after a signed phone confirmation."""

    def __init__(self, profile: TextTransportProfile, one_time_tokens: OneTimePairingTokenRegistry,
        credential_store: WindowsCompanionPairingStore, pairing_tokens: PairingTokenRegistry, *,
        clock: Callable[[], float]=time.time, key_factory: Callable[[int], bytes]=secrets.token_bytes) -> None:
        profile.validate()
        self._profile = profile
        self._one_time_tokens = one_time_tokens
        self._credential_store = credential_store
        self._pairing_tokens = pairing_tokens
        self._clock = clock
        self._key_factory = key_factory
        self._pending: dict[str, _PendingCompanionImePairing] = {}
        self._awaiting_commit: dict[str, _PendingCompanionImePairing] = {}
        self._completed: CompletedCompanionImePairing | None = None
        self._pending_guard = threading.RLock()

    def issue_one_time_token(self, *, ttl_seconds: float=300.0) -> OneTimePairingGrant:
        return self._one_time_tokens.issue(self._profile, ttl_seconds=ttl_seconds)

    @staticmethod
    def is_pair_request(value: bytes) -> bool:
        try:
            request = json.loads(value.decode("utf-8"))
        except (AttributeError, UnicodeDecodeError, json.JSONDecodeError):
            return False
        return isinstance(request, Mapping) and request.get("type") == "pair_request"

    def restore_persisted_pairing(self) -> StoredCompanionPairing | None:
        credential = self._credential_store.load(self._profile.device_id)
        if credential is None:
            return None
        if credential.pairing_id != self._profile.pairing_id:
            raise TextTransportScopeError("持久 Companion IME pairing_id 与 profile 不一致。")
        self._pairing_tokens.replace(credential.pairing_id, credential.shared_key())
        return credential

    def close(self) -> None:
        """Forget unused setup grants and unconfirmed keys; active pairing remains intact."""

        self._one_time_tokens.clear()
        with self._pending_guard:
            self._pending.clear()
            self._awaiting_commit.clear()

    def completion_status(self) -> CompletedCompanionImePairing | None:
        with self._pending_guard:
            return self._completed

    def handle_pair_request(self, value: bytes) -> bytes:
        try:
            request = json.loads(value.decode("utf-8"))
        except (AttributeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TextTransportContractError("Companion IME pair_request 不是有效 JSON。") from exc
        if not isinstance(request, Mapping) or set(request) != _PAIR_REQUEST_FIELDS:
            raise TextTransportContractError("Companion IME pair_request 字段无效。")
        if request["protocol_version"] != TEXT_TRANSPORT_PROTOCOL or request["type"] != "pair_request":
            raise TextTransportContractError("Companion IME pair_request 协议无效。")
        installation_id = request["installation_id"]
        client_nonce = request["client_nonce"]
        token = request["one_time_token"]
        if (not isinstance(installation_id, str) or not _PAIRING_ID.fullmatch(installation_id)
            or not isinstance(client_nonce, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{15,127}", client_nonce)
            or not isinstance(token, str) or not token or len(token) > 512):
            raise TextTransportContractError("Companion IME pair_request 内容无效。")
        self._one_time_tokens.consume(token, self._profile)
        existing = self._credential_store.load(self._profile.device_id)
        if existing is not None and existing.pairing_id != self._profile.pairing_id:
            raise TextTransportScopeError("持久 Companion IME pairing_id 与 profile 不一致。")
        if existing is not None and existing.installation_id != installation_id:
            raise TextTransportScopeError(
                "已有 Companion IME 配对只允许同一 installation repair；换设备前必须显式撤销。")
        # Repair/synchronization reuses the current active key. Silent rotation would
        # create an unavoidable split-brain window across two independently durable
        # stores. Explicit revocation is required before generating a replacement key.
        shared_key = existing.shared_key() if existing is not None else self._key_factory(32)
        if not isinstance(shared_key, bytes) or len(shared_key) != 32:
            raise TextTransportAuthenticationError("Companion IME shared key 生成失败。")
        now = float(self._clock())
        credential = StoredCompanionPairing(device_id=self._profile.device_id,
            pairing_id=self._profile.pairing_id, installation_id=installation_id,
            created_at_epoch=now, _shared_key=shared_key)
        pending = _PendingCompanionImePairing(credential=credential, client_nonce=client_nonce,
            expires_at_epoch=now + 60.0)
        with self._pending_guard:
            self._pending = {nonce: item for nonce, item in self._pending.items()
                if item.expires_at_epoch >= now}
            if client_nonce in self._pending:
                raise TextTransportReplayError("Companion IME pair_request nonce 已使用。")
            self._pending[client_nonce] = pending
        response = {"protocol_version": TEXT_TRANSPORT_PROTOCOL, "type": "pair_response",
            "installation_id": installation_id, "client_nonce": client_nonce,
            "pairing_id": self._profile.pairing_id, "device_id": self._profile.device_id,
            "shared_key": base64.b64encode(shared_key).decode("ascii")}
        return canonical_text_transport_json(response)

    def handle_pair_confirm(self, value: bytes) -> bytes:
        """Commit a prepared key only after the Android client proves receipt with HMAC."""

        try:
            envelope = json.loads(value.decode("utf-8"))
        except (AttributeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TextTransportContractError("Companion IME pair_confirm 不是有效 JSON。") from exc
        if not isinstance(envelope, Mapping) or set(envelope) != _PAIR_CONFIRM_ENVELOPE_FIELDS:
            raise TextTransportContractError("Companion IME pair_confirm envelope 无效。")
        confirm = envelope["confirm"]
        if not isinstance(confirm, Mapping) or set(confirm) != _PAIR_CONFIRM_FIELDS:
            raise TextTransportContractError("Companion IME pair_confirm 字段无效。")
        if (confirm["protocol_version"] != TEXT_TRANSPORT_PROTOCOL or confirm["type"] != "pair_confirm"
            or confirm["pairing_id"] != self._profile.pairing_id
            or confirm["device_id"] != self._profile.device_id):
            raise TextTransportScopeError("Companion IME pair_confirm 不属于当前 profile。")
        installation_id = confirm["installation_id"]
        client_nonce = confirm["client_nonce"]
        if (not isinstance(installation_id, str) or not _PAIRING_ID.fullmatch(installation_id)
            or not isinstance(client_nonce, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{15,127}", client_nonce)):
            raise TextTransportContractError("Companion IME pair_confirm 内容无效。")
        now = float(self._clock())
        with self._pending_guard:
            self._pending = {nonce: item for nonce, item in self._pending.items()
                if item.expires_at_epoch >= now}
            pending = self._pending.get(client_nonce)
        if pending is None or pending.credential.installation_id != installation_id:
            raise TextTransportAuthenticationError("Companion IME pair_confirm 没有对应的待确认配对。")
        shared_key = pending.credential.shared_key()
        verify_text_transport_signature(confirm, envelope["signature"], shared_key)
        with self._pending_guard:
            self._pending.pop(client_nonce, None)
            self._awaiting_commit[client_nonce] = pending
        ack = {**confirm, "type": "pair_confirm_ack", "status": "accepted"}
        return canonical_text_transport_json({"confirm_ack": ack,
            "signature": sign_text_transport_payload(ack, shared_key)})

    def handle_pair_commit(self, value: bytes) -> bytes:
        """Activate a key only after Android has promoted its pending slot to active."""

        try:
            envelope = json.loads(value.decode("utf-8"))
        except (AttributeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TextTransportContractError("Companion IME pair_commit 不是有效 JSON。") from exc
        if not isinstance(envelope, Mapping) or set(envelope) != _PAIR_COMMIT_ENVELOPE_FIELDS:
            raise TextTransportContractError("Companion IME pair_commit envelope 无效。")
        commit = envelope["commit"]
        if not isinstance(commit, Mapping) or set(commit) != _PAIR_CONFIRM_FIELDS:
            raise TextTransportContractError("Companion IME pair_commit 字段无效。")
        if (commit["protocol_version"] != TEXT_TRANSPORT_PROTOCOL or commit["type"] != "pair_commit"
            or commit["pairing_id"] != self._profile.pairing_id
            or commit["device_id"] != self._profile.device_id):
            raise TextTransportScopeError("Companion IME pair_commit 不属于当前 profile。")
        installation_id = commit["installation_id"]
        client_nonce = commit["client_nonce"]
        if (not isinstance(installation_id, str) or not _PAIRING_ID.fullmatch(installation_id)
            or not isinstance(client_nonce, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{15,127}", client_nonce)):
            raise TextTransportContractError("Companion IME pair_commit 内容无效。")
        now = float(self._clock())
        with self._pending_guard:
            self._awaiting_commit = {nonce: item for nonce, item in self._awaiting_commit.items()
                if item.expires_at_epoch >= now}
            pending = self._awaiting_commit.get(client_nonce)
        if pending is None or pending.credential.installation_id != installation_id:
            raise TextTransportAuthenticationError("Companion IME pair_commit 没有对应的已确认配对。")
        shared_key = pending.credential.shared_key()
        verify_text_transport_signature(commit, envelope["signature"], shared_key)
        try:
            self._credential_store.save(pending.credential)
            self._pairing_tokens.replace(self._profile.pairing_id, shared_key)
        finally:
            with self._pending_guard:
                self._awaiting_commit.pop(client_nonce, None)
        completed = CompletedCompanionImePairing(device_id=self._profile.device_id,
            pairing_id=self._profile.pairing_id, installation_id=installation_id, completed_at_epoch=now)
        with self._pending_guard:
            self._completed = completed
        ack = {**commit, "type": "pair_commit_ack", "status": "accepted"}
        return canonical_text_transport_json({"commit_ack": ack,
            "signature": sign_text_transport_payload(ack, shared_key)})


class NonceReplayGuard:
    """Consume each canonical action nonce once and prune expired entries."""

    def __init__(self) -> None:
        self._seen: dict[tuple[str, str, str, str], float] = {}
        self._guard = threading.Lock()

    def consume(self, scope: TextTransportActionScope, *, now_epoch: float) -> None:
        scope.require_fresh(now_epoch)
        key = (scope.device_id, scope.session_id, scope.action_id, scope.nonce)
        with self._guard:
            self._seen = {item: expiry for item, expiry in self._seen.items() if expiry >= now_epoch}
            if key in self._seen:
                raise TextTransportReplayError("Companion IME action nonce 已使用。")
            self._seen[key] = float(scope.expires_at_epoch)


@runtime_checkable
class CompanionImeBridgeChannel(Protocol):
    """Encrypted channel carrying one request and its one ACK."""

    def is_secure(self) -> bool: ...

    def is_available(self) -> bool: ...

    def ready_editor_session_id(self) -> str | None: ...

    def exchange_once(self, payload: bytes, *, timeout_seconds: float) -> bytes: ...


def _unavailable(scope: TextTransportActionScope, operation: str, reason_code: str,
    command_digest: str | None=None) -> TextTransportResult:
    result = TextTransportResult(protocol_version=TEXT_TRANSPORT_PROTOCOL, device_id=scope.device_id,
        action_id=scope.action_id, nonce=scope.nonce, operation=operation, status="unavailable", attempted=False,
        accepted=False, reason_code=reason_code, command_digest=command_digest, receipt_digest=None)
    result.validate()
    return result


def _unknown(command: TextTransportCommand, reason_code: str) -> TextTransportResult:
    result = TextTransportResult(protocol_version=TEXT_TRANSPORT_PROTOCOL, device_id=command.scope.device_id,
        action_id=command.scope.action_id, nonce=command.scope.nonce, operation=command.operation, status="unknown",
        attempted=True, accepted=False, reason_code=reason_code, command_digest=command.command_digest,
        receipt_digest=None)
    result.validate()
    return result


def build_companion_ime_ack(command: TextTransportCommand, pairing_key: bytes, *, status: str,
    acknowledged_at_epoch: float, reason_code: str | None=None) -> bytes:
    """Build the Android-compatible signed ACK for protocol tests/endpoints."""

    if status not in {"accepted", "rejected"}:
        raise TextTransportContractError("Companion IME ACK status 无效。")
    if status == "accepted" and reason_code is not None:
        raise TextTransportContractError("accepted ACK 不得携带失败原因。")
    if status == "rejected" and (not isinstance(reason_code, str) or not _REASON_CODE.fullmatch(reason_code)):
        raise TextTransportContractError("rejected ACK 缺少有效 reason_code。")
    if (isinstance(acknowledged_at_epoch, bool) or not isinstance(acknowledged_at_epoch, (int, float))
        or not math.isfinite(float(acknowledged_at_epoch))):
        raise TextTransportContractError("Companion IME ACK 时间无效。")
    ack = {"protocol_version": TEXT_TRANSPORT_PROTOCOL, "device_id": command.scope.device_id,
        "action_id": command.scope.action_id, "nonce": command.scope.nonce,
        "operation": TEXT_TRANSPORT_NETWORK_OPERATIONS[command.operation], "status": status,
        "reason_code": reason_code, "command_digest": command.command_digest,
        "acknowledged_at_epoch": float(acknowledged_at_epoch)}
    return canonical_text_transport_json({"ack": ack, "signature": sign_text_transport_payload(ack, pairing_key)})


def _decode_ack(value: bytes, command: TextTransportCommand, pairing_key: bytes) -> TextTransportResult:
    try:
        envelope = json.loads(value.decode("utf-8"))
    except (AttributeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TextTransportContractError("Companion IME ACK 不是有效 JSON。") from exc
    if not isinstance(envelope, Mapping) or set(envelope) != {"ack", "signature"}:
        raise TextTransportContractError("Companion IME ACK envelope 字段无效。")
    ack = envelope["ack"]
    if not isinstance(ack, Mapping) or set(ack) != _ACK_FIELDS:
        raise TextTransportContractError("Companion IME ACK 字段无效。")
    verify_text_transport_signature(ack, envelope["signature"], pairing_key)
    expected = {"protocol_version": TEXT_TRANSPORT_PROTOCOL, "device_id": command.scope.device_id,
        "action_id": command.scope.action_id, "nonce": command.scope.nonce,
        "operation": TEXT_TRANSPORT_NETWORK_OPERATIONS[command.operation], "command_digest": command.command_digest}
    if any(ack[name] != expected[name] for name in expected):
        raise TextTransportScopeError("Companion IME ACK 未绑定当前 action scope。")
    status = ack["status"]
    reason = ack["reason_code"]
    if (status not in {"accepted", "rejected"} or (status == "accepted" and reason is not None)
        or (status == "rejected" and (not isinstance(reason, str) or not _REASON_CODE.fullmatch(reason)))):
        raise TextTransportContractError("Companion IME ACK 状态无效。")
    acknowledged = ack["acknowledged_at_epoch"]
    if (isinstance(acknowledged, bool) or not isinstance(acknowledged, (int, float))
        or not math.isfinite(float(acknowledged))):
        raise TextTransportContractError("Companion IME ACK 时间无效。")
    receipt_digest = hashlib.sha256(value).hexdigest()
    result = TextTransportResult(protocol_version=TEXT_TRANSPORT_PROTOCOL, device_id=command.scope.device_id,
        action_id=command.scope.action_id, nonce=command.scope.nonce, operation=command.operation, status=status,
        attempted=True, accepted=status == "accepted", reason_code=reason, command_digest=command.command_digest,
        receipt_digest=receipt_digest)
    result.validate()
    return result


class CompanionImeTextTransport:
    """Production client: preflight once, send once, never retry or claim visual success."""

    def __init__(self, profile: TextTransportProfile, channel: CompanionImeBridgeChannel,
        pairing_tokens: PairingTokenRegistry, *, clock: Callable[[], float]=time.time,
        outbound_replay_guard: NonceReplayGuard | None=None) -> None:
        profile.validate()
        self._profile = profile
        self._channel = channel
        self._pairing_tokens = pairing_tokens
        self._clock = clock
        self._outbound_replay_guard = outbound_replay_guard or NonceReplayGuard()

    @property
    def profile(self) -> TextTransportProfile:
        return self._profile

    def mint_action_scope(self, *, session_id: str, task_id: str, revision: int, action_id: str,
        input_field_id: str, observation_fingerprint: str, prior_text_digest: str,
        fragment_text_digest: str, expected_text_digest: str) -> TextTransportActionScope:
        """Bind the current ready editor to one short-lived canonical action."""

        now = float(self._clock())
        try:
            editor_session_id = self._channel.ready_editor_session_id()
        except Exception as exc:
            raise TextTransportScopeError("Companion IME 当前编辑连接不可用。") from exc
        if not isinstance(editor_session_id, str) or not editor_session_id:
            raise TextTransportScopeError("Companion IME 当前没有 ready editor session。")
        scope = TextTransportActionScope(protocol_version=TEXT_TRANSPORT_PROTOCOL,
            device_id=self._profile.device_id, session_id=session_id, task_id=task_id,
            revision=revision, action_id=action_id, input_field_id=input_field_id,
            editor_session_id=editor_session_id, observation_fingerprint=observation_fingerprint,
            prior_text_digest=prior_text_digest, fragment_text_digest=fragment_text_digest,
            expected_text_digest=expected_text_digest, issued_at_epoch=now,
            expires_at_epoch=now + min(15.0, max(2.0, float(self._profile.ack_timeout_seconds) + 5.0)),
            nonce=secrets.token_hex(16))
        scope.validate()
        return scope

    def append_text(self, scope: TextTransportActionScope, text: str) -> TextTransportResult:
        return self._execute("append_text", scope, text)

    def clear_text(self, scope: TextTransportActionScope) -> TextTransportResult:
        return self._execute("clear_text", scope, None)

    def _execute(self, operation: str, scope: TextTransportActionScope, text: str | None) -> TextTransportResult:
        now = float(self._clock())
        scope.require_fresh(now)
        if scope.device_id != self._profile.device_id:
            raise TextTransportScopeError("Companion IME profile 与 action device_id 不一致。")
        if not self._profile.enabled:
            return _unavailable(scope, operation, "profile_disabled")
        if operation not in self._profile.capabilities:
            return _unavailable(scope, operation, "capability_unavailable")
        pairing_key = self._pairing_tokens.resolve(self._profile.pairing_id)
        if pairing_key is None:
            return _unavailable(scope, operation, "pairing_unavailable")
        command = TextTransportCommand.create(operation=operation, scope=scope, text=text,
            pairing_key=pairing_key)
        try:
            payload = command.to_wire_bytes()
        except TextTransportFrameSizeError:
            return _unavailable(scope, operation, "frame_too_large", command.command_digest)
        self._outbound_replay_guard.consume(scope, now_epoch=now)
        try:
            secure = self._channel.is_secure()
            available = self._channel.is_available() if secure else False
            editor_session_id = self._channel.ready_editor_session_id() if available else None
        except Exception:
            return _unavailable(scope, operation, "bridge_offline", command.command_digest)
        if not secure:
            return _unavailable(scope, operation, "bridge_insecure", command.command_digest)
        if not available:
            return _unavailable(scope, operation, "bridge_offline", command.command_digest)
        if editor_session_id != scope.editor_session_id:
            return _unavailable(scope, operation, "editor_session_mismatch", command.command_digest)
        try:
            ack = self._channel.exchange_once(payload,
                timeout_seconds=float(self._profile.ack_timeout_seconds))
        except Exception:
            return _unknown(command, "ack_unknown")
        try:
            return _decode_ack(ack, command, pairing_key)
        except TextTransportContractError:
            return _unknown(command, "ack_invalid")


class CompanionImeCommandVerifier:
    """Receiver-side HMAC, expiry, device and replay guard shared with bridge tests."""

    def __init__(self, profile: TextTransportProfile, pairing_tokens: PairingTokenRegistry, *,
        clock: Callable[[], float]=time.time, replay_guard: NonceReplayGuard | None=None) -> None:
        profile.validate()
        self._profile = profile
        self._pairing_tokens = pairing_tokens
        self._clock = clock
        self._replay_guard = replay_guard or NonceReplayGuard()

    def verify_once(self, value: bytes) -> TextTransportCommand:
        pairing_key = self._pairing_tokens.resolve(self._profile.pairing_id)
        if pairing_key is None:
            raise TextTransportAuthenticationError("Companion IME pairing 不可用。")
        now = float(self._clock())
        command = TextTransportCommand.from_wire_bytes(value, pairing_key=pairing_key, now_epoch=now)
        if not self._profile.enabled or command.scope.device_id != self._profile.device_id:
            raise TextTransportScopeError("Companion IME command 不属于当前 profile。")
        if command.operation not in self._profile.capabilities:
            raise TextTransportScopeError("Companion IME command capability 未启用。")
        self._replay_guard.consume(command.scope, now_epoch=now)
        return command


def _send_frame(connection: socket.socket, payload: bytes) -> None:
    if not isinstance(payload, bytes) or not payload or len(payload) > _MAX_FRAME_BYTES:
        raise TextTransportContractError("Companion IME bridge frame 大小无效。")
    connection.sendall(struct.pack("!I", len(payload)) + payload)


def _receive_exact(connection: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise ConnectionError("Companion IME bridge connection closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _receive_frame(connection: socket.socket) -> bytes:
    size = struct.unpack("!I", _receive_exact(connection, 4))[0]
    if size <= 0 or size > _MAX_FRAME_BYTES:
        raise TextTransportContractError("Companion IME bridge frame 大小无效。")
    return _receive_exact(connection, size)


def build_companion_ime_bridge_hello(profile: TextTransportProfile, pairing_key: bytes, *, nonce: str,
    issued_at_epoch: float, expires_at_epoch: float) -> bytes:
    """Build the TLS-session hello used by the Android outbound bridge."""

    profile.validate()
    hello = {"protocol_version": TEXT_TRANSPORT_PROTOCOL, "type": "bridge_hello",
        "device_id": profile.device_id, "pairing_id": profile.pairing_id,
        "issued_at_epoch": float(issued_at_epoch), "expires_at_epoch": float(expires_at_epoch), "nonce": nonce}
    if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{15,127}", nonce)
        or not math.isfinite(hello["issued_at_epoch"]) or not math.isfinite(hello["expires_at_epoch"])
        or hello["expires_at_epoch"] <= hello["issued_at_epoch"]):
        raise TextTransportContractError("Companion IME bridge hello 无效。")
    return canonical_text_transport_json({"hello": hello, "signature": sign_text_transport_payload(hello,
        pairing_key)})


def build_companion_ime_editor_ready(profile: TextTransportProfile, pairing_key: bytes, *,
    editor_session_id: str, nonce: str, issued_at_epoch: float, expires_at_epoch: float) -> bytes:
    """Build the signed ready frame for the currently active Android InputConnection."""

    profile.validate()
    ready = {"protocol_version": TEXT_TRANSPORT_PROTOCOL, "type": "editor_ready",
        "device_id": profile.device_id, "pairing_id": profile.pairing_id,
        "editor_session_id": editor_session_id, "issued_at_epoch": float(issued_at_epoch),
        "expires_at_epoch": float(expires_at_epoch), "nonce": nonce}
    if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}", editor_session_id)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{15,127}", nonce)
        or not math.isfinite(ready["issued_at_epoch"]) or not math.isfinite(ready["expires_at_epoch"])
        or ready["expires_at_epoch"] <= ready["issued_at_epoch"]):
        raise TextTransportContractError("Companion IME editor ready 无效。")
    return canonical_text_transport_json({"ready": ready, "signature": sign_text_transport_payload(ready,
        pairing_key)})


class TlsCompanionImeBridgeServer:
    """Length-prefixed TLS channel accepting one authenticated device connection."""

    def __init__(self, profile: TextTransportProfile, pairing_tokens: PairingTokenRegistry,
        ssl_context: ssl.SSLContext, *, host: str="0.0.0.0", port: int=0,
        clock: Callable[[], float]=time.time, handshake_timeout_seconds: float=5.0,
        pairing_authority: CompanionImePairingAuthority | None=None) -> None:
        profile.validate()
        if not isinstance(ssl_context, ssl.SSLContext):
            raise TypeError("ssl_context must be ssl.SSLContext")
        if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
            raise ValueError("port is invalid")
        self._profile = profile
        self._pairing_tokens = pairing_tokens
        self._ssl_context = ssl_context
        self._host = host
        self._port = port
        self._clock = clock
        self._handshake_timeout = float(handshake_timeout_seconds)
        self._pairing_authority = pairing_authority
        self._listener: socket.socket | None = None
        self._connection: ssl.SSLSocket | None = None
        self._editor_session_id: str | None = None
        self._connection_guard = threading.RLock()
        self._exchange_guard = threading.Lock()
        self._hello_nonces: dict[str, float] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __repr__(self) -> str:
        return (f"TlsCompanionImeBridgeServer(device_id={self._profile.device_id!r}, address={self.address!r}, "
            f"available={self.is_available()!r})")

    @property
    def address(self) -> tuple[str, int] | None:
        listener = self._listener
        if listener is None:
            return None
        host, port = listener.getsockname()[:2]
        return str(host), int(port)

    def start(self) -> tuple[str, int]:
        if self._thread is not None:
            address = self.address
            if address is None:
                raise RuntimeError("Companion IME TLS bridge 未正常启动。")
            return address
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((self._host, self._port))
        listener.listen(1)
        listener.settimeout(0.5)
        self._listener = listener
        self._stop.clear()
        self._thread = threading.Thread(target=self._accept_loop, name="companion-ime-tls-bridge", daemon=True)
        self._thread.start()
        address = self.address
        if address is None:
            raise RuntimeError("Companion IME TLS bridge 未正常启动。")
        return address

    def stop(self) -> None:
        self._stop.set()
        listener, self._listener = self._listener, None
        if listener is not None:
            try:
                listener.close()
            except OSError:
                pass
        self._drop_connection()
        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)

    def is_secure(self) -> bool:
        return True

    def is_available(self) -> bool:
        with self._connection_guard:
            return self._connection is not None and self._editor_session_id is not None

    def ready_editor_session_id(self) -> str | None:
        with self._connection_guard:
            return self._editor_session_id

    def exchange_once(self, payload: bytes, *, timeout_seconds: float) -> bytes:
        with self._exchange_guard:
            with self._connection_guard:
                connection = self._connection
            if connection is None:
                raise ConnectionError("Companion IME bridge offline")
            try:
                connection.settimeout(float(timeout_seconds))
                _send_frame(connection, payload)
                return _receive_frame(connection)
            except Exception:
                self._drop_connection(connection)
                raise

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            listener = self._listener
            if listener is None:
                return
            try:
                raw, _address = listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            connection: ssl.SSLSocket | None = None
            try:
                raw.settimeout(self._handshake_timeout)
                connection = self._ssl_context.wrap_socket(raw, server_side=True)
                hello = _receive_frame(connection)
                if (self._pairing_authority is not None
                    and self._pairing_authority.is_pair_request(hello)):
                    response = self._pairing_authority.handle_pair_request(hello)
                    _send_frame(connection, response)
                    confirm = _receive_frame(connection)
                    confirm_ack = self._pairing_authority.handle_pair_confirm(confirm)
                    _send_frame(connection, confirm_ack)
                    commit = _receive_frame(connection)
                    commit_ack = self._pairing_authority.handle_pair_commit(commit)
                    _send_frame(connection, commit_ack)
                    self._drop_connection()
                    connection.close()
                    continue
                ack = self._verify_hello(hello)
                _send_frame(connection, ack)
                ready = _receive_frame(connection)
                editor_session_id, ready_ack = self._verify_ready(ready)
                _send_frame(connection, ready_ack)
                connection.settimeout(None)
                with self._connection_guard:
                    previous = self._connection
                    self._connection = connection
                    self._editor_session_id = editor_session_id
                if previous is not None:
                    try:
                        previous.close()
                    except OSError:
                        pass
            except Exception:
                try:
                    (connection if connection is not None else raw).close()
                except OSError:
                    pass

    def _verify_hello(self, value: bytes) -> bytes:
        try:
            envelope = json.loads(value.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TextTransportContractError("Companion IME bridge hello 不是有效 JSON。") from exc
        if not isinstance(envelope, Mapping) or set(envelope) != {"hello", "signature"}:
            raise TextTransportContractError("Companion IME bridge hello envelope 无效。")
        hello = envelope["hello"]
        if not isinstance(hello, Mapping) or set(hello) != _HELLO_FIELDS:
            raise TextTransportContractError("Companion IME bridge hello 字段无效。")
        if (hello["protocol_version"] != TEXT_TRANSPORT_PROTOCOL or hello["type"] != "bridge_hello"
            or hello["device_id"] != self._profile.device_id or hello["pairing_id"] != self._profile.pairing_id):
            raise TextTransportScopeError("Companion IME bridge hello 不属于当前 profile。")
        key = self._pairing_tokens.resolve(self._profile.pairing_id)
        if key is None:
            raise TextTransportAuthenticationError("Companion IME pairing 不可用。")
        verify_text_transport_signature(hello, envelope["signature"], key)
        now = float(self._clock())
        issued, expires = hello["issued_at_epoch"], hello["expires_at_epoch"]
        nonce = hello["nonce"]
        if (isinstance(issued, bool) or isinstance(expires, bool) or not isinstance(issued, (int, float))
            or not isinstance(expires, (int, float)) or not math.isfinite(float(issued))
            or not math.isfinite(float(expires)) or now < float(issued) or now > float(expires)
            or not isinstance(nonce, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{15,127}", nonce)):
            raise TextTransportContractError("Companion IME bridge hello 已过期或无效。")
        self._hello_nonces = {item: expiry for item, expiry in self._hello_nonces.items() if expiry >= now}
        if nonce in self._hello_nonces:
            raise TextTransportReplayError("Companion IME bridge hello nonce 已使用。")
        self._hello_nonces[nonce] = float(expires)
        ack = {"protocol_version": TEXT_TRANSPORT_PROTOCOL, "type": "bridge_hello_ack",
            "device_id": self._profile.device_id, "pairing_id": self._profile.pairing_id, "nonce": nonce,
            "status": "accepted"}
        return canonical_text_transport_json({"hello_ack": ack, "signature": sign_text_transport_payload(ack, key)})

    def _verify_ready(self, value: bytes) -> tuple[str, bytes]:
        try:
            envelope = json.loads(value.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TextTransportContractError("Companion IME editor ready 不是有效 JSON。") from exc
        if not isinstance(envelope, Mapping) or set(envelope) != {"ready", "signature"}:
            raise TextTransportContractError("Companion IME editor ready envelope 无效。")
        ready = envelope["ready"]
        if not isinstance(ready, Mapping) or set(ready) != _READY_FIELDS:
            raise TextTransportContractError("Companion IME editor ready 字段无效。")
        if (ready["protocol_version"] != TEXT_TRANSPORT_PROTOCOL or ready["type"] != "editor_ready"
            or ready["device_id"] != self._profile.device_id or ready["pairing_id"] != self._profile.pairing_id):
            raise TextTransportScopeError("Companion IME editor ready 不属于当前 profile。")
        key = self._pairing_tokens.resolve(self._profile.pairing_id)
        if key is None:
            raise TextTransportAuthenticationError("Companion IME pairing 不可用。")
        verify_text_transport_signature(ready, envelope["signature"], key)
        now = float(self._clock())
        issued, expires = ready["issued_at_epoch"], ready["expires_at_epoch"]
        nonce, editor_session_id = ready["nonce"], ready["editor_session_id"]
        if (isinstance(issued, bool) or isinstance(expires, bool) or not isinstance(issued, (int, float))
            or not isinstance(expires, (int, float)) or not math.isfinite(float(issued))
            or not math.isfinite(float(expires)) or now < float(issued) or now > float(expires)
            or not isinstance(nonce, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{15,127}", nonce)
            or not isinstance(editor_session_id, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}", editor_session_id)):
            raise TextTransportContractError("Companion IME editor ready 已过期或无效。")
        replay_key = f"ready:{nonce}"
        self._hello_nonces = {item: expiry for item, expiry in self._hello_nonces.items() if expiry >= now}
        if replay_key in self._hello_nonces:
            raise TextTransportReplayError("Companion IME editor ready nonce 已使用。")
        self._hello_nonces[replay_key] = float(expires)
        ack = {"protocol_version": TEXT_TRANSPORT_PROTOCOL, "type": "editor_ready_ack",
            "device_id": self._profile.device_id, "pairing_id": self._profile.pairing_id,
            "editor_session_id": editor_session_id, "nonce": nonce, "status": "accepted"}
        return editor_session_id, canonical_text_transport_json({"ready_ack": ack,
            "signature": sign_text_transport_payload(ack, key)})

    def _drop_connection(self, expected: ssl.SSLSocket | None=None) -> None:
        with self._connection_guard:
            if expected is not None and self._connection is not expected:
                return
            connection, self._connection = self._connection, None
            self._editor_session_id = None
        if connection is not None:
            try:
                connection.close()
            except OSError:
                pass


__all__ = ["CompanionImeBridgeChannel", "CompanionImeCommandVerifier", "CompanionImePairingAuthority",
    "CompletedCompanionImePairing",
    "CompanionImeTextTransport", "NonceReplayGuard", "OneTimePairingGrant", "OneTimePairingTokenRegistry",
    "PairingTokenRegistry", "TlsCompanionImeBridgeServer", "build_companion_ime_ack",
    "build_companion_ime_bridge_hello", "build_companion_ime_editor_ready"]
