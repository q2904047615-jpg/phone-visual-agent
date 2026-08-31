import hashlib
import json
import socket
import ssl
import unittest

from agent.domain.text_transport import (
    EMPTY_TEXT_DIGEST,
    TEXT_TRANSPORT_MAX_FRAME_BYTES,
    TEXT_TRANSPORT_PROTOCOL,
    TextTransportActionScope,
    TextTransportAuthenticationError,
    TextTransportCommand,
    TextTransportContractError,
    TextTransportExpiredError,
    TextTransportFrameSizeError,
    TextTransportProfile,
    TextTransportReplayError,
    TextTransportScopeError,
    sign_text_transport_payload,
    text_digest,
)
from agent.domain.foreground_app_identity import FOREGROUND_APP_IDENTITY_PROTOCOL
from agent.infrastructure import companion_ime_transport as bridge_module
from agent.infrastructure.companion_ime_transport import (
    CompanionImeCommandVerifier,
    CompanionImeTextTransport,
    PairingTokenRegistry,
    TlsCompanionImeBridgeServer,
    build_companion_foreground_state,
    build_companion_ime_ack,
    build_companion_ime_bridge_hello,
    build_companion_ime_editor_ready,
)


NOW = 1_800_000_000.0
PAIRING_KEY = b"companion-ime-test-key-32-bytes!!"


def _profile(*, device_id: str="device-1", enabled: bool=True) -> TextTransportProfile:
    return TextTransportProfile(protocol_version=TEXT_TRANSPORT_PROTOCOL, profile_id="profile-1",
        device_id=device_id, pairing_id="pairing-1", enabled=enabled,
        capabilities=("append_text", "clear_text"), ack_timeout_seconds=2.0)


def _scope(text: str | None, *, device_id: str="device-1", nonce: str="nonce-000000000001",
    issued_at: float=NOW - 1, expires_at: float=NOW + 10, prior: str="") -> TextTransportActionScope:
    fragment = "" if text is None else text
    expected = "" if text is None else prior + text
    return TextTransportActionScope(protocol_version=TEXT_TRANSPORT_PROTOCOL, device_id=device_id,
        session_id="session-1", task_id="task-1", revision=7, action_id="action-1",
        input_field_id="input-field-1", editor_session_id="editor-session-1",
        observation_fingerprint="fingerprint-1",
        prior_text_digest=text_digest(prior), fragment_text_digest=text_digest(fragment),
        expected_text_digest=text_digest(expected), issued_at_epoch=issued_at, expires_at_epoch=expires_at,
        nonce=nonce)


def _registry() -> PairingTokenRegistry:
    registry = PairingTokenRegistry()
    registry.register("pairing-1", PAIRING_KEY)
    return registry


class _AckChannel:
    def __init__(self, *, status: str="accepted", reason_code: str | None=None) -> None:
        self.calls = 0
        self.status = status
        self.reason_code = reason_code

    def is_secure(self) -> bool:
        return True

    def is_available(self) -> bool:
        return True

    def ready_editor_session_id(self) -> str | None:
        return "editor-session-1"

    def exchange_once(self, payload: bytes, *, timeout_seconds: float) -> bytes:
        self.calls += 1
        self.timeout_seconds = timeout_seconds
        command = TextTransportCommand.from_wire_bytes(payload, pairing_key=PAIRING_KEY, now_epoch=NOW)
        return build_companion_ime_ack(command, PAIRING_KEY, status=self.status,
            reason_code=self.reason_code, acknowledged_at_epoch=NOW)


class _OfflineChannel:
    def __init__(self, *, secure: bool=True) -> None:
        self.secure = secure
        self.calls = 0

    def is_secure(self) -> bool:
        return self.secure

    def is_available(self) -> bool:
        return False

    def ready_editor_session_id(self) -> str | None:
        return None

    def exchange_once(self, payload: bytes, *, timeout_seconds: float) -> bytes:
        self.calls += 1
        raise AssertionError("offline transport must not send")


class _LostAckChannel(_AckChannel):
    def exchange_once(self, payload: bytes, *, timeout_seconds: float) -> bytes:
        self.calls += 1
        raise TimeoutError("payload text must never be copied to diagnostics")


class CompanionImeDomainContractTests(unittest.TestCase):
    def test_unicode_emoji_and_lf_round_trip_with_redacted_safe_views(self) -> None:
        value = "aaazjie？你好\n🙂"
        command = TextTransportCommand.create(operation="append_text", scope=_scope(value), text=value,
            pairing_key=PAIRING_KEY)
        decoded = TextTransportCommand.from_wire_bytes(command.to_wire_bytes(), pairing_key=PAIRING_KEY,
            now_epoch=NOW)

        self.assertEqual(value, decoded.text)
        self.assertEqual(text_digest(value), decoded.scope.fragment_text_digest)
        self.assertEqual("commit_text", json.loads(command.to_wire_bytes())["command"]["operation"])
        for safe_view in (repr(command), json.dumps(command.to_dict(), ensure_ascii=False)):
            self.assertNotIn(value, safe_view)
            self.assertNotIn("你好", safe_view)
            self.assertNotIn("🙂", safe_view)
            self.assertNotIn(command._signature, safe_view)

    def test_fragment_digest_mismatch_and_carriage_return_are_rejected(self) -> None:
        valid = _scope("hello")
        invalid = TextTransportActionScope(**{**valid.to_dict(), "fragment_text_digest": text_digest("other")})
        with self.assertRaises(TextTransportScopeError):
            TextTransportCommand.create(operation="append_text", scope=invalid, text="hello",
                pairing_key=PAIRING_KEY)
        with self.assertRaises(TextTransportContractError):
            TextTransportCommand.create(operation="append_text", scope=_scope("a\rb"), text="a\rb",
                pairing_key=PAIRING_KEY)

    def test_clear_has_no_plaintext_and_requires_empty_digests(self) -> None:
        command = TextTransportCommand.create(operation="clear_text", scope=_scope(None), text=None,
            pairing_key=PAIRING_KEY)
        payload = json.loads(command.to_wire_bytes())
        self.assertEqual("clear_text", payload["command"]["operation"])
        self.assertIsNone(payload["command"]["text"])
        self.assertEqual(EMPTY_TEXT_DIGEST, command.scope.expected_text_digest)

        invalid = TextTransportActionScope(**{**_scope(None).to_dict(),
            "expected_text_digest": text_digest("not-empty")})
        with self.assertRaises(TextTransportScopeError):
            TextTransportCommand.create(operation="clear_text", scope=invalid, text=None,
                pairing_key=PAIRING_KEY)

    def test_wire_and_profile_fields_are_strict(self) -> None:
        command = TextTransportCommand.create(operation="append_text", scope=_scope("abc"), text="abc",
            pairing_key=PAIRING_KEY)
        envelope = json.loads(command.to_wire_bytes())
        envelope["command"]["legacy_fallback"] = True
        envelope["signature"] = sign_text_transport_payload(envelope["command"], PAIRING_KEY)
        with self.assertRaisesRegex(TextTransportContractError, "额外字段"):
            TextTransportCommand.from_wire_bytes(json.dumps(envelope).encode(), pairing_key=PAIRING_KEY,
                now_epoch=NOW)

        profile = _profile().to_dict()
        profile["adb_serial"] = "forbidden"
        with self.assertRaisesRegex(TextTransportContractError, "额外字段"):
            TextTransportProfile.from_dict(profile)

    def test_signature_tampering_is_rejected(self) -> None:
        command = TextTransportCommand.create(operation="append_text", scope=_scope("abc"), text="abc",
            pairing_key=PAIRING_KEY)
        envelope = json.loads(command.to_wire_bytes())
        envelope["command"]["text"] = "abd"
        with self.assertRaises(TextTransportAuthenticationError):
            TextTransportCommand.from_wire_bytes(json.dumps(envelope).encode(), pairing_key=PAIRING_KEY,
                now_epoch=NOW)

    def test_complete_authenticated_envelope_enforces_the_shared_frame_limit(self) -> None:
        value = "a" * TEXT_TRANSPORT_MAX_FRAME_BYTES
        command = TextTransportCommand.create(operation="append_text", scope=_scope(value), text=value,
            pairing_key=PAIRING_KEY)
        with self.assertRaises(TextTransportFrameSizeError):
            command.to_wire_bytes()
        with self.assertRaises(TextTransportFrameSizeError):
            TextTransportCommand.from_wire_bytes(b"x" * (TEXT_TRANSPORT_MAX_FRAME_BYTES + 1),
                pairing_key=PAIRING_KEY, now_epoch=NOW)


class CompanionImeClientTests(unittest.TestCase):
    def test_accepted_ack_is_one_attempt_and_still_requires_visual_verification(self) -> None:
        channel = _AckChannel()
        transport = CompanionImeTextTransport(_profile(), channel, _registry(), clock=lambda: NOW)
        result = transport.append_text(_scope("你好🙂\nnext"), "你好🙂\nnext")

        self.assertEqual(1, channel.calls)
        self.assertEqual(2.0, channel.timeout_seconds)
        self.assertTrue(result.attempted)
        self.assertTrue(result.accepted)
        self.assertEqual("accepted", result.status)
        self.assertTrue(result.visual_verification_required)
        self.assertNotIn("success", result.to_dict())

    def test_rejected_ack_is_not_retried(self) -> None:
        channel = _AckChannel(status="rejected", reason_code="input_connection_unavailable")
        transport = CompanionImeTextTransport(_profile(), channel, _registry(), clock=lambda: NOW)
        result = transport.append_text(_scope("abc"), "abc")
        self.assertEqual(1, channel.calls)
        self.assertEqual("rejected", result.status)
        self.assertTrue(result.attempted)
        self.assertFalse(result.accepted)

    def test_ack_loss_is_unknown_after_one_send_and_never_retried(self) -> None:
        channel = _LostAckChannel()
        transport = CompanionImeTextTransport(_profile(), channel, _registry(), clock=lambda: NOW)
        result = transport.append_text(_scope("private-text"), "private-text")
        self.assertEqual(1, channel.calls)
        self.assertEqual("unknown", result.status)
        self.assertTrue(result.attempted)
        self.assertFalse(result.accepted)
        self.assertEqual("ack_unknown", result.reason_code)
        self.assertNotIn("private-text", repr(result))

    def test_offline_or_insecure_preflight_is_not_an_attempt(self) -> None:
        for secure, reason in ((True, "bridge_offline"), (False, "bridge_insecure")):
            with self.subTest(secure=secure):
                channel = _OfflineChannel(secure=secure)
                result = CompanionImeTextTransport(_profile(), channel, _registry(), clock=lambda: NOW).append_text(
                    _scope("abc", nonce=f"nonce-0000000000{int(secure)}"), "abc")
                self.assertEqual(0, channel.calls)
                self.assertFalse(result.attempted)
                self.assertEqual("unavailable", result.status)
                self.assertEqual(reason, result.reason_code)

    def test_expired_wrong_device_and_outbound_replay_stop_before_send(self) -> None:
        channel = _AckChannel()
        transport = CompanionImeTextTransport(_profile(), channel, _registry(), clock=lambda: NOW)
        with self.assertRaises(TextTransportExpiredError):
            transport.append_text(_scope("abc", issued_at=NOW - 20, expires_at=NOW - 1), "abc")
        with self.assertRaises(TextTransportScopeError):
            transport.append_text(_scope("abc", device_id="device-2"), "abc")
        scope = _scope("abc", nonce="nonce-000000000099")
        self.assertTrue(transport.append_text(scope, "abc").accepted)
        with self.assertRaises(TextTransportReplayError):
            transport.append_text(scope, "abc")
        self.assertEqual(1, channel.calls)

    def test_ready_editor_session_mismatch_stops_before_send(self) -> None:
        channel = _AckChannel()
        channel.ready_editor_session_id = lambda: "editor-session-new"
        result = CompanionImeTextTransport(_profile(), channel, _registry(), clock=lambda: NOW).append_text(
            _scope("abc"), "abc")
        self.assertEqual(0, channel.calls)
        self.assertFalse(result.attempted)
        self.assertEqual("editor_session_mismatch", result.reason_code)

    def test_receiver_replay_guard_and_wrong_profile_scope(self) -> None:
        registry = _registry()
        command = TextTransportCommand.create(operation="append_text", scope=_scope("abc"), text="abc",
            pairing_key=PAIRING_KEY)
        verifier = CompanionImeCommandVerifier(_profile(), registry, clock=lambda: NOW)
        self.assertEqual("abc", verifier.verify_once(command.to_wire_bytes()).text)
        with self.assertRaises(TextTransportReplayError):
            verifier.verify_once(command.to_wire_bytes())

        wrong = CompanionImeCommandVerifier(_profile(device_id="device-2"), registry, clock=lambda: NOW)
        with self.assertRaises(TextTransportScopeError):
            wrong.verify_once(command.to_wire_bytes())

    def test_pairing_secret_and_plaintext_are_absent_from_safe_objects(self) -> None:
        registry = _registry()
        self.assertNotIn(PAIRING_KEY.decode(), repr(registry))
        command = TextTransportCommand.create(operation="append_text", scope=_scope("secret-body"),
            text="secret-body", pairing_key=PAIRING_KEY)
        self.assertNotIn("secret-body", repr(command))
        self.assertNotIn("secret-body", json.dumps(command.to_dict()))
        self.assertNotIn(PAIRING_KEY.decode(), repr(command))

    def test_oversized_command_is_rejected_before_any_send_or_attempt(self) -> None:
        value = "你" * TEXT_TRANSPORT_MAX_FRAME_BYTES
        channel = _AckChannel()
        transport = CompanionImeTextTransport(_profile(), channel, _registry(), clock=lambda: NOW)
        scope = _scope(value, nonce="nonce-oversized-000001")

        for _index in range(2):
            result = transport.append_text(scope, value)
            self.assertEqual("unavailable", result.status)
            self.assertEqual("frame_too_large", result.reason_code)
            self.assertFalse(result.attempted)
        self.assertEqual(0, channel.calls)


class CompanionImeTlsBridgeTests(unittest.TestCase):
    def test_length_prefixed_frame_round_trip(self) -> None:
        left, right = socket.socketpair()
        try:
            bridge_module._send_frame(left, b'{"test":true}')
            self.assertEqual(b'{"test":true}', bridge_module._receive_frame(right))
        finally:
            left.close()
            right.close()

    def test_signed_bridge_hello_is_fresh_and_replay_guarded(self) -> None:
        profile = _profile()
        registry = _registry()
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        server = TlsCompanionImeBridgeServer(profile, registry, context, clock=lambda: NOW)
        hello = build_companion_ime_bridge_hello(profile, PAIRING_KEY, nonce="hello-000000000001",
            issued_at_epoch=NOW - 1, expires_at_epoch=NOW + 10)
        ack = json.loads(server._verify_hello(hello))
        self.assertEqual("accepted", ack["hello_ack"]["status"])
        with self.assertRaises(TextTransportReplayError):
            server._verify_hello(hello)
        ready = build_companion_ime_editor_ready(profile, PAIRING_KEY, editor_session_id="editor-session-1",
            nonce="ready-000000000001", issued_at_epoch=NOW - 1, expires_at_epoch=NOW + 10)
        editor_session_id, ready_ack_bytes = server._verify_ready(ready)
        self.assertEqual("editor-session-1", editor_session_id)
        ready_ack = json.loads(ready_ack_bytes)
        self.assertEqual("accepted", ready_ack["ready_ack"]["status"])
        with self.assertRaises(TextTransportReplayError):
            server._verify_ready(ready)
        self.assertTrue(server.is_secure())
        self.assertFalse(server.is_available())

    def test_signed_foreground_identity_is_fresh_replay_guarded_and_can_be_cleared(self) -> None:
        profile = _profile()
        clock = [NOW]
        server = TlsCompanionImeBridgeServer(profile, _registry(), ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER),
            clock=lambda: clock[0])
        state = build_companion_foreground_state(profile, PAIRING_KEY, package_name="com.tencent.mm",
            source="usage_stats", event_at_epoch=NOW - 20, observed_at_epoch=NOW,
            reason_code=None, nonce="foreground-000000001", issued_at_epoch=NOW - 1,
            expires_at_epoch=NOW + 10)

        ack = json.loads(server._verify_foreground_state(state))
        self.assertEqual(FOREGROUND_APP_IDENTITY_PROTOCOL, ack["foreground_state_ack"]["protocol_version"])
        identity = server.foreground_app_identity()
        self.assertIsNotNone(identity)
        self.assertEqual("com.tencent.mm", identity.package_name)
        self.assertEqual("usage_stats", identity.source)
        with self.assertRaises(TextTransportReplayError):
            server._verify_foreground_state(state)

        unavailable = build_companion_foreground_state(profile, PAIRING_KEY, package_name=None,
            source="unavailable", event_at_epoch=None, observed_at_epoch=NOW, reason_code="usage_access_not_granted",
            nonce="foreground-000000002", issued_at_epoch=NOW - 1, expires_at_epoch=NOW + 10)
        server._verify_foreground_state(unavailable)
        self.assertIsNone(server.foreground_app_identity())
        self.assertEqual("usage_access_not_granted", server.foreground_identity_status()["reason_code"])

    def test_foreground_identity_rejects_tamper_wrong_device_and_staleness(self) -> None:
        profile = _profile()
        clock = [NOW]
        server = TlsCompanionImeBridgeServer(profile, _registry(), ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER),
            clock=lambda: clock[0])
        state = build_companion_foreground_state(profile, PAIRING_KEY, package_name="com.android.settings",
            source="editor_info", event_at_epoch=NOW - 2, observed_at_epoch=NOW,
            reason_code=None, nonce="foreground-000000011", issued_at_epoch=NOW - 1,
            expires_at_epoch=NOW + 10)
        tampered = json.loads(state)
        tampered["foreground_state"]["package_name"] = "com.tencent.mm"
        with self.assertRaises(TextTransportAuthenticationError):
            server._verify_foreground_state(json.dumps(tampered).encode("utf-8"))

        wrong_state = build_companion_foreground_state(_profile(device_id="device-2"), PAIRING_KEY,
            package_name="com.android.settings", source="usage_stats", event_at_epoch=NOW - 2,
            observed_at_epoch=NOW, reason_code=None, nonce="foreground-000000012",
            issued_at_epoch=NOW - 1, expires_at_epoch=NOW + 10)
        with self.assertRaises(TextTransportScopeError):
            server._verify_foreground_state(wrong_state)

        server._verify_foreground_state(state)
        clock[0] = NOW + 7
        self.assertIsNone(server.foreground_app_identity())


if __name__ == "__main__":
    unittest.main()
