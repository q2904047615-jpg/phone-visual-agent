import base64
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from agent.domain.text_transport import (
    TEXT_TRANSPORT_PROTOCOL,
    TextTransportAuthenticationError,
    TextTransportContractError,
    TextTransportProfile,
    TextTransportScopeError,
    sign_text_transport_payload,
)
from agent.infrastructure.companion_ime_transport import (
    CompanionImePairingAuthority,
    OneTimePairingTokenRegistry,
    PairingTokenRegistry,
)
from agent.infrastructure.windows_companion_pairing_store import (
    StoredCompanionPairing,
    WindowsCompanionPairingStore,
    WindowsDpapiProtector,
)


NOW = 1_800_000_000.0
SHARED_KEY = bytes(range(32))


def _profile() -> TextTransportProfile:
    return TextTransportProfile(protocol_version=TEXT_TRANSPORT_PROTOCOL, profile_id="profile-1",
        device_id="device-1", pairing_id="pairing-1", enabled=True,
        capabilities=("append_text", "clear_text"), ack_timeout_seconds=2.0)


class _MaskProtector:
    @staticmethod
    def _mask(entropy: bytes) -> bytes:
        return hashlib.sha256(b"test-protector\0" + entropy).digest()

    def protect(self, plaintext: bytes, *, entropy: bytes) -> bytes:
        mask = self._mask(entropy)
        return b"protected-v1:" + bytes(value ^ mask[index % len(mask)] for index, value in enumerate(plaintext))

    def unprotect(self, ciphertext: bytes, *, entropy: bytes) -> bytes:
        if not ciphertext.startswith(b"protected-v1:"):
            raise ValueError("invalid protected value")
        value = ciphertext[len(b"protected-v1:"):]
        mask = self._mask(entropy)
        return bytes(item ^ mask[index % len(mask)] for index, item in enumerate(value))


def _request(token: str, *, extra: bool=False, installation_id: str="installation-1") -> bytes:
    value = {"protocol_version": TEXT_TRANSPORT_PROTOCOL, "type": "pair_request",
        "installation_id": installation_id, "client_nonce": "client-nonce-000001",
        "one_time_token": token}
    if extra:
        value["shared_key"] = "forbidden"
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _confirm(response: dict, *, key: bytes=SHARED_KEY) -> bytes:
    value = {"protocol_version": TEXT_TRANSPORT_PROTOCOL, "type": "pair_confirm",
        "installation_id": response["installation_id"], "client_nonce": response["client_nonce"],
        "pairing_id": response["pairing_id"], "device_id": response["device_id"]}
    return json.dumps({"confirm": value, "signature": sign_text_transport_payload(value, key)},
        ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _commit(response: dict, *, key: bytes=SHARED_KEY) -> bytes:
    value = {"protocol_version": TEXT_TRANSPORT_PROTOCOL, "type": "pair_commit",
        "installation_id": response["installation_id"], "client_nonce": response["client_nonce"],
        "pairing_id": response["pairing_id"], "device_id": response["device_id"]}
    return json.dumps({"commit": value, "signature": sign_text_transport_payload(value, key)},
        ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


class CompanionImePairingTests(unittest.TestCase):
    def _authority(self, root: Path, *, clock=lambda: NOW, key_factory=lambda _size: SHARED_KEY):
        profile = _profile()
        grants = OneTimePairingTokenRegistry(clock=clock,
            token_factory=lambda _size: "one-time-token-secret-000001")
        live = PairingTokenRegistry()
        store = WindowsCompanionPairingStore(root, protector=_MaskProtector())
        authority = CompanionImePairingAuthority(profile, grants, store, live, clock=clock,
            key_factory=key_factory)
        return authority, grants, live, store

    def test_exact_android_pair_request_returns_exact_bound_response(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            authority, _grants, live, store = self._authority(Path(temp))
            grant = authority.issue_one_time_token(ttl_seconds=60)
            token = grant.reveal_token()
            response = json.loads(authority.handle_pair_request(_request(token)))

            self.assertEqual({"protocol_version", "type", "installation_id", "client_nonce", "pairing_id",
                "device_id", "shared_key"}, set(response))
            self.assertEqual(TEXT_TRANSPORT_PROTOCOL, response["protocol_version"])
            self.assertEqual("pair_response", response["type"])
            self.assertEqual("installation-1", response["installation_id"])
            self.assertEqual("client-nonce-000001", response["client_nonce"])
            self.assertEqual("pairing-1", response["pairing_id"])
            self.assertEqual("device-1", response["device_id"])
            self.assertEqual(SHARED_KEY, base64.b64decode(response["shared_key"], validate=True))
            self.assertIsNone(live.resolve("pairing-1"))
            self.assertIsNone(store.load("device-1"))

            confirm_ack = json.loads(authority.handle_pair_confirm(_confirm(response)))
            self.assertEqual("accepted", confirm_ack["confirm_ack"]["status"])
            self.assertIsNone(live.resolve("pairing-1"))
            self.assertIsNone(store.load("device-1"))
            commit_ack = json.loads(authority.handle_pair_commit(_commit(response)))
            self.assertEqual("accepted", commit_ack["commit_ack"]["status"])
            self.assertEqual(SHARED_KEY, live.resolve("pairing-1"))
            self.assertEqual(SHARED_KEY, store.load("device-1").shared_key())
            self.assertEqual("installation-1", authority.completion_status().installation_id)

    def test_token_is_digest_only_redacted_and_consumed_exactly_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            authority, grants, _live, _store = self._authority(Path(temp))
            grant = authority.issue_one_time_token(ttl_seconds=60)
            token = grant.reveal_token()
            self.assertNotIn(token, repr(grant))
            self.assertNotIn(token, json.dumps(grant.to_dict()))
            self.assertNotIn(token, repr(grants))
            self.assertNotIn(token, repr(grants._grants))

            authority.handle_pair_request(_request(token))
            with self.assertRaises(TextTransportAuthenticationError):
                authority.handle_pair_request(_request(token))

            unused = authority.issue_one_time_token(ttl_seconds=60).reveal_token()
            authority.close()
            with self.assertRaises(TextTransportAuthenticationError):
                authority.handle_pair_request(_request(unused))

    def test_expired_token_and_extra_request_fields_are_fail_closed(self) -> None:
        now = [NOW]
        with tempfile.TemporaryDirectory() as temp:
            authority, _grants, _live, _store = self._authority(Path(temp), clock=lambda: now[0])
            grant = authority.issue_one_time_token(ttl_seconds=10)
            token = grant.reveal_token()
            with self.assertRaises(TextTransportContractError):
                authority.handle_pair_request(_request(token, extra=True))
            # A malformed request never consumes the valid grant.
            self.assertEqual("pair_response", json.loads(authority.handle_pair_request(_request(token)))["type"])

            next_grant = authority.issue_one_time_token(ttl_seconds=10)
            now[0] = NOW + 11
            with self.assertRaises(TextTransportAuthenticationError):
                authority.handle_pair_request(_request(next_grant.reveal_token()))

    def test_key_and_token_are_never_persisted_in_plaintext(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            authority, _grants, _live, store = self._authority(root)
            token = authority.issue_one_time_token(ttl_seconds=60).reveal_token()
            response = json.loads(authority.handle_pair_request(_request(token)))
            authority.handle_pair_confirm(_confirm(response))
            authority.handle_pair_commit(_commit(response))
            path = store._path("device-1")
            raw = path.read_bytes()

            self.assertNotIn(token.encode(), raw)
            self.assertNotIn(SHARED_KEY, raw)
            self.assertNotIn(base64.b64encode(SHARED_KEY), raw)
            record = json.loads(raw)
            self.assertEqual({"record_version", "protocol_version", "device_id", "pairing_id",
                "installation_id", "created_at_epoch", "protected_shared_key"}, set(record))
            safe = store.load("device-1")
            self.assertNotIn(base64.b64encode(SHARED_KEY).decode(), repr(safe))
            self.assertNotIn("protected_shared_key", safe.to_dict())

    def test_persistence_failure_consumes_token_before_any_response_or_live_key(self) -> None:
        class FailingStore:
            def save(self, _credential):
                raise OSError("disk unavailable")

            def load(self, _device_id):
                return None

        profile = _profile()
        grants = OneTimePairingTokenRegistry(clock=lambda: NOW,
            token_factory=lambda _size: "one-time-token-secret-000001")
        live = PairingTokenRegistry()
        authority = CompanionImePairingAuthority(profile, grants, FailingStore(), live, clock=lambda: NOW,
            key_factory=lambda _size: SHARED_KEY)
        token = authority.issue_one_time_token(ttl_seconds=60).reveal_token()
        response = json.loads(authority.handle_pair_request(_request(token)))
        self.assertIsNone(live.resolve("pairing-1"))
        authority.handle_pair_confirm(_confirm(response))
        with self.assertRaises(OSError):
            authority.handle_pair_commit(_commit(response))
        self.assertIsNone(live.resolve("pairing-1"))
        with self.assertRaises(TextTransportAuthenticationError):
            authority.handle_pair_request(_request(token))

    def test_persisted_pairing_restores_and_repair_reuses_the_active_key(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first, _grants, _live, _store = self._authority(root)
            token = first.issue_one_time_token(ttl_seconds=60).reveal_token()
            response = json.loads(first.handle_pair_request(_request(token)))
            first.handle_pair_confirm(_confirm(response))
            first.handle_pair_commit(_commit(response))

            profile = _profile()
            fresh_live = PairingTokenRegistry()
            def unexpected_key_factory(_size: int) -> bytes:
                raise AssertionError("repair must not silently rotate the active key")

            fresh_authority = CompanionImePairingAuthority(profile,
                OneTimePairingTokenRegistry(clock=lambda: NOW),
                WindowsCompanionPairingStore(root, protector=_MaskProtector()), fresh_live,
                clock=lambda: NOW, key_factory=unexpected_key_factory)
            restored = fresh_authority.restore_persisted_pairing()
            self.assertEqual(SHARED_KEY, restored.shared_key())
            self.assertEqual(SHARED_KEY, fresh_live.resolve("pairing-1"))

            fresh_authority._one_time_tokens = OneTimePairingTokenRegistry(clock=lambda: NOW,
                token_factory=lambda _size: "rotated-token-secret-000001")
            rotated_token = fresh_authority.issue_one_time_token(ttl_seconds=60).reveal_token()
            rotated_response = json.loads(fresh_authority.handle_pair_request(_request(rotated_token)))
            self.assertEqual(SHARED_KEY, base64.b64decode(rotated_response["shared_key"], validate=True))
            self.assertEqual(SHARED_KEY, fresh_live.resolve("pairing-1"))
            fresh_authority.handle_pair_confirm(_confirm(rotated_response))
            self.assertEqual(SHARED_KEY, fresh_live.resolve("pairing-1"))
            fresh_authority.handle_pair_commit(_commit(rotated_response))
            self.assertEqual(SHARED_KEY, fresh_live.resolve("pairing-1"))

    def test_unconfirmed_or_tampered_repair_never_replaces_the_existing_key(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            authority, _grants, live, _store = self._authority(root)
            initial = json.loads(authority.handle_pair_request(_request(
                authority.issue_one_time_token(ttl_seconds=60).reveal_token())))
            authority.handle_pair_confirm(_confirm(initial))
            authority.handle_pair_commit(_commit(initial))
            self.assertEqual(SHARED_KEY, live.resolve("pairing-1"))

            authority._one_time_tokens = OneTimePairingTokenRegistry(clock=lambda: NOW,
                token_factory=lambda _size: "repair-token-secret-000001")
            pending = json.loads(authority.handle_pair_request(_request(
                authority.issue_one_time_token(ttl_seconds=60).reveal_token())))
            self.assertEqual(SHARED_KEY, live.resolve("pairing-1"))
            with self.assertRaises(TextTransportAuthenticationError):
                authority.handle_pair_confirm(_confirm(pending, key=b"x" * 32))
            self.assertEqual(SHARED_KEY, live.resolve("pairing-1"))

    def test_repair_never_shares_one_active_key_with_another_installation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            authority, _grants, live, store = self._authority(Path(temp))
            initial = json.loads(authority.handle_pair_request(_request(
                authority.issue_one_time_token(ttl_seconds=60).reveal_token())))
            authority.handle_pair_confirm(_confirm(initial))
            authority.handle_pair_commit(_commit(initial))

            authority._one_time_tokens = OneTimePairingTokenRegistry(clock=lambda: NOW,
                token_factory=lambda _size: "other-install-token-000001")
            token = authority.issue_one_time_token(ttl_seconds=60).reveal_token()
            with self.assertRaisesRegex(TextTransportScopeError, "installation"):
                authority.handle_pair_request(_request(token, installation_id="installation-2"))
            self.assertEqual(SHARED_KEY, live.resolve("pairing-1"))
            self.assertEqual("installation-1", store.load("device-1").installation_id)

    def test_pair_request_detection_never_accepts_envelopes_or_other_types(self) -> None:
        self.assertTrue(CompanionImePairingAuthority.is_pair_request(_request("token")))
        self.assertFalse(CompanionImePairingAuthority.is_pair_request(b'{"type":"bridge_hello"}'))
        self.assertFalse(CompanionImePairingAuthority.is_pair_request(b'{"hello":{"type":"pair_request"}}'))
        self.assertFalse(CompanionImePairingAuthority.is_pair_request(b"not-json"))

    def test_android_repair_keeps_active_key_until_signed_confirm_ack(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "android" / "companion-ime" / "app" / "src" /
            "main" / "java" / "com" / "visualagent" / "companionime" / "security" /
            "PairingClient.java").read_text(encoding="utf-8")
        save_pending = source.index("pairingStore.savePending(record)")
        send_confirm = source.index("PairingMessages.pairConfirm(record, clientNonce)")
        verify_ack = source.index("PairingMessages.verifyPairConfirmAck(")
        promote = source.index("pairingStore.promotePending()")
        send_commit = source.index("PairingMessages.pairCommit(record, clientNonce)")
        verify_commit = source.index("PairingMessages.verifyPairCommitAck(")
        self.assertLess(save_pending, send_confirm)
        self.assertLess(send_confirm, verify_ack)
        self.assertLess(verify_ack, promote)
        self.assertLess(promote, send_commit)
        self.assertLess(send_commit, verify_commit)
        self.assertNotIn("pairingStore.save(record)", source)

    @unittest.skipIf(os.name == "nt", "Non-Windows fail-closed behavior only")
    def test_dpapi_adapter_fails_closed_off_windows(self) -> None:
        with self.assertRaises(RuntimeError):
            WindowsDpapiProtector().protect(SHARED_KEY, entropy=b"test")

    @unittest.skipUnless(os.name == "nt", "Windows DPAPI round trip only")
    def test_windows_dpapi_round_trip_uses_current_user_protection(self) -> None:
        protector = WindowsDpapiProtector()
        protected = protector.protect(SHARED_KEY, entropy=b"companion-ime-test-entropy")
        self.assertNotEqual(SHARED_KEY, protected)
        self.assertEqual(SHARED_KEY, protector.unprotect(protected,
            entropy=b"companion-ime-test-entropy"))

    def test_store_rejects_unknown_record_fields_and_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            store = WindowsCompanionPairingStore(Path(temp), protector=_MaskProtector())
            credential = StoredCompanionPairing(device_id="device-1", pairing_id="pairing-1",
                installation_id="installation-1", created_at_epoch=NOW, _shared_key=SHARED_KEY)
            path = store.save(credential)
            value = json.loads(path.read_text(encoding="utf-8"))
            value["plaintext_key"] = base64.b64encode(SHARED_KEY).decode()
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(TextTransportContractError):
                store.load("device-1")


if __name__ == "__main__":
    unittest.main()
