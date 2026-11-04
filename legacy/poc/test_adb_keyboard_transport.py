from __future__ import annotations
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from agent.domain.text_transport import (EMPTY_TEXT_DIGEST, TEXT_TRANSPORT_PROTOCOL,
    TextTransportProfile, TextTransportReplayError, text_digest)
from agent.infrastructure.adb_keyboard_transport import (ADB_KEYBOARD_IME_ID,
    ADB_KEYBOARD_RUNTIME_REGISTRY_VERSION, AdbKeyboardRuntimeRegistry, AdbKeyboardTextTransport)


class ScriptedRunner:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append((list(argv), dict(kwargs)))
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def completed(stdout: str = "", *, returncode: int = 0, stderr: str = ""):
    return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr=stderr)


def profile(*, serial: str = "serial-01") -> TextTransportProfile:
    return TextTransportProfile(protocol_version=TEXT_TRANSPORT_PROTOCOL, profile_id="adb-keyboard-01",
        device_id="device-local-01", adb_serial=serial, enabled=True,
        capabilities=("append_text", "clear_text"), command_timeout_seconds=5.0)


class AdbKeyboardTransportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.adb = Path(self.directory.name) / "adb.exe"
        self.adb.write_bytes(b"test")

    def tearDown(self):
        self.directory.cleanup()

    def transport(self, runner: ScriptedRunner) -> AdbKeyboardTextTransport:
        return AdbKeyboardTextTransport(profile(), self.adb, runner=runner, clock=lambda: 100.0,
            nonce_factory=lambda: "nonce-0000000000001")

    @staticmethod
    def scope(transport: AdbKeyboardTextTransport, *, fragment: str, expected: str):
        return transport.mint_action_scope(session_id="session-1", task_id="task-1", revision=1,
            action_id="action-1", input_field_id="field-1", observation_fingerprint="fingerprint-1",
            prior_text_digest=text_digest(""), fragment_text_digest=text_digest(fragment),
            expected_text_digest=text_digest(expected))

    @staticmethod
    def ready_results(final):
        return [completed("device\n"), completed(ADB_KEYBOARD_IME_ID + "\n"),
            completed(ADB_KEYBOARD_IME_ID + "\n"), final]

    def test_unicode_input_uses_one_fixed_base64_broadcast_after_read_only_preflight(self):
        runner = ScriptedRunner(self.ready_results(completed("Broadcast completed: result=0\n")))
        transport = self.transport(runner)
        text = "aaazjie？你好🙂"
        result = transport.append_text(self.scope(transport, fragment=text, expected=text), text)

        self.assertTrue(result.accepted)
        self.assertTrue(result.visual_verification_required)
        command = runner.calls[-1][0]
        self.assertEqual([str(self.adb), "-s", "serial-01", "shell", "am", "broadcast", "-a",
            "ADB_INPUT_B64", "--es", "msg", base64.b64encode(text.encode("utf-8")).decode("ascii")], command)
        self.assertNotIn(text, json.dumps(result.to_dict(), ensure_ascii=False))

    def test_clear_is_a_separate_fixed_broadcast_without_input_payload(self):
        runner = ScriptedRunner(self.ready_results(completed("Broadcast completed: result=0\n")))
        transport = self.transport(runner)
        result = transport.clear_text(self.scope(transport, fragment="", expected=""))

        self.assertTrue(result.accepted)
        self.assertEqual(EMPTY_TEXT_DIGEST, transport.profile.to_dict() and text_digest(""))
        self.assertEqual([str(self.adb), "-s", "serial-01", "shell", "am", "broadcast", "-a",
            "ADB_CLEAR_TEXT"], runner.calls[-1][0])

    def test_command_and_receipt_digests_keep_exact_wire_bytes(self):
        def expected(value):
            return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                separators=(',', ':')).encode('utf-8')).hexdigest()

        for text in ('你好🙂e\u0301\n', ''):
            with self.subTest(text=text):
                reply = completed('Broadcast completed: result=0\n')
                runner = ScriptedRunner(self.ready_results(reply))
                transport = self.transport(runner)
                scope = self.scope(transport, fragment=text, expected=text)
                operation = 'append_text' if text else 'clear_text'
                result = transport.append_text(scope, text) if text else transport.clear_text(scope)
                command_digest = expected({'protocol_version': TEXT_TRANSPORT_PROTOCOL,
                    'operation': operation, 'device_id': transport.profile.device_id,
                    'adb_serial': transport.profile.adb_serial, 'scope': scope.to_dict(),
                    'payload_digest': text_digest(text)})
                self.assertEqual(command_digest, result.command_digest)
                self.assertEqual(expected({'returncode': reply.returncode, 'stdout': reply.stdout,
                    'stderr': reply.stderr, 'command_digest': command_digest}), result.receipt_digest)

    def test_offline_device_stops_before_any_broadcast(self):
        runner = ScriptedRunner([completed("unknown\n", returncode=1)])
        transport = self.transport(runner)
        text = "hello"
        result = transport.append_text(self.scope(transport, fragment=text, expected=text), text)

        self.assertEqual("unavailable", result.status)
        self.assertEqual("device_offline", result.reason_code)
        self.assertFalse(result.attempted)
        self.assertEqual(1, len(runner.calls))

    def test_unselected_ime_stops_before_broadcast(self):
        runner = ScriptedRunner([completed("device\n"), completed(ADB_KEYBOARD_IME_ID + "\n"),
            completed("com.example.other/.Ime\n")])
        transport = self.transport(runner)
        text = "hello"
        result = transport.append_text(self.scope(transport, fragment=text, expected=text), text)

        self.assertEqual("ime_not_selected", result.reason_code)
        self.assertFalse(result.attempted)
        self.assertEqual(3, len(runner.calls))

    def test_consumed_scope_cannot_broadcast_twice(self):
        runner = ScriptedRunner(self.ready_results(completed("Broadcast completed: result=0\n")))
        transport = self.transport(runner)
        text = "hello"
        scope = self.scope(transport, fragment=text, expected=text)
        transport.append_text(scope, text)

        with self.assertRaises(TextTransportReplayError):
            transport.append_text(scope, text)
        self.assertEqual(4, len(runner.calls))

    def test_registry_binds_fixed_device_and_serial(self):
        registry = Path(self.directory.name) / "registry.json"
        registry.write_text(json.dumps({"version": ADB_KEYBOARD_RUNTIME_REGISTRY_VERSION, "devices": [{
            "profile": profile().to_dict(), "adb_executable": str(self.adb)}]}), encoding="utf-8")
        runtime = AdbKeyboardRuntimeRegistry(registry, runner=ScriptedRunner([]))

        self.assertEqual(("device-local-01",), runtime.configured_device_ids)
        self.assertEqual("serial-01", runtime.transport_for_device("device-local-01").profile.adb_serial)
        self.assertIsNone(runtime.transport_for_device("other-device"))

    def test_transport_exposes_no_page_observation_or_foreground_identity_authority(self):
        source = (Path(__file__).resolve().parent / "agent" / "infrastructure"
            / "adb_keyboard_transport.py").read_text(encoding="utf-8")
        runtime = AdbKeyboardRuntimeRegistry(Path(self.directory.name) / "missing.json")

        for forbidden in ("dumpsys", "foreground_app_identity", "foreground_identity_for_device",
            "ForegroundAppIdentity"):
            self.assertNotIn(forbidden, source)
        self.assertFalse(hasattr(runtime, "foreground_identity_for_device"))


if __name__ == "__main__":
    unittest.main()
