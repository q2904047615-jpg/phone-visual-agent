from __future__ import annotations

import json
from dataclasses import replace
import secrets
import subprocess
import unittest
from unittest.mock import Mock, patch

from agent.bootstrap.runtime import Runtime
from agent.domain.text_transport import EMPTY_TEXT_DIGEST, TEXT_TRANSPORT_PROTOCOL, text_digest
from agent.infrastructure.adb_keyboard_transport import ADB_KEYBOARD_IME_ID, ADB_KEYBOARD_RUNTIME_REGISTRY_VERSION
from agent.infrastructure.adb_package_launcher import AdbPackageLauncherError
from agent.infrastructure.connector_journal import ConnectorJournal
from agent.infrastructure.remote_connector import ConnectorProtocolError, validate_command
from agent.infrastructure.remote_device_tools import LocalDeviceTools, RemoteTextTransport, RemoteAppLauncher, describe_remote_tools
import test_remote_connector as connector_fixtures
from test_remote_connector import CONNECTOR, DEVICE, command


class RemoteDeviceToolsTests(unittest.TestCase):
    handle = connector_fixtures.ConnectorWorkerTests.handle
    roundtrip = connector_fixtures.ConnectorWorkerTests.roundtrip

    def setUp(self):
        connector_fixtures.ConnectorWorkerTests.setUp(self)
        self.hardware.machine_position = 1
        self.registry = self.path / "device_registry.json"
        self.registry.write_text(json.dumps({"version": 1, "default_device_id": DEVICE, "devices": [
            {"enabled": True, "device_id": DEVICE, "machine_position": 1},
            {"enabled": True, "device_id": "device-local-02", "machine_position": 2},
            {"enabled": False, "device_id": "device-local-03", "machine_position": 3},
        ]}), encoding="utf-8")
        self.adb = self.path / "adb.exe"
        self.adb.touch()
        self.text_path = self.path / "adb_keyboard_registry.json"
        self.app_path = self.path / "app_package_registry.json"
        profiles = []
        apps = []
        for station in (1, 2, 3):
            device = f"device-local-0{station}"
            profiles.append({"adb_executable": str(self.adb), "profile": {
                "protocol_version": TEXT_TRANSPORT_PROTOCOL, "profile_id": f"keyboard-{station}",
                "device_id": device, "adb_serial": f"serial-{station}", "enabled": True,
                "capabilities": ["append_text", "clear_text"], "command_timeout_seconds": 2}})
            apps.append({"device_id": device, "enabled": True, "adb_executable": str(self.adb),
                "adb_serial": f"serial-{station}", "apps": [{"launch_ref": f"trusted-{station}",
                "aliases": ["测试应用"], "package": f"org.example.app{station}"}]})
        self.text_path.write_text(json.dumps({"version": ADB_KEYBOARD_RUNTIME_REGISTRY_VERSION, "devices": profiles}), encoding="utf-8")
        self.app_path.write_text(json.dumps({"version": 1, "devices": apps}), encoding="utf-8")
        self.calls = []
        self.timeout_broadcast = False
        self.worker.tools = self.local_tools()

    def local_tools(self):
        tools = LocalDeviceTools(registry_path=self.registry, device_id=DEVICE,
            text_registry_path=self.text_path, app_registry_path=self.app_path)
        for transport in tools.texts._transports.values():
            transport._runner = self.adb_run
        for launcher in tools.launchers.values():
            launcher._runner = self.adb_run
        return tools

    def adb_run(self, argv, **kwargs):
        self.calls.append(argv)
        if "broadcast" in argv:
            if self.timeout_broadcast:
                raise subprocess.TimeoutExpired(argv, 2)
            output = "Broadcast completed: result=0"
        elif "get-state" in argv:
            output = "device"
        elif "ime" in argv or "default_input_method" in argv:
            output = ADB_KEYBOARD_IME_ID
        else:
            output = "Events injected: 1"
        return subprocess.CompletedProcess(argv, 0, output, "")

    def adapters(self):
        description = self.roundtrip(describe_remote_tools, self.proxy)
        return RemoteTextTransport(self.proxy, description), RemoteAppLauncher(self.proxy, description)

    def scope(self, transport, text="你好 🌍"):
        return transport.mint_action_scope(session_id="session", task_id="task", revision=1,
            action_id="action", input_field_id="field", observation_fingerprint="scene",
            prior_text_digest=EMPTY_TEXT_DIGEST, fragment_text_digest=text_digest(text), expected_text_digest=text_digest(text))

    def broadcasts(self):
        return [call for call in self.calls if "broadcast" in call]

    def test_unicode_newline_clear_and_launch_each_observe_then_one_operation(self):
        text, app = self.adapters()
        self.assertTrue(self.roundtrip(text.status)["ready"])
        for payload in ("你好 🌍", "\n"):
            self.roundtrip(self.proxy.vision_capture)
            receipt = self.roundtrip(text.append_text, self.scope(text, payload), payload)
            self.assertTrue(receipt.accepted)
            self.assertTrue(receipt.visual_verification_required)
        self.roundtrip(self.proxy.vision_capture)
        self.assertTrue(self.roundtrip(text.clear_text, self.scope(text, "")).accepted)
        self.roundtrip(self.proxy.vision_capture)
        target = app.resolve("", "测试应用")
        self.assertEqual(target.expected_app_id, "org.example.app1")
        self.assertEqual(self.roundtrip(app.launch, target.launch_ref)["status"], "accepted")
        self.assertEqual(len(self.broadcasts()), 3)
        self.assertTrue(all(call[2] == "serial-1" for call in self.calls))
        self.assertEqual(sum("monkey" in call for call in self.calls), 1)

    def test_station_two_uses_only_station_two_serial_and_launch_catalog(self):
        old_text, old_app = self.adapters()
        self.roundtrip(self.proxy.select_machine_position, 2)
        text, app = self.adapters()
        self.assertEqual(text.profile.device_id, DEVICE)
        self.assertEqual(text.profile.adb_serial, "serial-2")
        self.roundtrip(self.proxy.vision_capture)
        result = self.roundtrip(old_text.append_text, self.scope(old_text), "你好 🌍")
        self.assertFalse(result.attempted)
        self.roundtrip(self.proxy.vision_capture)
        self.assertTrue(self.roundtrip(text.append_text, self.scope(text), "你好 🌍").accepted)
        self.roundtrip(self.proxy.vision_capture)
        self.roundtrip(app.launch, app.resolve("", "测试应用").launch_ref)
        self.assertTrue(all(call[2] == "serial-2" for call in self.calls))
        self.assertIn("org.example.app2", self.calls[-1])
        self.assertEqual(len(self.broadcasts()), 1)

    def test_disabled_unconfigured_station_does_not_borrow_first_profile(self):
        for station in (3, 10):
            self.roundtrip(self.proxy.select_machine_position, station)
            description = self.roundtrip(describe_remote_tools, self.proxy)
            self.assertIsNone(description["text_profile"])
            self.assertEqual(description["apps"], [])
            self.assertEqual(self.roundtrip(self.proxy.vision_capture).size, self.hardware.frame.size)
        self.assertEqual(self.calls, [])

    def test_timeout_is_unknown_not_retried(self):
        text, _ = self.adapters()
        scope = self.scope(text)
        self.roundtrip(self.proxy.vision_capture)
        self.timeout_broadcast = True
        result = self.roundtrip(text.append_text, scope, "你好 🌍")
        self.assertEqual(result.status, "unknown")
        with self.assertRaises(ConnectorProtocolError):
            text.append_text(scope, "你好 🌍")
        self.assertEqual(len(self.broadcasts()), 1)

    def test_missing_capture_and_unknown_app_never_dispatch(self):
        text, app = self.adapters()
        self.assertFalse(text.append_text(self.scope(text), "你好 🌍").attempted)
        with self.assertRaises(AdbPackageLauncherError) as caught:
            app.launch("org.injected.arbitrary")
        self.assertFalse(caught.exception.attempted)
        self.assertEqual(self.calls, [])

    def test_scope_digest_foreign_device_extra_shell_and_bad_receipt(self):
        text, _ = self.adapters()
        scope = self.scope(text)
        self.roundtrip(self.proxy.vision_capture)
        for changed in ({"device_id": "foreign"}, {"fragment_text_digest": text_digest("tampered")}):
            with self.assertRaises((ConnectorProtocolError, ValueError)):
                text.append_text(replace(scope, **changed), "你好 🌍")
        args = {**text.binding, "scope": scope.to_dict(), "text": "你好 🌍", "shell": "bad"}
        with self.assertRaises(ConnectorProtocolError):
            validate_command({**command("text_append", args), "authority": {"capture_id": self.proxy._remote_capture_id}})
        with patch.object(self.proxy, "_call", return_value={"status": "accepted"}):
            self.assertEqual(text.append_text(scope, "你好 🌍").status, "unknown")
        self.assertEqual(self.calls, [])

    def test_duplicate_command_result_retransmission_and_nonce_survive_restart(self):
        text, _ = self.adapters()
        scope = self.scope(text)
        self.roundtrip(self.proxy.vision_capture)
        args = {**text.binding, "scope": scope.to_dict(), "text": "你好 🌍"}
        cmd = {**command("text_append", args), "authority": {"capture_id": self.proxy._remote_capture_id}}
        first = self.worker.execute(cmd)
        self.assertTrue(first["ok"])
        self.worker.journal = ConnectorJournal(self.worker.journal.path)
        self.assertEqual(self.worker.execute(cmd), first)
        self.worker.journal.acknowledge(cmd["command_id"])
        self.roundtrip(self.proxy.vision_capture)
        # New process has no in-memory consumed nonces; the durable journal owns it.
        self.worker.tools.texts.transport_for_device(DEVICE)._consumed_nonces.clear()
        replay = {**cmd, "command_id": secrets.token_hex(16), "authority": {"capture_id": self.proxy._remote_capture_id}}
        self.assertFalse(self.worker.execute(replay)["ok"])
        self.assertEqual(len(self.broadcasts()), 1)

    def test_connector_restart_invalidates_old_tool_binding(self):
        text, _ = self.adapters()
        self.worker.tools = self.local_tools()
        self.roundtrip(self.proxy.vision_capture)
        result = self.roundtrip(text.append_text, self.scope(text), "你好 🌍")
        self.assertFalse(result.attempted)
        self.assertEqual(self.calls, [])

    def test_local_station_restore_uses_latest_success_and_matching_identity(self):
        self.roundtrip(self.proxy.select_machine_position, 2)
        journal = ConnectorJournal(self.worker.journal.path)
        self.assertEqual(journal.last_confirmed_machine_position(connector_id=CONNECTOR, device_id=DEVICE), 2)
        self.assertIsNone(journal.last_confirmed_machine_position(connector_id=CONNECTOR, device_id="other"))
        journal.begin(command(args={"machine_position": 3}))
        self.assertIsNone(journal.last_confirmed_machine_position(connector_id=CONNECTOR, device_id=DEVICE))
        self.assertEqual(self.hardware.calls, [("position", 2)])

    def test_text_preflight_failure_has_zero_effects_and_launch_timeout_is_unknown(self):
        text, app = self.adapters()
        self.adb.unlink()
        self.roundtrip(self.proxy.vision_capture)
        result = self.roundtrip(text.append_text, self.scope(text), "你好 🌍")
        self.assertFalse(result.attempted)
        self.assertEqual(result.reason_code, "adb_missing")
        self.assertEqual(self.calls, [])
        self.adb.touch()
        launcher = self.worker.tools.launchers[DEVICE]
        with patch.object(launcher, "_runner", side_effect=subprocess.TimeoutExpired("adb", 10)) as runner:
            self.roundtrip(self.proxy.vision_capture)
            with self.assertRaises(AdbPackageLauncherError) as caught:
                self.roundtrip(app.launch, "trusted-1")
            self.assertTrue(caught.exception.attempted)
            self.assertEqual(runner.call_count, 1)

    def test_lost_text_result_ack_retransmits_without_another_broadcast(self):
        import httpx
        text, _ = self.adapters()
        scope = self.scope(text)
        self.roundtrip(self.proxy.vision_capture)
        cid = self.store.enqueue(connector_id=CONNECTOR, device_id=DEVICE, operation="text_append",
            args={**text.binding, "scope": scope.to_dict(), "text": "你好 🌍"},
            authority={"capture_id": self.proxy._remote_capture_id})
        self.reject_next_result = True
        with self.assertRaises(httpx.ReadError):
            self.worker.run_once()
        self.assertTrue(self.store.wait(cid)["value"]["accepted"])
        self.worker.run_once()
        self.assertEqual(len(self.broadcasts()), 1)
        self.assertEqual(self.worker.journal.pending(), [])

    def test_runtime_uses_remote_factories_and_never_cloud_adb(self):
        runtime = Runtime.__new__(Runtime)
        runtime.controller_for_device = Mock(return_value=self.proxy)
        runtime.adb_keyboard_runtime = Mock()
        runtime.app_package_registry_path = self.path / "cloud-does-not-have-windows-config"
        text = self.roundtrip(runtime.text_transport_for_device, DEVICE)
        app = self.roundtrip(runtime.app_launcher_for_device, DEVICE)
        self.assertIsInstance(text, RemoteTextTransport)
        self.assertIsInstance(app, RemoteAppLauncher)
        runtime.adb_keyboard_runtime.transport_for_device.assert_not_called()
        self.roundtrip(self.proxy.select_machine_position, 3)
        self.assertIsNone(self.roundtrip(runtime.text_transport_for_device, DEVICE))
        self.assertIsNone(self.roundtrip(runtime.app_launcher_for_device, DEVICE))


if __name__ == "__main__":
    unittest.main()
