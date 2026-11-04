from __future__ import annotations

import sys
from pathlib import Path
import unittest
from unittest.mock import Mock

from agent.domain.text_transport import TEXT_TRANSPORT_PROTOCOL, TextTransportProfile
from agent.infrastructure.adb_keyboard_transport import AdbKeyboardTextTransport
from agent.infrastructure import seller_window_adapter as seller


class OfflineEnvironmentTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform != "win32", "The fallback is only used off Windows")
    def test_seller_window_placeholder_never_dispatches_input(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Win32 input"):
            seller.user32.mouse_event(0, 0, 0, 0, 0)

    @unittest.skipUnless(sys.platform != "win32", "The path normalization regression is Linux-specific")
    def test_windows_adb_path_is_validated_before_filesystem_readiness(self) -> None:
        profile = TextTransportProfile(
            protocol_version=TEXT_TRANSPORT_PROTOCOL,
            profile_id="offline-profile",
            device_id="offline-device",
            adb_serial="offline-serial",
            enabled=True,
            capabilities=("append_text",),
            command_timeout_seconds=5.0,
        )
        transport = AdbKeyboardTextTransport(
            profile,
            Path(r"C:\platform-tools\adb.exe"),
        )
        self.assertEqual("adb_missing", transport.status()["reason_code"])

    @unittest.skipUnless(sys.platform != "win32", "The fail-closed guard is only exercised off Windows")
    def test_all_hardware_capture_entrypoints_fail_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "只支持 Windows"):
            seller.capture_client_passive(1)
        with self.assertRaisesRegex(RuntimeError, "只支持 Windows"):
            seller.click_client_point(1, 1, 1, 0, 0.1)

    def test_device_registry_stop_broadcast_covers_every_controller(self) -> None:
        from agent.infrastructure.device_controller_registry import DeviceControllerRegistry

        first, second = Mock(), Mock()
        registry = object.__new__(DeviceControllerRegistry)
        registry._controllers = {"device-a": first, "device-b": second}
        self.assertEqual(("device-a", "device-b"), registry.request_stop_all())
        first.request_stop.assert_called_once_with()
        second.request_stop.assert_called_once_with()

    def test_action_catalog_is_the_canonical_action_source(self) -> None:
        from agent.domain.action_catalog import (
            CANONICAL_ACTION_KINDS,
            PHYSICAL_ACTION_KINDS,
            PROMOTABLE_ACTION_KINDS,
        )
        from agent.domain.action_catalog import CANONICAL_ACTION_KINDS as exported

        self.assertEqual(CANONICAL_ACTION_KINDS, exported)
        self.assertTrue(PHYSICAL_ACTION_KINDS >= CANONICAL_ACTION_KINDS - {"wait_for_change", "launch_app"})
        self.assertEqual(PROMOTABLE_ACTION_KINDS, exported & PROMOTABLE_ACTION_KINDS)


if __name__ == "__main__":
    unittest.main()
