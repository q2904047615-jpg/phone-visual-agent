from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent.infrastructure.adb_pairing import AdbPairingError, AdbPairingService


class AdbPairingServiceTests(unittest.TestCase):
    def test_pair_connect_and_verify_use_fixed_sequence(self) -> None:
        calls = []

        def runner(argv, **kwargs):
            calls.append(argv)
            if argv[1] == "pair":
                return SimpleNamespace(returncode=0, stdout="Successfully paired to phone", stderr="")
            if argv[1] == "connect":
                return SimpleNamespace(returncode=0, stdout="connected to phone", stderr="")
            return SimpleNamespace(returncode=0, stdout="device\n", stderr="")

        result = AdbPairingService(Path("adb.exe"), "192.168.0.162:41165", runner=runner).pair_and_connect(
            pairing_host="192.168.0.162", pairing_port=42897, pairing_code="650406"
        )
        self.assertEqual({"paired": True, "connected": True, "state": "device", "adb_serial": "192.168.0.162:41165"}, result)
        self.assertEqual([
            ["adb.exe", "pair", "192.168.0.162:42897", "650406"],
            ["adb.exe", "connect", "192.168.0.162:41165"],
            ["adb.exe", "-s", "192.168.0.162:41165", "get-state"],
        ], calls)

    def test_invalid_code_is_rejected_before_transport(self) -> None:
        calls = []
        service = AdbPairingService(Path("adb.exe"), "192.168.0.162:41165", runner=lambda *args, **kwargs: calls.append(args))
        with self.assertRaisesRegex(AdbPairingError, "6 位数字"):
            service.pair_and_connect(pairing_host="192.168.0.162", pairing_port=42897, pairing_code="123")
        self.assertEqual([], calls)

    def test_pair_failure_does_not_connect(self) -> None:
        calls = []

        def runner(argv, **kwargs):
            calls.append(argv)
            return SimpleNamespace(returncode=1, stdout="", stderr="failed")

        with self.assertRaisesRegex(AdbPairingError, "pair失败"):
            AdbPairingService(Path("adb.exe"), "192.168.0.162:41165", runner=runner).pair_and_connect(
                pairing_host="192.168.0.162", pairing_port=42897, pairing_code="650406"
            )
        self.assertEqual(1, len(calls))


if __name__ == "__main__":
    unittest.main()
