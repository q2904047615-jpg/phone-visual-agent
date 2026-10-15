from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from agent.domain.ga_readiness import GA_READINESS_PROTOCOL, evaluate_device_registry


def _write_registry(root: Path, *, enabled_count: int) -> Path:
    devices = []
    for index in range(1, enabled_count + 1):
        calibration = root / f"calibration-{index}.json"
        calibration.write_text("{}", encoding="utf-8")
        devices.append({
            "device_id": f"virtual-{index}",
            "enabled": True,
            "window_title": "seller" if index <= 2 else f"seller-{index}",
            "machine_position": index,
            "calibration_path": calibration.name,
            "verified_actions": [],
        })
    path = root / "devices.json"
    path.write_text(json.dumps({"version": 1, "default_device_id": "virtual-1", "devices": devices}), encoding="utf-8")
    return path


class GAReadinessTests(unittest.TestCase):
    def test_current_single_device_is_physical_blocked_without_changing_registry(self) -> None:
        path = Path(__file__).with_name("device_registry.json")
        before = path.read_bytes()
        report = evaluate_device_registry(path)
        self.assertEqual(GA_READINESS_PROTOCOL, report.protocol_version)
        self.assertEqual(("device-local-01",), report.enabled_devices)
        self.assertEqual("blocked", report.offline_virtual_status)
        self.assertEqual("blocked_registry_not_ready", report.physical_multi_device_status)
        self.assertEqual("pass", report.offline_ga_status)
        self.assertEqual("pass", next(item.status for item in report.ga_scope if item.check_id == "users_teams_permissions"))
        self.assertEqual("pass", next(item.status for item in report.ga_scope if item.check_id == "api_sdk"))
        self.assertEqual(before, path.read_bytes())

    def test_virtual_two_device_registry_passes_offline_but_needs_real_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_registry(Path(temporary), enabled_count=2)
            report = evaluate_device_registry(path)
        self.assertEqual("pass", report.offline_virtual_status)
        self.assertEqual("blocked_no_real_evidence", report.physical_multi_device_status)
        self.assertEqual("pass", next(item.status for item in report.checks if item.check_id == "offline_virtual_multi_device"))
        self.assertEqual("pass", report.offline_ga_status)
        self.assertEqual("pass", next(item.status for item in report.ga_scope if item.check_id == "users_teams_permissions"))

    def test_missing_calibration_blocks_virtual_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = _write_registry(root, enabled_count=2)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["devices"][1]["calibration_path"] = "missing.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            report = evaluate_device_registry(path)
        self.assertEqual(("virtual-2",), report.missing_calibrations)
        self.assertEqual("blocked", report.offline_virtual_status)
        self.assertEqual("blocked_registry_not_ready", report.physical_multi_device_status)


if __name__ == "__main__":
    unittest.main()
