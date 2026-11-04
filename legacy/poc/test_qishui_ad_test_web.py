from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import web_app
from test_support.web_platform import _BaseApiEndToEndTests


class _FakeMonitor:
    def __init__(self):
        self.record = None

    def start(self, *, profile, goal):
        self.record = SimpleNamespace(
            test_id="test-1", profile=profile, goal=goal, phase="manual_observe",
            snapshot=lambda: {
                "test_id": "test-1", "feature_id": "qishui_ad_test",
                "status": "observing", "phase": "manual_observe", "session_id": "session-1",
                "device_id": profile.device_id, "app_alias": profile.app_alias,
                "duration_seconds": profile.duration_seconds, "physical_actions": 0,
                "confirmation_mode": "", "claim_action_allowed": False,
                "run_dir": str(Path("evidence")),
            },
        )
        return self.record

    def get(self, test_id):
        return self.record if test_id == "test-1" else None

    def observe(self, test_id, *, device_id):
        self.record.phase = "reward_or_end_page"
        self.record.snapshot = lambda: {
            "test_id": "test-1", "feature_id": "qishui_ad_test",
            "status": "awaiting_confirmation", "phase": "reward_or_end_page",
            "session_id": "session-1", "device_id": device_id,
            "confirmation_mode": "end", "physical_actions": 0,
            "claim_action_allowed": False,
        }
        return self.record

    def confirm(self, test_id, *, device_id, confirmed, mode, confirmation):
        self.record.phase = "completed"
        self.record.snapshot = lambda: {
            "test_id": "test-1", "feature_id": "qishui_ad_test",
            "status": "completed", "phase": "completed",
            "device_id": device_id, "physical_actions": 0,
            "claim_action_allowed": False,
        }
        return self.record

    def cancel(self, test_id, *, device_id):
        self.record.phase = "stopped"
        return self.record


class QishuiAdTestWebTests(_BaseApiEndToEndTests):
    def test_preset_is_manual_and_claim_is_disabled(self):
        response = self.client.get("/api/features/qishui-ad-test", headers=self.headers)
        self.assertEqual(200, response.status_code, response.text)
        payload = response.json()
        self.assertEqual("qishui_ad_test", payload["feature_id"])
        self.assertFalse(payload["automatic_loop_enabled"])
        self.assertEqual(1, payload["max_ads_per_round"])
        self.assertIn("不会自动领取金币", payload["notice"])

    def test_start_observe_and_end_confirmation_routes(self):
        fake = _FakeMonitor()
        with patch.object(web_app, "_QISHUI_AD_TEST_MONITOR", fake):
            started = self.client.post(
                "/api/features/qishui-ad-test/start",
                headers=self.headers,
                json={"device_id": "device-local-01", "duration_seconds": 30},
            )
            self.assertEqual(200, started.status_code, started.text)
            self.assertEqual("test-1", started.json()["test"]["test_id"])
            observed = self.client.post(
                "/api/features/qishui-ad-test/test-1/observe",
                headers=self.headers,
                json={"device_id": "device-local-01"},
            )
            self.assertEqual(200, observed.status_code, observed.text)
            self.assertEqual("awaiting_confirmation", observed.json()["test"]["status"])
            confirmed = self.client.post(
                "/api/features/qishui-ad-test/test-1/confirm",
                headers=self.headers,
                json={"device_id": "device-local-01", "confirmed": True, "mode": "end"},
            )
            self.assertEqual(200, confirmed.status_code, confirmed.text)
            self.assertEqual("completed", confirmed.json()["test"]["status"])
            self.assertFalse(confirmed.json()["test"]["claim_action_allowed"])

    def test_routes_require_local_token(self):
        response = self.client.get("/api/features/qishui-ad-test")
        self.assertEqual(403, response.status_code)


if __name__ == "__main__":
    unittest.main()
