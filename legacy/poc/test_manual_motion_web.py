from __future__ import annotations

import time
from pathlib import Path
import unittest

import web_app
from test_support.web_platform import _BaseApiEndToEndTests


class ManualMotionWebTests(_BaseApiEndToEndTests):
    def test_main_console_exposes_both_configurable_entries(self):
        source = (Path(__file__).resolve().parent / "static" / "app.js").read_text(encoding="utf-8")
        page = (Path(__file__).resolve().parent / "static" / "index.html").read_text(encoding="utf-8")
        self.assertIn('openManualMotionDialog("swipe")', source)
        self.assertIn('function startFixedClickTest()', source)
        self.assertIn('/api/manual-motion/start', source)
        self.assertIn('id="manualPointAX"', page)
        self.assertIn('id="manualPointBX"', page)
        self.assertIn('id="manualDurationHours"', page)
        self.assertIn('不会移动本机鼠标或激活窗口', page)

    def test_start_status_stop_are_additive_and_do_not_call_qwen(self):
        response = self.client.post(
            "/api/manual-motion/start",
            headers=self.headers,
            json={
                "kind": "swipe",
                "device_id": "device-local-01",
                "point_a": [500, 800],
                "point_b": [500, 200],
                "interval_seconds": 10,
                "duration_seconds": 100,
            },
        )
        self.assertEqual(200, response.status_code, response.text)
        motion = response.json()["motion"]
        self.assertIn(motion["status"], {"starting", "running"})
        self.assertFalse(response.json()["qwen_called"])
        listed = self.client.get("/api/device").json()["manual_motion_tests"]
        self.assertTrue(any(item["motion_id"] == motion["motion_id"] for item in listed))
        stopped = self.client.post(
            f"/api/manual-motion/{motion['motion_id']}/stop",
            headers=self.headers,
            json={},
        )
        self.assertEqual(200, stopped.status_code, stopped.text)
        deadline = time.time() + 2
        while time.time() < deadline:
            current = self.client.get(f"/api/manual-motion/{motion['motion_id']}", headers=self.headers).json()["motion"]
            if current["status"] == "stopped":
                break
            time.sleep(.01)
        self.assertEqual("stopped", current["status"])

    def test_invalid_coordinates_are_rejected_without_starting(self):
        response = self.client.post(
            "/api/manual-motion/start",
            headers=self.headers,
            json={
                "kind": "swipe",
                "device_id": "device-local-01",
                "point_a": [540, 800],
                "point_b": [500, 200],
                "interval_seconds": 5,
                "duration_seconds": 3600,
            },
        )
        self.assertEqual(422, response.status_code, response.text)


if __name__ == "__main__":
    unittest.main()
