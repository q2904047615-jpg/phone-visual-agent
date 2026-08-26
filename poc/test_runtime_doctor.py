from __future__ import annotations

import json
import unittest

from PIL import Image

from agent.domain.action_capabilities import KNOWN_ACTION_CAPABILITIES
from runtime_doctor import run_runtime_doctor


class FakeDoctorController:
    def __init__(self, frames=None, *, online=True, camera=True, busy=False):
        self.frames = list(frames or stable_frames())
        self.online = online
        self.camera = camera
        self.busy = busy
        self.capture_count = 0
        self.action_calls = []

    def device_status(self):
        return {
            "controller_online": self.online,
            "camera_online": self.camera,
            "busy": self.busy,
            "stop_requested": False,
            "window_title": "FAKE controller",
            "client_size": [540, 1038],
        }

    def hardware_capability_profile(self):
        enabled = {
            "tap_semantic",
            "swipe",
            "back",
            "home",
            "wait_for_change",
        }
        return {
            "protocol_version": "fixture",
            "device_id": "device-a",
            "actions": {
                action: {"enabled": action in enabled}
                for action in KNOWN_ACTION_CAPABILITIES
            },
        }

    def vision_capture(self):
        frame = self.frames[self.capture_count]
        self.capture_count += 1
        return frame.copy()

    def vision_tap_relative(self, *_args):
        self.action_calls.append("tap")
        raise AssertionError("doctor must never execute an action")


class FakeProvider:
    def __init__(self, *, configured=True, model=""):
        self.configured = configured
        self.model = model

    def status(self):
        return {
            "configured": self.configured,
            "model": self.model,
            "api_key": "secret-that-must-not-leak",
            "base_url": "https://secret.example/v1",
        }


class ExplodingProvider:
    def status(self):
        raise RuntimeError("secret-provider-error")


def stable_frames():
    return [Image.new("RGB", (540, 960), "#203040") for _ in range(4)]


def run(controller, *, active_session=None, deepseek=None, qwen=None):
    return run_runtime_doctor(
        device_id="device-a",
        controller=controller,
        deepseek_provider=deepseek or FakeProvider(model="deepseek-chat"),
        qwen_provider=qwen or FakeProvider(model="qwen3.7-plus"),
        active_session=active_session,
        protocols={"action": "2026-08-20-canonical-action-v1"},
        sleep=lambda _seconds: None,
    )


class RuntimeDoctorTests(unittest.TestCase):
    def test_ready_doctor_captures_stable_frames_and_never_acts(self):
        controller = FakeDoctorController()
        result = run(controller)

        self.assertTrue(result["ready"])
        self.assertEqual(0, result["physical_actions"])
        self.assertEqual(4, controller.capture_count)
        self.assertEqual([], controller.action_calls)
        self.assertTrue(result["camera"]["stability"]["stable"])
        self.assertEqual(4, len(result["camera"]["frame_fingerprints"]))
        self.assertIn("tap_semantic", result["capabilities"]["supported_actions"])

    def test_active_session_is_reported_without_camera_capture(self):
        controller = FakeDoctorController()
        result = run(controller, active_session="session-active")

        self.assertFalse(result["ready"])
        self.assertEqual(0, controller.capture_count)
        self.assertFalse(result["camera"]["captured"])
        self.assertIn("设备已有活动会话：session-active", result["blockers"])

    def test_unstable_frames_are_a_runtime_blocker(self):
        frames = stable_frames()
        frames[3] = Image.new("RGB", (540, 960), "white")
        result = run(FakeDoctorController(frames))

        self.assertFalse(result["ready"])
        self.assertTrue(any("不稳定" in item for item in result["blockers"]))

    def test_provider_errors_and_secrets_are_redacted(self):
        result = run(
            FakeDoctorController(),
            deepseek=ExplodingProvider(),
        )
        serialized = json.dumps(result, ensure_ascii=False)

        self.assertFalse(result["ready"])
        self.assertNotIn("secret-provider-error", serialized)
        self.assertNotIn("secret-that-must-not-leak", serialized)

    def test_wrong_qwen_model_is_reported_without_changing_it(self):
        result = run(
            FakeDoctorController(),
            qwen=FakeProvider(model="another-model"),
        )

        self.assertFalse(result["ready"])
        self.assertTrue(any("不是 qwen3.7-plus" in item for item in result["blockers"]))

if __name__ == "__main__":
    unittest.main()
