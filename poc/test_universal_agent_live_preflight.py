from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from universal_agent_orchestrator import (
    DeviceTaskRegistry,
    PhaseOneNavigationPolicy,
)
from run_universal_agent_live_preflight import run_read_only_preflight


class FakeReadOnlyController:
    def __init__(
        self,
        frames: list[Image.Image],
        *,
        controller_online: bool = True,
        camera_online: bool = True,
    ) -> None:
        self.frames = list(frames)
        self.capture_count = 0
        self.action_calls: list[str] = []
        self.controller_online = controller_online
        self.camera_online = camera_online

    def device_status(self) -> dict:
        return {
            "controller_online": self.controller_online,
            "camera_online": self.camera_online,
            "window_title": "FAKE read-only controller",
            "client_size": [540, 1038],
            "busy": False,
            "stop_requested": False,
            "error": None,
            "camera_error": None,
        }

    def vision_capture(self) -> Image.Image:
        frame = self.frames[self.capture_count]
        self.capture_count += 1
        return frame.copy()

    def vision_tap_relative(self, *_args) -> None:
        self.action_calls.append("tap")
        raise AssertionError("read-only preflight must not tap")

    def vision_swipe_up(self) -> None:
        self.action_calls.append("swipe")
        raise AssertionError("read-only preflight must not swipe")

    def vision_android_back(self) -> None:
        self.action_calls.append("back")
        raise AssertionError("read-only preflight must not go back")


class FakeProvider:
    def __init__(self, configured: bool, secret: str) -> None:
        self.configured = configured
        self.secret = secret

    def status(self) -> dict:
        return {
            "configured": self.configured,
            "api_key": self.secret,
            "base_url": f"https://example.invalid/{self.secret}",
        }


class ExplodingProvider:
    configured = False

    def status(self) -> dict:
        raise RuntimeError("provider failed with secret-token-that-must-not-leak")


def stable_frames() -> list[Image.Image]:
    return [Image.new("RGB", (540, 960), "#203040") for _ in range(4)]


class UniversalAgentLivePreflightTests(unittest.TestCase):
    def test_ready_preflight_captures_four_frames_without_any_action(self) -> None:
        controller = FakeReadOnlyController(stable_frames())
        deepseek = FakeProvider(True, "deepseek-secret-must-not-leak")
        qwen = FakeProvider(True, "qwen-secret-must-not-leak")

        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_read_only_preflight(
                device_id="device-a",
                controller=controller,
                deepseek_provider=deepseek,
                qwen_provider=qwen,
                registry=DeviceTaskRegistry(),
                policy=PhaseOneNavigationPolicy(),
                evidence_dir=Path(temp_dir),
                sleep=lambda _seconds: None,
            )

            self.assertTrue(result["ready"])
            self.assertEqual(result["physical_actions"], 0)
            self.assertEqual(result["camera"]["frame_count"], 4)
            self.assertTrue(result["camera"]["stability"]["stable"])
            self.assertEqual(controller.capture_count, 4)
            self.assertEqual(controller.action_calls, [])
            self.assertEqual(len(result["camera"]["evidence_paths"]), 4)
            for path in result["camera"]["evidence_paths"]:
                self.assertTrue(Path(path).is_file())

        serialized = json.dumps(result, ensure_ascii=False)
        self.assertNotIn("deepseek-secret-must-not-leak", serialized)
        self.assertNotIn("qwen-secret-must-not-leak", serialized)
        self.assertEqual(
            result["policy"]["version"], PhaseOneNavigationPolicy.VERSION
        )

    def test_blockers_are_reported_and_still_never_trigger_actions(self) -> None:
        frames = stable_frames()
        frames[2] = Image.new("RGB", (540, 960), "white")
        controller = FakeReadOnlyController(frames)
        registry = DeviceTaskRegistry()
        registry.reserve("device-b", "another-session")

        result = run_read_only_preflight(
            device_id="device-b",
            controller=controller,
            deepseek_provider=FakeProvider(False, "missing-deepseek"),
            qwen_provider=FakeProvider(False, "missing-qwen"),
            registry=registry,
            policy=PhaseOneNavigationPolicy(),
            sleep=lambda _seconds: None,
        )

        self.assertFalse(result["ready"])
        self.assertEqual(result["physical_actions"], 0)
        self.assertEqual(controller.action_calls, [])
        self.assertEqual(controller.capture_count, 4)
        self.assertFalse(result["device"]["exclusive_available"])
        self.assertFalse(result["camera"]["stability"]["stable"])
        self.assertGreaterEqual(len(result["blockers"]), 4)

    def test_offline_camera_is_not_captured(self) -> None:
        controller = FakeReadOnlyController(
            stable_frames(), controller_online=False, camera_online=False
        )

        result = run_read_only_preflight(
            device_id="device-c",
            controller=controller,
            deepseek_provider=FakeProvider(True, "hidden"),
            qwen_provider=FakeProvider(True, "hidden"),
            registry=DeviceTaskRegistry(),
            policy=PhaseOneNavigationPolicy(),
            sleep=lambda _seconds: None,
        )

        self.assertFalse(result["ready"])
        self.assertEqual(controller.capture_count, 0)
        self.assertEqual(controller.action_calls, [])
        self.assertEqual(result["physical_actions"], 0)

    def test_provider_status_exception_cannot_leak_secret_text(self) -> None:
        result = run_read_only_preflight(
            device_id="device-d",
            controller=FakeReadOnlyController(stable_frames()),
            deepseek_provider=ExplodingProvider(),
            qwen_provider=FakeProvider(True, "hidden"),
            registry=DeviceTaskRegistry(),
            policy=PhaseOneNavigationPolicy(),
            sleep=lambda _seconds: None,
        )

        serialized = json.dumps(result, ensure_ascii=False)
        self.assertNotIn("secret-token-that-must-not-leak", serialized)
        self.assertIn("DeepSeek provider 状态读取失败", result["blockers"])


if __name__ == "__main__":
    unittest.main()
