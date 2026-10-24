from __future__ import annotations

import io
import unittest
from unittest.mock import patch

from PIL import Image

from agent.domain.text_transport import TextTransportReplayError, text_digest, EMPTY_TEXT_DIGEST
from agent.infrastructure.observation_images import local_frame_fingerprint
from agent.infrastructure.orientation_safety import _mint_single_step_scene_credential
from agent.infrastructure.robot_controller import MockRobotController
from agent.infrastructure.simulated_text_transport import SimulatedTextTransport
from agent.infrastructure.simulated_phone import SimPhone


def _arm(controller: MockRobotController, action: str) -> None:
    frame = controller.vision_capture()
    fingerprint = local_frame_fingerprint(frame)
    credential = _mint_single_step_scene_credential(device_id=controller.device_id,
        scene_fingerprint=fingerprint, frame=frame)
    controller.arm_physical_execution(credential, action=action, scene_fingerprint=fingerprint)


def _scope(transport: SimulatedTextTransport, *, action_id: str, fragment: str, expected: str, prior: str = ""):
    return transport.mint_action_scope(session_id="s1", task_id="t1", revision=1, action_id=action_id,
        input_field_id="field-1", observation_fingerprint="obs-1", prior_text_digest=text_digest(prior),
        fragment_text_digest=text_digest(fragment) if fragment else EMPTY_TEXT_DIGEST,
        expected_text_digest=text_digest(expected) if expected else EMPTY_TEXT_DIGEST)


class SimulatedPhoneRenderingTests(unittest.TestCase):
    def test_render_is_deterministic_and_reflects_state(self) -> None:
        phone = SimPhone()
        first, second = phone.render(), phone.render()
        self.assertEqual((540, 960), first.size)
        self.assertEqual(first.tobytes(), second.tobytes())
        phone.open_app("设置")
        self.assertNotEqual(first.tobytes(), phone.render().tobytes())

    def test_grid_mapping_covers_full_frame(self) -> None:
        self.assertEqual((0, 0), SimPhone.grid_to_pixel(0, 0))
        self.assertEqual((539, 959), SimPhone.grid_to_pixel(1000, 1000))


class MockControllerSimulationTests(unittest.TestCase):
    def test_capture_shows_simulated_screen_not_static_placeholder(self) -> None:
        controller = MockRobotController(device_id="device-local-01")
        frame = controller.vision_capture()
        preview = Image.open(io.BytesIO(controller.capture_preview())).convert("RGB")
        self.assertEqual(frame.size, preview.size)
        self.assertEqual(controller.phone.render().tobytes(), frame.tobytes())

    def test_credential_gated_tap_changes_the_next_observation(self) -> None:
        controller = MockRobotController(device_id="device-local-01")
        before = controller.vision_capture()
        _arm(controller, "tap_semantic")
        # Settings icon centre on the simulated home screen: pixel (88, 688).
        controller.vision_tap_relative(163, 717)
        self.assertEqual("app:设置", controller.phone.state.screen)
        self.assertEqual([{"action": "tap", "coordinate": [163, 717]}], controller.executions)
        self.assertNotEqual(before.tobytes(), controller.vision_capture().tobytes())

    def test_tap_without_execution_credential_is_still_rejected(self) -> None:
        controller = MockRobotController(device_id="device-local-01")
        with self.assertRaises(Exception):
            controller.vision_tap_relative(163, 717)
        self.assertEqual("home", controller.phone.state.screen)
        self.assertEqual([], controller.executions)

    def test_system_navigation_and_swipe_drive_the_phone(self) -> None:
        controller = MockRobotController(device_id="device-local-01")
        controller.phone.open_app("设置")
        _arm(controller, "back")
        controller.vision_android_back()
        self.assertEqual("home", controller.phone.state.screen)
        _arm(controller, "open_recent_apps")
        controller.vision_android_recent_apps()
        self.assertEqual("recents", controller.phone.state.screen)
        controller.phone.open_app("设置")
        _arm(controller, "swipe")
        controller.vision_swipe_relative(500, 750, 500, 300, "up")
        self.assertGreater(controller.phone.state.settings_scroll, 0)


class SimulatedToolTests(unittest.TestCase):
    def test_text_transport_keeps_scope_nonce_and_digest_contract(self) -> None:
        phone = SimPhone()
        phone.new_note()
        phone.focus("draft")
        transport = SimulatedTextTransport(phone, "device-local-01")
        self.assertTrue(transport.status()["ready"])
        scope = _scope(transport, action_id="a1", fragment="买牛奶🥛", expected="买牛奶🥛")
        result = transport.append_text(scope, "买牛奶🥛")
        self.assertTrue(result.accepted)
        self.assertEqual("买牛奶🥛", phone.state.draft)
        with self.assertRaises(TextTransportReplayError):
            transport.append_text(scope, "买牛奶🥛")
        self.assertEqual("买牛奶🥛", phone.state.draft)
        cleared = transport.clear_text(_scope(transport, action_id="a2", fragment="", expected="", prior="买牛奶🥛"))
        self.assertTrue(cleared.accepted)
        self.assertEqual("", phone.state.draft)
        self.assertEqual(["ADB_INPUT_B64", "ADB_CLEAR_TEXT"], transport.broadcasts)

    def test_only_the_two_fixed_broadcasts_are_interpreted(self) -> None:
        transport = SimulatedTextTransport(SimPhone(), "device-local-01")
        result = transport._broadcast_runner(["adb", "-s", "x", "shell", "input", "tap", "1", "1"])
        self.assertEqual(1, result.returncode)
        self.assertEqual([], transport.broadcasts)


class RuntimeSimulatedToolGatingTests(unittest.TestCase):
    def test_simulated_text_is_opt_in_and_bound_per_device(self) -> None:
        import web_app

        runtime = web_app.runtime
        controller = MockRobotController(device_id="device-local-01")
        with patch.object(runtime, "controller_for_device", return_value=controller), \
             patch.object(runtime, "_simulated_text", {}), \
             patch.object(runtime.adb_keyboard_runtime, "transport_for_device", return_value=None):
            with patch.object(runtime, "simulated_phone_tools", False):
                self.assertIsNone(runtime.text_transport_for_device("device-local-01"))
            with patch.object(runtime, "simulated_phone_tools", True):
                text = runtime.text_transport_for_device("device-local-01")
                other = runtime.text_transport_for_device("device-local-02")
                self.assertIsInstance(text, SimulatedTextTransport)
                self.assertIs(text.phone, controller.phone)
                self.assertEqual("device-local-02", other.profile.device_id)
                self.assertIs(text, runtime.text_transport_for_device("device-local-01"))


if __name__ == "__main__":
    unittest.main()
