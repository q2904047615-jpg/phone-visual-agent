from __future__ import annotations

import ctypes
from contextlib import nullcontext
import json
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

import robot_gui_poc
import web_app
from agent.domain import EvidenceStoreError
from agent.infrastructure import (
    CameraPreviewUnavailable,
    DeviceCameraCoordinator,
    DeviceControllerRegistry,
    DeviceControllerRegistryError,
    DeviceRuntimeResourceError,
    DeviceRuntimeResourceRegistry,
    DeviceTaskRegistry,
    FileSystemAgentEvidenceStore,
    InterProcessLease,
)
from capability_acceptance import PROMOTABLE_ACTIONS
from agent.domain.canonical_action_protocol import GenericStepProposal
from agent.infrastructure.generic_scene_observer import (
    SINGLE_STEP_SCENE_OBSERVER_VERSION,
)
from robot_core import (
    DEFAULT_CONTROLLER_CONFIG,
    MockRobotController as _MockRobotController,
    RobotController as _RobotController,
    controller_client_has_camera,
    oriented_navigation_ratio,
    qwerty_keyboard_config_from_anchors,
    qwerty_key_point,
)
from orientation_safety import _mint_audited_credential
from agent.domain.ui_scene import UIScene


class _TestDirectionCredentialMixin:
    """Keep no-hardware tests behind the same one-shot gate."""

    def _consume_physical_execution(self, action, frame):
        credential = _mint_audited_credential(
            device_id=self.device_id,
            scene_fingerprint="test-scene",
            frame=frame,
            phone_content_rotation="upright",
            confidence=0.99,
            evidence=("合成手机界面轴线",),
        )
        self._physical_execution_gate.arm(
            credential,
            action=action,
            scene_fingerprint="test-scene",
        )
        return super()._consume_physical_execution(action, frame)


class RobotController(_TestDirectionCredentialMixin, _RobotController):
    def __init__(self, *args, device_id="test-device", **kwargs):
        super().__init__(*args, device_id=device_id, **kwargs)


class MockRobotController(_TestDirectionCredentialMixin, _MockRobotController):
    def __init__(self, *args, device_id="test-device", **kwargs):
        super().__init__(*args, device_id=device_id, **kwargs)


web_app.RobotController = RobotController
web_app.MockRobotController = MockRobotController
from intent_provider import DeepSeekIntentProvider, IntentProviderError
from agent.domain.vision_model import VisionAgentError


TEST_QWERTY_LAYOUT = {
    "type": "qwerty",
    "anchors": {
        "q": [115, 704],
        "p": [875, 704],
        "a": [157, 773],
        "l": [832, 773],
        "z": [241, 844],
        "m": [747, 844],
        "backspace": [862, 844],
    },
}


class PhysicalNavigationSafetyTests(unittest.TestCase):
    @staticmethod
    def _click_barrier_receipt(click_count=1):
        return {
            "version": "2026-08-19-seller-gui-click-barrier-v1",
            "channel": "left_button_atomic_click",
            "seller_event_barrier_confirmed": True,
            "round_trip_position_confirmed": True,
            "requested_mouse_hold_seconds": 0.35,
            "barrier_offset_pixels": 3,
            "changed_pixels": 240,
            "return_changed_pixels": 235,
            "barrier_elapsed_ms": 35.0,
            "mechanical_contact_ack": False,
            "click_count": click_count,
        }

    def test_live_preview_uses_passive_capture_without_active_capture_path(self):
        controller = RobotController(title="test")
        frame = Image.new("RGB", (540, 1038), "white")

        with (
            patch("robot_core.seller_gui.find_window", return_value=(123, "test")),
            patch(
                "robot_core.seller_gui.capture_client_passive",
                return_value=frame,
            ) as passive,
            patch("robot_core.seller_gui.capture_client") as active,
        ):
            payload = controller.capture_preview()

        self.assertTrue(payload.startswith(b"\xff\xd8"))
        passive.assert_called_once_with(123)
        active.assert_not_called()

    def test_passive_capture_never_invokes_window_activation(self):
        frame = Image.new("RGB", (540, 1038), "white")

        with (
            patch("robot_gui_poc._window_is_minimized", return_value=False),
            patch("robot_gui_poc._validate_camera_region_unoccluded") as validate,
            patch(
                "robot_gui_poc.client_geometry",
                return_value=(10, 20, 540, 1038),
            ),
            patch("robot_gui_poc.ImageGrab.grab", return_value=frame) as grab,
            patch("robot_gui_poc.ensure_camera_region_unoccluded") as activate,
        ):
            result = robot_gui_poc.capture_client_passive(123)

        self.assertEqual((540, 1038), result.size)
        validate.assert_called_once_with(123)
        grab.assert_called_once_with(
            bbox=(10, 20, 550, 1058),
            all_screens=True,
        )
        activate.assert_not_called()

    def test_passive_capture_keeps_minimized_window_minimized(self):
        with (
            patch("robot_gui_poc._window_is_minimized", return_value=True),
            patch("robot_gui_poc.ImageGrab.grab") as grab,
            patch("robot_gui_poc.ensure_camera_region_unoccluded") as activate,
        ):
            with self.assertRaisesRegex(RuntimeError, "已最小化"):
                robot_gui_poc.capture_client_passive(123)

        grab.assert_not_called()
        activate.assert_not_called()

    def test_agent_capture_holds_cursor_lease_only_during_the_capture(self):
        controller = RobotController(title="test")
        frame = Image.new("RGB", (540, 1038), "white")
        events = []

        class CursorLease:
            def __enter__(self):
                events.append("cursor_lease_enter")

            def __exit__(self, *_args):
                events.append("cursor_lease_exit")

        def capture(_hwnd):
            events.append("capture")
            return frame

        with (
            patch("robot_core.seller_gui.find_window", return_value=(123, "test")),
            patch(
                "robot_core.seller_gui.temporarily_park_cursor_outside_camera",
                return_value=CursorLease(),
            ) as cursor_lease,
            patch.object(controller, "_capture_phone", side_effect=capture),
        ):
            result = controller.vision_capture()

        self.assertIs(frame, result)
        cursor_lease.assert_called_once_with(123)
        self.assertEqual(
            ["cursor_lease_enter", "capture", "cursor_lease_exit"],
            events,
        )

    def test_controller_client_rejects_small_landscape_error_dialog(self):
        self.assertFalse(controller_client_has_camera(379, 169))
        self.assertFalse(controller_client_has_camera(540, 400))
        self.assertTrue(controller_client_has_camera(540, 1038))
        self.assertTrue(controller_client_has_camera(1440, 810))
        self.assertTrue(controller_client_has_camera(1440, 885))

    def test_navigation_ratio_rotates_counter_clockwise_landscape_feed(self):
        landscape = oriented_navigation_ratio(0.685, 0.976, landscape=True)
        self.assertAlmostEqual(landscape[0], 0.976)
        self.assertAlmostEqual(landscape[1], 0.315)
        self.assertEqual(
            oriented_navigation_ratio(0.685, 0.976, landscape=False),
            (0.685, 0.976),
        )

    def test_navigation_tap_forces_single_click_and_returns_exact_pixel(self):
        controller = RobotController(title="test")
        frame = Image.new("RGB", (540, 960), "white")

        with (
            patch("robot_core.seller_gui.find_window", return_value=(123, "test")),
            patch.object(controller, "_capture_phone", return_value=frame),
            patch.object(controller, "_checkpoint"),
            patch("robot_core.seller_gui.configure_single_click_count") as configure,
            patch(
                "robot_core.seller_gui.click_client_point",
                return_value=self._click_barrier_receipt(),
            ) as click,
            patch("robot_core.seller_gui.clear_seller_camera_overlay"),
            patch(
                "robot_core.load_controller_config",
                return_value={"tap_hold": 0.35},
            ),
        ):
            point = controller._vision_nav_tap(0.685, 0.976, action="back")

        self.assertEqual(point, (370, 937))
        configure.assert_called_once_with(123)
        click.assert_called_once_with(
            123,
            370,
            937,
            countdown=0,
            hold_seconds=0.35,
            require_event_barrier=True,
        )
        receipt = controller.consume_last_click_receipt()
        self.assertTrue(receipt["seller_event_barrier_confirmed"])
        self.assertFalse(receipt["mechanical_contact_ack"])
        self.assertIsNone(controller.consume_last_click_receipt())
        profile = controller.hardware_capability_profile()["actions"]["back"]
        self.assertEqual("gui_event_barrier", profile["transport_ack"])
        self.assertFalse(profile["mechanical_contact_ack"])

    def test_double_tap_sets_two_then_restores_single_click_count(self):
        controller = RobotController(
            title="test",
            verified_actions={"double_tap"},
        )
        frame = Image.new("RGB", (540, 960), "white")

        with (
            patch("robot_core.seller_gui.find_window", return_value=(123, "test")),
            patch.object(controller, "_capture_phone", return_value=frame),
            patch.object(controller, "_consume_physical_execution"),
            patch.object(controller, "_checkpoint"),
            patch(
                "tap_calibration.corrected_grid_point",
                return_value=(500.0, 500.0),
            ),
            patch("robot_core.seller_gui.configure_click_count") as configure,
            patch(
                "robot_core.seller_gui.configure_single_click_count"
            ) as restore,
            patch(
                "robot_core.seller_gui.click_client_point",
                return_value=self._click_barrier_receipt(2),
            ) as click,
            patch("robot_core.seller_gui.clear_seller_camera_overlay") as clear,
            patch(
                "robot_core.load_controller_config",
                return_value={"tap_hold": 0.35},
            ),
        ):
            point = controller.vision_double_tap_relative(500, 500)

        self.assertEqual(point, (270, 480))
        configure.assert_called_once_with(123, 2)
        restore.assert_called_once_with(123)
        click.assert_called_once_with(
            123,
            270,
            480,
            countdown=0,
            hold_seconds=0.35,
            require_event_barrier=True,
            click_count=2,
        )
        clear.assert_called_once_with(123)
        receipt = controller.consume_last_click_receipt()
        self.assertEqual(receipt["click_count"], 2)
        self.assertEqual(receipt["click_count_restored_to"], 1)

    def test_click_event_barrier_confirms_round_trip_and_restores_cursor(self):
        class FakeUser32:
            def __init__(self):
                self.positions = []
                self.events = []

            def ClientToScreen(self, _hwnd, point):
                point._obj.x += 10
                point._obj.y += 20
                return 1

            def GetCursorPos(self, point):
                point._obj.x = 7
                point._obj.y = 9
                return 1

            def ShowWindow(self, *_args):
                return 1

            def SetForegroundWindow(self, *_args):
                return 1

            def SetCursorPos(self, x, y):
                self.positions.append((x, y))
                return 1

            def mouse_event(self, event, *_args):
                self.events.append(event)

            def GetAsyncKeyState(self, *_args):
                return 0

        fake = FakeUser32()
        baseline = np.zeros((45, 180, 3), dtype=np.int16)
        with (
            patch("robot_gui_poc.user32", fake),
            patch("robot_gui_poc.client_geometry", return_value=(0, 0, 540, 1038)),
            patch("robot_gui_poc._stable_seller_position_baseline", return_value=baseline),
            patch("robot_gui_poc._capture_seller_position_overlay", return_value=baseline),
            patch(
                "robot_gui_poc._wait_for_seller_position_state",
                side_effect=((240, 0.01), (235, 0.02)),
            ) as wait_state,
            patch("robot_gui_poc.time.sleep"),
        ):
            receipt = robot_gui_poc.click_client_point(
                123,
                270,
                937,
                countdown=0,
                hold_seconds=0.35,
                require_event_barrier=True,
            )

        self.assertEqual(
            [robot_gui_poc.MOUSEEVENTF_LEFTDOWN, robot_gui_poc.MOUSEEVENTF_LEFTUP],
            fake.events,
        )
        self.assertEqual((7, 9), fake.positions[-1])
        self.assertIn((283, 957), fake.positions)
        self.assertIn((280, 957), fake.positions)
        self.assertEqual(2, wait_state.call_count)
        self.assertTrue(receipt["seller_event_barrier_confirmed"])
        self.assertTrue(receipt["round_trip_position_confirmed"])
        self.assertFalse(receipt["mechanical_contact_ack"])

    def test_click_event_barrier_timeout_restores_cursor_without_second_click(self):
        class FakeUser32:
            def __init__(self):
                self.positions = []
                self.events = []

            def ClientToScreen(self, _hwnd, point):
                return 1

            def GetCursorPos(self, point):
                point._obj.x = 11
                point._obj.y = 12
                return 1

            def ShowWindow(self, *_args):
                return 1

            def SetForegroundWindow(self, *_args):
                return 1

            def SetCursorPos(self, x, y):
                self.positions.append((x, y))
                return 1

            def mouse_event(self, event, *_args):
                self.events.append(event)

            def GetAsyncKeyState(self, *_args):
                return 0

        fake = FakeUser32()
        baseline = np.zeros((45, 180, 3), dtype=np.int16)
        with (
            patch("robot_gui_poc.user32", fake),
            patch("robot_gui_poc.client_geometry", return_value=(0, 0, 540, 1038)),
            patch("robot_gui_poc._stable_seller_position_baseline", return_value=baseline),
            patch(
                "robot_gui_poc._wait_for_seller_position_state",
                side_effect=RuntimeError("控制端事件栅栏超时"),
            ),
            patch("robot_gui_poc.time.sleep"),
        ):
            with self.assertRaisesRegex(RuntimeError, "事件栅栏超时"):
                robot_gui_poc.click_client_point(
                    123,
                    270,
                    937,
                    countdown=0,
                    hold_seconds=0.35,
                    require_event_barrier=True,
                )

        self.assertEqual(
            [robot_gui_poc.MOUSEEVENTF_LEFTDOWN, robot_gui_poc.MOUSEEVENTF_LEFTUP],
            fake.events,
        )
        self.assertEqual((11, 12), fake.positions[-1])

    def test_click_hold_exception_releases_button_and_restores_cursor(self):
        class FakeUser32:
            def __init__(self):
                self.positions = []
                self.events = []

            def ClientToScreen(self, _hwnd, point):
                return 1

            def GetCursorPos(self, point):
                point._obj.x = 13
                point._obj.y = 14
                return 1

            def ShowWindow(self, *_args):
                return 1

            def SetForegroundWindow(self, *_args):
                return 1

            def SetCursorPos(self, x, y):
                self.positions.append((x, y))
                return 1

            def mouse_event(self, event, *_args):
                self.events.append(event)

            def GetAsyncKeyState(self, *_args):
                return 0

        fake = FakeUser32()
        sleeps = iter((RuntimeError("hold interrupted"), None))

        def sleep_side_effect(_seconds):
            outcome = next(sleeps)
            if isinstance(outcome, Exception):
                raise outcome

        with (
            patch("robot_gui_poc.user32", fake),
            patch("robot_gui_poc.client_geometry", return_value=(0, 0, 540, 1038)),
            patch("robot_gui_poc.time.sleep", side_effect=sleep_side_effect),
        ):
            with self.assertRaisesRegex(RuntimeError, "hold interrupted"):
                robot_gui_poc.click_client_point(
                    123,
                    270,
                    937,
                    countdown=0,
                    hold_seconds=0.35,
                )

        self.assertEqual(
            [robot_gui_poc.MOUSEEVENTF_LEFTDOWN, robot_gui_poc.MOUSEEVENTF_LEFTUP],
            fake.events,
        )
        self.assertEqual((13, 14), fake.positions[-1])

    def test_drag_is_fail_closed_until_device_marks_it_verified(self):
        controller = RobotController(title="test")

        self.assertFalse(controller.hardware_capabilities()["drag"])
        with self.assertRaisesRegex(Exception, "尚未完成任意两点拖动真机验收"):
            controller.vision_drag_relative(100, 200, 700, 800)

    def test_unverified_input_and_long_press_fail_before_hardware_access(self):
        controller = RobotController(
            title="test",
            verified_actions={
                "tap_semantic",
                "dismiss_overlay",
                "swipe",
                "back",
                "wait_for_change",
            },
        )

        with patch("robot_core.seller_gui.find_window") as find_window:
            with self.assertRaisesRegex(Exception, "输入.*真机验收"):
                controller.vision_type_text_with_layout(
                    "通用Agent验收草稿", TEST_QWERTY_LAYOUT
                )
            with self.assertRaisesRegex(Exception, "长按.*真机验收"):
                controller.vision_long_press_relative(500, 500)

        find_window.assert_not_called()

    def test_verified_long_press_uses_stationary_touch_channel(self):
        controller = RobotController(
            title="test",
            verified_actions={"long_press"},
        )
        frame = Image.new("RGB", (540, 960), "white")

        with (
            patch("robot_core.seller_gui.find_window", return_value=(123, "test")),
            patch.object(controller, "_capture_phone", return_value=frame),
            patch.object(controller, "_consume_physical_execution"),
            patch.object(controller, "_checkpoint"),
            patch(
                "tap_calibration.corrected_grid_point",
                return_value=(500.0, 500.0),
            ),
            patch(
                "robot_core.seller_gui.long_press_client_point",
                return_value={
                    "version": "2026-08-16-seller-gui-contact-barrier-v3",
                    "channel": "right_button_stationary_touch",
                    "seller_event_barrier_confirmed": True,
                    "round_trip_position_confirmed": True,
                    "hold_started_after_barrier": True,
                    "requested_hold_seconds": 0.8,
                    "barrier_offset_pixels": 3,
                    "changed_pixels": 240,
                    "return_changed_pixels": 240,
                    "barrier_elapsed_ms": 35.0,
                    "post_barrier_settle_seconds": 0.45,
                },
            ) as long_press,
            patch("robot_core.seller_gui.click_client_point") as click,
            patch("robot_core.seller_gui.clear_seller_camera_overlay"),
        ):
            point = controller.vision_long_press_relative(500, 500, 0.8)

        self.assertEqual((270, 480), point)
        long_press.assert_called_once_with(
            123,
            270,
            480,
            hold_seconds=0.8,
        )
        click.assert_not_called()
        receipt = controller.consume_last_long_press_receipt()
        self.assertTrue(receipt["seller_event_barrier_confirmed"])
        self.assertIsNone(controller.consume_last_long_press_receipt())

    def test_seller_position_overlay_diff_separates_noise_and_move(self):
        baseline = np.zeros((45, 180, 3), dtype=np.int16)
        noise = baseline.copy()
        noise[0:20, 0:20, :] = 12
        moved = baseline.copy()
        moved[0:20, 0:20, :] = 13

        self.assertEqual(
            0,
            robot_gui_poc._seller_position_changed_pixels(baseline, noise),
        )
        self.assertEqual(
            400,
            robot_gui_poc._seller_position_changed_pixels(baseline, moved),
        )

    def test_seller_position_barrier_waits_for_change_then_return(self):
        baseline = np.zeros((45, 180, 3), dtype=np.int16)
        moved = baseline.copy()
        moved[0:20, 0:20, :] = 20

        with patch(
            "robot_gui_poc._capture_seller_position_overlay",
            side_effect=[baseline.copy(), moved, moved, baseline.copy()],
        ):
            changed, _ = robot_gui_poc._wait_for_seller_position_state(
                123,
                baseline,
                expect_changed=True,
            )
            returned, _ = robot_gui_poc._wait_for_seller_position_state(
                123,
                baseline,
                expect_changed=False,
            )

        self.assertEqual(400, changed)
        self.assertEqual(0, returned)

    def test_verified_text_profile_rejects_unverified_characters_before_hardware(self):
        controller = RobotController(
            title="test",
            verified_actions={"input_verified_text"},
        )

        with patch("robot_core.seller_gui.find_window") as find_window:
            for text in ("Agent123", "中文", ".com"):
                with self.subTest(text=text), self.assertRaisesRegex(
                    Exception,
                    "英.*字母",
                ):
                    controller.vision_type_text_with_layout(text, TEST_QWERTY_LAYOUT)

        find_window.assert_not_called()

    def test_verified_lowercase_word_uses_audited_visible_key_sequence(self):
        controller = RobotController(
            title="test",
            verified_actions={"input_verified_text"},
        )

        with (
            patch.object(controller, "vision_type_pinyin") as type_pinyin,
        ):
            controller.vision_type_text_with_layout("agent", TEST_QWERTY_LAYOUT)

        type_pinyin.assert_called_once_with("agent", "agent", TEST_QWERTY_LAYOUT)

    def test_universal_uppercase_profile_uses_same_audited_key_geometry(self):
        controller = RobotController(
            title="test",
            verified_actions={"input_verified_text"},
        )
        layout = {"type": "qwerty", "anchors": {}}
        with patch.object(controller, "vision_type_pinyin") as type_pinyin:
            controller.vision_type_text_with_layout("M", layout)
        type_pinyin.assert_called_once_with("M", "m", layout)

        controller.validate_verified_text(
            "M",
            {
                "focused": True,
                "value": "",
                "keyboard_layout": "qwerty",
                "keyboard_input_mode": "direct_latin",
                "keyboard_case_mode": "upper",
            },
            target_text="Meeting",
            input_method="direct_latin",
        )

    def test_unicode_text_transport_emits_down_and_up_for_every_character(self):
        class FakeUser32:
            def __init__(self):
                self.events = []

            def SendInput(self, count, events, _size):
                self.events = [
                    (events[index].ki.wScan, events[index].ki.dwFlags)
                    for index in range(count)
                ]
                return count

        fake = FakeUser32()
        with (
            patch.object(robot_gui_poc, "user32", fake),
            patch("robot_gui_poc.time.sleep"),
        ):
            robot_gui_poc.type_unicode_text("ab1")

        self.assertEqual(
            [ord("a"), ord("a"), ord("b"), ord("b"), ord("1"), ord("1")],
            [item[0] for item in fake.events],
        )
        self.assertEqual(
            [
                robot_gui_poc.KEYEVENTF_UNICODE,
                robot_gui_poc.KEYEVENTF_UNICODE | robot_gui_poc.KEYEVENTF_KEYUP,
            ]
            * 3,
            [item[1] for item in fake.events],
        )

    def test_single_click_configuration_uses_current_unicode_helper(self):
        with (
            patch("robot_gui_poc.ensure_window_fully_visible"),
            patch("robot_gui_poc.client_geometry", return_value=(0, 0, 540, 1010)),
            patch("robot_gui_poc.seller_control_point", return_value=(308, 992)),
            patch("robot_gui_poc.click_client_control"),
            patch.object(robot_gui_poc.user32, "keybd_event"),
            patch("robot_gui_poc.press_virtual_key") as press,
            patch("robot_gui_poc.type_unicode_text") as type_text,
            patch("robot_gui_poc.time.sleep"),
        ):
            robot_gui_poc.configure_single_click_count(123)

        type_text.assert_called_once_with("1")
        self.assertEqual(robot_gui_poc.VK_RETURN, press.call_args_list[-1].args[0])

    def test_verified_text_profile_requires_exact_focused_qwerty_transaction(self):
        controller = RobotController(
            title="test",
            verified_actions={"input_verified_text"},
        )
        valid = {
            "focused": True,
            "value": "",
            "keyboard_layout": "qwerty",
            "keyboard_input_mode": "direct_latin",
        }
        controller.validate_verified_text(
            "agent",
            valid,
            target_text="agent",
            input_method="direct_latin",
        )
        cases = (
            ({**valid, "value": "old"}, "精确前缀"),
            ({**valid, "keyboard_layout": "symbol"}, "QWERTY"),
            ({**valid, "keyboard_input_mode": "chinese_pinyin"}, "."),
            ({key: value for key, value in valid.items() if key != "keyboard_input_mode"}, "."),
            ({**valid, "focused": False}, "聚焦"),
        )
        for states, message in cases:
            with self.subTest(states=states), self.assertRaisesRegex(
                Exception,
                message,
            ):
                controller.validate_verified_text(
                    "agent",
                    states,
                    target_text="agent",
                    input_method="direct_latin",
                )

    def test_every_unverified_physical_primitive_fails_before_hardware_access(self):
        controller = RobotController(title="test", verified_actions=set())

        operations = (
            ("点击", lambda: controller.vision_tap_relative(500, 500)),
            ("主页导航", controller.vision_android_home),
            ("滑动", controller.vision_swipe_up),
            ("返回", controller.vision_android_back),
            (
                "输入",
                lambda: controller.vision_type_text_with_layout(
                    "草稿", TEST_QWERTY_LAYOUT
                ),
            ),
            ("拼音输入", lambda: controller.vision_type_pinyin("草稿", "caogao")),
            ("退格清空", lambda: controller.vision_clear_text(delete_count=2)),
            ("长按", lambda: controller.vision_long_press_relative(500, 500)),
            ("拖动", lambda: controller.vision_drag_relative(100, 100, 900, 900)),
        )
        with patch("robot_core.seller_gui.find_window") as find_window:
            for label, operation in operations:
                with self.subTest(label=label), self.assertRaisesRegex(
                    Exception,
                    "尚未完成|尚未验证",
                ):
                    operation()

        find_window.assert_not_called()

    def test_dismiss_capability_cannot_authorize_an_arbitrary_tap(self):
        controller = RobotController(
            title="test",
            verified_actions={"dismiss_overlay"},
        )

        with patch("robot_core.seller_gui.find_window") as find_window:
            with self.assertRaisesRegex(Exception, "点击.*真机验收"):
                controller.vision_tap_relative(500, 500)

        find_window.assert_not_called()

    def test_dismiss_uses_its_own_verified_physical_entry(self):
        controller = RobotController(
            title="test",
            verified_actions={"dismiss_overlay"},
        )

        with patch.object(
            controller,
            "_vision_press_relative",
            return_value=(270, 480),
        ) as press:
            point = controller.vision_dismiss_overlay_relative(500, 500)

        self.assertEqual(point, (270, 480))
        press.assert_called_once()

    def test_verified_drag_uses_two_calibrated_points_once(self):
        controller = RobotController(
            title="test",
            verified_actions={"drag"},
        )
        frame = Image.new("RGB", (540, 960), "white")

        with (
            patch("robot_core.seller_gui.find_window", return_value=(123, "test")),
            patch.object(controller, "_capture_phone", return_value=frame),
            patch.object(controller, "_checkpoint"),
            patch(
                "tap_calibration.corrected_grid_point",
                side_effect=[(100.0, 200.0), (700.0, 800.0)],
            ),
            patch("robot_core.seller_gui.drag_client_path") as drag,
            patch("robot_core.seller_gui.clear_seller_camera_overlay") as clear_overlay,
        ):
            result = controller.vision_drag_relative(100, 200, 700, 800)

        self.assertEqual(result, ((54, 192), (377, 767)))
        drag.assert_called_once_with(123, (54, 192), (377, 767))
        clear_overlay.assert_called_once_with(123)

    def test_system_navigation_reveal_is_independent_and_default_disabled(self):
        controller = RobotController(title="test")
        self.assertFalse(
            controller.hardware_capabilities()["reveal_system_navigation"]
        )
        controller = RobotController(
            title="test",
            verified_actions={"swipe", "drag"},
        )
        with (
            patch("robot_core.seller_gui.find_window") as find_window,
            patch("robot_core.seller_gui.drag_client_path") as drag,
        ):
            with self.assertRaisesRegex(Exception, "系统边缘唤出导航栏.*真机验收"):
                controller.vision_reveal_system_navigation()
        find_window.assert_not_called()
        drag.assert_not_called()

    def test_verified_system_navigation_reveal_uses_one_local_path(self):
        controller = RobotController(
            title="test",
            verified_actions={"reveal_system_navigation"},
        )
        frame = Image.new("RGB", (810, 1440), "white")
        derived = {
            "action": "reveal_system_navigation",
            "edge": "bottom",
            "frame_size": [810, 1440],
            "dom_path": [[0.504, 0.986], [0.505, 0.700]],
            "requested_grid": [[118, 500], [350, 500]],
            "corrected_grid": [[92, 495], [333, 495]],
        }
        with (
            patch("robot_core.seller_gui.find_window", return_value=(123, "test")),
            patch.object(controller, "_capture_phone", return_value=frame),
            patch.object(controller, "_checkpoint"),
            patch(
                "tap_calibration.reveal_system_navigation_path",
                return_value=derived,
            ) as derive,
            patch("robot_core.seller_gui.drag_client_path") as drag,
            patch("robot_core.seller_gui.clear_seller_camera_overlay") as clear_overlay,
        ):
            result = controller.vision_reveal_system_navigation()

        derive.assert_called_once_with((810, 1440), controller.calibration_path)
        drag.assert_called_once_with(123, (74, 712), (269, 712))
        clear_overlay.assert_called_once_with(123)
        self.assertEqual([[74, 712], [269, 712]], result["client_path"])
        self.assertEqual(derived["dom_path"], result["dom_path"])

    def test_system_navigation_reveal_never_retries_failed_drag(self):
        controller = RobotController(
            title="test",
            verified_actions={"reveal_system_navigation"},
        )
        frame = Image.new("RGB", (810, 1440), "white")
        with (
            patch("robot_core.seller_gui.find_window", return_value=(123, "test")),
            patch.object(controller, "_capture_phone", return_value=frame),
            patch.object(controller, "_checkpoint"),
            patch(
                "tap_calibration.reveal_system_navigation_path",
                return_value={"corrected_grid": [[92, 495], [333, 495]]},
            ),
            patch(
                "robot_core.seller_gui.drag_client_path",
                side_effect=RuntimeError("drag failed"),
            ) as drag,
        ):
            with self.assertRaisesRegex(RuntimeError, "drag failed"):
                controller.vision_reveal_system_navigation()
        drag.assert_called_once()

    def test_system_navigation_reveal_calibration_failure_is_zero_action(self):
        controller = RobotController(
            title="test",
            verified_actions={"reveal_system_navigation"},
        )
        frame = Image.new("RGB", (810, 1440), "white")
        with (
            patch("robot_core.seller_gui.find_window", return_value=(123, "test")),
            patch.object(controller, "_capture_phone", return_value=frame),
            patch(
                "tap_calibration.reveal_system_navigation_path",
                side_effect=RuntimeError("invalid calibration"),
            ),
            patch("robot_core.seller_gui.drag_client_path") as drag,
        ):
            with self.assertRaisesRegex(RuntimeError, "invalid calibration"):
                controller.vision_reveal_system_navigation()
        drag.assert_not_called()

    def test_mock_system_navigation_reveal_records_semantic_evidence(self):
        controller = MockRobotController(
            verified_actions={"reveal_system_navigation"}
        )
        result = controller.vision_reveal_system_navigation()

        self.assertEqual("reveal_system_navigation", result["action"])
        self.assertEqual("bottom", result["edge"])
        self.assertEqual(1, len(controller.executions))

TEST_NUMERIC_GRID_LAYOUT = {
    "type": "numeric_grid",
    "anchors": {
        "1": [304, 744],
        "3": [685, 744],
        "7": [304, 839],
        "9": [685, 839],
        "backspace": [850, 698],
    },
}


class LowLevelInputTests(unittest.TestCase):
    def test_input_structure_matches_windows_native_size(self) -> None:
        expected_size = 40 if ctypes.sizeof(ctypes.c_void_p) == 8 else 28
        self.assertEqual(ctypes.sizeof(robot_gui_poc.INPUT), expected_size)

    def test_nihao_maps_to_calibrated_qwerty_key_centers(self) -> None:
        keyboard = DEFAULT_CONTROLLER_CONFIG["pinyin_keyboard"]
        points = [
            qwerty_key_point(540, 960, key, keyboard)
            for key in "nihao"
        ]
        self.assertEqual(
            points,
            [
                (358, 810),
                (381, 676),
                (313, 742),
                (85, 742),
                (427, 676),
            ],
        )

    def test_dynamic_qwerty_anchors_rebuild_letter_centers(self) -> None:
        keyboard = qwerty_keyboard_config_from_anchors(
            TEST_QWERTY_LAYOUT["anchors"]
        )
        points = [
            qwerty_key_point(540, 960, key, keyboard)
            for key in "nihao"
        ]
        self.assertEqual(
            points,
            [
                (358, 810),
                (381, 676),
                (313, 742),
                (85, 742),
                (427, 676),
            ],
        )
        self.assertAlmostEqual(keyboard["backspace_x_ratio"], 0.862)
        self.assertAlmostEqual(keyboard["backspace_y_ratio"], 0.844)

    def test_dynamic_qwerty_normalizes_imprecise_row_endpoints(self) -> None:
        keyboard = qwerty_keyboard_config_from_anchors(
            {
                "q": [124, 698],
                "p": [874, 698],
                "a": [124, 768],
                "l": [874, 768],
                "z": [124, 838],
                "m": [749, 838],
                "backspace": [874, 838],
            }
        )
        points = [
            qwerty_key_point(540, 960, key, keyboard)
            for key in "nihao"
        ]
        self.assertEqual(
            points,
            [
                (359, 804),
                (382, 670),
                (314, 737),
                (89, 737),
                (427, 670),
            ],
        )
        self.assertEqual(keyboard["source"], "vision_anchors_normalized")

    def test_dynamic_qwerty_accepts_audited_bottom_row_near_frame_edge(self) -> None:
        keyboard = qwerty_keyboard_config_from_anchors(
            {
                "q": [110, 840],
                "p": [890, 840],
                "a": [160, 900],
                "l": [840, 900],
                "z": [260, 960],
                "m": [740, 960],
                "backspace": [890, 960],
            }
        )

        self.assertAlmostEqual(keyboard["rows"][2]["y"], 0.96)
        self.assertAlmostEqual(keyboard["backspace_y_ratio"], 0.96)

    def test_dynamic_qwerty_rejects_bottom_row_without_safe_center_margin(self) -> None:
        with self.assertRaisesRegex(Exception, "行位置或上下顺序异常"):
            qwerty_keyboard_config_from_anchors(
                {
                    "q": [110, 870],
                    "p": [890, 870],
                    "a": [160, 930],
                    "l": [840, 930],
                    "z": [260, 990],
                    "m": [740, 990],
                    "backspace": [890, 990],
                }
            )

    def test_dynamic_qwerty_rejects_non_qwerty_geometry(self) -> None:
        invalid = {
            **TEST_QWERTY_LAYOUT["anchors"],
            "p": [90, 704],
        }
        with self.assertRaisesRegex(Exception, "左右顺序错误"):
            qwerty_keyboard_config_from_anchors(invalid)


class DeepSeekIntentProviderTests(unittest.TestCase):
    @staticmethod
    def _response(content: str = '{"understood":false}') -> httpx.Response:
        request = httpx.Request(
            "POST",
            "https://api.deepseek.com/chat/completions",
        )
        return httpx.Response(
            200,
            request=request,
            json={
                "id": "deepseek-test-request",
                "choices": [{"message": {"content": content}}],
                "usage": {"total_tokens": 8},
            },
        )

    def test_uses_v4_flash_json_non_thinking_mode(self) -> None:
        provider = DeepSeekIntentProvider(
            api_key="test-key",
            max_attempts=1,
            retry_base_delay=0,
        )
        with patch(
            "intent_provider.httpx.post",
            return_value=self._response(),
        ) as mocked:
            result = provider.chat_json(
                [{"role": "user", "content": "json: test"}],
                max_tokens=123,
            )
        self.assertEqual(result, '{"understood":false}')
        body = mocked.call_args.kwargs["json"]
        self.assertEqual(body["model"], "deepseek-v4-flash")
        self.assertEqual(body["thinking"], {"type": "disabled"})
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertEqual(body["max_tokens"], 123)
        self.assertEqual(provider.status()["last_request_id"], "deepseek-test-request")

    def test_missing_key_fails_without_network_request(self) -> None:
        provider = DeepSeekIntentProvider(api_key="")
        with patch("intent_provider.httpx.post") as mocked:
            with self.assertRaisesRegex(IntentProviderError, "DEEPSEEK_API_KEY"):
                provider.chat_json([{"role": "user", "content": "json"}])
        mocked.assert_not_called()

    def test_uses_windows_user_environment_when_process_env_is_stale(self) -> None:
        with patch.dict("intent_provider.os.environ", {}, clear=True), patch(
            "intent_provider._read_windows_user_environment",
            return_value="registry-key",
        ):
            provider = DeepSeekIntentProvider()
        self.assertTrue(provider.configured)
        self.assertEqual(provider.api_key, "registry-key")


class DeviceControllerRegistryTests(unittest.TestCase):
    def test_default_real_device_advertises_only_actions_with_live_evidence(self) -> None:
        registry = DeviceControllerRegistry(
            web_app.DEVICE_REGISTRY_PATH,
            promotable_actions=PROMOTABLE_ACTIONS,
            mock=False,
        )
        controller = registry.controller(registry.default_device_id)

        self.assertTrue(controller.hardware_capabilities()["input_verified_text"])
        self.assertTrue(controller.hardware_capabilities()["long_press"])
        self.assertTrue(controller.hardware_capabilities()["drag"])

    def test_two_devices_have_independent_controllers_and_calibrations(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "devices.json"
            path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "default_device_id": "phone-a",
                        "devices": [
                            {
                                "device_id": "phone-a",
                                "enabled": True,
                                "window_title": "controller-a",
                                "calibration_path": "calibration-a.json",
                            },
                            {
                                "device_id": "phone-b",
                                "enabled": True,
                                "window_title": "controller-b",
                                "calibration_path": "calibration-b.json",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            registry = DeviceControllerRegistry(
                path,
                promotable_actions=PROMOTABLE_ACTIONS,
                mock=False,
            )

            first = registry.controller("phone-a")
            second = registry.controller("phone-b")

        self.assertIsNot(first, second)
        self.assertEqual(first.title, "controller-a")
        self.assertEqual(second.title, "controller-b")
        self.assertTrue(str(first.calibration_path).endswith("calibration-a.json"))
        self.assertTrue(str(second.calibration_path).endswith("calibration-b.json"))
        with self.assertRaisesRegex(DeviceControllerRegistryError, "未登记"):
            registry.controller("phone-c")

    def test_duplicate_enabled_window_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "devices.json"
            path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "default_device_id": "phone-a",
                        "devices": [
                            {"device_id": "phone-a", "enabled": True, "window_title": "same"},
                            {"device_id": "phone-b", "enabled": True, "window_title": "same"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "同一个机械臂控制窗口"):
                DeviceControllerRegistry(
                    path,
                    promotable_actions=PROMOTABLE_ACTIONS,
                )

    def test_mock_devices_remain_independent_without_mutating_descriptors(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "devices.json"
            path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "default_device_id": "phone-a",
                        "devices": [
                            {
                                "device_id": "phone-a",
                                "enabled": True,
                                "window_title": "controller-a",
                                "calibration_path": "a.json",
                            },
                            {
                                "device_id": "phone-b",
                                "enabled": True,
                                "window_title": "controller-b",
                                "calibration_path": "b.json",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            registry = DeviceControllerRegistry(
                path,
                promotable_actions=PROMOTABLE_ACTIONS,
                mock=True,
            )

            first = registry.controller("phone-a")
            second = registry.controller("phone-b")
            descriptors_before = registry.descriptors()

        self.assertIsInstance(first, _MockRobotController)
        self.assertIsInstance(second, _MockRobotController)
        self.assertIsNot(first, second)
        self.assertEqual(descriptors_before, registry.descriptors())

    def test_invalid_verified_actions_fails_before_controller_construction(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "devices.json"
            path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "default_device_id": "phone-a",
                        "devices": [
                            {
                                "device_id": "phone-a",
                                "enabled": True,
                                "verified_actions": ["back", ""],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                DeviceControllerRegistryError,
                "verified_actions 格式无效",
            ):
                DeviceControllerRegistry(
                    path,
                    promotable_actions=PROMOTABLE_ACTIONS,
                )


class DeviceCameraCoordinatorTests(unittest.TestCase):
    def test_task_lease_serves_cached_preview_without_another_capture(self) -> None:
        coordinator = DeviceCameraCoordinator()
        preview_calls = []

        def capture_preview(*, quality):
            preview_calls.append(quality)
            return b"live-preview"

        first, first_cached = coordinator.capture_preview(
            capture_preview,
            quality=72,
            cache_only=False,
        )
        with coordinator.serial_session():
            cached, cached_during_task = coordinator.capture_preview(
                capture_preview,
                quality=72,
                cache_only=True,
            )
            frame = coordinator.capture_agent_frame(
                lambda: Image.new("RGB", (540, 960), "#203040")
            )

        self.assertEqual(b"live-preview", first)
        self.assertFalse(first_cached)
        self.assertEqual(b"live-preview", cached)
        self.assertTrue(cached_during_task)
        self.assertEqual([72], preview_calls)
        self.assertEqual((540, 960), frame.size)

    def test_concurrent_preview_never_waits_or_captures_behind_task_lease(self) -> None:
        coordinator = DeviceCameraCoordinator()
        coordinator.capture_preview(
            lambda *, quality: b"primed-preview",
            quality=72,
            cache_only=False,
        )
        lease_started = threading.Event()
        release_lease = threading.Event()

        def hold_task_lease():
            with coordinator.serial_session():
                lease_started.set()
                release_lease.wait(2.0)

        worker = threading.Thread(target=hold_task_lease)
        worker.start()
        self.assertTrue(lease_started.wait(1.0))
        capture_calls = []
        started = time.monotonic()
        try:
            payload, cached = coordinator.capture_preview(
                lambda *, quality: capture_calls.append(quality) or b"wrong",
                quality=72,
                cache_only=False,
            )
        finally:
            release_lease.set()
            worker.join(2.0)

        self.assertLess(time.monotonic() - started, 0.5)
        self.assertEqual(b"primed-preview", payload)
        self.assertTrue(cached)
        self.assertEqual([], capture_calls)
        self.assertFalse(worker.is_alive())

    def test_cache_only_without_a_frame_fails_closed(self) -> None:
        coordinator = DeviceCameraCoordinator()

        with self.assertRaisesRegex(CameraPreviewUnavailable, "尚无可复用"):
            coordinator.capture_preview(
                lambda *, quality: b"must-not-run",
                quality=72,
                cache_only=True,
            )

    def test_two_device_coordinators_do_not_share_preview_cache(self) -> None:
        first = DeviceCameraCoordinator()
        second = DeviceCameraCoordinator()
        first.capture_preview(
            lambda *, quality: b"phone-a",
            quality=72,
            cache_only=False,
        )

        with self.assertRaises(CameraPreviewUnavailable):
            second.capture_preview(
                lambda *, quality: b"phone-b",
                quality=72,
                cache_only=True,
            )

        cached, is_cached = first.capture_preview(
            lambda *, quality: b"wrong",
            quality=72,
            cache_only=True,
        )
        self.assertEqual(b"phone-a", cached)
        self.assertTrue(is_cached)


class DeviceRuntimeResourceRegistryTests(unittest.TestCase):
    def test_same_device_reuses_resources_and_devices_remain_isolated(self) -> None:
        registry = DeviceRuntimeResourceRegistry(("phone-a",))

        self.assertIs(
            registry.coordination_lock("phone-a"),
            registry.coordination_lock("phone-a"),
        )
        self.assertIs(
            registry.camera_coordinator("phone-a"),
            registry.camera_coordinator("phone-a"),
        )
        self.assertIsNot(
            registry.coordination_lock("phone-a"),
            registry.coordination_lock("phone-b"),
        )
        self.assertIsNot(
            registry.camera_coordinator("phone-a"),
            registry.camera_coordinator("phone-b"),
        )

    def test_concurrent_first_lookup_returns_one_stable_resource(self) -> None:
        registry = DeviceRuntimeResourceRegistry()
        barrier = threading.Barrier(9)
        results = []

        def resolve() -> None:
            barrier.wait()
            results.append(registry.coordination_lock("phone-parallel"))

        workers = [threading.Thread(target=resolve) for _index in range(8)]
        for worker in workers:
            worker.start()
        barrier.wait()
        for worker in workers:
            worker.join(timeout=2)

        self.assertEqual(8, len(results))
        self.assertTrue(all(item is results[0] for item in results))

    def test_empty_device_fails_before_resource_creation(self) -> None:
        registry = DeviceRuntimeResourceRegistry()
        with self.assertRaisesRegex(DeviceRuntimeResourceError, "device_id 不能为空"):
            registry.coordination_lock("  ")
        with self.assertRaisesRegex(DeviceRuntimeResourceError, "device_id 不能为空"):
            registry.camera_coordinator("")


class ApiEndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.no_browser_patcher = patch.dict(
            "os.environ",
            {"ROBOT_WEB_NO_BROWSER": "1"},
        )
        cls.no_browser_patcher.start()
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.original_web_output_dir = web_app.WEB_OUTPUT_DIR
        web_app.WEB_OUTPUT_DIR = Path(cls.temp_dir.name) / "web_output"
        web_app.WEB_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        web_app.runtime.controller = MockRobotController()
        cls.client_context = TestClient(web_app.app)
        cls.client = cls.client_context.__enter__()
        cls.headers = {"X-Control-Token": web_app.CONTROL_TOKEN}

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client_context.__exit__(None, None, None)
        web_app.WEB_OUTPUT_DIR = cls.original_web_output_dir
        cls.temp_dir.cleanup()
        cls.no_browser_patcher.stop()

    def setUp(self) -> None:
        self.device_registry_patcher = patch.object(
            web_app.runtime,
            "device_task_registry",
            DeviceTaskRegistry(),
        )
        self.device_registry_patcher.start()
        web_app.runtime.agent_session_repository.clear()
        web_app.runtime.device_runtime_resources = DeviceRuntimeResourceRegistry(
            (web_app.runtime.device_controllers.default_device_id,)
        )

    def tearDown(self) -> None:
        self.device_registry_patcher.stop()


    def _universal_api_orchestrator(self, *, device_id="phone-01", graph=None):
        from test_universal_agent_orchestrator import (
            FakeDeepSeekPlanner,
            FakeExecutingAdapter,
            FakeQwenObserver,
            _graph,
            _scene,
            _trusted_factory,
        )
        from universal_agent_orchestrator import UniversalAgentOrchestrator

        initial = graph or _graph(device_id=device_id)
        planner = FakeDeepSeekPlanner(
            initial,
            replan_result=replace(initial, revision=initial.revision + 1),
        )
        qwen = FakeQwenObserver()
        target_app_id = (
            initial.goal.target_apps[0].app_id
            if len(initial.goal.target_apps) == 1
            else "sample.app"
        )
        active_subgoal = initial.active_subgoal()
        before_scene = (
            _scene(meaning="send_message", label="发送")
            if active_subgoal is not None
            and active_subgoal.external_impact == "external_state"
            else _scene()
        )
        adapter = FakeExecutingAdapter(
            replace(before_scene, app_id=target_app_id),
            replace(
                _scene(
                    fingerprint="frame-after-api",
                    meaning="open_more",
                    label="查看更多",
                ),
                app_id=target_app_id,
            ),
        )
        orchestrator = UniversalAgentOrchestrator(
            deepseek_planner=planner,
            qwen_observer=qwen,
            adapter_factory=lambda _device_id: adapter,
            trusted_observation_factory=_trusted_factory,
            evidence_store_factory=FileSystemAgentEvidenceStore,
            device_registry=DeviceTaskRegistry(),
        )
        return orchestrator, planner, qwen, adapter

    def _fake_capability_manager(self, *, device_id="capability-api-device"):
        calls = []

        class Session:
            def __init__(self):
                self.physical_actions = 0
                self.status = "awaiting_confirmation"

            def snapshot(self):
                return {
                    "session_id": "capability-session-001",
                    "status": self.status,
                    "physical_actions": self.physical_actions,
                    "confirmation_scope": {
                        "session_id": "capability-session-001",
                        "task_id": "task-001",
                        "device_id": device_id,
                        "revision": 1,
                        "subgoal_id": "subgoal-001",
                        "effect_ids": [],
                        "observation_id": "obs-001",
                        "fingerprint": "frame-001",
                    },
                }

        session = Session()
        trial = SimpleNamespace(
            trial_id="trial-api-001",
            device_id=device_id,
            candidate_action="drag",
            session=session,
        )
        trial.snapshot = lambda: {
            "trial_id": trial.trial_id,
            "device_id": trial.device_id,
            "candidate_action": trial.candidate_action,
            "session": session.snapshot(),
            "report": None,
            "promotion_scope": None,
            "promotion": None,
            "requires_restart": False,
        }

        class Manager:
            def __init__(self):
                self.promoted = False

            def start(self, *, device_id, candidate_action, text):
                calls.append(("start", device_id, candidate_action, text))
                return trial

            def get(self, trial_id):
                calls.append(("get", trial_id))
                if trial_id != trial.trial_id:
                    raise web_app.CapabilityAcceptanceError("不存在")
                return trial

            def confirm(self, trial_id, confirmation):
                calls.append(("confirm", trial_id, dict(confirmation)))
                if session.physical_actions:
                    raise web_app.CapabilityAcceptanceError("动作确认已使用")
                session.physical_actions = 1
                session.status = "paused"
                return {
                    "physical_actions": 1,
                    "action_outcome": "matched",
                }

            def promotion_scope(self, trial_id):
                calls.append(("promotion_scope", trial_id))
                return SimpleNamespace(
                    to_dict=lambda: {
                        "trial_id": trial.trial_id,
                        "device_id": trial.device_id,
                        "action": trial.candidate_action,
                        "report_sha256": "a" * 64,
                        "registry_sha256": "b" * 64,
                    }
                )

            def promote(self, trial_id, confirmation):
                calls.append(("promote", trial_id, dict(confirmation)))
                if self.promoted:
                    raise web_app.CapabilityAcceptanceError("晋级确认已使用")
                self.promoted = True
                return {
                    "device_id": trial.device_id,
                    "action": trial.candidate_action,
                    "requires_restart": True,
                }

            def cancel(self, trial_id):
                calls.append(("cancel", trial_id))
                session.status = "cancelled"

        return Manager(), trial, calls

    def test_home_and_device_are_available(self) -> None:
        self.assertEqual(self.client.get("/").status_code, 200)
        device = self.client.get("/api/device").json()
        self.assertTrue(device["controller_online"])
        self.assertTrue(device["camera_online"])
        self.assertEqual(device["default_device_id"], "device-local-01")
        self.assertEqual(device["devices"][0]["device_id"], "device-local-01")
        self.assertNotIn("legacy_free_agent", device)
        architecture = dict(device["execution_architecture"])
        universal = dict(architecture["universal_agent"])
        observer = universal.pop("observer")
        self.assertEqual(
            observer["observer_version"],
            SINGLE_STEP_SCENE_OBSERVER_VERSION,
        )
        self.assertEqual(observer["max_online_calls_per_observation"], 1)
        self.assertEqual(observer["model_role"], "single_step_fused_observation")
        self.assertEqual(observer["supported_app_scope"], "dynamic")
        architecture["universal_agent"] = universal
        self.assertEqual(architecture["active_orchestrator"], "universal_agent")
        self.assertEqual(architecture["controller"], "universal_action_controller")
        self.assertTrue(architecture["fixed_app_workflows_retired"])
        self.assertNotIn("background_compatibility_worker", architecture)
        self.assertTrue(universal["automatic_loop_enabled"])
        self.assertEqual(universal["automatic_loop_max_physical_actions"], 12)
        self.assertEqual(universal["supported_app_scope"], "dynamic")
        self.assertEqual(
            universal["action_protocol"],
            "2026-08-20-canonical-action-v1",
        )
        self.assertEqual(
            universal["controller_protocol"],
            web_app.UNIVERSAL_CONTROLLER_PROTOCOL_VERSION,
        )
        semantic_authority = universal["typed_effect_authority"]
        self.assertEqual(
            semantic_authority["authority_scope"],
            "typed_task_and_effect_policy",
        )
        self.assertFalse(
            semantic_authority["retired_remote_risk_diagnostics_enabled"]
        )
        self.assertEqual(
            semantic_authority["canonical_action_protocol"],
            "2026-08-20-canonical-action-v1",
        )
        self.assertEqual(
            universal["hardware_capability_profile"]["protocol_version"],
            "2026-08-18-device-capability-profile-v1",
        )
        self.assertTrue(
            universal["hardware_capability_profile"]["actions"]["tap_semantic"]
            ["fresh_visual_postcondition_required"]
        )
    def test_home_uses_generic_supervised_single_step_endpoints(self) -> None:
        home = self.client.get("/")
        script = self.client.get("/assets/app.js")
        protocol_adapter = self.client.get("/assets/protocol_adapter.js")
        styles = self.client.get("/assets/styles.css")
        self.assertEqual(home.status_code, 200)
        self.assertEqual(script.status_code, 200)
        self.assertEqual(protocol_adapter.status_code, 200)
        self.assertEqual(styles.status_code, 200)
        self.assertIn("你希望手机完成什么", home.text)
        self.assertIn("动态计划", home.text)
        self.assertIn("当前画面", home.text)
        self.assertIn("步骤记录", home.text)
        self.assertIn('id="pauseButton"', home.text)
        self.assertIn('id="stopButton"', home.text)
        self.assertIn('id="riskDialog"', home.text)
        self.assertIn('id="capabilityAcceptancePanel"', home.text)
        self.assertIn('id="promotionDialog"', home.text)
        self.assertIn('id="deviceId"', home.text)
        self.assertNotIn("微信工作流", home.text)
        self.assertNotIn("抖音工作流", home.text)
        self.assertIn('/assets/protocol_adapter.js', home.text)
        self.assertIn("/api/agent/generic-supervised/start", script.text)
        self.assertIn("/api/capability-acceptance/start", script.text)
        self.assertIn("createPromotionGrant", protocol_adapter.text)
        self.assertNotIn("/api/agent/generic-supervised/${view.sessionId}/auto", script.text)
        self.assertNotIn('id="confirmSafeLoop"', home.text)
        self.assertIn("nextSupervisedAgent", script.text)
        self.assertIn("togglePause", script.text)
        self.assertNotIn('api("/api/agent/supervised/start"', script.text)
        self.assertNotIn("wechatView", script.text)
        self.assertNotIn("douyinView", script.text)

    def test_capability_revision_must_match_loaded_service_code(self) -> None:
        runtime = web_app.Runtime.__new__(web_app.Runtime)
        runtime.loaded_code_revision = "loaded-revision"

        with patch.object(web_app, "current_code_revision", return_value="loaded-revision"):
            self.assertEqual(runtime.capability_code_revision(), "loaded-revision")
        with (
            patch.object(web_app, "current_code_revision", return_value="new-revision"),
            self.assertRaisesRegex(
                web_app.CapabilityAcceptanceError,
                "服务启动后代码状态发生变化",
            ),
        ):
            runtime.capability_code_revision()

    def test_generic_supervised_auto_request_is_strict_and_bounded(self) -> None:
        request = web_app.GenericSupervisedAutoRequest(device_id="phone-01")
        self.assertFalse(request.confirmed)
        self.assertIsNone(request.confirmation)
        self.assertEqual(request.max_physical_actions, 12)
        self.assertEqual(request.max_iterations, 24)
        bounded = web_app.GenericSupervisedAutoRequest(
            device_id="phone-01",
            max_physical_actions=20,
            max_iterations=40,
        )
        self.assertEqual(bounded.max_physical_actions, 20)
        self.assertEqual(bounded.max_iterations, 40)
        with self.assertRaises(ValueError):
            web_app.GenericSupervisedAutoRequest(
                device_id="phone-01",
                max_physical_actions=21,
            )
        with self.assertRaises(ValueError):
            web_app.GenericSupervisedAutoRequest(
                device_id="phone-01",
                max_physical_actions="1",
            )

    def test_capability_acceptance_requests_are_strict(self) -> None:
        with self.assertRaises(ValueError):
            web_app.CapabilityAcceptanceStartRequest(
                device_id="device-a",
                action="drag",
                text="拖动安全控件",
                unexpected="forbidden",
            )
        with self.assertRaises(ValueError):
            web_app.CapabilityActionConfirmationRequest(confirmed="true")
        with self.assertRaises(ValueError):
            web_app.CapabilityPromotionRequest(
                confirmed=True,
                trial_id="trial-a",
                device_id="device-a",
                action="drag",
                report_sha256="not-a-sha",
                registry_sha256="b" * 64,
            )

    def test_capability_acceptance_api_starts_at_zero_and_binds_action_scope(self) -> None:
        manager, trial, calls = self._fake_capability_manager()
        before_executions = list(web_app.runtime.controller.executions)
        with (
            patch.object(web_app.runtime, "capability_acceptance_manager", manager),
            patch.object(web_app, "_require_supervised_device_ready"),
            patch.object(
                web_app,
                "_supervised_hardware_lock",
                side_effect=lambda _device_id: nullcontext(),
            ),
        ):
            missing_token = self.client.post(
                "/api/capability-acceptance/start",
                json={
                    "device_id": trial.device_id,
                    "action": "drag",
                    "text": "拖动安全控件",
                },
            )
            started = self.client.post(
                "/api/capability-acceptance/start",
                headers=self.headers,
                json={
                    "device_id": trial.device_id,
                    "action": "drag",
                    "text": "拖动安全控件",
                },
            )

        self.assertEqual(missing_token.status_code, 403, missing_token.text)
        self.assertEqual(started.status_code, 200, started.text)
        payload = started.json()
        self.assertEqual(payload["physical_actions"], 0)
        scope = payload["trial"]["action_confirmation_scope"]
        self.assertEqual(scope["trial_id"], trial.trial_id)
        self.assertEqual(scope["action"], "drag")
        self.assertEqual([call[0] for call in calls], ["start"])
        self.assertEqual(before_executions, web_app.runtime.controller.executions)

    def test_capability_action_confirmation_is_exact_once(self) -> None:
        manager, trial, calls = self._fake_capability_manager()
        scope = {
            **trial.session.snapshot()["confirmation_scope"],
            "trial_id": trial.trial_id,
            "action": trial.candidate_action,
        }
        path = f"/api/capability-acceptance/{trial.trial_id}/confirm"
        before_executions = list(web_app.runtime.controller.executions)
        with (
            patch.object(web_app.runtime, "capability_acceptance_manager", manager),
            patch.object(web_app, "_require_supervised_device_ready"),
            patch.object(
                web_app,
                "_supervised_hardware_lock",
                side_effect=lambda _device_id: nullcontext(),
            ),
        ):
            first = self.client.post(
                path,
                headers=self.headers,
                json={"confirmed": True, "confirmation": scope},
            )
            replay = self.client.post(
                path,
                headers=self.headers,
                json={"confirmed": True, "confirmation": scope},
            )
            extra = self.client.post(
                path,
                headers=self.headers,
                json={
                    "confirmed": True,
                    "confirmation": {**scope, "unexpected": "forbidden"},
                },
            )

        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(first.json()["physical_actions"], 1)
        self.assertEqual(replay.status_code, 409, replay.text)
        self.assertEqual(extra.status_code, 422, extra.text)
        self.assertEqual([call[0] for call in calls].count("confirm"), 2)
        self.assertEqual(before_executions, web_app.runtime.controller.executions)

    def test_capability_outer_scope_mismatch_is_terminal_without_action(self) -> None:
        manager, trial, calls = self._fake_capability_manager()
        scope = {
            **trial.session.snapshot()["confirmation_scope"],
            "trial_id": trial.trial_id,
            "action": "long_press",
        }
        path = f"/api/capability-acceptance/{trial.trial_id}/confirm"
        with patch.object(web_app.runtime, "capability_acceptance_manager", manager):
            wrong = self.client.post(
                path,
                headers=self.headers,
                json={"confirmed": True, "confirmation": scope},
            )

        self.assertEqual(wrong.status_code, 409, wrong.text)
        self.assertEqual(wrong.json()["detail"]["physical_actions"], 0)
        self.assertEqual(trial.session.physical_actions, 0)
        self.assertEqual(trial.session.status, "cancelled")
        self.assertEqual([call[0] for call in calls].count("confirm"), 0)
        self.assertEqual([call[0] for call in calls].count("cancel"), 1)

    def test_capability_promotion_is_separate_zero_action_and_requires_restart(self) -> None:
        manager, trial, calls = self._fake_capability_manager()
        path = f"/api/capability-acceptance/{trial.trial_id}"
        before_executions = list(web_app.runtime.controller.executions)
        with (
            patch.object(web_app.runtime, "capability_acceptance_manager", manager),
            patch.object(
                web_app.runtime.vision_provider,
                "status",
                side_effect=AssertionError("promotion must not inspect Qwen"),
            ),
        ):
            preview = self.client.get(
                f"{path}/promotion-preview",
                headers=self.headers,
            )
            scope = preview.json()["promotion_scope"]
            refused = self.client.post(
                f"{path}/promote",
                headers=self.headers,
                json={"confirmed": False, **scope},
            )
            promoted = self.client.post(
                f"{path}/promote",
                headers=self.headers,
                json={"confirmed": True, **scope},
            )
            replay = self.client.post(
                f"{path}/promote",
                headers=self.headers,
                json={"confirmed": True, **scope},
            )

        self.assertEqual(preview.status_code, 200, preview.text)
        self.assertEqual(preview.json()["physical_actions"], 0)
        self.assertEqual(refused.status_code, 409, refused.text)
        self.assertEqual(promoted.status_code, 200, promoted.text)
        self.assertEqual(promoted.json()["physical_actions"], 0)
        self.assertTrue(promoted.json()["promotion"]["requires_restart"])
        self.assertEqual(replay.status_code, 409, replay.text)
        self.assertEqual([call[0] for call in calls].count("promote"), 2)
        self.assertEqual(before_executions, web_app.runtime.controller.executions)

    def test_capability_evidence_endpoint_is_token_and_trial_bound(self) -> None:
        manager, trial, _calls = self._fake_capability_manager()
        run_dir = Path(self.temp_dir.name) / "capability-evidence"
        run_dir.mkdir(exist_ok=True)
        frame = run_dir / "before-1.jpg"
        Image.new("RGB", (8, 8), "white").save(frame, format="JPEG")
        outside = Path(self.temp_dir.name) / "outside.jpg"
        Image.new("RGB", (8, 8), "black").save(outside, format="JPEG")
        trial.run_dir = run_dir
        trial.report_path = run_dir / "acceptance_report.json"
        trial.report_path.write_text(
            json.dumps(
                {
                    "before_frame_paths": [str(frame)],
                    "after_frame_paths": [],
                }
            ),
            encoding="utf-8",
        )
        path = f"/api/capability-acceptance/{trial.trial_id}/evidence/before/0"

        with patch.object(web_app.runtime, "capability_acceptance_manager", manager):
            forbidden = self.client.get(path)
            accepted = self.client.get(path, headers=self.headers)
            missing = self.client.get(
                f"/api/capability-acceptance/{trial.trial_id}/evidence/before/1",
                headers=self.headers,
            )
            trial.report_path.write_text(
                json.dumps(
                    {
                        "before_frame_paths": [str(outside)],
                        "after_frame_paths": [],
                    }
                ),
                encoding="utf-8",
            )
            escaped = self.client.get(path, headers=self.headers)

        self.assertEqual(forbidden.status_code, 403, forbidden.text)
        self.assertEqual(accepted.status_code, 200, accepted.text)
        self.assertEqual(accepted.headers["content-type"], "image/jpeg")
        self.assertEqual(missing.status_code, 404, missing.text)
        self.assertEqual(escaped.status_code, 404, escaped.text)

    def test_v3_confirm_request_requires_scope_and_forbids_extra_fields(self) -> None:
        orchestrator, _planner, _qwen, adapter = self._universal_api_orchestrator()
        session = orchestrator.start(
            session_id="api-scope",
            raw_goal="查看详情",
            device_id="phone-01",
            run_dir=web_app.WEB_OUTPUT_DIR / "api-scope",
        )
        web_app.runtime.agent_session_repository.add(session)
        path = f"/api/agent/generic-supervised/{session.session_id}/confirm"
        scope = session.snapshot()["confirmation_scope"]

        with (
            patch.object(web_app, "_require_supervised_device_ready"),
            patch.object(
                web_app.runtime,
                "universal_agent_orchestrator",
                orchestrator,
            ),
        ):
            missing = self.client.post(
                path,
                headers=self.headers,
                json={"confirmed": True},
            )
        self.assertEqual(missing.status_code, 409, missing.text)
        self.assertEqual(missing.json()["detail"]["physical_actions"], 0)
        self.assertEqual(adapter.execute_calls, 0)

        with self.assertRaises(ValueError):
            web_app.GenericSupervisedStepRequest(
                confirmed="true",
                confirmation=scope,
            )

        for payload in (
            {
                "confirmed": True,
                "confirmation": scope,
                "unexpected": "forbidden",
            },
            {
                "confirmed": True,
                "confirmation": {
                    **scope,
                    "unexpected": "forbidden",
                },
            },
        ):
            with self.subTest(payload=payload):
                with patch.object(web_app, "_require_supervised_device_ready"):
                    rejected = self.client.post(
                        path,
                        headers=self.headers,
                        json=payload,
                    )
                self.assertEqual(rejected.status_code, 422, rejected.text)
                self.assertEqual(adapter.execute_calls, 0)

    def test_v3_confirm_api_atomically_consumes_one_scope(self) -> None:
        orchestrator, _planner, _qwen, adapter = self._universal_api_orchestrator()
        session = orchestrator.start(
            session_id="api-confirm-once",
            raw_goal="查看详情",
            device_id="phone-01",
            run_dir=web_app.WEB_OUTPUT_DIR / "api-confirm-once",
        )
        web_app.runtime.agent_session_repository.add(session)
        path = f"/api/agent/generic-supervised/{session.session_id}/confirm"
        payload = {
            "confirmed": True,
            "confirmation": session.snapshot()["confirmation_scope"],
        }

        with (
            patch.object(web_app, "_require_supervised_device_ready"),
            patch.object(
                web_app.runtime,
                "universal_agent_orchestrator",
                orchestrator,
            ),
        ):
            first = self.client.post(path, headers=self.headers, json=payload)
            replay = self.client.post(path, headers=self.headers, json=payload)

        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(first.json()["execution"]["physical_actions"], 1)
        self.assertEqual(first.json()["session"]["task_graph"]["revision"], 2)
        self.assertEqual(len(first.json()["execution"]["after_frame_paths"]), 4)
        self.assertEqual(replay.status_code, 409, replay.text)
        self.assertEqual(replay.json()["detail"]["physical_actions"], 0)
        self.assertEqual(adapter.execute_calls, 1)

    def test_device_disconnect_invalidates_pending_confirmation(self) -> None:
        orchestrator, _planner, _qwen, adapter = self._universal_api_orchestrator()
        session = orchestrator.start(
            session_id="api-disconnect-invalidates",
            raw_goal="查看详情",
            device_id="phone-01",
            run_dir=web_app.WEB_OUTPUT_DIR / "api-disconnect-invalidates",
        )
        web_app.runtime.agent_session_repository.add(session)
        path = f"/api/agent/generic-supervised/{session.session_id}/confirm"
        payload = {
            "confirmed": True,
            "confirmation": session.snapshot()["confirmation_scope"],
        }

        def offline(_device_id=None) -> None:
            raise web_app.HTTPException(status_code=409, detail="控制端或摄像头离线。")

        with (
            patch.object(web_app, "_require_supervised_device_ready", offline),
            patch.object(
                web_app.runtime,
                "universal_agent_orchestrator",
                orchestrator,
            ),
        ):
            disconnected = self.client.post(
                path, headers=self.headers, json=payload
            )

        self.assertEqual(409, disconnected.status_code, disconnected.text)
        self.assertIsNone(session.snapshot()["confirmation_scope"])
        self.assertEqual("needs_reobservation", session.status)
        self.assertEqual(0, adapter.execute_calls)

        with (
            patch.object(web_app, "_require_supervised_device_ready"),
            patch.object(
                web_app.runtime,
                "universal_agent_orchestrator",
                orchestrator,
            ),
        ):
            stale = self.client.post(path, headers=self.headers, json=payload)
        self.assertEqual(409, stale.status_code, stale.text)
        self.assertEqual(0, adapter.execute_calls)

    def test_hardware_lock_rejects_another_process_lease(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            lease_dir = Path(temp)
            external = InterProcessLease(
                lease_dir / "physical_hardware_action.lease",
                owner_id="other-service",
                metadata={"purpose": "physical_hardware_action"},
            )
            self.assertTrue(external.acquire())
            try:
                with (
                    patch.object(web_app, "SHARED_DEVICE_LEASE_DIR", lease_dir),
                    self.assertRaisesRegex(
                        web_app.HTTPException, "另一进程已占用"
                    ),
                ):
                    with web_app._supervised_hardware_lock():
                        self.fail("cross-process lease must block the hardware lock")
            finally:
                external.release()

    def test_hardware_locks_allow_different_devices_but_reject_same_device(self) -> None:
        first_controller = SimpleNamespace(operation_lock=threading.Lock())
        second_controller = SimpleNamespace(operation_lock=threading.Lock())
        controllers = {
            "phone-a": first_controller,
            "phone-b": second_controller,
        }
        with tempfile.TemporaryDirectory() as temp:
            with (
                patch.object(web_app, "SHARED_DEVICE_LEASE_DIR", Path(temp)),
                patch.object(
                    web_app.runtime,
                    "controller_for_device",
                    side_effect=lambda device_id: controllers[device_id],
                ),
            ):
                with web_app._supervised_hardware_lock("phone-a"):
                    with web_app._supervised_hardware_lock("phone-b"):
                        self.assertTrue(first_controller.operation_lock.locked())
                        self.assertTrue(second_controller.operation_lock.locked())
                    with self.assertRaisesRegex(
                        web_app.HTTPException,
                        "占用|正在进行",
                    ):
                        with web_app._supervised_hardware_lock("phone-a"):
                            self.fail("同一设备不能取得第二个硬件锁")

        self.assertFalse(first_controller.operation_lock.locked())
        self.assertFalse(second_controller.operation_lock.locked())

    def test_v3_confirm_api_rejects_cross_device_scope(self) -> None:
        orchestrator, _planner, _qwen, adapter = self._universal_api_orchestrator()
        session = orchestrator.start(
            session_id="api-cross-device",
            raw_goal="查看详情",
            device_id="phone-01",
            run_dir=web_app.WEB_OUTPUT_DIR / "api-cross-device",
        )
        web_app.runtime.agent_session_repository.add(session)
        scope = session.snapshot()["confirmation_scope"]
        scope["device_id"] = "phone-02"

        with (
            patch.object(web_app, "_require_supervised_device_ready"),
            patch.object(
                web_app.runtime,
                "universal_agent_orchestrator",
                orchestrator,
            ),
        ):
            response = self.client.post(
                f"/api/agent/generic-supervised/{session.session_id}/confirm",
                headers=self.headers,
                json={"confirmed": True, "confirmation": scope},
            )
        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(response.json()["detail"]["physical_actions"], 0)
        self.assertEqual(adapter.execute_calls, 0)

    def test_generic_scene_preview_is_read_only(self) -> None:
        scene = UIScene(
            app_id="calculator",
            screen_id="app_home",
            summary="计算器首页",
            stable=True,
            confidence=0.96,
            fingerprint="local-frame",
        )
        before_executions = len(web_app.runtime.controller.executions)
        before_failures = set(
            web_app.WEB_OUTPUT_DIR.glob("generic_scene_failure_*")
        )
        with (
            patch.object(
                web_app.runtime.vision_provider,
                "status",
                return_value={"configured": True},
            ),
            patch.object(
                web_app.runtime.generic_scene_observer,
                "observe",
                return_value=scene,
            ),
        ):
            response = self.client.post(
                "/api/agent/generic-scene",
                headers=self.headers,
                json={"goal": {"objective": "在计算器输入7"}},
            )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertFalse(payload["executed"])
        self.assertFalse(payload["physical_action_requested"])
        self.assertEqual(payload["scene"]["app_id"], "calculator")
        self.assertEqual(
            len(web_app.runtime.controller.executions),
            before_executions,
        )
        self.assertEqual(
            before_failures,
            set(web_app.WEB_OUTPUT_DIR.glob("generic_scene_failure_*")),
        )

    def test_generic_scene_preview_failure_persists_redacted_raw_response(self) -> None:
        observer = web_app.runtime.generic_scene_observer
        raw = (
            '{"api_key":"preview-secret",'
            '"image":"data:image/jpeg;base64,QUJD",'
            '"unexpected":true}'
        )
        before_failures = set(
            web_app.WEB_OUTPUT_DIR.glob("generic_scene_failure_*")
        )
        with (
            patch.object(
                web_app.runtime.vision_provider,
                "status",
                return_value={"configured": True},
            ),
            patch.object(observer, "last_raw_response", raw),
            patch.object(
                observer,
                "last_diagnostics",
                {
                    "failed_stage": "parsing_targeted_refinement",
                    "error_type": "schema_validation",
                },
            ),
            patch.object(
                observer,
                "observe",
                side_effect=VisionAgentError("目标精查结果不符合最小增量协议"),
            ),
        ):
            response = self.client.post(
                "/api/agent/generic-scene",
                headers=self.headers,
                json={"goal": {"objective": "只读核对当前输入区域"}},
            )

        self.assertEqual(422, response.status_code, response.text)
        new_failures = (
            set(web_app.WEB_OUTPUT_DIR.glob("generic_scene_failure_*"))
            - before_failures
        )
        self.assertEqual(1, len(new_failures))
        artifacts = list(next(iter(new_failures)).glob("*_qwen_failure.json"))
        self.assertEqual(1, len(artifacts))
        artifact = json.loads(artifacts[0].read_text(encoding="utf-8"))
        serialized = json.dumps(artifact, ensure_ascii=False)
        self.assertNotIn("preview-secret", serialized)
        self.assertNotIn("data:image", serialized)
        self.assertIn("[REDACTED_SECRET]", serialized)
        self.assertIn("[REDACTED_IMAGE_DATA_URL]", serialized)

    def test_generic_scene_preview_diagnostic_failure_preserves_original_422(self) -> None:
        with (
            patch.object(
                web_app.runtime.vision_provider,
                "status",
                return_value={"configured": True},
            ),
            patch.object(
                web_app.runtime.generic_scene_observer,
                "observe",
                side_effect=VisionAgentError("原始只读观察错误"),
            ),
            patch.object(
                web_app,
                "persist_observer_failure_diagnostic",
                side_effect=OSError("disk unavailable"),
            ),
        ):
            response = self.client.post(
                "/api/agent/generic-scene",
                headers=self.headers,
                json={"goal": {"objective": "只读核对当前输入区域"}},
            )

        self.assertEqual(422, response.status_code, response.text)
        self.assertIn("原始只读观察错误", response.text)

    def test_generic_supervised_api_starts_and_enters_safe_auto_loop(self) -> None:
        orchestrator, planner, qwen, adapter = self._universal_api_orchestrator()
        before_executions = len(web_app.runtime.controller.executions)
        with (
            patch.object(web_app, "_require_supervised_device_ready"),
            patch.object(
                web_app.runtime,
                "universal_agent_orchestrator",
                orchestrator,
            ),
            patch.object(
                orchestrator,
                "run_autonomous_safe_loop",
                return_value={
                    "physical_actions": 0,
                    "iterations": 0,
                    "status": "awaiting_confirmation",
                    "pause_reason": "测试保留待执行安全动作",
                },
            ) as auto_loop,
        ):
            started = self.client.post(
                "/api/agent/generic-supervised/start",
                headers=self.headers,
                json={"text": "查看当前页面的详情", "device_id": "phone-01"},
            )
            self.assertEqual(started.status_code, 200, started.text)
            payload = started.json()
            session_id = payload["session"]["session_id"]
            self.assertEqual(payload["physical_actions"], 0)
            self.assertTrue(payload["automatic_loop_enabled"])
            self.assertEqual(
                payload["session"]["status"], "awaiting_confirmation"
            )
            self.assertEqual(len(planner.plan_calls), 1)
            self.assertEqual(len(qwen.calls), 1)
            self.assertEqual(adapter.capture_calls, 1)
            self.assertEqual(adapter.execute_calls, 0)
            auto_loop.assert_called_once()

            rejected = self.client.post(
                f"/api/agent/generic-supervised/{session_id}/confirm",
                headers=self.headers,
                json={"confirmed": False},
            )
            self.assertEqual(rejected.status_code, 409, rejected.text)
            self.assertEqual(rejected.json()["detail"]["physical_actions"], 0)

            cancelled = self.client.post(
                f"/api/agent/generic-supervised/{session_id}/cancel",
                headers=self.headers,
                json={"device_id": "phone-01"},
            )
        self.assertEqual(cancelled.status_code, 200, cancelled.text)
        self.assertEqual(cancelled.json()["session"]["status"], "cancelled")
        self.assertEqual(
            len(web_app.runtime.controller.executions),
            before_executions,
        )

    def test_generic_supervised_evidence_failure_remains_http_409(self) -> None:
        orchestrator, _planner, _qwen, _adapter = self._universal_api_orchestrator()
        with (
            patch.object(web_app, "_require_supervised_device_ready"),
            patch.object(
                web_app.runtime,
                "universal_agent_orchestrator",
                orchestrator,
            ),
            patch.object(
                orchestrator,
                "start",
                side_effect=EvidenceStoreError("simulated evidence disk failure"),
            ),
        ):
            response = self.client.post(
                "/api/agent/generic-supervised/start",
                headers=self.headers,
                json={
                    "text": "查看当前页面的详情",
                    "device_id": "phone-01",
                    "auto_advance": False,
                },
            )

        self.assertEqual(409, response.status_code, response.text)
        self.assertEqual(0, response.json()["detail"]["physical_actions"])
        self.assertIn("simulated evidence disk failure", response.text)

    def test_new_generic_session_clears_stop_from_an_earlier_task(self) -> None:
        orchestrator, _planner, _qwen, _adapter = self._universal_api_orchestrator()
        controller = web_app.runtime.controller
        original_start = orchestrator.start
        stop_state_at_task_boundary = []

        def start_after_boundary(**kwargs):
            stop_state_at_task_boundary.append(controller.stop_event.is_set())
            return original_start(**kwargs)

        try:
            stopped = self.client.post(
                "/api/stop",
                headers=self.headers,
                json={"device_id": "phone-01"},
            )
            self.assertEqual(200, stopped.status_code, stopped.text)
            self.assertTrue(controller.stop_event.is_set())

            with (
                patch.object(web_app, "_require_supervised_device_ready"),
                patch.object(
                    web_app.runtime,
                    "universal_agent_orchestrator",
                    orchestrator,
                ),
                patch.object(orchestrator, "start", side_effect=start_after_boundary),
            ):
                started = self.client.post(
                    "/api/agent/generic-supervised/start",
                    headers=self.headers,
                    json={
                        "text": "查看当前页面的详情",
                        "device_id": "phone-01",
                        "auto_advance": False,
                    },
                )

            self.assertEqual(200, started.status_code, started.text)
            self.assertEqual([False], stop_state_at_task_boundary)
            self.assertFalse(controller.stop_event.is_set())
            session_id = started.json()["session"]["session_id"]
            self.client.post(
                f"/api/agent/generic-supervised/{session_id}/cancel",
                headers=self.headers,
                json={"device_id": "phone-01"},
            )
        finally:
            controller.stop_event.clear()

    def test_stop_requested_after_new_task_boundary_is_not_cleared(self) -> None:
        orchestrator, _planner, _qwen, _adapter = self._universal_api_orchestrator()
        controller = web_app.runtime.controller
        original_start = orchestrator.start
        controller.stop_event.clear()

        def start_then_request_stop(**kwargs):
            self.assertFalse(controller.stop_event.is_set())
            controller.request_stop()
            return original_start(**kwargs)

        try:
            with (
                patch.object(web_app, "_require_supervised_device_ready"),
                patch.object(
                    web_app.runtime,
                    "universal_agent_orchestrator",
                    orchestrator,
                ),
                patch.object(
                    orchestrator,
                    "start",
                    side_effect=start_then_request_stop,
                ),
            ):
                started = self.client.post(
                    "/api/agent/generic-supervised/start",
                    headers=self.headers,
                    json={
                        "text": "查看当前页面的详情",
                        "device_id": "phone-01",
                        "auto_advance": False,
                    },
                )

            self.assertEqual(200, started.status_code, started.text)
            self.assertTrue(controller.stop_event.is_set())
            session_id = started.json()["session"]["session_id"]
            self.client.post(
                f"/api/agent/generic-supervised/{session_id}/cancel",
                headers=self.headers,
                json={"device_id": "phone-01"},
            )
        finally:
            controller.stop_event.clear()

    def test_generic_supervised_api_can_start_in_explicit_single_step_mode(self) -> None:
        orchestrator, planner, qwen, adapter = self._universal_api_orchestrator()
        before_executions = len(web_app.runtime.controller.executions)
        with (
            patch.object(web_app, "_require_supervised_device_ready"),
            patch.object(
                web_app.runtime,
                "universal_agent_orchestrator",
                orchestrator,
            ),
            patch.object(
                orchestrator,
                "run_autonomous_safe_loop",
                side_effect=AssertionError("single-step start must not run safe loop"),
            ) as auto_loop,
        ):
            started = self.client.post(
                "/api/agent/generic-supervised/start",
                headers=self.headers,
                json={
                    "text": "查看当前页面的详情",
                    "device_id": "phone-01",
                    "auto_advance": False,
                },
            )
            self.assertEqual(started.status_code, 200, started.text)
            payload = started.json()
            session_id = payload["session"]["session_id"]
            self.assertEqual("generic_supervised_single_step", payload["mode"])
            self.assertEqual(0, payload["physical_actions"])
            self.assertFalse(payload["automatic_loop_enabled"])
            self.assertEqual("awaiting_confirmation", payload["session"]["status"])
            self.assertEqual(1, len(planner.plan_calls))
            self.assertEqual(1, len(qwen.calls))
            self.assertEqual(1, adapter.capture_calls)
            self.assertEqual(0, adapter.execute_calls)
            auto_loop.assert_not_called()

            cancelled = self.client.post(
                f"/api/agent/generic-supervised/{session_id}/cancel",
                headers=self.headers,
                json={"device_id": "phone-01"},
            )

        self.assertEqual(200, cancelled.status_code, cancelled.text)
        self.assertEqual("cancelled", cancelled.json()["session"]["status"])
        self.assertEqual(
            before_executions,
            len(web_app.runtime.controller.executions),
        )

    def test_same_device_second_generic_session_returns_409(self) -> None:
        orchestrator, planner, qwen, adapter = self._universal_api_orchestrator()
        with tempfile.TemporaryDirectory() as temp:
            with (
                patch.object(web_app, "WEB_OUTPUT_DIR", Path(temp)),
                patch.object(web_app, "_require_supervised_device_ready"),
                patch.object(
                    web_app.runtime,
                    "universal_agent_orchestrator",
                    orchestrator,
                ),
                patch.object(
                    orchestrator,
                    "run_autonomous_safe_loop",
                    return_value={
                        "physical_actions": 0,
                        "iterations": 0,
                        "status": "awaiting_confirmation",
                        "pause_reason": "测试保留活动会话",
                    },
                ),
            ):
                first = self.client.post(
                    "/api/agent/generic-supervised/start",
                    headers=self.headers,
                    json={"text": "查看详情", "device_id": "phone-01"},
                )
                directories_after_first = sorted(Path(temp).iterdir())
                second = self.client.post(
                    "/api/agent/generic-supervised/start",
                    headers=self.headers,
                    json={"text": "返回上一页", "device_id": "phone-01"},
                )
                directories_after_second = sorted(Path(temp).iterdir())
                session_id = first.json()["session"]["session_id"]
                self.client.post(
                    f"/api/agent/generic-supervised/{session_id}/cancel",
                    headers=self.headers,
                    json={"device_id": "phone-01"},
                )

        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(second.status_code, 409, second.text)
        self.assertEqual(second.json()["detail"]["physical_actions"], 0)
        self.assertEqual(directories_after_second, directories_after_first)
        self.assertEqual(len(directories_after_first), 1)
        self.assertEqual(len(planner.plan_calls), 1)
        self.assertEqual(len(qwen.calls), 1)
        self.assertEqual(adapter.capture_calls, 1)
        self.assertEqual(adapter.execute_calls, 0)

    def test_safe_auto_executes_one_verified_action_without_confirmation(self) -> None:
        orchestrator, _planner, _qwen, adapter = self._universal_api_orchestrator()
        with (
            patch.object(web_app, "_require_supervised_device_ready"),
            patch.object(
                web_app.runtime,
                "universal_agent_orchestrator",
                orchestrator,
            ),
            patch.object(
                orchestrator,
                "run_autonomous_safe_loop",
                return_value={
                    "physical_actions": 0,
                    "iterations": 0,
                    "status": "awaiting_confirmation",
                    "pause_reason": "测试把执行留给 /auto",
                },
            ),
        ):
            started = self.client.post(
                "/api/agent/generic-supervised/start",
                headers=self.headers,
                json={"text": "查看详情", "device_id": "phone-01"},
            )
            session_id = started.json()["session"]["session_id"]

        with (
            patch.object(web_app, "_require_supervised_device_ready"),
            patch.object(
                web_app.runtime,
                "universal_agent_orchestrator",
                orchestrator,
            ),
        ):
            automatic = self.client.post(
                f"/api/agent/generic-supervised/{session_id}/auto",
                headers=self.headers,
                json={
                    "device_id": "phone-01",
                    "confirmed": False,
                    "max_physical_actions": 1,
                    "max_iterations": 1,
                },
            )
            self.client.post(
                f"/api/agent/generic-supervised/{session_id}/cancel",
                headers=self.headers,
                json={"device_id": "phone-01"},
            )

        self.assertEqual(automatic.status_code, 200, automatic.text)
        self.assertEqual(automatic.json()["execution"]["physical_actions"], 1)
        self.assertEqual(automatic.json()["session"]["physical_actions"], 1)
        self.assertGreaterEqual(adapter.capture_calls, 1)
        self.assertEqual(adapter.execute_calls, 1)

    def test_preview_requires_and_uses_exact_registered_device(self) -> None:
        class PreviewController:
            def __init__(self, marker: bytes):
                self.marker = marker
                self.calls = []

            def capture_preview(self, quality=76):
                self.calls.append(quality)
                return b"jpeg-" + self.marker

        phone_a = PreviewController(b"phone-a")
        phone_b = PreviewController(b"phone-b")

        def controller_for_device(device_id):
            controllers = {"phone-a": phone_a, "phone-b": phone_b}
            if device_id not in controllers:
                raise web_app.UniversalAgentOrchestratorError(
                    f"device_id 未登记或未启用：{device_id}。"
                )
            return controllers[device_id]

        with patch.object(
            web_app.runtime,
            "controller_for_device",
            side_effect=controller_for_device,
        ):
            missing = self.client.get("/api/preview.jpg")
            unknown = self.client.get("/api/preview.jpg?device_id=phone-x")
            first = self.client.get("/api/preview.jpg?device_id=phone-a")
            second = self.client.get("/api/preview.jpg?device_id=phone-b")

        self.assertEqual(422, missing.status_code)
        self.assertEqual(404, unknown.status_code)
        self.assertEqual(b"jpeg-phone-a", first.content)
        self.assertEqual(b"jpeg-phone-b", second.content)
        self.assertEqual([72], phone_a.calls)
        self.assertEqual([72], phone_b.calls)

    def test_preview_uses_one_cached_frame_while_device_is_coordinated(self) -> None:
        class PreviewController:
            def __init__(self):
                self.calls = []

            def capture_preview(self, quality=76):
                self.calls.append(quality)
                return f"jpeg-live-{len(self.calls)}".encode("ascii")

        controller = PreviewController()
        device_id = "phone-camera-lease"
        with patch.object(
            web_app.runtime,
            "controller_for_device",
            return_value=controller,
        ):
            first = self.client.get(f"/api/preview.jpg?device_id={device_id}")
            coordination = (
                web_app.runtime.device_runtime_resources.coordination_lock(
                    device_id
                )
            )
            self.assertTrue(coordination.acquire(blocking=False))
            try:
                cached = [
                    self.client.get(f"/api/preview.jpg?device_id={device_id}")
                    for _index in range(3)
                ]
            finally:
                coordination.release()

        self.assertEqual(200, first.status_code)
        self.assertEqual("live", first.headers["X-Camera-Source"])
        self.assertEqual([72], controller.calls)
        self.assertTrue(all(item.content == first.content for item in cached))
        self.assertTrue(
            all(item.headers["X-Camera-Source"] == "cache" for item in cached)
        )

    def test_device_status_identifies_each_active_generic_session_device(self) -> None:
        def active_session(session_id: str, device_id: str):
            payload = {
                "session_id": session_id,
                "device_id": device_id,
                "status": "awaiting_confirmation",
                "step_number": 1,
                "proposal": {"status": "action"},
            }
            return SimpleNamespace(
                session_id=session_id,
                device_id=device_id,
                status="awaiting_confirmation",
                snapshot=lambda payload=payload: dict(payload),
            )

        web_app.runtime.agent_session_repository.add(
            active_session("session-phone-a", "phone-a")
        )
        web_app.runtime.agent_session_repository.add(
            active_session("session-phone-b", "phone-b")
        )

        active = self.client.get("/api/device").json()[
            "generic_supervised_execution"
        ]["active_sessions"]

        self.assertEqual(
            {
                (item["session_id"], item["device_id"])
                for item in active
            },
            {
                ("session-phone-a", "phone-a"),
                ("session-phone-b", "phone-b"),
            },
        )

if __name__ == "__main__":
    unittest.main(verbosity=2)
