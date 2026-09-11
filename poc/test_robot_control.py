from __future__ import annotations
from PIL import Image
from agent.infrastructure.robot_controller import controller_client_has_camera
import ctypes
from agent.infrastructure.robot_controller import oriented_navigation_ratio
from unittest.mock import patch
from agent.infrastructure import seller_window_adapter as robot_gui_poc
import unittest
from test_support.web_platform import (
    MockRobotController,
    RobotController,
    _BasePhysicalNavigationSafetyTests,
)


class PhysicalNavigationSafetyTests(_BasePhysicalNavigationSafetyTests):
    def test_live_preview_uses_passive_capture_without_active_capture_path(self):
        controller = RobotController(title="test")
        frame = Image.new("RGB", (540, 1038), "white")

        with (
            patch("agent.infrastructure.robot_controller.seller_gui.find_window", return_value=(123, "test")),
            patch(
                "agent.infrastructure.robot_controller.seller_gui.capture_client_passive",
                return_value=frame,
            ) as passive,
            patch("agent.infrastructure.robot_controller.seller_gui.capture_client") as active,
        ):
            payload = controller.capture_preview()

        self.assertTrue(payload.startswith(b"\xff\xd8"))
        passive.assert_called_once_with(123)
        active.assert_not_called()

    def test_passive_capture_never_invokes_window_activation(self):
        frame = Image.new("RGB", (540, 1038), "white")

        with (
            patch("agent.infrastructure.seller_window_adapter._window_is_minimized", return_value=False),
            patch("agent.infrastructure.seller_window_adapter._validate_camera_region_unoccluded") as validate,
            patch(
                "agent.infrastructure.seller_window_adapter.client_geometry",
                return_value=(10, 20, 540, 1038),
            ),
            patch("agent.infrastructure.seller_window_adapter.ImageGrab.grab", return_value=frame) as grab,
            patch("agent.infrastructure.seller_window_adapter.ensure_camera_region_unoccluded") as activate,
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
            patch("agent.infrastructure.seller_window_adapter._window_is_minimized", return_value=True),
            patch("agent.infrastructure.seller_window_adapter.ImageGrab.grab") as grab,
            patch("agent.infrastructure.seller_window_adapter.ensure_camera_region_unoccluded") as activate,
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
            patch("agent.infrastructure.robot_controller.seller_gui.find_window", return_value=(123, "test")),
            patch(
                "agent.infrastructure.robot_controller.seller_gui.temporarily_park_cursor_outside_camera",
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
            patch("agent.infrastructure.robot_controller.seller_gui.find_window", return_value=(123, "test")),
            patch.object(controller, "_capture_phone", return_value=frame),
            patch.object(controller, "_checkpoint"),
            patch("agent.infrastructure.robot_controller.seller_gui.configure_single_click_count") as configure,
            patch(
                "agent.infrastructure.robot_controller.seller_gui.click_client_point",
                return_value=self._click_barrier_receipt(),
            ) as click,
            patch("agent.infrastructure.robot_controller.seller_gui.clear_seller_camera_overlay"),
            patch(
                "agent.infrastructure.robot_controller.load_controller_config",
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
            return_dispatch_receipt=True,
        )
        receipt = controller.consume_last_click_receipt()
        self.assertTrue(receipt["input_events_dispatched"])
        self.assertFalse(receipt["mechanical_contact_ack"])
        self.assertIsNone(controller.consume_last_click_receipt())
        profile = controller.hardware_capability_profile()["actions"]["back"]
        self.assertEqual("input_events_dispatched", profile["transport_ack"])
        self.assertFalse(profile["mechanical_contact_ack"])

    def test_double_tap_sets_two_then_restores_single_click_count(self):
        controller = RobotController(
            title="test",
            verified_actions={"double_tap"},
        )
        frame = Image.new("RGB", (540, 960), "white")

        with (
            patch("agent.infrastructure.robot_controller.seller_gui.find_window", return_value=(123, "test")),
            patch.object(controller, "_capture_phone", return_value=frame),
            patch.object(controller, "_consume_physical_execution"),
            patch.object(controller, "_checkpoint"),
            patch(
                "agent.infrastructure.tap_calibration.corrected_grid_point",
                return_value=(500.0, 500.0),
            ),
            patch("agent.infrastructure.robot_controller.seller_gui.configure_click_count") as configure,
            patch(
                "agent.infrastructure.robot_controller.seller_gui.configure_single_click_count"
            ) as restore,
            patch(
                "agent.infrastructure.robot_controller.seller_gui.click_client_point",
                return_value=self._click_barrier_receipt(2),
            ) as click,
            patch("agent.infrastructure.robot_controller.seller_gui.clear_seller_camera_overlay") as clear,
            patch(
                "agent.infrastructure.robot_controller.load_controller_config",
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
            return_dispatch_receipt=True,
            click_count=2,
        )
        clear.assert_called_once_with(123)
        receipt = controller.consume_last_click_receipt()
        self.assertEqual(receipt["click_count"], 2)
        self.assertEqual(receipt["click_count_restored_to"], 1)

    def test_click_dispatch_does_not_sample_video_or_move_away_and_restores_cursor(self):
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
        with (
            patch("agent.infrastructure.seller_window_adapter.user32", fake),
            patch("agent.infrastructure.seller_window_adapter.client_geometry", return_value=(0, 0, 540, 1038)),
            patch("agent.infrastructure.seller_window_adapter.ImageGrab.grab", side_effect=AssertionError("must not sample video")) as grab,
            patch("agent.infrastructure.seller_window_adapter.time.sleep"),
        ):
            receipt = robot_gui_poc.click_client_point(
                123,
                270,
                937,
                countdown=0,
                hold_seconds=0.35,
                return_dispatch_receipt=True,
            )

        self.assertEqual(
            [robot_gui_poc.MOUSEEVENTF_LEFTDOWN, robot_gui_poc.MOUSEEVENTF_LEFTUP],
            fake.events,
        )
        self.assertEqual((7, 9), fake.positions[-1])
        self.assertNotIn((283, 957), fake.positions)
        self.assertIn((280, 957), fake.positions)
        grab.assert_not_called()
        self.assertTrue(receipt["input_events_dispatched"])
        self.assertNotIn("round_trip_position_confirmed", receipt)
        self.assertFalse(receipt["mechanical_contact_ack"])

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
            patch("agent.infrastructure.seller_window_adapter.user32", fake),
            patch("agent.infrastructure.seller_window_adapter.client_geometry", return_value=(0, 0, 540, 1038)),
            patch("agent.infrastructure.seller_window_adapter.time.sleep", side_effect=sleep_side_effect),
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

        with patch("agent.infrastructure.robot_controller.seller_gui.find_window") as find_window:
            self.assertFalse(hasattr(controller, "vision_type_text_with_layout"))
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
            patch("agent.infrastructure.robot_controller.seller_gui.find_window", return_value=(123, "test")),
            patch.object(controller, "_capture_phone", return_value=frame),
            patch.object(controller, "_consume_physical_execution"),
            patch.object(controller, "_checkpoint"),
            patch(
                "agent.infrastructure.tap_calibration.corrected_grid_point",
                return_value=(500.0, 500.0),
            ),
            patch(
                "agent.infrastructure.robot_controller.seller_gui.long_press_client_point",
                return_value={
                    "version": "2026-08-16-seller-gui-contact-barrier-v3",
                    "channel": "right_button_stationary_touch",
                    "input_events_dispatched": True,

                    "hold_started_after_barrier": True,
                    "requested_hold_seconds": 0.8,
                    "barrier_offset_pixels": 3,
                    "changed_pixels": 240,
                    "return_changed_pixels": 240,
                    "barrier_elapsed_ms": 35.0,
                    "post_barrier_settle_seconds": 0.45,
                },
            ) as long_press,
            patch("agent.infrastructure.robot_controller.seller_gui.click_client_point") as click,
            patch("agent.infrastructure.robot_controller.seller_gui.clear_seller_camera_overlay"),
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
        self.assertTrue(receipt["input_events_dispatched"])
        self.assertIsNone(controller.consume_last_long_press_receipt())

    def test_pixel_barrier_runtime_is_retired(self):
        for name in ("_stable_seller_position_baseline", "_wait_for_seller_position_state",
                     "_round_trip_position_barrier", "_capture_seller_position_overlay"):
            self.assertFalse(hasattr(robot_gui_poc, name), name)

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
            patch("agent.infrastructure.seller_window_adapter.time.sleep"),
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
            patch("agent.infrastructure.seller_window_adapter.ensure_window_fully_visible"),
            patch("agent.infrastructure.seller_window_adapter.client_geometry", return_value=(0, 0, 540, 1010)),
            patch("agent.infrastructure.seller_window_adapter.seller_control_point", return_value=(308, 992)),
            patch("agent.infrastructure.seller_window_adapter.click_client_control"),
            patch.object(robot_gui_poc.user32, "keybd_event"),
            patch("agent.infrastructure.seller_window_adapter.press_virtual_key") as press,
            patch("agent.infrastructure.seller_window_adapter.type_unicode_text") as type_text,
            patch("agent.infrastructure.seller_window_adapter.time.sleep"),
        ):
            robot_gui_poc.configure_single_click_count(123)

        type_text.assert_called_once_with("1")
        self.assertEqual(robot_gui_poc.VK_RETURN, press.call_args_list[-1].args[0])

    def test_every_unverified_physical_primitive_fails_before_hardware_access(self):
        controller = RobotController(title="test", verified_actions=set())

        operations = (
            ("点击", lambda: controller.vision_tap_relative(500, 500)),
            ("主页导航", controller.vision_android_home),
            ("滑动", controller.vision_swipe_up),
            ("返回", controller.vision_android_back),
            ("长按", lambda: controller.vision_long_press_relative(500, 500)),
            ("拖动", lambda: controller.vision_drag_relative(100, 100, 900, 900)),
        )
        with patch("agent.infrastructure.robot_controller.seller_gui.find_window") as find_window:
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

        with patch("agent.infrastructure.robot_controller.seller_gui.find_window") as find_window:
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
            patch("agent.infrastructure.robot_controller.seller_gui.find_window", return_value=(123, "test")),
            patch.object(controller, "_capture_phone", return_value=frame),
            patch.object(controller, "_checkpoint"),
            patch(
                "agent.infrastructure.tap_calibration.corrected_grid_point",
                side_effect=[(100.0, 200.0), (700.0, 800.0)],
            ),
            patch("agent.infrastructure.robot_controller.seller_gui.drag_client_path") as drag,
            patch("agent.infrastructure.robot_controller.seller_gui.clear_seller_camera_overlay") as clear_overlay,
        ):
            result = controller.vision_drag_relative(100, 200, 700, 800)

        self.assertEqual(result, ((54, 192), (377, 767)))
        drag.assert_called_once_with(123, (54, 192), (377, 767))
        clear_overlay.assert_called_once_with(123)

    def test_verified_swipe_uses_distinct_path_and_records_receipt(self):
        controller = RobotController(title="test", verified_actions={"swipe"})
        frame = Image.new("RGB", (540, 960), "white")
        receipt = {
            "version": "2026-09-03-seller-gui-swipe-path-v1",
            "channel": "right_button_swipe_path",
            "right_button_down_dispatched": True, "input_events_dispatched": True,
            "right_button_up_dispatched": True,
            "interpolation_steps_completed": 6,


            "touch_down_seconds": 0.35,
            "movement_seconds": 0.3,
            "step_count": 6,
            "client_start": [377, 480],
            "client_end": [54, 480],
            "mechanical_contact_ack": False,
        }

        with (
            patch("agent.infrastructure.robot_controller.seller_gui.find_window", return_value=(123, "test")),
            patch.object(controller, "_capture_phone", return_value=frame),
            patch.object(controller, "_checkpoint"),
            patch("agent.infrastructure.tap_calibration.corrected_grid_point",
                side_effect=[(700.0, 500.0), (100.0, 500.0)]),
            patch("agent.infrastructure.robot_controller.seller_gui.swipe_client_path",
                return_value=receipt) as swipe,
            patch("agent.infrastructure.robot_controller.seller_gui.drag_client_path") as drag,
            patch("agent.infrastructure.robot_controller.seller_gui.clear_seller_camera_overlay") as clear_overlay,
        ):
            result = controller.vision_swipe_relative(700, 500, 100, 500, "left")

        self.assertEqual(((377, 480), (54, 480)), result)
        swipe.assert_called_once_with(123, (377, 480), (54, 480), touch_down_seconds=0.35,
            movement_seconds=0.3, steps=6)
        drag.assert_not_called()
        stored = controller.consume_last_swipe_receipt()
        self.assertEqual("left", stored["requested_direction"])
        self.assertFalse(stored["mechanical_contact_ack"])
        self.assertIsNone(controller.consume_last_swipe_receipt())
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
            patch("agent.infrastructure.robot_controller.seller_gui.find_window") as find_window,
            patch("agent.infrastructure.robot_controller.seller_gui.drag_client_path") as drag,
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
            patch("agent.infrastructure.robot_controller.seller_gui.find_window", return_value=(123, "test")),
            patch.object(controller, "_capture_phone", return_value=frame),
            patch.object(controller, "_checkpoint"),
            patch(
                "agent.infrastructure.tap_calibration.reveal_system_navigation_path",
                return_value=derived,
            ) as derive,
            patch("agent.infrastructure.robot_controller.seller_gui.drag_client_path") as drag,
            patch("agent.infrastructure.robot_controller.seller_gui.clear_seller_camera_overlay") as clear_overlay,
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
            patch("agent.infrastructure.robot_controller.seller_gui.find_window", return_value=(123, "test")),
            patch.object(controller, "_capture_phone", return_value=frame),
            patch.object(controller, "_checkpoint"),
            patch(
                "agent.infrastructure.tap_calibration.reveal_system_navigation_path",
                return_value={"corrected_grid": [[92, 495], [333, 495]]},
            ),
            patch(
                "agent.infrastructure.robot_controller.seller_gui.drag_client_path",
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
            patch("agent.infrastructure.robot_controller.seller_gui.find_window", return_value=(123, "test")),
            patch.object(controller, "_capture_phone", return_value=frame),
            patch(
                "agent.infrastructure.tap_calibration.reveal_system_navigation_path",
                side_effect=RuntimeError("invalid calibration"),
            ),
            patch("agent.infrastructure.robot_controller.seller_gui.drag_client_path") as drag,
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


class LowLevelInputTests(unittest.TestCase):
    def test_input_structure_matches_windows_native_size(self) -> None:
        expected_size = 40 if ctypes.sizeof(ctypes.c_void_p) == 8 else 28
        self.assertEqual(ctypes.sizeof(robot_gui_poc.INPUT), expected_size)


if __name__ == "__main__":
    unittest.main()
