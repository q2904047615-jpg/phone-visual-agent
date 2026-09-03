from __future__ import annotations

import unittest
from unittest.mock import patch

from PIL import Image

from agent.infrastructure import seller_window_adapter as seller


class _FakeCursorUser32:
    def __init__(self, cursor: tuple[int, int]):
        self.cursor = cursor
        self.positions: list[tuple[int, int]] = []
        self.mouse_events: list[int] = []

    def GetCursorPos(self, point):
        point._obj.x, point._obj.y = self.cursor
        return 1

    def SetCursorPos(self, x, y):
        self.cursor = (int(x), int(y))
        self.positions.append(self.cursor)
        return 1

    def ClientToScreen(self, _hwnd, point):
        point._obj.x += 10
        point._obj.y += 20
        return 1

    def ShowWindow(self, _hwnd, _command):
        return 1

    def SetForegroundWindow(self, _hwnd):
        return 1

    def mouse_event(self, event, *_args):
        self.mouse_events.append(int(event))

    def GetWindowRect(self, _hwnd, rect):
        rect._obj.left = 0
        rect._obj.top = 0
        rect._obj.right = 830
        rect._obj.bottom = 1600
        return 1

    def GetSystemMetrics(self, metric):
        return {76: 0, 77: 0, 78: 2560, 79: 1600}[metric]


class SellerDpiScalingTests(unittest.TestCase):
    def test_swipe_path_has_distinct_timing_and_honest_receipt(self) -> None:
        fake = _FakeCursorUser32((400, 400))
        sleeps: list[float] = []
        with (
            patch.object(seller, "user32", fake),
            patch.object(seller, "client_geometry", return_value=(10, 20, 810, 1515)),
            patch.object(seller, "seller_camera_height", return_value=1440),
            patch.object(seller, "_check_escape"),
            patch.object(seller, "_stable_seller_position_baseline", return_value=object()),
            patch.object(seller, "_round_trip_position_barrier", return_value=(3, 240, 235, 0.035)) as barrier,
            patch.object(seller, "sleep_interruptible", side_effect=lambda seconds: sleeps.append(seconds)),
            patch.object(seller.time, "sleep"),
        ):
            receipt = seller.swipe_client_path(123, (198, 797), (17, 797),
                touch_down_seconds=0.35, movement_seconds=0.3, steps=6)

        self.assertEqual([seller.MOUSEEVENTF_RIGHTDOWN, seller.MOUSEEVENTF_RIGHTUP], fake.mouse_events)
        self.assertEqual((208, 817), fake.positions[0])
        self.assertEqual((27, 817), fake.positions[-2])
        self.assertEqual((400, 400), fake.positions[-1])
        self.assertEqual(0.35, sleeps[0])
        self.assertEqual(6, len(sleeps[1:]))
        for delay in sleeps[1:]:
            self.assertAlmostEqual(0.05, delay)
        barrier.assert_called_once()
        self.assertTrue(receipt["right_button_down_dispatched"])
        self.assertTrue(receipt["right_button_up_dispatched"])
        self.assertTrue(receipt["seller_position_barrier_confirmed"])
        self.assertEqual(6, receipt["interpolation_steps_completed"])
        self.assertFalse(receipt["mechanical_contact_ack"])

    def test_swipe_path_releases_right_button_when_movement_fails(self) -> None:
        fake = _FakeCursorUser32((400, 400))
        calls = 0

        def fail_during_movement(_seconds: float) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("movement failed")

        with (
            patch.object(seller, "user32", fake),
            patch.object(seller, "client_geometry", return_value=(10, 20, 810, 1515)),
            patch.object(seller, "seller_camera_height", return_value=1440),
            patch.object(seller, "_check_escape"),
            patch.object(seller, "sleep_interruptible", side_effect=fail_during_movement),
            patch.object(seller.time, "sleep"),
        ):
            with self.assertRaisesRegex(RuntimeError, "movement failed"):
                seller.swipe_client_path(123, (198, 797), (17, 797))

        self.assertEqual([seller.MOUSEEVENTF_RIGHTDOWN, seller.MOUSEEVENTF_RIGHTUP], fake.mouse_events)
        self.assertEqual((400, 400), fake.cursor)

    def test_documented_100_percent_geometry_is_unchanged(self) -> None:
        self.assertEqual(seller.seller_ui_scale(540), 1.0)
        self.assertEqual(seller.seller_camera_height(540, 1010), 960)
        self.assertEqual(seller.seller_control_point(540, 1010, 130), (130, 992))

    def test_125_percent_geometry_scales_camera_and_toolbar(self) -> None:
        self.assertEqual(seller.seller_ui_scale(675), 1.25)
        self.assertEqual(seller.seller_camera_height(675, 1263), 1200)
        self.assertEqual(seller.seller_control_point(675, 1263, 130), (163, 1240))
        self.assertEqual(seller.seller_control_point(675, 1263, 308), (385, 1240))

    def test_150_percent_geometry_scales_camera_and_toolbar(self) -> None:
        self.assertEqual(seller.seller_ui_scale(810), 1.5)
        self.assertEqual(seller.seller_camera_height(810, 1515), 1440)
        self.assertEqual(seller.seller_control_point(810, 1515, 176), (264, 1488))
        self.assertEqual(seller.seller_control_point(810, 1515, 520), (780, 1488))
        self.assertEqual(seller.cursor_parking_client_point(810, 1515), (405, 1477))

    def test_cursor_parking_prefers_a_desktop_corner_outside_seller_window(self) -> None:
        point = seller.cursor_parking_screen_point(
            (0, 0, 830, 1600),
            (0, 0, 2560, 1600),
        )
        self.assertEqual((2557, 2), point)

    def test_cursor_parking_returns_none_when_window_covers_virtual_desktop(self) -> None:
        self.assertIsNone(
            seller.cursor_parking_screen_point(
                (0, 0, 1920, 1080),
                (0, 0, 1920, 1080),
            )
        )

    def test_cursor_lease_restores_original_position_after_capture(self) -> None:
        fake = _FakeCursorUser32((400, 400))
        with (
            patch.object(seller, "user32", fake),
            patch.object(
                seller,
                "client_geometry",
                return_value=(0, 0, 810, 1515),
            ),
            patch.object(seller.time, "sleep"),
        ):
            with seller.temporarily_park_cursor_outside_camera(123) as moved:
                self.assertTrue(moved)
                self.assertEqual((2557, 2), fake.cursor)

        self.assertEqual((400, 400), fake.cursor)
        self.assertEqual([(2557, 2), (400, 400)], fake.positions)

    def test_cursor_lease_does_not_move_pointer_already_outside_preview(self) -> None:
        fake = _FakeCursorUser32((1200, 400))
        with (
            patch.object(seller, "user32", fake),
            patch.object(
                seller,
                "client_geometry",
                return_value=(0, 0, 810, 1515),
            ),
            patch.object(seller.time, "sleep") as sleep,
        ):
            with seller.temporarily_park_cursor_outside_camera(123) as moved:
                self.assertFalse(moved)

        self.assertEqual((1200, 400), fake.cursor)
        self.assertEqual([], fake.positions)
        sleep.assert_not_called()

    def test_cursor_lease_preserves_position_selected_by_user(self) -> None:
        fake = _FakeCursorUser32((400, 400))
        with (
            patch.object(seller, "user32", fake),
            patch.object(
                seller,
                "client_geometry",
                return_value=(0, 0, 810, 1515),
            ),
            patch.object(seller.time, "sleep"),
        ):
            with seller.temporarily_park_cursor_outside_camera(123):
                fake.cursor = (1200, 700)

        self.assertEqual((1200, 700), fake.cursor)
        self.assertEqual([(2557, 2)], fake.positions)

    def test_cursor_lease_restores_original_position_after_exception(self) -> None:
        fake = _FakeCursorUser32((400, 400))
        with (
            patch.object(seller, "user32", fake),
            patch.object(
                seller,
                "client_geometry",
                return_value=(0, 0, 810, 1515),
            ),
            patch.object(seller.time, "sleep"),
        ):
            with self.assertRaisesRegex(RuntimeError, "capture failed"):
                with seller.temporarily_park_cursor_outside_camera(123):
                    raise RuntimeError("capture failed")

        self.assertEqual((400, 400), fake.cursor)

    def test_150_percent_landscape_uses_height_for_vertical_scale(self) -> None:
        self.assertEqual(seller.seller_layout_scale(1440, 810), 1.5)
        self.assertEqual(seller.seller_control_point(1440, 885, 520), (1387, 858))
        self.assertEqual(seller.seller_required_client_height(1440, 810), 885)
        self.assertTrue(seller.seller_layout_has_full_camera(1440, 810))
        self.assertTrue(seller.seller_layout_has_full_camera(1440, 885))
        self.assertEqual(seller.seller_camera_height(1440, 885), 810)

    def test_small_landscape_dialog_is_not_a_camera(self) -> None:
        self.assertFalse(seller.seller_layout_has_full_camera(540, 304))

    def test_150_percent_window_must_not_be_clipped_by_a_1440p_desktop(self) -> None:
        self.assertFalse(seller.seller_layout_has_full_camera(810, 1392))
        self.assertTrue(seller.seller_layout_has_full_camera(810, 1515))

    def test_camera_crop_uses_current_client_width_instead_of_fixed_pixels(self) -> None:
        image = Image.new("RGB", (675, 1263), "white")
        cropped = seller.camera_crop(image, seller.DEFAULT_CAMERA_HEIGHT)
        self.assertEqual(cropped.size, (675, 1200))

    def test_invalid_or_clipped_geometry_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "宽度"):
            seller.seller_ui_scale(0)
        with self.assertRaisesRegex(ValueError, "超出"):
            seller.seller_control_point(100, 20, 540)
        with self.assertRaisesRegex(ValueError, "相机外控制条"):
            seller.cursor_parking_client_point(540, 960)
        with self.assertRaisesRegex(ValueError, "虚拟桌面"):
            seller.cursor_parking_screen_point((0, 0, 10, 10), (0, 0, 0, 0))


if __name__ == "__main__":
    unittest.main()
