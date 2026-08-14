from __future__ import annotations

import unittest

from PIL import Image

import robot_gui_poc as seller


class SellerDpiScalingTests(unittest.TestCase):
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
