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


if __name__ == "__main__":
    unittest.main()
