from __future__ import annotations

import unittest

from PIL import Image, ImageDraw

from system_navigation_privacy import (
    SYSTEM_NAVIGATION_MASK_COLOR,
    privacy_minimized_system_navigation_view,
)


class SystemNavigationPrivacyViewTests(unittest.TestCase):
    def test_central_app_content_is_hidden_and_bottom_structure_is_retained(self):
        source = Image.new("RGB", (200, 400), "red")
        draw = ImageDraw.Draw(source)
        draw.rectangle((0, 0, 4, 399), fill="blue")
        draw.rectangle((195, 0, 199, 399), fill="blue")
        draw.rectangle((0, 370, 199, 399), fill="green")

        masked = privacy_minimized_system_navigation_view(source)

        self.assertEqual((255, 0, 0), source.getpixel((100, 200)))
        self.assertEqual(SYSTEM_NAVIGATION_MASK_COLOR, masked.getpixel((100, 200)))
        self.assertEqual((0, 128, 0), masked.getpixel((100, 390)))
        self.assertEqual((0, 0, 255), masked.getpixel((1, 200)))

    def test_masking_does_not_mutate_source(self):
        source = Image.new("RGB", (100, 200), "magenta")
        before = source.tobytes()

        masked = privacy_minimized_system_navigation_view(source)

        self.assertEqual(before, source.tobytes())
        self.assertNotEqual(before, masked.tobytes())


if __name__ == "__main__":
    unittest.main()
