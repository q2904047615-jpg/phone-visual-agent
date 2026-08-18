from __future__ import annotations

from PIL import Image


SYSTEM_NAVIGATION_PRIVACY_VIEW_VERSION = (
    "2026-08-19-system-navigation-privacy-view-v1"
)
SYSTEM_NAVIGATION_MASK_COLOR = (40, 40, 40)


def privacy_minimized_system_navigation_view(image: Image.Image) -> Image.Image:
    """Hide App content while retaining only device-edge/navigation structure."""

    source = image.convert("RGB")
    width, height = source.size
    if width < 32 or height < 32:
        raise ValueError("系统导航最小披露视图要求有效手机画布。")
    result = Image.new("RGB", source.size, SYSTEM_NAVIGATION_MASK_COLOR)
    side_width = max(2, min(width // 40, 20))
    bottom_height = max(24, min(height // 14, 120))
    result.paste(source.crop((0, 0, side_width, height)), (0, 0))
    result.paste(
        source.crop((width - side_width, 0, width, height)),
        (width - side_width, 0),
    )
    result.paste(
        source.crop((0, height - bottom_height, width, height)),
        (0, height - bottom_height),
    )
    return result
