from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class DouyinLocalSignals:
    """High-precision visual signals that do not depend on model wording."""

    live_preview_badge: tuple[int, int, int, int] | None = None
    live_room_close: tuple[int, int] | None = None

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        if self.live_preview_badge is not None:
            value["live_preview_badge"] = list(self.live_preview_badge)
        if self.live_room_close is not None:
            value["live_room_close"] = list(self.live_room_close)
        return value


def _connected_components(
    mask: np.ndarray,
    origin_x: int,
    origin_y: int,
    min_area: int = 20,
) -> list[tuple[int, int, int, int, int]]:
    """Return (area, left, top, right, bottom) for 8-connected components."""

    height, width = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    components: list[tuple[int, int, int, int, int]] = []
    for start_y in range(height):
        for start_x in range(width):
            if not mask[start_y, start_x] or seen[start_y, start_x]:
                continue
            stack = [(start_x, start_y)]
            seen[start_y, start_x] = True
            area = 0
            min_x = max_x = start_x
            min_y = max_y = start_y
            while stack:
                x, y = stack.pop()
                area += 1
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = x + dx, y + dy
                        if (
                            0 <= nx < width
                            and 0 <= ny < height
                            and mask[ny, nx]
                            and not seen[ny, nx]
                        ):
                            seen[ny, nx] = True
                            stack.append((nx, ny))
            if area >= min_area:
                components.append(
                    (
                        area,
                        origin_x + min_x,
                        origin_y + min_y,
                        origin_x + max_x + 1,
                        origin_y + max_y + 1,
                    )
                )
    return components


def detect_live_preview_badge(
    camera: Image.Image,
) -> tuple[int, int, int, int] | None:
    """Detect the magenta ``直播中`` badge used by a live-preview card."""

    rgb = np.asarray(camera.convert("RGB"), dtype=np.int16)
    height, width, _channels = rgb.shape
    x1 = int(width * 0.05)
    x2 = int(width * 0.36)
    y1 = int(height * 0.65)
    y2 = int(height * 0.83)
    roi = rgb[y1:y2, x1:x2]
    red = roi[:, :, 0]
    green = roi[:, :, 1]
    blue = roi[:, :, 2]
    magenta_mask = (
        (red > 140)
        & ((red - green) > 45)
        & ((red - blue) > 15)
        & (green < 170)
    )
    components = _connected_components(magenta_mask, x1, y1)
    candidates: list[tuple[int, int, int, int, int]] = []
    for component in components:
        area, left, top, right, bottom = component
        component_width = right - left
        component_height = bottom - top
        center_x = (left + right) / 2.0
        center_y = (top + bottom) / 2.0
        if (
            600 <= area <= 3000
            and 50 <= component_width <= 110
            and 18 <= component_height <= 45
            and width * 0.08 <= center_x <= width * 0.25
            and height * 0.68 <= center_y <= height * 0.81
        ):
            candidates.append(component)
    if not candidates:
        return None
    _area, left, top, right, bottom = max(candidates, key=lambda item: item[0])
    return left, top, right, bottom


def detect_live_room_close(camera: Image.Image) -> tuple[int, int] | None:
    """Detect the small white × at the top-right of a full live room."""

    rgb = np.asarray(camera.convert("RGB"), dtype=np.int16)
    height, width, _channels = rgb.shape
    x1 = int(width * 0.82)
    x2 = int(width * 0.94)
    y1 = int(height * 0.005)
    y2 = int(height * 0.055)
    roi = rgb[y1:y2, x1:x2]
    channel_min = roi.min(axis=2)
    channel_max = roi.max(axis=2)
    white_mask = (channel_min > 120) & ((channel_max - channel_min) < 55)
    components = _connected_components(white_mask, x1, y1)
    candidates: list[tuple[int, int, int, int, int]] = []
    for component in components:
        area, left, top, right, bottom = component
        component_width = right - left
        component_height = bottom - top
        center_x = (left + right) / 2.0
        center_y = (top + bottom) / 2.0
        local = white_mask[top - y1 : bottom - y1, left - x1 : right - x1]
        half_h = max(1, local.shape[0] // 2)
        half_w = max(1, local.shape[1] // 2)
        quadrant_counts = (
            int(local[:half_h, :half_w].sum()),
            int(local[:half_h, half_w:].sum()),
            int(local[half_h:, :half_w].sum()),
            int(local[half_h:, half_w:].sum()),
        )
        x_shape = min(quadrant_counts) >= max(3, round(area * 0.14))
        if (
            35 <= area <= 180
            and 8 <= component_width <= 20
            and 8 <= component_height <= 20
            and width * 0.85 <= center_x <= width * 0.92
            and height * 0.012 <= center_y <= height * 0.045
            and x_shape
        ):
            candidates.append(component)
    if not candidates:
        return None
    _area, left, top, right, bottom = max(candidates, key=lambda item: item[0])
    return (left + right) // 2, (top + bottom) // 2


def detect_douyin_local_signals(camera: Image.Image) -> DouyinLocalSignals:
    return DouyinLocalSignals(
        live_preview_badge=detect_live_preview_badge(camera),
        live_room_close=detect_live_room_close(camera),
    )
