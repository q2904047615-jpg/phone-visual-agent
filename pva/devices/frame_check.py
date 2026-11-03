"""Sanity check of a camera frame before it is shown to the model.

The phone sits in a dark enclosure, so its lit screen is the only bright area of
the camera image. A frame with no bright pixel at all means the camera is not
looking at the phone (or the screen is off). A screen that is no longer where it
was when the touch calibration was made means taps will land in the wrong place.

Dark app pages (a black video, "直播已结束") still show some white text, so they
are never treated as black. Page content can only darken parts of the screen, so
the position check looks for evidence content cannot produce: light outside the
expected screen area, or a bright screen pressed against a frame border that the
expected screen keeps clear of, or one larger than expected. A shift that only
moves an already clipped edge further out of the frame is not detectable.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

LIT_LEVEL = 60            # grey level that counts as lit screen
MIN_LIT_FRACTION = 0.001  # fewer lit pixels than this: nothing but the dark enclosure is visible
LINE_LIT_FRACTION = 0.3   # a row/column belongs to the screen when this much of it is lit
BRIGHT_PAGE = 0.8         # lit share of the bright box above which its edges are trusted
OUTSIDE_LIMIT = 0.02      # lit share allowed outside the expected screen area (plus tolerance)
CENTER_TOLERANCE = 0.05   # allowed shift of the screen edges, as a fraction of the frame
SIZE_TOLERANCE = 0.10     # allowed growth of screen width/height, as a fraction of the expected size

# The vendor window draws coordinate text along the top and buttons along the bottom.
_TOP_SKIP, _BOTTOM_SKIP = 0.04, 0.97

Region = tuple[float, float, float, float]   # x0, y0, x1, y1 as fractions of the frame


def _lit(image: Image.Image) -> np.ndarray:
    small = image.convert("L").resize((108, 192))
    return np.asarray(small, dtype=np.uint8) > LIT_LEVEL


def _band(lit: np.ndarray) -> np.ndarray:
    h = lit.shape[0]
    band = lit.copy()
    band[:int(h * _TOP_SKIP)] = False
    band[int(h * _BOTTOM_SKIP):] = False
    return band


def screen_region(image: Image.Image) -> Region | None:
    """Bounding box of the lit phone screen, or None when nothing is lit."""
    lit = _lit(image)
    h, w = lit.shape
    columns = np.where(_band(lit)[int(h * _TOP_SKIP):int(h * _BOTTOM_SKIP)].mean(axis=0) > LINE_LIT_FRACTION)[0]
    if len(columns) == 0:
        return None
    rows = np.where(lit[:int(h * _BOTTOM_SKIP), columns.min():columns.max() + 1].mean(axis=1) > LINE_LIT_FRACTION)[0]
    if len(rows) == 0:
        return None
    return (float(columns.min() / w), float(rows.min() / h), float((columns.max() + 1) / w), float((rows.max() + 1) / h))


def check_frame(image: Image.Image, expected: Region | None) -> str | None:
    """Return a problem description, or None when the frame looks usable."""
    lit = _band(_lit(image))
    h, w = lit.shape
    if lit.mean() < MIN_LIT_FRACTION:
        return "相机画面几乎全黑：相机可能没有对准手机，或手机已熄屏"
    if expected is None:
        return None
    ex0, ey0, ex1, ey1 = expected
    inside = np.zeros_like(lit)
    inside[max(0, int((ey0 - CENTER_TOLERANCE) * h)):int(np.ceil((ey1 + CENTER_TOLERANCE) * h)),
           max(0, int((ex0 - CENTER_TOLERANCE) * w)):int(np.ceil((ex1 + CENTER_TOLERANCE) * w))] = True
    outside = (lit & ~inside).sum() / max(1, lit.sum())
    region = screen_region(image)
    moved = outside > OUTSIDE_LIMIT
    if region is not None and not moved:
        x0, y0, x1, y1 = region
        box = lit[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
        if box.size and box.mean() >= BRIGHT_PAGE:
            # Content can shrink the bright box but cannot enlarge it or push it against
            # a frame border that the expected screen keeps clear of.
            moved = (_touches(region) - _touches(expected, margin=CENTER_TOLERANCE)
                     or (x1 - x0) > (1 + SIZE_TOLERANCE) * (ex1 - ex0)
                     or (y1 - y0) > (1 + SIZE_TOLERANCE) * (ey1 - ey0))
    if not moved:
        return None
    now = _fmt(region) if region else "（屏幕外出现亮光）"
    return (f"手机屏幕在画面中的位置变了：现在 {now}，配置的 screen_region 是 {_fmt(expected)}；"
            "相机或机械臂可能移位了，继续执行点击会偏")


def _touches(region: Region, margin: float = 0.01) -> set[str]:
    """Frame borders the region reaches (the bottom border is the vendor button strip)."""
    x0, y0, x1, y1 = region
    return {side for side, hit in (("left", x0 <= margin), ("top", y0 <= margin),
                                   ("right", x1 >= 1 - margin), ("bottom", y1 >= _BOTTOM_SKIP - margin)) if hit}


def _fmt(region: Region) -> str:
    return "[" + ", ".join(f"{v:.2f}" for v in region) + "]"
