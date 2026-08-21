"""Generic bridge to the seller-supplied robot control window.

This module contains only camera capture and primitive pointer/navigation
operations.  Task planning and App semantics belong to the canonical agent.
"""

from __future__ import annotations

import ctypes
import math
import sys
import time
from ctypes import wintypes
from pathlib import Path

import numpy as np
from PIL import Image, ImageGrab


# 新旧版本标题分别包含“智联新途机械臂控制端”和
# “智联新途AI机械臂控制端”，只匹配稳定前缀。
DEFAULT_WINDOW_TITLE = "智联新途"
BASELINE_CLIENT_WIDTH = 540
DEFAULT_CAMERA_HEIGHT = 960
MIN_AUTO_LAYOUT_WIDTH = 300
ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"


if sys.platform != "win32":
    raise SystemExit("这个 PoC 只能在 Windows 上运行。")


user32 = ctypes.windll.user32


def _enable_per_monitor_dpi_awareness() -> None:
    """Use physical pixels even when the seller window moves across monitors."""

    try:
        setter = user32.SetProcessDpiAwarenessContext
        setter.argtypes = [ctypes.c_void_p]
        setter.restype = wintypes.BOOL
        # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
        if setter(ctypes.c_void_p(-4)):
            return
    except (AttributeError, OSError, ValueError):
        pass
    try:
        user32.SetProcessDPIAware()
    except (AttributeError, OSError):
        pass


_enable_per_monitor_dpi_awareness()

EnumWindowsProc = ctypes.WINFUNCTYPE(
    wintypes.BOOL, wintypes.HWND, wintypes.LPARAM
)

WM_LBUTTONDOWN = 0x0201
WM_LBUTTONUP = 0x0202
MK_LBUTTON = 0x0001
SW_RESTORE = 9
GA_ROOT = 2
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
VK_ESCAPE = 0x1B
VK_HOME = 0x24
VK_DOWN = 0x28
VK_RETURN = 0x0D
VK_CONTROL = 0x11
VK_A = 0x41
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
INPUT_KEYBOARD = 1

# 卖家控制端底部控制条的固定横坐标。纵坐标使用“客户区底部向上偏移”
# 计算，以兼容窗口标题栏高度变化。
ACTION_BUTTON_X = 130
ACTION_DROPDOWN_X = 176
CLICK_COUNT_INPUT_X = 308
CONTROL_Y_FROM_BOTTOM = 18
BASELINE_TOOLBAR_HEIGHT = 50
SELLER_POSITION_OVERLAY_WIDTH = 180
SELLER_POSITION_OVERLAY_HEIGHT = 45
SELLER_POSITION_DIFF_CHANNEL_THRESHOLD = 12
SELLER_POSITION_CHANGED_PIXEL_MIN = 120
SELLER_POSITION_RETURN_PIXEL_MAX = 24
SELLER_POSITION_BARRIER_OFFSET = 3
SELLER_POSITION_BARRIER_TIMEOUT = 2.5
SELLER_TOUCH_DOWN_SETTLE_SECONDS = 0.45


class POINT(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]


# Explicit signatures are required on 64-bit Windows.  Without them ctypes
# may truncate HWND values returned by WindowFromPoint/GetAncestor.
user32.WindowFromPoint.argtypes = [POINT]
user32.WindowFromPoint.restype = wintypes.HWND
user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
user32.GetAncestor.restype = wintypes.HWND


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class INPUT_UNION(ctypes.Union):
    _fields_ = [
        ("mi", MOUSEINPUT),
        ("ki", KEYBDINPUT),
        ("hi", HARDWAREINPUT),
    ]


class INPUT(ctypes.Structure):
    _anonymous_ = ("data",)
    _fields_ = [
        ("type", wintypes.DWORD),
        ("data", INPUT_UNION),
    ]


class RECT(ctypes.Structure):
    _fields_ = [
        ("left", wintypes.LONG),
        ("top", wintypes.LONG),
        ("right", wintypes.LONG),
        ("bottom", wintypes.LONG),
    ]


class MONITORINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("rcMonitor", RECT),
        ("rcWork", RECT),
        ("dwFlags", wintypes.DWORD),
    ]


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_window(title_fragment: str) -> tuple[int, str]:
    matches: list[tuple[int, str]] = []

    @EnumWindowsProc
    def callback(hwnd: int, _lparam: int) -> bool:
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        title = buffer.value
        if title_fragment.lower() in title.lower():
            matches.append((hwnd, title))
        return True

    user32.EnumWindows(callback, 0)
    if not matches:
        raise RuntimeError(
            f"没有找到标题包含“{title_fragment}”的窗口。"
            "请先打开 main.exe，并保持控制端窗口可见。"
        )
    return matches[0]


def client_geometry(hwnd: int) -> tuple[int, int, int, int]:
    rect = RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        raise ctypes.WinError()
    top_left = POINT(0, 0)
    if not user32.ClientToScreen(hwnd, ctypes.byref(top_left)):
        raise ctypes.WinError()
    width = rect.right - rect.left
    height = rect.bottom - rect.top
    return top_left.x, top_left.y, width, height


def ensure_window_fully_visible(hwnd: int) -> None:
    """Resize/reposition the seller window so its camera and toolbar are usable."""

    user32.MonitorFromWindow.argtypes = [wintypes.HWND, wintypes.DWORD]
    user32.MonitorFromWindow.restype = wintypes.HMONITOR
    user32.GetMonitorInfoW.argtypes = [wintypes.HMONITOR, ctypes.c_void_p]
    user32.GetMonitorInfoW.restype = wintypes.BOOL
    window_rect = RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(window_rect)):
        raise ctypes.WinError()
    _left, _top, client_width, client_height = client_geometry(hwnd)
    required_height = seller_required_client_height(client_width, client_height)

    # Only auto-expand layouts that look like a real camera window.  Small
    # startup/error dialogs must continue to fail closed.
    portrait_candidate = client_height > client_width >= MIN_AUTO_LAYOUT_WIDTH
    desired_client_height = (
        max(client_height, required_height) if portrait_candidate else client_height
    )

    outer_width = window_rect.right - window_rect.left
    outer_height = window_rect.bottom - window_rect.top
    desired_outer_height = outer_height + desired_client_height - client_height

    MONITOR_DEFAULTTONEAREST = 2
    monitor = user32.MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST)
    info = MONITORINFO()
    info.cbSize = ctypes.sizeof(info)
    if not monitor or not user32.GetMonitorInfoW(monitor, ctypes.byref(info)):
        raise ctypes.WinError()
    work_width = info.rcWork.right - info.rcWork.left
    work_height = info.rcWork.bottom - info.rcWork.top
    if outer_width > work_width or desired_outer_height > work_height:
        raise RuntimeError(
            "控制端完整摄像区和操作栏大于当前显示器工作区，已拒绝执行。"
        )

    target_left = min(
        max(window_rect.left, info.rcWork.left),
        info.rcWork.right - outer_width,
    )
    target_top = min(
        max(window_rect.top, info.rcWork.top),
        info.rcWork.bottom - desired_outer_height,
    )
    if (
        target_left != window_rect.left
        or target_top != window_rect.top
        or desired_outer_height != outer_height
    ):
        SWP_NOZORDER = 0x0004
        SWP_NOACTIVATE = 0x0010
        if not user32.SetWindowPos(
            hwnd,
            0,
            target_left,
            target_top,
            outer_width,
            desired_outer_height,
            SWP_NOZORDER | SWP_NOACTIVATE,
        ):
            raise ctypes.WinError()
        time.sleep(0.2)


def window_dpi(hwnd: int) -> int:
    """Return the effective DPI for diagnostics without driving layout."""

    try:
        getter = user32.GetDpiForWindow
        getter.argtypes = [wintypes.HWND]
        getter.restype = wintypes.UINT
        value = int(getter(hwnd))
    except (AttributeError, OSError, ValueError):
        value = 96
    return value if value > 0 else 96


def seller_ui_scale(client_width: int) -> float:
    """Scale seller-control coordinates from its documented 540 px baseline."""

    if isinstance(client_width, bool) or int(client_width) <= 0:
        raise ValueError("控制端客户区宽度必须大于0。")
    return int(client_width) / BASELINE_CLIENT_WIDTH


def scale_seller_ui_value(value: int | float, client_width: int) -> int:
    if isinstance(value, bool) or float(value) < 0:
        raise ValueError("控制端基准坐标必须是非负数。")
    scaled = float(value) * seller_ui_scale(client_width)
    return int(math.floor(scaled + 0.5))


def seller_layout_scale(client_width: int, client_height: int) -> float:
    """Infer the seller UI scale for either phone orientation."""

    if isinstance(client_height, bool) or int(client_height) <= 0:
        raise ValueError("控制端客户区高度必须大于0。")
    if int(client_width) > int(client_height):
        return int(client_width) / DEFAULT_CAMERA_HEIGHT
    return seller_ui_scale(client_width)


def scale_seller_vertical_value(
    value: int | float,
    client_width: int,
    client_height: int,
) -> int:
    if isinstance(value, bool) or float(value) < 0:
        raise ValueError("控制端基准坐标必须是非负数。")
    scaled = float(value) * seller_layout_scale(client_width, client_height)
    return int(math.floor(scaled + 0.5))


def seller_required_client_height(
    client_width: int,
    client_height: int,
    baseline_height: int = DEFAULT_CAMERA_HEIGHT,
) -> int:
    """Return the smallest client height containing camera and both tool rows."""

    scale = seller_layout_scale(client_width, client_height)
    camera_baseline = (
        BASELINE_CLIENT_WIDTH if int(client_width) > int(client_height)
        else baseline_height
    )
    required = (camera_baseline + BASELINE_TOOLBAR_HEIGHT) * scale
    return int(math.floor(required + 0.5))


def seller_camera_height(
    client_width: int,
    client_height: int,
    baseline_height: int = DEFAULT_CAMERA_HEIGHT,
) -> int:
    """Map the 540x960 camera viewport to the current physical client size."""

    if isinstance(client_height, bool) or int(client_height) <= 0:
        raise ValueError("控制端客户区高度必须大于0。")
    if int(client_height) > int(client_width):
        scaled = scale_seller_ui_value(baseline_height, client_width)
        return min(int(client_height), max(1, scaled))

    return max(
        1,
        scale_seller_vertical_value(
            BASELINE_CLIENT_WIDTH,
            client_width,
            client_height,
        ),
    )


def seller_layout_has_full_camera(
    client_width: int,
    client_height: int,
    baseline_height: int = DEFAULT_CAMERA_HEIGHT,
) -> bool:
    """The real controller must include the full camera plus a bottom toolbar."""

    landscape = int(client_width) > int(client_height)
    if not landscape:
        return int(client_height) >= seller_required_client_height(
            client_width,
            client_height,
            baseline_height,
        )

    # A rotated 540x960 phone becomes 960x540.  The vendor window clips its
    # second toolbar row in this orientation; the camera and first-row controls
    # remain usable for the verified Back recovery action.  Small landscape
    # startup dialogs can share the title, so require a plausible camera size.
    scale = seller_layout_scale(client_width, client_height)
    expected_width = baseline_height * scale
    ratio = int(client_width) / int(client_height)
    return (
        int(client_width) >= 800
        and int(client_height) >= BASELINE_CLIENT_WIDTH * scale
        and int(client_width) >= expected_width * 0.95
        and 1.55 <= ratio <= 2.0
    )


def seller_control_point(
    client_width: int,
    client_height: int,
    baseline_x: int | float,
    baseline_y_from_bottom: int | float = CONTROL_Y_FROM_BOTTOM,
) -> tuple[int, int]:
    """Map one documented seller-toolbar point to the actual client pixels."""

    x = scale_seller_ui_value(baseline_x, client_width)
    bottom = scale_seller_vertical_value(
        baseline_y_from_bottom,
        client_width,
        client_height,
    )
    y = int(client_height) - bottom
    if not (0 <= x < int(client_width) and 0 <= y < int(client_height)):
        raise ValueError(
            f"缩放后的控制点 ({x}, {y}) 超出窗口客户区 "
            f"{client_width}×{client_height}。"
        )
    return x, y


def _root_window_at(screen_x: int, screen_y: int) -> int:
    candidate = user32.WindowFromPoint(POINT(screen_x, screen_y))
    if not candidate:
        return 0
    root = user32.GetAncestor(candidate, GA_ROOT)
    return int(root or candidate)


def _window_is_minimized(hwnd: int) -> bool:
    """Read window state without restoring or activating it."""

    try:
        return bool(user32.IsIconic(hwnd))
    except (AttributeError, OSError, ValueError):
        return False


def _validate_camera_region_unoccluded(
    hwnd: int,
    *,
    camera_height: int = DEFAULT_CAMERA_HEIGHT,
) -> None:
    """Prove the current desktop pixels belong to the seller controller."""

    left, top, width, height = client_geometry(hwnd)
    if not seller_layout_has_full_camera(width, height, camera_height):
        raise RuntimeError(
            "控制端窗口没有完整显示摄像区和底部操作栏，已拒绝执行；"
            "请恢复完整窗口或为卖家软件启用独立 DPI 兼容设置。"
        )
    visible_height = seller_camera_height(width, height, camera_height)
    if width <= 0 or visible_height <= 0:
        raise RuntimeError("控制端相机区域没有有效大小。")

    # Avoid borders and the seller toolbar.  Every point must belong to the
    # controller (or one of its child windows); one foreign owner means the
    # camera is still covered and the capture is unsafe.
    samples = (
        (0.20, 0.15),
        (0.50, 0.15),
        (0.80, 0.15),
        (0.20, 0.50),
        (0.50, 0.50),
        (0.80, 0.50),
        (0.20, 0.82),
        (0.50, 0.82),
        (0.80, 0.82),
    )
    foreign: list[tuple[int, int, int]] = []
    expected = int(hwnd)
    for x_ratio, y_ratio in samples:
        screen_x = left + min(width - 1, max(0, int(round((width - 1) * x_ratio))))
        screen_y = top + min(
            visible_height - 1,
            max(0, int(round((visible_height - 1) * y_ratio))),
        )
        owner = _root_window_at(screen_x, screen_y)
        if owner != expected:
            foreign.append((screen_x, screen_y, owner))
    if foreign:
        raise RuntimeError(
            "控制端相机区域仍被其他窗口遮挡，已拒绝把电脑桌面当成手机画面。"
        )


def ensure_camera_region_unoccluded(
    hwnd: int,
    *,
    camera_height: int = DEFAULT_CAMERA_HEIGHT,
) -> None:
    """Activate the seller controller for one explicitly requested task capture."""

    user32.ShowWindow(hwnd, SW_RESTORE)
    ensure_window_fully_visible(hwnd)
    user32.BringWindowToTop(hwnd)
    user32.SetForegroundWindow(hwnd)
    time.sleep(0.12)
    _validate_camera_region_unoccluded(hwnd, camera_height=camera_height)


def capture_client(hwnd: int) -> Image.Image:
    ensure_camera_region_unoccluded(hwnd)
    left, top, width, height = client_geometry(hwnd)
    if width <= 0 or height <= 0:
        raise RuntimeError("控制端窗口当前没有有效大小，可能已最小化。")
    return ImageGrab.grab(
        bbox=(left, top, left + width, top + height),
        all_screens=True,
    ).convert("RGB")


def capture_client_passive(hwnd: int) -> Image.Image:
    """Capture a visible preview without restoring, raising or focusing main.exe."""

    if _window_is_minimized(hwnd):
        raise RuntimeError("控制端已最小化，被动预览已暂停。")
    _validate_camera_region_unoccluded(hwnd)
    left, top, width, height = client_geometry(hwnd)
    if width <= 0 or height <= 0:
        raise RuntimeError("控制端窗口当前没有有效大小，被动预览已暂停。")
    return ImageGrab.grab(
        bbox=(left, top, left + width, top + height),
        all_screens=True,
    ).convert("RGB")


def camera_crop(image: Image.Image, camera_height: int) -> Image.Image:
    height = seller_camera_height(image.width, image.height, camera_height)
    return image.crop((0, 0, image.width, height))


def click_client_point(
    hwnd: int,
    x: int,
    y: int,
    countdown: int,
    hold_seconds: float,
    *,
    require_event_barrier: bool = False,
) -> dict[str, object] | None:
    _, _, width, height = client_geometry(hwnd)
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError(f"点击位置 ({x}, {y}) 超出窗口客户区 {width}×{height}。")
    if not (0.1 <= hold_seconds <= 2.0):
        raise ValueError("按住时间必须在 0.1～2.0 秒之间。")

    screen_point = POINT(x, y)
    if not user32.ClientToScreen(hwnd, ctypes.byref(screen_point)):
        raise ctypes.WinError()

    old_cursor = POINT()
    user32.GetCursorPos(ctypes.byref(old_cursor))
    user32.ShowWindow(hwnd, SW_RESTORE)
    user32.SetForegroundWindow(hwnd)
    if require_event_barrier:
        # Let Windows finish activating the seller window before establishing
        # the event-order baseline.  Per-key text input keeps its existing fast
        # path and is intentionally outside this single-action batch.
        time.sleep(0.1)

    for remaining in range(countdown, 0, -1):
        if user32.GetAsyncKeyState(VK_ESCAPE) & 0x8000:
            raise RuntimeError("用户按下 Esc，已取消执行。")
        print(f"{remaining} 秒后执行物理点击；按 Esc 取消……", flush=True)
        time.sleep(1)

    pressed = False
    changed_pixels = 0
    return_changed_pixels = 0
    barrier_seconds = 0.0
    try:
        user32.SetCursorPos(screen_point.x, screen_point.y)
        target_state = (
            _stable_seller_position_baseline(hwnd)
            if require_event_barrier
            else None
        )
        user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        pressed = True
        # 实机验证表明 0.08 秒过短：机械臂会下压，但手机可能收不到触摸。
        time.sleep(hold_seconds)
        user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        pressed = False

        if require_event_barrier:
            assert target_state is not None
            offset = (
                SELLER_POSITION_BARRIER_OFFSET
                if x + SELLER_POSITION_BARRIER_OFFSET < width
                else -SELLER_POSITION_BARRIER_OFFSET
            )
            barrier_started = time.monotonic()
            user32.SetCursorPos(screen_point.x + offset, screen_point.y)
            changed_pixels, _ = _wait_for_seller_position_state(
                hwnd,
                target_state,
                expect_changed=True,
            )
            offset_state = _capture_seller_position_overlay(hwnd)
            user32.SetCursorPos(screen_point.x, screen_point.y)
            return_changed_pixels, _ = _wait_for_seller_position_state(
                hwnd,
                offset_state,
                expect_changed=True,
            )
            barrier_seconds = time.monotonic() - barrier_started
        time.sleep(0.12)
    finally:
        if pressed:
            user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            time.sleep(0.12)
        user32.SetCursorPos(old_cursor.x, old_cursor.y)

    if not require_event_barrier:
        return None
    return {
        "version": "2026-08-19-seller-gui-click-barrier-v1",
        "channel": "left_button_atomic_click",
        "seller_event_barrier_confirmed": True,
        "round_trip_position_confirmed": True,
        "requested_mouse_hold_seconds": float(hold_seconds),
        "barrier_offset_pixels": abs(int(offset)),
        "changed_pixels": int(changed_pixels),
        "return_changed_pixels": int(return_changed_pixels),
        "barrier_elapsed_ms": round(barrier_seconds * 1000.0, 3),
        "mechanical_contact_ack": False,
    }


def _capture_seller_position_overlay(hwnd: int) -> np.ndarray:
    """Capture only the seller's PX/MM label without sampling the phone view."""

    left, top, width, height = client_geometry(hwnd)
    crop_width = min(SELLER_POSITION_OVERLAY_WIDTH, width)
    crop_height = min(SELLER_POSITION_OVERLAY_HEIGHT, height)
    if crop_width <= 0 or crop_height <= 0:
        raise RuntimeError("控制端坐标状态条当前不可见。")
    owner = _root_window_at(left + crop_width // 2, top + crop_height // 2)
    if owner != int(hwnd):
        raise RuntimeError("控制端坐标状态条被其他窗口遮挡。")
    image = ImageGrab.grab(
        bbox=(left, top, left + crop_width, top + crop_height),
        all_screens=True,
    ).convert("RGB")
    return np.asarray(image, dtype=np.int16).copy()


def _seller_position_changed_pixels(
    baseline: np.ndarray,
    current: np.ndarray,
) -> int:
    if baseline.shape != current.shape or baseline.ndim != 3:
        raise RuntimeError("控制端坐标状态条尺寸在动作期间发生变化。")
    delta = np.abs(current - baseline).max(axis=2)
    return int((delta > SELLER_POSITION_DIFF_CHANNEL_THRESHOLD).sum())


def _wait_for_seller_position_state(
    hwnd: int,
    baseline: np.ndarray,
    *,
    expect_changed: bool,
    timeout: float = SELLER_POSITION_BARRIER_TIMEOUT,
) -> tuple[int, float]:
    started = time.monotonic()
    deadline = started + max(0.1, float(timeout))
    last_count = 0
    while time.monotonic() < deadline:
        _check_escape("用户按下 Esc，已停止长按并准备释放触控笔。")
        current = _capture_seller_position_overlay(hwnd)
        last_count = _seller_position_changed_pixels(baseline, current)
        if expect_changed:
            if last_count >= SELLER_POSITION_CHANGED_PIXEL_MIN:
                return last_count, time.monotonic() - started
        elif last_count <= SELLER_POSITION_RETURN_PIXEL_MAX:
            return last_count, time.monotonic() - started
        time.sleep(0.02)
    state = "变化" if expect_changed else "恢复"
    raise RuntimeError(
        f"控制端没有在限定时间内确认坐标状态条{state}，已拒绝无确认长按。"
    )


def _stable_seller_position_baseline(hwnd: int) -> np.ndarray:
    deadline = time.monotonic() + 1.0
    previous = _capture_seller_position_overlay(hwnd)
    while time.monotonic() < deadline:
        time.sleep(0.03)
        current = _capture_seller_position_overlay(hwnd)
        if (
            _seller_position_changed_pixels(previous, current)
            <= SELLER_POSITION_RETURN_PIXEL_MAX
        ):
            return current
        previous = current
    raise RuntimeError("控制端坐标状态条在长按前不稳定。")


def long_press_client_point(
    hwnd: int,
    x: int,
    y: int,
    *,
    hold_seconds: float,
) -> dict[str, object]:
    """Hold contact only after the seller GUI processes a down/move barrier."""

    _, _, width, height = client_geometry(hwnd)
    camera_height = seller_camera_height(width, height)
    if not (0 <= x < width and 0 <= y < camera_height):
        raise ValueError(
            f"长按位置 ({x}, {y}) 超出摄像头客户区 {width}×{camera_height}。"
        )
    if not 0.5 <= float(hold_seconds) <= 2.0:
        raise ValueError("长按时间必须在0.5～2.0秒之间。")

    point = POINT(x, y)
    if not user32.ClientToScreen(hwnd, ctypes.byref(point)):
        raise ctypes.WinError()
    old_cursor = POINT()
    user32.GetCursorPos(ctypes.byref(old_cursor))
    user32.ShowWindow(hwnd, SW_RESTORE)
    user32.SetForegroundWindow(hwnd)
    time.sleep(0.1)
    user32.SetCursorPos(point.x, point.y)
    _check_escape("用户按下 Esc，已取消长按。")
    baseline = _stable_seller_position_baseline(hwnd)
    offset = (
        SELLER_POSITION_BARRIER_OFFSET
        if x + SELLER_POSITION_BARRIER_OFFSET < width
        else -SELLER_POSITION_BARRIER_OFFSET
    )
    pressed = False
    changed_pixels = 0
    return_changed_pixels = 0
    barrier_seconds = 0.0
    try:
        user32.mouse_event(MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
        pressed = True
        barrier_started = time.monotonic()
        user32.SetCursorPos(point.x + offset, point.y)
        changed_pixels, _ = _wait_for_seller_position_state(
            hwnd,
            baseline,
            expect_changed=True,
        )
        offset_state = _capture_seller_position_overlay(hwnd)
        user32.SetCursorPos(point.x, point.y)
        return_changed_pixels, _ = _wait_for_seller_position_state(
            hwnd,
            offset_state,
            expect_changed=True,
        )
        barrier_seconds = time.monotonic() - barrier_started
        # The seller GUI returning from its synchronous handler proves event
        # ordering, but its native Z command has no documented contact ACK.
        # Keep the pen stationary for a calibrated descent window before the
        # requested semantic hold interval starts.
        sleep_interruptible(SELLER_TOUCH_DOWN_SETTLE_SECONDS)
        sleep_interruptible(float(hold_seconds))
    finally:
        if pressed:
            user32.mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
            time.sleep(0.12)
        user32.SetCursorPos(old_cursor.x, old_cursor.y)
    return {
        "version": "2026-08-16-seller-gui-contact-barrier-v3",
        "channel": "right_button_stationary_touch",
        "seller_event_barrier_confirmed": True,
        "round_trip_position_confirmed": True,
        "hold_started_after_barrier": True,
        "requested_hold_seconds": float(hold_seconds),
        "barrier_offset_pixels": abs(int(offset)),
        "changed_pixels": int(changed_pixels),
        "return_changed_pixels": int(return_changed_pixels),
        "barrier_elapsed_ms": round(barrier_seconds * 1000.0, 3),
        "post_barrier_settle_seconds": SELLER_TOUCH_DOWN_SETTLE_SECONDS,
    }


def drag_client_path(
    hwnd: int,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    duration_seconds: float = 0.8,
    steps: int = 16,
) -> None:
    """Drive the seller UI's right-button touch-down/move/touch-up path."""

    _, _, width, height = client_geometry(hwnd)
    camera_height = seller_camera_height(width, height)
    for name, (x, y) in (("起点", start), ("终点", end)):
        if not (0 <= x < width and 0 <= y < camera_height):
            raise ValueError(
                f"拖动{name} ({x}, {y}) 超出摄像头客户区 "
                f"{width}×{camera_height}。"
            )
    if start == end:
        raise ValueError("拖动起点和终点不能相同。")
    if not 0.3 <= float(duration_seconds) <= 2.0:
        raise ValueError("拖动时间必须在0.3～2.0秒之间。")
    if isinstance(steps, bool) or not 4 <= int(steps) <= 60:
        raise ValueError("拖动插值步数必须在4～60之间。")

    start_point = POINT(*start)
    end_point = POINT(*end)
    if not user32.ClientToScreen(hwnd, ctypes.byref(start_point)):
        raise ctypes.WinError()
    if not user32.ClientToScreen(hwnd, ctypes.byref(end_point)):
        raise ctypes.WinError()

    old_cursor = POINT()
    user32.GetCursorPos(ctypes.byref(old_cursor))
    user32.ShowWindow(hwnd, SW_RESTORE)
    user32.SetForegroundWindow(hwnd)
    time.sleep(0.1)
    user32.SetCursorPos(start_point.x, start_point.y)
    _check_escape("用户按下 Esc，已取消拖动。")
    pressed = False
    try:
        user32.mouse_event(MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
        pressed = True
        time.sleep(0.12)
        step_count = int(steps)
        delay = max(0.01, (float(duration_seconds) - 0.12) / step_count)
        for index in range(1, step_count + 1):
            _check_escape("用户按下 Esc，已停止拖动。")
            ratio = index / step_count
            x = round(start_point.x + (end_point.x - start_point.x) * ratio)
            y = round(start_point.y + (end_point.y - start_point.y) * ratio)
            user32.SetCursorPos(x, y)
            time.sleep(delay)
    finally:
        if pressed:
            user32.mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
            time.sleep(0.12)
        user32.SetCursorPos(old_cursor.x, old_cursor.y)


def _check_escape(message: str = "用户按下 Esc，已停止执行。") -> None:
    if user32.GetAsyncKeyState(VK_ESCAPE) & 0x8000:
        raise RuntimeError(message)


def sleep_interruptible(seconds: float, poll_seconds: float = 0.1) -> None:
    """Sleep while keeping Esc responsive."""
    deadline = time.monotonic() + max(0.0, seconds)
    while True:
        _check_escape()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(poll_seconds, remaining))


def click_client_control(hwnd: int, x: int, y: int, hold: float = 0.08) -> None:
    """Click the seller software UI itself, not the camera/phone area."""
    _, _, width, height = client_geometry(hwnd)
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError(f"控制点 ({x}, {y}) 超出窗口客户区 {width}×{height}。")

    point = POINT(x, y)
    if not user32.ClientToScreen(hwnd, ctypes.byref(point)):
        raise ctypes.WinError()
    old_cursor = POINT()
    user32.GetCursorPos(ctypes.byref(old_cursor))
    user32.ShowWindow(hwnd, SW_RESTORE)
    user32.SetForegroundWindow(hwnd)
    time.sleep(0.1)
    user32.SetCursorPos(point.x, point.y)
    user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    time.sleep(hold)
    user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
    time.sleep(0.12)
    user32.SetCursorPos(old_cursor.x, old_cursor.y)


def press_virtual_key(key_code: int) -> None:
    user32.keybd_event(key_code, 0, 0, 0)
    time.sleep(0.05)
    user32.keybd_event(key_code, 0, KEYEVENTF_KEYUP, 0)
    time.sleep(0.08)


def configure_single_click_count(hwnd: int) -> None:
    """Force the seller software's 连点次数 field to one."""
    ensure_window_fully_visible(hwnd)
    _, _, width, height = client_geometry(hwnd)
    control_x, control_y = seller_control_point(
        width,
        height,
        CLICK_COUNT_INPUT_X,
    )
    click_client_control(hwnd, control_x, control_y)
    user32.keybd_event(VK_CONTROL, 0, 0, 0)
    press_virtual_key(VK_A)
    user32.keybd_event(VK_CONTROL, 0, KEYEVENTF_KEYUP, 0)
    type_unicode_digit("1")
    press_virtual_key(VK_RETURN)
    time.sleep(0.25)


def cursor_parking_client_point(width: int, height: int) -> tuple[int, int]:
    """Return a passive pointer position below the seller camera preview."""

    camera_height = seller_camera_height(width, height, DEFAULT_CAMERA_HEIGHT)
    if camera_height >= height:
        raise ValueError("卖家窗口没有可用于停放鼠标的相机外控制条。")
    return width // 2, camera_height + (height - camera_height) // 2


def cursor_parking_screen_point(
    window_rect: tuple[int, int, int, int],
    virtual_screen_rect: tuple[int, int, int, int],
) -> tuple[int, int] | None:
    """Choose a visible desktop corner that is definitely outside the seller window."""

    window_left, window_top, window_right, window_bottom = window_rect
    screen_left, screen_top, screen_right, screen_bottom = virtual_screen_rect
    if screen_right <= screen_left or screen_bottom <= screen_top:
        raise ValueError("虚拟桌面范围无效。")
    candidates = (
        (screen_left + 2, screen_top + 2),
        (screen_right - 3, screen_top + 2),
        (screen_left + 2, screen_bottom - 3),
        (screen_right - 3, screen_bottom - 3),
    )
    window_center = (
        (window_left + window_right) / 2,
        (window_top + window_bottom) / 2,
    )
    outside = [
        point
        for point in candidates
        if not (
            window_left <= point[0] < window_right
            and window_top <= point[1] < window_bottom
        )
    ]
    if not outside:
        return None
    return max(
        outside,
        key=lambda point: (point[0] - window_center[0]) ** 2
        + (point[1] - window_center[1]) ** 2,
    )


def move_cursor_outside_camera(hwnd: int) -> None:
    """Move the pointer outside the seller preview without clicking anything.

    The old title-bar parking point is still interpreted by seller v1.0.1018
    as a preview coordinate near y=30, leaving its opaque PX/MM tooltip over
    the top of the phone.  Prefer a point outside the whole seller window so a
    real mouse-leave event clears a tooltip left by physical key taps.  The
    documented bottom control strip remains a fallback when the seller window
    covers the complete virtual desktop.
    """

    left, top, width, height = client_geometry(hwnd)
    window = RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(window)):
        raise ctypes.WinError()
    SM_XVIRTUALSCREEN = 76
    SM_YVIRTUALSCREEN = 77
    SM_CXVIRTUALSCREEN = 78
    SM_CYVIRTUALSCREEN = 79
    screen_left = int(user32.GetSystemMetrics(SM_XVIRTUALSCREEN))
    screen_top = int(user32.GetSystemMetrics(SM_YVIRTUALSCREEN))
    screen_width = int(user32.GetSystemMetrics(SM_CXVIRTUALSCREEN))
    screen_height = int(user32.GetSystemMetrics(SM_CYVIRTUALSCREEN))
    screen_point = cursor_parking_screen_point(
        (window.left, window.top, window.right, window.bottom),
        (
            screen_left,
            screen_top,
            screen_left + screen_width,
            screen_top + screen_height,
        ),
    )
    if screen_point is None:
        client_x, client_y = cursor_parking_client_point(width, height)
        screen_point = (left + client_x, top + client_y)
    user32.SetCursorPos(*screen_point)
    time.sleep(0.25)


def configure_swipe(hwnd: int, direction: str) -> None:
    """Select one of the seller software's four documented swipe actions."""
    action_index = {
        "up": 0,
        "down": 1,
        "left": 2,
        "right": 3,
    }
    try:
        index = action_index[direction]
    except KeyError as exc:
        raise ValueError(f"Unsupported swipe direction: {direction}") from exc
    ensure_window_fully_visible(hwnd)
    _, _, width, height = client_geometry(hwnd)
    control_x, control_y = seller_control_point(
        width,
        height,
        ACTION_DROPDOWN_X,
    )
    click_client_control(hwnd, control_x, control_y)
    # 下拉选项顺序由卖家文档和实机确认：
    # 上划、下划、左划、右划、下拉、起点。
    press_virtual_key(VK_HOME)
    for _ in range(index):
        press_virtual_key(VK_DOWN)
    press_virtual_key(VK_RETURN)
    time.sleep(0.25)


def trigger_selected_action(hwnd: int) -> None:
    ensure_window_fully_visible(hwnd)
    _, _, width, height = client_geometry(hwnd)
    control_x, control_y = seller_control_point(width, height, ACTION_BUTTON_X)
    click_client_control(hwnd, control_x, control_y)
