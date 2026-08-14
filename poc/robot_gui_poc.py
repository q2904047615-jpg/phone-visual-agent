from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import json
import math
import sys
import time
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageGrab, ImageTk


# 新旧版本标题分别包含“智联新途机械臂控制端”和
# “智联新途AI机械臂控制端”，只匹配稳定前缀。
DEFAULT_WINDOW_TITLE = "智联新途"
BASELINE_CLIENT_WIDTH = 540
DEFAULT_CAMERA_HEIGHT = 960
MIN_AUTO_LAYOUT_WIDTH = 300
DEFAULT_THRESHOLD = 0.82
ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
TEMPLATE_DIR = ROOT / "templates"


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
DEFAULT_DOUYIN_TEMPLATE = TEMPLATE_DIR / "douyin_home.png"


@dataclass(frozen=True)
class HeartDetection:
    state: str
    center: tuple[int, int] | None
    bbox: tuple[int, int, int, int] | None
    area: int
    white_candidates: tuple[tuple[int, int, int, int, int], ...] = ()
    red_candidates: tuple[tuple[int, int, int, int, int], ...] = ()


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
    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)


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


def _next_pow2(value: int) -> int:
    return 1 << max(0, value - 1).bit_length()


def _window_sums(array: np.ndarray, h: int, w: int) -> np.ndarray:
    integral = np.pad(array, ((1, 0), (1, 0)), mode="constant")
    integral = integral.cumsum(axis=0).cumsum(axis=1)
    return (
        integral[h:, w:]
        - integral[:-h, w:]
        - integral[h:, :-w]
        + integral[:-h, :-w]
    )


def match_template(
    source_image: Image.Image, template_image: Image.Image
) -> tuple[float, int, int]:
    """Return normalized cross-correlation score and top-left position."""
    source = np.asarray(source_image.convert("L"), dtype=np.float64)
    template = np.asarray(template_image.convert("L"), dtype=np.float64)
    ih, iw = source.shape
    th, tw = template.shape
    if th < 4 or tw < 4:
        raise ValueError("模板太小，请至少框选 4×4 像素。")
    if th > ih or tw > iw:
        raise ValueError("模板尺寸大于摄像头画面。")

    fft_h = _next_pow2(ih + th - 1)
    fft_w = _next_pow2(iw + tw - 1)
    source_fft = np.fft.rfft2(source, s=(fft_h, fft_w))
    template_fft = np.fft.rfft2(
        np.flip(template, axis=(0, 1)), s=(fft_h, fft_w)
    )
    correlation_full = np.fft.irfft2(
        source_fft * template_fft, s=(fft_h, fft_w)
    )
    correlation = correlation_full[th - 1 : ih, tw - 1 : iw]

    count = float(th * tw)
    template_mean = float(template.mean())
    template_energy = float(np.square(template - template_mean).sum())
    if template_energy < 1e-9:
        raise ValueError("模板几乎是纯色，请框选带文字或图案的区域。")

    sums = _window_sums(source, th, tw)
    square_sums = _window_sums(np.square(source), th, tw)
    numerator = correlation - sums * template_mean
    source_energy = square_sums - np.square(sums) / count
    denominator = np.sqrt(np.maximum(source_energy, 1e-9) * template_energy)
    scores = numerator / denominator
    scores = np.nan_to_num(scores, nan=-1.0, posinf=-1.0, neginf=-1.0)

    flat_index = int(np.argmax(scores))
    y, x = np.unravel_index(flat_index, scores.shape)
    return float(scores[y, x]), int(x), int(y)


def annotate_match(
    image: Image.Image,
    template: Image.Image,
    score: float,
    x: int,
    y: int,
    output: Path,
) -> None:
    result = image.copy()
    draw = ImageDraw.Draw(result)
    draw.rectangle(
        (x, y, x + template.width, y + template.height),
        outline=(255, 0, 0),
        width=4,
    )
    draw.text((x + 4, max(0, y - 18)), f"score={score:.3f}", fill=(255, 0, 0))
    result.save(output)


def select_template(
    image: Image.Image,
    output: Path,
    title: str = "拖动鼠标框选要自动识别的按钮或图标",
) -> None:
    import tkinter as tk
    from tkinter import messagebox

    screen_w = user32.GetSystemMetrics(0)
    screen_h = user32.GetSystemMetrics(1)
    scale = min(1.0, (screen_w * 0.85) / image.width, (screen_h * 0.78) / image.height)
    shown_w = max(1, int(image.width * scale))
    shown_h = max(1, int(image.height * scale))
    shown = image.resize((shown_w, shown_h), Image.Resampling.LANCZOS)

    root = tk.Tk()
    root.title(title)
    root.attributes("-topmost", True)
    info = tk.Label(
        root,
        text="按住左键拖出矩形；请只框住有特征的图标/文字，不要框太大。按 Esc 取消。",
        padx=10,
        pady=8,
    )
    info.pack()
    canvas = tk.Canvas(root, width=shown_w, height=shown_h, cursor="crosshair")
    canvas.pack()
    tk_image = ImageTk.PhotoImage(shown)
    canvas.create_image(0, 0, anchor="nw", image=tk_image)

    state: dict[str, object] = {"start": None, "rect": None, "saved": False}

    def on_press(event: tk.Event) -> None:
        state["start"] = (event.x, event.y)
        if state["rect"] is not None:
            canvas.delete(state["rect"])
        state["rect"] = canvas.create_rectangle(
            event.x, event.y, event.x, event.y, outline="red", width=2
        )

    def on_drag(event: tk.Event) -> None:
        if state["start"] is None or state["rect"] is None:
            return
        sx, sy = state["start"]
        canvas.coords(state["rect"], sx, sy, event.x, event.y)

    def on_release(event: tk.Event) -> None:
        if state["start"] is None:
            return
        sx, sy = state["start"]
        x1, x2 = sorted((sx, event.x))
        y1, y2 = sorted((sy, event.y))
        if x2 - x1 < 8 or y2 - y1 < 8:
            messagebox.showwarning("框选太小", "请重新框选一个清晰的按钮或图标。")
            return
        original_box = (
            max(0, int(x1 / scale)),
            max(0, int(y1 / scale)),
            min(image.width, int(math.ceil(x2 / scale))),
            min(image.height, int(math.ceil(y2 / scale))),
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        image.crop(original_box).save(output)
        state["saved"] = True
        root.destroy()

    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<ButtonRelease-1>", on_release)
    root.bind("<Escape>", lambda _event: root.destroy())
    root.mainloop()
    if not state["saved"]:
        raise RuntimeError("未保存模板。")


def click_client_point(
    hwnd: int,
    x: int,
    y: int,
    countdown: int,
    hold_seconds: float,
) -> None:
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

    for remaining in range(countdown, 0, -1):
        if user32.GetAsyncKeyState(VK_ESCAPE) & 0x8000:
            raise RuntimeError("用户按下 Esc，已取消执行。")
        print(f"{remaining} 秒后执行物理点击；按 Esc 取消……", flush=True)
        time.sleep(1)

    user32.SetCursorPos(screen_point.x, screen_point.y)
    user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    # 实机验证表明 0.08 秒过短：机械臂会下压，但手机可能收不到触摸。
    time.sleep(hold_seconds)
    user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
    time.sleep(0.12)
    user32.SetCursorPos(old_cursor.x, old_cursor.y)


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


def type_unicode_digit(digit: str) -> None:
    """Type one ASCII digit without depending on the active input method."""
    if len(digit) != 1 or digit not in "0123456789":
        raise ValueError("只允许输入一个 ASCII 数字。")

    events = (INPUT * 2)()
    for index, flags in enumerate(
        (KEYEVENTF_UNICODE, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP)
    ):
        events[index].type = INPUT_KEYBOARD
        events[index].ki = KEYBDINPUT(
            wVk=0,
            wScan=ord(digit),
            dwFlags=flags,
            time=0,
            dwExtraInfo=0,
        )
    sent = user32.SendInput(len(events), events, ctypes.sizeof(INPUT))
    if sent != len(events):
        raise ctypes.WinError()
    time.sleep(0.08)


def configure_single_click_count(hwnd: int) -> None:
    """Force the seller software's 连点次数 field to one.

    Two clicks on the Douyin heart toggle like on and then off, so this is a
    required safety check for the like workflow.
    """
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


def move_cursor_outside_camera(hwnd: int) -> None:
    """Move the pointer into the seller toolbar, outside the camera preview.

    The old title-bar parking point is still interpreted by seller v1.0.1018
    as a preview coordinate near y=30, leaving its opaque PX/MM tooltip over
    the top of the phone.  The documented bottom control strip is inside the
    same window but outside the camera crop, so moving there clears the tooltip
    without clicking or operating the phone.
    """

    left, top, width, height = client_geometry(hwnd)
    client_x, client_y = cursor_parking_client_point(width, height)
    user32.SetCursorPos(left + client_x, top + client_y)
    time.sleep(0.18)


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


def configure_up_swipe(hwnd: int) -> None:
    """Keep the original helper for the verified Douyin workflow."""
    configure_swipe(hwnd, "up")


def trigger_selected_action(hwnd: int) -> None:
    ensure_window_fully_visible(hwnd)
    _, _, width, height = client_geometry(hwnd)
    control_x, control_y = seller_control_point(width, height, ACTION_BUTTON_X)
    click_client_control(hwnd, control_x, control_y)


def image_change_score(before: Image.Image, after: Image.Image) -> float:
    """Return a 0..1 visual change score on a downsampled camera frame."""
    if before.size != after.size:
        after = after.resize(before.size, Image.Resampling.BILINEAR)
    width, height = before.size
    # 避开左侧持续存在的白色遮挡，并尽量覆盖视频主体、作者头像和说明文字。
    box = (
        int(width * 0.30),
        int(height * 0.10),
        int(width * 0.96),
        int(height * 0.84),
    )
    a = before.crop(box).convert("L").resize((96, 128), Image.Resampling.BILINEAR)
    b = after.crop(box).convert("L").resize((96, 128), Image.Resampling.BILINEAR)
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    return float(np.abs(aa - bb).mean() / 255.0)


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


def _heart_component_candidates(
    components: list[tuple[int, int, int, int, int]],
    camera_width: int,
    camera_height: int,
    *,
    min_area: int,
    min_center_ratio: float,
) -> list[tuple[int, int, int, int, int]]:
    """Filter bright/red components to the Douyin right-side heart geometry."""
    expected_x = camera_width * 0.865
    min_center_y = camera_height * min_center_ratio
    max_center_y = camera_height * 0.57
    candidates: list[tuple[int, int, int, int, int]] = []
    for component in components:
        area, left, top, right, bottom = component
        width = right - left
        height = bottom - top
        center_x = (left + right) / 2.0
        center_y = (top + bottom) / 2.0
        if (
            area >= min_area
            and 22 <= width <= 58
            and 20 <= height <= 52
            and abs(center_x - expected_x) <= camera_width * 0.065
            and min_center_y <= center_y <= max_center_y
        ):
            candidates.append(component)
    return sorted(candidates, key=lambda item: item[0], reverse=True)


def detect_douyin_heart(camera: Image.Image) -> HeartDetection:
    """Detect the normal white heart or persistent pink/red liked heart.

    The search is deliberately limited to Douyin's right-side action rail.
    This avoids mistaking video content for a UI icon and also treats live
    cards without the normal heart as "missing".
    """
    rgb = np.asarray(camera.convert("RGB"), dtype=np.int16)
    height, width, _channels = rgb.shape
    x1 = max(0, int(width * 0.77))
    x2 = min(width, int(width * 0.94))
    y1 = max(0, int(height * 0.39))
    y2 = min(height, int(height * 0.59))
    roi = rgb[y1:y2, x1:x2]
    if roi.size == 0:
        return HeartDetection("missing", None, None, 0)

    channel_min = roi.min(axis=2)
    channel_max = roi.max(axis=2)
    white_mask = (channel_min > 155) & ((channel_max - channel_min) < 105)
    red_mask = (
        (roi[:, :, 0] > 165)
        & (roi[:, :, 0] > roi[:, :, 1] * 1.35)
        & (roi[:, :, 0] > roi[:, :, 2] * 1.10)
        & (roi[:, :, 1] < 175)
    )

    white_components = _connected_components(white_mask, x1, y1)
    red_components = _connected_components(red_mask, x1, y1)
    white_candidates = _heart_component_candidates(
        white_components,
        width,
        height,
        min_area=250,
        min_center_ratio=0.44,
    )
    red_candidates = _heart_component_candidates(
        red_components,
        width,
        height,
        min_area=300,
        # 关注按钮通常在 45% 高度附近；红心应明显位于它的下方。
        # 收紧下界，避免把头像旁的红色“+”误判成已点赞爱心。
        min_center_ratio=0.465,
    )

    # 红色“关注 +”位于爱心正上方约 5.4% 屏幕高度。亮色视频中，
    # 白心可能与背景连成一片；此时用关注按钮作为爱心定位锚点。
    expected_x = width * 0.865
    follow_candidates: list[tuple[int, int, int, int, int]] = []
    for component in red_components:
        area, left, top, right, bottom = component
        component_width = right - left
        component_height = bottom - top
        center_x = (left + right) / 2.0
        center_y = (top + bottom) / 2.0
        if (
            area >= 120
            and 14 <= component_width <= 42
            and 14 <= component_height <= 42
            and abs(center_x - expected_x) <= width * 0.065
            and height * 0.38 <= center_y <= height * 0.48
        ):
            follow_candidates.append(component)

    if follow_candidates:
        # 同一位置可能同时出现头像红点、红色“+”等多个组件。
        # 关注按钮是其中位置最低的一个，选最高的组件会把真正的
        # 关注按钮误当成下方红心。
        follow = max(
            follow_candidates,
            key=lambda item: (item[2] + item[4]) / 2.0,
        )
        _area, left, top, right, bottom = follow
        inferred_x = (left + right) // 2
        follow_y = (top + bottom) // 2
        inferred_y = follow_y + round(height * 0.054)

        white_heart_nearby = [
            component
            for component in white_candidates
            if abs(((component[2] + component[4]) / 2.0) - inferred_y)
            <= height * 0.025
        ]
        # 画面内容偶尔会在爱心旁出现粉色小块。只要同一位置存在清晰
        # 白心，就应优先判定为未点赞，不能让附近粉色内容覆盖它。
        if white_heart_nearby:
            area, left, top, right, bottom = max(
                white_heart_nearby,
                key=lambda item: item[0],
            )
            return HeartDetection(
                "unliked",
                ((left + right) // 2, (top + bottom) // 2),
                (left, top, right, bottom),
                area,
                tuple(white_candidates),
                tuple(red_candidates),
            )

        red_heart_nearby = [
            component
            for component in red_components
            if (
                component[0] >= 150
                and abs(((component[1] + component[3]) / 2.0) - inferred_x)
                <= width * 0.030
                and abs(((component[2] + component[4]) / 2.0) - inferred_y)
                <= height * 0.025
                and ((component[2] + component[4]) / 2.0) > follow_y + 20
            )
        ]
        if red_heart_nearby:
            area, left, top, right, bottom = max(
                red_heart_nearby,
                key=lambda item: item[0],
            )
            return HeartDetection(
                "liked",
                ((left + right) // 2, (top + bottom) // 2),
                (left, top, right, bottom),
                area,
                tuple(white_candidates),
                tuple(red_candidates),
            )

        inferred_box = (
            inferred_x - 20,
            inferred_y - 20,
            inferred_x + 20,
            inferred_y + 20,
        )
        return HeartDetection(
            "unliked",
            (inferred_x, inferred_y),
            inferred_box,
            0,
            tuple(white_candidates),
            tuple(red_candidates),
        )

    # 没有关注按钮的页面或离线合成图，继续使用独立爱心组件。
    if white_candidates:
        area, left, top, right, bottom = white_candidates[0]
        return HeartDetection(
            "unliked",
            ((left + right) // 2, (top + bottom) // 2),
            (left, top, right, bottom),
            area,
            tuple(white_candidates),
            tuple(red_candidates),
        )
    if red_candidates:
        # 已点赞页可能同时存在关注按钮和红心；红心始终位于关注按钮下方。
        area, left, top, right, bottom = max(
            red_candidates,
            key=lambda item: (item[2] + item[4]) / 2.0,
        )
        return HeartDetection(
            "liked",
            ((left + right) // 2, (top + bottom) // 2),
            (left, top, right, bottom),
            area,
            tuple(white_candidates),
            tuple(red_candidates),
        )
    return HeartDetection(
        "missing",
        None,
        None,
        0,
        tuple(white_candidates),
        tuple(red_candidates),
    )


def liked_transition_near_click(
    before: Image.Image,
    after: Image.Image,
    center: tuple[int, int],
) -> tuple[bool, dict[str, float | int]]:
    """Verify that the clicked white-heart patch became distinctly red.

    Douyin briefly draws a translucent red ripple after a successful like.
    During that animation the red heart can be split into several components,
    so a single-frame connected-component test may return ``missing`` even
    though the click succeeded. Comparing the exact same patch before and
    after is both more tolerant and safer than accepting red pixels elsewhere
    in the video.
    """
    before_rgb = np.asarray(before.convert("RGB"), dtype=np.int16)
    after_rgb = np.asarray(after.convert("RGB"), dtype=np.int16)
    if before_rgb.shape != after_rgb.shape:
        return False, {"before_red": 0, "after_red": 0, "red_gain": 0}

    height, width, _channels = after_rgb.shape
    center_x, center_y = center
    radius = 24
    left = max(0, center_x - radius)
    right = min(width, center_x + radius + 1)
    top = max(0, center_y - radius)
    bottom = min(height, center_y + radius + 1)
    if left >= right or top >= bottom:
        return False, {"before_red": 0, "after_red": 0, "red_gain": 0}

    def red_metrics(image: np.ndarray) -> tuple[int, float]:
        patch = image[top:bottom, left:right]
        red = patch[:, :, 0]
        green = patch[:, :, 1]
        blue = patch[:, :, 2]
        # The camera makes the animation pale pink, hence channel-difference
        # thresholds are more reliable here than a strict saturation ratio.
        mask = (
            (red > 130)
            & ((red - green) > 20)
            & ((red - blue) > 5)
        )
        redness = np.maximum(red - green, 0).mean()
        return int(mask.sum()), float(redness)

    before_red, before_redness = red_metrics(before_rgb)
    after_red, after_redness = red_metrics(after_rgb)
    red_gain = after_red - before_red
    redness_gain = after_redness - before_redness
    verified = (
        after_red >= 150
        and red_gain >= 100
        and after_red >= before_red * 1.6
        and redness_gain >= 8.0
    )
    return verified, {
        "before_red": before_red,
        "after_red": after_red,
        "red_gain": red_gain,
        "before_redness": round(before_redness, 3),
        "after_redness": round(after_redness, 3),
        "redness_gain": round(redness_gain, 3),
    }


def wait_for_liked_heart(
    hwnd: int,
    camera_height: int,
    before: Image.Image,
    center: tuple[int, int],
    initial_wait: float,
    timeout: float,
) -> tuple[Image.Image, HeartDetection, bool, str, float, dict[str, float | int]]:
    """Poll several frames until a persistent heart or red transition appears."""
    started = time.monotonic()
    deadline = started + timeout
    sleep_interruptible(min(initial_wait, timeout))
    last_camera = before
    last_detection = detect_douyin_heart(before)
    last_metrics: dict[str, float | int] = {}

    while True:
        last_camera = camera_crop(capture_client(hwnd), camera_height)
        last_detection = detect_douyin_heart(last_camera)
        if last_detection.state == "liked":
            return (
                last_camera,
                last_detection,
                True,
                "persistent_red_heart",
                time.monotonic() - started,
                last_metrics,
            )

        changed_to_red, last_metrics = liked_transition_near_click(
            before,
            last_camera,
            center,
        )
        if changed_to_red:
            center_x, center_y = center
            transition_detection = HeartDetection(
                "liked",
                center,
                (
                    center_x - 24,
                    center_y - 24,
                    center_x + 25,
                    center_y + 25,
                ),
                int(last_metrics["after_red"]),
                last_detection.white_candidates,
                last_detection.red_candidates,
            )
            return (
                last_camera,
                transition_detection,
                True,
                "before_after_red_transition",
                time.monotonic() - started,
                last_metrics,
            )

        now = time.monotonic()
        if now >= deadline:
            return (
                last_camera,
                last_detection,
                False,
                "timeout",
                now - started,
                last_metrics,
            )
        sleep_interruptible(min(0.25, deadline - now))


def annotate_heart_detection(
    image: Image.Image,
    detection: HeartDetection,
    output: Path,
) -> None:
    result = image.copy()
    draw = ImageDraw.Draw(result)
    width, height = result.size
    roi_box = (
        int(width * 0.77),
        int(height * 0.39),
        int(width * 0.94),
        int(height * 0.59),
    )
    draw.rectangle(roi_box, outline=(255, 210, 0), width=2)
    color = {
        "liked": (255, 30, 90),
        "unliked": (40, 210, 80),
        "missing": (255, 170, 0),
    }[detection.state]
    if detection.bbox is not None:
        draw.rectangle(detection.bbox, outline=color, width=4)
    draw.text(
        (roi_box[0], max(0, roi_box[1] - 20)),
        f"heart={detection.state} area={detection.area}",
        fill=color,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    result.save(output)


def douyin_home_score(
    camera: Image.Image,
    template_path: Path,
) -> tuple[float, int, int, Image.Image]:
    if not template_path.exists():
        raise FileNotFoundError(
            f"缺少抖音首页模板：{template_path}。"
            "请先在抖音首页运行 douyin-calibrate。"
        )
    template = Image.open(template_path).convert("RGB")
    # 只在底部左侧导航栏搜索。“朋友”等文字和“首页”外观相似，
    # 在整张画面搜索会偶尔匹配到第二个导航项。
    search_left = 0
    search_top = int(camera.height * 0.82)
    search_right = int(camera.width * 0.30)
    search_bottom = camera.height
    search_region = camera.crop(
        (search_left, search_top, search_right, search_bottom)
    )
    score, local_x, local_y = match_template(search_region, template)
    return score, search_left + local_x, search_top + local_y, template


def valid_douyin_home_marker(
    camera: Image.Image,
    template: Image.Image,
    score: float,
    x: int,
    y: int,
    threshold: float,
) -> bool:
    """Require the matched 首页 marker to be in Douyin's bottom-left nav."""
    center_x = x + template.width / 2.0
    center_y = y + template.height / 2.0
    return (
        score >= threshold
        and center_x <= camera.width * 0.28
        and center_y >= camera.height * 0.82
    )


def camera_ui_sharpness(camera: Image.Image) -> float:
    """Estimate whether Douyin's action rail and bottom controls are sharp.

    Video content itself may legitimately be soft, so only the right-side UI
    rail and bottom control band are measured. A swipe transition produces
    strong ghosting in both regions and must never be used as a click frame.
    """
    gray = np.asarray(camera.convert("L"), dtype=np.float32)
    height, width = gray.shape
    regions = (
        gray[int(height * 0.35):int(height * 0.75), int(width * 0.75):],
        gray[int(height * 0.80):, :],
    )
    scores: list[float] = []
    for region in regions:
        if region.shape[0] < 3 or region.shape[1] < 3:
            scores.append(0.0)
            continue
        laplacian = (
            region[1:-1, 2:]
            + region[1:-1, :-2]
            + region[2:, 1:-1]
            + region[:-2, 1:-1]
            - 4 * region[1:-1, 1:-1]
        )
        scores.append(float(laplacian.var()))
    return min(scores)


def detect_douyin_live_close(
    camera: Image.Image,
) -> tuple[int, int] | None:
    """Detect the small white × used to leave a full-screen live room."""
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
        local = white_mask[
            top - y1 : bottom - y1,
            left - x1 : right - x1,
        ]
        half_h = max(1, local.shape[0] // 2)
        half_w = max(1, local.shape[1] // 2)
        quadrant_counts = (
            int(local[:half_h, :half_w].sum()),
            int(local[:half_h, half_w:].sum()),
            int(local[half_h:, :half_w].sum()),
            int(local[half_h:, half_w:].sum()),
        )
        # “×”的两条对角线会同时经过四个象限；推荐页的搜索放大镜
        # 主要只有圆环和右下手柄，至少有一个象限明显为空。
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
    _area, left, top, right, bottom = max(
        candidates,
        key=lambda item: item[0],
    )
    return ((left + right) // 2, (top + bottom) // 2)


def detect_douyin_live_preview_badge(
    camera: Image.Image,
) -> tuple[int, int, int, int] | None:
    """Detect the pink 直播中 badge on a live-preview card in 推荐."""
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
            # 实机“直播中”徽标一直贴近左侧（中心约为画面宽度 16%）。
            # 更靠右的粉色块通常来自视频字幕或商品贴纸。
            and width * 0.08 <= center_x <= width * 0.25
            and height * 0.68 <= center_y <= height * 0.81
        ):
            candidates.append(component)
    if not candidates:
        return None
    _area, left, top, right, bottom = max(
        candidates,
        key=lambda item: item[0],
    )
    return (left, top, right, bottom)


def capture_ready_douyin_page(
    hwnd: int,
    camera_height: int,
    template_path: Path,
    threshold: float,
    timeout: float,
    classify_hold: float,
) -> tuple[Image.Image, float, int, int, Image.Image, str, float]:
    """Observe a sharp page until one classification stays stable long enough."""
    started = time.monotonic()
    deadline = started + timeout
    stable_key: str | None = None
    stable_since = started
    stable_samples = 0
    while True:
        _check_escape("用户按下 Esc，等待抖音页面稳定时已停止。")
        camera = camera_crop(capture_client(hwnd), camera_height)
        score, x, y, template = douyin_home_score(camera, template_path)
        home_ready = valid_douyin_home_marker(
            camera, template, score, x, y, threshold
        )
        sharpness = camera_ui_sharpness(camera)
        # 不同类型视频的底部导航曝光差异很大；只要正常爱心已稳定出现，
        # 就能确认上划动画结束，不再强制要求“首页”文字模板同时清晰。
        heart_state = detect_douyin_heart(camera).state
        heart_ready = heart_state != "missing"
        live_close = detect_douyin_live_close(camera)
        live_preview_badge = detect_douyin_live_preview_badge(camera)
        # 运动模糊帧可能把直播画面里的红色块误认成“关注 +”，并由此
        # 推算出一个不存在的白心。必须 UI 足够清晰且连续两帧满足条件，
        # 才允许后续点击。
        # 全屏直播没有推荐流的普通爱心栏。若“×”和普通爱心同时出现，
        # 应以爱心结构为准，避免把右上角搜索图标当成直播关闭按钮。
        full_live_ready = live_close is not None and not heart_ready
        candidate_key: str | None = None
        candidate_status: str | None = None
        if sharpness >= 500.0:
            if live_preview_badge is not None:
                candidate_key = "live_preview"
                candidate_status = "live_preview"
            elif full_live_ready:
                candidate_key = "live"
                candidate_status = "live"
            elif heart_ready:
                # 连续观察期间爱心颜色也必须保持一致，防止点赞动画或
                # 页面切换瞬间被当作一个已经稳定的普通页面。
                candidate_key = f"normal:{heart_state}"
                candidate_status = "normal"

        now = time.monotonic()
        if candidate_key is None:
            stable_key = None
            stable_since = now
            stable_samples = 0
        elif candidate_key != stable_key:
            stable_key = candidate_key
            stable_since = now
            stable_samples = 1
        else:
            stable_samples += 1

        elapsed = now - started
        stable_elapsed = now - stable_since
        if (
            candidate_status is not None
            and stable_samples >= 4
            and stable_elapsed >= classify_hold
        ):
            return (
                camera,
                score,
                x,
                y,
                template,
                candidate_status,
                elapsed,
            )
        if now >= deadline:
            return camera, score, x, y, template, "timeout", elapsed
        sleep_interruptible(0.30)


def append_jsonl(path: Path, event: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"time": dt.datetime.now().isoformat(timespec="seconds"), **event}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def locate(
    title: str,
    template_path: Path,
    camera_height: int,
) -> tuple[int, str, Image.Image, Image.Image, float, int, int]:
    hwnd, actual_title = find_window(title)
    full = capture_client(hwnd)
    camera = camera_crop(full, camera_height)
    if not template_path.exists():
        raise FileNotFoundError(f"模板不存在：{template_path}")
    template = Image.open(template_path).convert("RGB")
    score, x, y = match_template(camera, template)
    return hwnd, actual_title, camera, template, score, x, y


def command_doctor(args: argparse.Namespace) -> int:
    ensure_dirs()
    hwnd, title = find_window(args.title)
    image = capture_client(hwnd)
    camera = camera_crop(image, args.camera_height)
    output = OUTPUT_DIR / "doctor_window.png"
    image.save(output)
    print(f"找到窗口：{title}")
    print(f"窗口客户区：{image.width}×{image.height}")
    print(f"按摄像区处理：{camera.width}×{camera.height}")
    print(f"窗口 DPI：{window_dpi(hwnd)}")
    print(f"卖家界面尺度：{seller_ui_scale(image.width) * 100:.0f}%")
    print(f"诊断截图：{output}")
    if image.height <= camera.height:
        print("提醒：控制端底部操作栏不可见；请恢复完整窗口后再执行动作。")
    return 0


def command_select(args: argparse.Namespace) -> int:
    ensure_dirs()
    hwnd, title = find_window(args.title)
    image = camera_crop(capture_client(hwnd), args.camera_height)
    output = Path(args.output).resolve()
    select_template(image, output)
    print(f"模板已保存：{output}")
    print("下一步运行 find，确认红框是否识别正确。")
    return 0


def command_find(args: argparse.Namespace) -> int:
    ensure_dirs()
    template_path = Path(args.template).resolve()
    _, title, camera, template, score, x, y = locate(
        args.title, template_path, args.camera_height
    )
    output = OUTPUT_DIR / "last_match.png"
    annotate_match(camera, template, score, x, y, output)
    center = (x + template.width // 2, y + template.height // 2)
    print(f"窗口：{title}")
    print(f"匹配分数：{score:.3f}（阈值 {args.threshold:.3f}）")
    print(f"目标中心：{center}")
    print(f"标注截图：{output}")
    if score < args.threshold:
        print("结果：未达到阈值，不会执行点击。")
        return 2
    print("结果：识别通过；当前命令只是检查，不会驱动机械臂。")
    return 0


def command_run_once(args: argparse.Namespace) -> int:
    ensure_dirs()
    template_path = Path(args.template).resolve()
    hwnd, title, camera, template, score, x, y = locate(
        args.title, template_path, args.camera_height
    )
    output = OUTPUT_DIR / "last_match.png"
    annotate_match(camera, template, score, x, y, output)
    center_x = x + template.width // 2
    center_y = y + template.height // 2
    print(f"窗口：{title}")
    print(f"匹配分数：{score:.3f}；目标中心：({center_x}, {center_y})")
    print(f"标注截图：{output}")
    if score < args.threshold:
        print("未达到阈值，拒绝点击。")
        return 2
    if center_y >= args.camera_height:
        print("目标不在摄像头画面内，拒绝点击。")
        return 2
    if not args.execute:
        print("DRY-RUN：识别成功，但没有点击。加 --execute 才会驱动机械臂。")
        return 0
    click_client_point(
        hwnd, center_x, center_y, args.countdown, args.hold
    )
    print("已向控制软件点击一次。请观察机械臂是否完成对应的物理点击。")
    return 0


def command_sequence(args: argparse.Namespace) -> int:
    ensure_dirs()
    sequence_path = Path(args.file).resolve()
    actions = json.loads(sequence_path.read_text(encoding="utf-8"))
    if not isinstance(actions, list) or not actions:
        raise ValueError("动作文件必须是非空 JSON 数组。")
    if not args.execute:
        print("sequence 默认不执行。请先逐个用 find 验证模板，再加 --execute。")
        return 0

    for index, action in enumerate(actions, start=1):
        template_path = (sequence_path.parent / action["template"]).resolve()
        threshold = float(action.get("threshold", args.threshold))
        timeout = float(action.get("timeout", 20.0))
        after = float(action.get("after", 1.5))
        deadline = time.monotonic() + timeout
        print(f"[{index}/{len(actions)}] 等待模板：{template_path.name}")
        while True:
            if user32.GetAsyncKeyState(VK_ESCAPE) & 0x8000:
                raise RuntimeError("用户按下 Esc，动作序列已停止。")
            try:
                hwnd, _, camera, template, score, x, y = locate(
                    args.title, template_path, args.camera_height
                )
            except RuntimeError:
                score = -1.0
            if score >= threshold:
                center_x = x + template.width // 2
                center_y = y + template.height // 2
                print(
                    f"识别成功 score={score:.3f}，点击 ({center_x}, {center_y})"
                )
                click_client_point(
                    hwnd, center_x, center_y, args.countdown, args.hold
                )
                time.sleep(after)
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"等待模板 {template_path.name} 超时；最后分数 {score:.3f}"
                )
            time.sleep(0.5)
    print("动作序列执行完成。")
    return 0


def command_douyin_calibrate(args: argparse.Namespace) -> int:
    """Capture a stable Douyin-home marker selected by the user."""
    ensure_dirs()
    hwnd, title = find_window(args.title)
    camera = camera_crop(capture_client(hwnd), args.camera_height)
    output = Path(args.output).resolve()
    print(f"窗口：{title}")
    if args.box:
        try:
            box = tuple(int(value.strip()) for value in args.box.split(","))
        except ValueError as exc:
            raise ValueError("--box 格式必须是 x1,y1,x2,y2 四个整数。") from exc
        if len(box) != 4:
            raise ValueError("--box 格式必须是 x1,y1,x2,y2 四个整数。")
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= camera.width and 0 <= y1 < y2 <= camera.height):
            raise ValueError(
                f"--box {box} 超出摄像区 {camera.width}×{camera.height}。"
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        camera.crop(box).save(output)
        print(f"已按坐标自动框选：{box}")
    else:
        print("请框选抖音底部导航栏左下角的“首页”图标和文字。")
        print("不要框选正在播放的视频、点赞数或作者头像。")
        select_template(
            camera,
            output,
            title="抖音标定：只框选底部左下角“首页”图标和文字",
        )
    template = Image.open(output).convert("RGB")
    score, x, y = match_template(camera, template)
    annotated = OUTPUT_DIR / "douyin_calibration.png"
    annotate_match(camera, template, score, x, y, annotated)
    print(f"标定模板已保存：{output}")
    print(f"当前画面自匹配分数：{score:.3f}")
    print(f"标定检查图：{annotated}")
    print("下一步先运行 douyin-auto（不加 --execute）做 dry-run。")
    return 0


def command_douyin_auto(args: argparse.Namespace) -> int:
    ensure_dirs()
    if args.count < 1 or args.count > 100:
        raise ValueError("--count 必须在 1～100 之间。")
    if args.interval < 1.0:
        raise ValueError("--interval 不能小于 1 秒。")
    if args.max_failures < 1:
        raise ValueError("--max-failures 至少为 1。")

    template_path = Path(args.template).resolve()
    hwnd, title = find_window(args.title)
    first_camera = camera_crop(capture_client(hwnd), args.camera_height)
    first_score, x, y, template = douyin_home_score(first_camera, template_path)
    first_annotated = OUTPUT_DIR / "douyin_home_check.png"
    annotate_match(
        first_camera,
        template,
        first_score,
        x,
        y,
        first_annotated,
    )
    print(f"窗口：{title}")
    print(f"抖音首页标记分数：{first_score:.3f}（阈值 {args.threshold:.3f}）")
    print(f"首页检查图：{first_annotated}")
    if first_score < args.threshold:
        print("当前不是已标定的抖音首页，或有弹窗遮挡；拒绝执行。")
        return 2

    print(
        f"计划：上划 {args.count} 次，间隔 {args.interval:.1f} 秒，"
        f"连续 {args.max_failures} 次无法验证时停止。"
    )
    if not args.execute:
        print("DRY-RUN：检查通过，但不会驱动机械臂。")
        print("确认无误后加 --execute；运行中随时按 Esc 停止。")
        return 0

    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / f"douyin_{run_id}"
    log_path = run_dir / "events.jsonl"
    run_dir.mkdir(parents=True, exist_ok=True)
    append_jsonl(
        log_path,
        {
            "event": "start",
            "count": args.count,
            "interval": args.interval,
            "threshold": args.threshold,
            "window": title,
        },
    )

    print(f"{args.countdown} 秒后开始；按 Esc 取消……")
    for remaining in range(args.countdown, 0, -1):
        _check_escape()
        print(f"{remaining}……", flush=True)
        sleep_interruptible(1.0)

    configure_up_swipe(hwnd)
    failures = 0
    completed = 0

    for index in range(1, args.count + 1):
        _check_escape("用户按下 Esc，抖音自动上划已停止。")
        camera_before = camera_crop(capture_client(hwnd), args.camera_height)
        home_score, hx, hy, home_template = douyin_home_score(
            camera_before, template_path
        )
        if home_score < args.threshold:
            failure_path = run_dir / f"{index:03d}_not_home.png"
            annotate_match(
                camera_before,
                home_template,
                home_score,
                hx,
                hy,
                failure_path,
            )
            append_jsonl(
                log_path,
                {
                    "event": "stop_not_home",
                    "index": index,
                    "home_score": round(home_score, 4),
                    "screenshot": failure_path.name,
                },
            )
            raise RuntimeError(
                f"第 {index} 次动作前未识别到抖音首页 "
                f"(score={home_score:.3f})，已安全停止。"
            )

        # 先测量视频自身的自然变化，避免把正在播放的运动误判为换视频。
        sleep_interruptible(args.sample_gap)
        camera_natural = camera_crop(capture_client(hwnd), args.camera_height)
        natural_change = image_change_score(camera_before, camera_natural)

        print(f"[{index}/{args.count}] 执行上划……", flush=True)
        trigger_selected_action(hwnd)
        sleep_interruptible(args.post_wait)
        camera_after = camera_crop(capture_client(hwnd), args.camera_height)
        swipe_change = image_change_score(camera_natural, camera_after)
        required_change = max(
            args.min_change,
            natural_change * args.change_factor + args.change_margin,
        )
        verified = swipe_change >= required_change

        before_path = run_dir / f"{index:03d}_before.jpg"
        after_path = run_dir / f"{index:03d}_after.jpg"
        camera_before.save(before_path, quality=88)
        camera_after.save(after_path, quality=88)
        append_jsonl(
            log_path,
            {
                "event": "swipe",
                "index": index,
                "home_score": round(home_score, 4),
                "natural_change": round(natural_change, 4),
                "swipe_change": round(swipe_change, 4),
                "required_change": round(required_change, 4),
                "verified": verified,
                "before": before_path.name,
                "after": after_path.name,
            },
        )

        if verified:
            failures = 0
            completed += 1
            print(
                f"  已验证画面切换：{swipe_change:.3f} "
                f">= {required_change:.3f}"
            )
        else:
            failures += 1
            print(
                f"  未能确认换视频：{swipe_change:.3f} "
                f"< {required_change:.3f}；连续失败 {failures}/{args.max_failures}"
            )
            if failures >= args.max_failures:
                append_jsonl(
                    log_path,
                    {
                        "event": "stop_unverified",
                        "index": index,
                        "consecutive_failures": failures,
                    },
                )
                raise RuntimeError(
                    f"连续 {failures} 次无法确认视频切换，已安全停止。"
                )

        if index < args.count:
            remaining_wait = max(0.0, args.interval - args.post_wait)
            sleep_interruptible(remaining_wait)

    append_jsonl(
        log_path,
        {"event": "complete", "requested": args.count, "verified": completed},
    )
    print(f"完成：计划 {args.count} 次，验证成功 {completed} 次。")
    print(f"运行日志与前后截图：{run_dir}")
    return 0


def _confirm_like_batch(count: int) -> None:
    expected = f"确认执行{count}条点赞"
    print()
    print("即将通过机械臂改变当前抖音账号的点赞记录和推荐信号。")
    print(f"如需继续，请输入：{expected}")
    answer = input("> ").strip()
    if answer != expected:
        raise RuntimeError("确认文本不匹配，已取消执行。")


def command_douyin_like(args: argparse.Namespace) -> int:
    """Like unliked normal videos and verify the persistent red-heart state."""
    ensure_dirs()
    if not (1 <= args.count <= 50):
        raise ValueError("--count 必须在 1～50 之间。")
    if args.max_pages < 0:
        raise ValueError("--max-pages 不能为负数。")
    if not (0 <= args.countdown <= 30):
        raise ValueError("--countdown 必须在 0～30 秒之间。")
    if not (0.1 <= args.tap_hold <= 0.5):
        raise ValueError("--tap-hold 必须在 0.1～0.5 秒之间。")
    if not (0.3 <= args.verify_wait <= 10.0):
        raise ValueError("--verify-wait 必须在 0.3～10 秒之间。")
    if not (1.0 <= args.verify_timeout <= 10.0):
        raise ValueError("--verify-timeout 必须在 1～10 秒之间。")
    if args.verify_timeout < args.verify_wait:
        raise ValueError("--verify-timeout 不能小于 --verify-wait。")
    if not (0.8 <= args.post_swipe <= 30.0):
        raise ValueError("--post-swipe 必须在 0.8～30 秒之间。")
    if not (1.0 <= args.page_ready_timeout <= 15.0):
        raise ValueError("--page-ready-timeout 必须在 1～15 秒之间。")
    if not (0.8 <= args.classify_hold <= 5.0):
        raise ValueError("--classify-hold 必须在 0.8～5 秒之间。")
    if args.classify_hold >= args.page_ready_timeout:
        raise ValueError("--classify-hold 必须小于 --page-ready-timeout。")

    template_path = Path(args.template).resolve()
    hwnd, title = find_window(args.title)
    first_camera = camera_crop(capture_client(hwnd), args.camera_height)
    home_score, hx, hy, home_template = douyin_home_score(
        first_camera, template_path
    )
    first_heart = detect_douyin_heart(first_camera)
    first_live_close = detect_douyin_live_close(first_camera)
    first_live_preview = detect_douyin_live_preview_badge(first_camera)
    check_path = OUTPUT_DIR / "douyin_like_check.png"
    annotate_heart_detection(first_camera, first_heart, check_path)
    print(f"窗口：{title}")
    print(f"抖音首页标记分数：{home_score:.3f}（阈值 {args.threshold:.3f}）")
    print(
        f"当前爱心状态：{first_heart.state}"
        + (
            f"，中心 {first_heart.center}，面积 {first_heart.area}"
            if first_heart.center
            else ""
        )
    )
    print(f"检测检查图：{check_path}")
    home_marker_ready = valid_douyin_home_marker(
        first_camera,
        home_template,
        home_score,
        hx,
        hy,
        args.threshold,
    )
    heart_structure_ready = first_heart.state != "missing"
    live_close_ready = first_live_close is not None
    live_preview_ready = first_live_preview is not None
    print(
        "启动页验证依据："
        + (
            "首页导航"
            if home_marker_ready
            else "关注按钮—爱心结构"
            if heart_structure_ready
            else "直播关闭按钮"
            if live_close_ready
            else "直播预览标签"
            if live_preview_ready
            else "无"
        )
    )
    if not (
        home_marker_ready
        or heart_structure_ready
        or live_close_ready
        or live_preview_ready
    ):
        annotate_match(
            first_camera,
            home_template,
            home_score,
            hx,
            hy,
            OUTPUT_DIR / "douyin_like_not_home.png",
        )
        print(
            "当前既未识别到抖音首页导航，也未识别到正常爱心结构；"
            "拒绝执行。"
        )
        return 2
    if not args.execute:
        print("DRY-RUN：只完成识别检查，没有点击或上划。")
        print("检查图正确后，加 --execute 才会启动本地闭环。")
        return 0

    if not args.yes:
        _confirm_like_batch(args.count)

    max_pages = args.max_pages or max(args.count * 3, args.count + 5)
    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / f"douyin_like_{run_id}"
    log_path = run_dir / "events.jsonl"
    run_dir.mkdir(parents=True, exist_ok=True)
    append_jsonl(
        log_path,
        {
            "event": "start",
            "requested_likes": args.count,
            "max_pages": max_pages,
            "tap_hold": args.tap_hold,
            "verify_wait": args.verify_wait,
            "verify_timeout": args.verify_timeout,
            "post_swipe": args.post_swipe,
            "page_ready_timeout": args.page_ready_timeout,
            "classify_hold": args.classify_hold,
            "window": title,
        },
    )

    print("正在强制设置：连点次数=1，动作=上划。")
    configure_single_click_count(hwnd)
    configure_up_swipe(hwnd)
    print(f"{args.countdown} 秒后开始；运行中按 Esc 可随时停止。")
    for remaining in range(args.countdown, 0, -1):
        _check_escape()
        print(f"{remaining}……", flush=True)
        sleep_interruptible(1.0)

    liked = 0
    skipped_live_or_unknown = 0
    skipped_already_liked = 0
    inspected_pages = 0

    while liked < args.count and inspected_pages < max_pages:
        _check_escape("用户按下 Esc，抖音自动点赞已停止。")
        inspected_pages += 1
        (
            camera_before,
            home_score,
            home_x,
            home_y,
            home_template,
            page_status,
            ready_wait,
        ) = capture_ready_douyin_page(
            hwnd,
            args.camera_height,
            template_path,
            args.threshold,
            args.page_ready_timeout,
            args.classify_hold,
        )
        # 全屏直播必须同时满足“存在真正的 ×”和“没有普通爱心栏”。
        # 单独一个右上角白色图标不足以覆盖正常推荐页。
        status_heart = detect_douyin_heart(camera_before)
        if (
            detect_douyin_live_close(camera_before) is not None
            and status_heart.state == "missing"
        ):
            page_status = "live"
        elif detect_douyin_live_preview_badge(camera_before) is not None:
            page_status = "live_preview"
        page_ready = page_status == "normal"
        if page_status == "timeout":
            append_jsonl(
                log_path,
                {
                    "event": "page_ready_timeout",
                    "page": inspected_pages,
                    "home_score": round(home_score, 4),
                    "ready_wait_seconds": round(ready_wait, 3),
                },
            )
            print(
                f"  等待 {ready_wait:.1f} 秒后仍无普通爱心，"
                "按直播/未知页面处理。",
                flush=True,
            )
        elif page_status == "live":
            live_close = detect_douyin_live_close(camera_before)
            append_jsonl(
                log_path,
                {
                    "event": "live_detected",
                    "page": inspected_pages,
                    "close_button": list(live_close) if live_close else None,
                    "ready_wait_seconds": round(ready_wait, 3),
                },
            )
            print(
                f"  已识别全屏直播，等待 {ready_wait:.1f} 秒后确认画面稳定；"
                "不执行点赞。",
                flush=True,
            )
        elif page_status == "live_preview":
            live_badge = detect_douyin_live_preview_badge(camera_before)
            append_jsonl(
                log_path,
                {
                    "event": "live_preview_detected",
                    "page": inspected_pages,
                    "badge": list(live_badge) if live_badge else None,
                    "ready_wait_seconds": round(ready_wait, 3),
                },
            )
            print(
                f"  已识别推荐流直播预览，等待 {ready_wait:.1f} 秒确认稳定；"
                "禁止点击，直接跳过。",
                flush=True,
            )
        status_label = {
            "normal": "普通视频",
            "live_preview": "推荐流直播预览",
            "live": "全屏直播",
            "timeout": "其他/未知页面",
        }[page_status]
        print(
            f"  页面分类：{status_label}；连续稳定观察至少 "
            f"{args.classify_hold:.1f} 秒，总等待 {ready_wait:.1f} 秒。",
            flush=True,
        )

        detection_before = detect_douyin_heart(camera_before)
        # 在真正点击前重新抓一帧做第二道直播硬拦截，避免页面状态在
        # 识别与点击之间发生变化。
        if (
            page_ready
            and detection_before.state == "unliked"
            and detection_before.center
        ):
            guard_camera = camera_crop(
                capture_client(hwnd),
                args.camera_height,
            )
            guard_live_close = detect_douyin_live_close(guard_camera)
            guard_live_preview = detect_douyin_live_preview_badge(guard_camera)
            guard_heart = detect_douyin_heart(guard_camera)
            guard_full_live = (
                guard_live_close is not None
                and guard_heart.state == "missing"
            )
            if guard_full_live or guard_live_preview is not None:
                camera_before = guard_camera
                detection_before = guard_heart
                page_status = (
                    "live" if guard_full_live else "live_preview"
                )
                page_ready = False
                append_jsonl(
                    log_path,
                    {
                        "event": "live_click_guard",
                        "page": inspected_pages,
                        "page_status": page_status,
                    },
                )
                print("  点击前直播复检命中，已取消本次点赞。", flush=True)
        before_path = run_dir / f"{inspected_pages:03d}_before.jpg"
        annotate_heart_detection(camera_before, detection_before, before_path)

        if page_status == "timeout":
            append_jsonl(
                log_path,
                {
                    "event": "stop_unknown_page",
                    "page": inspected_pages,
                    "screenshot": before_path.name,
                },
            )
            raise RuntimeError(
                f"第 {inspected_pages} 个页面在等待 {ready_wait:.1f} 秒后"
                "仍不是普通推荐视频、直播预览或全屏直播。"
                f"为避免在错误页面继续操作，程序已停止。截图：{before_path}"
            )
        if (
            page_ready
            and detection_before.state == "unliked"
            and detection_before.center
        ):
            center_x, center_y = detection_before.center
            print(
                f"[成功 {liked}/{args.count}｜页面 {inspected_pages}/{max_pages}] "
                f"单击白色爱心 ({center_x}, {center_y})……",
                flush=True,
            )
            click_client_point(
                hwnd,
                center_x,
                center_y,
                countdown=0,
                hold_seconds=args.tap_hold,
            )
            move_cursor_outside_camera(hwnd)
            (
                camera_after,
                detection_after,
                verified,
                verification_method,
                verification_seconds,
                transition_metrics,
            ) = wait_for_liked_heart(
                hwnd,
                args.camera_height,
                camera_before,
                (center_x, center_y),
                args.verify_wait,
                args.verify_timeout,
            )
            after_path = run_dir / f"{inspected_pages:03d}_after.jpg"
            annotate_heart_detection(camera_after, detection_after, after_path)
            append_jsonl(
                log_path,
                {
                    "event": "like_attempt",
                    "page": inspected_pages,
                    "click": [center_x, center_y],
                    "before_state": detection_before.state,
                    "after_state": detection_after.state,
                    "verified": verified,
                    "verification_method": verification_method,
                    "verification_seconds": round(verification_seconds, 3),
                    "transition_metrics": transition_metrics,
                    "before": before_path.name,
                    "after": after_path.name,
                },
            )
            if not verified:
                raise RuntimeError(
                    f"第 {inspected_pages} 个页面点击后，连续等待 "
                    f"{verification_seconds:.1f} 秒仍未确认爱心变红 "
                    f"(检测结果 {detection_after.state})。为避免误操作已停止。"
                )
            liked += 1
            print(f"  已验证爱心持续变红：{liked}/{args.count}")
        elif page_ready and detection_before.state == "liked":
            skipped_already_liked += 1
            append_jsonl(
                log_path,
                {
                    "event": "skip_already_liked",
                    "page": inspected_pages,
                    "screenshot": before_path.name,
                },
            )
            print("  当前视频已经点赞，不重复点击，也不计入新增成功数。")
        else:
            skipped_live_or_unknown += 1
            append_jsonl(
                log_path,
                {
                    "event": "skip_no_normal_heart",
                    "page": inspected_pages,
                    "screenshot": before_path.name,
                },
            )
            print("  未检测到普通视频白色爱心，按直播/未知页面跳过。")

        if page_status == "live":
            live_close = detect_douyin_live_close(camera_before)
            if live_close is not None:
                print(
                    f"  单击直播关闭按钮 {live_close}，返回推荐流……",
                    flush=True,
                )
                click_client_point(
                    hwnd,
                    live_close[0],
                    live_close[1],
                    countdown=0,
                    hold_seconds=args.tap_hold,
                )
                move_cursor_outside_camera(hwnd)
                sleep_interruptible(1.0)
                append_jsonl(
                    log_path,
                    {
                        "event": "live_closed",
                        "page": inspected_pages,
                        "click": list(live_close),
                    },
                )

        if liked >= args.count:
            break
        print("  调用卖家软件内置上划……", flush=True)
        trigger_selected_action(hwnd)
        sleep_interruptible(args.post_swipe)

    if liked < args.count:
        append_jsonl(
            log_path,
            {
                "event": "stop_max_pages",
                "verified_likes": liked,
                "requested_likes": args.count,
                "inspected_pages": inspected_pages,
            },
        )
        raise RuntimeError(
            f"已检查 {inspected_pages} 个页面，只新增并验证 {liked} 条点赞；"
            "达到最大页面数后停止。"
        )

    append_jsonl(
        log_path,
        {
            "event": "complete",
            "verified_likes": liked,
            "inspected_pages": inspected_pages,
            "skipped_already_liked": skipped_already_liked,
            "skipped_live_or_unknown": skipped_live_or_unknown,
        },
    )
    print(
        f"完成：新增并验证 {liked}/{args.count} 条点赞；"
        f"共检查 {inspected_pages} 个页面，"
        f"跳过已点赞 {skipped_already_liked}，"
        f"跳过直播/未知 {skipped_live_or_unknown}。"
    )
    print(f"证据截图和日志：{run_dir}")
    return 0


def command_selftest(_args: argparse.Namespace) -> int:
    ensure_dirs()
    rng = np.random.default_rng(7)
    source_arr = rng.integers(0, 50, size=(180, 240), dtype=np.uint8)
    pattern = np.zeros((32, 44), dtype=np.uint8)
    pattern[3:29, 4:40] = 180
    pattern[10:22, 14:30] = 250
    expected_x, expected_y = 123, 77
    source_arr[
        expected_y : expected_y + pattern.shape[0],
        expected_x : expected_x + pattern.shape[1],
    ] = pattern
    score, x, y = match_template(Image.fromarray(source_arr), Image.fromarray(pattern))
    if (x, y) != (expected_x, expected_y) or score < 0.99:
        raise AssertionError(
            f"模板匹配自检失败：score={score:.4f}, pos=({x},{y})"
        )
    still = Image.fromarray(source_arr)
    changed_arr = source_arr.copy()
    changed_arr[20:140, 80:220] = 255 - changed_arr[20:140, 80:220]
    changed = Image.fromarray(changed_arr)
    if image_change_score(still, still) != 0.0:
        raise AssertionError("画面变化自检失败：相同图像的变化分数不为 0。")
    if image_change_score(still, changed) < 0.05:
        raise AssertionError("画面变化自检失败：未检测到明显变化。")

    def synthetic_heart(color: tuple[int, int, int] | None) -> Image.Image:
        image = Image.new("RGB", (540, 960), (25, 25, 25))
        if color is not None:
            draw = ImageDraw.Draw(image)
            draw.ellipse((448, 455, 471, 481), fill=color)
            draw.ellipse((466, 455, 489, 481), fill=color)
            draw.polygon(
                ((449, 469), (488, 469), (468, 500)),
                fill=color,
            )
        return image

    if detect_douyin_heart(synthetic_heart((240, 240, 240))).state != "unliked":
        raise AssertionError("爱心识别自检失败：没有识别出白色未点赞爱心。")
    if detect_douyin_heart(synthetic_heart((235, 45, 100))).state != "liked":
        raise AssertionError("爱心识别自检失败：没有识别出红色已点赞爱心。")
    if detect_douyin_heart(synthetic_heart(None)).state != "missing":
        raise AssertionError("爱心识别自检失败：空白画面被误判为爱心。")

    real_sample = OUTPUT_DIR / "douyin_20260727_135007" / "001_after.jpg"
    real_state = "未检查"
    if real_sample.exists():
        real_detection = detect_douyin_heart(
            Image.open(real_sample).convert("RGB")
        )
        real_state = real_detection.state
        if real_detection.state != "unliked":
            raise AssertionError(
                "爱心识别自检失败：历史真实截图应识别为白色未点赞，"
                f"实际为 {real_detection.state}。"
            )

    compact_sample = (
        OUTPUT_DIR
        / "douyin_like_20260727_153959"
        / "003_before.jpg"
    )
    compact_state = "未检查"
    if compact_sample.exists():
        compact_detection = detect_douyin_heart(
            Image.open(compact_sample).convert("RGB")
        )
        compact_state = compact_detection.state
        if compact_state != "unliked":
            raise AssertionError(
                "爱心识别自检失败：紧凑布局截图应识别为白色未点赞，"
                f"实际为 {compact_state}。"
            )

    bright_sample = (
        OUTPUT_DIR
        / "douyin_like_20260727_153959"
        / "005_not_home.jpg"
    )
    bright_state = "未检查"
    if bright_sample.exists():
        bright_detection = detect_douyin_heart(
            Image.open(bright_sample).convert("RGB")
        )
        bright_state = bright_detection.state
        if bright_state != "unliked":
            raise AssertionError(
                "爱心识别自检失败：亮色视频截图应识别为白色未点赞，"
                f"实际为 {bright_state}。"
            )

    animated_before = (
        OUTPUT_DIR
        / "douyin_like_20260727_155642"
        / "001_before.jpg"
    )
    animated_after = (
        OUTPUT_DIR
        / "douyin_like_20260727_155642"
        / "001_after.jpg"
    )
    animated_state = "未检查"
    if animated_before.exists() and animated_after.exists():
        transition_verified, transition_metrics = liked_transition_near_click(
            Image.open(animated_before).convert("RGB"),
            Image.open(animated_after).convert("RGB"),
            (465, 497),
        )
        animated_state = "liked" if transition_verified else "missing"
        if not transition_verified:
            raise AssertionError(
                "爱心识别自检失败：点赞动画前后帧应验证为已点赞，"
                f"实际指标为 {transition_metrics}。"
            )

    transition_blur = (
        OUTPUT_DIR
        / "douyin_like_20260727_160709"
        / "009_before.jpg"
    )
    stable_page = (
        OUTPUT_DIR
        / "douyin_like_20260727_160709"
        / "008_before.jpg"
    )
    blur_state = "未检查"
    if transition_blur.exists() and stable_page.exists():
        blur_score = camera_ui_sharpness(
            Image.open(transition_blur).convert("RGB")
        )
        stable_score = camera_ui_sharpness(
            Image.open(stable_page).convert("RGB")
        )
        blur_state = f"blocked({blur_score:.0f}<{stable_score:.0f})"
        if blur_score >= 500.0:
            raise AssertionError(
                "页面稳定性自检失败：直播过渡模糊帧未被拦截，"
                f"sharpness={blur_score:.1f}。"
            )
        if stable_score < 500.0:
            raise AssertionError(
                "页面稳定性自检失败：正常视频被误判为模糊，"
                f"sharpness={stable_score:.1f}。"
            )

    live_sample = (
        OUTPUT_DIR
        / "douyin_like_20260727_160709"
        / "009_after.jpg"
    )
    live_state = "未检查"
    if live_sample.exists():
        live_close = detect_douyin_live_close(
            Image.open(live_sample).convert("RGB")
        )
        live_state = str(live_close)
        if live_close is None:
            raise AssertionError(
                "直播识别自检失败：没有识别出右上角关闭按钮。"
            )

    normal_search_icon = (
        OUTPUT_DIR
        / "douyin_like_20260727_165555"
        / "011_before.jpg"
    )
    normal_search_state = "未检查"
    if normal_search_icon.exists():
        false_close = detect_douyin_live_close(
            Image.open(normal_search_icon).convert("RGB")
        )
        normal_search_state = str(false_close)
        if false_close is not None:
            raise AssertionError(
                "直播关闭按钮自检失败：普通推荐页的搜索放大镜被误判为 ×，"
                f"实际为 {false_close}。"
            )

    live_preview_states: list[str] = []
    for run_name, sample_name, expected in (
        ("douyin_like_20260727_162428", "002_before.jpg", True),
        ("douyin_like_20260727_162428", "007_before.jpg", True),
        ("douyin_like_20260727_162428", "001_before.jpg", False),
        ("douyin_like_20260727_162428", "006_before.jpg", False),
        # 163908 是发现误判的真实运行：004/008/013 是直播预览，
        # 007 虽含粉色内容但仍是普通视频。
        ("douyin_like_20260727_163908", "004_before.jpg", True),
        ("douyin_like_20260727_163908", "007_before.jpg", False),
        ("douyin_like_20260727_163908", "008_before.jpg", True),
        ("douyin_like_20260727_163908", "013_before.jpg", True),
    ):
        sample_path = OUTPUT_DIR / run_name / sample_name
        if not sample_path.exists():
            continue
        badge = detect_douyin_live_preview_badge(
            Image.open(sample_path).convert("RGB")
        )
        actual = badge is not None
        live_preview_states.append(
            f"{run_name[-6:]}/{sample_name[:3]}="
            f"{'live' if actual else 'normal'}"
        )
        if actual != expected:
            raise AssertionError(
                "直播预览识别自检失败："
                f"{run_name}/{sample_name} expected={expected}, "
                f"actual={actual}, "
                f"badge={badge}。"
            )

    latest_run = OUTPUT_DIR / "douyin_like_20260727_163908"
    heart_regression_states: list[str] = []
    for sample_name, expected in (
        # 014 的头像旁有红色关注按钮，但屏幕上的爱心仍是白色。
        ("014_before.jpg", "unliked"),
        # 001 是同一批次中已经成功点红的真实爱心。
        ("001_after.jpg", "liked"),
    ):
        sample_path = latest_run / sample_name
        if not sample_path.exists():
            continue
        actual = detect_douyin_heart(
            Image.open(sample_path).convert("RGB")
        ).state
        heart_regression_states.append(f"{sample_name[:3]}={actual}")
        if actual != expected:
            raise AssertionError(
                "爱心识别回归自检失败："
                f"{sample_name} expected={expected}, actual={actual}。"
            )

    pink_content_sample = (
        OUTPUT_DIR
        / "douyin_like_20260727_165555"
        / "008_before.jpg"
    )
    if pink_content_sample.exists():
        actual = detect_douyin_heart(
            Image.open(pink_content_sample).convert("RGB")
        ).state
        heart_regression_states.append(f"165555/008={actual}")
        if actual != "unliked":
            raise AssertionError(
                "爱心识别回归自检失败：普通视频中的粉色小块"
                f"不应覆盖白心，实际为 {actual}。"
            )
    print(
        f"自检通过：模板 score={score:.4f}, pos=({x},{y})；"
        f"爱心 synthetic=3/3，真实截图={real_state}，"
        f"紧凑布局={compact_state}，亮色视频={bright_state}，"
        f"点赞动画={animated_state}，直播过渡={blur_state}"
        f"，直播关闭={live_state}，普通页放大镜={normal_search_state}，"
        f"直播预览={','.join(live_preview_states) or '未检查'}，"
        f"误判回归={','.join(heart_regression_states) or '未检查'}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="智联新途机械臂控制端 GUI 自动点击 PoC"
    )
    parser.add_argument("--title", default=DEFAULT_WINDOW_TITLE)
    parser.add_argument("--camera-height", type=int, default=DEFAULT_CAMERA_HEIGHT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="检查控制端窗口并保存诊断截图")
    doctor.set_defaults(func=command_doctor)

    select = subparsers.add_parser("select", help="从当前摄像画面框选识别模板")
    select.add_argument(
        "--output", default=str(TEMPLATE_DIR / "target.png")
    )
    select.set_defaults(func=command_select)

    find = subparsers.add_parser("find", help="只识别并保存红框截图，不点击")
    find.add_argument("--template", default=str(TEMPLATE_DIR / "target.png"))
    find.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    find.set_defaults(func=command_find)

    run_once = subparsers.add_parser("run-once", help="识别模板并最多点击一次")
    run_once.add_argument("--template", default=str(TEMPLATE_DIR / "target.png"))
    run_once.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    run_once.add_argument("--countdown", type=int, default=3)
    run_once.add_argument(
        "--hold",
        type=float,
        default=0.35,
        help="鼠标按住时间（秒），实机默认 0.35",
    )
    run_once.add_argument(
        "--execute",
        action="store_true",
        help="明确允许真实点击；不加时只做 dry-run",
    )
    run_once.set_defaults(func=command_run_once)

    sequence = subparsers.add_parser("sequence", help="按 JSON 顺序识别并点击多个模板")
    sequence.add_argument("--file", required=True)
    sequence.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    sequence.add_argument("--countdown", type=int, default=1)
    sequence.add_argument("--hold", type=float, default=0.35)
    sequence.add_argument("--execute", action="store_true")
    sequence.set_defaults(func=command_sequence)

    douyin_calibrate = subparsers.add_parser(
        "douyin-calibrate",
        help="在抖音首页框选稳定标记（只需做一次）",
    )
    douyin_calibrate.add_argument(
        "--output",
        default=str(DEFAULT_DOUYIN_TEMPLATE),
        help="抖音首页标记模板保存路径",
    )
    douyin_calibrate.add_argument(
        "--box",
        help="可选：按 x1,y1,x2,y2 自动框选，跳过人工框选",
    )
    douyin_calibrate.set_defaults(func=command_douyin_calibrate)

    douyin_auto = subparsers.add_parser(
        "douyin-auto",
        help="识别抖音首页并自动上划换视频",
    )
    douyin_auto.add_argument(
        "--template",
        default=str(DEFAULT_DOUYIN_TEMPLATE),
        help="douyin-calibrate 生成的首页模板",
    )
    douyin_auto.add_argument(
        "--threshold",
        type=float,
        default=0.72,
        help="抖音首页标记匹配阈值",
    )
    douyin_auto.add_argument(
        "--count",
        type=int,
        default=3,
        help="本次最多上划次数（1～100）",
    )
    douyin_auto.add_argument(
        "--interval",
        type=float,
        default=8.0,
        help="两次上划开始时间的间隔秒数",
    )
    douyin_auto.add_argument(
        "--countdown",
        type=int,
        default=3,
        help="真实执行前倒计时秒数",
    )
    douyin_auto.add_argument(
        "--post-wait",
        type=float,
        default=2.2,
        help="触发上划后等待新视频稳定的秒数",
    )
    douyin_auto.add_argument(
        "--sample-gap",
        type=float,
        default=0.6,
        help="动作前测量视频自然变化的采样间隔",
    )
    douyin_auto.add_argument(
        "--min-change",
        type=float,
        default=0.10,
        help="确认换视频所需的最低画面变化分数",
    )
    douyin_auto.add_argument(
        "--change-factor",
        type=float,
        default=1.45,
        help="换视频变化相对自然变化的最低倍数",
    )
    douyin_auto.add_argument(
        "--change-margin",
        type=float,
        default=0.025,
        help="换视频变化相对自然变化追加的安全余量",
    )
    douyin_auto.add_argument(
        "--max-failures",
        type=int,
        default=2,
        help="连续多少次无法验证后自动停止",
    )
    douyin_auto.add_argument(
        "--execute",
        action="store_true",
        help="明确允许真实上划；不加时只做 dry-run",
    )
    douyin_auto.set_defaults(func=command_douyin_auto)

    douyin_like = subparsers.add_parser(
        "douyin-like",
        help="自动点赞普通视频并验证爱心持续变红",
    )
    douyin_like.add_argument(
        "--template",
        default=str(DEFAULT_DOUYIN_TEMPLATE),
        help="douyin-calibrate 生成的首页模板",
    )
    douyin_like.add_argument(
        "--threshold",
        type=float,
        default=0.55,
        help="抖音首页标记匹配阈值",
    )
    douyin_like.add_argument(
        "--count",
        type=int,
        default=10,
        help="需要新增并验证的点赞数量（1～50）",
    )
    douyin_like.add_argument(
        "--max-pages",
        type=int,
        default=0,
        help="最多检查页面数；0 表示自动取 count×3",
    )
    douyin_like.add_argument(
        "--countdown",
        type=int,
        default=3,
        help="真实执行前倒计时秒数",
    )
    douyin_like.add_argument(
        "--tap-hold",
        type=float,
        default=0.12,
        help="单次点按的软件鼠标按住时间（秒）",
    )
    douyin_like.add_argument(
        "--verify-wait",
        type=float,
        default=0.8,
        help="点按后开始验证前的最短等待秒数",
    )
    douyin_like.add_argument(
        "--verify-timeout",
        type=float,
        default=4.0,
        help="点按后连续等待爱心变红的最长秒数",
    )
    douyin_like.add_argument(
        "--post-swipe",
        type=float,
        default=1.6,
        help="调用内置上划后等待下一条页面的秒数",
    )
    douyin_like.add_argument(
        "--page-ready-timeout",
        type=float,
        default=6.0,
        help="等待页面完成分类的最长秒数",
    )
    douyin_like.add_argument(
        "--classify-hold",
        type=float,
        default=1.5,
        help="页面类型必须连续保持不变的观察秒数",
    )
    douyin_like.add_argument(
        "--execute",
        action="store_true",
        help="明确允许改变抖音账号点赞记录；不加时只做 dry-run",
    )
    douyin_like.add_argument(
        "--yes",
        action="store_true",
        help="跳过终端确认，仅供已在上层完成确认的自动调用方使用",
    )
    douyin_like.set_defaults(func=command_douyin_like)

    selftest = subparsers.add_parser("selftest", help="离线验证模板匹配算法")
    selftest.set_defaults(func=command_selftest)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\n已停止。", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
