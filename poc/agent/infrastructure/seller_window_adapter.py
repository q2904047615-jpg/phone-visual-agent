"""Camera capture and primitive input bridge for the seller-supplied robot window."""

from __future__ import annotations

from agent.domain.validation import reject_if
import ctypes
import math
import sys
import time
from contextlib import contextmanager
from ctypes import wintypes
from pathlib import Path

from PIL import Image, ImageGrab

# 新旧标题只共享稳定的“智联新途”前缀。
IS_WINDOWS = sys.platform == "win32"
DEFAULT_WINDOW_TITLE = "智联新途"
BASELINE_CLIENT_WIDTH = 540
DEFAULT_CAMERA_HEIGHT = 960
MIN_AUTO_LAYOUT_WIDTH = 300
ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "output"


if IS_WINDOWS:
    user32 = ctypes.windll.user32
else:
    # Keep the pure geometry/cropping helpers importable for offline tests.
    # Hardware entry points call _require_windows before reaching this value.
    class _OfflineUser32:
        """Patchable placeholder; it must never dispatch host input."""

        def keybd_event(self, *_args: object) -> None:
            raise RuntimeError("Win32 input is unavailable on this host")

        def mouse_event(self, *_args: object) -> None:
            raise RuntimeError("Win32 input is unavailable on this host")

        def __getattr__(self, _name: str) -> object:
            raise RuntimeError("Win32 window access is unavailable on this host")

    user32 = _OfflineUser32()


def _require_windows() -> None:
    if not IS_WINDOWS:
        raise RuntimeError(
            "卖家窗口控制只支持 Windows；当前进程可以运行无硬件的离线逻辑和测试。"
        )


def _win_error() -> OSError:
    """Return a platform-neutral Win32 error for mocked/offline tests."""

    factory = getattr(ctypes, "WinError", None)
    return factory() if callable(factory) else OSError("Win32 API call failed")


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


if IS_WINDOWS:
    _enable_per_monitor_dpi_awareness()

EnumWindowsProc = getattr(ctypes, "WINFUNCTYPE", ctypes.CFUNCTYPE)(
    wintypes.BOOL, wintypes.HWND, wintypes.LPARAM
)

SW_RESTORE = 9
GA_ROOT = 2
GA_ROOTOWNER = 3
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
VK_ESCAPE = 0x1B
GW_OWNER = 4
VK_HOME = 0x24
VK_DOWN = 0x28
VK_RETURN = 0x0D
VK_CONTROL = 0x11
VK_A = 0x41
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
INPUT_KEYBOARD = 1

# 底栏横坐标固定，纵坐标按客户区底部偏移以适应标题栏变化。
ACTION_BUTTON_X = 130
ACTION_DROPDOWN_X = 176
CLICK_COUNT_INPUT_X = 308
CONTROL_Y_FROM_BOTTOM = 18
BASELINE_TOOLBAR_HEIGHT = 50
SELLER_TOUCH_DOWN_SETTLE_SECONDS = 0.45
SELLER_SWIPE_TOUCH_DOWN_SECONDS = 0.35
SELLER_SWIPE_MOVEMENT_SECONDS = 0.30
SELLER_SWIPE_STEPS = 6
# The seller UI's first toolbar row contains ten machine-position buttons.
POSITION_BUTTON_Y_FROM_BOTTOM = 38


class POINT(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]


# Explicit 64-bit signatures prevent HWND truncation.
if IS_WINDOWS:
    user32.WindowFromPoint.argtypes = [POINT]
    user32.WindowFromPoint.restype = wintypes.HWND
    user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
    user32.GetAncestor.restype = wintypes.HWND
    user32.GetWindow.argtypes = [wintypes.HWND, wintypes.UINT]
    user32.GetWindow.restype = wintypes.HWND
    user32.GetForegroundWindow.restype = wintypes.HWND


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [('wVk', wintypes.WORD), ('wScan', wintypes.WORD), ('dwFlags', wintypes.DWORD), ('time', wintypes.DWORD),
        ('dwExtraInfo', ctypes.c_size_t)]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [('dx', wintypes.LONG), ('dy', wintypes.LONG), ('mouseData', wintypes.DWORD), ('dwFlags',
        wintypes.DWORD), ('time', wintypes.DWORD), ('dwExtraInfo', ctypes.c_size_t)]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [('uMsg', wintypes.DWORD), ('wParamL', wintypes.WORD), ('wParamH', wintypes.WORD)]


class INPUT_UNION(ctypes.Union):
    _fields_ = [('mi', MOUSEINPUT), ('ki', KEYBDINPUT), ('hi', HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _anonymous_ = ("data",)
    _fields_ = [('type', wintypes.DWORD), ('data', INPUT_UNION)]


class RECT(ctypes.Structure):
    _fields_ = [('left', wintypes.LONG), ('top', wintypes.LONG), ('right', wintypes.LONG), ('bottom', wintypes.LONG)]


class MONITORINFO(ctypes.Structure):
    _fields_ = [('cbSize', wintypes.DWORD), ('rcMonitor', RECT), ('rcWork', RECT), ('dwFlags', wintypes.DWORD)]


def find_window(title_fragment: str) -> tuple[int, str]:
    _require_windows()
    matches: list[tuple[int, str]] = []
    visible_titles: list[str] = []

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
        visible_titles.append(title)
        # The controller's message boxes carry the same title but are owned windows; when main.exe
        # hangs, Windows shows a same-titled "Ghost" stand-in that belongs to dwm.exe.
        if (title_fragment.lower() in title.lower() and not user32.GetWindow(hwnd, GW_OWNER)
                and _class_name(hwnd) != "Ghost"):
            matches.append((hwnd, title))
        return True

    user32.EnumWindows(callback, 0)
    if not matches:
        sample = "、".join(visible_titles[:5]) or "无"
        raise RuntimeError(
            f'没有找到标题包含“{title_fragment}”的可见窗口；'
            f'当前服务桌面可见窗口数={len(visible_titles)}，标题样本={sample}。'
            "请在运行服务的同一交互式桌面保持 main.exe 窗口可见。"
        )
    # Any message to a hung window blocks until it recovers, so stop before sending one.
    reject_if(user32.IsHungAppWindow(matches[0][0]),
              RuntimeError("卖家控制端 main.exe 没有响应（Windows 已标记为“未响应”）；请等它恢复或重启 main.exe 后再继续。"))
    return matches[0]


def _class_name(hwnd: int) -> str:
    buffer = ctypes.create_unicode_buffer(64)
    user32.GetClassNameW(hwnd, buffer, 64)
    return buffer.value


def _uia():
    """UI Automation client for this thread (reads and writes seller controls without keystrokes)."""

    import comtypes
    import comtypes.client

    try:
        comtypes.CoInitialize()
    except OSError:
        pass   # this thread already joined another COM apartment
    comtypes.client.GetModule("UIAutomationCore.dll")
    from comtypes.gen import UIAutomationClient as uia

    return uia, comtypes.client.CreateObject(uia.CUIAutomation, interface=uia.IUIAutomation)


class SellerPopupOpen(RuntimeError):
    """The controller shows a message box; main.exe refuses input until someone dismisses it."""


def seller_popups(hwnd: int) -> list[int]:
    """Visible windows owned by the controller, i.e. its message boxes."""

    if sys.platform != "win32":
        return []   # no seller controller (or its message boxes) exists off Windows, even when tests fake it
    popups: list[int] = []

    @EnumWindowsProc
    def callback(candidate: int, _lparam: int) -> bool:
        if user32.IsWindowVisible(candidate) and user32.GetWindow(candidate, GW_OWNER) == hwnd:
            popups.append(candidate)
        return True

    user32.EnumWindows(callback, 0)
    return popups


def seller_popup_text(hwnd: int) -> str | None:
    """The text of any message box the controller is showing, or None."""

    texts = []
    for popup in seller_popups(hwnd):
        try:
            uia, client = _uia()
            condition = client.CreatePropertyCondition(uia.UIA_ControlTypePropertyId, uia.UIA_TextControlTypeId)
            found = client.ElementFromHandle(popup).FindAll(uia.TreeScope_Descendants, condition)
            text = "".join(found.GetElement(i).CurrentName for i in range(found.Length)).strip()
        except Exception:   # best effort: the popup itself is what blocks the controller
            text = ""
        texts.append(text or "（无法读取提示文字）")
    return "；".join(texts) or None


def require_no_seller_popup(hwnd: int, when: str) -> None:
    """Fail before touching the controller while one of its message boxes is open."""

    text = seller_popup_text(hwnd)
    reject_if(bool(text), SellerPopupOpen(
        f"{when}控制端弹出提示“{text}”，在点掉之前控制端不会执行任何动作；请在控制端点掉提示后再继续。"))


def client_geometry(hwnd: int) -> tuple[int, int, int, int]:
    rect = RECT()
    reject_if(not user32.GetClientRect(hwnd, ctypes.byref(rect)), _win_error())
    top_left = POINT(0, 0)
    reject_if(not user32.ClientToScreen(hwnd, ctypes.byref(top_left)), _win_error())
    width = rect.right - rect.left
    height = rect.bottom - rect.top
    return top_left.x, top_left.y, width, height


def ensure_window_fully_visible(hwnd: int) -> None:
    """Resize/reposition the seller window so its camera and toolbar are usable."""

    _require_windows()

    user32.MonitorFromWindow.argtypes = [wintypes.HWND, wintypes.DWORD]
    user32.MonitorFromWindow.restype = wintypes.HMONITOR
    user32.GetMonitorInfoW.argtypes = [wintypes.HMONITOR, ctypes.c_void_p]
    user32.GetMonitorInfoW.restype = wintypes.BOOL
    window_rect = RECT()
    reject_if(not user32.GetWindowRect(hwnd, ctypes.byref(window_rect)), _win_error())
    _left, _top, client_width, client_height = client_geometry(hwnd)
    required_height = seller_required_client_height(client_width, client_height)

    # Auto-expand only plausible camera windows; small dialogs fail closed.
    portrait_candidate = client_height > client_width >= MIN_AUTO_LAYOUT_WIDTH
    desired_client_height = max(client_height, required_height) if portrait_candidate else client_height

    outer_width = window_rect.right - window_rect.left
    outer_height = window_rect.bottom - window_rect.top
    desired_outer_height = outer_height + desired_client_height - client_height

    MONITOR_DEFAULTTONEAREST = 2
    monitor = user32.MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST)
    info = MONITORINFO()
    info.cbSize = ctypes.sizeof(info)
    reject_if(not monitor or not user32.GetMonitorInfoW(monitor, ctypes.byref(info)), _win_error())
    work_width = info.rcWork.right - info.rcWork.left
    work_height = info.rcWork.bottom - info.rcWork.top
    reject_if(outer_width > work_width or desired_outer_height > work_height, RuntimeError('控制端完整摄像区和操作栏大于当前显示器工作区，已拒绝执行。'))

    target_left = min(max(window_rect.left, info.rcWork.left), info.rcWork.right - outer_width)
    target_top = min(max(window_rect.top, info.rcWork.top), info.rcWork.bottom - desired_outer_height)
    if target_left != window_rect.left or target_top != window_rect.top or desired_outer_height != outer_height:
        SWP_NOZORDER = 0x0004
        SWP_NOACTIVATE = 0x0010
        reject_if(
            not user32.SetWindowPos(hwnd, 0, target_left, target_top, outer_width, desired_outer_height,
            SWP_NOZORDER | SWP_NOACTIVATE),
            _win_error(),
        )
        time.sleep(0.2)


def seller_ui_scale(client_width: int) -> float:
    """Scale seller-control coordinates from its documented 540 px baseline."""

    reject_if(isinstance(client_width, bool) or int(client_width) <= 0, ValueError("控制端客户区宽度必须大于0。"))
    return int(client_width) / BASELINE_CLIENT_WIDTH


def scale_seller_ui_value(value: int | float, client_width: int) -> int:
    reject_if(isinstance(value, bool) or float(value) < 0, ValueError("控制端基准坐标必须是非负数。"))
    scaled = float(value) * seller_ui_scale(client_width)
    return int(math.floor(scaled + 0.5))


def seller_layout_scale(client_width: int, client_height: int) -> float:
    """Infer the seller UI scale for either phone orientation."""

    reject_if(isinstance(client_height, bool) or int(client_height) <= 0, ValueError("控制端客户区高度必须大于0。"))
    if int(client_width) > int(client_height):
        return int(client_width) / DEFAULT_CAMERA_HEIGHT
    return seller_ui_scale(client_width)


def scale_seller_vertical_value(value: int | float, client_width: int, client_height: int) -> int:
    reject_if(isinstance(value, bool) or float(value) < 0, ValueError("控制端基准坐标必须是非负数。"))
    scaled = float(value) * seller_layout_scale(client_width, client_height)
    return int(math.floor(scaled + 0.5))


def seller_required_client_height(client_width: int, client_height: int,
    baseline_height: int=DEFAULT_CAMERA_HEIGHT) -> int:
    """Return the smallest client height containing camera and both tool rows."""

    scale = seller_layout_scale(client_width, client_height)
    camera_baseline = BASELINE_CLIENT_WIDTH if int(client_width) > int(client_height) else baseline_height
    required = (camera_baseline + BASELINE_TOOLBAR_HEIGHT) * scale
    return int(math.floor(required + 0.5))


def seller_camera_height(client_width: int, client_height: int, baseline_height: int=DEFAULT_CAMERA_HEIGHT) -> int:
    """Map the 540x960 camera viewport to the current physical client size."""

    reject_if(isinstance(client_height, bool) or int(client_height) <= 0, ValueError("控制端客户区高度必须大于0。"))
    if int(client_height) > int(client_width):
        scaled = scale_seller_ui_value(baseline_height, client_width)
        return min(int(client_height), max(1, scaled))

    return max(1, scale_seller_vertical_value(BASELINE_CLIENT_WIDTH, client_width, client_height))


def seller_layout_has_full_camera(client_width: int, client_height: int,
    baseline_height: int=DEFAULT_CAMERA_HEIGHT) -> bool:
    """The real controller must include the full camera plus a bottom toolbar."""

    landscape = int(client_width) > int(client_height)
    if not landscape:
        return int(client_height) >= seller_required_client_height(client_width, client_height, baseline_height)

    # Rotated camera layouts may clip row two; require plausible camera size before Back recovery.
    scale = seller_layout_scale(client_width, client_height)
    expected_width = baseline_height * scale
    ratio = int(client_width) / int(client_height)
    return int(client_width) >= 800 and int(client_height) >= BASELINE_CLIENT_WIDTH * scale and (int(client_width) >=
        expected_width * 0.95) and (1.55 <= ratio <= 2.0)


def seller_control_point(client_width: int, client_height: int, baseline_x: int | float,
    baseline_y_from_bottom: int | float=CONTROL_Y_FROM_BOTTOM) -> tuple[int, int]:
    """Map one documented seller-toolbar point to the actual client pixels."""

    x = scale_seller_ui_value(baseline_x, client_width)
    bottom = scale_seller_vertical_value(baseline_y_from_bottom, client_width, client_height)
    y = int(client_height) - bottom
    reject_if(not (0 <= x < int(client_width) and 0 <= y < int(client_height)), ValueError(f'缩放后的控制点 ({x}, {y}) 超出窗口客户区 {client_width}×{client_height}。'))
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


def _validate_camera_region_unoccluded(hwnd: int, *, camera_height: int=DEFAULT_CAMERA_HEIGHT) -> None:
    """Prove the current desktop pixels belong to the seller controller."""

    left, top, width, height = client_geometry(hwnd)
    reject_if(not seller_layout_has_full_camera(width, height, camera_height), RuntimeError('控制端窗口没有完整显示摄像区和底部操作栏，已拒绝执行；请恢复完整窗口或为卖家软件启用独立 DPI 兼容设置。'))
    visible_height = seller_camera_height(width, height, camera_height)
    reject_if(width <= 0 or visible_height <= 0, RuntimeError("控制端相机区域没有有效大小。"))

    # Sample only camera interior owned by the controller tree; any foreign owner blocks capture.
    samples = ((0.2, 0.15), (0.5, 0.15), (0.8, 0.15), (0.2, 0.5), (0.5, 0.5), (0.8, 0.5), (0.2, 0.82), (0.5, 0.82),
        (0.8, 0.82))
    foreign: list[tuple[int, int, int]] = []
    expected = int(hwnd)
    for (x_ratio, y_ratio) in samples:
        screen_x = left + min(width - 1, max(0, int(round((width - 1) * x_ratio))))
        screen_y = top + min(visible_height - 1, max(0, int(round((visible_height - 1) * y_ratio))))
        owner = _root_window_at(screen_x, screen_y)
        if owner != expected:
            foreign.append((screen_x, screen_y, owner))
    reject_if(foreign, RuntimeError('控制端相机区域仍被其他窗口遮挡，已拒绝把电脑桌面当成手机画面。'))


def ensure_camera_region_unoccluded(hwnd: int, *, camera_height: int=DEFAULT_CAMERA_HEIGHT) -> None:
    """Activate the seller controller for one explicitly requested task capture."""

    _require_windows()

    user32.ShowWindow(hwnd, SW_RESTORE)
    ensure_window_fully_visible(hwnd)
    user32.BringWindowToTop(hwnd)
    user32.SetForegroundWindow(hwnd)
    time.sleep(0.12)
    _validate_camera_region_unoccluded(hwnd, camera_height=camera_height)


def capture_client(hwnd: int) -> Image.Image:
    _require_windows()
    ensure_camera_region_unoccluded(hwnd)
    left, top, width, height = client_geometry(hwnd)
    reject_if(width <= 0 or height <= 0, RuntimeError("控制端窗口当前没有有效大小，可能已最小化。"))
    return ImageGrab.grab(bbox=(left, top, left + width, top + height), all_screens=True).convert('RGB')


def capture_client_passive(hwnd: int) -> Image.Image:
    """Capture a visible preview without restoring, raising or focusing main.exe."""

    _require_windows()

    reject_if(_window_is_minimized(hwnd), RuntimeError("控制端已最小化，被动预览已暂停。"))
    _validate_camera_region_unoccluded(hwnd)
    left, top, width, height = client_geometry(hwnd)
    reject_if(width <= 0 or height <= 0, RuntimeError("控制端窗口当前没有有效大小，被动预览已暂停。"))
    return ImageGrab.grab(bbox=(left, top, left + width, top + height), all_screens=True).convert('RGB')


def camera_crop(image: Image.Image, camera_height: int) -> Image.Image:
    height = seller_camera_height(image.width, image.height, camera_height)
    return image.crop((0, 0, image.width, height))


@contextmanager
def _active_cursor_lease(hwnd: int, *, settle_seconds: float=0.1):
    """Temporarily activate the seller window and always restore the cursor."""

    original = POINT()
    reject_if(not user32.GetCursorPos(ctypes.byref(original)), _win_error())
    user32.ShowWindow(hwnd, SW_RESTORE)
    user32.SetForegroundWindow(hwnd)
    if settle_seconds > 0:
        time.sleep(settle_seconds)
    try:
        yield
    finally:
        user32.SetCursorPos(original.x, original.y)


def click_client_point(hwnd: int, x: int, y: int, countdown: int, hold_seconds: float, *,
    return_dispatch_receipt: bool=False, click_count: int=1) -> dict[str, object] | None:
    _require_windows()
    _, _, width, height = client_geometry(hwnd)
    reject_if(not (0 <= x < width and 0 <= y < height), ValueError(f"点击位置 ({x}, {y}) 超出窗口客户区 {width}×{height}。"))
    reject_if(not 0.1 <= hold_seconds <= 2.0, ValueError("按住时间必须在 0.1～2.0 秒之间。"))
    reject_if(click_count not in {1, 2}, ValueError("点击次数只允许1或2。"))

    screen_point = POINT(x, y)
    reject_if(not user32.ClientToScreen(hwnd, ctypes.byref(screen_point)), _win_error())

    pressed = False
    with _active_cursor_lease(hwnd, settle_seconds=0.1 if return_dispatch_receipt else 0.0):
        for remaining in range(countdown, 0, -1):
            _check_escape("用户按下 Esc，已取消执行。")
            print(f"{remaining} 秒后执行物理点击；按 Esc 取消……", flush=True)
            time.sleep(1)
        try:
            user32.SetCursorPos(screen_point.x, screen_point.y)
            user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            pressed = True
            # 实机验证表明 0.08 秒过短：机械臂会下压，但手机可能收不到触摸。
            time.sleep(hold_seconds)
            user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            pressed = False
            time.sleep(0.12)
        finally:
            if pressed:
                user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                time.sleep(0.12)

    if not return_dispatch_receipt:
        return None
    return {'version': '2026-09-08-seller-click-dispatch-v2', 'channel': 'left_button_atomic_click',
        'input_events_dispatched': True,
        'requested_mouse_hold_seconds': float(hold_seconds), 'click_count': int(click_count),
        'mechanical_contact_ack': False}


def long_press_client_point(hwnd: int, x: int, y: int, *, hold_seconds: float) -> dict[str, object]:
    """Dispatch one stationary right-button hold; visual effect is checked separately."""

    _require_windows()

    _, _, width, height = client_geometry(hwnd)
    camera_height = seller_camera_height(width, height)
    reject_if(not (0 <= x < width and 0 <= y < camera_height), ValueError(f'长按位置 ({x}, {y}) 超出摄像头客户区 {width}×{camera_height}。'))
    reject_if(not 0.5 <= float(hold_seconds) <= 2.0, ValueError("长按时间必须在0.5～2.0秒之间。"))

    point = POINT(x, y)
    reject_if(not user32.ClientToScreen(hwnd, ctypes.byref(point)), _win_error())
    pressed = False
    with _active_cursor_lease(hwnd):
        user32.SetCursorPos(point.x, point.y)
        _check_escape("用户按下 Esc，已取消长按。")
        try:
            user32.mouse_event(MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
            pressed = True
            sleep_interruptible(SELLER_TOUCH_DOWN_SETTLE_SECONDS)
            sleep_interruptible(float(hold_seconds))
        finally:
            if pressed:
                user32.mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
                time.sleep(0.12)
    return {'version': '2026-09-08-seller-hold-dispatch-v4', 'channel': 'right_button_stationary_touch',
        'input_events_dispatched': True, 'mechanical_contact_ack': False,
        'requested_hold_seconds': float(hold_seconds),
        'touch_down_settle_seconds': SELLER_TOUCH_DOWN_SETTLE_SECONDS}


def drag_client_path(hwnd: int, start: tuple[int, int], end: tuple[int, int], *, duration_seconds: float=0.8,
    steps: int=16) -> None:
    """Drive the seller UI's right-button touch-down/move/touch-up path."""

    _require_windows()

    _, _, width, height = client_geometry(hwnd)
    camera_height = seller_camera_height(width, height)
    for (name, (x, y)) in (('起点', start), ('终点', end)):
        reject_if(not (0 <= x < width and 0 <= y < camera_height), ValueError(f'拖动{name} ({x}, {y}) 超出摄像头客户区 {width}×{camera_height}。'))
    reject_if(start == end, ValueError("拖动起点和终点不能相同。"))
    reject_if(not 0.3 <= float(duration_seconds) <= 2.0, ValueError("拖动时间必须在0.3～2.0秒之间。"))
    reject_if(isinstance(steps, bool) or not 4 <= int(steps) <= 60, ValueError("拖动插值步数必须在4～60之间。"))

    start_point = POINT(*start)
    end_point = POINT(*end)
    reject_if(not user32.ClientToScreen(hwnd, ctypes.byref(start_point)), _win_error())
    reject_if(not user32.ClientToScreen(hwnd, ctypes.byref(end_point)), _win_error())

    pressed = False
    with _active_cursor_lease(hwnd):
        user32.SetCursorPos(start_point.x, start_point.y)
        _check_escape("用户按下 Esc，已取消拖动。")
        try:
            user32.mouse_event(MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
            pressed = True
            time.sleep(0.12)
            step_count = int(steps)
            delay = max(0.01, (float(duration_seconds) - 0.12) / step_count)
            for index in range(1, step_count + 1):
                _check_escape("用户按下 Esc，已停止拖动。")
                ratio = index / step_count
                user32.SetCursorPos(round(start_point.x + (end_point.x - start_point.x) * ratio),
                    round(start_point.y + (end_point.y - start_point.y) * ratio))
                time.sleep(delay)
        finally:
            if pressed:
                user32.mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
                time.sleep(0.12)


def swipe_client_path(hwnd: int, start: tuple[int, int], end: tuple[int, int], *,
    touch_down_seconds: float=SELLER_SWIPE_TOUCH_DOWN_SECONDS,
    movement_seconds: float=SELLER_SWIPE_MOVEMENT_SECONDS, steps: int=SELLER_SWIPE_STEPS) -> dict[str, object]:
    """Dispatch one bounded swipe and return an honest seller-event receipt.

    Unlike drag, a swipe first gives the actuator the same proven contact-establishment
    window as an atomic tap, then moves quickly and continuously. The receipt records
    dispatched mouse events only; it does not prove GUI processing or phone contact.
    """

    _require_windows()

    _, _, width, height = client_geometry(hwnd)
    camera_height = seller_camera_height(width, height)
    for (name, (x, y)) in (('起点', start), ('终点', end)):
        reject_if(not (0 <= x < width and 0 <= y < camera_height),
            ValueError(f'滑动{name} ({x}, {y}) 超出摄像头客户区 {width}×{camera_height}。'))
    reject_if(start == end, ValueError("滑动起点和终点不能相同。"))
    reject_if(not 0.1 <= float(touch_down_seconds) <= 0.45,
        ValueError("滑动触点建立时间必须在0.1～0.45秒之间。"))
    reject_if(not 0.15 <= float(movement_seconds) <= 0.6,
        ValueError("滑动移动时间必须在0.15～0.6秒之间。"))
    reject_if(isinstance(steps, bool) or not 4 <= int(steps) <= 12,
        ValueError("滑动插值步数必须在4～12之间。"))

    start_point = POINT(*start)
    end_point = POINT(*end)
    reject_if(not user32.ClientToScreen(hwnd, ctypes.byref(start_point)), _win_error())
    reject_if(not user32.ClientToScreen(hwnd, ctypes.byref(end_point)), _win_error())

    pressed = False
    right_up_dispatched = False
    with _active_cursor_lease(hwnd):
        user32.SetCursorPos(start_point.x, start_point.y)
        _check_escape("用户按下 Esc，已取消滑动。")
        try:
            user32.mouse_event(MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
            pressed = True
            sleep_interruptible(float(touch_down_seconds))
            step_count = int(steps)
            step_delay = float(movement_seconds) / step_count
            for index in range(1, step_count + 1):
                _check_escape("用户按下 Esc，已停止滑动。")
                ratio = index / step_count
                user32.SetCursorPos(round(start_point.x + (end_point.x - start_point.x) * ratio),
                    round(start_point.y + (end_point.y - start_point.y) * ratio))
                sleep_interruptible(step_delay)
            user32.mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
            pressed = False
            right_up_dispatched = True
            time.sleep(0.12)
        finally:
            if pressed:
                user32.mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
                time.sleep(0.12)

    return {'version': '2026-09-08-seller-swipe-dispatch-v2', 'channel': 'right_button_swipe_path',
        'right_button_down_dispatched': True, 'right_button_up_dispatched': right_up_dispatched,
        'interpolation_steps_completed': int(steps), 'input_events_dispatched': True, 'touch_down_seconds': float(touch_down_seconds),
        'movement_seconds': float(movement_seconds), 'step_count': int(steps),
        'client_start': [int(start[0]), int(start[1])], 'client_end': [int(end[0]), int(end[1])],
        'mechanical_contact_ack': False}


def _check_escape(message: str='用户按下 Esc，已停止执行。') -> None:
    _require_windows()
    reject_if(user32.GetAsyncKeyState(VK_ESCAPE) & 32768, RuntimeError(message))


def sleep_interruptible(seconds: float, poll_seconds: float=0.1) -> None:
    """Sleep while keeping Esc responsive."""
    deadline = time.monotonic() + max(0.0, seconds)
    while True:
        _check_escape()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(poll_seconds, remaining))


def click_client_control(hwnd: int, x: int, y: int, hold: float=0.08) -> None:
    """Click the seller software UI itself, not the camera/phone area."""
    _require_windows()
    _, _, width, height = client_geometry(hwnd)
    reject_if(not (0 <= x < width and 0 <= y < height), ValueError(f"控制点 ({x}, {y}) 超出窗口客户区 {width}×{height}。"))

    with _active_cursor_lease(hwnd):
        # Restoring/activating can move a window. Never reuse pre-activation
        # screen coordinates, and never send a global click into an occluder.
        _, _, current_width, current_height = client_geometry(hwnd)
        reject_if((current_width, current_height) != (width, height),
            RuntimeError('控制端尺寸在激活期间发生变化，未发送点击。'))
        point = POINT(x, y)
        reject_if(not user32.ClientToScreen(hwnd, ctypes.byref(point)), _win_error())
        reject_if(_root_window_at(point.x, point.y) != int(hwnd),
            RuntimeError('控制端按钮被其他窗口遮挡，未发送点击。'))
        reject_if(not user32.SetCursorPos(point.x, point.y), _win_error())
        actual = POINT()
        reject_if(not user32.GetCursorPos(ctypes.byref(actual)), _win_error())
        reject_if((actual.x, actual.y) != (point.x, point.y),
            RuntimeError('鼠标未到达控制端按钮，未发送点击。'))
        reject_if(_root_window_at(actual.x, actual.y) != int(hwnd),
            RuntimeError('控制端按钮被其他窗口遮挡，未发送点击。'))
        events = (INPUT * 1)()
        events[0].type = 0  # INPUT_MOUSE; coordinates were checked above.
        events[0].mi.dwFlags = MOUSEEVENTF_LEFTDOWN
        reject_if(user32.SendInput(1, events, ctypes.sizeof(INPUT)) != 1, _win_error())
        try:
            time.sleep(hold)
        finally:
            events[0].mi.dwFlags = MOUSEEVENTF_LEFTUP
            reject_if(user32.SendInput(1, events, ctypes.sizeof(INPUT)) != 1, _win_error())
        time.sleep(0.12)


def press_virtual_key(key_code: int) -> None:
    _require_windows()
    user32.keybd_event(key_code, 0, 0, 0)
    time.sleep(0.05)
    user32.keybd_event(key_code, 0, KEYEVENTF_KEYUP, 0)
    time.sleep(0.08)


def _require_seller_foreground(hwnd: int) -> None:
    """Keystrokes go to the foreground window; refuse unless that is the controller itself."""

    foreground = user32.GetForegroundWindow()
    # An open Qt dropdown list is its own top-level window owned by the controller, hence GA_ROOTOWNER.
    root = user32.GetAncestor(foreground, GA_ROOTOWNER) if foreground else 0
    reject_if(int(root or 0) != int(hwnd),
              RuntimeError("控制端不在前台，已拒绝发送按键（避免把按键打进其他窗口）。"))


class MachinePositionNotConfirmed(RuntimeError):
    """A station request was dispatched but its native selector did not confirm it."""


def machine_position_from_toolbar(toolbar: Image.Image) -> int | None:
    """Read the vendor's red 1-10 selector, never the phone/app image.

    Only a single unambiguous red button is accepted. Sampling both sides of
    each digit avoids text/hover borders and scales with the existing toolbar.
    This confirms UI selection only, not camera travel or mechanical contact.
    """
    width, height = toolbar.size
    if width < 100 or height < 20:
        return None
    rgb = toolbar.convert('RGB')
    selected = []
    for position in range(1, 11):
        samples = [rgb.getpixel((int((position - 1 + x) * width / 10), int(y * height)))
                   for x in (.22, .78) for y in (.16, .28, .40)]
        red = sum(r >= 150 and r > g * 1.6 and r > b * 1.6 for r, g, b in samples)
        if red >= 5:
            selected.append(position)
    return selected[0] if len(selected) == 1 else None


def read_machine_position(hwnd: int) -> int | None:
    """Read only the unobscured native toolbar, without activating the window."""
    _require_windows()
    if _window_is_minimized(hwnd):
        return None
    left, top, width, height = client_geometry(hwnd)
    if not seller_layout_has_full_camera(width, height):
        return None
    strip_height = scale_seller_vertical_value(BASELINE_TOOLBAR_HEIGHT, width, height)
    strip_top = top + height - strip_height
    for position in range(10):
        for x in (.22, .78):
            for y in (.16, .28, .40):
                if _root_window_at(left + int((position + x) * width / 10),
                                   strip_top + int(y * strip_height)) != int(hwnd):
                    return None
    image = ImageGrab.grab(bbox=(left, strip_top, left + width, top + height), all_screens=True)
    return machine_position_from_toolbar(image)


def select_machine_position(hwnd: int, machine_position: int) -> None:
    """Select one of the seller control window's documented 1-10 positions."""

    _require_windows()

    reject_if(isinstance(machine_position, bool) or not isinstance(machine_position, int)
        or not 1 <= machine_position <= 10, ValueError("卖家机位必须是1到10之间的整数。"))
    ensure_window_fully_visible(hwnd)
    _, _, width, height = client_geometry(hwnd)
    baseline_x = (float(machine_position) - 0.5) * BASELINE_CLIENT_WIDTH / 10.0
    x = scale_seller_ui_value(baseline_x, width)
    y = height - scale_seller_vertical_value(POSITION_BUTTON_Y_FROM_BOTTOM, width, height)
    reject_if(not (0 <= x < width and 0 <= y < height),
        ValueError(f"机位按钮坐标 ({x}, {y}) 超出窗口客户区 {width}×{height}。"))
    click_client_control(hwnd, x, y)
    # Observe only: never replay a click whose physical outcome is uncertain.
    observed = None
    for _ in range(10):
        time.sleep(0.2)
        observed = read_machine_position(hwnd)
        if observed == machine_position:
            return
    raise MachinePositionNotConfirmed(
        f'控制端未确认切换到{machine_position}号机位（当前选中：{observed or "未知"}）；未自动重复点击。')


def _click_count_field(hwnd: int):
    """UI Automation value pattern of the 连点次数 input (the edit box under its toolbar point)."""

    left, top, width, height = client_geometry(hwnd)
    control_x, control_y = seller_control_point(width, height, CLICK_COUNT_INPUT_X)
    x, y = left + control_x, top + control_y
    try:
        uia, client = _uia()
        condition = client.CreatePropertyCondition(uia.UIA_ControlTypePropertyId, uia.UIA_EditControlTypeId)
        edits = client.ElementFromHandle(hwnd).FindAll(uia.TreeScope_Descendants, condition)
        for index in range(edits.Length):
            edit = edits.GetElement(index)
            rect = edit.CurrentBoundingRectangle
            if rect.left <= x < rect.right and rect.top <= y < rect.bottom:
                return edit.GetCurrentPattern(uia.UIA_ValuePatternId).QueryInterface(uia.IUIAutomationValuePattern)
    except Exception as exc:   # comtypes missing, COM failure
        raise RuntimeError(f"无法通过 UI Automation 读取控制端连点次数：{exc}") from exc
    raise RuntimeError("在控制端底栏找不到连点次数输入框。")


def configure_click_count(hwnd: int, click_count: int) -> None:
    """Make the seller software's 连点次数 field exactly one or two.

    Written through UI Automation and read back. Never typed: keystrokes go to
    whichever window has focus, which is not main.exe when it cannot be brought
    to the foreground, and a stray key left in the field (e.g. "1 ") makes
    main.exe reject every click with a message box. Seen on the real rig on
    2026-11-04 (the keys went into a chat window); a read when already correct.
    """
    _require_windows()
    reject_if(click_count not in {1, 2}, ValueError("控制端连点次数只允许1或2。"))
    field = _click_count_field(hwnd)
    wanted = str(click_count)
    if field.CurrentValue != wanted:
        field.SetValue(wanted)
        time.sleep(0.05)
    value = field.CurrentValue
    reject_if(value != wanted, RuntimeError(f"控制端连点次数没能设为 {wanted}（现在是“{value}”）。"))


def configure_single_click_count(hwnd: int) -> None:
    """Force the seller software's 连点次数 field to one."""
    configure_click_count(hwnd, 1)


def cursor_parking_client_point(width: int, height: int) -> tuple[int, int]:
    """Return a passive pointer position below the seller camera preview."""

    camera_height = seller_camera_height(width, height, DEFAULT_CAMERA_HEIGHT)
    reject_if(camera_height >= height, ValueError("卖家窗口没有可用于停放鼠标的相机外控制条。"))
    return width // 2, camera_height + (height - camera_height) // 2


def cursor_parking_screen_point(window_rect: tuple[int, int, int, int], virtual_screen_rect: tuple[int, int, int,
    int]) -> tuple[int, int] | None:
    """Choose a visible desktop corner that is definitely outside the seller window."""

    window_left, window_top, window_right, window_bottom = window_rect
    screen_left, screen_top, screen_right, screen_bottom = virtual_screen_rect
    reject_if(screen_right <= screen_left or screen_bottom <= screen_top, ValueError("虚拟桌面范围无效。"))
    candidates = ((screen_left + 2, screen_top + 2), (screen_right - 3, screen_top + 2), (screen_left + 2,
        screen_bottom - 3), (screen_right - 3, screen_bottom - 3))
    window_center = ((window_left + window_right) / 2, (window_top + window_bottom) / 2)
    outside = [point for point in candidates if not (window_left <= point[0] < window_right
        and window_top <= point[1] < window_bottom)]
    if not outside:
        return None
    return max(outside, key=lambda point: (point[0] - window_center[0]) ** 2 + (point[1] - window_center[1]) ** 2)


@contextmanager
def temporarily_park_cursor_outside_camera(hwnd: int):
    """Temporarily park a pointer over the camera and restore it unless the user moved it."""

    original = POINT()
    reject_if(not user32.GetCursorPos(ctypes.byref(original)), _win_error())
    left, top, width, height = client_geometry(hwnd)
    camera_height = seller_camera_height(width, height, DEFAULT_CAMERA_HEIGHT)
    if not (left <= original.x < left + width and top <= original.y < top + camera_height):
        yield False
        return

    window = RECT()
    reject_if(not user32.GetWindowRect(hwnd, ctypes.byref(window)), _win_error())
    SM_XVIRTUALSCREEN = 76
    SM_YVIRTUALSCREEN = 77
    SM_CXVIRTUALSCREEN = 78
    SM_CYVIRTUALSCREEN = 79
    screen_left = int(user32.GetSystemMetrics(SM_XVIRTUALSCREEN))
    screen_top = int(user32.GetSystemMetrics(SM_YVIRTUALSCREEN))
    screen_width = int(user32.GetSystemMetrics(SM_CXVIRTUALSCREEN))
    screen_height = int(user32.GetSystemMetrics(SM_CYVIRTUALSCREEN))
    screen_point = cursor_parking_screen_point((window.left, window.top, window.right, window.bottom), (screen_left,
        screen_top, screen_left + screen_width, screen_top + screen_height))
    if screen_point is None:
        client_x, client_y = cursor_parking_client_point(width, height)
        screen_point = (left + client_x, top + client_y)
    reject_if(not user32.SetCursorPos(*screen_point), _win_error())
    try:
        time.sleep(0.25)
        yield True
    finally:
        current = POINT()
        cursor_read = user32.GetCursorPos(ctypes.byref(current))
        if not cursor_read or (current.x, current.y) == screen_point:
            user32.SetCursorPos(original.x, original.y)


def clear_seller_camera_overlay(hwnd: int) -> None:
    """Generate a transient mouse-leave without taking the user's cursor."""

    with temporarily_park_cursor_outside_camera(hwnd):
        pass


def configure_swipe(hwnd: int, direction: str) -> None:
    """Select one of the seller software's four documented swipe actions."""
    _require_windows()
    action_index = {'up': 0, 'down': 1, 'left': 2, 'right': 3}
    try:
        index = action_index[direction]
    except KeyError as exc:
        raise ValueError(f"Unsupported swipe direction: {direction}") from exc
    ensure_window_fully_visible(hwnd)
    _, _, width, height = client_geometry(hwnd)
    control_x, control_y = seller_control_point(width, height, ACTION_DROPDOWN_X)
    click_client_control(hwnd, control_x, control_y)
    # 下拉顺序：上划、下划、左划、右划、下拉、起点。Every key is checked against the foreground window.
    for key in (VK_HOME, *([VK_DOWN] * index), VK_RETURN):
        _require_seller_foreground(hwnd)
        press_virtual_key(key)
    time.sleep(0.25)


def trigger_selected_action(hwnd: int) -> None:
    _require_windows()
    ensure_window_fully_visible(hwnd)
    _, _, width, height = client_geometry(hwnd)
    control_x, control_y = seller_control_point(width, height, ACTION_BUTTON_X)
    click_client_control(hwnd, control_x, control_y)
