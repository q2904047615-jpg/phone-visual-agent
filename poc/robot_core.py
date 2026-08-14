from __future__ import annotations

import ctypes
import datetime as dt
import json
import re
import threading
import time
from ctypes import wintypes
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

import robot_gui_poc as legacy
from orientation_safety import (
    OrientationCredential,
    PhysicalExecutionGate,
)
from ocr_runtime import (
    OcrMatch,
    find_text as find_ocr_text,
    is_available as ocr_available,
    recognize as recognize_ocr,
)


WEB_TEMPLATE_DIR = legacy.TEMPLATE_DIR / "web"
WEB_OUTPUT_DIR = legacy.OUTPUT_DIR / "web"
WEB_CONFIG_PATH = legacy.ROOT / "web_workflows.json"

# A normal seller control window is portrait and tall enough to contain the
# phone camera view plus its bottom controls.  Startup/error dialogs can share
# the same title fragment, but are small landscape windows.  Keep controller
# presence separate from camera readiness so callers never execute against a
# blocking dialog that merely happens to be visible.
MIN_CAMERA_CLIENT_WIDTH = 300
MIN_CAMERA_CLIENT_HEIGHT = 500


def controller_client_has_camera(width: int, height: int) -> bool:
    if width < MIN_CAMERA_CLIENT_WIDTH or height < MIN_CAMERA_CLIENT_HEIGHT:
        return False
    if width > height:
        return width >= 800 and height >= 450 and 1.45 <= width / height <= 2.0
    return legacy.seller_layout_has_full_camera(width, height)


def oriented_navigation_ratio(
    x_ratio: float,
    y_ratio: float,
    *,
    landscape: bool,
) -> tuple[float, float]:
    """Map portrait Android navigation coordinates into the observed layout."""

    if landscape:
        # The camera feed rotates counter-clockwise when the phone enters
        # landscape: portrait (x, y) becomes landscape (y, 1 - x).
        return y_ratio, 1.0 - x_ratio
    return x_ratio, y_ratio


DEFAULT_CONFIG: dict[str, Any] = {
    "wechat": {
        "input_wait_per_character": 0.8,
        "ocr_language": "zh-Hans-CN",
        "ocr_scale": 3.0,
        "ocr_timeout": 10.0,
        "home_icon_y_offset": -46,
        "android_home_x_ratio": 0.50,
        "android_home_y_ratio": 0.976,
        "input_x_ratio": 0.50,
        "input_y_ratio": 0.89,
    },
    "douyin": {
        "threshold": 0.82,
        "page_ready_timeout": 8.0,
        "classify_hold": 1.5,
        "verify_wait": 0.6,
        "verify_timeout": 4.0,
        "tap_hold": 0.35,
    },
    "vision_agent": {
        "tap_hold": 0.35,
        "android_home_x_ratio": 0.50,
        "android_home_y_ratio": 0.976,
        "android_back_x_ratio": 0.685,
        "android_back_y_ratio": 0.976,
        "keyboard_backspace_x_ratio": 0.862,
        "keyboard_backspace_y_ratio": 0.844,
        "pinyin_keyboard": {
            "rows": [
                {
                    "keys": "qwertyuiop",
                    "x_start": 0.115,
                    "x_step": 0.0844,
                    "y": 0.704,
                },
                {
                    "keys": "asdfghjkl",
                    "x_start": 0.157,
                    "x_step": 0.0844,
                    "y": 0.773,
                },
                {
                    "keys": "zxcvbnm",
                    "x_start": 0.241,
                    "x_step": 0.0844,
                    "y": 0.844,
                },
            ],
            "key_hold": 0.18,
            "inter_key_wait": 0.12,
            "pre_key_wait": 0.45,
            "first_key_settle": 0.35,
        },
        "digit_long_press_hold": 0.78,
    },
}

TEMPLATE_FILES = {
    "wechat_home_icon": "wechat_home_icon.png",
    "wechat_file_transfer_entry": "wechat_file_transfer_entry.png",
    "wechat_file_transfer_title": "wechat_file_transfer_title.png",
    "wechat_input_field": "wechat_input_field.png",
    "wechat_send_button": "wechat_send_button.png",
    "douyin_comment_button": "douyin_comment_button.png",
    "douyin_comment_input": "douyin_comment_input.png",
    "douyin_comment_send": "douyin_comment_send.png",
}

WECHAT_REQUIRED = (
    "wechat_home_icon",
    "wechat_file_transfer_entry",
    "wechat_file_transfer_title",
    "wechat_input_field",
    "wechat_send_button",
)
DOUYIN_COMMENT_REQUIRED = (
    "douyin_comment_button",
    "douyin_comment_input",
    "douyin_comment_send",
)

INPUT_BUTTON_X_FROM_RIGHT = 20
CONTROL_Y_FROM_BOTTOM = legacy.CONTROL_Y_FROM_BOTTOM
GW_OWNER = 4
CF_UNICODETEXT = 13
GMEM_MOVEABLE = 0x0002

ALLOWED_TEXT_RE = re.compile(
    r"^[\u3400-\u4dbf\u4e00-\u9fffA-Za-z0-9 "
    r"，。！？、；：,.!?;:'\"（）()《》【】\[\]\-—_+@#%&*/=]+$"
)


class RobotWorkflowError(RuntimeError):
    """A user-facing workflow failure that must stop further physical actions."""


class WorkflowNotReady(RobotWorkflowError):
    """Required local calibration/template data is missing."""


class UnsafePageError(RobotWorkflowError):
    """The current page is unknown or differs from the expected safe state."""


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(base))
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_workflow_config() -> dict[str, Any]:
    if not WEB_CONFIG_PATH.exists():
        return json.loads(json.dumps(DEFAULT_CONFIG))
    try:
        raw = json.loads(WEB_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkflowNotReady(f"工作流配置损坏：{exc}") from exc
    if not isinstance(raw, dict):
        raise WorkflowNotReady("工作流配置必须是 JSON 对象。")
    return _deep_merge(DEFAULT_CONFIG, raw)


def qwerty_key_point(
    width: int,
    height: int,
    key: str,
    keyboard_config: dict[str, Any],
) -> tuple[int, int]:
    if len(key) != 1 or key not in "abcdefghijklmnopqrstuvwxyz":
        raise ValueError(f"不支持的拼音键：{key!r}")
    rows = keyboard_config.get("rows")
    if not isinstance(rows, list):
        raise WorkflowNotReady("拼音键盘配置缺少 rows。")
    for row in rows:
        if not isinstance(row, dict):
            continue
        keys = str(row.get("keys", ""))
        if key not in keys:
            continue
        index = keys.index(key)
        x_ratio = float(row["x_start"]) + index * float(row["x_step"])
        y_ratio = float(row["y"])
        if not (0.0 <= x_ratio <= 1.0 and 0.0 <= y_ratio <= 1.0):
            raise WorkflowNotReady("拼音键盘坐标超出画面范围。")
        return (
            min(width - 1, max(0, int(round(width * x_ratio)))),
            min(height - 1, max(0, int(round(height * y_ratio)))),
        )
    raise WorkflowNotReady(f"拼音键盘配置中没有按键 {key!r}。")


QWERTY_ANCHOR_KEYS = ("q", "p", "a", "l", "z", "m", "backspace")


def qwerty_keyboard_config_from_anchors(
    anchors: dict[str, Any],
    *,
    key_hold: float = 0.18,
    inter_key_wait: float = 0.12,
) -> dict[str, Any]:
    """Build and validate a QWERTY profile from seven relative anchor points.

    Anchor coordinates use the vision agent's 0..1000 coordinate space.  The
    visual model identifies only the row endpoints and backspace; all letter
    centers are then calculated locally and deterministically.
    """

    if not isinstance(anchors, dict):
        raise WorkflowNotReady("动态拼音键盘缺少 anchors。")

    points: dict[str, tuple[float, float]] = {}
    for key in QWERTY_ANCHOR_KEYS:
        value = anchors.get(key)
        if (
            not isinstance(value, (list, tuple))
            or len(value) != 2
            or any(
                isinstance(item, bool) or not isinstance(item, (int, float))
                for item in value
            )
        ):
            raise WorkflowNotReady(f"动态拼音键盘锚点 {key!r} 无效。")
        x, y = float(value[0]), float(value[1])
        if not (0.0 <= x <= 1000.0 and 0.0 <= y <= 1000.0):
            raise WorkflowNotReady(f"动态拼音键盘锚点 {key!r} 超出安全范围。")
        points[key] = (x, y)

    def row(
        first: str,
        last: str,
        key_count: int,
    ) -> tuple[float, float, float]:
        first_x, first_y = points[first]
        last_x, last_y = points[last]
        if last_x <= first_x:
            raise WorkflowNotReady("动态拼音键盘行锚点左右顺序错误。")
        if abs(first_y - last_y) > 45.0:
            raise WorkflowNotReady("动态拼音键盘同一行不够水平，拒绝执行。")
        step = (last_x - first_x) / (key_count - 1)
        if not 45.0 <= step <= 130.0:
            raise WorkflowNotReady("动态拼音键盘键距异常，拒绝执行。")
        return first_x, step, (first_y + last_y) / 2.0

    q_x, top_step, top_y = row("q", "p", 10)
    observed_a_x, _observed_middle_step, middle_y = row("a", "l", 9)
    observed_z_x, _observed_bottom_step, bottom_y = row("z", "m", 7)
    if not (450.0 <= top_y < middle_y < bottom_y <= 950.0):
        raise WorkflowNotReady("动态拼音键盘行位置或上下顺序异常，拒绝执行。")
    if not (35.0 <= middle_y - top_y <= 140.0):
        raise WorkflowNotReady("动态拼音键盘第一、二行间距异常。")
    if not (35.0 <= bottom_y - middle_y <= 140.0):
        raise WorkflowNotReady("动态拼音键盘第二、三行间距异常。")
    # Vision models reliably locate the keyboard rows and the Q/P endpoints,
    # but may report A/L and Z/M as the visible row bounds instead of the
    # literal key centers.  Use those four anchors as coarse QWERTY evidence,
    # then normalize the two indented rows from the measured Q..P key pitch.
    # This keeps letter clicks deterministic without trusting imprecise
    # per-letter visual coordinates.
    if not (q_x - top_step * 0.5 <= observed_a_x <= q_x + top_step * 2.0):
        raise WorkflowNotReady("动态拼音键盘第二行缩进异常。")
    if not (
        points["p"][0] - top_step * 2.0
        <= points["l"][0]
        <= points["p"][0] + top_step * 0.5
    ):
        raise WorkflowNotReady("动态拼音键盘第二行右端位置异常。")
    if not (
        q_x - top_step * 0.5 <= observed_z_x <= q_x + top_step * 3.0
    ):
        raise WorkflowNotReady("动态拼音键盘第三行缩进异常。")
    if not (
        points["p"][0] - top_step * 3.0
        <= points["m"][0]
        <= points["p"][0] + top_step * 0.5
    ):
        raise WorkflowNotReady("动态拼音键盘第三行右端位置异常。")

    a_x = q_x + top_step * 0.5
    middle_step = top_step
    z_x = q_x + top_step * 1.5
    bottom_step = top_step

    backspace_x, backspace_y = points["backspace"]
    if backspace_x <= z_x + bottom_step * 5.5 or abs(backspace_y - bottom_y) > 90.0:
        raise WorkflowNotReady("动态拼音键盘退格键位置异常。")

    return {
        "source": "vision_anchors_normalized",
        "rows": [
            {
                "keys": "qwertyuiop",
                "x_start": q_x / 1000.0,
                "x_step": top_step / 1000.0,
                "y": top_y / 1000.0,
            },
            {
                "keys": "asdfghjkl",
                "x_start": a_x / 1000.0,
                "x_step": middle_step / 1000.0,
                "y": middle_y / 1000.0,
            },
            {
                "keys": "zxcvbnm",
                "x_start": z_x / 1000.0,
                "x_step": bottom_step / 1000.0,
                "y": bottom_y / 1000.0,
            },
        ],
        "backspace_x_ratio": backspace_x / 1000.0,
        "backspace_y_ratio": backspace_y / 1000.0,
        "key_hold": max(0.08, min(float(key_hold), 0.50)),
        "inter_key_wait": max(0.05, min(float(inter_key_wait), 0.50)),
    }


def template_path(name: str) -> Path:
    try:
        filename = TEMPLATE_FILES[name]
    except KeyError as exc:
        raise KeyError(f"未知工作流模板：{name}") from exc
    return WEB_TEMPLATE_DIR / filename


def workflow_readiness() -> dict[str, Any]:
    def missing(names: tuple[str, ...]) -> list[str]:
        return [name for name in names if not template_path(name).is_file()]

    douyin_comment_missing = missing(DOUYIN_COMMENT_REQUIRED)
    wechat_ocr_ready = ocr_available()
    return {
        "wechat": {
            "ready": wechat_ocr_ready,
            "mode": "runtime_ocr",
            "missing_templates": [],
            "missing_capabilities": (
                [] if wechat_ocr_ready else ["windows_zh_hans_ocr"]
            ),
        },
        "douyin_like": {
            "ready": legacy.DEFAULT_DOUYIN_TEMPLATE.is_file(),
            "missing_templates": (
                [] if legacy.DEFAULT_DOUYIN_TEMPLATE.is_file() else ["douyin_home"]
            ),
        },
        "douyin_comment": {
            "ready": (
                legacy.DEFAULT_DOUYIN_TEMPLATE.is_file()
                and not douyin_comment_missing
            ),
            "missing_templates": (
                ([] if legacy.DEFAULT_DOUYIN_TEMPLATE.is_file() else ["douyin_home"])
                + douyin_comment_missing
            ),
        },
    }


def _ocr_matches_in_region(
    payload: dict[str, Any],
    target: str,
    width: int,
    height: int,
    region: tuple[float, float, float, float],
) -> list[OcrMatch]:
    left, top, right, bottom = region
    return [
        match
        for match in find_ocr_text(payload, target)
        if left * width <= match.center[0] <= right * width
        and top * height <= match.center[1] <= bottom * height
    ]


def classify_wechat_page(
    payload: dict[str, Any],
    width: int,
    height: int,
) -> str:
    """Classify the current WeChat navigation state from runtime OCR output."""
    bottom_region = (0.0, 0.78, 1.0, 0.98)
    has_bottom_navigation = any(
        _ocr_matches_in_region(payload, text, width, height, bottom_region)
        for text in ("通讯录", "发现")
    )
    if has_bottom_navigation:
        return "conversation_list"

    title_region = (0.0, 0.02, 1.0, 0.105)
    if _ocr_matches_in_region(
        payload,
        "文件传输助手",
        width,
        height,
        title_region,
    ):
        return "target_chat"

    conversation_region = (0.0, 0.08, 1.0, 0.82)
    if _ocr_matches_in_region(
        payload,
        "文件传输助手",
        width,
        height,
        conversation_region,
    ):
        return "conversation_list"

    if _ocr_matches_in_region(
        payload,
        "微信",
        width,
        height,
        (0.02, 0.08, 0.98, 0.92),
    ):
        return "android_home"
    return "unknown"


def classify_obscured_wechat_title(payload: dict[str, Any]) -> bool:
    """Recognize the visible suffix of 文件传输助手 below the PX/MM overlay."""
    text = "".join(
        str(line.get("text", "")) for line in (payload.get("lines") or [])
    )
    chinese_only = re.sub(r"[^\u3400-\u4dbf\u4e00-\u9fff]", "", text)
    marker = chinese_only.find("件传输")
    return marker >= 0 and "手" in chinese_only[marker + len("件传输") :]


def _set_clipboard_text(text: str) -> None:
    """Put Unicode text on the Windows clipboard without shelling out."""
    kernel32 = ctypes.windll.kernel32
    user32 = ctypes.windll.user32

    kernel32.GlobalAlloc.argtypes = (wintypes.UINT, ctypes.c_size_t)
    kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
    kernel32.GlobalLock.argtypes = (wintypes.HGLOBAL,)
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalUnlock.argtypes = (wintypes.HGLOBAL,)
    kernel32.GlobalUnlock.restype = wintypes.BOOL
    user32.SetClipboardData.argtypes = (wintypes.UINT, wintypes.HANDLE)
    user32.SetClipboardData.restype = wintypes.HANDLE

    encoded = (text + "\0").encode("utf-16-le")
    handle = kernel32.GlobalAlloc(GMEM_MOVEABLE, len(encoded))
    if not handle:
        raise ctypes.WinError()
    pointer = kernel32.GlobalLock(handle)
    if not pointer:
        raise ctypes.WinError()
    try:
        ctypes.memmove(pointer, encoded, len(encoded))
    finally:
        kernel32.GlobalUnlock(handle)

    if not user32.OpenClipboard(None):
        raise RobotWorkflowError("无法打开 Windows 剪贴板。")
    try:
        if not user32.EmptyClipboard():
            raise ctypes.WinError()
        if not user32.SetClipboardData(CF_UNICODETEXT, handle):
            raise ctypes.WinError()
        # Ownership transfers to the clipboard on success.
        handle = None
    finally:
        user32.CloseClipboard()
        if handle:
            kernel32.GlobalFree(handle)


def _visible_owned_windows(owner: int) -> set[int]:
    matches: set[int] = set()

    @legacy.EnumWindowsProc
    def callback(hwnd: int, _lparam: int) -> bool:
        if legacy.user32.IsWindowVisible(hwnd):
            window_owner = legacy.user32.GetWindow(hwnd, GW_OWNER)
            if window_owner == owner:
                matches.add(int(hwnd))
        return True

    legacy.user32.EnumWindows(callback, 0)
    return matches


def _window_title(hwnd: int) -> str:
    length = legacy.user32.GetWindowTextLengthW(hwnd)
    buffer = ctypes.create_unicode_buffer(max(1, length + 1))
    legacy.user32.GetWindowTextW(hwnd, buffer, len(buffer))
    return buffer.value


class RobotController:
    """Safe, single-machine adapter used by the local web task worker."""

    def __init__(
        self,
        title: str = legacy.DEFAULT_WINDOW_TITLE,
        *,
        calibration_path: Path | None = None,
        verified_actions: set[str] | frozenset[str] | None = None,
        device_id: str,
    ) -> None:
        self.title = title
        self.device_id = str(device_id or "").strip()
        self._physical_execution_gate = PhysicalExecutionGate(self.device_id)
        self.calibration_path = (
            Path(calibration_path)
            if calibration_path is not None
            else Path(__file__).with_name("tap_calibration.json")
        )
        self.stop_event = threading.Event()
        self.operation_lock = threading.Lock()
        # The browser MJPEG preview and the vision worker can otherwise call
        # the seller window capture routine at the same time. On Windows that
        # occasionally returns a transient, truncated client bitmap.
        self.capture_lock = threading.RLock()
        default_actions = {
            "tap_semantic",
            "dismiss_overlay",
            "swipe",
            "back",
            "home",
            "wait_for_change",
        }
        self.verified_actions = frozenset(
            default_actions if verified_actions is None else verified_actions
        )
        allowed_actions = default_actions | {
            "input_verified_text",
            "long_press",
            "drag",
            "reveal_system_navigation",
        }
        unexpected = self.verified_actions - allowed_actions
        if unexpected:
            raise ValueError(
                "设备已验证动作包含未知值：" + ", ".join(sorted(unexpected))
            )

    def hardware_capabilities(self) -> dict[str, bool]:
        return {
            action: action in self.verified_actions
            for action in (
                "tap_semantic",
                "dismiss_overlay",
                "swipe",
                "back",
                "home",
                "wait_for_change",
                "input_verified_text",
                "long_press",
                "drag",
                "reveal_system_navigation",
            )
        }

    def _require_verified_action(self, action: str, label: str) -> None:
        if action not in self.verified_actions:
            raise WorkflowNotReady(
                f"当前设备尚未完成{label}真机验收，拒绝执行。"
            )

    def arm_physical_execution(
        self,
        credential: OrientationCredential,
        *,
        action: str,
        scene_fingerprint: str,
    ) -> None:
        self._physical_execution_gate.arm(
            credential,
            action=action,
            scene_fingerprint=scene_fingerprint,
        )

    def clear_physical_execution_authorization(self) -> None:
        self._physical_execution_gate.clear()

    def _consume_physical_execution(
        self, action: str, frame: Image.Image
    ) -> OrientationCredential:
        return self._physical_execution_gate.consume(
            action=action,
            frame=frame,
        )

    def request_stop(self) -> None:
        self.stop_event.set()

    def clear_stop(self) -> None:
        self.stop_event.clear()

    def _checkpoint(self) -> None:
        if self.stop_event.is_set():
            raise RobotWorkflowError("用户已请求停止任务。")
        legacy._check_escape()

    def _sleep(self, seconds: float) -> None:
        deadline = time.monotonic() + max(0.0, seconds)
        while time.monotonic() < deadline:
            self._checkpoint()
            remaining = deadline - time.monotonic()
            self.stop_event.wait(min(0.1, max(0.0, remaining)))
        self._checkpoint()

    def device_status(self) -> dict[str, Any]:
        try:
            hwnd, title = legacy.find_window(self.title)
            _left, _top, width, height = legacy.client_geometry(hwnd)
            online = width > 0 and height > 0
            camera_online = online and controller_client_has_camera(width, height)
            error = None
            camera_error = (
                None
                if camera_online
                else (
                    "控制端当前显示启动/报错对话框，未检测到可用的手机摄像头画面。"
                    if online
                    else "控制端窗口不可用。"
                )
            )
        except Exception as exc:  # Device status must stay readable while offline.
            hwnd = 0
            title = ""
            width = height = 0
            online = False
            camera_online = False
            error = str(exc)
            camera_error = str(exc)
        return {
            "controller_online": online,
            "camera_online": camera_online,
            "window_title": title,
            "client_size": [width, height],
            "stop_requested": self.stop_event.is_set(),
            "busy": self.operation_lock.locked(),
            "error": error,
            "camera_error": camera_error,
            "readiness": workflow_readiness(),
        }

    def _capture_phone(self, hwnd: int) -> Image.Image:
        with self.capture_lock:
            return legacy.camera_crop(
                legacy.capture_client(hwnd), legacy.DEFAULT_CAMERA_HEIGHT
            )

    def _capture_phone_passive(self, hwnd: int) -> Image.Image:
        with self.capture_lock:
            return legacy.camera_crop(
                legacy.capture_client_passive(hwnd), legacy.DEFAULT_CAMERA_HEIGHT
            )

    def capture_preview(self, quality: int = 72) -> bytes:
        hwnd, _title = legacy.find_window(self.title)
        image = self._capture_phone_passive(hwnd)
        from io import BytesIO

        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
        return buffer.getvalue()

    def vision_capture(self) -> Image.Image:
        """Capture exactly the phone region seen by the visual agent."""
        hwnd, _title = legacy.find_window(self.title)
        legacy.move_cursor_outside_camera(hwnd)
        return self._capture_phone(hwnd)

    def vision_tap_relative(self, x: int, y: int) -> tuple[int, int]:
        """Tap a Qwen3-VL coordinate expressed on a 1000×1000 grid."""
        self._require_verified_action("tap_semantic", "点击")
        return self._vision_press_relative(
            x,
            y,
            action="tap_semantic",
            hold_seconds=float(load_workflow_config()["vision_agent"]["tap_hold"]),
        )

    def vision_dismiss_overlay_relative(self, x: int, y: int) -> tuple[int, int]:
        """Dismiss one observed overlay through its separately verified entry."""

        self._require_verified_action("dismiss_overlay", "关闭弹层")
        return self._vision_press_relative(
            x,
            y,
            action="dismiss_overlay",
            hold_seconds=float(load_workflow_config()["vision_agent"]["tap_hold"]),
        )

    def vision_long_press_relative(
        self,
        x: int,
        y: int,
        hold_seconds: float = 0.8,
    ) -> tuple[int, int]:
        """Long-press one calibrated visual target without changing its point."""

        self._require_verified_action("long_press", "长按")
        if not 0.5 <= float(hold_seconds) <= 2.0:
            raise ValueError("通用长按时间必须在0.5～2.0秒之间。")
        return self._vision_press_relative(
            x, y, action="long_press", hold_seconds=float(hold_seconds)
        )

    def vision_drag_relative(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """Drag between two calibrated visual points through the seller UI."""

        self._require_verified_action("drag", "任意两点拖动")
        values = (start_x, start_y, end_x, end_y)
        if any(not 0 <= value <= 1000 for value in values):
            raise ValueError("拖动视觉坐标必须全部在0～1000之间。")
        if (start_x, start_y) == (end_x, end_y):
            raise ValueError("拖动起点和终点不能相同。")
        hwnd, _title = legacy.find_window(self.title)
        frame = self._capture_phone(hwnd)
        self._consume_physical_execution("drag", frame)
        from tap_calibration import corrected_grid_point

        corrected_start = corrected_grid_point(
            start_x,
            start_y,
            (frame.width, frame.height),
            self.calibration_path,
        )
        corrected_end = corrected_grid_point(
            end_x,
            end_y,
            (frame.width, frame.height),
            self.calibration_path,
        )

        def to_pixel(point: tuple[float, float]) -> tuple[int, int]:
            return (
                min(
                    frame.width - 1,
                    max(0, int(round(point[0] * (frame.width - 1) / 1000))),
                ),
                min(
                    frame.height - 1,
                    max(0, int(round(point[1] * (frame.height - 1) / 1000))),
                ),
            )

        start = to_pixel(corrected_start)
        end = to_pixel(corrected_end)
        if start == end:
            raise ValueError("标定后的拖动起点和终点重合。")
        self._checkpoint()
        legacy.drag_client_path(hwnd, start, end)
        legacy.move_cursor_outside_camera(hwnd)
        return start, end

    def vision_reveal_system_navigation(self) -> dict[str, Any]:
        """Reveal transient system navigation with one locally derived edge path."""

        self._require_verified_action(
            "reveal_system_navigation",
            "系统边缘唤出导航栏",
        )
        hwnd, _title = legacy.find_window(self.title)
        frame = self._capture_phone(hwnd)
        self._consume_physical_execution(
            "reveal_system_navigation", frame
        )
        from tap_calibration import reveal_system_navigation_path

        evidence = reveal_system_navigation_path(
            (frame.width, frame.height),
            self.calibration_path,
        )
        corrected = evidence["corrected_grid"]

        def to_pixel(point: list[int]) -> tuple[int, int]:
            return (
                int(round(point[0] * (frame.width - 1) / 1000)),
                int(round(point[1] * (frame.height - 1) / 1000)),
            )

        start, end = (to_pixel(point) for point in corrected)
        if start == end:
            raise ValueError("系统边缘轨迹纠偏后起终点重合。")
        self._checkpoint()
        legacy.drag_client_path(hwnd, start, end)
        legacy.move_cursor_outside_camera(hwnd)
        return {
            **evidence,
            "client_path": [list(start), list(end)],
        }

    def _vision_press_relative(
        self,
        x: int,
        y: int,
        *,
        action: str,
        hold_seconds: float,
    ) -> tuple[int, int]:
        if not (0 <= x <= 1000 and 0 <= y <= 1000):
            raise ValueError("视觉 Agent 坐标必须在0～1000之间。")
        hwnd, _title = legacy.find_window(self.title)
        frame = self._capture_phone(hwnd)
        self._consume_physical_execution(action, frame)
        # Multi-position calibration corrects camera-to-physical XY distortion.
        # Dedicated Android navigation stays on its independently validated
        # ratios and intentionally does not pass through this transform.
        from tap_calibration import corrected_grid_point

        x, y = corrected_grid_point(
            x,
            y,
            (frame.width, frame.height),
            self.calibration_path,
        )
        point = (
            min(frame.width - 1, max(0, int(round(x * (frame.width - 1) / 1000)))),
            min(frame.height - 1, max(0, int(round(y * (frame.height - 1) / 1000)))),
        )
        self._checkpoint()
        # The seller control keeps the previous 连点次数 value. A stale value
        # of 2 can focus a text box with the first physical tap and then click
        # the old screen coordinate again after the keyboard moves the layout.
        # Every visual-agent tap is one atomic action, so force single-click.
        legacy.configure_single_click_count(hwnd)
        legacy.click_client_point(
            hwnd,
            point[0],
            point[1],
            countdown=0,
            hold_seconds=hold_seconds,
        )
        legacy.move_cursor_outside_camera(hwnd)
        return point

    def _vision_nav_tap(
        self, x_ratio: float, y_ratio: float, *, action: str
    ) -> tuple[int, int]:
        hwnd, _title = legacy.find_window(self.title)
        frame = self._capture_phone(hwnd)
        self._consume_physical_execution(action, frame)
        x_ratio, y_ratio = oriented_navigation_ratio(
            x_ratio,
            y_ratio,
            landscape=frame.width > frame.height,
        )
        point = (
            min(frame.width - 1, max(0, int(round(frame.width * x_ratio)))),
            min(frame.height - 1, max(0, int(round(frame.height * y_ratio)))),
        )
        self._checkpoint()
        # The seller control persists the previous "连点次数" value.
        # Navigation must be one atomic tap just like a visual target tap;
        # otherwise one Back/Home request can issue multiple physical taps and
        # make the observed transition non-deterministic.
        legacy.configure_single_click_count(hwnd)
        legacy.click_client_point(
            hwnd,
            point[0],
            point[1],
            countdown=0,
            hold_seconds=float(
                load_workflow_config()["vision_agent"]["tap_hold"]
            ),
        )
        legacy.move_cursor_outside_camera(hwnd)
        return point

    def vision_android_home(self) -> tuple[int, int]:
        # This is the independently verified Android system Home primitive.
        # It must never inherit ordinary semantic-tap authority because it is
        # not an App/browser "home page" element.
        self._require_verified_action("home", "Android系统Home")
        cfg = load_workflow_config()["vision_agent"]
        return self._vision_nav_tap(
            float(cfg["android_home_x_ratio"]),
            float(cfg["android_home_y_ratio"]),
            action="home",
        )

    def vision_android_back(self) -> tuple[int, int]:
        self._require_verified_action("back", "返回")
        cfg = load_workflow_config()["vision_agent"]
        return self._vision_nav_tap(
            float(cfg["android_back_x_ratio"]),
            float(cfg["android_back_y_ratio"]),
            action="back",
        )

    def _vision_swipe(self, direction: str) -> None:
        self._require_verified_action("swipe", "滑动")
        hwnd, _title = legacy.find_window(self.title)
        frame = self._capture_phone(hwnd)
        self._consume_physical_execution("swipe", frame)
        self._checkpoint()
        legacy.configure_swipe(hwnd, direction)
        legacy.trigger_selected_action(hwnd)
        legacy.move_cursor_outside_camera(hwnd)

    def vision_swipe_up(self) -> None:
        self._vision_swipe("up")

    def vision_swipe_down(self) -> None:
        self._vision_swipe("down")

    def vision_swipe_left(self) -> None:
        self._vision_swipe("left")

    def vision_swipe_right(self) -> None:
        self._vision_swipe("right")

    def vision_type_text(self, text: str) -> None:
        self._require_verified_action("input_verified_text", "输入文字")
        self._validate_verified_text_characters(text)
        # The seller controller's bulk-input command is not exact for digits
        # and symbol pages (a historical real-device run turned "1" into
        # ".com").  The first verified profile therefore uses only the
        # calibrated QWERTY letter path and fails closed for every other text.
        self.vision_type_pinyin(text, text)

    @staticmethod
    def _validate_verified_text_characters(text: str) -> None:
        if not isinstance(text, str) or not re.fullmatch(r"[a-z]{1,30}", text):
            raise WorkflowNotReady(
                "当前设备的安全文字输入仅验收了1～30个小写英文字母；"
                "大写、数字、中文和符号尚未验收。"
            )

    def validate_verified_text(self, text: str, input_states: dict[str, Any]) -> None:
        """Fail before hardware unless the current visual keyboard profile is exact."""

        self._require_verified_action("input_verified_text", "输入文字")
        self._validate_verified_text_characters(text)
        if input_states.get("focused") is not True:
            raise WorkflowNotReady("当前输入框没有可信聚焦证据。")
        if input_states.get("value") != "":
            raise WorkflowNotReady("当前安全文字输入只允许从视觉确认的空输入框开始。")
        if input_states.get("keyboard_layout") != "qwerty":
            raise WorkflowNotReady("当前安全文字输入要求画面确认标准 QWERTY 键盘。")
        if input_states.get("keyboard_input_mode") != "direct_latin":
            raise WorkflowNotReady(
                "当前安全文字输入要求画面确认 direct_latin 英文直输模式；"
                "中文拼音 QWERTY 会产生组合文本。"
            )

    def vision_type_pinyin(
        self,
        text: str,
        pinyin: str,
        keyboard_layout: dict[str, Any] | None = None,
    ) -> None:
        self._require_verified_action("input_verified_text", "输入文字")
        del text
        if not re.fullmatch(r"[a-z]{1,30}", pinyin):
            raise ValueError("拼音必须是1～30个小写英文字母。")
        cfg = load_workflow_config()["vision_agent"]
        configured_keyboard = cfg["pinyin_keyboard"]
        if keyboard_layout is None:
            keyboard_cfg = configured_keyboard
        else:
            if keyboard_layout.get("type") != "qwerty":
                raise WorkflowNotReady("当前键盘不是受支持的标准 QWERTY 布局。")
            keyboard_cfg = qwerty_keyboard_config_from_anchors(
                keyboard_layout.get("anchors"),
                key_hold=float(configured_keyboard["key_hold"]),
                inter_key_wait=float(configured_keyboard["inter_key_wait"]),
            )
        hwnd, _title = legacy.find_window(self.title)
        frame = self._capture_phone(hwnd)
        self._consume_physical_execution("input_verified_text", frame)
        if frame.width < 400 or frame.height < 700:
            raise RobotWorkflowError("键盘画面尺寸异常，拒绝执行拼音点击。")
        legacy.configure_single_click_count(hwnd)
        # Editing the seller control's click-count field changes focus and can
        # leave the physical actuator settling.  Starting the first phone tap
        # immediately after that UI edit caused an intermittent duplicated
        # first letter on real hardware.  Move away from the camera and give
        # the controller/actuator a short deterministic settling window.
        legacy.move_cursor_outside_camera(hwnd)
        self._sleep(float(configured_keyboard.get("pre_key_wait", 0.45)))
        for index, key in enumerate(pinyin):
            self._checkpoint()
            point = qwerty_key_point(frame.width, frame.height, key, keyboard_cfg)
            legacy.click_client_point(
                hwnd,
                point[0],
                point[1],
                countdown=0,
                hold_seconds=float(keyboard_cfg["key_hold"]),
            )
            wait_seconds = float(keyboard_cfg["inter_key_wait"])
            if index == 0:
                wait_seconds = max(
                    wait_seconds,
                    float(configured_keyboard.get("first_key_settle", 0.35)),
                )
            self._sleep(wait_seconds)
        legacy.move_cursor_outside_camera(hwnd)

    def vision_clear_text(
        self,
        keyboard_layout: dict[str, Any] | None = None,
        delete_count: int | None = None,
    ) -> None:
        """Clear a focused phone input field using the visible keyboard.

        The visual model may call this only after it has observed both the
        keyboard and the exact incorrect text. The validated character count
        determines the exact number of physical backspace taps.
        """
        self._require_verified_action("input_verified_text", "输入文字")
        if (
            isinstance(delete_count, bool)
            or not isinstance(delete_count, int)
            or not 1 <= delete_count <= 100
        ):
            raise WorkflowNotReady(
                "退格次数必须是视觉确认后的1～100之间整数，拒绝固定次数清空。"
            )
        cfg = load_workflow_config()["vision_agent"]
        backspace_x_ratio = float(cfg["keyboard_backspace_x_ratio"])
        backspace_y_ratio = float(cfg["keyboard_backspace_y_ratio"])
        if keyboard_layout is not None:
            layout_type = keyboard_layout.get("type")
            anchors = keyboard_layout.get("anchors") or {}
            if layout_type == "qwerty":
                dynamic_cfg = qwerty_keyboard_config_from_anchors(anchors)
                backspace_x_ratio = float(dynamic_cfg["backspace_x_ratio"])
                backspace_y_ratio = float(dynamic_cfg["backspace_y_ratio"])
            elif layout_type == "generic":
                backspace = anchors.get("backspace")
                if (
                    not isinstance(backspace, list)
                    or len(backspace) != 2
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        for value in backspace
                    )
                    or not all(0 <= float(value) <= 1000 for value in backspace)
                ):
                    raise WorkflowNotReady(
                        "非QWERTY键盘必须提供画面中真实可见的退格键中心。"
                    )
                backspace_x_ratio = float(backspace[0]) / 1000.0
                backspace_y_ratio = float(backspace[1]) / 1000.0
            else:
                raise WorkflowNotReady(
                    "当前键盘布局不支持安全退格。"
                )
        hwnd, _title = legacy.find_window(self.title)
        frame = self._capture_phone(hwnd)
        self._consume_physical_execution("input_verified_text", frame)
        point = (
            min(
                frame.width - 1,
                max(
                    0,
                    int(
                        round(
                            frame.width
                            * backspace_x_ratio
                        )
                    ),
                ),
            ),
            min(
                frame.height - 1,
                max(
                    0,
                    int(
                        round(
                            frame.height
                            * backspace_y_ratio
                        )
                    ),
                ),
            ),
        )
        legacy.configure_single_click_count(hwnd)
        for _index in range(delete_count):
            self._checkpoint()
            legacy.click_client_point(
                hwnd,
                point[0],
                point[1],
                countdown=0,
                hold_seconds=0.18,
            )
            self._sleep(0.08)
        legacy.move_cursor_outside_camera(hwnd)

    def _new_run_dir(self, operation: str) -> Path:
        WEB_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_operation = operation.replace(".", "_")
        run_dir = WEB_OUTPUT_DIR / f"{safe_operation}_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    def _save_frame(self, image: Image.Image, path: Path, label: str) -> None:
        result = image.copy()
        draw = ImageDraw.Draw(result)
        draw.rectangle((0, 0, min(result.width, 500), 28), fill=(20, 24, 34))
        draw.text((8, 7), label, fill=(245, 247, 255))
        result.save(path, quality=90)

    def _required_template(self, name: str) -> Path:
        path = template_path(name)
        if not path.is_file():
            raise WorkflowNotReady(
                f"缺少模板 {path.name}。请先按网页“设置”页说明采集。"
            )
        return path

    def _stable_template_match(
        self,
        hwnd: int,
        name: str,
        *,
        timeout: float = 8.0,
        threshold: float = 0.82,
        hold: float = 1.0,
        required_samples: int = 3,
    ) -> tuple[Image.Image, float, int, int, Image.Image]:
        path = self._required_template(name)
        template = Image.open(path).convert("RGB")
        deadline = time.monotonic() + timeout
        stable_since: float | None = None
        samples = 0
        last_center: tuple[int, int] | None = None
        best: tuple[Image.Image, float, int, int, Image.Image] | None = None
        while time.monotonic() < deadline:
            self._checkpoint()
            frame = self._capture_phone(hwnd)
            score, x, y = legacy.match_template(frame, template)
            center = (x + template.width // 2, y + template.height // 2)
            now = time.monotonic()
            if score >= threshold and (
                last_center is None
                or (abs(center[0] - last_center[0]) <= 6 and abs(center[1] - last_center[1]) <= 6)
            ):
                if stable_since is None:
                    stable_since = now
                    samples = 1
                else:
                    samples += 1
                last_center = center
                best = (frame, score, x, y, template)
                if samples >= required_samples and now - stable_since >= hold:
                    return best
            else:
                stable_since = None
                samples = 0
                last_center = None
            self._sleep(0.2)
        raise UnsafePageError(
            f"等待 {timeout:.1f} 秒仍未稳定识别到 {path.name}，拒绝继续操作。"
        )

    def _click_template(
        self,
        hwnd: int,
        name: str,
        *,
        timeout: float = 8.0,
        threshold: float = 0.82,
        hold: float = 0.35,
    ) -> tuple[int, int]:
        _frame, _score, x, y, template = self._stable_template_match(
            hwnd, name, timeout=timeout, threshold=threshold
        )
        center = (x + template.width // 2, y + template.height // 2)
        self._checkpoint()
        legacy.click_client_point(
            hwnd,
            center[0],
            center[1],
            countdown=0,
            hold_seconds=hold,
        )
        return center

    @staticmethod
    def _match_in_region(
        matches: list[OcrMatch],
        width: int,
        height: int,
        region: tuple[float, float, float, float],
    ) -> list[OcrMatch]:
        left, top, right, bottom = region
        return [
            match
            for match in matches
            if left * width <= match.center[0] <= right * width
            and top * height <= match.center[1] <= bottom * height
        ]

    def _stable_wechat_page(
        self,
        hwnd: int,
        *,
        timeout: float,
        required_samples: int = 2,
    ) -> tuple[Image.Image, dict[str, Any], str]:
        cfg = load_workflow_config()["wechat"]
        language = str(cfg["ocr_language"])
        scale = float(cfg["ocr_scale"])
        deadline = time.monotonic() + timeout
        last_status: str | None = None
        stable_samples = 0
        last_result: tuple[Image.Image, dict[str, Any], str] | None = None

        # The seller preview paints a black PX/MM tooltip at the most recent
        # camera click. It can cover WeChat's title and make OCR see only the
        # tooltip. Moving the desktop pointer to the title bar clears that
        # obstruction without operating the phone.
        legacy.move_cursor_outside_camera(hwnd)
        while time.monotonic() < deadline:
            self._checkpoint()
            frame = self._capture_phone(hwnd)
            payload = recognize_ocr(frame, language, scale=scale)
            status = classify_wechat_page(
                payload,
                frame.width,
                frame.height,
            )
            if status == "unknown":
                # The seller software leaves an opaque PX/MM overlay over the
                # left half of WeChat's title. OCR the still-visible title
                # suffix separately; this is used only for the exact
                # 文件传输助手 marker, never for arbitrary contacts.
                title_left = int(round(frame.width * 0.32))
                title_bottom = int(round(frame.height * 0.12))
                title_crop = frame.crop(
                    (title_left, 0, frame.width, title_bottom)
                )
                title_payload = recognize_ocr(
                    title_crop,
                    language,
                    scale=scale,
                )
                if classify_obscured_wechat_title(title_payload):
                    status = "target_chat"
            if status != "unknown" and status == last_status:
                stable_samples += 1
            elif status != "unknown":
                stable_samples = 1
            else:
                stable_samples = 0
            last_status = status
            last_result = (frame, payload, status)
            if status != "unknown" and stable_samples >= required_samples:
                return last_result
            self._sleep(0.25)

        final_status = last_result[2] if last_result else "unknown"
        raise UnsafePageError(
            f"等待 {timeout:.1f} 秒仍未稳定识别微信页面"
            f"（最后状态：{final_status}），拒绝继续操作。"
        )

    def _stable_ocr_match(
        self,
        hwnd: int,
        target: str,
        *,
        timeout: float = 10.0,
        region: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
        required_samples: int = 2,
        max_center_delta: int = 18,
    ) -> tuple[Image.Image, OcrMatch, dict[str, Any]]:
        cfg = load_workflow_config()["wechat"]
        language = str(cfg["ocr_language"])
        scale = float(cfg["ocr_scale"])
        deadline = time.monotonic() + timeout
        stable_samples = 0
        last_center: tuple[int, int] | None = None
        last_result: tuple[Image.Image, OcrMatch, dict[str, Any]] | None = None

        while time.monotonic() < deadline:
            self._checkpoint()
            frame = self._capture_phone(hwnd)
            payload = recognize_ocr(frame, language, scale=scale)
            matches = self._match_in_region(
                find_ocr_text(payload, target),
                frame.width,
                frame.height,
                region,
            )
            if matches:
                match = min(matches, key=lambda item: (item.top, item.left))
                center = match.center
                if last_center is not None and (
                    abs(center[0] - last_center[0]) <= max_center_delta
                    and abs(center[1] - last_center[1]) <= max_center_delta
                ):
                    stable_samples += 1
                else:
                    stable_samples = 1
                last_center = center
                last_result = (frame, match, payload)
                if stable_samples >= required_samples:
                    return last_result
            else:
                stable_samples = 0
                last_center = None
                last_result = None
            self._sleep(0.25)

        raise UnsafePageError(
            f"等待 {timeout:.1f} 秒仍未稳定识别到文字“{target}”，拒绝继续操作。"
        )

    def _click_ocr_text(
        self,
        hwnd: int,
        target: str,
        *,
        timeout: float = 10.0,
        region: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
        offset: tuple[int, int] = (0, 0),
        hold: float = 0.35,
    ) -> tuple[int, int]:
        frame, match, _payload = self._stable_ocr_match(
            hwnd,
            target,
            timeout=timeout,
            region=region,
        )
        point = (
            max(0, min(frame.width - 1, match.center[0] + offset[0])),
            max(0, min(frame.height - 1, match.center[1] + offset[1])),
        )
        self._checkpoint()
        legacy.click_client_point(
            hwnd,
            point[0],
            point[1],
            countdown=0,
            hold_seconds=hold,
        )
        return point

    def _invoke_seller_input(self, hwnd: int, text: str) -> None:
        self._require_verified_action("input_verified_text", "输入文字")
        if not text or len(text) > 100:
            raise ValueError("文字长度必须在 1～100 个字符之间。")
        if "\n" in text or "\r" in text or not ALLOWED_TEXT_RE.fullmatch(text):
            raise ValueError(
                "第一版仅支持中文、大小写字母、数字、空格和常用中英文标点。"
            )

        _set_clipboard_text(text)
        existing = _visible_owned_windows(hwnd)
        legacy.ensure_window_fully_visible(hwnd)
        _left, _top, width, height = legacy.client_geometry(hwnd)
        control_x, control_y = legacy.seller_control_point(
            width,
            height,
            legacy.BASELINE_CLIENT_WIDTH - INPUT_BUTTON_X_FROM_RIGHT,
            CONTROL_Y_FROM_BOTTOM,
        )
        legacy.click_client_control(
            hwnd,
            control_x,
            control_y,
        )

        deadline = time.monotonic() + 3.0
        dialog: int | None = None
        while time.monotonic() < deadline:
            self._checkpoint()
            new_windows = _visible_owned_windows(hwnd) - existing
            if new_windows:
                dialog = next(iter(new_windows))
                break
            self._sleep(0.1)
        if dialog is None:
            raise RobotWorkflowError(
                "点击卖家控制端“输入”后没有发现输入对话框，已停止。"
            )

        legacy.user32.ShowWindow(dialog, legacy.SW_RESTORE)
        legacy.user32.SetForegroundWindow(dialog)
        self._sleep(0.2)
        legacy.user32.keybd_event(legacy.VK_CONTROL, 0, 0, 0)
        legacy.press_virtual_key(0x56)  # V
        legacy.user32.keybd_event(
            legacy.VK_CONTROL, 0, legacy.KEYEVENTF_KEYUP, 0
        )
        legacy.press_virtual_key(legacy.VK_RETURN)

        close_deadline = time.monotonic() + 3.0
        while time.monotonic() < close_deadline:
            self._checkpoint()
            if not legacy.user32.IsWindow(dialog) or not legacy.user32.IsWindowVisible(dialog):
                return
            self._sleep(0.1)
        raise RobotWorkflowError(
            f"卖家输入对话框未关闭（标题“{_window_title(dialog)}”），拒绝继续。"
        )

    def execute(self, operation: str, params: dict[str, Any]) -> dict[str, Any]:
        del operation, params
        raise WorkflowNotReady(
            "旧多步 workflow 入口未绑定独立方向凭据且破坏一次动作语义，已禁用。"
        )

    def like_current_douyin(self, _params: dict[str, Any]) -> dict[str, Any]:
        raise WorkflowNotReady(
            "已废弃的 App 多步流程入口已禁用；必须使用通用单动作闭环。"
        )
        self._require_verified_action("tap_semantic", "点击")
        ready = workflow_readiness()["douyin_like"]
        if not ready["ready"]:
            raise WorkflowNotReady("抖音首页模板尚未采集。")
        cfg = load_workflow_config()["douyin"]
        run_dir = self._new_run_dir("douyin.like_current")
        hwnd, title = legacy.find_window(self.title)
        legacy.configure_single_click_count(hwnd)
        self._checkpoint()
        (
            before,
            home_score,
            _home_x,
            _home_y,
            _home_template,
            page_status,
            ready_wait,
        ) = legacy.capture_ready_douyin_page(
            hwnd,
            legacy.DEFAULT_CAMERA_HEIGHT,
            legacy.DEFAULT_DOUYIN_TEMPLATE,
            float(cfg["threshold"]),
            float(cfg["page_ready_timeout"]),
            float(cfg["classify_hold"]),
        )
        detection = legacy.detect_douyin_heart(before)
        before_path = run_dir / "before.jpg"
        legacy.annotate_heart_detection(before, detection, before_path)
        if page_status != "normal":
            raise UnsafePageError(
                f"当前页面分类为 {page_status}，不是普通视频；未执行点赞。"
            )
        if detection.state == "liked":
            return {
                "changed": False,
                "already_liked": True,
                "page_status": page_status,
                "ready_wait_seconds": round(ready_wait, 3),
                "home_score": round(home_score, 4),
                "evidence": [str(before_path)],
                "window_title": title,
            }
        if detection.state != "unliked" or not detection.center:
            raise UnsafePageError("当前普通视频没有稳定白色爱心，拒绝点击。")

        self._checkpoint()
        legacy.click_client_point(
            hwnd,
            detection.center[0],
            detection.center[1],
            countdown=0,
            hold_seconds=float(cfg["tap_hold"]),
        )
        legacy.move_cursor_outside_camera(hwnd)
        (
            after,
            after_detection,
            verified,
            verification_method,
            verification_seconds,
            transition_metrics,
        ) = legacy.wait_for_liked_heart(
            hwnd,
            legacy.DEFAULT_CAMERA_HEIGHT,
            before,
            detection.center,
            float(cfg["verify_wait"]),
            float(cfg["verify_timeout"]),
        )
        after_path = run_dir / "after.jpg"
        legacy.annotate_heart_detection(after, after_detection, after_path)
        if not verified:
            raise RobotWorkflowError(
                f"点击后等待 {verification_seconds:.1f} 秒仍未确认爱心变红。"
            )
        return {
            "changed": True,
            "already_liked": False,
            "page_status": page_status,
            "ready_wait_seconds": round(ready_wait, 3),
            "verification_method": verification_method,
            "verification_seconds": round(verification_seconds, 3),
            "transition_metrics": transition_metrics,
            "evidence": [str(before_path), str(after_path)],
            "window_title": title,
        }

    def comment_current_douyin(self, params: dict[str, Any]) -> dict[str, Any]:
        raise WorkflowNotReady(
            "已废弃的 App 多步流程入口已禁用；必须使用通用单动作闭环。"
        )
        self._require_verified_action("input_verified_text", "输入文字")
        self._require_verified_action("tap_semantic", "点击")
        text = str(params.get("text", "")).strip()
        ready = workflow_readiness()["douyin_comment"]
        if not ready["ready"]:
            raise WorkflowNotReady(
                "抖音评论工作流尚未就绪，缺少：" + "、".join(ready["missing_templates"])
            )
        cfg = load_workflow_config()["douyin"]
        run_dir = self._new_run_dir("douyin.comment_current")
        hwnd, title = legacy.find_window(self.title)
        (
            page,
            _score,
            _x,
            _y,
            _template,
            page_status,
            ready_wait,
        ) = legacy.capture_ready_douyin_page(
            hwnd,
            legacy.DEFAULT_CAMERA_HEIGHT,
            legacy.DEFAULT_DOUYIN_TEMPLATE,
            float(cfg["threshold"]),
            float(cfg["page_ready_timeout"]),
            float(cfg["classify_hold"]),
        )
        self._save_frame(page, run_dir / "01_page.jpg", "stable normal page")
        if page_status != "normal":
            raise UnsafePageError(
                f"当前页面分类为 {page_status}，不是普通视频；未执行评论。"
            )
        self._click_template(hwnd, "douyin_comment_button")
        self._click_template(hwnd, "douyin_comment_input")
        self._checkpoint()
        self._invoke_seller_input(hwnd, text)
        self._sleep(max(2.0, len(text) * 0.8))
        before_send = self._capture_phone(hwnd)
        self._save_frame(before_send, run_dir / "02_before_send.jpg", "before send")
        send_point = self._click_template(hwnd, "douyin_comment_send")
        self._sleep(1.5)
        after_send = self._capture_phone(hwnd)
        self._save_frame(after_send, run_dir / "03_after_send.jpg", "after send")
        change = legacy.image_change_score(before_send, after_send)
        if change < 0.008:
            raise RobotWorkflowError(
                "点击评论发送后画面没有足够变化，无法确认发送成功。"
            )
        return {
            "changed": True,
            "page_status": page_status,
            "ready_wait_seconds": round(ready_wait, 3),
            "send_point": list(send_point),
            "visual_change": round(change, 4),
            "evidence": [
                str(run_dir / "01_page.jpg"),
                str(run_dir / "02_before_send.jpg"),
                str(run_dir / "03_after_send.jpg"),
            ],
            "window_title": title,
        }

    def send_wechat_text(self, params: dict[str, Any]) -> dict[str, Any]:
        raise WorkflowNotReady(
            "已废弃的 App 多步流程入口已禁用；必须使用通用单动作闭环。"
        )
        self._require_verified_action("input_verified_text", "输入文字")
        self._require_verified_action("tap_semantic", "点击")
        text = str(params.get("text", "")).strip()
        ready = workflow_readiness()["wechat"]
        if not ready["ready"]:
            raise WorkflowNotReady(
                "微信工作流尚未就绪，缺少运行时中文 OCR。"
            )
        cfg = load_workflow_config()["wechat"]
        ocr_timeout = float(cfg["ocr_timeout"])
        run_dir = self._new_run_dir("wechat.send_text_to_file_transfer")
        hwnd, title = legacy.find_window(self.title)

        # Always start from Android home. Besides making the navigation
        # deterministic, this avoids trusting a chat title that may have been
        # obscured by the seller preview's coordinate overlay.
        legacy.configure_single_click_count(hwnd)
        initial_frame = self._capture_phone(hwnd)
        android_home_point = (
            int(round(initial_frame.width * float(cfg["android_home_x_ratio"]))),
            int(round(initial_frame.height * float(cfg["android_home_y_ratio"]))),
        )
        self._checkpoint()
        legacy.click_client_point(
            hwnd,
            android_home_point[0],
            android_home_point[1],
            countdown=0,
            hold_seconds=0.25,
        )
        legacy.move_cursor_outside_camera(hwnd)
        self._sleep(1.2)

        current_frame, _payload, page_status = self._stable_wechat_page(
            hwnd,
            timeout=ocr_timeout,
        )
        if page_status != "android_home":
            raise UnsafePageError(
                f"点击手机主页键后页面分类为 {page_status}，"
                "未确认回到安卓桌面，拒绝继续操作。"
            )

        self._click_ocr_text(
            hwnd,
            "微信",
            timeout=ocr_timeout,
            region=(0.02, 0.08, 0.98, 0.92),
            offset=(0, int(cfg["home_icon_y_offset"])),
        )
        legacy.move_cursor_outside_camera(hwnd)
        self._sleep(2.0)
        current_frame, _payload, page_status = self._stable_wechat_page(
            hwnd,
            timeout=ocr_timeout,
        )

        if page_status == "conversation_list":
            self._click_ocr_text(
                hwnd,
                "文件传输助手",
                timeout=ocr_timeout,
                region=(0.0, 0.08, 1.0, 0.82),
            )
            legacy.move_cursor_outside_camera(hwnd)
            self._sleep(1.0)
            current_frame, _payload, page_status = self._stable_wechat_page(
                hwnd,
                timeout=ocr_timeout,
            )

        if page_status != "target_chat":
            raise UnsafePageError(
                f"当前微信页面分类为 {page_status}，"
                "未确认进入文件传输助手，拒绝输入。"
            )

        verified_chat = current_frame
        self._save_frame(
            verified_chat, run_dir / "01_verified_chat.jpg", "verified chat title"
        )

        # WeChat's input field has no stable text before focus. Locate it by
        # the phone page geometry only after the target title is verified.
        input_point = (
            int(round(verified_chat.width * float(cfg["input_x_ratio"]))),
            int(round(verified_chat.height * float(cfg["input_y_ratio"]))),
        )
        self._checkpoint()
        legacy.click_client_point(
            hwnd,
            input_point[0],
            input_point[1],
            countdown=0,
            hold_seconds=0.25,
        )
        self._sleep(0.6)
        self._invoke_seller_input(hwnd, text)
        legacy.move_cursor_outside_camera(hwnd)
        self._sleep(max(2.0, len(text) * float(cfg["input_wait_per_character"])))

        # Re-verify the contact immediately before the irreversible send.
        _guard_frame, _guard_payload, guard_status = self._stable_wechat_page(
            hwnd,
            timeout=4.0,
        )
        if guard_status != "target_chat":
            raise UnsafePageError(
                f"发送前微信页面变为 {guard_status}，拒绝点击发送。"
            )
        before_send = self._capture_phone(hwnd)
        self._save_frame(before_send, run_dir / "02_before_send.jpg", "before send")
        send_point = self._click_ocr_text(
            hwnd,
            "发送",
            timeout=ocr_timeout,
            region=(0.45, 0.48, 1.0, 0.98),
        )
        legacy.move_cursor_outside_camera(hwnd)
        self._sleep(1.5)
        after_send = self._capture_phone(hwnd)
        self._save_frame(after_send, run_dir / "03_after_send.jpg", "after send")
        _after_frame, _after_payload, after_status = self._stable_wechat_page(
            hwnd,
            timeout=3.0,
        )
        if after_status != "target_chat":
            raise UnsafePageError(
                f"发送后微信页面变为 {after_status}，无法确认结果。"
            )
        change = legacy.image_change_score(before_send, after_send)
        if change < 0.006:
            raise RobotWorkflowError(
                "点击发送后画面变化不足，无法确认文件传输助手收到新消息。"
            )
        return {
            "changed": True,
            "android_home_point": list(android_home_point),
            "input_point": list(input_point),
            "send_point": list(send_point),
            "visual_change": round(change, 4),
            "evidence": [
                str(run_dir / "01_verified_chat.jpg"),
                str(run_dir / "02_before_send.jpg"),
                str(run_dir / "03_after_send.jpg"),
            ],
            "window_title": title,
        }


class MockRobotController(RobotController):
    """No-hardware controller for API tests and UI demonstrations."""

    def __init__(
        self,
        *,
        verified_actions: set[str] | frozenset[str] | None = None,
        device_id: str = "mock-default",
    ) -> None:
        all_actions = {
            "tap_semantic",
            "dismiss_overlay",
            "swipe",
            "back",
            "home",
            "wait_for_change",
            "input_verified_text",
            "long_press",
            "drag",
            "reveal_system_navigation",
        }
        super().__init__(
            title="MOCK",
            verified_actions=(
                all_actions if verified_actions is None else verified_actions
            ),
            device_id=device_id,
        )
        self.executions: list[dict[str, Any]] = []

    def device_status(self) -> dict[str, Any]:
        return {
            "controller_online": True,
            "camera_online": True,
            "window_title": "MOCK 智联新途机械臂控制端",
            "client_size": [540, 1038],
            "stop_requested": self.stop_event.is_set(),
            "busy": self.operation_lock.locked(),
            "error": None,
            "readiness": {
                "wechat": {"ready": True, "missing_templates": []},
                "douyin_like": {"ready": True, "missing_templates": []},
                "douyin_comment": {"ready": True, "missing_templates": []},
            },
        }

    def capture_preview(self, quality: int = 72) -> bytes:
        from io import BytesIO

        image = Image.new("RGB", (540, 960), (18, 24, 38))
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((68, 90, 472, 870), radius=42, fill=(31, 40, 58))
        draw.text((174, 450), "MOCK PHONE", fill=(228, 234, 247))
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        return buffer.getvalue()

    def vision_capture(self) -> Image.Image:
        from io import BytesIO

        return Image.open(BytesIO(self.capture_preview())).convert("RGB")

    def _consume_mock_execution(self, action: str) -> None:
        self._consume_physical_execution(action, self.vision_capture())

    def vision_tap_relative(self, x: int, y: int) -> tuple[int, int]:
        self._consume_mock_execution("tap_semantic")
        self.executions.append({"action": "tap", "coordinate": [x, y]})
        return x, y

    def vision_dismiss_overlay_relative(self, x: int, y: int) -> tuple[int, int]:
        self._consume_mock_execution("dismiss_overlay")
        self.executions.append({"action": "dismiss_overlay", "coordinate": [x, y]})
        return x, y

    def vision_long_press_relative(
        self,
        x: int,
        y: int,
        hold_seconds: float = 0.8,
    ) -> tuple[int, int]:
        self._consume_mock_execution("long_press")
        self.executions.append(
            {
                "action": "long_press",
                "coordinate": [x, y],
                "hold_seconds": hold_seconds,
            }
        )
        return x, y

    def vision_drag_relative(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        self._consume_mock_execution("drag")
        self.executions.append(
            {
                "action": "drag",
                "start": [start_x, start_y],
                "end": [end_x, end_y],
            }
        )
        return (start_x, start_y), (end_x, end_y)

    def vision_reveal_system_navigation(self) -> dict[str, Any]:
        self._require_verified_action(
            "reveal_system_navigation",
            "系统边缘唤出导航栏",
        )
        self._consume_mock_execution("reveal_system_navigation")
        evidence = {
            "action": "reveal_system_navigation",
            "edge": "bottom",
            "frame_size": [540, 960],
            "dom_path": [[0.5, 0.95], [0.5, 0.70]],
            "requested_grid": [[100, 500], [350, 500]],
            "corrected_grid": [[100, 500], [350, 500]],
            "client_path": [[54, 480], [189, 480]],
            "mock": True,
        }
        self.executions.append(dict(evidence))
        return evidence

    def vision_android_home(self) -> tuple[int, int]:
        self._consume_mock_execution("home")
        self.executions.append({"action": "android_home"})
        return 500, 976

    def vision_android_back(self) -> tuple[int, int]:
        self._consume_mock_execution("back")
        self.executions.append({"action": "android_back"})
        return 910, 976

    def vision_swipe_up(self) -> None:
        self._consume_mock_execution("swipe")
        self.executions.append({"action": "swipe_up"})

    def vision_swipe_down(self) -> None:
        self._consume_mock_execution("swipe")
        self.executions.append({"action": "swipe_down"})

    def vision_swipe_left(self) -> None:
        self._consume_mock_execution("swipe")
        self.executions.append({"action": "swipe_left"})

    def vision_swipe_right(self) -> None:
        self._consume_mock_execution("swipe")
        self.executions.append({"action": "swipe_right"})

    def vision_type_text(self, text: str) -> None:
        self._consume_mock_execution("input_verified_text")
        self.executions.append({"action": "type_text", "text": text})

    def vision_type_pinyin(
        self,
        text: str,
        pinyin: str,
        keyboard_layout: dict[str, Any] | None = None,
    ) -> None:
        self._consume_mock_execution("input_verified_text")
        record: dict[str, Any] = {
            "action": "type_pinyin",
            "text": text,
            "pinyin": pinyin,
        }
        if keyboard_layout is not None:
            record["keyboard_layout"] = keyboard_layout
        self.executions.append(record)

    def vision_clear_text(
        self,
        keyboard_layout: dict[str, Any] | None = None,
        delete_count: int | None = None,
    ) -> None:
        self._consume_mock_execution("input_verified_text")
        record: dict[str, Any] = {
            "action": "clear_text",
            "delete_count": delete_count,
        }
        if keyboard_layout is not None:
            record["keyboard_layout"] = keyboard_layout
        self.executions.append(record)

    def execute(self, operation: str, params: dict[str, Any]) -> dict[str, Any]:
        del operation, params
        raise WorkflowNotReady(
            "旧多步 workflow 入口未绑定独立方向凭据且破坏一次动作语义，已禁用。"
        )
