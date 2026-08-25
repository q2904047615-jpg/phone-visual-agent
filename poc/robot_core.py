from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

import robot_gui_poc as seller_gui
from orientation_safety import (
    OrientationCredential,
    PhysicalExecutionGate,
)
from verified_text_transaction import (
    MAX_DIRECT_LATIN_SEGMENT_CHARS,
    VerifiedTextTransactionError,
    plan_next_verified_input,
)


WEB_OUTPUT_DIR = seller_gui.OUTPUT_DIR / "web"
CONTROL_CONFIG_PATH = Path(__file__).with_name("controller_config.json")

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
    return seller_gui.seller_layout_has_full_camera(width, height)


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


DEFAULT_CONTROLLER_CONFIG: dict[str, Any] = {
    "tap_hold": 0.35,
    "android_home_x_ratio": 0.50,
    "android_home_y_ratio": 0.976,
    "android_back_x_ratio": 0.685,
    "android_back_y_ratio": 0.976,
    "keyboard_backspace_x_ratio": 0.862,
    "keyboard_backspace_y_ratio": 0.844,
    "pinyin_keyboard": {
        "rows": [
            {"keys": "qwertyuiop", "x_start": 0.115, "x_step": 0.0844, "y": 0.704},
            {"keys": "asdfghjkl", "x_start": 0.157, "x_step": 0.0844, "y": 0.773},
            {"keys": "zxcvbnm", "x_start": 0.241, "x_step": 0.0844, "y": 0.844},
        ],
        "key_hold": 0.18,
        "inter_key_wait": 0.12,
        "pre_key_wait": 0.45,
        "first_key_settle": 0.35,
    },
    "digit_long_press_hold": 0.78,
}


class RobotWorkflowError(RuntimeError):
    """A user-facing workflow failure that must stop further physical actions."""


class WorkflowNotReady(RobotWorkflowError):
    """Required local calibration/template data is missing."""


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(base))
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_controller_config() -> dict[str, Any]:
    if not CONTROL_CONFIG_PATH.exists():
        return json.loads(json.dumps(DEFAULT_CONTROLLER_CONFIG))
    try:
        raw = json.loads(CONTROL_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkflowNotReady(f"控制器配置损坏：{exc}") from exc
    if not isinstance(raw, dict):
        raise WorkflowNotReady("控制器配置必须是 JSON 对象。")
    return _deep_merge(DEFAULT_CONTROLLER_CONFIG, raw)


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
    # A complete audited keyboard can legitimately place its bottom row close
    # to the lower edge of the normalized frame.  Keep a small center margin,
    # while leaving row order and spacing as the authoritative geometry checks.
    if not (450.0 <= top_y < middle_y < bottom_y <= 980.0):
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


class RobotController:
    """Safe, single-machine adapter used by the local web task worker."""

    def __init__(
        self,
        title: str = seller_gui.DEFAULT_WINDOW_TITLE,
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
        self._stop_state_lock = threading.Lock()
        self.operation_lock = threading.Lock()
        # The browser MJPEG preview and the vision worker can otherwise call
        # the seller window capture routine at the same time. On Windows that
        # occasionally returns a transient, truncated client bitmap.
        self.capture_lock = threading.RLock()
        self._last_click_receipt: dict[str, Any] | None = None
        self._last_long_press_receipt: dict[str, Any] | None = None
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
            "double_tap",
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
                "double_tap",
                "long_press",
                "drag",
                "reveal_system_navigation",
            )
        }

    def hardware_capability_profile(self) -> dict[str, Any]:
        """Return typed device limits without claiming undocumented ACKs."""

        enabled = self.hardware_capabilities()
        actions: dict[str, dict[str, Any]] = {}
        for action, available in enabled.items():
            actions[action] = {
                "enabled": bool(available),
                "one_physical_action_per_receipt": action != "wait_for_change",
                "fresh_visual_postcondition_required": True,
                "transport_ack": "gui_event_barrier"
                if action
                in {
                    "tap_semantic",
                    "dismiss_overlay",
                    "back",
                    "home",
                    "double_tap",
                    "long_press",
                }
                else "local_call_return",
                "mechanical_contact_ack": False,
            }
        actions["long_press"]["duration_ms"] = {"min": 500, "max": 2000}
        actions["drag"]["duration_ms"] = {"fixed": 800}
        actions["input_verified_text"]["text"] = {
            "canonical_max_chars": 4000,
            "max_chars_per_physical_step": MAX_DIRECT_LATIN_SEGMENT_CHARS,
            "direct_latin_transport": "audited_visible_key_sequence",
            "non_lowercase_transport": "audited_visible_key_sequence",
            "max_fields": 32,
            "max_targets": 32,
            "segments": ["direct_latin", "chinese_pinyin", "visible_literal_key"],
            "newline": "visible_multiline_enter_key_only",
            "unsupported_character_policy": "structured_capability_gap",
        }
        actions["clear_verified_text"] = {
            **actions["input_verified_text"],
            "enabled": bool(enabled.get("input_verified_text")),
            "layouts": ["qwerty", "numeric", "symbol", "generic_visible_backspace"],
            "verified_delete_count": {"min": 1, "max": 100},
        }
        actions["double_tap"].update(
            {
                "enabled": bool(enabled.get("double_tap")),
                "transport": "seller_click_count_two_atomic_request",
                "canonical_action_count": 1,
                "contact_count": 2,
                "restores_click_count_to": 1,
                "gap_reason": (
                    None
                    if enabled.get("double_tap")
                    else "requires_double_tap_live_acceptance"
                ),
            }
        )
        actions["press_enter"] = {
            "enabled": bool(enabled.get("tap_semantic")),
            "primitive": "vision_tap_relative",
            "requires": [
                "fresh_visible_enter_key",
                "key_action_newline",
                "active_multiline_input_field",
                "exact_post_action_input_value",
            ],
        }
        actions["pinch"] = {
            "enabled": False,
            "gap_reason": "multi_touch_not_supported_by_single_contact_robot",
        }
        actions["hardware_key"] = {
            "enabled": False,
            "gap_reason": "hardware_key_transport_not_verified",
        }
        return {
            "protocol_version": "2026-08-18-device-capability-profile-v1",
            "device_id": self.device_id,
            "actions": actions,
        }

    def consume_last_long_press_receipt(self) -> dict[str, Any] | None:
        receipt = self._last_long_press_receipt
        self._last_long_press_receipt = None
        return dict(receipt) if receipt is not None else None

    def consume_last_click_receipt(self) -> dict[str, Any] | None:
        receipt = self._last_click_receipt
        self._last_click_receipt = None
        return dict(receipt) if receipt is not None else None

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
        with self._stop_state_lock:
            self.stop_event.set()

    def begin_new_task(self) -> None:
        """Acknowledge stop requests that predate this new task boundary."""

        with self._stop_state_lock:
            self.stop_event.clear()

    def _checkpoint(self) -> None:
        if self.stop_event.is_set():
            raise RobotWorkflowError("用户已请求停止任务。")
        seller_gui._check_escape()

    def _sleep(self, seconds: float) -> None:
        deadline = time.monotonic() + max(0.0, seconds)
        while time.monotonic() < deadline:
            self._checkpoint()
            remaining = deadline - time.monotonic()
            self.stop_event.wait(min(0.1, max(0.0, remaining)))
        self._checkpoint()

    def device_status(self) -> dict[str, Any]:
        try:
            hwnd, title = seller_gui.find_window(self.title)
            _left, _top, width, height = seller_gui.client_geometry(hwnd)
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
        }

    def _capture_phone(self, hwnd: int) -> Image.Image:
        with self.capture_lock:
            return seller_gui.camera_crop(
                seller_gui.capture_client(hwnd), seller_gui.DEFAULT_CAMERA_HEIGHT
            )

    def _capture_phone_passive(self, hwnd: int) -> Image.Image:
        with self.capture_lock:
            return seller_gui.camera_crop(
                seller_gui.capture_client_passive(hwnd), seller_gui.DEFAULT_CAMERA_HEIGHT
            )

    def capture_preview(self, quality: int = 72) -> bytes:
        hwnd, _title = seller_gui.find_window(self.title)
        image = self._capture_phone_passive(hwnd)
        from io import BytesIO

        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
        return buffer.getvalue()

    def vision_capture(self) -> Image.Image:
        """Capture exactly the phone region seen by the visual agent."""
        hwnd, _title = seller_gui.find_window(self.title)
        with seller_gui.temporarily_park_cursor_outside_camera(hwnd):
            return self._capture_phone(hwnd)

    def vision_tap_relative(self, x: int, y: int) -> tuple[int, int]:
        """Tap a Qwen3-VL coordinate expressed on a 1000×1000 grid."""
        self._require_verified_action("tap_semantic", "点击")
        return self._vision_press_relative(
            x,
            y,
            action="tap_semantic",
            hold_seconds=float(load_controller_config()["tap_hold"]),
        )

    def resolve_calibrated_target_grid_point(
        self,
        x: int,
        y: int,
        target_bounds: tuple[float, float, float, float],
        frame_size: tuple[int, int],
    ) -> tuple[int, int]:
        """Resolve one dual-audited local target inside measured coverage."""

        from tap_calibration import resolve_target_grid_point_within_calibration

        return resolve_target_grid_point_within_calibration(
            x,
            y,
            target_bounds,
            frame_size,
            self.calibration_path,
        )

    def vision_dismiss_overlay_relative(self, x: int, y: int) -> tuple[int, int]:
        """Dismiss one observed overlay through its separately verified entry."""

        self._require_verified_action("dismiss_overlay", "关闭弹层")
        return self._vision_press_relative(
            x,
            y,
            action="dismiss_overlay",
            hold_seconds=float(load_controller_config()["tap_hold"]),
        )

    def vision_double_tap_relative(self, x: int, y: int) -> tuple[int, int]:
        """Execute one canonical double-tap through seller click-count two."""

        self._require_verified_action("double_tap", "双击")
        return self._vision_press_relative(
            x,
            y,
            action="double_tap",
            hold_seconds=float(load_controller_config()["tap_hold"]),
            click_count=2,
        )

    def vision_long_press_relative(
        self,
        x: int,
        y: int,
        hold_seconds: float = 0.8,
    ) -> tuple[int, int]:
        """Long-press one calibrated visual target without changing its point."""

        self._require_verified_action("long_press", "长按")
        self._last_long_press_receipt = None
        if not 0.5 <= float(hold_seconds) <= 2.0:
            raise ValueError("通用长按时间必须在0.5～2.0秒之间。")
        if not (0 <= x <= 1000 and 0 <= y <= 1000):
            raise ValueError("视觉 Agent 坐标必须在0～1000之间。")
        hwnd, _title = seller_gui.find_window(self.title)
        frame = self._capture_phone(hwnd)
        self._consume_physical_execution("long_press", frame)
        from tap_calibration import corrected_grid_point

        corrected_x, corrected_y = corrected_grid_point(
            x,
            y,
            (frame.width, frame.height),
            self.calibration_path,
        )
        point = (
            min(
                frame.width - 1,
                max(0, int(round(corrected_x * (frame.width - 1) / 1000))),
            ),
            min(
                frame.height - 1,
                max(0, int(round(corrected_y * (frame.height - 1) / 1000))),
            ),
        )
        self._checkpoint()
        receipt = seller_gui.long_press_client_point(
            hwnd,
            point[0],
            point[1],
            hold_seconds=float(hold_seconds),
        )
        if not isinstance(receipt, dict):
            raise RuntimeError("控制端没有返回长按事件栅栏凭据。")
        self._last_long_press_receipt = dict(receipt)
        seller_gui.clear_seller_camera_overlay(hwnd)
        return point

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
        hwnd, _title = seller_gui.find_window(self.title)
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
        seller_gui.drag_client_path(hwnd, start, end)
        seller_gui.clear_seller_camera_overlay(hwnd)
        return start, end

    def vision_reveal_system_navigation(self) -> dict[str, Any]:
        """Reveal transient system navigation with one locally derived edge path."""

        self._require_verified_action(
            "reveal_system_navigation",
            "系统边缘唤出导航栏",
        )
        hwnd, _title = seller_gui.find_window(self.title)
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
        seller_gui.drag_client_path(hwnd, start, end)
        seller_gui.clear_seller_camera_overlay(hwnd)
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
        click_count: int = 1,
    ) -> tuple[int, int]:
        if not (0 <= x <= 1000 and 0 <= y <= 1000):
            raise ValueError("视觉 Agent 坐标必须在0～1000之间。")
        self._last_click_receipt = None
        hwnd, _title = seller_gui.find_window(self.title)
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
        # Seller click-count is persistent global UI state.  Bind it to this
        # one canonical request and always restore one afterwards so a later
        # ordinary tap can never inherit double-tap behavior.
        receipt = None
        try:
            seller_gui.configure_click_count(hwnd, click_count)
            receipt = seller_gui.click_client_point(
                hwnd,
                point[0],
                point[1],
                countdown=0,
                hold_seconds=hold_seconds,
                require_event_barrier=True,
                click_count=click_count,
            )
        finally:
            if click_count != 1:
                seller_gui.configure_single_click_count(hwnd)
            seller_gui.clear_seller_camera_overlay(hwnd)
        if not isinstance(receipt, dict):
            raise RuntimeError("控制端没有返回点击事件栅栏凭据。")
        if click_count != 1:
            receipt["click_count_restored_to"] = 1
        self._last_click_receipt = dict(receipt)
        return point

    def _vision_nav_tap(
        self, x_ratio: float, y_ratio: float, *, action: str
    ) -> tuple[int, int]:
        self._last_click_receipt = None
        hwnd, _title = seller_gui.find_window(self.title)
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
        seller_gui.configure_single_click_count(hwnd)
        receipt = seller_gui.click_client_point(
            hwnd,
            point[0],
            point[1],
            countdown=0,
            hold_seconds=float(
                load_controller_config()["tap_hold"]
            ),
            require_event_barrier=True,
        )
        if not isinstance(receipt, dict):
            raise RuntimeError("控制端没有返回单击事件栅栏凭据。")
        self._last_click_receipt = dict(receipt)
        seller_gui.clear_seller_camera_overlay(hwnd)
        return point

    def vision_android_home(self) -> tuple[int, int]:
        # This is the independently verified Android system Home primitive.
        # It must never inherit ordinary semantic-tap authority because it is
        # not an App/browser "home page" element.
        self._require_verified_action("home", "Android系统Home")
        cfg = load_controller_config()
        return self._vision_nav_tap(
            float(cfg["android_home_x_ratio"]),
            float(cfg["android_home_y_ratio"]),
            action="home",
        )

    def vision_android_back(self) -> tuple[int, int]:
        self._require_verified_action("back", "返回")
        cfg = load_controller_config()
        return self._vision_nav_tap(
            float(cfg["android_back_x_ratio"]),
            float(cfg["android_back_y_ratio"]),
            action="back",
        )

    def _vision_swipe(self, direction: str) -> None:
        self._require_verified_action("swipe", "滑动")
        hwnd, _title = seller_gui.find_window(self.title)
        frame = self._capture_phone(hwnd)
        self._consume_physical_execution("swipe", frame)
        self._checkpoint()
        seller_gui.configure_swipe(hwnd, direction)
        seller_gui.trigger_selected_action(hwnd)
        seller_gui.clear_seller_camera_overlay(hwnd)

    def vision_swipe_up(self) -> None:
        self._vision_swipe("up")

    def vision_swipe_down(self) -> None:
        self._vision_swipe("down")

    def vision_swipe_left(self) -> None:
        self._vision_swipe("left")

    def vision_swipe_right(self) -> None:
        self._vision_swipe("right")

    def vision_type_text_with_layout(
        self,
        text: str,
        keyboard_layout: dict[str, Any],
    ) -> None:
        """Universal-agent input path; never falls back to static geometry."""

        self._require_verified_action("input_verified_text", "输入文字")
        if not isinstance(keyboard_layout, dict):
            raise WorkflowNotReady("通用文字输入缺少本轮视觉键盘几何。")
        if not isinstance(text, str) or not re.fullmatch(r"[A-Za-z]{1,30}", text):
            raise WorkflowNotReady("通用英文分段必须是1～30个同一可见大小写状态的字母。")
        self.vision_type_pinyin(text, text.casefold(), keyboard_layout)

    def validate_verified_text(
        self,
        text: str,
        input_states: dict[str, Any],
        *,
        target_text: str | None = None,
        input_method: str | None = None,
        pinyin: str | None = None,
    ) -> None:
        """Fail before hardware unless the current visual keyboard profile is exact."""

        self._require_verified_action("input_verified_text", "输入文字")
        if target_text is None:
            raise WorkflowNotReady("文字输入缺少 canonical 目标全文。")
        try:
            step = plan_next_verified_input(target_text, input_states.get("value"))
        except (ValueError, VerifiedTextTransactionError) as exc:
            raise WorkflowNotReady(f"无法建立精确文字输入事务：{exc}") from exc
        if input_method is None:
            raise WorkflowNotReady("文字输入缺少 canonical 输入方式。")
        if step is None or text != step.segment or input_method != step.kind:
            raise WorkflowNotReady("设备收到的文字分段与本地精确事务不一致。")
        if step.kind == "direct_latin":
            if (
                step.required_case_mode
                and input_states.get("keyboard_case_mode") != step.required_case_mode
            ):
                raise WorkflowNotReady("当前键盘大小写状态与英文分段不一致。")
        elif step.kind == "chinese_pinyin":
            if pinyin != step.pinyin:
                raise WorkflowNotReady("设备收到的拼音与本地确定性结果不一致。")
        else:
            raise WorkflowNotReady("数字或符号仍要求独立可见键位审计。")
        if input_states.get("focused") is not True:
            raise WorkflowNotReady("当前输入框没有可信聚焦证据。")
        if input_states.get("keyboard_layout") != "qwerty":
            raise WorkflowNotReady("当前安全文字输入要求画面确认标准 QWERTY 键盘。")
        if input_states.get("keyboard_input_mode") != step.required_mode:
            raise WorkflowNotReady("当前键盘模式与下一确定性文字分段不一致。")
        if input_states.get("ime_preedit_text"):
            raise WorkflowNotReady("当前仍有未完成的输入法组合。")

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
        cfg = load_controller_config()
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
        hwnd, _title = seller_gui.find_window(self.title)
        frame = self._capture_phone(hwnd)
        self._consume_physical_execution("input_verified_text", frame)
        if frame.width < 400 or frame.height < 700:
            raise RobotWorkflowError("键盘画面尺寸异常，拒绝执行拼音点击。")
        seller_gui.configure_single_click_count(hwnd)
        # Editing the seller control's click-count field changes focus and can
        # leave the physical actuator settling.  Starting the first phone tap
        # immediately after that UI edit caused an intermittent duplicated
        # first letter on real hardware.  Move away from the camera and give
        # the controller/actuator a short deterministic settling window.
        seller_gui.clear_seller_camera_overlay(hwnd)
        self._sleep(float(configured_keyboard.get("pre_key_wait", 0.45)))
        for index, key in enumerate(pinyin):
            self._checkpoint()
            point = qwerty_key_point(frame.width, frame.height, key, keyboard_cfg)
            seller_gui.click_client_point(
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
        seller_gui.clear_seller_camera_overlay(hwnd)

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
        cfg = load_controller_config()
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
        hwnd, _title = seller_gui.find_window(self.title)
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
        seller_gui.configure_single_click_count(hwnd)
        for _index in range(delete_count):
            self._checkpoint()
            seller_gui.click_client_point(
                hwnd,
                point[0],
                point[1],
                countdown=0,
                hold_seconds=0.18,
            )
            self._sleep(0.08)
        seller_gui.clear_seller_camera_overlay(hwnd)

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
            "double_tap",
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

    def _record_mock_click_receipt(self, click_count: int = 1) -> None:
        self._last_click_receipt = {
            "version": "2026-08-25-mock-click-barrier-v1",
            "channel": "mock_atomic_click",
            "seller_event_barrier_confirmed": True,
            "round_trip_position_confirmed": True,
            "mechanical_contact_ack": False,
            "click_count": click_count,
        }

    def vision_tap_relative(self, x: int, y: int) -> tuple[int, int]:
        self._consume_mock_execution("tap_semantic")
        self.executions.append({"action": "tap", "coordinate": [x, y]})
        self._record_mock_click_receipt()
        return x, y

    def vision_dismiss_overlay_relative(self, x: int, y: int) -> tuple[int, int]:
        self._consume_mock_execution("dismiss_overlay")
        self.executions.append({"action": "dismiss_overlay", "coordinate": [x, y]})
        self._record_mock_click_receipt()
        return x, y

    def vision_double_tap_relative(self, x: int, y: int) -> tuple[int, int]:
        self._consume_mock_execution("double_tap")
        self.executions.append({"action": "double_tap", "coordinate": [x, y]})
        self._record_mock_click_receipt(2)
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
        self._last_long_press_receipt = {
            "version": "2026-08-25-mock-long-press-barrier-v1",
            "hold_started_after_barrier": True,
        }
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
        self._record_mock_click_receipt()
        return 500, 976

    def vision_android_back(self) -> tuple[int, int]:
        self._consume_mock_execution("back")
        self.executions.append({"action": "android_back"})
        self._record_mock_click_receipt()
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

    def vision_type_text_with_layout(
        self,
        text: str,
        keyboard_layout: dict[str, Any],
    ) -> None:
        self._consume_mock_execution("input_verified_text")
        self.executions.append(
            {
                "action": "type_text",
                "text": text,
                "keyboard_layout": keyboard_layout,
            }
        )

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
