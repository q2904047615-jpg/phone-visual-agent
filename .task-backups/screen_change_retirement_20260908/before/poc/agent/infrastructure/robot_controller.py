"""Seller-window backed Robot and Mock controller infrastructure."""

from __future__ import annotations

from agent.domain.validation import NormalizedPoint, reject_if
import json
import re
import threading
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from agent.infrastructure import seller_window_adapter as seller_gui
from agent.infrastructure.orientation_safety import OrientationCredential, PhysicalExecutionGate



POC_ROOT = Path(__file__).resolve().parents[2]
WEB_OUTPUT_DIR = seller_gui.OUTPUT_DIR / "web"
CONTROL_CONFIG_PATH = POC_ROOT / "controller_config.json"

# Controller presence and camera readiness stay separate because small dialogs share the seller title.
MIN_CAMERA_CLIENT_WIDTH = 300
MIN_CAMERA_CLIENT_HEIGHT = 500


def controller_client_has_camera(width: int, height: int) -> bool:
    if width < MIN_CAMERA_CLIENT_WIDTH or height < MIN_CAMERA_CLIENT_HEIGHT:
        return False
    if width > height:
        return width >= 800 and height >= 450 and 1.45 <= width / height <= 2.0
    return seller_gui.seller_layout_has_full_camera(width, height)


def oriented_navigation_ratio(x_ratio: float, y_ratio: float, *, landscape: bool) -> NormalizedPoint:
    """Map portrait Android navigation coordinates into the observed layout."""

    if landscape:
        # Landscape camera rotation maps portrait (x, y) to (y, 1 - x).
        return y_ratio, 1.0 - x_ratio
    return x_ratio, y_ratio


DEFAULT_CONTROLLER_CONFIG: dict[str, Any] = {'tap_hold': 0.35, 'android_home_x_ratio': 0.5,
    'android_home_y_ratio': 0.976, 'android_recents_x_ratio': 0.33, 'android_recents_y_ratio': 0.976,
    'android_back_x_ratio': 0.685, 'android_back_y_ratio': 0.976,
    'swipe_touch_down_seconds': 0.35, 'swipe_movement_seconds': 0.30, 'swipe_steps': 6}



class RobotWorkflowError(RuntimeError):
    """A user-facing workflow failure that must stop further physical actions."""


class WorkflowNotReady(RobotWorkflowError):
    """Required local calibration/template data is missing."""


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(base))
    for (key, value) in override.items():
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
    reject_if(not isinstance(raw, dict), WorkflowNotReady("控制器配置必须是 JSON 对象。"))
    return _deep_merge(DEFAULT_CONTROLLER_CONFIG, raw)








class RobotController:
    """Safe, single-machine adapter used by the local web task worker."""

    def __init__(self, title: str=seller_gui.DEFAULT_WINDOW_TITLE, *, calibration_path: Path | None=None,
        verified_actions: set[str] | frozenset[str] | None=None, device_id: str) -> None:
        self.title = title
        self.device_id = str(device_id or "").strip()
        self._physical_execution_gate = PhysicalExecutionGate(self.device_id)
        self.calibration_path = Path(
            calibration_path) if calibration_path is not None else POC_ROOT / 'tap_calibration.json'
        self.stop_event = threading.Event()
        self._stop_state_lock = threading.Lock()
        self.operation_lock = threading.Lock()
        # Serialize preview/vision capture to avoid transient truncated Windows bitmaps.
        self.capture_lock = threading.RLock()
        self._last_click_receipt: dict[str, Any] | None = None
        self._last_long_press_receipt: dict[str, Any] | None = None
        self._last_swipe_receipt: dict[str, Any] | None = None
        default_actions = {'tap_semantic', 'dismiss_overlay', 'swipe', 'back', 'home', 'open_recent_apps',
            'wait_for_change'}
        self.verified_actions = frozenset(default_actions if verified_actions is None else verified_actions)
        allowed_actions = default_actions | {'double_tap', 'long_press', 'drag',
            'reveal_system_navigation'}
        unexpected = self.verified_actions - allowed_actions
        reject_if(unexpected, ValueError('设备已验证动作包含未知值：' + ', '.join(sorted(unexpected))))

    def hardware_capabilities(self) -> dict[str, bool]:
        return {action: action in self.verified_actions for action in ('tap_semantic', 'dismiss_overlay', 'swipe',
            'back', 'home', 'open_recent_apps', 'wait_for_change', 'double_tap', 'long_press',
            'drag', 'reveal_system_navigation')}

    def hardware_capability_profile(self) -> dict[str, Any]:
        """Return typed device limits without claiming undocumented ACKs."""

        enabled = self.hardware_capabilities()
        actions: dict[str, dict[str, Any]] = {}
        for (action, available) in enabled.items():
            actions[action] = {'enabled': bool(available), 'one_physical_action_per_receipt': action !=
                'wait_for_change', 'fresh_visual_postcondition_required': True,
                'transport_ack': 'gui_event_barrier' if action in {'tap_semantic', 'dismiss_overlay', 'back', 'home',
                'open_recent_apps', 'double_tap', 'long_press'} else 'local_call_return',
                'mechanical_contact_ack': False}
        actions["long_press"]["duration_ms"] = {"min": 500, "max": 2000}
        actions["drag"]["duration_ms"] = {"fixed": 800}
        actions["swipe"].update({'relative_path_receipt': 'seller_position_barrier',
            'touch_down_ms': 350, 'movement_ms': 300, 'steps': 6})
        actions['double_tap'].update({'enabled': bool(enabled.get('double_tap')),
            'transport': 'seller_click_count_two_atomic_request', 'canonical_action_count': 1, 'contact_count': 2,
            'restores_click_count_to': 1, 'gap_reason': None if enabled.get(
            'double_tap') else 'requires_double_tap_live_acceptance'})
        actions['pinch'] = {'enabled': False, 'gap_reason': 'multi_touch_not_supported_by_single_contact_robot'}
        actions['hardware_key'] = {'enabled': False, 'gap_reason': 'hardware_key_transport_not_verified'}
        return {'protocol_version': '2026-08-18-device-capability-profile-v1', 'device_id': self.device_id,
            'actions': actions}

    def consume_last_long_press_receipt(self) -> dict[str, Any] | None:
        receipt = self._last_long_press_receipt
        self._last_long_press_receipt = None
        return dict(receipt) if receipt is not None else None

    def consume_last_click_receipt(self) -> dict[str, Any] | None:
        receipt = self._last_click_receipt
        self._last_click_receipt = None
        return dict(receipt) if receipt is not None else None

    def consume_last_swipe_receipt(self) -> dict[str, Any] | None:
        receipt = self._last_swipe_receipt
        self._last_swipe_receipt = None
        return dict(receipt) if receipt is not None else None

    def _require_verified_action(self, action: str, label: str) -> None:
        reject_if(action not in self.verified_actions, WorkflowNotReady(f'当前设备尚未完成{label}真机验收，拒绝执行。'))

    def arm_physical_execution(self, credential: OrientationCredential, *, action: str, scene_fingerprint: str) -> None:
        self._physical_execution_gate.arm(credential, action=action, scene_fingerprint=scene_fingerprint)

    def clear_physical_execution_authorization(self) -> None:
        self._physical_execution_gate.clear()

    def _consume_physical_execution(self, action: str, frame: Image.Image) -> OrientationCredential:
        return self._physical_execution_gate.consume(action=action, frame=frame)

    def request_stop(self) -> None:
        with self._stop_state_lock:
            self.stop_event.set()

    def begin_new_task(self) -> None:
        """Acknowledge stop requests that predate this new task boundary."""

        with self._stop_state_lock:
            self.stop_event.clear()

    def _checkpoint(self) -> None:
        reject_if(self.stop_event.is_set(), RobotWorkflowError("用户已请求停止任务。"))
        seller_gui._check_escape()

    def device_status(self) -> dict[str, Any]:
        try:
            hwnd, title = seller_gui.find_window(self.title)
            _left, _top, width, height = seller_gui.client_geometry(hwnd)
            online = width > 0 and height > 0
            camera_online = online and controller_client_has_camera(width, height)
            error = None
            camera_error = None if camera_online else '控制端当前显示启动/报错对话框，未检测到可用的手机摄像头画面。' if online else '控制端窗口不可用。'
        except Exception as exc:  # Device status must stay readable while offline.
            hwnd = 0
            title = ""
            width = height = 0
            online = False
            camera_online = False
            error = str(exc)
            camera_error = str(exc)
        return {'controller_online': online, 'camera_online': camera_online, 'window_title': title,
            'client_size': [width, height], 'stop_requested': self.stop_event.is_set(),
            'busy': self.operation_lock.locked(), 'error': error, 'camera_error': camera_error}

    def _capture_phone(self, hwnd: int) -> Image.Image:
        with self.capture_lock:
            return seller_gui.camera_crop(seller_gui.capture_client(hwnd), seller_gui.DEFAULT_CAMERA_HEIGHT)

    def _capture_phone_passive(self, hwnd: int) -> Image.Image:
        with self.capture_lock:
            return seller_gui.camera_crop(seller_gui.capture_client_passive(hwnd), seller_gui.DEFAULT_CAMERA_HEIGHT)

    def capture_preview(self, quality: int=72) -> bytes:
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

    def _authorized_frame(self, action: str) -> tuple[int, Image.Image]:
        hwnd, _title = seller_gui.find_window(self.title)
        frame = self._capture_phone(hwnd)
        self._consume_physical_execution(action, frame)
        return hwnd, frame

    @staticmethod
    def _grid_pixel(frame: Image.Image, point: NormalizedPoint) -> tuple[int, int]:
        return tuple(min(size - 1, max(0, int(round(value * (size - 1) / 1000))))
            for value, size in zip(point, frame.size))

    def _verified_click(self, x: int, y: int, *, action: str, label: str,
        click_count: int=1) -> tuple[int, int]:
        self._require_verified_action(action, label)
        return self._vision_press_relative(x, y, action=action,
            hold_seconds=float(load_controller_config()['tap_hold']), click_count=click_count)

    def vision_tap_relative(self, x: int, y: int) -> tuple[int, int]:
        """Tap a Qwen3-VL coordinate expressed on a 1000×1000 grid."""
        return self._verified_click(x, y, action='tap_semantic', label='点击')

    def vision_dismiss_overlay_relative(self, x: int, y: int) -> tuple[int, int]:
        """Dismiss one observed overlay through its separately verified entry."""

        return self._verified_click(x, y, action='dismiss_overlay', label='关闭弹层')

    def vision_double_tap_relative(self, x: int, y: int) -> tuple[int, int]:
        """Execute one canonical double-tap through seller click-count two."""

        return self._verified_click(x, y, action='double_tap', label='双击', click_count=2)

    def vision_long_press_relative(self, x: int, y: int, hold_seconds: float=0.8) -> tuple[int, int]:
        """Long-press one calibrated visual target without changing its point."""

        self._require_verified_action("long_press", "长按")
        self._last_long_press_receipt = None
        reject_if(not 0.5 <= float(hold_seconds) <= 2.0, ValueError("通用长按时间必须在0.5～2.0秒之间。"))
        reject_if(not (0 <= x <= 1000 and 0 <= y <= 1000), ValueError("视觉 Agent 坐标必须在0～1000之间。"))
        hwnd, frame = self._authorized_frame('long_press')
        from agent.infrastructure.tap_calibration import corrected_grid_point

        point = self._grid_pixel(frame, corrected_grid_point(x, y, frame.size, self.calibration_path))
        self._checkpoint()
        receipt = seller_gui.long_press_client_point(hwnd, point[0], point[1], hold_seconds=float(hold_seconds))
        reject_if(not isinstance(receipt, dict), RuntimeError("控制端没有返回长按事件栅栏凭据。"))
        self._last_long_press_receipt = dict(receipt)
        seller_gui.clear_seller_camera_overlay(hwnd)
        return point

    def vision_drag_relative(self, start_x: int, start_y: int, end_x: int, end_y: int) -> tuple[tuple[int, int],
        tuple[int, int]]:
        """Drag between two calibrated visual points through the seller UI."""

        return self._vision_path_relative(start_x, start_y, end_x, end_y, action='drag', label='任意两点拖动')

    def vision_swipe_relative(self, start_x: int, start_y: int, end_x: int, end_y: int,
        direction: str) -> tuple[tuple[int, int], tuple[int, int]]:
        """Swipe one observed object along a controller-derived calibrated path."""

        delta_x = end_x - start_x
        delta_y = end_y - start_y
        direction_matches = {'up': delta_y < 0 and abs(delta_y) > abs(delta_x),
            'down': delta_y > 0 and abs(delta_y) > abs(delta_x), 'left': delta_x < 0 and abs(delta_x) > abs(delta_y),
            'right': delta_x > 0 and abs(delta_x) > abs(delta_y)}.get(str(direction or '').strip().lower())
        reject_if(direction_matches is not True, ValueError("元素滑动轨迹与请求方向不一致。"))
        self._require_verified_action('swipe', '元素绑定滑动')
        values = (start_x, start_y, end_x, end_y)
        reject_if(any((not 0 <= value <= 1000 for value in values)), ValueError("元素绑定滑动视觉坐标必须全部在0～1000之间。"))
        hwnd, frame = self._authorized_frame('swipe')
        from agent.infrastructure.tap_calibration import corrected_grid_point

        start, end = (self._grid_pixel(frame, corrected_grid_point(x, y, frame.size, self.calibration_path))
            for x, y in ((start_x, start_y), (end_x, end_y)))
        reject_if(start == end, ValueError("标定后的元素绑定滑动起点和终点重合。"))
        cfg = load_controller_config()
        self._last_swipe_receipt = None
        self._checkpoint()
        receipt = seller_gui.swipe_client_path(hwnd, start, end,
            touch_down_seconds=float(cfg['swipe_touch_down_seconds']),
            movement_seconds=float(cfg['swipe_movement_seconds']), steps=int(cfg['swipe_steps']))
        reject_if(not isinstance(receipt, dict), RuntimeError("控制端没有返回滑动路径凭据。"))
        self._last_swipe_receipt = {**receipt, 'requested_direction': str(direction).strip().lower()}
        seller_gui.clear_seller_camera_overlay(hwnd)
        return start, end

    def _vision_path_relative(self, start_x: int, start_y: int, end_x: int, end_y: int, *, action: str,
        label: str) -> tuple[tuple[int, int], tuple[int, int]]:
        """Execute one calibrated two-point touch path under typed authority."""

        self._require_verified_action(action, label)
        values = (start_x, start_y, end_x, end_y)
        reject_if(any((not 0 <= value <= 1000 for value in values)), ValueError(f"{label}视觉坐标必须全部在0～1000之间。"))
        reject_if((start_x, start_y) == (end_x, end_y), ValueError(f"{label}起点和终点不能相同。"))
        hwnd, frame = self._authorized_frame(action)
        from agent.infrastructure.tap_calibration import corrected_grid_point

        start, end = (self._grid_pixel(frame, corrected_grid_point(x, y, frame.size, self.calibration_path))
            for x, y in ((start_x, start_y), (end_x, end_y)))
        reject_if(start == end, ValueError(f"标定后的{label}起点和终点重合。"))
        self._checkpoint()
        seller_gui.drag_client_path(hwnd, start, end)
        seller_gui.clear_seller_camera_overlay(hwnd)
        return start, end

    def vision_reveal_system_navigation(self) -> dict[str, Any]:
        """Reveal transient system navigation with one locally derived edge path."""

        self._require_verified_action('reveal_system_navigation', '系统边缘唤出导航栏')
        hwnd, frame = self._authorized_frame('reveal_system_navigation')
        from agent.infrastructure.tap_calibration import reveal_system_navigation_path

        evidence = reveal_system_navigation_path((frame.width, frame.height), self.calibration_path)
        corrected = evidence["corrected_grid"]

        start, end = (self._grid_pixel(frame, tuple(point)) for point in corrected)
        reject_if(start == end, ValueError("系统边缘轨迹纠偏后起终点重合。"))
        self._checkpoint()
        seller_gui.drag_client_path(hwnd, start, end)
        seller_gui.clear_seller_camera_overlay(hwnd)
        return {**evidence, 'client_path': [list(start), list(end)]}

    def _vision_press_relative(self, x: int, y: int, *, action: str, hold_seconds: float,
        click_count: int=1) -> tuple[int, int]:
        reject_if(not (0 <= x <= 1000 and 0 <= y <= 1000), ValueError("视觉 Agent 坐标必须在0～1000之间。"))
        self._last_click_receipt = None
        hwnd, frame = self._authorized_frame(action)
        # Calibration corrects visual XY; dedicated Android navigation keeps its own validated ratios.
        from agent.infrastructure.tap_calibration import corrected_grid_point

        point = self._grid_pixel(frame, corrected_grid_point(x, y, frame.size, self.calibration_path))
        self._checkpoint()
        # Seller click-count persists globally, so bind it to this request and restore one.
        receipt = None
        try:
            seller_gui.configure_click_count(hwnd, click_count)
            receipt = seller_gui.click_client_point(hwnd, point[0], point[1], countdown=0, hold_seconds=hold_seconds,
                require_event_barrier=True, click_count=click_count)
        finally:
            if click_count != 1:
                seller_gui.configure_single_click_count(hwnd)
            seller_gui.clear_seller_camera_overlay(hwnd)
        reject_if(not isinstance(receipt, dict), RuntimeError("控制端没有返回点击事件栅栏凭据。"))
        if click_count != 1:
            receipt["click_count_restored_to"] = 1
        self._last_click_receipt = dict(receipt)
        return point

    def _vision_nav_tap(self, x_ratio: float, y_ratio: float, *, action: str) -> tuple[int, int]:
        self._last_click_receipt = None
        hwnd, frame = self._authorized_frame(action)
        x_ratio, y_ratio = oriented_navigation_ratio(x_ratio, y_ratio, landscape=frame.width > frame.height)
        point = (min(frame.width - 1, max(0, int(round(frame.width * x_ratio)))), min(frame.height - 1, max(0,
            int(round(frame.height * y_ratio)))))
        self._checkpoint()
        # Navigation must reset persistent click-count state to one atomic tap.
        seller_gui.configure_single_click_count(hwnd)
        receipt = seller_gui.click_client_point(hwnd, point[0], point[1], countdown=0,
            hold_seconds=float(load_controller_config()['tap_hold']), require_event_barrier=True)
        reject_if(not isinstance(receipt, dict), RuntimeError("控制端没有返回单击事件栅栏凭据。"))
        self._last_click_receipt = dict(receipt)
        seller_gui.clear_seller_camera_overlay(hwnd)
        return point

    def vision_android_home(self) -> tuple[int, int]:
        # Android Home is an independent system primitive, not semantic App navigation.
        return self._vision_system_navigation('home', 'android_home', 'Android系统Home')

    def vision_android_back(self) -> tuple[int, int]:
        return self._vision_system_navigation('back', 'android_back', '返回')

    def vision_android_recent_apps(self) -> tuple[int, int]:
        return self._vision_system_navigation('open_recent_apps', 'android_recents', 'Android系统最近任务')

    def _vision_system_navigation(self, action: str, config_prefix: str, label: str) -> tuple[int, int]:
        self._require_verified_action(action, label)
        cfg = load_controller_config()
        return self._vision_nav_tap(float(cfg[f'{config_prefix}_x_ratio']), float(cfg[f'{config_prefix}_y_ratio']),
            action=action)

    def _vision_swipe(self, direction: str) -> None:
        self._require_verified_action("swipe", "滑动")
        hwnd, _frame = self._authorized_frame('swipe')
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





class MockRobotController(RobotController):
    """No-hardware controller for API tests and UI demonstrations."""

    def __init__(self, *, verified_actions: set[str] | frozenset[str] | None=None,
        device_id: str='mock-default') -> None:
        all_actions = {'tap_semantic', 'dismiss_overlay', 'swipe', 'back', 'home', 'open_recent_apps',
            'wait_for_change', 'double_tap', 'long_press', 'drag', 'reveal_system_navigation'}
        super().__init__(title='MOCK', verified_actions=all_actions if verified_actions is None else verified_actions,
            device_id=device_id)
        self.executions: list[dict[str, Any]] = []

    def device_status(self) -> dict[str, Any]:
        return {'controller_online': True, 'camera_online': True, 'window_title': 'MOCK 智联新途机械臂控制端',
            'client_size': [540, 1038], 'stop_requested': self.stop_event.is_set(),
            'busy': self.operation_lock.locked(), 'error': None}

    def capture_preview(self, quality: int=72) -> bytes:
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

    def _record_mock_click_receipt(self, click_count: int=1) -> None:
        self._last_click_receipt = {'version': '2026-08-25-mock-click-barrier-v1', 'channel': 'mock_atomic_click',
            'seller_event_barrier_confirmed': True, 'round_trip_position_confirmed': True,
            'mechanical_contact_ack': False, 'click_count': click_count}

    def _mock(self, authority: str, action: str, *, result: Any=None, click_count: int=0,
        **payload: Any) -> Any:
        self._consume_mock_execution(authority)
        self.executions.append({'action': action, **payload})
        if click_count:
            self._record_mock_click_receipt(click_count)
        return result

    def _verified_click(self, x: int, y: int, *, action: str, label: str,
        click_count: int=1) -> tuple[int, int]:
        del label
        recorded = {'tap_semantic': 'tap', 'dismiss_overlay': 'dismiss_overlay',
            'double_tap': 'double_tap'}[action]
        return self._mock(action, recorded, result=(x, y), click_count=click_count, coordinate=[x, y])

    def vision_long_press_relative(self, x: int, y: int, hold_seconds: float=0.8) -> tuple[int, int]:
        self._consume_mock_execution("long_press")
        self.executions.append({'action': 'long_press', 'coordinate': [x, y], 'hold_seconds': hold_seconds})
        self._last_long_press_receipt = {'version': '2026-08-25-mock-long-press-barrier-v1',
            'hold_started_after_barrier': True}
        return x, y

    def vision_drag_relative(self, start_x: int, start_y: int, end_x: int, end_y: int) -> tuple[tuple[int, int],
        tuple[int, int]]:
        return self._mock('drag', 'drag', result=((start_x, start_y), (end_x, end_y)), start=[start_x, start_y],
            end=[end_x, end_y])

    def vision_swipe_relative(self, start_x: int, start_y: int, end_x: int, end_y: int,
        direction: str) -> tuple[tuple[int, int], tuple[int, int]]:
        result = self._mock('swipe', 'swipe_relative', result=((start_x, start_y), (end_x, end_y)),
            direction=direction, start=[start_x, start_y], end=[end_x, end_y])
        self._last_swipe_receipt = {'version': '2026-09-03-mock-swipe-path-v1',
            'channel': 'mock_swipe_path', 'right_button_down_dispatched': True,
            'right_button_up_dispatched': True, 'interpolation_steps_completed': 6,
            'seller_position_barrier_confirmed': True, 'round_trip_position_confirmed': True,
            'touch_down_seconds': 0.35, 'movement_seconds': 0.3, 'step_count': 6,
            'client_start': [start_x, start_y], 'client_end': [end_x, end_y],
            'mechanical_contact_ack': False, 'requested_direction': direction}
        return result

    def vision_reveal_system_navigation(self) -> dict[str, Any]:
        self._require_verified_action('reveal_system_navigation', '系统边缘唤出导航栏')
        self._consume_mock_execution("reveal_system_navigation")
        evidence = {'action': 'reveal_system_navigation', 'edge': 'bottom', 'frame_size': [540, 960], 'dom_path': [[0.5,
            0.95], [0.5, 0.7]], 'requested_grid': [[100, 500], [350, 500]], 'corrected_grid': [[100, 500], [350, 500]],
            'client_path': [[54, 480], [189, 480]], 'mock': True}
        self.executions.append(dict(evidence))
        return evidence

    def _vision_system_navigation(self, action: str, config_prefix: str, label: str) -> tuple[int, int]:
        del config_prefix, label
        recorded, point = {'home': ('android_home', (500, 976)), 'back': ('android_back', (910, 976)),
            'open_recent_apps': ('android_recent_apps', (315, 976))}[action]
        return self._mock(action, recorded, result=point, click_count=1)

    def _vision_swipe(self, direction: str) -> None:
        self._mock('swipe', f'swipe_{direction}')
