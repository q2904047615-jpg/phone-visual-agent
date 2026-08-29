from __future__ import annotations

from agent.domain.validation import reject_if
import time
from typing import Any, Callable, Iterable, Mapping

from agent.domain import DeviceActionRequest, DeviceExecutionError, DeviceExecutionResult
from agent.infrastructure.orientation_safety import OrientationSafetyError


class RobotDeviceExecutor:
    """The only registry mapping resolved canonical actions to Robot methods."""

    _CLICK_TRANSPORTS = {
        'tap_semantic': ('vision_tap_relative', True, 1),
        'press_enter': ('vision_tap_relative', True, 1),
        'dismiss_overlay': ('vision_dismiss_overlay_relative', True, 1),
        'double_tap': ('vision_double_tap_relative', True, 2),
        'back': ('vision_android_back', False, 1),
        'home': ('vision_android_home', False, 1),
        'open_recent_apps': ('vision_android_recent_apps', False, 1),
    }
    _SIMPLE_TRANSPORTS = {
        'reveal_system_navigation': ('vision_reveal_system_navigation', lambda request: ()),
        'clear_verified_text': ('vision_clear_text', lambda request: (dict(request.keyboard_geometry or {}),
            request.delete_count)),
        'drag': ('vision_drag_relative', lambda request: (*request.point, *request.end_point)),
    }

    def __init__(self, robot: Any, *, sleep: Callable[[float], None]=time.sleep) -> None:
        self.robot = robot
        self.sleep = sleep
        self._handlers: dict[str, Callable[[DeviceActionRequest], DeviceExecutionResult]] = {
            'swipe': self._swipe,
            'input_verified_text': self._input_text,
            'long_press': self._long_press,
            'wait_for_change': self._wait,
        }

    def execute(self, request: DeviceActionRequest) -> DeviceExecutionResult:
        request.validate()
        click_spec = self._CLICK_TRANSPORTS.get(request.kind)
        if click_spec is not None:
            return self._click(request, *click_spec)
        simple_spec = self._SIMPLE_TRANSPORTS.get(request.kind)
        if simple_spec is not None:
            method, arguments = simple_spec
            return DeviceExecutionResult(physical_actions=1,
                transport_result=self._hardware_call(method, *arguments(request)))
        handler = self._handlers.get(request.kind)
        reject_if(handler is None, DeviceExecutionError(f"设备执行器没有动作处理器：{request.kind}"))
        return handler(request)

    def _method(self, name: str) -> Callable[..., Any]:
        method = getattr(self.robot, name, None)
        reject_if(not callable(method), DeviceExecutionError(f"机械控制端缺少 transport：{name}"))
        return method

    def _hardware_call(self, method_name: str, *args: Any) -> Any:
        method = self._method(method_name)
        try:
            return method(*args)
        except DeviceExecutionError:
            raise
        except OrientationSafetyError:
            raise
        except Exception as exc:
            raise DeviceExecutionError(f'设备 transport {method_name} 调用失败：{exc}', physical_actions=1) from exc

    def _consume_click_receipt(self, *, expected_count: int) -> dict[str, Any]:
        consumer = self._method("consume_last_click_receipt")
        raw = consumer()
        reject_if(
            not isinstance(raw, dict) or raw.get('seller_event_barrier_confirmed') is not True
            or raw.get('round_trip_position_confirmed') is not True or (raw.get('mechanical_contact_ack')
            is not False),
            DeviceExecutionError('机械控制端没有返回有效的单击事件栅栏凭据。', physical_actions=1),
        )
        click_count = raw.get("click_count", 1)
        reject_if(click_count != expected_count, DeviceExecutionError('点击事件栅栏的 click_count 与请求不一致。', physical_actions=1))
        return dict(raw)

    @staticmethod
    def _point(request: DeviceActionRequest) -> tuple[int, int]:
        assert request.point is not None
        return request.point

    def _click(self, request: DeviceActionRequest, method: str, needs_point: bool,
        click_count: int) -> DeviceExecutionResult:
        args = self._point(request) if needs_point else ()
        result = self._hardware_call(method, *args)
        return DeviceExecutionResult(physical_actions=1, transport_result=result,
            hardware_receipt=self._consume_click_receipt(expected_count=click_count))

    def _swipe(self, request: DeviceActionRequest) -> DeviceExecutionResult:
        assert request.direction is not None
        if request.point is not None and request.end_point is not None:
            return DeviceExecutionResult(physical_actions=1,
                transport_result=self._hardware_call('vision_swipe_relative', *request.point, *request.end_point,
                request.direction))
        return DeviceExecutionResult(physical_actions=1, transport_result=self._hardware_call(f'vision_swipe_{
            request.direction}'))

    def _input_text(self, request: DeviceActionRequest) -> DeviceExecutionResult:
        geometry = dict(request.keyboard_geometry or {})
        if request.input_method == 'chinese_pinyin':
            result = self._hardware_call('vision_type_pinyin', request.input_fragment, request.input_pinyin, geometry)
        else:
            result = self._hardware_call('vision_type_text_with_layout', request.input_fragment, geometry)
        return DeviceExecutionResult(physical_actions=1, transport_result=result)

    def _long_press(self, request: DeviceActionRequest) -> DeviceExecutionResult:
        result = self._hardware_call('vision_long_press_relative', *self._point(request), request.hold_seconds)
        raw = self._method("consume_last_long_press_receipt")()
        reject_if(not isinstance(raw, dict), DeviceExecutionError('机械控制端没有返回长按事件栅栏凭据。', physical_actions=1))
        return DeviceExecutionResult(physical_actions=1, transport_result=result, hardware_receipt=dict(raw))

    def _wait(self, request: DeviceActionRequest) -> DeviceExecutionResult:
        self.sleep(float(request.wait_seconds or 0.0))
        return DeviceExecutionResult(physical_actions=0)


class ReplayDeviceExecutor:
    """Read-only deterministic executor for offline task-sequence evaluation."""

    def __init__(self, script: Iterable[Mapping[str, Any]]) -> None:
        self._script = [dict(item) for item in script]
        self._index = 0
        self.requests: list[dict[str, Any]] = []

    @property
    def complete(self) -> bool:
        return self._index == len(self._script)

    def execute(self, request: DeviceActionRequest) -> DeviceExecutionResult:
        request.validate()
        reject_if(self._index >= len(self._script), DeviceExecutionError("离线回放收到脚本之外的额外动作。"))
        expected = self._script[self._index]
        expected_kind = str(expected.get("kind") or "")
        reject_if(expected_kind != request.kind, DeviceExecutionError(f'离线回放动作不匹配：{request.kind} != {expected_kind}'))
        expected_request = expected.get("request")
        actual = request.to_dict()
        if isinstance(expected_request, Mapping):
            for (key, value) in expected_request.items():
                reject_if(actual.get(str(key)) != value, DeviceExecutionError(f"离线回放参数不匹配：{key}"))
        self._index += 1
        self.requests.append(actual)
        return DeviceExecutionResult(execution_mode='offline_replay', physical_actions=0,
            replayed_actions=1 if request.kind != 'wait_for_change' else 0,
            transport_result=expected.get('transport_result'),
            hardware_receipt=dict(expected['hardware_receipt']) if isinstance(expected.get('hardware_receipt'),
            Mapping) else None, metadata={'script_index': self._index - 1})
