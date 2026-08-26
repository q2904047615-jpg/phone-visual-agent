from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Protocol, runtime_checkable

from canonical_action_protocol import SUPPORTED_ACTIONS
from orientation_safety import OrientationSafetyError


DEVICE_EXECUTOR_PROTOCOL = "2026-08-25-device-executor-v1"

# The canonical catalog is the only action-kind authority.  The executor owns
# only the transport handler mapping and must not maintain a second whitelist.
EXECUTABLE_ACTION_KINDS = SUPPORTED_ACTIONS


class DeviceExecutionError(RuntimeError):
    def __init__(self, message: str, *, physical_actions: int = 0) -> None:
        super().__init__(message)
        self.physical_actions = int(physical_actions)


@dataclass(frozen=True)
class DeviceActionRequest:
    """Transport-only projection of one Controller-resolved action."""

    kind: str
    point: tuple[int, int] | None = None
    end_point: tuple[int, int] | None = None
    direction: str | None = None
    hold_seconds: float | None = None
    input_fragment: str | None = None
    input_method: str | None = None
    input_pinyin: str | None = None
    keyboard_geometry: Mapping[str, Any] | None = None
    delete_count: int | None = None
    wait_seconds: float | None = None

    def validate(self) -> None:
        if self.kind not in EXECUTABLE_ACTION_KINDS:
            raise DeviceExecutionError(f"设备执行器不支持动作：{self.kind}")
        point_kinds = {
            "tap_semantic",
            "press_enter",
            "dismiss_overlay",
            "double_tap",
            "long_press",
            "drag",
        }
        if self.kind in point_kinds:
            self._validate_point(self.point, "动作落点")
        if self.kind == "drag":
            self._validate_point(self.end_point, "拖动终点")
            if self.point == self.end_point:
                raise DeviceExecutionError("拖动起点和终点不能相同。")
        if self.kind == "swipe":
            if self.direction not in {"up", "down", "left", "right"}:
                raise DeviceExecutionError("滑动方向无效。")
            has_start = self.point is not None
            has_end = self.end_point is not None
            if has_start != has_end:
                raise DeviceExecutionError(
                    "元素绑定滑动必须同时提供起点和终点。"
                )
            if has_start:
                self._validate_point(self.point, "元素滑动起点")
                self._validate_point(self.end_point, "元素滑动终点")
                if self.point == self.end_point:
                    raise DeviceExecutionError("元素滑动起点和终点不能相同。")
                assert self.point is not None and self.end_point is not None
                delta_x = self.end_point[0] - self.point[0]
                delta_y = self.end_point[1] - self.point[1]
                direction_matches = {
                    "up": delta_y < 0 and abs(delta_y) > abs(delta_x),
                    "down": delta_y > 0 and abs(delta_y) > abs(delta_x),
                    "left": delta_x < 0 and abs(delta_x) > abs(delta_y),
                    "right": delta_x > 0 and abs(delta_x) > abs(delta_y),
                }[self.direction]
                if not direction_matches:
                    raise DeviceExecutionError(
                        "元素滑动轨迹与请求方向不一致。"
                    )
        if self.kind == "long_press" and (
            isinstance(self.hold_seconds, bool)
            or not isinstance(self.hold_seconds, (int, float))
            or not 0.5 <= float(self.hold_seconds) <= 2.0
        ):
            raise DeviceExecutionError("长按时长必须在0.5～2.0秒之间。")
        if self.kind == "input_verified_text":
            if not isinstance(self.input_fragment, str) or not self.input_fragment:
                raise DeviceExecutionError("输入动作缺少确定性文字分段。")
            if self.input_method not in {"direct_latin", "chinese_pinyin"}:
                raise DeviceExecutionError("输入动作 transport 类型无效。")
            if not isinstance(self.keyboard_geometry, Mapping):
                raise DeviceExecutionError("输入动作缺少已审计键盘几何。")
            if self.input_method == "chinese_pinyin" and not self.input_pinyin:
                raise DeviceExecutionError("中文输入动作缺少拼音分段。")
        if self.kind == "clear_verified_text":
            if not isinstance(self.keyboard_geometry, Mapping):
                raise DeviceExecutionError("清空动作缺少已审计键盘几何。")
            if (
                isinstance(self.delete_count, bool)
                or not isinstance(self.delete_count, int)
                or not 1 <= self.delete_count <= 100
            ):
                raise DeviceExecutionError("清空动作退格次数无效。")
        if self.kind == "wait_for_change" and (
            isinstance(self.wait_seconds, bool)
            or not isinstance(self.wait_seconds, (int, float))
            or float(self.wait_seconds) < 0
        ):
            raise DeviceExecutionError("等待时长无效。")

    @staticmethod
    def _validate_point(value: tuple[int, int] | None, label: str) -> None:
        if (
            not isinstance(value, tuple)
            or len(value) != 2
            or any(
                isinstance(part, bool)
                or not isinstance(part, int)
                or not 0 <= part <= 1000
                for part in value
            )
        ):
            raise DeviceExecutionError(f"{label}必须是0～1000整数坐标。")

    def to_dict(self) -> dict[str, Any]:
        value = {
            "kind": self.kind,
            "point": list(self.point) if self.point is not None else None,
            "end_point": (
                list(self.end_point) if self.end_point is not None else None
            ),
            "direction": self.direction,
            "hold_seconds": self.hold_seconds,
            "input_fragment": self.input_fragment,
            "input_method": self.input_method,
            "input_pinyin": self.input_pinyin,
            "keyboard_geometry": (
                dict(self.keyboard_geometry)
                if isinstance(self.keyboard_geometry, Mapping)
                else None
            ),
            "delete_count": self.delete_count,
            "wait_seconds": self.wait_seconds,
        }
        return value


@dataclass(frozen=True)
class DeviceExecutionResult:
    protocol_version: str = DEVICE_EXECUTOR_PROTOCOL
    execution_mode: str = "hardware"
    physical_actions: int = 0
    replayed_actions: int = 0
    transport_result: Any = None
    hardware_receipt: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class DeviceExecutor(Protocol):
    @property
    def action_kinds(self) -> frozenset[str]: ...

    def execute(self, request: DeviceActionRequest) -> DeviceExecutionResult: ...


class RobotDeviceExecutor:
    """The only registry mapping resolved canonical actions to Robot methods."""

    def __init__(
        self,
        robot: Any,
        *,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.robot = robot
        self.sleep = sleep
        self._handlers: dict[
            str,
            Callable[[DeviceActionRequest], DeviceExecutionResult],
        ] = {
            "tap_semantic": self._tap,
            "press_enter": self._tap,
            "dismiss_overlay": self._dismiss_overlay,
            "double_tap": self._double_tap,
            "reveal_system_navigation": self._reveal_system_navigation,
            "swipe": self._swipe,
            "back": self._back,
            "home": self._home,
            "open_recent_apps": self._open_recent_apps,
            "input_verified_text": self._input_text,
            "clear_verified_text": self._clear_text,
            "long_press": self._long_press,
            "drag": self._drag,
            "wait_for_change": self._wait,
        }

    @property
    def action_kinds(self) -> frozenset[str]:
        return frozenset(self._handlers)

    def execute(self, request: DeviceActionRequest) -> DeviceExecutionResult:
        request.validate()
        handler = self._handlers.get(request.kind)
        if handler is None:
            raise DeviceExecutionError(f"设备执行器没有动作处理器：{request.kind}")
        return handler(request)

    def _method(self, name: str) -> Callable[..., Any]:
        method = getattr(self.robot, name, None)
        if not callable(method):
            raise DeviceExecutionError(f"机械控制端缺少 transport：{name}")
        return method

    def _hardware_call(
        self,
        method_name: str,
        *args: Any,
    ) -> Any:
        method = self._method(method_name)
        try:
            return method(*args)
        except DeviceExecutionError:
            raise
        except OrientationSafetyError:
            # The shared orientation gate rejected the call before the seller
            # transport was allowed to emit a physical action.  Preserve the
            # typed error so the adapter can retain its zero-action evidence.
            raise
        except Exception as exc:
            raise DeviceExecutionError(
                f"设备 transport {method_name} 调用失败：{exc}",
                physical_actions=1,
            ) from exc

    def _consume_click_receipt(self, *, expected_count: int) -> dict[str, Any]:
        consumer = self._method("consume_last_click_receipt")
        raw = consumer()
        if (
            not isinstance(raw, dict)
            or raw.get("seller_event_barrier_confirmed") is not True
            or raw.get("round_trip_position_confirmed") is not True
            or raw.get("mechanical_contact_ack") is not False
        ):
            raise DeviceExecutionError(
                "机械控制端没有返回有效的单击事件栅栏凭据。",
                physical_actions=1,
            )
        click_count = raw.get("click_count", 1)
        if click_count != expected_count:
            raise DeviceExecutionError(
                "点击事件栅栏的 click_count 与请求不一致。",
                physical_actions=1,
            )
        return dict(raw)

    def _point(self, request: DeviceActionRequest) -> tuple[int, int]:
        assert request.point is not None
        return request.point

    def _tap(self, request: DeviceActionRequest) -> DeviceExecutionResult:
        result = self._hardware_call("vision_tap_relative", *self._point(request))
        return DeviceExecutionResult(
            physical_actions=1,
            transport_result=result,
            hardware_receipt=self._consume_click_receipt(expected_count=1),
        )

    def _dismiss_overlay(self, request: DeviceActionRequest) -> DeviceExecutionResult:
        result = self._hardware_call(
            "vision_dismiss_overlay_relative",
            *self._point(request),
        )
        return DeviceExecutionResult(
            physical_actions=1,
            transport_result=result,
            hardware_receipt=self._consume_click_receipt(expected_count=1),
        )

    def _double_tap(self, request: DeviceActionRequest) -> DeviceExecutionResult:
        result = self._hardware_call(
            "vision_double_tap_relative",
            *self._point(request),
        )
        return DeviceExecutionResult(
            physical_actions=1,
            transport_result=result,
            hardware_receipt=self._consume_click_receipt(expected_count=2),
        )

    def _reveal_system_navigation(
        self,
        _request: DeviceActionRequest,
    ) -> DeviceExecutionResult:
        return DeviceExecutionResult(
            physical_actions=1,
            transport_result=self._hardware_call("vision_reveal_system_navigation"),
        )

    def _swipe(self, request: DeviceActionRequest) -> DeviceExecutionResult:
        assert request.direction is not None
        if request.point is not None and request.end_point is not None:
            return DeviceExecutionResult(
                physical_actions=1,
                transport_result=self._hardware_call(
                    "vision_swipe_relative",
                    *request.point,
                    *request.end_point,
                    request.direction,
                ),
            )
        return DeviceExecutionResult(
            physical_actions=1,
            transport_result=self._hardware_call(
                f"vision_swipe_{request.direction}"
            ),
        )

    def _back(self, _request: DeviceActionRequest) -> DeviceExecutionResult:
        result = self._hardware_call("vision_android_back")
        return DeviceExecutionResult(
            physical_actions=1,
            transport_result=result,
            hardware_receipt=self._consume_click_receipt(expected_count=1),
        )

    def _home(self, _request: DeviceActionRequest) -> DeviceExecutionResult:
        result = self._hardware_call("vision_android_home")
        return DeviceExecutionResult(
            physical_actions=1,
            transport_result=result,
            hardware_receipt=self._consume_click_receipt(expected_count=1),
        )

    def _open_recent_apps(
        self,
        _request: DeviceActionRequest,
    ) -> DeviceExecutionResult:
        result = self._hardware_call("vision_android_recent_apps")
        return DeviceExecutionResult(
            physical_actions=1,
            transport_result=result,
            hardware_receipt=self._consume_click_receipt(expected_count=1),
        )

    def _input_text(self, request: DeviceActionRequest) -> DeviceExecutionResult:
        geometry = dict(request.keyboard_geometry or {})
        if request.input_method == "chinese_pinyin":
            result = self._hardware_call(
                "vision_type_pinyin",
                request.input_fragment,
                request.input_pinyin,
                geometry,
            )
        else:
            result = self._hardware_call(
                "vision_type_text_with_layout",
                request.input_fragment,
                geometry,
            )
        return DeviceExecutionResult(
            physical_actions=1,
            transport_result=result,
        )

    def _clear_text(self, request: DeviceActionRequest) -> DeviceExecutionResult:
        result = self._hardware_call(
            "vision_clear_text",
            dict(request.keyboard_geometry or {}),
            request.delete_count,
        )
        return DeviceExecutionResult(
            physical_actions=1,
            transport_result=result,
        )

    def _long_press(self, request: DeviceActionRequest) -> DeviceExecutionResult:
        result = self._hardware_call(
            "vision_long_press_relative",
            *self._point(request),
            request.hold_seconds,
        )
        raw = self._method("consume_last_long_press_receipt")()
        if not isinstance(raw, dict):
            raise DeviceExecutionError(
                "机械控制端没有返回长按事件栅栏凭据。",
                physical_actions=1,
            )
        return DeviceExecutionResult(
            physical_actions=1,
            transport_result=result,
            hardware_receipt=dict(raw),
        )

    def _drag(self, request: DeviceActionRequest) -> DeviceExecutionResult:
        assert request.end_point is not None
        return DeviceExecutionResult(
            physical_actions=1,
            transport_result=self._hardware_call(
                "vision_drag_relative",
                *self._point(request),
                *request.end_point,
            ),
        )

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
    def action_kinds(self) -> frozenset[str]:
        return EXECUTABLE_ACTION_KINDS

    @property
    def complete(self) -> bool:
        return self._index == len(self._script)

    def execute(self, request: DeviceActionRequest) -> DeviceExecutionResult:
        request.validate()
        if self._index >= len(self._script):
            raise DeviceExecutionError("离线回放收到脚本之外的额外动作。")
        expected = self._script[self._index]
        expected_kind = str(expected.get("kind") or "")
        if expected_kind != request.kind:
            raise DeviceExecutionError(
                f"离线回放动作不匹配：{request.kind} != {expected_kind}"
            )
        expected_request = expected.get("request")
        actual = request.to_dict()
        if isinstance(expected_request, Mapping):
            for key, value in expected_request.items():
                if actual.get(str(key)) != value:
                    raise DeviceExecutionError(
                        f"离线回放参数不匹配：{key}"
                    )
        self._index += 1
        self.requests.append(actual)
        return DeviceExecutionResult(
            execution_mode="offline_replay",
            physical_actions=0,
            replayed_actions=1 if request.kind != "wait_for_change" else 0,
            transport_result=expected.get("transport_result"),
            hardware_receipt=(
                dict(expected["hardware_receipt"])
                if isinstance(expected.get("hardware_receipt"), Mapping)
                else None
            ),
            metadata={"script_index": self._index - 1},
        )
