from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

from .canonical_action_kinds import CANONICAL_ACTION_KINDS


DEVICE_EXECUTOR_PROTOCOL = "2026-08-25-device-executor-v1"

# The canonical catalog is the only action-kind authority. The executor owns
# only transport dispatch and must not maintain a second action whitelist.
EXECUTABLE_ACTION_KINDS = CANONICAL_ACTION_KINDS


class DeviceExecutionError(RuntimeError):
    def __init__(self, message: str, *, physical_actions: int = 0) -> None:
        super().__init__(message)
        self.physical_actions = int(physical_actions)


class DeviceTaskRegistryError(RuntimeError):
    """Invalid or conflicting ownership of one physical device."""


@runtime_checkable
class DeviceTaskRegistryPort(Protocol):
    TERMINAL_STATUSES: frozenset[str]

    def reserve(self, device_id: str, session_id: str) -> None: ...

    def release(self, device_id: str, session_id: str) -> None: ...

    def active_session(self, device_id: str) -> str | None: ...

    def device_lock(self, device_id: str) -> AbstractContextManager[None]: ...

    def is_locked_by_current_thread(self, device_id: str) -> bool: ...


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
        return {
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
