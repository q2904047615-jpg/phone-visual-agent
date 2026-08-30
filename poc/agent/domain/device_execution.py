from __future__ import annotations

from .validation import DataclassWire, reject_if
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

from .canonical_action_kinds import CANONICAL_ACTION_KINDS
from .text_transport import EMPTY_TEXT_DIGEST, TextTransportActionScope, text_digest


DEVICE_EXECUTOR_PROTOCOL = "2026-08-25-device-executor-v1"

# The canonical catalog is the only action-kind authority. The executor owns
# only transport dispatch and must not maintain a second action whitelist.
EXECUTABLE_ACTION_KINDS = CANONICAL_ACTION_KINDS


class DeviceExecutionError(RuntimeError):
    def __init__(self, message: str, *, physical_actions: int=0,
        metadata: Mapping[str, Any] | None=None) -> None:
        super().__init__(message)
        self.physical_actions = int(physical_actions)
        self.metadata = dict(metadata or {})


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
class DeviceActionRequest(DataclassWire):
    """Transport-only projection of one Controller-resolved action."""

    kind: str
    point: tuple[int, int] | None = None
    end_point: tuple[int, int] | None = None
    direction: str | None = None
    hold_seconds: float | None = None
    input_fragment: str | None = None
    input_method: str | None = None
    input_pinyin: str | None = None
    text_transport: str | None = None
    text_scope: TextTransportActionScope | None = None
    keyboard_geometry: Mapping[str, Any] | None = None
    delete_count: int | None = None
    launch_ref: str | None = None
    wait_seconds: float | None = None

    def validate(self) -> None:
        reject_if(self.kind not in EXECUTABLE_ACTION_KINDS, DeviceExecutionError(f"设备执行器不支持动作：{self.kind}"))
        point_kinds = {'tap_semantic', 'press_enter', 'dismiss_overlay', 'double_tap', 'long_press', 'drag'}
        if self.kind in point_kinds:
            self._validate_point(self.point, "动作落点")
        if self.kind == 'drag':
            self._validate_point(self.end_point, "拖动终点")
            reject_if(self.point == self.end_point, DeviceExecutionError("拖动起点和终点不能相同。"))
        if self.kind == 'swipe':
            reject_if(self.direction not in {'up', 'down', 'left', 'right'}, DeviceExecutionError("滑动方向无效。"))
            has_start = self.point is not None
            has_end = self.end_point is not None
            reject_if(has_start != has_end, DeviceExecutionError('元素绑定滑动必须同时提供起点和终点。'))
            if has_start:
                self._validate_point(self.point, "元素滑动起点")
                self._validate_point(self.end_point, "元素滑动终点")
                reject_if(self.point == self.end_point, DeviceExecutionError("元素滑动起点和终点不能相同。"))
                assert self.point is not None and self.end_point is not None
                delta_x = self.end_point[0] - self.point[0]
                delta_y = self.end_point[1] - self.point[1]
                direction_matches = {'up': delta_y < 0 and abs(delta_y) > abs(delta_x),
                    'down': delta_y > 0 and abs(delta_y) > abs(delta_x),
                    'left': delta_x < 0 and abs(delta_x) > abs(delta_y),
                    'right': delta_x > 0 and abs(delta_x) > abs(delta_y)}[self.direction]
                reject_if(not direction_matches, DeviceExecutionError('元素滑动轨迹与请求方向不一致。'))
        reject_if(self.kind == 'long_press' and (isinstance(self.hold_seconds, bool) or not isinstance(self.hold_seconds, (int, float)) or (not 0.5 <= float(self.hold_seconds) <= 2.0)), DeviceExecutionError("长按时长必须在0.5～2.0秒之间。"))
        if self.kind == 'input_verified_text':
            reject_if(not isinstance(self.input_fragment, str) or not self.input_fragment, DeviceExecutionError("输入动作缺少确定性文字分段。"))
            if self.text_transport == 'companion_ime':
                reject_if(self.input_method != 'unicode_commit' or self.input_pinyin is not None
                    or self.keyboard_geometry is not None or self.text_scope is None,
                    DeviceExecutionError("Companion IME 输入请求包含机械键盘字段或缺少授权 scope。"))
                assert self.text_scope is not None
                try:
                    self.text_scope.validate()
                except ValueError as exc:
                    raise DeviceExecutionError(f"Companion IME 输入 scope 无效：{exc}") from exc
                reject_if(text_digest(self.input_fragment) != self.text_scope.fragment_text_digest,
                    DeviceExecutionError("Companion IME 输入正文与授权摘要不一致。"))
            else:
                reject_if(self.text_transport not in {None, 'mechanical_keyboard'}, DeviceExecutionError("输入动作 transport 类型无效。"))
                reject_if(self.input_method not in {'direct_latin', 'chinese_pinyin'}, DeviceExecutionError("输入动作 transport 类型无效。"))
                reject_if(not isinstance(self.keyboard_geometry, Mapping), DeviceExecutionError("输入动作缺少已审计键盘几何。"))
                reject_if(self.input_method == 'chinese_pinyin' and (not self.input_pinyin), DeviceExecutionError("中文输入动作缺少拼音分段。"))
                reject_if(self.text_scope is not None, DeviceExecutionError("机械键盘输入不得携带 Companion scope。"))
        if self.kind == 'clear_verified_text':
            if self.text_transport == 'companion_ime':
                reject_if(self.keyboard_geometry is not None or self.delete_count is not None
                    or self.text_scope is None, DeviceExecutionError("Companion IME 清空请求包含机械键盘字段或缺少授权 scope。"))
                assert self.text_scope is not None
                try:
                    self.text_scope.validate()
                except ValueError as exc:
                    raise DeviceExecutionError(f"Companion IME 清空 scope 无效：{exc}") from exc
                reject_if(self.text_scope.fragment_text_digest != EMPTY_TEXT_DIGEST
                    or self.text_scope.expected_text_digest != EMPTY_TEXT_DIGEST,
                    DeviceExecutionError("Companion IME 清空 scope 没有绑定空 fragment/expected。"))
            else:
                reject_if(self.text_transport not in {None, 'mechanical_keyboard'}, DeviceExecutionError("清空动作 transport 类型无效。"))
                reject_if(not isinstance(self.keyboard_geometry, Mapping), DeviceExecutionError("清空动作缺少已审计键盘几何。"))
                reject_if(isinstance(self.delete_count, bool) or not isinstance(self.delete_count, int) or (not 1 <= self.delete_count <= 100), DeviceExecutionError("清空动作退格次数无效。"))
                reject_if(self.text_scope is not None, DeviceExecutionError("机械键盘清空不得携带 Companion scope。"))
        if self.kind not in {'input_verified_text', 'clear_verified_text'}:
            reject_if(self.text_transport is not None or self.text_scope is not None,
                DeviceExecutionError("非文字动作不得携带 text transport 或授权 scope。"))
        if self.kind == 'launch_app':
            reject_if(not isinstance(self.launch_ref, str) or not self.launch_ref or len(self.launch_ref) > 128
                or any((not (character.isalnum() or character in '._:-') for character in self.launch_ref)),
                DeviceExecutionError("App 直启请求缺少有效 launch_ref。"))
        else:
            reject_if(self.launch_ref is not None, DeviceExecutionError("非 App 直启动作不得携带 launch_ref。"))
        reject_if(self.kind == 'wait_for_change' and (isinstance(self.wait_seconds, bool) or not isinstance(self.wait_seconds, (int, float)) or float(self.wait_seconds) < 0), DeviceExecutionError("等待时长无效。"))

    @staticmethod
    def _validate_point(value: tuple[int, int] | None, label: str) -> None:
        reject_if(not isinstance(value, tuple) or len(value) != 2 or any((isinstance(part, bool) or not isinstance(part, int) or (not 0 <= part <= 1000) for part in value)), DeviceExecutionError(f"{label}必须是0～1000整数坐标。"))

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
    def execute(self, request: DeviceActionRequest) -> DeviceExecutionResult: ...
