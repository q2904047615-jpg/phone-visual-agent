from __future__ import annotations

from agent.domain.validation import reject_if
import json
import threading
from collections.abc import Collection
from pathlib import Path
from typing import Any

from agent.infrastructure.robot_controller import MockRobotController, RobotController
from agent.domain.action_capabilities import physical_capability_for_action


class DeviceControllerRegistryError(RuntimeError):
    """The configured device-controller registry is unavailable or invalid."""


class ProvisionalDeviceControllerError(DeviceControllerRegistryError):
    """A provisional capability controller cannot be created."""


class DeviceControllerRegistry:
    """Resolve one controller and calibration per configured device_id."""

    def __init__(self, path: Path, *, promotable_actions: Collection[str], mock: bool=False) -> None:
        self.path = Path(path)
        self._promotable_actions = frozenset((str(action or '').strip() for action in promotable_actions))
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise DeviceControllerRegistryError(f'设备注册表无法读取：{exc}') from exc
        reject_if(payload.get('version') != 1 or not isinstance(payload.get('devices'), list), DeviceControllerRegistryError('设备注册表版本或 devices 格式无效。'))
        self.default_device_id = str(payload.get('default_device_id') or '').strip()
        self._controllers: dict[str, RobotController] = {}
        self._descriptors: dict[str, dict[str, Any]] = {}
        enabled_windows: set[str] = set()
        window_positions: dict[str, set[int | None]] = {}
        window_locks: dict[str, threading.RLock] = {}
        window_capture_locks: dict[str, threading.RLock] = {}
        for raw in payload['devices']:
            if not isinstance(raw, dict) or raw.get('enabled') is not True:
                continue
            device_id = str(raw.get("device_id") or "").strip()
            window_title = str(raw.get("window_title") or "").strip()
            calibration_value = str(raw.get("calibration_path") or "").strip()
            raw_verified_actions = raw.get("verified_actions")
            if raw_verified_actions is None:
                verified_actions = None
            elif (not isinstance(raw_verified_actions, list) or not all((isinstance(item,
                str) and item.strip() for item in raw_verified_actions))):
                raise DeviceControllerRegistryError(f'设备 {device_id or 'missing'} 的 verified_actions 格式无效。')
            else:
                verified_actions = {str(item).strip() for item in raw_verified_actions}
            raw_machine_position = raw.get("machine_position")
            if raw_machine_position is None:
                machine_position = None
            else:
                reject_if(isinstance(raw_machine_position, bool) or not isinstance(raw_machine_position, int)
                    or not 1 <= raw_machine_position <= 10,
                    DeviceControllerRegistryError(f'设备 {device_id or "missing"} 的 machine_position 必须是1到10之间的整数。'))
                machine_position = raw_machine_position
            reject_if(not device_id or device_id in self._controllers, DeviceControllerRegistryError('设备注册表存在空或重复的 device_id。'))
            effective_window = window_title or "__default_window__"
            existing_positions = window_positions.get(effective_window)
            if existing_positions is not None:
                if machine_position is None or None in existing_positions:
                    raise DeviceControllerRegistryError('两台已启用设备不能绑定同一个机械臂控制窗口。')
                reject_if(machine_position in existing_positions,
                    DeviceControllerRegistryError('同一卖家控制窗口必须为每台设备绑定不同的 machine_position。'))
            else:
                existing_positions = set()
                window_positions[effective_window] = existing_positions
                enabled_windows.add(effective_window)
            existing_positions.add(machine_position)
            operation_lock = window_locks.setdefault(effective_window, threading.RLock())
            capture_lock = window_capture_locks.setdefault(effective_window, threading.RLock())
            calibration_path = Path(calibration_value or "tap_calibration.json")
            if not calibration_path.is_absolute():
                calibration_path = self.path.parent / calibration_path
            controller: RobotController
            if mock:
                controller = MockRobotController(device_id=device_id)
            elif window_title:
                controller = RobotController(window_title, calibration_path=calibration_path,
                    verified_actions=verified_actions, device_id=device_id, machine_position=machine_position,
                    operation_lock=operation_lock, capture_lock=capture_lock)
            else:
                controller = RobotController(calibration_path=calibration_path, verified_actions=verified_actions,
                    device_id=device_id, machine_position=machine_position, operation_lock=operation_lock,
                    capture_lock=capture_lock)
            self._controllers[device_id] = controller
            self._descriptors[device_id] = {'device_id': device_id, 'window_title': window_title,
                'machine_position': machine_position, 'calibration_path': str(calibration_path),
                'verified_actions': sorted(controller.verified_actions)}
        reject_if(not self._controllers or self.default_device_id not in self._controllers, DeviceControllerRegistryError('设备注册表必须包含已启用的 default_device_id。'))

    def controller(self, device_id: str) -> RobotController:
        resolved = str(device_id or "").strip()
        controller = self._controllers.get(resolved)
        reject_if(controller is None, DeviceControllerRegistryError(f'device_id 未登记或未启用：{resolved or 'missing'}。'))
        return controller

    def provisional_controller(self, device_id: str, candidate_action: str) -> RobotController:
        """Create an unregistered controller for one evidence-bound trial."""

        resolved_device = str(device_id or "").strip()
        action = str(candidate_action or "").strip()
        physical_action = physical_capability_for_action(action)
        reject_if(action not in self._promotable_actions, ProvisionalDeviceControllerError(f'动作 {action or 'missing'} 不能进入真机能力验收。'))
        try:
            original = self._controllers[resolved_device]
            descriptor = self._descriptors[resolved_device]
        except KeyError as exc:
            raise ProvisionalDeviceControllerError(f'device_id 未登记或未启用：{resolved_device or 'missing'}。') from exc
        reject_if(physical_action in original.verified_actions,
            ProvisionalDeviceControllerError(f'设备能力 {action} 对应的物理能力已经通过真机验收。'))
        verified_actions = set(original.verified_actions) | {physical_action}
        if isinstance(original, MockRobotController):
            return MockRobotController(verified_actions=verified_actions, device_id=resolved_device)
        return RobotController(descriptor['window_title'] or original.title,
            calibration_path=Path(descriptor['calibration_path']), verified_actions=verified_actions,
            device_id=resolved_device)

    def descriptors(self) -> list[dict[str, Any]]:
        return [dict(self._descriptors[key]) for key in sorted(self._descriptors)]
