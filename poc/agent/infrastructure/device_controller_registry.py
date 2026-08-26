from __future__ import annotations

import json
from collections.abc import Collection
from pathlib import Path
from typing import Any

from robot_core import MockRobotController, RobotController


class DeviceControllerRegistryError(RuntimeError):
    """The configured device-controller registry is unavailable or invalid."""


class ProvisionalDeviceControllerError(DeviceControllerRegistryError):
    """A provisional capability controller cannot be created."""


class DeviceControllerRegistry:
    """Resolve one controller and calibration per configured device_id."""

    def __init__(
        self,
        path: Path,
        *,
        promotable_actions: Collection[str],
        mock: bool = False,
    ) -> None:
        self.path = Path(path)
        self._promotable_actions = frozenset(
            str(action or "").strip() for action in promotable_actions
        )
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise DeviceControllerRegistryError(
                f"设备注册表无法读取：{exc}"
            ) from exc
        if payload.get("version") != 1 or not isinstance(
            payload.get("devices"),
            list,
        ):
            raise DeviceControllerRegistryError(
                "设备注册表版本或 devices 格式无效。"
            )
        self.default_device_id = str(
            payload.get("default_device_id") or ""
        ).strip()
        self._controllers: dict[str, RobotController] = {}
        self._descriptors: dict[str, dict[str, Any]] = {}
        enabled_windows: set[str] = set()
        for raw in payload["devices"]:
            if not isinstance(raw, dict) or raw.get("enabled") is not True:
                continue
            device_id = str(raw.get("device_id") or "").strip()
            window_title = str(raw.get("window_title") or "").strip()
            calibration_value = str(raw.get("calibration_path") or "").strip()
            raw_verified_actions = raw.get("verified_actions")
            if raw_verified_actions is None:
                verified_actions = None
            elif not isinstance(raw_verified_actions, list) or not all(
                isinstance(item, str) and item.strip()
                for item in raw_verified_actions
            ):
                raise DeviceControllerRegistryError(
                    f"设备 {device_id or 'missing'} 的 verified_actions 格式无效。"
                )
            else:
                verified_actions = {
                    str(item).strip() for item in raw_verified_actions
                }
            if not device_id or device_id in self._controllers:
                raise DeviceControllerRegistryError(
                    "设备注册表存在空或重复的 device_id。"
                )
            effective_window = window_title or "__default_window__"
            if effective_window in enabled_windows:
                raise DeviceControllerRegistryError(
                    "两台已启用设备不能绑定同一个机械臂控制窗口。"
                )
            enabled_windows.add(effective_window)
            calibration_path = Path(calibration_value or "tap_calibration.json")
            if not calibration_path.is_absolute():
                calibration_path = self.path.parent / calibration_path
            controller: RobotController
            if mock:
                controller = MockRobotController(device_id=device_id)
            elif window_title:
                controller = RobotController(
                    window_title,
                    calibration_path=calibration_path,
                    verified_actions=verified_actions,
                    device_id=device_id,
                )
            else:
                controller = RobotController(
                    calibration_path=calibration_path,
                    verified_actions=verified_actions,
                    device_id=device_id,
                )
            self._controllers[device_id] = controller
            self._descriptors[device_id] = {
                "device_id": device_id,
                "window_title": window_title,
                "calibration_path": str(calibration_path),
                "verified_actions": sorted(controller.verified_actions),
            }
        if not self._controllers or self.default_device_id not in self._controllers:
            raise DeviceControllerRegistryError(
                "设备注册表必须包含已启用的 default_device_id。"
            )

    def controller(self, device_id: str) -> RobotController:
        resolved = str(device_id or "").strip()
        controller = self._controllers.get(resolved)
        if controller is None:
            raise DeviceControllerRegistryError(
                f"device_id 未登记或未启用：{resolved or 'missing'}。"
            )
        return controller

    def provisional_controller(
        self,
        device_id: str,
        candidate_action: str,
    ) -> RobotController:
        """Create an unregistered controller for one evidence-bound trial."""

        resolved_device = str(device_id or "").strip()
        action = str(candidate_action or "").strip()
        if action not in self._promotable_actions:
            raise ProvisionalDeviceControllerError(
                f"动作 {action or 'missing'} 不能进入真机能力验收。"
            )
        try:
            original = self._controllers[resolved_device]
            descriptor = self._descriptors[resolved_device]
        except KeyError as exc:
            raise ProvisionalDeviceControllerError(
                f"device_id 未登记或未启用：{resolved_device or 'missing'}。"
            ) from exc
        if action in original.verified_actions:
            raise ProvisionalDeviceControllerError(
                f"设备能力 {action} 已经通过真机验收。"
            )
        verified_actions = set(original.verified_actions) | {action}
        if isinstance(original, MockRobotController):
            return MockRobotController(
                verified_actions=verified_actions,
                device_id=resolved_device,
            )
        return RobotController(
            descriptor["window_title"] or original.title,
            calibration_path=Path(descriptor["calibration_path"]),
            verified_actions=verified_actions,
            device_id=resolved_device,
        )

    def descriptors(self) -> list[dict[str, Any]]:
        return [
            dict(self._descriptors[key])
            for key in sorted(self._descriptors)
        ]
