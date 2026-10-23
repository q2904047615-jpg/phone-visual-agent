"""Device registry and per-controller exclusivity."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator

from pva.config import AppConfig
from pva.devices.base import Device, DeviceError, StopRequested


class DeviceBusy(RuntimeError):
    pass


def build_device(config: AppConfig, device_id: str) -> Device:
    entry = config.device(device_id)
    if entry.driver == "mock":
        from pva.devices.mock import MockDevice
        return MockDevice(entry, config.adb_executable)
    if entry.driver == "adb":
        from pva.devices.adb_device import AdbDevice
        return AdbDevice(entry, config.adb_executable)
    from pva.devices.robot import RobotDevice
    return RobotDevice(entry, config.controller, config.adb_executable)


class DeviceManager:
    """Owns every enabled device.

    ``reserve`` gives one owner (a task or plugin) exclusive use of a physical
    controller; devices that share a robot window share that reservation.
    ``hardware`` serialises individual physical operations (including preview
    capture and manual actions) on that controller.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.devices: dict[str, Device] = {
            entry.id: build_device(config, entry.id) for entry in config.devices if entry.enabled
        }
        if not self.devices:
            raise DeviceError("没有启用的设备。")
        self.default_device = config.default_device if config.default_device in self.devices else next(iter(self.devices))
        self._owners: dict[str, tuple[str, str]] = {}   # hardware_key -> (device_id, owner)
        self._locks: dict[str, threading.RLock] = {}
        self._guard = threading.Lock()

    def get(self, device_id: str | None) -> Device:
        device = self.devices.get(device_id or self.default_device)
        if device is None:
            raise DeviceError(f"未知或未启用的设备：{device_id}")
        return device

    def owner(self, device_id: str) -> str | None:
        device = self.get(device_id)
        entry = self._owners.get(device.hardware_key)
        return entry[1] if entry else None

    def reserve(self, device_id: str, owner: str) -> Device:
        device = self.get(device_id)
        with self._guard:
            current = self._owners.get(device.hardware_key)
            if current and current[1] != owner:
                holder = self.devices.get(current[0])
                raise DeviceBusy(f"{holder.name if holder else current[0]} 正在被任务 {current[1][:8]} 使用；"
                                 "同一控制端同时只能运行一个任务。")
            self._owners[device.hardware_key] = (device.id, owner)
        return device

    def release(self, device_id: str, owner: str) -> None:
        device = self.get(device_id)
        with self._guard:
            if self._owners.get(device.hardware_key, (None, None))[1] == owner:
                del self._owners[device.hardware_key]

    @contextmanager
    def hardware(self, device_id: str, timeout: float | None = None) -> Iterator[Device]:
        device = self.get(device_id)
        with self._guard:
            lock = self._locks.setdefault(device.hardware_key, threading.RLock())
        if not lock.acquire(timeout=-1 if timeout is None else timeout):
            raise DeviceBusy("控制端正在执行动作，请稍候。")
        try:
            yield device
        finally:
            lock.release()


__all__ = ["Device", "DeviceBusy", "DeviceError", "DeviceManager", "StopRequested", "build_device"]
