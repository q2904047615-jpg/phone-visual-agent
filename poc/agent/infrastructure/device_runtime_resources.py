from __future__ import annotations

import threading
from collections.abc import Iterable

from .camera_coordinator import DeviceCameraCoordinator


class DeviceRuntimeResourceError(RuntimeError):
    """A per-device runtime resource cannot be resolved."""


class DeviceRuntimeResourceRegistry:
    """Own stable coordination resources independently for each device."""

    def __init__(self, initial_device_ids: Iterable[str]=()) -> None:
        self._coordination_lock_guard = threading.RLock()
        self._coordination_locks: dict[str, threading.Lock] = {}
        self._camera_coordinator_guard = threading.RLock()
        self._camera_coordinators: dict[str, DeviceCameraCoordinator] = {}
        for device_id in initial_device_ids:
            resolved = self._resolved_device_id(device_id)
            self._coordination_locks[resolved] = threading.Lock()
            self._camera_coordinators[resolved] = DeviceCameraCoordinator()

    @staticmethod
    def _resolved_device_id(device_id: str) -> str:
        resolved = str(device_id or "").strip()
        if not resolved:
            raise DeviceRuntimeResourceError("device_id 不能为空。")
        return resolved

    def coordination_lock(self, device_id: str) -> threading.Lock:
        resolved = self._resolved_device_id(device_id)
        with self._coordination_lock_guard:
            return self._coordination_locks.setdefault(resolved, threading.Lock())

    def camera_coordinator(self, device_id: str) -> DeviceCameraCoordinator:
        resolved = self._resolved_device_id(device_id)
        with self._camera_coordinator_guard:
            return self._camera_coordinators.setdefault(resolved, DeviceCameraCoordinator())
