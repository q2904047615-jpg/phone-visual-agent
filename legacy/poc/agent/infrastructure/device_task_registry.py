from __future__ import annotations

from agent.domain.validation import reject_if
from contextlib import contextmanager
import hashlib
from pathlib import Path
import threading
from typing import Iterator

from agent.domain import DeviceTaskRegistryError
from agent.domain.session import TERMINAL_SESSION_STATUSES
from .device_exclusivity import InterProcessLease


class DeviceTaskRegistry:
    """Own active-session identity and one re-entrant lock per device."""

    TERMINAL_STATUSES = TERMINAL_SESSION_STATUSES

    def __init__(self, *, lease_directory: Path | None=None) -> None:
        self._guard = threading.RLock()
        self._locks: dict[str, threading.RLock] = {}
        self._active: dict[str, str] = {}
        self._owners: dict[str, tuple[int, int]] = {}
        self._lease_directory = Path(lease_directory) if lease_directory is not None else None
        self._leases: dict[str, InterProcessLease] = {}

    def _lease_path(self, device_id: str) -> Path | None:
        if self._lease_directory is None:
            return None
        digest = hashlib.sha256(device_id.encode("utf-8")).hexdigest()[:24]
        return self._lease_directory / f"device_{digest}.lease"

    @staticmethod
    def _id(value: str, field_name: str) -> str:
        result = str(value or "").strip()
        reject_if(not result, DeviceTaskRegistryError(f"{field_name} 不能为空。"))
        return result

    def reserve(self, device_id: str, session_id: str) -> None:
        device = self._id(device_id, "device_id")
        session = self._id(session_id, "session_id")
        with self._guard:
            active = self._active.get(device)
            reject_if(active is not None and active != session, DeviceTaskRegistryError(f'设备 {device} 已有活动任务：{active}。'))
            lease_path = self._lease_path(device)
            if lease_path is not None and device not in self._leases:
                lease = InterProcessLease(lease_path, owner_id=session, metadata={'device_id': device,
                    'session_id': session})
                if not lease.acquire():
                    payload = InterProcessLease.active_payload(lease_path) or {}
                    owner = str(payload.get("session_id") or "另一个进程")
                    raise DeviceTaskRegistryError(f'设备 {device} 已有活动任务：{owner}。')
                self._leases[device] = lease
            self._active[device] = session
            self._locks.setdefault(device, threading.RLock())

    def release(self, device_id: str, session_id: str) -> None:
        device = self._id(device_id, "device_id")
        session = self._id(session_id, "session_id")
        with self._guard:
            if self._active.get(device) == session:
                self._active.pop(device, None)
                lease = self._leases.pop(device, None)
                if lease is not None:
                    lease.release()

    def active_session(self, device_id: str) -> str | None:
        device = self._id(device_id, "device_id")
        with self._guard:
            local = self._active.get(device)
            if local is not None:
                return local
            lease_path = self._lease_path(device)
            if lease_path is None:
                return None
            payload = InterProcessLease.active_payload(lease_path) or {}
            return str(payload.get("session_id") or "").strip() or None

    @contextmanager
    def device_lock(self, device_id: str) -> Iterator[None]:
        device = self._id(device_id, "device_id")
        with self._guard:
            lock = self._locks.setdefault(device, threading.RLock())
        lock.acquire()
        thread_id = threading.get_ident()
        with self._guard:
            owner, depth = self._owners.get(device, (thread_id, 0))
            if depth and owner != thread_id:
                lock.release()
                raise DeviceTaskRegistryError(f"设备锁所有者异常：{device}。")
            self._owners[device] = (thread_id, depth + 1)
        try:
            yield
        finally:
            with self._guard:
                owner, depth = self._owners.get(device, (thread_id, 1))
                if owner == thread_id and depth <= 1:
                    self._owners.pop(device, None)
                elif owner == thread_id:
                    self._owners[device] = (owner, depth - 1)
            lock.release()

    def is_locked_by_current_thread(self, device_id: str) -> bool:
        device = self._id(device_id, "device_id")
        with self._guard:
            owner = self._owners.get(device)
            return bool(owner and owner[0] == threading.get_ident() and (owner[1] > 0))
