from __future__ import annotations

import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any


SHARED_DEVICE_LEASE_DIR = Path(
    os.environ.get(
        "PHONE_VISUAL_AGENT_LEASE_DIR",
        str(Path(tempfile.gettempdir()) / "phone_visual_agent_device_leases"),
    )
)


def _try_lock(descriptor: int) -> bool:
    if os.name == "nt":
        import msvcrt

        try:
            os.lseek(descriptor, 0, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False
    import fcntl

    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def _unlock(descriptor: int) -> None:
    if os.name == "nt":
        import msvcrt

        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(descriptor, fcntl.LOCK_UN)


class InterProcessLease:
    """An OS-backed non-blocking lease with inspectable JSON metadata."""

    def __init__(self, path: Path, *, owner_id: str, metadata: dict[str, Any]) -> None:
        self.path = Path(path)
        self.owner_id = str(owner_id)
        self.metadata = dict(metadata)
        self.token = uuid.uuid4().hex
        self.acquired = False
        self._descriptor: int | None = None

    @staticmethod
    def _open(path: Path) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(str(path), os.O_CREAT | os.O_RDWR)
        if os.fstat(descriptor).st_size == 0:
            os.write(descriptor, b" ")
            os.fsync(descriptor)
        return descriptor

    @staticmethod
    def _read_descriptor(descriptor: int) -> dict[str, Any] | None:
        try:
            os.lseek(descriptor, 1, os.SEEK_SET)
            raw = os.read(descriptor, 65536).decode("utf-8").strip()
            value = json.loads(raw) if raw else None
        except (OSError, UnicodeError, ValueError, TypeError):
            return None
        return value if isinstance(value, dict) else None

    @classmethod
    def active_payload(cls, path: Path) -> dict[str, Any] | None:
        descriptor = cls._open(Path(path))
        try:
            if _try_lock(descriptor):
                _unlock(descriptor)
                return None
            return cls._read_descriptor(descriptor) or {'session_id': 'unknown-process', 'unreadable': True}
        finally:
            os.close(descriptor)

    def acquire(self) -> bool:
        if self.acquired:
            return True
        descriptor = self._open(self.path)
        if not _try_lock(descriptor):
            os.close(descriptor)
            return False
        payload = {'pid': os.getpid(), 'owner_id': self.owner_id, 'token': self.token, **self.metadata}
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        os.lseek(descriptor, 1, os.SEEK_SET)
        os.write(descriptor, encoded)
        os.ftruncate(descriptor, 1 + len(encoded))
        os.fsync(descriptor)
        self._descriptor = descriptor
        self.acquired = True
        return True

    def release(self) -> None:
        descriptor = self._descriptor
        if not self.acquired or descriptor is None:
            return
        try:
            os.ftruncate(descriptor, 1)
            os.fsync(descriptor)
            _unlock(descriptor)
        finally:
            os.close(descriptor)
            self._descriptor = None
            self.acquired = False

    def __enter__(self) -> "InterProcessLease":
        if not self.acquire():
            raise RuntimeError("跨进程设备控制权已被占用。")
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.release()
