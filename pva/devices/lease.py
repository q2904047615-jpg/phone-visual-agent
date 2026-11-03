"""Cross-process device lease, compatible with the legacy console and the platform connector.

Every program that drives a physical phone holds an OS byte-range lock on
``<lease dir>/device_<sha256(device_id)[:24]>.lease`` while it owns the device;
the file body after the first byte is JSON describing the holder. Using the same
directory and file names as ``legacy/poc`` keeps this app, the old console and the
device connector from moving the same arm at the same time.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

DEFAULT_LEASE_DIR = Path(os.environ.get("PHONE_VISUAL_AGENT_LEASE_DIR",
                                        str(Path(tempfile.gettempdir()) / "phone_visual_agent_device_leases")))


def lease_path(lease_dir: Path, device_id: str) -> Path:
    digest = hashlib.sha256(device_id.encode("utf-8")).hexdigest()[:24]
    return Path(lease_dir) / f"device_{digest}.lease"


def _try_lock(descriptor: int) -> bool:
    try:
        if os.name == "nt":
            import msvcrt
            os.lseek(descriptor, 0, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def _unlock(descriptor: int) -> None:
    if os.name == "nt":
        import msvcrt
        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(descriptor, fcntl.LOCK_UN)


def _open(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(str(path), os.O_CREAT | os.O_RDWR)
    if os.fstat(descriptor).st_size == 0:
        os.write(descriptor, b" ")
    return descriptor


class InterProcessLease:
    def __init__(self, path: Path, *, owner_id: str, metadata: dict[str, Any]):
        self.path = Path(path)
        self.owner_id = owner_id
        self.metadata = dict(metadata)
        self._descriptor: int | None = None

    @staticmethod
    def active_payload(path: Path) -> dict[str, Any] | None:
        """The holder's JSON when another descriptor holds the lock, else None."""
        descriptor = _open(Path(path))
        try:
            if _try_lock(descriptor):
                _unlock(descriptor)
                return None
            try:
                os.lseek(descriptor, 1, os.SEEK_SET)
                value = json.loads(os.read(descriptor, 65536).decode("utf-8").strip() or "null")
            except (OSError, UnicodeError, ValueError):
                value = None
            return value if isinstance(value, dict) else {"session_id": "未知程序"}
        finally:
            os.close(descriptor)

    def acquire(self) -> bool:
        if self._descriptor is not None:
            return True
        descriptor = _open(self.path)
        if not _try_lock(descriptor):
            os.close(descriptor)
            return False
        payload = {"pid": os.getpid(), "owner_id": self.owner_id, "token": uuid.uuid4().hex, **self.metadata}
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        os.lseek(descriptor, 1, os.SEEK_SET)
        os.write(descriptor, encoded)
        os.ftruncate(descriptor, 1 + len(encoded))
        self._descriptor = descriptor
        return True

    def release(self) -> None:
        descriptor, self._descriptor = self._descriptor, None
        if descriptor is None:
            return
        try:
            os.ftruncate(descriptor, 1)
            _unlock(descriptor)
        finally:
            os.close(descriptor)
