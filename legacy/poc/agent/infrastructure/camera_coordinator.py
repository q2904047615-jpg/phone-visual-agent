from __future__ import annotations

from agent.domain.validation import reject_if
from contextlib import contextmanager
from io import BytesIO
import threading
from typing import Any, Callable, Iterator


class CameraPreviewUnavailable(RuntimeError):
    """The live camera is leased and no cached frame exists yet."""


class DeviceCameraCoordinator:
    """Give one closed-loop task priority over passive browser previews."""

    def __init__(self) -> None:
        self._serial_lock = threading.RLock()
        self._cached_preview: bytes | None = None

    @staticmethod
    def _jpeg(frame: Any, *, quality: int=72) -> bytes:
        buffer = BytesIO()
        frame.convert('RGB').save(buffer, format='JPEG', quality=max(1, min(95, int(quality))), optimize=True)
        return buffer.getvalue()

    @contextmanager
    def serial_session(self) -> Iterator[None]:
        with self._serial_lock:
            yield

    def capture_agent_frame(self, capture: Callable[[], Any]) -> Any:
        with self._serial_lock:
            frame = capture().convert("RGB")
            self._cached_preview = self._jpeg(frame)
            return frame

    def capture_preview(self, capture: Callable[..., bytes], *, quality: int, cache_only: bool) -> tuple[bytes, bool]:
        if cache_only:
            reject_if(self._cached_preview is None, CameraPreviewUnavailable('任务正在独占相机，尚无可复用的缓存画面。'))
            return self._cached_preview, True

        acquired = self._serial_lock.acquire(blocking=False)
        if not acquired:
            reject_if(self._cached_preview is None, CameraPreviewUnavailable('任务正在独占相机，尚无可复用的缓存画面。'))
            return self._cached_preview, True
        try:
            content = bytes(capture(quality=quality))
            self._cached_preview = content
            return content, False
        finally:
            self._serial_lock.release()
