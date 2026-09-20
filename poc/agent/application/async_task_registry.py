"""Bounded in-process task metadata for asynchronous HTTP work."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading
import time
from collections.abc import Callable
from typing import Any


class AsyncTaskRegistry:
    """Own task metadata and executor lifecycle without exposing its storage."""

    def __init__(
        self,
        *,
        ttl_seconds: float,
        max_tasks: int,
        max_workers: int = 2,
        thread_name_prefix: str = "agent-start",
    ) -> None:
        self.ttl_seconds = max(60.0, float(ttl_seconds))
        self.max_tasks = max(16, int(max_tasks))
        self._tasks: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=thread_name_prefix,
        )

    def prune(self, *, now: float | None = None) -> None:
        current = time.monotonic() if now is None else float(now)
        with self._lock:
            expired = [
                task_id
                for task_id, task in self._tasks.items()
                if current - float(task.get("created_monotonic", current))
                >= self.ttl_seconds * (2 if task.get("status") == "running" else 1)
            ]
            for task_id in expired:
                self._tasks.pop(task_id, None)

    def reserve(self, task_id: str) -> None:
        self.prune()
        with self._lock:
            if len(self._tasks) >= self.max_tasks:
                raise OverflowError("异步启动任务队列已满，请稍后重试。")
            self._tasks[task_id] = {
                "status": "running",
                "created_monotonic": time.monotonic(),
            }

    def submit(self, task_id: str, operation: Callable[[], Any]) -> None:
        try:
            self._executor.submit(self._run, task_id, operation)
        except RuntimeError:
            self.remove(task_id)
            raise

    def _run(self, task_id: str, operation: Callable[[], Any]) -> None:
        try:
            result = operation()
        except Exception as exc:
            self._finish(task_id, status="failed", error=str(exc))
        else:
            self._finish(task_id, status="completed", result=result)

    def _finish(self, task_id: str, **values: Any) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            self._tasks[task_id] = {
                "created_monotonic": task.get("created_monotonic", time.monotonic()),
                **values,
            }

    def remove(self, task_id: str) -> None:
        with self._lock:
            self._tasks.pop(task_id, None)

    def get(self, task_id: str) -> dict[str, Any] | None:
        self.prune()
        with self._lock:
            task = self._tasks.get(task_id)
            return dict(task) if task is not None else None

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


__all__ = ["AsyncTaskRegistry"]
