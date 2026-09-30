"""Bounded in-process task metadata for asynchronous HTTP work."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading
import time
from collections.abc import Callable
from typing import Any


class AsyncTaskCancelled(RuntimeError):
    """Raised internally when a queued asynchronous operation is cancelled."""


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

    def reserve(self, task_id: str, *, metadata: dict[str, Any] | None = None) -> None:
        self.prune()
        with self._lock:
            if len(self._tasks) >= self.max_tasks:
                raise OverflowError("异步启动任务队列已满，请稍后重试。")
            self._tasks[task_id] = {
                "status": "queued",
                "created_monotonic": time.monotonic(),
                "cancel_event": threading.Event(),
                "future": None,
                **dict(metadata or {}),
            }

    def submit(self, task_id: str, operation: Callable[[], Any]) -> None:
        try:
            future = self._executor.submit(self._run, task_id, operation)
        except RuntimeError:
            self.remove(task_id)
            raise
        with self._lock:
            task = self._tasks.get(task_id)
            if task is not None:
                task["future"] = future
                if task.get("status") == "queued":
                    task["status"] = "running"

    def _run(self, task_id: str, operation: Callable[[], Any]) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            cancel_event = task.get("cancel_event") if task is not None else None
            if task is None or (cancel_event is not None and cancel_event.is_set()):
                if task is not None:
                    self._finish(
                        task_id,
                        status="cancelled",
                        error="异步启动任务已取消，尚未执行。",
                        error_detail={
                            "code": "async_task_cancelled",
                            "message": "异步启动任务已取消，尚未执行。",
                        },
                    )
                return
        try:
            result = operation()
        except AsyncTaskCancelled as exc:
            self._finish(
                task_id,
                status="cancelled",
                error=str(exc),
                error_detail={"code": "async_task_cancelled", "message": str(exc)},
            )
        except Exception as exc:
            self._finish(
                task_id,
                status="failed",
                error=str(exc),
                error_detail=self._exception_detail(exc),
            )
        else:
            with self._lock:
                task = self._tasks.get(task_id)
                cancelled = bool(
                    task is not None
                    and task.get("cancel_event") is not None
                    and task["cancel_event"].is_set()
                )
            if cancelled:
                # The worker may already have entered device/model I/O. Do
                # not report a successful side-effecting operation as
                # cancelled; preserve the result and expose that cancellation
                # arrived too late for the caller to undo it.
                self._finish(
                    task_id,
                    status="completed",
                    cancel_requested=True,
                    error="已收到取消请求，但启动操作已经开始并完成，无法回滚。",
                    error_detail={
                        "code": "async_task_cancel_too_late",
                        "message": "启动操作已经开始并完成，无法回滚。",
                    },
                    result=result,
                )
            else:
                self._finish(task_id, status="completed", result=result)

    @staticmethod
    def _exception_detail(exc: Exception) -> Any:
        """Keep structured HTTP/application error details for the caller."""

        detail = getattr(exc, "detail", None)
        if detail is not None:
            return detail
        return {
            "code": type(exc).__name__,
            "message": str(exc),
        }

    def _finish(self, task_id: str, **values: Any) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            task.update(values)

    def cancellation_requested(self, task_id: str) -> bool:
        with self._lock:
            task = self._tasks.get(task_id)
            event = task.get("cancel_event") if task is not None else None
            return bool(event is not None and event.is_set())

    def cancel(self, task_id: str) -> dict[str, Any] | None:
        """Cancel a queued start, or mark a running start as cancelling.

        Python cannot safely terminate a worker already inside device/model
        I/O. In that case the operation is allowed to finish, but its result
        is never reported as a successful ticket. A queued worker is removed
        before it can call the operation.
        """

        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            status = str(task.get("status") or "")
            if status in {"completed", "failed", "cancelled"}:
                return self._public_task(task)
            event = task.get("cancel_event")
            if event is not None:
                event.set()
            future = task.get("future")
            # ``future`` is None only in the tiny reserve→submit window; that
            # worker cannot have started yet, so treat it as cancelled too.
            cancelled_before_start = bool(future is None or future.cancel())
            task["status"] = "cancelled" if cancelled_before_start else "cancelling"
            task["error"] = "异步启动任务已取消，等待当前操作退出。"
            task["error_detail"] = {
                "code": "async_task_cancelled",
                "message": "异步启动任务已取消，等待当前操作退出。",
            }
            return self._public_task(task)

    def active_for(self, key: str, value: Any) -> dict[str, Any] | None:
        """Return one queued/running ticket matching metadata, if any."""

        self.prune()
        with self._lock:
            for task in self._tasks.values():
                if task.get("status") in {"queued", "running", "cancelling"} and task.get(key) == value:
                    return self._public_task(task)
        return None

    @staticmethod
    def _public_task(task: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in task.items() if key not in {"cancel_event", "future"}}

    def remove(self, task_id: str) -> None:
        with self._lock:
            self._tasks.pop(task_id, None)

    def get(self, task_id: str) -> dict[str, Any] | None:
        self.prune()
        with self._lock:
            task = self._tasks.get(task_id)
            return self._public_task(task) if task is not None else None

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


__all__ = ["AsyncTaskCancelled", "AsyncTaskRegistry"]
