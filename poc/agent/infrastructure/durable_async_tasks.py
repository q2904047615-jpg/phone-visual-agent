"""Single-owner async-start receipts persisted across executor restarts.

Closures and credentials are never serialized. Interrupted operations are
reported for reconciliation, not reconstructed or automatically submitted.
"""
import json
import time
from pathlib import Path

from agent.application.async_task_registry import AsyncTaskRegistry
from agent.infrastructure.command_inbox import CommandInbox


class DurableAsyncTaskRegistry(AsyncTaskRegistry):
    def __init__(self, database: Path, **kwargs):
        self.store = CommandInbox(database)
        super().__init__(**kwargs)
        with self.store.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS async_start_receipts "
                       "(task_id TEXT PRIMARY KEY, updated_at REAL NOT NULL, payload TEXT NOT NULL)")
            rows = db.execute("SELECT task_id,payload FROM async_start_receipts").fetchall()
            for task_id, encoded in rows:
                task = json.loads(encoded)
                if task["status"] not in {"queued", "running", "cancelling"}:
                    continue
                dispatched = bool(task.get("dispatch_started"))
                task.update(status="failed", recovery_required=dispatched,
                            error_detail={"code":"async_start_outcome_unknown" if dispatched else "async_start_not_dispatched",
                                          "message":"执行进程已重启；请查看关联 Run 并按恢复合同核对，禁止自动重放。" if dispatched else "执行进程在启动前重启，本次尚未派发。"})
                db.execute("UPDATE async_start_receipts SET updated_at=?,payload=? WHERE task_id=?",
                           (time.time(), json.dumps(task, ensure_ascii=False), task_id))

    def _persist(self, task_id):
        task = self._tasks.get(task_id)
        if task is None:
            return
        value = self._public_task(task)
        value.pop("created_monotonic", None)
        with self.store.connect() as db:
            db.execute("INSERT INTO async_start_receipts VALUES(?,?,?) ON CONFLICT(task_id) "
                       "DO UPDATE SET updated_at=excluded.updated_at,payload=excluded.payload",
                       (task_id, time.time(), json.dumps(value, ensure_ascii=False)))

    def reserve(self, task_id, *, metadata=None):
        with self._lock:
            super().reserve(task_id, metadata=metadata)
            self._persist(task_id)

    def submit(self, task_id, operation):
        with self._lock:
            super().submit(task_id, operation)
            self._persist(task_id)

    def _run(self, task_id, operation):
        with self._lock:
            task = self._tasks.get(task_id)
            if task is not None and not task["cancel_event"].is_set():
                task["dispatch_started"] = True
                self._persist(task_id)
        return super()._run(task_id, operation)

    def _finish(self, task_id, **values):
        with self._lock:
            super()._finish(task_id, **values)
            self._persist(task_id)

    def cancel(self, task_id):
        with self._lock:
            result = super().cancel(task_id)
            self._persist(task_id)
            return result if result is not None else self.get(task_id)

    def get(self, task_id):
        with self._lock:
            result = super().get(task_id)
            if result is not None:
                return result
            with self.store.connect() as db:
                row = db.execute("SELECT payload FROM async_start_receipts WHERE task_id=?", (task_id,)).fetchone()
            return json.loads(row[0]) if row else None

    def remove(self, task_id):
        with self._lock:
            super().remove(task_id)
            with self.store.connect() as db:
                db.execute("DELETE FROM async_start_receipts WHERE task_id=?", (task_id,))

    def active_tasks(self):
        with self._lock:
            return [self._public_task(task) for task in self._tasks.values()
                    if task["status"] in {"queued", "running", "cancelling"}]

    def shutdown(self):
        # The executor OS lock must outlive every running closure. A forced
        # process stop instead releases the OS lock and recovers as unknown.
        self._executor.shutdown(wait=True, cancel_futures=True)

    def prune(self, *, now=None):
        # Never prune running closures. Match the existing TTL for completed
        # receipts; keep unresolved outcomes until explicit Runtime recovery.
        current = time.monotonic() if now is None else now
        with self._lock:
            for task_id, task in list(self._tasks.items()):
                if task["status"] in {"completed", "failed", "cancelled"} and current - task["created_monotonic"] >= self.ttl_seconds:
                    self._tasks.pop(task_id)
            with self.store.connect() as db:
                db.execute("DELETE FROM async_start_receipts WHERE updated_at<? "
                           "AND json_extract(payload,'$.status') IN ('completed','failed','cancelled') "
                           "AND COALESCE(json_extract(payload,'$.recovery_required'),0)=0",
                           (time.time() - self.ttl_seconds,))
