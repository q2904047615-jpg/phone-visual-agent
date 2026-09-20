from __future__ import annotations

import time
import unittest

from agent.application.async_task_registry import AsyncTaskRegistry


class AsyncTaskRegistryTests(unittest.TestCase):
    def test_completed_result_preserves_lifecycle_metadata(self) -> None:
        registry = AsyncTaskRegistry(ttl_seconds=60, max_tasks=16)
        try:
            registry.reserve("task-1")
            registry.submit("task-1", lambda: {"ok": True})
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                task = registry.get("task-1")
                if task and task.get("status") == "completed":
                    break
                time.sleep(0.01)
            else:
                self.fail("异步任务没有在测试期限内完成")
            self.assertEqual({"ok": True}, task["result"])
            self.assertIn("created_monotonic", task)
        finally:
            registry.shutdown()

    def test_expired_metadata_is_pruned_without_running_worker(self) -> None:
        registry = AsyncTaskRegistry(ttl_seconds=60, max_tasks=16)
        try:
            registry.reserve("task-1")
            created = registry.get("task-1")["created_monotonic"]
            registry.prune(now=created + 121)
            self.assertIsNone(registry.get("task-1"))
        finally:
            registry.shutdown()

    def test_submit_after_shutdown_rolls_back_reserved_metadata(self) -> None:
        registry = AsyncTaskRegistry(ttl_seconds=60, max_tasks=16)
        registry.reserve("task-1")
        registry.shutdown()
        with self.assertRaises(RuntimeError):
            registry.submit("task-1", lambda: None)
        self.assertIsNone(registry.get("task-1"))


if __name__ == "__main__":
    unittest.main()
