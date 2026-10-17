from pathlib import Path
import tempfile
import time
import unittest
from threading import Event

from agent.infrastructure.durable_async_tasks import DurableAsyncTaskRegistry


class DurableAsyncTaskTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "dispatch.sqlite3"

    def registry(self):
        registry = DurableAsyncTaskRegistry(self.path, ttl_seconds=3600, max_tasks=16)
        self.addCleanup(registry.shutdown)
        return registry

    def test_completed_receipt_survives_restart(self):
        first = self.registry()
        first.reserve("task", metadata={"session_id":"run"})
        first.submit("task", lambda: {"session":{"session_id":"run"}})
        deadline = time.monotonic() + 3
        while first.get("task")["status"] != "completed":
            self.assertLess(time.monotonic(), deadline)
            time.sleep(.01)
        first.shutdown()
        self.assertEqual("run", self.registry().get("task")["result"]["session"]["session_id"])

    def test_recovery_never_replays_pending_or_dispatched_closure(self):
        first = self.registry()
        first.reserve("pending", metadata={"session_id":"run-pending"})
        first.reserve("unknown", metadata={"session_id":"run-unknown"})
        with first._lock:
            first._tasks["unknown"].update(status="running", dispatch_started=True)
            first._persist("unknown")
        first.shutdown()
        restarted = self.registry()
        self.assertEqual("async_start_not_dispatched", restarted.get("pending")["error_detail"]["code"])
        self.assertEqual("async_start_outcome_unknown", restarted.get("unknown")["error_detail"]["code"])
        self.assertTrue(restarted.get("unknown")["recovery_required"])
        self.assertEqual({}, restarted._tasks)

    def test_cancel_before_submit_stays_cancelled_after_restart(self):
        registry = self.registry()
        registry.reserve("cancel")
        registry.cancel("cancel")
        registry.shutdown()
        self.assertEqual("cancelled", self.registry().get("cancel")["status"])

    def test_running_operation_is_not_pruned(self):
        registry = self.registry()
        release, entered = Event(), Event()
        self.addCleanup(release.set)
        registry.reserve("running")
        registry.submit("running", lambda: (entered.set(), release.wait(3), {} )[-1])
        self.assertTrue(entered.wait(2))
        registry.prune(now=time.monotonic() + 10000)
        self.assertEqual("running", registry.get("running")["status"])
        release.set()
        registry._executor.shutdown(wait=True)

    def test_submission_failure_does_not_leave_a_persistent_queued_ticket(self):
        registry = self.registry()
        registry.reserve("rejected")
        registry.shutdown()
        with self.assertRaises(RuntimeError):
            registry.submit("rejected", lambda: {})
        self.assertIsNone(registry.get("rejected"))
        self.assertIsNone(self.registry().get("rejected"))
