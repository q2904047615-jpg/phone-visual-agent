from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from agent.domain import DeviceTaskRegistryError
from agent.infrastructure import (
    DeviceTaskRegistry,
    InterProcessLease,
)


class InterProcessLeaseTests(unittest.TestCase):
    def test_device_lock_is_reentrant_for_same_thread(self) -> None:
        registry = DeviceTaskRegistry()

        with registry.device_lock("device-a"):
            self.assertTrue(registry.is_locked_by_current_thread("device-a"))
            with registry.device_lock("device-a"):
                self.assertTrue(
                    registry.is_locked_by_current_thread("device-a")
                )
            self.assertTrue(registry.is_locked_by_current_thread("device-a"))

        self.assertFalse(registry.is_locked_by_current_thread("device-a"))

    def test_real_second_process_cannot_acquire_same_lease(self) -> None:
        child_code = r'''
import sys
from pathlib import Path
from agent.infrastructure import InterProcessLease
lease = InterProcessLease(Path(sys.argv[1]), owner_id="child", metadata={"session_id": "child-session"})
if not lease.acquire():
    raise SystemExit(2)
print("READY", flush=True)
sys.stdin.readline()
lease.release()
'''
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "shared.lease"
            child = subprocess.Popen(
                [sys.executable, "-c", child_code, str(path)],
                cwd=Path(__file__).resolve().parent,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                assert child.stdout is not None
                self.assertEqual("READY", child.stdout.readline().strip())
                competing = InterProcessLease(
                    path,
                    owner_id="parent",
                    metadata={"session_id": "parent-session"},
                )
                self.assertFalse(competing.acquire())
                payload = InterProcessLease.active_payload(path)
                self.assertIsNotNone(payload)
                self.assertEqual("child-session", payload["session_id"])
            finally:
                if child.stdin is not None:
                    child.stdin.write("stop\n")
                    child.stdin.flush()
                    child.stdin.close()
                child.wait(timeout=5)
                if child.stdout is not None:
                    child.stdout.close()
                if child.stderr is not None:
                    child.stderr.close()
            self.assertEqual(0, child.returncode)
            self.assertIsNone(InterProcessLease.active_payload(path))

    def test_device_registry_blocks_same_device_but_allows_another_cross_process(self) -> None:
        child_code = r'''
import sys
from pathlib import Path
from agent.infrastructure import DeviceTaskRegistry
registry = DeviceTaskRegistry(lease_directory=Path(sys.argv[1]))
registry.reserve("device-a", "child-session-a")
print("READY", flush=True)
sys.stdin.readline()
registry.release("device-a", "child-session-a")
'''
        with tempfile.TemporaryDirectory() as temp:
            lease_dir = Path(temp)
            child = subprocess.Popen(
                [sys.executable, "-c", child_code, str(lease_dir)],
                cwd=Path(__file__).resolve().parent,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            parent = DeviceTaskRegistry(lease_directory=lease_dir)
            try:
                assert child.stdout is not None
                self.assertEqual("READY", child.stdout.readline().strip())
                self.assertEqual(
                    "child-session-a",
                    parent.active_session("device-a"),
                )
                with self.assertRaisesRegex(
                    DeviceTaskRegistryError,
                    "已有活动任务",
                ):
                    parent.reserve("device-a", "parent-session-a")

                parent.reserve("device-b", "parent-session-b")
                try:
                    self.assertEqual(
                        "parent-session-b",
                        parent.active_session("device-b"),
                    )
                    self.assertIsNone(child.poll())
                finally:
                    parent.release("device-b", "parent-session-b")
            finally:
                if child.stdin is not None:
                    child.stdin.write("stop\n")
                    child.stdin.flush()
                    child.stdin.close()
                child.wait(timeout=5)
                if child.stdout is not None:
                    child.stdout.close()
                if child.stderr is not None:
                    child.stderr.close()
            self.assertEqual(0, child.returncode)
            self.assertIsNone(parent.active_session("device-a"))
            self.assertIsNone(parent.active_session("device-b"))


if __name__ == "__main__":
    unittest.main()
