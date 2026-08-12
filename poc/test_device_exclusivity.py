from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from device_exclusivity import InterProcessLease


class InterProcessLeaseTests(unittest.TestCase):
    def test_real_second_process_cannot_acquire_same_lease(self) -> None:
        child_code = r'''
import sys
from pathlib import Path
from device_exclusivity import InterProcessLease
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


if __name__ == "__main__":
    unittest.main()
