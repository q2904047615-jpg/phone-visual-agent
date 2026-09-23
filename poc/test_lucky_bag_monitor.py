from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import time
import unittest

from features.lucky_bag.monitor import LuckyBagMonitor
from features.lucky_bag.profile import LuckyBagProfile, build_lucky_bag_goal


class _Gateway:
    def __init__(self, *, block_start: bool = False) -> None:
        self.calls: list[tuple[str, int, int]] = []
        self.auto_count = 0
        self.start_entered = threading.Event()
        self.release_start = threading.Event()
        self.block_start = block_start

    def start(self, *, goal: str, device_id: str, run_dir: Path):
        self.start_entered.set()
        if self.block_start:
            self.release_start.wait(timeout=2)
        return {"session": {"session_id": "session-1", "status": "running", "physical_actions": 0,
                            "execution_budget": {"observation_attempts": 0}}}

    def get(self, session_id: str):
        return {"session": {"session_id": session_id, "status": "running", "physical_actions": self.auto_count,
                            "execution_budget": {"observation_attempts": self.auto_count}}}

    def auto(self, session_id: str, *, max_physical_actions: int, max_observations: int):
        self.calls.append((session_id, max_physical_actions, max_observations))
        self.auto_count += 1
        status = "succeeded" if self.auto_count >= 2 else "budget_paused"
        return {"session": {"session_id": session_id, "status": status,
                            "physical_actions": self.auto_count,
                            "execution_budget": {"observation_attempts": self.auto_count},
                            "qwen_decision": {"reason": "未看到明确的没抽中结果"}}}

    def pause(self, session_id: str):
        return {"session": {"session_id": session_id, "status": "paused"}}

    def cancel(self, session_id: str):
        return {"session": {"session_id": session_id, "status": "cancelled"}}


class LuckyBagMonitorTests(unittest.TestCase):
    def test_resumes_cumulative_chunks_and_writes_one_notification(self) -> None:
        with TemporaryDirectory() as tmp:
            gateway = _Gateway()
            monitor = LuckyBagMonitor(
                gateway=gateway, output_root=Path(tmp), chunk_actions=2,
                chunk_observations=3,
            )
            profile = LuckyBagProfile(device_id="device-local-01", recipient="q2904047615@gmail.com")
            record = monitor.start(profile=profile, goal=build_lucky_bag_goal(profile))
            deadline = time.monotonic() + 2
            while record.status not in {"succeeded", "failed"} and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(record.status, "succeeded")
            self.assertEqual(gateway.calls, [("session-1", 2, 3), ("session-1", 3, 4)])
            outbox = Path(tmp) / "lucky_bag_notifications.jsonl"
            rows = [json.loads(line) for line in outbox.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["event_id"], record.monitor_id)
            self.assertEqual(rows[0]["subject"], "疑似中奖")
            monitor.shutdown()

    def test_restored_active_monitor_requires_explicit_resume(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            monitor_id = "persisted-monitor"
            now = time.time()
            state = [{
                "monitor_id": monitor_id,
                "feature_id": "lucky_bag",
                "status": "running",
                "session_id": "old-session",
                "device_id": "device-local-01",
                "recipient": "q2904047615@gmail.com",
                "duration_seconds": 60,
                "started_at": "2026-09-23T06:00:00+08:00",
                "updated_at": "2026-09-23T06:00:01+08:00",
                "deadline_epoch": now + 60,
                "detail": "运行中",
                "notified": False,
                "run_dir": str(root / "lucky_bag_persisted"),
                "goal": "观察当前直播间福袋",
                "profile": {
                    "device_id": "device-local-01",
                    "recipient": "q2904047615@gmail.com",
                    "duration_seconds": 60,
                    "app_alias": "抖音",
                    "subject": "疑似中奖",
                    "body": "疑似中奖",
                },
            }]
            (root / "lucky_bag_monitors.json").write_text(
                json.dumps(state, ensure_ascii=False), encoding="utf-8"
            )
            gateway = _Gateway()
            monitor = LuckyBagMonitor(gateway=gateway, output_root=root)
            restored = monitor.get(monitor_id)
            self.assertIsNotNone(restored)
            self.assertEqual("recovery_required", restored.status)
            self.assertEqual("", restored.session_id)
            resumed = monitor.resume(monitor_id)
            deadline = time.monotonic() + 2
            while resumed.status not in {"succeeded", "failed"} and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual("succeeded", resumed.status)
            self.assertEqual([("session-1", 20, 40), ("session-1", 21, 41)], gateway.calls)
            monitor.shutdown()

    def test_pause_prevents_worker_from_advancing(self) -> None:
        with TemporaryDirectory() as tmp:
            gateway = _Gateway(block_start=True)
            monitor = LuckyBagMonitor(gateway=gateway, output_root=Path(tmp))
            profile = LuckyBagProfile(device_id="device-local-01", recipient="a@example.com")
            record = monitor.start(profile=profile, goal="等待当前直播间的福袋")
            self.assertTrue(gateway.start_entered.wait(timeout=1))
            self.assertEqual(monitor.pause(record.monitor_id).status, "paused")
            gateway.release_start.set()
            time.sleep(0.05)
            self.assertEqual(record.status, "paused")
            self.assertEqual(gateway.calls, [])
            monitor.shutdown()


if __name__ == "__main__":
    unittest.main()
