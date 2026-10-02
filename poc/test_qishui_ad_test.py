from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from features.qishui_ad_test.monitor import QishuiAdTestMonitor
from features.qishui_ad_test.profile import QishuiAdTestProfile, build_qishui_ad_test_goal


class _Gateway:
    def __init__(self, sessions):
        self.sessions = list(sessions)
        self.confirm_calls = 0

    def start(self, *, goal: str, device_id: str, run_dir: Path):
        return {"session": self.sessions[0]}

    def get(self, session_id: str):
        return {"session": self.sessions[-1]}

    def observe(self, session_id: str, *, device_id: str):
        return {"session": self.sessions.pop(0) if len(self.sessions) > 1 else self.sessions[0]}

    def confirm(self, session_id: str, *, device_id: str, confirmation):
        self.confirm_calls += 1
        return {"session": {"session_id": session_id, "status": "needs_reobservation", "physical_actions": 1}}

    def cancel(self, session_id: str, *, device_id: str):
        return {"session": {"session_id": session_id, "status": "cancelled", "physical_actions": 0}}


def _session(status="awaiting_confirmation", action="back", **extra):
    payload = {
        "session_id": "session-1",
        "status": status,
        "physical_actions": 0,
        "proposal": {"status": "action", "action": {"action": action}},
        "qwen_decision": {"reason": "广告倒计时完成"},
        "evidence": ["frame-1.jpg"],
    }
    payload.update(extra)
    return payload


class QishuiAdTestTests(unittest.TestCase):
    def test_goal_is_single_manual_test_and_never_claims(self):
        profile = QishuiAdTestProfile(device_id="phone-1")
        goal = build_qishui_ad_test_goal(profile)
        self.assertIn("最多测试一个广告", goal)
        self.assertIn("不得点击广告内的安装、下载、跳转、领取", goal)

    def test_start_does_not_auto_advance(self):
        with TemporaryDirectory() as tmp:
            gateway = _Gateway([_session(action="wait_for_change")])
            monitor = QishuiAdTestMonitor(gateway=gateway, output_root=Path(tmp))
            record = monitor.start(
                profile=QishuiAdTestProfile(device_id="phone-1"),
                goal="manual test",
            )
            self.assertEqual("observing", record.status)
            self.assertEqual(0, record.physical_actions)
            self.assertEqual(0, gateway.confirm_calls)

    def test_reward_page_waits_for_explicit_end_confirmation(self):
        with TemporaryDirectory() as tmp:
            gateway = _Gateway([_session(status="succeeded", action=None)])
            monitor = QishuiAdTestMonitor(gateway=gateway, output_root=Path(tmp))
            record = monitor.start(profile=QishuiAdTestProfile(device_id="phone-1"), goal="manual")
            updated = monitor.observe(record.test_id, device_id="phone-1")
            self.assertEqual("awaiting_confirmation", updated.status)
            self.assertEqual("end", updated.confirmation_mode)
            completed = monitor.confirm(record.test_id, device_id="phone-1", confirmed=True,
                mode="end", confirmation=None)
            self.assertEqual("completed", completed.status)
            self.assertEqual(0, completed.physical_actions)

    def test_only_confirmed_back_can_execute_one_action(self):
        with TemporaryDirectory() as tmp:
            gateway = _Gateway([_session()])
            monitor = QishuiAdTestMonitor(gateway=gateway, output_root=Path(tmp))
            record = monitor.start(profile=QishuiAdTestProfile(device_id="phone-1"), goal="manual")
            updated = monitor.observe(record.test_id, device_id="phone-1")
            self.assertEqual("back", updated.confirmation_mode)
            with self.assertRaises(ValueError):
                monitor.confirm(record.test_id, device_id="phone-1", confirmed=True,
                    mode="back", confirmation=None)
            completed = monitor.confirm(record.test_id, device_id="phone-1", confirmed=True,
                mode="back", confirmation={"scope": "test"})
            self.assertEqual("completed", completed.status)
            self.assertEqual(1, completed.physical_actions)
            self.assertEqual(1, gateway.confirm_calls)

    def test_non_back_action_is_never_confirmed(self):
        with TemporaryDirectory() as tmp:
            gateway = _Gateway([_session(action="tap_semantic")])
            monitor = QishuiAdTestMonitor(gateway=gateway, output_root=Path(tmp))
            record = monitor.start(profile=QishuiAdTestProfile(device_id="phone-1"), goal="manual")
            updated = monitor.observe(record.test_id, device_id="phone-1")
            self.assertEqual("observing", updated.status)
            self.assertEqual(0, gateway.confirm_calls)


if __name__ == "__main__":
    unittest.main()
