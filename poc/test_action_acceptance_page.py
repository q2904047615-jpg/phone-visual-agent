from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from touch_calibration_server import ACTION_PAGE_PATH, ActionEventStore


class ActionAcceptancePageTests(unittest.TestCase):
    def test_page_exposes_only_generic_safe_action_modes(self) -> None:
        page = ACTION_PAGE_PATH.read_text(encoding="utf-8")
        for marker in (
            "mode=swipe",
            "mode=tap",
            "mode=back",
            "mode=input",
            "mode=long_press",
            "mode=drag",
            "mode=sequence",
            "连续闭环：滑动→点击→返回",
            "tap_semantic",
            "input_verified_text",
            "系统返回",
            "长按目标",
            "拖动目标",
            "目标文字：agent",
            "input.value === 'agent'",
        ):
            self.assertIn(marker, page)
        for app_name in ("微信", "抖音", "支付宝"):
            self.assertNotIn(app_name, page)

    def test_action_event_store_records_one_supported_event(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ActionEventStore(Path(directory))
            event = store.append(
                {
                    "kind": "drag",
                    "status": "passed",
                    "client_event_id": "event-1",
                    "details": {"source_in_destination": True},
                }
            )
            saved = json.loads(store.path.read_text(encoding="utf-8"))

        self.assertEqual(0, event["sequence"])
        self.assertEqual("drag", saved["kind"])
        self.assertEqual("passed", saved["status"])

    def test_action_event_store_records_generic_three_step_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ActionEventStore(Path(directory))
            for index, kind in enumerate(("swipe", "tap_semantic", "back"), 1):
                store.append(
                    {
                        "kind": kind,
                        "status": "passed",
                        "client_event_id": f"sequence-{index}",
                        "details": {"sequence_step": index},
                    }
                )
            snapshot = store.snapshot()

        self.assertEqual([0, 1, 2], [item["sequence"] for item in snapshot["events"]])
        self.assertEqual(
            ["swipe", "tap_semantic", "back"],
            [item["kind"] for item in snapshot["events"]],
        )

    def test_action_event_store_rejects_unknown_or_duplicate_events(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ActionEventStore(Path(directory))
            base = {
                "kind": "long_press",
                "status": "passed",
                "client_event_id": "event-1",
                "details": {},
            }
            store.append(base)
            with self.assertRaisesRegex(ValueError, "重复"):
                store.append(base)
            with self.assertRaisesRegex(ValueError, "不支持"):
                store.append({**base, "kind": "send", "client_event_id": "event-2"})


if __name__ == "__main__":
    unittest.main()
