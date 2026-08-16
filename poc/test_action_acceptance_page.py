from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from touch_calibration_server import (
    ACTION_PAGE_PATH,
    PAGE_PATH,
    ActionEventStore,
    PageStateStore,
    root_action_location,
)


class ActionAcceptancePageTests(unittest.TestCase):
    def test_touch_calibration_page_requires_fullscreen_edge_grid(self) -> None:
        page = PAGE_PATH.read_text(encoding="utf-8")
        self.assertIn("requestFullscreen", page)
        self.assertIn("webkitRequestFullscreen", page)
        self.assertIn("webkitRequestFullScreen", page)
        self.assertIn("webkitCurrentFullScreenElement", page)
        self.assertIn("request.call(root)", page)
        self.assertNotIn("navigationUI: 'hide'", page)
        self.assertIn("viewport_coverage", page)
        self.assertIn("ratio >= 0.92 && ratio <= 1.08", page)
        self.assertIn("fullscreen_lost_during_calibration", page)
        self.assertIn("[.05,.05]", page)
        self.assertIn("[.95,.95]", page)
        self.assertIn("/api/page-state", page)

    def test_page_state_store_rejects_unknown_phase(self) -> None:
        store = PageStateStore()
        with self.assertRaisesRegex(ValueError, "不支持"):
            store.update(
                {
                    "phase": "unknown",
                    "fullscreen": False,
                    "calibration_mode": "setup",
                    "viewport_coverage": {},
                    "viewport_width": 393,
                    "viewport_height": 685,
                    "sequence": 0,
                }
            )

    def test_page_state_store_records_fullscreen_calibration(self) -> None:
        store = PageStateStore()
        state = store.update(
            {
                "phase": "calibration",
                "fullscreen": True,
                "calibration_mode": "fullscreen",
                "fullscreen_attempted": True,
                "viewport_coverage": {},
                "viewport_width": 810,
                "viewport_height": 1440,
                "sequence": 2,
            }
        )
        self.assertTrue(state["fullscreen"])
        self.assertEqual("calibration", store.snapshot()["phase"])

    def test_page_state_store_accepts_only_measured_viewport_fallback(self) -> None:
        store = PageStateStore()
        state = store.update(
            {
                "phase": "calibration",
                "fullscreen": False,
                "calibration_mode": "viewport_coverage",
                "fullscreen_attempted": True,
                "viewport_coverage": {
                    "eligible": True,
                    "width_ratio": 0.95,
                    "height_ratio": 0.93,
                },
                "viewport_width": 393,
                "viewport_height": 806,
                "sequence": 0,
            }
        )
        self.assertEqual("viewport_coverage", state["calibration_mode"])
        with self.assertRaisesRegex(ValueError, "缺少可信"):
            store.update(
                {
                    "phase": "calibration",
                    "fullscreen": False,
                    "calibration_mode": "viewport_coverage",
                    "fullscreen_attempted": True,
                    "viewport_coverage": {
                        "eligible": True,
                        "width_ratio": 0.95,
                        "height_ratio": 0.90,
                    },
                    "viewport_width": 393,
                    "viewport_height": 780,
                    "sequence": 0,
                }
            )

    def test_page_state_store_rejects_string_viewport_claims(self) -> None:
        store = PageStateStore()
        with self.assertRaisesRegex(ValueError, "缺少可信"):
            store.update(
                {
                    "phase": "calibration",
                    "fullscreen": "false",
                    "calibration_mode": "viewport_coverage",
                    "fullscreen_attempted": "true",
                    "viewport_coverage": {
                        "eligible": "true",
                        "width_ratio": 0.95,
                        "height_ratio": 0.95,
                    },
                    "viewport_width": 393,
                    "viewport_height": 806,
                    "sequence": 0,
                }
            )
        with self.assertRaisesRegex(ValueError, "必须是数值"):
            store.update(
                {
                    "phase": "calibration",
                    "fullscreen": False,
                    "calibration_mode": "viewport_coverage",
                    "fullscreen_attempted": True,
                    "viewport_coverage": {
                        "eligible": True,
                        "width_ratio": "0.95",
                        "height_ratio": 0.95,
                    },
                    "viewport_width": 393,
                    "viewport_height": 806,
                    "sequence": 0,
                }
            )

    def test_page_exposes_only_generic_safe_action_modes(self) -> None:
        page = ACTION_PAGE_PATH.read_text(encoding="utf-8")
        for marker in (
            "mode=swipe",
            "mode=tap",
            "mode=back",
            "mode=input",
            "mode=long_press",
            "mode=drag",
            "mode=replan",
            "mode=sequence",
            "异常重规划：页面自行切换只读入口",
            "打开初始只读入口",
            "打开替代只读入口",
            "setTimeout(showReplanReplacement, 120000)",
            "连续闭环：滑动→点击→返回",
            "tap_semantic",
            "input_verified_text",
            "系统返回",
            "长按目标",
            "拖动目标",
            "目标文字：agent",
            "input.value === 'agent'",
            'href="/actions">返回验收模式选择',
        ):
            self.assertIn(marker, page)
        for app_name in ("微信", "抖音", "支付宝"):
            self.assertNotIn(app_name, page)
        self.assertNotIn("placeholder=", page)

    def test_root_can_redirect_only_to_explicit_generic_action_modes(self) -> None:
        self.assertIsNone(root_action_location(None))
        self.assertEqual("/actions?mode=input", root_action_location("input"))
        self.assertEqual(
            "/actions?mode=long_press",
            root_action_location("long_press"),
        )
        self.assertEqual("/actions?mode=drag", root_action_location("drag"))
        with self.assertRaisesRegex(ValueError, "不支持"):
            root_action_location("account_action")

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
