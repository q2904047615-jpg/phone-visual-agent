from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import web_app
from test_support.web_platform import _BaseApiEndToEndTests


class _FakeMonitor:
    def __init__(self) -> None:
        self.record = None

    def start(self, *, profile, goal):
        self.record = SimpleNamespace(
            profile=profile,
            goal=goal,
            snapshot=lambda: {
                "monitor_id": "monitor-1",
                "feature_id": "lucky_bag",
                "status": "starting",
                "session_id": "",
                "device_id": profile.device_id,
                "recipient": profile.recipient,
                "duration_seconds": profile.duration_seconds,
                "started_at": "now",
                "updated_at": "now",
                "detail": "",
                "notified": False,
                "run_dir": str(Path("evidence")),
            },
        )
        return self.record

    def get(self, monitor_id):
        return self.record if monitor_id == "monitor-1" else None

    def list(self):
        return [self.record] if self.record is not None else []


class LuckyBagWebFeatureTests(_BaseApiEndToEndTests):
    def test_lucky_bag_feature_returns_goal_preset(self) -> None:
        response = self.client.get(
            "/api/features/lucky-bag",
            headers=self.headers,
        )
        self.assertEqual(200, response.status_code, response.text)
        payload = response.json()
        self.assertEqual("lucky_bag", payload["feature_id"])
        self.assertEqual("dedicated_monitor", payload["stage"])
        self.assertIn("预填评论", payload["goal"])
        self.assertEqual("疑似中奖", payload["profile"]["subject"])

    def test_lucky_bag_page_uses_strict_start_payload(self) -> None:
        source = (Path(__file__).resolve().parent / "static" / "app.js").read_text(encoding="utf-8")
        self.assertIn("function luckyBagStartPayload(profile)", source)
        self.assertIn("JSON.stringify(luckyBagStartPayload(state.luckyBagProfile))", source)
        self.assertNotIn("JSON.stringify(state.luckyBagProfile || {})", source)

    def test_lucky_bag_page_exposes_live_qwen_feedback(self) -> None:
        source = (Path(__file__).resolve().parent / "static" / "app.js").read_text(encoding="utf-8")
        page = (Path(__file__).resolve().parent / "static" / "index.html").read_text(encoding="utf-8")
        self.assertIn("async function getLuckyBagSession(sessionId)", source)
        self.assertIn("/api/agent/generic-supervised/", source)
        self.assertIn("renderLuckyBagAgentFeedback", source)
        self.assertIn("cancelLuckyBagMonitor", source)
        self.assertIn("failed_reason", source)
        self.assertNotIn('id="luckyBagFeedback"', page)
        self.assertNotIn("Qwen 最近判断摘要", page)
        self.assertIn('id="actionContent"', page)
        self.assertIn('id="chatMessages"', page)
        self.assertIn('conversation-message', source)
        self.assertIn('renderConversation', source)
        self.assertIn('id="continueTaskButton"', page)
        self.assertIn("pagehide", source)
        self.assertIn("settlePageExitPause", source)

    def test_page_uses_chat_first_task_entry(self) -> None:
        source = (Path(__file__).resolve().parent / "static" / "app.js").read_text(encoding="utf-8")
        page = (Path(__file__).resolve().parent / "static" / "index.html").read_text(encoding="utf-8")
        self.assertIn("async function startLuckyBagModule()", source)
        self.assertIn("startLuckyBagModule);", source)
        self.assertNotIn('state.chatUserMessage = String(feature.goal || "");', source)
        self.assertNotIn('state.chatUserMessage = String(response.goal || state.chatUserMessage || "");', source)
        self.assertIn('class="agent-workspace"', page)
        self.assertIn('class="chat-panel"', page)
        self.assertIn('id="agentText"', page)
        self.assertIn('发送', page)
        self.assertIn('福袋模块', page)

    def test_lucky_bag_feature_requires_local_control_token(self) -> None:
        response = self.client.get("/api/features/lucky-bag")
        self.assertEqual(403, response.status_code)

    def test_start_and_get_monitor_are_additive_routes(self) -> None:
        fake = _FakeMonitor()
        with patch.object(web_app, "_LUCKY_BAG_MONITOR", fake):
            response = self.client.post(
                "/api/features/lucky-bag/start",
                headers=self.headers,
                json={
                    "device_id": "device-local-01",
                    "recipient": "a@example.com",
                    "duration_seconds": 120,
                },
            )
            self.assertEqual(200, response.status_code, response.text)
            payload = response.json()
            self.assertEqual("monitoring", payload["stage"])
            self.assertEqual("monitor-1", payload["monitor"]["monitor_id"])
            self.assertEqual(120, payload["profile"]["duration_seconds"])

            status = self.client.get(
                "/api/features/lucky-bag/monitor-1",
                headers=self.headers,
            )
            self.assertEqual(200, status.status_code, status.text)
            self.assertEqual("monitor-1", status.json()["monitor"]["monitor_id"])

            listing = self.client.get(
                "/api/features/lucky-bag/monitors",
                headers=self.headers,
            )
            self.assertEqual(200, listing.status_code, listing.text)
            self.assertEqual("monitor-1", listing.json()["monitors"][0]["monitor"]["monitor_id"])


if __name__ == "__main__":
    unittest.main()
