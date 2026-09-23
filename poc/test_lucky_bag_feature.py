from __future__ import annotations

import base64
import json
from pathlib import Path
import tempfile
import unittest

from features.lucky_bag import LuckyBagProfile, build_lucky_bag_goal
from features.lucky_bag.gmail import DurableNotificationRouter, GmailApiNotificationSink
from features.notifications import JsonlNotificationOutbox, NotificationEvent


class _TokenResponse:
    def read(self):
        return b'{"access_token":"refreshed"}'


class _Response:
    def read(self):
        return b'{"id":"message-1"}'


class LuckyBagFeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = LuckyBagProfile(
            device_id="device-local-01",
            recipient="q2904047615@gmail.com",
        )

    def test_profile_builds_goal_without_coordinates_or_fixed_steps(self) -> None:
        goal = build_lucky_bag_goal(self.profile)
        self.assertIn("预填评论", goal)
        self.assertIn("不重复发送评论", goal)
        self.assertIn("没有明确显示没抽中", goal)
        self.assertIn("当前 Android 实时画面", goal)
        self.assertNotIn("点击(", goal)
        self.assertNotIn("x=", goal)
        self.assertNotIn("y=", goal)

    def test_profile_rejects_invalid_runtime_values(self) -> None:
        with self.assertRaises(ValueError):
            LuckyBagProfile(device_id="", recipient="a@example.com")
        with self.assertRaises(ValueError):
            LuckyBagProfile(device_id="d", recipient="a@example.com", duration_seconds=0)

    def test_notification_outbox_is_provider_neutral_and_append_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "notifications.jsonl"
            outbox = JsonlNotificationOutbox(path)
            event = NotificationEvent(
                event_id="event-1",
                recipient=self.profile.recipient,
                subject="疑似中奖",
                body="疑似中奖",
                observed_at="2026-09-23T06:00:00+08:00",
                screenshot_path="evidence/win.jpg",
                reason="开奖结果未显示明确没抽中",
            )
            outbox.publish(event)
            outbox.publish(event)
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(2, len(rows))
            self.assertEqual("event-1", rows[0]["event_id"])
            self.assertEqual("evidence/win.jpg", rows[0]["screenshot_path"])

    def test_gmail_sink_refreshes_before_send(self) -> None:
        calls = []

        def opener(request, timeout):
            calls.append(request.full_url)
            return _TokenResponse() if "oauth2.googleapis.com" in request.full_url else _Response()

        sink = GmailApiNotificationSink(
            sender="q2904047615@gmail.com",
            refresh_token="refresh",
            client_id="client",
            client_secret="secret",
            opener=opener,
        )
        sink.publish(NotificationEvent(
            event_id="event-refresh",
            recipient=self.profile.recipient,
            subject="疑似中奖",
            body="疑似中奖",
            observed_at="2026-09-23T06:00:00+08:00",
        ))
        self.assertEqual(calls, [
            "https://oauth2.googleapis.com/token",
            "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
        ])
    def test_gmail_sink_builds_api_request_and_router_persists_first(self) -> None:
        captured = {}

        def opener(request, timeout):
            captured["request"] = request
            captured["timeout"] = timeout
            return _Response()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "notifications.jsonl"
            sink = GmailApiNotificationSink(
                access_token="token",
                sender="q2904047615@gmail.com",
                opener=opener,
            )
            router = DurableNotificationRouter(
                outbox=JsonlNotificationOutbox(path),
                remote=sink,
            )
            event = NotificationEvent(
                event_id="event-2",
                recipient=self.profile.recipient,
                subject="疑似中奖",
                body="疑似中奖",
                observed_at="2026-09-23T06:00:00+08:00",
            )
            router.publish(event)
            payload = json.loads(captured["request"].data.decode("utf-8"))
            raw = base64.urlsafe_b64decode(payload["raw"] + "===")
            self.assertIn(b"Subject: =?utf-8?", raw)
            self.assertEqual("Bearer token", captured["request"].get_header("Authorization"))
            self.assertEqual(1, len(path.read_text(encoding="utf-8").splitlines()))


if __name__ == "__main__":
    unittest.main()
