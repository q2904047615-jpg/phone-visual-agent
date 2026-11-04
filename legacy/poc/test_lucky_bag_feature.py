from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from PIL import Image

from features.lucky_bag import COMMENT_POINT, SEND_POINT, LUCKY_BAG_SLOT_POINTS, LuckyBagProfile, build_lucky_bag_goal
from features.lucky_bag.profile import COMMENT_SEND_DELAY_SECONDS, NO_LUCKY_BAG_REOBSERVE_SECONDS
from features.lucky_bag.gmail import (
    DurableNotificationRouter,
    GmailApiNotificationSink,
    gmail_configuration_status,
)
from features.notifications import JsonlNotificationOutbox, NotificationEvent
from features.lucky_bag.fast_runner import FastLuckyBagSessionGateway


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

    def test_profile_builds_goal_with_approved_fixed_form_steps(self) -> None:
        goal = build_lucky_bag_goal(self.profile)
        self.assertIn("预填评论", goal)
        self.assertIn("不重复发送评论", goal)
        self.assertIn("开奖结果", goal)
        self.assertIn("wait_seconds 填为 300", goal)
        self.assertIn("图1是用户提供的福袋入口近照", goal)
        self.assertIn("五张用户参考图", goal)
        self.assertIn("图2是直播间中的福袋入口", goal)
        self.assertIn("图3是发送后短暂出现的“成功参与福袋”提示", goal)
        self.assertIn("图4是重新打开同一个福袋后显示“已参与”", goal)
        self.assertIn("图5是开奖后的“没抽中福袋”和“知道了”", goal)
        self.assertIn("红色或粉红色布袋/礼包图标", goal)
        self.assertIn("黄色、金色绳结", goal)
        self.assertIn("黑色圆角倒计时标签", goal)
        self.assertIn("07:07", goal)
        self.assertNotIn("图2是打开后的“福袋”详情页", goal)
        self.assertNotIn("图3是评论框中已经自动填好的评论", goal)
        self.assertIn("先点击该入口打开", goal)
        self.assertIn("普通礼物、红包、游戏礼包和推荐卡没有这种红粉色袋身加金黄色绳结的组合时才排除", goal)
        self.assertIn("点击去发表评论后固定等待 3 秒", goal)
        self.assertIn("开奖后只调用一次Qwen判断中奖或未中奖", goal)
        self.assertEqual(300, NO_LUCKY_BAG_REOBSERVE_SECONDS)
        self.assertEqual(3, COMMENT_SEND_DELAY_SECONDS)
        self.assertIn("可参与福袋可见时等待", goal)
        self.assertIn("当前 Android 实时画面", goal)
        self.assertIn("[0.498, 0.868]", goal)
        self.assertIn("[0.832, 0.476]", goal)
        self.assertEqual((498, 868), COMMENT_POINT)
        self.assertEqual((832, 476), SEND_POINT)

    def test_profile_exposes_five_project_owned_visual_references(self) -> None:
        paths = self.profile.visual_reference_paths
        self.assertEqual(5, len(paths))
        self.assertTrue(all(path.is_file() for path in paths))
        self.assertEqual("00_lucky_bag_closeup.jpg", paths[0].name)
        self.assertEqual("06_not_selected.jpg", paths[-1].name)
        self.assertNotIn("02_lucky_bag_detail.jpg", {path.name for path in paths})
        self.assertNotIn("03_prefilled_comment.jpg", {path.name for path in paths})

    def test_profile_marks_lucky_bag_goal_as_active_ordered_execution(self) -> None:
        goal = build_lucky_bag_goal(self.profile)
        self.assertIn("【ACTIVE_EXECUTION_POLICY:lucky_bag】", goal)
        self.assertIn("当前画面出现与上述外观一致的福袋入口", goal)
        self.assertIn("禁止选择wait_for_change", goal)

    def test_gmail_configuration_status_is_secret_free_when_unconfigured(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            status = gmail_configuration_status()
        self.assertEqual(
            {
                "configured": False,
                "mode": "local_outbox_only",
                "message": "Gmail 未配置，中奖通知只能写入本地队列。",
            },
            status,
        )

    def test_profile_rejects_invalid_runtime_values(self) -> None:
        with self.assertRaises(ValueError):
            LuckyBagProfile(device_id="", recipient="a@example.com")
        with self.assertRaises(ValueError):
            LuckyBagProfile(device_id="d", recipient="a@example.com", duration_seconds=0)

    def test_fast_runner_uses_find_verify_result_observations_and_fixed_form_points(self) -> None:
        class Controller:
            device_id = "device-local-01"

            def __init__(self):
                self.points = []

            def begin_new_task(self):
                pass

            def vision_capture(self):
                return Image.new("RGB", (540, 960), "black")

            def arm_physical_execution(self, *args, **kwargs):
                pass

            def vision_tap_relative(self, x, y):
                self.points.append((x, y))

            def consume_last_click_receipt(self):
                return {"input_events_dispatched": True}

            def clear_physical_execution_authorization(self):
                pass

        class Observer:
            def __init__(self):
                self.calls = 0
                self.last_raw_response = ""
                self.detail_counts = []
                self.detail_sizes = []

            def observe_with_decision(self, **kwargs):
                self.calls += 1
                details = tuple(kwargs.get("current_detail_images", ()))
                self.detail_counts.append(len(details))
                self.detail_sizes.append(tuple(image.size for image in details))
                if self.calls == 1:
                    self.last_raw_response = '{"reply":"发现福袋","decision":{}}'
                    return object(), {"status": "action", "action": "tap", "point": [130, 118]}
                if self.calls == 2:
                    self.last_raw_response = '{"reply":"已确认参与","decision":{}}'
                    return object(), {"status": "action", "action": "wait_for_change", "wait_seconds": 300,
                                      "postcondition": {"status": "confirmed", "fact": "已参与，剩余5分钟"}}
                self.last_raw_response = '{"reply":"没中奖","decision":{}}'
                return object(), {"status": "action", "action": "tap", "point": [420, 730],
                                  "postcondition": {"status": "not_confirmed", "fact": "没抽中福袋"}}

        controller = Controller()
        observer = Observer()
        runner = FastLuckyBagSessionGateway(
            controller_for_device=lambda _device: controller,
            capture_frame=lambda _device: controller.vision_capture(),
            observer=observer,
            ensure_device_ready=lambda _device: None,
        )
        profile = self.profile
        with patch("features.lucky_bag.fast_runner.save_task_frames", return_value=("a.jpg",) * 4), \
             patch("features.lucky_bag.fast_runner.local_frame_fingerprint", return_value="frame"), \
             patch("features.lucky_bag.fast_runner._mint_single_step_scene_credential", return_value=object()):
            started = runner.start(goal="福袋", device_id=profile.device_id, run_dir=Path("poc"))
            session_id = started["session"]["session_id"]
            runner._sessions[session_id].profile = profile
            runner.auto(session_id, max_physical_actions=1, max_observations=2)
            self.assertEqual(1, observer.calls)
            self.assertEqual([1], observer.detail_counts)
            self.assertEqual([((729, 864),)], observer.detail_sizes)
            runner.auto(session_id, max_physical_actions=2, max_observations=2)
            self.assertEqual(1, observer.calls)
            self.assertGreaterEqual(
                runner._sessions[session_id].next_observation_at - time.monotonic(), 2.5
            )
            # The production path waits three seconds here; make the unit test
            # advance the clock without sleeping.
            runner._sessions[session_id].next_observation_at = 0
            result = runner.auto(session_id, max_physical_actions=3, max_observations=4)
            self.assertEqual(2, observer.calls)
            runner._sessions[session_id].next_observation_at = 0
            result = runner.auto(session_id, max_physical_actions=4, max_observations=6)
        self.assertEqual("running", result["session"]["status"])
        self.assertEqual(3, observer.calls)
        self.assertEqual([1, 0, 0], observer.detail_counts)
        self.assertEqual([((729, 864),), (), ()], observer.detail_sizes)
        self.assertEqual([(130, 118), COMMENT_POINT, SEND_POINT, (420, 730)], controller.points)
        self.assertEqual("find_bag", result["session"]["current_phase"])
        self.assertIn("未中奖", result["session"]["detail"])

    def test_fast_runner_win_result_keeps_monitor_running_without_dismiss_click(self) -> None:
        class Controller:
            device_id = "device-local-01"
            def begin_new_task(self): pass
            def vision_capture(self): return Image.new("RGB", (540, 960), "black")
            def arm_physical_execution(self, *args, **kwargs): pass
            def __init__(self): self.points = []
            def vision_tap_relative(self, *point): self.points.append(point)
            def consume_last_click_receipt(self): return {"input_events_dispatched": True}
            def clear_physical_execution_authorization(self): pass

        class Observer:
            def __init__(self): self.calls = 0; self.last_raw_response = ""
            def observe_with_decision(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    self.last_raw_response = '{"reply":"发现福袋","decision":{}}'
                    return object(), {"status": "action", "action": "tap", "point": [130, 118]}
                if self.calls == 2:
                    self.last_raw_response = '{"reply":"已确认参与","decision":{}}'
                    return object(), {"status": "action", "action": "wait_for_change", "wait_seconds": 1,
                                      "postcondition": {"status": "confirmed", "fact": "已参与"}}
                self.last_raw_response = '{"reply":"恭喜你中奖了","decision":{}}'
                return object(), {"status": "finish", "action": None, "point": None,
                                  "postcondition": {"status": "confirmed", "fact": "已中奖"}}

        controller, observer = Controller(), Observer()
        runner = FastLuckyBagSessionGateway(
            controller_for_device=lambda _device: controller,
            capture_frame=lambda _device: controller.vision_capture(),
            observer=observer,
            ensure_device_ready=lambda _device: None,
        )
        with patch("features.lucky_bag.fast_runner.save_task_frames", return_value=("a.jpg",) * 4), \
             patch("features.lucky_bag.fast_runner.local_frame_fingerprint", return_value="frame"), \
             patch("features.lucky_bag.fast_runner._mint_single_step_scene_credential", return_value=object()):
            started = runner.start(goal="福袋", device_id=self.profile.device_id, run_dir=Path("poc"))
            session_id = started["session"]["session_id"]
            runner._sessions[session_id].profile = self.profile
            runner.auto(session_id, max_physical_actions=1, max_observations=2)
            runner.auto(session_id, max_physical_actions=2, max_observations=2)
            runner._sessions[session_id].next_observation_at = 0
            runner.auto(session_id, max_physical_actions=3, max_observations=4)
            runner._sessions[session_id].next_observation_at = 0
            result = runner.auto(session_id, max_physical_actions=4, max_observations=6)
        self.assertEqual("running", result["session"]["status"])
        self.assertEqual(3, observer.calls)
        self.assertEqual([(130, 118), COMMENT_POINT, SEND_POINT], controller.points)
        self.assertIn("已中奖", result["session"]["detail"])
        self.assertTrue(result["session"]["win_detected"])

    def test_lucky_bag_entry_is_selected_from_three_raw_slots(self) -> None:
        class Controller:
            def vision_capture(self):
                return Image.new("RGB", (540, 960), "black")

        slot = FastLuckyBagSessionGateway._match_lucky_bag_slot([219, 118], Controller())
        self.assertEqual(2, slot)
        self.assertEqual((118, 113), LUCKY_BAG_SLOT_POINTS[slot])

        class CurrentFrameController:
            def vision_capture(self):
                return Image.new("RGB", (810, 1440), "black")

        self.assertEqual(
            2,
            FastLuckyBagSessionGateway._match_lucky_bag_slot([146, 79], CurrentFrameController()),
        )

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
