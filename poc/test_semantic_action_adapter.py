import tempfile
import unittest
from pathlib import Path

from PIL import Image

from live_semantic_dry_run import RichObservationCapture
from semantic_action_adapter import (
    EnsureAppActionAdapter,
    ObserveActionAdapter,
    PhysicalActionVerificationError,
    SemanticActionRouter,
    SemanticActionAdapterError,
    SwipeUpActionAdapter,
    TapHeartActionAdapter,
)
from semantic_executor import SemanticAction
from state_controller import PageObservation
from target_locator import TargetResolution
from task_orchestrator import GoalSpec


def capture(observation: PageObservation) -> RichObservationCapture:
    frames = tuple(Image.new("RGB", (540, 960), "black") for _ in range(4))
    return RichObservationCapture(observation=observation, frames=frames)


class FakeSensor:
    def __init__(self, *observations: PageObservation) -> None:
        self.captures = [capture(item) for item in observations]
        self.calls = []

    def observe_rich(self, goal, **kwargs):
        self.calls.append((goal, kwargs))
        if not self.captures:
            raise AssertionError("发生了计划外的额外观察。")
        return self.captures.pop(0)


class FakeResolver:
    def __init__(self, *verified_states: str | None) -> None:
        self.calls = []
        self.verified_states = list(verified_states)

    def resolve(self, **kwargs):
        self.calls.append(kwargs)
        proposed = kwargs["proposed"]
        verified_state = (
            self.verified_states.pop(0) if self.verified_states else None
        )
        return TargetResolution(
            target=kwargs["target"],
            method="test_guarded_target",
            proposed_coordinate=proposed,
            resolved_coordinate=proposed,
            pixel_center=(270, 480),
            samples=4,
            spread_px=0.0,
            detail="test",
            verified_state=verified_state,
        )


class RaisingSensor(FakeSensor):
    def observe_rich(self, goal, **kwargs):
        self.calls.append((goal, kwargs))
        if not self.captures:
            raise RuntimeError("模拟点赞后摄像头读取失败")
        return self.captures.pop(0)


class EnsureAppActionAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.goal = GoalSpec.from_operation("douyin.search", {"keyword": "机械臂"})
        self.action = SemanticAction(
            node_id="open_app",
            action="ensure_app",
            params={"app_id": "douyin"},
        )

    def test_unsupported_action_is_rejected_without_observation_or_tap(self) -> None:
        sensor = FakeSensor()
        taps = []
        adapter = EnsureAppActionAdapter(
            sensor,
            lambda x, y: taps.append((x, y)),
            post_tap_wait=0,
        )
        with self.assertRaisesRegex(SemanticActionAdapterError, "只允许 ensure_app"):
            adapter.execute(
                SemanticAction("like", "tap_semantic", {"target": "heart"}),
                self.goal,
            )
        self.assertEqual(sensor.calls, [])
        self.assertEqual(taps, [])

    def test_already_in_target_app_succeeds_with_zero_taps(self) -> None:
        sensor = FakeSensor(
            PageObservation(
                state="douyin_video",
                base_state="douyin_video",
                confidence=0.98,
                stable=True,
                reason="抖音视频页",
            )
        )
        taps = []
        adapter = EnsureAppActionAdapter(
            sensor,
            lambda x, y: taps.append((x, y)),
            post_tap_wait=0,
        )
        result = adapter.execute(self.action, self.goal)
        self.assertTrue(result.action_result.success)
        self.assertFalse(result.robot_action_called)
        self.assertTrue(result.action_result.details["no_op"])
        self.assertEqual(taps, [])
        self.assertEqual(len(sensor.calls), 1)

    def test_android_home_taps_once_and_requires_verified_target_app(self) -> None:
        sensor = FakeSensor(
            PageObservation(
                state="android_home",
                base_state="android_home",
                confidence=0.99,
                stable=True,
                reason="桌面",
                targets={"douyin_icon": (800, 700)},
                target_bounds={"douyin_icon": (740, 640, 860, 760)},
            ),
            PageObservation(
                state="douyin_home",
                base_state="douyin_home",
                confidence=0.96,
                stable=True,
                reason="抖音首页",
            ),
        )
        taps = []
        resolver = FakeResolver()
        adapter = EnsureAppActionAdapter(
            sensor,
            lambda x, y: taps.append((x, y)) or (x + 1, y + 2),
            target_resolver=resolver,
            post_tap_wait=0,
        )
        with tempfile.TemporaryDirectory() as folder:
            result = adapter.execute(
                self.action,
                self.goal,
                evidence_dir=Path(folder),
            )
            self.assertEqual(len(result.evidence), 8)
        self.assertTrue(result.action_result.success)
        self.assertTrue(result.robot_action_called)
        self.assertEqual(taps, [(800, 700)])
        self.assertEqual(result.command_point, (801, 702))
        self.assertEqual(len(sensor.calls), 2)
        self.assertEqual(len(resolver.calls), 1)

    def test_failed_post_observation_never_retries_tap(self) -> None:
        sensor = FakeSensor(
            PageObservation(
                state="android_home",
                base_state="android_home",
                confidence=0.99,
                stable=True,
                reason="桌面",
                targets={"douyin_icon": (800, 700)},
                target_bounds={"douyin_icon": (740, 640, 860, 760)},
            ),
            PageObservation(
                state="android_home",
                base_state="android_home",
                confidence=0.99,
                stable=True,
                reason="仍在桌面",
            ),
        )
        taps = []
        adapter = EnsureAppActionAdapter(
            sensor,
            lambda x, y: taps.append((x, y)) or (x, y),
            target_resolver=FakeResolver(),
            post_tap_wait=0,
        )
        result = adapter.execute(self.action, self.goal)
        self.assertFalse(result.action_result.success)
        self.assertEqual(taps, [(800, 700)])
        self.assertEqual(len(sensor.calls), 2)

    def test_low_confidence_stops_before_resolver_and_tap(self) -> None:
        sensor = FakeSensor(
            PageObservation(
                state="android_home",
                base_state="android_home",
                confidence=0.60,
                stable=True,
                reason="模糊",
                targets={"douyin_icon": (800, 700)},
            )
        )
        taps = []
        resolver = FakeResolver()
        adapter = EnsureAppActionAdapter(
            sensor,
            lambda x, y: taps.append((x, y)),
            target_resolver=resolver,
            post_tap_wait=0,
        )
        with self.assertRaisesRegex(SemanticActionAdapterError, "置信度"):
            adapter.execute(self.action, self.goal)
        self.assertEqual(taps, [])
        self.assertEqual(resolver.calls, [])

    def test_goal_and_action_app_must_match(self) -> None:
        sensor = FakeSensor()
        adapter = EnsureAppActionAdapter(
            sensor,
            lambda _x, _y: (_x, _y),
            post_tap_wait=0,
        )
        with self.assertRaisesRegex(SemanticActionAdapterError, "不一致"):
            adapter.execute(
                SemanticAction(
                    node_id="open_app",
                    action="ensure_app",
                    params={"app_id": "wechat"},
                ),
                self.goal,
            )
        self.assertEqual(sensor.calls, [])


class ObserveActionAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.goal = GoalSpec.from_operation(
            "douyin.batch_interact",
            {
                "keyword": None,
                "target_count": 3,
                "like": True,
                "comment": False,
            },
        )
        self.action = SemanticAction("observe_video", "observe", {})

    def test_ordinary_video_is_reported_with_zero_physical_actions(self) -> None:
        sensor = FakeSensor(
            PageObservation(
                state="douyin_video",
                base_state="douyin_video",
                confidence=0.99,
                stable=True,
                reason="普通视频稳定",
                heart_state="unliked",
            )
        )
        result = ObserveActionAdapter(sensor).execute(self.action, self.goal)
        self.assertEqual(result.classification, "ordinary_video")
        self.assertTrue(result.safe_for_next_action)
        self.assertFalse(result.robot_action_called)
        self.assertEqual(result.action_result.details["physical_actions"], 0)
        self.assertEqual(len(sensor.calls), 1)

    def test_live_page_is_classified_without_trying_to_close_it(self) -> None:
        sensor = FakeSensor(
            PageObservation(
                state="douyin_live",
                base_state="douyin_live",
                confidence=0.98,
                stable=True,
                reason="直播页稳定",
            )
        )
        result = ObserveActionAdapter(sensor).execute(self.action, self.goal)
        self.assertEqual(result.classification, "live")
        self.assertTrue(result.safe_for_next_action)
        forbidden = sensor.calls[0][1]["explicitly_forbidden"]
        self.assertIn("close_overlay", forbidden)
        self.assertIn("swipe", forbidden)

    def test_unknown_or_low_confidence_is_not_safe_for_next_action(self) -> None:
        sensor = FakeSensor(
            PageObservation(
                state="unknown",
                base_state="unknown",
                confidence=0.40,
                stable=False,
                reason="画面模糊",
            )
        )
        result = ObserveActionAdapter(sensor).execute(self.action, self.goal)
        self.assertEqual(result.classification, "unknown")
        self.assertFalse(result.safe_for_next_action)
        self.assertFalse(result.robot_action_called)

    def test_non_observe_action_is_rejected_before_camera_capture(self) -> None:
        sensor = FakeSensor()
        with self.assertRaisesRegex(SemanticActionAdapterError, "只允许 observe"):
            ObserveActionAdapter(sensor).execute(
                SemanticAction("like", "tap_semantic", {"target": "heart"}),
                self.goal,
            )
        self.assertEqual(sensor.calls, [])


class TapHeartActionAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.goal = GoalSpec.from_operation(
            "douyin.batch_interact",
            {
                "keyword": None,
                "target_count": 3,
                "like": True,
                "comment": False,
            },
        )
        self.action = SemanticAction(
            node_id="tap_heart",
            action="tap_semantic",
            params={"target": "heart"},
        )

    @staticmethod
    def observation(heart_state: str, *, confidence: float = 0.99) -> PageObservation:
        return PageObservation(
            state="douyin_video",
            base_state="douyin_video",
            confidence=confidence,
            stable=True,
            reason=f"普通视频，爱心={heart_state}",
            heart_state=heart_state,
            targets={"heart": (470, 450)},
            target_bounds={"heart": (440, 420, 500, 480)},
        )

    def test_one_white_heart_tap_requires_both_detectors_to_confirm_red(self) -> None:
        sensor = FakeSensor(self.observation("unliked"), self.observation("liked"))
        taps = []
        adapter = TapHeartActionAdapter(
            sensor,
            lambda x, y: taps.append((x, y)) or (x + 2, y - 1),
            target_resolver=FakeResolver("unliked", "liked"),
            post_tap_wait=0,
        )
        result = adapter.execute(self.action, self.goal)
        self.assertTrue(result.action_result.success)
        self.assertEqual(taps, [(470, 450)])
        self.assertEqual(result.command_point, (472, 449))
        self.assertEqual(result.action_result.details["physical_actions"], 1)
        self.assertEqual(result.action_result.details["retry_count"], 0)
        self.assertEqual(len(sensor.calls), 2)

    def test_qwen_already_liked_stops_before_local_resolver_and_tap(self) -> None:
        sensor = FakeSensor(self.observation("liked"))
        resolver = FakeResolver("liked")
        taps = []
        adapter = TapHeartActionAdapter(
            sensor,
            lambda x, y: taps.append((x, y)) or (x, y),
            target_resolver=resolver,
            post_tap_wait=0,
        )
        with self.assertRaisesRegex(SemanticActionAdapterError, "只允许点击白心"):
            adapter.execute(self.action, self.goal)
        self.assertEqual(taps, [])
        self.assertEqual(resolver.calls, [])

    def test_local_detector_disagreement_stops_before_tap(self) -> None:
        sensor = FakeSensor(self.observation("unliked"))
        taps = []
        adapter = TapHeartActionAdapter(
            sensor,
            lambda x, y: taps.append((x, y)) or (x, y),
            target_resolver=FakeResolver("liked"),
            post_tap_wait=0,
        )
        with self.assertRaisesRegex(SemanticActionAdapterError, "独立确认白心"):
            adapter.execute(self.action, self.goal)
        self.assertEqual(taps, [])
        self.assertEqual(len(sensor.calls), 1)

    def test_post_tap_unliked_result_fails_without_second_tap(self) -> None:
        sensor = FakeSensor(
            self.observation("unliked"),
            self.observation("unliked"),
        )
        taps = []
        adapter = TapHeartActionAdapter(
            sensor,
            lambda x, y: taps.append((x, y)) or (x, y),
            target_resolver=FakeResolver("unliked", "unliked"),
            post_tap_wait=0,
        )
        result = adapter.execute(self.action, self.goal)
        self.assertFalse(result.action_result.success)
        self.assertEqual(taps, [(470, 450)])
        self.assertEqual(result.action_result.details["retry_count"], 0)
        self.assertEqual(len(sensor.calls), 2)

    def test_post_tap_camera_failure_reports_one_action_and_never_retries(self) -> None:
        sensor = RaisingSensor(self.observation("unliked"))
        taps = []
        adapter = TapHeartActionAdapter(
            sensor,
            lambda x, y: taps.append((x, y)) or (x, y),
            target_resolver=FakeResolver("unliked"),
            post_tap_wait=0,
        )
        with self.assertRaises(PhysicalActionVerificationError) as caught:
            adapter.execute(self.action, self.goal)
        self.assertEqual(caught.exception.physical_actions, 1)
        self.assertIn("禁止补点", str(caught.exception))
        self.assertEqual(taps, [(470, 450)])
        self.assertEqual(len(sensor.calls), 2)

    def test_wrong_target_is_rejected_without_observation_or_tap(self) -> None:
        sensor = FakeSensor()
        taps = []
        adapter = TapHeartActionAdapter(
            sensor,
            lambda x, y: taps.append((x, y)) or (x, y),
            post_tap_wait=0,
        )
        with self.assertRaisesRegex(SemanticActionAdapterError, "只允许点击抖音爱心"):
            adapter.execute(
                SemanticAction("tap_comment", "tap_semantic", {"target": "comment"}),
                self.goal,
            )
        self.assertEqual(sensor.calls, [])
        self.assertEqual(taps, [])


class SwipeUpActionAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.goal = GoalSpec.from_operation(
            "douyin.batch_interact",
            {
                "keyword": None,
                "target_count": 3,
                "like": True,
                "comment": False,
            },
        )
        self.action = SemanticAction(
            node_id="skip_known_non_video_page",
            action="swipe",
            params={"direction": "up"},
        )

    @staticmethod
    def observation(
        state: str,
        identity: str,
        *,
        confidence: float = 0.99,
        blocking_overlays: tuple[str, ...] = (),
    ) -> PageObservation:
        return PageObservation(
            state=state,
            base_state=state,
            confidence=confidence,
            stable=True,
            reason=state,
            overlays=blocking_overlays,
            blocking_overlays=blocking_overlays,
            page_fingerprint=identity,
            visible_texts=[f"@{identity}", f"标题-{identity}"],
        )

    def test_live_preview_swipes_once_and_verifies_new_page(self) -> None:
        sensor = FakeSensor(
            self.observation("douyin_live_preview", "live-a"),
            self.observation("douyin_video", "video-b"),
        )
        swipes = []
        adapter = SwipeUpActionAdapter(
            sensor,
            lambda: swipes.append("up"),
            post_swipe_wait=0,
        )
        result = adapter.execute(self.action, self.goal)
        self.assertTrue(result.action_result.success)
        self.assertEqual(swipes, ["up"])
        self.assertEqual(result.action_result.details["physical_actions"], 1)
        self.assertEqual(result.action_result.details["retry_count"], 0)
        self.assertEqual(len(sensor.calls), 2)

    def test_ad_page_can_be_skipped_once(self) -> None:
        sensor = FakeSensor(
            self.observation("douyin_ad", "ad-a"),
            self.observation("douyin_video", "video-b"),
        )
        swipes = []
        result = SwipeUpActionAdapter(
            sensor,
            lambda: swipes.append("up"),
            post_swipe_wait=0,
        ).execute(self.action, self.goal)
        self.assertTrue(result.action_result.success)
        self.assertEqual(swipes, ["up"])

    def test_false_loading_label_on_stable_live_preview_does_not_deadlock_skip(self) -> None:
        sensor = FakeSensor(
            self.observation(
                "douyin_live_preview",
                "live-a",
                blocking_overlays=("loading",),
            ),
            self.observation("douyin_video", "video-b"),
        )
        swipes = []
        result = SwipeUpActionAdapter(
            sensor,
            lambda: swipes.append("up"),
            post_swipe_wait=0,
        ).execute(self.action, self.goal)
        self.assertTrue(result.action_result.success)
        self.assertEqual(swipes, ["up"])
        self.assertEqual(result.action_result.details["physical_actions"], 1)

    def test_real_dialog_on_live_preview_still_blocks_before_swipe(self) -> None:
        sensor = FakeSensor(
            self.observation(
                "douyin_live_preview",
                "live-a",
                blocking_overlays=("app_dialog",),
            )
        )
        swipes = []
        with self.assertRaisesRegex(SemanticActionAdapterError, "存在阻塞弹层"):
            SwipeUpActionAdapter(
                sensor,
                lambda: swipes.append("up"),
                post_swipe_wait=0,
            ).execute(self.action, self.goal)
        self.assertEqual(swipes, [])

    def test_ordinary_video_stops_before_swipe(self) -> None:
        sensor = FakeSensor(self.observation("douyin_video", "video-a"))
        swipes = []
        with self.assertRaisesRegex(SemanticActionAdapterError, "不是直播预览或广告"):
            SwipeUpActionAdapter(
                sensor,
                lambda: swipes.append("up"),
                post_swipe_wait=0,
            ).execute(self.action, self.goal)
        self.assertEqual(swipes, [])
        self.assertEqual(len(sensor.calls), 1)

    def test_full_live_room_stops_before_swipe(self) -> None:
        sensor = FakeSensor(self.observation("douyin_live_room", "room-a"))
        swipes = []
        with self.assertRaisesRegex(SemanticActionAdapterError, "上划前不是稳定"):
            SwipeUpActionAdapter(
                sensor,
                lambda: swipes.append("up"),
                post_swipe_wait=0,
            ).execute(self.action, self.goal)
        self.assertEqual(swipes, [])

    def test_low_confidence_stops_before_swipe(self) -> None:
        sensor = FakeSensor(
            self.observation("douyin_live_preview", "live-a", confidence=0.60)
        )
        swipes = []
        with self.assertRaisesRegex(SemanticActionAdapterError, "置信度低于"):
            SwipeUpActionAdapter(
                sensor,
                lambda: swipes.append("up"),
                post_swipe_wait=0,
            ).execute(self.action, self.goal)
        self.assertEqual(swipes, [])

    def test_same_page_after_swipe_stops_without_retry(self) -> None:
        sensor = FakeSensor(
            self.observation("douyin_live_preview", "live-a"),
            self.observation("douyin_live_preview", "live-a"),
        )
        swipes = []
        with self.assertRaises(PhysicalActionVerificationError) as caught:
            SwipeUpActionAdapter(
                sensor,
                lambda: swipes.append("up"),
                post_swipe_wait=0,
            ).execute(self.action, self.goal)
        self.assertEqual(caught.exception.physical_actions, 1)
        self.assertIn("禁止自动补划", str(caught.exception))
        self.assertEqual(swipes, ["up"])
        self.assertEqual(len(sensor.calls), 2)

    def test_wrong_direction_stops_before_observation(self) -> None:
        sensor = FakeSensor()
        with self.assertRaisesRegex(SemanticActionAdapterError, "只允许向上"):
            SwipeUpActionAdapter(
                sensor,
                lambda: None,
                post_swipe_wait=0,
            ).execute(
                SemanticAction("swipe_down", "swipe", {"direction": "down"}),
                self.goal,
            )
        self.assertEqual(sensor.calls, [])


class RecordingAdapter:
    def __init__(self, result) -> None:
        self.result = result
        self.calls = []

    def execute(self, action, goal, **kwargs):
        self.calls.append((action, goal, kwargs))
        return self.result


class SemanticActionRouterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.goal = GoalSpec.from_operation(
            "douyin.batch_interact",
            {"target_count": 1, "like": True, "comment": False},
        )
        self.observe = RecordingAdapter("observed")
        self.ensure = RecordingAdapter("ensured")
        self.heart = RecordingAdapter("liked")
        self.swipe = RecordingAdapter("swiped")
        self.router = SemanticActionRouter(
            observe=self.observe,
            ensure_app=self.ensure,
            tap_heart=self.heart,
            swipe_up=self.swipe,
        )

    def test_observe_routes_to_only_read_only_adapter(self) -> None:
        action = SemanticAction("observe", "observe", {})
        self.assertEqual(self.router.execute(action, self.goal), "observed")
        self.assertEqual(len(self.observe.calls), 1)
        self.assertEqual(self.ensure.calls, [])
        self.assertEqual(self.heart.calls, [])

    def test_ensure_app_routes_to_only_app_adapter(self) -> None:
        action = SemanticAction(
            "open_app", "ensure_app", {"app_id": "douyin"}
        )
        self.assertEqual(self.router.execute(action, self.goal), "ensured")
        self.assertEqual(len(self.ensure.calls), 1)
        self.assertEqual(self.observe.calls, [])
        self.assertEqual(self.heart.calls, [])

    def test_only_heart_target_can_reach_tap_adapter(self) -> None:
        action = SemanticAction("tap_heart", "tap_semantic", {"target": "heart"})
        self.assertEqual(self.router.execute(action, self.goal), "liked")
        self.assertEqual(len(self.heart.calls), 1)

    def test_up_swipe_routes_to_only_swipe_adapter(self) -> None:
        action = SemanticAction("skip", "swipe", {"direction": "up"})
        self.assertEqual(self.router.execute(action, self.goal), "swiped")
        self.assertEqual(len(self.swipe.calls), 1)
        self.assertEqual(self.observe.calls, [])
        self.assertEqual(self.ensure.calls, [])
        self.assertEqual(self.heart.calls, [])

    def test_unknown_or_other_tap_stops_before_every_adapter(self) -> None:
        for action in (
            SemanticAction("swipe", "swipe_semantic", {"direction": "up"}),
            SemanticAction("comment", "tap_semantic", {"target": "comments"}),
        ):
            with self.assertRaisesRegex(
                SemanticActionAdapterError, "尚未进入统一白名单"
            ):
                self.router.execute(action, self.goal)
        self.assertEqual(self.observe.calls, [])
        self.assertEqual(self.ensure.calls, [])
        self.assertEqual(self.heart.calls, [])

    def test_observe_rejects_stale_capture_instead_of_reusing_it(self) -> None:
        action = SemanticAction("observe", "observe", {})
        stale = capture(
            PageObservation(
                state="douyin_video",
                base_state="douyin_video",
                confidence=0.99,
                stable=True,
                reason="旧观察",
            )
        )
        with self.assertRaisesRegex(SemanticActionAdapterError, "重新采集"):
            self.router.execute(action, self.goal, before_capture=stale)
        self.assertEqual(self.observe.calls, [])


if __name__ == "__main__":
    unittest.main()
