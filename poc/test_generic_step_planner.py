from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from generic_action_adapter import GenericActionAdapterError, GenericSingleActionAdapter
from generic_intent import GenericIntentDraft
from generic_scene_observer import _local_frame_fingerprint
from generic_step_planner import (
    GenericStepPlanner,
    GenericStepPlanningError,
    GenericStepProposal,
)
from generic_supervised_runtime import GenericSupervisedSession
from semantic_executor import SemanticAction
from ui_scene import UIElement, UIScene
from vision_agent import VisionAgentError


class FakeTextProvider:
    configured = True

    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def chat_json(self, messages, max_tokens=700):
        self.calls += 1
        self.messages = messages
        return json.dumps(self.payload, ensure_ascii=False)


class FakeSceneObserver:
    def __init__(self, scenes):
        self.scenes = list(scenes)
        self.calls = 0

    def observe(self, *, frames, goal_context=None):
        self.calls += 1
        result = self.scenes.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


class FakeRobot:
    def __init__(self):
        self.actions = []

    def vision_tap_relative(self, x, y):
        self.actions.append(("tap", x, y))
        return (x, y)

    def vision_swipe_up(self):
        self.actions.append(("swipe", "up"))

    def vision_android_back(self):
        self.actions.append(("back",))
        return (500, 950)


class SequenceCapture:
    def __init__(self, colors):
        self.colors = list(colors)
        self.calls = 0

    def __call__(self):
        index = min(self.calls, len(self.colors) - 1)
        self.calls += 1
        return Image.new("RGB", (540, 960), self.colors[index])


class SecondPostCaptureFailureAdapter(GenericSingleActionAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.post_capture_calls = 0

    def _capture_stable_post_action_frames(self, **kwargs):
        self.post_capture_calls += 1
        if self.post_capture_calls == 2:
            raise GenericActionAdapterError(
                "动作后画面在限定时间内没有稳定",
                evidence=("second_capture_timeout.jpg",),
            )
        return super()._capture_stable_post_action_frames(**kwargs)


def goal():
    return GenericIntentDraft(
        understood=True,
        app_id="settings",
        app_name="设置",
        objective="打开蓝牙设置",
        success_criteria={"screen": "蓝牙设置"},
    )


def scene(
    fingerprint,
    *,
    screen_id="android_home",
    element_id="e1",
    bounds=(0.2, 0.3, 0.4, 0.5),
    app_id=None,
):
    return UIScene(
        app_id=(
            app_id
            if app_id is not None
            else ("settings" if screen_id != "android_home" else "unknown")
        ),
        screen_id=screen_id,
        summary="测试页面",
        elements=(
            UIElement(
                element_id=element_id,
                role="icon",
                meaning="app_icon",
                label="设置",
                bounds=bounds,
                confidence=0.96,
            ),
        ),
        stable=True,
        confidence=0.95,
        fingerprint=fingerprint,
    )


class GenericStepPlannerTests(unittest.TestCase):
    def test_proposes_only_one_existing_element(self):
        provider = FakeTextProvider(
            {
                "status": "action",
                "action": {
                    "kind": "tap_semantic",
                    "element_id": "e1",
                    "target": "app_icon",
                    "role": "icon",
                    "label": "设置",
                    "states": {},
                    "expected_effect": {},
                },
                "reason": "设置图标清晰可见",
                "completion_evidence": [],
            }
        )
        proposal = GenericStepPlanner(provider).propose(goal(), scene("a"))
        self.assertEqual(proposal.status, "action")
        self.assertEqual(proposal.action.params["element_id"], "e1")
        self.assertEqual(provider.calls, 1)

    def test_rejects_raw_coordinates(self):
        provider = FakeTextProvider(
            {
                "status": "action",
                "action": {
                    "kind": "tap_semantic",
                    "element_id": "e1",
                    "target": "app_icon",
                    "x": 300,
                },
                "reason": "bad",
                "completion_evidence": [],
            }
        )
        with self.assertRaisesRegex(GenericStepPlanningError, "协议外字段"):
            GenericStepPlanner(provider).propose(goal(), scene("a"))

    def test_finished_requires_visible_evidence(self):
        provider = FakeTextProvider(
            {
                "status": "finished",
                "action": None,
                "reason": "完成",
                "completion_evidence": [],
            }
        )
        with self.assertRaisesRegex(GenericStepPlanningError, "可见证据"):
            GenericStepPlanner(provider).propose(goal(), scene("a"))


class GenericActionAdapterTests(unittest.TestCase):
    def _adapter(self, observer, robot):
        return GenericSingleActionAdapter(
            capture=lambda: Image.new("RGB", (540, 960), "gray"),
            observer=observer,
            robot=robot,
            frame_interval=0,
            post_action_settle=0,
        )

    def test_confirmed_tap_executes_exactly_once_and_reobserves(self):
        planned = scene("planned", bounds=(0.1, 0.2, 0.3, 0.4))
        fresh = scene("before", element_id="fresh", bounds=(0.11, 0.21, 0.31, 0.41))
        after = scene("after", screen_id="app_home", element_id="after")
        observer = FakeSceneObserver([fresh, after])
        robot = FakeRobot()
        action = SemanticAction(
            node_id="generic_step_1",
            action="tap_semantic",
            params={
                "element_id": "e1",
                "target": "app_icon",
                "role": "icon",
                "label": "设置",
            },
        )
        with tempfile.TemporaryDirectory() as temp:
            result = self._adapter(observer, robot).execute(
                requested_action=action,
                planned_scene=planned,
                goal=goal(),
                confirmed=True,
                evidence_dir=Path(temp),
            )
        self.assertEqual(robot.actions, [("tap", 210, 310)])
        self.assertEqual(result.physical_actions, 1)
        self.assertEqual(observer.calls, 2)

    def test_execution_result_keeps_exact_four_verified_after_frames(self):
        gray = Image.new("RGB", (540, 960), "gray")
        white = Image.new("RGB", (540, 960), "white")
        before_fingerprint = _local_frame_fingerprint(gray)
        after_fingerprint = _local_frame_fingerprint(white)
        planned = scene(before_fingerprint)
        fresh = scene(before_fingerprint, element_id="fresh")
        after = scene(
            after_fingerprint,
            screen_id="app_home",
            element_id="after",
        )
        observer = FakeSceneObserver([fresh, after])
        robot = FakeRobot()
        adapter = GenericSingleActionAdapter(
            capture=SequenceCapture(["gray"] * 4 + ["white"] * 4),
            observer=observer,
            robot=robot,
            frame_interval=0,
            post_action_settle=0,
        )
        action = SemanticAction(
            node_id="generic_step_1",
            action="tap_semantic",
            params={"element_id": "e1", "target": "app_icon"},
        )

        with tempfile.TemporaryDirectory() as temp:
            result = adapter.execute(
                requested_action=action,
                planned_scene=planned,
                goal=goal(),
                confirmed=True,
                evidence_dir=Path(temp),
            )

        self.assertEqual(4, len(result.after_frames))
        self.assertTrue(
            all(frame.getpixel((0, 0)) == (255, 255, 255) for frame in result.after_frames)
        )
        self.assertEqual(4, len(result.after_frame_paths))

    def test_after_frame_fingerprint_matches_after_scene(self):
        gray = Image.new("RGB", (540, 960), "gray")
        white = Image.new("RGB", (540, 960), "white")
        before_fingerprint = _local_frame_fingerprint(gray)
        after_fingerprint = _local_frame_fingerprint(white)
        adapter = GenericSingleActionAdapter(
            capture=SequenceCapture(["gray"] * 4 + ["white"] * 4),
            observer=FakeSceneObserver(
                [
                    scene(before_fingerprint, element_id="fresh"),
                    scene(after_fingerprint, screen_id="app_home", element_id="after"),
                ]
            ),
            robot=FakeRobot(),
            frame_interval=0,
            post_action_settle=0,
        )
        action = SemanticAction(
            node_id="generic_step_1",
            action="tap_semantic",
            params={"element_id": "e1", "target": "app_icon"},
        )

        result = adapter.execute(
            requested_action=action,
            planned_scene=scene(before_fingerprint),
            goal=goal(),
            confirmed=True,
        )

        selected_fingerprint = _local_frame_fingerprint(result.after_frames[0])
        self.assertEqual(result.after_scene.fingerprint, selected_fingerprint)

    def test_rebind_rejects_same_label_when_meaning_changes(self):
        planned = UIScene(
            app_id="unknown",
            screen_id="android_home",
            summary="主屏幕",
            elements=(
                UIElement(
                    element_id="douyin_app_icon",
                    role="icon",
                    meaning="抖音应用启动入口",
                    label="抖音",
                    bounds=(0.69, 0.63, 0.84, 0.76),
                    confidence=0.98,
                    states={"goal_relevant": True},
                ),
            ),
            stable=True,
            confidence=0.97,
            fingerprint="planned",
        )
        fresh = UIScene(
            app_id="unknown",
            screen_id="android_home",
            summary="主屏幕",
            elements=(
                UIElement(
                    element_id="e1",
                    role="icon",
                    meaning="抖音应用图标",
                    label="抖音",
                    bounds=(0.70, 0.64, 0.85, 0.77),
                    confidence=0.97,
                    states={"goal_relevant": True},
                ),
            ),
            stable=True,
            confidence=0.96,
            fingerprint="planned",
        )
        after = scene(
            "after",
            screen_id="app_home",
            element_id="after",
            app_id="douyin",
        )
        observer = FakeSceneObserver([fresh, after])
        robot = FakeRobot()
        action = SemanticAction(
            node_id="generic_step_1",
            action="tap_semantic",
            params={
                "element_id": "douyin_app_icon",
                "target": "抖音应用启动入口",
                "role": "icon",
                "label": "抖音",
                "states": {"goal_relevant": True},
            },
        )
        with self.assertRaisesRegex(GenericActionAdapterError, "语义"):
            self._adapter(observer, robot).execute(
                requested_action=action,
                planned_scene=planned,
                goal=goal(),
                confirmed=True,
            )
        self.assertEqual(robot.actions, [])

    def test_changed_target_region_before_confirmation_stops_without_robot_action(self):
        planned = scene("planned")
        fresh = scene("fresh", bounds=(0.65, 0.65, 0.85, 0.85))
        observer = FakeSceneObserver([fresh])
        robot = FakeRobot()
        action = SemanticAction(
            node_id="generic_step_1",
            action="tap_semantic",
            params={"element_id": "e1", "target": "app_icon"},
        )

        with self.assertRaisesRegex(GenericActionAdapterError, "区域"):
            self._adapter(observer, robot).execute(
                requested_action=action,
                planned_scene=planned,
                goal=goal(),
                confirmed=True,
            )
        self.assertEqual(robot.actions, [])

    def test_changed_target_state_before_confirmation_stops_without_robot_action(self):
        planned = UIScene(
            app_id="unknown",
            screen_id="android_home",
            summary="主屏幕",
            elements=(
                UIElement(
                    element_id="e1",
                    role="icon",
                    meaning="app_icon",
                    label="设置",
                    bounds=(0.2, 0.3, 0.4, 0.5),
                    confidence=0.96,
                    states={"enabled": True},
                ),
            ),
            fingerprint="planned",
        )
        fresh = UIScene(
            app_id="unknown",
            screen_id="android_home",
            summary="主屏幕",
            elements=(
                UIElement(
                    element_id="fresh",
                    role="icon",
                    meaning="app_icon",
                    label="设置",
                    bounds=(0.2, 0.3, 0.4, 0.5),
                    confidence=0.96,
                    states={"enabled": False},
                ),
            ),
            fingerprint="fresh",
        )
        robot = FakeRobot()
        with self.assertRaisesRegex(GenericActionAdapterError, "语义|状态"):
            self._adapter(FakeSceneObserver([fresh]), robot).execute(
                requested_action=SemanticAction(
                    node_id="generic_step_1",
                    action="tap_semantic",
                    params={"element_id": "e1", "target": "app_icon"},
                ),
                planned_scene=planned,
                goal=goal(),
                confirmed=True,
            )
        self.assertEqual([], robot.actions)

    def test_each_execution_uses_unique_evidence_paths(self):
        action = SemanticAction(
            node_id="generic_step_1",
            action="tap_semantic",
            params={"element_id": "e1", "target": "app_icon"},
        )
        with tempfile.TemporaryDirectory() as temp:
            evidence_dir = Path(temp)
            first = self._adapter(
                FakeSceneObserver(
                    [scene("same", element_id="fresh-1"), scene("after-1", screen_id="app_home")]
                ),
                FakeRobot(),
            ).execute(
                requested_action=action,
                planned_scene=scene("same"),
                goal=goal(),
                confirmed=True,
                evidence_dir=evidence_dir,
            )
            first_bytes = {path: Path(path).read_bytes() for path in first.evidence}
            second = self._adapter(
                FakeSceneObserver(
                    [scene("same", element_id="fresh-2"), scene("after-2", screen_id="app_home")]
                ),
                FakeRobot(),
            ).execute(
                requested_action=action,
                planned_scene=scene("same"),
                goal=goal(),
                confirmed=True,
                evidence_dir=evidence_dir,
            )

            self.assertTrue(set(first.evidence).isdisjoint(second.evidence))
            self.assertEqual(
                first_bytes,
                {path: Path(path).read_bytes() for path in first.evidence},
            )

    def test_changed_screen_before_confirmation_stops_without_robot_action(self):
        planned = scene("planned")
        changed = scene("before", screen_id="app_home")
        observer = FakeSceneObserver([changed])
        robot = FakeRobot()
        action = SemanticAction(
            node_id="generic_step_1",
            action="tap_semantic",
            params={"element_id": "e1", "target": "app_icon"},
        )
        with self.assertRaisesRegex(GenericActionAdapterError, "已变化"):
            self._adapter(observer, robot).execute(
                requested_action=action,
                planned_scene=planned,
                goal=goal(),
                confirmed=True,
            )
        self.assertEqual(robot.actions, [])

    def test_goal_polluted_home_app_is_normalized_before_confirmation(self):
        planned = scene("planned", app_id="douyin")
        fresh = scene("before", element_id="fresh", app_id="unknown")
        after = scene(
            "after",
            screen_id="app_home",
            element_id="after",
            app_id="douyin",
        )
        observer = FakeSceneObserver([fresh, after])
        robot = FakeRobot()
        action = SemanticAction(
            node_id="generic_step_1",
            action="tap_semantic",
            params={"element_id": "e1", "target": "app_icon"},
        )
        result = self._adapter(observer, robot).execute(
            requested_action=action,
            planned_scene=planned,
            goal=goal(),
            confirmed=True,
        )
        self.assertEqual(planned.foreground_app_id, "launcher")
        self.assertEqual(fresh.foreground_app_id, "launcher")
        self.assertEqual(result.physical_actions, 1)
        self.assertEqual(len(robot.actions), 1)

    def test_failed_post_verification_never_retries_physical_action(self):
        planned = scene("same")
        fresh = scene("same", element_id="fresh")
        unchanged = scene("same", element_id="after")
        observer = FakeSceneObserver([fresh, unchanged, unchanged])
        robot = FakeRobot()
        action = SemanticAction(
            node_id="generic_step_1",
            action="tap_semantic",
            params={"element_id": "e1", "target": "app_icon"},
        )
        with self.assertRaisesRegex(GenericActionAdapterError, "没有可验证变化") as ctx:
            self._adapter(observer, robot).execute(
                requested_action=action,
                planned_scene=planned,
                goal=goal(),
                confirmed=True,
            )
        self.assertEqual(ctx.exception.physical_actions, 1)
        self.assertEqual(len(robot.actions), 1)

    def test_post_action_waits_until_four_frame_window_is_locally_stable(self):
        planned = scene("planned")
        fresh = scene("before", element_id="fresh")
        after = scene("after", screen_id="app_home", element_id="after")
        observer = FakeSceneObserver([fresh, after])
        robot = FakeRobot()
        # Four gray frames are used by confirmation.  The post-action window
        # then sees a black/white transition before four consecutive white
        # frames finally settle.
        capture = SequenceCapture(
            ["gray"] * 4 + ["black", "white", "white", "white", "white"]
        )
        adapter = GenericSingleActionAdapter(
            capture=capture,
            observer=observer,
            robot=robot,
            frame_interval=0,
            post_action_settle=0,
            post_action_timeout=1,
        )
        action = SemanticAction(
            node_id="generic_step_1",
            action="tap_semantic",
            params={"element_id": "e1", "target": "app_icon"},
        )
        result = adapter.execute(
            requested_action=action,
            planned_scene=planned,
            goal=goal(),
            confirmed=True,
        )
        self.assertEqual(result.physical_actions, 1)
        self.assertEqual(robot.actions, [("tap", 300, 400)])
        self.assertEqual(capture.calls, 9)
        self.assertEqual(observer.calls, 2)

    def test_post_action_allows_second_observation_without_repeating_action(self):
        planned = scene("planned")
        fresh = scene("before", element_id="fresh")
        transitional = UIScene(
            app_id="unknown",
            screen_id="loading",
            summary="过渡中",
            elements=(),
            stable=False,
            confidence=0.30,
            fingerprint="transition",
        )
        after = scene("after", screen_id="app_home", element_id="after")
        observer = FakeSceneObserver([fresh, transitional, after])
        robot = FakeRobot()
        adapter = GenericSingleActionAdapter(
            capture=lambda: Image.new("RGB", (540, 960), "gray"),
            observer=observer,
            robot=robot,
            frame_interval=0,
            post_action_settle=0,
            post_action_timeout=1,
        )
        action = SemanticAction(
            node_id="generic_step_1",
            action="tap_semantic",
            params={"element_id": "e1", "target": "app_icon"},
        )
        result = adapter.execute(
            requested_action=action,
            planned_scene=planned,
            goal=goal(),
            confirmed=True,
        )
        self.assertEqual(result.physical_actions, 1)
        self.assertEqual(robot.actions, [("tap", 300, 400)])
        self.assertEqual(observer.calls, 3)

    def test_post_action_format_failure_recaptures_once_without_repeating_action(self):
        planned = scene("planned")
        fresh = scene("before", element_id="fresh")
        after = scene("after", screen_id="app_home", element_id="after")
        observer = FakeSceneObserver(
            [fresh, VisionAgentError("模型返回的 JSON 无法解析"), after]
        )
        robot = FakeRobot()
        capture = SequenceCapture(["gray"] * 12)
        adapter = GenericSingleActionAdapter(
            capture=capture,
            observer=observer,
            robot=robot,
            frame_interval=0,
            post_action_settle=0,
            post_action_timeout=1,
        )
        action = SemanticAction(
            node_id="generic_step_1",
            action="tap_semantic",
            params={"element_id": "e1", "target": "app_icon"},
        )

        with tempfile.TemporaryDirectory() as temp:
            result = adapter.execute(
                requested_action=action,
                planned_scene=planned,
                goal=goal(),
                confirmed=True,
                evidence_dir=Path(temp),
            )

        self.assertEqual(robot.actions, [("tap", 300, 400)])
        self.assertEqual(result.physical_actions, 1)
        self.assertEqual(observer.calls, 3)
        self.assertEqual(capture.calls, 12)
        self.assertEqual(len(result.evidence), 12)
        self.assertEqual(len(result.after_frame_paths), 4)
        self.assertEqual(
            result.observation_errors,
            ("第1轮动作后观察失败：模型返回的 JSON 无法解析",),
        )
        self.assertEqual(
            result.to_dict()["observation_errors"],
            ["第1轮动作后观察失败：模型返回的 JSON 无法解析"],
        )

    def test_two_post_action_format_failures_stop_after_one_robot_action(self):
        observer = FakeSceneObserver(
            [
                scene("before", element_id="fresh"),
                VisionAgentError("模型返回的 JSON 无法解析"),
                VisionAgentError("模型返回的 JSON 无法解析"),
            ]
        )
        robot = FakeRobot()
        adapter = GenericSingleActionAdapter(
            capture=SequenceCapture(["gray"] * 12),
            observer=observer,
            robot=robot,
            frame_interval=0,
            post_action_settle=0,
            post_action_timeout=1,
        )

        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(GenericActionAdapterError) as caught:
                adapter.execute(
                    requested_action=SemanticAction(
                        node_id="generic_step_1",
                        action="tap_semantic",
                        params={"element_id": "e1", "target": "app_icon"},
                    ),
                    planned_scene=scene("planned"),
                    goal=goal(),
                    confirmed=True,
                    evidence_dir=Path(temp),
                )

        self.assertEqual(caught.exception.physical_actions, 1)
        self.assertEqual(len(caught.exception.evidence), 12)
        self.assertEqual(observer.calls, 3)
        self.assertEqual(len(robot.actions), 1)
        self.assertEqual(
            caught.exception.observation_errors,
            (
                "第1轮动作后观察失败：模型返回的 JSON 无法解析",
                "第2轮动作后观察失败：模型返回的 JSON 无法解析",
            ),
        )
        self.assertIn("第1轮动作后观察失败", str(caught.exception))
        self.assertIn("第2轮动作后观察失败", str(caught.exception))

    def test_post_action_observation_limit_is_hard_capped_at_two(self):
        observer = FakeSceneObserver(
            [
                scene("before", element_id="fresh"),
                VisionAgentError("模型返回的 JSON 无法解析：first"),
                VisionAgentError("模型返回的 JSON 无法解析：second"),
                scene("after", screen_id="app_home", element_id="after"),
            ]
        )
        robot = FakeRobot()
        capture = SequenceCapture(["gray"] * 16)
        adapter = GenericSingleActionAdapter(
            capture=capture,
            observer=observer,
            robot=robot,
            frame_interval=0,
            post_action_settle=0,
            post_action_timeout=1,
            post_action_max_observations=5,
        )

        with self.assertRaises(GenericActionAdapterError) as caught:
            adapter.execute(
                requested_action=SemanticAction(
                    node_id="generic_step_1",
                    action="tap_semantic",
                    params={"element_id": "e1", "target": "app_icon"},
                ),
                planned_scene=scene("planned"),
                goal=goal(),
                confirmed=True,
            )

        self.assertEqual(adapter.post_action_max_observations, 2)
        self.assertEqual(caught.exception.physical_actions, 1)
        self.assertEqual(observer.calls, 3)
        self.assertEqual(capture.calls, 12)
        self.assertEqual(robot.actions, [("tap", 300, 400)])

    def test_second_capture_failure_keeps_first_format_error_and_evidence(self):
        observer = FakeSceneObserver(
            [
                scene("before", element_id="fresh"),
                VisionAgentError("模型返回的 JSON 无法解析：first"),
            ]
        )
        robot = FakeRobot()
        adapter = SecondPostCaptureFailureAdapter(
            capture=SequenceCapture(["gray"] * 8),
            observer=observer,
            robot=robot,
            frame_interval=0,
            post_action_settle=0,
            post_action_timeout=1,
        )

        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(GenericActionAdapterError) as caught:
                adapter.execute(
                    requested_action=SemanticAction(
                        node_id="generic_step_1",
                        action="tap_semantic",
                        params={"element_id": "e1", "target": "app_icon"},
                    ),
                    planned_scene=scene("planned"),
                    goal=goal(),
                    confirmed=True,
                    evidence_dir=Path(temp),
                )

        self.assertEqual(caught.exception.physical_actions, 1)
        self.assertEqual(
            caught.exception.observation_errors,
            ("第1轮动作后观察失败：模型返回的 JSON 无法解析：first",),
        )
        self.assertEqual(len(caught.exception.evidence), 9)
        self.assertEqual(caught.exception.evidence[-1], "second_capture_timeout.jpg")
        self.assertEqual(observer.calls, 2)
        self.assertEqual(robot.actions, [("tap", 300, 400)])

    def test_post_action_non_format_failure_is_not_retried(self):
        observer = FakeSceneObserver(
            [scene("before", element_id="fresh"), VisionAgentError("请求超时")]
        )
        robot = FakeRobot()
        capture = SequenceCapture(["gray"] * 8)
        adapter = GenericSingleActionAdapter(
            capture=capture,
            observer=observer,
            robot=robot,
            frame_interval=0,
            post_action_settle=0,
            post_action_timeout=1,
        )

        with self.assertRaises(GenericActionAdapterError) as caught:
            adapter.execute(
                requested_action=SemanticAction(
                    node_id="generic_step_1",
                    action="tap_semantic",
                    params={"element_id": "e1", "target": "app_icon"},
                ),
                planned_scene=scene("planned"),
                goal=goal(),
                confirmed=True,
            )

        self.assertEqual(caught.exception.physical_actions, 1)
        self.assertEqual(observer.calls, 2)
        self.assertEqual(capture.calls, 8)
        self.assertEqual(len(robot.actions), 1)

    def test_post_action_timeout_stops_without_calling_model_or_tapping_again(self):
        planned = scene("planned")
        fresh = scene("before", element_id="fresh")
        observer = FakeSceneObserver([fresh])
        robot = FakeRobot()
        capture = SequenceCapture(
            ["gray"] * 4 + ["black", "white", "black", "white"]
        )
        adapter = GenericSingleActionAdapter(
            capture=capture,
            observer=observer,
            robot=robot,
            frame_interval=0,
            post_action_settle=0,
            post_action_timeout=0,
        )
        action = SemanticAction(
            node_id="generic_step_1",
            action="tap_semantic",
            params={"element_id": "e1", "target": "app_icon"},
        )
        with self.assertRaisesRegex(
            GenericActionAdapterError,
            "限定时间内没有稳定",
        ) as ctx:
            adapter.execute(
                requested_action=action,
                planned_scene=planned,
                goal=goal(),
                confirmed=True,
            )
        self.assertEqual(ctx.exception.physical_actions, 1)
        self.assertEqual(observer.calls, 1)
        self.assertEqual(robot.actions, [("tap", 300, 400)])


class GenericSupervisedSessionTests(unittest.TestCase):
    def _proposal(self):
        return GenericStepProposal(
            status="action",
            action=SemanticAction(
                node_id="generic_step_1",
                action="tap_semantic",
                params={"element_id": "e1", "target": "app_icon"},
            ),
            reason="设置图标清晰可见",
        )

    def test_confirmation_executes_one_action_then_pauses(self):
        initial = scene("initial")
        before = scene("before", element_id="fresh")
        after = scene("after", screen_id="app_home", element_id="after")
        observer = FakeSceneObserver([before, after])
        robot = FakeRobot()
        adapter = GenericSingleActionAdapter(
            capture=lambda: Image.new("RGB", (540, 960), "gray"),
            observer=observer,
            robot=robot,
            frame_interval=0,
            post_action_settle=0,
        )
        planner = GenericStepPlanner(
            FakeTextProvider(
                {
                    "status": "finished",
                    "action": None,
                    "reason": "已完成",
                    "completion_evidence": ["蓝牙设置标题可见"],
                }
            )
        )
        with tempfile.TemporaryDirectory() as temp:
            session = GenericSupervisedSession.start(
                session_id="s1",
                goal=goal(),
                scene=initial,
                proposal=self._proposal(),
                planner=planner,
                adapter=adapter,
                run_dir=Path(temp),
            )
            self.assertEqual(session.status, "awaiting_confirmation")
            with self.assertRaisesRegex(GenericActionAdapterError, "明确确认"):
                session.confirm(confirmed=False)
            result = session.confirm(confirmed=True)
            self.assertEqual(result.physical_actions, 1)
            self.assertEqual(robot.actions, [("tap", 300, 400)])
            self.assertEqual(session.status, "paused_after_action")
            self.assertIsNone(session.proposal)
            with self.assertRaisesRegex(GenericActionAdapterError, "不能执行"):
                session.confirm(confirmed=True)

    def test_confirmed_terminal_scene_change_finishes_without_replanning(self):
        initial = scene("initial", screen_id="video_detail", app_id="douyin")
        before = scene("video-a", screen_id="video_detail", app_id="douyin")
        after = scene("video-b", screen_id="video_detail", app_id="douyin")
        observer = FakeSceneObserver([before, after])
        robot = FakeRobot()
        adapter = GenericSingleActionAdapter(
            capture=lambda: Image.new("RGB", (540, 960), "gray"),
            observer=observer,
            robot=robot,
            frame_interval=0,
            post_action_settle=0,
        )
        proposal = GenericStepProposal(
            status="action",
            action=SemanticAction(
                node_id="generic_step_1",
                action="swipe",
                params={
                    "direction": "up",
                    "expected_effect": {
                        "scene_changed": True,
                        "goal_complete_on_success": True,
                    },
                },
            ),
            reason="上划后目标内容应改变",
        )
        provider = FakeTextProvider({})
        with tempfile.TemporaryDirectory() as temp:
            session = GenericSupervisedSession.start(
                session_id="terminal-change",
                goal=goal(),
                scene=initial,
                proposal=proposal,
                planner=GenericStepPlanner(provider),
                adapter=adapter,
                run_dir=Path(temp),
            )
            result = session.confirm(confirmed=True)

        self.assertEqual(result.physical_actions, 1)
        self.assertEqual(session.status, "succeeded")
        self.assertEqual(session.proposal.status, "finished")
        self.assertEqual(provider.calls, 0)
        self.assertEqual(robot.actions, [("swipe", "up")])
        self.assertIn("场景指纹发生变化", session.history[-1]["completion_evidence"][0])

    def test_terminal_claim_without_machine_verifiable_effect_only_pauses(self):
        initial = scene("initial")
        before = scene("before", element_id="fresh")
        after = scene("after", screen_id="app_home", element_id="after")
        adapter = GenericSingleActionAdapter(
            capture=lambda: Image.new("RGB", (540, 960), "gray"),
            observer=FakeSceneObserver([before, after]),
            robot=FakeRobot(),
            frame_interval=0,
            post_action_settle=0,
        )
        proposal = GenericStepProposal(
            status="action",
            action=SemanticAction(
                node_id="generic_step_1",
                action="tap_semantic",
                params={
                    "element_id": "e1",
                    "target": "app_icon",
                    "expected_effect": {
                        "description": "应该完成",
                        "goal_complete_on_success": True,
                    },
                },
            ),
            reason="模型声称完成但没有机器证据",
        )
        with tempfile.TemporaryDirectory() as temp:
            session = GenericSupervisedSession.start(
                session_id="unproven-terminal",
                goal=goal(),
                scene=initial,
                proposal=proposal,
                planner=GenericStepPlanner(FakeTextProvider({})),
                adapter=adapter,
                run_dir=Path(temp),
            )
            session.confirm(confirmed=True)

        self.assertEqual(session.status, "paused_after_action")
        self.assertIsNone(session.proposal)

    def test_next_only_plans_and_does_not_touch_robot(self):
        initial = scene("initial")
        robot = FakeRobot()
        planner = GenericStepPlanner(
            FakeTextProvider(
                {
                    "status": "finished",
                    "action": None,
                    "reason": "已完成",
                    "completion_evidence": ["蓝牙设置标题可见"],
                }
            )
        )
        adapter = GenericSingleActionAdapter(
            capture=lambda: Image.new("RGB", (540, 960), "gray"),
            observer=FakeSceneObserver([]),
            robot=robot,
            frame_interval=0,
            post_action_settle=0,
        )
        with tempfile.TemporaryDirectory() as temp:
            session = GenericSupervisedSession.start(
                session_id="s2",
                goal=goal(),
                scene=initial,
                proposal=self._proposal(),
                planner=planner,
                adapter=adapter,
                run_dir=Path(temp),
            )
            session.status = "paused_after_action"
            session.proposal = None
            proposal = session.plan_next(
                scene("done", screen_id="bluetooth_settings")
            )
        self.assertEqual(proposal.status, "finished")
        self.assertEqual(session.status, "succeeded")
        self.assertEqual(robot.actions, [])

    def test_safe_loop_executes_navigation_then_finishes(self):
        initial = scene("initial")
        before = scene("before", element_id="fresh")
        after = scene(
            "after",
            screen_id="app_home",
            element_id="after",
            app_id="settings",
        )
        observer = FakeSceneObserver([before, after])
        robot = FakeRobot()
        adapter = GenericSingleActionAdapter(
            capture=lambda: Image.new("RGB", (540, 960), "gray"),
            observer=observer,
            robot=robot,
            frame_interval=0,
            post_action_settle=0,
        )
        planner = GenericStepPlanner(
            FakeTextProvider(
                {
                    "status": "finished",
                    "action": None,
                    "reason": "设置已经打开",
                    "completion_evidence": ["设置页面标题可见"],
                }
            )
        )
        with tempfile.TemporaryDirectory() as temp:
            session = GenericSupervisedSession.start(
                session_id="auto1",
                goal=goal(),
                scene=initial,
                proposal=self._proposal(),
                planner=planner,
                adapter=adapter,
                run_dir=Path(temp),
            )
            first = session.run_safe_loop(confirmed=True)
            self.assertEqual(first["physical_actions"], 1)
            self.assertEqual(session.status, "paused_after_action")
            summary = session.run_safe_loop(confirmed=True)
        self.assertEqual(summary["physical_actions"], 0)
        self.assertEqual(session.status, "succeeded")
        self.assertTrue(session.automatic_loop_enabled)
        self.assertEqual(robot.actions, [("tap", 300, 400)])

    def test_safe_loop_pauses_before_account_effect(self):
        proposal = GenericStepProposal(
            status="action",
            action=SemanticAction(
                node_id="generic_step_1",
                action="tap_semantic",
                params={"element_id": "e1", "target": "like"},
            ),
            reason="点赞按钮可见",
        )
        robot = FakeRobot()
        adapter = GenericSingleActionAdapter(
            capture=lambda: Image.new("RGB", (540, 960), "gray"),
            observer=FakeSceneObserver([]),
            robot=robot,
            frame_interval=0,
            post_action_settle=0,
        )
        with tempfile.TemporaryDirectory() as temp:
            session = GenericSupervisedSession.start(
                session_id="auto2",
                goal=goal(),
                scene=scene("video"),
                proposal=proposal,
                planner=GenericStepPlanner(FakeTextProvider({})),
                adapter=adapter,
                run_dir=Path(temp),
            )
            summary = session.run_safe_loop(confirmed=True)
        self.assertEqual(summary["physical_actions"], 0)
        self.assertEqual(session.status, "awaiting_confirmation")
        self.assertIn("账号状态", session.auto_pause_reason)
        self.assertEqual(robot.actions, [])

    def test_natural_language_like_button_is_reported_as_account_effect(self):
        proposal = GenericStepProposal(
            status="action",
            action=SemanticAction(
                node_id="generic_step_1",
                action="tap_semantic",
                params={
                    "element_id": "e1",
                    "target": "点赞按钮",
                    "expected_effect": {
                        "element_state": {
                            "meaning": "点赞按钮",
                            "states": {"is_liked": True},
                        }
                    },
                },
            ),
            reason="白色爱心按钮可见",
        )
        with tempfile.TemporaryDirectory() as temp:
            session = GenericSupervisedSession.start(
                session_id="account-effect-cn",
                goal=goal(),
                scene=scene("video"),
                proposal=proposal,
                planner=GenericStepPlanner(FakeTextProvider({})),
                adapter=GenericSingleActionAdapter(
                    capture=lambda: Image.new("RGB", (540, 960), "gray"),
                    observer=FakeSceneObserver([]),
                    robot=FakeRobot(),
                    frame_interval=0,
                    post_action_settle=0,
                ),
                run_dir=Path(temp),
            )
            snapshot = session.snapshot()
        self.assertTrue(snapshot["current_action"]["account_effect_possible"])


if __name__ == "__main__":
    unittest.main()
