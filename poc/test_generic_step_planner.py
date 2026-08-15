from __future__ import annotations

import json
import tempfile
import unittest
from unittest.mock import patch
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from capability_acceptance import _validate_live_promotion_source
from generic_action_adapter import (
    GenericActionAdapterError,
    GenericSingleActionAdapter as _GenericSingleActionAdapter,
)
from generic_intent import GenericIntentDraft
from generic_scene_observer import GenericSceneObserver, _local_frame_fingerprint
from orientation_safety import (
    _claim_audit_seal,
    _mint_audited_credential,
)
from generic_step_planner import (
    GenericStepPlanner,
    GenericStepPlanningError,
    GenericStepProposal,
)
from generic_supervised_runtime import GenericSupervisedSession
from semantic_executor import SemanticAction
from ui_scene import CameraAlignmentFacts, SystemUIFacts, UIElement, UIScene
from universal_action_controller import UniversalActionController, UniversalActionError
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


class RawSceneProvider:
    configured = True

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.max_tokens_seen = []

    def _chat(self, messages, *, max_tokens, **_kwargs):
        self.calls += 1
        self.max_tokens_seen.append(max_tokens)
        value = self.responses.pop(0)
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        return value

    def status(self):
        return {"configured": True, "model": "offline-sequence"}


class FakeSceneObserver:
    def __init__(
        self,
        scenes,
        *,
        audit_rotation="upright",
        audit_confidence=0.96,
        geometry_scenes=None,
    ):
        self.scenes = list(scenes)
        self.calls = 0
        self.geometry_audit_calls = []
        self.geometry_scenes = list(geometry_scenes or ())
        self.audit_rotation = audit_rotation
        self.audit_confidence = audit_confidence

    def observe(self, *, frames, goal_context=None):
        self.calls += 1
        result = self.scenes.pop(0)
        if isinstance(result, BaseException):
            raise result
        if result.camera_alignment.phone_content_rotation == "unknown":
            result = replace(
                result,
                camera_alignment=aligned_camera_facts(),
            )
        return result

    def audit_camera_alignment(self, *, frames, device_id, scene_fingerprint):
        return _mint_audited_credential(
            device_id=device_id,
            scene_fingerprint=scene_fingerprint,
            frame=frames[-1],
            phone_content_rotation=self.audit_rotation,
            confidence=self.audit_confidence,
            evidence=("测试手机界面轴线",),
        )

    def audit_element_geometry(self, *, frames, scene, element_ids):
        self.geometry_audit_calls.append(tuple(element_ids))
        return self.geometry_scenes.pop(0) if self.geometry_scenes else scene


class FakeRobot:
    def __init__(self):
        self.actions = []
        self.device_id = "test-device"
        self._armed = None

    def arm_physical_execution(self, credential, *, action, scene_fingerprint):
        credential.assert_authorizes(
            device_id=self.device_id,
            scene_fingerprint=scene_fingerprint,
            frame_size=credential.frame_size,
        )
        _claim_audit_seal(credential)
        self._armed = action

    def clear_physical_execution_authorization(self):
        self._armed = None

    def _consume(self, action):
        if self._armed != action:
            raise RuntimeError("missing test physical authorization")
        self._armed = None

    def vision_tap_relative(self, x, y):
        self._consume("tap_semantic")
        self.actions.append(("tap", x, y))
        return (x, y)

    def vision_dismiss_overlay_relative(self, x, y):
        self._consume("dismiss_overlay")
        self.actions.append(("dismiss", x, y))
        return (x, y)

    def vision_swipe_up(self):
        self._consume("swipe")
        self.actions.append(("swipe", "up"))

    def vision_reveal_system_navigation(self):
        self._consume("reveal_system_navigation")
        self.actions.append(("reveal_system_navigation",))
        return (500, 950)

    def vision_android_back(self):
        self._consume("back")
        self.actions.append(("back",))
        return (500, 950)

    def vision_android_home(self):
        self._consume("home")
        self.actions.append(("home",))
        return (500, 950)

    def vision_type_text(self, text):
        self._consume("input_verified_text")
        self.actions.append(("input", text))

    def vision_long_press_relative(self, x, y, hold_seconds):
        self._consume("long_press")
        self.actions.append(("long_press", x, y, hold_seconds))
        return (x, y)

    def vision_drag_relative(self, start_x, start_y, end_x, end_y):
        self._consume("drag")
        self.actions.append(("drag", start_x, start_y, end_x, end_y))


class GenericSingleActionAdapter(_GenericSingleActionAdapter):
    def __init__(self, *args, device_id="test-device", **kwargs):
        super().__init__(*args, device_id=device_id, **kwargs)


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


def aligned_camera_facts() -> CameraAlignmentFacts:
    return CameraAlignmentFacts(
        camera_layout_orientation="portrait",
        phone_content_rotation="upright",
        confidence=0.96,
        evidence=("手机界面轴线与相机画布正向一致",),
    )


def scene(
    fingerprint,
    *,
    screen_id="android_home",
    element_id="e1",
    bounds=(0.2, 0.3, 0.4, 0.5),
    app_id=None,
    system_ui=None,
    camera_alignment=None,
):
    current = UIScene(
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
        system_ui=system_ui or SystemUIFacts(),
        camera_alignment=camera_alignment or aligned_camera_facts(),
    )
    return current


def compact_scene_raw(*, label="设置"):
    payload = scene("model-scene").to_dict()
    payload["elements"][0]["label"] = label
    for element in payload["elements"]:
        element["bounds"] = [value * 1000 for value in element["bounds"]]
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


class GenericStepPlannerTests(unittest.TestCase):
    def test_reveal_system_navigation_has_no_geometry_parameters(self):
        provider = FakeTextProvider(
            {
                "status": "action",
                "action": {
                    "kind": "reveal_system_navigation",
                    "expected_effect": {
                        "system_ui": {"navigation_bar_visible": True}
                    },
                },
                "reason": "当前处于沉浸态且系统导航栏隐藏",
                "completion_evidence": [],
            }
        )
        current = scene(
            "a",
            system_ui=SystemUIFacts(
                immersive_or_fullscreen=True,
                navigation_bar_visible=False,
            ),
        )

        proposal = GenericStepPlanner(provider).propose(goal(), current)

        self.assertEqual("reveal_system_navigation", proposal.action.action)
        self.assertEqual(
            {"expected_effect": {"system_ui": {"navigation_bar_visible": True}}},
            proposal.action.params,
        )

    def test_reveal_system_navigation_rejects_direction(self):
        provider = FakeTextProvider(
            {
                "status": "action",
                "action": {
                    "kind": "reveal_system_navigation",
                    "direction": "up",
                    "expected_effect": {
                        "system_ui": {"navigation_bar_visible": True}
                    },
                },
                "reason": "bad",
                "completion_evidence": [],
            }
        )
        current = scene(
            "a",
            system_ui=SystemUIFacts(
                immersive_or_fullscreen=True,
                navigation_bar_visible=False,
            ),
        )

        with self.assertRaisesRegex(GenericStepPlanningError, "不能携带"):
            GenericStepPlanner(provider).propose(goal(), current)

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


class RevealSystemNavigationControllerTests(unittest.TestCase):
    @staticmethod
    def action() -> SemanticAction:
        return SemanticAction(
            node_id="reveal-navigation",
            action="reveal_system_navigation",
            params={
                "expected_effect": {
                    "system_ui": {"navigation_bar_visible": True}
                }
            },
        )

    def test_requires_structured_system_ui_pre_and_post_facts(self):
        controller = UniversalActionController()
        before = scene(
            "same",
            system_ui=SystemUIFacts(
                immersive_or_fullscreen=True,
                navigation_bar_visible=False,
            ),
        )
        resolved = controller.resolve_one(self.action(), before, confirmed=True)

        controller.verify_after_action(
            resolved,
            before,
            scene(
                "same",
                system_ui=SystemUIFacts(
                    immersive_or_fullscreen=True,
                    navigation_bar_visible=True,
                ),
            ),
        )

        with self.assertRaisesRegex(UniversalActionError, "结构化导航栏可见证据"):
            controller.verify_after_action(
                resolved,
                before,
                scene("different", screen_id="navigation_bar_visible_summary_only"),
            )

    def test_rejects_unknown_or_already_visible_system_ui_precondition(self):
        controller = UniversalActionController()
        for facts in (
            SystemUIFacts(
                immersive_or_fullscreen="unknown",
                navigation_bar_visible="unknown",
            ),
            SystemUIFacts(
                immersive_or_fullscreen=True,
                navigation_bar_visible=True,
            ),
        ):
            with self.subTest(facts=facts), self.assertRaisesRegex(
                UniversalActionError,
                "沉浸态且导航栏隐藏",
            ):
                controller.resolve_one(
                    self.action(),
                    scene("before", system_ui=facts),
                    confirmed=True,
                )


class GenericActionAdapterTests(unittest.TestCase):
    def _adapter(self, observer, robot):
        return GenericSingleActionAdapter(
            capture=lambda: Image.new("RGB", (540, 960), "gray"),
            observer=observer,
            robot=robot,
            frame_interval=0,
            post_action_settle=0,
        )

    def _assert_public_observation_failure_before_robot(self, responses):
        provider = RawSceneProvider(responses)
        robot = FakeRobot()
        adapter = self._adapter(GenericSceneObserver(provider), robot)
        action = SemanticAction(
            node_id="blocked-observer-json",
            action="tap_semantic",
            params={"element_id": "e1", "target": "app_icon"},
        )

        with self.assertRaises(GenericActionAdapterError) as caught:
            adapter.execute(
                requested_action=action,
                planned_scene=scene("planned"),
                goal=goal(),
                confirmed=True,
            )

        self.assertEqual(0, caught.exception.physical_actions)
        self.assertEqual([], robot.actions)
        self.assertIsNone(robot._armed)
        self.assertEqual([1200], provider.max_tokens_seen)

    def test_public_execute_irreparable_observation_never_calls_robot(self):
        self._assert_public_observation_failure_before_robot(
            ["not-json"]
        )

    def test_public_execute_duplicate_key_observation_never_calls_robot(self):
        valid = compact_scene_raw()
        duplicate = valid.replace(
            '"summary":',
            '"summary":"duplicate","summary":',
            1,
        )
        self._assert_public_observation_failure_before_robot(
            [duplicate]
        )

    def test_public_execute_ambiguous_observation_never_calls_robot(self):
        first = compact_scene_raw(label="设置")
        second = compact_scene_raw(label="蓝牙")
        with patch(
            "generic_scene_observer._single_json_structural_edits",
            side_effect=lambda _raw: iter((first, second)),
        ):
            self._assert_public_observation_failure_before_robot(
                ['{"bad":}']
            )

    def test_reports_only_callable_device_actions(self):
        adapter = self._adapter(FakeSceneObserver([]), FakeRobot())

        supported = adapter.supported_action_kinds()

        self.assertIn("tap_semantic", supported)
        self.assertIn("reveal_system_navigation", supported)
        self.assertIn("input_verified_text", supported)
        self.assertIn("long_press", supported)
        self.assertIn("drag", supported)
        self.assertIn("back", supported)
        self.assertIn("home", supported)
        self.assertIn("wait_for_change", supported)
        self.assertIn("swipe", supported)

    def test_orientation_mismatch_blocks_tap_back_home_and_drag_before_robot(self):
        mismatch = CameraAlignmentFacts(
            camera_layout_orientation="portrait",
            phone_content_rotation="rotated_90",
            confidence=0.95,
            evidence=("手机界面文字需旋转九十度才正向",),
        )
        ordinary_scene = scene("planned", camera_alignment=mismatch)
        drag_scene = UIScene(
            app_id="settings",
            screen_id="arrange",
            summary="两个可拖动对象",
            elements=(
                UIElement(
                    element_id="source",
                    role="icon",
                    meaning="source_item",
                    label="源",
                    bounds=(0.15, 0.25, 0.25, 0.35),
                    confidence=0.96,
                ),
                UIElement(
                    element_id="destination",
                    role="icon",
                    meaning="destination_slot",
                    label="目标",
                    bounds=(0.60, 0.25, 0.70, 0.35),
                    confidence=0.96,
                ),
            ),
            confidence=0.96,
            fingerprint="planned",
            camera_alignment=mismatch,
        )
        cases = (
            (
                "tap_semantic",
                ordinary_scene,
                SemanticAction(
                    node_id="blocked-tap",
                    action="tap_semantic",
                    params={"element_id": "e1", "target": "app_icon"},
                ),
            ),
            (
                "back",
                ordinary_scene,
                SemanticAction(node_id="blocked-back", action="back", params={}),
            ),
            (
                "home",
                ordinary_scene,
                SemanticAction(node_id="blocked-home", action="home", params={}),
            ),
            (
                "drag",
                drag_scene,
                SemanticAction(
                    node_id="blocked-drag",
                    action="drag",
                    params={
                        "source_element_id": "source",
                        "destination_element_id": "destination",
                        "expected_effect": {"scene_changed": True},
                    },
                ),
            ),
        )
        frames = tuple(Image.new("RGB", (540, 960), "gray") for _ in range(4))
        for kind, planned, action in cases:
            with self.subTest(kind=kind):
                robot = FakeRobot()
                adapter = GenericSingleActionAdapter(
                    capture=SequenceCapture(["gray"] * 4),
                    observer=FakeSceneObserver(
                        (
                            [planned]
                            if action.action
                            in GenericSingleActionAdapter.GEOMETRY_BOUND_KINDS
                            else []
                        ),
                        audit_rotation="rotated_90",
                    ),
                    robot=robot,
                    frame_interval=0,
                    post_action_settle=0,
                )
                with self.assertRaisesRegex(
                    GenericActionAdapterError,
                    "方向不一致或未知",
                ) as caught:
                    adapter.execute(
                        requested_action=action,
                        planned_scene=planned,
                        planned_frames=frames,
                        goal=goal(),
                        confirmed=True,
                    )
                self.assertEqual(0, caught.exception.physical_actions)
                self.assertEqual([], robot.actions)

    def test_orientation_gate_rejects_unknown_low_confidence_or_local_mismatch(self):
        cases = (
            ("rotated_90", 0.95, "方向不一致或未知"),
            ("rotated_180", 0.95, "方向不一致或未知"),
            ("rotated_270", 0.95, "方向不一致或未知"),
            ("unknown", 0.95, "方向不一致或未知"),
            ("upright", 0.4, "置信度不足"),
        )
        frames = tuple(Image.new("RGB", (540, 960), "gray") for _ in range(4))
        for rotation, confidence, error in cases:
            with self.subTest(rotation=rotation, confidence=confidence):
                robot = FakeRobot()
                adapter = GenericSingleActionAdapter(
                    capture=SequenceCapture(["gray"] * 4),
                    observer=FakeSceneObserver(
                        [],
                        audit_rotation=rotation,
                        audit_confidence=confidence,
                    ),
                    robot=robot,
                    frame_interval=0,
                    post_action_settle=0,
                )
                with self.assertRaisesRegex(GenericActionAdapterError, error) as caught:
                    adapter.execute(
                        requested_action=SemanticAction(
                            node_id="blocked-back",
                            action="back",
                            params={},
                        ),
                        planned_scene=scene("planned"),
                        planned_frames=frames,
                        goal=goal(),
                        confirmed=True,
                    )
                self.assertEqual(0, caught.exception.physical_actions)
                self.assertEqual([], robot.actions)

    def test_reveal_system_navigation_calls_one_dedicated_robot_action(self):
        hidden = SystemUIFacts(
            immersive_or_fullscreen=True,
            navigation_bar_visible=False,
        )
        visible = SystemUIFacts(
            immersive_or_fullscreen=True,
            navigation_bar_visible=True,
        )
        planned = scene("planned", system_ui=hidden)
        after = scene("planned", system_ui=visible)
        robot = FakeRobot()
        adapter = GenericSingleActionAdapter(
            capture=SequenceCapture(["gray"] * 4 + ["white"] * 4),
            observer=FakeSceneObserver([after]),
            robot=robot,
            frame_interval=0,
            post_action_settle=0,
        )

        result = adapter.execute(
            requested_action=SemanticAction(
                node_id="generic_step_1",
                action="reveal_system_navigation",
                params={
                    "expected_effect": {
                        "system_ui": {"navigation_bar_visible": True}
                    }
                },
            ),
            planned_scene=planned,
            planned_frames=tuple(
                Image.new("RGB", (540, 960), "gray") for _ in range(4)
            ),
            goal=goal(),
            confirmed=True,
        )

        self.assertEqual([("reveal_system_navigation",)], robot.actions)
        self.assertEqual(1, result.physical_actions)

    def test_reveal_system_navigation_failure_does_not_fallback_to_swipe(self):
        hidden = SystemUIFacts(
            immersive_or_fullscreen=True,
            navigation_bar_visible=False,
        )
        planned = scene("planned", system_ui=hidden)
        invalid_after = scene(
            "after",
            screen_id="navigation_bar_visible_summary_only",
        )
        robot = FakeRobot()
        adapter = GenericSingleActionAdapter(
            capture=SequenceCapture(["gray"] * 4 + ["white"] * 8),
            observer=FakeSceneObserver([invalid_after, invalid_after]),
            robot=robot,
            frame_interval=0,
            post_action_settle=0,
        )

        result = adapter.execute(
            requested_action=SemanticAction(
                node_id="generic_step_1",
                action="reveal_system_navigation",
                params={
                    "expected_effect": {
                        "system_ui": {"navigation_bar_visible": True}
                    }
                },
            ),
            planned_scene=planned,
            planned_frames=tuple(
                Image.new("RGB", (540, 960), "gray") for _ in range(4)
            ),
            goal=goal(),
            confirmed=True,
        )

        self.assertEqual(1, result.physical_actions)
        self.assertEqual("mismatched", result.action_outcome)
        self.assertTrue(result.verification_errors)
        self.assertEqual([("reveal_system_navigation",)], robot.actions)

    def test_confirmed_home_executes_exactly_once_and_reobserves(self):
        planned = scene(
            "planned",
            screen_id="settings_home",
            element_id="settings-title",
            app_id="settings",
        )
        fresh = scene(
            "before",
            screen_id="settings_home",
            element_id="settings-title",
            app_id="settings",
        )
        after = scene(
            "after",
            screen_id="android_home",
            element_id="launcher-icon",
            app_id="launcher",
        )
        observer = FakeSceneObserver([fresh, after])
        robot = FakeRobot()
        action = SemanticAction(
            node_id="return-to-launcher",
            action="home",
            params={"expected_effect": {"scene_changed": True, "app_id": "launcher"}},
        )

        result = self._adapter(observer, robot).execute(
            requested_action=action,
            planned_scene=planned,
            goal=goal(),
            confirmed=True,
        )

        self.assertEqual([("home",)], robot.actions)
        self.assertEqual("home", result.resolved_action.kind)
        self.assertEqual(1, result.physical_actions)
        self.assertEqual(2, observer.calls)

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

    def test_confirmed_input_executes_exact_text_once_and_reobserves(self):
        def input_scene(fingerprint, element_id):
            return UIScene(
                app_id="browser",
                screen_id="search",
                summary="输入框已聚焦",
                elements=(
                    UIElement(
                        element_id=element_id,
                        role="input",
                        meaning="搜索输入框",
                        label="搜索",
                        bounds=(0.1, 0.1, 0.9, 0.2),
                        confidence=0.96,
                        states={
                            "focused": True,
                            "value": "",
                            "keyboard_layout": "qwerty",
                            "keyboard_input_mode": "direct_latin",
                            "goal_relevant": True,
                        },
                    ),
                ),
                stable=True,
                confidence=0.95,
                fingerprint=fingerprint,
            )

        planned = input_scene("planned", "field-planned")
        fresh = input_scene("before", "field-fresh")
        after = UIScene(
            app_id="browser",
            screen_id="search",
            summary="输入框已显示新文字",
            elements=(
                UIElement(
                    element_id="field-after",
                    role="input",
                    meaning="搜索输入框",
                    label="搜索",
                    bounds=(0.1, 0.1, 0.9, 0.2),
                    confidence=0.96,
                    states={"focused": True, "value": "agent"},
                ),
            ),
            stable=True,
            confidence=0.95,
            fingerprint="after",
        )
        observer = FakeSceneObserver([fresh, after])
        robot = FakeRobot()
        action = SemanticAction(
            node_id="input-step",
            action="input_verified_text",
            params={
                "element_id": "field-planned",
                "target": "搜索输入框",
                "role": "input",
                "label": "搜索",
                "states": {"focused": True},
                "text": "agent",
            },
        )

        result = self._adapter(observer, robot).execute(
            requested_action=action,
            planned_scene=planned,
            goal=goal(),
            confirmed=True,
        )

        self.assertEqual([("input", "agent")], robot.actions)
        self.assertEqual(1, result.physical_actions)
        self.assertEqual(2, observer.calls)

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
        self.assertEqual(4, len(result.before_frames))
        self.assertIsInstance(result.after_frames, tuple)
        self.assertIsInstance(result.before_frames, tuple)
        self.assertEqual(4, len(result.before_frame_paths))
        self.assertTrue(
            all(frame.getpixel((0, 0)) == (255, 255, 255) for frame in result.after_frames)
        )
        self.assertEqual(4, len(result.after_frame_paths))
        credential = result.orientation_credential
        self.assertIsNotNone(credential)
        _validate_live_promotion_source(
            report={
                "candidate_action": "tap_semantic",
                "device_id": credential.device_id,
                "physical_actions": 1,
                "action_outcome": "matched",
                "execution": {
                    "orientation_credential": credential.to_dict(),
                },
                "before_frame_paths": list(result.before_frame_paths),
            },
            orientation_credential=credential,
            execution_result=result,
        )

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

    def test_rebind_accepts_same_navigation_class_after_model_wording_drift(self):
        planned = UIScene(
            app_id="unknown",
            screen_id="unknown",
            summary="主屏幕",
            elements=(
                UIElement(
                    element_id="browser_app_icon",
                    role="icon",
                    meaning="启动浏览器应用",
                    label="浏览器",
                    bounds=(0.12, 0.03, 0.33, 0.17),
                    confidence=0.98,
                    states={"goal_relevant": True},
                ),
            ),
            stable=True,
            confidence=0.97,
            fingerprint="planned",
        )
        fresh = UIScene(
            app_id="launcher",
            screen_id="home_screen",
            summary="主屏幕",
            elements=(
                UIElement(
                    element_id="browser_icon",
                    role="icon",
                    meaning="launch_browser_app",
                    label="浏览器",
                    bounds=(0.13, 0.03, 0.32, 0.16),
                    confidence=0.98,
                    states={"goal_relevant": True},
                ),
            ),
            stable=True,
            confidence=0.96,
            fingerprint="fresh",
        )
        after = UIScene(
            app_id="browser",
            screen_id="browser_home",
            summary="浏览器首页",
            stable=True,
            confidence=0.95,
            fingerprint="after",
        )
        robot = FakeRobot()
        result = self._adapter(
            FakeSceneObserver([fresh, after]),
            robot,
        ).execute(
            requested_action=SemanticAction(
                node_id="open-browser",
                action="tap_semantic",
                params={
                    "element_id": "browser_app_icon",
                    "target": "启动浏览器应用",
                    "role": "icon",
                    "label": "浏览器",
                    "states": {"goal_relevant": True},
                },
            ),
            planned_scene=planned,
            goal=goal(),
            confirmed=True,
        )

        self.assertEqual([("tap", 225, 95)], robot.actions)
        self.assertEqual("launch_browser_app", result.rebound_action.params["target"])

    def test_rebind_rejects_risky_meaning_drift_even_when_label_and_region_match(self):
        planned = UIScene(
            app_id="settings",
            screen_id="edit",
            summary="编辑页",
            elements=(
                UIElement(
                    element_id="return_button",
                    role="button",
                    meaning="return",
                    label="返回",
                    bounds=(0.1, 0.1, 0.3, 0.2),
                    confidence=0.96,
                    states={"enabled": True},
                ),
            ),
            fingerprint="planned",
        )
        fresh = UIScene(
            app_id="settings",
            screen_id="edit",
            summary="编辑页",
            elements=(
                UIElement(
                    element_id="save_return_button",
                    role="button",
                    meaning="save_and_return",
                    label="返回",
                    bounds=(0.1, 0.1, 0.3, 0.2),
                    confidence=0.96,
                    states={"enabled": True},
                ),
            ),
            fingerprint="fresh",
        )
        robot = FakeRobot()

        with self.assertRaisesRegex(GenericActionAdapterError, "语义"):
            self._adapter(FakeSceneObserver([fresh]), robot).execute(
                requested_action=SemanticAction(
                    node_id="return",
                    action="tap_semantic",
                    params={
                        "element_id": "return_button",
                        "target": "return",
                        "role": "button",
                        "label": "返回",
                        "states": {"enabled": True},
                    },
                ),
                planned_scene=planned,
                goal=goal(),
                confirmed=True,
            )

        self.assertEqual([], robot.actions)

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

    def test_matching_local_frames_override_model_screen_id_wording_drift(self):
        planned = scene("planned", screen_id="generic_action_verification_page")
        after = scene("after", screen_id="blue_endpoint_visible")
        observer = FakeSceneObserver([after])
        robot = FakeRobot()
        capture = SequenceCapture(["gray"] * 4 + ["white"] * 4)
        adapter = GenericSingleActionAdapter(
            capture=capture,
            observer=observer,
            robot=robot,
            frame_interval=0,
            post_action_settle=0,
        )
        action = SemanticAction(
            node_id="generic_step_1",
            action="swipe",
            params={"direction": "up", "expected_effect": {"scene_changed": True}},
        )

        result = adapter.execute(
            requested_action=action,
            planned_scene=planned,
            planned_frames=tuple(Image.new("RGB", (540, 960), "gray") for _ in range(4)),
            goal=goal(),
            confirmed=True,
        )

        self.assertEqual([("swipe", "up")], robot.actions)
        self.assertEqual(1, result.physical_actions)
        self.assertEqual(1, observer.calls)

    def test_matching_planned_frames_require_fresh_geometry_interpretation(self):
        planned = scene("planned")
        fresh = scene("fresh")
        after = scene("after", screen_id="app_home", element_id="after")
        observer = FakeSceneObserver([fresh, after])
        robot = FakeRobot()
        adapter = GenericSingleActionAdapter(
            capture=SequenceCapture(["gray"] * 4 + ["white"] * 4),
            observer=observer,
            robot=robot,
            frame_interval=0,
            post_action_settle=0,
        )

        result = adapter.execute(
            requested_action=SemanticAction(
                node_id="generic_step_1",
                action="tap_semantic",
                params={"element_id": "e1", "target": "app_icon"},
            ),
            planned_scene=planned,
            planned_frames=tuple(
                Image.new("RGB", (540, 960), "gray") for _ in range(4)
            ),
            goal=goal(),
            confirmed=True,
        )

        self.assertEqual(1, result.physical_actions)
        self.assertEqual([("tap", 300, 400)], robot.actions)
        self.assertEqual(2, observer.calls)
        self.assertEqual("fresh", result.before_scene.fingerprint)

    def test_planned_frame_identity_cannot_bypass_fresh_geometry_drift(self):
        planned = scene("planned", bounds=(0.12, 0.46, 0.58, 0.51))
        fresh = scene("fresh", bounds=(0.12, 0.225, 0.45, 0.265))
        observer = FakeSceneObserver([fresh])
        robot = FakeRobot()
        adapter = GenericSingleActionAdapter(
            capture=SequenceCapture(["gray"] * 4),
            observer=observer,
            robot=robot,
            frame_interval=0,
            post_action_settle=0,
        )

        with self.assertRaisesRegex(
            GenericActionAdapterError,
            "目标区域已明显移动",
        ):
            adapter.execute(
                requested_action=SemanticAction(
                    node_id="generic_step_1",
                    action="tap_semantic",
                    params={"element_id": "e1", "target": "app_icon"},
                ),
                planned_scene=planned,
                planned_frames=tuple(
                    Image.new("RGB", (540, 960), "gray") for _ in range(4)
                ),
                goal=goal(),
                confirmed=True,
            )

        self.assertEqual([], robot.actions)
        self.assertEqual(1, observer.calls)

    def test_drag_uses_two_independently_audited_endpoint_scenes(self):
        def drag_scene(fingerprint, source_bounds, destination_bounds):
            return UIScene(
                app_id="local.acceptance",
                screen_id="drag-board",
                summary="本地拖动验收页面",
                elements=(
                    UIElement(
                        element_id="source",
                        role="image",
                        meaning="draggable_purple_block",
                        label="起点",
                        bounds=source_bounds,
                        confidence=0.98,
                        states={"fully_visible": True},
                        evidence=("紫色圆角方块，中心写有起点二字",),
                    ),
                    UIElement(
                        element_id="destination",
                        role="container",
                        meaning="drop_target_green_zone",
                        label="绿色终点",
                        bounds=destination_bounds,
                        confidence=0.98,
                        states={"fully_visible": True},
                        evidence=("绿色虚线框区域，内部写有绿色终点",),
                    ),
                ),
                stable=True,
                confidence=0.98,
                fingerprint=fingerprint,
                camera_alignment=aligned_camera_facts(),
            )

        destination_bounds = (0.62, 0.66, 0.78, 0.82)
        planned = drag_scene(
            "planned",
            (0.12, 0.46, 0.58, 0.51),
            destination_bounds,
        )
        fresh = drag_scene(
            "fresh",
            (0.12, 0.225, 0.45, 0.265),
            destination_bounds,
        )
        audited_source = (0.18, 0.20, 0.32, 0.30)
        planned_audited = drag_scene(
            "planned",
            audited_source,
            destination_bounds,
        )
        fresh_audited = drag_scene(
            "fresh",
            audited_source,
            destination_bounds,
        )
        after = drag_scene(
            "after",
            (0.63, 0.68, 0.73, 0.78),
            destination_bounds,
        )
        observer = FakeSceneObserver(
            [fresh, after],
            geometry_scenes=[planned_audited, fresh_audited],
        )
        robot = FakeRobot()
        adapter = GenericSingleActionAdapter(
            capture=SequenceCapture(["gray"] * 4 + ["white"] * 4),
            observer=observer,
            robot=robot,
            frame_interval=0,
            post_action_settle=0,
        )

        result = adapter.execute(
            requested_action=SemanticAction(
                node_id="drag-audited",
                action="drag",
                params={
                    "source_element_id": "source",
                    "source_target": "draggable_purple_block",
                    "source_role": "image",
                    "source_label": "起点",
                    "source_states": {"fully_visible": True},
                    "destination_element_id": "destination",
                    "destination_target": "drop_target_green_zone",
                    "destination_role": "container",
                    "destination_label": "绿色终点",
                    "destination_states": {"fully_visible": True},
                    "expected_effect": {"scene_changed": True},
                },
            ),
            planned_scene=planned,
            planned_frames=tuple(
                Image.new("RGB", (540, 960), "gray") for _ in range(4)
            ),
            goal=goal(),
            confirmed=True,
        )

        self.assertEqual(
            [("source", "destination"), ("source", "destination")],
            observer.geometry_audit_calls,
        )
        self.assertEqual(
            [("drag", 250, 250, 700, 740)],
            robot.actions,
        )
        self.assertEqual(audited_source, result.before_scene.get_element("source").bounds)
        self.assertEqual(1, result.physical_actions)

    def test_drag_rebind_accepts_safe_meaning_synonyms_for_exact_labelled_endpoints(self):
        def make_scene(fingerprint, source_meaning, destination_meaning):
            return UIScene(
                app_id="local.acceptance",
                screen_id="drag-board",
                summary="通用拖动验收页面",
                elements=(
                    UIElement(
                        element_id="source",
                        role="container",
                        meaning=source_meaning,
                        label="起点",
                        bounds=(0.18, 0.68, 0.38, 0.82),
                        confidence=0.98,
                        states={"goal_relevant": True, "fully_visible": True},
                        evidence=("紧凑方块内逐字显示起点",),
                    ),
                    UIElement(
                        element_id="destination",
                        role="container",
                        meaning=destination_meaning,
                        label="绿色终点",
                        bounds=(0.55, 0.65, 0.85, 0.88),
                        confidence=0.98,
                        states={"goal_relevant": True, "fully_visible": True},
                        evidence=("绿色虚线区域内逐字显示绿色终点",),
                    ),
                ),
                stable=True,
                confidence=0.98,
                fingerprint=fingerprint,
                camera_alignment=aligned_camera_facts(),
            )

        planned = make_scene("planned", "drag_source_object", "drop_target_zone")
        fresh = make_scene("fresh", "draggable_source_block", "drag_target_region")
        after = make_scene("after", "draggable_source_block", "drag_target_region")
        moved_source = replace(
            after.elements[0],
            bounds=(0.62, 0.68, 0.78, 0.82),
        )
        after = replace(after, elements=(moved_source, after.elements[1]))
        observer = FakeSceneObserver(
            [fresh, after],
            geometry_scenes=[fresh, fresh],
        )
        robot = FakeRobot()
        adapter = GenericSingleActionAdapter(
            capture=SequenceCapture(["gray"] * 4 + ["white"] * 4),
            observer=observer,
            robot=robot,
            frame_interval=0,
            post_action_settle=0,
        )

        result = adapter.execute(
            requested_action=SemanticAction(
                node_id="drag-synonyms",
                action="drag",
                params={
                    "source_element_id": "source",
                    "source_target": "drag_source_object",
                    "source_role": "container",
                    "source_label": "起点",
                    "source_states": {"goal_relevant": True, "fully_visible": True},
                    "destination_element_id": "destination",
                    "destination_target": "drop_target_zone",
                    "destination_role": "container",
                    "destination_label": "绿色终点",
                    "destination_states": {"goal_relevant": True, "fully_visible": True},
                    "expected_effect": {"scene_changed": True},
                },
            ),
            planned_scene=planned,
            planned_frames=tuple(
                Image.new("RGB", (540, 960), "gray") for _ in range(4)
            ),
            goal=goal(),
            confirmed=True,
        )

        self.assertEqual(1, result.physical_actions)
        self.assertEqual([("drag", 280, 750, 700, 765)], robot.actions)

    def test_changed_local_frames_stop_even_when_model_screen_id_matches(self):
        planned = scene("planned", screen_id="same_screen")
        fresh = scene("before", screen_id="same_screen")
        observer = FakeSceneObserver([fresh])
        robot = FakeRobot()
        adapter = GenericSingleActionAdapter(
            capture=SequenceCapture(["white"] * 4),
            observer=observer,
            robot=robot,
            frame_interval=0,
            post_action_settle=0,
        )
        action = SemanticAction(
            node_id="generic_step_1",
            action="swipe",
            params={"direction": "up", "expected_effect": {"scene_changed": True}},
        )

        with self.assertRaisesRegex(GenericActionAdapterError, "本地真实画面已变化"):
            adapter.execute(
                requested_action=action,
                planned_scene=planned,
                planned_frames=tuple(
                    Image.new("RGB", (540, 960), "black") for _ in range(4)
                ),
                goal=goal(),
                confirmed=True,
            )

        self.assertEqual([], robot.actions)

    def test_confirmation_low_confidence_observation_retries_once_before_action(self):
        planned = scene("planned")
        fresh = scene("before", element_id="fresh")
        after = scene("after", screen_id="app_home", element_id="after")
        observer = FakeSceneObserver(
            [
                RuntimeError("页面不稳定或整体置信度不足，不能建立可信候选。"),
                fresh,
                after,
            ]
        )
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

        self.assertEqual(3, observer.calls)
        self.assertEqual(1, result.physical_actions)
        self.assertEqual(1, len(robot.actions))

    def test_confirmation_second_low_confidence_failure_stops_without_action(self):
        observer = FakeSceneObserver(
            [
                RuntimeError("页面不稳定或整体置信度不足，不能建立可信候选。"),
                RuntimeError("页面不稳定或整体置信度不足，不能建立可信候选。"),
            ]
        )
        robot = FakeRobot()
        action = SemanticAction(
            node_id="generic_step_1",
            action="tap_semantic",
            params={"element_id": "e1", "target": "app_icon"},
        )

        with self.assertRaisesRegex(GenericActionAdapterError, "第2轮动作前观察失败"):
            self._adapter(observer, robot).execute(
                requested_action=action,
                planned_scene=scene("planned"),
                goal=goal(),
                confirmed=True,
            )

        self.assertEqual(2, observer.calls)
        self.assertEqual([], robot.actions)

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
        result = self._adapter(observer, robot).execute(
            requested_action=action,
            planned_scene=planned,
            goal=goal(),
            confirmed=True,
        )
        self.assertEqual(result.physical_actions, 1)
        self.assertEqual(result.action_outcome, "mismatched")
        self.assertTrue(result.verification_errors)
        self.assertIn("没有可验证", result.verification_errors[-1])
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

    def test_confirmed_dismiss_uses_dedicated_physical_entry(self):
        planned = scene("planned", bounds=(0.1, 0.2, 0.3, 0.4))
        fresh = scene("before", element_id="fresh", bounds=(0.11, 0.21, 0.31, 0.41))
        after = scene("after", screen_id="app_home", element_id="after")
        robot = FakeRobot()
        action = SemanticAction(
            node_id="generic_step_1",
            action="dismiss_overlay",
            params={"element_id": "e1", "target": "close"},
        )

        result = self._adapter(
            FakeSceneObserver([fresh, after]),
            robot,
        ).execute(
            requested_action=action,
            planned_scene=planned,
            goal=goal(),
            confirmed=True,
        )

        self.assertEqual(result.physical_actions, 1)
        self.assertEqual(robot.actions, [("dismiss", 210, 310)])

    def test_second_capture_failure_keeps_first_semantic_mismatch(self):
        unchanged = scene("camera-noise-only", element_id="after")
        observer = FakeSceneObserver(
            [scene("before", element_id="fresh"), unchanged]
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
        self.assertEqual(len(caught.exception.verification_errors), 1)
        self.assertIn("语义变化", caught.exception.verification_errors[0])
        self.assertIn("语义变化", str(caught.exception))
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

    def test_legacy_boolean_confirmation_is_disabled_without_execution(self):
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
            with self.assertRaisesRegex(GenericActionAdapterError, "完整确认作用域"):
                session.confirm(confirmed=False)
            with self.assertRaisesRegex(GenericActionAdapterError, "完整确认作用域"):
                session.confirm(confirmed=True)
            self.assertEqual(robot.actions, [])
            self.assertEqual(session.status, "awaiting_confirmation")

    def test_legacy_boolean_confirmation_cannot_execute_terminal_scene_change(self):
        initial = scene("initial", screen_id="video_detail", app_id="douyin")
        before = scene("video-a", screen_id="video_detail", app_id="douyin")
        after = UIScene(
            app_id="douyin",
            screen_id="video_detail",
            summary="下一条公开视频可见",
            elements=(
                UIElement(
                    element_id="next-video",
                    role="image",
                    meaning="next_public_video",
                    label="下一条视频",
                    bounds=(0.05, 0.05, 0.95, 0.90),
                    confidence=0.96,
                ),
            ),
            stable=True,
            confidence=0.95,
            fingerprint="video-b",
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
            with self.assertRaisesRegex(GenericActionAdapterError, "完整确认作用域"):
                session.confirm(confirmed=True)

        self.assertEqual(session.status, "awaiting_confirmation")
        self.assertEqual(provider.calls, 0)
        self.assertEqual(robot.actions, [])

    def test_legacy_boolean_confirmation_cannot_submit_unproven_terminal_claim(self):
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
            with self.assertRaisesRegex(GenericActionAdapterError, "完整确认作用域"):
                session.confirm(confirmed=True)

        self.assertEqual(session.status, "awaiting_confirmation")
        self.assertEqual(adapter.robot.actions, [])

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

    def test_legacy_safe_loop_requires_exact_scope_and_never_executes(self):
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
            with self.assertRaisesRegex(
                GenericActionAdapterError,
                "完整确认作用域",
            ):
                session.run_safe_loop(confirmed=True)
        self.assertEqual(session.status, "awaiting_confirmation")
        self.assertEqual(robot.actions, [])

    def test_safe_loop_rejects_more_than_one_iteration_before_execution(self):
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
                session_id="auto-bounded",
                goal=goal(),
                scene=scene("initial"),
                proposal=self._proposal(),
                planner=GenericStepPlanner(FakeTextProvider({})),
                adapter=adapter,
                run_dir=Path(temp),
            )
            with self.assertRaisesRegex(
                GenericActionAdapterError,
                "每次确认只允许一个动作轮次",
            ):
                session.run_safe_loop(
                    confirmed=True,
                    confirmation={},
                    max_iterations=2,
                )

        self.assertEqual(robot.actions, [])

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
            with self.assertRaisesRegex(GenericActionAdapterError, "完整确认作用域"):
                session.run_safe_loop(confirmed=True)
        self.assertEqual(session.status, "awaiting_confirmation")
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
