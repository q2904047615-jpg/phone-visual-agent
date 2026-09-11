from __future__ import annotations
from agent.domain.ui_scene import CameraAlignmentFacts
from PIL import Image
from PIL import ImageDraw
from PIL import ImageOps
from pathlib import Path
from agent.domain.semantic_action import SemanticAction
from agent.domain.ui_scene import SystemUIFacts
from agent.domain.ui_scene import UIElement
from agent.domain.ui_scene import UIScene
from dataclasses import replace
import tempfile
import unittest
from test_support.generic_action_adapter import (
    ClickReceiptRobot,
    FakeRobot,
    FakeSceneObserver,
    GenericSingleActionAdapter,
    SequenceCapture,
    _BaseGenericActionAdapterTests,
    aligned_camera_facts,
    goal,
    scene,
    textured_phone_frame,
)


class GenericActionAdapterTests(_BaseGenericActionAdapterTests):
    def test_adapter_passes_the_device_runtime_action_set_to_qwen(self):
        class RuntimeActionObserver(FakeSceneObserver):
            supports_runtime_action_contract = True

            def __init__(self, scenes):
                super().__init__(scenes)
                self.available_action_sets = []

            def observe_with_decision(
                self,
                *,
                frames,
                goal_context=None,
                available_action_kinds=None,
            ):
                self.available_action_sets.append(frozenset(available_action_kinds or ()))
                return super().observe_with_decision(frames=frames, goal_context=goal_context)

        observer = RuntimeActionObserver([scene("fresh")])
        adapter = self._adapter(observer, FakeRobot())

        adapter.capture_scene(
            goal(),
            evidence_dir=None,
            prefix="before_step_1",
        )

        self.assertEqual(
            [adapter.supported_action_kinds()],
            observer.available_action_sets,
        )

    def test_reports_only_callable_device_actions(self):
        adapter = self._adapter(FakeSceneObserver([]), FakeRobot())

        supported = adapter.supported_action_kinds()

        self.assertIn("tap_semantic", supported)
        self.assertIn("reveal_system_navigation", supported)
        self.assertIn("input_verified_text", supported)
        self.assertIn("press_enter", supported)
        self.assertIn("clear_verified_text", supported)
        self.assertIn("long_press", supported)
        self.assertIn("drag", supported)
        self.assertIn("back", supported)
        self.assertIn("home", supported)
        self.assertIn("wait_for_change", supported)
        self.assertIn("scroll", supported)
        self.assertIn("swipe_element", supported)

    def test_optional_camera_alignment_metadata_does_not_veto_ordinary_action(self):
        cases = (
            ("omitted", CameraAlignmentFacts()),
            (
                "low_model_confidence",
                CameraAlignmentFacts(
                    camera_layout_orientation="portrait",
                    phone_content_rotation="upright",
                    confidence=0.01,
                    evidence=("模型仅提供诊断性方向描述",),
                ),
            ),
            (
                "model_layout_mismatch",
                CameraAlignmentFacts(
                    camera_layout_orientation="landscape",
                    phone_content_rotation="unknown",
                    confidence=0.0,
                    evidence=(),
                ),
            ),
            (
                "model_reports_rotated_content",
                CameraAlignmentFacts(
                    camera_layout_orientation="portrait",
                    phone_content_rotation="rotated_90",
                    confidence=0.99,
                    evidence=("模型诊断为旋转画面",),
                ),
            ),
        )
        frames = tuple(Image.new("RGB", (540, 960), "gray") for _ in range(4))

        class PreserveOptionalAlignmentObserver(FakeSceneObserver):
            def _next_scene(self, *, goal_context=None):
                self.calls += 1
                self.goal_contexts.append(goal_context)
                result = self.scenes.pop(0)
                if isinstance(result, BaseException):
                    raise result
                return result

        for label, alignment in cases:
            with self.subTest(label=label):
                planned = scene("planned", camera_alignment=alignment)
                after = scene("after", screen_id="app_home", element_id="after")
                robot = FakeRobot()
                adapter = GenericSingleActionAdapter(
                    capture=SequenceCapture(["gray"] * 4 + ["white"] * 4),
                    observer=PreserveOptionalAlignmentObserver([after]),
                    robot=robot,
                    frame_interval=0,
                    post_action_settle=0,
                )

                result = adapter.execute(
                    requested_action=SemanticAction(
                        node_id="ordinary-tap",
                        action="tap_semantic",
                        params={"tap_point": (0.3, 0.4), "element_id": "e1", "target": "app_icon"},
                    ),
                    planned_scene=planned,
                    planned_frames=frames,
                    goal=goal(),
                    confirmed=True,
                )

                self.assertEqual(1, result.physical_actions)
                self.assertEqual([("tap", 300, 400)], robot.actions)
                self.assertEqual("portrait", result.orientation_credential.camera_layout_orientation)
                self.assertEqual(1.0, result.orientation_credential.confidence)

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

    def test_reveal_system_navigation_leaves_summary_only_result_to_qwen_without_fallback(self):
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
                    params={},
                ),
                planned_scene=planned,
                planned_frames=tuple(
                    Image.new("RGB", (540, 960), "gray") for _ in range(4)
                ),
                goal=goal(),
                confirmed=True,
            )

        self.assertEqual([("reveal_system_navigation",)], robot.actions)
        self.assertEqual("finish", result.after_model_decision["status"])
        self.assertEqual((), result.controller_transition_evidence)

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
            params={},
        )

        prepared_1_adapter = self._adapter(observer, robot)
        prepared_1_goal = goal()
        prepared_1_scene, prepared_1_frames, _, _ = prepared_1_adapter.capture_scene(
            prepared_1_goal, prefix="test_orchestrator_before", evidence_dir=None)
        result = prepared_1_adapter.execute(
            requested_action=action,
            planned_scene=prepared_1_scene, planned_frames=prepared_1_frames,
            goal=prepared_1_goal,
            confirmed=True,
        )

        self.assertEqual([("home",)], robot.actions)
        self.assertEqual("home", result.resolved_action.kind)
        self.assertEqual(1, result.physical_actions)
        self.assertEqual(2, observer.calls)
        self.assertEqual(
            "single_step_scene_orientation",
            result.orientation_credential.source,
        )

    def test_home_records_single_click_transport_receipt(self):
        planned = scene("planned", screen_id="settings_home", app_id="settings")
        fresh = scene("before", screen_id="settings_home", app_id="settings")
        after = scene("after", screen_id="android_home", app_id="launcher")
        observer = FakeSceneObserver([fresh, after])
        robot = ClickReceiptRobot()

        prepared_3_adapter = self._adapter(observer, robot)
        prepared_3_goal = goal()
        prepared_3_scene, prepared_3_frames, _, _ = prepared_3_adapter.capture_scene(
            prepared_3_goal, prefix="test_orchestrator_before", evidence_dir=None)
        result = prepared_3_adapter.execute(
            requested_action=SemanticAction(
                node_id="return-to-launcher",
                action="home",
                params={},
            ),
            planned_scene=prepared_3_scene, planned_frames=prepared_3_frames,
            goal=prepared_3_goal,
            confirmed=True,
        )

        self.assertEqual([("home",)], robot.actions)
        self.assertTrue(result.hardware_receipt["input_events_dispatched"])
        self.assertFalse(result.hardware_receipt["mechanical_contact_ack"])
        self.assertIsNone(robot.consume_last_click_receipt())

    def test_confirmed_tap_executes_exactly_once_and_reobserves(self):
        planned = scene("planned", bounds=(0.1, 0.2, 0.3, 0.4))
        fresh = scene("before", bounds=(0.11, 0.21, 0.31, 0.41))
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
                "tap_point": (0.21, 0.31),
            },
        )
        with tempfile.TemporaryDirectory() as temp:
            prepared_17_adapter = self._adapter(observer, robot)
            prepared_17_goal = goal()
            prepared_17_scene, prepared_17_frames, _, _ = prepared_17_adapter.capture_scene(
                prepared_17_goal, evidence_dir=Path(temp), prefix="test_orchestrator_before")
            result = prepared_17_adapter.execute(
                requested_action=action,
                planned_scene=prepared_17_scene, planned_frames=prepared_17_frames,
                goal=prepared_17_goal,
                confirmed=True,
                evidence_dir=Path(temp),
            )
        self.assertEqual(robot.actions, [("tap", 210, 310)])
        self.assertEqual(result.physical_actions, 1)
        self.assertEqual(observer.calls, 2)

    def test_changed_non_authoritative_state_does_not_reselect_or_veto(self):
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
                    element_id="e1",
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
        prepared_11_adapter = self._adapter(
            FakeSceneObserver([fresh, scene("after", screen_id="app_home")]),
            robot,
        )
        prepared_11_goal = goal()
        prepared_11_scene, prepared_11_frames, _, _ = prepared_11_adapter.capture_scene(
            prepared_11_goal, prefix="test_orchestrator_before", evidence_dir=None)
        result = prepared_11_adapter.execute(
            requested_action=SemanticAction(
                node_id="generic_step_1",
                action="tap_semantic",
                params={"element_id": "e1", "target": "app_icon", "tap_point": (0.3, 0.4)},
            ),
            planned_scene=prepared_11_scene, planned_frames=prepared_11_frames,
            goal=prepared_11_goal,
            confirmed=True,
        )
        self.assertEqual([("tap", 300, 400)], robot.actions)
        self.assertEqual("executed", result.action_outcome)

    def test_drag_reuses_single_step_endpoint_geometry(self):
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
        observer = FakeSceneObserver([after])
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
            [("drag", 350, 485, 700, 740)],
            robot.actions,
        )
        self.assertEqual(
            planned.get_element("source").bounds,
            result.before_scene.get_element("source").bounds,
        )
        self.assertEqual(1, result.physical_actions)

    def test_long_press_reuses_single_step_target_geometry(self):
        states = {"goal_relevant": True, "fully_visible": True}

        def long_press_scene(
            fingerprint,
            bounds,
            *,
            meaning="long_press_target_area",
            passed=False,
        ):
            return UIScene(
                app_id="local.acceptance",
                screen_id="long-press-board",
                summary="本地长按验收页面",
                elements=(
                    UIElement(
                        element_id="target",
                        role="button",
                        meaning=meaning,
                        label="长按我 · 不要移动",
                        bounds=bounds,
                        confidence=1.0,
                        states=states,
                        evidence=("黄色虚线框内逐字显示长按目标",),
                    ),
                ),
                overlays=(("long_press 验收通过",) if passed else ()),
                stable=True,
                confidence=1.0,
                fingerprint=fingerprint,
                camera_alignment=aligned_camera_facts(),
            )

        planned = long_press_scene("planned", (0.09, 0.53, 0.91, 0.76))
        fresh = long_press_scene(
            "fresh",
            (0.09, 0.47, 0.91, 0.69),
            meaning="interaction_zone",
        )
        audited_bounds = (0.10, 0.48, 0.90, 0.70)
        planned_audited = long_press_scene("planned", audited_bounds)
        fresh_audited = long_press_scene(
            "fresh",
            audited_bounds,
            meaning="interaction_zone",
        )
        after = long_press_scene(
            "after",
            audited_bounds,
            meaning="interaction_zone",
            passed=True,
        )
        observer = FakeSceneObserver([after])
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
                node_id="long-press-audited",
                action="long_press",
                params={
                    "element_id": "target",
                    "target": "long_press_target",
                    "role": "button",
                    "label": "长按我 · 不要移动",
                    "states": states,
                    "duration_ms": 800,
                    "tap_point": (0.5, 0.645),

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
            [("long_press", 500, 645, 0.8)],
            robot.actions,
        )
        self.assertEqual(
            planned.get_element("target").bounds,
            result.before_scene.get_element("target").bounds,
        )
        self.assertEqual(1, result.physical_actions)

    def test_double_tap_crosses_adapter_as_one_canonical_action(self):
        states = {"goal_relevant": True, "fully_visible": True}
        target = UIElement(
            element_id="target",
            role="list_item",
            meaning="image_preview",
            label="预览图",
            bounds=(0.20, 0.30, 0.80, 0.70),
            confidence=1.0,
            states=states,
            evidence=("预览图完整可见",),
        )
        planned = UIScene(
            app_id="sample.app",
            screen_id="preview-list",
            summary="预览列表",
            elements=(target,),
            stable=True,
            confidence=1.0,
            fingerprint="planned",
            camera_alignment=aligned_camera_facts(),
        )
        after = replace(
            planned,
            screen_id="preview-detail",
            overlays=("预览已打开",),
            fingerprint="after",
        )
        robot = FakeRobot()
        result = GenericSingleActionAdapter(
            capture=SequenceCapture(["gray"] * 4),
            observer=FakeSceneObserver([after]),
            robot=robot,
            frame_interval=0,
            post_action_settle=0,
        ).execute(
            requested_action=SemanticAction(
                node_id="double-preview",
                action="double_tap",
                params={
                    "element_id": "target",
                    "target": "image_preview",
                    "role": "list_item",
                    "label": "预览图",
                    "states": states,
                    "tap_point": (0.5, 0.5),

                },
            ),
            planned_scene=planned,
            planned_frames=tuple(
                Image.new("RGB", (540, 960), "gray") for _ in range(4)
            ),
            goal=goal(),
            confirmed=True,
        )

        self.assertEqual([("double_tap", 500, 500)], robot.actions)
        self.assertEqual(1, result.physical_actions)
        self.assertEqual(2, result.hardware_receipt["click_count"])

    def test_drag_rebind_reuses_exact_endpoint_ids_despite_wording_drift(self):
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
        observer = FakeSceneObserver([after])
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

    def test_goal_polluted_home_app_is_normalized_before_confirmation(self):
        planned = scene("planned", app_id="douyin")
        fresh = scene("before", app_id="unknown")
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
            params={"tap_point": (0.3, 0.4), "element_id": "e1", "target": "app_icon"},
        )
        prepared_12_adapter = self._adapter(observer, robot)
        prepared_12_goal = goal()
        prepared_12_scene, prepared_12_frames, _, _ = prepared_12_adapter.capture_scene(
            prepared_12_goal, prefix="test_orchestrator_before", evidence_dir=None)
        result = prepared_12_adapter.execute(
            requested_action=action,
            planned_scene=prepared_12_scene, planned_frames=prepared_12_frames,
            goal=prepared_12_goal,
            confirmed=True,
        )
        self.assertEqual(planned.foreground_app_id, "launcher")
        self.assertEqual(fresh.foreground_app_id, "launcher")
        self.assertEqual(result.physical_actions, 1)
        self.assertEqual(len(robot.actions), 1)

    def test_ordinary_unchanged_screen_is_a_valid_fresh_receipt(self):
        planned = scene("same")
        fresh = scene("same")
        unchanged = scene("same", element_id="after")
        observer = FakeSceneObserver([fresh, unchanged, unchanged])
        robot = FakeRobot()
        action = SemanticAction(
            node_id="generic_step_1",
            action="tap_semantic",
            params={"tap_point": (0.3, 0.4), "element_id": "e1", "target": "app_icon"},
        )
        prepared_13_adapter = self._adapter(observer, robot)
        prepared_13_goal = goal()
        prepared_13_scene, prepared_13_frames, _, _ = prepared_13_adapter.capture_scene(
            prepared_13_goal, prefix="test_orchestrator_before", evidence_dir=None)
        result = prepared_13_adapter.execute(
            requested_action=action,
            planned_scene=prepared_13_scene, planned_frames=prepared_13_frames,
            goal=prepared_13_goal,
            confirmed=True,
        )
        self.assertEqual(result.physical_actions, 1)
        self.assertEqual(result.action_outcome, "executed")
        self.assertEqual(result.verification_errors, ())
        self.assertEqual(observer.calls, 2)
        self.assertEqual(len(robot.actions), 1)

    def test_continuous_action_does_not_reject_changed_background(self):
        reference = textured_phone_frame()
        off_phone = ImageOps.invert(reference)
        recovered = reference.copy()
        ImageDraw.Draw(recovered).rectangle(
            (180, 300, 360, 340),
            fill="white",
            outline="black",
        )
        capture = SequenceCapture([off_phone] * 4 + [recovered] * 4)
        adapter = GenericSingleActionAdapter(
            capture=capture,
            observer=FakeSceneObserver([]),
            robot=FakeRobot(),
            frame_interval=0,
            post_action_settle=0,
            post_action_timeout=1,
            post_action_continuous_timeout=45,
        )

        frames, _paths = adapter._capture_stable_post_action_frames(
            deadline=10**9,
            evidence_dir=None,
            prefix="camera_returns",
            clarity_reference_frames=tuple(reference.copy() for _ in range(4)),
            require_relative_clarity=True,
        )

        self.assertEqual(4, capture.calls)
        self.assertEqual(4, len(frames))
        self.assertEqual(off_phone.tobytes(), frames[-1].tobytes())

    def test_confirmed_dismiss_uses_dedicated_physical_entry(self):
        planned = scene("planned", bounds=(0.1, 0.2, 0.3, 0.4))
        fresh = scene("before", bounds=(0.11, 0.21, 0.31, 0.41))
        after = scene("after", screen_id="app_home", element_id="after")
        robot = FakeRobot()
        action = SemanticAction(
            node_id="generic_step_1",
            action="dismiss_overlay",
            params={"element_id": "e1", "target": "close", "tap_point": (0.21, 0.31)},
        )

        prepared_14_adapter = self._adapter(
            FakeSceneObserver([fresh, after]),
            robot,
        )
        prepared_14_goal = goal()
        prepared_14_scene, prepared_14_frames, _, _ = prepared_14_adapter.capture_scene(
            prepared_14_goal, prefix="test_orchestrator_before", evidence_dir=None)
        result = prepared_14_adapter.execute(
            requested_action=action,
            planned_scene=prepared_14_scene, planned_frames=prepared_14_frames,
            goal=prepared_14_goal,
            confirmed=True,
        )

        self.assertEqual(result.physical_actions, 1)
        self.assertEqual(robot.actions, [("dismiss", 210, 310)])


if __name__ == "__main__":
    unittest.main()
