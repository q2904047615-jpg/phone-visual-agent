from __future__ import annotations
from agent.domain.ui_scene import CameraAlignmentFacts
from agent.domain.confirmation_authority import ConfirmationAuthority
from agent.application.action_adapter import GenericActionAdapterError
from agent.domain.generic_goal import GenericIntentDraft
from PIL import Image
from PIL import ImageFilter
from pathlib import Path
from agent.domain.universal_action_controller import ResolvedSemanticAction
from agent.domain.semantic_action import SemanticAction
from types import SimpleNamespace
from agent.domain.ui_scene import UIElement
from agent.domain.ui_scene import UIScene
from agent.domain.vision_model import VisionAgentError
from agent.infrastructure.generic_action_adapter import GenericSingleActionAdapter as _GenericSingleActionAdapter
from agent.infrastructure.capability_acceptance import _validate_live_promotion_source
from agent.domain.validation import canonical_digest
import hashlib
import json
from agent.infrastructure.observation_images import local_frame_fingerprint
from unittest.mock import patch
from dataclasses import replace
import tempfile
import unittest
from test_support.generic_action_adapter import (
    ClickReceiptRobot,
    FakeRobot,
    FakeSceneObserver,
    GenericSingleActionAdapter,
    PhysicalGateCanvasErrorRobot,
    RawFailureSceneObserver,
    RuntimeActionRecordingObserver,
    SecondPostCaptureFailureAdapter,
    SequenceCapture,
    _BaseGenericActionAdapterTests,
    aligned_camera_facts,
    compact_scene_raw,
    goal,
    scene,
    textured_phone_frame,
)


class GenericActionAdapterTests(_BaseGenericActionAdapterTests):
    def test_post_action_goal_appends_actual_action_without_mutating_history(self) -> None:
        goal = GenericIntentDraft(understood=True, app_id="messenger", app_name="消息工具",
            objective="输入新文字", entities={"history": [{"action": "home"}]})
        projected = _GenericSingleActionAdapter._post_action_goal(goal, authority=SimpleNamespace(revision=1),
            requested=SemanticAction(node_id='node-1', action='input_verified_text', params={'text': '新文字'}),
            resolved=ResolvedSemanticAction(node_id="node-1", kind="input_verified_text"), physical_actions=1)
        self.assertEqual(1, len(goal.entities["history"]))
        self.assertEqual(2, len(projected.entities["history"]))
        receipt = projected.entities["history"][-1]
        self.assertEqual("executed", receipt["transport_outcome"])
        self.assertEqual("input_verified_text", receipt["action"]["kind"])
        self.assertIsNone(receipt["visual_outcome"])

    def test_effect_finish_rejects_unchanged_raw_frames_after_one_action(self) -> None:
        frame = textured_phone_frame()
        fingerprint = local_frame_fingerprint(frame)
        planned = scene(fingerprint)
        target = planned.elements[0]
        action = SemanticAction(node_id='effect-without-visual-transition', action='tap_semantic', params={"tap_point": (0.3, 0.4),
            'element_id': target.element_id,
            'target': target.meaning,
            'role': target.role,
            'label': target.label,
            'states': dict(target.states),
        })
        authority = ConfirmationAuthority(session_id='session-effect', task_id='task-effect',
            device_id='test-device', revision=1, step_id='effect-subgoal', effect_ids=('effect-1',),
            observation_id='observation-effect', fingerprint=fingerprint, decision_node_id=action.node_id,
            action_digest=canonical_digest(action.to_dict()), consumed=True)
        robot = FakeRobot()
        adapter = GenericSingleActionAdapter(
            capture=SequenceCapture([frame.copy() for _ in range(8)]),
            observer=FakeSceneObserver([replace(planned, fingerprint='after-same-pixels')]),
            robot=robot,
            frame_interval=0,
            post_action_settle=0,
        )

        with self.assertRaisesRegex(GenericActionAdapterError, '真实帧没有可归因的新变化') as raised:
            adapter.execute(requested_action=action, planned_scene=planned,
                planned_frames=tuple(frame.copy() for _ in range(4)), goal=goal(), confirmed=True,
                action_authority=authority)

        self.assertEqual(1, raised.exception.physical_actions)
        self.assertEqual([('tap', 300, 400)], robot.actions)
        credential = raised.exception.execution_metadata['effect_visual_transition']
        self.assertFalse(credential['material'])
        self.assertLess(credential['max_tile_median_delta'], credential['minimum_tile_delta'])

    def test_capture_scene_forwards_current_subgoal_action_scope_to_qwen(self) -> None:
        observer = RuntimeActionRecordingObserver([scene("scoped")])
        adapter = self._adapter(observer, FakeRobot())
        scoped = frozenset({"tap_semantic", "home", "back"})

        adapter.capture_scene(
            goal(),
            evidence_dir=None,
            prefix="scoped-actions",
            available_action_kinds=scoped,
        )

        self.assertEqual([scoped], observer.available_action_sets)

    def test_production_execute_requires_the_current_qwen_frames(self) -> None:
        robot = FakeRobot()
        adapter = _GenericSingleActionAdapter(
            capture=lambda: Image.new("RGB", (540, 960), "gray"),
            observer=FakeSceneObserver([]),
            robot=robot,
            device_id="test-device",
            frame_interval=0,
            post_action_settle=0,
        )

        with self.assertRaisesRegex(GenericActionAdapterError, "产生该 Qwen 动作的当前截图帧"):
            adapter.execute(
                requested_action=SemanticAction(
                    node_id="must-bind-current-frame",
                    action="tap_semantic",
                    params={"tap_point": (0.3, 0.4), "element_id": "e1", "target": "app_icon"},
                ),
                planned_scene=scene("planned"),
                goal=goal(),
                confirmed=True,
            )

        self.assertEqual([], robot.actions)

    def test_post_action_scene_is_never_rewritten_from_transport_expectations(self):
        self.assertFalse(hasattr(GenericSingleActionAdapter, "_primary_input_confirmation_reusable"))
        self.assertFalse(hasattr(GenericSingleActionAdapter, "_rebind_action"))
        self.assertFalse(hasattr(GenericSingleActionAdapter, "_reconcile_literal_key_visual_wrap"))
        self.assertFalse(hasattr(GenericSingleActionAdapter, "_reconcile_verified_text_horizontal_suffix"))

        prior = "releasecandidate"
        fragment = "continuation"
        observed_suffix = "didatecontinuation"
        before = self._literal_input_scene("before", value=prior, include_key=False)
        after = self._literal_input_scene("after", value=observed_suffix, include_key=False)
        resolved = ResolvedSemanticAction(
            node_id="no-post-rewrite",
            kind="input_verified_text",
            text=prior + fragment,
            input_fragment=fragment,
            prior_input_value=prior,
            expected_input_value=prior + fragment,
            target_element_id="local_audited_input_1",
            before_fingerprint=before.fingerprint,
        )
        adapter = self._adapter(FakeSceneObserver([after]), FakeRobot())
        stable_frames = [Image.new("RGB", (540, 960), "white") for _ in range(4)]

        with patch.object(
            adapter,
            "_capture_stable_post_action_frames",
            return_value=(stable_frames, ()),
        ), self.assertRaisesRegex(GenericActionAdapterError, "文字不匹配"):
            adapter._observe_stable_post_action_scene(
                goal(),
                before=before,
                before_frames=tuple(stable_frames),
                resolved=resolved,
                evidence_dir=None,
                evidence_prefix="no-post-rewrite",
            )

        self.assertEqual(
            observed_suffix,
            after.get_element("local_audited_input_1").states["value"],
        )

    def test_failed_observation_persists_bounded_redacted_qwen_response(self):
        raw = (
            '{"api_key":"secret-api-value","Authorization":"Bearer bearer-value",'
            '"image":"data:image/jpeg;base64,QUJDREVGRw==",'
            '"url":"https://example.test/path?token=query-secret",'
            '"text":"visible"} password=plain-secret '
            + ("x" * 17000)
        )
        observer = RawFailureSceneObserver(raw)
        with tempfile.TemporaryDirectory() as temp:
            evidence_dir = Path(temp)
            with self.assertRaises(GenericActionAdapterError) as caught:
                self._adapter(observer, FakeRobot()).capture_scene(
                    goal(),
                    evidence_dir=evidence_dir,
                    prefix="before_step_1",
                )
            diagnostic_paths = [
                Path(path)
                for path in caught.exception.evidence
                if path.endswith("_qwen_failure.json")
            ]
            self.assertEqual(1, len(diagnostic_paths))
            artifact = json.loads(diagnostic_paths[0].read_text(encoding="utf-8"))

        self.assertEqual("parsing_targeted_refinement", artifact["failed_stage"])
        self.assertEqual("schema_validation", artifact["error_type"])
        self.assertEqual(len(raw), artifact["raw_response_length"])
        self.assertEqual(
            hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            artifact["raw_response_sha256"],
        )
        self.assertTrue(artifact["redacted_response_truncated"])
        self.assertLessEqual(len(artifact["redacted_raw_response"]), 16000)
        serialized = json.dumps(artifact, ensure_ascii=False)
        for secret in (
            "secret-api-value",
            "bearer-value",
            "query-secret",
            "plain-secret",
            "QUJDREVGRw==",
        ):
            self.assertNotIn(secret, serialized)
        self.assertNotIn("data:image", serialized)
        self.assertIn("[REDACTED_SECRET]", serialized)
        self.assertIn("[REDACTED_IMAGE_DATA_URL]", serialized)

    def test_successful_observation_does_not_write_qwen_failure_artifact(self):
        with tempfile.TemporaryDirectory() as temp:
            evidence_dir = Path(temp)
            self._adapter(FakeSceneObserver([scene("fresh")]), FakeRobot()).capture_scene(
                goal(),
                evidence_dir=evidence_dir,
                prefix="before_step_1",
            )
            self.assertEqual([], list(evidence_dir.glob("*_qwen_failure.json")))

    def test_scene_only_observer_cannot_reenter_the_runtime_contract(self):
        class ObsoleteSceneOnlyObserver:
            def observe(self, *, frames, goal_context=None):
                del frames, goal_context
                return scene("obsolete-scene-only")

        with self.assertRaisesRegex(
            GenericActionAdapterError,
            r"不支持同一截图响应中的 scene \+ decision 合同",
        ):
            self._adapter(ObsoleteSceneOnlyObserver(), FakeRobot()).capture_scene(
                goal(),
                evidence_dir=None,
                prefix="before_step_1",
            )

    def test_diagnostic_write_failure_preserves_primary_observation_error(self):
        observer = RawFailureSceneObserver('{"broken":true}')
        with tempfile.TemporaryDirectory() as temp, patch(
            "agent.infrastructure.generic_action_adapter._persist_qwen_failure_diagnostic",
            side_effect=OSError("disk unavailable"),
        ):
            with self.assertRaisesRegex(
                GenericActionAdapterError, "目标精查严格协议拒绝"
            ):
                self._adapter(observer, FakeRobot()).capture_scene(
                    goal(),
                    evidence_dir=Path(temp),
                    prefix="before_step_1",
                )

        self.assertEqual("OSError", observer.last_diagnostics["diagnostic_persistence_error"])

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

    def test_fixed_system_home_uses_frame_binding_when_app_content_is_hidden(self):
        class ExactAlignmentObserver(FakeSceneObserver):
            def _next_scene(self, *, goal_context=None):
                self.calls += 1
                self.goal_contexts.append(goal_context)
                result = self.scenes.pop(0)
                if isinstance(result, BaseException):
                    raise result
                return result

        hidden_content_alignment = CameraAlignmentFacts(
            camera_layout_orientation="portrait",
            phone_content_rotation="unknown",
            confidence=0.95,
            evidence=("中央App内容被隐私遮罩，当前完整画布保持稳定",),
        )
        planned = scene(
            "planned-hidden-content",
            screen_id="unknown",
            app_id="unknown",
            camera_alignment=hidden_content_alignment,
        )
        fresh = scene(
            "fresh-hidden-content",
            screen_id="unknown",
            app_id="unknown",
            camera_alignment=hidden_content_alignment,
        )
        after = scene(
            "launcher-after-fixed-home",
            screen_id="android_home",
            app_id="launcher",
        )
        observer = ExactAlignmentObserver([fresh, after])
        robot = FakeRobot()

        prepared_2_adapter = self._adapter(observer, robot)
        prepared_2_goal = goal()
        prepared_2_scene, prepared_2_frames, _, _ = prepared_2_adapter.capture_scene(
            prepared_2_goal, prefix="test_orchestrator_before", evidence_dir=None)
        result = prepared_2_adapter.execute(
            requested_action=SemanticAction(
                node_id="fixed-system-home",
                action="home",
                params={

                },
            ),
            planned_scene=prepared_2_scene, planned_frames=prepared_2_frames,
            goal=prepared_2_goal,
            confirmed=True,
        )

        self.assertEqual([("home",)], robot.actions)
        self.assertEqual(1, result.physical_actions)
        self.assertEqual(
            "unknown",
            result.orientation_credential.phone_content_rotation,
        )
        self.assertEqual(2, observer.calls)

    def test_invalid_home_click_receipt_stops_before_fresh_screenshot_observation(self):
        planned = scene("planned", screen_id="settings_home", app_id="settings")
        fresh = scene("before", screen_id="settings_home", app_id="settings")
        observer = FakeSceneObserver([fresh])
        robot = ClickReceiptRobot(valid=False)

        with self.assertRaisesRegex(GenericActionAdapterError, "单击事件派发凭据"):
            prepared_16_adapter = self._adapter(observer, robot)
            prepared_16_goal = goal()
            prepared_16_scene, prepared_16_frames, _, _ = prepared_16_adapter.capture_scene(
                prepared_16_goal, prefix="test_orchestrator_before", evidence_dir=None)
            prepared_16_adapter.execute(
                requested_action=SemanticAction(
                    node_id="return-to-launcher",
                    action="home",
                    params={},
                ),
                planned_scene=prepared_16_scene, planned_frames=prepared_16_frames,
                goal=prepared_16_goal,
                confirmed=True,
            )

        self.assertEqual([("home",)], robot.actions)
        self.assertEqual(1, observer.calls)

    def test_execution_result_keeps_exact_four_verified_after_frames(self):
        gray = Image.new("RGB", (540, 960), "gray")
        white = Image.new("RGB", (540, 960), "white")
        before_fingerprint = local_frame_fingerprint(gray)
        after_fingerprint = local_frame_fingerprint(white)
        planned = scene(before_fingerprint)
        after = scene(
            after_fingerprint,
            screen_id="app_home",
            element_id="after",
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
        action = SemanticAction(
            node_id="generic_step_1",
            action="tap_semantic",
            params={"tap_point": (0.3, 0.4), "element_id": "e1", "target": "app_icon"},
        )

        with tempfile.TemporaryDirectory() as temp:
            result = adapter.execute(
                requested_action=action,
                planned_scene=planned,
                planned_frames=(gray, gray.copy(), gray.copy(), gray.copy()),
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
                "action_outcome": "executed",
                "execution": result.to_dict(),
                "before_frame_paths": list(result.before_frame_paths),
            },
            orientation_credential=credential,
            execution_result=result,
        )

    def test_after_frame_fingerprint_matches_after_scene(self):
        gray = Image.new("RGB", (540, 960), "gray")
        white = Image.new("RGB", (540, 960), "white")
        before_fingerprint = local_frame_fingerprint(gray)
        after_fingerprint = local_frame_fingerprint(white)
        adapter = GenericSingleActionAdapter(
            capture=SequenceCapture(["gray"] * 4 + ["white"] * 4),
            observer=FakeSceneObserver(
                [scene(after_fingerprint, screen_id="app_home", element_id="after")]
            ),
            robot=FakeRobot(),
            frame_interval=0,
            post_action_settle=0,
        )
        action = SemanticAction(
            node_id="generic_step_1",
            action="tap_semantic",
            params={"tap_point": (0.3, 0.4), "element_id": "e1", "target": "app_icon"},
        )

        result = adapter.execute(
            requested_action=action,
            planned_scene=scene(before_fingerprint),
            planned_frames=(gray, gray.copy(), gray.copy(), gray.copy()),
            goal=goal(),
            confirmed=True,
        )

        selected_fingerprint = local_frame_fingerprint(result.after_frames[0])
        self.assertEqual(result.after_scene.fingerprint, selected_fingerprint)

    def test_each_execution_uses_unique_evidence_paths(self):
        action = SemanticAction(
            node_id="generic_step_1",
            action="tap_semantic",
            params={"tap_point": (0.3, 0.4), "element_id": "e1", "target": "app_icon"},
        )
        with tempfile.TemporaryDirectory() as temp:
            evidence_dir = Path(temp)
            prepared_19_adapter = self._adapter(
                FakeSceneObserver(
                    [scene("same"), scene("after-1", screen_id="app_home")]
                ),
                FakeRobot(),
            )
            prepared_19_goal = goal()
            prepared_19_scene, prepared_19_frames, _, _ = prepared_19_adapter.capture_scene(
                prepared_19_goal, evidence_dir=evidence_dir, prefix="test_orchestrator_before")
            first = prepared_19_adapter.execute(
                requested_action=action,
                planned_scene=prepared_19_scene, planned_frames=prepared_19_frames,
                goal=prepared_19_goal,
                confirmed=True,
                evidence_dir=evidence_dir,
            )
            first_bytes = {path: Path(path).read_bytes() for path in first.evidence}
            prepared_20_adapter = self._adapter(
                FakeSceneObserver(
                    [scene("same"), scene("after-2", screen_id="app_home")]
                ),
                FakeRobot(),
            )
            prepared_20_goal = goal()
            prepared_20_scene, prepared_20_frames, _, _ = prepared_20_adapter.capture_scene(
                prepared_20_goal, evidence_dir=evidence_dir, prefix="test_orchestrator_before")
            second = prepared_20_adapter.execute(
                requested_action=action,
                planned_scene=prepared_20_scene, planned_frames=prepared_20_frames,
                goal=prepared_20_goal,
                confirmed=True,
                evidence_dir=evidence_dir,
            )

            self.assertTrue(set(first.evidence).isdisjoint(second.evidence))
            self.assertEqual(
                first_bytes,
                {path: Path(path).read_bytes() for path in first.evidence},
            )

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
            action="scroll",
            params={"direction": "up", },
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

    def test_physical_gate_canvas_error_preserves_zero_action_and_evidence(self):
        planned = scene("planned", screen_id="generic_action_verification_page")
        observer = FakeSceneObserver([])
        robot = PhysicalGateCanvasErrorRobot()
        adapter = GenericSingleActionAdapter(
            capture=SequenceCapture(["gray"] * 4),
            observer=observer,
            robot=robot,
            frame_interval=0,
            post_action_settle=0,
        )
        action = SemanticAction(
            node_id="generic_step_1",
            action="scroll",
            params={"direction": "up", },
        )

        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(
                GenericActionAdapterError,
                "共享物理执行门.*画布尺寸不匹配",
            ) as caught:
                adapter.execute(
                    requested_action=action,
                    planned_scene=planned,
                    planned_frames=tuple(
                        Image.new("RGB", (540, 960), "gray") for _ in range(4)
                    ),
                    goal=goal(),
                    confirmed=True,
                    evidence_dir=Path(temp),
                )
            self.assertEqual(4, len(caught.exception.evidence))
            self.assertTrue(all(Path(item).is_file() for item in caught.exception.evidence))
        self.assertEqual(0, caught.exception.physical_actions)
        self.assertEqual([], robot.actions)

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
                params={"tap_point": (0.3, 0.4), "element_id": "e1", "target": "app_icon"},
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
        self.assertEqual(1, observer.calls)
        self.assertEqual("planned", result.before_scene.fingerprint)

    def test_matching_frame_identity_reuses_original_geometry(self):
        planned = scene("planned", bounds=(0.12, 0.46, 0.58, 0.51))
        fresh = scene("fresh", bounds=(0.12, 0.225, 0.45, 0.265))
        observer = FakeSceneObserver(
            [scene("after", screen_id="app_home", element_id="after")]
        )
        robot = FakeRobot()
        adapter = GenericSingleActionAdapter(
            capture=SequenceCapture(["gray"] * 4),
            observer=observer,
            robot=robot,
            frame_interval=0,
            post_action_settle=0,
        )

        result = adapter.execute(
            requested_action=SemanticAction(
                node_id="generic_step_1",
                action="tap_semantic",
                params={"element_id": "e1", "target": "app_icon", "tap_point": (0.35, 0.485)},
            ),
            planned_scene=planned,
            planned_frames=tuple(
                Image.new("RGB", (540, 960), "gray") for _ in range(4)
            ),
            goal=goal(),
            confirmed=True,
        )

        self.assertEqual([("tap", 350, 485)], robot.actions)
        self.assertEqual("planned", result.before_scene.fingerprint)
        self.assertEqual(1, observer.calls)

    def test_matching_pixels_do_not_accept_second_model_long_press_veto(self):
        states = {"goal_relevant": True, "fully_visible": True}
        target = UIElement(
            element_id="target",
            role="button",
            meaning="long_press_target_area",
            label="长按我 · 不要移动",
            bounds=(0.10, 0.48, 0.90, 0.70),
            confidence=1.0,
            states=states,
            evidence=("黄色虚线框内逐字显示长按目标",),
        )
        planned = UIScene(
            app_id="local.acceptance",
            screen_id="long-press-board",
            summary="本地长按验收页面",
            elements=(target,),
            stable=True,
            confidence=1.0,
            fingerprint="planned",
            camera_alignment=aligned_camera_facts(),
        )
        after = replace(
            planned,
            overlays=("long_press 验收通过",),
            fingerprint="after",
        )
        robot = FakeRobot()
        adapter = GenericSingleActionAdapter(
            capture=SequenceCapture(["gray"] * 4),
            observer=FakeSceneObserver([after]),
            robot=robot,
            frame_interval=0,
            post_action_settle=0,
        )

        result = adapter.execute(
                requested_action=SemanticAction(
                    node_id="long-press-risky",
                    action="long_press",
                    params={
                        "element_id": "target",
                        "target": "long_press_target_area",
                        "role": "button",
                        "label": "长按我 · 不要移动",
                        "states": states,
                        "duration_ms": 800,
                        "tap_point": (0.5, 0.59),

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
        self.assertEqual([("long_press", 500, 590, 0.8)], robot.actions)

    def test_changed_local_pixels_do_not_veto_or_claim_identity_verified(self):
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
            action="scroll",
            params={"direction": "up", },
        )

        result = adapter.execute(
            requested_action=action,
            planned_scene=planned,
            planned_frames=tuple(Image.new("RGB", (540, 960), "black") for _ in range(4)),
            goal=goal(),
            confirmed=True,
        )
        self.assertEqual([("swipe", "up")], robot.actions)
        self.assertEqual(1, observer.calls)
        self.assertFalse(result.confirmation_frame_identity_verified)
        self.assertIsNone(result.confirmation_frame_delta)

    def test_observer_exception_stops_without_retry_or_action(self):
        planned = scene("planned")
        fresh = scene("before", element_id="fresh")
        after = scene("after", screen_id="app_home", element_id="after")
        observer = FakeSceneObserver(
            [
                RuntimeError("观察器返回不可解析响应。"),
                fresh,
                after,
            ]
        )
        robot = FakeRobot()
        action = SemanticAction(
            node_id="generic_step_1",
            action="tap_semantic",
            params={"tap_point": (0.3, 0.4), "element_id": "e1", "target": "app_icon"},
        )

        with self.assertRaisesRegex(GenericActionAdapterError, "第1轮动作前观察失败"):
            prepared_21_adapter = self._adapter(observer, robot)
            prepared_21_goal = goal()
            prepared_21_scene, prepared_21_frames, _, _ = prepared_21_adapter.capture_scene(
                prepared_21_goal, prefix="test_orchestrator_before", evidence_dir=None)
            prepared_21_adapter.execute(
                requested_action=action,
                planned_scene=prepared_21_scene, planned_frames=prepared_21_frames,
                goal=prepared_21_goal,
                confirmed=True,
            )

        self.assertEqual(1, observer.calls)
        self.assertEqual(0, len(robot.actions))

    def test_post_action_accepts_first_complete_changing_window(self):
        planned = scene("planned")
        after = scene("after", screen_id="app_home", element_id="after")
        observer = FakeSceneObserver([after])
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
            params={"tap_point": (0.3, 0.4), "element_id": "e1", "target": "app_icon"},
        )
        result = adapter.execute(
            requested_action=action,
            planned_scene=planned,
            planned_frames=tuple(
                Image.new("RGB", (540, 960), "gray") for _ in range(4)
            ),
            goal=goal(),
            confirmed=True,
        )
        self.assertEqual(result.physical_actions, 1)
        self.assertEqual(robot.actions, [("tap", 300, 400)])
        self.assertEqual(capture.calls, 8)
        self.assertEqual(observer.calls, 1)

    def test_continuous_actions_receive_adaptive_post_action_budget(self):
        production_defaults = GenericSingleActionAdapter(
            capture=SequenceCapture(["gray"]),
            observer=FakeSceneObserver([]),
            robot=FakeRobot(),
        )
        self.assertEqual(10, production_defaults.post_action_timeout)
        self.assertEqual(45, production_defaults.post_action_continuous_timeout)

        adapter = GenericSingleActionAdapter(
            capture=SequenceCapture(["gray"]),
            observer=FakeSceneObserver([]),
            robot=FakeRobot(),
            post_action_timeout=10,
            post_action_continuous_timeout=45,
        )
        for kind in (
            "clear_verified_text",
            "drag",
            "input_verified_text",
            "long_press",
        ):
            with self.subTest(kind=kind):
                resolved = ResolvedSemanticAction(
                    node_id="continuous",
                    kind=kind,
                )
                self.assertTrue(
                    adapter._requires_extended_post_action_timeout(resolved)
                )
                self.assertEqual(45, adapter._post_action_timeout_for(resolved))

        navigation = ResolvedSemanticAction(
            node_id="navigation",
            kind="tap_semantic",
        )
        self.assertFalse(
            adapter._requires_extended_post_action_timeout(navigation)
        )
        self.assertEqual(10, adapter._post_action_timeout_for(navigation))

        explicit_test_budget = GenericSingleActionAdapter(
            capture=SequenceCapture(["gray"]),
            observer=FakeSceneObserver([]),
            robot=FakeRobot(),
            post_action_timeout=1,
        )
        self.assertEqual(1, explicit_test_budget.post_action_timeout)
        self.assertEqual(1, explicit_test_budget.post_action_continuous_timeout)

    def test_navigation_post_action_does_not_compare_changed_page_sharpness(self):
        sharp = textured_phone_frame()
        blurred = sharp.filter(ImageFilter.GaussianBlur(2))
        capture = SequenceCapture([blurred] * 4)
        adapter = GenericSingleActionAdapter(
            capture=capture,
            observer=FakeSceneObserver([]),
            robot=FakeRobot(),
            frame_interval=0,
            post_action_settle=0,
            post_action_timeout=1,
        )

        frames, _paths = adapter._capture_stable_post_action_frames(
            deadline=10**9,
            evidence_dir=None,
            prefix="navigation_changed_page",
            clarity_reference_frames=tuple(sharp.copy() for _ in range(4)),
            require_relative_clarity=False,
        )

        self.assertEqual(capture.calls, 4)
        self.assertEqual(len(frames), 4)

    def test_post_action_format_failure_does_not_resample_model(self):
        planned = scene("planned")
        fresh = scene("before")
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
            params={"tap_point": (0.3, 0.4), "element_id": "e1", "target": "app_icon"},
        )

        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(GenericActionAdapterError) as caught:
                prepared_24_adapter = adapter
                prepared_24_goal = goal()
                prepared_24_scene, prepared_24_frames, _, _ = prepared_24_adapter.capture_scene(
                    prepared_24_goal, evidence_dir=Path(temp), prefix="test_orchestrator_before")
                prepared_24_adapter.execute(
                    requested_action=action,
                    planned_scene=prepared_24_scene, planned_frames=prepared_24_frames,
                    goal=prepared_24_goal,
                    confirmed=True,
                    evidence_dir=Path(temp),
                )

        self.assertEqual(robot.actions, [("tap", 300, 400)])
        self.assertEqual(caught.exception.physical_actions, 1)
        self.assertEqual(observer.calls, 2)
        self.assertEqual(capture.calls, 12)
        self.assertEqual(len(caught.exception.evidence), 8)
        self.assertEqual(
            caught.exception.observation_errors,
            ("第1轮动作后观察失败：模型返回的 JSON 无法解析",),
        )

    def test_first_post_action_format_failure_stops_after_one_robot_action(self):
        observer = FakeSceneObserver(
            [
                scene("before"),
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
                prepared_25_adapter = adapter
                prepared_25_goal = goal()
                prepared_25_scene, prepared_25_frames, _, _ = prepared_25_adapter.capture_scene(
                    prepared_25_goal, evidence_dir=Path(temp), prefix="test_orchestrator_before")
                prepared_25_adapter.execute(
                    requested_action=SemanticAction(
                        node_id="generic_step_1",
                        action="tap_semantic",
                        params={"tap_point": (0.3, 0.4), "element_id": "e1", "target": "app_icon"},
                    ),
                    planned_scene=prepared_25_scene, planned_frames=prepared_25_frames,
                    goal=prepared_25_goal,
                    confirmed=True,
                    evidence_dir=Path(temp),
                )

        self.assertEqual(caught.exception.physical_actions, 1)
        self.assertEqual(len(caught.exception.evidence), 8)
        self.assertEqual(observer.calls, 2)
        self.assertEqual(len(robot.actions), 1)
        self.assertEqual(
            caught.exception.observation_errors,
            ("第1轮动作后观察失败：模型返回的 JSON 无法解析",),
        )
        self.assertIn("第1轮动作后观察失败", str(caught.exception))
        self.assertNotIn("第2轮动作后观察失败", str(caught.exception))

    def test_each_action_uses_exactly_one_fresh_screenshot_observation(self):
        observer = FakeSceneObserver(
            [
                scene("before"),
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
        )

        with self.assertRaises(GenericActionAdapterError) as caught:
            prepared_22_adapter = adapter
            prepared_22_goal = goal()
            prepared_22_scene, prepared_22_frames, _, _ = prepared_22_adapter.capture_scene(
                prepared_22_goal, prefix="test_orchestrator_before", evidence_dir=None)
            prepared_22_adapter.execute(
                requested_action=SemanticAction(
                    node_id="generic_step_1",
                    action="tap_semantic",
                    params={"tap_point": (0.3, 0.4), "element_id": "e1", "target": "app_icon"},
                ),
                planned_scene=prepared_22_scene, planned_frames=prepared_22_frames,
                goal=prepared_22_goal,
                confirmed=True,
            )

        self.assertEqual(caught.exception.physical_actions, 1)
        self.assertEqual(observer.calls, 2)
        self.assertEqual(capture.calls, 12)
        self.assertEqual(robot.actions, [("tap", 300, 400)])

    def test_format_error_stops_before_any_second_capture(self):
        observer = FakeSceneObserver(
            [
                scene("before"),
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
                prepared_26_adapter = adapter
                prepared_26_goal = goal()
                prepared_26_scene, prepared_26_frames, _, _ = prepared_26_adapter.capture_scene(
                    prepared_26_goal, evidence_dir=Path(temp), prefix="test_orchestrator_before")
                prepared_26_adapter.execute(
                    requested_action=SemanticAction(
                        node_id="generic_step_1",
                        action="tap_semantic",
                        params={"tap_point": (0.3, 0.4), "element_id": "e1", "target": "app_icon"},
                    ),
                    planned_scene=prepared_26_scene, planned_frames=prepared_26_frames,
                    goal=prepared_26_goal,
                    confirmed=True,
                    evidence_dir=Path(temp),
                )

        self.assertEqual(caught.exception.physical_actions, 1)
        self.assertEqual(
            caught.exception.observation_errors,
            ("第1轮动作后观察失败：模型返回的 JSON 无法解析：first",),
        )
        self.assertEqual(len(caught.exception.evidence), 8)
        self.assertNotIn("second_capture_timeout.jpg", caught.exception.evidence)
        self.assertEqual(observer.calls, 2)
        self.assertEqual(robot.actions, [("tap", 300, 400)])

    def test_post_action_non_format_failure_is_not_retried(self):
        observer = FakeSceneObserver(
            [scene("before"), VisionAgentError("请求超时")]
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
            prepared_23_adapter = adapter
            prepared_23_goal = goal()
            prepared_23_scene, prepared_23_frames, _, _ = prepared_23_adapter.capture_scene(
                prepared_23_goal, prefix="test_orchestrator_before", evidence_dir=None)
            prepared_23_adapter.execute(
                requested_action=SemanticAction(
                    node_id="generic_step_1",
                    action="tap_semantic",
                    params={"tap_point": (0.3, 0.4), "element_id": "e1", "target": "app_icon"},
                ),
                planned_scene=prepared_23_scene, planned_frames=prepared_23_frames,
                goal=prepared_23_goal,
                confirmed=True,
            )

        self.assertEqual(caught.exception.physical_actions, 1)
        self.assertEqual(observer.calls, 2)
        self.assertEqual(capture.calls, 12)
        self.assertEqual(len(robot.actions), 1)

    def test_post_action_timeout_stops_without_calling_model_or_tapping_again(self):
        planned = scene("planned")
        observer = FakeSceneObserver([])
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
            params={"tap_point": (0.3, 0.4), "element_id": "e1", "target": "app_icon"},
        )
        with self.assertRaisesRegex(
            GenericActionAdapterError,
            "未能采集到连续4帧",
        ) as ctx:
            adapter.execute(
                requested_action=action,
                planned_scene=planned,
                planned_frames=tuple(
                    Image.new("RGB", (540, 960), "gray") for _ in range(4)
                ),
                goal=goal(),
                confirmed=True,
            )
        self.assertEqual(ctx.exception.physical_actions, 1)
        self.assertEqual(observer.calls, 0)
        self.assertEqual(robot.actions, [("tap", 300, 400)])


if __name__ == "__main__":
    unittest.main()
