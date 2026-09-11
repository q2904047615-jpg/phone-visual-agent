"""Offline counterexamples for duplicate post-action verdicts and raw-frame protection."""

from contextlib import nullcontext
from dataclasses import replace
import unittest
from unittest.mock import patch

from PIL import Image

from agent.domain.confirmation_authority import ConfirmationAuthority
from agent.domain import ui_scene
from agent.domain.universal_action_controller import UniversalActionController, UniversalActionError
from agent.domain.validation import canonical_digest
from agent.infrastructure.observation_images import local_frame_fingerprint
import test_generic_action_adapter as fixtures
import test_launch_app_action as launches


class PostActionAuthorityTests(unittest.TestCase):
    def test_launch_postcheck_does_not_rejudge_app_identity(self):
        self.assertFalse(hasattr(ui_scene, "scene_matches_app_identity"))
        controller = UniversalActionController()
        for app_id, app_name in (("sample_app", "示例应用"), ("gallery", "图片工具")):
            before, action = launches.LaunchAppVerificationTests._candidate_action(
                app_id=app_id, app_name=app_name)
            resolved = controller.resolve_one(action, before, confirmed=True)
            for observed in ("unknown", "launcher", "com.example.other"):
                with self.subTest(target=app_id, observed=observed):
                    after = launches._scene(app_id=observed, fingerprint="fresh-post-launch")
                    self.assertEqual((), controller.verify_after_action(resolved, before, after))

    def test_navigation_postcheck_preserves_the_same_response_action_or_finish(self):
        for nav_fact in ("unknown", False, True):
            for decision_status in ("action", "finish"):
                with self.subTest(nav_fact=nav_fact, decision_status=decision_status):
                    before = fixtures.scene("before", system_ui=fixtures.SystemUIFacts(
                        immersive_or_fullscreen=True, navigation_bar_visible=False))
                    after = fixtures.scene("after", system_ui=fixtures.SystemUIFacts(
                        immersive_or_fullscreen=True, navigation_bar_visible=nav_fact))

                    class CurrentDecisionObserver(fixtures.FakeSceneObserver):
                        def observe_with_decision(self, **kwargs):
                            current, _ = super().observe_with_decision(**kwargs)
                            return current, {"status": decision_status, "action": "back"
                                if decision_status == "action" else None,
                                "reason": "依据当前新图继续动作或完成"}

                    robot = fixtures.FakeRobot()
                    observer = CurrentDecisionObserver([after])
                    adapter = fixtures.GenericSingleActionAdapter(
                        capture=fixtures.SequenceCapture(["gray"] * 4 + ["white"] * 4),
                        observer=observer, robot=robot, frame_interval=0, post_action_settle=0)
                    result = adapter.execute(requested_action=fixtures.SemanticAction(node_id="reveal",
                        action="reveal_system_navigation", params={}), planned_scene=before,
                        planned_frames=tuple(Image.new("RGB", (540, 960), "gray") for _ in range(4)),
                        goal=fixtures.goal(), confirmed=True)
                    self.assertEqual(1, result.physical_actions)
                    self.assertEqual(1, observer.calls)
                    self.assertEqual((), result.controller_transition_evidence)
                    self.assertEqual(decision_status, result.after_model_decision["status"])
                    self.assertEqual([("reveal_system_navigation",)], robot.actions)

    def test_execution_identity_and_stability_checks_remain(self):
        before, action = launches.LaunchAppVerificationTests._candidate_action()
        controller = UniversalActionController()
        resolved = controller.resolve_one(action, before, confirmed=True)
        after = launches._scene(app_id="com.example.sample", fingerprint="after")
        with self.assertRaisesRegex(UniversalActionError, "fingerprint"):
            controller.verify_after_action(replace(resolved, before_fingerprint="stale"), before, after)
        with self.assertRaisesRegex(UniversalActionError, "仍在变化"):
            controller.verify_after_action(resolved, before, replace(after, stable=False))

    @staticmethod
    def execute_unchanged_effect(*, bypass_change_gate=False):
        frame = fixtures.textured_phone_frame()
        fingerprint = local_frame_fingerprint(frame)
        before = fixtures.scene(fingerprint)
        target = before.elements[0]
        action = fixtures.SemanticAction(node_id="unchanged-effect", action="tap_semantic", params={
            "element_id": target.element_id, "target": target.meaning, "role": target.role,
            "label": target.label, "states": dict(target.states)})
        authority = ConfirmationAuthority(session_id="counterexample", task_id="task-1", device_id="test-device",
            revision=1, step_id="effect", effect_ids=("effect-1",), observation_id="obs-1",
            fingerprint=fingerprint, decision_node_id=action.node_id,
            action_digest=canonical_digest(action.to_dict()), consumed=True)
        robot = fixtures.FakeRobot()
        observer = fixtures.FakeSceneObserver([replace(before, fingerprint="new-capture-same-pixels")])
        adapter = fixtures.GenericSingleActionAdapter(
            capture=fixtures.SequenceCapture([frame.copy() for _ in range(8)]), observer=observer,
            robot=robot, frame_interval=0, post_action_settle=0)
        # Ablation only in an isolated test: production code and saved frame contents are unchanged.
        gate = patch("agent.infrastructure.generic_action_adapter.measure_material_visual_transition",
            return_value={"material": True, "test_only_ablation": True}) if bypass_change_gate else nullcontext()
        with gate:
            result = adapter.execute(requested_action=action, planned_scene=before,
                planned_frames=tuple(frame.copy() for _ in range(4)), goal=fixtures.goal(),
                confirmed=True, action_authority=authority)
        return result, robot, observer

    def test_raw_frame_gate_rejects_no_change_even_with_a_fresh_capture_and_finish(self):
        with self.assertRaisesRegex(fixtures.GenericActionAdapterError, "真实帧没有可归因的新变化") as raised:
            self.execute_unchanged_effect()
        self.assertEqual(1, raised.exception.physical_actions)

    def test_removing_raw_frame_gate_alone_leaves_no_equivalent_adapter_check(self):
        result, robot, observer = self.execute_unchanged_effect(bypass_change_gate=True)
        self.assertEqual("executed", result.action_outcome)
        self.assertEqual("finish", result.after_model_decision["status"])
        self.assertEqual(1, result.physical_actions)
        self.assertEqual(1, observer.calls)
        self.assertEqual(1, len(robot.actions))
        self.assertTrue(all(a.tobytes() == b.tobytes() for a, b in zip(result.before_frames, result.after_frames)))


if __name__ == "__main__":
    unittest.main()
