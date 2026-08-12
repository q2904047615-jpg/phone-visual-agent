from __future__ import annotations

import unittest
from types import SimpleNamespace

from generic_step_planner import GenericStepProposal
from semantic_executor import SemanticAction
from ui_scene import UIElement, UIScene
from universal_agent_orchestrator import PhaseOneNavigationPolicy


def _scene(
    *,
    meaning: str = "open_details",
    label: str = "查看详情",
    role: str = "button",
    bounds: tuple[float, float, float, float] = (0.1, 0.2, 0.5, 0.3),
    confidence: float = 0.96,
) -> UIScene:
    return UIScene(
        app_id="sample.app",
        screen_id="home",
        summary="显示一个可进入的详情入口",
        elements=(
            UIElement(
                element_id="candidate-1",
                role=role,
                meaning=meaning,
                label=label,
                bounds=bounds,
                confidence=confidence,
                evidence=("画面中可见目标",),
            ),
        ),
        stable=True,
        confidence=0.95,
        fingerprint="frame-a",
    )


def _decision(
    scene: UIScene,
    *,
    action_kind: str = "tap_semantic",
    direction: str = "up",
    decision_confidence: float = 0.94,
) -> SimpleNamespace:
    element = scene.elements[0]
    params = {
        "element_id": element.element_id,
        "target": element.meaning,
        "meaning": element.meaning,
        "role": element.role,
        "label": element.label,
        "expected_effect": {"scene_changed": True},
    }
    if action_kind == "swipe":
        params = {
            "direction": direction,
            "expected_effect": {"scene_changed": True},
        }
    elif action_kind in {"back", "wait_for_change"}:
        params = {"expected_effect": {"scene_changed": action_kind == "back"}}
    action = SemanticAction(node_id="node-1", action=action_kind, params=params)
    observation = SimpleNamespace(
        device_id="device-1",
        fingerprint=scene.fingerprint,
        scene=scene,
        candidate_conflicts=(),
    )
    return SimpleNamespace(
        task_id="task-1",
        device_id="device-1",
        revision=1,
        fingerprint=scene.fingerprint,
        confidence=decision_confidence,
        proposal=GenericStepProposal(status="action", action=action),
        trusted_observation=observation,
        target_region=SimpleNamespace(
            kind=(
                "element"
                if action_kind in {"tap_semantic", "dismiss_overlay"}
                else "system_navigation"
                if action_kind == "back"
                else "screen"
            ),
            element_id=(
                element.element_id
                if action_kind in {"tap_semantic", "dismiss_overlay"}
                else ""
            ),
            bounds=(
                element.bounds
                if action_kind in {"tap_semantic", "dismiss_overlay"}
                else (0.0, 0.0, 1.0, 1.0)
            ),
        ),
    )


def _context(*, impact: str = "navigation_only") -> SimpleNamespace:
    return SimpleNamespace(
        task_id="task-1",
        device_id="device-1",
        revision=1,
        current_external_impact=impact,
    )


class PhaseOneNavigationPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = PhaseOneNavigationPolicy()

    def test_allows_swipe_for_navigation_only_subgoal(self) -> None:
        scene = _scene()

        result = self.policy.evaluate(
            task_context=_context(),
            trusted_observation=_decision(scene).trusted_observation,
            decision=_decision(scene, action_kind="swipe"),
        )

        self.assertTrue(result.allowed)
        self.assertEqual("swipe", result.canonical_class)

    def test_allows_back_for_navigation_only_subgoal(self) -> None:
        scene = _scene()
        decision = _decision(scene, action_kind="back")

        result = self.policy.evaluate(
            task_context=_context(),
            trusted_observation=decision.trusted_observation,
            decision=decision,
        )

        self.assertTrue(result.allowed)
        self.assertEqual("back", result.canonical_class)

    def test_allows_canonical_navigation_tap(self) -> None:
        scene = _scene(meaning="open_details", label="查看详情", role="list_item")
        decision = _decision(scene)

        result = self.policy.evaluate(
            task_context=_context(),
            trusted_observation=decision.trusted_observation,
            decision=decision,
        )

        self.assertTrue(result.allowed)
        self.assertEqual("open", result.canonical_class)

    def test_rejects_toggle_input_and_keyboard_roles(self) -> None:
        for role in ("toggle", "input", "keyboard_key"):
            with self.subTest(role=role):
                scene = _scene(role=role)
                decision = _decision(scene)

                result = self.policy.evaluate(
                    task_context=_context(),
                    trusted_observation=decision.trusted_observation,
                    decision=decision,
                )

                self.assertFalse(result.allowed)
                self.assertIn("角色", result.reason)

    def test_rejects_external_state_even_when_confirmed(self) -> None:
        scene = _scene()
        decision = _decision(scene)

        result = self.policy.evaluate(
            task_context=_context(impact="external_state"),
            trusted_observation=decision.trusted_observation,
            decision=decision,
        )

        self.assertFalse(result.allowed)
        self.assertIn("external_state", result.reason)

    def test_rejects_unknown_impact_before_qwen_or_robot(self) -> None:
        scene = _scene()
        decision = _decision(scene)

        result = self.policy.evaluate(
            task_context=_context(impact="unknown"),
            trusted_observation=decision.trusted_observation,
            decision=decision,
        )

        self.assertFalse(result.allowed)
        self.assertIn("unknown", result.reason)

    def test_rejects_account_effect_semantics(self) -> None:
        for meaning, label in (
            ("follow_creator", "关注"),
            ("open_payment", "打开支付"),
            ("view_and_send", "查看并发送"),
        ):
            with self.subTest(meaning=meaning):
                scene = _scene(meaning=meaning, label=label)
                decision = _decision(scene)

                result = self.policy.evaluate(
                    task_context=_context(),
                    trusted_observation=decision.trusted_observation,
                    decision=decision,
                )

                self.assertFalse(result.allowed)

    def test_rejects_noncanonical_or_ambiguous_tap_meaning(self) -> None:
        scene = _scene(meaning="primary_action", label="继续")
        decision = _decision(scene)

        result = self.policy.evaluate(
            task_context=_context(),
            trusted_observation=decision.trusted_observation,
            decision=decision,
        )

        self.assertFalse(result.allowed)
        self.assertIn("导航语义", result.reason)

    def test_rejects_out_of_bounds_or_stale_candidate(self) -> None:
        scene = _scene()
        decision = _decision(scene)
        decision.fingerprint = "frame-stale"

        result = self.policy.evaluate(
            task_context=_context(),
            trusted_observation=decision.trusted_observation,
            decision=decision,
        )

        self.assertFalse(result.allowed)
        self.assertIn("fingerprint", result.reason)


if __name__ == "__main__":
    unittest.main()
