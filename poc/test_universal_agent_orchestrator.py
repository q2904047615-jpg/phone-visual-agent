from __future__ import annotations

import unittest
from types import SimpleNamespace

from deepseek_task_graph import (
    CompletionCondition,
    DynamicTaskGraph,
    GraphGoal,
    Subgoal,
    TargetApp,
)
from generic_step_planner import GenericStepProposal
from semantic_executor import SemanticAction
from ui_scene import UIElement, UIScene
from universal_agent_orchestrator import (
    ObservationBridge,
    PhaseOneNavigationPolicy,
    UniversalAgentOrchestratorError,
)


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


def _graph(*, device_id: str = "device-1") -> DynamicTaskGraph:
    graph = DynamicTaskGraph(
        task_id="task-1",
        device_id=device_id,
        revision=1,
        status="running",
        goal=GraphGoal(
            objective="在图片工具中查看公开分类详情",
            target_apps=(TargetApp(app_id="gallery", app_name="图片工具"),),
            entities={"category": "风景", "expected_text": "风景"},
        ),
        constraints=("仅查看公开信息",),
        completion_conditions=(
            CompletionCondition(
                condition_id="condition-1",
                description="页面显示风景分类详情",
                evidence_required=("详情标题可见",),
            ),
        ),
        risk_actions=(),
        subgoals=(
            Subgoal(
                subgoal_id="subgoal-1",
                objective="查看风景分类详情",
                status="active",
                depends_on=(),
                constraints=("不得改变任何账号状态",),
                completion_conditions=("详情标题可见",),
                completion_evidence=(),
                risk_action_ids=(),
                external_impact="navigation_only",
            ),
        ),
        active_subgoal_id="subgoal-1",
        raw_user_goal="看看图片工具里的风景分类",
    )
    graph.validate()
    return graph


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


class ObservationBridgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bridge = ObservationBridge()

    def test_projects_dynamic_graph_to_generic_goal_without_business_steps(self) -> None:
        goal = self.bridge.goal_draft(_graph())

        payload = goal.to_dict()
        self.assertTrue(goal.understood)
        self.assertEqual("gallery", goal.app_id)
        self.assertNotIn("operation", payload)
        self.assertNotIn("steps", str(payload).casefold())
        self.assertNotIn("coordinate", str(payload).casefold())

    def test_projection_preserves_goal_entities_constraints_and_completion(self) -> None:
        graph = _graph()

        goal = self.bridge.goal_draft(graph)

        self.assertEqual("风景", goal.entities["category"])
        self.assertIn("仅查看公开信息", goal.constraints)
        self.assertIn("不得改变任何账号状态", goal.constraints)
        self.assertEqual(
            "页面显示风景分类详情",
            goal.success_criteria["condition-1"]["description"],
        )

    def test_builds_observed_state_only_from_visible_evidence(self) -> None:
        scene = _scene()
        observation = SimpleNamespace(
            device_id="device-1",
            observation_id="obs-1",
            fingerprint=scene.fingerprint,
            scene=scene,
        )

        observed = self.bridge.observed_state(
            graph=_graph(),
            trusted_observation=observation,
            action_outcome="matched",
            verification={
                "visible_evidence": ["页面标题已变化"],
                "internal_debug_note": "不得进入证据",
            },
        )

        self.assertIn("页面标题已变化", observed.visible_evidence)
        self.assertTrue(any("查看详情" in item for item in observed.visible_evidence))
        self.assertNotIn("不得进入证据", observed.visible_evidence)
        observed.validate()

    def test_action_failure_is_not_reported_as_completion(self) -> None:
        scene = _scene()
        observation = SimpleNamespace(
            device_id="device-1",
            observation_id="obs-1",
            fingerprint=scene.fingerprint,
            scene=scene,
        )

        observed = self.bridge.observed_state(
            graph=_graph(),
            trusted_observation=observation,
            action_outcome="mismatched",
            verification={
                "completion_evidence": ["模型声称完成"],
                "blocked_reasons": ["页面没有发生预期变化"],
            },
        )

        self.assertNotIn("模型声称完成", observed.visible_evidence)
        self.assertEqual(("页面没有发生预期变化",), observed.blocked_reasons)

    def test_graph_and_observation_device_mismatch_fails_closed(self) -> None:
        scene = _scene()
        observation = SimpleNamespace(
            device_id="device-other",
            observation_id="obs-1",
            fingerprint=scene.fingerprint,
            scene=scene,
        )

        with self.assertRaisesRegex(UniversalAgentOrchestratorError, "device_id"):
            self.bridge.observed_state(
                graph=_graph(),
                trusted_observation=observation,
                action_outcome="not_applicable",
                verification={},
            )


if __name__ == "__main__":
    unittest.main()
