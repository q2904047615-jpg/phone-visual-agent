from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from deepseek_task_graph import (
    CompletionCondition,
    DynamicTaskGraph,
    GraphGoal,
    RiskAction,
    Subgoal,
    TargetApp,
)
from generic_step_planner import GenericStepProposal
from semantic_executor import SemanticAction
from ui_scene import UIElement, UIScene
from universal_agent_orchestrator import (
    AgentEvidenceStore,
    EvidenceStoreError,
    ObservationBridge,
    PhaseOneNavigationPolicy,
    UniversalAgentOrchestrator,
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


def _external_graph(*, impact: str = "external_state") -> DynamicTaskGraph:
    risk_type = (
        "message_or_communication"
        if impact == "external_state"
        else "unknown_external_effect"
    )
    graph = DynamicTaskGraph(
        task_id="task-1",
        device_id="device-1",
        revision=1,
        status="awaiting_confirmation",
        goal=GraphGoal(
            objective=(
                "向目标联系人发送需求询问"
                if impact == "external_state"
                else "处理影响尚不明确的目标状态"
            ),
            target_apps=(TargetApp(app_id="sample", app_name="示例工具"),),
            entities={"contact": "目标联系人"},
        ),
        constraints=("任何外部影响都必须失败关闭",),
        completion_conditions=(
            CompletionCondition(
                condition_id="condition-1",
                description="目标外部状态已经产生",
                evidence_required=("页面显示结果",),
            ),
        ),
        risk_actions=(
            RiskAction(
                risk_id="risk-1",
                description=(
                    "发送需求询问" if impact == "external_state" else "影响不明确"
                ),
                external_effect=(
                    "向外部对象发送消息"
                    if impact == "external_state"
                    else "可能改变外部状态"
                ),
                risk_type=risk_type,
                risk_level="high",
                subgoal_ids=("subgoal-1",),
            ),
        ),
        subgoals=(
            Subgoal(
                subgoal_id="subgoal-1",
                objective=(
                    "目标联系人收到需求询问"
                    if impact == "external_state"
                    else "目标状态达到但影响仍不明确"
                ),
                status="active",
                depends_on=(),
                constraints=("必须等待明确确认",),
                completion_conditions=("页面显示结果",),
                completion_evidence=(),
                risk_action_ids=("risk-1",),
                external_impact=impact,
            ),
        ),
        active_subgoal_id="subgoal-1",
        raw_user_goal="测试风险目标",
    )
    graph.validate()
    return graph


def _completed_graph(graph: DynamicTaskGraph) -> DynamicTaskGraph:
    completed = replace(
        graph,
        revision=graph.revision + 1,
        status="completed",
        completion_conditions=tuple(
            replace(item, satisfied=True, evidence=("详情标题可见",))
            for item in graph.completion_conditions
        ),
        subgoals=tuple(
            replace(item, status="completed", completion_evidence=("详情标题可见",))
            for item in graph.subgoals
        ),
        active_subgoal_id=None,
    )
    completed.validate()
    return completed


class FakeDeepSeekPlanner:
    def __init__(self, graph: DynamicTaskGraph, *, replan_result=None) -> None:
        self.graph = graph
        self.replan_result = replan_result
        self.plan_calls = []
        self.replan_calls = []

    def plan(self, raw_goal, *, device_id, task_id=None):
        self.plan_calls.append((raw_goal, device_id, task_id))
        return self.graph

    def replan(self, graph, observation, *, trigger, reason):
        self.replan_calls.append((graph, observation, trigger, reason))
        return self.replan_result or graph


class FakeTrustedObservation:
    def __init__(self, *, device_id: str, scene: UIScene, observation_id="obs-start"):
        self.device_id = device_id
        self.scene = scene
        self.observation_id = observation_id
        self.fingerprint = scene.fingerprint
        self.candidate_conflicts = ()

    def to_dict(self):
        return {
            "observation_id": self.observation_id,
            "device_id": self.device_id,
            "fingerprint": self.fingerprint,
            "scene": self.scene.to_dict(),
            "candidate_conflicts": [],
        }


class FakeAdapter:
    def __init__(self, scene: UIScene) -> None:
        self.scene = scene
        self.capture_calls = 0
        self.execute_calls = 0

    def capture_scene(self, goal, *, evidence_dir, prefix):
        self.capture_calls += 1
        frames = [Image.new("RGB", (540, 960), "white") for _ in range(4)]
        return self.scene, frames, tuple(
            str(evidence_dir / f"{prefix}_{index}.jpg") for index in range(1, 5)
        )

    def execute(self, **_kwargs):
        self.execute_calls += 1
        raise AssertionError("start must not execute a physical action")


class FakeQwenObserver:
    def __init__(self, status="action", *, mutate_identity=None) -> None:
        self.status = status
        self.mutate_identity = mutate_identity
        self.calls = []

    def decide(self, *, frames, task_context, trusted_observation, decision_number=1):
        self.calls.append((frames, task_context, trusted_observation, decision_number))
        if self.status == "action":
            decision = _decision(trusted_observation.scene)
        else:
            proposal = GenericStepProposal(
                status=self.status,
                action=None,
                reason="当前画面不能继续" if self.status == "blocked" else "完成证据可见",
                completion_evidence=("详情标题可见",) if self.status == "finished" else (),
            )
            decision = SimpleNamespace(
                proposal=proposal,
                target_region=None,
                confidence=0.95,
            )
        decision.task_id = task_context["task_id"]
        decision.device_id = task_context["device_id"]
        decision.revision = task_context["revision"]
        decision.observation_id = trusted_observation.observation_id
        decision.fingerprint = trusted_observation.fingerprint
        decision.trusted_observation = trusted_observation
        if self.mutate_identity:
            setattr(decision, self.mutate_identity[0], self.mutate_identity[1])
        decision.to_dict = lambda: {
            "task_id": decision.task_id,
            "device_id": decision.device_id,
            "revision": decision.revision,
            "observation_id": decision.observation_id,
            "fingerprint": decision.fingerprint,
            "status": decision.proposal.status,
            "next_action": (
                decision.proposal.action.to_dict() if decision.proposal.action else None
            ),
            "reason": decision.proposal.reason,
        }
        return decision


def _trusted_factory(*, frames, device_id, scene, observation_id=None):
    if len(frames) < 4:
        raise AssertionError("trusted observation requires four frames")
    return FakeTrustedObservation(
        device_id=device_id,
        scene=scene,
        observation_id=observation_id or "obs-start",
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


class AgentEvidenceStoreTests(unittest.TestCase):
    def test_writes_all_authoritative_json_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            store = AgentEvidenceStore(Path(temp))
            graph = _graph()

            store.write_session({"session_id": "session-1", "status": "observing"})
            store.write_task_graph(graph)
            store.write_risk_audit(graph)
            store.write_trusted_observation(1, {"observation_id": "obs-1"})
            store.write_qwen_decision(1, {"status": "action"})
            store.write_controller_decision(1, {"allowed": True})
            store.write_verification(1, {"matched": True})
            store.write_report({"physical_actions": 0})

            expected = {
                "session.json",
                "task_graph_revision_1.json",
                "risk_audit_revision_1.json",
                "trusted_observation_step_1.json",
                "qwen_decision_step_1.json",
                "controller_decision_step_1.json",
                "verification_step_1.json",
                "report.json",
            }
            self.assertTrue(expected.issubset({item.name for item in Path(temp).iterdir()}))
            payload = json.loads((Path(temp) / "task_graph_revision_1.json").read_text("utf-8"))
            self.assertEqual("task-1", payload["task_id"])

    def test_json_write_uses_replace_and_never_leaves_partial_target(self) -> None:
        def fail_replace(_source: Path, _target: Path) -> None:
            raise OSError("simulated disk failure")

        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "session.json"
            target.write_text('{"status":"old"}', encoding="utf-8")
            store = AgentEvidenceStore(Path(temp), replace_file=fail_replace)

            with self.assertRaisesRegex(EvidenceStoreError, "原子写入失败"):
                store.write_session({"status": "new"})

            self.assertEqual('{"status":"old"}', target.read_text("utf-8"))
            self.assertEqual([], list(Path(temp).glob(".session.json.*.tmp")))

    def test_rejects_path_traversal_and_non_serializable_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            store = AgentEvidenceStore(Path(temp))

            with self.assertRaises(EvidenceStoreError):
                store.write_json("../outside.json", {})
            with self.assertRaises(EvidenceStoreError):
                store.write_json("bad.json", {"value": object()})

            self.assertFalse((Path(temp).parent / "outside.json").exists())


class UniversalAgentStartTests(unittest.TestCase):
    def _orchestrator(self, planner, qwen, adapter):
        return UniversalAgentOrchestrator(
            deepseek_planner=planner,
            qwen_observer=qwen,
            adapter_factory=lambda _device_id: adapter,
            trusted_observation_factory=_trusted_factory,
        )

    def test_start_plans_observes_and_decides_with_zero_physical_actions(self) -> None:
        graph = _graph()
        planner = FakeDeepSeekPlanner(graph)
        qwen = FakeQwenObserver()
        adapter = FakeAdapter(_scene())
        with tempfile.TemporaryDirectory() as temp:
            session = self._orchestrator(planner, qwen, adapter).start(
                session_id="session-1",
                raw_goal="看看公开分类详情",
                device_id="device-1",
                run_dir=Path(temp),
            )

        self.assertEqual("awaiting_confirmation", session.status)
        self.assertEqual(0, session.physical_actions)
        self.assertEqual(1, adapter.capture_calls)
        self.assertEqual(0, adapter.execute_calls)
        self.assertEqual(1, len(qwen.calls))

    def test_start_uses_at_least_four_frames_for_trusted_observation(self) -> None:
        seen = []

        def recording_factory(**kwargs):
            seen.append(len(kwargs["frames"]))
            return _trusted_factory(**kwargs)

        adapter = FakeAdapter(_scene())
        with tempfile.TemporaryDirectory() as temp:
            orchestrator = UniversalAgentOrchestrator(
                deepseek_planner=FakeDeepSeekPlanner(_graph()),
                qwen_observer=FakeQwenObserver(),
                adapter_factory=lambda _device_id: adapter,
                trusted_observation_factory=recording_factory,
            )
            orchestrator.start(
                session_id="session-1",
                raw_goal="查看详情",
                device_id="device-1",
                run_dir=Path(temp),
            )

        self.assertEqual([4], seen)

    def test_external_state_blocks_before_qwen_and_robot(self) -> None:
        adapter = FakeAdapter(_scene())
        qwen = FakeQwenObserver()
        with tempfile.TemporaryDirectory() as temp:
            session = self._orchestrator(
                FakeDeepSeekPlanner(_external_graph()), qwen, adapter
            ).start(
                session_id="session-risk",
                raw_goal="向目标联系人发送需求询问",
                device_id="device-1",
                run_dir=Path(temp),
            )

        self.assertEqual("blocked", session.status)
        self.assertEqual(0, len(qwen.calls))
        self.assertEqual(0, adapter.capture_calls)
        self.assertEqual(0, adapter.execute_calls)
        self.assertEqual(0, session.physical_actions)

    def test_unknown_impact_blocks_before_qwen_and_robot(self) -> None:
        adapter = FakeAdapter(_scene())
        qwen = FakeQwenObserver()
        with tempfile.TemporaryDirectory() as temp:
            session = self._orchestrator(
                FakeDeepSeekPlanner(_external_graph(impact="unknown")), qwen, adapter
            ).start(
                session_id="session-unknown",
                raw_goal="处理影响尚不明确的状态",
                device_id="device-1",
                run_dir=Path(temp),
            )

        self.assertEqual("blocked", session.status)
        self.assertEqual([], qwen.calls)
        self.assertEqual(0, adapter.capture_calls)
        self.assertEqual(0, session.physical_actions)

    def test_qwen_blocked_has_no_confirmation_entry(self) -> None:
        adapter = FakeAdapter(_scene())
        with tempfile.TemporaryDirectory() as temp:
            session = self._orchestrator(
                FakeDeepSeekPlanner(_graph()), FakeQwenObserver("blocked"), adapter
            ).start(
                session_id="session-blocked",
                raw_goal="查看详情",
                device_id="device-1",
                run_dir=Path(temp),
            )

        self.assertEqual("blocked", session.status)
        self.assertIsNone(session.confirmation_authority)
        self.assertEqual(0, adapter.execute_calls)

    def test_qwen_finished_requires_deepseek_completion_revision(self) -> None:
        initial = _graph()
        completed = _completed_graph(initial)
        with tempfile.TemporaryDirectory() as temp:
            session = self._orchestrator(
                FakeDeepSeekPlanner(initial, replan_result=completed),
                FakeQwenObserver("finished"),
                FakeAdapter(_scene()),
            ).start(
                session_id="session-finished",
                raw_goal="查看详情",
                device_id="device-1",
                run_dir=Path(temp),
            )

        self.assertEqual("succeeded", session.status)
        self.assertEqual(2, session.task_graph.revision)
        self.assertEqual(0, session.physical_actions)

        with tempfile.TemporaryDirectory() as temp:
            not_completed = self._orchestrator(
                FakeDeepSeekPlanner(initial, replan_result=replace(initial, revision=2)),
                FakeQwenObserver("finished"),
                FakeAdapter(_scene()),
            ).start(
                session_id="session-unproven",
                raw_goal="查看详情",
                device_id="device-1",
                run_dir=Path(temp),
            )
        self.assertEqual("blocked", not_completed.status)

    def test_protocol_or_identity_mismatch_fails_with_zero_actions(self) -> None:
        adapter = FakeAdapter(_scene())
        qwen = FakeQwenObserver(mutate_identity=("device_id", "device-other"))
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(UniversalAgentOrchestratorError, "device_id"):
                self._orchestrator(FakeDeepSeekPlanner(_graph()), qwen, adapter).start(
                    session_id="session-stale",
                    raw_goal="查看详情",
                    device_id="device-1",
                    run_dir=Path(temp),
                )

        self.assertEqual(0, adapter.execute_calls)


if __name__ == "__main__":
    unittest.main()
