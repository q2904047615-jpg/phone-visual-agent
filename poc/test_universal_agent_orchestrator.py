from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from PIL import Image

from agent.application.action_adapter import GenericActionAdapterError
from agent.application.universal_agent_orchestrator import (
    ObservationBridge,
    UniversalAgentOrchestrator,
    UniversalAgentOrchestratorError,
)
from agent.domain.canonical_action_protocol import (
    GenericStepProposal,
    canonical_candidate_expected_result,
    compile_canonical_action_catalog,
)
from agent.domain.semantic_action import SemanticAction
from agent.domain.task_graph import (
    CompletionCondition,
    DynamicTaskGraph,
    GraphGoal,
    Subgoal,
    TargetApp,
)
from agent.domain.ui_scene import SystemUIFacts, UIElement, UIScene
from agent.domain.universal_action_controller import ResolvedSemanticAction
from agent.infrastructure import DeviceTaskRegistry, FileSystemAgentEvidenceStore
from agent.infrastructure.generic_action_adapter import GenericActionExecutionResult


def _scene(
    *,
    fingerprint: str = "frame-a",
    meaning: str = "open_details",
    label: str = "查看详情",
    app_id: str = "gallery",
) -> UIScene:
    return UIScene(
        app_id=app_id,
        screen_id="home",
        summary=f"{label}当前可见",
        elements=(
            UIElement(
                element_id="candidate-1",
                role="button",
                meaning=meaning,
                label=label,
                bounds=(0.1, 0.2, 0.5, 0.3),
                confidence=0.97,
                states={
                    "goal_relevant": True,
                    "fully_visible": True,
                    "enabled": True,
                },
                evidence=(f"{label}清晰可见",),
            ),
        ),
        stable=True,
        confidence=0.96,
        fingerprint=fingerprint,
        system_ui=SystemUIFacts(),
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
                constraints=(),
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


class FakeDeepSeekPlanner:
    def __init__(
        self,
        graph: DynamicTaskGraph,
        *,
        replan_result: DynamicTaskGraph | None = None,
        replan_error: Exception | None = None,
    ) -> None:
        self.graph = graph
        self.replan_result = replan_result
        self.replan_error = replan_error
        self.plan_calls: list[tuple] = []
        self.replan_calls: list[tuple] = []

    def plan(self, raw_goal, *, device_id, task_id=None):
        self.plan_calls.append((raw_goal, device_id, task_id))
        return self.graph

    def replan(self, *args, **kwargs):
        self.replan_calls.append((args, kwargs))
        if self.replan_error is not None:
            raise self.replan_error
        return self.replan_result or self.graph


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


def _trusted_factory(*, frames, device_id, scene, observation_id=None):
    if len(frames) < 4:
        raise AssertionError("trusted observation requires four frames")
    return FakeTrustedObservation(
        device_id=device_id,
        scene=scene,
        observation_id=observation_id or "obs-start",
    )


class FakeAdapter:
    def __init__(self, scene: UIScene) -> None:
        self.scene = scene
        self.capture_calls = 0
        self.execute_calls = 0

    def capture_scene(self, goal, *, evidence_dir, prefix):
        del goal
        self.capture_calls += 1
        frames = [Image.new("RGB", (540, 960), "white") for _ in range(4)]
        return self.scene, frames, tuple(
            str(evidence_dir / f"{prefix}_{index}.jpg") for index in range(1, 5)
        )

    def execute(self, **_kwargs):
        self.execute_calls += 1
        raise AssertionError("start must not execute a physical action")


class FakeExecutingAdapter(FakeAdapter):
    def __init__(
        self,
        scene: UIScene,
        after_scene: UIScene,
        *,
        execute_error: GenericActionAdapterError | None = None,
        action_outcome: str = "matched",
        verification_errors: tuple[str, ...] = (),
    ) -> None:
        super().__init__(scene)
        self.after_scene = after_scene
        self.execute_error = execute_error
        self.action_outcome = action_outcome
        self.verification_errors = verification_errors
        self.action_authorities = []

    def execute(
        self,
        *,
        requested_action,
        planned_scene,
        planned_frames=None,
        goal,
        confirmed,
        evidence_dir,
        action_authority=None,
    ):
        del planned_frames, goal, confirmed, evidence_dir
        self.action_authorities.append(action_authority)
        self.execute_calls += 1
        if self.execute_error is not None:
            raise self.execute_error
        after_frames = tuple(
            Image.new("RGB", (540, 960), "white") for _ in range(4)
        )
        self.scene = self.after_scene
        return GenericActionExecutionResult(
            requested_action=requested_action,
            rebound_action=requested_action,
            resolved_action=ResolvedSemanticAction(
                node_id=requested_action.node_id,
                kind=requested_action.action,
                normalized_point=(0.3, 0.25),
                target_element_id=str(requested_action.params.get("element_id") or ""),
                before_fingerprint=planned_scene.fingerprint,
                expected_effect=dict(requested_action.params.get("expected_effect") or {}),
            ),
            before_scene=planned_scene,
            after_scene=self.after_scene,
            planned_scene_fingerprint=planned_scene.fingerprint,
            confirmation_frame_identity_verified=True,
            confirmation_frame_delta=0.0,
            physical_actions=1,
            action_outcome=self.action_outcome,
            verification_errors=self.verification_errors,
            controller_transition_evidence=(),
            robot_result={"ok": True},
            evidence=("before-1.jpg", "after-1.jpg"),
            after_frames=after_frames,
            after_frame_paths=(
                "after-1.jpg",
                "after-2.jpg",
                "after-3.jpg",
                "after-4.jpg",
            ),
        )


class FakeQwenObserver:
    """Emit one scripted decision per fresh observation; never invoke DeepSeek."""

    def __init__(
        self,
        status="action",
        *,
        after_status="finish",
        mutate_identity=None,
        action_kind="tap_semantic",
    ) -> None:
        self.status = status
        self.after_status = after_status
        self.mutate_identity = mutate_identity
        self.action_kind = action_kind
        self.calls = []
        self.text_transport_profiles = []

    def decide(
        self,
        *,
        frames,
        task_context,
        trusted_observation,
        decision_number=1,
        available_action_kinds=None,
        text_transport_profile=None,
        launch_target=None,
    ):
        del launch_target
        self.calls.append((frames, task_context, trusted_observation, decision_number))
        self.text_transport_profiles.append(text_transport_profile)
        current_status = self.status if len(self.calls) == 1 else self.after_status
        completion_evidence = ()
        if current_status == "action":
            semantic_ir = getattr(task_context, "semantic_ir", None)
            if semantic_ir is None:
                raise AssertionError("FakeQwenObserver 缺少 canonical TaskSemanticIR")
            catalog = compile_canonical_action_catalog(
                trusted_observation.scene,
                semantic_ir,
                available_action_kinds or (),
            )
            matches = [
                item
                for item in catalog.candidates
                if item.action_kind == self.action_kind
            ]
            if len(matches) != 1:
                proposal = GenericStepProposal(
                    status="blocked",
                    reason=f"没有唯一 {self.action_kind} canonical candidate",
                )
                current_status = "blocked"
            else:
                candidate = matches[0]
                expected = canonical_candidate_expected_result(
                    candidate, trusted_observation.scene
                )
                action = SemanticAction(
                    node_id=f"qwen-model-{decision_number}",
                    action=candidate.action_kind,
                    params={
                        **candidate.parameters,
                        "formal_candidate_id": candidate.candidate_id,
                        "formal_transition": candidate.transition.to_dict(),
                        "expected_effect": expected,
                    },
                )
                proposal = GenericStepProposal(
                    status="action",
                    action=action,
                    reason="模型在当前截图中明确选择了唯一 canonical candidate。",
                )
        elif current_status == "finish":
            proposal = GenericStepProposal(
                status="finish",
                reason="模型根据当前新截图判定当前目标完成。",
            )
            completion_evidence = (trusted_observation.scene.summary,)
        else:
            proposal = GenericStepProposal(
                status="blocked",
                reason="模型根据当前截图无法选择合法动作。",
            )

        decision = SimpleNamespace(
            task_id=task_context["task_id"],
            device_id=task_context["device_id"],
            revision=task_context["revision"],
            observation_id=trusted_observation.observation_id,
            fingerprint=trusted_observation.fingerprint,
            trusted_observation=trusted_observation,
            target_region=None,
            confidence=0.95,
            proposal=proposal,
            completion_evidence=completion_evidence,
        )
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
                decision.proposal.action.to_dict()
                if decision.proposal.action
                else None
            ),
            "reason": decision.proposal.reason,
            "completion_evidence": list(decision.completion_evidence),
        }
        return decision


def _confirmation(session) -> dict:
    return dict(session.snapshot()["confirmation_scope"])


class UniversalAgentSingleAuthorityLoopTests(unittest.TestCase):
    @staticmethod
    def orchestrator(planner, qwen, adapter):
        return UniversalAgentOrchestrator(
            deepseek_planner=planner,
            qwen_observer=qwen,
            adapter_factory=lambda _device_id: adapter,
            trusted_observation_factory=_trusted_factory,
            evidence_store_factory=FileSystemAgentEvidenceStore,
            device_registry=DeviceTaskRegistry(),
        )

    def test_start_only_observes_and_stages_model_action(self) -> None:
        planner = FakeDeepSeekPlanner(_graph())
        qwen = FakeQwenObserver()
        adapter = FakeAdapter(_scene())
        with tempfile.TemporaryDirectory() as temp:
            session = self.orchestrator(planner, qwen, adapter).start(
                session_id="session-start",
                raw_goal="查看详情",
                device_id="device-1",
                run_dir=Path(temp),
            )

        self.assertEqual("awaiting_confirmation", session.status)
        self.assertEqual(1, len(planner.plan_calls))
        self.assertEqual([], planner.replan_calls)
        self.assertEqual(1, len(qwen.calls))
        self.assertEqual(0, adapter.execute_calls)

    def test_action_then_new_screenshot_finish_completes_without_replan(self) -> None:
        planner = FakeDeepSeekPlanner(
            _graph(), replan_error=AssertionError("DeepSeek must not replan")
        )
        qwen = FakeQwenObserver(after_status="finish")
        adapter = FakeExecutingAdapter(
            _scene(), _scene(fingerprint="frame-after", label="详情页")
        )
        orchestrator = self.orchestrator(planner, qwen, adapter)
        with tempfile.TemporaryDirectory() as temp:
            session = orchestrator.start(
                session_id="session-finish",
                raw_goal="查看详情",
                device_id="device-1",
                run_dir=Path(temp),
            )
            orchestrator.confirm_one(session, _confirmation(session))

        self.assertEqual("succeeded", session.status)
        self.assertEqual(2, session.task_graph.revision)
        self.assertEqual(1, session.physical_actions)
        self.assertEqual(2, len(qwen.calls))
        self.assertEqual([], planner.replan_calls)
        self.assertNotEqual(
            qwen.calls[0][2].observation_id, qwen.calls[1][2].observation_id
        )

    def test_action_after_new_screenshot_does_not_pretend_goal_finished(self) -> None:
        planner = FakeDeepSeekPlanner(_graph())
        qwen = FakeQwenObserver(after_status="action")
        adapter = FakeExecutingAdapter(
            _scene(), _scene(fingerprint="frame-after", label="下一入口")
        )
        orchestrator = self.orchestrator(planner, qwen, adapter)
        with tempfile.TemporaryDirectory() as temp:
            session = orchestrator.start(
                session_id="session-next-action",
                raw_goal="继续查看详情",
                device_id="device-1",
                run_dir=Path(temp),
            )
            orchestrator.confirm_one(session, _confirmation(session))

        self.assertEqual("awaiting_confirmation", session.status)
        self.assertEqual(1, session.task_graph.revision)
        self.assertEqual(1, session.physical_actions)
        self.assertEqual(2, len(qwen.calls))
        self.assertEqual([], planner.replan_calls)

    def test_finish_on_first_screenshot_uses_zero_actions(self) -> None:
        planner = FakeDeepSeekPlanner(_graph())
        qwen = FakeQwenObserver(status="finish")
        adapter = FakeAdapter(_scene(label="详情页"))
        with tempfile.TemporaryDirectory() as temp:
            session = self.orchestrator(planner, qwen, adapter).start(
                session_id="session-already-finished",
                raw_goal="确认详情已显示",
                device_id="device-1",
                run_dir=Path(temp),
            )

        self.assertEqual("succeeded", session.status)
        self.assertEqual(0, session.physical_actions)
        self.assertEqual(1, len(qwen.calls))
        self.assertEqual([], planner.replan_calls)

    def test_model_blocked_stops_without_local_fallback(self) -> None:
        planner = FakeDeepSeekPlanner(_graph())
        qwen = FakeQwenObserver(status="blocked")
        adapter = FakeAdapter(_scene())
        with tempfile.TemporaryDirectory() as temp:
            session = self.orchestrator(planner, qwen, adapter).start(
                session_id="session-blocked",
                raw_goal="查看详情",
                device_id="device-1",
                run_dir=Path(temp),
            )

        self.assertEqual("blocked", session.status)
        self.assertEqual(0, session.physical_actions)
        self.assertIsNone(session.confirmation_authority)
        self.assertEqual(0, adapter.execute_calls)

    def test_stale_action_scope_cannot_execute(self) -> None:
        planner = FakeDeepSeekPlanner(_graph())
        qwen = FakeQwenObserver()
        adapter = FakeExecutingAdapter(_scene(), _scene(fingerprint="after"))
        orchestrator = self.orchestrator(planner, qwen, adapter)
        with tempfile.TemporaryDirectory() as temp:
            session = orchestrator.start(
                session_id="session-stale",
                raw_goal="查看详情",
                device_id="device-1",
                run_dir=Path(temp),
            )
            stale = _confirmation(session)
            stale["fingerprint"] = "old-frame"
            with self.assertRaisesRegex(
                UniversalAgentOrchestratorError, "scope"
            ):
                orchestrator.confirm_one(session, stale)

        self.assertEqual(0, adapter.execute_calls)
        self.assertEqual(0, session.physical_actions)


class ObservationBridgeTests(unittest.TestCase):
    def test_projection_contains_goal_not_business_steps(self) -> None:
        payload = ObservationBridge().goal_draft(_graph()).to_dict()
        self.assertEqual("gallery", payload["app_id"])
        self.assertNotIn("steps", str(payload).casefold())
        self.assertNotIn("coordinate", str(payload).casefold())


if __name__ == "__main__":
    unittest.main()
