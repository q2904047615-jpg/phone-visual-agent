from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from PIL import Image

from agent.application.action_adapter import AppLaunchTarget, GenericActionAdapterError
from agent.application.universal_agent_orchestrator import (
    ObservationBridge,
    UniversalAgentOrchestrator,
    UniversalAgentOrchestratorError,
)
from agent.domain.canonical_action_protocol import (
    GenericStepProposal,
    bind_same_response_action,
    normalize_model_step_decision,
)
from agent.domain.task_graph import (
    CompletionCondition,
    DynamicTaskGraph,
    GraphGoal,
    RiskAction,
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


def _input_scene(
    *,
    fingerprint: str = "input-frame-a",
    value: str = "",
    label: str = "消息输入框",
    element_id: str = "message-input",
    input_field_id: str = "primary_input",
    bounds: tuple[float, float, float, float] = (0.08, 0.78, 0.82, 0.9),
    focused: bool = True,
    state_overrides: dict | None = None,
) -> UIScene:
    states = {
        "goal_relevant": True,
        "fully_visible": True,
        "enabled": True,
        "focused": focused,
        "value": value,
        "input_field_id": input_field_id,
        "primary_input_geometry_verified": True,
        "geometry_audit_source": "input_structure_audit",
    }
    states.update(state_overrides or {})
    return UIScene(
        app_id="messenger",
        screen_id="conversation",
        summary=f"{label}当前可见",
        elements=(
            UIElement(
                element_id=element_id,
                role="input",
                meaning="application_text_input",
                label=label,
                bounds=bounds,
                confidence=0.97,
                states=states,
                evidence=(f"{label}边界清晰可见",),
            ),
        ),
        stable=True,
        confidence=0.96,
        fingerprint=fingerprint,
        system_ui=SystemUIFacts(),
    )


def _same_frame_model_decision(scene: UIScene, *, status: str,
    action_kind: str) -> dict:
    if status == "finish":
        return {
            "status": "finish",
            "evidence_refs": ["scene.summary"],
            "confidence": 0.95,
            "reason": "当前同帧 scene 已证明目标完成。",
        }
    if status != "action":
        return {"status": status}
    payload = {
        "status": "action",
        "action": action_kind,
        "confidence": 0.95,
        "reason": "当前同帧 scene 选择一个 canonical 动作。",
    }
    if action_kind in {
        "tap_semantic",
        "dismiss_overlay",
        "input_verified_text",
        "press_enter",
        "clear_verified_text",
        "double_tap",
        "long_press",
    }:
        if len(scene.elements) != 1:
            raise AssertionError("测试 scene 必须只有一个同帧动作目标。")
        payload["element_id"] = scene.elements[0].element_id
    elif action_kind == "swipe":
        payload["direction"] = "up"
    return payload


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


def _effect_graph(
    kind: str,
    *,
    confirmation_required: bool = True,
    device_id: str = "device-1",
) -> DynamicTaskGraph:
    graph = DynamicTaskGraph(
        task_id=f"task-{kind}",
        device_id=device_id,
        revision=1,
        status="awaiting_confirmation" if confirmation_required else "running",
        goal=GraphGoal(
            objective=f"完成一次{kind}目标",
            target_apps=(TargetApp(app_id="gallery", app_name="图片工具"),),
            entities={},
        ),
        constraints=(),
        completion_conditions=(
            CompletionCondition(
                condition_id="condition-1",
                description="当前页面显示目标已经完成",
                evidence_required=("完成状态可见",),
            ),
        ),
        risk_actions=(
            RiskAction(
                risk_id="effect-1",
                subgoal_ids=("subgoal-1",),
                confirmation_required=confirmation_required,
                effect_kind=kind,
                expected_result_texts=("当前页面显示目标已经完成",),
            ),
        ),
        subgoals=(
            Subgoal(
                subgoal_id="subgoal-1",
                objective=f"执行{kind}动作",
                status="active",
                depends_on=(),
                constraints=(),
                completion_conditions=("当前页面显示目标已经完成",),
                completion_evidence=(),
                risk_action_ids=("effect-1",),
                external_impact="external_state",
            ),
        ),
        active_subgoal_id="subgoal-1",
        raw_user_goal=f"完成一次{kind}目标",
    )
    graph.validate()
    return graph


class FakeDeepSeekPlanner:
    """DeepSeek is invoked once to define the typed high-level goal."""

    def __init__(self, graph: DynamicTaskGraph) -> None:
        self.graph = graph
        self.plan_calls: list[tuple] = []

    def plan(self, raw_goal, *, device_id, task_id=None):
        self.plan_calls.append((raw_goal, device_id, task_id))
        return self.graph


class FakeTrustedObservation:
    def __init__(self, *, device_id: str, scene: UIScene, observation_id="obs-start"):
        self.device_id = device_id
        self.scene = scene
        self.observation_id = observation_id
        self.fingerprint = scene.fingerprint
        self.candidate_conflicts = ()

    def get_candidate(self, element_id: str):
        return self.scene.get_element(element_id)

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
    def __init__(
        self,
        scene: UIScene,
        *,
        launch_target: AppLaunchTarget | None = None,
    ) -> None:
        self.scene = scene
        self.launch_target = launch_target
        self.capture_calls = 0
        self.execute_calls = 0
        self.initial_model_decision = _same_frame_model_decision(
            scene, status="action", action_kind="tap_semantic"
        )
        self.followup_model_decision = dict(self.initial_model_decision)

    def configure_model_decisions(self, *, status: str, after_status: str,
        action_kind: str, after_action_kind: str | None = None) -> None:
        self.initial_model_decision = _same_frame_model_decision(
            self.scene, status=status, action_kind=action_kind
        )
        followup_scene = getattr(self, "after_scene", self.scene)
        self.followup_model_decision = _same_frame_model_decision(
            followup_scene,
            status=after_status,
            action_kind=after_action_kind or action_kind,
        )

    def capture_scene(self, goal, *, evidence_dir, prefix):
        del goal
        self.capture_calls += 1
        frames = [Image.new("RGB", (540, 960), "white") for _ in range(4)]
        model_decision = (
            self.initial_model_decision
            if self.capture_calls == 1
            else self.followup_model_decision
        )
        return self.scene, frames, tuple(
            str(evidence_dir / f"{prefix}_{index}.jpg") for index in range(1, 5)
        ), dict(model_decision)

    def execute(self, **_kwargs):
        self.execute_calls += 1
        raise AssertionError("start must not execute a physical action")

    def resolve_app_launch_target(self, app_id: str, app_name: str):
        self.launch_resolution = (app_id, app_name)
        return self.launch_target


class FakeExecutingAdapter(FakeAdapter):
    def __init__(
        self,
        scene: UIScene,
        after_scene: UIScene,
        *,
        execute_error: GenericActionAdapterError | None = None,
        action_outcome: str = "matched",
        verification_errors: tuple[str, ...] = (),
        launch_target: AppLaunchTarget | None = None,
    ) -> None:
        super().__init__(scene, launch_target=launch_target)
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
        params = requested_action.params
        return GenericActionExecutionResult(
            requested_action=requested_action,
            rebound_action=requested_action,
            resolved_action=ResolvedSemanticAction(
                node_id=requested_action.node_id,
                kind=requested_action.action,
                normalized_point=(0.3, 0.25),
                text=params.get("text"),
                input_fragment=params.get("input_fragment"),
                text_transport=params.get("text_transport"),
                input_field_id=params.get("input_field_id"),
                prior_input_value=params.get("prior_input_value"),
                expected_input_value=params.get("expected_input_value"),
                launch_ref=params.get("launch_ref"),
                expected_package_id=params.get("expected_app_id"),
                target_app_id=params.get("target_app_id"),
                target_app_name=params.get("target_app_name"),
                target_element_id=str(params.get("element_id") or ""),
                before_fingerprint=planned_scene.fingerprint,
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
            after_model_decision=self.followup_model_decision,
            after_frame_paths=(
                "after-1.jpg",
                "after-2.jpg",
                "after-3.jpg",
                "after-4.jpg",
            ),
        )


class StaleOnceExecutingAdapter(FakeExecutingAdapter):
    """Reject one old screenshot before any physical action."""

    def execute(self, **kwargs):
        if self.execute_error is not None:
            error = self.execute_error
            self.execute_error = None
            self.execute_calls += 1
            raise error
        return super().execute(**kwargs)


class FocusCorrectionExecutingAdapter(FakeExecutingAdapter):
    """Use a third scene for the one read-only correction after a focus click."""

    def __init__(
        self,
        scene: UIScene,
        after_scene: UIScene,
        correction_scene: UIScene,
    ) -> None:
        super().__init__(scene, after_scene)
        self.correction_scene = correction_scene

    def capture_scene(self, goal, *, evidence_dir, prefix):
        if self.capture_calls:
            self.scene = self.correction_scene
            self.followup_model_decision = _same_frame_model_decision(
                self.correction_scene,
                status="action",
                action_kind="tap_semantic",
            )
        return super().capture_scene(goal, evidence_dir=evidence_dir, prefix=prefix)


class FakeQwenObserver:
    """Emit one scripted decision per fresh observation; never invoke DeepSeek."""

    def __init__(
        self,
        status="action",
        *,
        after_status="finish",
        mutate_identity=None,
        action_kind="tap_semantic",
        after_action_kind=None,
    ) -> None:
        self.status = status
        self.after_status = after_status
        self.mutate_identity = mutate_identity
        self.action_kind = action_kind
        self.after_action_kind = after_action_kind
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
        model_decision,
    ):
        self.calls.append((frames, task_context, trusted_observation, decision_number))
        self.text_transport_profiles.append(text_transport_profile)
        current_status = model_decision.get("status") if isinstance(model_decision, dict) else None
        if current_status in {"action", "finish"}:
            payload = normalize_model_step_decision(model_decision)
        else:
            payload = dict(model_decision or {})
        completion_evidence = ()
        if current_status == "action":
            action = bind_same_response_action(
                payload,
                context=task_context,
                observation=trusted_observation,
                available_action_kinds=available_action_kinds or (),
                launch_target=launch_target,
                text_transport_profile=text_transport_profile,
            )
            proposal = GenericStepProposal(
                status="action",
                action=action,
                reason=payload["reason"],
            )
        elif current_status == "finish":
            proposal = GenericStepProposal(
                status="finish",
                reason=payload["reason"],
            )
            completion_evidence = (trusted_observation.scene.summary,)
        else:
            proposal = SimpleNamespace(
                status=current_status,
                action=None,
                reason="旧模型返回了协议外第三种状态。",
                validate=lambda _scene: None,
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

class FailOnNthQwenObserver(FakeQwenObserver):
    def __init__(self, fail_on_call: int, *, error: Exception | None = None) -> None:
        super().__init__(after_status="action")
        self.fail_on_call = fail_on_call
        self.error = error or RuntimeError(f"第 {fail_on_call} 次 Qwen 观察失败")

    def decide(self, **kwargs):
        decision = super().decide(**kwargs)
        if len(self.calls) == self.fail_on_call:
            raise self.error
        return decision


def _confirmation(session) -> dict:
    return dict(session.snapshot()["confirmation_scope"])


class UniversalAgentSingleAuthorityLoopTests(unittest.TestCase):
    @staticmethod
    def orchestrator(planner, qwen, adapter):
        adapter.configure_model_decisions(
            status=qwen.status,
            after_status=qwen.after_status,
            action_kind=qwen.action_kind,
            after_action_kind=qwen.after_action_kind,
        )
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
        self.assertEqual(1, len(qwen.calls))
        self.assertEqual(0, adapter.execute_calls)

    def test_action_then_new_screenshot_finish_completes_in_one_loop(self) -> None:
        planner = FakeDeepSeekPlanner(_graph())
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
        self.assertEqual(1, len(planner.plan_calls))
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
        self.assertEqual(1, len(planner.plan_calls))

    def test_automatic_send_effect_rejects_a_second_action_after_one_physical_effect(self) -> None:
        planner = FakeDeepSeekPlanner(
            _effect_graph("send_message", confirmation_required=False)
        )
        qwen = FakeQwenObserver(after_status="action", action_kind="tap_semantic")
        adapter = FakeExecutingAdapter(
            _scene(meaning="send_message", label="发送"),
            _scene(
                fingerprint="send-effect-after",
                meaning="send_message",
                label="发送",
            ),
        )
        orchestrator = self.orchestrator(planner, qwen, adapter)

        with tempfile.TemporaryDirectory() as temp:
            session = orchestrator.start(
                session_id="session-send-effect-action-after-action",
                raw_goal="发送当前已准备的内容",
                device_id="device-1",
                run_dir=Path(temp),
            )
            self.assertIsNone(session.effect_confirmation_authority)
            self.assertEqual(("effect-1",), session.confirmation_authority.effect_ids)
            result = orchestrator.run_autonomous_safe_loop(session)

        self.assertEqual("failed", result["status"])
        self.assertEqual("failed", session.status)
        self.assertIn("外部效果子目标已执行一次物理动作", session.failed_reason)
        self.assertIn("未自动重复", session.failed_reason)
        self.assertEqual(1, result["physical_actions"])
        self.assertEqual(1, adapter.execute_calls)
        self.assertEqual(1, session.physical_actions)
        self.assertEqual(2, len(qwen.calls))
        self.assertEqual(1, len(adapter.action_authorities))
        self.assertEqual(("effect-1",), adapter.action_authorities[0].effect_ids)
        self.assertIsNone(session.confirmation_authority)
        self.assertIsNone(session.controller_decision)

    def test_automatic_send_effect_accepts_finish_after_exactly_one_physical_effect(self) -> None:
        planner = FakeDeepSeekPlanner(
            _effect_graph("send_message", confirmation_required=False)
        )
        qwen = FakeQwenObserver(after_status="finish", action_kind="tap_semantic")
        adapter = FakeExecutingAdapter(
            _scene(meaning="send_message", label="发送"),
            _scene(
                fingerprint="send-effect-finished",
                meaning="sent_content",
                label="已发送内容",
            ),
        )
        orchestrator = self.orchestrator(planner, qwen, adapter)

        with tempfile.TemporaryDirectory() as temp:
            session = orchestrator.start(
                session_id="session-send-effect-finish-after-action",
                raw_goal="发送当前已准备的内容",
                device_id="device-1",
                run_dir=Path(temp),
            )
            result = orchestrator.run_autonomous_safe_loop(session)

        self.assertEqual("succeeded", result["status"])
        self.assertEqual("succeeded", session.status)
        self.assertEqual(1, result["physical_actions"])
        self.assertEqual(1, adapter.execute_calls)
        self.assertEqual(1, session.physical_actions)
        self.assertEqual(2, len(qwen.calls))
        self.assertEqual(1, len(adapter.action_authorities))
        self.assertEqual(("effect-1",), adapter.action_authorities[0].effect_ids)
        self.assertIsNone(session.confirmation_authority)

    def test_automatic_publish_effect_rejects_a_different_followup_action(self) -> None:
        planner = FakeDeepSeekPlanner(
            _effect_graph("publish_content", confirmation_required=False)
        )
        qwen = FakeQwenObserver(
            after_status="action",
            action_kind="tap_semantic",
            after_action_kind="swipe",
        )
        adapter = FakeExecutingAdapter(
            _scene(meaning="publish_content", label="发布"),
            _scene(
                fingerprint="publish-effect-after",
                meaning="published_content",
                label="发布后页面",
            ),
        )
        orchestrator = self.orchestrator(planner, qwen, adapter)

        with tempfile.TemporaryDirectory() as temp:
            session = orchestrator.start(
                session_id="session-publish-effect-different-followup",
                raw_goal="发布当前已准备的内容",
                device_id="device-1",
                run_dir=Path(temp),
            )
            result = orchestrator.run_autonomous_safe_loop(session)

        self.assertEqual("failed", result["status"])
        self.assertEqual("failed", session.status)
        self.assertIn("外部效果子目标已执行一次物理动作", session.failed_reason)
        self.assertIn("未自动重复", session.failed_reason)
        self.assertEqual(1, result["physical_actions"])
        self.assertEqual(1, adapter.execute_calls)
        self.assertEqual(1, session.physical_actions)
        self.assertEqual(2, len(qwen.calls))
        self.assertEqual(1, len(adapter.action_authorities))
        self.assertEqual(("effect-1",), adapter.action_authorities[0].effect_ids)
        self.assertIsNone(session.confirmation_authority)
        self.assertIsNone(session.controller_decision)

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
        self.assertEqual(1, len(planner.plan_calls))

    def test_authentication_and_payment_wait_for_one_explicit_effect_confirmation(self) -> None:
        for kind in ("authentication", "financial_transaction"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                planner = FakeDeepSeekPlanner(_effect_graph(kind))
                qwen = FakeQwenObserver(after_status="finish")
                adapter = FakeExecutingAdapter(
                    _scene(),
                    _scene(fingerprint=f"{kind}-after", label="完成状态"),
                )
                orchestrator = self.orchestrator(planner, qwen, adapter)
                session = orchestrator.start(
                    session_id=f"session-{kind}",
                    raw_goal=f"完成一次{kind}目标",
                    device_id="device-1",
                    run_dir=Path(temp),
                )

                self.assertEqual("awaiting_effect_confirmation", session.status)
                self.assertEqual(0, adapter.capture_calls)
                self.assertEqual(0, len(qwen.calls))
                effect_scope = dict(session.snapshot()["effect_confirmation_scope"])
                orchestrator.approve_effects(session, effect_scope)

                self.assertEqual("succeeded", session.status)
                self.assertEqual(1, adapter.execute_calls)
                self.assertEqual(1, session.physical_actions)
                self.assertEqual(2, len(qwen.calls))
                self.assertIsNone(
                    orchestrator.device_registry.active_session(session.device_id)
                )

    def test_launch_action_uses_only_trusted_registry_mapping(self) -> None:
        planner = FakeDeepSeekPlanner(_graph())
        qwen = FakeQwenObserver(action_kind="launch_app", after_status="finish")
        adapter = FakeExecutingAdapter(
            _scene(app_id="launcher"),
            _scene(fingerprint="gallery-open", app_id="gallery", label="图片工具首页"),
            launch_target=AppLaunchTarget(
                launch_ref="trusted.registry.gallery",
                expected_app_id="com.example.gallery",
            ),
        )
        orchestrator = self.orchestrator(planner, qwen, adapter)
        with tempfile.TemporaryDirectory() as temp:
            session = orchestrator.start(
                session_id="session-launch",
                raw_goal="打开图片工具",
                device_id="device-1",
                run_dir=Path(temp),
            )
            action = session.qwen_decision.proposal.action
            self.assertEqual("launch_app", action.action)
            self.assertEqual("trusted.registry.gallery", action.params["launch_ref"])
            self.assertEqual("com.example.gallery", action.params["expected_app_id"])
            self.assertEqual(("gallery", "图片工具"), adapter.launch_resolution)
            orchestrator.confirm_one(session, _confirmation(session))

        self.assertEqual("succeeded", session.status)
        self.assertEqual(1, adapter.execute_calls)

    def test_exact_input_uses_typed_current_subgoal(self) -> None:
        exact_text = "aaazjie？你好"
        planner = FakeDeepSeekPlanner(_graph())
        qwen = FakeQwenObserver(
            action_kind="input_verified_text",
            after_status="finish",
        )
        adapter = FakeExecutingAdapter(
            _input_scene(),
            _input_scene(fingerprint="input-after", value=exact_text),
        )
        orchestrator = self.orchestrator(planner, qwen, adapter)
        with tempfile.TemporaryDirectory() as temp:
            session = orchestrator.start(
                session_id="session-exact-input",
                raw_goal="在当前输入框输入指定文字",
                exact_input_text=exact_text,
                device_id="device-1",
                run_dir=Path(temp),
            )
            action = session.qwen_decision.proposal.action
            self.assertEqual("input_verified_text", action.action)
            self.assertEqual("primary_input", action.params["input_field_id"])
            self.assertEqual(exact_text, action.params["text"])
            self.assertEqual("", action.params["prior_input_value"])
            self.assertEqual(exact_text, action.params["expected_input_value"])
            orchestrator.confirm_one(session, _confirmation(session))

        self.assertEqual("succeeded", session.status)
        self.assertEqual(0, len(planner.plan_calls))
        self.assertEqual(1, session.physical_actions)

    def test_public_snapshot_keeps_compatibility_keys_without_runtime_authority(self) -> None:
        planner = FakeDeepSeekPlanner(_graph())
        qwen = FakeQwenObserver()
        adapter = FakeAdapter(_scene())
        with tempfile.TemporaryDirectory() as temp:
            session = self.orchestrator(planner, qwen, adapter).start(
                session_id="session-snapshot",
                raw_goal="查看详情",
                device_id="device-1",
                run_dir=Path(temp),
            )
            snapshot = session.snapshot()

        self.assertEqual("awaiting_confirmation", snapshot["status"])
        self.assertTrue(snapshot["confirmation_ready"])
        self.assertEqual([], snapshot["corrective_retry_history"])
        self.assertIsNone(snapshot["corrective_retry_protocol"])
        self.assertIsNone(snapshot["verified_app_surface_lineage"])
        self.assertIsNone(snapshot["effect_verification"])
        self.assertEqual("tap_semantic", snapshot["proposal"]["action"]["action"])

    def test_hard_adapter_error_is_terminal_and_releases_device_lease(self) -> None:
        planner = FakeDeepSeekPlanner(_graph())
        qwen = FakeQwenObserver()
        adapter = FakeExecutingAdapter(
            _scene(),
            _scene(fingerprint="unused-after"),
            execute_error=GenericActionAdapterError(
                "机械执行后未取得可信回执",
                physical_actions=1,
                evidence=("execution-error.json",),
            ),
        )
        orchestrator = self.orchestrator(planner, qwen, adapter)
        with tempfile.TemporaryDirectory() as temp:
            session = orchestrator.start(
                session_id="session-hard-adapter-error",
                raw_goal="查看详情",
                device_id="device-1",
                run_dir=Path(temp),
            )
            with self.assertRaisesRegex(
                GenericActionAdapterError,
                "机械执行后未取得可信回执",
            ):
                orchestrator.confirm_one(session, _confirmation(session))

        self.assertEqual("failed", session.status)
        self.assertEqual(1, session.physical_actions)
        self.assertEqual("机械执行后未取得可信回执", session.failed_reason)
        self.assertIsNone(orchestrator.device_registry.active_session(session.device_id))

    def test_autonomous_loop_discards_zero_action_stale_frame_and_reobserves(self) -> None:
        planner = FakeDeepSeekPlanner(_graph())
        qwen = FakeQwenObserver(after_status="finish")
        adapter = StaleOnceExecutingAdapter(
            _scene(),
            _scene(fingerprint="unused-after"),
            execute_error=GenericActionAdapterError(
                "确认时本地真实画面已变化",
                physical_actions=0,
                evidence=("fresh-frame.jpg",),
            ),
        )
        orchestrator = self.orchestrator(planner, qwen, adapter)
        with tempfile.TemporaryDirectory() as temp:
            session = orchestrator.start(
                session_id="session-stale-reobserve",
                raw_goal="查看详情",
                device_id="device-1",
                run_dir=Path(temp),
            )
            result = orchestrator.run_autonomous_safe_loop(session)

        self.assertEqual("succeeded", result["status"])
        self.assertEqual(0, result["physical_actions"])
        self.assertEqual(2, adapter.capture_calls)
        self.assertEqual(1, adapter.execute_calls)
        self.assertEqual(2, len(qwen.calls))
        self.assertIsNot(qwen.calls[0][0][0], qwen.calls[1][0][0])
        self.assertEqual("finish", session.qwen_decision.proposal.status)
        self.assertIsNone(session.confirmation_authority)

    def test_autonomous_loop_stops_repeated_unprogressed_input_focus_across_visual_drift(self) -> None:
        """Caret/keyboard state drift may not reopen one already-clicked typed field."""
        initial_scene = _input_scene(
            fingerprint="focus-before",
            element_id="input-visible-1",
            bounds=(0.08, 0.78, 0.82, 0.9),
            focused=False,
            state_overrides={
                "soft_keyboard_visible": False,
                "keyboard_layout": "unknown",
                "keyboard_input_mode": "unknown",
            },
        )
        after_scene = _input_scene(
            fingerprint="focus-after",
            element_id="input-renumbered-42",
            bounds=(0.085, 0.779, 0.823, 0.902),
            focused=True,
            state_overrides={
                "soft_keyboard_visible": True,
                "keyboard_layout": "qwerty",
                "keyboard_input_mode": "direct_latin",
            },
        )
        correction_scene = _input_scene(
            fingerprint="focus-correction-caret-blink",
            element_id="input-renumbered-99",
            bounds=(0.083, 0.781, 0.825, 0.904),
            focused=False,
            state_overrides={
                "soft_keyboard_visible": False,
                "keyboard_layout": "unknown",
                "keyboard_input_mode": "unknown",
            },
        )
        qwen = FakeQwenObserver(after_status="action", action_kind="tap_semantic")
        adapter = FocusCorrectionExecutingAdapter(
            initial_scene,
            after_scene,
            correction_scene,
        )
        orchestrator = self.orchestrator(FakeDeepSeekPlanner(_graph()), qwen, adapter)

        with tempfile.TemporaryDirectory() as temp:
            session = orchestrator.start(
                session_id="session-stop-repeated-input-focus",
                raw_goal="在当前输入框输入指定文字",
                exact_input_text="cross-app text",
                device_id="device-1",
                run_dir=Path(temp),
            )
            with self.assertRaisesRegex(
                UniversalAgentOrchestratorError,
                "输入框聚焦动作.*仍无推进.*未重复点击",
            ):
                orchestrator.run_autonomous_safe_loop(
                    session,
                    max_physical_actions=6,
                    max_iterations=12,
                )

        self.assertEqual(1, adapter.execute_calls)
        self.assertEqual(1, session.physical_actions)
        self.assertEqual(2, adapter.capture_calls)
        self.assertEqual(3, len(qwen.calls))
        self.assertEqual("failed", session.status)
        self.assertEqual("primary_input", initial_scene.elements[0].states["input_field_id"])
        self.assertEqual("primary_input", after_scene.elements[0].states["input_field_id"])
        self.assertEqual("primary_input", correction_scene.elements[0].states["input_field_id"])

    def test_fresh_focus_progress_can_advance_to_input_instead_of_being_repeat_blocked(self) -> None:
        initial_scene = _input_scene(
            fingerprint="focus-progress-before",
            element_id="input-before",
            focused=False,
        )
        after_scene = _input_scene(
            fingerprint="focus-progress-after",
            element_id="input-after",
            bounds=(0.081, 0.781, 0.821, 0.901),
            focused=True,
        )
        qwen = FakeQwenObserver(
            after_status="action",
            action_kind="tap_semantic",
            after_action_kind="input_verified_text",
        )
        adapter = FakeExecutingAdapter(initial_scene, after_scene)
        orchestrator = self.orchestrator(FakeDeepSeekPlanner(_graph()), qwen, adapter)

        with tempfile.TemporaryDirectory() as temp:
            session = orchestrator.start(
                session_id="session-focus-progresses-to-input",
                raw_goal="在当前输入框输入指定文字",
                exact_input_text="cross-app text",
                device_id="device-1",
                run_dir=Path(temp),
            )
            orchestrator.confirm_one(session, _confirmation(session))

        self.assertEqual(1, adapter.execute_calls)
        self.assertEqual(1, session.physical_actions)
        self.assertEqual(2, len(qwen.calls))
        self.assertEqual("awaiting_confirmation", session.status)
        self.assertEqual(
            "input_verified_text",
            session.qwen_decision.proposal.action.action,
        )
        self.assertEqual(
            "primary_input",
            session.qwen_decision.proposal.action.params["input_field_id"],
        )

    def test_focused_input_is_not_physically_tapped_again_when_qwen_repeats_focus(self) -> None:
        initial_scene = _input_scene(fingerprint="focus-before", focused=False)
        focused_scene = _input_scene(
            fingerprint="focus-after",
            element_id="renumbered-focused-input",
            bounds=(0.082, 0.779, 0.824, 0.903),
            focused=True,
        )
        qwen = FakeQwenObserver(after_status="action", action_kind="tap_semantic")
        adapter = FakeExecutingAdapter(initial_scene, focused_scene)
        orchestrator = self.orchestrator(FakeDeepSeekPlanner(_graph()), qwen, adapter)

        with tempfile.TemporaryDirectory() as temp:
            session = orchestrator.start(
                session_id="session-no-retap-focused-input",
                raw_goal="在当前输入框输入指定文字",
                exact_input_text="cross-app text",
                device_id="device-1",
                run_dir=Path(temp),
            )
            with self.assertRaisesRegex(
                UniversalAgentOrchestratorError,
                "输入框聚焦动作.*仍无推进.*未重复点击",
            ):
                orchestrator.run_autonomous_safe_loop(session)

        self.assertEqual(1, adapter.execute_calls)
        self.assertEqual(1, session.physical_actions)
        self.assertEqual(2, adapter.capture_calls)
        self.assertEqual(3, len(qwen.calls))

    def test_autonomous_loop_keeps_physical_action_error_terminal(self) -> None:
        planner = FakeDeepSeekPlanner(_graph())
        qwen = FakeQwenObserver(after_status="finish")
        adapter = FakeExecutingAdapter(
            _scene(),
            _scene(fingerprint="unused-after"),
            execute_error=GenericActionAdapterError(
                "机械执行后未取得可信回执",
                physical_actions=1,
            ),
        )
        orchestrator = self.orchestrator(planner, qwen, adapter)
        with tempfile.TemporaryDirectory() as temp:
            session = orchestrator.start(
                session_id="session-autonomous-hard-error",
                raw_goal="查看详情",
                device_id="device-1",
                run_dir=Path(temp),
            )
            with self.assertRaisesRegex(GenericActionAdapterError, "未取得可信回执"):
                orchestrator.run_autonomous_safe_loop(session)

        self.assertEqual("failed", session.status)
        self.assertEqual(1, session.physical_actions)
        self.assertIsNone(orchestrator.device_registry.active_session(session.device_id))

    def test_autonomous_loop_keeps_zero_action_non_stale_error_terminal(self) -> None:
        planner = FakeDeepSeekPlanner(_graph())
        qwen = FakeQwenObserver(after_status="finish")
        adapter = FakeExecutingAdapter(
            _scene(),
            _scene(fingerprint="unused-after"),
            execute_error=GenericActionAdapterError(
                "确认前控制器拒绝动作：输入字段硬合同不匹配",
                physical_actions=0,
            ),
        )
        orchestrator = self.orchestrator(planner, qwen, adapter)
        with tempfile.TemporaryDirectory() as temp:
            session = orchestrator.start(
                session_id="session-zero-action-hard-error",
                raw_goal="查看详情",
                device_id="device-1",
                run_dir=Path(temp),
            )
            with self.assertRaisesRegex(GenericActionAdapterError, "硬合同不匹配"):
                orchestrator.run_autonomous_safe_loop(session)

        self.assertEqual("failed", session.status)
        self.assertEqual(0, session.physical_actions)
        self.assertEqual(1, len(qwen.calls))
        self.assertIsNone(orchestrator.device_registry.active_session(session.device_id))

    def test_nth_step_failure_is_persisted_and_releases_device_lease(self) -> None:
        planner = FakeDeepSeekPlanner(_graph())
        qwen = FailOnNthQwenObserver(3)
        adapter = FakeExecutingAdapter(
            _scene(), _scene(fingerprint="frame-after", label="下一入口")
        )
        orchestrator = self.orchestrator(planner, qwen, adapter)
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp)
            session = orchestrator.start(
                session_id="session-nth-step-failure",
                raw_goal="连续查看详情",
                device_id="device-1",
                run_dir=run_dir,
            )
            self.assertEqual(
                session.session_id,
                orchestrator.device_registry.active_session(session.device_id),
            )

            with self.assertRaisesRegex(RuntimeError, "第 3 次 Qwen 观察失败"):
                orchestrator.run_autonomous_safe_loop(
                    session,
                    max_physical_actions=12,
                    max_iterations=24,
                )

            report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))

        self.assertEqual(3, len(qwen.calls))
        self.assertEqual(2, session.physical_actions)
        self.assertEqual("failed", session.status)
        self.assertEqual("第 3 次 Qwen 观察失败", session.failed_reason)
        self.assertFalse(session.automatic_loop_enabled)
        self.assertIsNone(orchestrator.device_registry.active_session(session.device_id))
        self.assertEqual("failed", report["session"]["status"])
        self.assertEqual(
            "第 3 次 Qwen 观察失败",
            report["session"]["failed_reason"],
        )
        self.assertFalse(report["session"]["automatic_loop_enabled"])

    def test_obsolete_model_blocked_status_is_rejected(self) -> None:
        planner = FakeDeepSeekPlanner(_graph())
        qwen = FakeQwenObserver(status="blocked")
        adapter = FakeAdapter(_scene())
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(
                UniversalAgentOrchestratorError, "只允许action或finish"
            ):
                self.orchestrator(planner, qwen, adapter).start(
                    session_id="session-obsolete-blocked",
                    raw_goal="查看详情",
                    device_id="device-1",
                    run_dir=Path(temp),
                )

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
