from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from deepseek_task_graph import (
    CompletionCondition,
    ControllerTransitionEvidenceRef,
    DynamicTaskGraph,
    GraphGoal,
    RiskAction,
    Subgoal,
    TaskGraphError,
    TargetApp,
    VerifiedActionTransition,
    _graph_from_payload,
)
from generic_step_planner import GenericStepProposal
from generic_scene_observer import _safe_goal_context
from generic_action_adapter import (
    GenericActionAdapterError,
    GenericActionExecutionResult,
)
from semantic_action import SemanticAction
from ui_scene import SystemUIFacts, UIElement, UIScene
from universal_action_controller import ResolvedSemanticAction
from vision_agent import VisionAgentError
from qwen_visual_decision import QwenTaskContext, _scene_matches_target_app_surface
from task_semantic_ir import compile_formal_semantic_authority
from canonical_action_protocol import compile_canonical_action_catalog
from universal_agent_orchestrator import (
    AgentEvidenceStore,
    EvidenceStoreError,
    DeviceTaskRegistry,
    ObservationBridge,
    PhaseOneNavigationPolicy,
    UniversalAgentOrchestrator,
    UniversalAgentOrchestratorError,
    UniversalAgentSessionState,
    VerifiedAppSurfaceLineage,
    _action_digest,
    _action_equivalence_digest,
    _validate_visible_completion_condition_progress,
)


def _scene(
    *,
    fingerprint: str = "frame-a",
    meaning: str = "open_details",
    label: str = "查看详情",
    role: str = "button",
    bounds: tuple[float, float, float, float] = (0.1, 0.2, 0.5, 0.3),
    confidence: float = 0.96,
    scene_confidence: float = 0.95,
    states: dict | None = None,
    evidence: tuple[str, ...] = ("画面中可见目标",),
    system_ui=None,
    app_id: str = "gallery",
) -> UIScene:
    current = UIScene(
        app_id=app_id,
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
                states={
                    "goal_relevant": True,
                    "fully_visible": True,
                    "scrollable": True,
                    **(states or {}),
                },
                evidence=evidence,
            ),
        ),
        stable=True,
        confidence=scene_confidence,
        fingerprint=fingerprint,
        system_ui=system_ui or SystemUIFacts(),
    )
    return current


def _decision(
    scene: UIScene,
    *,
    action_kind: str = "tap_semantic",
    direction: str = "up",
    decision_confidence: float = 0.94,
    goal_complete_on_success: bool = False,
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
    if goal_complete_on_success:
        params["expected_effect"]["goal_complete_on_success"] = True
    if element.states:
        params["states"] = dict(element.states)
    if action_kind == "swipe":
        params = {
            "direction": direction,
            "expected_effect": {"content_changed": True},
        }
    elif action_kind == "reveal_system_navigation":
        params = {
            "expected_effect": {
                "system_ui": {"navigation_bar_visible": True}
            }
        }
    elif action_kind in {"back", "home", "wait_for_change"}:
        params = {
            "expected_effect": {
                "scene_changed": action_kind in {"back", "home"},
            }
        }
    elif action_kind == "input_verified_text":
        params["text"] = "agent"
        params["states"] = dict(element.states)
        params["expected_effect"] = {
            "element_state": {
                "meaning": element.meaning,
                "states": {"value": "agent"},
            }
        }
    elif action_kind == "clear_verified_text":
        params["states"] = dict(element.states)
        params["expected_effect"] = {
            "element_state": {
                "meaning": element.meaning,
                "states": {"value": ""},
            }
        }
    elif action_kind == "long_press":
        params["duration_ms"] = 800
    action = SemanticAction(node_id="node-1", action=action_kind, params=params)
    observation = SimpleNamespace(
        device_id="device-1",
        fingerprint=scene.fingerprint,
        scene=scene,
        candidate_conflicts=(),
    )
    observation.target_local_candidate = scene.unique_trusted_goal_element
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
                if action_kind
                in {
                    "tap_semantic", "dismiss_overlay", "input_verified_text",
                    "clear_verified_text", "long_press",
                }
                else "system_navigation"
                if action_kind in {"back", "home", "reveal_system_navigation"}
                else "screen"
            ),
            element_id=(
                element.element_id
                if action_kind
                in {
                    "tap_semantic", "dismiss_overlay", "input_verified_text",
                    "clear_verified_text", "long_press",
                }
                else ""
            ),
            bounds=(
                element.bounds
                if action_kind
                in {
                    "tap_semantic", "dismiss_overlay", "input_verified_text",
                    "clear_verified_text", "long_press",
                }
                else (0.0, 0.0, 1.0, 1.0)
            ),
        ),
    )


def _context(
    *,
    impact: str = "navigation_only",
    external_action_allowed: bool = False,
    target_apps: tuple[dict, ...] = (),
    entities: dict | None = None,
    subgoal_objective: str = "打开目标详情",
    subgoal_constraints: tuple[str, ...] = (),
    subgoal_completion_conditions: tuple[str, ...] = ("目标详情可见",),
    risk_actions: tuple[dict, ...] = (),
    risk_action_ids: tuple[str, ...] = (),
) -> SimpleNamespace:
    execution_class = {
        "read_only": "observe",
        "navigation_only": "navigate",
        "external_state": "effect",
        "unknown": "unknown",
    }.get(impact, impact)
    return SimpleNamespace(
        task_id="task-1",
        device_id="device-1",
        revision=1,
        current_execution_class=execution_class,
        effect_action_allowed=external_action_allowed,
        goal={
            "target_apps": list(target_apps),
            "entities": dict(entities if entities is not None else {"target": "目标详情"}),
        },
        current_subgoal={
            "subgoal_id": "subgoal-1",
            "objective": subgoal_objective,
            "status": "active",
            "constraints": list(subgoal_constraints),
            "completion_conditions": list(subgoal_completion_conditions),
            "effect_ids": list(risk_action_ids),
            "execution_class": execution_class,
        },
        effect_intents=tuple(risk_actions),
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


def _multifield_graph(
    *,
    fields: tuple[tuple[str, str, str], ...] = (
        ("subject_field", "主题", "first"),
        ("body_field", "正文", "second"),
    ),
    active_subgoal_id: str = "input_subject",
) -> DynamicTaskGraph:
    field_by_id = {field_id: (label, text) for field_id, label, text in fields}
    subject_label, subject_text = field_by_id["subject_field"]
    body_label, body_text = field_by_id["body_field"]
    statuses = {
        "input_subject": (
            "active" if active_subgoal_id == "input_subject" else "completed"
        ),
        "input_body": (
            "pending"
            if active_subgoal_id == "input_subject"
            else "active" if active_subgoal_id == "input_body" else "completed"
        ),
        "verify_fields": (
            "active" if active_subgoal_id == "verify_fields" else "pending"
        ),
    }

    def completed_evidence(subgoal_id: str) -> tuple[str, ...]:
        return (
            ("主题字段已逐字核对",)
            if subgoal_id == "input_subject"
            else ("正文字段已逐字核对",)
        ) if statuses[subgoal_id] == "completed" else ()

    graph = DynamicTaskGraph(
        task_id="multifield-observation-task",
        device_id="device-local-01",
        revision=1,
        status="running",
        goal=GraphGoal(
            objective=(
                f"在{subject_label}字段输入 {subject_text}，再在{body_label}字段输入 "
                f"{body_text}，最后同时逐字核对"
            ),
            target_apps=(
                TargetApp(
                    app_id="current_foreground",
                    app_name="当前前台应用",
                ),
            ),
            entities={
                "input_fields": [
                    {
                        "field_id": field_id,
                        "field_label": label,
                        "text": text,
                    }
                    for field_id, label, text in fields
                ],
                "target_surface": "current_surface",
            },
        ),
        constraints=("不要发送或提交",),
        completion_conditions=(
            CompletionCondition(
                condition_id="both_fields_exact",
                description=(
                    f"{subject_label}字段逐字为 {subject_text} 且"
                    f"{body_label}字段逐字为 {body_text}"
                ),
                evidence_required=("两个字段当前值同时可见",),
            ),
        ),
        risk_actions=(),
        subgoals=(
            Subgoal(
                subgoal_id="input_subject",
                objective=f"在{subject_label}字段输入 {subject_text}",
                status=statuses["input_subject"],
                depends_on=(),
                constraints=("不要发送或提交",),
                completion_conditions=(
                    f"{subject_label}字段逐字为 {subject_text}",
                ),
                completion_evidence=completed_evidence("input_subject"),
                risk_action_ids=(),
                external_impact="navigation_only",
            ),
            Subgoal(
                subgoal_id="input_body",
                objective=f"在{body_label}字段输入 {body_text}",
                status=statuses["input_body"],
                depends_on=("input_subject",),
                constraints=("不要发送或提交",),
                completion_conditions=(
                    f"{body_label}字段逐字为 {body_text}",
                ),
                completion_evidence=completed_evidence("input_body"),
                risk_action_ids=(),
                external_impact="navigation_only",
            ),
            Subgoal(
                subgoal_id="verify_fields",
                objective=(
                    f"同时逐字核对{subject_label}为 {subject_text}、"
                    f"{body_label}为 {body_text}"
                ),
                status=statuses["verify_fields"],
                depends_on=("input_subject", "input_body"),
                constraints=("只读核对",),
                completion_conditions=(
                    f"{subject_label}字段逐字为 {subject_text}",
                    f"{body_label}字段逐字为 {body_text}",
                ),
                completion_evidence=(),
                risk_action_ids=(),
                external_impact="read_only",
            ),
        ),
        active_subgoal_id=active_subgoal_id,
        raw_user_goal=(
            f"在{subject_label}字段输入 {subject_text}，再在{body_label}字段输入 "
            f"{body_text}，最后同时逐字核对；不要发送或提交"
        ),
    )
    graph.validate()
    return graph


def _external_graph(*, impact: str = "external_state") -> DynamicTaskGraph:
    risk_type = (
        "transaction_or_payment"
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
                "向商户付款20元"
                if impact == "external_state"
                else "处理影响尚不明确的目标状态"
            ),
            target_apps=(TargetApp(app_id="gallery", app_name="示例工具"),),
            entities={
                "merchant": "商户",
                "amount": "20",
                "currency": "CNY",
            },
        ),
        constraints=("任何外部影响都必须失败关闭",),
        completion_conditions=(
            CompletionCondition(
                condition_id="condition-1",
                description="已向商户付款20元",
                evidence_required=("页面显示结果",),
            ),
        ),
        risk_actions=(
            RiskAction(
                risk_id="risk-1",
                description=(
                    "向商户付款20元" if impact == "external_state" else "影响不明确"
                ),
                external_effect=(
                    "已向商户付款20元"
                    if impact == "external_state"
                    else "可能改变外部状态"
                ),
                risk_type=risk_type,
                risk_level="high",
                subgoal_ids=("subgoal-1",),
                effect_kind=(
                    "financial_transaction"
                    if impact == "external_state"
                    else ""
                ),
                target_roles=(("merchant",) if impact == "external_state" else ()),
                payload_roles=(
                    ("amount", "currency") if impact == "external_state" else ()
                ),
                expected_result_texts=(
                    ("已向商户付款20元",) if impact == "external_state" else ()
                ),
            ),
        ),
        subgoals=(
            Subgoal(
                subgoal_id="subgoal-1",
                objective=(
                    "已向商户付款20元"
                    if impact == "external_state"
                    else "目标状态达到但影响仍不明确"
                ),
                status="active",
                depends_on=(),
                constraints=("必须等待明确确认",),
                completion_conditions=(
                    ("已向商户付款20元",)
                    if impact == "external_state"
                    else ("页面显示结果",)
                ),
                completion_evidence=(),
                risk_action_ids=("risk-1",),
                external_impact=impact,
            ),
        ),
        active_subgoal_id="subgoal-1",
        raw_user_goal=(
            "向商户付款20元"
            if impact == "external_state"
            else "测试未知效果目标"
        ),
    )
    graph.validate()
    return graph


def _ordinary_send_graph(*, revision: int = 1) -> DynamicTaskGraph:
    raw_goal = "向测试收件人乙发送 freshsendproof"
    return _graph_from_payload(
        {
            "status": "ready",
            "goal": {
                "objective": raw_goal,
                "target_apps": [
                    {"app_id": "sample.messaging", "app_name": "示例消息工具"}
                ],
                "entities": {
                    "recipient": "测试收件人乙",
                    "input_text": "freshsendproof",
                },
            },
            "constraints": ["只发送一次指定正文"],
            "completion_conditions": [
                {
                    "condition_id": "message_sent",
                    "description": "测试收件人乙的会话中显示 freshsendproof",
                    "evidence_required": [
                        "新鲜画面逐字显示已发送正文 freshsendproof"
                    ],
                    "satisfied": False,
                    "evidence": [],
                }
            ],
            "effect_intents": [
                {
                    "effect_id": "effect_send_fresh",
                    "kind": "send_message",
                    "target_entity_roles": ["recipient"],
                    "payload_entity_roles": ["input_text"],
                    "source_subgoal_ids": ["send_fresh_message"],
                    "expected_results": [
                        "测试收件人乙的会话中显示 freshsendproof"
                    ],
                }
            ],
            "subgoals": [
                {
                    "subgoal_id": "send_fresh_message",
                    "objective": "向测试收件人乙发送正文 freshsendproof",
                    "status": "active",
                    "depends_on": [],
                    "constraints": ["只允许一次普通发送动作"],
                    "completion_conditions": [
                        "测试收件人乙的会话中显示 freshsendproof"
                    ],
                    "completion_evidence": [],
                    "effect_ids": ["effect_send_fresh"],
                    "execution_class": "effect",
                }
            ],
            "active_subgoal_id": "send_fresh_message",
            "clarification_questions": [],
        },
        task_id="task-ordinary-send-fresh",
        device_id="device-1",
        revision=revision,
        raw_user_goal=raw_goal,
    )


def _complete_with_current_visual_claim(
    graph: DynamicTaskGraph,
    observation: object,
) -> DynamicTaskGraph:
    visual_refs = tuple(
        item.ref_id for item in observation.visual_claim_evidence_refs
    )
    if not visual_refs:
        raise AssertionError("send result completion requires a current visual claim")
    completed = replace(
        graph,
        revision=graph.revision + 1,
        status="completed",
        completion_conditions=tuple(
            replace(item, satisfied=True, evidence=(visual_refs[0],))
            for item in graph.completion_conditions
        ),
        subgoals=tuple(
            replace(
                item,
                status="completed",
                completion_evidence=(visual_refs[0],),
            )
            for item in graph.subgoals
        ),
        active_subgoal_id=None,
    )
    completed.validate()
    return completed


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
    def __init__(
        self,
        graph: DynamicTaskGraph,
        *,
        replan_result=None,
        replan_error: Exception | None = None,
    ) -> None:
        self.graph = graph
        self.replan_result = replan_result
        self.replan_error = replan_error
        self.plan_calls = []
        self.replan_calls = []

    def plan(self, raw_goal, *, device_id, task_id=None):
        self.plan_calls.append((raw_goal, device_id, task_id))
        return self.graph

    def replan(self, graph, observation, *, trigger, reason):
        self.replan_calls.append((graph, observation, trigger, reason))
        if self.replan_error is not None:
            raise self.replan_error
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


class FakeExecutingAdapter(FakeAdapter):
    def __init__(
        self,
        scene: UIScene,
        after_scene: UIScene,
        *,
        execute_error: GenericActionAdapterError | None = None,
        action_outcome: str = "matched",
        verification_errors: tuple[str, ...] = (),
        controller_completion_evidence: tuple[str, ...] = (),
    ) -> None:
        super().__init__(scene)
        self.after_scene = after_scene
        self.execute_error = execute_error
        self.action_outcome = action_outcome
        self.verification_errors = verification_errors
        self.controller_completion_evidence = controller_completion_evidence

    def execute(
        self,
        *,
        requested_action,
        planned_scene,
        planned_frames=None,
        goal,
        confirmed,
        evidence_dir,
    ):
        self.execute_calls += 1
        if self.execute_error is not None:
            raise self.execute_error
        after_frames = tuple(
            Image.new("RGB", (540, 960), "white") for _ in range(4)
        )
        return GenericActionExecutionResult(
            requested_action=requested_action,
            rebound_action=requested_action,
            resolved_action=ResolvedSemanticAction(
                node_id=requested_action.node_id,
                kind=requested_action.action,
                normalized_point=(0.3, 0.25),
                target_element_id="candidate-1",
                before_fingerprint=planned_scene.fingerprint,
                expected_effect=dict(
                    requested_action.params.get("expected_effect") or {}
                ),
            ),
            before_scene=planned_scene,
            after_scene=self.after_scene,
            planned_scene_fingerprint=planned_scene.fingerprint,
            confirmation_frame_identity_verified=True,
            confirmation_frame_delta=0.0,
            physical_actions=1,
            action_outcome=self.action_outcome,
            verification_errors=self.verification_errors,
            controller_completion_evidence=(
                self.controller_completion_evidence
            ),
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
    def __init__(
        self,
        status="action",
        *,
        mutate_identity=None,
        action_kind="tap_semantic",
        goal_complete_on_success=False,
    ) -> None:
        self.status = status
        self.mutate_identity = mutate_identity
        self.action_kind = action_kind
        self.goal_complete_on_success = goal_complete_on_success
        self.calls = []

    def decide(
        self,
        *,
        frames,
        task_context,
        trusted_observation,
        decision_number=1,
        available_action_kinds=None,
    ):
        self.calls.append((frames, task_context, trusted_observation, decision_number))
        if self.status == "action":
            decision = _decision(
                trusted_observation.scene,
                action_kind=self.action_kind,
                goal_complete_on_success=self.goal_complete_on_success,
            )
            semantic_ir = getattr(task_context, "semantic_ir", None)
            if semantic_ir is None:
                raise AssertionError("FakeQwenObserver 缺少 canonical TaskSemanticIR")
            catalog = compile_canonical_action_catalog(
                trusted_observation.scene,
                semantic_ir,
                available_action_kinds or (),
            )
            requested = decision.proposal.action
            matches = [
                item
                for item in catalog.candidates
                if item.action_kind == requested.action
                and (
                    requested.action not in {
                        "tap_semantic",
                        "dismiss_overlay",
                        "input_verified_text",
                        "clear_verified_text",
                        "long_press",
                    }
                    or str(item.parameters.get("element_id") or "")
                    == str(requested.params.get("element_id") or "")
                )
                and (
                    requested.action != "swipe"
                    or str(item.parameters.get("direction") or "")
                    == str(requested.params.get("direction") or "")
                )
            ]
            if len(matches) != 1:
                decision = SimpleNamespace(
                    proposal=GenericStepProposal(
                        status="blocked",
                        action=None,
                        reason=(
                            "当前 canonical action catalog 没有唯一匹配候选："
                            f"{requested.action}；现有候选="
                            f"{[(item.action_kind, item.parameters) for item in catalog.candidates]}"
                        ),
                    ),
                    target_region=None,
                    confidence=1.0,
                )
            else:
                candidate = matches[0]
                params = dict(requested.params)
                params.update(
                    {
                        "formal_candidate_id": candidate.candidate_id,
                        "formal_report_digest": catalog.report_digest,
                        "formal_transition": candidate.transition.to_dict(),
                    }
                )
                decision.proposal = replace(
                    decision.proposal,
                    action=replace(requested, params=params),
                )
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


class SequenceDeepSeekPlanner(FakeDeepSeekPlanner):
    def __init__(self, graph: DynamicTaskGraph, *replan_results: DynamicTaskGraph):
        super().__init__(graph)
        self.replan_results = list(replan_results)

    def replan(self, graph, observation, *, trigger, reason):
        self.replan_calls.append((graph, observation, trigger, reason))
        if not self.replan_results:
            raise AssertionError("unexpected DeepSeek replan call")
        return self.replan_results.pop(0)


class SequenceQwenObserver:
    def __init__(self, *statuses: str) -> None:
        self.statuses = list(statuses)
        self.calls = []

    def decide(self, **kwargs):
        if not self.statuses:
            raise AssertionError("unexpected Qwen decision call")
        self.calls.append(kwargs)
        return FakeQwenObserver(self.statuses.pop(0)).decide(**kwargs)


class SequenceExecutingAdapter(FakeAdapter):
    def __init__(
        self,
        scene: UIScene,
        *after_steps: tuple[UIScene, str, tuple[str, ...]],
    ) -> None:
        super().__init__(scene)
        self.after_steps = list(after_steps)

    def execute(self, **kwargs):
        self.execute_calls += 1
        if not self.after_steps:
            raise AssertionError("unexpected physical action")
        after_scene, outcome, errors = self.after_steps.pop(0)
        result = FakeExecutingAdapter(
            kwargs["planned_scene"],
            after_scene,
            action_outcome=outcome,
            verification_errors=errors,
        ).execute(**kwargs)
        self.scene = after_scene
        return result


class SequenceCaptureAdapter(FakeAdapter):
    def __init__(self, *scenes: UIScene) -> None:
        if not scenes:
            raise ValueError("at least one scene is required")
        super().__init__(scenes[0])
        self.scenes = list(scenes)

    def capture_scene(self, goal, *, evidence_dir, prefix):
        if not self.scenes:
            raise AssertionError("unexpected capture")
        self.scene = self.scenes.pop(0)
        return super().capture_scene(goal, evidence_dir=evidence_dir, prefix=prefix)


def _trusted_factory(*, frames, device_id, scene, observation_id=None):
    if len(frames) < 4:
        raise AssertionError("trusted observation requires four frames")
    return FakeTrustedObservation(
        device_id=device_id,
        scene=scene,
        observation_id=observation_id or "obs-start",
    )


class VisibleCompletionConditionProgressTests(unittest.TestCase):
    @staticmethod
    def _observed(*refs: str) -> SimpleNamespace:
        return SimpleNamespace(
            visual_claim_evidence_refs=tuple(
                SimpleNamespace(ref_id=value) for value in refs
            ),
            visible_evidence=(),
            grounded_visual_facts=(),
        )

    def test_allows_only_current_typed_evidence_state_progress(self) -> None:
        graph = _graph()
        ref = "visual_claim:obs-current:claim-current"
        revised = replace(
            graph,
            completion_conditions=(
                replace(
                    graph.completion_conditions[0],
                    satisfied=True,
                    evidence=(ref,),
                ),
            ),
        )

        _validate_visible_completion_condition_progress(
            graph,
            revised,
            self._observed(ref),
        )

    def test_rejects_definition_rewrite_and_unknown_evidence(self) -> None:
        graph = _graph()
        ref = "visual_claim:obs-current:claim-current"
        cases = (
            (
                replace(
                    graph,
                    completion_conditions=graph.completion_conditions
                    + (
                        CompletionCondition(
                            condition_id="added-condition",
                            description="新增状态可见",
                            evidence_required=("新增状态可见",),
                        ),
                    ),
                ),
                "不得增加、删除或重排全局完成条件",
            ),
            (
                replace(
                    graph,
                    completion_conditions=(
                        replace(
                            graph.completion_conditions[0],
                            description="被改写的完成条件",
                        ),
                    ),
                ),
                "不得改写全局完成条件定义",
            ),
            (
                replace(
                    graph,
                    completion_conditions=(
                        replace(
                            graph.completion_conditions[0],
                            satisfied=True,
                            evidence=("visual_claim:old:unknown",),
                        ),
                    ),
                ),
                "当前观察之外的全局完成证据",
            ),
        )
        for revised, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(
                    UniversalAgentOrchestratorError,
                    message,
                ):
                    _validate_visible_completion_condition_progress(
                        graph,
                        revised,
                        self._observed(ref),
                    )

    def test_rejects_satisfied_condition_rollback_or_evidence_deletion(self) -> None:
        ref = "visual_claim:obs-old:claim-old"
        graph = replace(
            _graph(),
            completion_conditions=(
                replace(
                    _graph().completion_conditions[0],
                    satisfied=True,
                    evidence=(ref,),
                ),
            ),
        )
        cases = (
            (
                replace(
                    graph,
                    completion_conditions=(
                        replace(
                            graph.completion_conditions[0],
                            satisfied=False,
                            evidence=(),
                        ),
                    ),
                ),
                "不得撤销已满足",
            ),
            (
                replace(
                    graph,
                    completion_conditions=(
                        replace(graph.completion_conditions[0], evidence=()),
                    ),
                ),
                "不得删除既有全局完成证据",
            ),
        )
        for revised, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(
                    UniversalAgentOrchestratorError,
                    message,
                ):
                    _validate_visible_completion_condition_progress(
                        graph,
                        revised,
                        self._observed(),
                    )


class PhaseOneNavigationPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = PhaseOneNavigationPolicy()

    def test_missing_canonical_protocol_is_rejected(self) -> None:
        current_scene = _scene()
        result = self.policy.evaluate(
            task_context=_context(),
            trusted_observation=_decision(current_scene).trusted_observation,
            decision=_decision(current_scene),
        )
        self.assertFalse(result.allowed)
        self.assertIn("canonical action protocol", result.reason)


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

    def test_projects_formal_surface_goals_without_inventing_target_apps(self) -> None:
        identities = {
            "device": ("device", "设备界面"),
            "system": ("system", "系统界面"),
            "current_surface": ("current_surface", "当前界面"),
        }
        for surface, expected_identity in identities.items():
            with self.subTest(surface=surface):
                graph = replace(
                    _graph(),
                    goal=GraphGoal(
                        objective="确认正式目标表面可见",
                        target_apps=(),
                        entities={"target_surface": surface},
                    ),
                    raw_user_goal="确认当前手机表面可见",
                )
                graph.validate()

                draft = self.bridge.goal_draft(graph)

                self.assertEqual(expected_identity, (draft.app_id, draft.app_name))
                self.assertEqual(surface, draft.entities["target_surface"])
                self.assertEqual([], draft.entities["target_apps"])
                self.assertEqual(
                    surface,
                    draft.entities["active_subgoal_visual_context"][
                        "goal_entities"
                    ]["target_surface"],
                )

    def test_rejects_missing_app_and_missing_formal_surface(self) -> None:
        graph = replace(
            _graph(),
            status="blocked",
            goal=GraphGoal(
                objective="等待补充目标",
                target_apps=(),
                entities={},
            ),
            active_subgoal_id=None,
            subgoals=tuple(
                replace(item, status="blocked") for item in _graph().subgoals
            ),
            clarification_questions=("请补充目标 App 或目标表面",),
        )
        graph.validate()

        with self.assertRaisesRegex(
            UniversalAgentOrchestratorError,
            "目标 App 或正式目标 surface",
        ):
            self.bridge.goal_draft(graph)

    def test_projection_preserves_goal_entities_constraints_and_completion(self) -> None:
        graph = _graph()

        goal = self.bridge.goal_draft(graph)

        self.assertEqual("风景", goal.entities["category"])
        self.assertEqual(
            "看看图片工具里的风景分类",
            goal.entities["original_goal_visual_context"],
        )
        self.assertNotIn(
            "original_goal_visual_context",
            graph.to_qwen_context()["goal"]["entities"],
        )
        self.assertIn("仅查看公开信息", goal.constraints)
        self.assertIn("不得改变任何账号状态", goal.constraints)
        self.assertEqual(
            "页面显示风景分类详情",
            goal.success_criteria["condition-1"]["description"],
        )
        focus = goal.entities["active_subgoal_visual_context"]
        self.assertEqual(graph.active_subgoal_id, focus["subgoal_id"])
        self.assertEqual(graph.active_subgoal().objective, focus["objective"])
        self.assertEqual(
            list(graph.active_subgoal().completion_conditions),
            focus["completion_conditions"],
        )

    def test_projects_only_typed_active_input_transaction_marker(self) -> None:
        graph = DynamicTaskGraph(
            task_id="task-input-candidate",
            device_id="device-1",
            revision=1,
            status="running",
            goal=GraphGoal(
                objective="完成当前未提交的中文输入",
                target_apps=(
                    TargetApp(
                        app_id="current_foreground",
                        app_name="当前前台应用",
                    ),
                ),
                entities={
                    "input_text": "你好",
                    "target_ui_label": "你好",
                },
            ),
            constraints=("不要发送或提交",),
            completion_conditions=(
                CompletionCondition(
                    condition_id="input-display",
                    description="当前输入框逐字显示你好",
                    evidence_required=("输入框显示你好",),
                ),
            ),
            risk_actions=(),
            subgoals=(
                Subgoal(
                    subgoal_id="select-candidate",
                    objective="选择唯一逐字候选你好",
                    status="active",
                    depends_on=(),
                    constraints=("不要发送或提交",),
                    completion_conditions=("候选你好被选中",),
                    completion_evidence=(),
                    risk_action_ids=(),
                    external_impact="navigation_only",
                ),
                Subgoal(
                    subgoal_id="verify-input",
                    objective="确认输入框逐字显示你好",
                    status="pending",
                    depends_on=("select-candidate",),
                    constraints=("不要发送或提交",),
                    completion_conditions=("输入框显示你好",),
                    completion_evidence=(),
                    risk_action_ids=(),
                    external_impact="read_only",
                ),
            ),
            active_subgoal_id="select-candidate",
            raw_user_goal="选择当前拼音候选你好并确认输入值，不要发送",
        )
        graph.validate()

        focus = self.bridge.goal_draft(graph).entities[
            "active_subgoal_visual_context"
        ]

        self.assertEqual(
            "你好",
            focus["goal_entities"]["active_input_transaction_text"],
        )
        self.assertEqual(
            "input_field_1",
            focus["goal_entities"]["active_input_field_id"],
        )
        self.assertFalse(focus["goal_entities"]["active_input_multiline"])

        unrelated = replace(
            graph,
            goal=replace(
                graph.goal,
                objective="显示结果列表",
                entities={
                    "input_text": "搜索",
                    "target_ui_label": "搜索",
                },
            ),
            completion_conditions=(
                CompletionCondition(
                    condition_id="results-visible",
                    description="结果列表已显示",
                    evidence_required=("结果列表可见",),
                ),
            ),
            subgoals=(
                Subgoal(
                    subgoal_id="show-results",
                    objective="显示结果列表",
                    status="active",
                    depends_on=(),
                    constraints=(),
                    completion_conditions=("结果列表可见",),
                    completion_evidence=(),
                    risk_action_ids=(),
                    external_impact="navigation_only",
                ),
            ),
            active_subgoal_id="show-results",
            raw_user_goal="显示结果列表",
        )
        unrelated.validate()

        unrelated_focus = self.bridge.goal_draft(unrelated).entities[
            "active_subgoal_visual_context"
        ]
        self.assertNotIn(
            "active_input_transaction_text",
            unrelated_focus["goal_entities"],
        )

    def test_multifield_live_context_projects_only_current_typed_field(self) -> None:
        graph = _multifield_graph()

        goal = self.bridge.goal_draft(graph)
        focus = goal.entities["active_subgoal_visual_context"]
        observation_context = {
            "device_id": graph.device_id,
            "objective": focus["objective"],
            "entities": goal.entities,
            "constraints": list(goal.constraints),
            "completion_conditions": list(focus["completion_conditions"]),
            "execution_class": focus["execution_class"],
        }

        self.assertEqual(
            graph.goal.entities["input_fields"],
            goal.entities["input_fields"],
        )
        self.assertNotIn("input_fields", focus["goal_entities"])
        self.assertEqual(
            {
                "active_input_transaction_text": "first",
                "active_input_field_id": "subject_field",
                "active_input_field_label": "主题",
                "active_input_multiline": False,
            },
            {
                key: focus["goal_entities"][key]
                for key in (
                    "active_input_transaction_text",
                    "active_input_field_id",
                    "active_input_field_label",
                    "active_input_multiline",
                )
            },
        )
        self.assertEqual("current_surface", focus["goal_entities"]["target_surface"])
        self.assertIsInstance(_safe_goal_context(observation_context), dict)

        historical_context = json.loads(json.dumps(observation_context))
        historical_context["entities"]["active_subgoal_visual_context"][
            "goal_entities"
        ]["input_fields"] = list(graph.goal.entities["input_fields"])
        with self.assertRaisesRegex(VisionAgentError, "目标上下文嵌套过深"):
            _safe_goal_context(historical_context)

    def test_multifield_projection_tracks_field_identity_across_order_and_labels(self) -> None:
        cases = (
            (
                (
                    ("subject_field", "标题", "alpha"),
                    ("body_field", "备注", "beta"),
                ),
                "input_subject",
                ("subject_field", "标题", "alpha"),
            ),
            (
                (
                    ("body_field", "内容", "delta"),
                    ("subject_field", "名称", "gamma"),
                ),
                "input_body",
                ("body_field", "内容", "delta"),
            ),
        )
        for fields, active_subgoal_id, expected in cases:
            with self.subTest(active_subgoal_id=active_subgoal_id):
                graph = _multifield_graph(
                    fields=fields,
                    active_subgoal_id=active_subgoal_id,
                )

                focus = self.bridge.goal_draft(graph).entities[
                    "active_subgoal_visual_context"
                ]["goal_entities"]

                self.assertNotIn("input_fields", focus)
                self.assertEqual(expected[0], focus["active_input_field_id"])
                self.assertEqual(expected[1], focus["active_input_field_label"])
                self.assertEqual(expected[2], focus["active_input_transaction_text"])

    def test_multifield_final_verify_keeps_two_typed_desired_states(self) -> None:
        graph = _multifield_graph(active_subgoal_id="verify_fields")

        goal = self.bridge.goal_draft(graph)
        focus = goal.entities["active_subgoal_visual_context"]["goal_entities"]

        self.assertEqual(2, len(goal.entities["input_fields"]))
        self.assertNotIn("input_fields", focus)
        self.assertEqual(
            {"subject_field": "first", "body_field": "second"},
            focus["desired_input_values"],
        )
        self.assertEqual(
            {"subject_field": "主题", "body_field": "正文"},
            focus["desired_input_labels"],
        )
        self.assertNotIn("active_input_transaction_text", focus)

    def test_current_open_app_node_projects_exact_app_label_without_mutating_graph(self) -> None:
        for app_id, app_name, verb in (
            ("browser", "浏览器", "打开"),
            ("music", "音乐", "启动"),
        ):
            with self.subTest(app_id=app_id):
                graph = replace(
                    _graph(),
                    goal=GraphGoal(
                        objective=f"先{verb}{app_name}，再读取页面标题",
                        target_apps=(TargetApp(app_id=app_id, app_name=app_name),),
                        entities={"target_surface": "device"},
                    ),
                    subgoals=(
                        replace(
                            _graph().subgoals[0],
                            subgoal_id=f"open_{app_id}",
                            objective=f"{verb}{app_name}",
                            completion_conditions=(f"{app_name}主界面可见",),
                        ),
                    ),
                    active_subgoal_id=f"open_{app_id}",
                )
                graph.validate()

                goal = self.bridge.goal_draft(graph)
                focus = goal.entities["active_subgoal_visual_context"]

                self.assertEqual(app_name, focus["goal_entities"]["target_ui_label"])
                self.assertNotIn("target_ui_label", graph.goal.entities)

    def test_multi_app_node_projects_only_the_uniquely_referenced_current_app(self) -> None:
        graph = replace(
            _graph(),
            goal=GraphGoal(
                objective="依次查看浏览器和音乐",
                target_apps=(
                    TargetApp(app_id="browser", app_name="浏览器"),
                    TargetApp(app_id="music", app_name="音乐"),
                ),
                entities={"target_surface": "device"},
            ),
            subgoals=(
                replace(
                    _graph().subgoals[0],
                    objective="启动音乐",
                    completion_conditions=("音乐主界面可见",),
                ),
            ),
        )
        graph.validate()

        focus = self.bridge.goal_draft(graph).entities["active_subgoal_visual_context"]

        self.assertEqual("音乐", focus["goal_entities"]["target_ui_label"])

    def test_non_entry_ambiguous_and_reference_nodes_do_not_project_app_label(self) -> None:
        cases = (
            (
                "read",
                (TargetApp(app_id="browser", app_name="浏览器"),),
                "读取浏览器页面标题",
                "read_only",
            ),
            (
                "ambiguous",
                (
                    TargetApp(app_id="browser", app_name="浏览器"),
                    TargetApp(app_id="music", app_name="音乐"),
                ),
                "打开浏览器或音乐",
                "navigation_only",
            ),
            (
                "reference",
                (TargetApp(app_id="current_foreground", app_name="当前前台应用"),),
                "打开当前前台应用",
                "navigation_only",
            ),
            (
                "home",
                (TargetApp(app_id="browser", app_name="浏览器"),),
                "返回手机桌面",
                "navigation_only",
            ),
        )
        for name, target_apps, objective, impact in cases:
            with self.subTest(name=name):
                graph = replace(
                    _graph(),
                    goal=replace(_graph().goal, target_apps=target_apps),
                    subgoals=(
                        replace(
                            _graph().subgoals[0],
                            objective=objective,
                            external_impact=impact,
                        ),
                    ),
                )
                graph.validate()

                focus = self.bridge.goal_draft(graph).entities[
                    "active_subgoal_visual_context"
                ]

                self.assertNotIn("target_ui_label", focus["goal_entities"])

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
        self.assertTrue(observed.grounded_visual_facts)
        self.assertTrue(
            any("查看详情" in item for item in observed.grounded_visual_facts)
        )
        self.assertTrue(observed.visual_claim_evidence_refs)
        self.assertTrue(
            all(
                item.scene_id == observed.scene_id
                and item.ref_id.startswith(f"visual_claim:{observed.scene_id}:")
                for item in observed.visual_claim_evidence_refs
            )
        )
        self.assertTrue(
            any(
                item.fact == "页面标题已变化"
                for item in observed.visual_claim_evidence_refs
            )
        )
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
            store.write_effect_policy_snapshot(graph)
            store.write_trusted_observation(1, {"observation_id": "obs-1"})
            store.write_qwen_decision(1, {"status": "action"})
            store.write_controller_decision(1, {"allowed": True})
            store.write_verification(1, {"matched": True})
            store.write_report({"physical_actions": 0})

            expected = {
                "session.json",
                "task_graph_revision_1.json",
                "effect_policy_revision_1.json",
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

    def test_start_preserves_literal_newline_in_raw_goal(self) -> None:
        graph = _graph()
        planner = FakeDeepSeekPlanner(graph)
        orchestrator = self._orchestrator(
            planner,
            FakeQwenObserver(),
            FakeAdapter(_scene()),
        )
        raw_goal = (
            "在当前多行输入框逐字输入且不要发送：first line\nsecond line"
        )

        with tempfile.TemporaryDirectory() as temp:
            session = orchestrator.start(
                session_id="session-newline-authority",
                raw_goal=f"  {raw_goal}  ",
                device_id="device-1",
                run_dir=Path(temp),
            )

        self.assertEqual(raw_goal, session.raw_goal)
        self.assertEqual(raw_goal, planner.plan_calls[0][0])

    def test_named_app_already_foreground_completes_open_node_without_action(self) -> None:
        base = _graph()
        open_app = replace(
            base.subgoals[0],
            subgoal_id="open_wechat",
            objective="打开微信应用",
            completion_conditions=("微信应用已打开",),
            constraints=(),
            external_impact="navigation_only",
        )
        next_step = Subgoal(
            subgoal_id="continue_in_wechat",
            objective="继续处理当前页面",
            status="pending",
            depends_on=(open_app.subgoal_id,),
            constraints=(),
            completion_conditions=("后续目标完成",),
            completion_evidence=(),
            risk_action_ids=(),
            external_impact="navigation_only",
        )
        initial = replace(
            base,
            goal=replace(
                base.goal,
                objective="打开微信并继续处理当前页面",
                target_apps=(TargetApp(app_id="wechat", app_name="微信"),),
            ),
            subgoals=(open_app, next_step),
            active_subgoal_id=open_app.subgoal_id,
            raw_user_goal="打开微信并继续处理当前页面",
        )
        initial.validate()
        scene = UIScene(
            app_id="wechat",
            screen_id="chat_window",
            summary="微信文件传输助手聊天界面。",
            elements=(),
            stable=True,
            confidence=1.0,
            fingerprint="wechat-already-foreground",
        )
        revised = replace(
            initial,
            revision=2,
            subgoals=(
                replace(
                    open_app,
                    status="completed",
                    completion_evidence=(scene.summary,),
                ),
                replace(next_step, status="active"),
            ),
            active_subgoal_id=next_step.subgoal_id,
        )
        revised.validate()
        planner = SequenceDeepSeekPlanner(initial, revised)
        adapter = FakeAdapter(scene)
        orchestrator = self._orchestrator(
            planner,
            FakeQwenObserver(),
            adapter,
        )

        with tempfile.TemporaryDirectory() as temp:
            session = UniversalAgentSessionState(
                session_id="session-open-app-idempotent",
                raw_goal=initial.raw_user_goal,
                device_id=initial.device_id,
                run_dir=Path(temp),
                adapter=adapter,
                evidence_store=AgentEvidenceStore(Path(temp)),
                task_graph=initial,
            )
            result = orchestrator._try_advance_visible_presence_subgoal(
                session,
                graph=initial,
                trusted_observation=FakeTrustedObservation(
                    device_id=initial.device_id,
                    scene=scene,
                ),
            )
            wrong_app = orchestrator._try_advance_visible_presence_subgoal(
                session,
                graph=initial,
                trusted_observation=FakeTrustedObservation(
                    device_id=initial.device_id,
                    scene=replace(
                        scene,
                        app_id="browser",
                        fingerprint="browser-foreground",
                    ),
                ),
            )

        self.assertEqual(revised, result)
        self.assertIsNone(wrong_app)
        self.assertEqual(1, len(planner.replan_calls))

    def test_app_foreground_presence_phrase_is_narrow(self) -> None:
        self.assertTrue(
            UniversalAgentOrchestrator._is_idempotent_app_foreground_completion(
                "浏览器应用已启动"
            )
        )
        for value in (
            "应用已重新加载",
            "微信应用已打开并登录",
            "付款应用已完成付款",
            "应用图标可见",
        ):
            with self.subTest(value=value):
                self.assertFalse(
                    UniversalAgentOrchestrator._is_idempotent_app_foreground_completion(
                        value
                    )
                )

    def test_named_app_page_ignores_incomplete_control_sharing_app_name(self) -> None:
        base = _graph()
        open_settings = replace(
            base.subgoals[0],
            subgoal_id="open_settings",
            objective="打开设置应用",
            completion_conditions=("设置主界面可见",),
            constraints=(),
        )
        input_wifi = Subgoal(
            subgoal_id="input_wifi",
            objective="在搜索输入框中输入 wifi",
            status="pending",
            depends_on=(open_settings.subgoal_id,),
            constraints=("不得选择任何搜索结果",),
            completion_conditions=("输入框中显示 'wifi'",),
            completion_evidence=(),
            risk_action_ids=(),
            external_impact="navigation_only",
        )
        initial = replace(
            base,
            goal=replace(
                base.goal,
                objective="打开设置并在搜索输入框输入 wifi",
                target_apps=(
                    TargetApp(app_id="com.android.settings", app_name="设置"),
                ),
                entities={"input_text": "wifi"},
            ),
            subgoals=(open_settings, input_wifi),
            active_subgoal_id=open_settings.subgoal_id,
            raw_user_goal="打开设置并在搜索输入框输入 wifi",
        )
        initial.validate()
        scene = UIScene(
            app_id="com.android.settings",
            screen_id="settings_main",
            summary="设置应用主界面，可见搜索框和连接选项。",
            elements=(
                UIElement(
                    element_id="search-input",
                    role="input",
                    meaning="search_settings",
                    label="搜索系统设置项",
                    bounds=(0.12, 0.15, 0.9, 0.21),
                    confidence=0.95,
                    states={"goal_relevant": True, "value": ""},
                ),
                UIElement(
                    element_id="page-title",
                    role="text",
                    meaning="page_title",
                    label="设置",
                    bounds=(0.12, 0.08, 0.3, 0.14),
                    confidence=1.0,
                    states={"goal_relevant": True, "fully_visible": True},
                ),
            ),
            stable=True,
            confidence=1.0,
            fingerprint="settings-main-current",
        )
        trusted = FakeTrustedObservation(
            device_id=initial.device_id,
            scene=scene,
        )
        observed = ObservationBridge().observed_state(
            graph=initial,
            trusted_observation=trusted,
            action_outcome="not_applicable",
            verification={"visible_evidence": [scene.summary]},
        )
        current_summary_ref = next(
            item.ref_id
            for item in observed.visual_claim_evidence_refs
            if item.fact == scene.summary
        )
        revised = replace(
            initial,
            revision=2,
            completion_conditions=tuple(
                replace(
                    item,
                    satisfied=True,
                    evidence=(current_summary_ref,),
                )
                for item in initial.completion_conditions
            ),
            subgoals=(
                replace(
                    open_settings,
                    status="completed",
                    completion_evidence=(scene.summary,),
                ),
                replace(input_wifi, status="active"),
            ),
            active_subgoal_id=input_wifi.subgoal_id,
        )
        revised.validate()
        planner = SequenceDeepSeekPlanner(initial, revised)
        adapter = FakeAdapter(scene)
        orchestrator = self._orchestrator(
            planner,
            FakeQwenObserver(),
            adapter,
        )

        with tempfile.TemporaryDirectory() as temp:
            session = UniversalAgentSessionState(
                session_id="session-settings-page-presence",
                raw_goal=initial.raw_user_goal,
                device_id=initial.device_id,
                run_dir=Path(temp),
                adapter=adapter,
                evidence_store=AgentEvidenceStore(Path(temp)),
                task_graph=initial,
            )
            result = orchestrator._try_advance_visible_presence_subgoal(
                session,
                graph=initial,
                trusted_observation=trusted,
            )

        self.assertEqual(revised, result)
        self.assertEqual(1, len(planner.replan_calls))
        observed = planner.replan_calls[0][1]
        self.assertIn(scene.summary, observed.visible_evidence)
        self.assertTrue(
            any('"screen_id":"settings_main"' in item for item in observed.visible_evidence)
        )

    def test_browser_reload_element_cannot_prove_phone_desktop_presence(self) -> None:
        base = _graph()
        desktop_graph = replace(
            base,
            goal=replace(
                base.goal,
                objective="让手机桌面在前台稳定可见",
                target_apps=(
                    TargetApp(
                        app_id="current_foreground",
                        app_name="当前前台应用",
                    ),
                ),
                entities={},
            ),
            completion_conditions=(
                CompletionCondition(
                    condition_id="desktop_visible",
                    description="手机桌面在前台稳定可见",
                    evidence_required=("手机桌面界面在前台显示",),
                ),
            ),
            subgoals=(
                replace(
                    base.subgoals[0],
                    subgoal_id="ensure-desktop",
                    objective="手机桌面在前台可见",
                    completion_conditions=("手机桌面在前台显示",),
                    external_impact="navigation_only",
                ),
            ),
            active_subgoal_id="ensure-desktop",
            raw_user_goal="让手机桌面在前台稳定可见",
        )
        desktop_graph.validate()
        browser_scene = replace(
            _scene(
                meaning="reload",
                label="",
                role="icon",
                states={"goal_relevant": True, "fully_visible": True},
                evidence=("完整圆弧和箭头头部",),
            ),
            app_id="browser",
            screen_id="generic_acceptance_page",
            summary="浏览器显示通用动作真机验收页和刷新图标",
        )
        planner = FakeDeepSeekPlanner(
            desktop_graph,
            replan_result=_completed_graph(desktop_graph),
        )
        qwen = FakeQwenObserver(action_kind="home")
        adapter = FakeAdapter(browser_scene)

        with tempfile.TemporaryDirectory() as temp:
            session = self._orchestrator(planner, qwen, adapter).start(
                session_id="session-desktop-from-browser",
                raw_goal=desktop_graph.raw_user_goal,
                device_id="device-1",
                run_dir=Path(temp),
            )

        self.assertEqual("awaiting_confirmation", session.status)
        self.assertEqual([], planner.replan_calls)
        self.assertEqual(1, len(qwen.calls))
        self.assertEqual("home", session.qwen_decision.proposal.action.action)
        self.assertEqual(0, session.physical_actions)
        self.assertEqual(0, adapter.execute_calls)

    @staticmethod
    def _named_app_page_graph(*, app_id: str, app_name: str) -> DynamicTaskGraph:
        base = _graph()
        graph = replace(
            base,
            goal=replace(
                base.goal,
                objective=f"确认{app_name}首页可见",
                target_apps=(TargetApp(app_id=app_id, app_name=app_name),),
                entities={"target_ui_label": app_name},
            ),
            subgoals=(
                replace(
                    base.subgoals[0],
                    subgoal_id="named-app-page-visible",
                    objective=f"{app_name}首页可见",
                    completion_conditions=(f"{app_name}应用界面可见",),
                    external_impact="navigation_only",
                ),
                Subgoal(
                    subgoal_id="safe-followup",
                    objective="下一安全目标可见",
                    status="pending",
                    depends_on=("named-app-page-visible",),
                    constraints=("只读",),
                    completion_conditions=("下一安全目标可见",),
                    completion_evidence=(),
                    risk_action_ids=(),
                    external_impact="navigation_only",
                ),
            ),
            active_subgoal_id="named-app-page-visible",
            raw_user_goal=f"只读确认{app_name}首页",
        )
        graph.validate()
        return graph

    @staticmethod
    def _advance_named_app_page_graph(graph: DynamicTaskGraph) -> DynamicTaskGraph:
        revised = replace(
            graph,
            revision=graph.revision + 1,
            subgoals=(
                replace(
                    graph.subgoals[0],
                    status="completed",
                    completion_evidence=(graph.subgoals[0].completion_conditions[0],),
                ),
                replace(graph.subgoals[1], status="active"),
            ),
            active_subgoal_id="safe-followup",
        )
        revised.validate()
        return revised

    def test_named_app_source_does_not_rebind_launcher_completion_state(self) -> None:
        base = _graph()
        first_home = replace(
            base.subgoals[0],
            subgoal_id="return_to_desktop_1",
            objective="从当前页面返回手机桌面",
            status="completed",
            completion_conditions=("手机桌面界面可见",),
            completion_evidence=("手机桌面界面可见",),
            external_impact="navigation_only",
        )
        open_settings = Subgoal(
            subgoal_id="open_settings",
            objective="打开设置并确认设置主界面可见",
            status="completed",
            depends_on=(first_home.subgoal_id,),
            constraints=(),
            completion_conditions=("设置主界面可见",),
            completion_evidence=("设置主界面可见",),
            risk_action_ids=(),
            external_impact="navigation_only",
        )
        final_home = Subgoal(
            subgoal_id="return_to_desktop_2",
            objective="从设置界面返回手机桌面并确认桌面可见",
            status="active",
            depends_on=(open_settings.subgoal_id,),
            constraints=(),
            completion_conditions=("手机桌面界面可见",),
            completion_evidence=(),
            risk_action_ids=(),
            external_impact="navigation_only",
        )
        followup = Subgoal(
            subgoal_id="safe-followup",
            objective="下一安全目标可见",
            status="pending",
            depends_on=(final_home.subgoal_id,),
            constraints=(),
            completion_conditions=("下一安全目标可见",),
            completion_evidence=(),
            risk_action_ids=(),
            external_impact="navigation_only",
        )
        previous = replace(
            base,
            goal=replace(
                base.goal,
                objective=(
                    "从当前页面返回手机桌面，打开设置并确认设置主界面可见，"
                    "然后再次返回手机桌面"
                ),
                target_apps=(TargetApp(app_id="settings", app_name="设置"),),
                entities={"target_surface": "device"},
            ),
            subgoals=(first_home, open_settings, final_home, followup),
            active_subgoal_id="return_to_desktop_2",
            raw_user_goal=(
                "从当前页面返回手机桌面，打开设置并确认设置主界面可见，"
                "然后再次返回手机桌面"
            ),
        )
        previous.validate()
        revised = replace(
            previous,
            revision=previous.revision + 1,
            subgoals=(
                first_home,
                open_settings,
                replace(
                    final_home,
                    status="completed",
                    completion_evidence=("手机桌面界面可见",),
                ),
                replace(followup, status="active"),
            ),
            active_subgoal_id=followup.subgoal_id,
        )
        revised.validate()
        launcher = replace(
            _scene(
                meaning="open_settings",
                label="设置",
                states={"goal_relevant": True, "fully_visible": True},
            ),
            app_id="launcher",
            screen_id="home_screen",
            summary="手机桌面可见，包含设置入口。",
        )

        UniversalAgentOrchestrator._validate_newly_completed_named_app_surfaces(
            previous=previous,
            revised=revised,
            trusted_observation=FakeTrustedObservation(
                device_id="device-1",
                scene=launcher,
            ),
        )

    @staticmethod
    def _read_only_locate_graph() -> DynamicTaskGraph:
        base = _graph()
        graph = replace(
            base,
            goal=replace(
                base.goal,
                objective="定位当前可见输入框并把内容替换为 agent",
                target_apps=(),
                entities={
                    "input_text": "agent",
                    "target_surface": "current_surface",
                },
            ),
            subgoals=(
                replace(
                    base.subgoals[0],
                    subgoal_id="locate-input",
                    objective="定位当前可见输入框",
                    completion_conditions=("唯一目标输入框完整可见",),
                    external_impact="read_only",
                ),
                Subgoal(
                    subgoal_id="replace-input",
                    objective="把当前输入框内容替换为 agent",
                    status="pending",
                    depends_on=("locate-input",),
                    constraints=("不要提交、搜索或发送",),
                    completion_conditions=("输入框显示 agent",),
                    completion_evidence=(),
                    risk_action_ids=(),
                    external_impact="navigation_only",
                ),
            ),
            active_subgoal_id="locate-input",
            raw_user_goal="把当前输入框内容替换为 agent，不要搜索或提交",
        )
        graph.validate()
        return graph

    @staticmethod
    def _advance_locate_graph(graph: DynamicTaskGraph) -> DynamicTaskGraph:
        revised = replace(
            graph,
            revision=graph.revision + 1,
            subgoals=(
                replace(
                    graph.subgoals[0],
                    status="completed",
                    completion_evidence=("唯一目标输入框完整可见",),
                ),
                replace(graph.subgoals[1], status="active"),
            ),
            active_subgoal_id="replace-input",
        )
        revised.validate()
        return revised

    @staticmethod
    def _read_only_multi_locate_graph() -> DynamicTaskGraph:
        base = _graph()
        graph = replace(
            base,
            goal=replace(
                base.goal,
                objective="让紫色方块位于绿色终点的本机临时位置",
                entities={"source": "紫色方块", "destination": "绿色终点"},
            ),
            subgoals=(
                replace(
                    base.subgoals[0],
                    subgoal_id="locate-endpoints",
                    objective="紫色方块和绿色终点在当前页面可见",
                    completion_conditions=("紫色方块和绿色终点均出现在当前画面中",),
                    external_impact="read_only",
                ),
                Subgoal(
                    subgoal_id="move-source",
                    objective="紫色方块位于绿色终点的本机临时位置",
                    status="pending",
                    depends_on=("locate-endpoints",),
                    constraints=("不得保存、提交或修改账号数据",),
                    completion_conditions=("紫色方块与绿色终点重合",),
                    completion_evidence=(),
                    risk_action_ids=(),
                    external_impact="navigation_only",
                ),
            ),
            active_subgoal_id="locate-endpoints",
            raw_user_goal="只改变当前页面未保存的临时布局",
        )
        graph.validate()
        return graph

    @staticmethod
    def _advance_multi_locate_graph(graph: DynamicTaskGraph) -> DynamicTaskGraph:
        revised = replace(
            graph,
            revision=graph.revision + 1,
            subgoals=(
                replace(
                    graph.subgoals[0],
                    status="completed",
                    completion_evidence=("紫色方块和绿色终点均可见",),
                ),
                replace(graph.subgoals[1], status="active"),
            ),
            active_subgoal_id="move-source",
        )
        revised.validate()
        return revised

    @staticmethod
    def _two_endpoint_scene(*, clipped: bool = False) -> UIScene:
        scene = _scene()
        instruction = replace(
            scene.elements[0],
            element_id="instruction",
            role="button",
            meaning="select_drag_task",
            label="动作：把紫色方块拖到绿色终点",
            bounds=(0.1, 0.39, 0.9, 0.47),
            states={"goal_relevant": True, "fully_visible": True},
            evidence=("任务描述同时写有紫色方块和绿色终点",),
        )
        source = replace(
            scene.elements[0],
            element_id="source",
            role="image",
            meaning="purple_start_block",
            label="起点",
            bounds=(0.0 if clipped else 0.18, 0.68, 0.38, 0.82),
            states={"goal_relevant": False, "fully_visible": True},
            evidence=("粉紫色方块",),
        )
        destination = UIElement(
            element_id="destination",
            role="container",
            meaning="green_end_point",
            label="绿色终点",
            bounds=(0.55, 0.63, 0.85, 0.85),
            confidence=0.98,
            states={"goal_relevant": False, "fully_visible": True},
            evidence=("绿色虚线终点区域",),
        )
        return replace(scene, elements=(instruction, source, destination))

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

    def test_start_advances_one_presence_only_read_only_subgoal_without_action(self) -> None:
        initial = self._read_only_locate_graph()
        revised = self._advance_locate_graph(initial)
        planner = FakeDeepSeekPlanner(initial, replan_result=revised)
        scene = _scene(
            meaning="当前可编辑输入框",
            label="旧内容",
            role="input",
            states={
                "goal_relevant": True,
                "fully_visible": True,
                "focused": True,
                "value": "",
                "keyboard_layout": "qwerty",
                "keyboard_input_mode": "direct_latin",
            },
        )

        class InputQwen(FakeQwenObserver):
            def decide(self, **kwargs):
                decision = super().decide(**kwargs)
                input_decision = _decision(
                    kwargs["trusted_observation"].scene,
                    action_kind="input_verified_text",
                )
                decision.proposal = input_decision.proposal
                decision.target_region = input_decision.target_region
                return decision

        qwen = InputQwen()
        adapter = FakeAdapter(scene)
        with tempfile.TemporaryDirectory() as temp:
            session = self._orchestrator(planner, qwen, adapter).start(
                session_id="session-read-only-locate",
                raw_goal=initial.raw_user_goal,
                device_id="device-1",
                run_dir=Path(temp),
            )

        self.assertEqual("needs_reobservation", session.status)
        self.assertEqual(2, session.task_graph.revision)
        self.assertEqual("replace-input", session.task_graph.active_subgoal_id)
        self.assertEqual(["subgoal_completed"], [call[2] for call in planner.replan_calls])
        self.assertEqual(0, len(qwen.calls))
        self.assertEqual(1, adapter.capture_calls)
        self.assertEqual(0, adapter.execute_calls)
        self.assertEqual(0, session.physical_actions)

    def test_start_advances_explicit_two_element_presence_checkpoint(self) -> None:
        initial = self._read_only_multi_locate_graph()
        revised = self._advance_multi_locate_graph(initial)
        planner = FakeDeepSeekPlanner(initial, replan_result=revised)
        qwen = FakeQwenObserver()
        adapter = FakeAdapter(self._two_endpoint_scene())

        with tempfile.TemporaryDirectory() as temp:
            session = self._orchestrator(planner, qwen, adapter).start(
                session_id="session-two-endpoints",
                raw_goal=initial.raw_user_goal,
                device_id="device-1",
                run_dir=Path(temp),
            )

        self.assertEqual(2, session.task_graph.revision)
        self.assertEqual("move-source", session.task_graph.active_subgoal_id)
        self.assertEqual(["subgoal_completed"], [call[2] for call in planner.replan_calls])
        self.assertEqual(0, len(qwen.calls))
        self.assertEqual("needs_reobservation", session.status)
        self.assertEqual(1, adapter.capture_calls)
        self.assertEqual(0, adapter.execute_calls)
        self.assertEqual(0, session.physical_actions)
        visible_evidence = planner.replan_calls[0][1].visible_evidence
        self.assertTrue(any("element_id=source" in item for item in visible_evidence))
        self.assertTrue(any("element_id=destination" in item for item in visible_evidence))

    def test_multi_element_presence_requires_safe_interior_bounds(self) -> None:
        initial = self._read_only_multi_locate_graph()
        planner = FakeDeepSeekPlanner(
            initial,
            replan_result=self._advance_multi_locate_graph(initial),
        )
        adapter = FakeAdapter(self._two_endpoint_scene(clipped=True))

        with tempfile.TemporaryDirectory() as temp:
            session = self._orchestrator(
                planner,
                FakeQwenObserver(),
                adapter,
            ).start(
                session_id="session-clipped-two-endpoints",
                raw_goal=initial.raw_user_goal,
                device_id="device-1",
                run_dir=Path(temp),
            )

        self.assertEqual("blocked", session.status)
        self.assertEqual([], planner.replan_calls)
        self.assertEqual(0, adapter.execute_calls)
        self.assertEqual(0, session.physical_actions)

    def test_multi_element_presence_rejects_aggregate_instruction_alone(self) -> None:
        initial = self._read_only_multi_locate_graph()
        planner = FakeDeepSeekPlanner(
            initial,
            replan_result=self._advance_multi_locate_graph(initial),
        )
        scene = self._two_endpoint_scene()
        adapter = FakeAdapter(replace(scene, elements=(scene.elements[0],)))

        with tempfile.TemporaryDirectory() as temp:
            session = self._orchestrator(
                planner,
                FakeQwenObserver(),
                adapter,
            ).start(
                session_id="session-unbound-two-endpoints",
                raw_goal=initial.raw_user_goal,
                device_id="device-1",
                run_dir=Path(temp),
            )

        self.assertEqual("blocked", session.status)
        self.assertEqual([], planner.replan_calls)
        self.assertEqual(0, adapter.execute_calls)

    def test_start_does_not_advance_presence_checkpoint_without_full_visibility(self) -> None:
        initial = self._read_only_locate_graph()
        planner = FakeDeepSeekPlanner(
            initial,
            replan_result=self._advance_locate_graph(initial),
        )
        scene = _scene(
            meaning="当前可编辑输入框",
            label="旧内容",
            role="input",
            states={"goal_relevant": True, "fully_visible": False},
        )
        qwen = FakeQwenObserver()
        adapter = FakeAdapter(scene)
        with tempfile.TemporaryDirectory() as temp:
            session = self._orchestrator(planner, qwen, adapter).start(
                session_id="session-clipped-locate",
                raw_goal=initial.raw_user_goal,
                device_id="device-1",
                run_dir=Path(temp),
            )

        self.assertEqual("blocked", session.status)
        self.assertEqual([], planner.replan_calls)
        self.assertEqual([], qwen.calls)
        self.assertEqual(0, adapter.execute_calls)
        self.assertEqual(0, session.physical_actions)

    def test_start_does_not_use_presence_to_verify_element_value(self) -> None:
        initial = self._read_only_locate_graph()
        verify_value = replace(
            initial,
            subgoals=(
                replace(
                    initial.subgoals[0],
                    objective="验证输入框文字内容是否为 agent",
                    completion_conditions=("输入框文字等于 agent",),
                ),
                initial.subgoals[1],
            ),
        )
        verify_value.validate()
        navigation = replace(
            verify_value,
            revision=verify_value.revision + 1,
            subgoals=(
                replace(
                    verify_value.subgoals[0],
                    status="pending",
                    depends_on=("show-target-page",),
                ),
                verify_value.subgoals[1],
                Subgoal(
                    subgoal_id="show-target-page",
                    objective="目标输入框所在页面可见",
                    status="active",
                    depends_on=(),
                    constraints=("不得提交、搜索或发送",),
                    completion_conditions=("目标输入框所在页面可见",),
                    completion_evidence=(),
                    risk_action_ids=(),
                    external_impact="navigation_only",
                ),
            ),
            active_subgoal_id="show-target-page",
        )
        navigation.validate()
        planner = FakeDeepSeekPlanner(
            verify_value,
            replan_result=navigation,
        )
        scene = _scene(
            meaning="当前可编辑输入框",
            label="Agent123",
            role="input",
            states={"goal_relevant": True, "fully_visible": True},
        )
        qwen = FakeQwenObserver()
        adapter = FakeAdapter(scene)
        with tempfile.TemporaryDirectory() as temp:
            session = self._orchestrator(planner, qwen, adapter).start(
                session_id="session-verify-value",
                raw_goal=verify_value.raw_user_goal,
                device_id="device-1",
                run_dir=Path(temp),
            )

        self.assertEqual("needs_reobservation", session.status)
        self.assertEqual(2, session.task_graph.revision)
        self.assertEqual("show-target-page", session.task_graph.active_subgoal_id)
        self.assertEqual(
            ["observation_changed"],
            [call[2] for call in planner.replan_calls],
        )
        self.assertEqual(0, len(qwen.calls))
        self.assertEqual(0, adapter.execute_calls)
        self.assertEqual(0, session.physical_actions)

    def test_start_rejects_read_only_replan_that_changes_future_semantics(self) -> None:
        initial = self._read_only_locate_graph()
        valid = self._advance_locate_graph(initial)
        mutated = replace(
            valid,
            subgoals=(
                valid.subgoals[0],
                replace(
                    valid.subgoals[1],
                    completion_conditions=("目标状态清晰可见",),
                ),
            ),
        )
        mutated.validate()
        planner = FakeDeepSeekPlanner(initial, replan_result=mutated)
        scene = _scene(
            meaning="当前可编辑输入框",
            label="旧内容",
            role="input",
            states={"goal_relevant": True, "fully_visible": True},
        )
        adapter = FakeAdapter(scene)
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(
                UniversalAgentOrchestratorError,
                "只能改变子目标状态",
            ):
                self._orchestrator(
                    planner,
                    FakeQwenObserver(),
                    adapter,
                ).start(
                    session_id="session-mutated-read-only",
                    raw_goal=initial.raw_user_goal,
                    device_id="device-1",
                    run_dir=Path(temp),
                )

        self.assertEqual(0, adapter.execute_calls)

    def test_refresh_reobserves_and_redecides_without_physical_action(self) -> None:
        adapter = FakeAdapter(_scene())
        qwen = FakeQwenObserver()
        with tempfile.TemporaryDirectory() as temp:
            orchestrator = self._orchestrator(
                FakeDeepSeekPlanner(_graph()), qwen, adapter
            )
            session = orchestrator.start(
                session_id="session-refresh",
                raw_goal="查看详情",
                device_id="device-1",
                run_dir=Path(temp),
            )
            old_scope = dict(session.snapshot()["confirmation_scope"])

            orchestrator.refresh_decision(session)

        self.assertEqual(2, adapter.capture_calls)
        self.assertEqual(0, adapter.execute_calls)
        self.assertEqual(2, len(qwen.calls))
        self.assertEqual(0, session.physical_actions)
        self.assertEqual("awaiting_confirmation", session.status)
        self.assertNotEqual(
            old_scope["observation_id"],
            session.snapshot()["confirmation_scope"]["observation_id"],
        )

    def test_payment_waits_for_local_effect_confirmation_before_qwen_and_robot(self) -> None:
        adapter = FakeAdapter(_scene())
        qwen = FakeQwenObserver()
        with tempfile.TemporaryDirectory() as temp:
            session = self._orchestrator(
                FakeDeepSeekPlanner(_external_graph()), qwen, adapter
            ).start(
                session_id="session-risk",
                raw_goal="向商户付款20元",
                device_id="device-1",
                run_dir=Path(temp),
            )

        self.assertEqual("awaiting_effect_confirmation", session.status)
        self.assertTrue(session.snapshot()["effect_confirmation_ready"])
        self.assertEqual(["risk-1"], session.snapshot()["effect_confirmation_scope"]["effect_ids"])
        self.assertRegex(
            session.snapshot()["effect_confirmation_scope"]["intent_digest"],
            r"^[0-9a-f]{64}$",
        )
        preview = session.snapshot()["effect_confirmation_preview"]
        self.assertEqual("typed_effects", preview["kind"])
        self.assertEqual(["risk-1"], preview["effect_ids"])
        self.assertEqual("financial_transaction", preview["effects"][0]["kind"])
        self.assertEqual(0, len(qwen.calls))
        self.assertEqual(1, adapter.capture_calls)
        self.assertEqual(0, adapter.execute_calls)
        self.assertEqual(0, session.physical_actions)

    def test_unknown_impact_waits_for_effect_confirmation_before_qwen_and_robot(self) -> None:
        adapter = FakeAdapter(_scene())
        qwen = FakeQwenObserver()
        unknown_graph = _external_graph(impact="unknown")
        with tempfile.TemporaryDirectory() as temp:
            session = self._orchestrator(
                FakeDeepSeekPlanner(
                    unknown_graph,
                    replan_result=replace(unknown_graph, revision=2),
                ),
                qwen,
                adapter,
            ).start(
                session_id="session-unknown",
                raw_goal="处理影响尚不明确的状态",
                device_id="device-1",
                run_dir=Path(temp),
            )

        self.assertEqual("awaiting_effect_confirmation", session.status)
        self.assertEqual([], qwen.calls)
        self.assertEqual(1, adapter.capture_calls)
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

    def test_qwen_finished_may_advance_one_completed_subgoal_then_reobserve(self) -> None:
        base = _graph()
        first = replace(
            base.subgoals[0],
            objective="确认当前公开列表可见",
            completion_conditions=("公开列表可见",),
        )
        second = Subgoal(
            subgoal_id="subgoal-2",
            objective="查看下一公开详情",
            status="pending",
            depends_on=(first.subgoal_id,),
            constraints=("不得改变任何账号状态",),
            completion_conditions=("下一公开详情可见",),
            completion_evidence=(),
            risk_action_ids=(),
            external_impact="navigation_only",
        )
        initial = replace(
            base,
            subgoals=(first, second),
            active_subgoal_id=first.subgoal_id,
        )
        initial.validate()
        revised = replace(
            initial,
            revision=2,
            subgoals=(
                replace(
                    first,
                    status="completed",
                    completion_evidence=("公开列表可见",),
                ),
                replace(second, status="active"),
            ),
            active_subgoal_id=second.subgoal_id,
        )
        revised.validate()

        with tempfile.TemporaryDirectory() as temp:
            session = self._orchestrator(
                FakeDeepSeekPlanner(initial, replan_result=revised),
                FakeQwenObserver("finished"),
                FakeAdapter(_scene()),
            ).start(
                session_id="session-finished-prefix",
                raw_goal="先确认公开列表，再查看下一公开详情",
                device_id="device-1",
                run_dir=Path(temp),
            )

        self.assertEqual("needs_reobservation", session.status)
        self.assertEqual(2, session.task_graph.revision)
        self.assertEqual("completed", session.task_graph.subgoals[0].status)
        self.assertEqual("subgoal-2", session.task_graph.active_subgoal_id)
        self.assertIsNone(session.controller_decision)
        self.assertIsNone(session.confirmation_authority)
        self.assertEqual(0, session.physical_actions)

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

    def test_start_persists_deepseek_failure_before_any_device_observation(self) -> None:
        class FailingPlanner:
            last_raw_response = json.dumps(
                {
                    "completion_conditions": [
                        {"evidence_required": ["tap the visible result"]}
                    ]
                }
            )

            def plan(self, *_args, **_kwargs):
                raise TaskGraphError(
                    "DeepSeek 高层任务图包含低层动作表达："
                    "completion_conditions.evidence_required"
                )

        adapter = FakeAdapter(_scene())
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp)
            with self.assertRaisesRegex(TaskGraphError, "低层动作表达"):
                self._orchestrator(
                    FailingPlanner(), FakeQwenObserver(), adapter
                ).start(
                    session_id="session-deepseek-failure",
                    raw_goal="查看本机公开信息",
                    device_id="device-1",
                    run_dir=run_dir,
                )
            diagnostics = list(run_dir.glob("*_deepseek_failure.json"))
            session = json.loads((run_dir / "session.json").read_text("utf-8"))
            report = json.loads((run_dir / "report.json").read_text("utf-8"))

        self.assertEqual(1, len(diagnostics))
        self.assertEqual("failed", session["status"])
        self.assertEqual(0, session["physical_actions"])
        self.assertIn(str(diagnostics[0]), session["evidence"])
        self.assertIn(str(diagnostics[0]), report["session"]["evidence"])
        self.assertEqual(0, adapter.capture_calls)
        self.assertEqual(0, adapter.execute_calls)

    def test_diagnostic_write_failure_does_not_mask_deepseek_error(self) -> None:
        class FailingPlanner:
            last_raw_response = "{}"

            def plan(self, *_args, **_kwargs):
                raise TaskGraphError("original task graph failure")

        adapter = FakeAdapter(_scene())
        with tempfile.TemporaryDirectory() as temp, patch(
            "universal_agent_orchestrator.persist_deepseek_failure_diagnostic",
            side_effect=OSError("diagnostic disk failure"),
        ):
            with self.assertRaisesRegex(TaskGraphError, "original task graph failure"):
                self._orchestrator(
                    FailingPlanner(), FakeQwenObserver(), adapter
                ).start(
                    session_id="session-diagnostic-failure",
                    raw_goal="查看本机公开信息",
                    device_id="device-1",
                    run_dir=Path(temp),
                )
            session = json.loads((Path(temp) / "session.json").read_text("utf-8"))

        self.assertEqual("failed", session["status"])
        self.assertEqual(0, session["physical_actions"])
        self.assertEqual(0, adapter.capture_calls)
        self.assertEqual(0, adapter.execute_calls)

    def test_launcher_app_entry_cannot_prove_named_target_app_page(self) -> None:
        cases = (
            ("browser", "浏览器", "open_browser"),
            ("phone_manager", "手机管家", "open_phone_manager"),
            ("settings", "设置", "open_settings"),
        )
        for app_id, app_name, meaning in cases:
            with self.subTest(app_id=app_id):
                graph = self._named_app_page_graph(
                    app_id=app_id,
                    app_name=app_name,
                )
                planner = FakeDeepSeekPlanner(graph)
                qwen = FakeQwenObserver()
                launcher = replace(
                    _scene(
                        meaning=meaning,
                        label=app_name,
                        states={"goal_relevant": True, "fully_visible": True},
                    ),
                    app_id="launcher",
                    screen_id="home_screen",
                    summary=f"手机主桌面可见，桌面上有{app_name}入口图标。",
                )
                adapter = FakeAdapter(launcher)

                with tempfile.TemporaryDirectory() as temp:
                    session = self._orchestrator(planner, qwen, adapter).start(
                        session_id=f"session-launcher-{app_id}",
                        raw_goal=graph.raw_user_goal,
                        device_id="device-1",
                        run_dir=Path(temp),
                    )

                self.assertEqual("awaiting_confirmation", session.status)
                self.assertEqual([], planner.replan_calls)
                self.assertEqual(1, len(qwen.calls))
                self.assertEqual("action", session.qwen_decision.proposal.status)
                self.assertEqual(0, session.physical_actions)

    def test_launcher_entry_cannot_prove_named_target_app_is_foreground(self) -> None:
        cases = (
            ("browser", "浏览器", "open_browser"),
            ("settings", "设置", "open_settings"),
            ("chat_app", "聊天工具", "open_chat"),
        )
        for app_id, app_name, meaning in cases:
            with self.subTest(app_id=app_id):
                base = self._named_app_page_graph(
                    app_id=app_id,
                    app_name=app_name,
                )
                graph = replace(
                    base,
                    goal=replace(
                        base.goal,
                        objective=f"确认{app_name}应用在前台可见",
                    ),
                    subgoals=(
                        replace(
                            base.subgoals[0],
                            objective=f"{app_name}应用在前台可见",
                            completion_conditions=(
                                f"{app_name}应用在前台可见",
                            ),
                        ),
                        base.subgoals[1],
                    ),
                    raw_user_goal=f"确认{app_name}应用在前台可见",
                )
                graph.validate()
                planner = FakeDeepSeekPlanner(
                    graph,
                    replan_result=self._advance_named_app_page_graph(graph),
                )
                launcher = replace(
                    _scene(
                        meaning=meaning,
                        label=app_name,
                        states={"goal_relevant": True, "fully_visible": True},
                    ),
                    app_id="launcher",
                    screen_id="home_screen",
                    summary=f"手机桌面显示{app_name}入口图标。",
                )
                qwen = FakeQwenObserver()

                with tempfile.TemporaryDirectory() as temp:
                    session = self._orchestrator(
                        planner,
                        qwen,
                        FakeAdapter(launcher),
                    ).start(
                        session_id=f"session-foreground-{app_id}",
                        raw_goal=graph.raw_user_goal,
                        device_id="device-1",
                        run_dir=Path(temp),
                    )

                self.assertEqual("awaiting_confirmation", session.status)
                self.assertEqual([], planner.replan_calls)
                self.assertEqual(1, len(qwen.calls))
                self.assertEqual(0, session.physical_actions)

    def test_replan_launcher_entry_cannot_complete_named_foreground_app(self) -> None:
        for app_id, app_name, meaning in (
            ("wechat", "微信", "open_wechat"),
            ("settings", "设置", "open_settings"),
            ("browser", "浏览器", "open_browser"),
        ):
            with self.subTest(app_id=app_id):
                previous = self._named_app_page_graph(
                    app_id=app_id,
                    app_name=app_name,
                )
                revised = self._advance_named_app_page_graph(previous)
                launcher = replace(
                    _scene(
                        meaning=meaning,
                        label=app_name,
                        states={"goal_relevant": True, "fully_visible": True},
                    ),
                    app_id="launcher",
                    screen_id="home_screen",
                    summary=f"手机桌面显示{app_name}入口图标。",
                )

                with self.assertRaisesRegex(
                    UniversalAgentOrchestratorError,
                    "入口不能证明目标 App 页面已在前台",
                ):
                    UniversalAgentOrchestrator._validate_graph_identity(
                        revised,
                        device_id="device-1",
                        previous=previous,
                        trusted_observation=SimpleNamespace(scene=launcher),
                    )

    @staticmethod
    def _named_app_launch_transition_proof(
        *,
        app_id: str,
        app_name: str,
        after_app_id: str,
    ) -> tuple[DynamicTaskGraph, DynamicTaskGraph, dict]:
        previous = UniversalAgentStartTests._named_app_page_graph(
            app_id=app_id,
            app_name=app_name,
        )
        revised = UniversalAgentStartTests._advance_named_app_page_graph(
            previous
        )
        before_scene = replace(
            _scene(
                fingerprint=f"before-{app_id}",
                meaning=f"open_{app_id}",
                label=app_name,
                states={"goal_relevant": True, "fully_visible": True},
            ),
            app_id="launcher",
            screen_id="home_screen",
            summary=f"手机主桌面显示{app_name}入口。",
        )
        after_scene = replace(
            _scene(
                fingerprint=f"after-{app_id}",
                meaning="current_page_title",
                label="当前页面",
                role="text",
            ),
            app_id=after_app_id,
            screen_id=f"{after_app_id}_home",
            summary=f"{app_name}打开后的当前功能页面。",
        )
        before = FakeTrustedObservation(
            device_id="device-1",
            scene=before_scene,
            observation_id=f"obs-before-{app_id}",
        )
        after = FakeTrustedObservation(
            device_id="device-1",
            scene=after_scene,
            observation_id=f"obs-after-{app_id}",
        )
        element = before_scene.elements[0]
        action = SemanticAction(
            node_id=f"node-open-{app_id}",
            action="tap_semantic",
            params={
                "element_id": element.element_id,
                "target": element.meaning,
                "role": element.role,
                "label": element.label,
                "formal_transition": {
                    "transition_id": f"transition-open-{app_id}",
                    "precondition_claim_ids": ["claim-launcher-entry"],
                    "expectations": [
                        {
                            "subject_ref": "surface_current",
                            "predicate": "surface.active_ref",
                            "operator": "equals",
                            "value": f"surface_{app_id}",
                        }
                    ],
                    "exploratory": False,
                },
                "expected_effect": {
                    "scene_changed": True,
                    "goal_complete_on_success": True,
                },
            },
        )
        resolved = ResolvedSemanticAction(
            node_id=action.node_id,
            kind=action.action,
            normalized_point=(0.3, 0.25),
            target_element_id=element.element_id,
            before_fingerprint=before.fingerprint,
            expected_effect=dict(action.params["expected_effect"]),
        )
        result = SimpleNamespace(rebound_action=action, resolved_action=resolved)
        receipt = VerifiedActionTransition(
            receipt_id=f"receipt-open-{app_id}",
            session_id=f"session-open-{app_id}",
            task_id=previous.task_id,
            device_id=previous.device_id,
            prior_revision=previous.revision,
            subgoal_id=previous.active_subgoal_id,
            decision_node_id=action.node_id,
            action_digest=_action_digest(action),
            rebound_action_digest=_action_digest(action),
            resolved_action_digest=_action_digest(resolved),
            action_kind=action.action,
            before_observation_id=before.observation_id,
            before_fingerprint=before.fingerprint,
            after_observation_id=after.observation_id,
            after_fingerprint=after.fingerprint,
            physical_actions=1,
            outcome="matched",
            errors=(),
            controller_completion_evidence=(f"已打开{app_name}",),
        )
        ref = ControllerTransitionEvidenceRef(
            ref_id=f"controller_transition:{receipt.receipt_id}:1",
            receipt_id=receipt.receipt_id,
            subgoal_id=receipt.subgoal_id,
            text=f"已打开{app_name}",
        )
        kwargs = {
            "device_id": "device-1",
            "previous": previous,
            "trusted_observation": after,
            "session_id": receipt.session_id,
            "verified_transition": receipt,
            "controller_transition_evidence_refs": (ref,),
            "before_observation": before,
            "previous_decision": SimpleNamespace(
                proposal=GenericStepProposal(status="action", action=action),
                trusted_observation=before,
            ),
            "execution_result": result,
        }
        return previous, revised, kwargs

    def test_matched_launcher_transition_proves_functionally_classified_app_surface(
        self,
    ) -> None:
        for app_id, app_name, functional_app_id in (
            ("browser", "浏览器", "news_aggregator"),
            ("music", "音乐", "media_library"),
        ):
            with self.subTest(app_id=app_id):
                _previous, revised, kwargs = self._named_app_launch_transition_proof(
                    app_id=app_id,
                    app_name=app_name,
                    after_app_id=functional_app_id,
                )
                UniversalAgentOrchestrator._validate_graph_identity(
                    revised,
                    **kwargs,
                )

    def test_named_app_launch_transition_requires_complete_exact_binding(self) -> None:
        _previous, revised, base = self._named_app_launch_transition_proof(
            app_id="browser",
            app_name="浏览器",
            after_app_id="news_aggregator",
        )
        cases: list[tuple[str, dict]] = []
        cases.append(("missing_controller_ref", {**base, "controller_transition_evidence_refs": ()}))
        cases.append(("wrong_session", {**base, "session_id": "session-other"}))
        cases.append(
            (
                "wrong_after_fingerprint",
                {
                    **base,
                    "verified_transition": replace(
                        base["verified_transition"],
                        after_fingerprint="after-other",
                    ),
                },
            )
        )
        mismatched = replace(
            base["verified_transition"],
            outcome="mismatched",
            errors=("目标页面未出现",),
        )
        cases.append(("mismatched", {**base, "verified_transition": mismatched}))

        old_action = base["previous_decision"].proposal.action
        wrong_surface_action = replace(
            old_action,
            params={
                **old_action.params,
                "formal_transition": {
                    **old_action.params["formal_transition"],
                    "expectations": [
                        {
                            "subject_ref": "surface_current",
                            "predicate": "surface.active_ref",
                            "operator": "equals",
                            "value": "surface_music",
                        }
                    ],
                },
            },
        )
        wrong_surface_result = SimpleNamespace(
            rebound_action=wrong_surface_action,
            resolved_action=base["execution_result"].resolved_action,
        )
        cases.append(
            (
                "wrong_surface",
                {
                    **base,
                    "previous_decision": SimpleNamespace(
                        proposal=GenericStepProposal(
                            status="action",
                            action=wrong_surface_action,
                        ),
                        trusted_observation=base["before_observation"],
                    ),
                    "verified_transition": replace(
                        base["verified_transition"],
                        action_digest=_action_digest(wrong_surface_action),
                        rebound_action_digest=_action_digest(wrong_surface_action),
                    ),
                    "execution_result": wrong_surface_result,
                },
            )
        )

        launcher_after = FakeTrustedObservation(
            device_id="device-1",
            scene=replace(
                base["trusted_observation"].scene,
                app_id="launcher",
                screen_id="home_screen",
            ),
            observation_id=base["trusted_observation"].observation_id,
        )
        launcher_receipt = replace(
            base["verified_transition"],
            after_fingerprint=launcher_after.fingerprint,
        )
        cases.append(
            (
                "still_launcher",
                {
                    **base,
                    "trusted_observation": launcher_after,
                    "verified_transition": launcher_receipt,
                },
            )
        )

        for name, kwargs in cases:
            with self.subTest(case=name), self.assertRaisesRegex(
                UniversalAgentOrchestratorError,
                "入口不能证明目标 App 页面已在前台",
            ):
                UniversalAgentOrchestrator._validate_graph_identity(
                    revised,
                    **kwargs,
                )

    def test_unique_visible_title_advances_with_verified_surface_lineage(self) -> None:
        base = self._advance_named_app_page_graph(
            self._named_app_page_graph(app_id="browser", app_name="浏览器")
        )
        source = replace(
            base.subgoals[0],
            completion_evidence=("controller_transition:receipt-browser:1",),
        )
        read = replace(
            base.subgoals[1],
            objective="读取浏览器打开后页面的主标题或错误提示",
            completion_conditions=("已获取页面主标题或错误提示文本",),
            external_impact="read_only",
        )
        finish = Subgoal(
            subgoal_id="return-home",
            objective="返回手机桌面",
            status="pending",
            depends_on=(read.subgoal_id,),
            constraints=(),
            completion_conditions=("手机桌面可见",),
            completion_evidence=(),
            risk_action_ids=(),
            external_impact="navigation_only",
        )
        graph = replace(base, subgoals=(source, read, finish))
        graph.validate()
        visible_fact = (
            "当前可信画面读取结果：element_id=title-1, role=text, "
            "meaning=page_title, label=要闻。"
        )
        revised = replace(
            graph,
            revision=graph.revision + 1,
            subgoals=(
                source,
                replace(read, status="completed", completion_evidence=(visible_fact,)),
                replace(finish, status="active"),
            ),
            active_subgoal_id=finish.subgoal_id,
        )
        revised.validate()
        scene = replace(
            _scene(
                fingerprint="news-current",
                meaning="page_title",
                label="要闻",
                role="text",
                states={"goal_relevant": True, "fully_visible": True},
            ),
            app_id="news_aggregator",
            screen_id="news_feed",
            summary="当前页面顶部主标题为要闻。",
            elements=(
                replace(
                    _scene(meaning="page_title", label="要闻", role="text").elements[0],
                    element_id="title-1",
                ),
            ),
        )
        observation = FakeTrustedObservation(
            device_id="device-1", scene=scene, observation_id="obs-title"
        )
        lineage = VerifiedAppSurfaceLineage(
            session_id="session-title",
            task_id=graph.task_id,
            device_id=graph.device_id,
            app_id="browser",
            app_name="浏览器",
            surface_id="surface_browser",
            source_receipt_id="receipt-browser",
            source_subgoal_id=source.subgoal_id,
            functional_foreground_app_id="news_aggregator",
            physical_actions=1,
        )
        planner = FakeDeepSeekPlanner(graph, replan_result=revised)
        orchestrator = self._orchestrator(planner, FakeQwenObserver(), FakeAdapter(scene))
        session = SimpleNamespace(
            session_id="session-title",
            device_id="device-1",
            verified_app_surface_lineage=lineage,
            physical_actions=1,
        )

        actual = orchestrator._try_advance_visible_text_read_subgoal(
            session,
            graph=graph,
            trusted_observation=observation,
        )

        self.assertEqual(revised, actual)
        self.assertEqual("subgoal_completed", planner.replan_calls[0][2])
        self.assertIn(visible_fact, planner.replan_calls[0][1].visible_evidence)

        with self.assertRaisesRegex(
            UniversalAgentOrchestratorError,
            "入口不能证明目标 App 页面已在前台",
        ):
            UniversalAgentOrchestrator._validate_graph_identity(
                revised,
                device_id="device-1",
                previous=graph,
                trusted_observation=observation,
                session_id="session-title",
                verified_app_surface_lineage=lineage,
                physical_actions=2,
            )

    def test_verified_app_entry_lineage_rebinds_runtime_package_for_next_step(self) -> None:
        base = self._named_app_page_graph(app_id="wechat", app_name="微信")
        graph = self._advance_named_app_page_graph(base)
        source = replace(
            graph.subgoals[0],
            completion_evidence=("controller_transition:receipt-wechat:1",),
        )
        graph = replace(graph, subgoals=(source, *graph.subgoals[1:]))
        graph.validate()
        semantic_ir = compile_formal_semantic_authority(graph).semantic_ir
        context = replace(
            QwenTaskContext.from_dict(graph.to_qwen_context()),
            semantic_ir=semantic_ir,
        )
        scene = replace(
            _scene(),
            app_id="com.tencent.mm",
            screen_id="wechat_chat_list",
        )
        lineage = VerifiedAppSurfaceLineage(
            session_id="session-wechat",
            task_id=graph.task_id,
            device_id=graph.device_id,
            app_id="wechat",
            app_name="微信",
            surface_id="surface_wechat",
            source_receipt_id="receipt-wechat",
            source_subgoal_id=source.subgoal_id,
            functional_foreground_app_id="com.tencent.mm",
            physical_actions=1,
        )
        session = SimpleNamespace(
            session_id="session-wechat",
            device_id=graph.device_id,
            task_graph=graph,
            physical_actions=1,
            verified_app_surface_lineage=lineage,
        )

        rebound = UniversalAgentOrchestrator._bind_verified_lineage_to_qwen_context(
            session,
            context,
            SimpleNamespace(scene=scene),
        )

        target = next(
            item for item in rebound.semantic_ir.surfaces
            if item.surface_id == "surface_wechat"
        )
        self.assertEqual("com.tencent.mm", target.app_id)
        self.assertTrue(_scene_matches_target_app_surface(scene, target))

        session.physical_actions = 2
        unchanged = UniversalAgentOrchestrator._bind_verified_lineage_to_qwen_context(
            session,
            context,
            SimpleNamespace(scene=scene),
        )
        original = next(
            item for item in unchanged.semantic_ir.surfaces
            if item.surface_id == "surface_wechat"
        )
        self.assertEqual("wechat", original.app_id)

    def test_verified_lineage_survives_exact_display_name_observation(self) -> None:
        lineage = VerifiedAppSurfaceLineage(
            session_id="session-wechat",
            task_id="task-wechat",
            device_id="device-1",
            app_id="wechat",
            app_name="微信",
            surface_id="surface_wechat",
            source_receipt_id="receipt-wechat",
            source_subgoal_id="open-wechat",
            functional_foreground_app_id="com.tencent.mm",
            physical_actions=1,
        )
        self.assertTrue(
            UniversalAgentOrchestrator._lineage_matches_observed_foreground(
                lineage,
                "微信",
            )
        )
        self.assertTrue(
            UniversalAgentOrchestrator._lineage_matches_observed_foreground(
                lineage,
                "com.tencent.mm",
            )
        )
        self.assertFalse(
            UniversalAgentOrchestrator._lineage_matches_observed_foreground(
                lineage,
                "launcher",
            )
        )
        self.assertFalse(
            UniversalAgentOrchestrator._lineage_matches_observed_foreground(
                lineage,
                "com.example.other",
            )
        )

    def test_visible_text_read_rejects_exact_or_ambiguous_results(self) -> None:
        subgoal = SimpleNamespace(
            external_impact="read_only",
            objective="读取主标题是否为指定文字",
            completion_conditions=("主标题等于指定文字",),
        )
        self.assertFalse(
            UniversalAgentOrchestrator._is_visible_text_read_subgoal(subgoal)
        )

        graph = self._named_app_page_graph(app_id="browser", app_name="浏览器")
        read = replace(
            graph.subgoals[0],
            objective="读取页面主标题或错误提示",
            completion_conditions=("已获取主标题或错误提示文本",),
            external_impact="read_only",
        )
        graph = replace(graph, subgoals=(read, graph.subgoals[1]))
        graph.validate()
        first = _scene(meaning="page_title", label="标题一", role="text").elements[0]
        ambiguous_scene = replace(
            _scene(),
            elements=(first, replace(first, element_id="title-2", label="标题二")),
        )
        planner = FakeDeepSeekPlanner(graph)
        result = self._orchestrator(
            planner, FakeQwenObserver(), FakeAdapter(ambiguous_scene)
        )._try_advance_visible_text_read_subgoal(
            SimpleNamespace(
                session_id="session-ambiguous",
                device_id="device-1",
                verified_app_surface_lineage=None,
                physical_actions=0,
            ),
            graph=graph,
            trusted_observation=FakeTrustedObservation(
                device_id="device-1", scene=ambiguous_scene
            ),
        )
        self.assertIsNone(result)
        self.assertEqual([], planner.replan_calls)

    def test_replan_structured_foreground_app_and_launcher_checkpoint_pass(self) -> None:
        previous = self._named_app_page_graph(
            app_id="wechat",
            app_name="微信",
        )
        revised = self._advance_named_app_page_graph(previous)
        wechat_scene = replace(
            _scene(role="text", meaning="wechat_home_title", label="微信"),
            app_id="wechat",
            screen_id="wechat_home",
            summary="微信应用首页可见。",
        )
        UniversalAgentOrchestrator._validate_graph_identity(
            revised,
            device_id="device-1",
            previous=previous,
            trusted_observation=SimpleNamespace(scene=wechat_scene),
        )

        launcher_previous = replace(
            previous,
            goal=replace(previous.goal, objective="先确认系统主桌面可见"),
            subgoals=(
                replace(
                    previous.subgoals[0],
                    objective="系统主桌面在前台可见",
                    completion_conditions=("系统主桌面在前台可见",),
                ),
                previous.subgoals[1],
            ),
            raw_user_goal="先确认系统主桌面可见，再进入微信",
        )
        launcher_previous.validate()
        launcher_revised = self._advance_named_app_page_graph(launcher_previous)
        launcher_scene = replace(
            _scene(meaning="open_wechat", label="微信"),
            app_id="launcher",
            screen_id="home_screen",
            summary="系统主桌面可见。",
        )
        UniversalAgentOrchestrator._validate_graph_identity(
            launcher_revised,
            device_id="device-1",
            previous=launcher_previous,
            trusted_observation=SimpleNamespace(scene=launcher_scene),
        )

    def test_matching_target_app_can_prove_foreground_wording(self) -> None:
        base = self._named_app_page_graph(
            app_id="local_tool",
            app_name="本地工具",
        )
        graph = replace(
            base,
            subgoals=(
                replace(
                    base.subgoals[0],
                    objective="本地工具应用在前台可见",
                    completion_conditions=(
                        "本地工具应用在前台可见",
                    ),
                ),
                base.subgoals[1],
            ),
        )
        graph.validate()
        planner = FakeDeepSeekPlanner(
            graph,
            replan_result=self._advance_named_app_page_graph(graph),
        )
        foreground = replace(
            _scene(
                meaning="local_tool_title",
                label="本地工具",
                role="text",
                states={"goal_relevant": True, "fully_visible": True},
            ),
            app_id="local_tool",
            screen_id="local_tool_home",
            summary="本地工具应用在前台可见。",
        )

        with tempfile.TemporaryDirectory() as temp:
            session = self._orchestrator(
                planner,
                FakeQwenObserver(),
                FakeAdapter(foreground),
            ).start(
                session_id="session-matching-foreground",
                raw_goal=graph.raw_user_goal,
                device_id="device-1",
                run_dir=Path(temp),
            )

        self.assertEqual(2, session.task_graph.revision)
        self.assertEqual("safe-followup", session.task_graph.active_subgoal_id)
        self.assertEqual(1, len(planner.replan_calls))
        self.assertEqual(0, session.physical_actions)
        self.assertIn(
            "foreground_app",
            UniversalAgentOrchestrator._presence_surface_classes(
                "Local tool is in the foreground"
            ),
        )

    def test_settings_search_page_cannot_prove_named_launcher_page(self) -> None:
        base = self._named_app_page_graph(
            app_id="current_foreground",
            app_name="当前前台应用",
        )
        graph = replace(
            base,
            goal=replace(base.goal, objective="让手机主桌面页面成为当前前台"),
            subgoals=(
                replace(
                    base.subgoals[0],
                    objective="手机主桌面页面可见",
                    completion_conditions=("手机主桌面页面在前台可见",),
                ),
                base.subgoals[1],
            ),
            raw_user_goal="让手机主桌面页面成为当前前台",
        )
        graph.validate()
        planner = FakeDeepSeekPlanner(
            graph,
            replan_result=self._advance_named_app_page_graph(graph),
        )
        settings = replace(
            _scene(
                meaning="search_settings",
                label="搜索系统设置项",
                role="input",
                states={"goal_relevant": True, "fully_visible": True},
            ),
            app_id="com.android.settings",
            screen_id="settings_search",
            summary="设置搜索页面，底部软键盘可见。",
        )

        with tempfile.TemporaryDirectory() as temp:
            session = self._orchestrator(
                planner,
                FakeQwenObserver(),
                FakeAdapter(settings),
            ).start(
                session_id="session-settings-not-launcher",
                raw_goal=graph.raw_user_goal,
                device_id="device-1",
                run_dir=Path(temp),
            )

        self.assertEqual("blocked", session.status)
        self.assertEqual([], planner.replan_calls)
        self.assertEqual(0, session.physical_actions)

    def test_matching_foreground_app_can_prove_named_target_app_page(self) -> None:
        graph = self._named_app_page_graph(
            app_id="local_tool",
            app_name="本地工具",
        )
        planner = FakeDeepSeekPlanner(
            graph,
            replan_result=self._advance_named_app_page_graph(graph),
        )
        page = replace(
            _scene(
                meaning="local_tool_home_title",
                label="本地工具",
                role="text",
                states={"goal_relevant": True, "fully_visible": True},
            ),
            app_id="local_tool",
            screen_id="local_tool_home",
            summary="本地工具首页可见。",
        )

        with tempfile.TemporaryDirectory() as temp:
            session = self._orchestrator(
                planner,
                FakeQwenObserver(),
                FakeAdapter(page),
            ).start(
                session_id="session-target-app-page",
                raw_goal=graph.raw_user_goal,
                device_id="device-1",
                run_dir=Path(temp),
            )

        self.assertEqual(2, session.task_graph.revision)
        self.assertEqual("safe-followup", session.task_graph.active_subgoal_id)
        self.assertEqual(["subgoal_completed"], [call[2] for call in planner.replan_calls])
        self.assertEqual(0, session.physical_actions)

    def test_start_advances_unique_list_item_selected_by_visible_title_prefix(self) -> None:
        self.assertEqual(
            ("通用动作真机验",),
            UniversalAgentOrchestrator._presence_title_prefixes(
                "定位标题开头为通用动作真机验的唯一卡片"
            ),
        )
        self.assertEqual(
            ("weekly report",),
            UniversalAgentOrchestrator._presence_title_prefixes(
                "locate the card whose title starts with Weekly Report card"
            ),
        )
        base = _graph()
        locate = replace(
            base.subgoals[0],
            subgoal_id="locate-card",
            objective="定位标题开头为通用动作真机验的唯一卡片",
            completion_conditions=(
                "标题开头为通用动作真机验的卡片在列表中可见",
            ),
            external_impact="read_only",
        )
        click = Subgoal(
            subgoal_id="click-card",
            objective="点击已定位的唯一卡片",
            status="pending",
            depends_on=("locate-card",),
            constraints=("只点击该卡片",),
            completion_conditions=("目标页面已打开",),
            completion_evidence=(),
            risk_action_ids=(),
            external_impact="navigation_only",
        )
        initial = replace(
            base,
            goal=replace(
                base.goal,
                objective="点击标题开头为通用动作真机验的唯一卡片",
                target_apps=(),
                entities={
                    "target_ui_label": "通用动作真机验",
                    "target_surface": "current_surface",
                },
            ),
            subgoals=(locate, click),
            active_subgoal_id="locate-card",
            raw_user_goal="点击当前卡片列表中标题开头为通用动作真机验的唯一卡片",
        )
        initial.validate()
        visible_fact = "标题开头为通用动作真机验的卡片在列表中可见"
        revised = replace(
            initial,
            revision=2,
            subgoals=(
                replace(
                    locate,
                    status="completed",
                    completion_evidence=(visible_fact,),
                ),
                replace(click, status="active"),
            ),
            active_subgoal_id="click-card",
        )
        revised.validate()
        scene = _scene(
            meaning="app_task_card",
            label="通用动作真机验...",
            role="list_item",
            bounds=(0.53, 0.34, 0.94, 0.70),
            evidence=(visible_fact,),
            app_id="unknown",
        )
        planner = FakeDeepSeekPlanner(initial, replan_result=revised)
        adapter = FakeAdapter(scene)

        with tempfile.TemporaryDirectory() as temp:
            session = self._orchestrator(
                planner,
                FakeQwenObserver(),
                adapter,
            ).start(
                session_id="session-title-prefix-card",
                raw_goal=initial.raw_user_goal,
                device_id="device-1",
                run_dir=Path(temp),
            )

        self.assertEqual("needs_reobservation", session.status)
        self.assertEqual(2, session.task_graph.revision)
        self.assertEqual("click-card", session.task_graph.active_subgoal_id)
        self.assertEqual(0, adapter.execute_calls)
        self.assertEqual(0, session.physical_actions)

        for unsafe_scene in (
            replace(scene, elements=()),
            replace(
                scene,
                elements=(
                    scene.elements[0],
                    replace(scene.elements[0], element_id="duplicate-card"),
                ),
            ),
            replace(
                scene,
                elements=(
                    replace(
                        scene.elements[0],
                        states={"goal_relevant": True, "fully_visible": False},
                    ),
                ),
            ),
            replace(
                scene,
                elements=(
                    replace(
                        scene.elements[0],
                        label="其他卡片",
                        evidence=("仅可见其他卡片",),
                    ),
                ),
            ),
        ):
            unsafe_planner = FakeDeepSeekPlanner(initial, replan_result=revised)
            with tempfile.TemporaryDirectory() as temp:
                unsafe_session = self._orchestrator(
                    unsafe_planner,
                    FakeQwenObserver(),
                    FakeAdapter(unsafe_scene),
                ).start(
                    session_id="session-unsafe-title-prefix-card",
                    raw_goal=initial.raw_user_goal,
                    device_id="device-1",
                    run_dir=Path(temp),
                )
            self.assertEqual("blocked", unsafe_session.status)
            self.assertEqual([], unsafe_planner.replan_calls)
            self.assertEqual(0, unsafe_session.physical_actions)

    def test_exact_target_app_id_can_ground_deep_page_without_app_title(self) -> None:
        graph = self._named_app_page_graph(
            app_id="local_tool",
            app_name="本地工具",
        )
        planner = FakeDeepSeekPlanner(
            graph,
            replan_result=self._advance_named_app_page_graph(graph),
        )
        deep_page = replace(
            _scene(
                meaning="draft_input",
                label="",
                role="input",
                states={"goal_relevant": True, "fully_visible": True},
            ),
            app_id="local_tool",
            screen_id="conversation_detail",
            summary="当前是一个深层会话页面，底部输入区域可见。",
        )

        with tempfile.TemporaryDirectory() as temp:
            session = self._orchestrator(
                planner,
                FakeQwenObserver(),
                FakeAdapter(deep_page),
            ).start(
                session_id="session-target-app-deep-page",
                raw_goal=graph.raw_user_goal,
                device_id="device-1",
                run_dir=Path(temp),
            )

        self.assertEqual(2, session.task_graph.revision)
        self.assertEqual("safe-followup", session.task_graph.active_subgoal_id)
        self.assertEqual(["subgoal_completed"], [call[2] for call in planner.replan_calls])
        self.assertEqual(0, session.physical_actions)

    def test_shared_generic_app_token_does_not_match_other_foreground(self) -> None:
        graph = self._named_app_page_graph(
            app_id="target_app",
            app_name="目标工具",
        )
        other_app_scene = replace(
            _scene(),
            app_id="other_app",
            screen_id="other_app_home",
            summary="另一个工具首页可见。",
        )

        self.assertFalse(
            UniversalAgentOrchestrator._scene_foreground_matches_target_app_page(
                scene=other_app_scene,
                target_apps=graph.goal.target_apps,
            )
        )
        unknown_scene = replace(
            other_app_scene,
            app_id="unknown",
            screen_id="unknown_screen",
        )
        self.assertFalse(
            UniversalAgentOrchestrator._scene_foreground_matches_target_app_page(
                scene=unknown_scene,
                target_apps=graph.goal.target_apps,
            )
        )

    def test_target_app_does_not_block_launcher_home_presence_checkpoint(self) -> None:
        base = self._named_app_page_graph(app_id="camera", app_name="相机")
        graph = replace(
            base,
            goal=replace(base.goal, objective="先确认手机主桌面可见"),
            subgoals=(
                replace(
                    base.subgoals[0],
                    objective="手机主桌面可见",
                    completion_conditions=("手机主桌面可见",),
                ),
                base.subgoals[1],
            ),
            raw_user_goal="先确认主桌面再进入相机",
        )
        graph.validate()
        planner = FakeDeepSeekPlanner(
            graph,
            replan_result=self._advance_named_app_page_graph(graph),
        )
        launcher = replace(
            _scene(
                meaning="open_camera",
                label="相机",
                states={"goal_relevant": True, "fully_visible": True},
            ),
            app_id="launcher",
            screen_id="home_screen",
            summary="手机主桌面可见。",
        )

        with tempfile.TemporaryDirectory() as temp:
            session = self._orchestrator(
                planner,
                FakeQwenObserver(),
                FakeAdapter(launcher),
            ).start(
                session_id="session-launcher-home-checkpoint",
                raw_goal=graph.raw_user_goal,
                device_id="device-1",
                run_dir=Path(temp),
            )

        self.assertEqual("needs_reobservation", session.status)
        self.assertEqual("safe-followup", session.task_graph.active_subgoal_id)
        self.assertEqual(["subgoal_completed"], [call[2] for call in planner.replan_calls])
        self.assertEqual(0, session.physical_actions)

    def test_read_only_successor_requires_reobservation_without_releasing_device(self) -> None:
        initial = self._read_only_locate_graph()
        initial = replace(
            initial,
            subgoals=(
                initial.subgoals[0],
                replace(
                    initial.subgoals[1],
                    objective="核对另一个只读结果",
                    completion_conditions=("完成只读核对",),
                    external_impact="read_only",
                ),
            ),
        )
        initial.validate()
        revised = self._advance_locate_graph(initial)
        planner = FakeDeepSeekPlanner(initial, replan_result=revised)
        qwen = FakeQwenObserver()
        adapter = FakeAdapter(
            _scene(
                meaning="当前可编辑输入框",
                label="旧内容",
                role="input",
                states={
                    "goal_relevant": True,
                    "fully_visible": True,
                    "focused": True,
                    "value": "",
                    "keyboard_layout": "qwerty",
                    "keyboard_input_mode": "direct_latin",
                },
            )
        )

        with tempfile.TemporaryDirectory() as temp:
            orchestrator = self._orchestrator(planner, qwen, adapter)
            session = orchestrator.start(
                session_id="session-read-only-reobserve",
                raw_goal=initial.raw_user_goal,
                device_id="device-1",
                run_dir=Path(temp),
            )
            self.assertEqual("needs_reobservation", session.status)
            self.assertEqual(
                session.session_id,
                orchestrator.device_registry.active_session(session.device_id),
            )
            self.assertEqual(0, adapter.execute_calls)
            self.assertEqual(0, session.physical_actions)
            orchestrator.cancel(session)


def _confirmation(session) -> dict:
    return dict(session.snapshot()["confirmation_scope"])


def _effect_confirmation(session) -> dict:
    return dict(session.snapshot()["effect_confirmation_scope"])


def _failure_artifacts(temp: str, *, step_number: int = 1):
    root = Path(temp)
    transition = json.loads(
        (root / f"post_action_transition_step_{step_number}.json").read_text(
            encoding="utf-8"
        )
    )
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    return transition, report


def _confirmation_failure_artifacts(temp: str, *, step_number: int = 1):
    root = Path(temp)
    transition = json.loads(
        (root / f"confirmation_failure_step_{step_number}.json").read_text(
            encoding="utf-8"
        )
    )
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    return transition, report


class UniversalAgentOfflineClosedLoopTests(unittest.TestCase):
    @staticmethod
    def _orchestrator(planner, qwen, adapter):
        return UniversalAgentOrchestrator(
            deepseek_planner=planner,
            qwen_observer=qwen,
            adapter_factory=lambda _device_id: adapter,
            trusted_observation_factory=_trusted_factory,
        )

    @staticmethod
    def _unknown_app_graph() -> DynamicTaskGraph:
        base = _graph()
        graph = replace(
            base,
            goal=replace(
                base.goal,
                objective="在首次出现的资料工具中查看唯一条目及其下一层只读内容",
                target_apps=(
                    TargetApp(
                        app_id="unseen.reference.workspace",
                        app_name="陌生资料工具",
                    ),
                ),
            ),
            raw_user_goal="查看眼前唯一资料，再查看它的下一层只读内容",
        )
        graph.validate()
        return graph

    def test_unknown_app_multistep_requires_two_separate_confirmations(self) -> None:
        initial = self._unknown_app_graph()
        revision_two = replace(initial, revision=2)
        completed = _completed_graph(revision_two)
        planner = SequenceDeepSeekPlanner(initial, revision_two, completed)
        qwen = SequenceQwenObserver("action", "action")
        adapter = SequenceExecutingAdapter(
            _scene(app_id="unseen.reference.workspace"),
            (_scene(fingerprint="frame-b", label="进入公开内容", app_id="unseen.reference.workspace"), "matched", ()),
            (_scene(fingerprint="frame-c", label="公开内容已显示", app_id="unseen.reference.workspace"), "matched", ()),
        )
        with tempfile.TemporaryDirectory() as temp:
            orchestrator = self._orchestrator(planner, qwen, adapter)
            session = orchestrator.start(
                session_id="session-unseen-multistep",
                raw_goal=initial.raw_user_goal,
                device_id="device-1",
                run_dir=Path(temp),
            )

            first = orchestrator.confirm_one(session, _confirmation(session))
            self.assertEqual(
                "awaiting_confirmation",
                session.status,
                session.failed_reason,
            )
            self.assertEqual(1, session.physical_actions)
            second = orchestrator.confirm_one(session, _confirmation(session))

        self.assertEqual(1, first.physical_actions)
        self.assertEqual(1, second.physical_actions)
        self.assertEqual(2, adapter.execute_calls)
        self.assertEqual(2, session.physical_actions)
        self.assertEqual(2, len(session.history))
        self.assertEqual("succeeded", session.status)
        self.assertEqual(3, session.task_graph.revision)
        self.assertEqual("unseen.reference.workspace", session.goal_draft.app_id)

    def test_new_active_subgoal_gets_goal_conditioned_reobservation(self) -> None:
        base = self._unknown_app_graph()
        first = replace(
            base.subgoals[0],
            subgoal_id="return_home",
            objective="返回手机桌面",
            completion_conditions=("手机桌面可见",),
        )
        second = Subgoal(
            subgoal_id="read_title",
            objective="读取当前页面主标题",
            status="pending",
            depends_on=(first.subgoal_id,),
            constraints=("仅读取",),
            completion_conditions=("页面主标题已读取",),
            completion_evidence=(),
            risk_action_ids=(),
            external_impact="read_only",
        )
        initial = replace(
            base,
            subgoals=(first, second),
            active_subgoal_id=first.subgoal_id,
        )
        initial.validate()
        revised = replace(
            initial,
            revision=2,
            subgoals=(
                replace(
                    first,
                    status="completed",
                    completion_evidence=("手机桌面可见",),
                ),
                replace(second, status="active"),
            ),
            active_subgoal_id=second.subgoal_id,
        )
        revised.validate()
        completed = _completed_graph(revised)
        completed = replace(
            completed,
            subgoals=(
                completed.subgoals[0],
                replace(
                    completed.subgoals[1],
                    completion_evidence=(
                        "当前可信画面读取结果：element_id=candidate-1, "
                        "role=text, meaning=page_title, label=公开页面主标题。",
                    ),
                ),
            ),
        )
        completed.validate()

        after_scene = replace(
            _scene(
                fingerprint="frame-b",
                app_id="unseen.reference.workspace",
            ),
            elements=(),
            summary="动作后页面稳定，但旧目标观察没有标题候选",
        )
        title_scene = _scene(
            fingerprint="frame-b",
            app_id="unseen.reference.workspace",
            meaning="page_title",
            label="公开页面主标题",
            role="text",
            states={"goal_relevant": True, "fully_visible": True},
        )

        class GoalConditionedAdapter(SequenceExecutingAdapter):
            def __init__(self):
                super().__init__(
                    _scene(app_id="unseen.reference.workspace"),
                    (after_scene, "matched", ()),
                )
                self.captured_subgoals = []

            def capture_scene(self, goal, *, evidence_dir, prefix):
                focus = goal.entities["active_subgoal_visual_context"]
                self.captured_subgoals.append(focus["subgoal_id"])
                if focus["subgoal_id"] == "read_title":
                    self.scene = title_scene
                return super().capture_scene(
                    goal,
                    evidence_dir=evidence_dir,
                    prefix=prefix,
                )

        planner = SequenceDeepSeekPlanner(initial, revised, completed)
        qwen = SequenceQwenObserver("action")
        adapter = GoalConditionedAdapter()
        with tempfile.TemporaryDirectory() as temp:
            orchestrator = self._orchestrator(planner, qwen, adapter)
            session = orchestrator.start(
                session_id="session-goal-conditioned-reobservation",
                raw_goal=initial.raw_user_goal,
                device_id="device-1",
                run_dir=Path(temp),
            )

            orchestrator.confirm_one(session, _confirmation(session))
            self.assertEqual("needs_reobservation", session.status)
            self.assertEqual(1, len(qwen.calls))
            self.assertEqual((), session.trusted_observation.scene.elements)
            self.assertEqual(
                "advanced_to_goal_conditioned_reobservation",
                session.last_post_action_transition["disposition"],
            )

            orchestrator.refresh_decision(session)

        self.assertEqual(["return_home", "read_title"], adapter.captured_subgoals)
        self.assertEqual(1, adapter.execute_calls)
        self.assertEqual(1, session.physical_actions)
        self.assertEqual(1, len(qwen.calls))
        self.assertEqual(
            "公开页面主标题",
            session.trusted_observation.scene.elements[0].label,
        )
        self.assertEqual("succeeded", session.status)

    def test_unchanged_screen_is_mismatch_evidence_and_replans_once(self) -> None:
        initial = self._unknown_app_graph()
        revised = replace(initial, revision=2)
        planner = SequenceDeepSeekPlanner(initial, revised)
        qwen = SequenceQwenObserver("action", "blocked")
        unchanged = _scene(app_id="unseen.reference.workspace")
        adapter = SequenceExecutingAdapter(
            unchanged,
            (
                unchanged,
                "mismatched",
                ("动作后画面未变化，预期语义结果未出现。",),
            ),
        )
        with tempfile.TemporaryDirectory() as temp:
            orchestrator = self._orchestrator(planner, qwen, adapter)
            session = orchestrator.start(
                session_id="session-no-effect",
                raw_goal=initial.raw_user_goal,
                device_id="device-1",
                run_dir=Path(temp),
            )
            result = orchestrator.confirm_one(session, _confirmation(session))

        self.assertEqual("mismatched", result.action_outcome)
        self.assertEqual(result.before_scene.fingerprint, result.after_scene.fingerprint)
        self.assertEqual(1, adapter.execute_calls)
        self.assertEqual(1, session.physical_actions)
        self.assertEqual("action_result_mismatch", planner.replan_calls[0][2])
        self.assertEqual(2, len(qwen.calls))
        self.assertEqual("blocked", session.status)

    def test_candidate_disappears_after_action_and_old_candidate_is_not_reused(self) -> None:
        initial = self._unknown_app_graph()
        revised = replace(initial, revision=2)
        planner = SequenceDeepSeekPlanner(initial, revised)
        qwen = SequenceQwenObserver("action", "blocked")
        disappeared = replace(
            _scene(fingerprint="frame-no-candidate", app_id="unseen.reference.workspace"),
            summary="动作后目标候选已经不在当前画面",
            elements=(),
        )
        adapter = SequenceExecutingAdapter(
            _scene(app_id="unseen.reference.workspace"),
            (disappeared, "matched", ()),
        )
        with tempfile.TemporaryDirectory() as temp:
            orchestrator = self._orchestrator(planner, qwen, adapter)
            session = orchestrator.start(
                session_id="session-candidate-disappeared",
                raw_goal=initial.raw_user_goal,
                device_id="device-1",
                run_dir=Path(temp),
            )
            orchestrator.confirm_one(session, _confirmation(session))

        second_observation = qwen.calls[1]["trusted_observation"]
        self.assertEqual((), second_observation.scene.elements)
        self.assertEqual("frame-no-candidate", second_observation.fingerprint)
        self.assertEqual(1, adapter.execute_calls)
        self.assertEqual(1, session.physical_actions)
        self.assertEqual("blocked", session.status)
        self.assertIsNone(session.snapshot()["confirmation_scope"])

    def test_refresh_finished_is_reviewed_by_deepseek_without_action(self) -> None:
        initial = self._unknown_app_graph()
        completed = _completed_graph(initial)
        planner = SequenceDeepSeekPlanner(initial, completed)
        qwen = SequenceQwenObserver("action", "finished")
        adapter = SequenceCaptureAdapter(
            _scene(app_id="unseen.reference.workspace"),
            _scene(fingerprint="frame-complete", label="公开内容已显示", app_id="unseen.reference.workspace"),
        )
        with tempfile.TemporaryDirectory() as temp:
            orchestrator = self._orchestrator(planner, qwen, adapter)
            session = orchestrator.start(
                session_id="session-refresh-completed",
                raw_goal=initial.raw_user_goal,
                device_id="device-1",
                run_dir=Path(temp),
            )
            decision = orchestrator.refresh_decision(session)

        self.assertEqual("finished", decision.proposal.status)
        self.assertEqual("observation_changed", planner.replan_calls[0][2])
        self.assertEqual("succeeded", session.status)
        self.assertEqual(2, session.task_graph.revision)
        self.assertEqual(0, adapter.execute_calls)
        self.assertEqual(0, session.physical_actions)

    def test_refresh_changes_fingerprint_and_invalidates_old_action_scope(self) -> None:
        initial = self._unknown_app_graph()
        planner = SequenceDeepSeekPlanner(initial, replace(initial, revision=2))
        qwen = SequenceQwenObserver("action", "action")
        adapter = SequenceCaptureAdapter(
            _scene(app_id="unseen.reference.workspace"),
            _scene(fingerprint="frame-new-candidate", label="新的唯一入口", app_id="unseen.reference.workspace"),
        )
        with tempfile.TemporaryDirectory() as temp:
            orchestrator = self._orchestrator(planner, qwen, adapter)
            session = orchestrator.start(
                session_id="session-stale-fingerprint",
                raw_goal=initial.raw_user_goal,
                device_id="device-1",
                run_dir=Path(temp),
            )
            stale_scope = _confirmation(session)
            stale_authority = session.confirmation_authority
            orchestrator.refresh_decision(session)
            with self.assertRaisesRegex(
                UniversalAgentOrchestratorError,
                "不一致",
            ):
                orchestrator.confirm_one(session, stale_scope)

        self.assertNotEqual(
            stale_scope["fingerprint"],
            session.trusted_observation.fingerprint,
        )
        self.assertTrue(stale_authority.consumed)
        self.assertEqual("fresh_observation_requested", stale_authority.invalid_reason)
        self.assertEqual(initial.revision + 1, session.task_graph.revision)
        self.assertEqual(
            session.task_graph.revision,
            qwen.calls[1]["task_context"]["revision"],
        )
        self.assertEqual(0, adapter.execute_calls)
        self.assertEqual(0, session.physical_actions)

    def test_navigation_presence_binds_page_to_scene_and_input_to_control(self) -> None:
        initial = UniversalAgentStartTests._read_only_locate_graph()
        initial = replace(
            initial,
            subgoals=(
                replace(
                    initial.subgoals[0],
                    objective="当前本地页面的唯一输入框可见",
                    completion_conditions=(
                        "当前本地页面的唯一输入框在画面中可见",
                    ),
                    external_impact="navigation_only",
                ),
                initial.subgoals[1],
            ),
        )
        initial.validate()
        revised = UniversalAgentStartTests._advance_locate_graph(initial)
        planner = FakeDeepSeekPlanner(initial, replan_result=revised)
        input_element = UIElement(
            element_id="input-1",
            role="input",
            meaning="target_text_input",
            label="",
            bounds=(0.13, 0.50, 0.87, 0.60),
            confidence=1.0,
            states={
                "goal_relevant": True,
                "fully_visible": True,
                "value": "",
            },
            evidence=("位于目标文字下方的空矩形输入框",),
        )
        instruction = UIElement(
            element_id="instruction-1",
            role="text",
            meaning="target_instruction",
            label="目标文字：agent",
            bounds=(0.13, 0.45, 0.45, 0.49),
            confidence=1.0,
            states={"goal_relevant": True, "fully_visible": True},
            evidence=("输入框上方明确指示目标文字为agent",),
        )
        scene = UIScene(
            app_id="unknown",
            screen_id="通用动作真机验收页",
            summary="页面包含目标文字agent的输入任务及一个空输入框。",
            elements=(input_element, instruction),
            stable=True,
            confidence=1.0,
            fingerprint="live-shape-before-focus",
        )

        with tempfile.TemporaryDirectory() as temp:
            session = self._orchestrator(
                planner,
                FakeQwenObserver(),
                FakeAdapter(scene),
            ).start(
                session_id="session-live-shape-presence",
                raw_goal=initial.raw_user_goal,
                device_id="device-1",
                run_dir=Path(temp),
            )

        self.assertEqual("needs_reobservation", session.status)
        self.assertEqual(2, session.task_graph.revision)
        self.assertEqual("replace-input", session.task_graph.active_subgoal_id)
        self.assertEqual(1, len(planner.replan_calls))
        visible_evidence = planner.replan_calls[0][1].visible_evidence
        self.assertTrue(any("element_id=input-1" in item for item in visible_evidence))
        self.assertFalse(any("element_id=instruction-1" in item for item in visible_evidence))
        self.assertEqual(0, session.physical_actions)

    def test_navigation_presence_rejects_instruction_mention_without_safe_input(self) -> None:
        initial = UniversalAgentStartTests._read_only_locate_graph()
        initial = replace(
            initial,
            subgoals=(
                replace(
                    initial.subgoals[0],
                    objective="当前页面的唯一输入框可见",
                    completion_conditions=("当前页面的唯一输入框可见",),
                    external_impact="navigation_only",
                ),
                initial.subgoals[1],
            ),
        )
        initial.validate()
        instruction = UIElement(
            element_id="instruction-only",
            role="text",
            meaning="target_instruction",
            label="请在输入框中填写内容",
            bounds=(0.13, 0.45, 0.55, 0.49),
            confidence=1.0,
            states={"goal_relevant": True, "fully_visible": True},
            evidence=("说明文字提到输入框",),
        )
        scene = UIScene(
            app_id="unknown",
            screen_id="local-page",
            summary="当前页面显示输入任务说明。",
            elements=(instruction,),
            stable=True,
            confidence=1.0,
            fingerprint="instruction-without-input",
        )
        planner = FakeDeepSeekPlanner(
            initial,
            replan_result=UniversalAgentStartTests._advance_locate_graph(initial),
        )

        with tempfile.TemporaryDirectory() as temp:
            session = self._orchestrator(
                planner,
                FakeQwenObserver(),
                FakeAdapter(scene),
            ).start(
                session_id="session-no-real-input",
                raw_goal=initial.raw_user_goal,
                device_id="device-1",
                run_dir=Path(temp),
            )

        self.assertEqual("blocked", session.status)
        self.assertEqual([], planner.replan_calls)
        self.assertEqual(1, session.task_graph.revision)
        self.assertEqual(0, session.physical_actions)

    def test_live_shape_input_goal_succeeds_with_two_actions_in_one_session(self) -> None:
        initial = UniversalAgentStartTests._read_only_locate_graph()
        initial = replace(
            initial,
            goal=replace(
                initial.goal,
                objective="让当前唯一输入框中的内容最终为 agent",
                entities={
                    "input_text": "agent",
                    "target_surface": "current_surface",
                },
            ),
            completion_conditions=(
                replace(
                    initial.completion_conditions[0],
                    description="当前唯一输入框显示 agent",
                    evidence_required=("输入框结构化值为 agent",),
                ),
            ),
            subgoals=(
                replace(
                    initial.subgoals[0],
                    subgoal_id="ensure-input-visible",
                    objective="当前页面的唯一输入框可见",
                    completion_conditions=("当前页面的唯一输入框可见",),
                    external_impact="navigation_only",
                ),
                replace(
                    initial.subgoals[1],
                    subgoal_id="set-input-text",
                    objective="当前唯一输入框中的内容为 agent",
                    depends_on=("ensure-input-visible",),
                    completion_conditions=("输入框结构化值为 agent",),
                ),
            ),
            active_subgoal_id="ensure-input-visible",
            raw_user_goal=(
                "让当前页面唯一输入框中的内容最终为 agent；"
                "不得搜索、提交、发送、保存或发布"
            ),
        )
        initial.validate()

        before = UIScene(
            app_id="unknown",
            screen_id="local-input-page",
            summary="当前页面显示一个空的唯一输入框。",
            elements=(
                UIElement(
                    element_id="candidate-1",
                    role="input",
                    meaning="target_text_input",
                    label="",
                    bounds=(0.13, 0.50, 0.87, 0.60),
                    confidence=1.0,
                    states={
                        "goal_relevant": True,
                        "fully_visible": True,
                        "value": "",
                    },
                    evidence=("页面中央唯一空输入框",),
                ),
                UIElement(
                    element_id="instruction-1",
                    role="text",
                    meaning="target_instruction",
                    label="目标文字：agent",
                    bounds=(0.13, 0.45, 0.45, 0.49),
                    confidence=1.0,
                    states={"goal_relevant": True, "fully_visible": True},
                    evidence=("输入任务说明",),
                ),
            ),
            stable=True,
            confidence=1.0,
            fingerprint="input-empty-unfocused",
        )
        focused_input = replace(
            before.elements[0],
            states={
                "goal_relevant": True,
                "fully_visible": True,
                "focused": True,
                "value": "",
                "keyboard_layout": "qwerty",
                "keyboard_input_mode": "direct_latin",
            },
        )
        focused = replace(
            before,
            summary="唯一输入框已聚焦，英文直输键盘可见。",
            elements=(focused_input, before.elements[1]),
            fingerprint="input-empty-focused",
        )
        filled_input = replace(
            focused_input,
            states={**focused_input.states, "value": "agent"},
        )
        filled = replace(
            focused,
            summary="唯一输入框结构化值为 agent。",
            elements=(filled_input, before.elements[1]),
            fingerprint="input-filled-agent",
        )

        presence_advanced = replace(
            initial,
            revision=2,
            subgoals=(
                replace(
                    initial.subgoals[0],
                    status="completed",
                    completion_evidence=(before.summary,),
                ),
                replace(initial.subgoals[1], status="active"),
            ),
            active_subgoal_id="set-input-text",
        )
        presence_advanced.validate()
        focus_replanned = replace(presence_advanced, revision=3)
        focus_replanned.validate()
        completed = replace(
            focus_replanned,
            revision=4,
            status="completed",
            completion_conditions=(
                replace(
                    focus_replanned.completion_conditions[0],
                    satisfied=True,
                    evidence=(filled.summary,),
                ),
            ),
            subgoals=(
                focus_replanned.subgoals[0],
                replace(
                    focus_replanned.subgoals[1],
                    status="completed",
                    completion_evidence=(filled.summary,),
                ),
            ),
            active_subgoal_id=None,
        )
        completed.validate()

        class FocusThenInputQwen:
            def __init__(self) -> None:
                self.action_kinds = ["tap_semantic", "input_verified_text"]
                self.calls = []

            def decide(self, **kwargs):
                if not self.action_kinds:
                    raise AssertionError("unexpected Qwen decision call")
                self.calls.append(kwargs)
                action_kind = self.action_kinds.pop(0)
                decision = FakeQwenObserver(action_kind=action_kind).decide(
                    **kwargs
                )
                if action_kind == "input_verified_text":
                    decision.proposal.action.params["text"] = "agent"
                return decision

        planner = SequenceDeepSeekPlanner(
            initial,
            presence_advanced,
            focus_replanned,
            completed,
        )
        qwen = FocusThenInputQwen()
        adapter = SequenceExecutingAdapter(
            before,
            (focused, "matched", ()),
            (filled, "matched", ()),
        )

        with tempfile.TemporaryDirectory() as temp:
            orchestrator = self._orchestrator(planner, qwen, adapter)
            session = orchestrator.start(
                session_id="session-live-shape-two-actions",
                raw_goal=initial.raw_user_goal,
                device_id="device-1",
                run_dir=Path(temp),
            )
            self.assertEqual("needs_reobservation", session.status)
            orchestrator.refresh_decision(session)
            self.assertEqual(
                "awaiting_confirmation",
                session.status,
                session.failed_reason,
            )
            self.assertEqual(0, session.physical_actions)
            orchestrator.confirm_one(session, _confirmation(session))
            self.assertEqual("awaiting_confirmation", session.status)
            self.assertEqual(1, session.physical_actions)
            orchestrator.confirm_one(session, _confirmation(session))

        self.assertEqual("succeeded", session.status)
        self.assertEqual(2, session.physical_actions)
        self.assertEqual(2, adapter.execute_calls)
        self.assertEqual(2, len(session.history))
        self.assertEqual(4, session.task_graph.revision)
        self.assertIsNone(session.task_graph.active_subgoal_id)
        self.assertEqual(
            ["subgoal_completed", "action_result_matched", "action_result_matched"],
            [call[2] for call in planner.replan_calls],
        )
        receipts = [
            call[1].verified_action_transition for call in planner.replan_calls[1:]
        ]
        self.assertTrue(all(receipt is not None for receipt in receipts))
        self.assertTrue(all(receipt.physical_actions == 1 for receipt in receipts))
        self.assertNotEqual(receipts[0].receipt_id, receipts[1].receipt_id)
        self.assertNotEqual(
            receipts[0].after_observation_id,
            receipts[1].after_observation_id,
        )
        self.assertEqual(
            ["input-empty-focused", "input-filled-agent"],
            [item["after_fingerprint"] for item in session.history],
        )

    def test_reload_visible_state_is_not_zero_action_presence_completion(self) -> None:
        subgoal = SimpleNamespace(
            objective="当前本地页面完成重新载入",
            completion_conditions=("页面重新载入后的可见状态",),
        )

        self.assertFalse(
            UniversalAgentOrchestrator._is_presence_only_read_only_subgoal(
                subgoal
            )
        )

    def test_unique_focused_input_mints_one_local_zero_action_state_fact(self) -> None:
        focused_scene = _scene(
            role="input",
            meaning="application_text_input",
            label="",
            states={"focused": True, "value": ""},
        )
        observation = FakeTrustedObservation(
            device_id="device-1",
            scene=focused_scene,
        )
        subgoal = SimpleNamespace(
            completion_conditions=("输入框处于聚焦状态",),
        )

        fact = UniversalAgentOrchestrator._zero_action_visible_state_fact(
            subgoal,
            observation,
        )

        self.assertEqual(
            "当前可信画面的局部控件状态："
            "element_id=candidate-1, role=input, focused=true。",
            fact,
        )

    def test_focus_state_fact_fails_closed_for_untrusted_or_unrelated_shapes(self) -> None:
        base = _scene(
            role="input",
            meaning="application_text_input",
            label="",
            states={"focused": True, "value": ""},
        )
        focus_subgoal = SimpleNamespace(
            completion_conditions=("输入框处于聚焦状态",),
        )
        value_subgoal = SimpleNamespace(
            completion_conditions=("输入框内容为 codex",),
        )
        duplicate = replace(
            base,
            elements=(
                base.elements[0],
                replace(base.elements[0], element_id="candidate-2"),
            ),
        )
        cases = {
            "not_focused": replace(
                base,
                elements=(
                    replace(
                        base.elements[0],
                        states={**base.elements[0].states, "focused": False},
                    ),
                ),
            ),
            "wrong_role": replace(
                base,
                elements=(replace(base.elements[0], role="button"),),
            ),
            "low_confidence": replace(
                base,
                elements=(replace(base.elements[0], confidence=0.70),),
            ),
            "not_goal_relevant": replace(
                base,
                elements=(
                    replace(
                        base.elements[0],
                        states={
                            **base.elements[0].states,
                            "goal_relevant": False,
                        },
                    ),
                ),
            ),
            "duplicate": duplicate,
        }
        for name, scene in cases.items():
            with self.subTest(case=name):
                observation = FakeTrustedObservation(
                    device_id="device-1",
                    scene=scene,
                )
                self.assertIsNone(
                    UniversalAgentOrchestrator._zero_action_visible_state_fact(
                        focus_subgoal,
                        observation,
                    )
                )

        conflicted = FakeTrustedObservation(
            device_id="device-1",
            scene=base,
        )
        conflicted.candidate_conflicts = (
            {"kind": "ambiguous", "element_ids": ["candidate-1"]},
        )
        self.assertIsNone(
            UniversalAgentOrchestrator._zero_action_visible_state_fact(
                focus_subgoal,
                conflicted,
            )
        )
        self.assertIsNone(
            UniversalAgentOrchestrator._zero_action_visible_state_fact(
                value_subgoal,
                FakeTrustedObservation(device_id="device-1", scene=base),
            )
        )

    def test_visible_page_and_focused_input_advance_as_one_safe_prefix(self) -> None:
        base = _graph()
        open_page = replace(
            base.subgoals[0],
            subgoal_id="local-tool-page-visible",
            objective="本地工具首页可见",
            completion_conditions=("本地工具应用界面可见",),
            external_impact="navigation_only",
        )
        focus = Subgoal(
            subgoal_id="focus-input",
            objective="让目标输入框获得焦点",
            status="pending",
            depends_on=(open_page.subgoal_id,),
            constraints=(),
            completion_conditions=("输入框处于聚焦状态",),
            completion_evidence=(),
            risk_action_ids=(),
            external_impact="navigation_only",
        )
        type_text = Subgoal(
            subgoal_id="type-text",
            objective="输入框内容为 codex",
            status="pending",
            depends_on=(focus.subgoal_id,),
            constraints=(),
            completion_conditions=("输入框内容逐字为 codex",),
            completion_evidence=(),
            risk_action_ids=(),
            external_impact="navigation_only",
        )
        initial = replace(
            base,
            goal=replace(
                base.goal,
                objective="确认本地工具输入框内容为 codex",
                target_apps=(
                    TargetApp(app_id="local_tool", app_name="本地工具"),
                ),
                entities={"target_ui_label": "输入框", "input_text": "codex"},
            ),
            subgoals=(open_page, focus, type_text),
            active_subgoal_id=open_page.subgoal_id,
            raw_user_goal="确认本地工具输入框内容为 codex",
        )
        initial.validate()
        scene = replace(
            _scene(
                role="input",
                meaning="application_text_input",
                label="",
                states={"focused": True, "value": ""},
                app_id="local_tool",
            ),
            screen_id="local_tool_home",
            summary="本地工具应用界面可见，唯一输入框已聚焦。",
        )
        scene = replace(
            scene,
            elements=(
                scene.elements[0],
                UIElement(
                    element_id="page-title",
                    role="text",
                    meaning="page_title",
                    label="本地工具",
                    bounds=(0.30, 0.04, 0.70, 0.10),
                    confidence=1.0,
                    states={"goal_relevant": False, "fully_visible": True},
                    evidence=("页面顶部唯一主标题",),
                ),
            ),
        )
        focus_fact = (
            "当前可信画面的局部控件状态："
            "element_id=candidate-1, role=input, focused=true。"
        )
        revised = replace(
            initial,
            revision=2,
            subgoals=(
                replace(
                    open_page,
                    status="completed",
                    completion_evidence=(scene.summary,),
                ),
                replace(
                    focus,
                    status="completed",
                    completion_evidence=(focus_fact,),
                ),
                replace(type_text, status="active"),
            ),
            active_subgoal_id=type_text.subgoal_id,
        )
        revised.validate()
        planner = FakeDeepSeekPlanner(initial, replan_result=revised)
        qwen = FakeQwenObserver()
        adapter = FakeAdapter(scene)

        with tempfile.TemporaryDirectory() as temp:
            session = self._orchestrator(planner, qwen, adapter).start(
                session_id="session-visible-focus-prefix",
                raw_goal=initial.raw_user_goal,
                device_id="device-1",
                run_dir=Path(temp),
            )

        self.assertEqual("needs_reobservation", session.status)
        self.assertEqual(2, session.task_graph.revision)
        self.assertEqual(
            ["completed", "completed", "active"],
            [item.status for item in session.task_graph.subgoals],
        )
        self.assertEqual("type-text", session.task_graph.active_subgoal_id)
        self.assertEqual([], qwen.calls)
        self.assertEqual(0, session.physical_actions)
        self.assertEqual(0, adapter.execute_calls)

    def test_visible_prefix_narrows_model_only_focus_claim_without_failing(self) -> None:
        base = _graph()
        page = replace(
            base.subgoals[0],
            subgoal_id="chat-visible",
            objective="文件传输助手会话页面可见",
            completion_conditions=("文件传输助手会话页面可见",),
            external_impact="navigation_only",
        )
        focus = Subgoal(
            subgoal_id="focus-input",
            objective="聚焦当前唯一空白输入框",
            status="pending",
            depends_on=(page.subgoal_id,),
            constraints=(),
            completion_conditions=("输入框处于聚焦状态",),
            completion_evidence=(),
            risk_action_ids=(),
            external_impact="navigation_only",
        )
        type_text = Subgoal(
            subgoal_id="type-text",
            objective="输入框内容为 longinput",
            status="pending",
            depends_on=(focus.subgoal_id,),
            constraints=(),
            completion_conditions=("输入框内容逐字为 longinput",),
            completion_evidence=(),
            risk_action_ids=(),
            external_impact="navigation_only",
        )
        initial = replace(
            base,
            goal=replace(
                base.goal,
                objective="在文件传输助手输入 longinput 但不发送",
                target_apps=(TargetApp(app_id="wechat", app_name="微信"),),
                entities={
                    "recipient": "文件传输助手",
                    "target_ui_label": "文件传输助手",
                    "input_text": "longinput",
                },
            ),
            subgoals=(page, focus, type_text),
            active_subgoal_id=page.subgoal_id,
            raw_user_goal="在文件传输助手输入 longinput 但不发送",
        )
        initial.validate()
        scene = replace(
            _scene(
                role="input",
                meaning="message_input",
                label="",
                states={"value": ""},
                app_id="wechat",
            ),
            screen_id="文件传输助手",
            # This wording is model prose, not a structured focus fact.
            summary="文件传输助手会话页面可见，空输入框已激活。",
        )
        scene = replace(
            scene,
            elements=(
                scene.elements[0],
                UIElement(
                    element_id="page-title",
                    role="text",
                    meaning="page_title",
                    label="文件传输助手",
                    bounds=(0.30, 0.04, 0.70, 0.10),
                    confidence=1.0,
                    states={"goal_relevant": False, "fully_visible": True},
                    evidence=("页面顶部唯一会话标题",),
                ),
            ),
        )
        model_revised = replace(
            initial,
            revision=2,
            subgoals=(
                replace(
                    page,
                    status="completed",
                    completion_evidence=(scene.summary,),
                ),
                replace(
                    focus,
                    status="completed",
                    completion_evidence=(scene.summary,),
                ),
                replace(type_text, status="active"),
            ),
            active_subgoal_id=type_text.subgoal_id,
        )
        model_revised.validate()
        planner = FakeDeepSeekPlanner(initial, replan_result=model_revised)
        qwen = FakeQwenObserver()
        adapter = FakeAdapter(scene)

        with tempfile.TemporaryDirectory() as temp:
            session = self._orchestrator(planner, qwen, adapter).start(
                session_id="session-narrow-model-focus",
                raw_goal=initial.raw_user_goal,
                device_id="device-1",
                run_dir=Path(temp),
            )

        self.assertEqual("needs_reobservation", session.status)
        self.assertEqual(2, session.task_graph.revision)
        self.assertEqual(
            ["completed", "active", "pending"],
            [item.status for item in session.task_graph.subgoals],
        )
        self.assertEqual("focus-input", session.task_graph.active_subgoal_id)
        self.assertEqual((), session.task_graph.subgoals[1].completion_evidence)
        self.assertEqual([], qwen.calls)
        self.assertEqual(0, session.physical_actions)
        self.assertEqual(0, adapter.execute_calls)

    def test_visible_prefix_narrowing_never_activates_external_successor(self) -> None:
        external_base = _external_graph()
        external = replace(
            external_base.subgoals[0],
            status="pending",
            depends_on=("page-visible",),
        )
        page = Subgoal(
            subgoal_id="page-visible",
            objective="目标页面可见",
            status="active",
            depends_on=(),
            constraints=(),
            completion_conditions=("目标页面可见",),
            completion_evidence=(),
            risk_action_ids=(),
            external_impact="navigation_only",
        )
        tail = Subgoal(
            subgoal_id="result-visible",
            objective="结果页面可见",
            status="pending",
            depends_on=(external.subgoal_id,),
            constraints=(),
            completion_conditions=("结果页面可见",),
            completion_evidence=(),
            risk_action_ids=(),
            external_impact="read_only",
        )
        previous = replace(
            external_base,
            status="running",
            subgoals=(page, external, tail),
            active_subgoal_id=page.subgoal_id,
        )
        previous.validate()
        revised = replace(
            previous,
            revision=2,
            subgoals=(
                replace(
                    page,
                    status="completed",
                    completion_evidence=("目标页面可见",),
                ),
                replace(
                    external,
                    status="completed",
                    completion_evidence=("模型声称外部结果可见",),
                ),
                replace(tail, status="active"),
            ),
            active_subgoal_id=tail.subgoal_id,
        )
        revised.validate()

        narrowed = UniversalAgentOrchestrator._narrow_unproven_visible_successor(
            previous=previous,
            revised=revised,
            current_subgoal_id=page.subgoal_id,
            accepted_prefix=(page.subgoal_id,),
            unsupported_subgoal_id=external.subgoal_id,
        )

        self.assertIsNone(narrowed)

    def test_already_visible_navigation_destination_is_presence_completion(self) -> None:
        cases = (
            ("打开设置应用", "设置主界面可见"),
            ("返回手机桌面", "手机桌面可见"),
            ("进入目标详情页", "目标详情页面可见"),
        )

        for objective, completion_condition in cases:
            with self.subTest(objective=objective):
                subgoal = SimpleNamespace(
                    objective=objective,
                    completion_conditions=(completion_condition,),
                )
                self.assertTrue(
                    UniversalAgentOrchestrator._is_presence_only_read_only_subgoal(
                        subgoal
                    )
                )

    def test_absence_or_dismissal_is_not_zero_action_presence_completion(self) -> None:
        cases = (
            ("当前可见软键盘被收起且不可见", "当前画面中无软键盘"),
            ("关闭当前可见弹层", "弹层已消失"),
            ("Hide the visible keyboard", "The keyboard is not visible"),
            ("Dismiss the current dialog", "The dialog is absent"),
        )

        for objective, completion_condition in cases:
            with self.subTest(objective=objective):
                subgoal = SimpleNamespace(
                    objective=objective,
                    completion_conditions=(completion_condition,),
                )
                self.assertFalse(
                    UniversalAgentOrchestrator._is_presence_only_read_only_subgoal(
                        subgoal
                    )
                )

    def test_start_consumes_only_the_visible_safe_presence_prefix_before_qwen(self) -> None:
        base = _graph()
        initial = replace(
            base,
            goal=replace(
                base.goal,
                objective="进入验收模式选择列表第二项后查看目标页标题",
                entities={"target_ui_label": "验收模式选择"},
            ),
            subgoals=(
                replace(
                    base.subgoals[0],
                    subgoal_id="list-visible",
                    objective="验收模式选择列表在当前页面可见",
                    completion_conditions=("验收模式选择列表可见",),
                ),
                Subgoal(
                    subgoal_id="second-visible",
                    objective="列表中从上往下第二项可见",
                    status="pending",
                    depends_on=("list-visible",),
                    constraints=(),
                    completion_conditions=("从上往下第二项可见",),
                    completion_evidence=(),
                    risk_action_ids=(),
                    external_impact="navigation_only",
                ),
                Subgoal(
                    subgoal_id="target-page-visible",
                    objective="第二项对应的目标页面可见",
                    status="pending",
                    depends_on=("second-visible",),
                    constraints=(),
                    completion_conditions=("目标页面标题可见",),
                    completion_evidence=(),
                    risk_action_ids=(),
                    external_impact="navigation_only",
                ),
            ),
            active_subgoal_id="list-visible",
        )
        initial.validate()
        list_advanced = replace(
            initial,
            revision=2,
            subgoals=(
                replace(
                    initial.subgoals[0],
                    status="completed",
                    completion_evidence=("验收模式选择列表可见",),
                ),
                replace(initial.subgoals[1], status="active"),
                initial.subgoals[2],
            ),
            active_subgoal_id="second-visible",
        )
        list_advanced.validate()
        second_advanced = replace(
            list_advanced,
            revision=3,
            subgoals=(
                list_advanced.subgoals[0],
                replace(
                    list_advanced.subgoals[1],
                    status="completed",
                    completion_evidence=("从上往下第二项可见",),
                ),
                replace(list_advanced.subgoals[2], status="active"),
            ),
            active_subgoal_id="target-page-visible",
        )
        second_advanced.validate()
        planner = SequenceDeepSeekPlanner(initial, list_advanced, second_advanced)
        qwen = FakeQwenObserver(status="blocked")
        base_scene = _scene(
            meaning="acceptance_mode_option",
            label="语义点击",
            role="list_item",
            states={"goal_relevant": True, "fully_visible": True},
        )
        target = replace(
            base_scene.elements[0],
            evidence=(
                "位于第一项正下方，从上往下第二个蓝色圆角矩形条目",
            ),
        )
        scene = replace(
            base_scene,
            summary="验收模式选择列表页，包含多个选项按钮",
            elements=(target,),
        )
        adapter = FakeAdapter(scene)

        with tempfile.TemporaryDirectory() as temp:
            session = self._orchestrator(planner, qwen, adapter).start(
                session_id="session-visible-prefix",
                raw_goal=initial.raw_user_goal,
                device_id="device-1",
                run_dir=Path(temp),
            )

        self.assertEqual(3, session.task_graph.revision)
        self.assertEqual("target-page-visible", session.task_graph.active_subgoal_id)
        self.assertEqual(2, len(planner.replan_calls))
        self.assertEqual(0, len(qwen.calls))
        self.assertEqual("needs_reobservation", session.status)
        self.assertEqual(0, adapter.execute_calls)
        self.assertEqual(0, session.physical_actions)

    def test_start_advances_visible_navigation_checkpoint_before_qwen(self) -> None:
        base = _graph()
        initial = replace(
            base,
            goal=replace(
                base.goal,
                objective="进入验收模式选择列表第二项后返回",
            ),
            subgoals=(
                replace(
                    base.subgoals[0],
                    subgoal_id="list-visible",
                    objective="验收模式选择列表可见",
                    completion_conditions=("验收模式选择列表可见",),
                ),
                Subgoal(
                    subgoal_id="second-item-visible",
                    objective="验收模式选择列表第二项可见",
                    status="pending",
                    depends_on=("list-visible",),
                    constraints=("仅浏览本地只读页面",),
                    completion_conditions=("列表第二项可见",),
                    completion_evidence=(),
                    risk_action_ids=(),
                    external_impact="navigation_only",
                ),
                Subgoal(
                    subgoal_id="second-page-visible",
                    objective="列表第二项对应页面可见",
                    status="pending",
                    depends_on=("second-item-visible",),
                    constraints=("仅浏览本地只读页面",),
                    completion_conditions=("第二项页面可见",),
                    completion_evidence=(),
                    risk_action_ids=(),
                    external_impact="navigation_only",
                ),
            ),
            active_subgoal_id="list-visible",
            raw_user_goal="打开本地只读列表第二项后返回",
        )
        initial.validate()
        revised = replace(
            initial,
            revision=initial.revision + 1,
            subgoals=(
                replace(
                    initial.subgoals[0],
                    status="completed",
                    completion_evidence=("验收模式选择列表可见",),
                ),
                replace(
                    initial.subgoals[1],
                    status="completed",
                    completion_evidence=("列表第二项可见",),
                ),
                replace(initial.subgoals[2], status="active"),
            ),
            active_subgoal_id="second-page-visible",
        )
        revised.validate()
        scene = _scene()
        title = replace(
            scene.elements[0],
            element_id="title",
            role="text",
            meaning="acceptance_mode_list_title",
            label="验收模式选择",
            bounds=(0.1, 0.08, 0.5, 0.14),
            states={"goal_relevant": True, "fully_visible": True},
            evidence=("页面标题显示验收模式选择",),
        )
        second_item = replace(
            scene.elements[0],
            element_id="second-item",
            role="list_item",
            meaning="acceptance_mode_entry",
            label="语义点击",
            bounds=(0.12, 0.34, 0.88, 0.44),
            states={"goal_relevant": True, "fully_visible": True},
            evidence=("验收模式选择列表第二项完整可见",),
        )
        scene = replace(
            scene,
            summary="当前显示验收模式选择列表，第二项完整可见。",
            elements=(title, second_item),
        )
        planner = FakeDeepSeekPlanner(initial, replan_result=revised)
        qwen = FakeQwenObserver("blocked")
        adapter = FakeAdapter(scene)

        with tempfile.TemporaryDirectory() as temp:
            session = self._orchestrator(planner, qwen, adapter).start(
                session_id="session-visible-navigation",
                raw_goal=initial.raw_user_goal,
                device_id="device-1",
                run_dir=Path(temp),
            )

        self.assertEqual(2, session.task_graph.revision)
        self.assertEqual("second-page-visible", session.task_graph.active_subgoal_id)
        self.assertEqual(["subgoal_completed"], [call[2] for call in planner.replan_calls])
        self.assertEqual(0, len(qwen.calls))
        self.assertEqual("needs_reobservation", session.status)
        self.assertEqual(1, adapter.capture_calls)
        self.assertEqual(0, adapter.execute_calls)
        self.assertEqual(0, session.physical_actions)

    def test_refresh_drift_replan_failure_blocks_before_qwen(self) -> None:
        initial = self._unknown_app_graph()
        planner = FakeDeepSeekPlanner(
            initial,
            replan_error=RuntimeError("deepseek unavailable"),
        )
        qwen = SequenceQwenObserver("action")
        adapter = SequenceCaptureAdapter(
            _scene(),
            _scene(fingerprint="frame-drifted", label="页面已变化"),
        )
        with tempfile.TemporaryDirectory() as temp:
            orchestrator = self._orchestrator(planner, qwen, adapter)
            session = orchestrator.start(
                session_id="session-refresh-replan-unavailable",
                raw_goal=initial.raw_user_goal,
                device_id="device-1",
                run_dir=Path(temp),
            )
            stale_authority = session.confirmation_authority

            decision = orchestrator.refresh_decision(session)

        self.assertEqual("blocked", decision.proposal.status)
        self.assertEqual("blocked", session.status)
        self.assertIn("deepseek unavailable", session.failed_reason)
        self.assertTrue(stale_authority.consumed)
        self.assertEqual("fresh_observation_requested", stale_authority.invalid_reason)
        self.assertEqual(1, len(planner.replan_calls))
        self.assertEqual("observation_changed", planner.replan_calls[0][2])
        self.assertEqual(1, len(qwen.calls))
        self.assertEqual(0, adapter.execute_calls)
        self.assertEqual(0, session.physical_actions)

    def test_refresh_insufficient_observation_blocks_before_qwen(self) -> None:
        initial = self._unknown_app_graph()
        planner = FakeDeepSeekPlanner(initial)
        qwen = SequenceQwenObserver("action")
        adapter = SequenceCaptureAdapter(_scene(), _scene(fingerprint="frame-drifted"))
        observation_calls = 0

        def insufficient_factory(**kwargs):
            nonlocal observation_calls
            observation_calls += 1
            if observation_calls == 2:
                raise RuntimeError("四帧不稳定")
            return _trusted_factory(**kwargs)

        with tempfile.TemporaryDirectory() as temp:
            orchestrator = UniversalAgentOrchestrator(
                deepseek_planner=planner,
                qwen_observer=qwen,
                adapter_factory=lambda _device_id: adapter,
                trusted_observation_factory=insufficient_factory,
            )
            session = orchestrator.start(
                session_id="session-refresh-insufficient-evidence",
                raw_goal=initial.raw_user_goal,
                device_id="device-1",
                run_dir=Path(temp),
            )
            stale_authority = session.confirmation_authority

            decision = orchestrator.refresh_decision(session)

        self.assertEqual("blocked", decision.proposal.status)
        self.assertEqual("blocked", session.status)
        self.assertIn("四帧不稳定", session.failed_reason)
        self.assertTrue(stale_authority.consumed)
        self.assertEqual([], planner.replan_calls)
        self.assertEqual(1, len(qwen.calls))
        self.assertEqual(0, adapter.execute_calls)
        self.assertEqual(0, session.physical_actions)


class UniversalAgentConfirmTests(unittest.TestCase):
    @staticmethod
    def _input_graph() -> DynamicTaskGraph:
        base = _graph()
        graph = replace(
            base,
            goal=replace(
                base.goal,
                objective="当前输入框逐字显示指定文本且尚未提交",
                entities={"input_text": "live21"},
            ),
            constraints=("不得提交当前文字",),
            completion_conditions=(
                replace(
                    base.completion_conditions[0],
                    description="当前输入框逐字显示 live21 且尚未提交",
                    evidence_required=("当前输入框逐字显示 live21",),
                ),
            ),
            subgoals=(
                replace(
                    base.subgoals[0],
                    objective="当前输入框逐字显示 live21 且尚未提交",
                    constraints=("不得提交当前文字",),
                    completion_conditions=("当前输入框逐字显示 live21",),
                ),
            ),
            raw_user_goal="在当前输入框输入 live21，但不要提交",
        )
        graph.validate()
        return graph

    @staticmethod
    def _input_scene(
        value: str,
        *,
        fingerprint: str,
        auxiliary: UIElement | None = None,
    ) -> UIScene:
        elements = [
            UIElement(
                element_id="input-1",
                role="input",
                meaning="application_text_input",
                label="",
                bounds=(0.08, 0.58, 0.82, 0.66),
                confidence=0.98,
                states={
                    "goal_relevant": True,
                    "fully_visible": True,
                    "focused": True,
                    "visible": True,
                    "value": value,
                    "keyboard_layout": "qwerty",
                    "keyboard_input_mode": "direct_latin",
                    "keyboard_case_mode": "lower",
                },
                evidence=("唯一聚焦输入框",),
            )
        ]
        if auxiliary is not None:
            elements.append(auxiliary)
        scene = UIScene(
            app_id="example",
            screen_id="compose",
            summary="输入框与软键盘可见",
            elements=tuple(elements),
            stable=True,
            confidence=0.97,
            fingerprint=fingerprint,
        )
        scene.validate()
        return scene

    def test_verified_direct_input_fragment_keeps_high_level_graph(self) -> None:
        graph = self._input_graph()
        before = self._input_scene("", fingerprint="input-before")
        after = self._input_scene("live", fingerprint="input-after")
        action = SemanticAction(
            node_id="input-step",
            action="input_verified_text",
            params={"text": "live21"},
        )
        result = SimpleNamespace(
            action_outcome="matched",
            physical_actions=1,
            verification_errors=(),
            before_scene=before,
            after_scene=after,
            resolved_action=ResolvedSemanticAction(
                node_id="input-step",
                kind="input_verified_text",
                text="live21",
                input_fragment="live",
                input_method="direct_latin",
                prior_input_value="",
                expected_input_value="live",
                target_element_id="input-1",
                before_fingerprint=before.fingerprint,
                expected_effect={
                    "element_state": {
                        "meaning": "application_text_input",
                        "states": {"value": "live"},
                    }
                },
            ),
        )
        decision = SimpleNamespace(
            proposal=GenericStepProposal(status="action", action=action)
        )

        self.assertTrue(
            UniversalAgentOrchestrator._verified_input_transaction_microstep(
                graph=graph,
                previous_decision=decision,
                result=result,
                before_observation=SimpleNamespace(fingerprint=before.fingerprint),
                new_observation=SimpleNamespace(fingerprint=after.fingerprint),
            )
        )

    def test_verified_exact_input_focus_keeps_local_transaction(self) -> None:
        graph = self._input_graph()
        before_base = self._input_scene("", fingerprint="focus-before")
        before = replace(
            before_base,
            elements=(
                replace(
                    before_base.elements[0],
                    states={
                        **before_base.elements[0].states,
                        "focused": False,
                        "value": "正文",
                        "placeholder": "正文",
                        "input_field_id": "input_field_1",
                    },
                ),
            ),
        )
        after_base = self._input_scene("", fingerprint="focus-after")
        after = replace(
            after_base,
            elements=(
                replace(
                    after_base.elements[0],
                    states={
                        **after_base.elements[0].states,
                        "input_field_id": "input_field_1",
                    },
                ),
            ),
        )
        effect = {
            "element_state": {
                "meaning": "application_text_input",
                "states": {"focused": True},
            }
        }
        action = SemanticAction(
            node_id="focus-input",
            action="tap_semantic",
            params={"expected_effect": effect},
        )
        result = SimpleNamespace(
            action_outcome="matched",
            physical_actions=1,
            verification_errors=(),
            before_scene=before,
            after_scene=after,
            resolved_action=ResolvedSemanticAction(
                node_id="focus-input",
                kind="tap_semantic",
                target_element_id="input-1",
                before_fingerprint=before.fingerprint,
                expected_effect=effect,
            ),
        )

        self.assertTrue(
            UniversalAgentOrchestrator._verified_input_transaction_microstep(
                graph=graph,
                previous_decision=SimpleNamespace(
                    proposal=GenericStepProposal(status="action", action=action)
                ),
                result=result,
                before_observation=SimpleNamespace(fingerprint=before.fingerprint),
                new_observation=SimpleNamespace(fingerprint=after.fingerprint),
            )
        )

    def test_verified_enter_key_keeps_multiline_transaction(self) -> None:
        graph = self._input_graph()
        graph = replace(
            graph,
            goal=replace(
                graph.goal,
                entities={"input_text": "first\nsecond"},
            ),
        )
        enter = UIElement(
            element_id="enter-key",
            role="button",
            meaning="input_exact_enter_key",
            label="↵",
            bounds=(0.78, 0.78, 0.94, 0.9),
            confidence=0.98,
            states={
                "goal_relevant": True,
                "fully_visible": True,
                "input_enter_key": True,
                "key_action": "newline",
                "key_value": "\n",
                "prior_input_value": "first",
                "expected_input_value": "first\n",
                "input_element_id": "input-1",
            },
            evidence=("多行输入框的唯一换行键",),
        )
        before = self._input_scene(
            "first",
            fingerprint="enter-before",
            auxiliary=enter,
        )
        after = self._input_scene("first\n", fingerprint="enter-after")
        effect = {
            "element_state": {
                "meaning": "application_text_input",
                "states": {"value": "first\n"},
            }
        }
        action = SemanticAction(
            node_id="enter-step",
            action="press_enter",
            params={"expected_effect": effect},
        )
        result = SimpleNamespace(
            action_outcome="matched",
            physical_actions=1,
            verification_errors=(),
            before_scene=before,
            after_scene=after,
            resolved_action=ResolvedSemanticAction(
                node_id="enter-step",
                kind="press_enter",
                target_element_id="enter-key",
                before_fingerprint=before.fingerprint,
                expected_effect=effect,
            ),
        )

        self.assertTrue(
            UniversalAgentOrchestrator._verified_input_transaction_microstep(
                graph=graph,
                previous_decision=SimpleNamespace(
                    proposal=GenericStepProposal(status="action", action=action)
                ),
                result=result,
                before_observation=SimpleNamespace(fingerprint=before.fingerprint),
                new_observation=SimpleNamespace(fingerprint=after.fingerprint),
            )
        )

    def test_verified_first_line_exits_microstep_for_formal_newline_successor(
        self,
    ) -> None:
        base = self._input_graph()
        first = replace(
            base.subgoals[0],
            subgoal_id="input_first_line",
            objective="在输入框中输入 first",
            completion_conditions=("输入框内容为 first",),
        )
        newline = replace(
            base.subgoals[0],
            subgoal_id="press_enter",
            objective="点击手机键盘右下角换行键",
            status="pending",
            depends_on=("input_first_line",),
            completion_conditions=("输入框内容为 first 加换行",),
        )
        second = replace(
            base.subgoals[0],
            subgoal_id="input_second_line",
            objective="在输入框第二行输入 second",
            status="pending",
            depends_on=("press_enter",),
            completion_conditions=("输入框内容为 first 换行 second",),
        )
        graph = replace(
            base,
            goal=replace(
                base.goal,
                objective="第一行输入 first，换行后第二行输入 second",
                entities={"input_text": "first\nsecond"},
            ),
            subgoals=(first, newline, second),
            active_subgoal_id="input_first_line",
            raw_user_goal="第一行输入 first，换行后第二行输入 second",
        )
        graph.validate()
        before = self._input_scene("", fingerprint="first-line-before")
        after = self._input_scene("first", fingerprint="first-line-after")
        expected_effect = {
            "element_state": {
                "meaning": "application_text_input",
                "states": {"value": "first"},
            }
        }
        action = SemanticAction(
            node_id="first-line-step",
            action="input_verified_text",
            params={"text": "first\nsecond"},
        )
        result = SimpleNamespace(
            action_outcome="matched",
            physical_actions=1,
            verification_errors=(),
            before_scene=before,
            after_scene=after,
            resolved_action=ResolvedSemanticAction(
                node_id="first-line-step",
                kind="input_verified_text",
                text="first\nsecond",
                input_fragment="first",
                input_method="direct_latin",
                prior_input_value="",
                expected_input_value="first",
                target_element_id="input-1",
                before_fingerprint=before.fingerprint,
                expected_effect=expected_effect,
            ),
        )

        self.assertFalse(
            UniversalAgentOrchestrator._verified_input_transaction_microstep(
                graph=graph,
                previous_decision=SimpleNamespace(
                    proposal=GenericStepProposal(status="action", action=action)
                ),
                result=result,
                before_observation=SimpleNamespace(
                    fingerprint=before.fingerprint
                ),
                new_observation=SimpleNamespace(fingerprint=after.fingerprint),
            )
        )
        unrelated = replace(
            graph,
            subgoals=(
                first,
                replace(newline, objective="打开下一页"),
                second,
            ),
        )
        unrelated.validate()
        self.assertTrue(
            UniversalAgentOrchestrator._verified_input_transaction_microstep(
                graph=unrelated,
                previous_decision=SimpleNamespace(
                    proposal=GenericStepProposal(status="action", action=action)
                ),
                result=result,
                before_observation=SimpleNamespace(
                    fingerprint=before.fingerprint
                ),
                new_observation=SimpleNamespace(fingerprint=after.fingerprint),
            )
        )

    def test_verified_direct_input_canonical_value_exits_microstep(self) -> None:
        graph = self._input_graph()
        graph = replace(
            graph,
            goal=replace(
                graph.goal,
                objective="当前输入框逐字显示 agent 且尚未提交",
                entities={"input_text": "agent"},
            ),
            completion_conditions=(
                replace(
                    graph.completion_conditions[0],
                    description="当前输入框逐字显示 agent 且尚未提交",
                    evidence_required=("当前输入框逐字显示 agent",),
                ),
            ),
            subgoals=(
                replace(
                    graph.subgoals[0],
                    objective="当前输入框逐字显示 agent 且尚未提交",
                    completion_conditions=("当前输入框逐字显示 agent",),
                ),
            ),
            raw_user_goal="在当前输入框输入 agent，但不要提交",
        )
        graph.validate()
        before = self._input_scene("", fingerprint="direct-final-before")
        after = self._input_scene("agent", fingerprint="direct-final-after")
        expected_effect = {
            "element_state": {
                "meaning": "application_text_input",
                "states": {"value": "agent"},
            },
            "goal_complete_on_success": True,
        }
        action = SemanticAction(
            node_id="direct-final-step",
            action="input_verified_text",
            params={"text": "agent", "expected_effect": expected_effect},
        )
        result = SimpleNamespace(
            action_outcome="matched",
            physical_actions=1,
            verification_errors=(),
            before_scene=before,
            after_scene=after,
            resolved_action=ResolvedSemanticAction(
                node_id="direct-final-step",
                kind="input_verified_text",
                text="agent",
                input_fragment="agent",
                input_method="direct_latin",
                prior_input_value="",
                expected_input_value="agent",
                target_element_id="input-1",
                before_fingerprint=before.fingerprint,
                expected_effect=expected_effect,
            ),
        )

        self.assertFalse(
            UniversalAgentOrchestrator._verified_input_transaction_microstep(
                graph=graph,
                previous_decision=SimpleNamespace(
                    proposal=GenericStepProposal(status="action", action=action)
                ),
                result=result,
                before_observation=SimpleNamespace(fingerprint=before.fingerprint),
                new_observation=SimpleNamespace(fingerprint=after.fingerprint),
            )
        )
        self.assertTrue(
            UniversalAgentOrchestrator._verified_input_transaction_microstep(
                graph=graph,
                previous_decision=SimpleNamespace(
                    proposal=GenericStepProposal(status="action", action=action)
                ),
                result=result,
                before_observation=SimpleNamespace(fingerprint=before.fingerprint),
                new_observation=SimpleNamespace(fingerprint=after.fingerprint),
                allow_terminal=True,
            )
        )

    def test_local_exact_input_completion_builds_valid_terminal_graph(self) -> None:
        graph = self._input_graph()

        completed = UniversalAgentOrchestrator._complete_local_exact_input_graph(
            graph,
            new_observation=SimpleNamespace(
                observation_id="obs-exact-input-complete",
                fingerprint="exact-input-fingerprint",
            ),
        )

        self.assertEqual("completed", completed.status)
        self.assertEqual(graph.revision + 1, completed.revision)
        self.assertIsNone(completed.active_subgoal_id)
        self.assertTrue(all(item.satisfied for item in completed.completion_conditions))
        self.assertTrue(all(item.status == "completed" for item in completed.subgoals))

    def test_verified_pinyin_preedit_uses_fresh_rebound_scene_and_continues_to_candidate(self) -> None:
        graph = self._input_graph()
        graph = replace(
            graph,
            goal=replace(
                graph.goal,
                objective="当前输入框逐字显示你好且尚未提交",
                entities={"input_text": "你好"},
            ),
            raw_user_goal="在当前输入框输入你好，但不要提交",
        )
        graph.validate()
        before_base = self._input_scene("", fingerprint="fresh-before")
        before_field = replace(
            before_base.elements[0],
            states={
                **before_base.elements[0].states,
                "keyboard_input_mode": "chinese_pinyin",
            },
        )
        before = replace(before_base, elements=(before_field,))
        after_field = replace(
            before_field,
            states={
                **before_field.states,
                "ime_preedit_text": "nihao",
                "ime_exact_candidate_text": "你好",
            },
        )
        candidate = UIElement(
            element_id="local_audited_ime_candidate_1",
            role="button",
            meaning="ime_exact_candidate",
            label="你好",
            bounds=(0.1, 0.6, 0.25, 0.64),
            confidence=1.0,
            states={
                "goal_relevant": True,
                "fully_visible": True,
                "ime_candidate": True,
                "input_element_id": "input-1",
                "prior_input_value": "",
                "expected_input_value": "你好",
                "pinyin": "nihao",
            },
            evidence=("输入结构审计确认拼音 nihao 的唯一逐字候选：你好",),
        )
        after = replace(
            before,
            elements=(after_field, candidate),
            fingerprint="fresh-after",
        )
        after.validate()
        expected_effect = {
            "element_state": {
                "meaning": "application_text_input",
                "states": {
                    "value": "",
                    "ime_preedit_text": "nihao",
                    "ime_exact_candidate_text": "你好",
                },
            }
        }
        action = SemanticAction(
            node_id="pinyin-step",
            action="input_verified_text",
            params={"text": "你好"},
        )
        resolved = ResolvedSemanticAction(
            node_id="pinyin-step",
            kind="input_verified_text",
            text="你好",
            input_fragment="你好",
            input_method="chinese_pinyin",
            input_pinyin="nihao",
            prior_input_value="",
            expected_input_value="你好",
            target_element_id="input-1",
            before_fingerprint=before.fingerprint,
            expected_effect=expected_effect,
        )
        result = SimpleNamespace(
            action_outcome="matched",
            physical_actions=1,
            verification_errors=(),
            before_scene=before,
            after_scene=after,
            resolved_action=resolved,
        )
        decision = SimpleNamespace(
            proposal=GenericStepProposal(status="action", action=action)
        )

        self.assertTrue(
            UniversalAgentOrchestrator._verified_input_transaction_microstep(
                graph=graph,
                previous_decision=decision,
                result=result,
                before_observation=SimpleNamespace(fingerprint="planned-before"),
                new_observation=SimpleNamespace(fingerprint=after.fingerprint),
            )
        )
        self.assertFalse(
            UniversalAgentOrchestrator._verified_input_transaction_microstep(
                graph=graph,
                previous_decision=decision,
                result=SimpleNamespace(
                    **{
                        **vars(result),
                        "resolved_action": replace(
                        resolved,
                        before_fingerprint="wrong-fresh-before",
                        ),
                    }
                ),
                before_observation=SimpleNamespace(fingerprint="planned-before"),
                new_observation=SimpleNamespace(fingerprint=after.fingerprint),
            )
        )

        final_after = self._input_scene("你好", fingerprint="candidate-after")
        final_effect = {
            "element_state": {
                "meaning": "application_text_input",
                "states": {"value": "你好"},
            },
            "goal_complete_on_success": True,
        }
        final_action = SemanticAction(
            node_id="candidate-step",
            action="tap_semantic",
            params={"expected_effect": final_effect},
        )
        final_result = SimpleNamespace(
            action_outcome="matched",
            physical_actions=1,
            verification_errors=(),
            before_scene=after,
            after_scene=final_after,
            resolved_action=ResolvedSemanticAction(
                node_id="candidate-step",
                kind="tap_semantic",
                target_element_id=candidate.element_id,
                before_fingerprint=after.fingerprint,
                expected_effect=final_effect,
            ),
        )

        self.assertFalse(
            UniversalAgentOrchestrator._verified_input_transaction_microstep(
                graph=graph,
                previous_decision=SimpleNamespace(
                    proposal=GenericStepProposal(
                        status="action",
                        action=final_action,
                    )
                ),
                result=final_result,
                before_observation=SimpleNamespace(fingerprint=after.fingerprint),
                new_observation=SimpleNamespace(
                    fingerprint=final_after.fingerprint
                ),
            )
        )

    def test_verified_literal_key_keeps_high_level_graph(self) -> None:
        graph = self._input_graph()
        key = UIElement(
            element_id="key-2",
            role="button",
            meaning="input_exact_literal_key",
            label="2",
            bounds=(0.18, 0.78, 0.25, 0.85),
            confidence=0.98,
            states={
                "goal_relevant": True,
                "fully_visible": True,
                "input_literal_key": True,
                "input_element_id": "input-1",
                "key_value": "2",
                "prior_input_value": "live",
                "expected_input_value": "live2",
            },
            evidence=("数字键 2 完整可见",),
        )
        before = self._input_scene(
            "live", fingerprint="key-before", auxiliary=key
        )
        after = self._input_scene("live2", fingerprint="key-after")
        expected_effect = {
            "element_state": {
                "meaning": "application_text_input",
                "states": {"value": "live2"},
            }
        }
        action = SemanticAction(
            node_id="key-step",
            action="tap_semantic",
            params={"expected_effect": expected_effect},
        )
        result = SimpleNamespace(
            action_outcome="matched",
            physical_actions=1,
            verification_errors=(),
            before_scene=before,
            after_scene=after,
            resolved_action=ResolvedSemanticAction(
                node_id="key-step",
                kind="tap_semantic",
                target_element_id="key-2",
                before_fingerprint=before.fingerprint,
                expected_effect=expected_effect,
            ),
        )
        decision = SimpleNamespace(
            proposal=GenericStepProposal(status="action", action=action)
        )

        self.assertTrue(
            UniversalAgentOrchestrator._verified_input_transaction_microstep(
                graph=graph,
                previous_decision=decision,
                result=result,
                before_observation=SimpleNamespace(fingerprint=before.fingerprint),
                new_observation=SimpleNamespace(fingerprint=after.fingerprint),
            )
        )

        final_key = replace(
            key,
            element_id="key-1",
            label="1",
            states={
                **key.states,
                "key_value": "1",
                "prior_input_value": "live2",
                "expected_input_value": "live21",
            },
        )
        final_before = self._input_scene(
            "live2",
            fingerprint="final-key-before",
            auxiliary=final_key,
        )
        final_after = self._input_scene(
            "live21",
            fingerprint="final-key-after",
        )
        final_effect = {
            "element_state": {
                "meaning": "application_text_input",
                "states": {"value": "live21"},
            },
            "goal_complete_on_success": True,
        }
        final_action = SemanticAction(
            node_id="final-key-step",
            action="tap_semantic",
            params={"expected_effect": final_effect},
        )
        final_result = SimpleNamespace(
            action_outcome="matched",
            physical_actions=1,
            verification_errors=(),
            before_scene=final_before,
            after_scene=final_after,
            resolved_action=ResolvedSemanticAction(
                node_id="final-key-step",
                kind="tap_semantic",
                target_element_id=final_key.element_id,
                before_fingerprint=final_before.fingerprint,
                expected_effect=final_effect,
            ),
        )

        self.assertFalse(
            UniversalAgentOrchestrator._verified_input_transaction_microstep(
                graph=graph,
                previous_decision=SimpleNamespace(
                    proposal=GenericStepProposal(
                        status="action",
                        action=final_action,
                    )
                ),
                result=final_result,
                before_observation=SimpleNamespace(
                    fingerprint=final_before.fingerprint
                ),
                new_observation=SimpleNamespace(
                    fingerprint=final_after.fingerprint
                ),
            )
        )

    def test_input_microstep_rejects_wrong_after_value(self) -> None:
        graph = self._input_graph()
        before = self._input_scene("", fingerprint="wrong-before")
        after = self._input_scene("lixe", fingerprint="wrong-after")
        action = SemanticAction(
            node_id="wrong-step",
            action="input_verified_text",
            params={"text": "live21"},
        )
        result = SimpleNamespace(
            action_outcome="matched",
            physical_actions=1,
            verification_errors=(),
            before_scene=before,
            after_scene=after,
            resolved_action=ResolvedSemanticAction(
                node_id="wrong-step",
                kind="input_verified_text",
                text="live21",
                input_fragment="live",
                input_method="direct_latin",
                prior_input_value="",
                expected_input_value="live",
                target_element_id="input-1",
                before_fingerprint=before.fingerprint,
                expected_effect={
                    "element_state": {
                        "meaning": "application_text_input",
                        "states": {"value": "live"},
                    }
                },
            ),
        )
        decision = SimpleNamespace(
            proposal=GenericStepProposal(status="action", action=action)
        )

        self.assertFalse(
            UniversalAgentOrchestrator._verified_input_transaction_microstep(
                graph=graph,
                previous_decision=decision,
                result=result,
                before_observation=SimpleNamespace(fingerprint=before.fingerprint),
                new_observation=SimpleNamespace(fingerprint=after.fingerprint),
            )
        )

    def test_verified_input_microstep_skips_deepseek_and_mints_fresh_scope(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            orchestrator, session, planner, qwen, adapter = self._started(temp)
            old_scope = dict(session.snapshot()["confirmation_scope"])
            with patch.object(
                orchestrator,
                "_verified_input_transaction_microstep",
                return_value=True,
            ):
                result = orchestrator.confirm_one(
                    session,
                    _confirmation(session),
                )
            persisted = json.loads(
                (Path(temp) / "post_action_transition_step_1.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(1, result.physical_actions)
        self.assertEqual(1, adapter.execute_calls)
        self.assertEqual([], planner.replan_calls)
        self.assertEqual(1, session.task_graph.revision)
        self.assertEqual("awaiting_confirmation", session.status)
        self.assertEqual(2, len(qwen.calls))
        self.assertTrue(persisted["input_transaction_progress"])
        self.assertEqual("advanced_to_new_confirmation", persisted["disposition"])
        new_scope = session.snapshot()["confirmation_scope"]
        self.assertNotEqual(old_scope["observation_id"], new_scope["observation_id"])
        self.assertNotEqual(old_scope["fingerprint"], new_scope["fingerprint"])
        self.assertNotEqual(old_scope["action_digest"], new_scope["action_digest"])

    def test_local_exact_input_terminal_skips_deepseek_and_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            orchestrator, session, planner, _qwen, _adapter = self._started(temp)
            session.task_graph = self._input_graph()
            session.local_exact_input_authority = True
            with patch.object(
                orchestrator,
                "_verified_input_transaction_microstep",
                return_value=True,
            ), patch.object(
                orchestrator,
                "_input_transaction_reached_canonical",
                return_value=True,
            ):
                result = orchestrator.confirm_one(
                    session,
                    _confirmation(session),
                )
            persisted = json.loads(
                (Path(temp) / "post_action_transition_step_1.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(1, result.physical_actions)
        self.assertEqual([], planner.replan_calls)
        self.assertEqual("succeeded", session.status)
        self.assertEqual("completed", session.task_graph.status)
        self.assertIsNone(session.task_graph.active_subgoal_id)
        self.assertTrue(persisted["input_transaction_completed"])
        self.assertEqual("task_completed", persisted["disposition"])

    def test_authority_digest_is_exact_while_progress_digest_is_semantic(self) -> None:
        first = SemanticAction(
            node_id="decision-a",
            action="tap_semantic",
            params={
                "element_id": "candidate-a",
                "target": "reload_current_page",
                "role": "button",
                "label": "刷新",
                "expected_effect": {"scene_changed": True},
            },
        )
        second = replace(
            first,
            node_id="decision-b",
            params={**first.params, "element_id": "candidate-b"},
        )

        self.assertNotEqual(_action_digest(first), _action_digest(second))
        self.assertEqual(
            _action_equivalence_digest(first),
            _action_equivalence_digest(second),
        )

    def _started(
        self,
        temp: str,
        *,
        planner=None,
        qwen=None,
        adapter=None,
        policy=None,
        evidence_store_factory=None,
    ):
        initial = _graph()
        planner = planner or FakeDeepSeekPlanner(
            initial,
            replan_result=replace(initial, revision=2),
        )
        qwen = qwen or FakeQwenObserver()
        adapter = adapter or FakeExecutingAdapter(
            _scene(),
            _scene(fingerprint="frame-b", meaning="open_more", label="查看更多"),
        )
        orchestrator = UniversalAgentOrchestrator(
            deepseek_planner=planner,
            qwen_observer=qwen,
            adapter_factory=lambda _device_id: adapter,
            trusted_observation_factory=_trusted_factory,
            policy=policy,
            evidence_store_factory=evidence_store_factory,
        )
        session = orchestrator.start(
            session_id="session-confirm",
            raw_goal="查看详情",
            device_id="device-1",
            run_dir=Path(temp),
        )
        return orchestrator, session, planner, qwen, adapter

    def test_exact_confirmation_executes_one_action_then_pauses(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            orchestrator, session, _planner, _qwen, adapter = self._started(temp)

            result = orchestrator.confirm_one(session, _confirmation(session))

        self.assertEqual(1, result.physical_actions)
        self.assertEqual(1, session.physical_actions)
        self.assertEqual(1, adapter.execute_calls)
        self.assertEqual("awaiting_confirmation", session.status)
        self.assertEqual(2, session.step_number)
        self.assertEqual(2, session.task_graph.revision)

    def test_confirmation_accepts_fresh_execution_scene_with_new_fingerprint(self) -> None:
        class FreshConfirmationAdapter(FakeExecutingAdapter):
            def execute(self, **kwargs):
                result = super().execute(**kwargs)
                fresh = _scene(fingerprint="frame-confirmed")
                return replace(
                    result,
                    before_scene=fresh,
                    resolved_action=replace(
                        result.resolved_action,
                        before_fingerprint=fresh.fingerprint,
                    ),
                )

        adapter = FreshConfirmationAdapter(
            _scene(),
            _scene(fingerprint="frame-b"),
        )
        with tempfile.TemporaryDirectory() as temp:
            orchestrator, session, planner, _qwen, adapter = self._started(
                temp,
                adapter=adapter,
            )
            result = orchestrator.confirm_one(session, _confirmation(session))

        receipt = planner.replan_calls[0][1].verified_action_transition
        self.assertEqual(1, result.physical_actions)
        self.assertEqual("frame-confirmed", result.before_scene.fingerprint)
        self.assertEqual("frame-a", result.planned_scene_fingerprint)
        self.assertEqual("frame-a", receipt.before_fingerprint)
        self.assertEqual("awaiting_confirmation", session.status)

    def test_confirmation_rejects_wrong_planned_scene_binding(self) -> None:
        class WrongPlannedBindingAdapter(FakeExecutingAdapter):
            def execute(self, **kwargs):
                return replace(
                    super().execute(**kwargs),
                    planned_scene_fingerprint="wrong-planned-frame",
                )

        adapter = WrongPlannedBindingAdapter(
            _scene(),
            _scene(fingerprint="frame-b"),
        )
        with tempfile.TemporaryDirectory() as temp:
            orchestrator, session, _planner, _qwen, adapter = self._started(
                temp,
                adapter=adapter,
            )
            with self.assertRaisesRegex(
                UniversalAgentOrchestratorError,
                "确认 scope 的规划画面",
            ):
                orchestrator.confirm_one(session, _confirmation(session))

        self.assertEqual(1, adapter.execute_calls)
        self.assertEqual(1, session.physical_actions)
        self.assertEqual(
            "post_action_failure",
            session.last_post_action_transition["transition_kind"],
        )

    def test_confirmation_rejects_wrong_execution_scene_binding(self) -> None:
        class WrongExecutionBindingAdapter(FakeExecutingAdapter):
            def execute(self, **kwargs):
                result = super().execute(**kwargs)
                return replace(
                    result,
                    resolved_action=replace(
                        result.resolved_action,
                        before_fingerprint="wrong-execution-frame",
                    ),
                )

        adapter = WrongExecutionBindingAdapter(
            _scene(),
            _scene(fingerprint="frame-b"),
        )
        with tempfile.TemporaryDirectory() as temp:
            orchestrator, session, _planner, _qwen, adapter = self._started(
                temp,
                adapter=adapter,
            )
            with self.assertRaisesRegex(
                UniversalAgentOrchestratorError,
                "复核后的执行前画面",
            ):
                orchestrator.confirm_one(session, _confirmation(session))

        self.assertEqual(1, adapter.execute_calls)
        self.assertEqual(1, session.physical_actions)
        self.assertEqual(
            "post_action_failure",
            session.last_post_action_transition["transition_kind"],
        )

    def test_consecutive_swipe_remains_a_valid_multistep_navigation(self) -> None:
        initial = _graph()
        planner = FakeDeepSeekPlanner(
            initial,
            replan_result=replace(initial, revision=2),
        )
        adapter = FakeExecutingAdapter(
            _scene(
                role="container",
                meaning="content_viewport",
                label="",
                states={"scroll_axis": "vertical"},
            ),
            _scene(
                fingerprint="frame-b",
                role="container",
                meaning="content_viewport",
                label="",
                states={"scroll_axis": "vertical"},
            ),
        )
        qwen = FakeQwenObserver(action_kind="swipe")
        with tempfile.TemporaryDirectory() as temp:
            orchestrator, session, _planner, _qwen, adapter = self._started(
                temp,
                planner=planner,
                qwen=qwen,
                adapter=adapter,
            )
            result = orchestrator.confirm_one(session, _confirmation(session))

        self.assertEqual(1, result.physical_actions)
        self.assertEqual(1, adapter.execute_calls)
        self.assertEqual(1, session.physical_actions)
        self.assertEqual("awaiting_confirmation", session.status)
        self.assertIsNotNone(session.snapshot()["confirmation_scope"])
        self.assertEqual(
            "advanced_to_new_confirmation",
            session.last_post_action_transition["disposition"],
        )
        self.assertEqual(
            "action_result_matched",
            planner.replan_calls[0][2],
        )
        receipt = planner.replan_calls[0][1].verified_action_transition
        self.assertEqual(session.session_id, receipt.session_id)
        self.assertEqual(1, receipt.physical_actions)

    def test_typed_controller_completion_finishes_in_one_action(self) -> None:
        initial = _graph()
        planner = FakeDeepSeekPlanner(
            initial,
            replan_result=_completed_graph(initial),
        )
        adapter = FakeExecutingAdapter(
            _scene(),
            _scene(fingerprint="frame-b"),
            controller_completion_evidence=(
                "控制器确认动作前后场景指纹发生变化：frame-a -> frame-b",
            ),
        )
        qwen = FakeQwenObserver(goal_complete_on_success=True)
        with tempfile.TemporaryDirectory() as temp:
            orchestrator, session, _planner, qwen, adapter = self._started(
                temp,
                planner=planner,
                qwen=qwen,
                adapter=adapter,
            )
            result = orchestrator.confirm_one(session, _confirmation(session))
            persisted_transition = json.loads(
                (Path(temp) / "post_action_transition_step_1.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(1, result.physical_actions)
        self.assertEqual(1, adapter.execute_calls)
        self.assertEqual(1, session.physical_actions)
        self.assertEqual(1, len(qwen.calls))
        self.assertEqual("succeeded", session.status)
        observed = planner.replan_calls[0][1]
        self.assertEqual(1, len(observed.controller_transition_evidence_refs))
        self.assertEqual(
            "task_completed",
            session.last_post_action_transition["disposition"],
        )
        self.assertEqual(
            session.last_post_action_transition,
            persisted_transition,
        )

    def test_ordinary_send_succeeds_once_with_fresh_visual_effect_receipt(self) -> None:
        initial = _ordinary_send_graph()
        before = _scene(
            meaning="send_message",
            label="发送",
            app_id="sample.messaging",
        )
        after = _scene(
            fingerprint="frame-send-result",
            meaning="message_sent_result",
            label="freshsendproof",
            role="text",
            evidence=("新消息气泡逐字显示 freshsendproof",),
            app_id="sample.messaging",
        )

        class SendCompletionPlanner(FakeDeepSeekPlanner):
            def replan(self, graph, observation, *, trigger, reason):
                self.replan_calls.append((graph, observation, trigger, reason))
                return _complete_with_current_visual_claim(graph, observation)

        planner = SendCompletionPlanner(initial)
        qwen = FakeQwenObserver()
        adapter = FakeExecutingAdapter(before, after)
        with tempfile.TemporaryDirectory() as temp:
            orchestrator = UniversalAgentOrchestrator(
                deepseek_planner=planner,
                qwen_observer=qwen,
                adapter_factory=lambda _device_id: adapter,
                trusted_observation_factory=_trusted_factory,
            )
            session = orchestrator.start(
                session_id="session-ordinary-send-fresh",
                raw_goal=initial.raw_user_goal,
                device_id="device-1",
                run_dir=Path(temp),
            )
            self.assertEqual("awaiting_confirmation", session.status)
            self.assertIsNone(session.snapshot()["effect_confirmation_scope"])

            result = orchestrator.confirm_one(session, _confirmation(session))

        self.assertEqual(1, result.physical_actions)
        self.assertEqual(1, adapter.execute_calls)
        self.assertEqual(1, session.physical_actions)
        self.assertEqual("succeeded", session.status)
        self.assertEqual("completed", session.task_graph.status)
        self.assertIsNone(session.task_graph.active_subgoal_id)
        self.assertEqual(1, len(qwen.calls))
        receipt = planner.replan_calls[0][1].verified_action_transition
        self.assertEqual("matched", receipt.outcome)
        self.assertEqual(1, receipt.physical_actions)
        self.assertEqual("frame-send-result", receipt.after_fingerprint)
        self.assertEqual("send_message", session.effect_previews[0]["effect_kind"])
        self.assertEqual(
            ["freshsendproof"],
            [item["value"] for item in session.effect_previews[0]["payloads"]],
        )
        self.assertEqual("automatic", session.effect_previews[0]["policy"])
        self.assertIsNone(session.effect_verification)

    def test_ordinary_send_uses_one_zero_action_result_refresh_without_resending(
        self,
    ) -> None:
        initial = _ordinary_send_graph()
        before = _scene(
            meaning="send_message",
            label="发送",
            app_id="sample.messaging",
        )
        after = _scene(
            fingerprint="frame-send-result",
            meaning="message_sent_result",
            label="freshsendproof",
            role="text",
            evidence=("新消息气泡逐字显示 freshsendproof",),
            app_id="sample.messaging",
        )

        class DeferredSendCompletionPlanner(FakeDeepSeekPlanner):
            def replan(self, graph, observation, *, trigger, reason):
                self.replan_calls.append((graph, observation, trigger, reason))
                if len(self.replan_calls) == 1:
                    return replace(graph, revision=graph.revision + 1)
                return _complete_with_current_visual_claim(graph, observation)

        planner = DeferredSendCompletionPlanner(initial)
        qwen = FakeQwenObserver()
        adapter = FakeExecutingAdapter(before, after)
        with tempfile.TemporaryDirectory() as temp:
            orchestrator = UniversalAgentOrchestrator(
                deepseek_planner=planner,
                qwen_observer=qwen,
                adapter_factory=lambda _device_id: adapter,
                trusted_observation_factory=_trusted_factory,
            )
            session = orchestrator.start(
                session_id="session-ordinary-send-read-only-proof",
                raw_goal=initial.raw_user_goal,
                device_id="device-1",
                run_dir=Path(temp),
            )
            result = orchestrator.confirm_one(session, _confirmation(session))
            self.assertEqual("needs_effect_verification", session.status)
            self.assertEqual(1, session.physical_actions)
            self.assertIsNone(session.snapshot()["confirmation_scope"])

            adapter.scene = after
            refresh = orchestrator.refresh_decision(session)

        self.assertEqual(1, result.physical_actions)
        self.assertEqual("finished", refresh.proposal.status)
        self.assertEqual("succeeded", session.status)
        self.assertEqual(1, adapter.execute_calls)
        self.assertEqual(1, session.physical_actions)
        self.assertEqual(1, len(qwen.calls))
        self.assertEqual(
            ["action_result_matched", "observation_changed"],
            [call[2] for call in planner.replan_calls],
        )
        self.assertEqual("verified", session.effect_verification["status"])
        self.assertEqual(1, session.effect_verification["verification_attempts"])
        self.assertEqual(
            "frame-send-result",
            session.effect_verification["verification_fingerprint"],
        )

    def test_wait_for_change_is_zero_action_observation_transition(self) -> None:
        class WaitAdapter(FakeExecutingAdapter):
            def execute(self, **kwargs):
                return replace(super().execute(**kwargs), physical_actions=0)

        initial = _graph()
        planner = FakeDeepSeekPlanner(
            initial,
            replan_result=replace(initial, revision=2),
        )
        qwen = FakeQwenObserver(action_kind="wait_for_change")
        adapter = WaitAdapter(_scene(), _scene())
        with tempfile.TemporaryDirectory() as temp:
            orchestrator, session, _planner, _qwen, adapter = self._started(
                temp,
                planner=planner,
                qwen=qwen,
                adapter=adapter,
            )
            result = orchestrator.confirm_one(session, _confirmation(session))

        self.assertEqual(0, result.physical_actions)
        self.assertEqual(0, session.physical_actions)
        self.assertEqual(1, adapter.execute_calls)
        self.assertEqual("observation_changed", planner.replan_calls[0][2])
        self.assertIsNone(
            planner.replan_calls[0][1].verified_action_transition
        )
        self.assertEqual(
            "wait_observation",
            session.last_post_action_transition["transition_kind"],
        )
        self.assertEqual("awaiting_confirmation", session.status)

    def test_wrong_rebound_binding_fails_after_counting_one_action(self) -> None:
        class WrongBindingAdapter(FakeExecutingAdapter):
            def execute(self, **kwargs):
                result = super().execute(**kwargs)
                return replace(
                    result,
                    rebound_action=replace(
                        result.rebound_action,
                        node_id="wrong-node",
                    ),
                )

        adapter = WrongBindingAdapter(
            _scene(),
            _scene(fingerprint="frame-b"),
        )
        with tempfile.TemporaryDirectory() as temp:
            orchestrator, session, planner, qwen, adapter = self._started(
                temp,
                adapter=adapter,
            )
            with self.assertRaisesRegex(
                UniversalAgentOrchestratorError,
                "confirmed/requested/rebound/resolved",
            ):
                orchestrator.confirm_one(session, _confirmation(session))
            transition, report = _failure_artifacts(temp)

        self.assertEqual(1, adapter.execute_calls)
        self.assertEqual(1, session.physical_actions)
        self.assertEqual([], planner.replan_calls)
        self.assertEqual(1, len(qwen.calls))
        self.assertEqual("failed", session.status)
        self.assertEqual("validating_execution_result", transition["failed_stage"])
        self.assertEqual(1, transition["request_physical_actions"])
        self.assertEqual(transition, session.last_post_action_transition)
        self.assertEqual("failed", report["session"]["status"])
        self.assertEqual(
            transition,
            report["session"]["last_post_action_transition"],
        )

    def test_bad_after_frames_persist_the_same_failed_terminal_state(self) -> None:
        class BadFramesAdapter(FakeExecutingAdapter):
            def execute(self, **kwargs):
                result = super().execute(**kwargs)
                return replace(
                    result,
                    after_frames=result.after_frames[:3],
                    after_frame_paths=result.after_frame_paths[:3],
                )

        adapter = BadFramesAdapter(
            _scene(),
            _scene(fingerprint="frame-b"),
        )
        with tempfile.TemporaryDirectory() as temp:
            orchestrator, session, planner, _qwen, adapter = self._started(
                temp,
                adapter=adapter,
            )
            with self.assertRaisesRegex(
                UniversalAgentOrchestratorError,
                "原始帧证据",
            ):
                orchestrator.confirm_one(session, _confirmation(session))
            transition, report = _failure_artifacts(temp)

        self.assertEqual(1, adapter.execute_calls)
        self.assertEqual(1, session.physical_actions)
        self.assertEqual([], planner.replan_calls)
        self.assertEqual("failed", session.status)
        self.assertEqual(
            "validating_post_action_evidence",
            transition["failed_stage"],
        )
        self.assertEqual(transition, session.last_post_action_transition)
        self.assertEqual("failed", report["session"]["status"])
        self.assertEqual(
            transition,
            report["session"]["last_post_action_transition"],
        )

    def test_bad_after_fingerprint_persists_the_same_failed_terminal_state(self) -> None:
        calls = 0

        def wrong_after_factory(*, frames, device_id, scene, observation_id=None):
            nonlocal calls
            calls += 1
            observed = FakeTrustedObservation(
                device_id=device_id,
                scene=scene,
                observation_id=observation_id or "obs-start",
            )
            if calls == 2:
                observed.fingerprint = "wrong-after-fingerprint"
            return observed

        initial = _graph()
        planner = FakeDeepSeekPlanner(
            initial,
            replan_result=replace(initial, revision=2),
        )
        qwen = FakeQwenObserver()
        adapter = FakeExecutingAdapter(
            _scene(),
            _scene(fingerprint="frame-b"),
        )
        with tempfile.TemporaryDirectory() as temp:
            orchestrator = UniversalAgentOrchestrator(
                deepseek_planner=planner,
                qwen_observer=qwen,
                adapter_factory=lambda _device_id: adapter,
                trusted_observation_factory=wrong_after_factory,
            )
            session = orchestrator.start(
                session_id="session-bad-after-fp",
                raw_goal="查看详情",
                device_id="device-1",
                run_dir=Path(temp),
            )
            with self.assertRaisesRegex(
                UniversalAgentOrchestratorError,
                "after scene fingerprint",
            ):
                orchestrator.confirm_one(session, _confirmation(session))
            transition, report = _failure_artifacts(temp)

        self.assertEqual(1, adapter.execute_calls)
        self.assertEqual(1, session.physical_actions)
        self.assertEqual([], planner.replan_calls)
        self.assertEqual("failed", session.status)
        self.assertEqual("building_trusted_observation", transition["failed_stage"])
        self.assertEqual(transition, session.last_post_action_transition)
        self.assertEqual("failed", report["session"]["status"])
        self.assertEqual(
            transition,
            report["session"]["last_post_action_transition"],
        )

    def test_action_then_read_only_completion_gets_new_deepseek_revision(self) -> None:
        initial = _graph()
        read_only = replace(
            initial,
            revision=2,
            subgoals=(
                replace(
                    initial.subgoals[0],
                    status="completed",
                    completion_evidence=("上一页可见",),
                ),
                Subgoal(
                    subgoal_id="confirm-visible",
                    objective="确认上一页可见",
                    status="active",
                    depends_on=("subgoal-1",),
                    constraints=("不得执行物理动作",),
                    completion_conditions=("当前画面显示上一页",),
                    completion_evidence=(),
                    risk_action_ids=(),
                    external_impact="read_only",
                ),
            ),
            active_subgoal_id="confirm-visible",
        )
        read_only.validate()
        completed = _completed_graph(read_only)

        class SequentialPlanner(FakeDeepSeekPlanner):
            def replan(self, graph, observation, *, trigger, reason):
                self.replan_calls.append((graph, observation, trigger, reason))
                return read_only if graph.revision == 1 else completed

        planner = SequentialPlanner(initial)
        qwen = SequenceQwenObserver("action", "finished")
        with tempfile.TemporaryDirectory() as temp:
            orchestrator, session, _planner, _qwen, adapter = self._started(
                temp,
                planner=planner,
                qwen=qwen,
            )

            result = orchestrator.confirm_one(session, _confirmation(session))
            self.assertEqual("needs_reobservation", session.status)
            self.assertEqual(1, len(qwen.calls))
            adapter.scene = adapter.after_scene
            orchestrator.refresh_decision(session)

        self.assertEqual(1, result.physical_actions)
        self.assertEqual(1, adapter.execute_calls)
        self.assertEqual(1, session.physical_actions)
        self.assertEqual("succeeded", session.status)
        self.assertEqual(3, session.task_graph.revision)
        self.assertEqual("completed", session.task_graph.status)
        self.assertEqual(
            ["action_result_matched", "subgoal_completed"],
            [call[2] for call in planner.replan_calls],
        )
        self.assertIsNotNone(
            planner.replan_calls[0][1].verified_action_transition
        )
        self.assertIsNone(
            planner.replan_calls[1][1].verified_action_transition
        )
        self.assertEqual(
            (),
            planner.replan_calls[1][1].controller_transition_evidence_refs,
        )
        self.assertEqual(2, len(qwen.calls))

    def test_read_only_checkpoint_can_advance_to_later_navigation(self) -> None:
        initial = _graph()
        read_only = replace(
            initial,
            revision=2,
            subgoals=(
                replace(
                    initial.subgoals[0],
                    status="completed",
                    completion_evidence=("上一页可见",),
                ),
                Subgoal(
                    subgoal_id="confirm-visible",
                    objective="确认上一页可见",
                    status="active",
                    depends_on=("subgoal-1",),
                    constraints=("不得执行物理动作",),
                    completion_conditions=("当前画面显示上一页",),
                    completion_evidence=(),
                    risk_action_ids=(),
                    external_impact="read_only",
                ),
                Subgoal(
                    subgoal_id="continue-navigation",
                    objective="继续查看下一项公开信息",
                    status="pending",
                    depends_on=("confirm-visible",),
                    constraints=("不得改变任何账号状态",),
                    completion_conditions=("下一项公开信息可见",),
                    completion_evidence=(),
                    risk_action_ids=(),
                    external_impact="navigation_only",
                ),
            ),
            active_subgoal_id="confirm-visible",
        )
        read_only.validate()
        navigation = replace(
            read_only,
            revision=3,
            subgoals=(
                read_only.subgoals[0],
                replace(
                    read_only.subgoals[1],
                    status="completed",
                    completion_evidence=("上一页可见",),
                ),
                replace(read_only.subgoals[2], status="active"),
            ),
            active_subgoal_id="continue-navigation",
        )
        navigation.validate()

        class SequentialPlanner(FakeDeepSeekPlanner):
            def replan(self, graph, observation, *, trigger, reason):
                self.replan_calls.append((graph, observation, trigger, reason))
                return read_only if graph.revision == 1 else navigation

        planner = SequentialPlanner(initial)
        qwen = SequenceQwenObserver("action", "finished", "action")
        with tempfile.TemporaryDirectory() as temp:
            orchestrator, session, _planner, _qwen, adapter = self._started(
                temp,
                planner=planner,
                qwen=qwen,
            )

            result = orchestrator.confirm_one(session, _confirmation(session))
            self.assertEqual("needs_reobservation", session.status)
            adapter.scene = adapter.after_scene
            orchestrator.refresh_decision(session)
            self.assertEqual("needs_reobservation", session.status)
            orchestrator.refresh_decision(session)

        self.assertEqual(1, result.physical_actions)
        self.assertEqual(1, adapter.execute_calls)
        self.assertEqual(1, session.physical_actions)
        self.assertEqual("awaiting_confirmation", session.status)
        self.assertEqual(3, session.task_graph.revision)
        self.assertEqual("continue-navigation", session.task_graph.active_subgoal_id)
        self.assertEqual(3, len(qwen.calls))
        self.assertEqual(3, qwen.calls[-1]["task_context"]["revision"])

    def test_safe_loop_executes_one_confirmed_action_then_pauses_for_new_confirmation(self) -> None:
        initial = _graph()

        class SequentialPlanner(FakeDeepSeekPlanner):
            def replan(self, graph, observation, *, trigger, reason):
                self.replan_calls.append((graph, observation, trigger, reason))
                if graph.revision == 1:
                    return replace(initial, revision=2)
                return _completed_graph(graph)

        class SequentialAdapter(FakeExecutingAdapter):
            def __init__(self):
                super().__init__(
                    _scene(),
                    _scene(fingerprint="frame-b", meaning="open_more", label="查看更多"),
                )
                self.after_scenes = [
                    self.after_scene,
                    _scene(
                        fingerprint="frame-c",
                        meaning="open_final",
                        label="打开最终详情",
                    ),
                ]

            def execute(self, **kwargs):
                self.after_scene = self.after_scenes[self.execute_calls]
                return super().execute(**kwargs)

        planner = SequentialPlanner(initial)
        adapter = SequentialAdapter()
        with tempfile.TemporaryDirectory() as temp:
            orchestrator, session, _planner, qwen, _adapter = self._started(
                temp,
                planner=planner,
                adapter=adapter,
            )

            result = orchestrator.run_safe_loop(
                session,
                _confirmation(session),
                max_physical_actions=1,
                max_iterations=1,
            )

        self.assertEqual(1, result["physical_actions"])
        self.assertEqual(1, result["iterations"])
        self.assertEqual("awaiting_confirmation", result["status"])
        self.assertEqual(1, adapter.execute_calls)
        self.assertEqual(1, session.physical_actions)
        self.assertEqual(2, session.task_graph.revision)
        self.assertEqual(2, len(qwen.calls))
        self.assertFalse(session.automatic_loop_enabled)

    def test_safe_loop_rejects_more_than_one_physical_action_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            orchestrator, session, _planner, _qwen, adapter = self._started(temp)

            with self.assertRaisesRegex(
                UniversalAgentOrchestratorError,
                "每次确认最多执行一个物理动作",
            ):
                orchestrator.run_safe_loop(
                    session,
                    _confirmation(session),
                    max_physical_actions=2,
                    max_iterations=1,
                )

        self.assertEqual(0, adapter.execute_calls)
        self.assertEqual(0, session.physical_actions)

    def test_safe_loop_rejects_external_risk_stage_with_zero_actions(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            orchestrator = UniversalAgentOrchestrator(
                deepseek_planner=FakeDeepSeekPlanner(_external_graph()),
                qwen_observer=FakeQwenObserver(),
                adapter_factory=lambda _device_id: FakeAdapter(_scene()),
                trusted_observation_factory=_trusted_factory,
            )
            session = orchestrator.start(
                session_id="session-auto-risk",
                raw_goal="发送一条消息",
                device_id="device-1",
                run_dir=Path(temp),
            )
            with self.assertRaisesRegex(
                UniversalAgentOrchestratorError,
                "awaiting_effect_confirmation",
            ):
                orchestrator.run_safe_loop(
                    session,
                    _effect_confirmation(session),
                )

        self.assertEqual(0, session.physical_actions)

    def test_confirmation_requires_observation_id_and_fingerprint(self) -> None:
        for missing in ("observation_id", "fingerprint"):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as temp:
                orchestrator, session, _planner, _qwen, adapter = self._started(temp)
                confirmation = _confirmation(session)
                confirmation.pop(missing)

                with self.assertRaisesRegex(
                    UniversalAgentOrchestratorError, "确认作用域"
                ):
                    orchestrator.confirm_one(session, confirmation)

                self.assertEqual(0, adapter.execute_calls)

    def test_replay_and_authority_mismatches_fail(self) -> None:
        mutations = {
            "session_id": "other-session",
            "task_id": "other-task",
            "device_id": "device-other",
            "revision": 99,
            "subgoal_id": "other-subgoal",
            "effect_ids": ["other-risk"],
            "observation_id": "obs-other",
            "fingerprint": "frame-other",
        }
        for field_name, value in mutations.items():
            with self.subTest(field=field_name), tempfile.TemporaryDirectory() as temp:
                orchestrator, session, _planner, _qwen, adapter = self._started(temp)
                confirmation = _confirmation(session)
                confirmation[field_name] = value

                with self.assertRaises(UniversalAgentOrchestratorError):
                    orchestrator.confirm_one(session, confirmation)

                self.assertEqual(0, adapter.execute_calls)

        with tempfile.TemporaryDirectory() as temp:
            orchestrator, session, _planner, _qwen, adapter = self._started(temp)
            confirmation = _confirmation(session)
            orchestrator.confirm_one(session, confirmation)
            with self.assertRaisesRegex(
                UniversalAgentOrchestratorError, "使用|推进|不一致"
            ):
                orchestrator.confirm_one(session, confirmation)
            self.assertEqual(1, adapter.execute_calls)

    def test_bad_confirmation_scope_is_not_reported_as_post_action(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            orchestrator, session, _planner, _qwen, adapter = self._started(temp)
            confirmation = _confirmation(session)
            confirmation["revision"] = 99

            with self.assertRaisesRegex(UniversalAgentOrchestratorError, "不一致"):
                orchestrator.confirm_one(session, confirmation)
            transition, report = _confirmation_failure_artifacts(temp)
            post_action_exists = (
                Path(temp) / "post_action_transition_step_1.json"
            ).exists()

        self.assertEqual(0, adapter.execute_calls)
        self.assertEqual(0, session.physical_actions)
        self.assertEqual("failed", session.status)
        self.assertEqual("confirmation_failure", transition["transition_kind"])
        self.assertEqual("validating_confirmation", transition["failed_stage"])
        self.assertEqual(0, transition["request_physical_actions"])
        self.assertIsNone(session.last_post_action_transition)
        self.assertEqual(transition, session.last_confirmation_failure)
        self.assertFalse(post_action_exists)
        self.assertEqual("failed", report["session"]["status"])
        self.assertEqual(
            transition,
            report["session"]["last_confirmation_failure"],
        )
        self.assertIsNone(report["session"]["last_post_action_transition"])

    def test_policy_is_rechecked_immediately_before_execute(self) -> None:
        class CountingReportStore(AgentEvidenceStore):
            def __init__(self, run_dir):
                super().__init__(run_dir)
                self.report_writes = 0

            def write_report(self, report):
                self.report_writes += 1
                return super().write_report(report)

        class FlipPolicy(PhaseOneNavigationPolicy):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def evaluate(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    return super().evaluate(**kwargs)
                return SimpleNamespace(
                    allowed=False,
                    reason="策略状态已变化",
                    canonical_class="",
                )

        policy = FlipPolicy()
        stores = []

        def evidence_store_factory(run_dir):
            store = CountingReportStore(run_dir)
            stores.append(store)
            return store

        with tempfile.TemporaryDirectory() as temp:
            orchestrator, session, _planner, qwen, adapter = self._started(
                temp,
                policy=policy,
                evidence_store_factory=evidence_store_factory,
            )

            with self.assertRaisesRegex(UniversalAgentOrchestratorError, "策略状态已变化"):
                orchestrator.confirm_one(session, _confirmation(session))
            report = json.loads(
                (Path(temp) / "report.json").read_text(encoding="utf-8")
            )
            post_action_exists = (
                Path(temp) / "post_action_transition_step_1.json"
            ).exists()
            confirmation_failure_exists = (
                Path(temp) / "confirmation_failure_step_1.json"
            ).exists()

        self.assertEqual(2, policy.calls)
        self.assertEqual(0, adapter.execute_calls)
        self.assertEqual(0, session.physical_actions)
        self.assertEqual("blocked", session.status)
        self.assertEqual("策略状态已变化", session.failed_reason)
        self.assertIsNone(session.last_post_action_transition)
        self.assertIsNone(session.last_confirmation_failure)
        self.assertFalse(post_action_exists)
        self.assertFalse(confirmation_failure_exists)
        self.assertEqual(2, stores[0].report_writes)
        self.assertEqual("blocked", report["session"]["status"])
        self.assertEqual("策略状态已变化", report["session"]["failed_reason"])

    def test_new_observation_fingerprint_and_revision_are_required(self) -> None:
        initial = _graph()
        same_scene_adapter = FakeExecutingAdapter(_scene(), _scene())
        with tempfile.TemporaryDirectory() as temp:
            orchestrator, session, _planner, qwen, adapter = self._started(
                temp,
                planner=FakeDeepSeekPlanner(
                    initial, replan_result=replace(initial, revision=2)
                ),
                adapter=same_scene_adapter,
            )
            with self.assertRaisesRegex(UniversalAgentOrchestratorError, "fingerprint"):
                orchestrator.confirm_one(session, _confirmation(session))
        self.assertEqual(1, adapter.execute_calls)
        self.assertEqual(1, session.physical_actions)
        self.assertEqual(1, len(qwen.calls))

        with tempfile.TemporaryDirectory() as temp:
            orchestrator, session, _planner, qwen, adapter = self._started(
                temp,
                planner=FakeDeepSeekPlanner(initial, replan_result=initial),
            )
            orchestrator.confirm_one(session, _confirmation(session))
        self.assertEqual("blocked", session.status)
        self.assertIn("revision", session.failed_reason)
        self.assertEqual(1, adapter.execute_calls)
        self.assertEqual(1, len(qwen.calls))

        with tempfile.TemporaryDirectory() as temp:
            orchestrator, session, _planner, qwen, adapter = self._started(
                temp,
                planner=FakeDeepSeekPlanner(
                    initial,
                    replan_result=replace(initial, revision=3),
                ),
            )
            orchestrator.confirm_one(session, _confirmation(session))
        self.assertEqual("blocked", session.status)
        self.assertIn("+ 1", session.failed_reason)
        self.assertEqual(1, adapter.execute_calls)
        self.assertEqual(1, len(qwen.calls))

    def test_post_action_observation_must_bind_device_and_after_scene(self) -> None:
        calls = 0

        def drifting_factory(*, frames, device_id, scene, observation_id=None):
            nonlocal calls
            calls += 1
            return FakeTrustedObservation(
                device_id=(device_id if calls == 1 else "other-device"),
                scene=scene,
                observation_id=observation_id or "obs-start",
            )

        initial = _graph()
        planner = FakeDeepSeekPlanner(
            initial,
            replan_result=replace(initial, revision=2),
        )
        qwen = FakeQwenObserver()
        adapter = FakeExecutingAdapter(
            _scene(),
            _scene(fingerprint="frame-b", meaning="open_more", label="查看更多"),
        )
        with tempfile.TemporaryDirectory() as temp:
            orchestrator = UniversalAgentOrchestrator(
                deepseek_planner=planner,
                qwen_observer=qwen,
                adapter_factory=lambda _device_id: adapter,
                trusted_observation_factory=drifting_factory,
            )
            session = orchestrator.start(
                session_id="session-post-drift",
                raw_goal="查看详情",
                device_id="device-1",
                run_dir=Path(temp),
            )
            with self.assertRaisesRegex(
                UniversalAgentOrchestratorError,
                "device/after scene fingerprint",
            ):
                orchestrator.confirm_one(session, _confirmation(session))

        self.assertEqual(1, adapter.execute_calls)
        self.assertEqual(1, session.physical_actions)
        self.assertEqual(1, len(qwen.calls))
        self.assertEqual([], planner.replan_calls)
        self.assertIsNone(session.snapshot()["confirmation_scope"])

    def test_semantic_noop_is_recorded_and_replanned_without_retry(self) -> None:
        initial = _graph()
        planner = FakeDeepSeekPlanner(
            initial,
            replan_result=replace(initial, revision=2),
        )
        adapter = FakeExecutingAdapter(
            _scene(),
            _scene(fingerprint="camera-noise-only"),
            action_outcome="mismatched",
            verification_errors=(
                "第2轮动作结果不匹配：动作后页面没有可验证的语义变化。",
            ),
        )
        with tempfile.TemporaryDirectory() as temp:
            orchestrator, session, _planner, _qwen, adapter = self._started(
                temp,
                planner=planner,
                adapter=adapter,
            )

            result = orchestrator.confirm_one(session, _confirmation(session))

            verification = json.loads(
                (Path(temp) / "verification_step_1.json").read_text(encoding="utf-8")
            )

        self.assertEqual("mismatched", result.action_outcome)
        self.assertEqual(1, adapter.execute_calls)
        self.assertEqual(1, session.physical_actions)
        self.assertEqual(1, len(planner.replan_calls))
        _graph_before, observed, trigger, reason = planner.replan_calls[0]
        self.assertEqual("mismatched", observed.last_action_outcome)
        self.assertEqual("action_result_mismatch", trigger)
        self.assertIn("必须重规划", reason)
        self.assertFalse(verification["matched"])
        self.assertEqual("mismatched", verification["action_outcome"])
        self.assertIn("语义变化", verification["blocked_reasons"][0])

    def test_safe_loop_stops_after_one_semantic_noop(self) -> None:
        initial = _graph()
        planner = FakeDeepSeekPlanner(
            initial,
            replan_result=replace(initial, revision=2),
        )
        adapter = FakeExecutingAdapter(
            _scene(),
            _scene(fingerprint="camera-noise-only"),
            action_outcome="mismatched",
            verification_errors=("动作后页面没有可验证的语义变化。",),
        )
        with tempfile.TemporaryDirectory() as temp:
            orchestrator, session, _planner, _qwen, adapter = self._started(
                temp,
                planner=planner,
                adapter=adapter,
            )

            result = orchestrator.run_safe_loop(
                session,
                _confirmation(session),
                max_physical_actions=1,
                max_iterations=1,
            )

        self.assertEqual(1, result["physical_actions"])
        self.assertEqual(1, result["iterations"])
        self.assertIn("预期语义变化", result["pause_reason"])
        self.assertEqual(1, adapter.execute_calls)
        self.assertEqual(1, session.physical_actions)


class DeviceTaskRegistryTests(unittest.TestCase):
    def _orchestrator(self, registry, adapter, *, qwen=None, device_id="device-1"):
        return UniversalAgentOrchestrator(
            deepseek_planner=FakeDeepSeekPlanner(
                _graph(device_id=device_id),
                replan_result=replace(_graph(device_id=device_id), revision=2),
            ),
            qwen_observer=qwen or FakeQwenObserver(),
            adapter_factory=lambda _device_id: adapter,
            trusted_observation_factory=_trusted_factory,
            device_registry=registry,
        )

    def test_two_different_devices_keep_independent_active_sessions(self) -> None:
        registry = DeviceTaskRegistry()
        first_adapter = FakeAdapter(_scene())
        second_adapter = FakeAdapter(_scene(fingerprint="device-b-frame"))
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_session = self._orchestrator(
                registry,
                first_adapter,
                device_id="device-a",
            ).start(
                session_id="session-device-a",
                raw_goal="查看设备 A 的当前详情",
                device_id="device-a",
                run_dir=Path(first),
            )
            second_session = self._orchestrator(
                registry,
                second_adapter,
                device_id="device-b",
            ).start(
                session_id="session-device-b",
                raw_goal="查看设备 B 的当前详情",
                device_id="device-b",
                run_dir=Path(second),
            )

        self.assertEqual("awaiting_confirmation", first_session.status)
        self.assertEqual("awaiting_confirmation", second_session.status)
        self.assertEqual("session-device-a", registry.active_session("device-a"))
        self.assertEqual("session-device-b", registry.active_session("device-b"))
        self.assertEqual(0, first_adapter.execute_calls)
        self.assertEqual(0, second_adapter.execute_calls)

    def test_second_active_session_on_same_device_is_rejected(self) -> None:
        registry = DeviceTaskRegistry()
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            self._orchestrator(registry, FakeAdapter(_scene())).start(
                session_id="session-first",
                raw_goal="查看详情",
                device_id="device-1",
                run_dir=Path(first),
            )
            second_qwen = FakeQwenObserver()
            with self.assertRaisesRegex(UniversalAgentOrchestratorError, "已有活动任务"):
                self._orchestrator(
                    registry,
                    FakeAdapter(_scene()),
                    qwen=second_qwen,
                ).start(
                    session_id="session-second",
                    raw_goal="查看另一个页面",
                    device_id="device-1",
                    run_dir=Path(second),
                )

        self.assertEqual([], second_qwen.calls)
        self.assertEqual("session-first", registry.active_session("device-1"))

    def test_shared_lease_rejects_a_second_process_registry(self) -> None:
        with tempfile.TemporaryDirectory() as lease_dir:
            first = DeviceTaskRegistry(lease_directory=Path(lease_dir))
            second = DeviceTaskRegistry(lease_directory=Path(lease_dir))
            first.reserve("device-shared", "session-first")
            try:
                self.assertEqual(
                    "session-first", second.active_session("device-shared")
                )
                with self.assertRaisesRegex(
                    UniversalAgentOrchestratorError, "已有活动任务"
                ):
                    second.reserve("device-shared", "session-second")
            finally:
                first.release("device-shared", "session-first")

    def test_unlocked_stale_lease_file_is_not_an_active_session(self) -> None:
        with tempfile.TemporaryDirectory() as lease_dir:
            registry = DeviceTaskRegistry(lease_directory=Path(lease_dir))
            lease_path = registry._lease_path("device-stale")
            assert lease_path is not None
            lease_path.parent.mkdir(parents=True, exist_ok=True)
            lease_path.write_text(" stale metadata", encoding="utf-8")

            self.assertIsNone(registry.active_session("device-stale"))
            registry.reserve("device-stale", "new-session")
            registry.release("device-stale", "new-session")

    def test_terminal_session_releases_device(self) -> None:
        registry = DeviceTaskRegistry()
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            blocked = self._orchestrator(
                registry,
                FakeAdapter(_scene()),
                qwen=FakeQwenObserver("blocked"),
            ).start(
                session_id="session-blocked",
                raw_goal="查看详情",
                device_id="device-1",
                run_dir=Path(first),
            )
            next_session = self._orchestrator(
                registry,
                FakeAdapter(_scene()),
            ).start(
                session_id="session-next",
                raw_goal="查看另一个详情",
                device_id="device-1",
                run_dir=Path(second),
            )

        self.assertEqual("blocked", blocked.status)
        self.assertEqual("awaiting_confirmation", next_session.status)
        self.assertEqual("session-next", registry.active_session("device-1"))

    def test_pause_invalidates_confirmation_but_keeps_session_inspectable(self) -> None:
        registry = DeviceTaskRegistry()
        adapter = FakeAdapter(_scene())
        with tempfile.TemporaryDirectory() as temp:
            orchestrator = self._orchestrator(registry, adapter)
            session = orchestrator.start(
                session_id="session-pause",
                raw_goal="查看详情",
                device_id="device-1",
                run_dir=Path(temp),
            )
            authority = session.confirmation_authority

            orchestrator.pause(session)

        self.assertEqual("paused", session.status)
        self.assertTrue(authority.consumed)
        self.assertEqual("paused", session.snapshot()["status"])
        self.assertIsNone(registry.active_session("device-1"))
        self.assertEqual(0, adapter.execute_calls)

    def test_pause_releases_device_even_when_terminal_snapshot_write_fails(self) -> None:
        registry = DeviceTaskRegistry()
        adapter = FakeAdapter(_scene())
        with tempfile.TemporaryDirectory() as temp:
            orchestrator = self._orchestrator(registry, adapter)
            session = orchestrator.start(
                session_id="session-pause-write-failure",
                raw_goal="查看详情",
                device_id="device-1",
                run_dir=Path(temp),
            )
            authority = session.confirmation_authority
            with patch.object(
                orchestrator,
                "_write_terminal_snapshot",
                side_effect=OSError("disk full"),
            ):
                with self.assertRaisesRegex(OSError, "disk full"):
                    orchestrator.pause(session)

        self.assertEqual("paused", session.status)
        self.assertTrue(authority.consumed)
        self.assertIsNone(registry.active_session("device-1"))
        self.assertEqual(0, adapter.execute_calls)

    def test_cancel_releases_device_and_never_calls_robot(self) -> None:
        registry = DeviceTaskRegistry()
        adapter = FakeAdapter(_scene())
        with tempfile.TemporaryDirectory() as temp:
            orchestrator = self._orchestrator(registry, adapter)
            session = orchestrator.start(
                session_id="session-cancel",
                raw_goal="查看详情",
                device_id="device-1",
                run_dir=Path(temp),
            )

            orchestrator.cancel(session)

        self.assertEqual("cancelled", session.status)
        self.assertIsNone(registry.active_session("device-1"))
        self.assertEqual(0, adapter.execute_calls)

    def test_observe_execute_and_post_observe_share_device_lock(self) -> None:
        registry = DeviceTaskRegistry()

        class LockCheckingAdapter(FakeExecutingAdapter):
            def capture_scene(self, *args, **kwargs):
                if not registry.is_locked_by_current_thread("device-1"):
                    raise AssertionError("capture must hold device lock")
                return super().capture_scene(*args, **kwargs)

            def execute(self, **kwargs):
                if not registry.is_locked_by_current_thread("device-1"):
                    raise AssertionError("execute and post-observe must hold device lock")
                return super().execute(**kwargs)

        adapter = LockCheckingAdapter(
            _scene(),
            _scene(fingerprint="frame-b", meaning="open_more", label="查看更多"),
        )
        with tempfile.TemporaryDirectory() as temp:
            orchestrator = self._orchestrator(registry, adapter)
            session = orchestrator.start(
                session_id="session-lock",
                raw_goal="查看详情",
                device_id="device-1",
                run_dir=Path(temp),
            )
            orchestrator.confirm_one(session, _confirmation(session))

        self.assertEqual(1, adapter.execute_calls)


class UniversalAgentConfirmFailureTests(unittest.TestCase):
    _started = UniversalAgentConfirmTests._started

    def test_action_failure_is_not_retried(self) -> None:
        adapter = FakeExecutingAdapter(
            _scene(),
            _scene(fingerprint="frame-b", meaning="open_more", label="查看更多"),
            execute_error=GenericActionAdapterError(
                "动作后画面没有可验证变化",
                physical_actions=1,
                evidence=("failed-after.jpg",),
            ),
        )
        with tempfile.TemporaryDirectory() as temp:
            orchestrator, session, _planner, _qwen, adapter = self._started(
                temp, adapter=adapter
            )
            with self.assertRaises(GenericActionAdapterError):
                orchestrator.confirm_one(session, _confirmation(session))

        self.assertEqual(1, adapter.execute_calls)
        self.assertEqual(1, session.physical_actions)
        self.assertEqual("failed", session.status)

    def test_pre_action_drift_consumes_scope_then_replans_without_action(self) -> None:
        initial = _graph()
        planner = FakeDeepSeekPlanner(
            initial,
            replan_result=replace(initial, revision=2),
        )
        adapter = FakeExecutingAdapter(
            _scene(),
            _scene(fingerprint="unused-after"),
            execute_error=GenericActionAdapterError(
                "确认时本地真实画面已变化",
                physical_actions=0,
                evidence=("fresh-before-1.jpg",),
            ),
        )
        with tempfile.TemporaryDirectory() as temp:
            orchestrator, session, _planner, qwen, adapter = self._started(
                temp,
                planner=planner,
                adapter=adapter,
            )
            stale_authority = session.confirmation_authority
            with self.assertRaisesRegex(
                GenericActionAdapterError,
                "真实画面已变化",
            ):
                orchestrator.confirm_one(session, _confirmation(session))

            self.assertEqual("needs_reobservation", session.status)
            self.assertTrue(stale_authority.consumed)
            self.assertEqual("consumed_before_execution", stale_authority.invalid_reason)
            self.assertEqual(0, session.physical_actions)
            self.assertEqual(1, adapter.execute_calls)
            self.assertEqual(
                "needs_reobservation",
                session.last_confirmation_failure["disposition"],
            )
            self.assertEqual(
                session.session_id,
                orchestrator.device_registry.active_session(session.device_id),
            )

            adapter.scene = _scene(
                fingerprint="frame-drifted",
                meaning="open_alternative",
                label="打开替代只读入口",
            )
            decision = orchestrator.refresh_decision(session)

        self.assertEqual("action", decision.proposal.status)
        self.assertEqual("awaiting_confirmation", session.status)
        self.assertEqual(initial.revision + 1, session.task_graph.revision)
        self.assertEqual("observation_changed", planner.replan_calls[0][2])
        self.assertEqual(2, len(qwen.calls))
        self.assertEqual(1, adapter.execute_calls)
        self.assertEqual(0, session.physical_actions)

    def test_replan_failure_keeps_after_frames_and_blocks_old_plan(self) -> None:
        planner = FakeDeepSeekPlanner(
            _graph(),
            replan_error=RuntimeError("deepseek unavailable"),
        )
        qwen = FakeQwenObserver()
        with tempfile.TemporaryDirectory() as temp:
            orchestrator, session, _planner, qwen, adapter = self._started(
                temp, planner=planner, qwen=qwen
            )

            result = orchestrator.confirm_one(session, _confirmation(session))

        self.assertEqual(1, result.physical_actions)
        self.assertEqual("blocked", session.status)
        self.assertIn("deepseek unavailable", session.failed_reason)
        self.assertIn("after-1.jpg", session.evidence_paths)
        self.assertEqual(1, adapter.execute_calls)
        self.assertEqual(1, len(qwen.calls))

    def test_post_action_task_graph_failure_persists_redacted_candidate_diff(self) -> None:
        initial = _graph()
        candidate = initial.to_dict()
        candidate["goal"]["objective"] = "模型错误改写后的目标"
        candidate["authorization"] = "Bearer post-action-secret"
        planner = FakeDeepSeekPlanner(
            initial,
            replan_error=TaskGraphError(
                "重规划不能改写用户目标、目标 App 或目标实体。"
            ),
        )
        planner.last_raw_response = json.dumps(candidate, ensure_ascii=False)

        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp)
            orchestrator, session, _planner, _qwen, adapter = self._started(
                temp,
                planner=planner,
            )

            result = orchestrator.confirm_one(session, _confirmation(session))
            diagnostics = list(run_dir.glob("*post_action_replan*_deepseek_failure.json"))
            artifact = json.loads(diagnostics[0].read_text(encoding="utf-8"))
            report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))

        self.assertEqual(1, result.physical_actions)
        self.assertEqual("blocked", session.status)
        self.assertEqual(1, session.physical_actions)
        self.assertEqual(1, len(planner.replan_calls))
        self.assertEqual(1, adapter.execute_calls)
        self.assertEqual(1, len(diagnostics))
        self.assertEqual("post_action_replan", artifact["failed_stage"])
        self.assertEqual(
            ["goal_objective"],
            artifact["structured_candidate_diff"]["changed_fields"],
        )
        self.assertNotIn(
            "post-action-secret",
            json.dumps(artifact, ensure_ascii=False),
        )
        self.assertIn(str(diagnostics[0]), session.evidence_paths)
        self.assertIn(str(diagnostics[0]), report["session"]["evidence"])
        self.assertEqual("blocked_replan_failure", session.last_post_action_transition["disposition"])

    def test_post_action_task_graph_failure_without_raw_keeps_existing_terminal(self) -> None:
        planner = FakeDeepSeekPlanner(
            _graph(),
            replan_error=TaskGraphError("候选解析前失败"),
        )
        planner.last_raw_response = ""

        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp)
            orchestrator, session, _planner, _qwen, adapter = self._started(
                temp,
                planner=planner,
            )

            result = orchestrator.confirm_one(session, _confirmation(session))
            diagnostics = list(run_dir.glob("*_deepseek_failure.json"))

        self.assertEqual(1, result.physical_actions)
        self.assertEqual("blocked", session.status)
        self.assertIn("候选解析前失败", session.failed_reason)
        self.assertEqual(1, session.physical_actions)
        self.assertEqual(1, len(planner.replan_calls))
        self.assertEqual(1, adapter.execute_calls)
        self.assertEqual([], diagnostics)
        self.assertEqual("blocked_replan_failure", session.last_post_action_transition["disposition"])

    def test_next_qwen_decision_uses_only_new_revision_and_observation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            orchestrator, session, _planner, qwen, _adapter = self._started(temp)
            first_observation = session.trusted_observation

            orchestrator.confirm_one(session, _confirmation(session))

        self.assertEqual(2, len(qwen.calls))
        _frames, context, observation, decision_number = qwen.calls[1]
        self.assertEqual(2, context["revision"])
        self.assertEqual(2, decision_number)
        self.assertNotEqual(first_observation.observation_id, observation.observation_id)
        self.assertNotEqual(first_observation.fingerprint, observation.fingerprint)

    def test_evidence_failure_before_action_keeps_zero_physical_actions(self) -> None:
        class FailSecondControllerStore(AgentEvidenceStore):
            def __init__(self, run_dir):
                super().__init__(run_dir)
                self.controller_writes = 0

            def write_controller_decision(self, step_number, decision):
                self.controller_writes += 1
                if self.controller_writes == 2:
                    raise EvidenceStoreError("simulated controller evidence failure")
                return super().write_controller_decision(step_number, decision)

        with tempfile.TemporaryDirectory() as temp:
            orchestrator, session, _planner, _qwen, adapter = self._started(
                temp,
                evidence_store_factory=FailSecondControllerStore,
            )
            with self.assertRaises(EvidenceStoreError):
                orchestrator.confirm_one(session, _confirmation(session))

        self.assertEqual(0, adapter.execute_calls)
        self.assertEqual(0, session.physical_actions)

    def test_evidence_failure_after_action_blocks_any_next_action(self) -> None:
        class FailVerificationStore(AgentEvidenceStore):
            def write_verification(self, step_number, verification):
                raise EvidenceStoreError("simulated verification evidence failure")

        with tempfile.TemporaryDirectory() as temp:
            orchestrator, session, _planner, qwen, adapter = self._started(
                temp,
                evidence_store_factory=FailVerificationStore,
            )
            confirmation = _confirmation(session)
            with self.assertRaises(EvidenceStoreError):
                orchestrator.confirm_one(session, confirmation)
            with self.assertRaises(UniversalAgentOrchestratorError):
                orchestrator.confirm_one(session, confirmation)

        self.assertEqual(1, adapter.execute_calls)
        self.assertEqual(1, session.physical_actions)
        self.assertEqual(1, len(qwen.calls))

if __name__ == "__main__":
    unittest.main()
