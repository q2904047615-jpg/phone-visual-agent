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
    DynamicTaskGraph,
    GraphGoal,
    RiskAction,
    Subgoal,
    TargetApp,
)
from generic_step_planner import GenericStepProposal
from generic_action_adapter import (
    GenericActionAdapterError,
    GenericActionExecutionResult,
)
from semantic_executor import SemanticAction
from ui_scene import SystemUIFacts, UIElement, UIScene
from universal_action_controller import ResolvedSemanticAction
from universal_agent_orchestrator import (
    AgentEvidenceStore,
    EvidenceStoreError,
    DeviceTaskRegistry,
    ObservationBridge,
    PhaseOneNavigationPolicy,
    UniversalAgentOrchestrator,
    UniversalAgentOrchestratorError,
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
    system_ui=None,
) -> UIScene:
    current = UIScene(
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
                states=states or {},
                evidence=("画面中可见目标",),
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
    if element.states:
        params["states"] = dict(element.states)
    if action_kind == "swipe":
        params = {
            "direction": direction,
            "expected_effect": {"scene_changed": True},
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
        params["text"] = "蓝牙设置"
        params["states"] = dict(element.states)
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
                in {"tap_semantic", "dismiss_overlay", "input_verified_text", "long_press"}
                else "system_navigation"
                if action_kind in {"back", "home", "reveal_system_navigation"}
                else "screen"
            ),
            element_id=(
                element.element_id
                if action_kind
                in {"tap_semantic", "dismiss_overlay", "input_verified_text", "long_press"}
                else ""
            ),
            bounds=(
                element.bounds
                if action_kind
                in {"tap_semantic", "dismiss_overlay", "input_verified_text", "long_press"}
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
    return SimpleNamespace(
        task_id="task-1",
        device_id="device-1",
        revision=1,
        current_external_impact=impact,
        external_action_allowed=external_action_allowed,
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
            "risk_action_ids": list(risk_action_ids),
            "external_impact": impact,
        },
        risk_actions=tuple(risk_actions),
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
    ) -> None:
        super().__init__(scene)
        self.after_scene = after_scene
        self.execute_error = execute_error
        self.action_outcome = action_outcome
        self.verification_errors = verification_errors

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
                expected_effect={"scene_changed": True},
            ),
            before_scene=planned_scene,
            after_scene=self.after_scene,
            physical_actions=1,
            action_outcome=self.action_outcome,
            verification_errors=self.verification_errors,
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
        kwargs.pop("available_action_kinds", None)
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

    def test_allows_structured_system_navigation_reveal(self) -> None:
        scene = _scene(
            system_ui=SystemUIFacts(
                immersive_or_fullscreen=True,
                navigation_bar_visible=False,
            )
        )
        decision = _decision(scene, action_kind="reveal_system_navigation")

        result = self.policy.evaluate(
            task_context=_context(),
            trusted_observation=decision.trusted_observation,
            decision=decision,
            available_action_kinds=frozenset({"reveal_system_navigation"}),
        )

        self.assertTrue(result.allowed)
        self.assertEqual("reveal_system_navigation", result.canonical_class)

    def test_rejects_system_navigation_reveal_for_read_only_or_risk(self) -> None:
        scene = _scene(
            system_ui=SystemUIFacts(
                immersive_or_fullscreen=True,
                navigation_bar_visible=False,
            )
        )
        decision = _decision(scene, action_kind="reveal_system_navigation")
        for context, message in (
            (_context(impact="read_only"), "read_only"),
            (_context(risk_action_ids=("risk-1",)), "风险动作"),
        ):
            with self.subTest(message=message):
                result = self.policy.evaluate(
                    task_context=context,
                    trusted_observation=decision.trusted_observation,
                    decision=decision,
                )
                self.assertFalse(result.allowed)
                self.assertIn(message, result.reason)

    def test_rejects_system_navigation_reveal_for_unknown_or_visible_bar(self) -> None:
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
            scene = _scene(system_ui=facts)
            decision = _decision(scene, action_kind="reveal_system_navigation")
            result = self.policy.evaluate(
                task_context=_context(),
                trusted_observation=decision.trusted_observation,
                decision=decision,
            )
            self.assertFalse(result.allowed)
            self.assertIn("沉浸态且导航栏隐藏", result.reason)

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

    def test_allows_home_for_navigation_only_subgoal(self) -> None:
        scene = _scene()
        decision = _decision(scene, action_kind="home")

        result = self.policy.evaluate(
            task_context=_context(),
            trusted_observation=decision.trusted_observation,
            decision=decision,
            available_action_kinds=frozenset({"home"}),
        )

        self.assertTrue(result.allowed)
        self.assertEqual("home", result.canonical_class)

    def test_allows_observed_refresh_icon_as_generic_navigation(self) -> None:
        scene = _scene(meaning="refresh_page", label="刷新", role="icon")
        decision = _decision(scene)

        result = self.policy.evaluate(
            task_context=_context(),
            trusted_observation=decision.trusted_observation,
            decision=decision,
        )

        self.assertTrue(result.allowed)
        self.assertEqual("refresh", result.canonical_class)

    def test_allows_exact_formal_target_app_entry_without_open_word(self) -> None:
        scene = _scene(meaning="settings_app", label="设置", role="icon")
        decision = _decision(scene)

        result = self.policy.evaluate(
            task_context=_context(
                target_apps=({"app_id": "settings", "app_name": "系统设置"},),
            ),
            trusted_observation=decision.trusted_observation,
            decision=decision,
        )

        self.assertTrue(result.allowed)
        self.assertEqual("open", result.canonical_class)

    def test_rejects_unrelated_app_entry_without_navigation_semantics(self) -> None:
        scene = _scene(meaning="camera_app", label="相机", role="icon")
        decision = _decision(scene)

        result = self.policy.evaluate(
            task_context=_context(
                target_apps=({"app_id": "settings", "app_name": "系统设置"},),
            ),
            trusted_observation=decision.trusted_observation,
            decision=decision,
        )

        self.assertFalse(result.allowed)
        self.assertIn("无法证明", result.reason)

    def test_rejects_action_missing_from_device_capabilities(self) -> None:
        scene = _scene()
        decision = _decision(scene, action_kind="swipe")

        result = self.policy.evaluate(
            task_context=_context(),
            trusted_observation=decision.trusted_observation,
            decision=decision,
            available_action_kinds=frozenset({"back", "wait_for_change"}),
        )

        self.assertFalse(result.allowed)
        self.assertIn("没有本地验证动作能力", result.reason)

    def test_allows_verified_text_input_only_for_focused_input(self) -> None:
        scene = _scene(
            meaning="搜索输入框",
            label="搜索",
            role="input",
            states={
                "focused": True,
                "value": "",
                "keyboard_layout": "qwerty",
                "keyboard_input_mode": "direct_latin",
                "goal_relevant": True,
            },
        )
        decision = _decision(scene, action_kind="input_verified_text")

        result = self.policy.evaluate(
            task_context=_context(),
            trusted_observation=decision.trusted_observation,
            decision=decision,
        )

        self.assertTrue(result.allowed)
        self.assertEqual("input", result.canonical_class)

        chinese_scene = _scene(
            meaning="搜索输入框",
            label="搜索",
            role="input",
            states={
                "focused": True,
                "value": "",
                "keyboard_layout": "qwerty",
                "keyboard_input_mode": "chinese_pinyin",
                "goal_relevant": True,
            },
        )
        chinese_decision = _decision(
            chinese_scene,
            action_kind="input_verified_text",
        )
        denied = self.policy.evaluate(
            task_context=_context(),
            trusted_observation=chinese_decision.trusted_observation,
            decision=chinese_decision,
        )
        self.assertFalse(denied.allowed)
        self.assertIn("direct_latin", denied.reason)

    def test_allows_only_observed_chinese_to_direct_latin_mode_switch(self) -> None:
        input_element = UIElement(
            element_id="field",
            role="input",
            meaning="search_input",
            label="搜索",
            bounds=(0.08, 0.02, 0.60, 0.08),
            confidence=0.96,
            states={
                "goal_relevant": True,
                "focused": True,
                "value": "",
                "keyboard_layout": "qwerty",
                "keyboard_input_mode": "chinese_pinyin",
            },
        )
        switch = UIElement(
            element_id="mode-switch",
            role="button",
            meaning="switch_keyboard_input_mode",
            label="中",
            bounds=(0.68, 0.88, 0.78, 0.95),
            confidence=0.96,
            states={
                "keyboard_input_mode_switch": True,
                "current_mode": "chinese_pinyin",
                "target_mode": "direct_latin",
            },
            evidence=("键面显示中",),
        )
        scene = UIScene(
            app_id="sample.app",
            screen_id="search",
            summary="空白搜索框已聚焦，中文拼音 QWERTY 可见",
            elements=(input_element, switch),
            stable=True,
            confidence=0.95,
            fingerprint="frame-mode-switch",
        )
        decision = _decision(scene)
        decision.proposal = GenericStepProposal(
            status="action",
            action=SemanticAction(
                node_id="switch-mode",
                action="tap_semantic",
                params={
                    "element_id": switch.element_id,
                    "target": switch.meaning,
                    "meaning": switch.meaning,
                    "role": switch.role,
                    "label": switch.label,
                    "states": dict(switch.states),
                    "expected_effect": {"scene_changed": True},
                },
            ),
        )
        decision.target_region = SimpleNamespace(
            kind="element",
            element_id=switch.element_id,
            bounds=switch.bounds,
        )

        result = self.policy.evaluate(
            task_context=_context(),
            trusted_observation=decision.trusted_observation,
            decision=decision,
        )
        self.assertTrue(result.allowed)
        self.assertEqual("switch_keyboard_input_mode", result.canonical_class)

        unsafe_switch = replace(
            switch,
            states={**switch.states, "target_mode": "unknown"},
        )
        unsafe_scene = replace(scene, elements=(input_element, unsafe_switch))
        unsafe_decision = _decision(unsafe_scene)
        unsafe_decision.proposal = replace(
            decision.proposal,
            action=replace(
                decision.proposal.action,
                params={
                    **decision.proposal.action.params,
                    "states": dict(unsafe_switch.states),
                },
            ),
        )
        unsafe_decision.target_region = SimpleNamespace(
            kind="element",
            element_id=unsafe_switch.element_id,
            bounds=unsafe_switch.bounds,
        )
        denied = self.policy.evaluate(
            task_context=_context(),
            trusted_observation=unsafe_decision.trusted_observation,
            decision=unsafe_decision,
        )
        self.assertFalse(denied.allowed)

    def test_allows_external_effect_only_with_scope_confirmation(self) -> None:
        scene = _scene(meaning="send_message", label="发送", role="button")
        decision = _decision(scene)

        denied = self.policy.evaluate(
            task_context=_context(impact="external_state"),
            trusted_observation=decision.trusted_observation,
            decision=decision,
        )
        allowed = self.policy.evaluate(
            task_context=_context(
                impact="external_state",
                external_action_allowed=True,
            ),
            trusted_observation=decision.trusted_observation,
            decision=decision,
        )

        self.assertFalse(denied.allowed)
        self.assertTrue(allowed.allowed)
        self.assertEqual("external", allowed.canonical_class)

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

    def test_low_scene_confidence_allows_exact_unique_goal_element_tap(self) -> None:
        scene = _scene(
            meaning="open_tab_list",
            label="窗口",
            role="button",
            states={"goal_relevant": True},
            scene_confidence=0.6,
        )
        decision = _decision(scene)

        result = self.policy.evaluate(
            task_context=_context(),
            trusted_observation=decision.trusted_observation,
            decision=decision,
        )

        self.assertTrue(result.allowed)

    def test_low_scene_confidence_still_rejects_screen_action(self) -> None:
        scene = _scene(
            states={"goal_relevant": True},
            scene_confidence=0.6,
        )
        decision = _decision(scene, action_kind="swipe")

        result = self.policy.evaluate(
            task_context=_context(),
            trusted_observation=decision.trusted_observation,
            decision=decision,
        )

        self.assertFalse(result.allowed)
        self.assertIn("局部证据", result.reason)

    def test_allows_generic_forward_navigation_tap(self) -> None:
        scene = _scene(
            meaning="浏览器前进按钮",
            label=">",
            role="button",
        )
        decision = _decision(scene)

        result = self.policy.evaluate(
            task_context=_context(impact="navigation_only"),
            trusted_observation=decision.trusted_observation,
            decision=decision,
        )

        self.assertTrue(result.allowed)
        self.assertEqual("forward", result.canonical_class)

    def test_allows_generic_app_and_page_entry_navigation(self) -> None:
        for meaning, label in (
            ("浏览器应用入口", "浏览器"),
            ("启动浏览器应用", "浏览器"),
            ("账户页面入口", "账户"),
            ("application_entry", "工具"),
            ("page_entry", "帮助"),
            ("launch_application", "工具"),
            ("start_application", "工具"),
        ):
            with self.subTest(meaning=meaning):
                scene = _scene(meaning=meaning, label=label, role="icon")
                decision = _decision(scene)
                result = self.policy.evaluate(
                    task_context=_context(),
                    trusted_observation=decision.trusted_observation,
                    decision=decision,
                )
                self.assertTrue(result.allowed)
                self.assertEqual("open", result.canonical_class)

    def test_external_effect_entry_stays_forbidden(self) -> None:
        for meaning, label in (
            ("支付入口", "支付"),
            ("send_message_entry", "消息"),
            ("删除入口", "删除"),
        ):
            with self.subTest(meaning=meaning):
                scene = _scene(meaning=meaning, label=label, role="icon")
                decision = _decision(scene)
                result = self.policy.evaluate(
                    task_context=_context(),
                    trusted_observation=decision.trusted_observation,
                    decision=decision,
                )
                self.assertFalse(result.allowed)

    def test_allows_canonical_candidate_after_duplicate_alias_collapse(self) -> None:
        scene = _scene(meaning="open_browser", label="浏览器", role="icon")
        decision = _decision(scene)
        decision.trusted_observation.candidate_conflicts = (
            {
                "kind": "duplicate_visual_object_collapsed",
                "canonical_element_id": "candidate-1",
                "element_ids": ["candidate-1", "candidate-label"],
            },
        )

        result = self.policy.evaluate(
            task_context=_context(),
            trusted_observation=decision.trusted_observation,
            decision=decision,
        )

        self.assertTrue(result.allowed)

    def test_rejects_unresolved_overlap_conflict_for_candidate(self) -> None:
        scene = _scene(meaning="open_browser", label="浏览器", role="icon")
        decision = _decision(scene)
        decision.trusted_observation.candidate_conflicts = (
            {
                "kind": "overlapping_semantic_conflict",
                "element_ids": ["candidate-1", "other-action"],
                "iou": 0.8,
            },
        )

        result = self.policy.evaluate(
            task_context=_context(),
            trusted_observation=decision.trusted_observation,
            decision=decision,
        )

        self.assertFalse(result.allowed)
        self.assertIn("冲突", result.reason)

    def test_rejects_toggle_and_keyboard_roles(self) -> None:
        for role in ("toggle", "keyboard_key"):
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

    def test_allows_exact_input_candidate_tap_only_as_local_focus(self) -> None:
        scene = _scene(
            meaning="顶部搜索输入框",
            label="旧文字",
            role="input",
            states={"goal_relevant": True, "fully_visible": True},
        )
        decision = _decision(scene)

        result = self.policy.evaluate(
            task_context=_context(impact="navigation_only"),
            trusted_observation=decision.trusted_observation,
            decision=decision,
        )

        self.assertTrue(result.allowed)
        self.assertEqual("focus_input", result.canonical_class)

    def test_rejects_redundant_tap_on_already_focused_input(self) -> None:
        scene = _scene(
            meaning="application_text_input",
            label="",
            role="input",
            states={
                "goal_relevant": True,
                "fully_visible": True,
                "focused": True,
                "value": "",
                "keyboard_layout": "qwerty",
                "keyboard_input_mode": "chinese_pinyin",
            },
        )
        decision = _decision(scene)

        result = self.policy.evaluate(
            task_context=_context(impact="navigation_only"),
            trusted_observation=decision.trusted_observation,
            decision=decision,
        )

        self.assertFalse(result.allowed)
        self.assertIn("已由当前画面证明聚焦", result.reason)

    def test_allows_only_geometrically_bound_local_text_clear_control(self) -> None:
        input_element = UIElement(
            element_id="field",
            role="input",
            meaning="settings_search_input",
            label="搜索框",
            bounds=(0.08, 0.05, 0.72, 0.12),
            confidence=0.96,
            states={
                "goal_relevant": True,
                "focused": True,
                "value": "agent.com",
                "keyboard_layout": "qwerty",
                "keyboard_input_mode": "chinese_pinyin",
            },
        )
        clear_element = UIElement(
            element_id="clear",
            role="icon",
            meaning="clear_local_text",
            label="×",
            bounds=(0.64, 0.065, 0.70, 0.105),
            confidence=0.94,
            states={"local_text_clear": True},
        )
        scene = UIScene(
            app_id="sample.app",
            screen_id="search",
            summary="非空输入框和独立清空图标可见",
            elements=(input_element, clear_element),
            stable=True,
            confidence=0.95,
            fingerprint="frame-a",
        )
        decision = _decision(scene)
        decision.proposal = GenericStepProposal(
            status="action",
            action=SemanticAction(
                node_id="clear-local",
                action="tap_semantic",
                params={
                    "element_id": "clear",
                    "target": "clear_local_text",
                    "meaning": "clear_local_text",
                    "role": "icon",
                    "label": "×",
                    "states": {"local_text_clear": True},
                    "expected_effect": {"scene_changed": True},
                },
            ),
        )
        decision.target_region = SimpleNamespace(
            kind="element",
            element_id="clear",
            bounds=clear_element.bounds,
        )

        result = self.policy.evaluate(
            task_context=_context(impact="navigation_only"),
            trusted_observation=decision.trusted_observation,
            decision=decision,
        )

        self.assertTrue(result.allowed)
        self.assertEqual("clear_local_text", result.canonical_class)

        far_clear = replace(clear_element, bounds=(0.7, 0.7, 0.76, 0.76))
        far_scene = replace(scene, elements=(input_element, far_clear))
        far_decision = _decision(far_scene)
        far_decision.proposal = decision.proposal
        far_decision.target_region = SimpleNamespace(
            kind="element",
            element_id="clear",
            bounds=far_clear.bounds,
        )
        denied = self.policy.evaluate(
            task_context=_context(impact="navigation_only"),
            trusted_observation=far_decision.trusted_observation,
            decision=far_decision,
        )
        self.assertFalse(denied.allowed)
        self.assertIn("几何绑定", denied.reason)

        cancel_element = replace(
            clear_element,
            label="×",
            bounds=(0.77, 0.014, 0.88, 0.048),
            evidence=("右侧‘取消’按钮",),
        )
        cancel_scene = replace(scene, elements=(input_element, cancel_element))
        cancel_decision = _decision(cancel_scene)
        cancel_decision.proposal = GenericStepProposal(
            status="action",
            action=SemanticAction(
                node_id="clear-local-cancel",
                action="tap_semantic",
                params={
                    "element_id": "clear",
                    "target": "clear_local_text",
                    "meaning": "clear_local_text",
                    "role": "icon",
                    "label": "×",
                    "states": {"local_text_clear": True},
                    "expected_effect": {"scene_changed": True},
                },
            ),
        )
        cancel_decision.target_region = SimpleNamespace(
            kind="element",
            element_id="clear",
            bounds=cancel_element.bounds,
        )
        denied_cancel = self.policy.evaluate(
            task_context=_context(impact="navigation_only"),
            trusted_observation=cancel_decision.trusted_observation,
            decision=cancel_decision,
        )
        self.assertFalse(denied_cancel.allowed)

        robot_icon = replace(
            clear_element,
            label="",
            evidence=("蓝色输入法机器人图标",),
        )
        robot_scene = replace(scene, elements=(input_element, robot_icon))
        robot_decision = _decision(robot_scene)
        robot_decision.proposal = GenericStepProposal(
            status="action",
            action=SemanticAction(
                node_id="robot-not-clear",
                action="tap_semantic",
                params={
                    "element_id": "clear",
                    "target": "clear_local_text",
                    "meaning": "clear_local_text",
                    "role": "icon",
                    "label": "",
                    "states": {"local_text_clear": True},
                    "expected_effect": {"scene_changed": True},
                },
            ),
        )
        robot_decision.target_region = SimpleNamespace(
            kind="element",
            element_id="clear",
            bounds=robot_icon.bounds,
        )
        denied_robot = self.policy.evaluate(
            task_context=_context(impact="navigation_only"),
            trusted_observation=robot_decision.trusted_observation,
            decision=robot_decision,
        )
        self.assertFalse(denied_robot.allowed)
        self.assertIn("×", denied_robot.reason)
        self.assertIn("取消", denied_cancel.reason)

    def test_read_only_subgoal_cannot_focus_input(self) -> None:
        scene = _scene(
            meaning="顶部搜索输入框",
            label="旧文字",
            role="input",
            states={"goal_relevant": True, "fully_visible": True},
        )
        decision = _decision(scene)

        result = self.policy.evaluate(
            task_context=_context(impact="read_only"),
            trusted_observation=decision.trusted_observation,
            decision=decision,
        )

        self.assertFalse(result.allowed)
        self.assertIn("read_only", result.reason)

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
            ("confirm_and_return", "确认并返回"),
            ("approve_and_close", "同意并关闭"),
            ("accept_and_back", "确定并返回"),
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

    def test_allows_unique_structurally_goal_bound_unknown_navigation_tap(self) -> None:
        scene = _scene(
            meaning="purple_diamond_choice",
            label="紫色菱形",
            role="button",
            states={"goal_relevant": True},
        )
        element = replace(
            scene.elements[0],
            evidence=("紫色菱形入口与结构化目标实体一致",),
        )
        scene = replace(scene, elements=(element,))
        decision = _decision(scene)

        result = self.policy.evaluate(
            task_context=_context(
                entities={"target_object": "紫色菱形入口"},
                subgoal_objective="激活紫色菱形入口以显示后续内容",
                subgoal_completion_conditions=("后续内容已经可见",),
            ),
            trusted_observation=decision.trusted_observation,
            decision=decision,
        )

        self.assertTrue(result.allowed)
        self.assertEqual("goal_bound_tap", result.canonical_class)

    def test_goal_bound_unknown_tap_fails_closed_without_every_gate(self) -> None:
        base_scene = _scene(
            meaning="purple_diamond_choice",
            label="紫色菱形",
            role="button",
            states={"goal_relevant": True},
        )
        base_element = replace(
            base_scene.elements[0],
            evidence=("紫色菱形入口与结构化目标实体一致",),
        )
        base_scene = replace(base_scene, elements=(base_element,))
        safe_context = _context(
            entities={"target_object": "紫色菱形入口"},
            subgoal_objective="激活紫色菱形入口以显示后续内容",
            subgoal_completion_conditions=("后续内容已经可见",),
        )

        cases = []

        unrelated_decision = _decision(base_scene)
        cases.append(
            (
                "目标实体",
                _context(
                    entities={"target_object": "绿色圆形入口"},
                    subgoal_objective="激活绿色圆形入口以显示后续内容",
                    subgoal_completion_conditions=("后续内容已经可见",),
                ),
                unrelated_decision,
            )
        )

        no_binding_scene = replace(
            base_scene,
            elements=(replace(base_element, states={}),),
        )
        cases.append(("goal_relevant", safe_context, _decision(no_binding_scene)))

        missing_state_binding = _decision(base_scene)
        missing_state_binding.proposal.action.params.pop("states")
        cases.append(("逐项复用候选 states", safe_context, missing_state_binding))

        conflicted = _decision(base_scene)
        conflicted.trusted_observation.candidate_conflicts = (
            {
                "kind": "overlapping_semantic_conflict",
                "element_ids": [base_element.element_id, "candidate-alias"],
            },
        )
        cases.append(("冲突", safe_context, conflicted))

        second = replace(
            base_element,
            element_id="second-choice",
            meaning="second_unknown_choice",
            label="另一个紫色菱形",
            bounds=(0.55, 0.2, 0.85, 0.3),
        )
        multiple_scene = replace(base_scene, elements=(base_element, second))
        cases.append(("唯一高置信", safe_context, _decision(multiple_scene)))

        no_postcondition = _decision(base_scene)
        no_postcondition.proposal.action.params["expected_effect"] = {}
        cases.append(("结构化动作后预期", safe_context, no_postcondition))

        risky_context = _context(
            entities={"target_object": "紫色菱形入口"},
            subgoal_objective="激活紫色菱形入口以显示后续内容",
            subgoal_completion_conditions=("后续内容已经可见",),
            risk_actions=({"risk_id": "risk-1"},),
            risk_action_ids=("risk-1",),
        )
        cases.append(("无风险动作", risky_context, _decision(base_scene)))

        external_context = _context(
            impact="external_state",
            entities={"target_object": "紫色菱形入口"},
            subgoal_objective="激活紫色菱形入口以显示后续内容",
            subgoal_completion_conditions=("后续内容已经可见",),
        )
        cases.append(("external_state", external_context, _decision(base_scene)))

        unknown_context = _context(
            impact="unknown",
            entities={"target_object": "紫色菱形入口"},
            subgoal_objective="激活紫色菱形入口以显示后续内容",
            subgoal_completion_conditions=("后续内容已经可见",),
        )
        cases.append(("unknown", unknown_context, _decision(base_scene)))

        text_scene = replace(
            base_scene,
            elements=(replace(base_element, role="text"),),
        )
        cases.append(("可点击角色", safe_context, _decision(text_scene)))

        for message, context, decision in cases:
            with self.subTest(message=message):
                result = self.policy.evaluate(
                    task_context=context,
                    trusted_observation=decision.trusted_observation,
                    decision=decision,
                )
                self.assertFalse(result.allowed)
                self.assertIn(message, result.reason)

    def test_goal_bound_unknown_tap_rejects_destructive_or_transaction_semantics(self) -> None:
        destructive_scene = _scene(
            meaning="erase_record",
            label="Delete",
            role="button",
            states={"goal_relevant": True},
        )
        destructive_decision = _decision(destructive_scene)
        destructive = self.policy.evaluate(
            task_context=_context(
                entities={"target_object": "delete record"},
                subgoal_objective="delete the selected record",
                subgoal_completion_conditions=("record is deleted",),
            ),
            trusted_observation=destructive_decision.trusted_observation,
            decision=destructive_decision,
        )
        self.assertFalse(destructive.allowed)
        self.assertIn("账号或外部状态", destructive.reason)

        transaction_scene = _scene(
            meaning="membership_plan",
            label="Membership plan",
            role="button",
            states={"goal_relevant": True},
        )
        transaction_element = replace(
            transaction_scene.elements[0],
            evidence=("membership plan matches the structured target",),
        )
        transaction_scene = replace(
            transaction_scene,
            elements=(transaction_element,),
        )
        transaction_decision = _decision(transaction_scene)
        transaction = self.policy.evaluate(
            task_context=_context(
                entities={"target_object": "purchase membership plan"},
                subgoal_objective="purchase the membership plan",
                subgoal_completion_conditions=("membership purchase is complete",),
            ),
            trusted_observation=transaction_decision.trusted_observation,
            decision=transaction_decision,
        )
        self.assertFalse(transaction.allowed)
        self.assertIn("交易语义", transaction.reason)

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

    def test_drag_rejects_conflicted_or_forbidden_endpoints(self) -> None:
        source = UIElement(
            element_id="source",
            role="list_item",
            meaning="draggable_item",
            label="项目",
            bounds=(0.10, 0.20, 0.20, 0.30),
            confidence=0.95,
        )
        destination = UIElement(
            element_id="destination",
            role="container",
            meaning="drop_zone",
            label="目标",
            bounds=(0.70, 0.20, 0.90, 0.40),
            confidence=0.95,
        )
        scene = UIScene(
            app_id="sample.app",
            screen_id="board",
            summary="拖动场景",
            elements=(source, destination),
            stable=True,
            confidence=0.95,
            fingerprint="drag-frame",
        )
        action = SemanticAction(
            node_id="drag-1",
            action="drag",
            params={
                "source_element_id": source.element_id,
                "source_target": source.meaning,
                "source_role": source.role,
                "source_label": source.label,
                "destination_element_id": destination.element_id,
                "destination_target": destination.meaning,
                "destination_role": destination.role,
                "destination_label": destination.label,
                "expected_effect": {"scene_changed": True},
            },
        )
        observation = SimpleNamespace(
            device_id="device-1",
            fingerprint=scene.fingerprint,
            scene=scene,
            candidate_conflicts=(
                {"kind": "overlap", "element_ids": [source.element_id, "alias"]},
            ),
        )
        observation.target_local_candidate = scene.unique_trusted_goal_element
        decision = SimpleNamespace(
            task_id="task-1",
            device_id="device-1",
            revision=1,
            fingerprint=scene.fingerprint,
            confidence=0.94,
            proposal=GenericStepProposal(status="action", action=action),
            trusted_observation=observation,
            target_region=SimpleNamespace(
                kind="element_path",
                element_id=source.element_id,
                bounds=source.bounds,
                destination_element_id=destination.element_id,
                destination_bounds=destination.bounds,
            ),
        )

        result = self.policy.evaluate(
            task_context=_context(),
            trusted_observation=observation,
            decision=decision,
        )
        self.assertFalse(result.allowed)
        self.assertIn("冲突", result.reason)

        keyboard_source = replace(source, role="keyboard_key")
        keyboard_scene = replace(scene, elements=(keyboard_source, destination))
        observation.scene = keyboard_scene
        observation.fingerprint = keyboard_scene.fingerprint
        observation.candidate_conflicts = ()
        decision.trusted_observation = observation
        decision.proposal.action.params["source_role"] = "keyboard_key"
        result = self.policy.evaluate(
            task_context=_context(),
            trusted_observation=observation,
            decision=decision,
        )
        self.assertFalse(result.allowed)
        self.assertIn("禁止", result.reason)


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

    @staticmethod
    def _read_only_locate_graph() -> DynamicTaskGraph:
        base = _graph()
        graph = replace(
            base,
            goal=replace(
                base.goal,
                objective="定位当前可见输入框并把内容替换为 Agent123",
                entities={"input_text": "Agent123"},
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
                    objective="把当前输入框内容替换为 Agent123",
                    status="pending",
                    depends_on=("locate-input",),
                    constraints=("不要提交、搜索或发送",),
                    completion_conditions=("输入框显示 Agent123",),
                    completion_evidence=(),
                    risk_action_ids=(),
                    external_impact="navigation_only",
                ),
            ),
            active_subgoal_id="locate-input",
            raw_user_goal="把当前输入框内容替换为 Agent123，不要搜索或提交",
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

        self.assertEqual("awaiting_confirmation", session.status)
        self.assertEqual(2, session.task_graph.revision)
        self.assertEqual("replace-input", session.task_graph.active_subgoal_id)
        self.assertEqual(["subgoal_completed"], [call[2] for call in planner.replan_calls])
        self.assertEqual(1, len(qwen.calls))
        self.assertEqual(2, qwen.calls[0][1]["revision"])
        self.assertEqual(1, adapter.capture_calls)
        self.assertEqual(0, adapter.execute_calls)
        self.assertEqual(0, session.physical_actions)

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
                    objective="验证输入框文字内容是否为 Agent123",
                    completion_conditions=("输入框文字等于 Agent123",),
                ),
                initial.subgoals[1],
            ),
        )
        verify_value.validate()
        planner = FakeDeepSeekPlanner(
            verify_value,
            replan_result=self._advance_locate_graph(verify_value),
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

        self.assertEqual("blocked", session.status)
        self.assertEqual([], planner.replan_calls)
        self.assertEqual([], qwen.calls)
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

    def test_external_state_waits_for_risk_confirmation_before_qwen_and_robot(self) -> None:
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

        self.assertEqual("awaiting_risk_confirmation", session.status)
        self.assertTrue(session.snapshot()["risk_confirmation_ready"])
        self.assertEqual(["risk-1"], session.snapshot()["risk_confirmation_scope"]["risk_ids"])
        self.assertEqual(0, len(qwen.calls))
        self.assertEqual(0, adapter.capture_calls)
        self.assertEqual(0, adapter.execute_calls)
        self.assertEqual(0, session.physical_actions)

    def test_unknown_impact_waits_for_risk_confirmation_before_qwen_and_robot(self) -> None:
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

        self.assertEqual("awaiting_risk_confirmation", session.status)
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


def _confirmation(session) -> dict:
    graph = session.task_graph
    observation = session.trusted_observation
    current = graph.active_subgoal()
    return {
        "session_id": session.session_id,
        "task_id": graph.task_id,
        "device_id": graph.device_id,
        "revision": graph.revision,
        "subgoal_id": current.subgoal_id,
        "risk_ids": sorted(current.risk_action_ids),
        "observation_id": observation.observation_id,
        "fingerprint": observation.fingerprint,
    }


def _risk_confirmation(session) -> dict:
    return dict(session.snapshot()["risk_confirmation_scope"])


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
            _scene(),
            (_scene(fingerprint="frame-b", label="进入公开内容"), "matched", ()),
            (_scene(fingerprint="frame-c", label="公开内容已显示"), "matched", ()),
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
            self.assertEqual("awaiting_confirmation", session.status)
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

    def test_unchanged_screen_is_mismatch_evidence_and_replans_once(self) -> None:
        initial = self._unknown_app_graph()
        revised = replace(initial, revision=2)
        planner = SequenceDeepSeekPlanner(initial, revised)
        qwen = SequenceQwenObserver("action", "blocked")
        unchanged = _scene()
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
            _scene(fingerprint="frame-no-candidate"),
            summary="动作后目标候选已经不在当前画面",
            elements=(),
        )
        adapter = SequenceExecutingAdapter(
            _scene(),
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
            _scene(),
            _scene(fingerprint="frame-complete", label="公开内容已显示"),
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
        self.assertEqual("subgoal_completed", planner.replan_calls[0][2])
        self.assertEqual("succeeded", session.status)
        self.assertEqual(2, session.task_graph.revision)
        self.assertEqual(0, adapter.execute_calls)
        self.assertEqual(0, session.physical_actions)

    def test_refresh_changes_fingerprint_and_invalidates_old_action_scope(self) -> None:
        initial = self._unknown_app_graph()
        planner = SequenceDeepSeekPlanner(initial)
        qwen = SequenceQwenObserver("action", "action")
        adapter = SequenceCaptureAdapter(
            _scene(),
            _scene(fingerprint="frame-new-candidate", label="新的唯一入口"),
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
        self.assertEqual(0, adapter.execute_calls)
        self.assertEqual(0, session.physical_actions)


class UniversalAgentRiskConfirmationTests(unittest.TestCase):
    def _started(self, temp: str, *, impact: str = "external_state"):
        initial = _external_graph(impact=impact)
        planner = FakeDeepSeekPlanner(
            initial,
            replan_result=replace(initial, revision=2),
        )
        qwen = FakeQwenObserver()
        adapter = FakeExecutingAdapter(
            _scene(),
            _scene(fingerprint="frame-b", meaning="open_more", label="查看更多"),
        )
        orchestrator = UniversalAgentOrchestrator(
            deepseek_planner=planner,
            qwen_observer=qwen,
            adapter_factory=lambda _device_id: adapter,
            trusted_observation_factory=_trusted_factory,
        )
        session = orchestrator.start(
            session_id="session-risk-confirm",
            raw_goal="向目标联系人发送需求询问",
            device_id="device-1",
            run_dir=Path(temp),
        )
        return orchestrator, session, qwen, adapter

    def test_risk_approval_observes_once_then_requires_action_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            orchestrator, session, qwen, adapter = self._started(temp)

            decision = orchestrator.approve_risks(
                session,
                _risk_confirmation(session),
            )

        self.assertEqual("action", decision.proposal.status)
        self.assertEqual("awaiting_confirmation", session.status)
        self.assertEqual(("risk-1",), session.confirmed_risk_ids)
        self.assertEqual(1, len(qwen.calls))
        self.assertEqual(1, adapter.capture_calls)
        self.assertEqual(0, adapter.execute_calls)
        self.assertEqual(0, session.physical_actions)
        self.assertIsNotNone(session.snapshot()["confirmation_scope"])

    def test_risk_scope_mismatch_is_consumed_without_observation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            orchestrator, session, qwen, adapter = self._started(temp)
            scope = _risk_confirmation(session)
            scope["revision"] = 99

            with self.assertRaisesRegex(UniversalAgentOrchestratorError, "不一致"):
                orchestrator.approve_risks(session, scope)

        self.assertEqual([], qwen.calls)
        self.assertEqual(0, adapter.capture_calls)
        self.assertEqual(0, adapter.execute_calls)
        self.assertTrue(session.risk_confirmation_authority.consumed)

    def test_action_confirmation_executes_once_then_new_revision_requires_new_risk(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            orchestrator, session, qwen, adapter = self._started(temp)
            old_risk_scope = _risk_confirmation(session)
            orchestrator.approve_risks(session, old_risk_scope)

            result = orchestrator.confirm_one(session, _confirmation(session))
            new_risk_scope = dict(session.snapshot()["risk_confirmation_scope"])
            with self.assertRaisesRegex(
                UniversalAgentOrchestratorError,
                "不一致",
            ):
                orchestrator.approve_risks(session, old_risk_scope)

        self.assertEqual(1, result.physical_actions)
        self.assertEqual(1, adapter.execute_calls)
        self.assertEqual(1, session.physical_actions)
        self.assertEqual(2, session.task_graph.revision)
        self.assertEqual("awaiting_risk_confirmation", session.status)
        self.assertEqual((), session.confirmed_risk_ids)
        self.assertEqual(1, len(qwen.calls))
        self.assertIsNone(session.snapshot()["confirmation_scope"])
        self.assertEqual(2, new_risk_scope["revision"])
        self.assertNotEqual(old_risk_scope, new_risk_scope)
        self.assertFalse(session.snapshot()["risk_confirmation_ready"])
        self.assertIsNone(session.snapshot()["risk_confirmation_scope"])


class UniversalAgentConfirmTests(unittest.TestCase):
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
        qwen = FakeQwenObserver()
        with tempfile.TemporaryDirectory() as temp:
            orchestrator, session, _planner, _qwen, adapter = self._started(
                temp,
                planner=planner,
                qwen=qwen,
            )

            result = orchestrator.confirm_one(session, _confirmation(session))

        self.assertEqual(1, result.physical_actions)
        self.assertEqual(1, adapter.execute_calls)
        self.assertEqual(1, session.physical_actions)
        self.assertEqual("succeeded", session.status)
        self.assertEqual(3, session.task_graph.revision)
        self.assertEqual("completed", session.task_graph.status)
        self.assertEqual(
            ["observation_changed", "subgoal_completed"],
            [call[2] for call in planner.replan_calls],
        )
        self.assertEqual(1, len(qwen.calls))

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
        qwen = FakeQwenObserver()
        with tempfile.TemporaryDirectory() as temp:
            orchestrator, session, _planner, _qwen, adapter = self._started(
                temp,
                planner=planner,
                qwen=qwen,
            )

            result = orchestrator.confirm_one(session, _confirmation(session))

        self.assertEqual(1, result.physical_actions)
        self.assertEqual(1, adapter.execute_calls)
        self.assertEqual(1, session.physical_actions)
        self.assertEqual("awaiting_confirmation", session.status)
        self.assertEqual(3, session.task_graph.revision)
        self.assertEqual("continue-navigation", session.task_graph.active_subgoal_id)
        self.assertEqual(2, len(qwen.calls))
        self.assertEqual(3, qwen.calls[-1][1]["revision"])

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
                "awaiting_risk_confirmation",
            ):
                orchestrator.run_safe_loop(
                    session,
                    _risk_confirmation(session),
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
            "risk_ids": ["other-risk"],
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

    def test_policy_is_rechecked_immediately_before_execute(self) -> None:
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
        with tempfile.TemporaryDirectory() as temp:
            orchestrator, session, _planner, qwen, adapter = self._started(
                temp, policy=policy
            )

            with self.assertRaisesRegex(UniversalAgentOrchestratorError, "策略状态已变化"):
                orchestrator.confirm_one(session, _confirmation(session))

        self.assertEqual(2, policy.calls)
        self.assertEqual(0, adapter.execute_calls)

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
