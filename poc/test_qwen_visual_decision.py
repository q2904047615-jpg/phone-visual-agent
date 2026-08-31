from __future__ import annotations

from dataclasses import replace
import unittest

from PIL import Image, ImageDraw

from agent.application.qwen_visual_decision import QwenVisualDecisionObserver
from agent.domain.qwen_task_context import QwenTaskContext
from agent.domain.task_graph import (
    CompletionCondition,
    DynamicTaskGraph,
    GraphGoal,
    Subgoal,
    TargetApp,
    build_exact_action_task_graph,
)
from agent.domain.task_semantic_ir import compile_formal_semantic_authority
from agent.domain.ui_scene import UIElement, UIScene
from agent.domain.vision_model import VisionAgentError
from agent.infrastructure.observation_images import local_frame_fingerprint
from agent.infrastructure.trusted_observation_frames import (
    build_trusted_observation,
    validate_trusted_observation_against_frames,
)


def patterned_frames() -> list[Image.Image]:
    image = Image.new("RGB", (240, 480), "black")
    draw = ImageDraw.Draw(image)
    for y in range(0, 480, 12):
        for x in range(0, 240, 12):
            if (x // 12 + y // 12) % 2:
                draw.rectangle((x, y, x + 5, y + 5), fill="white")
    return [image.copy() for _ in range(4)]


def action_payload(*, element_id: str, confidence: float = 0.91) -> dict:
    return {
        "status": "action",
        "action": "tap_semantic",
        "element_id": element_id,
        "source_element_id": None,
        "destination_element_id": None,
        "direction": None,
        "evidence_refs": [],
        "confidence": confidence,
        "reason": "当前画面中的目标控件与用户目标精确对应",
    }


def finish_payload(*, evidence_refs: list[str]) -> dict:
    return {
        "status": "finish",
        "action": None,
        "element_id": None,
        "source_element_id": None,
        "destination_element_id": None,
        "direction": None,
        "evidence_refs": evidence_refs,
        "confidence": 0.93,
        "reason": "当前截图已经直接证明本目标完成",
    }


def system_action_payload(*, action: str) -> dict:
    return {
        "status": "action",
        "action": action,
        "element_id": None,
        "source_element_id": None,
        "destination_element_id": None,
        "direction": None,
        "evidence_refs": [],
        "confidence": 0.92,
        "reason": "当前前台应用不是目标应用，先返回主屏幕继续寻找目标入口",
    }


def legacy_blocked_payload(*, reason: str = "当前画面没有目标应用入口") -> dict:
    return {
        "status": "blocked",
        "action": None,
        "element_id": None,
        "source_element_id": None,
        "destination_element_id": None,
        "direction": None,
        "evidence_refs": [],
        "confidence": 0.88,
        "reason": reason,
    }


class SameResponseDecisionSource:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.fingerprints: list[str] = []

    def decision_for(self, fingerprint: str) -> dict:
        self.fingerprints.append(fingerprint)
        return dict(self.payload)


class StatusOnlyProvider:
    configured = True

    def status(self) -> dict:
        return {"configured": True, "model": "offline-qwen"}


class QwenSameResponseDecisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frames = patterned_frames()
        fingerprint = local_frame_fingerprint(self.frames[-1])
        self.scene = UIScene(
            app_id="sample.app",
            screen_id="sample_home",
            summary="页面显示目标控件和一个无关控件",
            elements=(
                UIElement(
                    element_id="target-button",
                    role="button",
                    meaning="目标控件",
                    label="目标控件",
                    bounds=(0.10, 0.20, 0.55, 0.32),
                    confidence=0.97,
                    states={"fully_visible": True, "enabled": True},
                    evidence=("目标控件文字清晰可见",),
                ),
                UIElement(
                    element_id="other-button",
                    role="button",
                    meaning="无关控件",
                    label="无关控件",
                    bounds=(0.10, 0.40, 0.55, 0.52),
                    confidence=0.98,
                    states={"fully_visible": True, "enabled": True},
                    evidence=("无关控件文字清晰可见",),
                ),
            ),
            stable=True,
            confidence=0.98,
            fingerprint=fingerprint,
        )
        self.observation = build_trusted_observation(
            frames=self.frames,
            device_id="device-local-01",
            scene=self.scene,
            observation_id="obs_0123456789abcdef0123456789abcdef",
        )
        graph = build_exact_action_task_graph(
            "点击目标控件",
            action_kind="tap_semantic",
            target_label="目标控件",
            device_id="device-local-01",
            task_id="task_same_response_decision",
        )
        graph = replace(graph, status="running")
        graph.validate()
        context = QwenTaskContext.from_dict(graph.to_qwen_context())
        self.context = replace(
            context,
            semantic_ir=compile_formal_semantic_authority(graph).semantic_ir,
        )
        self.context.validate()

    def decide(
        self,
        payload: dict,
        *,
        task_context: QwenTaskContext | None = None,
        trusted_observation=None,
        available_action_kinds: set[str] | None = None,
    ):
        source = SameResponseDecisionSource(payload)
        observer = QwenVisualDecisionObserver(
            StatusOnlyProvider(),
            decision_source=source,
            trusted_observation_frame_validator=(
                validate_trusted_observation_against_frames
            ),
        )
        decision = observer.decide(
            frames=self.frames,
            task_context=task_context or self.context,
            trusted_observation=trusted_observation or self.observation,
            available_action_kinds=available_action_kinds or {"tap_semantic"},
        )
        return source, observer, decision

    def wrong_app_target_case(self) -> tuple[QwenTaskContext, object]:
        graph = DynamicTaskGraph(
            task_id="task-open-target-app",
            device_id="device-local-01",
            revision=1,
            status="running",
            goal=GraphGoal(
                objective="打开目标应用",
                target_apps=(
                    TargetApp(app_id="target_app", app_name="目标应用"),
                ),
                entities={"target_ui_label": "目标应用"},
            ),
            constraints=(),
            completion_conditions=(
                CompletionCondition(
                    condition_id="goal-complete",
                    description="目标应用已在前台",
                    evidence_required=("目标应用前台画面",),
                ),
            ),
            risk_actions=(),
            subgoals=(
                Subgoal(
                    subgoal_id="open-target-app",
                    objective="打开目标应用",
                    status="active",
                    depends_on=(),
                    constraints=(),
                    completion_conditions=("目标应用已在前台",),
                    completion_evidence=(),
                    risk_action_ids=(),
                    external_impact="navigation_only",
                ),
            ),
            active_subgoal_id="open-target-app",
            raw_user_goal="打开目标应用",
        )
        graph.validate()
        context = QwenTaskContext.from_dict(graph.to_qwen_context())
        context = replace(
            context,
            semantic_ir=compile_formal_semantic_authority(graph).semantic_ir,
        )
        context.validate()
        scene = UIScene(
            app_id="other.app",
            screen_id="other_main",
            summary="当前显示另一个应用的主页面，画面内没有目标应用入口",
            elements=(),
            stable=True,
            confidence=0.98,
            fingerprint=self.scene.fingerprint,
        )
        observation = build_trusted_observation(
            frames=self.frames,
            device_id="device-local-01",
            scene=scene,
            observation_id="obs_fedcba9876543210fedcba9876543210",
        )
        return context, observation

    def test_model_named_element_maps_to_exactly_one_canonical_action(self) -> None:
        source, observer, decision = self.decide(
            action_payload(element_id="target-button")
        )

        self.assertEqual("action", decision.proposal.status)
        self.assertEqual("tap_semantic", decision.proposal.action.action)
        self.assertEqual(
            "target-button", decision.proposal.action.params["element_id"]
        )
        self.assertEqual([self.observation.fingerprint], source.fingerprints)
        self.assertEqual(0, observer.last_diagnostics["model_calls"])
        self.assertTrue(
            observer.last_diagnostics["decision_from_same_observation_response"]
        )

    def test_local_code_does_not_fallback_when_model_reference_is_wrong(self) -> None:
        with self.assertRaisesRegex(
            VisionAgentError, "未精确映射唯一canonical candidate"
        ):
            self.decide(action_payload(element_id="missing-button"))

    def test_low_confidence_is_reported_but_does_not_create_a_second_veto(self) -> None:
        _source, _observer, decision = self.decide(
            action_payload(element_id="target-button", confidence=0.12)
        )

        self.assertEqual("action", decision.proposal.status)
        self.assertAlmostEqual(0.12, decision.confidence)

    def test_finish_uses_only_current_scene_evidence(self) -> None:
        _source, observer, decision = self.decide(
            finish_payload(evidence_refs=["scene.summary", "element:target-button"])
        )

        self.assertEqual("finish", decision.proposal.status)
        self.assertIsNone(decision.proposal.action)
        self.assertEqual(
            (
                "页面显示目标控件和一个无关控件",
                "目标控件文字清晰可见",
            ),
            decision.completion_evidence,
        )
        self.assertEqual(1, observer.status()["model_finish_count"])

    def test_finish_rejects_evidence_outside_current_scene(self) -> None:
        with self.assertRaisesRegex(VisionAgentError, "未知scene证据"):
            self.decide(finish_payload(evidence_refs=["history:old-frame"]))

    def test_decision_wire_shape_is_exact(self) -> None:
        malformed = action_payload(element_id="target-button")
        malformed["choice_id"] = "legacy-local-selector"

        with self.assertRaisesRegex(VisionAgentError, "协议外字段"):
            self.decide(malformed)

    def test_model_home_choice_maps_when_target_app_is_not_foreground(self) -> None:
        context, observation = self.wrong_app_target_case()

        _source, observer, decision = self.decide(
            system_action_payload(action="home"),
            task_context=context,
            trusted_observation=observation,
            available_action_kinds={"home"},
        )

        self.assertEqual("action", decision.proposal.status)
        self.assertEqual("home", decision.proposal.action.action)
        self.assertEqual(
            ["home"],
            [item["action"] for item in observer.last_diagnostics["canonical_choices"]],
        )

    def test_obsolete_model_blocked_responses_are_rejected(self) -> None:
        context, observation = self.wrong_app_target_case()
        historical_reasons = (
            "当前画面没有目标应用入口，无法继续。",
            "home只能返回主屏幕，不能直接进入设置。",
        )
        for reason in historical_reasons:
            with self.subTest(reason=reason), self.assertRaisesRegex(
                VisionAgentError, "只允许action或finish"
            ):
                self.decide(
                    legacy_blocked_payload(reason=reason),
                    task_context=context,
                    trusted_observation=observation,
                    available_action_kinds={"home"},
                )


if __name__ == "__main__":
    unittest.main()
