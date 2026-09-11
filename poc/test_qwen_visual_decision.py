from __future__ import annotations
from dataclasses import replace
import unittest
from PIL import Image, ImageDraw
from agent.application.qwen_visual_decision import QwenVisualDecisionObserver
from agent.domain.qwen_task_context import QwenTaskContext
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


def action_payload(*, element_id: str, confidence: float = 0.91, role: str = "button",
    meaning: str = "目标控件", label: str = "目标控件") -> dict:
    return {
        "status": "action",
        "action": "tap_semantic",
        "target": {"element_id": element_id, "role": role, "meaning": meaning, "label": label,
            "evidence": [label or meaning]},
        "tap_point": [325, 260],
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
        self.context = QwenTaskContext(task_id="task_same_response_decision", device_id="device-local-01",
            revision=1, raw_goal="点击目标控件")
        self.context.validate()

    def decide(
        self,
        payload: dict,
        *,
        task_context: QwenTaskContext | None = None,
        trusted_observation=None,
        available_action_kinds: set[str] | None = None,
        launch_target: dict[str, str] | None = None,
    ):
        observer = QwenVisualDecisionObserver(
            StatusOnlyProvider(),
            trusted_observation_frame_validator=(
                validate_trusted_observation_against_frames
            ),
        )
        decision = observer.decide(
            frames=self.frames,
            task_context=task_context or self.context,
            trusted_observation=trusted_observation or self.observation,
            available_action_kinds=available_action_kinds or {"tap_semantic"},
            launch_target=launch_target,
            model_decision=payload,
        )
        return payload, observer, decision

    def wrong_app_target_case(self) -> tuple[QwenTaskContext, object]:
        context = QwenTaskContext(task_id="task-open-target-app", device_id="device-local-01",
            revision=1, raw_goal="打开目标应用")
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

    def test_model_named_element_binds_directly_to_current_scene_action(self) -> None:
        payload, observer, decision = self.decide(
            action_payload(element_id="target-button")
        )

        self.assertEqual("action", decision.proposal.status)
        self.assertEqual("tap_semantic", decision.proposal.action.action)
        self.assertEqual(
            "target-button", decision.proposal.action.params["element_id"]
        )
        self.assertEqual("target-button", payload["target"]["element_id"])
        self.assertEqual(0, observer.last_diagnostics["model_calls"])
        self.assertTrue(
            observer.last_diagnostics["decision_from_same_observation_response"]
        )
        self.assertEqual("direct_current_frame", observer.last_diagnostics["canonical_binding"])
        self.assertNotIn("canonical_choices", observer.last_diagnostics)
        self.assertNotIn("expected_result", decision.to_dict())
        self.assertFalse({"expected_effect", "formal_candidate_id", "formal_transition"}
            .intersection(decision.proposal.action.params))

    def test_local_code_rejects_geometry_in_strict_direct_target(self) -> None:
        payload = action_payload(element_id="missing-button")
        payload["target"]["bounds"] = [0.1, 0.2, 0.3, 0.4]
        with self.assertRaisesRegex(VisionAgentError, "decision.target不得携带几何"):
            self.decide(payload)

    def test_current_container_or_dialog_is_not_rejected_only_by_role(self) -> None:
        for role in ("container", "dialog"):
            with self.subTest(role=role):
                scene = replace(self.scene, elements=(replace(self.scene.elements[0], role=role),
                    self.scene.elements[1]))
                observation = build_trusted_observation(frames=self.frames, device_id="device-local-01",
                    scene=scene, observation_id=f"obs_{role}000000000000000000000000")
                _source, _observer, decision = self.decide(action_payload(element_id="target-button", role=role),
                    trusted_observation=observation)
                self.assertEqual(role, decision.proposal.action.params["role"])

    def test_overlapping_optional_element_does_not_rewrite_qwen_selected_element(self) -> None:
        selected = self.scene.elements[0]
        optional = replace(
            self.scene.elements[1],
            element_id="optional-overlap",
            bounds=selected.bounds,
            label="可选重叠说明",
            meaning="optional_context",
        )
        scene = replace(self.scene, elements=(selected, optional))
        observation = build_trusted_observation(
            frames=self.frames,
            device_id="device-local-01",
            scene=scene,
            observation_id="obs_44444444444444444444444444444444",
        )

        _source, _observer, decision = self.decide(
            action_payload(element_id=selected.element_id),
            trusted_observation=observation,
        )

        self.assertEqual(2, len(observation.scene.elements))
        self.assertEqual(selected.element_id, decision.proposal.action.params["element_id"])

    def test_scene_element_and_decision_confidence_are_diagnostic_only(self) -> None:
        low_scene = replace(
            self.scene,
            elements=tuple(replace(item, confidence=0.01) for item in self.scene.elements),
            confidence=0.01,
        )
        low_observation = build_trusted_observation(
            frames=self.frames,
            device_id="device-local-01",
            scene=low_scene,
            observation_id="obs_33333333333333333333333333333333",
        )
        _source, _observer, decision = self.decide(
            action_payload(element_id="target-button", confidence=0.12),
            trusted_observation=low_observation,
        )

        self.assertEqual("action", decision.proposal.status)
        self.assertFalse(hasattr(decision, "confidence"))

    def test_minimal_action_omits_unrelated_null_reason_and_confidence(self) -> None:
        _source, _observer, decision = self.decide(
            {
                "status": "action",
                "action": "tap_semantic",
                "target": {"element_id": "target-button", "role": "button", "meaning": "目标控件",
                    "label": "目标控件", "evidence": ["目标控件"]},
                "tap_point": [325, 260],
            }
        )

        self.assertEqual("action", decision.proposal.status)
        self.assertEqual("target-button", decision.proposal.action.params["element_id"])
        self.assertFalse(hasattr(decision, "confidence"))

    def test_finish_uses_only_current_scene_evidence(self) -> None:
        _source, observer, decision = self.decide(
            finish_payload(evidence_refs=["scene.summary", "element:target-button"])
        )

        self.assertEqual("finish", decision.proposal.status)
        self.assertIsNone(decision.proposal.action)
        self.assertEqual("当前截图已经直接证明本目标完成", decision.proposal.reason)
        self.assertNotIn("completion_evidence", decision.to_dict())
        self.assertEqual(1, observer.status()["model_finish_count"])

    def test_minimal_finish_omits_action_nulls_reason_and_confidence(self) -> None:
        _source, _observer, decision = self.decide(
            {
                "status": "finish",
                "evidence_refs": ["scene.summary"],
            }
        )

        self.assertEqual("finish", decision.proposal.status)
        self.assertIsNone(decision.proposal.action)
        self.assertFalse(hasattr(decision, "confidence"))
        self.assertTrue(decision.proposal.reason)

    def test_duplicate_finish_evidence_is_deduplicated(self) -> None:
        _source, _observer, decision = self.decide(
            finish_payload(
                evidence_refs=[
                    "scene.summary",
                    "scene.summary",
                    "element:target-button",
                    "element:target-button",
                ]
            )
        )

        self.assertEqual("当前截图已经直接证明本目标完成", decision.proposal.reason)

    def test_old_reference_strings_neither_authorize_nor_veto_finish(self) -> None:
        payload = finish_payload(evidence_refs=["history:old-frame"])
        _, _, decision = self.decide(payload)
        self.assertEqual(payload['reason'], decision.proposal.reason)
        self.assertNotIn('history:old-frame', decision.proposal.reason)

    def test_finish_is_not_vetoed_by_redundant_launch_registry_lineage(self) -> None:
        context, observation = self.wrong_app_target_case()
        conflicting_scene = replace(observation.scene, app_id="com.tencent.mm",
            summary="微信当前可见，系统设置尚未打开")
        conflicting_observation = build_trusted_observation(frames=self.frames,
            device_id="device-local-01", scene=conflicting_scene,
            observation_id="obs_11111111111111111111111111111111")

        _source, _observer, decision = self.decide(
            finish_payload(evidence_refs=["scene.summary"]), task_context=context,
            trusted_observation=conflicting_observation,
            available_action_kinds={"home", "launch_app"},
            launch_target={"launch_ref": "settings", "expected_app_id": "com.android.settings"})

        self.assertEqual("finish", decision.proposal.status)
        self.assertEqual("当前截图已经直接证明本目标完成", decision.proposal.reason)

    def test_finish_allows_matching_trusted_target_package(self) -> None:
        context, observation = self.wrong_app_target_case()
        matching_scene = replace(observation.scene, app_id="com.android.settings",
            summary="系统设置主页面当前可见")
        matching_observation = build_trusted_observation(frames=self.frames,
            device_id="device-local-01", scene=matching_scene,
            observation_id="obs_22222222222222222222222222222222")

        _source, _observer, decision = self.decide(finish_payload(evidence_refs=["scene.summary"]),
            task_context=context, trusted_observation=matching_observation,
            available_action_kinds={"home", "launch_app"},
            launch_target={"launch_ref": "settings", "expected_app_id": "com.android.settings"})
        self.assertEqual("finish", decision.proposal.status)

    def test_neutral_choice_metadata_does_not_veto_a_valid_action(self) -> None:
        payload = {
            "status": "action",
            "action": "tap_semantic",
            "target": {"element_id": "target-button", "role": "button", "meaning": "目标控件",
                "label": "目标控件", "evidence": ["目标控件"]},
            "tap_point": [325, 260],
            "choice_id": "qwen-observation-local-note",
        }

        _source, _observer, decision = self.decide(payload)

        self.assertEqual("action", decision.proposal.status)
        self.assertEqual("target-button", decision.proposal.action.params["element_id"])

    def test_optional_action_evidence_refs_do_not_veto_or_become_completion(self) -> None:
        payload = action_payload(element_id="target-button")
        payload["evidence_refs"] = ["scene.summary", "element:target-button"]

        _source, _observer, decision = self.decide(payload)

        self.assertEqual("action", decision.proposal.status)
        self.assertNotIn("completion_evidence", decision.to_dict())
        self.assertEqual("target-button", decision.proposal.action.params["element_id"])

    def test_multi_action_plan_and_raw_coordinates_are_rejected(self) -> None:
        injected_fields = {
            "actions": [{"action": "tap_semantic", "element_id": "target-button"}],
            "plan": ["点击目标控件", "再点击其它控件"],
            "coordinates": [120, 240],
            "metadata": {"execution_plan": ["点击目标控件", "再点击其它控件"]},
        }
        for field, value in injected_fields.items():
            with self.subTest(field=field), self.assertRaisesRegex(
                VisionAgentError, "多动作、计划或裸坐标"
            ):
                payload = {
                    "status": "action",
                    "action": "tap_semantic",
                    "element_id": "target-button",
                    field: value,
                }
                self.decide(payload)

    def test_cross_variant_references_are_rejected(self) -> None:
        invalid_payloads = (
            (
                {
                    "status": "action",
                    "action": "tap_semantic",
                    "target": {"element_id": "target-button", "role": "button", "meaning": "目标控件",
                        "label": "目标控件", "evidence": ["目标控件"]},
                    "direction": "down",
                    "tap_point": [325, 260],
                },
                "点按动作必须且只能使用decision.target",
            ),
            (
                {
                    "status": "action",
                    "action": "home",
                    "element_id": "target-button",
                },
                "系统动作不得携带元素或方向字段",
            ),
            (
                {
                    "status": "finish",
                    "action": "tap_semantic",
                    "element_id": "target-button",
                    "evidence_refs": ["scene.summary"],
                },
                "finish必须陈述当前截图完成事实",
            ),
        )
        for payload, message in invalid_payloads:
            with self.subTest(payload=payload), self.assertRaisesRegex(
                VisionAgentError, message
            ):
                self.decide(payload)

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
        self.assertEqual(["home"], observer.last_diagnostics["device_action_kinds"])
        self.assertNotIn("canonical_choices", observer.last_diagnostics)

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
