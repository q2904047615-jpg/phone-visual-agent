from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from PIL import Image

from generic_scene_observer import _local_frame_fingerprint
from generic_step_planner import GenericStepPlanningError
from qwen_visual_decision import (
    QWEN_VISUAL_DECISION_PROTOCOL_VERSION,
    QwenTaskContext,
    QwenVisualDecisionObserver,
    TrustedObservation,
    _decision_retry_prompt,
)
from ui_scene import UIElement, UIScene, UISceneError
from vision_agent import VisionAgentError


ROOT = Path(__file__).resolve().parent
ASSET_ROOT = ROOT / "evals" / "qwen_visual_decision" / "images"
REPLAY_ROOT = ROOT / "evals" / "vision_replay" / "images"


class FakeProvider:
    configured = True

    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls = 0
        self.messages: list[dict] = []

    def status(self) -> dict:
        return {"configured": True, "model": "fake-qwen"}

    def _chat(self, messages, max_tokens, **kwargs) -> str:
        self.calls += 1
        self.messages = messages
        self.last_call_options = dict(kwargs)
        return json.dumps(self.payload, ensure_ascii=False)


class SequenceProvider(FakeProvider):
    def __init__(self, payloads: list[dict | BaseException]) -> None:
        super().__init__({})
        self.payloads = list(payloads)

    def _chat(self, messages, max_tokens, **kwargs) -> str:
        self.calls += 1
        self.messages = messages
        if not self.payloads:
            raise AssertionError("模型被调用超过一次初始请求和一次修复重试")
        value = self.payloads.pop(0)
        if isinstance(value, BaseException):
            raise value
        return json.dumps(value, ensure_ascii=False)


def load_sequence(name: str) -> list[Image.Image]:
    paths = sorted((ASSET_ROOT / name).glob("frame_*.jpg"))
    if len(paths) != 4:
        raise AssertionError(f"真实四帧测试序列缺失：{name}")
    frames: list[Image.Image] = []
    for path in paths:
        with Image.open(path) as image:
            frames.append(image.convert("RGB"))
    return frames


def load_replay_image(name: str) -> Image.Image:
    with Image.open(REPLAY_ROOT / name) as image:
        return image.convert("RGB")


def repeated_frames(path: Path) -> list[Image.Image]:
    with Image.open(path) as image:
        frame = image.convert("RGB")
    return [frame.copy() for _ in range(4)]


def launcher_elements() -> tuple[UIElement, ...]:
    return (
        UIElement(
            element_id="settings_icon",
            role="icon",
            meaning="open_settings",
            label="设置",
            bounds=(0.68, 0.20, 0.86, 0.35),
            confidence=0.96,
            states={"goal_relevant": True},
            evidence=("设置",),
        ),
        UIElement(
            element_id="unlabelled_camera_icon",
            role="icon",
            meaning="open_camera",
            label="",
            bounds=(0.69, 0.82, 0.88, 0.95),
            confidence=0.92,
            states={"goal_relevant": False},
            evidence=("相机图形",),
        ),
    )


def overlay_elements() -> tuple[UIElement, ...]:
    return (
        UIElement(
            element_id="input_value",
            role="input",
            meaning="current_text_input",
            label=".com",
            bounds=(0.05, 0.62, 0.78, 0.72),
            confidence=0.94,
            states={"focused": True},
            evidence=(".com",),
        ),
        UIElement(
            element_id="unlabelled_close_icon",
            role="icon",
            meaning="close_top_overlay",
            label="",
            bounds=(0.90, 0.08, 0.98, 0.14),
            confidence=0.91,
            states={"overlay_control": True},
            evidence=("关闭图形",),
        ),
        UIElement(
            element_id="text_submit_button",
            role="button",
            meaning="submit_current_content",
            label="发布",
            bounds=(0.81, 0.62, 0.97, 0.72),
            confidence=0.96,
            states={"enabled": True},
            evidence=("发布",),
        ),
    )


def settings_list_elements() -> tuple[UIElement, ...]:
    return (
        UIElement(
            element_id="visible_settings_list",
            role="container",
            meaning="scrollable_settings_list",
            label="",
            bounds=(0.08, 0.22, 0.92, 0.92),
            confidence=0.95,
            states={"scrollable": True},
            evidence=("多个纵向列表项",),
        ),
        UIElement(
            element_id="visible_text_item",
            role="list_item",
            meaning="open_device_information",
            label="我的设备",
            bounds=(0.16, 0.45, 0.88, 0.56),
            confidence=0.96,
            states={},
            evidence=("我的设备",),
        ),
    )


def scene_for(
    frames: list[Image.Image],
    *,
    elements: tuple[UIElement, ...] | None = None,
    screen_id: str = "launcher_home",
    app_id: str = "launcher",
    summary: str = "桌面应用网格清晰可见",
    overlays: tuple[str, ...] = (),
) -> UIScene:
    sharpness = []
    from observation_images import measure_frame_sharpness

    for frame in frames:
        sharpness.append(measure_frame_sharpness(frame))
    selected = max(range(len(frames)), key=sharpness.__getitem__)
    return UIScene(
        app_id=app_id,
        screen_id=screen_id,
        summary=summary,
        elements=elements if elements is not None else launcher_elements(),
        overlays=overlays,
        stable=True,
        confidence=0.95,
        fingerprint=_local_frame_fingerprint(frames[selected]),
    )


def trusted_observation(
    frames: list[Image.Image],
    *,
    elements: tuple[UIElement, ...] | None = None,
    scene: UIScene | None = None,
    observation_id: str = "obs_0123456789abcdef0123456789abcdef",
) -> TrustedObservation:
    return TrustedObservation.from_scene(
        frames=frames,
        device_id="offline_phone_01",
        scene=scene or scene_for(frames, elements=elements),
        observation_id=observation_id,
    )


def task_context(
    *,
    task_id: str = "task_offline_01",
    revision: int = 3,
    external: bool = False,
    confirmed: bool = False,
) -> dict:
    risk_ids = ["risk_send"] if external else []
    impact = "external_state" if external else "navigation_only"
    return {
        "protocol_version": "2026-08-11-deepseek-task-graph-v3",
        "task_id": task_id,
        "device_id": "offline_phone_01",
        "revision": revision,
        "task_status": "awaiting_confirmation" if external else "running",
        "goal": {
            "objective": "打开当前目标页面" if not external else "提交当前内容",
            "target_apps": [{"app_id": "generic", "app_name": "目标应用"}],
            "entities": {},
        },
        "global_constraints": ["每轮只允许一个动作", "看不清时停止"],
        "goal_completion_conditions": [
            {
                "condition_id": "visible_result",
                "description": "目标结果清晰可见",
                "evidence_required": ["当前画面证据"],
                "satisfied": False,
                "evidence": [],
            }
        ],
        "current_subgoal": {
            "subgoal_id": "current_target",
            "objective": "打开设置" if not external else "提交当前内容",
            "status": "active",
            "depends_on": [],
            "constraints": ["只使用当前画面中的可信控件"],
            "completion_conditions": ["目标页面可见"],
            "completion_evidence": [],
            "risk_action_ids": risk_ids,
            "external_impact": impact,
        },
        "current_external_impact": impact,
        "risk_actions": (
            [
                {
                    "risk_id": "risk_send",
                    "description": "提交将改变外部状态",
                    "external_effect": "内容会被提交",
                    "risk_type": "data_mutation",
                    "risk_level": "high",
                    "subgoal_ids": ["current_target"],
                    "confirmation_required": True,
                }
            ]
            if external
            else []
        ),
        "confirmation_gate": {
            "required": external,
            "state": "confirmed" if confirmed else "awaiting_confirmation" if external else "not_required",
            "risk_ids": risk_ids,
            "scope": {
                "task_id": task_id,
                "device_id": "offline_phone_01",
                "revision": revision,
                "subgoal_id": "current_target",
            },
            "external_state_action_allowed": bool(external and confirmed),
        },
    }


def action_payload(
    context: dict,
    observation: TrustedObservation,
    *,
    element_id: str = "settings_icon",
) -> dict:
    element = observation.get_candidate(element_id)
    return {
        "protocol_version": QWEN_VISUAL_DECISION_PROTOCOL_VERSION,
        "task_id": context["task_id"],
        "device_id": context["device_id"],
        "revision": context["revision"],
        "observation_id": observation.observation_id,
        "fingerprint": observation.fingerprint,
        "page_state": {
            "foreground_app_id": observation.scene.foreground_app_id,
            "screen_id": observation.scene.screen_id,
            "summary": observation.scene.summary,
            "overlays": list(observation.scene.overlays),
        },
        "status": "action",
        "next_action": {
            "kind": "tap_semantic",
            "element_id": element.element_id,
            "target": element.meaning,
            "role": element.role,
            "label": element.label,
            "states": dict(element.states),
        },
        "target_region": {
            "kind": "element",
            "element_id": element.element_id,
            "bounds": [round(item * 1000) for item in element.bounds],
            "description": element.label or element.meaning,
        },
        "expected_result": {"scene_changed": True},
        "confidence": 0.92,
        "reason": "可信候选唯一且清晰。",
        "completion_evidence_element_ids": [],
    }


def blocked_payload(context: dict, observation: TrustedObservation) -> dict:
    value = action_payload(context, observation)
    value.update(
        {
            "status": "blocked",
            "next_action": None,
            "target_region": None,
            "expected_result": {},
            "confidence": 0.4,
            "reason": "没有可靠且唯一的可信候选。",
        }
    )
    return value


class QwenVisualDecisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frames = load_sequence("launcher_stable")
        self.context = task_context()
        self.observation = trusted_observation(self.frames)

    def decide(self, provider, *, context=None, frames=None, observation=None):
        observer = QwenVisualDecisionObserver(provider)
        decision = observer.decide(
            frames=frames or self.frames,
            task_context=context or self.context,
            trusted_observation=observation or self.observation,
        )
        return observer, decision

    def test_real_stable_sequence_uses_four_distinct_frames(self) -> None:
        fingerprints = {hash(frame.tobytes()) for frame in self.frames}
        self.assertEqual(len(fingerprints), 4)
        self.assertTrue(self.observation.local_stability.stable)
        self.assertEqual(self.observation.local_stability.frame_count, 4)
        self.assertEqual(self.observation.scene.fingerprint, self.observation.fingerprint)

    def test_input_action_must_copy_structured_text_exactly(self) -> None:
        context = task_context()
        context["goal"]["entities"] = {"input_text": "蓝牙设置"}
        context["current_subgoal"]["objective"] = "在已聚焦输入框输入查询词"
        field = UIElement(
            element_id="query_field",
            role="input",
            meaning="搜索输入框",
            label="搜索",
            bounds=(0.08, 0.12, 0.92, 0.22),
            confidence=0.97,
            states={"focused": True},
            evidence=("输入光标可见",),
        )
        observation = trusted_observation(self.frames, elements=(field,))
        payload = action_payload(context, observation, element_id="query_field")
        payload["next_action"].update(
            {"kind": "input_verified_text", "text": "蓝牙设置"}
        )
        payload["expected_result"] = {"content_changed": True}

        _observer, decision = self.decide(
            FakeProvider(payload),
            context=context,
            observation=observation,
        )

        self.assertEqual("input_verified_text", decision.proposal.action.action)
        self.assertEqual("蓝牙设置", decision.proposal.action.params["text"])

    def test_input_action_rejects_model_invented_text(self) -> None:
        context = task_context()
        context["goal"]["entities"] = {"input_text": "蓝牙设置"}
        field = UIElement(
            element_id="query_field",
            role="input",
            meaning="搜索输入框",
            label="搜索",
            bounds=(0.08, 0.12, 0.92, 0.22),
            confidence=0.97,
            states={"focused": True},
            evidence=("输入光标可见",),
        )
        observation = trusted_observation(self.frames, elements=(field,))
        payload = action_payload(context, observation, element_id="query_field")
        payload["next_action"].update(
            {"kind": "input_verified_text", "text": "打开蓝牙设置"}
        )

        observer, decision = self.decide(
            FakeProvider(payload),
            context=context,
            observation=observation,
        )

        self.assertEqual("blocked", decision.proposal.status)
        self.assertIn("input_text", decision.proposal.reason)
        self.assertTrue(observer.last_diagnostics["retry_failure_blocked"])

    def test_offline_manifest_uses_full_context_and_multiple_page_types(self) -> None:
        manifest = json.loads(
            (ROOT / "evals" / "qwen_visual_decision" / "cases.json").read_text(
                encoding="utf-8"
            )
        )
        page_types = {str(case["page_type"]) for case in manifest["cases"]}
        self.assertGreaterEqual(len(page_types), 3)
        first_frames = manifest["cases"][0]["frames"]
        self.assertEqual(len(first_frames), 4)
        self.assertEqual(len(set(first_frames)), 4)
        for case in manifest["cases"]:
            parsed = QwenTaskContext.from_dict(case["task_context"])
            self.assertEqual(
                parsed.protocol_version,
                "2026-08-11-deepseek-task-graph-v3",
            )
            self.assertEqual(parsed.device_id, "offline_phone_01")
            for field in (
                "protocol_version",
                "task_id",
                "device_id",
                "revision",
                "current_subgoal",
                "global_constraints",
                "current_external_impact",
                "risk_actions",
                "confirmation_gate",
            ):
                self.assertIn(field, case["task_context"])
            scope = case["task_context"]["confirmation_gate"]["scope"]
            self.assertEqual(
                set(scope),
                {"task_id", "device_id", "revision", "subgoal_id"},
            )
            self.assertEqual(scope["task_id"], case["task_context"]["task_id"])
            self.assertEqual(scope["device_id"], case["task_context"]["device_id"])
            self.assertEqual(scope["revision"], case["task_context"]["revision"])
            self.assertEqual(
                scope["subgoal_id"],
                case["task_context"]["current_subgoal"]["subgoal_id"],
            )

    def test_action_binds_to_preexisting_trusted_candidate(self) -> None:
        provider = FakeProvider(action_payload(self.context, self.observation))
        _observer, decision = self.decide(provider)
        self.assertEqual(provider.last_call_options["max_attempts"], 2)
        self.assertEqual(decision.proposal.action.params["element_id"], "settings_icon")
        self.assertEqual(
            decision.target_region.bounds,
            self.observation.get_candidate("settings_icon").bounds,
        )
        self.assertIs(decision.trusted_observation, self.observation)
        self.assertNotIn("elements", decision.page_state.to_dict())

    def test_action_semantic_fields_are_canonicalized_from_trusted_candidate(self) -> None:
        payload = action_payload(self.context, self.observation)
        payload["next_action"].update(
            {
                "target": "模型近义词",
                "role": "模型角色",
                "label": "模型标签",
                "states": {"invented": True},
            }
        )

        _observer, decision = self.decide(FakeProvider(payload))

        candidate = self.observation.get_candidate("settings_icon")
        params = decision.proposal.action.params
        self.assertEqual(candidate.meaning, params["target"])
        self.assertEqual(candidate.role, params["role"])
        self.assertEqual(candidate.label, params["label"])
        self.assertEqual(candidate.states, params["states"])

    def test_prompt_uses_documented_thousand_scale_without_mutating_candidate(self) -> None:
        original_bounds = self.observation.get_candidate("settings_icon").bounds
        prompt = self.observation.prompt_dict()
        candidate = next(
            item for item in prompt["candidates"] if item["element_id"] == "settings_icon"
        )
        self.assertEqual(prompt["candidate_bounds_scale"], 1000)
        self.assertEqual(candidate["bounds"], [680, 200, 860, 350])
        self.assertEqual(
            self.observation.get_candidate("settings_icon").bounds,
            original_bounds,
        )

    def test_observation_context_is_small_but_decision_context_remains_complete(self) -> None:
        parsed = QwenTaskContext.from_dict(self.context)
        observation_context = parsed.to_observation_context()
        self.assertEqual(observation_context["device_id"], self.context["device_id"])
        self.assertEqual(
            observation_context["objective"],
            self.context["current_subgoal"]["objective"],
        )
        self.assertIn("constraints", observation_context)
        self.assertNotIn("task_id", observation_context)
        for field in (
            "protocol_version",
            "task_id",
            "device_id",
            "revision",
            "current_subgoal",
            "global_constraints",
            "current_external_impact",
            "risk_actions",
            "confirmation_gate",
        ):
            self.assertIn(field, parsed.to_dict())

    def test_forged_mars_element_and_self_authored_page_state_are_rejected(self) -> None:
        forged = action_payload(self.context, self.observation)
        forged["page_state"]["elements"] = [
            {
                "element_id": "mars_entry",
                "role": "button",
                "meaning": "open_mars",
                "label": "火星入口",
                "bounds": [100, 100, 300, 200],
                "confidence": 0.99,
            }
        ]
        forged["next_action"].update(
            {
                "element_id": "mars_entry",
                "target": "open_mars",
                "role": "button",
                "label": "火星入口",
            }
        )
        forged["target_region"].update(
            {"element_id": "mars_entry", "bounds": [100, 100, 300, 200]}
        )
        provider = SequenceProvider([forged, forged])
        observer = QwenVisualDecisionObserver(provider)
        decision = observer.decide(
            frames=self.frames,
            task_context=self.context,
            trusted_observation=self.observation,
        )
        self.assertEqual(provider.calls, 2)
        self.assertEqual(decision.proposal.status, "blocked")
        self.assertIn("禁止携带候选元素", decision.reason)
        self.assertTrue(observer.last_diagnostics["first_output_rejected"])
        self.assertFalse(
            observer.last_diagnostics["candidate_action_from_first_output"]
        )
        self.assertTrue(observer.last_diagnostics["retry_failure_blocked"])
        self.assertEqual(observer.status()["final_blocked_rate"], 1.0)

    def test_forged_element_id_without_page_elements_is_rejected(self) -> None:
        forged = action_payload(self.context, self.observation)
        forged["next_action"]["element_id"] = "mars_entry"
        forged["target_region"]["element_id"] = "mars_entry"
        provider = SequenceProvider([forged, forged])
        _observer, decision = self.decide(provider)
        self.assertEqual(decision.proposal.status, "blocked")
        self.assertIn("不存在元素", decision.reason)

    def test_output_task_revision_and_fingerprint_must_match(self) -> None:
        for field, replacement in (
            ("task_id", "task_old"),
            ("revision", 2),
            ("fingerprint", "old_fingerprint"),
            ("observation_id", "obs_ffffffffffffffffffffffffffffffff"),
        ):
            with self.subTest(field=field):
                bad = action_payload(self.context, self.observation)
                bad[field] = replacement
                provider = SequenceProvider([bad, bad])
                _observer, decision = self.decide(provider)
                self.assertEqual(decision.proposal.status, "blocked")
                self.assertIn("不匹配或已过期", decision.reason)

    def test_old_decision_rejected_after_task_revision_changes(self) -> None:
        _observer, decision = self.decide(
            FakeProvider(action_payload(self.context, self.observation))
        )
        newer = task_context(revision=4)
        with self.assertRaisesRegex(GenericStepPlanningError, "已过期或不匹配"):
            decision.validate_fresh(QwenTaskContext.from_dict(newer), self.observation)

    def test_old_decision_rejected_after_fingerprint_changes(self) -> None:
        _observer, decision = self.decide(
            FakeProvider(action_payload(self.context, self.observation))
        )
        new_frames = load_sequence("launcher_changed")
        new_observation = trusted_observation(
            new_frames,
            observation_id="obs_fedcba9876543210fedcba9876543210",
        )
        with self.assertRaisesRegex(GenericStepPlanningError, "已过期或不匹配"):
            decision.validate_fresh(self.context, new_observation)

    def test_confirmation_gate_blocks_before_qwen_call(self) -> None:
        context = task_context(external=True, confirmed=False)
        provider = FakeProvider(action_payload(task_context(), self.observation))
        observer, decision = self.decide(provider, context=context)
        self.assertEqual(provider.calls, 0)
        self.assertEqual(decision.proposal.status, "blocked")
        self.assertIn("确认门未满足", decision.reason)
        self.assertEqual(
            observer.last_diagnostics["local_safety_block"], "confirmation_gate"
        )
        self.assertEqual(observer.status()["final_blocked_rate"], 1.0)

    def test_missing_or_ambiguous_exact_text_blocks_before_qwen(self) -> None:
        missing_context = task_context(task_id="task_exact_missing", revision=12)
        missing_context["goal"]["entities"] = {"expected_text": "火星入口"}
        missing_provider = FakeProvider(action_payload(self.context, self.observation))
        missing_observer, missing_decision = self.decide(
            missing_provider,
            context=missing_context,
        )
        self.assertEqual(missing_provider.calls, 0)
        self.assertEqual(missing_decision.proposal.status, "blocked")
        self.assertEqual(
            missing_observer.last_diagnostics["local_safety_block"],
            "exact_text_missing",
        )

        duplicate_elements = (
            UIElement(
                element_id="duplicate_1",
                role="button",
                meaning="confirm_first",
                label="确定",
                bounds=(0.1, 0.2, 0.3, 0.3),
                confidence=0.95,
            ),
            UIElement(
                element_id="duplicate_2",
                role="button",
                meaning="confirm_second",
                label="确定",
                bounds=(0.6, 0.2, 0.8, 0.3),
                confidence=0.95,
            ),
        )
        duplicate_observation = trusted_observation(
            self.frames,
            elements=duplicate_elements,
            observation_id="obs_33333333333333333333333333333333",
        )
        ambiguous_context = task_context(task_id="task_exact_ambiguous", revision=13)
        ambiguous_context["goal"]["entities"] = {"exact_text": "确定"}
        ambiguous_provider = FakeProvider(
            action_payload(
                ambiguous_context,
                duplicate_observation,
                element_id="duplicate_1",
            )
        )
        observer = QwenVisualDecisionObserver(ambiguous_provider)
        decision = observer.decide(
            frames=self.frames,
            task_context=ambiguous_context,
            trusted_observation=duplicate_observation,
        )
        self.assertEqual(ambiguous_provider.calls, 0)
        self.assertEqual(decision.proposal.status, "blocked")
        self.assertEqual(
            observer.last_diagnostics["local_safety_block"],
            "exact_text_ambiguous",
        )

    def test_read_only_exact_text_uses_structured_role_for_completion(self) -> None:
        elements = (
            UIElement(
                element_id="input_value",
                role="input",
                meaning="current_text_input",
                label=".com",
                bounds=(0.05, 0.55, 0.75, 0.65),
                confidence=0.96,
                evidence=(".com",),
            ),
            UIElement(
                element_id="candidate_value",
                role="keyboard_key",
                meaning="candidate_shortcut",
                label=".com",
                bounds=(0.10, 0.70, 0.28, 0.77),
                confidence=0.93,
                evidence=(".com",),
            ),
            UIElement(
                element_id="preview_text",
                role="text",
                meaning="input_preview",
                label=".com",
                bounds=(0.35, 0.70, 0.53, 0.77),
                confidence=0.91,
                evidence=(".com",),
            ),
        )
        scene = scene_for(
            self.frames,
            elements=elements,
            app_id="generic_surface",
            screen_id="text_entry",
            summary="输入框与输入辅助区域清晰可见",
        )
        observation = trusted_observation(
            self.frames,
            scene=scene,
            observation_id="obs_44444444444444444444444444444444",
        )
        context = task_context(task_id="task_verify_input", revision=15)
        context["current_external_impact"] = "read_only"
        context["current_subgoal"]["external_impact"] = "read_only"
        context["goal"]["entities"] = {
            "expected_text": ".com",
            "expected_role": "input",
        }
        payload = action_payload(context, observation, element_id="input_value")
        payload.update(
            {
                "status": "finished",
                "next_action": None,
                "target_region": None,
                "expected_result": {},
                "completion_evidence_element_ids": ["input_value"],
            }
        )
        observer = QwenVisualDecisionObserver(FakeProvider(payload))
        decision = observer.decide(
            frames=self.frames,
            task_context=context,
            trusted_observation=observation,
        )
        self.assertEqual(decision.proposal.status, "finished")
        self.assertEqual(decision.proposal.completion_evidence, ("input_value:.com",))

    def test_same_visual_object_duplicates_collapse_without_changing_bounds(self) -> None:
        elements = (
            UIElement(
                element_id="confirm_text",
                role="text",
                meaning="confirm_current_dialog",
                label="确定",
                bounds=(0.40, 0.60, 0.60, 0.68),
                confidence=0.92,
                evidence=("确定",),
            ),
            UIElement(
                element_id="confirm_button",
                role="button",
                meaning="confirm_current_dialog",
                label="确定",
                bounds=(0.38, 0.58, 0.62, 0.70),
                confidence=0.95,
                evidence=("确定",),
            ),
        )
        original_scene = scene_for(self.frames, elements=elements)
        observation = trusted_observation(
            self.frames,
            scene=original_scene,
            observation_id="obs_55555555555555555555555555555555",
        )
        self.assertEqual([item.element_id for item in observation.scene.elements], ["confirm_button"])
        self.assertEqual(dict(observation.candidate_aliases)["confirm_text"], "confirm_button")
        self.assertEqual(
            observation.get_candidate("confirm_button").bounds,
            elements[1].bounds,
        )
        with self.assertRaisesRegex(UISceneError, "不存在元素"):
            observation.get_candidate("confirm_text")
        context = task_context(task_id="task_confirm_unique", revision=16)
        context["goal"]["entities"] = {
            "expected_text": "确定",
            "expected_role": "button",
        }
        payload = action_payload(context, observation, element_id="confirm_button")
        observer = QwenVisualDecisionObserver(FakeProvider(payload))
        decision = observer.decide(
            frames=self.frames,
            task_context=context,
            trusted_observation=observation,
        )
        self.assertEqual(decision.proposal.status, "action")

    def test_overlapping_semantic_conflict_is_recorded_not_silently_merged(self) -> None:
        elements = (
            UIElement(
                element_id="left_action",
                role="button",
                meaning="accept_change",
                label="接受",
                bounds=(0.35, 0.50, 0.65, 0.62),
                confidence=0.94,
            ),
            UIElement(
                element_id="right_action",
                role="button",
                meaning="reject_change",
                label="拒绝",
                bounds=(0.36, 0.51, 0.66, 0.63),
                confidence=0.94,
            ),
        )
        observation = trusted_observation(
            self.frames,
            scene=scene_for(self.frames, elements=elements),
            observation_id="obs_66666666666666666666666666666666",
        )
        self.assertEqual(len(observation.scene.elements), 2)
        self.assertTrue(
            any(
                item["kind"] == "overlapping_semantic_conflict"
                for item in observation.candidate_conflicts
            )
        )

    def test_action_cannot_ignore_unique_exact_text_candidate(self) -> None:
        context = task_context(task_id="task_exact_select", revision=14)
        context["goal"]["entities"] = {"target_text": "设置"}
        wrong = action_payload(
            context,
            self.observation,
            element_id="unlabelled_camera_icon",
        )
        provider = SequenceProvider([wrong, wrong])
        _observer, decision = self.decide(provider, context=context)
        self.assertEqual(decision.proposal.status, "blocked")
        self.assertIn("逐字一致唯一候选", decision.reason)

    def test_confirmed_external_context_can_propose_one_bound_action(self) -> None:
        context = task_context(external=True, confirmed=True)
        response = action_payload(context, self.observation)
        provider = FakeProvider(response)
        _observer, decision = self.decide(provider, context=context)
        self.assertEqual(provider.calls, 1)
        self.assertEqual(decision.proposal.status, "action")

    def test_device_capability_set_rejects_unavailable_action(self) -> None:
        response = action_payload(self.context, self.observation)
        provider = SequenceProvider([response, response])
        observer = QwenVisualDecisionObserver(provider)

        decision = observer.decide(
            frames=self.frames,
            task_context=self.context,
            trusted_observation=self.observation,
            available_action_kinds={"wait_for_change"},
        )

        self.assertEqual(provider.calls, 2)
        self.assertEqual(decision.proposal.status, "blocked")
        self.assertIn("没有本地验证动作能力", decision.reason)
        self.assertEqual(
            observer.last_diagnostics["available_action_kinds"],
            ["wait_for_change"],
        )

    def test_decision_service_disconnect_returns_blocked_with_diagnostics(self) -> None:
        provider = SequenceProvider(
            [VisionAgentError("千问视觉连接连续1次中断：Server disconnected")]
        )
        observer, decision = self.decide(provider)
        self.assertEqual(provider.calls, 1)
        self.assertEqual(decision.proposal.status, "blocked")
        self.assertEqual(observer.last_diagnostics["error_type"], "service_disconnect")
        self.assertEqual(observer.last_diagnostics["model_calls"], 1)
        self.assertEqual(
            len(observer.last_diagnostics["model_call_elapsed_seconds"]),
            1,
        )
        self.assertIn("未形成候选动作", observer.last_diagnostics["safe_stop_reason"])

    def test_disconnect_during_format_retry_discards_first_output(self) -> None:
        invalid = action_payload(self.context, self.observation)
        invalid["target_region"]["bounds"] = [1, 1, 10, 10]
        provider = SequenceProvider(
            [
                invalid,
                VisionAgentError("千问视觉连接连续1次中断：Server disconnected"),
            ]
        )
        observer, decision = self.decide(provider)
        self.assertEqual(provider.calls, 2)
        self.assertEqual(decision.proposal.status, "blocked")
        self.assertTrue(observer.last_diagnostics["first_output_rejected"])
        self.assertFalse(observer.last_diagnostics["candidate_action_from_first_output"])
        self.assertEqual(observer.last_diagnostics["error_type"], "service_disconnect")

    def test_multiple_actions_field_is_rejected(self) -> None:
        bad = action_payload(self.context, self.observation)
        bad["actions"] = [bad["next_action"], bad["next_action"]]
        provider = SequenceProvider([bad, bad])
        _observer, decision = self.decide(provider)
        self.assertEqual(decision.proposal.status, "blocked")
        self.assertIn("协议外字段：actions", decision.reason)

    def test_modified_candidate_bounds_are_rejected(self) -> None:
        bad = action_payload(self.context, self.observation)
        bad["target_region"]["bounds"] = [690, 210, 850, 340]
        provider = SequenceProvider([bad, bad])
        _observer, decision = self.decide(provider)
        self.assertEqual(decision.proposal.status, "blocked")
        self.assertIn("原始 bounds", decision.reason)

    def test_invalid_first_output_gets_exactly_one_retry(self) -> None:
        invalid = action_payload(self.context, self.observation)
        invalid["target_region"]["bounds"] = [1, 1, 10, 10]
        valid = action_payload(self.context, self.observation)
        provider = SequenceProvider([invalid, valid])
        observer, decision = self.decide(provider)
        self.assertEqual(provider.calls, 2)
        self.assertEqual(decision.proposal.status, "action")
        self.assertTrue(observer.last_diagnostics["protocol_retry_used"])
        self.assertFalse(
            observer.last_diagnostics["candidate_action_from_first_output"]
        )
        status = observer.status()
        self.assertEqual(status["first_pass_rate"], 0.0)
        self.assertEqual(status["repair_retry_rate"], 1.0)

    def test_known_action_field_aliases_are_normalized_before_trust_checks(self) -> None:
        payload = action_payload(self.context, self.observation)
        action = payload["next_action"]
        action["action_type"] = action.pop("kind")
        action["target_element_id"] = action.pop("element_id")

        _observer, decision = self.decide(FakeProvider(payload))

        self.assertEqual("action", decision.proposal.status)
        self.assertEqual("tap_semantic", decision.proposal.action.action)
        self.assertEqual(
            "settings_icon",
            decision.proposal.action.params["element_id"],
        )

    def test_nested_swipe_params_use_only_local_verified_direction(self) -> None:
        payload = action_payload(self.context, self.observation)
        payload["next_action"] = {
            "type": "swipe",
            "params": {"direction": "up", "distance": 300},
        }
        payload["target_region"] = {
            "element_id": "settings_icon",
            "bounds": [680, 200, 860, 350],
            "description": "模型建议的局部滚动区域",
        }
        payload["expected_result"] = {"list_content_changed": True}

        _observer, decision = self.decide(FakeProvider(payload))

        self.assertEqual("action", decision.proposal.status)
        self.assertEqual("swipe", decision.proposal.action.action)
        self.assertEqual("up", decision.proposal.action.params["direction"])
        self.assertNotIn("distance", decision.proposal.action.params)
        for irrelevant in (
            "element_id",
            "target",
            "role",
            "label",
            "states",
            "duration_ms",
            "source_element_id",
            "destination_element_id",
        ):
            self.assertNotIn(irrelevant, decision.proposal.action.params)
        self.assertEqual("screen", decision.target_region.kind)
        self.assertEqual("", decision.target_region.element_id)
        self.assertEqual((0.0, 0.0, 1.0, 1.0), decision.target_region.bounds)

    def test_conflicting_nested_action_param_is_rejected(self) -> None:
        bad = action_payload(self.context, self.observation)
        bad["next_action"] = {
            "kind": "swipe",
            "direction": "down",
            "params": {"direction": "up"},
        }
        provider = SequenceProvider([bad, bad])

        _observer, decision = self.decide(provider)

        self.assertEqual("blocked", decision.proposal.status)
        self.assertIn("params.direction", decision.reason)

    def test_nested_action_params_still_reject_raw_coordinates(self) -> None:
        bad = action_payload(self.context, self.observation)
        bad["next_action"] = {
            "type": "swipe",
            "params": {"direction": "up", "x": 400, "y": 600},
        }
        provider = SequenceProvider([bad, bad])

        _observer, decision = self.decide(provider)

        self.assertEqual("blocked", decision.proposal.status)
        self.assertIn("禁止字段", decision.reason)

    def test_nested_action_params_must_be_an_object(self) -> None:
        bad = action_payload(self.context, self.observation)
        bad["next_action"] = {"type": "swipe", "params": "direction=up"}
        provider = SequenceProvider([bad, bad])

        _observer, decision = self.decide(provider)

        self.assertEqual("blocked", decision.proposal.status)
        self.assertIn("params 必须是JSON对象", decision.reason)

    def test_action_alias_and_nested_target_region_are_normalized(self) -> None:
        payload = action_payload(self.context, self.observation)
        action = payload["next_action"]
        action["action"] = action.pop("kind")
        action["target_region"] = payload.pop("target_region")

        _observer, decision = self.decide(FakeProvider(payload))

        self.assertEqual("action", decision.proposal.status)
        self.assertEqual("tap_semantic", decision.proposal.action.action)
        self.assertEqual("settings_icon", decision.target_region.element_id)

    def test_nested_expected_result_is_promoted_and_verified(self) -> None:
        payload = action_payload(self.context, self.observation)
        expected = payload.pop("expected_result")
        payload["next_action"]["expected_result"] = expected

        _observer, decision = self.decide(FakeProvider(payload))

        self.assertEqual("action", decision.proposal.status)
        self.assertEqual(expected, decision.expected_result)
        self.assertEqual(
            expected,
            decision.proposal.action.params["expected_effect"],
        )

    def test_expected_result_aliases_normalize_to_controller_contract(self) -> None:
        payload = action_payload(self.context, self.observation)
        payload["expected_result"] = {
            "screen_change": True,
            "new_foreground_app_id": "settings",
            "new_screen_id": "settings_main",
        }

        _observer, decision = self.decide(FakeProvider(payload))

        self.assertEqual(
            {
                "scene_changed": True,
                "app_id": "settings",
                "screen_id": "settings_main",
            },
            decision.expected_result,
        )
        self.assertEqual(
            decision.expected_result,
            decision.proposal.action.params["expected_effect"],
        )

    def test_expected_result_conflicting_alias_is_rejected(self) -> None:
        bad = action_payload(self.context, self.observation)
        bad["expected_result"] = {
            "app_id": "settings",
            "new_foreground_app_id": "browser",
        }
        provider = SequenceProvider([bad, bad])

        _observer, decision = self.decide(provider)

        self.assertEqual("blocked", decision.proposal.status)
        self.assertIn("expected_result", decision.reason)
        self.assertIn("冲突", decision.reason)

    def test_expected_result_unsupported_claim_is_rejected(self) -> None:
        bad = action_payload(self.context, self.observation)
        bad["expected_result"] = {
            "scene_changed": True,
            "no_search_or_login_ui": True,
        }
        provider = SequenceProvider([bad, bad])

        _observer, decision = self.decide(provider)

        self.assertEqual("blocked", decision.proposal.status)
        self.assertIn("协议外字段", decision.reason)

    def test_conflicting_nested_expected_result_is_rejected(self) -> None:
        bad = action_payload(self.context, self.observation)
        bad["next_action"]["expected_result"] = {"scene_changed": False}
        provider = SequenceProvider([bad, bad])

        _observer, decision = self.decide(provider)

        self.assertEqual("blocked", decision.proposal.status)
        self.assertIn("与顶层 expected_result 冲突", decision.reason)

    def test_conflicting_nested_target_region_is_rejected(self) -> None:
        bad = action_payload(self.context, self.observation)
        nested = copy.deepcopy(bad["target_region"])
        nested["element_id"] = "other_candidate"
        bad["next_action"]["target_region"] = nested
        provider = SequenceProvider([bad, bad])

        _observer, decision = self.decide(provider)

        self.assertEqual("blocked", decision.proposal.status)
        self.assertIn("与顶层 target_region 冲突", decision.reason)

    def test_conflicting_action_field_alias_is_rejected(self) -> None:
        bad = action_payload(self.context, self.observation)
        bad["next_action"]["target_element_id"] = "other_candidate"
        provider = SequenceProvider([bad, bad])

        _observer, decision = self.decide(provider)

        self.assertEqual("blocked", decision.proposal.status)
        self.assertIn("target_element_id 与 element_id 冲突", decision.reason)

    def test_missing_target_region_kind_is_derived_from_verified_action(self) -> None:
        payload = action_payload(self.context, self.observation)
        payload["target_region"].pop("kind")

        _observer, decision = self.decide(FakeProvider(payload))

        self.assertEqual("action", decision.proposal.status)
        self.assertEqual("element", decision.target_region.kind)

    def test_missing_target_region_is_built_from_trusted_candidate(self) -> None:
        payload = action_payload(self.context, self.observation)
        payload["target_region"] = None

        _observer, decision = self.decide(FakeProvider(payload))

        candidate = self.observation.get_candidate("settings_icon")
        self.assertEqual("action", decision.proposal.status)
        self.assertEqual("element", decision.target_region.kind)
        self.assertEqual(candidate.element_id, decision.target_region.element_id)
        self.assertEqual(candidate.bounds, decision.target_region.bounds)
        self.assertEqual(candidate.label, decision.target_region.description)

    def test_missing_target_region_description_uses_trusted_candidate(self) -> None:
        payload = action_payload(self.context, self.observation)
        payload["target_region"]["description"] = ""

        _observer, decision = self.decide(FakeProvider(payload))

        self.assertEqual("设置", decision.target_region.description)

    def test_known_target_region_aliases_are_normalized(self) -> None:
        payload = action_payload(self.context, self.observation)
        region = payload["target_region"]
        region["region_type"] = region.pop("kind")
        region["target_element_id"] = region.pop("element_id")
        region["target_bounds"] = region.pop("bounds")

        _observer, decision = self.decide(FakeProvider(payload))

        self.assertEqual("action", decision.proposal.status)
        self.assertEqual("element", decision.target_region.kind)
        self.assertEqual("settings_icon", decision.target_region.element_id)

    def test_conflicting_target_region_alias_is_rejected(self) -> None:
        bad = action_payload(self.context, self.observation)
        bad["target_region"]["region_type"] = "screen"
        provider = SequenceProvider([bad, bad])

        _observer, decision = self.decide(provider)

        self.assertEqual("blocked", decision.proposal.status)
        self.assertIn("region_type 与 kind 冲突", decision.reason)

    def test_retry_prompt_makes_status_and_action_fields_mutually_exclusive(self) -> None:
        prompt = _decision_retry_prompt(
            QwenTaskContext.from_dict(self.context),
            self.observation,
            error=VisionAgentError("finished/blocked 不能携带 next_action"),
            decision_number=1,
            available_action_kinds=frozenset({"tap_semantic", "back"}),
        )

        self.assertIn('status="action"', prompt)
        self.assertIn('status="blocked"', prompt)
        self.assertIn('status="finished"', prompt)
        self.assertIn("不要混合三种形状", prompt)
        self.assertIn("绝不能保留B/C的status", prompt)

    def test_unstable_real_page_sequence_does_not_call_qwen(self) -> None:
        overlay = load_replay_image("douyin_digit_local_input_com.jpg")
        moving = [self.frames[0], overlay, self.frames[1], overlay.copy()]
        provider = FakeProvider(action_payload(self.context, self.observation))
        observer = QwenVisualDecisionObserver(provider)
        with self.assertRaisesRegex(VisionAgentError, "不稳定"):
            observer.decide(
                frames=moving,
                task_context=self.context,
                trusted_observation=self.observation,
            )
        self.assertEqual(provider.calls, 0)

    def test_motion_blur_cannot_establish_trusted_observation(self) -> None:
        blurred = load_replay_image("motion_blur_after_pinyin.jpg")
        with self.assertRaisesRegex(VisionAgentError, "仍然模糊"):
            TrustedObservation.from_scene(
                frames=[blurred.copy() for _ in range(4)],
                device_id="offline_phone_01",
                scene=scene_for([blurred.copy() for _ in range(4)]),
            )

    def test_scene_fingerprint_must_come_from_current_frames(self) -> None:
        scene = scene_for(self.frames)
        stale = UIScene(
            app_id=scene.app_id,
            screen_id=scene.screen_id,
            summary=scene.summary,
            elements=scene.elements,
            stable=True,
            confidence=scene.confidence,
            fingerprint="stale_fingerprint",
        )
        with self.assertRaisesRegex(VisionAgentError, "fingerprint"):
            TrustedObservation.from_scene(
                frames=self.frames,
                device_id="offline_phone_01",
                scene=stale,
            )

    def test_finished_uses_only_trusted_evidence_ids(self) -> None:
        payload = action_payload(self.context, self.observation)
        payload.update(
            {
                "status": "finished",
                "next_action": None,
                "target_region": None,
                "expected_result": {},
                "confidence": 0.94,
                "completion_evidence_element_ids": ["settings_icon"],
            }
        )
        _observer, decision = self.decide(FakeProvider(payload))
        self.assertEqual(decision.proposal.status, "finished")
        self.assertEqual(
            decision.proposal.completion_evidence,
            ("settings_icon:设置",),
        )

    def test_finished_with_forged_evidence_id_is_rejected(self) -> None:
        bad = action_payload(self.context, self.observation)
        bad.update(
            {
                "status": "finished",
                "next_action": None,
                "target_region": None,
                "expected_result": {},
                "completion_evidence_element_ids": ["mars_entry"],
            }
        )
        provider = SequenceProvider([bad, bad])
        _observer, decision = self.decide(provider)
        self.assertEqual(decision.proposal.status, "blocked")
        self.assertIn("不存在元素", decision.reason)

    def test_unlabelled_icon_can_only_be_selected_by_existing_id(self) -> None:
        payload = action_payload(
            self.context,
            self.observation,
            element_id="unlabelled_camera_icon",
        )
        _observer, decision = self.decide(FakeProvider(payload))
        self.assertEqual(
            decision.proposal.action.params["element_id"],
            "unlabelled_camera_icon",
        )
        self.assertEqual(decision.proposal.action.params.get("label", ""), "")

    def test_redacted_list_page_supports_only_one_screen_swipe(self) -> None:
        frames = repeated_frames(ASSET_ROOT / "settings_list_redacted.png")
        scene = scene_for(
            frames,
            elements=settings_list_elements(),
            app_id="system_surface",
            screen_id="scrollable_list",
            summary="脱敏后的纵向列表清晰可见",
        )
        observation = trusted_observation(
            frames,
            scene=scene,
            observation_id="obs_11111111111111111111111111111111",
        )
        context = task_context(task_id="task_scroll_list", revision=8)
        payload = action_payload(context, observation, element_id="visible_text_item")
        payload.update(
            {
                "next_action": {"kind": "swipe", "direction": "up"},
                "target_region": {
                    "kind": "screen",
                    "element_id": None,
                    "bounds": [0, 0, 1000, 1000],
                    "description": "当前可滚动列表整屏区域",
                },
                "expected_result": {"list_content_changed": True},
            }
        )
        observer = QwenVisualDecisionObserver(FakeProvider(payload))
        decision = observer.decide(
            frames=frames,
            task_context=context,
            trusted_observation=observation,
        )
        self.assertEqual(decision.proposal.action.action, "swipe")
        self.assertEqual(decision.proposal.action.params["direction"], "up")

    def test_overlay_page_covers_close_input_text_button_and_finished(self) -> None:
        frames = repeated_frames(REPLAY_ROOT / "douyin_digit_local_input_com.jpg")
        scene = scene_for(
            frames,
            elements=overlay_elements(),
            app_id="media_surface",
            screen_id="input_overlay",
            summary="输入弹层、输入框和文字提交按钮清晰可见",
            overlays=("input_overlay",),
        )
        observation = trusted_observation(
            frames,
            scene=scene,
            observation_id="obs_22222222222222222222222222222222",
        )

        close_context = task_context(task_id="task_close_overlay", revision=9)
        close_payload = action_payload(
            close_context,
            observation,
            element_id="unlabelled_close_icon",
        )
        close_payload["next_action"]["kind"] = "dismiss_overlay"
        close_observer = QwenVisualDecisionObserver(FakeProvider(close_payload))
        close_decision = close_observer.decide(
            frames=frames,
            task_context=close_context,
            trusted_observation=observation,
        )
        self.assertEqual(close_decision.proposal.action.action, "dismiss_overlay")

        submit_context = task_context(
            task_id="task_submit_text_button",
            revision=10,
            external=True,
            confirmed=True,
        )
        submit_payload = action_payload(
            submit_context,
            observation,
            element_id="text_submit_button",
        )
        submit_observer = QwenVisualDecisionObserver(FakeProvider(submit_payload))
        submit_decision = submit_observer.decide(
            frames=frames,
            task_context=submit_context,
            trusted_observation=observation,
        )
        self.assertEqual(
            submit_decision.proposal.action.params["label"],
            "发布",
        )

        finished_context = task_context(task_id="task_input_finished", revision=11)
        finished_context["goal"]["entities"] = {"expected_text": ".com"}
        finished_payload = action_payload(
            finished_context,
            observation,
            element_id="input_value",
        )
        finished_payload.update(
            {
                "status": "finished",
                "next_action": None,
                "target_region": None,
                "expected_result": {},
                "completion_evidence_element_ids": ["input_value"],
            }
        )
        finished_observer = QwenVisualDecisionObserver(FakeProvider(finished_payload))
        finished_decision = finished_observer.decide(
            frames=frames,
            task_context=finished_context,
            trusted_observation=observation,
        )
        self.assertEqual(finished_decision.proposal.status, "finished")
        self.assertEqual(
            finished_decision.proposal.completion_evidence,
            ("input_value:.com",),
        )

    def test_full_deepseek_context_is_preserved_in_prompt(self) -> None:
        provider = FakeProvider(blocked_payload(self.context, self.observation))
        self.decide(provider)
        prompt = provider.messages[-1]["content"][0]["text"]
        for field in (
            "protocol_version",
            "task_id",
            "device_id",
            "revision",
            "current_subgoal",
            "global_constraints",
            "current_external_impact",
            "risk_actions",
            "confirmation_gate",
        ):
            self.assertIn(field, prompt)


if __name__ == "__main__":
    unittest.main()
