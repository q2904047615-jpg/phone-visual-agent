from __future__ import annotations

import copy
import json
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageDraw

from canonical_action_protocol import compile_canonical_action_catalog
from generic_scene_observer import _local_frame_fingerprint
from generic_step_planner import GenericStepPlanningError
from qwen_visual_decision import (
    QWEN_VISUAL_DECISION_PROTOCOL_VERSION,
    QwenTaskContext,
    QwenVisualDecisionObserver,
    TrustedObservation,
    _decision_retry_prompt,
    _exact_text_candidate_block,
    _identity_text_candidate_block,
    _launcher_app_entry_candidate_ids,
    _required_exact_candidate_ids,
    _scene_matches_target_app_surface,
    _selection_choices,
)
from ui_scene import SystemUIFacts, UIElement, UIScene, UISceneError
from vision_agent import VisionAgentError
from vision_agent import _image_data_url
from system_navigation_privacy import privacy_minimized_system_navigation_view
from task_semantic_ir import (
    ConstraintIntent,
    DesiredState,
    EffectIntent,
    InputFieldIntent,
    SemanticEntity,
    SemanticSubgoal,
    SourceSpan,
    SurfaceRef,
    TaskSemanticIR,
)


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


class RawSequenceProvider(FakeProvider):
    def __init__(self, responses: list[str]) -> None:
        super().__init__({})
        self.responses = list(responses)

    def _chat(self, messages, max_tokens, **kwargs) -> str:
        self.calls += 1
        self.messages = messages
        if not self.responses:
            raise AssertionError("模型被调用超过一次初始请求和一次修复重试")
        return self.responses.pop(0)


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
            states={"goal_relevant": True, "fully_visible": True},
            evidence=("设置",),
        ),
        UIElement(
            element_id="unlabelled_camera_icon",
            role="icon",
            meaning="open_camera",
            label="",
            bounds=(0.69, 0.82, 0.88, 0.95),
            confidence=0.92,
            states={"goal_relevant": False, "fully_visible": True},
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
            states={
                "fully_visible": True,
                "scrollable": True,
                "scroll_axis": "vertical",
            },
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
    stable_tail_start = max(0, len(frames) - min(3, len(frames)))
    selected = max(
        range(stable_tail_start, len(frames)),
        key=sharpness.__getitem__,
    )
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
    effect_ids = ["effect_send"] if external else []
    execution_class = "effect" if external else "navigate"
    return {
        "protocol_version": "2026-08-20-deepseek-typed-task-graph-v4",
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
            "effect_ids": effect_ids,
            "execution_class": execution_class,
        },
        "current_execution_class": execution_class,
        "effect_intents": (
            [
                {
                    "effect_id": "effect_send",
                    "kind": "data_mutation",
                    "target_entity_roles": [],
                    "payload_entity_roles": [],
                    "source_subgoal_ids": ["current_target"],
                    "expected_results": ["内容会被提交"],
                    "local_policy": {
                        "effect_id": "effect_send",
                        "confirmation_required": True,
                        "policy_level": "high",
                    },
                }
            ]
            if external
            else []
        ),
        "effect_gate": {
            "required": external,
            "state": "confirmed" if confirmed else "awaiting_confirmation" if external else "not_required",
            "effect_ids": effect_ids,
            "scope": {
                "task_id": task_id,
                "device_id": "offline_phone_01",
                "revision": revision,
                "subgoal_id": "current_target",
            },
            "effect_action_allowed": bool(external and confirmed),
        },
    }


def test_context_with_semantic_ir(
    value: dict | QwenTaskContext,
    observation: TrustedObservation,
    available_action_kinds: frozenset[str],
) -> QwenTaskContext:
    """Upgrade parser-era fixtures to the sole canonical action protocol."""

    context = value if isinstance(value, QwenTaskContext) else QwenTaskContext.from_dict(value)
    if context.semantic_ir is not None:
        return context

    raw_goal = " ".join(
        (
            str(context.goal.get("objective") or ""),
            str(context.current_subgoal.get("objective") or ""),
            *(str(item.label) for item in observation.scene.elements if item.label),
        )
    ) or "test task"
    entities: list[SemanticEntity] = []
    seen_values: set[str] = set()
    for element in observation.scene.elements:
        value_text = str(element.label or "").strip()
        if not value_text or value_text.casefold() in seen_values:
            continue
        seen_values.add(value_text.casefold())
        entities.append(
            SemanticEntity(
                entity_id=f"entity_visible_{len(entities) + 1}",
                entity_type="ui_label",
                role="target_ui_label",
                value=value_text,
                authority="planner_context",
            )
        )

    goal_entities = context.goal.get("entities") or {}
    payload_text = ""
    if isinstance(goal_entities, dict):
        for key in ("input_text", "text", "content", "message"):
            candidate = goal_entities.get(key)
            if isinstance(candidate, str):
                payload_text = candidate
                break
    if not payload_text:
        for element in observation.scene.elements:
            expected = element.states.get("expected_input_value")
            if isinstance(expected, str) and expected:
                payload_text = expected
                break
    if not payload_text and "clear_verified_text" in available_action_kinds:
        for element in observation.scene.elements:
            current = element.states.get("value")
            if element.role == "input" and isinstance(current, str) and current:
                payload_text = current
                break
    payload_ref = ""
    if payload_text:
        payload_ref = "entity_input_text"
        entities.append(
            SemanticEntity(
                entity_id=payload_ref,
                entity_type="text",
                role="input_text",
                value=payload_text,
                authority="planner_context",
            )
        )

    active_text = " ".join(
        (
            str(context.current_subgoal.get("objective") or ""),
            *(
                str(item)
                for item in context.current_subgoal.get(
                    "completion_conditions",
                    [],
                )
            ),
        )
    ).casefold()
    required_actions: set[str] = set()
    if "clear_verified_text" in available_action_kinds and any(
        token in active_text for token in ("清空", "清除", "空白", "为空", "clear")
    ):
        required_actions.add("clear_verified_text")
    if "back" in available_action_kinds and any(
        token in active_text for token in ("收起", "隐藏", "不可见", "dismiss", "hide")
    ):
        required_actions.add("back")
    if len(available_action_kinds) == 1:
        required_actions.update(available_action_kinds)
    constraints = tuple(
        ConstraintIntent(
            constraint_id=f"constraint_action_{index}",
            kind="required_action",
            value=action,
            authoritative=True,
        )
        for index, action in enumerate(
            sorted(
                required_actions
                & {
                    "tap_semantic",
                    "swipe",
                    "long_press",
                    "drag",
                    "input_verified_text",
                    "press_enter",
                    "clear_verified_text",
                    "home",
                    "back",
                    "dismiss_overlay",
                    "reveal_system_navigation",
                }
            ),
            start=1,
        )
    )
    subgoal_id = str(context.current_subgoal["subgoal_id"])
    input_fields = (
        (
            InputFieldIntent(
                field_id="field_test_input",
                payload_ref=payload_ref,
                source_subgoal_ids=(subgoal_id,),
            ),
        )
        if payload_ref
        else ()
    )

    effect_refs: tuple[str, ...] = ()
    effects: tuple[EffectIntent, ...] = ()
    if context.current_execution_class == "effect":
        effect_kind = "generic_effect"
        visible_meanings = {item.meaning for item in observation.scene.elements}
        for candidate_kind in (
            "send_message",
            "publish_content",
            "relationship_change",
            "membership_change",
        ):
            if candidate_kind in visible_meanings:
                effect_kind = candidate_kind
                break
        effect_refs = ("effect_test",)
        effects = (
            EffectIntent(
                effect_id="effect_test",
                kind=effect_kind,
                payload_refs=(payload_ref,) if payload_ref else (),
                source_subgoal_ids=(subgoal_id,),
            ),
        )

    semantic_ir = TaskSemanticIR(
        task_id=context.task_id,
        device_id=context.device_id,
        revision=context.revision,
        raw_goal=raw_goal,
        surfaces=(SurfaceRef(surface_id="surface_current", kind="current_surface"),),
        entities=tuple(entities),
        effects=effects,
        constraints=constraints,
        desired_states=(
            DesiredState(
                state_id="state_active_goal",
                subject_ref="surface_current",
                predicate="surface.state_visible",
                value=active_text or "current state visible",
                source_subgoal_id=subgoal_id,
            ),
        ),
        subgoals=(
            SemanticSubgoal(
                subgoal_id=subgoal_id,
                surface_ref="surface_current",
                status="active",
                external_impact=(
                    "external_state"
                    if context.current_execution_class == "effect"
                    else "navigation_only"
                ),
                constraint_refs=tuple(item.constraint_id for item in constraints),
                entity_refs=tuple(item.entity_id for item in entities),
                desired_state_refs=("state_active_goal",),
                effect_refs=effect_refs,
            ),
        ),
        input_fields=input_fields,
    )
    return replace(context, semantic_ir=semantic_ir)


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


def minimal_selection_payload(
    *,
    status: str,
    choice_id: str | None = None,
    completes_current_subgoal_on_success: bool = False,
    completion_evidence_element_ids: list[str] | None = None,
) -> dict:
    return {
        "status": status,
        "choice_id": choice_id,
        "completes_current_subgoal_on_success": (
            completes_current_subgoal_on_success
        ),
        "confidence": 0.94,
        "reason": "当前可信画面与活动子目标支持该选择。",
        "completion_evidence_element_ids": (
            completion_evidence_element_ids or []
        ),
    }


class QwenVisualDecisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frames = load_sequence("launcher_stable")
        self.context = task_context()
        self.observation = trusted_observation(self.frames)

    def test_reload_literal_alias_requires_unique_local_visual_audit(self) -> None:
        audited_reload = UIElement(
            element_id="local_audited_reload_control_1",
            role="icon",
            meaning="reload",
            label="",
            bounds=(0.82, 0.01, 0.86, 0.04),
            confidence=0.95,
            states={
                "goal_relevant": True,
                "fully_visible": True,
                "reload_visual_audit": True,
                "independent_geometry_verified": True,
            },
            evidence=("圆弧与箭头组成的独立图标",),
        )
        observation = trusted_observation(
            self.frames,
            elements=(audited_reload,),
        )
        raw = task_context(task_id="task_reload_alias")
        raw["goal"]["entities"] = {"target_ui_label": "刷新图标"}
        raw["current_subgoal"].update(
            objective="点击当前浏览器顶部可见的刷新图标",
            completion_conditions=["刷新图标已被点击"],
        )
        context = test_context_with_semantic_ir(
            raw,
            observation,
            frozenset({"tap_semantic"}),
        )
        self.assertIsNone(_exact_text_candidate_block(context, observation))
        self.assertEqual(
            {"local_audited_reload_control_1"},
            _required_exact_candidate_ids(context, observation),
        )

        for states, required_text in (
            ({"goal_relevant": True, "fully_visible": True}, "刷新图标"),
            (audited_reload.states, "支付图标"),
        ):
            with self.subTest(states=states, required_text=required_text):
                candidate = replace(audited_reload, states=states)
                blocked_observation = trusted_observation(
                    self.frames,
                    elements=(candidate,),
                    observation_id="obs_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                )
                blocked_raw = copy.deepcopy(raw)
                blocked_raw["goal"]["entities"] = {
                    "target_ui_label": required_text
                }
                blocked_context = test_context_with_semantic_ir(
                    blocked_raw,
                    blocked_observation,
                    frozenset({"tap_semantic"}),
                )
                self.assertEqual(
                    "exact_text_missing",
                    _exact_text_candidate_block(
                        blocked_context,
                        blocked_observation,
                    )[1],
                )

    def test_cross_surface_app_entry_offers_home_before_app_controls(self) -> None:
        parsed = QwenTaskContext.from_dict(task_context())
        semantic_ir = TaskSemanticIR(
            task_id=parsed.task_id,
            device_id=parsed.device_id,
            revision=parsed.revision,
            raw_goal="打开目标应用",
            surfaces=(
                SurfaceRef(
                    surface_id="surface_target",
                    kind="app",
                    app_id="target",
                    app_name="目标应用",
                ),
            ),
            entities=(
                SemanticEntity(
                    entity_id="entity_target",
                    entity_type="ui_label",
                    role="target_ui_label",
                    value="目标应用",
                    source_span=SourceSpan(2, 6),
                    authority="user_literal",
                ),
            ),
            effects=(),
            subgoals=(
                SemanticSubgoal(
                    subgoal_id="current_target",
                    surface_ref="surface_target",
                    status="active",
                    external_impact="navigation_only",
                ),
            ),
        )
        parsed = replace(parsed, semantic_ir=semantic_ir)
        current_scene = UIScene(
            app_id="source",
            screen_id="source_home",
            summary="当前位于另一个应用",
            elements=(
                UIElement(
                    element_id="target-entry",
                    role="button",
                    meaning="open_target",
                    label="目标应用",
                    bounds=(0.2, 0.2, 0.7, 0.3),
                    confidence=0.97,
                    states={"visible": True, "fully_visible": True},
                ),
            ),
            stable=True,
            confidence=0.96,
            fingerprint=scene_for(self.frames).fingerprint,
        )
        observation = trusted_observation(
            self.frames,
            scene=current_scene,
        )
        choices = _selection_choices(
            parsed,
            observation,
            frozenset({"home", "tap_semantic"}),
        )
        self.assertEqual(["home"], [item["action"] for item in choices])

    def test_target_app_surface_rejects_near_package_and_non_title_mentions(self) -> None:
        target = SimpleNamespace(app_id="settings", app_name="设置")
        exact_display_name = scene_for(
            self.frames,
            app_id="设置",
            screen_id="settings_main",
            elements=(),
        )
        self.assertTrue(
            _scene_matches_target_app_surface(exact_display_name, target)
        )

        near_display_name = scene_for(
            self.frames,
            app_id="设置助手",
            screen_id="helper_main",
            elements=(),
        )
        self.assertFalse(
            _scene_matches_target_app_surface(near_display_name, target)
        )

        exact_package = scene_for(
            self.frames,
            app_id="com.android.settings",
            screen_id="settings_main",
            elements=(),
        )
        self.assertTrue(_scene_matches_target_app_surface(exact_package, target))

        opaque_package_with_title = scene_for(
            self.frames,
            app_id="com.vendor.controlcenter",
            screen_id="main",
            elements=(
                UIElement(
                    element_id="title",
                    role="text",
                    meaning="page_title",
                    label="设置",
                    bounds=(0.1, 0.05, 0.4, 0.12),
                    confidence=0.98,
                    states={"fully_visible": True},
                ),
            ),
        )
        self.assertTrue(
            _scene_matches_target_app_surface(opaque_package_with_title, target)
        )

        near_package_with_body_mention = scene_for(
            self.frames,
            app_id="com.android.settings_helper",
            screen_id="helper_main",
            elements=(
                UIElement(
                    element_id="body-mention",
                    role="text",
                    meaning="body_text",
                    label="设置",
                    bounds=(0.1, 0.2, 0.4, 0.26),
                    confidence=0.98,
                    states={"fully_visible": True},
                ),
            ),
        )
        self.assertFalse(
            _scene_matches_target_app_surface(near_package_with_body_mention, target)
        )

    def test_exact_foreground_app_name_keeps_in_app_literal_navigation_choice(self) -> None:
        parsed = QwenTaskContext.from_dict(task_context())
        semantic_ir = TaskSemanticIR(
            task_id=parsed.task_id,
            device_id=parsed.device_id,
            revision=parsed.revision,
            raw_goal="进入微信中的文件传输助手",
            surfaces=(
                SurfaceRef(
                    surface_id="surface_wechat",
                    kind="app",
                    app_id="wechat",
                    app_name="微信",
                ),
            ),
            entities=(
                SemanticEntity(
                    entity_id="entity_recipient",
                    entity_type="party",
                    role="recipient",
                    value="文件传输助手",
                    source_span=SourceSpan(6, 12),
                    authority="user_literal",
                ),
            ),
            effects=(),
            subgoals=(
                SemanticSubgoal(
                    subgoal_id=str(
                        parsed.current_subgoal.get("subgoal_id") or "locate_target"
                    ),
                    surface_ref="surface_wechat",
                    status="active",
                    external_impact="navigation_only",
                    entity_refs=("entity_recipient",),
                ),
            ),
        )
        parsed = replace(parsed, semantic_ir=semantic_ir)
        current_scene = UIScene(
            app_id="微信",
            screen_id="微信消息列表",
            summary="微信消息列表中可见文件传输助手。",
            elements=(
                UIElement(
                    element_id="file-transfer",
                    role="list_item",
                    meaning="chat_entry",
                    label="文件传输助手",
                    bounds=(0.08, 0.22, 0.93, 0.32),
                    confidence=0.99,
                    states={"visible": True, "fully_visible": True},
                ),
            ),
            stable=True,
            confidence=0.98,
            fingerprint=scene_for(self.frames).fingerprint,
        )
        observation = trusted_observation(self.frames, scene=current_scene)

        choices = _selection_choices(
            parsed,
            observation,
            frozenset({"home", "tap_semantic"}),
        )

        self.assertIn(
            ("tap_semantic", "file-transfer"),
            tuple((item["action"], item.get("element_id")) for item in choices),
        )

    def test_launcher_defers_inner_literal_only_for_unique_typed_app_entry(self) -> None:
        raw = task_context()
        raw["goal"]["entities"].update(
            recipient="文件传输助手",
            target_ui_label="文件传输助手",
        )
        raw["current_subgoal"].update(
            objective="打开微信并进入文件传输助手聊天页面",
            completion_conditions=["文件传输助手聊天页面可见"],
        )
        parsed = QwenTaskContext.from_dict(raw)
        parsed = replace(
            parsed,
            semantic_ir=TaskSemanticIR(
                task_id=parsed.task_id,
                device_id=parsed.device_id,
                revision=parsed.revision,
                raw_goal="打开微信并进入文件传输助手聊天页面",
                surfaces=(
                    SurfaceRef(
                        surface_id="surface_wechat",
                        kind="app",
                        app_id="wechat",
                        app_name="微信",
                    ),
                ),
                entities=(
                    SemanticEntity(
                        entity_id="entity_recipient",
                        entity_type="party",
                        role="recipient",
                        value="文件传输助手",
                        source_span=SourceSpan(7, 13),
                        authority="user_literal",
                    ),
                ),
                effects=(),
                subgoals=(
                    SemanticSubgoal(
                        subgoal_id=str(parsed.current_subgoal["subgoal_id"]),
                        surface_ref="surface_wechat",
                        status="active",
                        external_impact="navigation_only",
                    ),
                ),
            ),
        )

        def launcher_scene(*elements: UIElement) -> UIScene:
            return UIScene(
                app_id="launcher",
                screen_id="home_screen",
                summary="手机主屏幕",
                elements=elements,
                stable=True,
                confidence=0.99,
                fingerprint=scene_for(self.frames).fingerprint,
            )

        app_entry = UIElement(
            element_id="open-wechat",
            role="button",
            meaning="open_wechat",
            label="微信",
            bounds=(0.36, 0.63, 0.59, 0.78),
            confidence=0.99,
            states={"goal_relevant": True, "fully_visible": True},
        )
        observation = trusted_observation(
            self.frames,
            scene=launcher_scene(app_entry),
        )

        self.assertEqual(
            ("open-wechat",),
            _launcher_app_entry_candidate_ids(parsed, observation),
        )
        self.assertIsNone(_exact_text_candidate_block(parsed, observation))
        choices = _selection_choices(
            parsed,
            observation,
            frozenset({"home", "swipe", "tap_semantic"}),
        )
        self.assertEqual(
            [("tap_semantic", "open-wechat")],
            [(item["action"], item.get("element_id")) for item in choices],
        )

        variants = (
            launcher_scene(
                app_entry,
                replace(
                    app_entry,
                    element_id="duplicate",
                    bounds=(0.66, 0.63, 0.89, 0.78),
                ),
            ),
            launcher_scene(replace(app_entry, label="微信助手")),
            launcher_scene(
                replace(app_entry, states={"goal_relevant": True, "fully_visible": False})
            ),
            replace(
                launcher_scene(app_entry),
                app_id="微信",
                screen_id="微信消息列表",
            ),
        )
        for scene in variants:
            with self.subTest(app_id=scene.app_id, count=len(scene.elements)):
                variant = trusted_observation(self.frames, scene=scene)
                self.assertEqual((), _launcher_app_entry_candidate_ids(parsed, variant))
                self.assertIsNotNone(_exact_text_candidate_block(parsed, variant))

    def test_recipient_exact_text_applies_only_to_bound_subgoal(self) -> None:
        context = task_context()
        context["goal"]["entities"]["recipient"] = "张三"
        context["current_subgoal"]["objective"] = "聊天应用在前台可见"
        parsed = QwenTaskContext.from_dict(context)
        self.assertNotIn("张三", parsed.exact_text_requirements)

        context["current_subgoal"]["objective"] = "张三的聊天页面在前台可见"
        context["current_subgoal"]["completion_conditions"] = [
            "当前聊天标题逐字显示张三"
        ]
        parsed = QwenTaskContext.from_dict(context)
        self.assertNotIn("张三", parsed.exact_text_requirements)
        self.assertIn("张三", parsed.identity_text_requirements)

        context["current_subgoal"]["objective"] = "打开唯一匹配的张三聊天入口"
        context["goal"]["entities"]["target_ui_label"] = "张三"
        parsed = QwenTaskContext.from_dict(context)
        self.assertEqual(("张三",), parsed.exact_text_requirements)
        self.assertNotIn("张三", parsed.identity_text_requirements)

    def test_duplicate_recipient_target_label_becomes_identity_on_input_subgoal(self) -> None:
        context = task_context(task_id="task_recipient_input_identity", revision=5)
        context["goal"]["entities"] = {
            "recipient": "文件传输助手",
            "target_ui_label": "文件传输助手",
            "input_text": "longinputvalidation2026:123+45-6@7.",
        }
        context["current_subgoal"].update(
            subgoal_id="type_input",
            objective=(
                "在文件传输助手的唯一消息输入框中逐字输入 "
                "longinputvalidation2026:123+45-6@7. 并保持未发送"
            ),
            completion_conditions=[
                "输入框中的文字为 longinputvalidation2026:123+45-6@7.",
                "消息未发送",
            ],
        )
        context["effect_gate"]["scope"]["subgoal_id"] = "type_input"
        parsed = QwenTaskContext.from_dict(context)

        self.assertNotIn("文件传输助手", parsed.exact_text_requirements)
        self.assertEqual(
            ("文件传输助手",),
            parsed.identity_text_requirements,
        )

        title = UIElement(
            element_id="page-title",
            role="text",
            meaning="page_title",
            label="文件传输助手",
            bounds=(0.35, 0.04, 0.65, 0.09),
            confidence=1.0,
            states={"goal_relevant": False, "fully_visible": True},
            evidence=("顶部唯一会话标题",),
        )
        input_element = UIElement(
            element_id="input-1",
            role="input",
            meaning="application_text_input",
            label="",
            bounds=(0.14, 0.915, 0.63, 0.965),
            confidence=0.95,
            states={
                "goal_relevant": True,
                "fully_visible": True,
                "value": "",
                "soft_keyboard_visible": False,
            },
            evidence=("empty input bar",),
        )
        observation = trusted_observation(
            self.frames,
            scene=scene_for(
                self.frames,
                elements=(title, input_element),
                app_id="wechat",
                screen_id="chat_window_file_transfer_helper",
                summary="具名会话页面与唯一空输入框可见",
            ),
        )

        parsed = test_context_with_semantic_ir(
            context,
            observation,
            frozenset({"tap_semantic", "input_verified_text"}),
        )
        self.assertIsNone(_exact_text_candidate_block(parsed, observation))
        self.assertIsNone(_identity_text_candidate_block(parsed, observation))
        choices = _selection_choices(
            parsed,
            observation,
            frozenset({"tap_semantic", "input_verified_text"}),
        )
        self.assertEqual(
            [("tap_semantic", "input-1")],
            [(item["action"], item.get("element_id")) for item in choices],
        )

        missing_identity = trusted_observation(
            self.frames,
            scene=replace(observation.scene, elements=(input_element,)),
            observation_id="obs_abcdef0123456789abcdef0123456789",
        )
        self.assertIsNotNone(
            _identity_text_candidate_block(parsed, missing_identity)
        )

        different_field = copy.deepcopy(context)
        different_field["goal"]["entities"]["target_ui_label"] = "备注"
        different_field["current_subgoal"]["objective"] = (
            "在文件传输助手页面的备注输入框输入指定文字"
        )
        different_parsed = QwenTaskContext.from_dict(different_field)
        self.assertIn("备注", different_parsed.exact_text_requirements)
        self.assertIn(
            "文件传输助手",
            different_parsed.identity_text_requirements,
        )

        multi_recipient = copy.deepcopy(context)
        del multi_recipient["goal"]["entities"]["recipient"]
        multi_recipient["goal"]["entities"]["recipients"] = ["张三", "李四"]
        multi_recipient["goal"]["entities"]["target_ui_label"] = "李四"
        multi_recipient["current_subgoal"]["objective"] = (
            "在李四的唯一消息输入框中输入指定文字并保持未发送"
        )
        multi_parsed = QwenTaskContext.from_dict(multi_recipient)
        self.assertEqual(("李四",), multi_parsed.identity_text_requirements)
        self.assertNotIn("李四", multi_parsed.exact_text_requirements)
        self.assertNotIn("张三", multi_parsed.exact_text_requirements)

    def test_non_element_action_uses_exact_title_only_as_surface_identity(self) -> None:
        raw = task_context(task_id="task_swipe_identity", revision=17)
        raw["goal"]["entities"] = {"target_ui_label": "文件传输助手"}
        raw["current_subgoal"].update(
            objective="在当前文件传输助手聊天中向上滑动一次",
            completion_conditions=["聊天记录区域内容发生变化"],
        )
        parsed = QwenTaskContext.from_dict(raw)

        def with_required_action(action: str) -> QwenTaskContext:
            return replace(
                parsed,
                semantic_ir=TaskSemanticIR(
                    task_id=parsed.task_id,
                    device_id=parsed.device_id,
                    revision=parsed.revision,
                    raw_goal="在当前文件传输助手聊天中向上滑动一次",
                    surfaces=(
                        SurfaceRef(
                            surface_id="surface_current",
                            kind="current_surface",
                        ),
                    ),
                    entities=(),
                    effects=(),
                    constraints=(
                        ConstraintIntent(
                            constraint_id="constraint_action",
                            kind="required_action",
                            value=action,
                            authoritative=True,
                        ),
                    ),
                    subgoals=(
                        SemanticSubgoal(
                            subgoal_id=str(parsed.current_subgoal["subgoal_id"]),
                            surface_ref="surface_current",
                            status="active",
                            external_impact="navigation_only",
                            constraint_refs=("constraint_action",),
                        ),
                    ),
                ),
            )

        title = UIElement(
            element_id="page_title",
            role="text",
            meaning="page_title",
            label="文件传输助手",
            bounds=(0.35, 0.01, 0.65, 0.06),
            confidence=0.99,
            states={"goal_relevant": True, "fully_visible": True},
        )
        message_list = UIElement(
            element_id="message_list",
            role="container",
            meaning="message_list_area",
            label="聊天记录列表",
            bounds=(0.0, 0.06, 1.0, 0.7),
            confidence=0.99,
            states={
                "fully_visible": True,
                "scrollable": True,
                "scroll_axis": "vertical",
            },
            evidence=("多条聊天记录纵向排列",),
        )
        observation = trusted_observation(
            self.frames,
            elements=(title, message_list),
            observation_id="obs_17171717171717171717171717171717",
        )

        swipe_context = with_required_action("swipe")
        self.assertIsNone(_exact_text_candidate_block(swipe_context, observation))
        self.assertEqual(set(), _required_exact_candidate_ids(swipe_context, observation))
        choices = _selection_choices(
            swipe_context,
            observation,
            frozenset({"swipe"}),
        )
        self.assertEqual(
            {"up", "down", "left", "right"},
            {item["direction"] for item in choices},
        )

        missing_identity = trusted_observation(
            self.frames,
            elements=(message_list,),
            observation_id="obs_18181818181818181818181818181818",
        )
        self.assertEqual(
            "exact_text_missing",
            _exact_text_candidate_block(swipe_context, missing_identity)[1],
        )

        tap_context = with_required_action("tap_semantic")
        self.assertEqual(
            "exact_text_missing",
            _exact_text_candidate_block(tap_context, observation)[1],
        )

    def test_non_element_surface_identity_accepts_only_generic_type_suffix(self) -> None:
        raw = task_context(task_id="task_surface_suffix", revision=19)
        raw["goal"]["entities"] = {"target_ui_label": "设置页面"}
        raw["current_subgoal"].update(
            objective="收起键盘并保持在设置页面",
            completion_conditions=["软键盘已收起"],
        )
        parsed = QwenTaskContext.from_dict(raw)
        parsed = replace(
            parsed,
            semantic_ir=TaskSemanticIR(
                task_id=parsed.task_id,
                device_id=parsed.device_id,
                revision=parsed.revision,
                raw_goal="收起键盘并保持在设置页面",
                surfaces=(SurfaceRef(surface_id="surface_current", kind="current_surface"),),
                entities=(),
                effects=(),
                constraints=(
                    ConstraintIntent(
                        constraint_id="constraint_action",
                        kind="required_action",
                        value="back",
                        authoritative=True,
                    ),
                ),
                subgoals=(
                    SemanticSubgoal(
                        subgoal_id=str(parsed.current_subgoal["subgoal_id"]),
                        surface_ref="surface_current",
                        status="active",
                        external_impact="navigation_only",
                        constraint_refs=("constraint_action",),
                    ),
                ),
            ),
        )

        def identity(label: str, element_id: str = "title") -> UIElement:
            bounds = (
                (0.3, 0.01, 0.7, 0.07)
                if element_id != "title_b"
                else (0.3, 0.09, 0.7, 0.15)
            )
            return UIElement(
                element_id=element_id,
                role="text",
                meaning="page_title",
                label=label,
                bounds=bounds,
                confidence=0.99,
                states={"goal_relevant": True, "fully_visible": True},
            )

        settings = trusted_observation(self.frames, elements=(identity("设置"),))
        self.assertIsNone(_exact_text_candidate_block(parsed, settings))

        raw_chat = task_context(task_id="task_chat_suffix", revision=20)
        raw_chat["goal"]["entities"] = {"target_ui_label": "文件传输助手聊天页面"}
        raw_chat["current_subgoal"].update(
            objective="收起键盘并保持在文件传输助手聊天页面",
            completion_conditions=["软键盘已收起"],
        )
        chat_context = replace(
            QwenTaskContext.from_dict(raw_chat),
            semantic_ir=replace(parsed.semantic_ir, task_id="task_chat_suffix", revision=20),
        )
        chat = trusted_observation(
            self.frames,
            elements=(identity("文件传输助手"),),
            observation_id="obs_20202020202020202020202020202020",
        )
        self.assertIsNone(_exact_text_candidate_block(chat_context, chat))

        partial = trusted_observation(
            self.frames,
            elements=(identity("文件传输"),),
            observation_id="obs_21212121212121212121212121212121",
        )
        self.assertEqual(
            "exact_text_missing",
            _exact_text_candidate_block(chat_context, partial)[1],
        )

        ambiguous = trusted_observation(
            self.frames,
            elements=(identity("设置", "title_a"), identity("设置页面", "title_b")),
            observation_id="obs_22222222222222222222222222222222",
        )
        self.assertEqual(
            "exact_text_ambiguous",
            _exact_text_candidate_block(parsed, ambiguous)[1],
        )

    def test_page_descriptor_does_not_become_input_element_label(self) -> None:
        raw = task_context(task_id="task_page_input", revision=21)
        raw["goal"]["entities"] = {
            "target_ui_label": "文件传输助手聊天页面",
            "input_text": "live21",
        }
        raw["current_subgoal"].update(
            objective="在唯一空白消息输入框中输入 live21",
            completion_conditions=["输入框内容为 live21"],
        )
        context = QwenTaskContext.from_dict(raw)
        title = UIElement(
            element_id="title",
            role="text",
            meaning="page_title",
            label="文件传输助手",
            bounds=(0.3, 0.01, 0.7, 0.07),
            confidence=0.99,
            states={"goal_relevant": False, "fully_visible": True},
        )
        field = UIElement(
            element_id="field",
            role="input",
            meaning="application_text_input",
            label="",
            bounds=(0.1, 0.8, 0.9, 0.9),
            confidence=0.99,
            states={"goal_relevant": True, "fully_visible": True, "value": ""},
        )
        observation = trusted_observation(
            self.frames,
            elements=(title, field),
            observation_id="obs_23232323232323232323232323232323",
        )

        self.assertIsNone(_exact_text_candidate_block(context, observation))
        self.assertEqual(set(), _required_exact_candidate_ids(context, observation))

        missing_title = trusted_observation(
            self.frames,
            elements=(field,),
            observation_id="obs_24242424242424242424242424242424",
        )
        self.assertEqual(
            "exact_text_missing",
            _exact_text_candidate_block(context, missing_title)[1],
        )

    def decide(
        self,
        provider,
        *,
        context=None,
        frames=None,
        observation=None,
        available_action_kinds=None,
    ):
        observer = QwenVisualDecisionObserver(provider)
        resolved_observation = observation or self.observation
        resolved_actions = frozenset(
            available_action_kinds
            or {
                "tap_semantic",
                "dismiss_overlay",
                "swipe",
                "back",
                "home",
                "reveal_system_navigation",
                "input_verified_text",
                "clear_verified_text",
                "long_press",
                "drag",
                "wait_for_change",
            }
        )
        resolved_context = test_context_with_semantic_ir(
            context or self.context,
            resolved_observation,
            resolved_actions,
        )
        decision = observer.decide(
            frames=frames or self.frames,
            task_context=resolved_context,
            trusted_observation=resolved_observation,
            available_action_kinds=resolved_actions,
        )
        return observer, decision

    def test_real_stable_sequence_uses_four_distinct_frames(self) -> None:
        fingerprints = {hash(frame.tobytes()) for frame in self.frames}
        self.assertEqual(len(fingerprints), 4)
        self.assertTrue(self.observation.local_stability.stable)
        self.assertEqual(self.observation.local_stability.frame_count, 4)
        self.assertEqual(self.observation.scene.fingerprint, self.observation.fingerprint)

    def test_minimal_selection_hydrates_existing_formal_decision(self) -> None:
        parsed = test_context_with_semantic_ir(
            self.context,
            self.observation,
            frozenset({"tap_semantic"}),
        )
        choices = _selection_choices(
            parsed,
            self.observation,
            frozenset({"tap_semantic"}),
        )
        choice = next(
            item
            for item in choices
            if item.get("element_id") == "settings_icon"
        )
        provider = FakeProvider(
            minimal_selection_payload(
                status="action",
                choice_id=choice["choice_id"],
            )
        )

        observer, decision = self.decide(
            provider,
            available_action_kinds={"tap_semantic"},
        )

        self.assertEqual("action", decision.proposal.status)
        self.assertEqual("tap_semantic", decision.proposal.action.action)
        self.assertEqual(
            "settings_icon",
            decision.proposal.action.params["element_id"],
        )
        self.assertEqual(
            self.observation.get_candidate("settings_icon").bounds,
            decision.target_region.bounds,
        )
        self.assertEqual({"scene_changed": True}, decision.expected_result)
        self.assertEqual(1, provider.calls)
        self.assertEqual(
            {"type": "json_object"},
            provider.last_call_options["response_format"],
        )
        prompt = provider.messages[-1]["content"][0]["text"]
        self.assertIn("choice_id", prompt)
        self.assertIn("completes_current_subgoal_on_success", prompt)
        self.assertIn('"expected_result":{"scene_changed":true}', prompt)
        self.assertIn("禁止复制、改写或另行输出", prompt)
        self.assertIn("不要identity", prompt)
        self.assertNotIn('"protocol_version":"逐字复制输入"', prompt)

    def test_minimal_terminal_selection_binds_only_the_completion_claim(self) -> None:
        parsed = test_context_with_semantic_ir(
            self.context,
            self.observation,
            frozenset({"tap_semantic"}),
        )
        choices = _selection_choices(
            parsed,
            self.observation,
            frozenset({"tap_semantic"}),
        )
        choice = next(
            item
            for item in choices
            if item.get("element_id") == "settings_icon"
        )
        provider = FakeProvider(
            minimal_selection_payload(
                status="action",
                choice_id=choice["choice_id"],
                completes_current_subgoal_on_success=True,
            )
        )

        _observer, decision = self.decide(
            provider,
            available_action_kinds={"tap_semantic"},
        )

        self.assertEqual(
            {
                "scene_changed": True,
                "goal_complete_on_success": True,
            },
            decision.expected_result,
        )
        self.assertEqual(
            decision.expected_result,
            decision.proposal.action.params["expected_effect"],
        )

    def test_minimal_finished_cannot_claim_future_action_completion(self) -> None:
        context = task_context()
        context["current_execution_class"] = "observe"
        context["current_subgoal"]["execution_class"] = "observe"
        provider = FakeProvider(
            minimal_selection_payload(
                status="finished",
                completes_current_subgoal_on_success=True,
                completion_evidence_element_ids=["scene"],
            )
        )

        observer, decision = self.decide(provider, context=context)

        self.assertEqual("blocked", decision.proposal.status)
        self.assertEqual(1, provider.calls)
        self.assertFalse(observer.last_diagnostics["protocol_retry_used"])
        self.assertIn("动作后完成当前子目标", decision.reason)

    def test_minimal_selection_invalid_choice_fails_closed_without_retry(self) -> None:
        provider = FakeProvider(
            minimal_selection_payload(
                status="action",
                choice_id="invented_choice",
            )
        )

        observer, decision = self.decide(provider)

        self.assertEqual("blocked", decision.proposal.status)
        self.assertEqual(1, provider.calls)
        self.assertFalse(observer.last_diagnostics["protocol_retry_used"])
        self.assertIn("choice_id", decision.reason)

    def test_minimal_selection_requires_explicit_subgoal_completion_boolean(self) -> None:
        payload = minimal_selection_payload(
            status="action",
            choice_id="choice_1",
        )
        payload.pop("completes_current_subgoal_on_success")

        observer, decision = self.decide(FakeProvider(payload))

        self.assertEqual("blocked", decision.proposal.status)
        self.assertEqual(1, observer.last_diagnostics["model_calls"])
        self.assertFalse(observer.last_diagnostics["protocol_retry_used"])
        self.assertIn("completes_current_subgoal_on_success", decision.reason)

    def test_minimal_finished_uses_current_trusted_scene(self) -> None:
        context = task_context()
        context["current_execution_class"] = "observe"
        context["current_subgoal"]["execution_class"] = "observe"
        provider = FakeProvider(
            minimal_selection_payload(
                status="finished",
                completion_evidence_element_ids=["scene"],
            )
        )

        _observer, decision = self.decide(provider, context=context)

        self.assertEqual("finished", decision.proposal.status)
        self.assertEqual(
            (f"scene:{self.observation.scene.summary}",),
            decision.proposal.completion_evidence,
        )

    def test_minimal_input_choice_binds_deepseek_text_locally(self) -> None:
        context = task_context(task_id="task_minimal_input", revision=3)
        context["goal"]["entities"] = {"input_text": "agent"}
        context["current_subgoal"]["objective"] = "在当前输入框输入目标文字"
        field = UIElement(
            element_id="query_field",
            role="input",
            meaning="current_text_input",
            label="",
            bounds=(0.08, 0.12, 0.92, 0.22),
            confidence=0.97,
            states={
                "focused": True,
                "value": "",
                "fully_visible": True,
                "keyboard_layout": "qwerty",
                "keyboard_input_mode": "direct_latin",
                "goal_relevant": True,
            },
            evidence=("英文直输输入框已聚焦",),
        )
        observation = trusted_observation(self.frames, elements=(field,))
        provider = FakeProvider(
            minimal_selection_payload(
                status="action",
                choice_id="choice_1",
            )
        )

        _observer, decision = self.decide(
            provider,
            context=context,
            observation=observation,
            available_action_kinds={"input_verified_text"},
        )

        self.assertEqual("input_verified_text", decision.proposal.action.action)
        self.assertEqual("agent", decision.proposal.action.params["text"])
        self.assertEqual(
            {
                "element_state": {
                    "meaning": "current_text_input",
                    "states": {"value": "agent"},
                }
            },
            decision.expected_result,
        )

        exact_context = copy.deepcopy(context)
        exact_context["goal"]["entities"]["target_ui_label"] = "长文本"
        exact_context["current_subgoal"]["objective"] = (
            "在当前唯一的长文本输入框输入目标文字"
        )
        exact_field = replace(field, label="长文本")
        exact_observation = trusted_observation(
            self.frames,
            elements=(exact_field,),
            observation_id="obs_eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
        )
        _observer, exact_decision = self.decide(
            FakeProvider(
                minimal_selection_payload(status="action", choice_id="choice_1")
            ),
            context=exact_context,
            observation=exact_observation,
            available_action_kinds={"input_verified_text"},
        )
        self.assertEqual("action", exact_decision.proposal.status)
        self.assertEqual(
            "input_verified_text", exact_decision.proposal.action.action
        )
        self.assertEqual(
            "query_field", exact_decision.proposal.action.params["element_id"]
        )

    def test_typed_input_field_remains_exact_target_after_placeholder_disappears(self) -> None:
        target_text = "abcdefghijklmnopqrstuvwxyzabcdefghijk"
        prefix = "abcdefghijklmnopqrst"
        raw = task_context(task_id="task_input_continuity", revision=8)
        raw["goal"]["entities"] = {
            "input_text": target_text,
            "target_ui_label": "长文本",
        }
        raw["current_subgoal"]["objective"] = "继续在长文本输入框输入剩余文字"
        field = UIElement(
            element_id="local_audited_input_1",
            role="input",
            meaning="application_text_input",
            label=prefix,
            bounds=(0.135, 0.49, 0.865, 0.615),
            confidence=0.99,
            states={
                "focused": True,
                "value": prefix,
                "fully_visible": True,
                "goal_relevant": True,
                "keyboard_layout": "qwerty",
                "keyboard_input_mode": "direct_latin",
                "input_field_id": "field_test_input",
            },
            evidence=(f"应用输入框当前文字：{prefix}",),
        )
        observation = trusted_observation(
            self.frames,
            elements=(field,),
            observation_id="obs_12121212121212121212121212121212",
        )
        context = test_context_with_semantic_ir(
            raw,
            observation,
            frozenset({"input_verified_text"}),
        )
        context = replace(
            context,
            semantic_ir=replace(
                context.semantic_ir,
                input_fields=(
                    replace(
                        context.semantic_ir.input_fields[0],
                        field_id="input_field_1",
                    ),
                ),
            ),
        )
        field = replace(
            field,
            states={**field.states, "input_field_id": "input_field_1"},
        )
        observation = trusted_observation(
            self.frames,
            elements=(field,),
            observation_id="obs_12121212121212121212121212121212",
        )
        self.assertIsNone(_exact_text_candidate_block(context, observation))
        self.assertEqual(
            {"local_audited_input_1"},
            _required_exact_candidate_ids(context, observation),
        )
        _observer, decision = self.decide(
            FakeProvider(
                minimal_selection_payload(status="action", choice_id="choice_1")
            ),
            context=context,
            observation=observation,
            available_action_kinds={"input_verified_text"},
        )
        self.assertEqual("action", decision.proposal.status)
        self.assertEqual(target_text, decision.proposal.action.params["text"])

        for changed_states in (
            {"input_field_id": "field_other"},
            {"value": "wrong-prefix"},
        ):
            wrong_field = replace(
                field,
                label=str(changed_states.get("value", prefix)),
                states={**field.states, **changed_states},
            )
            wrong_observation = trusted_observation(
                self.frames,
                elements=(wrong_field,),
                observation_id=(
                    "obs_34343434343434343434343434343434"
                    if "value" not in changed_states
                    else "obs_56565656565656565656565656565656"
                ),
            )
            self.assertEqual(
                "exact_text_missing",
                _exact_text_candidate_block(context, wrong_observation)[1],
            )

    def test_typed_input_prefix_identity_varies_by_field_label_and_payload(self) -> None:
        target_text = "releasecandidatecontinuation"
        prefix = "releasecandidate"
        raw = task_context(task_id="task_input_continuity_variation", revision=5)
        raw["goal"]["entities"] = {
            "input_text": target_text,
            "target_ui_label": "备注",
        }
        raw["current_subgoal"]["objective"] = "继续填写备注字段"
        field = UIElement(
            element_id="local_audited_notes_input",
            role="input",
            meaning="application_text_input",
            label=prefix,
            bounds=(0.12, 0.31, 0.88, 0.46),
            confidence=0.98,
            states={
                "focused": True,
                "value": prefix,
                "fully_visible": True,
                "goal_relevant": True,
                "keyboard_layout": "qwerty",
                "keyboard_input_mode": "direct_latin",
                "input_field_id": "notes_field",
                "input_field_label": "备注",
            },
            evidence=(f"备注字段当前文字：{prefix}",),
        )
        observation = trusted_observation(
            self.frames,
            elements=(field,),
            observation_id="obs_78787878787878787878787878787878",
        )
        context = test_context_with_semantic_ir(
            raw,
            observation,
            frozenset({"input_verified_text"}),
        )
        context = replace(
            context,
            semantic_ir=replace(
                context.semantic_ir,
                input_fields=(
                    replace(
                        context.semantic_ir.input_fields[0],
                        field_id="notes_field",
                        field_label="备注",
                    ),
                ),
            ),
        )
        self.assertIsNone(_exact_text_candidate_block(context, observation))
        self.assertEqual(
            {"local_audited_notes_input"},
            _required_exact_candidate_ids(context, observation),
        )

        unrelated_nonempty = replace(
            field,
            element_id="other_input",
            label="draft",
            states={
                **field.states,
                "value": "draft",
                "input_field_id": "other_field",
                "input_field_label": "标题",
            },
        )
        empty_active = replace(
            field,
            label="备注",
            states={**field.states, "value": ""},
        )
        empty_observation = trusted_observation(
            self.frames,
            elements=(empty_active, unrelated_nonempty),
            observation_id="obs_89898989898989898989898989898989",
        )
        self.assertIsNone(_exact_text_candidate_block(context, empty_observation))
        self.assertEqual(
            {"local_audited_notes_input"},
            _required_exact_candidate_ids(context, empty_observation),
        )

    def test_minimal_clear_choice_uses_local_empty_postcondition_without_text(self) -> None:
        context = task_context(task_id="task_minimal_clear", revision=4)
        context["current_subgoal"]["objective"] = "恢复唯一错误草稿输入框为空"
        field = UIElement(
            element_id="draft_field",
            role="input",
            meaning="draft_input",
            label="",
            bounds=(0.08, 0.12, 0.92, 0.22),
            confidence=0.97,
            states={
                "focused": True,
                "value": "lxs,",
                "fully_visible": True,
                "keyboard_layout": "qwerty",
                "keyboard_input_mode": "direct_latin",
                "goal_relevant": True,
            },
            evidence=("唯一输入框中逐字可见 lxs,",),
        )
        observation = trusted_observation(self.frames, elements=(field,))

        _observer, decision = self.decide(
            FakeProvider(
                minimal_selection_payload(status="action", choice_id="choice_1")
            ),
            context=context,
            observation=observation,
            available_action_kinds={"clear_verified_text"},
        )

        self.assertEqual("clear_verified_text", decision.proposal.action.action)
        self.assertNotIn("text", decision.proposal.action.params)
        self.assertEqual(
            {
                "element_state": {
                    "meaning": "draft_input",
                    "states": {"value": ""},
                }
            },
            decision.expected_result,
        )

    def test_minimal_press_enter_choice_uses_local_newline_candidate(self) -> None:
        context = task_context(task_id="task_minimal_enter", revision=4)
        context["goal"]["entities"] = {"input_text": "first\nsecond"}
        context["current_subgoal"]["objective"] = "在当前多行正文中插入真实换行"
        field = UIElement(
            element_id="field",
            role="input",
            meaning="application_text_input",
            label="正文",
            bounds=(0.08, 0.1, 0.92, 0.3),
            confidence=0.98,
            states={
                "focused": True,
                "value": "first",
                "input_multiline": True,
                "fully_visible": True,
                "goal_relevant": False,
            },
            evidence=("正文多行输入框已聚焦",),
        )
        enter = UIElement(
            element_id="enter",
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
                "input_element_id": "field",
            },
            evidence=("本地输入结构审计确认换行键",),
        )
        observation = trusted_observation(self.frames, elements=(field, enter))
        _observer, decision = self.decide(
            FakeProvider(
                minimal_selection_payload(status="action", choice_id="choice_1")
            ),
            context=context,
            observation=observation,
            available_action_kinds={"press_enter"},
        )
        self.assertEqual("press_enter", decision.proposal.action.action)
        self.assertEqual("enter", decision.proposal.action.params["element_id"])
        self.assertEqual(
            {
                "element_state": {
                    "meaning": "application_text_input",
                    "states": {"value": "first\n"},
                }
            },
            decision.expected_result,
        )

    def test_explicit_clear_subgoal_offers_only_verified_clear(self) -> None:
        context = task_context(task_id="task_clear_only", revision=5)
        context["current_subgoal"]["objective"] = "当前唯一临时草稿区域内容为空白"
        context["current_subgoal"]["completion_conditions"] = [
            "草稿区域显示为空白",
            "键盘仍然可见",
        ]
        field = UIElement(
            element_id="draft_field",
            role="input",
            meaning="unique_temporary_draft_area",
            label="lxs,",
            bounds=(0.08, 0.12, 0.92, 0.22),
            confidence=0.97,
            states={
                "focused": True,
                "value": "lxs,",
                "fully_visible": True,
                "keyboard_layout": "qwerty",
                "keyboard_input_mode": "direct_latin",
                "goal_relevant": True,
            },
            evidence=("唯一输入框中逐字可见 lxs,",),
        )
        observation = trusted_observation(self.frames, elements=(field,))

        observer, decision = self.decide(
            FakeProvider(
                minimal_selection_payload(status="action", choice_id="choice_1")
            ),
            context=context,
            observation=observation,
            available_action_kinds={
                "back",
                "clear_verified_text",
                "home",
                "long_press",
                "swipe",
                "tap_semantic",
            },
        )

        self.assertEqual("clear_verified_text", decision.proposal.action.action)
        self.assertEqual(
            ["clear_verified_text"],
            observer.last_diagnostics["available_action_kinds"],
        )

    def test_minimal_selection_rejects_model_authored_expected_result(self) -> None:
        payload = minimal_selection_payload(
            status="action",
            choice_id="choice_1",
        )
        payload["expected_result"] = {
            "element_state": {"query_field": {"focused": True}}
        }

        observer, decision = self.decide(FakeProvider(payload))

        self.assertEqual("blocked", decision.proposal.status)
        self.assertEqual(1, observer.last_diagnostics["model_calls"])
        self.assertFalse(observer.last_diagnostics["protocol_retry_used"])
        self.assertIn("expected_result", decision.reason)

    def test_truncated_minimal_json_blocks_after_one_call(self) -> None:
        provider = RawSequenceProvider(['{"status":"action"'])

        observer, decision = self.decide(provider)

        self.assertEqual("blocked", decision.proposal.status)
        self.assertEqual(1, provider.calls)
        self.assertFalse(observer.last_diagnostics["protocol_retry_used"])
        self.assertIn("JSON", decision.reason)

    def test_input_action_must_copy_structured_text_exactly(self) -> None:
        context = task_context()
        context["goal"]["entities"] = {"input_text": "agent"}
        context["current_subgoal"]["objective"] = "在已聚焦输入框输入查询词"
        field = UIElement(
            element_id="query_field",
            role="input",
            meaning="搜索输入框",
            label="搜索",
            bounds=(0.08, 0.12, 0.92, 0.22),
            confidence=0.97,
            states={
                "focused": True,
                "value": "",
                "fully_visible": True,
                "keyboard_layout": "qwerty",
                "keyboard_input_mode": "direct_latin",
                "goal_relevant": True,
            },
            evidence=("输入光标可见",),
        )
        observation = trusted_observation(self.frames, elements=(field,))
        payload = action_payload(context, observation, element_id="query_field")
        payload["next_action"].update(
            {"kind": "input_verified_text", "text": "agent"}
        )
        payload["expected_result"] = {
            "element_state": {
                "meaning": "搜索输入框",
                "states": {"value": "agent"},
            }
        }

        _observer, decision = self.decide(
            FakeProvider(payload),
            context=context,
            observation=observation,
        )

        self.assertEqual("input_verified_text", decision.proposal.action.action)
        self.assertEqual("agent", decision.proposal.action.params["text"])

    def test_visible_keyboard_dismissal_offers_qwen_only_certified_back(self) -> None:
        context = task_context(task_id="task_hide_keyboard", revision=19)
        context["current_subgoal"]["objective"] = "收起当前已显示的软键盘"
        context["current_subgoal"]["completion_conditions"] = ["软键盘不可见"]
        field = UIElement(
            element_id="query_field",
            role="input",
            meaning="current_text_input",
            label="agent",
            bounds=(0.08, 0.12, 0.92, 0.22),
            confidence=0.97,
            states={
                "focused": True,
                "value": "agent",
                "fully_visible": True,
                "keyboard_layout": "qwerty",
                "keyboard_input_mode": "chinese_pinyin",
                "goal_relevant": True,
            },
            evidence=("输入光标和QWERTY软键盘可见",),
        )
        key = UIElement(
            element_id="keyboard_done_key",
            role="keyboard_key",
            meaning="dismiss_keyboard",
            label="⌄",
            bounds=(0.88, 0.86, 0.98, 0.96),
            confidence=0.95,
            states={"goal_relevant": True},
            evidence=("键盘右下角按键",),
        )
        observation = trusted_observation(self.frames, elements=(field, key))
        provider = FakeProvider(
            minimal_selection_payload(
                status="action",
                choice_id="choice_1",
            )
        )

        observer, decision = self.decide(
            provider,
            context=context,
            observation=observation,
        )

        self.assertEqual(1, provider.calls)
        self.assertEqual("action", decision.proposal.status)
        self.assertEqual("back", decision.proposal.action.action)
        self.assertEqual({"scene_changed": True}, decision.expected_result)
        self.assertNotIn("selection_context", decision.proposal.action.params)
        choices = _selection_choices(
            test_context_with_semantic_ir(
                context,
                observation,
                frozenset({"back"}),
            ),
            observation,
            frozenset({"back"}),
        )
        self.assertEqual(1, len(choices))
        self.assertNotIn("selection_context", choices[0])
        self.assertEqual(["back"], observer.last_diagnostics["available_action_kinds"])
        self.assertFalse(observer.last_diagnostics["protocol_retry_used"])

    def test_keyboard_dismissal_contract_ignores_non_active_goal_mentions(self) -> None:
        context = task_context(task_id="task_keep_input", revision=20)
        context["goal"]["objective"] = "输入agent后收起软键盘"
        context["current_subgoal"]["objective"] = "点击当前输入框使其聚焦"
        context["current_subgoal"]["completion_conditions"] = ["输入框已聚焦"]
        field = UIElement(
            element_id="query_field",
            role="input",
            meaning="current_text_input",
            label="",
            bounds=(0.08, 0.12, 0.92, 0.22),
            confidence=0.97,
            states={
                "goal_relevant": True,
                "fully_visible": True,
                "focused": False,
                "value": "",
            },
            evidence=("空输入框可见",),
        )
        observation = trusted_observation(self.frames, elements=(field,))
        provider = FakeProvider(
            action_payload(context, observation, element_id="query_field")
        )

        observer, decision = self.decide(
            provider,
            context=context,
            observation=observation,
        )

        self.assertEqual("tap_semantic", decision.proposal.action.action)
        self.assertIn(
            "tap_semantic",
            observer.last_diagnostics["available_action_kinds"],
        )
        choices = _selection_choices(
            test_context_with_semantic_ir(
                context,
                observation,
                frozenset({"back", "tap_semantic"}),
            ),
            observation,
            frozenset({"back", "tap_semantic"}),
        )
        self.assertTrue(choices)
        self.assertTrue(
            all("selection_context" not in choice for choice in choices)
        )

    def test_keyboard_key_is_never_an_element_action_target(self) -> None:
        key = UIElement(
            element_id="keyboard_candidate",
            role="keyboard_key",
            meaning="candidate_shortcut",
            label=".com",
            bounds=(0.10, 0.70, 0.28, 0.77),
            confidence=0.93,
            states={"goal_relevant": True},
            evidence=(".com",),
        )
        observation = trusted_observation(self.frames, elements=(key,))
        payload = action_payload(
            self.context,
            observation,
            element_id="keyboard_candidate",
        )
        provider = SequenceProvider([payload, payload])

        _observer, decision = self.decide(provider, observation=observation)

        self.assertEqual("blocked", decision.proposal.status)
        self.assertIn("keyboard_key", decision.reason)

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
            states={
                "focused": True,
                "value": "",
                "fully_visible": True,
                "keyboard_layout": "qwerty",
                "keyboard_input_mode": "chinese_pinyin",
                "goal_relevant": True,
            },
            evidence=("输入光标可见",),
        )
        observation = trusted_observation(self.frames, elements=(field,))
        payload = action_payload(context, observation, element_id="query_field")
        payload["next_action"].update(
            {"kind": "input_verified_text", "text": "打开蓝牙设置"}
        )
        payload["expected_result"] = {
            "element_state": {
                "meaning": "搜索输入框",
                "states": {
                    "value": "",
                    "ime_preedit_text": "lanyashezhi",
                    "ime_exact_candidate_text": "蓝牙设置",
                },
            }
        }

        provider = FakeProvider(payload)
        observer, decision = self.decide(
            provider,
            context=context,
            observation=observation,
        )

        self.assertEqual("blocked", decision.proposal.status)
        self.assertIn("input_text", decision.proposal.reason)
        self.assertEqual(1, provider.calls)
        self.assertFalse(observer.last_diagnostics["protocol_retry_used"])

    def test_exact_ime_candidate_is_a_separate_locally_bound_choice(self) -> None:
        context = task_context()
        context["goal"]["entities"] = {"input_text": "你好"}
        context["current_subgoal"]["objective"] = "在消息输入框输入你好但不要发送"
        field = UIElement(
            element_id="field",
            role="input",
            meaning="application_text_input",
            label="消息",
            bounds=(0.08, 0.12, 0.92, 0.22),
            confidence=0.97,
            states={
                "focused": True,
                "value": "",
                "fully_visible": True,
                "keyboard_layout": "qwerty",
                "keyboard_input_mode": "chinese_pinyin",
                "ime_preedit_text": "nihao",
                "ime_exact_candidate_text": "你好",
                "goal_relevant": False,
            },
        )
        candidate = UIElement(
            element_id="local_audited_ime_candidate_1",
            role="button",
            meaning="ime_exact_candidate",
            label="你好",
            bounds=(0.08, 0.42, 0.22, 0.48),
            confidence=0.98,
            states={
                "goal_relevant": True,
                "fully_visible": True,
                "ime_candidate": True,
                "input_element_id": "field",
                "prior_input_value": "",
                "expected_input_value": "你好",
                "pinyin": "nihao",
            },
            evidence=("拼音nihao的唯一逐字候选你好",),
        )
        observation = trusted_observation(self.frames, elements=(field, candidate))
        provider = FakeProvider(
            minimal_selection_payload(status="action", choice_id="choice_1")
        )

        _observer, decision = self.decide(
            provider,
            context=context,
            observation=observation,
            available_action_kinds={"tap_semantic", "input_verified_text"},
        )

        self.assertEqual("tap_semantic", decision.proposal.action.action)
        self.assertEqual(
            "local_audited_ime_candidate_1",
            decision.proposal.action.params["element_id"],
        )
        self.assertEqual(
            {
                "element_state": {
                    "meaning": "application_text_input",
                    "states": {"value": "你好"},
                }
            },
            decision.expected_result,
        )

    def test_literal_key_and_keyboard_switches_are_local_exact_choices(self) -> None:
        context = task_context()
        context["goal"]["entities"] = {
            "input_text": "draft 8",
            "target_ui_label": "消息",
        }
        context["current_subgoal"]["objective"] = "消息草稿逐字为draft 8"
        parsed = QwenTaskContext.from_dict(context)
        field = UIElement(
            element_id="field", role="input", meaning="application_text_input",
            label="消息", bounds=(0.08, 0.12, 0.92, 0.22), confidence=0.97,
            states={
                "focused": True, "value": "draft", "keyboard_layout": "qwerty",
                "keyboard_input_mode": "direct_latin", "goal_relevant": False,
            },
        )
        literal = UIElement(
            element_id="local_audited_literal_key_1", role="button",
            meaning="input_exact_literal_key", label="空格",
            bounds=(0.31, 0.87, 0.69, 0.97), confidence=0.98,
            states={
                "goal_relevant": True, "fully_visible": True,
                "input_literal_key": True, "key_value": " ",
                "prior_input_value": "draft", "expected_input_value": "draft ",
                "input_element_id": "field",
            },
        )
        mode_switch = UIElement(
            element_id="local_audited_keyboard_mode_switch_1",
            role="button",
            meaning="switch_keyboard_input_mode",
            label="中",
            bounds=(0.72, 0.87, 0.82, 0.97),
            confidence=0.98,
            states={
                "goal_relevant": True,
                "fully_visible": True,
                "keyboard_input_mode_switch": True,
                "current_mode": "direct_latin",
                "target_mode": "chinese_pinyin",
                "prior_input_value": "draft",
                "next_input_value": "你好",
                "input_element_id": "field",
            },
        )
        observation = trusted_observation(
            self.frames,
            elements=(field, literal, mode_switch),
        )
        parsed = test_context_with_semantic_ir(
            context,
            observation,
            frozenset({"tap_semantic", "input_verified_text"}),
        )
        choices = _selection_choices(
            parsed, observation, frozenset({"tap_semantic", "input_verified_text"})
        )
        self.assertFalse(
            any(item["action"] == "input_verified_text" for item in choices),
            "a visible literal-key step must not expose an invalid batch-input choice",
        )
        choice = next(item for item in choices if item["element_id"] == literal.element_id)
        self.assertEqual("tap_semantic", choice["action"])
        self.assertEqual(
            {"element_state": {"meaning": "application_text_input", "states": {"value": "draft "}}},
            choice["expected_result"],
        )
        mode_choice = next(
            item for item in choices if item["element_id"] == mode_switch.element_id
        )
        self.assertEqual("tap_semantic", mode_choice["action"])
        self.assertEqual(
            {
                "element_state": {
                    "meaning": "application_text_input",
                    "states": {
                        "value": "draft",
                        "keyboard_input_mode": "chinese_pinyin",
                    },
                }
            },
            mode_choice["expected_result"],
        )
        available = frozenset({"tap_semantic", "input_verified_text", "clear_verified_text"})
        literal_choice = next(
            item for item in choices if item["element_id"] == literal.element_id
        )
        decision = QwenVisualDecisionObserver(
            FakeProvider(
                minimal_selection_payload(
                    status="action",
                    choice_id=literal_choice["choice_id"],
                )
            )
        ).decide(
            frames=self.frames,
            task_context=parsed,
            trusted_observation=observation,
            available_action_kinds=available,
        )
        rebuilt = compile_canonical_action_catalog(
            observation.scene,
            parsed.semantic_ir,
            available,
        )
        self.assertEqual(
            rebuilt.report_digest,
            decision.proposal.action.params["formal_report_digest"],
        )

    def test_search_result_prohibition_keeps_unique_unfocused_input_focus_choice(self) -> None:
        raw_goal = "在搜索输入框输入wifi，不得选择任何搜索结果"
        literal_start = raw_goal.index("wifi")
        payload = SemanticEntity(
            entity_id="entity_input_text",
            entity_type="text",
            role="input_text",
            value="wifi",
            source_span=SourceSpan(literal_start, literal_start + 4),
            authority="user_literal",
        )
        effect = EffectIntent(
            effect_id="effect_input_text",
            kind="input_text",
            payload_refs=(payload.entity_id,),
            source_subgoal_ids=("input_wifi",),
        )
        required_action = ConstraintIntent(
            constraint_id="constraint_input_action",
            kind="required_action",
            value="input_verified_text",
            source_text="输入wifi",
            authoritative=True,
        )
        semantic_ir = TaskSemanticIR(
            task_id="task_offline_01",
            device_id="offline_phone_01",
            revision=3,
            raw_goal=raw_goal,
            surfaces=(
                SurfaceRef(
                    "surface_settings",
                    "app",
                    app_id="settings",
                    app_name="设置",
                ),
            ),
            entities=(payload,),
            effects=(effect,),
            constraints=(required_action,),
            subgoals=(
                SemanticSubgoal(
                    subgoal_id="input_wifi",
                    surface_ref="surface_settings",
                    status="active",
                    external_impact="navigation_only",
                    constraint_refs=(required_action.constraint_id,),
                    effect_refs=(effect.effect_id,),
                ),
            ),
            input_fields=(
                InputFieldIntent(
                    field_id="field_search",
                    payload_ref=payload.entity_id,
                    source_subgoal_ids=("input_wifi",),
                ),
            ),
        )
        context = task_context()
        context["goal"]["objective"] = raw_goal
        context["goal"]["entities"] = {"input_text": "wifi"}
        context["current_subgoal"].update(
            {
                "subgoal_id": "input_wifi",
                "objective": "在搜索输入框中输入 wifi",
                "constraints": ["不得提交搜索", "不得选择任何搜索结果"],
                "completion_conditions": ["输入框中显示 'wifi'"],
            }
        )
        context["effect_gate"]["scope"]["subgoal_id"] = "input_wifi"
        parsed = replace(QwenTaskContext.from_dict(context), semantic_ir=semantic_ir)
        field = UIElement(
            element_id="local_audited_input_1",
            role="input",
            meaning="application_text_input",
            label="搜索系统设置项",
            bounds=(0.12, 0.15, 0.9, 0.21),
            confidence=0.95,
            states={
                "goal_relevant": True,
                "fully_visible": True,
                "value": "",
                "soft_keyboard_visible": False,
            },
            evidence=("应用输入框为空", "search icon"),
        )
        page_title = UIElement(
            element_id="page-title",
            role="text",
            meaning="page_title",
            label="设置",
            bounds=(0.12, 0.08, 0.3, 0.14),
            confidence=1.0,
            states={"goal_relevant": False, "fully_visible": True},
        )
        observation = trusted_observation(
            self.frames,
            scene=scene_for(
                self.frames,
                elements=(field, page_title),
                app_id="com.android.settings",
                screen_id="settings_main",
                summary="设置主页面",
            ),
        )

        choices = _selection_choices(
            parsed,
            observation,
            frozenset({"tap_semantic", "input_verified_text", "home"}),
        )

        focus = [item for item in choices if item["action"] == "tap_semantic"]
        self.assertEqual(1, len(focus))
        self.assertEqual(field.element_id, focus[0]["element_id"])
        self.assertEqual(
            {
                "element_state": {
                    "meaning": "application_text_input",
                    "states": {"focused": True},
                }
            },
            focus[0]["expected_result"],
        )
        self.assertNotIn("input_verified_text", [item["action"] for item in choices])

    def test_recipient_title_is_identity_evidence_not_input_action_target(self) -> None:
        context = task_context()
        context["goal"]["entities"] = {
            "recipient": "张三",
            "input_text": "agent",
        }
        context["current_subgoal"]["objective"] = "在张三的聊天页输入消息草稿"
        title = UIElement(
            element_id="conversation-title",
            role="text",
            meaning="conversation_identity",
            label="张三",
            bounds=(0.35, 0.03, 0.65, 0.09),
            confidence=0.98,
            states={"fully_visible": True, "goal_relevant": False},
            evidence=("聊天页标题逐字显示张三",),
        )
        field = UIElement(
            element_id="message-field",
            role="input",
            meaning="application_text_input",
            label="消息",
            bounds=(0.08, 0.12, 0.92, 0.22),
            confidence=0.97,
            states={
                "focused": True,
                "value": "",
                "fully_visible": True,
                "keyboard_layout": "qwerty",
                "keyboard_input_mode": "direct_latin",
                "goal_relevant": True,
            },
        )
        observation = trusted_observation(self.frames, elements=(title, field))
        provider = FakeProvider(
            minimal_selection_payload(status="action", choice_id="choice_1")
        )

        _observer, decision = self.decide(
            provider,
            context=context,
            observation=observation,
            available_action_kinds={"input_verified_text"},
        )

        self.assertEqual("input_verified_text", decision.proposal.action.action)
        self.assertEqual("message-field", decision.proposal.action.params["element_id"])

        missing_identity = trusted_observation(self.frames, elements=(field,))
        observer, blocked = self.decide(
            FakeProvider(minimal_selection_payload(status="action", choice_id="choice_1")),
            context=context,
            observation=missing_identity,
            available_action_kinds={"input_verified_text"},
        )
        self.assertEqual("blocked", blocked.proposal.status)
        self.assertIn("收件人", blocked.proposal.reason)
        self.assertEqual("identity_missing", observer.last_diagnostics["local_safety_block"])

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
                "2026-08-20-deepseek-typed-task-graph-v4",
            )
            self.assertEqual(parsed.device_id, "offline_phone_01")
            for field in (
                "protocol_version",
                "task_id",
                "device_id",
                "revision",
                "current_subgoal",
                "global_constraints",
                "current_execution_class",
                "effect_intents",
                "effect_gate",
            ):
                self.assertIn(field, case["task_context"])
            scope = case["task_context"]["effect_gate"]["scope"]
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

    def test_low_scene_confidence_allows_only_unique_strong_goal_element(self) -> None:
        low_scene = replace(
            scene_for(self.frames),
            confidence=0.6,
        )
        observation = trusted_observation(self.frames, scene=low_scene)
        payload = action_payload(self.context, observation)

        _observer, decision = self.decide(
            FakeProvider(payload),
            observation=observation,
        )

        self.assertEqual("action", decision.proposal.status)
        self.assertEqual(0.92, decision.confidence)
        self.assertEqual(
            "settings_icon",
            observation.target_local_candidate().element_id,
        )

    def test_low_scene_confidence_rejects_multiple_goal_elements(self) -> None:
        second = replace(
            launcher_elements()[1],
            element_id="other_goal",
            states={"goal_relevant": True},
        )
        low_scene = replace(
            scene_for(self.frames),
            elements=(launcher_elements()[0], second),
            confidence=0.6,
        )

        with self.assertRaisesRegex(VisionAgentError, "整体置信度不足"):
            trusted_observation(self.frames, scene=low_scene)

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
            "current_execution_class",
            "effect_intents",
            "effect_gate",
        ):
            self.assertIn(field, parsed.to_dict())

    def test_prompt_requires_completion_check_before_any_action(self) -> None:
        from qwen_visual_decision import _decision_prompt

        prompt = _decision_prompt(
            QwenTaskContext.from_dict(self.context),
            self.observation,
            decision_number=1,
            available_action_kinds=frozenset({"tap_semantic"}),
        )

        self.assertIn("必须先做完成判定", prompt)
        self.assertIn("禁止再点击", prompt)
        self.assertIn("已选中tab", prompt)

    def test_prompt_treats_negative_constraints_as_candidate_filter(self) -> None:
        from qwen_visual_decision import _decision_prompt

        prompt = _decision_prompt(
            QwenTaskContext.from_dict(self.context),
            self.observation,
            decision_number=1,
            available_action_kinds=frozenset({"tap_semantic", "back"}),
        )

        self.assertIn("候选选择前的硬过滤条件", prompt)
        self.assertIn("即使它看起来是最短路径", prompt)
        self.assertIn("应使用无element_id、无坐标的back", prompt)
        self.assertIn("绝不能把back伪装成页面元素tap_semantic", prompt)

    def test_negative_constraint_removes_candidate_from_model_surface_only(self) -> None:
        from qwen_visual_decision import _decision_observation_prompt_dict

        candidate = self.observation.scene.elements[0]
        context = json.loads(json.dumps(self.context, ensure_ascii=False))
        context["current_subgoal"]["constraints"] = [
            f"不要再次使用{candidate.label}入口"
        ]
        prompt_observation = _decision_observation_prompt_dict(
            QwenTaskContext.from_dict(context),
            self.observation,
        )

        self.assertNotIn(
            candidate.element_id,
            {item["element_id"] for item in prompt_observation["candidates"]},
        )
        self.assertEqual(
            candidate.element_id,
            self.observation.scene.elements[0].element_id,
        )

    def test_keyboard_geometry_is_local_credential_not_qwen_prompt_data(self) -> None:
        from qwen_visual_decision import _decision_observation_prompt_dict

        geometry = {
            "type": "qwerty",
            "anchors": {
                "q": [115, 704], "p": [875, 704],
                "a": [157, 773], "l": [832, 773],
                "z": [241, 844], "m": [747, 844],
                "backspace": [875, 844],
            },
            "source": "input_structure_audit",
        }
        input_element = UIElement(
            element_id="audited-input",
            role="input",
            meaning="application_text_input",
            label="输入",
            bounds=(0.1, 0.1, 0.9, 0.2),
            confidence=0.98,
            states={
                "goal_relevant": True,
                "focused": True,
                "value": "",
                "keyboard_layout": "qwerty",
                "keyboard_input_mode": "direct_latin",
                "keyboard_geometry": geometry,
            },
        )
        scene = replace(
            self.observation.scene,
            elements=(input_element,),
            fingerprint=self.observation.fingerprint,
        )
        observation = replace(self.observation, scene=scene)

        projected = _decision_observation_prompt_dict(
            QwenTaskContext.from_dict(self.context),
            observation,
        )

        self.assertNotIn("keyboard_geometry", projected["candidates"][0]["states"])
        self.assertEqual(
            geometry,
            observation.scene.elements[0].states["keyboard_geometry"],
        )

    def test_page_element_scope_constraint_removes_button_candidate(self) -> None:
        from qwen_visual_decision import _decision_observation_prompt_dict

        context = json.loads(json.dumps(self.context, ensure_ascii=False))
        context["current_subgoal"]["constraints"] = [
            "禁止通过任何页面正文链接、按钮或元素跳转"
        ]
        prompt_observation = _decision_observation_prompt_dict(
            QwenTaskContext.from_dict(context),
            self.observation,
        )

        self.assertEqual([], prompt_observation["candidates"])

    def test_named_page_button_constraint_keeps_unrelated_list_candidate(self) -> None:
        from qwen_visual_decision import _decision_observation_prompt_dict

        context = json.loads(json.dumps(self.context, ensure_ascii=False))
        context["current_subgoal"]["constraints"] = [
            "不要触碰页面练习按钮、浏览器栏"
        ]
        prompt_observation = _decision_observation_prompt_dict(
            QwenTaskContext.from_dict(context),
            self.observation,
        )

        self.assertEqual(
            [item.element_id for item in self.observation.scene.elements],
            [item["element_id"] for item in prompt_observation["candidates"]],
        )

    def test_prompt_allows_one_bounded_swipe_for_clipped_navigation_list(self) -> None:
        from qwen_visual_decision import _decision_prompt

        prompt = _decision_prompt(
            QwenTaskContext.from_dict(self.context),
            self.observation,
            decision_number=1,
            available_action_kinds=frozenset({"swipe", "tap_semantic"}),
        )

        self.assertIn("目标字面标签或目标区域尚未出现在可信候选中", prompt)
        self.assertIn("边缘存在", prompt)
        self.assertIn("被裁切的后续内容", prompt)
        self.assertIn("连续引导轨/连接线明确接触该边缘", prompt)
        self.assertIn('expected_result只写{"content_changed":true}', prompt)
        self.assertIn("动作后必须重新观察，不能连续执行", prompt)

    def test_drag_prompt_uses_flat_endpoint_fields_and_container_destination(self) -> None:
        from qwen_visual_decision import _decision_prompt

        prompt = _decision_prompt(
            QwenTaskContext.from_dict(self.context),
            self.observation,
            decision_number=1,
            available_action_kinds=frozenset({"drag"}),
        )

        self.assertIn("source_element_id/source_target/source_role", prompt)
        self.assertIn("destination_element_id/destination_target/destination_role", prompt)
        self.assertIn("绝不能返回source或destination嵌套对象", prompt)
        self.assertIn("container可以逐字复制为destination_element_id", prompt)
        self.assertIn("代表单个源物体的", prompt)

    def test_flat_drag_payload_binds_two_trusted_candidates(self) -> None:
        source = UIElement(
            element_id="source",
            role="image",
            meaning="drag_source_block",
            label="起点",
            bounds=(0.19, 0.67, 0.38, 0.81),
            confidence=0.98,
            states={"goal_relevant": True, "fully_visible": True},
            evidence=("紫色方块内逐字显示起点",),
        )
        destination = UIElement(
            element_id="destination",
            role="container",
            meaning="drop_target_zone",
            label="绿色终点",
            bounds=(0.53, 0.63, 0.83, 0.86),
            confidence=0.98,
            states={"goal_relevant": True, "fully_visible": True},
            evidence=("绿色虚线区域内逐字显示绿色终点",),
        )
        observation = trusted_observation(
            self.frames,
            elements=(source, destination),
        )
        payload = action_payload(
            self.context,
            observation,
            element_id="source",
        )
        payload["next_action"] = {
            "kind": "drag",
            "source_element_id": source.element_id,
            "source_target": source.meaning,
            "source_role": source.role,
            "source_label": source.label,
            "source_states": dict(source.states),
            "destination_element_id": destination.element_id,
            "destination_target": destination.meaning,
            "destination_role": destination.role,
            "destination_label": destination.label,
            "destination_states": dict(destination.states),
        }
        payload["target_region"] = {
            "kind": "element_path",
            "element_id": source.element_id,
            "bounds": [round(item * 1000) for item in source.bounds],
            "destination_element_id": destination.element_id,
            "destination_bounds": [
                round(item * 1000) for item in destination.bounds
            ],
            "description": "起点到绿色终点",
        }

        _observer, decision = self.decide(
            FakeProvider(payload),
            observation=observation,
        )

        self.assertEqual("action", decision.proposal.status)
        self.assertEqual("drag", decision.proposal.action.action)
        self.assertEqual(
            source.element_id,
            decision.proposal.action.params["source_element_id"],
        )
        self.assertEqual(
            destination.element_id,
            decision.proposal.action.params["destination_element_id"],
        )
        self.assertEqual("element_path", decision.target_region.kind)

    def test_prompt_forbids_redundant_focus_on_focused_input(self) -> None:
        from qwen_visual_decision import _decision_prompt

        prompt = _decision_prompt(
            QwenTaskContext.from_dict(self.context),
            self.observation,
            decision_number=1,
            available_action_kinds=frozenset({"tap_semantic", "input_verified_text"}),
        )

        self.assertIn("states.focused=true时禁止再用tap_semantic重复聚焦", prompt)

    def test_unfocused_input_hides_verified_input_until_fresh_focus(self) -> None:
        context_payload = copy.deepcopy(self.context)
        context_payload["goal"]["entities"] = {"input_text": "agent"}
        context = QwenTaskContext.from_dict(context_payload)
        field = UIElement(
            element_id="target_field",
            role="input",
            meaning="target_text_input",
            label="",
            bounds=(0.1, 0.3, 0.9, 0.4),
            confidence=0.99,
            states={"goal_relevant": True, "fully_visible": True, "value": ""},
            evidence=("唯一空输入框",),
        )
        observation = trusted_observation(self.frames, elements=(field,))
        typed_context = test_context_with_semantic_ir(
            context,
            observation,
            frozenset({"tap_semantic", "input_verified_text"}),
        )
        choices = _selection_choices(
            typed_context,
            observation,
            frozenset({"tap_semantic", "input_verified_text"}),
        )
        self.assertEqual(["tap_semantic"], [item["action"] for item in choices])
        self.assertEqual("target_field", choices[0]["element_id"])

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
            task_context=test_context_with_semantic_ir(
                self.context,
                self.observation,
                frozenset({"tap_semantic"}),
            ),
            trusted_observation=self.observation,
        )
        self.assertEqual(provider.calls, 1)
        self.assertEqual(decision.proposal.status, "blocked")
        self.assertIn("禁止携带候选元素", decision.reason)
        self.assertTrue(observer.last_diagnostics["first_output_rejected"])
        self.assertFalse(
            observer.last_diagnostics["candidate_action_from_first_output"]
        )
        self.assertFalse(observer.last_diagnostics["retry_failure_blocked"])
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

    def test_effect_gate_blocks_before_qwen_call(self) -> None:
        context = task_context(external=True, confirmed=False)
        provider = FakeProvider(action_payload(task_context(), self.observation))
        observer, decision = self.decide(provider, context=context)
        self.assertEqual(provider.calls, 0)
        self.assertEqual(decision.proposal.status, "blocked")
        self.assertIn("确认门未满足", decision.reason)
        self.assertEqual(
            observer.last_diagnostics["local_safety_block"], "effect_gate"
        )
        self.assertEqual(observer.status()["final_blocked_rate"], 1.0)

    def test_missing_or_ambiguous_exact_text_blocks_before_qwen(self) -> None:
        missing_context = task_context(task_id="task_exact_missing", revision=12)
        missing_context["goal"]["entities"] = {"target_ui_label": "火星入口"}
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
        ambiguous_context["goal"]["entities"] = {"target_ui_label": "确定"}
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
            task_context=test_context_with_semantic_ir(
                ambiguous_context,
                duplicate_observation,
                frozenset({"tap_semantic"}),
            ),
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
        context["current_execution_class"] = "observe"
        context["current_subgoal"]["execution_class"] = "observe"
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
            task_context=test_context_with_semantic_ir(
                context,
                observation,
                frozenset({"tap_semantic"}),
            ),
            trusted_observation=observation,
        )
        self.assertEqual(decision.proposal.status, "finished")
        self.assertEqual(decision.proposal.completion_evidence, ("input_value:.com",))

    def test_read_only_physical_action_is_blocked_without_remote_repair(self) -> None:
        context = task_context()
        context["current_execution_class"] = "observe"
        context["current_subgoal"]["execution_class"] = "observe"
        invalid = action_payload(context, self.observation)
        finished = copy.deepcopy(invalid)
        finished.update(
            {
                "status": "finished",
                "next_action": None,
                "target_region": None,
                "expected_result": {},
                "completion_evidence_element_ids": ["scene"],
                "reason": "当前可信场景摘要已经证明只读结果。",
            }
        )
        provider = SequenceProvider([invalid, finished])

        _observer, decision = self.decide(provider, context=context)

        self.assertEqual("blocked", decision.proposal.status)
        self.assertEqual(1, provider.calls)

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
            task_context=test_context_with_semantic_ir(
                context,
                observation,
                frozenset({"tap_semantic"}),
            ),
            trusted_observation=observation,
        )
        self.assertEqual(decision.proposal.status, "action")

    def test_exact_label_overlapping_ime_duplicates_collapse_to_local_audit(self) -> None:
        generic = UIElement(
            element_id="e1",
            role="button",
            meaning="select_candidate",
            label="你好",
            bounds=(0.095, 0.615, 0.215, 0.655),
            confidence=1.0,
            states={"goal_relevant": False, "fully_visible": True},
            evidence=("拼音候选栏首个候选",),
        )
        audited = UIElement(
            element_id="local_audited_ime_candidate_1",
            role="button",
            meaning="ime_exact_candidate",
            label="你好",
            bounds=(0.11, 0.61, 0.23, 0.65),
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
            evidence=("输入结构审计确认唯一逐字候选",),
        )

        observation = trusted_observation(
            self.frames,
            scene=scene_for(self.frames, elements=(generic, audited)),
            observation_id="obs_77777777777777777777777777777777",
        )

        self.assertEqual(
            [item.element_id for item in observation.scene.elements],
            [audited.element_id],
        )
        self.assertEqual(
            dict(observation.candidate_aliases)[generic.element_id],
            audited.element_id,
        )
        self.assertTrue(
            any(
                item["kind"] == "duplicate_visual_object_collapsed"
                and item["canonical_element_id"] == audited.element_id
                for item in observation.candidate_conflicts
            )
        )

    def test_separate_same_label_buttons_remain_ambiguous(self) -> None:
        elements = (
            UIElement(
                element_id="first_save",
                role="button",
                meaning="save_first_item",
                label="保存",
                bounds=(0.1, 0.4, 0.3, 0.48),
                confidence=0.97,
            ),
            UIElement(
                element_id="second_save",
                role="button",
                meaning="save_second_item",
                label="保存",
                bounds=(0.1, 0.6, 0.3, 0.68),
                confidence=0.97,
            ),
        )

        observation = trusted_observation(
            self.frames,
            scene=scene_for(self.frames, elements=elements),
            observation_id="obs_88888888888888888888888888888888",
        )

        self.assertEqual(len(observation.scene.elements), 2)
        self.assertEqual(observation.candidate_aliases, ())

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
        context["goal"]["entities"] = {"target_ui_label": "设置"}
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
            task_context=test_context_with_semantic_ir(
                self.context,
                self.observation,
                frozenset({"wait_for_change"}),
            ),
            trusted_observation=self.observation,
            available_action_kinds={"wait_for_change"},
        )

        self.assertEqual(provider.calls, 1)
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

    def test_invalid_output_does_not_contact_remote_repair_response(self) -> None:
        invalid = action_payload(self.context, self.observation)
        invalid["target_region"]["bounds"] = [1, 1, 10, 10]
        provider = SequenceProvider(
            [
                invalid,
                VisionAgentError("千问视觉连接连续1次中断：Server disconnected"),
            ]
        )
        observer, decision = self.decide(provider)
        self.assertEqual(provider.calls, 1)
        self.assertEqual(decision.proposal.status, "blocked")
        self.assertTrue(observer.last_diagnostics["first_output_rejected"])
        self.assertFalse(observer.last_diagnostics["candidate_action_from_first_output"])
        self.assertFalse(observer.last_diagnostics["protocol_retry_used"])

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

    def test_invalid_first_output_is_blocked_without_remote_retry(self) -> None:
        invalid = action_payload(self.context, self.observation)
        invalid["target_region"]["bounds"] = [1, 1, 10, 10]
        valid = action_payload(self.context, self.observation)
        provider = SequenceProvider([invalid, valid])
        observer, decision = self.decide(provider)
        self.assertEqual(provider.calls, 1)
        self.assertEqual(decision.proposal.status, "blocked")
        self.assertFalse(observer.last_diagnostics["protocol_retry_used"])
        self.assertFalse(
            observer.last_diagnostics["candidate_action_from_first_output"]
        )
        status = observer.status()
        self.assertEqual(status["first_pass_rate"], 0.0)
        self.assertEqual(status["repair_retry_rate"], 0.0)

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
            "params": {"direction": "up", "distance": "medium"},
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

    def test_home_is_a_bound_system_navigation_action(self) -> None:
        payload = action_payload(self.context, self.observation)
        payload["next_action"] = {"kind": "home"}
        payload["target_region"] = {
            "kind": "system_navigation",
            "bounds": [0, 0, 1000, 1000],
            "description": "Android系统Home键",
        }
        payload["expected_result"] = {"scene_changed": True}

        _observer, decision = self.decide(FakeProvider(payload))

        self.assertEqual("action", decision.proposal.status)
        self.assertEqual("home", decision.proposal.action.action)
        self.assertEqual("system_navigation", decision.target_region.kind)
        self.assertEqual("", decision.target_region.element_id)
        self.assertEqual((0.0, 0.0, 1.0, 1.0), decision.target_region.bounds)

    def test_explicit_system_home_sends_only_privacy_minimized_image(self) -> None:
        context = task_context(task_id="task_home_privacy", revision=5)
        context["current_subgoal"]["objective"] = "返回手机桌面"
        context["current_subgoal"]["completion_conditions"] = ["手机桌面可见"]
        payload = action_payload(context, self.observation)
        payload["next_action"] = {"kind": "home"}
        payload["target_region"] = {
            "kind": "system_navigation",
            "bounds": [0, 0, 1000, 1000],
            "description": "Android系统Home键",
        }
        payload["expected_result"] = {"scene_changed": True}
        provider = FakeProvider(payload)

        _observer, decision = self.decide(
            provider,
            context=context,
            available_action_kinds={"home"},
        )

        sent_url = provider.messages[1]["content"][1]["image_url"]["url"]
        selected = self.frames[self.observation.selected_frame_index].convert("RGB")
        self.assertEqual(
            _image_data_url(privacy_minimized_system_navigation_view(selected)),
            sent_url,
        )
        self.assertEqual("home", decision.proposal.action.action)

    def test_privacy_home_keeps_full_canonical_catalog_digest(self) -> None:
        raw_context = task_context(task_id="task_home_digest", revision=7)
        raw_context["current_subgoal"]["objective"] = "回到手机主屏幕"
        raw_context["current_subgoal"]["completion_conditions"] = [
            "手机主屏幕可见"
        ]
        parsed = QwenTaskContext.from_dict(raw_context)
        semantic_ir = TaskSemanticIR(
            task_id=parsed.task_id,
            device_id=parsed.device_id,
            revision=parsed.revision,
            raw_goal="回到手机主屏幕",
            surfaces=(SurfaceRef("surface_launcher", "launcher"),),
            entities=(),
            effects=(),
            constraints=(
                ConstraintIntent(
                    constraint_id="constraint.home",
                    kind="required_action",
                    value="home",
                    authoritative=True,
                ),
            ),
            subgoals=(
                SemanticSubgoal(
                    subgoal_id="current_target",
                    surface_ref="surface_launcher",
                    status="active",
                    external_impact="navigation_only",
                    constraint_refs=("constraint.home",),
                ),
            ),
        )
        context = replace(parsed, semantic_ir=semantic_ir)
        current_scene = scene_for(
            self.frames,
            app_id="unknown",
            screen_id="unknown",
            elements=(),
        )
        observation = trusted_observation(self.frames, scene=current_scene)
        available = frozenset({"home", "tap_semantic", "swipe"})
        provider = FakeProvider(
            minimal_selection_payload(status="action", choice_id="choice_1")
        )

        _observer, decision = self.decide(
            provider,
            context=context,
            observation=observation,
            available_action_kinds=available,
        )
        expected = compile_canonical_action_catalog(
            observation.scene,
            semantic_ir,
            available,
        )

        self.assertEqual("home", decision.proposal.action.action)
        self.assertEqual(
            expected.report_digest,
            decision.proposal.action.params["formal_report_digest"],
        )

    def test_explicit_system_home_privacy_view_rejects_other_actions(self) -> None:
        context = task_context(task_id="task_home_only", revision=6)
        context["current_subgoal"]["objective"] = "返回手机桌面"
        context["current_subgoal"]["completion_conditions"] = ["手机桌面可见"]
        forged = action_payload(context, self.observation)
        provider = FakeProvider(forged)

        _observer, decision = self.decide(
            provider,
            context=context,
            available_action_kinds={"home"},
        )

        self.assertEqual("blocked", decision.proposal.status)
        self.assertIn("tap_semantic", decision.reason)

    def _reveal_system_navigation_payload(self) -> tuple[dict, TrustedObservation]:
        current_scene = scene_for(self.frames)
        object.__setattr__(
            current_scene,
            "system_ui",
            SystemUIFacts(
                immersive_or_fullscreen=True,
                navigation_bar_visible=False,
            ),
        )
        observation = trusted_observation(self.frames, scene=current_scene)
        payload = action_payload(self.context, observation)
        payload["next_action"] = {"kind": "reveal_system_navigation"}
        payload["target_region"] = {
            "kind": "system_navigation",
            "bounds": [0, 0, 1000, 1000],
            "description": "Android系统导航栏",
        }
        payload["expected_result"] = {
            "system_ui": {"navigation_bar_visible": True}
        }
        return payload, observation

    def test_reveal_system_navigation_is_coordinate_free_system_action(self) -> None:
        payload, observation = self._reveal_system_navigation_payload()

        _observer, decision = self.decide(
            FakeProvider(payload),
            observation=observation,
        )

        self.assertEqual("action", decision.proposal.status)
        self.assertEqual(
            "reveal_system_navigation",
            decision.proposal.action.action,
        )
        self.assertEqual(
            {"expected_effect": {"system_ui": {"navigation_bar_visible": True}}},
            decision.proposal.action.params,
        )
        self.assertEqual("system_navigation", decision.target_region.kind)
        self.assertEqual(
            True,
            observation.prompt_dict()["system_ui"]["immersive_or_fullscreen"],
        )

    def test_reveal_system_navigation_rejects_model_coordinates(self) -> None:
        bad, observation = self._reveal_system_navigation_payload()
        bad["next_action"]["x"] = 500
        bad["next_action"]["y"] = 990
        provider = SequenceProvider([bad, bad])

        _observer, decision = self.decide(provider, observation=observation)

        self.assertEqual("blocked", decision.proposal.status)
        self.assertIn("协议外字段", decision.reason)

    def test_reveal_system_navigation_rejects_model_distance(self) -> None:
        bad, observation = self._reveal_system_navigation_payload()
        bad["next_action"]["distance"] = "short"
        provider = SequenceProvider([bad, bad])

        _observer, decision = self.decide(provider, observation=observation)

        self.assertEqual("blocked", decision.proposal.status)
        self.assertIn("distance", decision.reason)

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

    def test_redundant_action_bounds_must_match_trusted_candidate(self) -> None:
        payload = action_payload(self.context, self.observation)
        payload["next_action"]["bounds"] = [680, 200, 860, 350]

        _observer, decision = self.decide(FakeProvider(payload))

        self.assertEqual("action", decision.proposal.status)
        self.assertNotIn("bounds", decision.proposal.action.params)
        self.assertEqual(
            self.observation.get_candidate("settings_icon").bounds,
            decision.target_region.bounds,
        )

    def test_redundant_action_bounds_cannot_change_candidate_region(self) -> None:
        bad = action_payload(self.context, self.observation)
        bad["next_action"]["bounds"] = [100, 100, 300, 300]
        provider = SequenceProvider([bad, bad])

        _observer, decision = self.decide(provider)

        self.assertEqual("blocked", decision.proposal.status)
        self.assertIn("bounds", decision.reason)

    def test_screen_action_cannot_carry_redundant_element_bounds(self) -> None:
        bad = action_payload(self.context, self.observation)
        bad["next_action"] = {
            "kind": "swipe",
            "direction": "up",
            "bounds": [680, 200, 860, 350],
        }
        provider = SequenceProvider([bad, bad])

        _observer, decision = self.decide(provider)

        self.assertEqual("blocked", decision.proposal.status)
        self.assertIn("bounds", decision.reason)

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

    def test_overlay_absence_under_system_ui_normalizes_only_for_dismissal(self) -> None:
        payload = action_payload(self.context, self.observation)
        payload.update(
            {
                "next_action": {"kind": "back"},
                "target_region": {
                    "kind": "system_navigation",
                    "bounds": [0, 0, 1000, 1000],
                    "description": "Android系统返回键",
                },
                "expected_result": {"system_ui": {"overlays": []}},
            }
        )

        _observer, decision = self.decide(FakeProvider(payload))

        self.assertEqual("action", decision.proposal.status)
        self.assertEqual("back", decision.proposal.action.action)
        self.assertEqual({"scene_changed": True}, decision.expected_result)

        invalid = action_payload(self.context, self.observation)
        invalid["expected_result"] = {"system_ui": {"overlays": []}}
        _observer, blocked = self.decide(SequenceProvider([invalid, invalid]))
        self.assertEqual("blocked", blocked.proposal.status)
        self.assertIn("system_ui", blocked.reason)

        keyboard_payload = action_payload(self.context, self.observation)
        keyboard_payload.update(
            {
                "next_action": {"kind": "back"},
                "target_region": {
                    "kind": "system_navigation",
                    "bounds": [0, 0, 1000, 1000],
                    "description": "Android系统返回键",
                },
                "expected_result": {
                    "system_ui": {"soft_keyboard_visible": False}
                },
            }
        )
        _observer, keyboard_decision = self.decide(FakeProvider(keyboard_payload))
        self.assertEqual("action", keyboard_decision.proposal.status)
        self.assertEqual({"scene_changed": True}, keyboard_decision.expected_result)

        invalid_keyboard = action_payload(self.context, self.observation)
        invalid_keyboard["expected_result"] = {
            "system_ui": {"soft_keyboard_visible": False}
        }
        _observer, blocked_keyboard = self.decide(
            SequenceProvider([invalid_keyboard, invalid_keyboard])
        )
        self.assertEqual("blocked", blocked_keyboard.proposal.status)
        self.assertIn("system_ui", blocked_keyboard.reason)

        true_keyboard = copy.deepcopy(keyboard_payload)
        true_keyboard["expected_result"] = {
            "system_ui": {"soft_keyboard_visible": True}
        }
        _observer, blocked_true = self.decide(
            SequenceProvider([true_keyboard, true_keyboard])
        )
        self.assertEqual("blocked", blocked_true.proposal.status)
        self.assertIn("system_ui", blocked_true.reason)

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
        self.assertIn('"status":"action"', prompt)
        self.assertIn('"element_id":"逐字复制可信候选ID"', prompt)
        self.assertIn('"target_region":{"kind":"element"', prompt)
        self.assertIn('"expected_result":{"scene_changed":true}', prompt)
        self.assertIn('"source_element_id":"逐字复制起点候选ID"', prompt)
        self.assertIn('"destination_element_id":"逐字复制终点候选ID"', prompt)
        self.assertIn('"kind":"element_path"', prompt)
        self.assertIn("绝不能返回source或destination对象", prompt)
        self.assertIn(
            '"element_state":{"meaning":"逐字复制输入候选meaning",'
            '"states":{"value":"逐字复制goal.entities.input_text"}}',
            prompt,
        )
        self.assertIn("element_state和states都必须是JSON对象", prompt)
        self.assertIn('"status":"blocked","next_action":null', prompt)
        self.assertIn('"status":"finished","next_action":null', prompt)

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

    def test_trusted_observation_uses_converged_tail_not_sharper_leading_frame(self) -> None:
        base = self.frames[-1].copy().convert("RGB")
        leading = base.copy()
        draw = ImageDraw.Draw(leading)
        for y in range(220, 700, 8):
            for x in range(140, 400, 8):
                color = "white" if ((x + y) // 8) % 2 else "black"
                draw.rectangle((x, y, x + 7, y + 7), fill=color)
        frames = [leading, base.copy(), base.copy(), base.copy()]
        scene = scene_for(frames)

        observation = TrustedObservation.from_scene(
            frames=frames,
            device_id="offline_phone_01",
            scene=scene,
        )

        self.assertGreaterEqual(observation.selected_frame_index, 1)
        self.assertEqual(_local_frame_fingerprint(base), observation.fingerprint)

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

    def test_refresh_event_cannot_finish_from_static_single_frame_evidence(self) -> None:
        cases = (
            ("refresh", "当前页面内容已刷新，以获取最新版本", "当前页面内容已刷新"),
            ("navigation", "已经导航进入目标页面", "导航已经完成"),
            ("retrieval", "已经重新获取服务器内容", "重新获取已经完成"),
        )
        for suffix, objective, condition in cases:
            with self.subTest(suffix=suffix):
                context = task_context(task_id=f"task_{suffix}_static")
                context["current_subgoal"]["objective"] = objective
                context["current_subgoal"]["completion_conditions"] = [condition]
                invalid = action_payload(context, self.observation)
                invalid.update(
                    {
                        "status": "finished",
                        "next_action": None,
                        "target_region": None,
                        "expected_result": {},
                        "completion_evidence_element_ids": ["settings_icon"],
                        "reason": "当前静态页面包含目标内容，所以动作已经发生。",
                    }
                )
                provider = SequenceProvider(
                    [invalid, blocked_payload(context, self.observation)]
                )

                _observer, decision = self.decide(provider, context=context)

                self.assertEqual("blocked", decision.proposal.status)
                self.assertEqual(1, provider.calls)

    def test_refresh_event_accepts_literal_dynamic_success_evidence(self) -> None:
        context = task_context(task_id="task_refresh_dynamic")
        context["current_subgoal"]["objective"] = "当前页面内容已刷新"
        context["current_subgoal"]["completion_conditions"] = ["刷新已经完成"]
        refreshed = UIElement(
            element_id="refresh-result",
            role="text",
            meaning="refresh_status",
            label="刷新成功",
            bounds=(0.2, 0.1, 0.8, 0.16),
            confidence=0.97,
            states={"goal_relevant": True},
            evidence=("页面逐字显示刷新成功",),
        )
        observation = trusted_observation(self.frames, elements=(refreshed,))
        payload = action_payload(
            context,
            observation,
            element_id="refresh-result",
        )
        payload.update(
            {
                "status": "finished",
                "next_action": None,
                "target_region": None,
                "expected_result": {},
                "completion_evidence_element_ids": ["refresh-result"],
            }
        )

        _observer, decision = self.decide(
            FakeProvider(payload),
            context=context,
            observation=observation,
        )

        self.assertEqual("finished", decision.proposal.status)
        self.assertEqual(("refresh-result:刷新成功",), decision.proposal.completion_evidence)

    def test_static_update_or_version_label_cannot_prove_refresh_event(self) -> None:
        cases = (
            ("latest-version", "latest version"),
            ("last-updated", "last updated: 2026-08-01"),
            ("last-updated-zh", "最后更新：2026-08-01"),
            ("update-time-zh", "更新时间：2026-08-01"),
        )
        for suffix, label in cases:
            with self.subTest(label=label):
                context = task_context(task_id=f"task_refresh_{suffix}")
                context["current_subgoal"]["objective"] = "当前页面内容已刷新"
                context["current_subgoal"]["completion_conditions"] = [
                    "刷新已经完成"
                ]
                static_label = UIElement(
                    element_id="static-update-label",
                    role="text",
                    meaning="update_metadata",
                    label=label,
                    bounds=(0.2, 0.1, 0.8, 0.16),
                    confidence=0.97,
                    states={"goal_relevant": True},
                    evidence=(f"页面逐字显示 {label}",),
                )
                observation = trusted_observation(
                    self.frames,
                    elements=(static_label,),
                )
                invalid = action_payload(
                    context,
                    observation,
                    element_id="static-update-label",
                )
                invalid.update(
                    {
                        "status": "finished",
                        "next_action": None,
                        "target_region": None,
                        "expected_result": {},
                        "completion_evidence_element_ids": ["static-update-label"],
                    }
                )
                blocked = copy.deepcopy(invalid)
                blocked.update(
                    {
                        "status": "blocked",
                        "completion_evidence_element_ids": [],
                        "confidence": 0.4,
                        "reason": "单帧静态更新标签不能证明本次刷新发生。",
                    }
                )
                provider = SequenceProvider([invalid, blocked])

                _observer, decision = self.decide(
                    provider,
                    context=context,
                    observation=observation,
                )

                self.assertEqual("blocked", decision.proposal.status)
                self.assertEqual(1, provider.calls)

    def test_exact_duplicate_json_response_is_accepted(self) -> None:
        payload = action_payload(self.context, self.observation)
        raw = json.dumps(payload, ensure_ascii=False)

        provider = RawSequenceProvider([raw + "\n" + raw])
        _observer, decision = self.decide(provider)

        self.assertEqual("action", decision.proposal.status)
        self.assertEqual(1, provider.calls)

    def test_conflicting_duplicate_json_response_is_safely_blocked(self) -> None:
        first = action_payload(self.context, self.observation)
        second = copy.deepcopy(first)
        second["confidence"] = 0.81
        raw = (
            json.dumps(first, ensure_ascii=False)
            + "\n"
            + json.dumps(second, ensure_ascii=False)
        )

        provider = RawSequenceProvider([raw, raw])
        _observer, decision = self.decide(provider)

        self.assertEqual("blocked", decision.proposal.status)
        self.assertIn("多个互相冲突", decision.reason)
        self.assertEqual(1, provider.calls)

    def test_action_discards_model_authored_completion_evidence(self) -> None:
        payload = action_payload(self.context, self.observation)
        payload["completion_evidence_element_ids"] = ["settings_icon"]

        _observer, decision = self.decide(FakeProvider(payload))

        self.assertEqual("action", decision.proposal.status)
        self.assertEqual((), decision.completion_evidence_element_ids)
        self.assertEqual((), decision.proposal.completion_evidence)

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
            task_context=test_context_with_semantic_ir(
                context,
                observation,
                frozenset({"swipe"}),
            ),
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
            task_context=test_context_with_semantic_ir(
                close_context,
                observation,
                frozenset({"dismiss_overlay"}),
            ),
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
            task_context=test_context_with_semantic_ir(
                submit_context,
                observation,
                frozenset({"tap_semantic"}),
            ),
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
            task_context=test_context_with_semantic_ir(
                finished_context,
                observation,
                frozenset({"tap_semantic"}),
            ),
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
            "current_execution_class",
            "effect_intents",
            "effect_gate",
        ):
            self.assertIn(field, prompt)


if __name__ == "__main__":
    unittest.main()
