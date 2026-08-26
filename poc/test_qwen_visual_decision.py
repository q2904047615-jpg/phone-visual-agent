from __future__ import annotations

import copy
import json
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageDraw

from agent.domain.canonical_action_protocol import (
    CanonicalActionProtocolError as GenericStepPlanningError,
    compile_canonical_action_catalog,
    scene_matches_target_app_surface,
)
from generic_scene_observer import _local_frame_fingerprint
from qwen_visual_decision import (
    QWEN_VISUAL_DECISION_PROTOCOL_VERSION,
    QwenTaskContext,
    QwenVisualDecisionObserver,
    TrustedObservation,
    _exact_text_candidate_block,
    _identity_text_candidate_block,
    _hydrate_canonical_selection,
    _launcher_app_entry_candidate_ids,
    _required_exact_candidate_ids,
    _deterministic_exact_selection_payload,
    _selection_choices,
)
from agent.domain.ui_scene import SystemUIFacts, UIElement, UIScene, UISceneError
from vision_agent import _image_data_url
from agent.domain.vision_model import VisionAgentError
from agent.domain.task_semantic_ir import (
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
SAMPLE_IMAGE_ROOT = ROOT / "evals" / "qwen_visual_decision" / "images"


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


class StructuredAppSurfaceSelectionRegressionTests(unittest.TestCase):
    def test_runtime_package_and_structured_screen_select_named_child_entry(self) -> None:
        target = SimpleNamespace(app_id="sample_chat", app_name="示例聊天")
        unrelated_scene = UIScene(
            app_id="com.vendor.runtime",
            screen_id="generic_main_list",
            summary="普通列表",
            elements=(),
            stable=True,
            confidence=0.99,
        )
        launcher_scene = UIScene(
            app_id="launcher",
            screen_id="home_screen",
            summary="系统桌面",
            elements=(),
            stable=True,
            confidence=0.99,
        )
        self.assertFalse(scene_matches_target_app_surface(unrelated_scene, target))
        self.assertFalse(scene_matches_target_app_surface(launcher_scene, target))

        task_id = "task_structured_app_surface"
        device_id = "device-local-01"
        revision = 1
        semantic_ir = TaskSemanticIR(
            task_id=task_id,
            device_id=device_id,
            revision=revision,
            raw_goal="进入示例聊天中的文件传输助手",
            surfaces=(
                SurfaceRef(
                    surface_id="surface_sample_chat",
                    kind="app",
                    app_id="sample_chat",
                    app_name="示例聊天",
                ),
            ),
            entities=(
                SemanticEntity(
                    entity_id="entity_recipient",
                    entity_type="party",
                    role="recipient",
                    value="文件传输助手",
                    source_span=SourceSpan(8, 14),
                    authority="user_literal",
                ),
            ),
            effects=(),
            subgoals=(
                SemanticSubgoal(
                    subgoal_id="find_recipient",
                    surface_ref="surface_sample_chat",
                    status="active",
                    external_impact="navigation_only",
                    entity_refs=("entity_recipient",),
                ),
            ),
        )
        parsed = QwenTaskContext(
            protocol_version="2026-08-20-deepseek-typed-task-graph-v4",
            task_id=task_id,
            device_id=device_id,
            revision=revision,
            task_status="running",
            goal={
                "objective": "进入示例聊天中的文件传输助手",
                "target_apps": [
                    {"app_id": "sample_chat", "app_name": "示例聊天"}
                ],
                "entities": {"recipient": "文件传输助手"},
            },
            global_constraints=(),
            goal_completion_conditions=(),
            current_subgoal={
                "subgoal_id": "find_recipient",
                "objective": "进入文件传输助手页面",
                "status": "active",
                "depends_on": (),
                "constraints": (),
                "completion_conditions": ("文件传输助手页面可见",),
                "completion_evidence": (),
                "effect_ids": (),
                "execution_class": "navigate",
            },
            current_execution_class="navigate",
            effect_intents=(),
            effect_gate={
                "required": False,
                "state": "not_required",
                "effect_ids": [],
                "effect_action_allowed": False,
                "scope": {
                    "task_id": task_id,
                    "device_id": device_id,
                    "revision": revision,
                    "subgoal_id": "find_recipient",
                },
            },
            semantic_ir=semantic_ir,
        )
        parsed.validate()
        current_scene = UIScene(
            app_id="com.vendor.runtime",
            screen_id="sample_chat_main_list",
            summary="示例聊天列表中可见文件传输助手。",
            elements=(
                UIElement(
                    element_id="file-transfer",
                    role="list_item",
                    meaning="chat_entry_file_transfer",
                    label="文件传输助手",
                    bounds=(0.08, 0.22, 0.93, 0.32),
                    confidence=0.99,
                    states={
                        "goal_relevant": True,
                        "visible": True,
                        "fully_visible": True,
                    },
                ),
            ),
            stable=True,
            confidence=0.98,
        )
        self.assertTrue(scene_matches_target_app_surface(current_scene, target))
        observation = SimpleNamespace(
            scene=current_scene,
            target_local_candidate=current_scene.unique_trusted_goal_element,
        )
        choices = _selection_choices(
            parsed,
            observation,
            frozenset({"back", "home", "tap_semantic", "wait_for_change"}),
        )

        selected = _deterministic_exact_selection_payload(
            parsed,
            choices,
            observation=observation,
        )

        self.assertIsNotNone(selected)
        self.assertEqual("action", selected["status"])
        selected_choice = next(
            item for item in choices if item["choice_id"] == selected["choice_id"]
        )
        self.assertEqual("tap_semantic", selected_choice["action"])
        self.assertEqual("file-transfer", selected_choice["element_id"])


class PagedViewportSelectionRegressionTests(unittest.TestCase):
    def test_paged_viewport_does_not_self_select_a_direction(self) -> None:
        task_id = "task_paged_launcher"
        device_id = "device-local-01"
        surface = SurfaceRef(
            "surface_target",
            "app",
            app_id="sample.app",
            app_name="示例应用",
        )
        semantic_ir = TaskSemanticIR(
            task_id=task_id,
            device_id=device_id,
            revision=2,
            raw_goal="打开示例应用",
            surfaces=(surface,),
            entities=(),
            effects=(),
            subgoals=(
                SemanticSubgoal(
                    "open_target_app",
                    surface.surface_id,
                    "active",
                    "navigation_only",
                ),
            ),
        )
        context = QwenTaskContext(
            protocol_version="2026-08-20-deepseek-typed-task-graph-v4",
            task_id=task_id,
            device_id=device_id,
            revision=2,
            task_status="running",
            goal={
                "objective": "打开示例应用",
                "target_apps": [
                    {"app_id": "sample.app", "app_name": "示例应用"}
                ],
                "entities": {},
            },
            global_constraints=(),
            goal_completion_conditions=(),
            current_subgoal={
                "subgoal_id": "open_target_app",
                "objective": "打开示例应用",
                "status": "active",
                "depends_on": (),
                "constraints": (),
                "completion_conditions": ("示例应用主界面可见",),
                "completion_evidence": (),
                "effect_ids": (),
                "execution_class": "navigate",
            },
            current_execution_class="navigate",
            effect_intents=(),
            effect_gate={
                "required": False,
                "state": "not_required",
                "effect_ids": [],
                "effect_action_allowed": False,
                "scope": {
                    "task_id": task_id,
                    "device_id": device_id,
                    "revision": 2,
                    "subgoal_id": "open_target_app",
                },
            },
            semantic_ir=semantic_ir,
        )
        current_scene = UIScene(
            app_id="launcher",
            screen_id="home_screen",
            summary="三页桌面的第二页，目标应用未出现",
            elements=(
                UIElement(
                    element_id="pages",
                    role="container",
                    meaning="paged_viewport",
                    label="桌面分页区",
                    bounds=(0.02, 0.10, 0.98, 0.90),
                    confidence=1.0,
                    states={
                        "goal_relevant": True,
                        "fully_visible": True,
                        "scrollable": True,
                        "scroll_axis": "horizontal",
                        "page_index": 1,
                        "page_count": 3,
                    },
                    evidence=("三个分页圆点中第二个高亮",),
                ),
            ),
            stable=True,
            confidence=1.0,
        )
        observation = SimpleNamespace(
            scene=current_scene,
            target_local_candidate=lambda: None,
        )
        choices = _selection_choices(
            context,
            observation,
            frozenset({"back", "swipe", "tap_semantic", "wait_for_change"}),
        )

        def selected_direction():
            payload = _deterministic_exact_selection_payload(
                context,
                choices,
                observation=observation,
            )
            if payload is None:
                return None
            selected = next(
                item for item in choices if item["choice_id"] == payload["choice_id"]
            )
            return selected.get("direction")

        self.assertIsNone(selected_direction())


class ElementBoundSwipeSelectionRegressionTests(unittest.TestCase):
    @staticmethod
    def context(*, source_text: str) -> QwenTaskContext:
        constraint = ConstraintIntent(
            constraint_id="constraint.swipe",
            kind="required_action",
            value="swipe",
            source_text=source_text,
            authoritative=True,
        )
        semantic_ir = TaskSemanticIR(
            task_id="task_element_swipe",
            device_id="device-local-01",
            revision=1,
            raw_goal=source_text,
            surfaces=(SurfaceRef("surface_current", "current_surface"),),
            entities=(),
            effects=(),
            constraints=(constraint,),
            subgoals=(
                SemanticSubgoal(
                    "dismiss_visible_object",
                    "surface_current",
                    "active",
                    "navigation_only",
                    constraint_refs=(constraint.constraint_id,),
                ),
            ),
        )
        return QwenTaskContext(
            protocol_version="2026-08-20-deepseek-typed-task-graph-v4",
            task_id=semantic_ir.task_id,
            device_id=semantic_ir.device_id,
            revision=semantic_ir.revision,
            task_status="running",
            goal={"objective": source_text, "target_apps": [], "entities": {}},
            global_constraints=(),
            goal_completion_conditions=(),
            current_subgoal={
                "subgoal_id": "dismiss_visible_object",
                "objective": source_text,
                "status": "active",
                "depends_on": (),
                "constraints": (),
                "completion_conditions": ("目标不再可见",),
                "completion_evidence": (),
                "effect_ids": (),
                "execution_class": "navigate",
            },
            current_execution_class="navigate",
            effect_intents=(),
            effect_gate={
                "required": False,
                "state": "not_required",
                "effect_ids": [],
                "effect_action_allowed": False,
                "scope": {
                    "task_id": semantic_ir.task_id,
                    "device_id": semantic_ir.device_id,
                    "revision": semantic_ir.revision,
                    "subgoal_id": "dismiss_visible_object",
                },
            },
            semantic_ir=semantic_ir,
        )

    @staticmethod
    def observation(current_scene: UIScene) -> SimpleNamespace:
        return SimpleNamespace(
            observation_id="obs_1234567890abcdef",
            device_id="device-local-01",
            fingerprint=current_scene.fingerprint,
            scene=current_scene,
            get_candidate=current_scene.get_element,
            target_local_candidate=current_scene.unique_trusted_goal_element,
        )

    def test_targeted_swipe_hydrates_element_while_viewport_swipe_stays_screen(self):
        target = UIElement(
            element_id="preview",
            role="container",
            meaning="application_preview_card",
            label="示例应用",
            bounds=(0.27, 0.29, 0.73, 0.81),
            confidence=0.99,
            states={"goal_relevant": True, "fully_visible": True},
            evidence=("唯一完整可见的应用预览卡片",),
        )
        target_scene = UIScene(
            app_id="system",
            screen_id="recent_tasks",
            summary="唯一预览卡片可见",
            elements=(target,),
            stable=True,
            confidence=0.99,
            fingerprint="target-before",
        )
        target_context = self.context(source_text="向上划掉当前唯一预览卡片")
        target_observation = self.observation(target_scene)
        target_choices = _selection_choices(
            target_context,
            target_observation,
            frozenset({"swipe"}),
        )
        target_decision = _hydrate_canonical_selection(
            {
                "status": "action",
                "choice_id": target_choices[0]["choice_id"],
                "confidence": 1.0,
                "reason": "唯一元素绑定滑动",
            },
            context=target_context,
            observation=target_observation,
            choices=target_choices,
        )

        self.assertEqual("element", target_decision.target_region.kind)
        self.assertEqual("preview", target_decision.target_region.element_id)
        self.assertEqual("preview", target_decision.proposal.action.params["element_id"])
        self.assertEqual(
            target.states,
            target_decision.proposal.action.params["states"],
        )

        viewport = replace(
            target,
            element_id="viewport",
            meaning="content_viewport",
            label="内容区",
            states={
                "goal_relevant": True,
                "fully_visible": True,
                "scrollable": True,
                "scroll_axis": "vertical",
            },
        )
        viewport_scene = replace(
            target_scene,
            elements=(viewport,),
            fingerprint="viewport-before",
        )
        viewport_context = self.context(source_text="向上滑动当前内容区")
        viewport_observation = self.observation(viewport_scene)
        viewport_choices = _selection_choices(
            viewport_context,
            viewport_observation,
            frozenset({"swipe"}),
        )
        up_choice = next(
            item for item in viewport_choices if item.get("direction") == "up"
        )
        viewport_decision = _hydrate_canonical_selection(
            {
                "status": "action",
                "choice_id": up_choice["choice_id"],
                "confidence": 1.0,
                "reason": "视口滑动",
            },
            context=viewport_context,
            observation=viewport_observation,
            choices=viewport_choices,
        )
        self.assertEqual("screen", viewport_decision.target_region.kind)
        self.assertNotIn(
            "element_id",
            viewport_decision.proposal.action.params,
        )


class CanonicalEffectSelectionRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        frame = Image.new("RGB", (240, 480), "black")
        draw = ImageDraw.Draw(frame)
        for y in range(0, 480, 12):
            for x in range(0, 240, 12):
                if (x // 12 + y // 12) % 2:
                    draw.rectangle((x, y, x + 5, y + 5), fill="white")
        self.frames = [frame.copy() for _ in range(4)]
        fingerprint = _local_frame_fingerprint(frame)
        self.scene = UIScene(
            app_id="chat",
            screen_id="conversation",
            summary="会话标题、现有正文和发送按钮可见",
            elements=(
                UIElement(
                    element_id="send-control",
                    role="button",
                    meaning="send_message",
                    label="发送",
                    bounds=(0.78, 0.54, 0.96, 0.60),
                    confidence=1.0,
                    states={"goal_relevant": False, "fully_visible": True},
                ),
                UIElement(
                    element_id="conversation-title",
                    role="text",
                    meaning="page_title",
                    label="文件传输助手",
                    bounds=(0.34, 0.02, 0.66, 0.06),
                    confidence=1.0,
                    states={"goal_relevant": False, "fully_visible": True},
                ),
                UIElement(
                    element_id="draft-input",
                    role="input",
                    meaning="application_text_input",
                    label="freshsendproof",
                    bounds=(0.14, 0.57, 0.70, 0.63),
                    confidence=1.0,
                    states={
                        "goal_relevant": True,
                        "fully_visible": True,
                        "focused": True,
                        "value": "freshsendproof",
                    },
                ),
                UIElement(
                    element_id="mode-switch",
                    role="button",
                    meaning="switch_keyboard_input_mode",
                    label="英",
                    bounds=(0.73, 0.92, 0.81, 0.98),
                    confidence=1.0,
                    states={
                        "goal_relevant": False,
                        "fully_visible": True,
                        "keyboard_input_mode_switch": True,
                        "current_mode": "direct_latin",
                        "target_mode": "chinese_pinyin",
                        "prior_input_value": "freshsendproof",
                        "next_input_value": "",
                        "input_element_id": "draft-input",
                    },
                ),
                UIElement(
                    element_id="other-control",
                    role="button",
                    meaning="open_other",
                    label="其他",
                    bounds=(0.05, 0.12, 0.22, 0.18),
                    confidence=1.0,
                    states={"goal_relevant": False, "fully_visible": True},
                ),
            ),
            stable=True,
            confidence=1.0,
            fingerprint=fingerprint,
        )
        self.observation = TrustedObservation.from_scene(
            frames=self.frames,
            device_id="device-local-01",
            scene=self.scene,
            observation_id="obs_effectbinding000000000000000000",
        )
        raw_goal = "文件传输助手 freshsendproof 其他"
        recipient_start = raw_goal.index("文件传输助手")
        input_start = raw_goal.index("freshsendproof")
        other_start = raw_goal.index("其他")
        semantic_ir = TaskSemanticIR(
            task_id="task_effect_binding",
            device_id="device-local-01",
            revision=1,
            raw_goal=raw_goal,
            surfaces=(SurfaceRef("surface_current", "current_surface"),),
            entities=(
                SemanticEntity(
                    entity_id="entity_recipient",
                    entity_type="party",
                    role="recipient",
                    value="文件传输助手",
                    source_span=SourceSpan(
                        recipient_start,
                        recipient_start + len("文件传输助手"),
                    ),
                    authority="user_literal",
                ),
                SemanticEntity(
                    entity_id="entity_input_text",
                    entity_type="text",
                    role="input_text",
                    value="freshsendproof",
                    source_span=SourceSpan(
                        input_start,
                        input_start + len("freshsendproof"),
                    ),
                    authority="user_literal",
                ),
                SemanticEntity(
                    entity_id="entity_other_label",
                    entity_type="ui_label",
                    role="target_ui_label",
                    value="其他",
                    source_span=SourceSpan(
                        other_start,
                        other_start + len("其他"),
                    ),
                    authority="user_literal",
                ),
            ),
            effects=(
                EffectIntent(
                    effect_id="effect_send",
                    kind="send_message",
                    target_refs=("entity_recipient",),
                    payload_refs=("entity_input_text",),
                    source_subgoal_ids=("send_existing_text",),
                    expected_result_texts=(
                        "最新消息气泡逐字为 freshsendproof 且输入框为空",
                    ),
                ),
            ),
            subgoals=(
                SemanticSubgoal(
                    subgoal_id="send_existing_text",
                    surface_ref="surface_current",
                    status="active",
                    external_impact="external_state",
                    entity_refs=("entity_other_label",),
                    effect_refs=("effect_send",),
                ),
            ),
        )
        self.context = QwenTaskContext(
            protocol_version="2026-08-20-deepseek-typed-task-graph-v4",
            task_id="task_effect_binding",
            device_id="device-local-01",
            revision=1,
            task_status="running",
            goal={
                "understood": True,
                "objective": "只发送一次当前已有正文",
                "entities": {
                    "recipient": "文件传输助手",
                    "input_text": "freshsendproof",
                    "target_ui_label": "其他",
                },
            },
            global_constraints=("只发送一次",),
            goal_completion_conditions=(
                {
                    "condition_id": "sent",
                    "description": "最新消息气泡逐字为 freshsendproof 且输入框为空",
                    "evidence_required": ["最新消息气泡和空输入框可见"],
                    "satisfied": False,
                },
            ),
            current_subgoal={
                "subgoal_id": "send_existing_text",
                "objective": "只发送一次输入框内现有正文 freshsendproof",
                "status": "active",
                "depends_on": [],
                "constraints": ["只发送一次"],
                "completion_conditions": ["已发送一次现有正文 freshsendproof"],
                "completion_evidence": [],
                "effect_ids": ["effect_send"],
                "execution_class": "effect",
            },
            current_execution_class="effect",
            effect_intents=(
                {
                    "effect_id": "effect_send",
                    "kind": "send_message",
                    "target_entity_roles": ["recipient"],
                    "payload_entity_roles": ["input_text"],
                    "source_subgoal_ids": ["send_existing_text"],
                    "expected_results": [
                        "最新消息气泡逐字为 freshsendproof 且输入框为空"
                    ],
                    "local_policy": {
                        "effect_id": "effect_send",
                        "confirmation_required": False,
                        "policy_level": "low",
                    },
                },
            ),
            effect_gate={
                "required": False,
                "state": "not_required",
                "effect_ids": [],
                "effect_action_allowed": True,
                "scope": {
                    "task_id": "task_effect_binding",
                    "device_id": "device-local-01",
                    "revision": 1,
                    "subgoal_id": "send_existing_text",
                },
            },
            semantic_ir=semantic_ir,
        )
        self.context.validate()
        self.available = frozenset({"tap_semantic", "clear_verified_text"})

    def _decide(self):
        provider = FakeProvider({})
        decision = QwenVisualDecisionObserver(provider).decide(
            frames=self.frames,
            task_context=self.context,
            trusted_observation=self.observation,
            available_action_kinds=self.available,
        )
        return provider, decision

    def test_effect_bound_canonical_control_outranks_generic_goal_element(self) -> None:
        provider, decision = self._decide()

        self.assertEqual(0, provider.calls)
        self.assertEqual("action", decision.proposal.status)
        self.assertEqual(
            "send-control",
            decision.proposal.action.params["element_id"],
        )

    def test_provider_payload_cannot_override_canonical_effect_control(self) -> None:
        provider, decision = self._decide()

        provider.payload = {"status": "action", "choice_id": "other-control"}
        self.assertEqual(0, provider.calls)
        self.assertEqual("send-control", decision.proposal.action.params["element_id"])
