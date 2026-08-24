from __future__ import annotations

import copy
import json
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageDraw

from canonical_action_protocol import (
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
    _launcher_app_entry_candidate_ids,
    _required_exact_candidate_ids,
    _deterministic_exact_selection_payload,
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
