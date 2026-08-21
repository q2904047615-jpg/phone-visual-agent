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

    def setUp(self) -> None:
        self.frames = load_sequence("launcher_stable")
        self.context = task_context()
        self.observation = trusted_observation(self.frames)

    def test_semantic_reload_label_does_not_become_literal_preblock(self) -> None:
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
        self.assertEqual(set(), _required_exact_candidate_ids(context, observation))

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
                self.assertIsNone(
                    _exact_text_candidate_block(
                        blocked_context,
                        blocked_observation,
                    )
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
        self.assertNotIn("备注", different_parsed.exact_text_requirements)
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
        self.assertIsNone(
            _exact_text_candidate_block(swipe_context, missing_identity)
        )

        tap_context = with_required_action("tap_semantic")
        self.assertIsNone(_exact_text_candidate_block(tap_context, observation))

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
        self.assertIsNone(_exact_text_candidate_block(chat_context, partial))

        ambiguous = trusted_observation(
            self.frames,
            elements=(identity("设置", "title_a"), identity("设置页面", "title_b")),
            observation_id="obs_22222222222222222222222222222222",
        )
        self.assertIsNone(_exact_text_candidate_block(parsed, ambiguous))

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
        self.assertIsNone(_exact_text_candidate_block(context, missing_title))

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

    def test_full_decision_payload_cannot_reenter_current_runtime(self) -> None:
        payload = minimal_selection_payload(status="blocked")
        payload["protocol_version"] = QWEN_VISUAL_DECISION_PROTOCOL_VERSION
        provider = SequenceProvider([payload, payload])

        _observer, decision = self.decide(provider)

        self.assertEqual("blocked", decision.proposal.status)
        self.assertEqual(2, provider.calls)
        self.assertIn("协议外字段", decision.reason)

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
        self.assertEqual(set(), _required_exact_candidate_ids(context, observation))
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
            self.assertIsNone(
                _exact_text_candidate_block(context, wrong_observation)
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
        self.assertEqual(set(), _required_exact_candidate_ids(context, observation))

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
        self.assertEqual(set(), _required_exact_candidate_ids(context, empty_observation))

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
