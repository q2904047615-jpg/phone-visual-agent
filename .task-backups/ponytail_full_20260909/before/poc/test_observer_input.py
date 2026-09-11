from __future__ import annotations
from PIL import Image
from agent.infrastructure.generic_scene_observer import SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION
from agent.infrastructure.generic_scene_observer import SingleStepGenericSceneObserver
from agent.domain.vision_model import VisionAgentError
from agent.domain.visual_evidence import VisualObstruction
import json
from unittest.mock import patch
import unittest
from test_support.generic_scene_observer import (
    SequenceProvider,
    _BaseSingleStepGenericSceneObserverTests,
    audited_application_input,
    input_audit_payload,
    scene_payload,
    stable_frames,
    unique_goal_element,
)


class SingleStepGenericSceneObserverTests(_BaseSingleStepGenericSceneObserverTests):
    def test_target_only_clear_binds_visible_preedit_to_unique_focused_field(
        self,
    ) -> None:
        scene = scene_payload()
        scene.update(
            {
                "foreground_app_id": "com.example.messaging",
                "screen_id": "conversation",
                "summary": "会话页中唯一输入框已聚焦",
                "elements": [],
            }
        )
        audit = input_audit_payload(
            application_inputs=[
                audited_application_input(
                    structure_id="composer",
                    bounds=[100, 500, 760, 600],
                    text="",
                    placeholder="",
                    visible_editable_cues=["aaazjie"],
                    preedit_text="aaazjie",
                    caret_line_index=0,
                )
            ],
            ime_preedit_regions=[
                {
                    "region_id": "preedit",
                    "bounds": [160, 525, 330, 570],
                    "text": "aaazjie",
                    "confidence": 0.99,
                    "candidates": [
                        {
                            "text": "aaazjie",
                            "bounds": [160, 610, 330, 645],
                            "confidence": 0.99,
                            "fully_visible": True,
                        }
                    ],
                }
            ],
            keyboard={
                "visible": True,
                "bounds": [0, 650, 1000, 1000],
                "layout": "qwerty",
                "input_mode": "direct_latin",
                "case_mode": "lower",
                "mode_switch": None,
                "backspace_key": {
                    "label": "⌫",
                    "bounds": [830, 810, 930, 880],
                    "confidence": 0.99,
                    "fully_visible": True,
                },
            },
        )
        context = {
            "entities": {
                "active_subgoal_visual_context": {
                    "subgoal_id": "clear_preedit",
                    "objective": "清空当前唯一聚焦输入框中的预编辑",
                    "constraints": ["不要发送"],
                    "completion_conditions": ["输入框和预编辑都为空"],
                    "execution_class": "navigate",
                    "goal_entities": {
                        "active_input_field_id": "current_input",
                        "active_input_operation": "clear_verified_text",
                        "active_input_target_only": True,
                        "active_input_multiline": False,
                    },
                }
            }
        }

        observed = SingleStepGenericSceneObserver(
            SequenceProvider(
                [
                    {
                        "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
                        "coordinate_space": {
                            "kind": "normalized_1000",
                            "width": 1000,
                            "height": 1000,
                        },
                        "scene": scene,
                        "input_structure": audit,
                    }
                ]
            )
        ).observe(
            frames=stable_frames(),
            goal_context=context,
            device_id="device-local-01",
        )

        field = unique_goal_element(observed)
        self.assertIsNotNone(field)
        self.assertEqual(
            "current_input",
            field.states["input_field_id"],
        )
        self.assertEqual("aaazjie", field.states["ime_preedit_text"])
        self.assertEqual("", field.states["value"])

    def test_target_only_clear_rejects_unbound_preedit_and_backspace(self) -> None:
        scene = scene_payload()
        scene["elements"] = []
        audit = input_audit_payload(
            ime_preedit_regions=[
                {
                    "region_id": "preedit",
                    "bounds": [160, 525, 330, 570],
                    "text": "aaazjie",
                    "confidence": 0.99,
                    "candidates": [],
                }
            ],
            keyboard={
                "visible": True,
                "bounds": [0, 650, 1000, 1000],
                "layout": "qwerty",
                "input_mode": "direct_latin",
                "case_mode": "lower",
                "mode_switch": None,
                "backspace_key": {
                    "label": "⌫",
                    "bounds": [830, 810, 930, 880],
                    "confidence": 0.99,
                    "fully_visible": True,
                },
            },
        )
        context = {
            "entities": {
                "active_subgoal_visual_context": {
                    "subgoal_id": "clear_preedit",
                    "objective": "清空当前唯一聚焦输入框中的预编辑",
                    "constraints": ["不要发送"],
                    "completion_conditions": ["输入框和预编辑都为空"],
                    "execution_class": "navigate",
                    "goal_entities": {
                        "active_input_field_id": "current_input",
                        "active_input_operation": "clear_verified_text",
                        "active_input_target_only": True,
                        "active_input_multiline": False,
                    },
                }
            }
        }

        observed = SingleStepGenericSceneObserver(SequenceProvider([{
            "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
            "coordinate_space": {"kind": "normalized_1000", "width": 1000, "height": 1000},
            "scene": scene, "input_structure": audit,
        }])).observe(frames=stable_frames(), goal_context=context, device_id="device-local-01")

        self.assertIsNone(unique_goal_element(observed))

    def test_single_step_observer_uses_one_request_for_scene_and_input(self) -> None:
        scene = scene_payload()
        scene.update(
            {
                "foreground_app_id": "wechat",
                "screen_id": "chat",
                "summary": "聊天页输入框可见",
                "elements": [],
            }
        )
        audit = input_audit_payload(
            application_inputs=[
                audited_application_input(
                    structure_id="message",
                    bounds=[100, 720, 900, 820],
                    text="",
                    placeholder="消息",
                    field_labels=["消息"],
                    visible_editable_cues=["消息输入框完整边框", "插入光标"],
                    caret_line_index=0,
                )
            ]
        )
        provider = SequenceProvider(
            [
                {
                    "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
                    "coordinate_space": {
                        "kind": "normalized_1000",
                        "width": 1000,
                        "height": 1000,
                    },
                    "scene": scene,
                    "input_structure": audit,
                }
            ]
        )
        context = {
            "entities": {
                "active_subgoal_visual_context": {
                    "subgoal_id": "input_message",
                    "objective": "在消息输入框输入abc",
                    "constraints": [],
                    "completion_conditions": ["消息输入框内容为abc"],
                    "execution_class": "navigate",
                    "goal_entities": {
                        "active_input_transaction_text": "abc",
                        "active_input_field_id": "message_field",
                        "active_input_field_label": "消息",
                        "active_input_multiline": False,
                    },
                }
            }
        }

        observed = SingleStepGenericSceneObserver(provider).observe(
            frames=stable_frames(),
            goal_context=context,
            device_id="device-local-01",
        )

        self.assertEqual(provider.calls, 1)
        self.assertEqual(provider.call_options["max_attempts"], 1)
        prompt = json.dumps(provider.messages_seen[0], ensure_ascii=False)
        self.assertIn("A blank input is valid", prompt)
        self.assertIn("does not need placeholder, caret", prompt)
        self.assertIn("Text input, clearing and newline use ADB Keyboard", prompt)
        self.assertIn("scene中的role=input只是可选页面上下文", prompt)
        self.assertNotIn('local_audited_input_1', prompt)
        self.assertEqual(
            unique_goal_element(observed).element_id,
            "local_audited_input_1",
        )

    def test_input_target_miss_does_not_create_an_execution_target(self):
        wrong_scene = scene_payload()
        wrong_scene.update(
            {
                "foreground_app_id": "com.example.messaging",
                "screen_id": "wrong_named_conversation",
                "summary": "进入了相邻会话，当前没有消息输入框",
                "elements": [],
            }
        )
        envelope = {
            "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
            "coordinate_space": {
                "kind": "normalized_1000",
                "width": 1000,
                "height": 1000,
            },
            "scene": wrong_scene,
            "input_structure": input_audit_payload(),
        }
        context = {
            "entities": {
                "active_subgoal_visual_context": {
                    "subgoal_id": "input_message",
                    "objective": "在消息输入框输入abc",
                    "constraints": [],
                    "completion_conditions": ["消息输入框内容为abc"],
                    "execution_class": "navigate",
                    "goal_entities": {
                        "active_input_transaction_text": "abc",
                        "active_input_field_id": "message_field",
                        "active_input_field_label": "消息",
                        "active_input_multiline": False,
                        "observation_phase": "untrusted-successor-preview",
                    },
                }
            }
        }

        provider = SequenceProvider([envelope])
        observed = SingleStepGenericSceneObserver(provider).observe(
            frames=stable_frames(), goal_context=context, device_id="device-local-01")

        self.assertIsNone(unique_goal_element(observed))
        self.assertEqual(1, provider.calls)

    def test_scene_only_input_does_not_create_second_input_authority(self):
        compact_scene = scene_payload()
        compact_scene.update(
            {
                "foreground_app_id": "com.example.messaging",
                "screen_id": "named_conversation",
                "summary": "指定会话页底部有一个空白编辑面",
                "elements": [
                    {
                        "element_id": "coarse-input",
                        "role": "input",
                        "bounds": [120, 910, 780, 960],
                        "states": {
                            "goal_relevant": False,
                            "fully_visible": True,
                        },
                    }
                ],
            }
        )
        envelope = {
            "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
            "coordinate_space": {
                "kind": "normalized_1000",
                "width": 1000,
                "height": 1000,
            },
            "scene": compact_scene,
            "input_structure": input_audit_payload(),
        }
        context = {
            "entities": {
                "active_subgoal_visual_context": {
                    "subgoal_id": "input_message",
                    "objective": "在消息输入框输入abc",
                    "constraints": [],
                    "completion_conditions": ["消息输入框内容为abc"],
                    "execution_class": "navigate",
                    "goal_entities": {
                        "active_input_transaction_text": "abc",
                        "active_input_field_id": "message_field",
                        "active_input_field_label": "消息",
                        "active_input_multiline": False,
                    },
                }
            }
        }

        for with_spoofed_phase in (False, True):
            with self.subTest(with_spoofed_phase=with_spoofed_phase):
                current_context = json.loads(json.dumps(context, ensure_ascii=False))
                if with_spoofed_phase:
                    current_context["entities"]["active_subgoal_visual_context"]["goal_entities"][
                        "observation_phase"
                    ] = "untrusted-successor-preview"
                observed = SingleStepGenericSceneObserver(SequenceProvider([envelope])).observe(
                    frames=stable_frames(),
                    goal_context=current_context,
                    device_id="device-local-01",
                )

                target = unique_goal_element(observed)
                self.assertIsNone(target)
                self.assertFalse(any(item.role == 'input' for item in observed.elements))

    def test_ambiguous_or_unbound_focus_surfaces_do_not_create_local_targets_or_rewrite_qwen_facts(self):
        base_input = {
            "element_id": "coarse-input-1",
            "role": "input",
            "meaning": "form_text_field",
            "label": "备注",
            "bounds": [100, 300, 900, 390],
            "confidence": 0.99,
            "states": {"goal_relevant": True, "fully_visible": True},
            "evidence": ["表单中完整可见的备注编辑区域"],
        }
        context = {
            "entities": {
                "active_subgoal_visual_context": {
                    "subgoal_id": "input_notes",
                    "objective": "在备注字段输入release",
                    "constraints": [],
                    "completion_conditions": ["备注字段为release"],
                    "execution_class": "navigate",
                    "goal_entities": {
                        "active_input_transaction_text": "release",
                        "active_input_field_id": "notes_field",
                        "active_input_field_label": "备注",
                        "active_input_multiline": False,
                    },
                }
            }
        }
        cases = (
            (
                "two-inputs",
                [
                    base_input,
                    {
                        **base_input,
                        "element_id": "coarse-input-2",
                        "label": "正文",
                        "bounds": [100, 430, 900, 520],
                        "evidence": ["表单中另一个完整可见的正文编辑区域"],
                    },
                ],
                input_audit_payload(),
            ),
            (
                "keyboard-visible",
                [base_input],
                input_audit_payload(
                    keyboard={
                        "visible": True,
                        "bounds": [50, 600, 950, 990],
                        "layout": "numeric",
                        "input_mode": "unknown",
                        "mode_switch": None,
                    }
                ),
            ),
        )
        for name, elements, audit in cases:
            with self.subTest(name=name):
                candidate_scene = scene_payload()
                candidate_scene.update(
                    {
                        "foreground_app_id": "com.example.form",
                        "screen_id": "edit_form",
                        "elements": elements,
                    }
                )
                envelope = {
                    "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
                    "coordinate_space": {
                        "kind": "normalized_1000",
                        "width": 1000,
                        "height": 1000,
                    },
                    "scene": candidate_scene,
                    "input_structure": audit,
                }
                observed = SingleStepGenericSceneObserver(SequenceProvider([envelope])).observe(
                    frames=stable_frames(), goal_context=context, device_id="device-local-01")
                self.assertFalse(any(item.element_id.startswith("local_audited_") for item in observed.elements))
                self.assertFalse(any(item.role == 'input' for item in observed.elements))
                self.assertIsNone(unique_goal_element(observed))

    def test_single_step_observer_uses_only_input_structure_for_blank_value(
        self,
    ) -> None:
        context = {
            "entities": {
                "active_subgoal_visual_context": {
                    "subgoal_id": "input_message",
                    "objective": "在输入框中输入消息",
                    "constraints": [],
                    "completion_conditions": ["输入框显示指定消息"],
                    "execution_class": "navigate",
                    "goal_entities": {
                        "active_input_transaction_text": "aaazjie？你好",
                        "active_input_field_id": "message_field",
                        "active_input_multiline": False,
                    },
                }
            }
        }
        cases = (
            (
                "harmless-extra",
                "com.example.messaging",
                "named_conversation",
                {"bounds": [120, 910, 780, 960], "text": "", "visual_note": "optional metadata"},
            ),
            (
                "low-confidence",
                "com.example.notes",
                "edit_note",
                {"bounds": [100, 300, 900, 390], "confidence": 0.01},
            ),
            (
                "no-fully-visible",
                "com.example.forms",
                "edit_form",
                {"bounds": [80, 420, 920, 510]},
            ),
        )
        for name, app_id, screen_id, audit_item in cases:
            with self.subTest(name=name):
                scene = scene_payload()
                scene.update(
                    {
                        "foreground_app_id": app_id,
                        "screen_id": screen_id,
                        "summary": "当前页面有一个空输入框",
                        "elements": [],
                    }
                )
                focused_audit_item = dict(audit_item)
                focused_audit_item["caret_line_index"] = 0
                audit = input_audit_payload(application_inputs=[focused_audit_item])
                provider = SequenceProvider(
                    [
                        {
                            "protocol_version": (SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION),
                            "coordinate_space": {
                                "kind": "normalized_1000",
                                "width": 1000,
                                "height": 1000,
                            },
                            "scene": scene,
                            "input_structure": audit,
                            "decision": {"status": "action", "action": "input_verified_text", "text": "sample",
                                "element_id": "local_audited_input_1"},
                        }
                    ]
                )

                observer = SingleStepGenericSceneObserver(provider)
                observed, decision = observer.observe_with_decision(
                    frames=stable_frames(),
                    goal_context=context,
                    device_id="device-local-01",
                )

                self.assertEqual(provider.calls, 1)
                field = unique_goal_element(observed)
                self.assertEqual("local_audited_input_1", field.element_id)
                self.assertEqual("", field.states["value"])
                self.assertNotIn("same_frame_input_surface_evidence", field.states)
                self.assertEqual("input_verified_text", decision["action"])
                self.assertEqual(1.0, decision["confidence"])
                self.assertTrue(decision["reason"])

    def test_blank_input_preserves_explicit_same_frame_focus_without_keyboard(
        self,
    ) -> None:
        context = {
            "entities": {
                "active_subgoal_visual_context": {
                    "subgoal_id": "input_text",
                    "objective": "在当前唯一输入框中输入文字",
                    "constraints": [],
                    "completion_conditions": ["输入框显示指定文字"],
                    "execution_class": "navigate",
                    "goal_entities": {
                        "active_input_transaction_text": "sample text",
                        "active_input_field_id": "current_field",
                        "active_input_multiline": False,
                    },
                }
            }
        }
        cases = (
            ("cursor", ["cursor"], None, True),
            ("caret", ["caret"], None, True),
            ("chinese-cursor", ["光标"], None, True),
            ("insertion-mark", ["插入符"], None, True),
            ("insertion-mark-in-field", ["插入符位于空输入框内"], None, True),
            ("caret-line-zero", [], 0, True),
            ("no-cue", [], None, False),
            ("border", ["complete input border"], None, False),
            ("outline", ["input outline"], None, False),
            ("negated-cursor", ["no cursor"], None, False),
            ("hidden-caret", ["caret hidden"], None, False),
            ("invisible-chinese-cursor", ["光标不可见"], None, False),
        )
        for index, (name, cues, caret_line_index, expected_focused) in enumerate(cases, start=1):
            with self.subTest(name=name):
                scene = scene_payload()
                scene.update(
                    {
                        "foreground_app_id": f"com.example.editor{index}",
                        "screen_id": "edit_text",
                        "summary": "当前页面有一个空输入框",
                        "elements": [],
                    }
                )
                audit = input_audit_payload(
                    application_inputs=[
                        audited_application_input(
                            structure_id="current-input",
                            bounds=[100, 720, 900, 820],
                            text="",
                            visible_editable_cues=cues, focused=expected_focused,
                            caret_line_index=caret_line_index,
                        )
                    ],
                    keyboard={
                        "visible": False,
                        "bounds": None,
                        "layout": "unknown",
                        "input_mode": "unknown",
                        "mode_switch": None,
                    },
                )

                observed = SingleStepGenericSceneObserver(SequenceProvider([{
                    "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
                    "coordinate_space": {
                        "kind": "normalized_1000",
                        "width": 1000,
                        "height": 1000,
                    },
                    "scene": scene,
                    "input_structure": audit,
                }])).observe(
                    frames=stable_frames(),
                    goal_context=context,
                    device_id="device-local-01",
                )

                field = observed.get_element("local_audited_input_1")
                self.assertEqual(expected_focused, field.states.get("focused") is True)
                self.assertNotIn("keyboard_geometry", field.states)

    def test_single_step_prompt_requires_focus_before_each_typed_operation(self) -> None:
        for operation in ("input_verified_text", "clear_verified_text", "press_enter"):
            with self.subTest(operation=operation):
                scene = scene_payload()
                scene.update({"foreground_app_id": "com.example.notes", "screen_id": "editor",
                    "summary": "当前页面有一个空白编辑框", "elements": []})
                audit = input_audit_payload(application_inputs=[audited_application_input(
                    structure_id="note-body", bounds=[100, 700, 900, 820], text="", focused=None,
                    visible_editable_cues=[])])
                provider = SequenceProvider([{
                    "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
                    "coordinate_space": {"kind": "normalized_1000", "width": 1000, "height": 1000},
                    "scene": scene, "input_structure": audit,
                    "decision": {"status": "action", "action": "tap_semantic",
                        "element_id": "local_audited_input_1", "tap_point": [500, 900]},
                }])
                context = {"entities": {"active_subgoal_visual_context": {
                    "subgoal_id": "edit_note", "objective": "修改指定文字", "constraints": [],
                    "completion_conditions": ["编辑框达到目标状态"], "execution_class": "navigate",
                    "goal_entities": {"active_input_transaction_text": "sample text",
                        "active_input_field_id": "note_body", "active_input_operation": operation,
                        "active_input_multiline": operation == "press_enter"}}}}

                observed, decision = SingleStepGenericSceneObserver(provider).observe_with_decision(
                    frames=stable_frames(), goal_context=context, device_id="device-local-01")

                field = observed.get_element("local_audited_input_1")
                self.assertIsNot(field.states.get("focused"), True)
                self.assertEqual("current_input", field.states["input_field_id"])
                self.assertEqual("tap_semantic", decision["action"])
                prompt = json.dumps(provider.messages_seen[0], ensure_ascii=False)
                for text in ("未知或未聚焦时先点该字段再取新图", "不选择Enter键",
                    "先单独clear_verified_text", "旧聊天气泡", "不能一次清空并输入"):
                    self.assertIn(text, prompt)

    def test_same_response_input_is_projected_once_without_scene_identity(self) -> None:
        scene = scene_payload()
        scene.update({
            "foreground_app_id": "com.tencent.mm",
            "screen_id": "chat_file_transfer_assistant",
            "summary": "文件传输助手聊天界面，底部空白输入框可见",
            "elements": [{
                "element_id": "e2",
                "role": "input",
                "meaning": "message_input_field",
                "label": "",
                "bounds": [120, 1160, 780, 1230],
                "confidence": 1.0,
                "states": {"goal_relevant": True, "fully_visible": True},
                "evidence": ["底部工具栏中央空白长条区域"],
            }],
        })
        context = {"entities": {"active_subgoal_visual_context": {
            "subgoal_id": "input_text",
            "objective": "在消息输入框输入 aaazjie？你好",
            "constraints": [],
            "completion_conditions": ["输入框内容为 aaazjie？你好"],
            "execution_class": "navigate",
            "goal_entities": {
                "active_input_transaction_text": "aaazjie？你好",
                "active_input_field_id": "message_field",
                "active_input_multiline": False,
            },
        }}}
        envelope = {
            "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
            "coordinate_space": {"kind": "axis_grid", "width": 1000, "height": 1280},
            "scene": scene,
            "input_structure": input_audit_payload(
                application_inputs=[{"bounds": [120, 1160, 780, 1230], "text": ""}],
            ),
            "decision": {"status": "action", "action": "tap_semantic", "element_id": "e2",
                "tap_point": [450, 1195]},
        }
        observer = SingleStepGenericSceneObserver(SequenceProvider([envelope]))

        observed, model_decision = observer.observe_with_decision(
            frames=[Image.new("RGB", (720, 1280), (30, 40, 50)) for _ in range(4)],
            goal_context=context,
            device_id="device-local-01",
        )

        field = observed.get_element("local_audited_input_1")
        self.assertIsNotNone(field)
        self.assertEqual("", field.states["value"])
        self.assertEqual(1, sum(item.role == 'input' for item in observed.elements))
        self.assertEqual("e2", model_decision["target"]["element_id"])

    def test_input_audit_uses_current_text_and_ignores_qwen_input_goal_marker(self) -> None:
        context = {"entities": {"active_subgoal_visual_context": {
            "subgoal_id": "input_text",
            "objective": "在当前输入框继续输入",
            "constraints": [],
            "completion_conditions": ["输入框显示目标文字"],
            "execution_class": "navigate",
            "goal_entities": {
                "active_input_transaction_text": "fresh-value",
                "active_input_field_id": "message_field",
                "active_input_multiline": False,
            },
        }}}
        for current_text, other_marker in (("fresh-value", False), ("first\nsecond", True)):
            with self.subTest(current_text=current_text):
                current_context = json.loads(json.dumps(context, ensure_ascii=False))
                current_context["entities"]["active_subgoal_visual_context"]["goal_entities"][
                    "active_input_transaction_text"] = current_text
                scene = scene_payload()
                scene.update({
                    "foreground_app_id": "com.example.messaging",
                    "screen_id": "conversation",
                    "summary": "当前页面有一个消息输入框和一段页面说明",
                    "elements": [{
                        "element_id": "message-input",
                        "role": "input",
                        "meaning": "application_text_input",
                        "label": "",
                        "bounds": [100, 500, 800, 590],
                        "confidence": 1.0,
                        "states": {"goal_relevant": False, "fully_visible": True,
                            "value": "stale-scene-value"},
                        "evidence": ["当前截图中的完整编辑栏"],
                    }, {
                        "element_id": "page-note",
                        "role": "text",
                        "meaning": "page_note",
                        "label": "页面说明",
                        "bounds": [100, 200, 500, 260],
                        "confidence": 1.0,
                        "states": {"goal_relevant": other_marker, "fully_visible": True},
                        "evidence": ["页面说明逐字可见"],
                    }],
                })
                audit = input_audit_payload(application_inputs=[{
                    "element_id": "message-input",
                    "bounds": [100, 500, 800, 590],
                    "text": current_text,
                    "visible_editable_cues": ["stale-lineage-value", "cursor"],
                }])
                observer = SingleStepGenericSceneObserver(SequenceProvider([{
                    "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
                    "coordinate_space": {"kind": "normalized_1000", "width": 1000, "height": 1000},
                    "scene": scene,
                    "input_structure": audit,
                    "decision": {"status": "action", "action": "input_verified_text", "text": "sample",
                        "element_id": "message-input"},
                }]))

                observed = observer.observe(
                    frames=stable_frames(), goal_context=current_context, device_id="device-local-01")

                field = observed.get_element("local_audited_input_1")
                self.assertEqual(current_text, field.states["value"])
                self.assertIs(field.states["goal_relevant"], True)
                self.assertIs(other_marker, observed.get_element("page-note").states["goal_relevant"])

    def test_cross_app_blank_form_keeps_scene_identity_and_input_structure_value(self) -> None:
        scene = scene_payload()
        scene.update({
            "foreground_app_id": "com.example.forms",
            "screen_id": "new_contact_form",
            "summary": "联系人表单中唯一空白备注输入框可见",
            "elements": [{
                "element_id": "form-note",
                "role": "input",
                "meaning": "note_input_field",
                "label": "备注",
                "bounds": [90, 300, 910, 400],
                "confidence": 0.82,
                "states": {
                    "goal_relevant": True,
                    "fully_visible": True,
                    "value": "scene-transcription-must-not-win",
                },
                "evidence": ["备注标签右侧完整编辑栏"],
            }],
        })
        context = {"entities": {"active_subgoal_visual_context": {
            "subgoal_id": "input_note",
            "objective": "在备注输入框输入 follow-up",
            "constraints": [],
            "completion_conditions": ["备注输入框内容为 follow-up"],
            "execution_class": "navigate",
            "goal_entities": {
                "active_input_transaction_text": "follow-up",
                "active_input_field_id": "note_field",
                "active_input_field_label": "备注",
                "active_input_multiline": False,
            },
        }}}
        provider = SequenceProvider([{
            "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
            "coordinate_space": {"kind": "normalized_1000", "width": 1000, "height": 1000},
            "scene": scene,
            "input_structure": input_audit_payload(application_inputs=[{
                "element_id": "form-note",
                "bounds": [90, 300, 910, 400],
                "text": "",
            }]),
            "decision": {"status": "action", "action": "tap_semantic", "element_id": "form-note",
                "tap_point": [500, 350]},
        }])
        observer = SingleStepGenericSceneObserver(provider)

        observed, model_decision = observer.observe_with_decision(
            frames=stable_frames(), goal_context=context, device_id="device-local-01")

        field = observed.get_element("local_audited_input_1")
        self.assertIsNotNone(field)
        self.assertEqual("", field.states["value"])
        self.assertEqual("current_input", field.states["input_field_id"])
        self.assertEqual(1, sum(item.role == 'input' for item in observed.elements))
        self.assertEqual("form-note", model_decision["target"]["element_id"])

    def test_blank_input_with_invalid_bounds_is_still_rejected(self) -> None:
        scene = scene_payload()
        scene.update(
            {
                "foreground_app_id": "com.example.forms",
                "screen_id": "edit_form",
                "summary": "当前页面有一个空输入框",
                "elements": [],
            }
        )
        context = {
            "entities": {
                "active_subgoal_visual_context": {
                    "subgoal_id": "input_message",
                    "objective": "在输入框输入abc",
                    "constraints": [],
                    "completion_conditions": ["输入框内容为abc"],
                    "execution_class": "navigate",
                    "goal_entities": {
                        "active_input_transaction_text": "abc",
                        "active_input_field_id": "message_field",
                        "active_input_multiline": False,
                    },
                }
            }
        }
        provider = SequenceProvider(
            [
                {
                    "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
                    "coordinate_space": {
                        "kind": "normalized_1000",
                        "width": 1000,
                        "height": 1000,
                    },
                    "scene": scene,
                    "input_structure": input_audit_payload(
                        application_inputs=[{"bounds": [100, 300, 100, 390]}]
                    ),
                    "decision": {
                        "status": "action",
                        "action": "input_verified_text", "text": "sample",
                        "element_id": "local_audited_input_1",
                    },
                }
            ]
        )

        with self.assertRaisesRegex(VisionAgentError, "bounds"):
            SingleStepGenericSceneObserver(provider).observe(
                frames=stable_frames(),
                goal_context=context,
                device_id="device-local-01",
            )

    def test_optional_invalid_and_surplus_input_facts_do_not_veto_valid_blank_input(self) -> None:
        scene = scene_payload()
        scene.update({
            "foreground_app_id": "com.example.forms",
            "screen_id": "edit_form",
            "summary": "当前页面有一个空输入框",
            "elements": [],
        })
        context = {"entities": {"active_subgoal_visual_context": {
            "subgoal_id": "input_message",
            "objective": "在输入框输入abc",
            "constraints": [],
            "completion_conditions": ["输入框内容为abc"],
            "execution_class": "navigate",
            "goal_entities": {
                "active_input_transaction_text": "abc",
                "active_input_field_id": "message_field",
                "active_input_multiline": False,
            },
        }}}
        invalid_inputs = [{"bounds": [100, 300, 100, 390], "text": "noise"} for _ in range(6)]
        invalid_preedits = [{
            "region_id": f"optional-{index}",
            "bounds": [100, 500, 100, 550],
            "text": "noise",
            "candidates": [{"text": "noise", "bounds": [80, 610, 80, 650]}] * 10,
        } for index in range(6)]
        audit = input_audit_payload(
            application_inputs=[{"bounds": [100, 300, 900, 390], "text": ""}, *invalid_inputs],
            ime_preedit_regions=invalid_preedits,
            keyboard={
                "visible": True,
                "bounds": [0, 650, 0, 1000],
                "layout": "qwerty",
                "input_mode": "direct_latin",
                "mode_switch": None,
            },
        )
        provider = SequenceProvider([{
            "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
            "coordinate_space": {"kind": "normalized_1000", "width": 1000, "height": 1000},
            "scene": scene,
            "input_structure": audit,
            "decision": {"status": "action", "action": "input_verified_text", "text": "sample",
                "element_id": "local_audited_input_1"},
        }])

        observed = SingleStepGenericSceneObserver(provider).observe(
            frames=stable_frames(), goal_context=context, device_id="device-local-01")

        field = unique_goal_element(observed)
        self.assertEqual("local_audited_input_1", field.element_id)
        self.assertEqual("", field.states["value"])

    def test_local_obstruction_is_diagnostic_and_does_not_rewrite_qwen_input_fact(self) -> None:
        scene = scene_payload()
        scene["elements"] = [{
            "element_id": "search-field",
            "role": "input",
            "meaning": "search_input",
            "label": "搜索",
            "bounds": [100, 50, 700, 150],
            "confidence": 1.0,
            "states": {"goal_relevant": True, "fully_visible": True, "focused": True},
            "evidence": ["当前截图显示完整搜索框"],
        }]
        provider = SequenceProvider([{
            "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
            "coordinate_space": {"kind": "normalized_1000", "width": 1000, "height": 1000},
            "scene": scene,
            "decision": {"status": "action", "action": "tap_semantic",
                "element_id": "search-field", "tap_point": [500, 500]},
        }])
        obstruction = VisualObstruction(kind="top_edge_opaque_band", bounds=(100, 0, 700, 100),
            reason="本地检测到顶边遮挡")

        with patch("agent.infrastructure.generic_scene_observer.consensus_top_edge_obstructions",
            return_value=(obstruction,)):
            observer = SingleStepGenericSceneObserver(provider)
            observed, model_decision = observer.observe_with_decision(
                frames=stable_frames(), goal_context={"objective": "聚焦搜索框"},
                device_id="device-local-01")

        self.assertEqual((), observed.elements)
        self.assertEqual("search-field", model_decision["target"]["element_id"])
        self.assertEqual("input", model_decision["target"]["role"])
        self.assertEqual([obstruction.to_dict()], observer.last_diagnostics["local_visual_obstructions"])

    def test_single_step_observer_keeps_input_when_optional_preedit_bounds_are_broad(
        self,
    ) -> None:
        context = {
            "entities": {
                "active_subgoal_visual_context": {
                    "subgoal_id": "input_message",
                    "objective": "在输入框中输入指定文字",
                    "constraints": [],
                    "completion_conditions": ["输入框显示指定文字"],
                    "execution_class": "navigate",
                    "goal_entities": {
                        "active_input_transaction_text": "aaazjie？你好",
                        "active_input_field_id": "message_field",
                        "active_input_multiline": False,
                    },
                }
            }
        }
        for app_id, screen_id in (
            ("com.example.messaging", "named_conversation"),
            ("com.example.notes", "edit_note"),
        ):
            with self.subTest(app_id=app_id, screen_id=screen_id):
                input_bounds = [130, 500, 720, 590]
                scene = scene_payload()
                scene.update(
                    {
                        "foreground_app_id": app_id,
                        "screen_id": screen_id,
                        "summary": "当前页面的唯一输入框内有带下划线的拉丁预编辑",
                        "elements": [
                            {
                                "element_id": "e1",
                                "role": "input",
                                "meaning": "application_text_input",
                                "label": "",
                                "bounds": input_bounds,
                                "confidence": 1.0,
                                "states": {
                                    "goal_relevant": True,
                                    "fully_visible": True,
                                },
                                "evidence": ["唯一完整输入表面内可见下划线 aaazjie"],
                            }
                        ],
                    }
                )
                audit = input_audit_payload(
                    application_inputs=[
                        audited_application_input(
                            structure_id="message",
                            bounds=input_bounds,
                            text="",
                            placeholder="",
                            visible_editable_cues=["aaazjie"],
                            preedit_text="aaazjie",
                            caret_line_index=0,
                        )
                    ],
                    ime_preedit_regions=[
                        {
                            "region_id": "preedit",
                            # The optional preedit geometry is deliberately as
                            # broad as the whole input surface.  Its own
                            # inaccuracy must not erase the independently
                            # established application input or exact candidate.
                            "bounds": input_bounds,
                            "text": "aaazjie",
                            "confidence": 0.99,
                            "candidates": [
                                {
                                    "text": "aaazjie",
                                    "bounds": [80, 610, 300, 645],
                                    "confidence": 0.99,
                                    "fully_visible": True,
                                }
                            ],
                        }
                    ],
                    keyboard={
                        "visible": True,
                        "bounds": [0, 650, 1000, 1000],
                        "layout": "qwerty",
                        "input_mode": "direct_latin",
                        "case_mode": "lower",
                        "mode_switch": None,
                    },
                )
                observed = SingleStepGenericSceneObserver(
                    SequenceProvider(
                        [
                            {
                                "protocol_version": (SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION),
                                "coordinate_space": {
                                    "kind": "normalized_1000",
                                    "width": 1000,
                                    "height": 1000,
                                },
                                "scene": scene,
                                "input_structure": audit,
                            }
                        ]
                    )
                ).observe(
                    frames=stable_frames(),
                    goal_context=context,
                    device_id="device-local-01",
                )

                field = observed.get_element("local_audited_input_1")
                candidates = [element for element in observed.elements if element.meaning == "ime_exact_candidate"]
                self.assertIsNotNone(field)
                self.assertEqual("", field.states["value"])
                self.assertEqual("aaazjie", field.states["ime_preedit_text"])
                self.assertEqual([], candidates)

    def test_single_step_observer_fills_the_only_missing_nested_audit_version(self) -> None:
        scene = scene_payload()
        scene.update(
            {
                "foreground_app_id": "sample_chat",
                "screen_id": "conversation",
                "summary": "消息输入框可见",
                "elements": [],
            }
        )
        audit = input_audit_payload(
            application_inputs=[
                audited_application_input(
                    structure_id="message",
                    bounds=[100, 720, 900, 820],
                    text="",
                    placeholder="消息",
                    field_labels=["消息"],
                    visible_editable_cues=["消息输入框完整边框", "插入光标"],
                    caret_line_index=0,
                )
            ]
        )
        audit.pop("protocol_version")
        provider = SequenceProvider(
            [
                {
                    "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
                    "coordinate_space": {
                        "kind": "normalized_1000",
                        "width": 1000,
                        "height": 1000,
                    },
                    "scene": scene,
                    "input_structure": audit,
                }
            ]
        )
        context = {
            "entities": {
                "active_subgoal_visual_context": {
                    "subgoal_id": "input_message",
                    "objective": "在消息输入框输入abc",
                    "constraints": [],
                    "completion_conditions": ["消息输入框内容为abc"],
                    "execution_class": "navigate",
                    "goal_entities": {
                        "active_input_transaction_text": "abc",
                        "active_input_field_id": "message_field",
                        "active_input_field_label": "消息",
                        "active_input_multiline": False,
                    },
                }
            }
        }

        observed = SingleStepGenericSceneObserver(provider).observe(
            frames=stable_frames(), goal_context=context, device_id="device-local-01")

        self.assertEqual(provider.calls, 1)
        self.assertEqual("local_audited_input_1", unique_goal_element(observed).element_id)

    def test_single_step_observer_ignores_harmless_nested_audit_metadata(self) -> None:
        scene = scene_payload()
        audit = input_audit_payload(application_inputs=[{"bounds": [100, 720, 900, 820], "text": ""}])
        audit.pop("protocol_version")
        audit["unexpected"] = True
        provider = SequenceProvider(
            [
                {
                    "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
                    "coordinate_space": {
                        "kind": "normalized_1000",
                        "width": 1000,
                        "height": 1000,
                    },
                    "scene": scene,
                    "input_structure": audit,
                }
            ]
        )
        context = {
            "entities": {
                "active_subgoal_visual_context": {
                    "subgoal_id": "input_message",
                    "objective": "在消息输入框输入abc",
                    "constraints": [],
                    "completion_conditions": ["消息输入框内容为abc"],
                    "execution_class": "navigate",
                    "goal_entities": {
                        "active_input_transaction_text": "abc",
                        "active_input_field_id": "message_field",
                        "active_input_field_label": "消息",
                        "active_input_multiline": False,
                    },
                }
            }
        }

        observed = SingleStepGenericSceneObserver(provider).observe(
            frames=stable_frames(), goal_context=context, device_id="device-local-01")

        self.assertEqual(provider.calls, 1)
        self.assertEqual("local_audited_input_1", unique_goal_element(observed).element_id)

    def test_single_step_observer_rejects_explicit_conflicting_nested_audit_version(self) -> None:
        scene = scene_payload()
        audit = input_audit_payload(application_inputs=[])
        audit["protocol_version"] = "conflicting-version"
        provider = SequenceProvider([{
            "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
            "coordinate_space": {"kind": "normalized_1000", "width": 1000, "height": 1000},
            "scene": scene, "input_structure": audit,
        }])
        context = {"entities": {"active_subgoal_visual_context": {
            "subgoal_id": "input_message", "objective": "在消息输入框输入abc", "constraints": [],
            "completion_conditions": ["消息输入框内容为abc"], "execution_class": "navigate",
            "goal_entities": {"active_input_transaction_text": "abc",
                "active_input_field_id": "message_field", "active_input_multiline": False},
        }}}

        with self.assertRaisesRegex(VisionAgentError, "协议版本不匹配"):
            SingleStepGenericSceneObserver(provider).observe(
                frames=stable_frames(), goal_context=context, device_id="device-local-01")
        self.assertEqual(1, provider.calls)

    def test_single_step_non_input_ignores_unselected_overflow_geometry(self) -> None:
        scene = scene_payload()
        scene.update(
            {
                "foreground_app_id": "com.vendor.runtime",
                "screen_id": "sample_app_conversation",
                "summary": "示例应用当前页面可见",
                "elements": [
                    {
                        "element_id": "title",
                        "role": "text",
                        "meaning": "page_title",
                        "label": "当前会话",
                        "bounds": [300, 20, 700, 80],
                        "confidence": 1.0,
                        "states": {"goal_relevant": True, "fully_visible": True},
                        "evidence": ["顶部标题"],
                    },
                    {
                        "element_id": "unusable-bottom-control",
                        "role": "button",
                        "meaning": "unrelated_action",
                        "label": "其他操作",
                        "bounds": [780, 950, 960, 1040],
                        "confidence": 1.0,
                        "states": {"goal_relevant": True, "fully_visible": True},
                        "evidence": ["底部控件"],
                    },
                ],
            }
        )
        provider = SequenceProvider(
            [
                {
                    "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
                    "coordinate_space": {
                        "kind": "axis_grid",
                        "width": 1000,
                        "height": 960,
                    },
                    "scene": scene,
                    "input_structure": None,
                }
            ]
        )

        observed = SingleStepGenericSceneObserver(provider).observe(
            frames=stable_frames(),
            goal_context={
                "app_id": "sample_app",
                "app_name": "示例应用",
                "entities": {
                    "active_subgoal_visual_context": {
                        "subgoal_id": "launch_sample_app",
                        "objective": "打开示例应用",
                        "constraints": [],
                        "completion_conditions": ["示例应用已打开"],
                        "execution_class": "navigate",
                        "goal_entities": {},
                    }
                },
            },
            device_id="device-local-01",
        )

        self.assertEqual(1, provider.calls)
        self.assertEqual(["title"], [item.element_id for item in observed.elements])

    def test_single_step_input_revokes_unselected_overflow_compact_geometry(self) -> None:
        scene = scene_payload()
        scene["elements"] = [
            {
                "element_id": "input-overflow",
                "role": "input",
                "meaning": "message_input",
                "label": "",
                "bounds": [80, 960, 720, 1030],
                "confidence": 1.0,
                "states": {"goal_relevant": True, "fully_visible": True},
                "evidence": ["输入框"],
            }
        ]
        provider = SequenceProvider(
            [
                {
                    "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
                    "coordinate_space": {
                        "kind": "normalized_1000",
                        "width": 1000,
                        "height": 1000,
                    },
                    "scene": scene,
                    "input_structure": input_audit_payload(application_inputs=[]),
                }
            ]
        )
        context = {
            "entities": {
                "active_subgoal_visual_context": {
                    "subgoal_id": "input_message",
                    "objective": "在输入框输入abc",
                    "constraints": [],
                    "completion_conditions": ["输入框内容为abc"],
                    "execution_class": "navigate",
                    "goal_entities": {
                        "active_input_transaction_text": "abc",
                        "active_input_field_id": "message_field",
                        "active_input_multiline": False,
                    },
                }
            }
        }

        observed = SingleStepGenericSceneObserver(provider).observe(
            frames=stable_frames(), goal_context=context, device_id="device-local-01")

        self.assertEqual(1, provider.calls)
        self.assertIsNone(unique_goal_element(observed))


if __name__ == "__main__":
    unittest.main()
