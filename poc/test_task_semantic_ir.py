import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from deepseek_task_graph import (
    DeepSeekTaskGraphPlanner,
    TaskGraphError,
    _graph_from_payload,
)
from deepseek_failure_diagnostics import persist_deepseek_failure_diagnostic
from task_semantic_ir import (
    AUTOMATIC,
    CONFIRMATION_REQUIRED,
    CriticalBinding,
    EffectIntent,
    LocalRiskPolicyConfig,
    SemanticEntity,
    SemanticRiskAuthorityReport,
    SourceSpan,
    TaskSemanticIR,
    TaskSemanticIRError,
    compile_runtime_graph_semantics,
    compile_formal_semantic_authority,
    apply_formal_semantic_risk_policy,
    load_local_risk_policy,
    local_risk_policy_from_dict,
)


RAW_GOAL = "打开微信，进入文件传输助手，输入“你好”，然后发送。"


def current_send_failure_payload(constraint="确认发送对象为文件传输助手"):
    return {
        "status": "ready",
        "goal": {
            "objective": RAW_GOAL,
            "target_apps": [{"app_id": "wechat", "app_name": "微信"}],
            "entities": {"recipient": "文件传输助手", "input_text": "你好"},
        },
        "constraints": ["不得发送除指定消息外的任何内容"],
        "completion_conditions": [
            {
                "condition_id": "message_sent",
                "description": "消息“你好”已发送给文件传输助手",
                "evidence_required": ["聊天界面显示已发送的消息“你好”"],
                "satisfied": False,
                "evidence": [],
            }
        ],
        "effect_intents": [
            {
                "effect_id": "send_message",
                "kind": "send_message",
                "target_entity_roles": ["recipient"],
                "payload_entity_roles": ["input_text"],
                "source_subgoal_ids": ["send_message"],
                "expected_results": ["消息“你好”已发送"],
            }
        ],
        "subgoals": [
            {
                "subgoal_id": "open_wechat",
                "objective": "打开微信应用",
                "status": "active",
                "depends_on": [],
                "constraints": ["仅导航"],
                "completion_conditions": ["微信主界面可见"],
                "completion_evidence": [],
                "effect_ids": [],
                "execution_class": "navigate",
            },
            {
                "subgoal_id": "send_message",
                "objective": "输入并发送消息“你好”",
                "status": "pending",
                "depends_on": ["open_wechat"],
                "constraints": [constraint],
                "completion_conditions": ["消息“你好”已发送"],
                "completion_evidence": [],
                "effect_ids": ["send_message"],
                "execution_class": "effect",
            },
        ],
        "active_subgoal_id": "open_wechat",
        "clarification_questions": [],
    }


def graph_from_payload(payload=None):
    return _graph_from_payload(
        payload or current_send_failure_payload(),
        task_id="54e036d284fa42b5ae4a1501a96f82bc",
        device_id="device-local-01",
        revision=1,
        raw_user_goal=RAW_GOAL,
    )


def required_actions_for_objective(objective: str) -> set[str]:
    payload = current_send_failure_payload()
    payload["goal"] = {
        "objective": objective,
        "target_apps": [
            {"app_id": "current_foreground", "app_name": "当前前台应用"}
        ],
        "entities": {},
    }
    payload["effect_intents"] = []
    payload["constraints"] = []
    payload["completion_conditions"] = [
        {
            "condition_id": "gesture_completed",
            "description": "目标手势后的页面状态可见",
            "evidence_required": ["动作后的稳定画面"],
            "satisfied": False,
            "evidence": [],
        }
    ]
    payload["subgoals"] = [
        {
            "subgoal_id": "gesture_once",
            "objective": objective,
            "status": "active",
            "depends_on": [],
            "constraints": [],
            "completion_conditions": ["目标手势后的页面状态可见"],
            "completion_evidence": [],
            "effect_ids": [],
            "execution_class": "navigate",
        }
    ]
    payload["active_subgoal_id"] = "gesture_once"
    semantic_ir = compile_formal_semantic_authority(
        _graph_from_payload(
            payload,
            task_id="gesture-task",
            device_id="device-local-01",
            revision=1,
            raw_user_goal=objective,
        )
    ).semantic_ir
    constraints = {item.constraint_id: item for item in semantic_ir.constraints}
    return {
        str(constraints[ref].value)
        for ref in semantic_ir.subgoals[0].constraint_refs
        if constraints[ref].kind == "required_action"
    }


class OneResponseProvider:
    configured = True

    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def chat_json(self, messages, max_tokens=2000):
        self.calls += 1
        return json.dumps(self.payload, ensure_ascii=False)


class RawResponseProvider:
    configured = True

    def __init__(self, raw):
        self.raw = raw

    def chat_json(self, messages, max_tokens=2000):
        return self.raw


class TaskSemanticIRTests(unittest.TestCase):
    def test_negated_action_mentions_do_not_mint_required_actions(self):
        cases = (
            (
                "向上滑动一次，不点击任何列表项",
                {"swipe"},
            ),
            (
                "Swipe up once without clicking any list item",
                {"swipe"},
            ),
            (
                "不要滑动页面，点击继续",
                {"tap_semantic"},
            ),
            (
                "不要点击取消，改为点击继续",
                {"tap_semantic"},
            ),
        )
        for objective, expected in cases:
            with self.subTest(objective=objective):
                self.assertEqual(expected, required_actions_for_objective(objective))

    def test_phone_home_screen_phrase_compiles_launcher_and_home_action(self):
        payload = current_send_failure_payload()
        payload["goal"] = {
            "objective": "回到手机主屏幕",
            "target_apps": [
                {"app_id": "current_foreground", "app_name": "当前前台应用"}
            ],
            "entities": {},
        }
        payload["effect_intents"] = []
        payload["subgoals"] = [
            {
                "subgoal_id": "go_home",
                "objective": "回到手机主屏幕",
                "status": "active",
                "depends_on": [],
                "constraints": [],
                "completion_conditions": ["手机主屏幕可见"],
                "completion_evidence": [],
                "effect_ids": [],
                "execution_class": "navigate",
            }
        ]
        payload["active_subgoal_id"] = "go_home"
        semantic_ir = compile_formal_semantic_authority(
            _graph_from_payload(
                payload,
                task_id="phone-home-task",
                device_id="device-local-01",
                revision=1,
                raw_user_goal="回到手机主屏幕",
            )
        ).semantic_ir
        subgoal = semantic_ir.subgoals[0]
        constraints = {item.constraint_id: item for item in semantic_ir.constraints}
        surfaces = {item.surface_id: item for item in semantic_ir.surfaces}

        self.assertEqual("launcher", surfaces[subgoal.surface_ref].kind)
        self.assertIn(
            "home",
            {
                constraints[ref].value
                for ref in subgoal.constraint_refs
                if constraints[ref].kind == "required_action"
            },
        )

    def test_app_main_screen_phrase_does_not_mint_system_home(self):
        payload = current_send_failure_payload()
        payload["goal"] = {
            "objective": "返回浏览器主屏幕",
            "target_apps": [{"app_id": "browser", "app_name": "浏览器"}],
            "entities": {},
        }
        payload["effect_intents"] = []
        payload["subgoals"] = [
            {
                "subgoal_id": "browser_main",
                "objective": "返回浏览器主屏幕",
                "status": "active",
                "depends_on": [],
                "constraints": [],
                "completion_conditions": ["浏览器主屏幕可见"],
                "completion_evidence": [],
                "effect_ids": [],
                "execution_class": "navigate",
            }
        ]
        payload["active_subgoal_id"] = "browser_main"
        semantic_ir = compile_formal_semantic_authority(
            _graph_from_payload(
                payload,
                task_id="app-main-screen-task",
                device_id="device-local-01",
                revision=1,
                raw_user_goal="返回浏览器主屏幕",
            )
        ).semantic_ir
        constraints = {item.constraint_id: item for item in semantic_ir.constraints}

        self.assertNotIn("launcher", {item.kind for item in semantic_ir.surfaces})
        self.assertNotIn(
            "home",
            {
                constraints[ref].value
                for ref in semantic_ir.subgoals[0].constraint_refs
                if constraints[ref].kind == "required_action"
            },
        )

    def test_single_input_clear_and_type_share_typed_field_ownership(self):
        payload = current_send_failure_payload()
        payload["goal"]["entities"] = {
            "input_text": "longinputvalidation2026:1234567890ABC",
            "target_ui_label": "输入框",
        }
        payload["effect_intents"] = []
        payload["subgoals"] = [
            {
                "subgoal_id": "clear_input",
                "objective": "删除当前输入框中的现有草稿，使输入框变为空白",
                "status": "active",
                "depends_on": [],
                "constraints": ["不得发送"],
                "completion_conditions": ["输入框显示为空"],
                "completion_evidence": [],
                "effect_ids": [],
                "execution_class": "navigate",
            },
            {
                "subgoal_id": "enter_text",
                "objective": (
                    "在输入框中输入 longinputvalidation2026:1234567890ABC"
                ),
                "status": "pending",
                "depends_on": ["clear_input"],
                "constraints": ["不得发送"],
                "completion_conditions": [
                    "输入框逐字等于 longinputvalidation2026:1234567890ABC"
                ],
                "completion_evidence": [],
                "effect_ids": [],
                "execution_class": "navigate",
            },
        ]
        payload["active_subgoal_id"] = "clear_input"
        graph = _graph_from_payload(
            payload,
            task_id="clear-then-input-task",
            device_id="device-local-01",
            revision=1,
            raw_user_goal=(
                "先清空当前草稿，再输入 longinputvalidation2026:1234567890ABC"
            ),
        )

        semantic_ir = compile_formal_semantic_authority(graph).semantic_ir
        constraints = {item.constraint_id: item for item in semantic_ir.constraints}
        clear_subgoal = next(
            item for item in semantic_ir.subgoals if item.subgoal_id == "clear_input"
        )

        self.assertIn("clear_input", semantic_ir.input_fields[0].source_subgoal_ids)
        self.assertIn("enter_text", semantic_ir.input_fields[0].source_subgoal_ids)
        self.assertIn(
            "clear_verified_text",
            {
                constraints[ref].value
                for ref in clear_subgoal.constraint_refs
                if constraints[ref].kind == "required_action"
            },
        )

    def test_multi_field_clear_binds_only_unique_visible_field_label(self):
        payload = current_send_failure_payload()
        payload["goal"]["entities"] = {
            "input_fields": [
                {"field_id": "subject", "field_label": "主题", "text": "周报"},
                {"field_id": "body", "field_label": "正文", "text": "本周完成"},
            ]
        }
        payload["effect_intents"] = []
        payload["subgoals"] = [
            {
                "subgoal_id": "clear_body",
                "objective": "清空正文输入框中的现有草稿",
                "status": "active",
                "depends_on": [],
                "constraints": [],
                "completion_conditions": ["正文输入框为空"],
                "completion_evidence": [],
                "effect_ids": [],
                "execution_class": "navigate",
            },
            {
                "subgoal_id": "enter_subject",
                "objective": "在主题字段输入周报",
                "status": "pending",
                "depends_on": ["clear_body"],
                "constraints": [],
                "completion_conditions": ["主题字段逐字为周报"],
                "completion_evidence": [],
                "effect_ids": [],
                "execution_class": "navigate",
            },
            {
                "subgoal_id": "enter_body",
                "objective": "在正文字段输入本周完成",
                "status": "pending",
                "depends_on": ["enter_subject"],
                "constraints": [],
                "completion_conditions": ["正文字段逐字为本周完成"],
                "completion_evidence": [],
                "effect_ids": [],
                "execution_class": "navigate",
            },
        ]
        payload["active_subgoal_id"] = "clear_body"
        semantic_ir = compile_formal_semantic_authority(
            _graph_from_payload(
                payload,
                task_id="multi-clear-task",
                device_id="device-local-01",
                revision=1,
                raw_user_goal="清空正文后填写主题周报和正文本周完成",
            )
        ).semantic_ir
        fields = {item.field_id: item for item in semantic_ir.input_fields}

        self.assertNotIn("clear_body", fields["subject"].source_subgoal_ids)
        self.assertIn("clear_body", fields["body"].source_subgoal_ids)

    def test_input_carrier_presence_does_not_mint_input_action(self):
        payload = current_send_failure_payload()
        payload["subgoals"][0]["objective"] = (
            "文件传输助手聊天页面和唯一空白消息输入框可见"
        )
        payload["subgoals"][0]["completion_conditions"] = [
            "唯一空白消息输入框可见"
        ]
        payload["subgoals"][1]["objective"] = (
            "在唯一空白消息输入框中输入“你好”"
        )
        graph = graph_from_payload(payload)

        authority = compile_formal_semantic_authority(graph)
        constraints = {
            item.constraint_id: item for item in authority.semantic_ir.constraints
        }
        actions_by_subgoal = {
            item.subgoal_id: {
                constraints[ref].value
                for ref in item.constraint_refs
                if constraints[ref].kind == "required_action"
            }
            for item in authority.semantic_ir.subgoals
        }

        self.assertNotIn("input_verified_text", actions_by_subgoal["open_wechat"])
        self.assertIn("input_verified_text", actions_by_subgoal["send_message"])
        self.assertEqual(
            ("send_message",),
            authority.semantic_ir.input_fields[0].source_subgoal_ids,
        )

    def test_input_capability_state_does_not_mint_input_action(self):
        for objective, completion in (
            ("定位主题字段并使其可见可输入", "主题字段可见且可输入"),
            ("确认主题字段当前可以正常输入且可编辑", "主题字段可编辑"),
        ):
            with self.subTest(objective=objective):
                payload = current_send_failure_payload()
                payload["goal"]["entities"] = {
                    "input_fields": [
                        {
                            "field_id": "subject",
                            "field_label": "主题",
                            "text": "first",
                        },
                        {
                            "field_id": "body",
                            "field_label": "正文",
                            "text": "second",
                        },
                    ]
                }
                payload["effect_intents"] = []
                payload["subgoals"] = [
                    {
                        **payload["subgoals"][0],
                        "objective": objective,
                        "constraints": ["不得输入任何内容"],
                        "completion_conditions": [completion],
                    },
                    {
                        "subgoal_id": "input_subject",
                        "objective": "在主题字段精确输入first",
                        "status": "pending",
                        "depends_on": ["open_wechat"],
                        "constraints": ["不得发送或提交"],
                        "completion_conditions": ["主题字段内容为first"],
                        "completion_evidence": [],
                        "effect_ids": [],
                        "execution_class": "navigate",
                    },
                    {
                        "subgoal_id": "input_body",
                        "objective": "在正文字段精确输入second",
                        "status": "pending",
                        "depends_on": ["input_subject"],
                        "constraints": ["不得发送或提交"],
                        "completion_conditions": ["正文字段内容为second"],
                        "completion_evidence": [],
                        "effect_ids": [],
                        "execution_class": "navigate",
                    },
                ]
                authority = compile_formal_semantic_authority(
                    graph_from_payload(payload)
                )
                constraints = {
                    item.constraint_id: item
                    for item in authority.semantic_ir.constraints
                }
                actions_by_subgoal = {
                    item.subgoal_id: {
                        constraints[ref].value
                        for ref in item.constraint_refs
                        if constraints[ref].kind == "required_action"
                    }
                    for item in authority.semantic_ir.subgoals
                }

                self.assertNotIn(
                    "input_verified_text",
                    actions_by_subgoal["open_wechat"],
                )
                self.assertIn(
                    "input_verified_text",
                    actions_by_subgoal["input_subject"],
                )
                self.assertIn(
                    "input_verified_text",
                    actions_by_subgoal["input_body"],
                )
                fields = {
                    item.field_id: item
                    for item in authority.semantic_ir.input_fields
                }
                self.assertEqual(
                    ("input_subject",),
                    fields["subject"].source_subgoal_ids,
                )
                self.assertEqual(
                    ("input_body",),
                    fields["body"].source_subgoal_ids,
                )

    def test_english_input_carrier_presence_does_not_mint_input_action(self):
        payload = current_send_failure_payload()
        payload["subgoals"][0]["objective"] = (
            "The only input field is visible and empty"
        )
        payload["subgoals"][0]["completion_conditions"] = [
            "The input field is visible"
        ]
        payload["subgoals"][1]["objective"] = "Type 你好 in the input field"
        graph = graph_from_payload(payload)

        authority = compile_formal_semantic_authority(graph)
        constraints = {
            item.constraint_id: item for item in authority.semantic_ir.constraints
        }
        actions_by_subgoal = {
            item.subgoal_id: {
                constraints[ref].value
                for ref in item.constraint_refs
                if constraints[ref].kind == "required_action"
            }
            for item in authority.semantic_ir.subgoals
        }

        self.assertNotIn("input_verified_text", actions_by_subgoal["open_wechat"])
        self.assertIn("input_verified_text", actions_by_subgoal["send_message"])
        self.assertEqual(
            ("send_message",),
            authority.semantic_ir.input_fields[0].source_subgoal_ids,
        )

    def test_input_validation_title_without_typed_payload_does_not_mint_input_action(self):
        payload = current_send_failure_payload()
        payload["goal"]["objective"] = (
            "刷新当前页面，直到看到标题为长文本输入验收且有空白输入框"
        )
        payload["goal"]["entities"] = {}
        payload["effect_intents"] = []
        payload["completion_conditions"] = [
            {
                "condition_id": "validation_page_visible",
                "description": "长文本输入验收标题和空白输入框可见",
                "evidence_required": ["标题和空白输入框同时可见"],
                "satisfied": False,
                "evidence": [],
            }
        ]
        payload["subgoals"] = [
            {
                "subgoal_id": "sg_refresh_until_title",
                "objective": "刷新当前页面，直到看到标题为长文本输入验收且有空白输入框",
                "status": "active",
                "depends_on": [],
                "constraints": [],
                "completion_conditions": ["长文本输入验收标题和空白输入框可见"],
                "completion_evidence": [],
                "effect_ids": [],
                "execution_class": "navigate",
            }
        ]
        payload["active_subgoal_id"] = "sg_refresh_until_title"

        authority = compile_formal_semantic_authority(
            _graph_from_payload(
                payload,
                task_id="input-validation-title-task",
                device_id="device-local-01",
                revision=1,
                raw_user_goal=payload["goal"]["objective"],
            )
        )
        constraints = {
            item.constraint_id: item for item in authority.semantic_ir.constraints
        }
        action_values = {
            constraints[ref].value
            for ref in authority.semantic_ir.subgoals[0].constraint_refs
            if constraints[ref].kind == "required_action"
        }

        self.assertNotIn("input_verified_text", action_values)
        self.assertEqual((), authority.semantic_ir.input_fields)

    def test_current_send_failure_projects_to_automatic_typed_effect(self):
        report = compile_runtime_graph_semantics(graph_from_payload())

        self.assertFalse(report.authoritative)
        self.assertFalse(report.execution_allowed)
        self.assertEqual([item.kind for item in report.semantic_ir.effects], ["send_message"])
        self.assertEqual(
            [item.policy for item in report.risk_decisions],
            [AUTOMATIC],
        )
        effect = report.semantic_ir.effects[0]
        role_by_ref = {
            item.entity_id: item.role for item in report.semantic_ir.entities
        }
        self.assertEqual(
            {role_by_ref[item] for item in effect.target_refs},
            {"recipient"},
        )
        self.assertEqual(
            {role_by_ref[item] for item in effect.payload_refs},
            {"input_text"},
        )

    def test_unique_bound_effect_result_is_projected_to_exact_source_text(self):
        payload = current_send_failure_payload()
        raw_goal = (
            "进入文件传输助手，输入精确正文 freshsendproof，只发送一次；"
            "验证新我方消息气泡正文逐字为 freshsendproof 且输入框为空。"
        )
        payload["goal"]["objective"] = raw_goal
        payload["effect_intents"][0]["effect_id"] = "send_message_freshsendproof"
        payload["effect_intents"][0]["expected_results"] = [
            "发送后新消息气泡正文逐字为 freshsendproof"
        ]
        payload["goal"]["entities"] = {
            "recipient": "文件传输助手",
            "input_text": "freshsendproof",
        }
        send = payload["subgoals"][1]
        send["effect_ids"] = ["send_message_freshsendproof"]
        send["completion_conditions"] = ["消息已发送，且发送后输入框为空"]
        payload["subgoals"].append(
            {
                "subgoal_id": "verify_sent_message",
                "objective": "验证新消息气泡正文与输入框状态",
                "status": "pending",
                "depends_on": ["send_message"],
                "constraints": [],
                "completion_conditions": [
                    "新消息气泡正文逐字为 freshsendproof",
                    "输入框为空",
                ],
                "completion_evidence": [],
                "effect_ids": [],
                "execution_class": "observe",
            }
        )

        graph = DeepSeekTaskGraphPlanner(OneResponseProvider(payload)).plan(
            raw_goal,
            device_id="device-local-01",
            task_id="unique-effect-result",
        )

        effect = graph.risk_actions[0]
        self.assertEqual(
            ("消息已发送，且发送后输入框为空",),
            effect.expected_result_texts,
        )
        self.assertEqual("send_message", effect.effect_kind)
        self.assertEqual(("send_message",), effect.subgoal_ids)
        self.assertEqual(("recipient",), effect.target_roles)
        self.assertEqual(("input_text",), effect.payload_roles)
        self.assertEqual("文件传输助手", graph.goal.entities["recipient"])
        self.assertEqual("freshsendproof", graph.goal.entities["input_text"])

        semantic_ir = compile_formal_semantic_authority(graph).semantic_ir
        input_field = semantic_ir.input_fields[0]
        self.assertNotIn(
            "verify_sent_message", input_field.source_subgoal_ids
        )
        verify = next(
            item
            for item in semantic_ir.subgoals
            if item.subgoal_id == "verify_sent_message"
        )
        desired = {
            item.state_id: item for item in semantic_ir.desired_states
        }
        self.assertFalse(
            any(
                desired[state_id].predicate == "input.value_equals"
                for state_id in verify.desired_state_refs
            )
        )

    def test_effect_result_literal_is_not_rebound_as_input_across_wording(self):
        payload = current_send_failure_payload()
        payload["goal"]["entities"] = {
            "recipient": "当前目标",
            "input_text": "release candidate",
        }
        payload["subgoals"][1]["objective"] = (
            "提交编辑器内现有正文 release candidate"
        )
        payload["subgoals"][1]["completion_conditions"] = ["提交动作已执行"]
        payload["subgoals"].append(
            {
                "subgoal_id": "verify_result",
                "objective": "核对提交结果与编辑器状态",
                "status": "pending",
                "depends_on": ["send_message"],
                "constraints": ["只读核对"],
                "completion_conditions": [
                    "保存结果预览逐字显示 release candidate",
                    "编辑器为空",
                ],
                "completion_evidence": [],
                "effect_ids": [],
                "execution_class": "observe",
            }
        )
        graph = _graph_from_payload(
            payload,
            task_id="generic-effect-result-literal",
            device_id="device-local-01",
            revision=1,
            raw_user_goal=payload["goal"]["objective"],
        )

        semantic_ir = compile_formal_semantic_authority(graph).semantic_ir
        self.assertNotIn(
            "verify_result", semantic_ir.input_fields[0].source_subgoal_ids
        )
        desired = {
            item.state_id: item for item in semantic_ir.desired_states
        }
        verify = next(
            item
            for item in semantic_ir.subgoals
            if item.subgoal_id == "verify_result"
        )
        self.assertEqual(
            {"observation.matches_description"},
            {desired[state_id].predicate for state_id in verify.desired_state_refs},
        )

    def test_unique_bound_result_projection_is_effect_kind_and_wording_agnostic(self):
        payload = current_send_failure_payload()
        payload["effect_intents"][0].update(
            {
                "kind": "data_mutation",
                "expected_results": ["提交后可看到新的保存结果"],
            }
        )
        payload["subgoals"][1]["completion_conditions"] = [
            "指定记录已保存且编辑框为空"
        ]

        graph = DeepSeekTaskGraphPlanner(OneResponseProvider(payload)).plan(
            RAW_GOAL,
            device_id="device-local-01",
            task_id="unique-data-mutation-result",
        )

        effect = graph.risk_actions[0]
        self.assertEqual("data_mutation", effect.effect_kind)
        self.assertEqual(
            ("指定记录已保存且编辑框为空",),
            effect.expected_result_texts,
        )

    def test_effect_result_projection_keeps_ambiguous_or_invalid_graphs_closed(self):
        def conditions(values):
            return lambda payload: payload["subgoals"][1].update(
                {"completion_conditions": values}
            )

        def effect(**values):
            return lambda payload: payload["effect_intents"][0].update(values)

        mutations = (
            conditions([]),
            conditions(["消息已发送", "输入框为空"]),
            conditions(["不得发送消息"]),
            conditions(["消息尚未发送"]),
            effect(source_subgoal_ids=["open_wechat", "send_message"]),
            effect(source_subgoal_ids=["missing_subgoal"]),
            effect(target_entity_roles=["missing_recipient"]),
            effect(payload_entity_roles=["missing_payload"]),
            lambda payload: payload["goal"]["entities"].update(
                {"recipient": " 文件传输助手"}
            ),
            lambda payload: payload["goal"]["entities"].update(
                {"input_text": "bad\rpayload"}
            ),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(case=index):
                payload = current_send_failure_payload()
                payload["effect_intents"][0]["expected_results"] = [
                    "发送结果可见"
                ]
                mutate(payload)
                planner = DeepSeekTaskGraphPlanner(OneResponseProvider(payload))

                with self.assertRaises(TaskGraphError):
                    planner.plan(
                        RAW_GOAL,
                        device_id="device-local-01",
                        task_id=f"closed-effect-result-{index}",
                    )

    def test_constraint_wording_never_changes_shadow_risk(self):
        variants = (
            "确认发送对象为文件传输助手",
            "聊天对象必须为文件传输助手",
            "仅向文件传输助手发送指定文字",
        )
        reports = [
            compile_runtime_graph_semantics(
                graph_from_payload(current_send_failure_payload(value))
            )
            for value in variants
        ]

        self.assertEqual(
            {report.risk_decisions[0].policy for report in reports},
            {AUTOMATIC},
        )
        self.assertEqual(
            {report.semantic_ir.semantic_digest for report in reports},
            {reports[0].semantic_ir.semantic_digest},
        )

    def test_constraint_location_never_changes_shadow_risk(self):
        in_subgoal = current_send_failure_payload()
        at_graph = current_send_failure_payload()
        moved = at_graph["subgoals"][1]["constraints"].pop()
        at_graph["constraints"].append(moved)

        reports = (
            compile_runtime_graph_semantics(graph_from_payload(in_subgoal)),
            compile_runtime_graph_semantics(graph_from_payload(at_graph)),
        )
        self.assertEqual(
            [report.risk_decisions[0].policy for report in reports],
            [AUTOMATIC, AUTOMATIC],
        )
        self.assertEqual(
            reports[0].semantic_ir.semantic_digest,
            reports[1].semantic_ir.semantic_digest,
        )

    def test_default_policy_matches_frozen_fixture_cases(self):
        fixture_path = (
            Path(__file__).parent
            / "fixtures"
            / "semantic_ir"
            / "role_aware_risk_cases.json"
        )
        cases = json.loads(fixture_path.read_text(encoding="utf-8"))["cases"]
        policy = LocalRiskPolicyConfig()

        for index, case in enumerate(cases, start=1):
            with self.subTest(case_id=case["case_id"]):
                effect = EffectIntent(
                    effect_id=f"effect_case_{index}",
                    kind=case["effect_kind"],
                    expected_result_texts=(case.get("expected_result") or "结果可验证",),
                )
                self.assertEqual(
                    policy.decide(effect).policy,
                    case["expected_policy"],
                )

    def test_default_policy_requires_confirmation_only_for_login_and_payment(self):
        policy = LocalRiskPolicyConfig()
        confirmation_kinds = {"authentication", "financial_transaction"}
        automatic_kinds = {
            "send_message",
            "publish_content",
            "relationship_change",
            "membership_change",
            "sensitive_permission_change",
            "irreversible_account_deletion",
            "irreversible_data_deletion",
            "data_mutation",
            "generic_effect",
        }

        for index, kind in enumerate(sorted(confirmation_kinds), start=1):
            with self.subTest(kind=kind):
                decision = policy.decide(
                    EffectIntent(effect_id=f"effect_confirm_{index}", kind=kind)
                )
                self.assertEqual(CONFIRMATION_REQUIRED, decision.policy)
        for index, kind in enumerate(sorted(automatic_kinds), start=1):
            with self.subTest(kind=kind):
                decision = policy.decide(
                    EffectIntent(effect_id=f"effect_auto_{index}", kind=kind)
                )
                self.assertEqual(AUTOMATIC, decision.policy)

    def test_policy_override_is_typed_and_versioned(self):
        policy = LocalRiskPolicyConfig(
            policy_id="team_policy",
            version=3,
            overrides=(("send_message", CONFIRMATION_REQUIRED),),
        )
        decision = policy.decide(
            EffectIntent(effect_id="effect_send", kind="send_message")
        )
        self.assertEqual(decision.policy, CONFIRMATION_REQUIRED)
        self.assertEqual(decision.policy_id, "team_policy")
        self.assertEqual(decision.policy_version, 3)

    def test_frozen_default_policy_file_is_strict_and_loadable(self):
        policy = load_local_risk_policy(
            Path(__file__).parent / "config" / "local_risk_policy.v1.json"
        )
        self.assertEqual(policy, LocalRiskPolicyConfig())

    def test_policy_file_rejects_extra_fields(self):
        payload = LocalRiskPolicyConfig().to_dict()
        payload["silent_allow_all"] = True
        with self.assertRaisesRegex(TaskSemanticIRError, "字段不匹配"):
            local_risk_policy_from_dict(payload)

    def test_duplicate_policy_override_is_rejected(self):
        policy = LocalRiskPolicyConfig(
            overrides=(
                ("send_message", AUTOMATIC),
                ("send_message", CONFIRMATION_REQUIRED),
            )
        )
        with self.assertRaisesRegex(TaskSemanticIRError, "override 重复"):
            policy.validate()

    def test_shadow_report_can_never_grant_execution(self):
        report = compile_runtime_graph_semantics(graph_from_payload())

        with self.assertRaisesRegex(TaskSemanticIRError, "不得携带执行权限"):
            replace(report, authoritative=True).validate()
        with self.assertRaisesRegex(TaskSemanticIRError, "不得携带执行权限"):
            replace(report, execution_allowed=True).validate()

    def test_literal_source_span_must_bind_exact_value(self):
        entity = SemanticEntity(
            entity_id="entity_recipient",
            entity_type="party",
            role="recipient",
            value="另一个人",
            source_span=SourceSpan(start=5, end=12),
            authority="user_literal",
        )
        with self.assertRaisesRegex(TaskSemanticIRError, "未逐字绑定"):
            entity.validate(RAW_GOAL)

    def test_unknown_entity_reference_is_rejected(self):
        semantic_ir = TaskSemanticIR(
            task_id="9abc",
            device_id="device-local-01",
            revision=1,
            raw_goal="更新目标",
            surfaces=(),
            entities=(),
            effects=(
                EffectIntent(
                    effect_id="effect_update",
                    kind="data_mutation",
                    target_refs=("entity_missing",),
                ),
            ),
        )
        with self.assertRaisesRegex(TaskSemanticIRError, "引用未知实体"):
            semantic_ir.validate()

    def test_binding_must_reference_effect_role(self):
        entity = SemanticEntity(
            entity_id="entity_target",
            entity_type="opaque",
            role="target",
            value="卡片",
        )
        semantic_ir = TaskSemanticIR(
            task_id="9abc",
            device_id="device-local-01",
            revision=1,
            raw_goal="更新卡片",
            surfaces=(),
            entities=(entity,),
            effects=(EffectIntent(effect_id="effect_update", kind="data_mutation"),),
            critical_bindings=(
                CriticalBinding(
                    binding_id="binding_target",
                    effect_id="effect_update",
                    binding_kind="effect_target_equals",
                    entity_ref="entity_target",
                ),
            ),
        )
        with self.assertRaisesRegex(TaskSemanticIRError, "未绑定 effect"):
            semantic_ir.validate()

    def test_formal_authority_removes_blanket_send_confirmation(self):
        provider = OneResponseProvider(current_send_failure_payload())
        planner = DeepSeekTaskGraphPlanner(
            provider,
        )

        graph = planner.plan(RAW_GOAL, device_id="device-local-01", task_id="9abc")

        self.assertEqual(provider.calls, 1)
        self.assertEqual(graph.status, "ready")
        self.assertFalse(graph.risk_actions[0].confirmation_required)
        self.assertIsInstance(
            planner.last_semantic_authority,
            SemanticRiskAuthorityReport,
        )
        self.assertFalse(hasattr(planner, "last_semantic_shadow"))

    def test_unknown_effect_is_rejected_by_new_transport(self):
        payload = current_send_failure_payload()
        payload["effect_intents"][0]["kind"] = "unknown_external_effect"
        payload["subgoals"][1]["objective"] = "处理当前对象"
        payload["subgoals"][1]["completion_conditions"] = ["处理结果可见"]
        planner = DeepSeekTaskGraphPlanner(
            OneResponseProvider(payload),
        )
        with self.assertRaises(TaskGraphError) as caught:
            planner.plan(RAW_GOAL, device_id="device-local-01", task_id="9abc")

        with tempfile.TemporaryDirectory() as temp:
            paths = persist_deepseek_failure_diagnostic(
                planner,
                evidence_dir=Path(temp),
                prefix="current_send_failure",
                failed_stage="initial_task_graph",
                error=caught.exception,
            )
            artifact = json.loads(Path(paths[0]).read_text(encoding="utf-8"))

        self.assertNotIn("typed_effect_authority", artifact)
        self.assertIn("正式效果类型无效", artifact["error_message"])

    def test_retired_shadow_compiler_is_absent_from_planner(self):
        provider = OneResponseProvider(current_send_failure_payload())
        planner = DeepSeekTaskGraphPlanner(
            provider,
        )
        graph = planner.plan(
            RAW_GOAL,
            device_id="device-local-01",
            task_id="9abc",
        )
        self.assertEqual(graph.status, "ready")
        self.assertFalse(hasattr(planner, "last_semantic_shadow"))

    def test_formal_cutover_diff_is_complete_and_bound_to_graph(self):
        graph = graph_from_payload()
        authority = compile_formal_semantic_authority(graph)

        self.assertEqual(len(authority.policy_traces), 1)
        trace = authority.policy_traces[0]
        self.assertTrue(trace.allowed)
        self.assertEqual(trace.formal_policy, AUTOMATIC)
        projected = apply_formal_semantic_risk_policy(graph, authority)
        self.assertFalse(projected.risk_actions[0].confirmation_required)

    def test_effect_preview_covers_target_payload_policy_and_digest(self):
        authority = compile_formal_semantic_authority(graph_from_payload())
        self.assertEqual(1, len(authority.effect_previews))
        preview = authority.effect_previews[0]
        self.assertEqual("send_message", preview.effect_kind)
        self.assertEqual(["recipient"], [item.role for item in preview.targets])
        self.assertEqual(["input_text"], [item.role for item in preview.payloads])
        self.assertEqual(AUTOMATIC, preview.policy)
        self.assertRegex(preview.preview_digest, r"^[0-9a-f]{64}$")
        changed = replace(
            preview,
            payloads=(replace(preview.payloads[0], value="另一段文字"),),
        )
        self.assertNotEqual(preview.preview_digest, changed.preview_digest)

    def test_effect_result_with_payload_text_is_not_misclassified_as_input_value(self):
        authority = compile_formal_semantic_authority(graph_from_payload())
        effect_states = tuple(
            item
            for item in authority.semantic_ir.desired_states
            if item.source_subgoal_id == "send_message"
        )
        self.assertEqual(1, len(effect_states))
        self.assertEqual("effect.result_visible", effect_states[0].predicate)
        requirement = next(
            item
            for item in authority.semantic_ir.evidence_requirements
            if item.desired_state_ref == effect_states[0].state_id
        )
        self.assertEqual(
            ("visual_claim", "effect_receipt"),
            requirement.allowed_sources,
        )

        payload = current_send_failure_payload()
        next(
            item for item in payload["subgoals"] if item["subgoal_id"] == "send_message"
        )["completion_conditions"] = ["发送动作已执行"]
        payload["effect_intents"][0]["expected_results"] = ["发送动作已执行"]
        receipt_only = compile_formal_semantic_authority(
            graph_from_payload(payload)
        )
        applied = next(
            item
            for item in receipt_only.semantic_ir.desired_states
            if item.source_subgoal_id == "send_message"
        )
        self.assertEqual("effect.applied", applied.predicate)
        applied_requirement = next(
            item
            for item in receipt_only.semantic_ir.evidence_requirements
            if item.desired_state_ref == applied.state_id
        )
        self.assertEqual(("effect_receipt",), applied_requirement.allowed_sources)

    def test_multiple_recipients_and_input_fields_compile_to_typed_refs(self):
        payload = current_send_failure_payload()
        payload["goal"]["entities"] = {
            "recipients": ["张三", "李四"],
            "input_fields": [
                {"field_id": "subject", "field_label": "主题", "text": "主题"},
                {
                    "field_id": "body",
                    "field_label": "正文",
                    "text": "第一行\n第二行",
                },
            ],
        }
        open_subgoal = payload["subgoals"][0]
        send_subgoal = payload["subgoals"][1]
        send_subgoal.update(
            {
                "objective": "向张三和李四发送指定内容",
                "depends_on": ["fill_body"],
                "completion_conditions": ["指定内容已发送给张三和李四"],
            }
        )
        payload["subgoals"] = [
            open_subgoal,
            {
                "subgoal_id": "fill_subject",
                "objective": "在主题字段输入主题",
                "status": "pending",
                "depends_on": ["open_wechat"],
                "constraints": ["不得发送或提交"],
                "completion_conditions": ["主题字段逐字为主题"],
                "completion_evidence": [],
                "effect_ids": [],
                "execution_class": "navigate",
            },
            {
                "subgoal_id": "fill_body",
                "objective": "在正文字段输入第一行\n第二行",
                "status": "pending",
                "depends_on": ["fill_subject"],
                "constraints": ["不得发送或提交"],
                "completion_conditions": ["正文字段逐字为第一行\n第二行"],
                "completion_evidence": [],
                "effect_ids": [],
                "execution_class": "navigate",
            },
            send_subgoal,
        ]
        payload["effect_intents"][0]["expected_results"] = [
            "指定内容已发送给张三和李四"
        ]
        graph = _graph_from_payload(
            payload,
            task_id="multi-field-task",
            device_id="device-local-01",
            revision=1,
            raw_user_goal="给张三和李四填写主题和两行正文后发送",
        )
        authority = compile_formal_semantic_authority(graph)
        roles = [item.role for item in authority.semantic_ir.entities]
        self.assertEqual(2, roles.count("recipient"))
        self.assertEqual(2, roles.count("input_text"))
        self.assertEqual(
            {"subject", "body"},
            {item.field_id for item in authority.semantic_ir.input_fields},
        )
        self.assertEqual(
            {"主题", "正文"},
            {item.field_label for item in authority.semantic_ir.input_fields},
        )
        fields = {
            item.field_id: item for item in authority.semantic_ir.input_fields
        }
        self.assertEqual(("fill_subject",), fields["subject"].source_subgoal_ids)
        self.assertEqual(("fill_body",), fields["body"].source_subgoal_ids)
        self.assertTrue(
            any(item.multiline for item in authority.semantic_ir.input_fields)
        )
        required_actions = {
            item.value
            for item in authority.semantic_ir.constraints
            if item.kind == "required_action"
        }
        self.assertIn("press_enter", required_actions)

        changed = replace(graph, revision=2)
        with self.assertRaisesRegex(TaskSemanticIRError, "未绑定当前任务图"):
            apply_formal_semantic_risk_policy(changed, authority)

    def test_multifield_input_ownership_and_final_exact_states_are_typed(self):
        for subject_label, body_label, subject_text, body_text in (
            ("主题", "正文", "first", "second"),
            ("标题", "备注", "alpha", "beta"),
        ):
            with self.subTest(subject_label=subject_label, body_label=body_label):
                payload = {
                    "status": "ready",
                    "goal": {
                        "objective": (
                            f"在{subject_label}字段输入 {subject_text}，"
                            f"再在{body_label}字段输入 {body_text}，最后同时核对"
                        ),
                        "target_apps": [
                            {
                                "app_id": "current_foreground",
                                "app_name": "当前前台应用",
                            }
                        ],
                        "entities": {
                            "input_fields": [
                                {
                                    "field_id": "subject",
                                    "field_label": subject_label,
                                    "text": subject_text,
                                },
                                {
                                    "field_id": "body",
                                    "field_label": body_label,
                                    "text": body_text,
                                },
                            ]
                        },
                    },
                    "constraints": ["不得发送或提交"],
                    "completion_conditions": [
                        {
                            "condition_id": "both_fields_exact",
                            "description": (
                                f"{subject_label}字段逐字为 {subject_text} 且"
                                f"{body_label}字段逐字为 {body_text}"
                            ),
                            "evidence_required": ["两个字段当前值同时可见"],
                            "satisfied": False,
                            "evidence": [],
                        }
                    ],
                    "effect_intents": [],
                    "subgoals": [
                        {
                            "subgoal_id": "fill_subject",
                            "objective": (
                                f"在{subject_label}字段输入 {subject_text}"
                            ),
                            "status": "active",
                            "depends_on": [],
                            "constraints": ["不得发送或提交"],
                            "completion_conditions": [
                                f"{subject_label}字段逐字为 {subject_text}"
                            ],
                            "completion_evidence": [],
                            "effect_ids": [],
                            "execution_class": "navigate",
                        },
                        {
                            "subgoal_id": "fill_body",
                            "objective": f"在{body_label}字段输入 {body_text}",
                            "status": "pending",
                            "depends_on": ["fill_subject"],
                            "constraints": ["不得发送或提交"],
                            "completion_conditions": [
                                f"{body_label}字段逐字为 {body_text}"
                            ],
                            "completion_evidence": [],
                            "effect_ids": [],
                            "execution_class": "navigate",
                        },
                        {
                            "subgoal_id": "verify_fields",
                            "objective": "同时核对两个字段当前值",
                            "status": "pending",
                            "depends_on": ["fill_body"],
                            "constraints": ["只读核对"],
                            "completion_conditions": [
                                f"{subject_label}字段逐字为 {subject_text} 且"
                                f"{body_label}字段逐字为 {body_text}"
                            ],
                            "completion_evidence": [],
                            "effect_ids": [],
                            "execution_class": "observe",
                        },
                    ],
                    "active_subgoal_id": "fill_subject",
                    "clarification_questions": [],
                }
                graph = _graph_from_payload(
                    payload,
                    task_id="multifield-exact-task",
                    device_id="device-local-01",
                    revision=1,
                    raw_user_goal=payload["goal"]["objective"],
                )
                semantic_ir = compile_formal_semantic_authority(graph).semantic_ir
                fields = {
                    item.field_id: item for item in semantic_ir.input_fields
                }
                self.assertEqual(
                    {"fill_subject", "verify_fields"},
                    set(fields["subject"].source_subgoal_ids),
                )
                self.assertEqual(
                    {"fill_body", "verify_fields"},
                    set(fields["body"].source_subgoal_ids),
                )

                verify = next(
                    item
                    for item in semantic_ir.subgoals
                    if item.subgoal_id == "verify_fields"
                )
                desired = {
                    item.state_id: item for item in semantic_ir.desired_states
                }
                exact_states = [
                    desired[state_id]
                    for state_id in verify.desired_state_refs
                    if desired[state_id].predicate == "input.value_equals"
                ]
                self.assertEqual(
                    {fields["subject"].payload_ref, fields["body"].payload_ref},
                    {item.subject_ref for item in exact_states},
                )
                self.assertEqual(
                    {subject_text, body_text},
                    {item.value for item in exact_states},
                )

                combined_payload = json.loads(
                    json.dumps(payload, ensure_ascii=False)
                )
                combined_payload["subgoals"] = [
                    {
                        "subgoal_id": "fill_both",
                        "objective": (
                            f"在{subject_label}字段输入 {subject_text}，"
                            f"并在{body_label}字段输入 {body_text}"
                        ),
                        "status": "active",
                        "depends_on": [],
                        "constraints": ["不得发送或提交"],
                        "completion_conditions": [
                            f"{subject_label}字段逐字为 {subject_text} 且"
                            f"{body_label}字段逐字为 {body_text}"
                        ],
                        "completion_evidence": [],
                        "effect_ids": [],
                        "execution_class": "navigate",
                    }
                ]
                combined_payload["active_subgoal_id"] = "fill_both"
                with self.assertRaisesRegex(
                    TaskSemanticIRError,
                    "必须且只能绑定一个 InputFieldIntent",
                ):
                    compile_formal_semantic_authority(
                        _graph_from_payload(
                            combined_payload,
                            task_id="multifield-combined-task",
                            device_id="device-local-01",
                            revision=1,
                            raw_user_goal=combined_payload["goal"]["objective"],
                        )
                    )

    def test_newline_key_wording_compiles_to_press_enter(self):
        for wording in (
            "点击手机键盘右下角换行键",
            "press the new line key",
        ):
            with self.subTest(wording=wording):
                payload = {
                    "status": "ready",
                    "goal": {
                        "objective": "第一行输入 first，换行后第二行输入 second",
                        "target_apps": [
                            {
                                "app_id": "current_foreground",
                                "app_name": "当前前台应用",
                            }
                        ],
                        "entities": {"input_text": "first\nsecond"},
                    },
                    "constraints": ["不得发送或提交"],
                    "completion_conditions": [
                        {
                            "condition_id": "input_complete",
                            "description": "输入框内容为 first 换行 second",
                            "evidence_required": ["输入框显示两行目标文字"],
                            "satisfied": False,
                            "evidence": [],
                        }
                    ],
                    "effect_intents": [],
                    "subgoals": [
                        {
                            "subgoal_id": "input_first_line",
                            "objective": "在输入框中输入 first",
                            "status": "active",
                            "depends_on": [],
                            "constraints": ["不得发送或提交"],
                            "completion_conditions": ["输入框内容为 first"],
                            "completion_evidence": [],
                            "effect_ids": [],
                            "execution_class": "navigate",
                        },
                        {
                            "subgoal_id": "press_enter",
                            "objective": wording,
                            "status": "pending",
                            "depends_on": ["input_first_line"],
                            "constraints": ["不得发送或提交"],
                            "completion_conditions": [
                                "输入框内容为 first 加换行"
                            ],
                            "completion_evidence": [],
                            "effect_ids": [],
                            "execution_class": "navigate",
                        },
                    ],
                    "active_subgoal_id": "input_first_line",
                    "clarification_questions": [],
                }
                graph = _graph_from_payload(
                    payload,
                    task_id="newline-wording-task",
                    device_id="device-local-01",
                    revision=1,
                    raw_user_goal=(
                        "第一行输入 first，点击换行键，第二行输入 second"
                    ),
                )
                semantic_ir = compile_formal_semantic_authority(graph).semantic_ir
                constraints = {
                    item.constraint_id: item for item in semantic_ir.constraints
                }
                newline = next(
                    item
                    for item in semantic_ir.subgoals
                    if item.subgoal_id == "press_enter"
                )
                required_actions = {
                    constraints[ref].value
                    for ref in newline.constraint_refs
                    if constraints[ref].kind == "required_action"
                }
                self.assertIn("press_enter", required_actions)

    def test_unknown_effect_cannot_cross_formal_cutover(self):
        payload = current_send_failure_payload()
        payload["effect_intents"][0]["kind"] = "unknown_external_effect"
        with self.assertRaisesRegex(TaskGraphError, "正式效果类型无效"):
            graph_from_payload(payload)

    def test_new_request_clears_stale_authority_before_json_parse(self):
        planner = DeepSeekTaskGraphPlanner(
            RawResponseProvider("{"),
        )
        planner.last_semantic_authority = compile_formal_semantic_authority(
            graph_from_payload()
        )
        planner.last_semantic_authority_error = "stale"

        with self.assertRaises(TaskGraphError):
            planner.plan("新的目标", device_id="device-local-01", task_id="9abc")

        self.assertIsNone(planner.last_semantic_authority)
        self.assertEqual(planner.last_semantic_authority_error, "")


if __name__ == "__main__":
    unittest.main()
