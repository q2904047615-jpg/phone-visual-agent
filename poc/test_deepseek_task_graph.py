import copy
import json
import unittest

from deepseek_task_graph import (
    ControllerTransitionEvidenceRef,
    DEEPSEEK_TASK_GRAPH_PROTOCOL_VERSION,
    DeepSeekTaskGraphPlanner,
    ObservedState,
    TaskGraphError,
    VerifiedActionTransition,
    _compact_identity_text,
    _infer_external_risk_types,
    _named_visual_identity_anchor,
    _quoted_visual_identity_anchor,
)


class FakeProvider:
    configured = True

    def __init__(
        self,
        *payloads,
        audit_payloads=None,
        audit_raw_responses=None,
        audit_error=None,
    ):
        self.payloads = list(payloads)
        self.audit_payloads = list(audit_payloads or [])
        self.audit_raw_responses = list(audit_raw_responses or [])
        self.audit_error = audit_error
        self.messages = []
        self.last_graph_payload = None

    def chat_json(self, messages, max_tokens=2000):
        self.messages.append(messages)
        prompt = messages[0]["content"]
        if "semantic-risk-audit-v1" in prompt:
            if self.audit_error is not None:
                raise self.audit_error
            if self.audit_raw_responses:
                return self.audit_raw_responses.pop(0)
            payload = (
                self.audit_payloads.pop(0)
                if self.audit_payloads
                else audit_payload_for_graph(self.last_graph_payload)
            )
            return json.dumps(payload, ensure_ascii=False)
        self.last_graph_payload = self.payloads.pop(0)
        return json.dumps(self.last_graph_payload, ensure_ascii=False)


def base_payload():
    return {
        "status": "ready",
        "goal": {
            "objective": "在地图应用中找到图书馆并保存地点",
            "target_apps": [{"app_id": "maps", "app_name": "地图"}],
            "entities": {"place": "图书馆"},
        },
        "constraints": ["不要发起导航"],
        "completion_conditions": [
            {
                "condition_id": "place_saved",
                "description": "目标地点已保存",
                "evidence_required": ["页面显示已收藏状态"],
                "satisfied": False,
                "evidence": [],
            }
        ],
        "risk_actions": [
            {
                "risk_id": "save_place",
                "description": "保存目标地点",
                "external_effect": "更改账号收藏数据",
                "risk_type": "data_mutation",
                "risk_level": "medium",
                "subgoal_ids": ["save_target"],
                "confirmation_required": True,
            }
        ],
        "subgoals": [
            {
                "subgoal_id": "locate_target",
                "objective": "目标地点详情可见",
                "status": "active",
                "depends_on": [],
                "constraints": ["地点名称必须匹配"],
                "completion_conditions": ["页面显示目标地点详情"],
                "completion_evidence": [],
                "risk_action_ids": [],
                "external_impact": "navigation_only",
            },
            {
                "subgoal_id": "save_target",
                "objective": "目标地点进入已保存状态",
                "status": "pending",
                "depends_on": ["locate_target"],
                "constraints": ["执行前等待用户确认"],
                "completion_conditions": ["页面显示已收藏状态"],
                "completion_evidence": [],
                "risk_action_ids": ["save_place"],
                "external_impact": "external_state",
            },
        ],
        "active_subgoal_id": "locate_target",
        "clarification_questions": [],
    }


def observation():
    return ObservedState(
        scene_id="scene-2",
        summary="地图已经显示目标地点详情",
        visible_evidence=("页面标题为城市图书馆", "页面显示已收藏状态"),
        last_action_outcome="matched",
    )


def mismatch_observation(graph):
    return ObservedState(
        scene_id="scene-2",
        summary="地图没有显示预期结果",
        visible_evidence=("页面仍显示原内容",),
        last_action_outcome="mismatched",
        blocked_reasons=("预期语义变化未出现",),
        verified_action_transition=VerifiedActionTransition(
            receipt_id="receipt-test-mismatch",
            session_id="session-test",
            task_id=graph.task_id,
            device_id=graph.device_id,
            prior_revision=graph.revision,
            subgoal_id=graph.active_subgoal_id,
            decision_node_id="decision-test",
            action_digest="a" * 64,
            rebound_action_digest="b" * 64,
            resolved_action_digest="c" * 64,
            action_kind="tap_semantic",
            before_observation_id="scene-1",
            before_fingerprint="before-fingerprint",
            after_observation_id="scene-2",
            after_fingerprint="after-fingerprint",
            physical_actions=1,
            outcome="mismatched",
            errors=("预期语义变化未出现",),
        ),
    )


def matched_controller_observation(graph, *, receipt_id="receipt-matched"):
    current = graph.active_subgoal()
    receipt = VerifiedActionTransition(
        receipt_id=receipt_id,
        session_id="session-test",
        task_id=graph.task_id,
        device_id=graph.device_id,
        prior_revision=graph.revision,
        subgoal_id=current.subgoal_id,
        decision_node_id="decision-test",
        action_digest="a" * 64,
        rebound_action_digest="b" * 64,
        resolved_action_digest="c" * 64,
        action_kind="tap_semantic",
        before_observation_id=f"scene-{graph.revision}",
        before_fingerprint=f"before-{graph.revision}",
        after_observation_id=f"scene-{graph.revision + 1}",
        after_fingerprint=f"after-{graph.revision}",
        physical_actions=1,
        outcome="matched",
        controller_completion_evidence=("控制器确认一次性导航完成",),
    )
    ref = ControllerTransitionEvidenceRef(
        ref_id=f"controller_transition:{receipt_id}:1",
        receipt_id=receipt_id,
        subgoal_id=current.subgoal_id,
        text="控制器确认一次性导航完成",
    )
    return ObservedState(
        scene_id=receipt.after_observation_id,
        summary="动作后的可信画面",
        visible_evidence=("动作后的页面可见",),
        last_action_outcome="matched",
        verified_action_transition=receipt,
        controller_transition_evidence_refs=(ref,),
    )


def single_subgoal_payload(objective, *, external_impact):
    payload = base_payload()
    payload["goal"]["objective"] = objective
    payload["completion_conditions"] = [
        {
            "condition_id": "result_visible",
            "description": "目标结果可见",
            "evidence_required": ["页面显示目标结果"],
            "satisfied": False,
            "evidence": [],
        }
    ]
    payload["risk_actions"] = []
    payload["subgoals"] = [
        {
            "subgoal_id": "target_state",
            "objective": objective,
            "status": "active",
            "depends_on": [],
            "constraints": [],
            "completion_conditions": ["目标结果可见"],
            "completion_evidence": [],
            "risk_action_ids": [],
            "external_impact": external_impact,
        }
    ]
    payload["active_subgoal_id"] = "target_state"
    payload["status"] = "ready"
    return payload


def active_external_payload():
    payload = base_payload()
    payload["status"] = "awaiting_confirmation"
    payload["subgoals"][0]["status"] = "skipped"
    payload["subgoals"][1]["status"] = "active"
    payload["subgoals"][1]["depends_on"] = []
    payload["active_subgoal_id"] = "save_target"
    return payload


def purely_forbidden_draft_payload():
    return {
        "status": "ready",
        "goal": {
            "objective": (
                "打开指定聊天页面，在唯一消息输入框中保留未发送的英文 codex 草稿，"
                "并停在发送按钮之前；不得发送、删除或选择其他联系人。"
            ),
            "target_apps": [{"app_id": "chat_app", "app_name": "聊天应用"}],
            "entities": {
                "target_ui_label": "指定聊天页面",
                "input_text": "codex",
            },
        },
        "constraints": [
            "不得发送、删除或选择其他联系人。",
            "不得点击发送按钮。",
        ],
        "completion_conditions": [
            {
                "condition_id": "chat_visible",
                "description": "指定聊天页面在前台可见",
                "evidence_required": ["指定聊天页面可见"],
                "satisfied": False,
                "evidence": [],
            },
            {
                "condition_id": "draft_visible",
                "description": "底部唯一消息输入框中包含未发送的英文 codex 草稿",
                "evidence_required": ["输入框逐字显示 codex"],
                "satisfied": False,
                "evidence": [],
            },
            {
                "condition_id": "before_send",
                "description": "发送按钮可见且 codex 仍可核对",
                "evidence_required": ["发送按钮可见", "codex 可见"],
                "satisfied": False,
                "evidence": [],
            },
        ],
        "risk_actions": [
            {
                "risk_id": "send_message",
                "description": "发送消息给指定联系人",
                "external_effect": "向指定联系人发送消息并产生通信记录",
                "risk_type": "message_or_communication",
                "risk_level": "high",
                "subgoal_ids": ["open_chat", "enter_draft", "verify_before_send"],
                "confirmation_required": True,
            },
            {
                "risk_id": "delete_content",
                "description": "删除输入框中的内容",
                "external_effect": "删除未发送的草稿内容",
                "risk_type": "data_deletion",
                "risk_level": "medium",
                "subgoal_ids": ["enter_draft"],
                "confirmation_required": True,
            },
            {
                "risk_id": "select_other_contact",
                "description": "选择其他联系人",
                "external_effect": "改变当前聊天对象",
                "risk_type": "account_relationship_change",
                "risk_level": "medium",
                "subgoal_ids": ["open_chat"],
                "confirmation_required": True,
            },
        ],
        "subgoals": [
            {
                "subgoal_id": "open_chat",
                "objective": "指定聊天页面在前台可见",
                "status": "active",
                "depends_on": [],
                "constraints": ["不得选择其他联系人"],
                "completion_conditions": ["指定聊天页面可见"],
                "completion_evidence": [],
                "risk_action_ids": ["select_other_contact"],
                "external_impact": "external_state",
            },
            {
                "subgoal_id": "enter_draft",
                "objective": "底部唯一消息输入框中包含未发送的英文 codex 草稿",
                "status": "pending",
                "depends_on": ["open_chat"],
                "constraints": ["不得发送或删除"],
                "completion_conditions": ["输入框逐字显示 codex"],
                "completion_evidence": [],
                "risk_action_ids": ["send_message", "delete_content"],
                "external_impact": "external_state",
            },
            {
                "subgoal_id": "verify_before_send",
                "objective": "停在发送按钮之前且 codex 仍可核对",
                "status": "pending",
                "depends_on": ["enter_draft"],
                "constraints": ["不得点击发送按钮"],
                "completion_conditions": ["发送按钮可见", "codex 可见"],
                "completion_evidence": [],
                "risk_action_ids": ["send_message"],
                "external_impact": "read_only",
            },
        ],
        "active_subgoal_id": "open_chat",
        "clarification_questions": [],
    }


def local_input_preparation_payload():
    """Reproduce the final stage-three risk graph without App-specific logic."""

    return {
        "status": "ready",
        "goal": {
            "objective": (
                "进入指定页面，在底部唯一消息输入框中保留未发送的英文 codex 草稿，"
                "并停在发送按钮之前；不得发送、提交、删除、转发、发布或产生外部影响。"
            ),
            "target_apps": [
                {"app_id": "current_foreground", "app_name": "当前前台应用"}
            ],
            "entities": {
                "target_ui_label": "指定页面",
                "input_text": "codex",
            },
        },
        "constraints": [
            "不得发送、提交、删除、转发或发布该草稿",
            "不得产生任何账号及外部影响",
        ],
        "completion_conditions": [
            {
                "condition_id": "final_state",
                "description": (
                    "指定页面在前台可见，底部唯一消息输入框内容为 codex，"
                    "且未发送，发送按钮可见但未触发"
                ),
                "evidence_required": [
                    "指定页面可见",
                    "底部唯一消息输入框内容为 codex",
                    "未发送草稿",
                    "发送按钮可见且未触发",
                ],
                "satisfied": False,
                "evidence": [],
            }
        ],
        "risk_actions": [
            {
                "risk_id": "send_message",
                "description": "发送消息",
                "external_effect": "向外部联系人发送消息并产生通信记录",
                "risk_type": "message_or_communication",
                "risk_level": "high",
                "subgoal_ids": ["navigate", "prepare_input", "type_text"],
                "confirmation_required": True,
            }
        ],
        "subgoals": [
            {
                "subgoal_id": "navigate",
                "objective": "打开指定页面",
                "status": "active",
                "depends_on": [],
                "constraints": [],
                "completion_conditions": ["指定页面在前台可见"],
                "completion_evidence": [],
                "risk_action_ids": [],
                "external_impact": "navigation_only",
            },
            {
                "subgoal_id": "prepare_input",
                "objective": "底部唯一消息输入框可见且可编辑",
                "status": "pending",
                "depends_on": ["navigate"],
                "constraints": ["不得发送、提交、删除、转发或发布"],
                "completion_conditions": ["底部唯一消息输入框可见且可编辑"],
                "completion_evidence": [],
                "risk_action_ids": [],
                "external_impact": "navigation_only",
            },
            {
                "subgoal_id": "type_text",
                "objective": "当前输入框内容为 codex",
                "status": "pending",
                "depends_on": ["prepare_input"],
                "constraints": ["不得发送、提交、删除、转发或发布"],
                "completion_conditions": ["当前输入框内容为 codex 且未发送"],
                "completion_evidence": [],
                "risk_action_ids": ["send_message"],
                "external_impact": "navigation_only",
            },
        ],
        "active_subgoal_id": "navigate",
        "clarification_questions": [],
    }


def audit_sources_for_graph(payload):
    sources = [
        ("raw_goal", "raw_goal", None),
        ("goal.objective", "goal_objective", None),
    ]
    sources.extend(
        (f"constraints.{index}", "goal_constraint", None)
        for index, _ in enumerate(payload["constraints"])
    )
    for condition in payload["completion_conditions"]:
        condition_id = condition["condition_id"]
        sources.append(
            (
                f"completion_conditions.{condition_id}.description",
                "goal_completion_condition",
                None,
            )
        )
        sources.extend(
            (
                f"completion_conditions.{condition_id}.evidence_required.{index}",
                "goal_completion_condition",
                None,
            )
            for index, _ in enumerate(condition["evidence_required"])
        )
    for subgoal in payload["subgoals"]:
        subgoal_id = subgoal["subgoal_id"]
        sources.append(
            (
                f"subgoals.{subgoal_id}.objective",
                "subgoal_objective",
                subgoal_id,
            )
        )
        sources.extend(
            (
                f"subgoals.{subgoal_id}.constraints.{index}",
                "subgoal_constraint",
                subgoal_id,
            )
            for index, _ in enumerate(subgoal["constraints"])
        )
        sources.extend(
            (
                f"subgoals.{subgoal_id}.completion_conditions.{index}",
                "subgoal_completion_condition",
                subgoal_id,
            )
            for index, _ in enumerate(subgoal["completion_conditions"])
        )
    return sources


def audit_payload_for_graph(payload, *, overrides=None, confidence=0.99):
    overrides = overrides or {}
    subgoals = {item["subgoal_id"]: item for item in payload["subgoals"]}
    risks = {item["risk_id"]: item for item in payload["risk_actions"]}
    graph_impacts = {item["external_impact"] for item in payload["subgoals"]}
    if "external_state" in graph_impacts:
        graph_impact = "external_state"
    elif "unknown" in graph_impacts:
        graph_impact = "unknown"
    elif "navigation_only" in graph_impacts:
        graph_impact = "navigation_only"
    else:
        graph_impact = "read_only"
    graph_risk_types = sorted({item["risk_type"] for item in payload["risk_actions"]})
    assessments = []
    for source_id, source_kind, subgoal_id in audit_sources_for_graph(payload):
        if subgoal_id is None:
            impact = graph_impact
            risk_types = graph_risk_types
        else:
            subgoal = subgoals[subgoal_id]
            impact = subgoal["external_impact"]
            risk_types = sorted(
                {
                    risks[risk_id]["risk_type"]
                    for risk_id in subgoal["risk_action_ids"]
                }
            )
        if impact == "unknown" and not risk_types:
            risk_types = ["unknown_external_effect"]
        if impact in {"read_only", "navigation_only"}:
            risk_types = []
        override = overrides.get(source_id)
        if override:
            impact = override.get("external_impact", impact)
            risk_types = override.get("risk_types", risk_types)
            item_confidence = override.get("confidence", confidence)
        else:
            item_confidence = confidence
        assessments.append(
            {
                "source_id": source_id,
                "source_kind": source_kind,
                "subgoal_id": subgoal_id,
                "external_impact": impact,
                "risk_types": risk_types,
                "reason": "独立语义审计结论",
                "confidence": item_confidence,
            }
        )
    return {"assessments": assessments}


class DeepSeekTaskGraphTests(unittest.TestCase):
    def test_verified_action_transition_is_separate_and_requires_one_action(self):
        receipt = VerifiedActionTransition(
            receipt_id="receipt-test",
            session_id="session-test",
            task_id="task-test",
            device_id="phone-1",
            prior_revision=1,
            subgoal_id="locate_target",
            decision_node_id="decision-test",
            action_digest="a" * 64,
            rebound_action_digest="b" * 64,
            resolved_action_digest="c" * 64,
            action_kind="tap_semantic",
            before_observation_id="scene-1",
            before_fingerprint="before",
            after_observation_id="scene-2",
            after_fingerprint="after",
            physical_actions=1,
            outcome="matched",
            controller_completion_evidence=("本地控制器证明",),
        )
        observed = ObservedState(
            scene_id="scene-2",
            summary="当前页面",
            visible_evidence=("页面可见事实",),
            last_action_outcome="matched",
            verified_action_transition=receipt,
            controller_transition_evidence_refs=(
                ControllerTransitionEvidenceRef(
                    ref_id="controller_transition:receipt-test:1",
                    receipt_id="receipt-test",
                    subgoal_id="locate_target",
                    text="本地控制器证明",
                ),
            ),
        )

        payload = observed.to_dict()
        self.assertEqual(["页面可见事实"], payload["visible_evidence"])
        self.assertEqual(
            ["本地控制器证明"],
            payload["verified_action_transition"][
                "controller_completion_evidence"
            ],
        )
        self.assertEqual(
            "controller_transition:receipt-test:1",
            payload["controller_transition_evidence_refs"][0]["ref_id"],
        )
        with self.assertRaisesRegex(TaskGraphError, "必须且只能证明 1 次"):
            VerifiedActionTransition(
                **{**receipt.__dict__, "physical_actions": 0}
            ).validate()

    def test_action_result_matched_requires_scope_bound_receipt(self):
        graph = DeepSeekTaskGraphPlanner(FakeProvider(base_payload())).plan(
            "目标", device_id="phone-1"
        )
        with self.assertRaisesRegex(TaskGraphError, "缺少本地"):
            DeepSeekTaskGraphPlanner(FakeProvider(base_payload())).replan(
                graph,
                observation(),
                trigger="action_result_matched",
                reason="动作验证匹配",
            )

    def test_protocol_remains_v3(self):
        self.assertEqual(
            DEEPSEEK_TASK_GRAPH_PROTOCOL_VERSION,
            "2026-08-11-deepseek-task-graph-v3",
        )

    def test_builds_generic_graph_with_device_isolation(self):
        provider = FakeProvider(base_payload())
        graph = DeepSeekTaskGraphPlanner(provider).plan(
            "请在地图中找到图书馆并保存，但不要开始导航",
            device_id="phone-01",
            task_id="task-map-001",
        )
        self.assertEqual(graph.task_id, "task-map-001")
        self.assertEqual(graph.device_id, "phone-01")
        self.assertEqual(graph.revision, 1)
        self.assertEqual(graph.active_subgoal().subgoal_id, "locate_target")
        self.assertEqual(graph.goal.target_apps[0].app_id, "maps")
        prompt = provider.messages[0][0]["content"]
        self.assertIn("任意 App", prompt)
        self.assertIn("不能输出坐标", prompt)
        self.assertIn("具名入口在列表中可见", prompt)
        self.assertIn("入口可见绝不能证明", prompt)

    def test_qwen_context_exposes_only_current_subgoal_and_linked_risks(self):
        graph = DeepSeekTaskGraphPlanner(FakeProvider(base_payload())).plan(
            "目标",
            device_id="phone-01",
            task_id="task-01",
        )
        context = graph.to_qwen_context()
        self.assertEqual(context["task_id"], "task-01")
        self.assertEqual(context["device_id"], "phone-01")
        self.assertEqual(context["current_subgoal"]["subgoal_id"], "locate_target")
        self.assertEqual(context["current_external_impact"], "navigation_only")
        self.assertEqual(context["risk_actions"], [])
        self.assertNotIn("subgoals", context)
        self.assertFalse(context["confirmation_gate"]["required"])
        self.assertFalse(
            context["confirmation_gate"]["external_state_action_allowed"]
        )

    def test_task_id_is_preserved_across_replanning(self):
        initial = base_payload()
        revised = copy.deepcopy(initial)
        graph = DeepSeekTaskGraphPlanner(FakeProvider(initial)).plan(
            "目标",
            device_id="phone-01",
            task_id="task-keep",
        )
        result = DeepSeekTaskGraphPlanner(FakeProvider(revised)).replan(
            graph,
            observation(),
            trigger="observation_changed",
            reason="画面内容发生变化",
        )
        self.assertEqual(result.task_id, "task-keep")
        self.assertEqual(result.device_id, "phone-01")

    def test_supports_cross_app_goal_without_app_specific_branches(self):
        payload = base_payload()
        payload["goal"] = {
            "objective": "读取日历中的安排并在笔记中整理摘要",
            "target_apps": [
                {"app_id": "calendar", "app_name": "日历"},
                {"app_id": "notes", "app_name": "笔记"},
            ],
            "entities": {"date": "明天"},
        }
        payload["constraints"] = ["不得修改日历事件"]
        graph = DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
            "看看明天的安排并整理到笔记里，不要改日历",
            device_id="phone-02",
        )
        self.assertEqual([item.app_id for item in graph.goal.target_apps], ["calendar", "notes"])

    def test_cross_app_goal_can_replan_without_changing_identity(self):
        payload = base_payload()
        payload["goal"] = {
            "objective": "读取日历安排并在笔记中整理摘要",
            "target_apps": [
                {"app_id": "calendar", "app_name": "日历"},
                {"app_id": "notes", "app_name": "笔记"},
            ],
            "entities": {"date": "明天"},
        }
        graph = DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
            "整理明天安排",
            device_id="phone-cross",
            task_id="task-cross",
        )
        result = DeepSeekTaskGraphPlanner(FakeProvider(copy.deepcopy(payload))).replan(
            graph,
            observation(),
            trigger="observation_changed",
            reason="页面内容更新",
        )
        self.assertEqual(result.task_id, "task-cross")
        self.assertEqual(result.device_id, "phone-cross")
        self.assertEqual(result.revision, 2)
        self.assertEqual(
            [item.app_id for item in result.goal.target_apps],
            ["calendar", "notes"],
        )

    def test_ambiguous_goal_can_return_blocked_clarification(self):
        payload = base_payload()
        payload["status"] = "blocked"
        payload["goal"]["target_apps"] = []
        payload["subgoals"] = []
        payload["risk_actions"] = []
        payload["active_subgoal_id"] = None
        payload["clarification_questions"] = ["请说明希望使用哪个地图应用。"]
        graph = DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
            "帮我找个地方",
            device_id="phone-1",
        )
        self.assertEqual(graph.status, "blocked")
        self.assertIsNone(graph.active_subgoal())

    def test_initial_plan_cannot_claim_global_condition_satisfied(self):
        payload = base_payload()
        payload["completion_conditions"][0]["satisfied"] = True
        payload["completion_conditions"][0]["evidence"] = ["页面显示已收藏状态"]
        with self.assertRaisesRegex(TaskGraphError, "初始规划没有观察证据"):
            DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan("目标", device_id="phone-1")

    def test_initial_plan_aligns_one_explicit_safe_status_error_locally(self):
        invalid = base_payload()
        invalid["subgoals"][0]["status"] = "pending"
        repaired = base_payload()
        provider = FakeProvider(invalid, repaired)
        graph = DeepSeekTaskGraphPlanner(provider).plan(
            "目标",
            device_id="phone-1",
        )
        self.assertEqual(graph.active_subgoal_id, "locate_target")
        self.assertEqual(graph.active_subgoal().status, "active")
        self.assertEqual(len(provider.messages), 2)
        self.assertIn("临时标签页也属于 navigation_only", provider.messages[0][0]["content"])
        self.assertIn("goal.entities.target_ui_label", provider.messages[0][0]["content"])
        self.assertIn("字面标签当成动作指令", provider.messages[0][0]["content"])
        self.assertIn("目标页面不再被遮挡，主要内容可见", provider.messages[0][0]["content"])
        self.assertIn("当前输入框内容为 X", provider.messages[0][0]["content"])
        self.assertIn("本机临时结果区域显示该表达式的答案", provider.messages[0][0]["content"])

    def test_initial_plan_repairs_unique_frontier_without_remote_retry(self):
        first = base_payload()
        first["subgoals"][1]["status"] = "active"
        provider = FakeProvider(first)

        graph = DeepSeekTaskGraphPlanner(provider).plan(
            "打开一个本机临时页面",
            device_id="phone-1",
        )

        self.assertEqual("locate_target", graph.active_subgoal_id)
        graph_prompts = [
            call for call in provider.messages
            if "semantic-risk-audit-v1" not in call[0]["content"]
        ]
        self.assertEqual(1, len(graph_prompts))

    def test_initial_plan_normalizes_self_contradictory_local_navigation_risk(self):
        payload = single_subgoal_payload(
            "当前浏览器显示空白标签页",
            external_impact="navigation_only",
        )
        payload["subgoals"][0]["status"] = "pending"
        payload["active_subgoal_id"] = None
        payload["risk_actions"] = [
            {
                "risk_id": "local_tab_state",
                "description": "临时标签页状态变化",
                "external_effect": "无外部状态影响",
                "risk_type": "unknown_external_effect",
                "risk_level": "low",
                "subgoal_ids": ["target_state"],
                "confirmation_required": True,
            }
        ]
        payload["subgoals"][0]["risk_action_ids"] = ["local_tab_state"]

        graph = DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
            "新建一个空白标签页",
            device_id="phone-1",
        )

        self.assertEqual((), graph.risk_actions)
        self.assertEqual("target_state", graph.active_subgoal_id)
        self.assertEqual("active", graph.active_subgoal().status)

    def test_initial_plan_normalizes_transient_tab_mislabeled_as_data_mutation(self):
        payload = single_subgoal_payload(
            "在当前浏览器打开空白标签页",
            external_impact="external_state",
        )
        payload["subgoals"][0]["status"] = "pending"
        payload["active_subgoal_id"] = None
        payload["risk_actions"] = [
            {
                "risk_id": "tab_mutation",
                "description": "临时标签页集合变化",
                "external_effect": "仅改变浏览器内部标签页，不影响账号数据或外部系统",
                "risk_type": "data_mutation",
                "risk_level": "low",
                "subgoal_ids": ["target_state"],
                "confirmation_required": True,
            }
        ]
        payload["subgoals"][0]["risk_action_ids"] = ["tab_mutation"]

        graph = DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
            "新建一个空白标签页",
            device_id="phone-1",
        )

        self.assertEqual((), graph.risk_actions)
        self.assertEqual("navigation_only", graph.active_subgoal().external_impact)
        self.assertEqual("active", graph.active_subgoal().status)

    def test_initial_plan_aligns_explicit_safe_active_id_with_pending_status(self):
        payload = single_subgoal_payload(
            "空白标签页在当前浏览器中可见",
            external_impact="navigation_only",
        )
        payload["subgoals"][0]["status"] = "pending"

        graph = DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
            "新建一个空白标签页",
            device_id="phone-1",
        )

        self.assertEqual("target_state", graph.active_subgoal_id)
        self.assertEqual("active", graph.active_subgoal().status)

    def test_initial_plan_repairs_only_unique_dependency_frontier(self):
        payload = single_subgoal_payload(
            "当前页面的输入区域可见后内容为314159",
            external_impact="navigation_only",
        )
        payload["goal"]["entities"]["input_text"] = "314159"
        payload["subgoals"][0]["subgoal_id"] = "input_visible"
        payload["subgoals"].append(
            {
                "subgoal_id": "input_value_ready",
                "objective": "当前输入框内容为314159",
                "status": "active",
                "depends_on": ["input_visible"],
                "constraints": [],
                "completion_conditions": ["输入框内容为314159"],
                "completion_evidence": [],
                "risk_action_ids": [],
                "external_impact": "navigation_only",
            }
        )
        payload["active_subgoal_id"] = "input_visible"

        graph = DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
            "让当前页面的输入区域显示314159",
            device_id="phone-1",
        )

        self.assertEqual("input_visible", graph.active_subgoal_id)
        self.assertEqual(
            ["active", "pending"],
            [item.status for item in graph.subgoals],
        )

    def test_initial_plan_does_not_choose_between_independent_frontiers(self):
        payload = single_subgoal_payload(
            "确认两个独立区域",
            external_impact="read_only",
        )
        payload["subgoals"].append(
            {
                "subgoal_id": "other_root",
                "objective": "另一个独立区域可见",
                "status": "active",
                "depends_on": [],
                "constraints": [],
                "completion_conditions": ["另一个独立区域可见"],
                "completion_evidence": [],
                "risk_action_ids": [],
                "external_impact": "read_only",
            }
        )

        with self.assertRaisesRegex(TaskGraphError, "只能有一个活动子目标"):
            DeepSeekTaskGraphPlanner(FakeProvider(payload, payload, payload)).plan(
                "确认两个独立区域",
                device_id="phone-1",
            )

    def test_replan_repairs_downstream_active_marker_from_unique_frontier(self):
        initial = single_subgoal_payload(
            "输入区域可见后内容为314159",
            external_impact="navigation_only",
        )
        initial["goal"]["entities"]["input_text"] = "314159"
        initial["subgoals"][0].update(
            subgoal_id="input_visible",
            objective="输入区域可见",
            completion_conditions=["输入区域可见"],
        )
        initial["subgoals"].extend(
            [
                {
                    "subgoal_id": "input_focused",
                    "objective": "输入区域已聚焦",
                    "status": "pending",
                    "depends_on": ["input_visible"],
                    "constraints": [],
                    "completion_conditions": ["输入区域已聚焦"],
                    "completion_evidence": [],
                    "risk_action_ids": [],
                    "external_impact": "navigation_only",
                },
                {
                    "subgoal_id": "input_value_ready",
                    "objective": "输入框内容为314159",
                    "status": "pending",
                    "depends_on": ["input_focused"],
                    "constraints": [],
                    "completion_conditions": ["输入框内容为314159"],
                    "completion_evidence": [],
                    "risk_action_ids": [],
                    "external_impact": "navigation_only",
                },
            ]
        )
        initial["active_subgoal_id"] = "input_visible"
        revised = copy.deepcopy(initial)
        revised["status"] = "running"
        revised["subgoals"][0].update(
            status="completed",
            completion_evidence=["输入区域可见"],
        )
        revised["subgoals"][1]["status"] = "active"
        revised["subgoals"][2]["status"] = "active"
        revised["active_subgoal_id"] = "input_focused"
        planner = DeepSeekTaskGraphPlanner(FakeProvider(initial, revised))
        graph = planner.plan(
            "让当前页面的输入区域显示314159",
            device_id="phone-1",
        )

        result = planner.replan(
            graph,
            ObservedState(
                scene_id="scene-input-visible",
                summary="输入区域可见",
                visible_evidence=("输入区域可见",),
                grounded_visual_facts=(
                    '{"meaning":"application_text_input","role":"input"}',
                ),
                last_action_outcome="not_applicable",
            ),
            trigger="subgoal_completed",
            reason="可信画面已经证明输入区域可见。",
        )

        self.assertEqual("input_focused", result.active_subgoal_id)
        self.assertEqual(
            ["completed", "active", "pending"],
            [item.status for item in result.subgoals],
        )

    def test_initial_plan_never_normalizes_real_external_effect_risk(self):
        payload = single_subgoal_payload(
            "退出当前账号",
            external_impact="navigation_only",
        )
        payload["risk_actions"] = [
            {
                "risk_id": "logout",
                "description": "退出当前账号",
                "external_effect": "改变当前账号登录状态",
                "risk_type": "unknown_external_effect",
                "risk_level": "low",
                "subgoal_ids": ["target_state"],
                "confirmation_required": True,
            }
        ]
        payload["subgoals"][0]["risk_action_ids"] = ["logout"]

        with self.assertRaisesRegex(TaskGraphError, "外部状态变化但未声明"):
            DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
                "退出当前账号",
                device_id="phone-1",
            )

    def test_initial_plan_does_not_trust_no_effect_text_for_non_navigation_goal(self):
        payload = single_subgoal_payload(
            "把此人纳入小组",
            external_impact="navigation_only",
        )
        payload["risk_actions"] = [
            {
                "risk_id": "claimed_safe",
                "description": "改变小组成员",
                "external_effect": "无外部状态影响",
                "risk_type": "unknown_external_effect",
                "risk_level": "low",
                "subgoal_ids": ["target_state"],
                "confirmation_required": True,
            }
        ]
        payload["subgoals"][0]["risk_action_ids"] = ["claimed_safe"]

        with self.assertRaises(TaskGraphError):
            DeepSeekTaskGraphPlanner(FakeProvider(payload, copy.deepcopy(payload))).plan(
                "把此人纳入小组",
                device_id="phone-1",
            )

    def test_initial_plan_repairs_one_missing_required_field_error(self):
        invalid = base_payload()
        del invalid["goal"]["entities"]
        repaired = base_payload()
        provider = FakeProvider(invalid, repaired)

        graph = DeepSeekTaskGraphPlanner(provider).plan(
            "点击浏览器",
            device_id="phone-1",
        )

        self.assertEqual({"place": "图书馆"}, graph.goal.entities)
        self.assertEqual(len(provider.messages), 3)
        self.assertIn("goal 缺少字段：entities", provider.messages[1][0]["content"])

    def test_read_only_completion_review_repairs_unchanged_active_node(self):
        initial = single_subgoal_payload(
            "确认目标结果可见",
            external_impact="read_only",
        )
        unchanged = copy.deepcopy(initial)
        unchanged["status"] = "running"
        completed = copy.deepcopy(initial)
        completed["status"] = "completed"
        completed["completion_conditions"][0]["satisfied"] = True
        completed["completion_conditions"][0]["evidence"] = ["页面显示目标结果"]
        completed["subgoals"][0]["status"] = "completed"
        completed["subgoals"][0]["completion_evidence"] = ["页面显示目标结果"]
        completed["active_subgoal_id"] = None
        provider = FakeProvider(initial, unchanged, completed)
        planner = DeepSeekTaskGraphPlanner(provider)
        graph = planner.plan("确认目标结果可见", device_id="phone-1")

        result = planner.replan(
            graph,
            ObservedState(
                scene_id="scene-read-only",
                summary="目标结果已经显示",
                visible_evidence=("页面显示目标结果",),
                last_action_outcome="matched",
            ),
            trigger="subgoal_completed",
            reason="当前可信画面用于只读完成复核。",
        )

        self.assertEqual("completed", result.status)
        self.assertEqual(2, result.revision)
        graph_prompts = [
            call[0]["content"]
            for call in provider.messages
            if "semantic-risk-audit-v1" not in call[0]["content"]
        ]
        self.assertEqual(3, len(graph_prompts))
        self.assertIn("read_only 完成复核不能继续保留", graph_prompts[-1])

    def test_replan_rejects_named_page_completion_without_grounded_identity(self):
        objective = "原来的只读通用动作验收页面可见"
        initial = single_subgoal_payload(objective, external_impact="read_only")
        initial["completion_conditions"][0].update(
            description=objective,
            evidence_required=["画面显示原来的只读通用动作验收页面"],
        )
        initial["subgoals"][0]["completion_conditions"] = [objective]
        claimed = "当前设置列表构成原来的只读通用动作验收页面的可见证据"
        completed = copy.deepcopy(initial)
        completed["status"] = "completed"
        completed["completion_conditions"][0].update(
            satisfied=True,
            evidence=[claimed],
        )
        completed["subgoals"][0].update(
            status="completed",
            completion_evidence=[claimed],
        )
        completed["active_subgoal_id"] = None
        provider = FakeProvider(initial, completed, copy.deepcopy(completed))
        planner = DeepSeekTaskGraphPlanner(provider)
        graph = planner.plan(objective, device_id="phone-1")

        with self.assertRaisesRegex(TaskGraphError, "身份锚点"):
            planner.replan(
                graph,
                ObservedState(
                    scene_id="scene-settings",
                    summary="设置页面，包含多个设置入口",
                    visible_evidence=(claimed,),
                    grounded_visual_facts=(
                        '{"label":"设置列表","meaning":"settings_list","role":"container"}',
                    ),
                    last_action_outcome="matched",
                ),
                trigger="observation_changed",
                reason="动作后重新观察。",
            )
        replan_prompts = [
            call[0]["content"]
            for call in provider.messages
            if "高层任务图重规划器" in call[0]["content"]
        ]
        self.assertEqual(2, len(replan_prompts))
        self.assertIn(
            "其名称必须\n    能从 grounded_visual_facts",
            replan_prompts[0],
        )
        self.assertIn(
            "应跳过或替换尚未完成的\n    具名页面节点",
            replan_prompts[1],
        )

    def test_generic_element_presence_is_not_a_named_page_identity(self):
        self.assertEqual(
            "",
            _named_visual_identity_anchor(
                (
                    "目标控件（目标入口）在当前页面可见",
                    "目标元素在当前页面可见",
                )
            ),
        )
        self.assertEqual(
            "",
            _named_visual_identity_anchor(
                ("紫色方块和绿色终点在当前页面可见",)
            ),
        )
        self.assertTrue(
            _named_visual_identity_anchor(("原来的只读通用动作验收页面可见",))
        )
        self.assertEqual(
            "",
            _named_visual_identity_anchor(("通过页面返回流程回到结果页",)),
        )
        self.assertTrue(
            _named_visual_identity_anchor(("跨境订单结果页面可见",)),
        )

    def test_current_page_title_read_is_not_a_named_page_identity(self):
        self.assertEqual(
            "",
            _named_visual_identity_anchor(
                (
                    "当前页面主标题逐字可见",
                    "当前页面主标题文字逐字可见",
                )
            ),
        )
        self.assertEqual(
            "",
            _named_visual_identity_anchor(
                ("The current page main title text is visible verbatim",)
            ),
        )

    def test_quoted_ui_title_requires_its_complete_literal_identity(self):
        anchor = _quoted_visual_identity_anchor(
            ("“选择单一验收模式”标题清晰可见",)
        )

        self.assertEqual("选择单一验收模式", anchor)
        self.assertFalse(
            anchor in _compact_identity_text("返回验收模式选择 / page_title")
        )
        self.assertTrue(
            anchor in _compact_identity_text("选择单一验收模式 / page_title")
        )

    def test_current_page_input_state_is_not_a_named_page_identity(self):
        self.assertEqual(
            "",
            _named_visual_identity_anchor(
                (
                    "当前页面唯一输入框中的内容为 agent",
                    "当前本地页面唯一输入框已完整显示 agent",
                    "The current page input value is agent",
                )
            ),
        )
        self.assertEqual(
            "设置",
            _named_visual_identity_anchor(
                ("当前设置页面的搜索输入框可见且可交互",)
            ),
        )

    def test_current_page_transition_state_is_not_a_named_page_identity(self):
        self.assertEqual(
            "",
            _named_visual_identity_anchor(
                (
                    "当前本地页面完成重新载入",
                    "页面重新载入后的可见状态",
                    "The current local page has been reloaded",
                    "Page state after reload is visible",
                )
            ),
        )
        self.assertTrue(
            _named_visual_identity_anchor(("系统设置页面已打开",))
        )
        self.assertTrue(
            _named_visual_identity_anchor(("跨境订单结果页面可见",))
        )

    def test_replan_accepts_current_page_reload_transition_without_page_anchor(self):
        objective = "当前本地页面完成重新载入"
        transition_evidence = "controller_transition:receipt-reload:1"
        visible_evidence = "页面重新载入后的可见状态"
        initial = single_subgoal_payload(objective, external_impact="navigation_only")
        initial["completion_conditions"][0].update(
            description="当前本地页面已完成重新载入",
            evidence_required=[visible_evidence],
        )
        initial["subgoals"][0]["completion_conditions"] = [visible_evidence]
        completed = copy.deepcopy(initial)
        completed["status"] = "completed"
        completed["completion_conditions"][0].update(
            satisfied=True,
            evidence=[visible_evidence],
        )
        completed["subgoals"][0].update(
            status="completed",
            completion_evidence=[transition_evidence],
        )
        completed["active_subgoal_id"] = None
        provider = FakeProvider(initial, completed)
        planner = DeepSeekTaskGraphPlanner(provider)
        graph = planner.plan(objective, device_id="phone-1")
        matched = matched_controller_observation(
            graph,
            receipt_id="receipt-reload",
        )

        revised = planner.replan(
            graph,
            ObservedState(
                scene_id=matched.scene_id,
                summary="页面已恢复为空输入框且软键盘不可见",
                visible_evidence=(visible_evidence,),
                grounded_visual_facts=(
                    '{"meaning":"application_text_input","role":"input",'
                    '"states":{"value":""}}',
                    '{"overlay":"soft_keyboard","visible":false}',
                ),
                last_action_outcome="matched",
                verified_action_transition=matched.verified_action_transition,
                controller_transition_evidence_refs=(
                    matched.controller_transition_evidence_refs
                ),
            ),
            trigger="action_result_matched",
            reason="动作后重新观察。",
        )

        self.assertEqual("completed", revised.status)

    def test_replan_accepts_grounded_input_state_without_page_title_anchor(self):
        objective = "当前页面唯一输入框中的内容为 agent"
        initial = single_subgoal_payload(objective, external_impact="navigation_only")
        initial["goal"]["entities"]["input_text"] = "agent"
        initial["completion_conditions"][0].update(
            description=objective,
            evidence_required=["输入框中的文字为 agent"],
        )
        initial["subgoals"][0]["completion_conditions"] = [
            "输入框中的文字为 agent"
        ]
        completed = copy.deepcopy(initial)
        completed["status"] = "completed"
        completed["completion_conditions"][0].update(
            satisfied=True,
            evidence=["输入框中的文字为 agent"],
        )
        completed["subgoals"][0].update(
            status="completed",
            completion_evidence=["输入框中的文字为 agent"],
        )
        completed["active_subgoal_id"] = None
        provider = FakeProvider(initial, completed)
        planner = DeepSeekTaskGraphPlanner(provider)
        graph = planner.plan(objective, device_id="phone-1")

        revised = planner.replan(
            graph,
            ObservedState(
                scene_id="scene-input",
                summary="输入框显示 agent",
                visible_evidence=("输入框中的文字为 agent",),
                grounded_visual_facts=(
                    '{"meaning":"application_text_input","role":"input",'
                    '"states":{"value":"agent"}}',
                ),
                last_action_outcome="matched",
            ),
            trigger="observation_changed",
            reason="动作后重新观察。",
        )

        self.assertEqual("completed", revised.status)

    def test_replan_accepts_named_page_completion_with_grounded_identity(self):
        objective = "原来的只读通用动作验收页面可见"
        initial = single_subgoal_payload(objective, external_impact="read_only")
        initial["completion_conditions"][0].update(
            description=objective,
            evidence_required=["画面显示原来的只读通用动作验收页面"],
        )
        initial["subgoals"][0]["completion_conditions"] = [objective]
        evidence = "通用动作真机验收页标题清晰可见"
        completed = copy.deepcopy(initial)
        completed["status"] = "completed"
        completed["completion_conditions"][0].update(
            satisfied=True,
            evidence=[evidence],
        )
        completed["subgoals"][0].update(
            status="completed",
            completion_evidence=[evidence],
        )
        completed["active_subgoal_id"] = None
        provider = FakeProvider(initial, completed)
        planner = DeepSeekTaskGraphPlanner(provider)
        graph = planner.plan(objective, device_id="phone-1")

        revised = planner.replan(
            graph,
            ObservedState(
                scene_id="scene-acceptance",
                summary=evidence,
                visible_evidence=(evidence,),
                grounded_visual_facts=(
                    '{"label":"通用动作真机验收页","role":"text"}',
                ),
                last_action_outcome="matched",
            ),
            trigger="observation_changed",
            reason="动作后重新观察。",
        )

        self.assertEqual("completed", revised.status)

    def test_current_page_goal_repairs_missing_app_to_foreground_context(self):
        invalid = base_payload()
        invalid["goal"]["target_apps"] = []
        repaired = base_payload()
        repaired["goal"]["target_apps"] = [
            {"app_id": "current_foreground", "app_name": "当前前台应用"}
        ]
        provider = FakeProvider(invalid, repaired)

        graph = DeepSeekTaskGraphPlanner(provider).plan(
            "让当前页面显示目标内容",
            device_id="phone-1",
        )

        self.assertEqual("current_foreground", graph.goal.target_apps[0].app_id)
        self.assertEqual(3, len(provider.messages))
        self.assertIn("current_foreground", provider.messages[1][0]["content"])

    def test_protocol_field_named_like_a_repairable_error_is_still_rejected(self):
        invalid = base_payload()
        invalid["goal"]["缺少字段"] = "不能利用字段名触发修复"
        provider = FakeProvider(invalid, base_payload())

        with self.assertRaisesRegex(TaskGraphError, "协议外字段"):
            DeepSeekTaskGraphPlanner(provider).plan(
                "点击浏览器",
                device_id="phone-1",
            )

        self.assertEqual(len(provider.messages), 1)

    def test_initial_plan_repairs_one_low_level_protocol_violation(self):
        invalid = base_payload()
        invalid["goal"]["objective"] = "向上滑动一次，让蓝色终点进入画面"
        repaired = base_payload()
        repaired["constraints"] = [
            "不要点击其他控件",
            "页面内容只允许向上移动一次",
        ]
        provider = FakeProvider(invalid, repaired)

        graph = DeepSeekTaskGraphPlanner(provider).plan(
            "向上滑动一次，让蓝色终点进入画面",
            device_id="phone-1",
        )

        self.assertEqual(repaired["goal"]["objective"], graph.goal.objective)
        self.assertEqual(3, len(provider.messages))
        self.assertIn("goal.objective", provider.messages[1][0]["content"])
        self.assertIn("不要输出点击、滑动、输入", provider.messages[1][0]["content"])
        self.assertIn("只保留动作希望达到的可见结果状态", provider.messages[1][0]["content"])
        self.assertIn("逐字保留“不要点击其他控件”", provider.messages[1][0]["content"])
        self.assertIn("页面内容只允许向上移动一次", provider.messages[1][0]["content"])
        self.assertIn("目标页面不再被遮挡，主要内容可见", provider.messages[1][0]["content"])
        self.assertIn("当前输入框内容为 X", provider.messages[1][0]["content"])
        self.assertIn("本机临时结果区域显示该表达式的答案", provider.messages[1][0]["content"])
        self.assertEqual(
            ("不要点击其他控件", "页面内容只允许向上移动一次"),
            graph.constraints,
        )

    def test_initial_plan_second_low_level_protocol_violation_stays_blocked(self):
        first = base_payload()
        first["goal"]["objective"] = "向上滑动一次，让蓝色终点进入画面"
        second = copy.deepcopy(first)
        provider = FakeProvider(first, second)

        with self.assertRaisesRegex(TaskGraphError, "包含低层动作表达"):
            DeepSeekTaskGraphPlanner(provider).plan(
                "向上滑动一次，让蓝色终点进入画面",
                device_id="phone-1",
            )

        self.assertEqual(2, len(provider.messages))

    def test_explicit_action_like_ui_label_is_preserved_only_as_entity(self):
        payload = single_subgoal_payload(
            "“长按目标”对应的本地验收页面可见",
            external_impact="navigation_only",
        )
        payload["goal"]["entities"] = {}
        payload["completion_conditions"][0]["description"] = (
            "长按目标页面的黄色虚线区域可见"
        )
        payload["completion_conditions"][0]["evidence_required"] = [
            "长按目标页面显示黄色虚线区域"
        ]
        payload["subgoals"][0]["completion_conditions"] = [
            "长按目标页面显示黄色虚线区域"
        ]

        graph = DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
            "当前前台进入本地验收模式。"
            "目标入口的可见文字是“长按目标”；"
            "完成状态为黄色虚线区域可见。",
            device_id="phone-1",
        )

        self.assertEqual("长按目标", graph.goal.entities["target_ui_label"])
        state_text = " ".join(
            [
                graph.goal.objective,
                *(item.description for item in graph.completion_conditions),
                *(item.objective for item in graph.subgoals),
                *(
                    condition
                    for item in graph.subgoals
                    for condition in item.completion_conditions
                ),
            ]
        )
        self.assertNotIn("长按目标", state_text)
        self.assertNotIn("可见文字为目标入口", state_text)
        self.assertIn("目标入口", state_text)

    def test_quoted_action_like_label_before_ui_role_is_preserved_only_as_entity(self):
        variants = (
            ("语义点击", "入口"),
            ("拖动项目", "按钮"),
            ("输入文字", "选项"),
        )
        for label, role_noun in variants:
            with self.subTest(label=label, role_noun=role_noun):
                payload = single_subgoal_payload(
                    f"“{label}”{role_noun}对应的本地页面可见",
                    external_impact="navigation_only",
                )
                payload["goal"]["entities"] = {}

                graph = DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
                    f"查看“{label}”{role_noun}对应页面后结束。",
                    device_id="phone-1",
                )

                self.assertEqual(label, graph.goal.entities["target_ui_label"])
                state_text = " ".join(
                    (
                        graph.goal.objective,
                        *(item.description for item in graph.completion_conditions),
                        *(item.objective for item in graph.subgoals),
                        *(
                            condition
                            for item in graph.subgoals
                            for condition in item.completion_conditions
                        ),
                    )
                )
                self.assertNotIn(label, state_text)
                self.assertIn("目标入口", state_text)

    def test_multiple_quoted_action_like_ui_labels_fail_closed(self):
        payload = single_subgoal_payload(
            "“语义点击”入口和“拖动项目”按钮均可见",
            external_impact="navigation_only",
        )
        payload["goal"]["entities"] = {}

        with self.assertRaisesRegex(TaskGraphError, "多个动作词字面 UI 标签"):
            DeepSeekTaskGraphPlanner(
                FakeProvider(payload, copy.deepcopy(payload))
            ).plan(
                "核对“语义点击”入口和“拖动项目”按钮。",
                device_id="phone-1",
            )

    def test_explicit_label_normalizes_presence_only_subgoal_wording(self):
        payload = single_subgoal_payload(
            "目标控件（可见文字为“长按我 · 不要移动”）在当前页面可见",
            external_impact="read_only",
        )
        payload["goal"]["entities"] = {}
        payload["subgoals"][0]["completion_conditions"] = [
            "目标控件在当前页面可见"
        ]

        graph = DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
            "目标控件的可见文字是“长按我 · 不要移动”。",
            device_id="phone-1",
        )

        self.assertEqual(
            "长按我 · 不要移动",
            graph.goal.entities["target_ui_label"],
        )
        self.assertNotIn("文字", graph.subgoals[0].objective)
        self.assertIn("目标入口", graph.subgoals[0].objective)

    def test_action_rederived_from_literal_label_is_normalized_to_state(self):
        payload = single_subgoal_payload(
            "目标控件被长按且未移动",
            external_impact="navigation_only",
        )
        payload["goal"]["entities"] = {}
        payload["subgoals"][0]["completion_conditions"] = [
            "目标控件被长按",
            "目标控件未移动",
        ]

        graph = DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
            "目标控件的可见文字是“长按我 · 不要移动”。",
            device_id="phone-1",
        )

        state_text = " ".join(
            (
                graph.subgoals[0].objective,
                *graph.subgoals[0].completion_conditions,
            )
        )
        self.assertNotIn("长按", state_text)
        self.assertIn("当前页面可见的本机临时结果状态", state_text)

    def test_drag_rederived_from_literal_label_is_normalized_to_state(self):
        payload = single_subgoal_payload(
            "目标元素被拖动至目标区域",
            external_impact="navigation_only",
        )
        payload["goal"]["entities"] = {}
        payload["subgoals"][0]["completion_conditions"] = [
            "目标元素被拖动到目标区域"
        ]

        graph = DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
            "目标入口的可见文字是“拖动目标”。",
            device_id="phone-1",
        )

        state_text = " ".join(
            (
                graph.subgoals[0].objective,
                *graph.subgoals[0].completion_conditions,
            )
        )
        self.assertNotIn("拖动", state_text)
        self.assertIn("本机临时目标位置", state_text)

    def test_ui_label_normalization_rejects_conflicting_model_entity(self):
        payload = single_subgoal_payload(
            "长按目标对应的本地页面可见",
            external_impact="navigation_only",
        )
        payload["goal"]["entities"]["target_ui_label"] = "拖动目标"

        with self.assertRaisesRegex(TaskGraphError, "target_ui_label.*冲突"):
            DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
                "目标入口的可见文字是“长按目标”。",
                device_id="phone-1",
            )

    def test_rejects_low_level_action_fields(self):
        payload = base_payload()
        payload["subgoals"][0]["steps"] = [{"tap": [10, 20]}]
        with self.assertRaisesRegex(TaskGraphError, "协议外字段"):
            DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan("目标", device_id="phone-1")

    def test_rejects_low_level_instruction_in_subgoal_text(self):
        payload = base_payload()
        payload["subgoals"][0]["objective"] = "点击搜索结果中的图书馆"
        with self.assertRaisesRegex(TaskGraphError, "包含低层动作表达"):
            DeepSeekTaskGraphPlanner(FakeProvider(payload, copy.deepcopy(payload))).plan(
                "目标",
                device_id="phone-1",
            )

    def test_rejects_generic_low_level_primitives_coordinates_and_commands(self):
        forbidden_objectives = (
            "向上滑动页面",
            "长按当前项目",
            "拖动卡片到顶部",
            "移动到裸坐标(120, 340)",
            "按下音量键",
            "在输入框输入文字abc",
            "在输入框输入密码",
            "输入关键词天气",
            "输入abc",
            "运行 PowerShell 系统命令",
        )
        for objective in forbidden_objectives:
            with self.subTest(objective=objective):
                payload = base_payload()
                payload["subgoals"][0]["objective"] = objective
                with self.assertRaisesRegex(TaskGraphError, "包含低层动作表达"):
                    DeepSeekTaskGraphPlanner(
                        FakeProvider(payload, copy.deepcopy(payload))
                    ).plan(
                        "目标",
                        device_id="phone-1",
                    )

    def test_allows_input_as_observed_state_noun(self):
        payload = base_payload()
        payload["completion_conditions"][0]["evidence_required"] = [
            "未出现搜索输入或搜索结果",
            "输入框保持为空",
        ]
        graph = DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
            "仅查看页面",
            device_id="phone-1",
        )
        self.assertEqual(
            graph.completion_conditions[0].evidence_required,
            ("未出现搜索输入或搜索结果", "输入框保持为空"),
        )

    def test_allows_system_navigation_keys_as_visible_state_nouns(self):
        payload = base_payload()
        payload["completion_conditions"][0]["evidence_required"] = [
            "屏幕底部可见返回键、主页键和多任务按键"
        ]

        graph = DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
            "恢复系统导航区域可见状态",
            device_id="phone-1",
        )

        self.assertEqual(
            (
                "屏幕底部可见返回键、主页键和多任务按键",
            ),
            graph.completion_conditions[0].evidence_required,
        )

    def test_rejects_system_navigation_key_press_instruction(self):
        payload = base_payload()
        payload["completion_conditions"][0]["evidence_required"] = [
            "按返回键后页面返回"
        ]

        with self.assertRaisesRegex(TaskGraphError, "包含低层动作表达"):
            DeepSeekTaskGraphPlanner(
                FakeProvider(payload, copy.deepcopy(payload))
            ).plan(
                "目标",
                device_id="phone-1",
            )

    def test_rejects_low_level_instruction_in_completion_condition(self):
        payload = base_payload()
        payload["subgoals"][0]["completion_conditions"] = ["点击收藏按钮后完成"]
        with self.assertRaisesRegex(TaskGraphError, "包含低层动作表达"):
            DeepSeekTaskGraphPlanner(FakeProvider(payload, copy.deepcopy(payload))).plan(
                "目标",
                device_id="phone-1",
            )

    def test_rejects_low_level_instruction_hidden_as_constraint(self):
        payload = base_payload()
        payload["subgoals"][0]["constraints"] = ["点击第一个搜索结果"]
        with self.assertRaisesRegex(TaskGraphError, "包含低层动作表达"):
            DeepSeekTaskGraphPlanner(FakeProvider(payload, copy.deepcopy(payload))).plan(
                "目标",
                device_id="phone-1",
            )

    def test_allows_negated_low_level_safety_constraint(self):
        payload = base_payload()
        payload["subgoals"][0]["constraints"] = ["不要点击广告"]
        graph = DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
            "目标",
            device_id="phone-1",
        )
        self.assertEqual(graph.subgoals[0].constraints, ("不要点击广告",))

    def test_allows_long_negated_low_level_safety_constraint(self):
        payload = base_payload()
        payload["constraints"] = ["不要执行任何改变状态的操作"]
        payload["subgoals"][0]["constraints"] = list(payload["constraints"])
        graph = DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
            "确认当前只读结果",
            device_id="phone-1",
        )
        self.assertEqual(graph.constraints, ("不要执行任何改变状态的操作",))

    def test_negation_does_not_cover_later_positive_instruction(self):
        payload = base_payload()
        payload["subgoals"][0]["constraints"] = [
            "不要点击广告，但点击确定按钮"
        ]
        with self.assertRaisesRegex(TaskGraphError, "包含低层动作表达"):
            DeepSeekTaskGraphPlanner(
                FakeProvider(payload, copy.deepcopy(payload))
            ).plan(
                "目标",
                device_id="phone-1",
            )

    def test_rejects_control_data_hidden_in_entities(self):
        payload = base_payload()
        payload["goal"]["entities"]["coordinate"] = [10, 20]
        with self.assertRaisesRegex(TaskGraphError, "低层控制字段"):
            DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan("目标", device_id="phone-1")

    def test_empty_optional_input_text_is_normalized_as_absent(self):
        for empty_value in ("", None):
            with self.subTest(empty_value=empty_value):
                payload = base_payload()
                payload["goal"]["entities"]["input_text"] = empty_value

                graph = DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
                    "只查看页面，不输入",
                    device_id="phone-1",
                )

                self.assertNotIn("input_text", graph.goal.entities)

    def test_whitespace_input_text_is_not_silently_removed(self):
        payload = base_payload()
        payload["goal"]["entities"]["input_text"] = " "

        graph = DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
            "在输入框里输入一个空格",
            device_id="phone-1",
        )

        self.assertEqual(" ", graph.goal.entities["input_text"])

    def test_risk_action_cannot_disable_confirmation(self):
        payload = base_payload()
        payload["risk_actions"][0]["confirmation_required"] = False
        with self.assertRaisesRegex(TaskGraphError, "必须等待用户确认"):
            DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan("目标", device_id="phone-1")

    def test_deepseek_cannot_generate_confirmation_results(self):
        payload = base_payload()
        payload["confirmed_risk_ids"] = ["save_place"]
        with self.assertRaisesRegex(TaskGraphError, "协议外字段"):
            DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
                "目标",
                device_id="phone-1",
            )

    def test_risk_type_must_use_cross_app_vocabulary(self):
        payload = base_payload()
        payload["risk_actions"][0]["risk_type"] = "custom_page_flow"
        with self.assertRaisesRegex(TaskGraphError, "通用风险类型无效"):
            DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
                "目标",
                device_id="phone-1",
            )

    def test_external_state_subgoal_cannot_omit_risk(self):
        payload = base_payload()
        payload["risk_actions"] = []
        payload["subgoals"][1]["objective"] = "向联系人发送消息"
        payload["subgoals"][1]["completion_conditions"] = ["消息发送成功"]
        payload["subgoals"][1]["risk_action_ids"] = []
        with self.assertRaisesRegex(TaskGraphError, "必须关联风险并失败关闭"):
            DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
                "目标",
                device_id="phone-1",
            )

    def test_external_state_change_cannot_be_mislabeled_read_only(self):
        payload = base_payload()
        payload["risk_actions"] = []
        payload["subgoals"][1]["objective"] = "发送地点信息给联系人"
        payload["subgoals"][1]["completion_conditions"] = ["地点信息发送成功"]
        payload["subgoals"][1]["risk_action_ids"] = []
        payload["subgoals"][1]["external_impact"] = "read_only"
        with self.assertRaisesRegex(TaskGraphError, "外部状态变化但未声明"):
            DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
                "目标",
                device_id="phone-1",
            )

    def test_directly_negated_external_effect_is_a_safety_constraint(self):
        objective = "打开浏览器首页，仅查看，不搜索、不登录"
        payload = single_subgoal_payload(
            objective,
            external_impact="navigation_only",
        )
        graph = DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
            objective,
            device_id="phone-1",
        )
        self.assertEqual(graph.subgoals[0].external_impact, "navigation_only")

    def test_coordinated_negation_covers_each_prohibited_effect(self):
        objective = "浏览器首页可见"
        payload = single_subgoal_payload(
            objective,
            external_impact="navigation_only",
        )
        payload["subgoals"][0]["constraints"] = [
            "仅打开浏览器，不进行搜索或登录"
        ]
        graph = DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
            objective,
            device_id="phone-1",
        )
        self.assertEqual(graph.subgoals[0].external_impact, "navigation_only")

    def test_universal_quantifier_inside_negation_remains_a_safety_constraint(self):
        objective = "执行系统返回"
        payload = single_subgoal_payload(
            objective,
            external_impact="navigation_only",
        )
        payload["subgoals"][0]["constraints"] = [
            "不得进行任何输入、提交或账号数据操作"
        ]

        graph = DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
            objective,
            device_id="phone-1",
        )

        self.assertEqual(graph.subgoals[0].external_impact, "navigation_only")

    def test_exiting_current_app_is_navigation_not_account_state_change(self):
        objective = "执行系统返回，回到上一级页面或退出当前应用"
        payload = single_subgoal_payload(
            objective,
            external_impact="navigation_only",
        )

        graph = DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
            objective,
            device_id="phone-1",
        )

        self.assertEqual(graph.subgoals[0].external_impact, "navigation_only")

    def test_exiting_account_still_requires_external_state_risk(self):
        objective = "退出当前账号"
        payload = single_subgoal_payload(
            objective,
            external_impact="navigation_only",
        )

        with self.assertRaisesRegex(TaskGraphError, "外部状态变化但未声明"):
            DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
                objective,
                device_id="phone-1",
            )

    def test_negation_word_does_not_hide_a_positive_external_effect(self):
        objective = "不要忘记登录并同步数据"
        payload = single_subgoal_payload(
            objective,
            external_impact="navigation_only",
        )
        with self.assertRaisesRegex(TaskGraphError, "外部状态变化但未声明"):
            DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
                objective,
                device_id="phone-1",
            )

    def test_relationship_communication_membership_and_role_effects_fail_closed(self):
        dangerous_goals = (
            "加好友",
            "拉他进群",
            "发私信询问需求",
            "邀请用户入群",
            "取消关注该用户",
            "给对方留言",
            "建群或修改群成员",
            "把用户设为管理员",
        )
        for objective in dangerous_goals:
            with self.subTest(objective=objective):
                payload = single_subgoal_payload(
                    objective,
                    external_impact="read_only",
                )
                with self.assertRaisesRegex(TaskGraphError, "外部状态变化但未声明"):
                    DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
                        objective,
                        device_id="phone-1",
                    )

    def test_synonymous_external_effects_cannot_bypass_classification(self):
        synonymous_goals = (
            "添加联系人",
            "把对方列为好友",
            "请该用户加入团队",
            "撤销管理员权限",
            "给联系人发消息",
            "将成员移出团队",
        )
        for objective in synonymous_goals:
            with self.subTest(objective=objective):
                payload = single_subgoal_payload(
                    objective,
                    external_impact="read_only",
                )
                with self.assertRaisesRegex(TaskGraphError, "外部状态变化但未声明"):
                    DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
                        objective,
                        device_id="phone-1",
                    )

    def test_declared_risk_type_must_match_inferred_external_effect(self):
        payload = single_subgoal_payload(
            "向联系人发送消息",
            external_impact="external_state",
        )
        payload["risk_actions"] = [copy.deepcopy(base_payload()["risk_actions"][0])]
        payload["risk_actions"][0]["subgoal_ids"] = ["target_state"]
        payload["subgoals"][0]["risk_action_ids"] = ["save_place"]
        payload["status"] = "awaiting_confirmation"
        with self.assertRaisesRegex(TaskGraphError, "缺少匹配的通用风险类型"):
            DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
                "向联系人发送消息",
                device_id="phone-1",
            )

    def test_external_effect_hidden_in_constraints_cannot_bypass_classification(self):
        payload = single_subgoal_payload(
            "处理联系人请求",
            external_impact="read_only",
        )
        payload["subgoals"][0]["constraints"] = ["完成前给联系人发消息"]
        with self.assertRaisesRegex(TaskGraphError, "外部状态变化但未声明"):
            DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
                "处理联系人请求",
                device_id="phone-1",
            )

    def test_semantic_audit_rejects_mixed_read_and_external_effect_phrasings(self):
        cases = (
            ("查看资料后把他弄进群里", "membership_change"),
            ("查看资料后和他聊聊需求", "message_or_communication"),
            ("查看后把他的权限升一下", "permission_role_change"),
            ("查看后替他买了", "transaction_or_payment"),
            ("核对资料再把此人纳入小组", "membership_change"),
            ("读完简介之后跟对方碰一碰想法", "message_or_communication"),
            ("了解情况后把他的身份提到管理级别", "permission_role_change"),
            ("看完说明替对方把东西拿下", "transaction_or_payment"),
        )
        for objective, risk_type in cases:
            with self.subTest(objective=objective):
                payload = single_subgoal_payload(
                    objective,
                    external_impact="read_only",
                )
                source_id = "subgoals.target_state.objective"
                audit = audit_payload_for_graph(
                    payload,
                    overrides={
                        source_id: {
                            "external_impact": "external_state",
                            "risk_types": [risk_type],
                        }
                    },
                )
                provider = FakeProvider(payload, audit_payloads=[audit])
                with self.assertRaisesRegex(TaskGraphError, "语义风险审计.*冲突"):
                    DeepSeekTaskGraphPlanner(provider).plan(
                        objective,
                        device_id="phone-1",
                    )

    def test_semantic_audit_preserves_field_boundaries_for_navigation(self):
        objective = "打开联系人页面"
        payload = single_subgoal_payload(
            objective,
            external_impact="navigation_only",
        )
        payload["subgoals"][0]["completion_conditions"] = [objective]
        audit = audit_payload_for_graph(payload)
        provider = FakeProvider(payload, audit_payloads=[audit])
        planner = DeepSeekTaskGraphPlanner(provider)

        graph = planner.plan(objective, device_id="phone-1")

        self.assertEqual(graph.active_subgoal().external_impact, "navigation_only")
        self.assertEqual(planner.risk_audit_call_count, 1)
        audit_prompt = provider.messages[1][0]["content"]
        self.assertIn('"source_id": "subgoals.target_state.objective"', audit_prompt)
        self.assertIn(
            '"source_id": "subgoals.target_state.completion_conditions.0"',
            audit_prompt,
        )
        self.assertIn("不登录", audit_prompt)
        self.assertIn("不要忘记登录", audit_prompt)
        self.assertIn("确认某内容可见", audit_prompt)
        self.assertIn("不得仅因“确认”一词返回 unknown", audit_prompt)

    def test_semantic_audit_allows_disagreement_between_safe_impacts(self):
        cases = (
            ("navigation_only", "read_only"),
            ("read_only", "navigation_only"),
        )
        for graph_impact, audit_impact in cases:
            with self.subTest(graph_impact=graph_impact, audit_impact=audit_impact):
                payload = single_subgoal_payload(
                    "目标页面可见",
                    external_impact=graph_impact,
                )
                audit = audit_payload_for_graph(payload)
                for assessment in audit["assessments"]:
                    if assessment["subgoal_id"] == "target_state":
                        assessment["external_impact"] = audit_impact
                        assessment["risk_types"] = []
                graph = DeepSeekTaskGraphPlanner(
                    FakeProvider(payload, audit_payloads=[audit])
                ).plan(
                    "目标页面可见",
                    device_id="phone-1",
                )
                self.assertEqual(
                    graph.active_subgoal().external_impact,
                    graph_impact,
                )

    def test_false_positive_audit_cannot_turn_direct_negation_into_external_state(self):
        objective = "打开浏览器首页，仅查看，不搜索、不登录"
        payload = single_subgoal_payload(
            objective,
            external_impact="navigation_only",
        )
        audit = audit_payload_for_graph(
            payload,
            overrides={
                "raw_goal": {
                    "external_impact": "external_state",
                    "risk_types": ["account_or_permission_change"],
                },
                "goal.objective": {
                    "external_impact": "external_state",
                    "risk_types": ["account_or_permission_change"],
                },
                "subgoals.target_state.objective": {
                    "external_impact": "external_state",
                    "risk_types": ["account_or_permission_change"],
                },
            },
        )
        planner = DeepSeekTaskGraphPlanner(
            FakeProvider(payload, audit_payloads=[audit])
        )

        graph = planner.plan(objective, device_id="phone-1")

        self.assertEqual(graph.active_subgoal().external_impact, "navigation_only")
        corrected = {
            item.source_id: item
            for item in planner.last_risk_audit.assessments
            if item.source_id
            in {"raw_goal", "goal.objective", "subgoals.target_state.objective"}
        }
        self.assertTrue(
            all(item.external_impact == "navigation_only" for item in corrected.values())
        )

    def test_false_positive_audit_cannot_turn_transient_tab_into_external_state(self):
        objective = "新建一个空白标签页"
        payload = single_subgoal_payload(
            objective,
            external_impact="navigation_only",
        )
        audit = audit_payload_for_graph(
            payload,
            overrides={
                "raw_goal": {
                    "external_impact": "external_state",
                    "risk_types": ["unknown_external_effect"],
                },
                "goal.objective": {
                    "external_impact": "external_state",
                    "risk_types": ["unknown_external_effect"],
                },
                "subgoals.target_state.objective": {
                    "external_impact": "external_state",
                    "risk_types": ["unknown_external_effect"],
                },
            },
        )

        graph = DeepSeekTaskGraphPlanner(
            FakeProvider(payload, audit_payloads=[audit])
        ).plan(objective, device_id="phone-1")

        self.assertEqual("navigation_only", graph.active_subgoal().external_impact)

    def test_false_positive_data_mutation_audit_cannot_turn_transient_tab_external(self):
        objective = "打开一个空白标签页"
        payload = single_subgoal_payload(
            objective,
            external_impact="navigation_only",
        )
        audit = audit_payload_for_graph(
            payload,
            overrides={
                "raw_goal": {
                    "external_impact": "external_state",
                    "risk_types": ["data_mutation"],
                },
                "goal.objective": {
                    "external_impact": "external_state",
                    "risk_types": ["data_mutation"],
                },
                "subgoals.target_state.objective": {
                    "external_impact": "external_state",
                    "risk_types": ["data_mutation"],
                },
            },
        )

        graph = DeepSeekTaskGraphPlanner(
            FakeProvider(payload, audit_payloads=[audit])
        ).plan(objective, device_id="phone-1")

        self.assertEqual("navigation_only", graph.active_subgoal().external_impact)

    def test_false_positive_audit_cannot_turn_unobscured_page_state_external(self):
        objectives = (
            "当前主界面无遮挡，主要内容清晰可见",
            "遮挡层不再可见且当前页面内容清晰可见",
        )
        for objective in objectives:
            with self.subTest(objective=objective):
                payload = single_subgoal_payload(
                    objective,
                    external_impact="navigation_only",
                )
                audit = audit_payload_for_graph(
                    payload,
                    overrides={
                        source_id: {
                            "external_impact": "external_state",
                            "risk_types": ["unknown_external_effect"],
                        }
                        for source_id in {
                            "raw_goal",
                            "goal.objective",
                            "completion_conditions.result_visible.description",
                            "completion_conditions.result_visible.evidence_required.0",
                            "subgoals.target_state.objective",
                            "subgoals.target_state.completion_conditions.0",
                        }
                    },
                )
                planner = DeepSeekTaskGraphPlanner(
                    FakeProvider(payload, audit_payloads=[audit])
                )

                graph = planner.plan(objective, device_id="phone-1")

                self.assertEqual(
                    "navigation_only",
                    graph.active_subgoal().external_impact,
                )
                self.assertEqual((), graph.active_subgoal().risk_action_ids)
                self.assertTrue(
                    all(
                        item.external_impact == "navigation_only"
                        and not item.risk_types
                        for item in planner.last_risk_audit.assessments
                    )
                )

    def test_unobscured_page_state_does_not_hide_external_effect(self):
        objective = "当前主界面无遮挡并登录账号"
        payload = single_subgoal_payload(
            objective,
            external_impact="navigation_only",
        )

        with self.assertRaisesRegex(TaskGraphError, "外部状态|通用风险"):
            DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
                objective,
                device_id="phone-1",
            )

    def test_false_positive_audit_cannot_turn_local_keyboard_mode_external(self):
        objective = "修改验收页软键盘输入法为英文直输模式"
        payload = single_subgoal_payload(
            objective,
            external_impact="navigation_only",
        )
        payload["goal"]["entities"] = {
            "target_page": "验收页",
            "keyboard_input_mode": "direct_latin",
        }
        payload["goal"]["target_apps"] = [
            {"app_id": "current_foreground", "app_name": "当前前台应用"}
        ]
        audit = audit_payload_for_graph(
            payload,
            overrides={
                source_id: {
                    "external_impact": "external_state",
                    "risk_types": ["data_mutation"],
                }
                for source_id in {
                    "raw_goal",
                    "goal.objective",
                    "completion_conditions.result_visible.description",
                    "completion_conditions.result_visible.evidence_required.0",
                    "subgoals.target_state.objective",
                    "subgoals.target_state.completion_conditions.0",
                }
            },
        )
        planner = DeepSeekTaskGraphPlanner(
            FakeProvider(payload, audit_payloads=[audit])
        )

        graph = planner.plan(objective, device_id="phone-1")

        self.assertEqual("navigation_only", graph.active_subgoal().external_impact)
        self.assertEqual((), graph.risk_actions)
        self.assertTrue(
            all(
                item.external_impact == "navigation_only" and not item.risk_types
                for item in planner.last_risk_audit.assessments
            )
        )

    def test_unknown_audit_cannot_override_strict_local_literal_action_state(self):
        objective = "目标控件处于当前页面可见的本机临时结果状态"
        payload = single_subgoal_payload(
            objective,
            external_impact="navigation_only",
        )
        payload["goal"]["target_apps"] = [
            {"app_id": "current_foreground", "app_name": "当前前台应用"}
        ]
        payload["goal"]["entities"] = {
            "target_ui_label": "长按我 · 不要移动"
        }
        payload["constraints"] = ["不得发布内容"]
        payload["subgoals"][0]["constraints"] = ["不得发布内容"]
        raw_goal = (
            "目标控件的可见文字是“长按我 · 不要移动”。"
            "目标控件处于当前页面可见的本机临时结果状态。"
            "不得发布内容。"
        )
        audit = audit_payload_for_graph(payload)
        for assessment in audit["assessments"]:
            assessment["external_impact"] = "unknown"
            assessment["risk_types"] = ["unknown_external_effect"]
        planner = DeepSeekTaskGraphPlanner(
            FakeProvider(payload, audit_payloads=[audit])
        )

        graph = planner.plan(raw_goal, device_id="phone-1")

        self.assertEqual("navigation_only", graph.active_subgoal().external_impact)
        self.assertEqual((), graph.risk_actions)
        self.assertTrue(
            all(
                item.external_impact == "navigation_only" and not item.risk_types
                for item in planner.last_risk_audit.assessments
            )
        )

    def test_local_literal_action_reconciliation_rejects_risky_label(self):
        objective = "目标控件处于当前页面可见的本机临时结果状态"
        payload = single_subgoal_payload(
            objective,
            external_impact="navigation_only",
        )
        payload["goal"]["target_apps"] = [
            {"app_id": "current_foreground", "app_name": "当前前台应用"}
        ]
        payload["goal"]["entities"] = {
            "target_ui_label": "长按并删除数据"
        }
        raw_goal = (
            "目标控件的可见文字是“长按并删除数据”。"
            "目标控件处于当前页面可见的本机临时结果状态。"
            "不得发布内容。"
        )
        audit = audit_payload_for_graph(payload)
        for assessment in audit["assessments"]:
            assessment["external_impact"] = "unknown"
            assessment["risk_types"] = ["unknown_external_effect"]

        with self.assertRaisesRegex(TaskGraphError, "外部状态|unknown"):
            DeepSeekTaskGraphPlanner(
                FakeProvider(payload, audit_payloads=[audit])
            ).plan(raw_goal, device_id="phone-1")

    def test_local_keyboard_mode_exception_never_hides_account_change(self):
        objective = "修改账号权限并将软键盘输入法改为英文直输模式"
        payload = single_subgoal_payload(
            objective,
            external_impact="navigation_only",
        )

        with self.assertRaisesRegex(TaskGraphError, "外部状态|通用风险"):
            DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
                objective,
                device_id="phone-1",
            )

    def test_default_input_method_setting_stays_external(self):
        objective = "修改系统默认输入法为英文直输模式"
        payload = single_subgoal_payload(
            objective,
            external_impact="navigation_only",
        )
        payload["goal"]["target_apps"] = [
            {"app_id": "current_foreground", "app_name": "当前前台应用"}
        ]

        with self.assertRaisesRegex(TaskGraphError, "外部状态|通用风险"):
            DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
                objective,
                device_id="phone-1",
            )

    def test_false_positive_audit_cannot_turn_reversible_navigation_external(self):
        objective = "activate browser page reload"
        payload = single_subgoal_payload(
            objective,
            external_impact="navigation_only",
        )
        payload["goal"]["entities"] = {
            "navigation_effect": "trigger_reload",
            "resource_scope": "current visible page",
        }
        payload["subgoals"][0]["completion_conditions"] = [
            "current visible page has been reloaded"
        ]
        audit = audit_payload_for_graph(
            payload,
            overrides={
                source_id: {
                    "external_impact": "external_state",
                    "risk_types": ["data_mutation"],
                }
                for source_id in {
                    "raw_goal",
                    "goal.objective",
                    "completion_conditions.result_visible.description",
                    "completion_conditions.result_visible.evidence_required.0",
                    "subgoals.target_state.objective",
                    "subgoals.target_state.completion_conditions.0",
                }
            },
        )
        planner = DeepSeekTaskGraphPlanner(
            FakeProvider(payload, audit_payloads=[audit])
        )

        graph = planner.plan(objective, device_id="phone-1")

        self.assertEqual("navigation_only", graph.active_subgoal().external_impact)
        self.assertEqual((), graph.active_subgoal().risk_action_ids)
        self.assertEqual((), graph.risk_actions)
        self.assertTrue(
            all(
                item.external_impact == "navigation_only" and not item.risk_types
                for item in planner.last_risk_audit.assessments
            )
        )

    def test_reversible_navigation_reconciliation_fails_closed_for_risky_entities(self):
        objective = "activate current page reload"
        payload = single_subgoal_payload(
            objective,
            external_impact="navigation_only",
        )
        payload["goal"]["entities"] = {
            "navigation_effect": "trigger_reload",
            "follow_up_effect": "save content",
        }
        audit = audit_payload_for_graph(
            payload,
            overrides={
                "raw_goal": {
                    "external_impact": "external_state",
                    "risk_types": ["data_mutation"],
                }
            },
        )

        with self.assertRaisesRegex(TaskGraphError, "语义风险审计.*冲突"):
            DeepSeekTaskGraphPlanner(
                FakeProvider(payload, audit_payloads=[audit])
            ).plan(objective, device_id="phone-1")

    def test_semantic_audit_prompt_classifies_command_wrappers_by_effect(self):
        objective = "activate the visible control to reload the current view"
        payload = single_subgoal_payload(
            objective,
            external_impact="navigation_only",
        )
        provider = FakeProvider(payload)

        DeepSeekTaskGraphPlanner(provider).plan(objective, device_id="phone-1")

        audit_prompt = provider.messages[1][0]["content"]
        self.assertIn("activate/trigger/激活/触发", audit_prompt)
        self.assertIn("必须按其实际语义", audit_prompt)
        self.assertIn("navigation_only", audit_prompt)

    def test_semantic_audit_prompt_documents_keyboard_mode_boundary(self):
        objective = "当前软键盘显示英文直输模式"
        payload = single_subgoal_payload(
            objective,
            external_impact="navigation_only",
        )
        payload["goal"]["target_apps"] = [
            {"app_id": "current_foreground", "app_name": "当前前台应用"}
        ]
        provider = FakeProvider(payload)

        DeepSeekTaskGraphPlanner(provider).plan(objective, device_id="phone-1")

        audit_prompt = provider.messages[1][0]["content"]
        self.assertIn("当前可见软键盘", audit_prompt)
        self.assertIn("系统默认输入法", audit_prompt)
        self.assertIn("账号同步", audit_prompt)

    def test_false_positive_audit_cannot_turn_explicit_unsubmitted_input_external(self):
        raw_goal = (
            "把顶部搜索输入框中的现有文字替换为 Agent123，"
            "不要搜索、提交、发送、保存或发布。"
        )
        objective = (
            "顶部搜索输入框中的文字为 Agent123，且未发生搜索、提交、发送、保存或发布。"
        )
        payload = single_subgoal_payload(
            objective,
            external_impact="navigation_only",
        )
        payload["goal"]["entities"]["input_text"] = "Agent123"
        payload["constraints"] = ["不要搜索、提交、发送、保存或发布。"]
        payload["subgoals"][0]["constraints"] = list(payload["constraints"])
        audit = audit_payload_for_graph(
            payload,
            overrides={
                "raw_goal": {
                    "external_impact": "external_state",
                    "risk_types": ["data_mutation"],
                },
                "goal.objective": {
                    "external_impact": "external_state",
                    "risk_types": ["data_mutation"],
                },
                "completion_conditions.result_visible.description": {
                    "external_impact": "external_state",
                    "risk_types": ["data_mutation"],
                },
                "subgoals.target_state.objective": {
                    "external_impact": "external_state",
                    "risk_types": ["data_mutation"],
                },
                "subgoals.target_state.completion_conditions.0": {
                    "external_impact": "external_state",
                    "risk_types": ["data_mutation"],
                },
            },
        )
        provider = FakeProvider(payload, audit_payloads=[audit])
        planner = DeepSeekTaskGraphPlanner(provider)

        graph = planner.plan(raw_goal, device_id="phone-1")

        self.assertEqual("navigation_only", graph.active_subgoal().external_impact)
        self.assertEqual((), graph.active_subgoal().risk_action_ids)
        corrected = {
            item.source_id: item
            for item in planner.last_risk_audit.assessments
            if item.source_id
            in {
                "raw_goal",
                "goal.objective",
                "subgoals.target_state.objective",
            }
        }
        self.assertTrue(
            all(item.external_impact == "navigation_only" for item in corrected.values())
        )

    def test_current_temporary_draft_empty_state_removes_invented_external_risk(self):
        raw_goal = (
            "将当前页面唯一临时草稿区域恢复为空白状态，完成时停留原页面并保持键盘可见；"
            "不得产生任何账号或外部影响。"
        )
        objective = "当前页面唯一临时草稿区域内容为空，且页面保持原样，键盘可见"
        payload = single_subgoal_payload(
            objective,
            external_impact="external_state",
        )
        payload["goal"]["objective"] = raw_goal
        payload["goal"]["target_apps"] = [
            {"app_id": "current_foreground", "app_name": "当前前台应用"}
        ]
        payload["goal"]["entities"] = {
            "target_ui_label": "唯一临时草稿区域"
        }
        payload["constraints"] = ["不得产生任何账号或外部影响"]
        payload["risk_actions"] = [
            {
                "risk_id": "accidental_external_effect",
                "description": "过程中可能误触发送、保存或发布等操作",
                "external_effect": "可能发送消息、发布内容或保存数据",
                "risk_type": "unknown_external_effect",
                "risk_level": "medium",
                "subgoal_ids": ["target_state"],
                "confirmation_required": True,
            }
        ]
        payload["subgoals"][0]["constraints"] = list(payload["constraints"])
        payload["subgoals"][0]["completion_conditions"] = [objective]
        payload["subgoals"][0]["risk_action_ids"] = [
            "accidental_external_effect"
        ]

        normalized = copy.deepcopy(payload)
        normalized["risk_actions"] = []
        normalized["subgoals"][0]["risk_action_ids"] = []
        normalized["subgoals"][0]["external_impact"] = "navigation_only"
        planner = DeepSeekTaskGraphPlanner(
            FakeProvider(
                payload,
                audit_payloads=[audit_payload_for_graph(normalized)],
            )
        )

        graph = planner.plan(raw_goal, device_id="phone-1")

        self.assertEqual((), graph.risk_actions)
        self.assertEqual("navigation_only", graph.active_subgoal().external_impact)
        self.assertEqual((), graph.active_subgoal().risk_action_ids)
        self.assertNotIn("input_text", graph.goal.entities)

    def test_saved_or_cloud_draft_empty_state_remains_external(self):
        raw_goal = (
            "将当前页面已保存的云端草稿恢复为空白；不得产生其他外部影响。"
        )
        objective = "当前页面已保存的云端草稿内容为空"
        payload = single_subgoal_payload(
            objective,
            external_impact="external_state",
        )
        payload["goal"]["objective"] = raw_goal
        payload["goal"]["target_apps"] = [
            {"app_id": "current_foreground", "app_name": "当前前台应用"}
        ]
        payload["constraints"] = ["不得产生其他外部影响"]
        payload["risk_actions"] = [
            {
                "risk_id": "mutate_saved_draft",
                "description": "修改已保存的云端草稿",
                "external_effect": "云端草稿数据被修改",
                "risk_type": "data_mutation",
                "risk_level": "high",
                "subgoal_ids": ["target_state"],
                "confirmation_required": True,
            }
        ]
        payload["subgoals"][0]["constraints"] = list(payload["constraints"])
        payload["subgoals"][0]["completion_conditions"] = [objective]
        payload["subgoals"][0]["risk_action_ids"] = ["mutate_saved_draft"]

        graph = DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
            raw_goal,
            device_id="phone-1",
        )

        self.assertEqual("external_state", graph.active_subgoal().external_impact)
        self.assertEqual(("mutate_saved_draft",), graph.active_subgoal().risk_action_ids)

    def test_symbolic_local_input_inherits_global_effect_boundary(self):
        for input_text in ("12+34", "12÷3"):
            with self.subTest(input_text=input_text):
                raw_goal = (
                    f"在本机工具中算出 {input_text} 并停留查看；"
                    "不得保存、分享、发送或改变任何账号状态。"
                )
                objective = f"当前输入框内容为 {input_text}"
                payload = single_subgoal_payload(
                    objective,
                    external_impact="navigation_only",
                )
                payload["goal"]["entities"]["input_text"] = input_text
                payload["constraints"] = [
                    "不得保存、分享、发送或改变任何账号状态。"
                ]
                payload["subgoals"][0]["constraints"] = []
                audit = audit_payload_for_graph(
                    payload,
                    overrides={
                        "raw_goal": {
                            "external_impact": "external_state",
                            "risk_types": ["data_mutation"],
                        },
                        "goal.objective": {
                            "external_impact": "external_state",
                            "risk_types": ["data_mutation"],
                        },
                        "subgoals.target_state.objective": {
                            "external_impact": "external_state",
                            "risk_types": ["data_mutation"],
                        },
                        "subgoals.target_state.completion_conditions.0": {
                            "external_impact": "external_state",
                            "risk_types": ["data_mutation"],
                        },
                    },
                )
                planner = DeepSeekTaskGraphPlanner(
                    FakeProvider(payload, audit_payloads=[audit])
                )

                graph = planner.plan(raw_goal, device_id="phone-1")

                self.assertEqual(
                    "navigation_only",
                    graph.active_subgoal().external_impact,
                )
                corrected = {
                    item.source_id: item
                    for item in planner.last_risk_audit.assessments
                    if item.source_id
                    in {
                        "raw_goal",
                        "goal.objective",
                        "subgoals.target_state.objective",
                    }
                }
                self.assertTrue(
                    all(
                        item.external_impact == "navigation_only"
                        and not item.risk_types
                        for item in corrected.values()
                    )
                )

    def test_risk_bound_active_initial_graph_enters_confirmation_gate(self):
        for impact, status in (
            ("external_state", "ready"),
            ("unknown", "running"),
        ):
            with self.subTest(impact=impact, status=status):
                payload = single_subgoal_payload(
                    "本机硬件状态处于关闭状态",
                    external_impact=impact,
                )
                payload["status"] = status
                payload["risk_actions"] = [
                    {
                        "risk_id": "hardware_state_change",
                        "description": "改变本机硬件状态",
                        "external_effect": "本机硬件状态发生变化",
                        "risk_type": "unknown_external_effect",
                        "risk_level": "low",
                        "subgoal_ids": ["target_state"],
                        "confirmation_required": True,
                    }
                ]
                payload["subgoals"][0]["risk_action_ids"] = [
                    "hardware_state_change"
                ]

                graph = DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
                    "使本机硬件状态发生变化",
                    device_id="phone-1",
                )

                self.assertEqual("awaiting_confirmation", graph.status)
                self.assertEqual(
                    ("hardware_state_change",),
                    graph.active_subgoal().risk_action_ids,
                )

    def test_initial_confirmation_status_never_mints_missing_risk(self):
        payload = single_subgoal_payload(
            "本机硬件状态处于关闭状态",
            external_impact="external_state",
        )
        payload["status"] = "ready"

        with self.assertRaisesRegex(TaskGraphError, "必须关联风险"):
            DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
                "改变本机硬件状态",
                device_id="phone-1",
            )

    def test_unsubmitted_input_without_explicit_effect_boundary_stays_external(self):
        objective = "顶部搜索输入框中的文字为 Agent123"
        payload = single_subgoal_payload(
            objective,
            external_impact="navigation_only",
        )
        payload["goal"]["entities"]["input_text"] = "Agent123"
        audit = audit_payload_for_graph(
            payload,
            overrides={
                "raw_goal": {
                    "external_impact": "external_state",
                    "risk_types": ["data_mutation"],
                },
                "goal.objective": {
                    "external_impact": "external_state",
                    "risk_types": ["data_mutation"],
                },
                "subgoals.target_state.objective": {
                    "external_impact": "external_state",
                    "risk_types": ["data_mutation"],
                },
            },
        )

        with self.assertRaisesRegex(TaskGraphError, "任务图分类冲突"):
            DeepSeekTaskGraphPlanner(
                FakeProvider(payload, audit_payloads=[audit])
            ).plan(objective, device_id="phone-1")

    def test_explicit_unsubmitted_input_correction_does_not_depend_on_model_risk_label(self):
        raw_goal = (
            "把顶部搜索输入框中的现有文字替换为 Agent123，"
            "不要搜索、提交、发送、保存或发布。"
        )
        objective = (
            "顶部搜索输入框中的文字为 Agent123，且未发生搜索、提交、发送、保存或发布。"
        )
        payload = single_subgoal_payload(
            objective,
            external_impact="navigation_only",
        )
        payload["goal"]["entities"]["input_text"] = "Agent123"
        payload["constraints"] = ["不要搜索、提交、发送、保存或发布。"]
        payload["subgoals"][0]["constraints"] = list(payload["constraints"])
        audit = audit_payload_for_graph(
            payload,
            overrides={
                "raw_goal": {
                    "external_impact": "external_state",
                    "risk_types": ["transaction_or_payment"],
                },
                "goal.objective": {
                    "external_impact": "external_state",
                    "risk_types": ["account_relationship_change"],
                },
                "subgoals.target_state.objective": {
                    "external_impact": "unknown",
                    "risk_types": ["unknown_external_effect"],
                },
            },
        )

        planner = DeepSeekTaskGraphPlanner(
            FakeProvider(payload, audit_payloads=[audit])
        )
        graph = planner.plan(raw_goal, device_id="phone-1")

        self.assertEqual("navigation_only", graph.active_subgoal().external_impact)
        corrected = {
            item.source_id: item
            for item in planner.last_risk_audit.assessments
            if item.source_id
            in {"raw_goal", "goal.objective", "subgoals.target_state.objective"}
        }
        self.assertTrue(
            all(item.external_impact == "navigation_only" for item in corrected.values())
        )
        self.assertTrue(all(not item.risk_types for item in corrected.values()))

    def test_explicit_unsubmitted_input_ignores_model_risk_severity(self):
        raw_goal = (
            "让当前输入框内容为314159并保持未提交；"
            "不要搜索、提交、发送、保存或发布。"
        )
        payload = single_subgoal_payload(
            "当前输入框内容为314159且保持未提交",
            external_impact="external_state",
        )
        payload["goal"]["entities"]["input_text"] = "314159"
        payload["constraints"] = ["不要搜索、提交、发送、保存或发布。"]
        payload["subgoals"][0]["constraints"] = list(payload["constraints"])
        payload["risk_actions"] = [
            {
                "risk_id": "model_input_risk",
                "description": "模型认为输入可能改变数据",
                "external_effect": "未知外部效果",
                "risk_type": "data_mutation",
                "risk_level": "high",
                "subgoal_ids": ["target_state"],
                "confirmation_required": True,
            }
        ]
        payload["subgoals"][0]["risk_action_ids"] = ["model_input_risk"]

        graph = DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
            raw_goal,
            device_id="phone-1",
        )

        self.assertEqual((), graph.risk_actions)
        self.assertEqual("navigation_only", graph.active_subgoal().external_impact)

    def test_explicit_unsubmitted_editable_region_removes_model_data_risk(self):
        samples = (
            (
                "在文本区域中输入 codex，使其保留为未提交草稿",
                "codex",
            ),
            (
                "在编辑区域中输入 note，使其保留为未提交文字",
                "note",
            ),
        )
        for objective, input_text in samples:
            with self.subTest(objective=objective):
                raw_goal = (
                    objective
                    + "；不得搜索、提交、发送、保存或发布，也不得改变外部状态。"
                )
                payload = single_subgoal_payload(
                    objective,
                    external_impact="external_state",
                )
                payload["goal"]["entities"]["input_text"] = input_text
                payload["constraints"] = [
                    "不得搜索、提交、发送、保存或发布",
                    "不得改变外部状态",
                ]
                payload["subgoals"][0]["constraints"] = list(
                    payload["constraints"]
                )
                payload["subgoals"][0]["completion_conditions"] = [
                    f"文本区域中包含 {input_text}",
                ]
                payload["risk_actions"] = [
                    {
                        "risk_id": "model_input_risk",
                        "description": "模型认为输入可能改变数据",
                        "external_effect": "可能触发自动保存或网络请求",
                        "risk_type": "data_mutation",
                        "risk_level": "medium",
                        "subgoal_ids": ["target_state"],
                        "confirmation_required": True,
                    }
                ]
                payload["subgoals"][0]["risk_action_ids"] = [
                    "model_input_risk"
                ]

                graph = DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
                    raw_goal,
                    device_id="phone-1",
                )

                self.assertEqual((), graph.risk_actions)
                self.assertEqual(
                    "navigation_only",
                    graph.active_subgoal().external_impact,
                )

    def test_editable_input_carrier_adjective_is_not_a_data_mutation(self):
        samples = (
            (
                "当前浏览器顶部可编辑的地址输入区域内容为 codex",
                "codex",
                ["不得打开网址、搜索、提交、发送、保存或发布。"],
            ),
            (
                "The editable text field shows note",
                "note",
                ["Do not search, submit, send, save, or publish."],
            ),
        )
        for objective, input_text, constraints in samples:
            with self.subTest(objective=objective):
                payload = single_subgoal_payload(
                    objective,
                    external_impact="navigation_only",
                )
                payload["goal"]["objective"] = objective
                payload["goal"]["entities"]["input_text"] = input_text
                payload["constraints"] = constraints
                payload["subgoals"][0]["constraints"] = list(constraints)
                payload["subgoals"][0]["completion_conditions"] = [objective]

                graph = DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
                    objective + "；" + "；".join(constraints),
                    device_id="phone-1",
                )

                self.assertEqual((), graph.risk_actions)
                self.assertEqual(
                    "navigation_only",
                    graph.active_subgoal().external_impact,
                )

    def test_canonical_input_literal_binds_through_descriptive_modifiers(self):
        samples = (
            "当前浏览器顶部可编辑的地址输入区域内容为英文 codex",
            "当前文本框内容为 ASCII codex",
            "The editable text field shows the literal codex",
            '当前输入框内容为英文“codex”',
        )
        constraints = ["不得搜索、提交、发送、保存或发布。"]
        for objective in samples:
            with self.subTest(objective=objective):
                payload = single_subgoal_payload(
                    objective,
                    external_impact="navigation_only",
                )
                payload["goal"]["objective"] = objective
                payload["goal"]["entities"]["input_text"] = "codex"
                payload["constraints"] = list(constraints)
                payload["subgoals"][0]["constraints"] = list(constraints)
                payload["subgoals"][0]["completion_conditions"] = [objective]

                graph = DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
                    objective + "；" + "；".join(constraints),
                    device_id="phone-1",
                )

                self.assertEqual("navigation_only", graph.active_subgoal().external_impact)
                self.assertEqual("codex", graph.goal.entities["input_text"])

    def test_canonical_input_literal_must_bind_same_state_clause_exactly(self):
        samples = (
            "当前输入框内容为 codex2",
            "当前输入框内容为 codex.com",
            "当前输入框内容为 codec；目标标记 codex 可见",
            "当前输入框内容为 codec",
        )
        constraints = ["不得搜索、提交、发送、保存或发布。"]
        for objective in samples:
            with self.subTest(objective=objective):
                payload = single_subgoal_payload(
                    objective,
                    external_impact="navigation_only",
                )
                payload["goal"]["objective"] = objective
                payload["goal"]["entities"]["input_text"] = "codex"
                payload["constraints"] = list(constraints)
                payload["subgoals"][0]["constraints"] = list(constraints)
                payload["subgoals"][0]["completion_conditions"] = [objective]

                with self.assertRaisesRegex(
                    TaskGraphError,
                    "子目标输入状态未绑定 canonical input_text",
                ):
                    DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
                        objective + "；" + "；".join(constraints),
                        device_id="phone-1",
                    )

    def test_editable_input_carrier_never_hides_saved_result(self):
        objective = "可编辑的地址输入区域内容为 codex，且草稿已保存"
        payload = single_subgoal_payload(
            objective,
            external_impact="external_state",
        )
        payload["goal"]["entities"]["input_text"] = "codex"
        payload["constraints"] = ["不得发送、提交或发布。"]
        payload["subgoals"][0]["constraints"] = list(payload["constraints"])
        payload["subgoals"][0]["completion_conditions"] = [objective]
        payload["risk_actions"] = [
            {
                "risk_id": "saved_result",
                "description": "草稿已保存",
                "external_effect": "草稿已保存到外部数据",
                "risk_type": "data_mutation",
                "risk_level": "medium",
                "subgoal_ids": ["target_state"],
                "confirmation_required": True,
            }
        ]
        payload["subgoals"][0]["risk_action_ids"] = ["saved_result"]
        payload["status"] = "awaiting_confirmation"

        graph = DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
            objective + "；不得发送、提交或发布。",
            device_id="phone-1",
        )

        self.assertEqual(("saved_result",), graph.active_subgoal().risk_action_ids)
        self.assertEqual("external_state", graph.active_subgoal().external_impact)

    def test_initial_plan_removes_only_purely_forbidden_effect_risks(self):
        payload = purely_forbidden_draft_payload()
        raw_goal = payload["goal"]["objective"]
        normalized_payload = copy.deepcopy(payload)
        normalized_payload["risk_actions"] = []
        for index, impact in enumerate(
            ("navigation_only", "navigation_only", "read_only")
        ):
            normalized_payload["subgoals"][index]["risk_action_ids"] = []
            normalized_payload["subgoals"][index]["external_impact"] = impact

        graph = DeepSeekTaskGraphPlanner(
            FakeProvider(
                copy.deepcopy(payload),
                copy.deepcopy(payload),
                audit_payloads=[audit_payload_for_graph(normalized_payload)],
            )
        ).plan(raw_goal, device_id="phone-1")

        self.assertEqual((), graph.risk_actions)
        self.assertEqual(
            ["navigation_only", "navigation_only", "read_only"],
            [item.external_impact for item in graph.subgoals],
        )
        self.assertTrue(all(not item.risk_action_ids for item in graph.subgoals))

    def test_initial_plan_accepts_purely_negative_risk_control_state(self):
        payload = purely_forbidden_draft_payload()
        payload["goal"]["target_apps"] = [
            {"app_id": "current_foreground", "app_name": "当前前台应用"}
        ]
        payload["constraints"] = [
            "不得发送、提交、删除、转发、发布该草稿",
            "不得选择其他联系人",
            "不得产生任何账号及外部影响",
        ]
        payload["completion_conditions"] = [
            {
                "condition_id": "final_state_visible",
                "description": (
                    "指定聊天页面在前台可见，底部唯一消息输入框中包含未发送的英文 "
                    "codex 草稿，且发送按钮未被触发，codex 仍可见"
                ),
                "evidence_required": [
                    "指定聊天页面可见",
                    "底部唯一消息输入框内容为 codex",
                    "发送按钮未被触发",
                    "codex 文字可见",
                ],
                "satisfied": False,
                "evidence": [],
            }
        ]
        payload["risk_actions"] = [payload["risk_actions"][0]]
        payload["risk_actions"][0]["subgoal_ids"] = [
            "open_chat",
            "enter_draft",
            "verify_before_send",
        ]
        for subgoal in payload["subgoals"]:
            subgoal["risk_action_ids"] = ["send_message"]
            subgoal["external_impact"] = (
                "read_only"
                if subgoal["subgoal_id"] == "verify_before_send"
                else "external_state"
            )
        payload["subgoals"][1]["constraints"] = [payload["constraints"][0]]
        payload["subgoals"][2]["objective"] = (
            "确认发送按钮未被触发且 codex 仍可见"
        )
        payload["subgoals"][2]["constraints"] = [payload["constraints"][0]]
        payload["subgoals"][2]["completion_conditions"] = [
            "发送按钮未被触发",
            "codex 文字可见",
        ]
        normalized_payload = copy.deepcopy(payload)
        normalized_payload["risk_actions"] = []
        for index, impact in enumerate(
            ("navigation_only", "navigation_only", "read_only")
        ):
            normalized_payload["subgoals"][index]["risk_action_ids"] = []
            normalized_payload["subgoals"][index]["external_impact"] = impact

        graph = DeepSeekTaskGraphPlanner(
            FakeProvider(
                copy.deepcopy(payload),
                copy.deepcopy(payload),
                audit_payloads=[audit_payload_for_graph(normalized_payload)],
            )
        ).plan(payload["goal"]["objective"], device_id="phone-1")

        self.assertEqual((), graph.risk_actions)
        self.assertEqual(
            ["navigation_only", "navigation_only", "read_only"],
            [item.external_impact for item in graph.subgoals],
        )

    def test_local_unsent_input_chain_removes_negated_risk_through_keyboard_dismissal(self):
        payload = local_input_preparation_payload()
        payload["goal"]["objective"] = (
            "在当前页面唯一已聚焦的空白输入框中保留未发送的英文 codex 草稿，"
            "然后收起软键盘；最终仍停留当前聊天页面并能看见 codex，"
            "且不得发送、提交、保存、发布、转发、选择联系人或产生任何外部影响。"
        )
        payload["constraints"] = [
            "不得发送、提交、保存、发布、转发该草稿",
            "不得选择联系人",
            "不得产生任何账号及外部影响",
        ]
        payload["completion_conditions"] = [
            {
                "condition_id": "final_state",
                "description": (
                    "当前聊天页面可见，且输入框中包含未发送的英文 codex "
                    "草稿，软键盘已收起。"
                ),
                "evidence_required": [
                    "当前聊天页面在前台可见",
                    "输入框内容为 codex",
                    "软键盘未显示",
                ],
                "satisfied": False,
                "evidence": [],
            }
        ]
        payload["risk_actions"] = [
            {
                "risk_id": "no_external_effect",
                "description": (
                    "确保不发送、提交、保存、发布、转发草稿，"
                    "不选择联系人，不产生任何账号或外部影响。"
                ),
                "external_effect": "无外部影响",
                "risk_type": "unknown_external_effect",
                "risk_level": "low",
                "subgoal_ids": [
                    "ensure_input",
                    "ensure_keyboard",
                    "ensure_stay",
                ],
                "confirmation_required": True,
            }
        ]
        payload["subgoals"] = [
            {
                "subgoal_id": "ensure_input",
                "objective": "当前唯一已聚焦的空白输入框内容为 codex，且未发送。",
                "status": "active",
                "depends_on": [],
                "constraints": list(payload["constraints"]),
                "completion_conditions": [
                    "输入框内容为 codex",
                    "输入框仍处于未发送状态",
                ],
                "completion_evidence": [],
                "risk_action_ids": ["no_external_effect"],
                "external_impact": "navigation_only",
            },
            {
                "subgoal_id": "ensure_keyboard",
                "objective": "软键盘已收起。",
                "status": "pending",
                "depends_on": ["ensure_input"],
                "constraints": list(payload["constraints"]),
                "completion_conditions": ["软键盘未显示"],
                "completion_evidence": [],
                "risk_action_ids": ["no_external_effect"],
                "external_impact": "navigation_only",
            },
            {
                "subgoal_id": "ensure_stay",
                "objective": "当前聊天页面在前台可见，且能看到 codex 草稿。",
                "status": "pending",
                "depends_on": ["ensure_keyboard"],
                "constraints": list(payload["constraints"]),
                "completion_conditions": [
                    "当前聊天页面在前台可见",
                    "输入框内容为 codex",
                ],
                "completion_evidence": [],
                "risk_action_ids": ["no_external_effect"],
                "external_impact": "navigation_only",
            },
        ]
        payload["active_subgoal_id"] = "ensure_input"
        normalized_payload = copy.deepcopy(payload)
        normalized_payload["risk_actions"] = []
        for subgoal in normalized_payload["subgoals"]:
            subgoal["risk_action_ids"] = []

        graph = DeepSeekTaskGraphPlanner(
            FakeProvider(
                copy.deepcopy(payload),
                audit_payloads=[audit_payload_for_graph(normalized_payload)],
            )
        ).plan(payload["goal"]["objective"], device_id="phone-1")

        self.assertEqual((), graph.risk_actions)
        self.assertTrue(all(not item.risk_action_ids for item in graph.subgoals))
        self.assertTrue(
            all(item.external_impact == "navigation_only" for item in graph.subgoals)
        )

    def test_negated_button_state_is_evidence_not_a_low_level_instruction(self):
        payload = purely_forbidden_draft_payload()
        payload["goal"]["objective"] = (
            "在底部唯一消息输入框中保留未发送的英文 codex 草稿，"
            "且不发送、不提交、不删除、不转发、不发布、不选择其他联系人。"
        )
        payload["risk_actions"][2]["description"] = "切换到其他联系人"
        payload["risk_actions"][2]["external_effect"] = "改变当前聊天对象"
        payload["subgoals"][2]["completion_conditions"][0] = "发送按钮未被点击"
        payload["completion_conditions"][2]["evidence_required"] = [
            "发送按钮未被激活或点击",
            "codex 文字可见",
        ]
        normalized_payload = copy.deepcopy(payload)
        normalized_payload["risk_actions"] = []
        for index, impact in enumerate(
            ("navigation_only", "navigation_only", "read_only")
        ):
            normalized_payload["subgoals"][index]["risk_action_ids"] = []
            normalized_payload["subgoals"][index]["external_impact"] = impact

        graph = DeepSeekTaskGraphPlanner(
            FakeProvider(
                copy.deepcopy(payload),
                audit_payloads=[audit_payload_for_graph(normalized_payload)],
            )
        ).plan(payload["goal"]["objective"], device_id="phone-1")

        self.assertEqual((), graph.risk_actions)
        self.assertEqual("ready", graph.status)
        self.assertEqual("open_chat", graph.active_subgoal_id)

    def test_positive_or_misplaced_click_text_is_not_completion_evidence(self):
        for evidence in ("发送按钮已点击", "不得点击发送按钮"):
            with self.subTest(evidence=evidence):
                payload = purely_forbidden_draft_payload()
                payload["completion_conditions"][2]["evidence_required"] = [evidence]
                with self.assertRaisesRegex(TaskGraphError, "低层动作表达"):
                    DeepSeekTaskGraphPlanner(
                        FakeProvider(
                            copy.deepcopy(payload),
                            copy.deepcopy(payload),
                            copy.deepcopy(payload),
                        )
                    ).plan(payload["goal"]["objective"], device_id="phone-1")

    def test_positive_or_misplaced_click_text_is_not_subgoal_completion(self):
        for completion in ("发送按钮已点击", "不得点击发送按钮"):
            with self.subTest(completion=completion):
                payload = purely_forbidden_draft_payload()
                payload["subgoals"][2]["completion_conditions"] = [completion]
                with self.assertRaisesRegex(TaskGraphError, "低层动作表达"):
                    DeepSeekTaskGraphPlanner(
                        FakeProvider(
                            copy.deepcopy(payload),
                            copy.deepcopy(payload),
                            copy.deepcopy(payload),
                        )
                    ).plan(payload["goal"]["objective"], device_id="phone-1")

    def test_input_preparation_state_removes_shared_purely_forbidden_risk(self):
        payload = local_input_preparation_payload()
        normalized_payload = copy.deepcopy(payload)
        normalized_payload["risk_actions"] = []
        for subgoal in normalized_payload["subgoals"]:
            subgoal["risk_action_ids"] = []

        graph = DeepSeekTaskGraphPlanner(
            FakeProvider(
                copy.deepcopy(payload),
                audit_payloads=[audit_payload_for_graph(normalized_payload)],
            )
        ).plan(payload["goal"]["objective"], device_id="phone-1")

        self.assertEqual((), graph.risk_actions)
        self.assertEqual(
            ["navigation_only", "navigation_only", "navigation_only"],
            [item.external_impact for item in graph.subgoals],
        )
        self.assertTrue(all(not item.risk_action_ids for item in graph.subgoals))

    def test_input_preparation_state_is_cross_app_and_bilingual(self):
        payload = local_input_preparation_payload()
        payload["goal"] = {
            "objective": (
                "The local notes page is open and the only text field shows note as "
                "an unsubmitted draft; do not save, submit, publish, or send it."
            ),
            "target_apps": [{"app_id": "notes", "app_name": "Notes"}],
            "entities": {"target_ui_label": "local notes", "input_text": "note"},
        }
        payload["constraints"] = ["Do not save, submit, publish, or send the draft."]
        payload["risk_actions"] = [
            {
                "risk_id": "save_draft",
                "description": "save the draft",
                "external_effect": "save draft data",
                "risk_type": "data_mutation",
                "risk_level": "high",
                "subgoal_ids": ["prepare_input", "type_text"],
                "confirmation_required": True,
            }
        ]
        payload["subgoals"][0]["objective"] = "Open the local notes page"
        payload["subgoals"][0]["completion_conditions"] = [
            "The local notes page is visible"
        ]
        payload["subgoals"][1]["objective"] = (
            "The only text field is visible and focused"
        )
        payload["subgoals"][1]["constraints"] = list(payload["constraints"])
        payload["subgoals"][1]["completion_conditions"] = [
            "The only text field is visible and focused"
        ]
        payload["subgoals"][2]["objective"] = (
            "The text field shows note and remains unsubmitted"
        )
        payload["subgoals"][2]["constraints"] = list(payload["constraints"])
        payload["subgoals"][2]["completion_conditions"] = [
            "The text field shows note and remains unsubmitted"
        ]
        payload["subgoals"][2]["risk_action_ids"] = ["save_draft"]
        payload["completion_conditions"][0]["description"] = (
            "The local notes page is visible and the text field shows note as an "
            "unsubmitted draft"
        )
        payload["completion_conditions"][0]["evidence_required"] = [
            "The local notes page is visible",
            "The text field shows note",
            "The draft remains unsubmitted",
        ]
        normalized_payload = copy.deepcopy(payload)
        normalized_payload["risk_actions"] = []
        for subgoal in normalized_payload["subgoals"]:
            subgoal["risk_action_ids"] = []

        graph = DeepSeekTaskGraphPlanner(
            FakeProvider(
                copy.deepcopy(payload),
                audit_payloads=[audit_payload_for_graph(normalized_payload)],
            )
        ).plan(payload["goal"]["objective"], device_id="phone-1")

        self.assertEqual((), graph.risk_actions)
        self.assertTrue(all(not item.risk_action_ids for item in graph.subgoals))

    def test_input_preparation_state_must_share_canonical_dependency_chain(self):
        payload = local_input_preparation_payload()
        payload["subgoals"][1]["depends_on"] = []
        payload["subgoals"][2]["depends_on"] = ["navigate"]

        with self.assertRaises(TaskGraphError):
            DeepSeekTaskGraphPlanner(
                FakeProvider(copy.deepcopy(payload), copy.deepcopy(payload))
            ).plan(payload["goal"]["objective"], device_id="phone-1")

    def test_input_preparation_state_never_hides_positive_saved_result(self):
        payload = local_input_preparation_payload()
        payload["goal"]["objective"] = (
            "进入指定页面，让输入框可见并保存 codex 草稿。"
        )
        payload["constraints"] = []
        payload["completion_conditions"][0]["description"] = "codex 草稿已保存"
        payload["completion_conditions"][0]["evidence_required"] = ["已保存标记"]
        payload["risk_actions"][0].update(
            {
                "risk_id": "save_draft",
                "description": "保存 codex 草稿",
                "external_effect": "草稿数据已保存",
                "risk_type": "data_mutation",
                "subgoal_ids": ["prepare_input"],
            }
        )
        payload["subgoals"] = [payload["subgoals"][1]]
        payload["subgoals"][0].update(
            {
                "status": "active",
                "depends_on": [],
                "objective": "输入框可见且 codex 草稿已保存",
                "constraints": [],
                "completion_conditions": ["codex 草稿已保存"],
                "risk_action_ids": ["save_draft"],
                "external_impact": "external_state",
            }
        )
        payload["active_subgoal_id"] = "prepare_input"
        payload["status"] = "awaiting_confirmation"

        graph = DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
            payload["goal"]["objective"], device_id="phone-1"
        )

        self.assertEqual(("save_draft",), graph.active_subgoal().risk_action_ids)
        self.assertEqual("external_state", graph.active_subgoal().external_impact)

    def test_input_preparation_state_requires_canonical_input_text(self):
        payload = local_input_preparation_payload()
        payload["goal"]["entities"].pop("input_text")

        with self.assertRaises(TaskGraphError):
            DeepSeekTaskGraphPlanner(
                FakeProvider(copy.deepcopy(payload), copy.deepcopy(payload))
            ).plan(payload["goal"]["objective"], device_id="phone-1")

    def test_purely_forbidden_effect_normalization_is_cross_app_and_bilingual(self):
        payload = single_subgoal_payload(
            "The editable text field shows note and remains an unsubmitted draft",
            external_impact="external_state",
        )
        payload["goal"]["target_apps"] = [
            {"app_id": "notes_app", "app_name": "Notes"}
        ]
        payload["goal"]["entities"]["input_text"] = "note"
        payload["constraints"] = ["Do not submit or delete the draft."]
        payload["subgoals"][0]["constraints"] = list(payload["constraints"])
        payload["subgoals"][0]["completion_conditions"] = [
            "The editable text field shows note and remains unsubmitted"
        ]
        payload["risk_actions"] = [
            {
                "risk_id": "submit_draft",
                "description": "submit the draft",
                "external_effect": "submit draft content to an external service",
                "risk_type": "data_mutation",
                "risk_level": "high",
                "subgoal_ids": ["target_state"],
                "confirmation_required": True,
            },
            {
                "risk_id": "delete_draft",
                "description": "delete the draft",
                "external_effect": "delete draft content",
                "risk_type": "data_deletion",
                "risk_level": "medium",
                "subgoal_ids": ["target_state"],
                "confirmation_required": True,
            },
        ]
        payload["subgoals"][0]["risk_action_ids"] = [
            "submit_draft",
            "delete_draft",
        ]

        graph = DeepSeekTaskGraphPlanner(
            FakeProvider(copy.deepcopy(payload), copy.deepcopy(payload))
        ).plan(
            "Keep note in the editable text field as an unsubmitted draft; "
            "do not submit or delete the draft.",
            device_id="phone-1",
        )

        self.assertEqual((), graph.risk_actions)
        self.assertEqual("navigation_only", graph.active_subgoal().external_impact)

    def test_purely_forbidden_effect_normalization_keeps_positive_send_result(self):
        payload = purely_forbidden_draft_payload()
        payload["goal"]["objective"] = (
            "在唯一消息输入框中保留 codex，然后发送 codex 消息给指定联系人；不得删除草稿。"
        )
        payload["constraints"] = ["不得删除草稿。"]
        payload["completion_conditions"].append(
            {
                "condition_id": "message_sent",
                "description": "发送 codex 消息给指定联系人已经完成",
                "evidence_required": ["聊天记录显示发送 codex 消息给指定联系人已经完成"],
                "satisfied": False,
                "evidence": [],
            }
        )
        payload["completion_conditions"] = [payload["completion_conditions"][-1]]
        payload["risk_actions"] = [payload["risk_actions"][0]]
        payload["risk_actions"][0]["subgoal_ids"] = ["send_draft"]
        payload["subgoals"] = [
            {
                "subgoal_id": "send_draft",
                "objective": "发送 codex 消息给指定联系人已经完成",
                "status": "active",
                "depends_on": [],
                "constraints": ["不得删除草稿"],
                "completion_conditions": ["发送 codex 消息给指定联系人已经完成"],
                "completion_evidence": [],
                "risk_action_ids": ["send_message"],
                "external_impact": "external_state",
            }
        ]
        payload["active_subgoal_id"] = "send_draft"
        payload["status"] = "awaiting_confirmation"

        graph = DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
            payload["goal"]["objective"],
            device_id="phone-1",
        )

        self.assertEqual(("send_message",), graph.active_subgoal().risk_action_ids)
        self.assertEqual("awaiting_confirmation", graph.status)

    def test_purely_forbidden_effect_normalization_requires_direct_prohibition(self):
        payload = purely_forbidden_draft_payload()
        payload["goal"]["objective"] = (
            "在唯一消息输入框中保留 codex，不要忘记发送 codex 消息给联系人。"
        )
        payload["constraints"] = ["不要忘记发送 codex 消息给联系人。"]
        payload["completion_conditions"] = [payload["completion_conditions"][1]]
        payload["risk_actions"] = [payload["risk_actions"][0]]
        payload["risk_actions"][0]["subgoal_ids"] = ["enter_draft"]
        payload["subgoals"] = [payload["subgoals"][1]]
        payload["subgoals"][0]["subgoal_id"] = "enter_draft"
        payload["subgoals"][0]["status"] = "active"
        payload["subgoals"][0]["depends_on"] = []
        payload["subgoals"][0]["constraints"] = list(payload["constraints"])
        payload["subgoals"][0]["risk_action_ids"] = ["send_message"]
        payload["active_subgoal_id"] = "enter_draft"
        payload["status"] = "awaiting_confirmation"

        graph = DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
            payload["goal"]["objective"],
            device_id="phone-1",
        )

        self.assertEqual(("send_message",), graph.active_subgoal().risk_action_ids)
        self.assertEqual("external_state", graph.active_subgoal().external_impact)

    def test_read_only_risk_control_state_never_hides_completed_send_effect(self):
        payload = single_subgoal_payload(
            "发送按钮可见，且发送 codex 消息给指定联系人已经完成",
            external_impact="read_only",
        )
        payload["constraints"] = ["不得点击发送按钮"]
        payload["subgoals"][0]["constraints"] = list(payload["constraints"])
        payload["subgoals"][0]["completion_conditions"] = [
            "发送按钮可见",
            "发送 codex 消息给指定联系人已经完成",
        ]

        with self.assertRaises(TaskGraphError):
            DeepSeekTaskGraphPlanner(
                FakeProvider(copy.deepcopy(payload), copy.deepcopy(payload))
            ).plan(
                "确认发送按钮可见且消息已发送；不得点击发送按钮。",
                device_id="phone-1",
            )

    def test_unsubmitted_input_workflow_removes_one_shared_false_risk(self):
        raw_goal = (
            "选择能留下未提交文字的模式，让唯一文本区域显示英文 codex，"
            "并停在能核对文字的页面；不得搜索、发送、提交、保存或发布。"
        )
        for description in (
            "仅修改未提交的临时文本，不触发任何外部效果。",
            "只涉及本地临时草稿且没有外部影响。",
        ):
            with self.subTest(description=description):
                payload = single_subgoal_payload(
                    "能留下未提交文字的模式被选中，且该模式页面可见。",
                    external_impact="external_state",
                )
                payload["goal"]["entities"] = {
                    "target_ui_label": "能留下未提交文字的模式",
                    "input_text": "codex",
                }
                payload["constraints"] = [
                    "不得搜索、发送、提交、保存或发布。",
                ]
                payload["risk_actions"] = [
                    {
                        "risk_id": "no_external_effect",
                        "description": description,
                        "external_effect": "无外部影响",
                        "risk_type": "unknown_external_effect",
                        "risk_level": "low",
                        "subgoal_ids": [
                            "select_mode",
                            "enter_text",
                            "verify_text",
                        ],
                        "confirmation_required": True,
                    }
                ]
                payload["subgoals"] = [
                    {
                        "subgoal_id": "select_mode",
                        "objective": "能留下未提交文字的模式被选中，且该模式页面可见。",
                        "status": "active",
                        "depends_on": [],
                        "constraints": list(payload["constraints"]),
                        "completion_conditions": ["目标模式页面可见"],
                        "completion_evidence": [],
                        "risk_action_ids": ["no_external_effect"],
                        "external_impact": "external_state",
                    },
                    {
                        "subgoal_id": "enter_text",
                        "objective": "该模式中的唯一文本区域内容为英文 codex。",
                        "status": "pending",
                        "depends_on": ["select_mode"],
                        "constraints": list(payload["constraints"]),
                        "completion_conditions": ["唯一文本区域内容为英文 codex"],
                        "completion_evidence": [],
                        "risk_action_ids": ["no_external_effect"],
                        "external_impact": "external_state",
                    },
                    {
                        "subgoal_id": "verify_text",
                        "objective": "停在能核对这段文字的页面，且唯一文本区域显示英文 codex。",
                        "status": "pending",
                        "depends_on": ["enter_text"],
                        "constraints": list(payload["constraints"]),
                        "completion_conditions": ["当前页面可核对文字"],
                        "completion_evidence": [],
                        "risk_action_ids": ["no_external_effect"],
                        "external_impact": "read_only",
                    },
                ]
                payload["active_subgoal_id"] = "select_mode"

                graph = DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
                    raw_goal,
                    device_id="phone-1",
                )

                self.assertEqual((), graph.risk_actions)
                self.assertEqual(
                    ["navigation_only", "navigation_only", "read_only"],
                    [item.external_impact for item in graph.subgoals],
                )
                self.assertTrue(all(not item.risk_action_ids for item in graph.subgoals))

    def test_unsubmitted_input_workflow_keeps_shared_risk_off_dependency_chain(self):
        payload = single_subgoal_payload(
            "当前文本区域内容为 codex 且保持未提交",
            external_impact="external_state",
        )
        payload["goal"]["entities"]["input_text"] = "codex"
        payload["constraints"] = ["不得搜索、发送、提交、保存或发布。"]
        payload["risk_actions"] = [
            {
                "risk_id": "shared_risk",
                "description": "仅修改未提交的临时文本，不触发任何外部效果。",
                "external_effect": "无外部影响",
                "risk_type": "unknown_external_effect",
                "risk_level": "low",
                "subgoal_ids": ["target_state", "unrelated_review"],
                "confirmation_required": True,
            }
        ]
        payload["subgoals"][0]["constraints"] = list(payload["constraints"])
        payload["subgoals"][0]["risk_action_ids"] = ["shared_risk"]
        payload["subgoals"].append(
            {
                "subgoal_id": "unrelated_review",
                "objective": "另一个独立区域可见",
                "status": "pending",
                "depends_on": [],
                "constraints": [],
                "completion_conditions": ["另一个独立区域可见"],
                "completion_evidence": [],
                "risk_action_ids": ["shared_risk"],
                "external_impact": "external_state",
            }
        )

        graph = DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
            "让当前文本区域显示 codex 并保持未提交；不得发送、提交、保存或发布。",
            device_id="phone-1",
        )

        self.assertEqual(
            ("shared_risk",),
            tuple(item.risk_id for item in graph.risk_actions),
        )
        self.assertTrue(
            all("shared_risk" in item.risk_action_ids for item in graph.subgoals)
        )

    def test_unsubmitted_input_workflow_never_removes_shared_send_effect(self):
        payload = single_subgoal_payload(
            "当前文本区域内容为 codex 且保持未提交",
            external_impact="external_state",
        )
        payload["goal"]["entities"]["input_text"] = "codex"
        payload["constraints"] = ["不得提交、保存或发布。"]
        payload["risk_actions"] = [
            {
                "risk_id": "send_effect",
                "description": "输入完成后发送文字",
                "external_effect": "消息将发送给外部联系人",
                "risk_type": "message_or_communication",
                "risk_level": "low",
                "subgoal_ids": ["target_state", "send_text"],
                "confirmation_required": True,
            }
        ]
        payload["subgoals"][0]["constraints"] = list(payload["constraints"])
        payload["subgoals"][0]["risk_action_ids"] = ["send_effect"]
        payload["subgoals"].append(
            {
                "subgoal_id": "send_text",
                "objective": "文字已发送给外部联系人",
                "status": "pending",
                "depends_on": ["target_state"],
                "constraints": [],
                "completion_conditions": ["外部联系人收到文字"],
                "completion_evidence": [],
                "risk_action_ids": ["send_effect"],
                "external_impact": "external_state",
            }
        )

        with self.assertRaises(TaskGraphError):
            DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
                "让当前文本区域显示 codex 并保持未提交，然后发送文字。",
                device_id="phone-1",
            )

    def test_editable_region_does_not_downgrade_visible_saved_draft(self):
        objective = "文本区域中包含 codex，且草稿已保存"
        payload = single_subgoal_payload(
            objective,
            external_impact="external_state",
        )
        payload["goal"]["entities"]["input_text"] = "codex"
        payload["constraints"] = ["不得发送或发布"]
        payload["subgoals"][0]["constraints"] = list(payload["constraints"])
        payload["subgoals"][0]["completion_conditions"] = [
            "文本区域中包含 codex",
            "草稿已保存",
        ]
        payload["risk_actions"] = [
            {
                "risk_id": "save_effect",
                "description": "保存草稿",
                "external_effect": "草稿已保存到外部数据",
                "risk_type": "data_mutation",
                "risk_level": "medium",
                "subgoal_ids": ["target_state"],
                "confirmation_required": True,
            }
        ]
        payload["subgoals"][0]["risk_action_ids"] = ["save_effect"]
        payload["status"] = "awaiting_confirmation"

        graph = DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
            objective + "；不得发送或发布。",
            device_id="phone-1",
        )

        self.assertEqual("external_state", graph.active_subgoal().external_impact)
        self.assertEqual(("save_effect",), graph.active_subgoal().risk_action_ids)

    def test_explicit_unsubmitted_input_never_downgrades_real_save_effect(self):
        objective = (
            "当前输入框内容为314159并保持未提交；草稿已保存；"
            "不要搜索、提交、发送或发布。"
        )
        payload = single_subgoal_payload(
            "当前输入框内容为314159并保持未提交",
            external_impact="external_state",
        )
        payload["goal"]["entities"]["input_text"] = "314159"
        payload["subgoals"][0]["completion_conditions"] = [
            "输入框内容为314159",
            "草稿已保存",
        ]
        payload["risk_actions"] = [
            {
                "risk_id": "save_effect",
                "description": "保存草稿",
                "external_effect": "草稿已保存到外部数据",
                "risk_type": "data_mutation",
                "risk_level": "high",
                "subgoal_ids": ["target_state"],
                "confirmation_required": True,
            }
        ]
        payload["subgoals"][0]["risk_action_ids"] = ["save_effect"]
        payload["status"] = "awaiting_confirmation"

        graph = DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
            objective,
            device_id="phone-1",
        )

        self.assertEqual("external_state", graph.active_subgoal().external_impact)
        self.assertEqual(("save_effect",), graph.active_subgoal().risk_action_ids)

    def test_generic_modify_word_is_safe_only_inside_explicit_unsubmitted_input(self):
        raw_goal = (
            "把当前输入框中的现有文字修改为 Agent123，"
            "不要搜索、提交、发送、保存或发布。"
        )
        objective = (
            "当前输入框文字修改为 Agent123，且未发生搜索、提交、发送、保存或发布。"
        )
        payload = single_subgoal_payload(
            objective,
            external_impact="navigation_only",
        )
        payload["goal"]["entities"]["input_text"] = "Agent123"
        payload["constraints"] = ["不要搜索、提交、发送、保存或发布。"]
        payload["subgoals"][0]["objective"] = "当前输入框文字修改为 Agent123"
        payload["subgoals"][0]["constraints"] = list(payload["constraints"])
        payload["subgoals"][0]["completion_conditions"] = [
            "当前输入框文字显示为 Agent123"
        ]
        audit = audit_payload_for_graph(
            payload,
            overrides={
                "raw_goal": {
                    "external_impact": "external_state",
                    "risk_types": ["unknown_external_effect"],
                },
                "goal.objective": {
                    "external_impact": "external_state",
                    "risk_types": ["unknown_external_effect"],
                },
                "subgoals.target_state.objective": {
                    "external_impact": "external_state",
                    "risk_types": ["unknown_external_effect"],
                },
            },
        )

        graph = DeepSeekTaskGraphPlanner(
            FakeProvider(payload, audit_payloads=[audit])
        ).plan(raw_goal, device_id="phone-1")

        self.assertEqual("navigation_only", graph.active_subgoal().external_impact)
        self.assertEqual((), graph.risk_actions)

    def test_generic_modify_word_does_not_hide_concrete_external_result(self):
        objective = (
            "当前输入框文字修改为 Agent123，内容已保存，"
            "不要搜索、提交、发送或发布。"
        )
        payload = single_subgoal_payload(
            objective,
            external_impact="navigation_only",
        )
        payload["goal"]["entities"]["input_text"] = "Agent123"
        payload["subgoals"][0]["objective"] = objective
        payload["subgoals"][0]["constraints"] = [
            "不要搜索、提交、发送或发布。"
        ]

        with self.assertRaisesRegex(TaskGraphError, "外部状态变化但未声明"):
            DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
                objective,
                device_id="phone-1",
            )

    def test_no_trigger_coordinated_effects_are_all_treated_as_negated(self):
        text = "没有触发搜索、提交、发送、保存或发布等操作"

        self.assertEqual(frozenset(), _infer_external_risk_types(text))

    def test_unsubmitted_input_exception_never_hides_send_or_save_effect(self):
        for effect in ("消息已发送给联系人", "草稿已保存", "内容已发布"):
            with self.subTest(effect=effect):
                objective = (
                    "顶部搜索输入框中的文字为 Agent123，不触发搜索；" + effect
                )
                payload = single_subgoal_payload(
                    objective,
                    external_impact="navigation_only",
                )
                payload["goal"]["entities"]["input_text"] = "Agent123"

                with self.assertRaisesRegex(TaskGraphError, "外部状态变化但未声明"):
                    DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
                        objective,
                        device_id="phone-1",
                    )

    def test_external_result_state_words_are_detected_as_data_mutation(self):
        for result_state in (
            "草稿已保存",
            "表单已提交",
            "文件已上传",
            "记录已修改",
        ):
            with self.subTest(result_state=result_state):
                payload = single_subgoal_payload(
                    result_state,
                    external_impact="navigation_only",
                )
                with self.assertRaisesRegex(TaskGraphError, "外部状态变化但未声明"):
                    DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
                        result_state,
                        device_id="phone-1",
                    )

    def test_semantic_risk_prompt_documents_unsubmitted_input_boundary(self):
        objective = "查看页面"
        payload = single_subgoal_payload(objective, external_impact="read_only")
        provider = FakeProvider(payload)

        DeepSeekTaskGraphPlanner(provider).plan(objective, device_id="phone-1")

        audit_prompts = [
            call[0]["content"]
            for call in provider.messages
            if "semantic-risk-audit-v1" in call[0]["content"]
        ]
        self.assertEqual(1, len(audit_prompts))
        self.assertIn("未提交临时文字", audit_prompts[0])
        self.assertIn("输入并搜索/发送/保存", audit_prompts[0])
        self.assertIn("本地只读页面的跨页浏览闭环", audit_prompts[0])
        self.assertIn("翻进", audit_prompts[0])

    def test_local_navigation_exception_never_hides_explicit_external_effect(self):
        objective = "新建空白标签页后登录当前账号"
        payload = single_subgoal_payload(
            objective,
            external_impact="navigation_only",
        )

        with self.assertRaisesRegex(TaskGraphError, "外部状态变化但未声明"):
            DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
                objective,
                device_id="phone-1",
            )

    def test_safe_negated_goal_replans_once_after_mixed_false_positive_audit(self):
        objective = "打开浏览器首页，仅查看，不搜索、不登录"
        first_payload = single_subgoal_payload(
            objective,
            external_impact="navigation_only",
        )
        second_payload = single_subgoal_payload(
            objective,
            external_impact="navigation_only",
        )
        false_positive = audit_payload_for_graph(
            first_payload,
            overrides={
                "raw_goal": {
                    "external_impact": "external_state",
                    "risk_types": [
                        "account_or_permission_change",
                        "unknown_external_effect",
                    ],
                },
                "goal.objective": {
                    "external_impact": "external_state",
                    "risk_types": [
                        "account_or_permission_change",
                        "unknown_external_effect",
                    ],
                },
            },
        )
        safe_audit = audit_payload_for_graph(second_payload)
        provider = FakeProvider(
            first_payload,
            second_payload,
            audit_payloads=[false_positive, safe_audit],
        )
        planner = DeepSeekTaskGraphPlanner(provider)

        graph = planner.plan(objective, device_id="phone-1")

        self.assertEqual(graph.status, "ready")
        self.assertEqual(graph.active_subgoal().external_impact, "navigation_only")
        self.assertEqual(planner.risk_audit_call_count, 2)
        graph_prompts = [
            call[0]["content"]
            for call in provider.messages
            if "semantic-risk-audit-v1" not in call[0]["content"]
        ]
        self.assertEqual(len(graph_prompts), 2)
        self.assertIn("独立重新规划", graph_prompts[1])

    def test_safe_negated_goal_stops_when_second_audit_still_conflicts(self):
        objective = "打开浏览器首页，仅查看，不搜索、不登录"
        first_payload = single_subgoal_payload(
            objective,
            external_impact="navigation_only",
        )
        second_payload = single_subgoal_payload(
            objective,
            external_impact="navigation_only",
        )
        false_positive = audit_payload_for_graph(
            first_payload,
            overrides={
                "raw_goal": {
                    "external_impact": "external_state",
                    "risk_types": [
                        "account_or_permission_change",
                        "unknown_external_effect",
                    ],
                },
            },
        )
        provider = FakeProvider(
            first_payload,
            second_payload,
            audit_payloads=[false_positive, false_positive],
        )
        planner = DeepSeekTaskGraphPlanner(provider)

        with self.assertRaisesRegex(TaskGraphError, "全局目标包含外部状态"):
            planner.plan(objective, device_id="phone-1")

        self.assertEqual(planner.risk_audit_call_count, 2)

    def test_semantic_audit_can_use_an_independent_provider(self):
        payload = single_subgoal_payload("查看资料", external_impact="read_only")
        graph_provider = FakeProvider(payload)
        audit_provider = FakeProvider(
            audit_payloads=[audit_payload_for_graph(payload)]
        )
        planner = DeepSeekTaskGraphPlanner(
            graph_provider,
            risk_audit_provider=audit_provider,
        )

        graph = planner.plan("查看资料", device_id="phone-1")

        self.assertEqual(graph.active_subgoal().external_impact, "read_only")
        self.assertEqual(len(graph_provider.messages), 1)
        self.assertEqual(len(audit_provider.messages), 1)
        self.assertEqual(planner.risk_audit_call_count, 1)

    def test_semantic_audit_timeout_fails_closed_to_unknown(self):
        payload = single_subgoal_payload("查看资料", external_impact="read_only")
        provider = FakeProvider(payload, audit_error=TimeoutError("audit timeout"))
        planner = DeepSeekTaskGraphPlanner(provider)

        with self.assertRaisesRegex(TaskGraphError, "unknown.*风险"):
            planner.plan("查看资料", device_id="phone-1")

        self.assertTrue(planner.last_risk_audit.failed_closed)
        self.assertTrue(
            all(
                item.external_impact == "unknown"
                for item in planner.last_risk_audit.assessments
            )
        )

    def test_semantic_audit_invalid_json_fails_closed_to_unknown(self):
        payload = single_subgoal_payload("查看资料", external_impact="read_only")
        provider = FakeProvider(payload, audit_raw_responses=["not-json"])
        planner = DeepSeekTaskGraphPlanner(provider)

        with self.assertRaisesRegex(TaskGraphError, "unknown.*风险"):
            planner.plan("查看资料", device_id="phone-1")

        self.assertTrue(planner.last_risk_audit.failed_closed)

    def test_failed_audit_accepts_only_unknown_graph_with_confirmation_risk(self):
        payload = single_subgoal_payload(
            "处理当前对象",
            external_impact="unknown",
        )
        payload["status"] = "awaiting_confirmation"
        payload["risk_actions"] = [
            {
                "risk_id": "unknown_effect",
                "description": "外部影响尚不明确",
                "external_effect": "可能改变外部状态",
                "risk_type": "unknown_external_effect",
                "risk_level": "high",
                "subgoal_ids": ["target_state"],
                "confirmation_required": True,
            }
        ]
        payload["subgoals"][0]["risk_action_ids"] = ["unknown_effect"]
        planner = DeepSeekTaskGraphPlanner(
            FakeProvider(payload, audit_error=TimeoutError("audit timeout"))
        )

        graph = planner.plan("处理当前对象", device_id="phone-1")

        self.assertEqual(graph.status, "awaiting_confirmation")
        self.assertEqual(graph.active_subgoal().external_impact, "unknown")
        self.assertTrue(planner.last_risk_audit.failed_closed)

    def test_semantic_audit_low_confidence_fails_closed_to_unknown(self):
        payload = single_subgoal_payload("查看资料", external_impact="read_only")
        audit = audit_payload_for_graph(payload, confidence=0.49)
        planner = DeepSeekTaskGraphPlanner(
            FakeProvider(payload, audit_payloads=[audit])
        )

        with self.assertRaisesRegex(TaskGraphError, "unknown.*风险"):
            planner.plan("查看资料", device_id="phone-1")

        self.assertFalse(planner.last_risk_audit.failed_closed)
        self.assertTrue(
            all(
                item.external_impact == "unknown"
                for item in planner.last_risk_audit.assessments
            )
        )

    def test_semantic_audit_low_level_reason_is_isolated_from_control_data(self):
        payload = single_subgoal_payload("查看资料", external_impact="read_only")
        audit = audit_payload_for_graph(payload)
        audit["assessments"][0]["reason"] = "需要点击右上角按钮"
        planner = DeepSeekTaskGraphPlanner(
            FakeProvider(payload, audit_payloads=[audit])
        )

        graph = planner.plan("查看资料", device_id="phone-1")

        self.assertEqual("ready", graph.status)
        self.assertEqual(1, planner.risk_audit_call_count)
        assessment = planner.last_risk_audit.assessments[0]
        self.assertNotIn("点击", assessment.reason)
        self.assertIn("已隔离", assessment.reason)
        self.assertEqual("read_only", assessment.external_impact)

    def test_replan_runs_a_fresh_semantic_risk_audit(self):
        initial = base_payload()
        revised = copy.deepcopy(initial)
        provider = FakeProvider(initial, revised)
        planner = DeepSeekTaskGraphPlanner(provider)
        graph = planner.plan("最初的用户目标", device_id="phone-1")

        result = planner.replan(
            graph,
            observation(),
            trigger="observation_changed",
            reason="画面发生变化",
        )

        self.assertEqual(result.revision, 2)
        self.assertEqual(planner.risk_audit_call_count, 2)
        self.assertEqual(
            sum(
                "semantic-risk-audit-v1" in item[0]["content"]
                for item in provider.messages
            ),
            2,
        )
        self.assertIn("最初的用户目标", provider.messages[3][0]["content"])

    def test_active_external_state_subgoal_is_promoted_to_confirmation(self):
        payload = base_payload()
        payload["status"] = "running"
        payload["subgoals"][0]["status"] = "skipped"
        payload["subgoals"][1]["status"] = "active"
        payload["subgoals"][1]["depends_on"] = []
        payload["active_subgoal_id"] = "save_target"
        graph = DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
            "目标",
            device_id="phone-1",
        )

        self.assertEqual("awaiting_confirmation", graph.status)
        self.assertEqual("save_target", graph.active_subgoal_id)
        self.assertEqual(("save_place",), graph.active_subgoal().risk_action_ids)

    def test_unknown_impact_enters_confirmation_without_auto_advance(self):
        payload = base_payload()
        payload["risk_actions"] = [
            {
                "risk_id": "unknown_effect",
                "description": "当前影响范围无法确定",
                "external_effect": "可能改变外部状态",
                "risk_type": "unknown_external_effect",
                "risk_level": "high",
                "subgoal_ids": ["locate_target"],
                "confirmation_required": True,
            }
        ]
        payload["subgoals"][0]["external_impact"] = "unknown"
        payload["subgoals"][0]["risk_action_ids"] = ["unknown_effect"]
        payload["subgoals"][1]["objective"] = "目标地点详情保持可见"
        payload["subgoals"][1]["completion_conditions"] = ["目标地点详情可见"]
        payload["subgoals"][1]["external_impact"] = "read_only"
        payload["subgoals"][1]["risk_action_ids"] = []
        graph = DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
            "目标",
            device_id="phone-1",
        )

        self.assertEqual("awaiting_confirmation", graph.status)
        self.assertEqual("locate_target", graph.active_subgoal_id)
        self.assertEqual("unknown", graph.active_subgoal().external_impact)
        self.assertEqual(("unknown_effect",), graph.active_subgoal().risk_action_ids)

    def test_read_only_and_navigation_subgoals_are_allowed(self):
        payload = base_payload()
        payload["goal"]["objective"] = "查看图书馆详情"
        payload["completion_conditions"] = [
            {
                "condition_id": "details_visible",
                "description": "目标地点详情可见",
                "evidence_required": ["页面显示目标地点详情"],
                "satisfied": False,
                "evidence": [],
            }
        ]
        payload["risk_actions"] = []
        payload["subgoals"] = [
            payload["subgoals"][0],
            {
                "subgoal_id": "observe_details",
                "objective": "目标地点信息可被观察",
                "status": "pending",
                "depends_on": ["locate_target"],
                "constraints": [],
                "completion_conditions": ["目标地点信息可见"],
                "completion_evidence": [],
                "risk_action_ids": [],
                "external_impact": "read_only",
            },
        ]
        graph = DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
            "查看图书馆详情",
            device_id="phone-1",
        )
        self.assertEqual(graph.subgoals[0].external_impact, "navigation_only")
        self.assertEqual(graph.subgoals[1].external_impact, "read_only")

    def test_read_only_relationship_inspection_goals_are_allowed(self):
        safe_goals = (
            "查看好友列表",
            "查看群成员",
            "检查是否已关注",
        )
        for objective in safe_goals:
            with self.subTest(objective=objective):
                payload = single_subgoal_payload(
                    objective,
                    external_impact="read_only",
                )
                graph = DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
                    objective,
                    device_id="phone-1",
                )
                self.assertEqual(graph.active_subgoal().external_impact, "read_only")

    def test_contact_page_navigation_is_allowed(self):
        payload = single_subgoal_payload(
            "打开联系人页面",
            external_impact="navigation_only",
        )
        graph = DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
            "打开联系人页面",
            device_id="phone-1",
        )
        self.assertEqual(graph.active_subgoal().external_impact, "navigation_only")

    def test_unproven_safe_classification_must_be_unknown(self):
        payload = single_subgoal_payload(
            "处理当前对象",
            external_impact="read_only",
        )
        audit = audit_payload_for_graph(
            payload,
            overrides={
                "subgoals.target_state.objective": {
                    "external_impact": "unknown",
                    "risk_types": ["unknown_external_effect"],
                }
            },
        )
        with self.assertRaisesRegex(TaskGraphError, "unknown.*风险"):
            DeepSeekTaskGraphPlanner(
                FakeProvider(payload, audit_payloads=[audit])
            ).plan(
                "处理当前对象",
                device_id="phone-1",
            )

    def test_qwen_context_closes_gate_for_external_state_subgoal(self):
        payload = base_payload()
        payload["status"] = "awaiting_confirmation"
        payload["subgoals"][0]["status"] = "skipped"
        payload["subgoals"][1]["status"] = "active"
        payload["subgoals"][1]["depends_on"] = []
        payload["active_subgoal_id"] = "save_target"
        graph = DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
            "目标",
            device_id="phone-1",
        )
        gate = graph.to_qwen_context()["confirmation_gate"]
        context = graph.to_qwen_context()
        self.assertEqual(context["current_external_impact"], "external_state")
        self.assertNotIn("subgoals", context)
        self.assertEqual(len(context["risk_actions"]), 1)
        self.assertTrue(gate["required"])
        self.assertEqual(gate["state"], "awaiting_confirmation")
        self.assertEqual(gate["risk_ids"], ["save_place"])
        self.assertEqual(
            gate["scope"],
            {
                "task_id": graph.task_id,
                "device_id": graph.device_id,
                "revision": graph.revision,
                "subgoal_id": "save_target",
            },
        )
        self.assertFalse(gate["external_state_action_allowed"])

        confirmed_gate = graph.to_qwen_context(
            confirmed_risk_ids=("save_place",),
            confirmed_task_id=graph.task_id,
            confirmed_device_id=graph.device_id,
            confirmed_subgoal_id="save_target",
            confirmed_revision=graph.revision,
        )["confirmation_gate"]
        self.assertEqual(confirmed_gate["state"], "confirmed")
        self.assertTrue(confirmed_gate["external_state_action_allowed"])

    def test_confirmation_cannot_be_reused_for_an_unrelated_risk(self):
        graph = DeepSeekTaskGraphPlanner(FakeProvider(base_payload())).plan(
            "目标",
            device_id="phone-1",
        )
        with self.assertRaisesRegex(TaskGraphError, "不属于 current_subgoal"):
            graph.to_qwen_context(
                confirmed_risk_ids=("save_place",),
                confirmed_task_id=graph.task_id,
                confirmed_device_id=graph.device_id,
                confirmed_subgoal_id="locate_target",
                confirmed_revision=graph.revision,
            )

    def test_confirmation_cannot_cross_task(self):
        graph = DeepSeekTaskGraphPlanner(FakeProvider(active_external_payload())).plan(
            "目标",
            device_id="phone-1",
        )
        with self.assertRaisesRegex(TaskGraphError, "跨 task"):
            graph.to_qwen_context(
                confirmed_risk_ids=("save_place",),
                confirmed_task_id="other-task",
                confirmed_device_id=graph.device_id,
                confirmed_subgoal_id="save_target",
                confirmed_revision=graph.revision,
            )

    def test_confirmation_cannot_cross_device(self):
        graph = DeepSeekTaskGraphPlanner(FakeProvider(active_external_payload())).plan(
            "目标",
            device_id="phone-1",
        )
        with self.assertRaisesRegex(TaskGraphError, "跨 device"):
            graph.to_qwen_context(
                confirmed_risk_ids=("save_place",),
                confirmed_task_id=graph.task_id,
                confirmed_device_id="phone-2",
                confirmed_subgoal_id="save_target",
                confirmed_revision=graph.revision,
            )

    def test_confirmation_cannot_cross_subgoal(self):
        graph = DeepSeekTaskGraphPlanner(FakeProvider(active_external_payload())).plan(
            "目标",
            device_id="phone-1",
        )
        with self.assertRaisesRegex(TaskGraphError, "跨子目标复用"):
            graph.to_qwen_context(
                confirmed_risk_ids=("save_place",),
                confirmed_task_id=graph.task_id,
                confirmed_device_id=graph.device_id,
                confirmed_subgoal_id="locate_target",
                confirmed_revision=graph.revision,
            )

    def test_confirmation_cannot_cross_revision(self):
        payload = active_external_payload()
        graph = DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
            "目标",
            device_id="phone-1",
            task_id="task-confirmation-scope",
        )
        revised = DeepSeekTaskGraphPlanner(FakeProvider(copy.deepcopy(payload))).replan(
            graph,
            observation(),
            trigger="observation_changed",
            reason="场景发生变化",
        )
        with self.assertRaisesRegex(TaskGraphError, "跨 revision 复用"):
            revised.to_qwen_context(
                confirmed_risk_ids=("save_place",),
                confirmed_task_id=revised.task_id,
                confirmed_device_id=revised.device_id,
                confirmed_subgoal_id="save_target",
                confirmed_revision=graph.revision,
            )

    def test_rejects_dependency_cycle(self):
        payload = base_payload()
        payload["subgoals"][0]["depends_on"] = ["save_target"]
        payload["subgoals"][0]["status"] = "pending"
        payload["active_subgoal_id"] = None
        payload["status"] = "blocked"
        payload["subgoals"][1]["status"] = "blocked"
        with self.assertRaisesRegex(TaskGraphError, "形成环"):
            DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan("目标", device_id="phone-1")

    def test_replan_replaces_remaining_path_and_records_revision(self):
        initial = base_payload()
        revised = copy.deepcopy(initial)
        revised["status"] = "awaiting_confirmation"
        revised["subgoals"][0]["status"] = "completed"
        revised["subgoals"][0]["completion_evidence"] = ["页面标题为城市图书馆"]
        revised["subgoals"][1]["status"] = "active"
        revised["active_subgoal_id"] = "save_target"
        revised["subgoals"].append(
            {
                "subgoal_id": "verify_saved",
                "objective": "确认保存状态仍然可见",
                "status": "pending",
                "depends_on": ["save_target"],
                "constraints": [],
                "completion_conditions": ["保存状态有可见证据"],
                "completion_evidence": [],
                "risk_action_ids": [],
                "external_impact": "read_only",
            }
        )
        provider = FakeProvider(initial, revised)
        planner = DeepSeekTaskGraphPlanner(provider)
        graph = planner.plan("目标", device_id="phone-1")
        result = planner.replan(
            graph,
            observation(),
            trigger="subgoal_completed",
            reason="当前画面已显示目标地点详情",
        )
        self.assertEqual(result.revision, 2)
        self.assertEqual(result.active_subgoal_id, "save_target")
        self.assertEqual(result.replan_history[-1].added_subgoal_ids, ("verify_saved",))
        self.assertEqual(result.replan_history[-1].trigger, "subgoal_completed")
        replan_prompt = provider.messages[2][0]["content"]
        self.assertIn("最多3项", replan_prompt)
        self.assertIn("不得复制整个观察对象或 JSON", replan_prompt)

    def test_replan_can_skip_stale_subgoal_after_mismatch(self):
        initial = base_payload()
        revised = copy.deepcopy(initial)
        revised["subgoals"][0]["status"] = "skipped"
        revised["subgoals"][1]["depends_on"] = []
        revised["subgoals"][1]["status"] = "active"
        revised["active_subgoal_id"] = "save_target"
        revised["status"] = "awaiting_confirmation"
        graph = DeepSeekTaskGraphPlanner(FakeProvider(initial)).plan("目标", device_id="phone-1")
        planner = DeepSeekTaskGraphPlanner(FakeProvider(revised))
        result = planner.replan(
            graph,
            observation(),
            trigger="action_mismatch",
            reason="页面路径与旧计划不同",
        )
        self.assertEqual(result.replan_history[-1].skipped_subgoal_ids, ("locate_target",))

    def test_replan_accepts_verified_action_result_mismatch_trigger(self):
        initial = base_payload()
        revised = copy.deepcopy(initial)
        revised["subgoals"][0]["status"] = "skipped"
        revised["subgoals"][1]["depends_on"] = []
        revised["subgoals"][1]["status"] = "active"
        revised["active_subgoal_id"] = "save_target"
        revised["status"] = "awaiting_confirmation"
        graph = DeepSeekTaskGraphPlanner(FakeProvider(initial)).plan(
            "目标", device_id="phone-1"
        )

        result = DeepSeekTaskGraphPlanner(FakeProvider(revised)).replan(
            graph,
            mismatch_observation(graph),
            trigger="action_result_mismatch",
            reason="物理动作已执行，但新画面未证明预期结果",
        )

        self.assertEqual(2, result.revision)
        self.assertEqual(
            "action_result_mismatch", result.replan_history[-1].trigger
        )
        self.assertEqual(
            ("locate_target",), result.replan_history[-1].skipped_subgoal_ids
        )

    def test_typed_controller_transition_completes_only_bound_navigation(self):
        initial = base_payload()
        revised = copy.deepcopy(initial)
        ref_id = "controller_transition:receipt-matched:1"
        revised["status"] = "awaiting_confirmation"
        revised["subgoals"][0]["status"] = "completed"
        revised["subgoals"][0]["completion_evidence"] = [ref_id]
        revised["subgoals"][1]["status"] = "active"
        revised["active_subgoal_id"] = "save_target"
        graph = DeepSeekTaskGraphPlanner(FakeProvider(initial)).plan(
            "目标", device_id="phone-1"
        )

        result = DeepSeekTaskGraphPlanner(FakeProvider(revised)).replan(
            graph,
            matched_controller_observation(graph),
            trigger="action_result_matched",
            reason="一次性导航动作已由控制器验证",
        )

        self.assertEqual("completed", result.subgoals[0].status)
        self.assertEqual((ref_id,), result.subgoals[0].completion_evidence)
        self.assertEqual(
            "receipt-matched",
            result.replan_history[-1].consumed_action_transition_receipt_id,
        )

    def test_replan_repairs_matched_controller_transition_left_unconsumed(self):
        initial = base_payload()
        unconsumed = copy.deepcopy(initial)
        repaired = copy.deepcopy(initial)
        ref_id = "controller_transition:receipt-matched:1"
        repaired["status"] = "awaiting_confirmation"
        repaired["subgoals"][0]["status"] = "completed"
        repaired["subgoals"][0]["completion_evidence"] = [ref_id]
        repaired["subgoals"][1]["status"] = "active"
        repaired["active_subgoal_id"] = "save_target"
        graph = DeepSeekTaskGraphPlanner(FakeProvider(initial)).plan(
            "目标", device_id="phone-1"
        )
        provider = FakeProvider(unconsumed, repaired)

        result = DeepSeekTaskGraphPlanner(provider).replan(
            graph,
            matched_controller_observation(graph),
            trigger="action_result_matched",
            reason="一次性导航动作已由控制器验证",
        )

        self.assertEqual("completed", result.subgoals[0].status)
        self.assertEqual((ref_id,), result.subgoals[0].completion_evidence)
        graph_prompts = [
            call[0]["content"]
            for call in provider.messages
            if "semantic-risk-audit-v1" not in call[0]["content"]
        ]
        self.assertEqual(2, len(graph_prompts))
        self.assertIn(
            "matched controller_transition 未完成其绑定的 navigation_only 子目标",
            graph_prompts[-1],
        )
        self.assertIn("不得自行生成第二动作", graph_prompts[-1])

    def test_replan_second_unconsumed_matched_transition_stays_blocked(self):
        initial = base_payload()
        first = copy.deepcopy(initial)
        second = copy.deepcopy(initial)
        graph = DeepSeekTaskGraphPlanner(FakeProvider(initial)).plan(
            "目标", device_id="phone-1"
        )
        provider = FakeProvider(first, second)

        with self.assertRaisesRegex(
            TaskGraphError,
            "matched controller_transition 未完成其绑定的 navigation_only 子目标",
        ):
            DeepSeekTaskGraphPlanner(provider).replan(
                graph,
                matched_controller_observation(graph),
                trigger="action_result_matched",
                reason="一次性导航动作已由控制器验证",
            )

        graph_prompts = [
            call
            for call in provider.messages
            if "semantic-risk-audit-v1" not in call[0]["content"]
        ]
        self.assertEqual(2, len(graph_prompts))

    def test_replan_repairs_subgoal_id_used_as_completion_evidence(self):
        initial = base_payload()
        invalid = copy.deepcopy(initial)
        invalid["status"] = "awaiting_confirmation"
        invalid["subgoals"][0]["status"] = "completed"
        invalid["subgoals"][0]["completion_evidence"] = ["locate_target"]
        invalid["subgoals"][1]["status"] = "active"
        invalid["active_subgoal_id"] = "save_target"
        repaired = copy.deepcopy(invalid)
        repaired["subgoals"][0]["completion_evidence"] = [
            "controller_transition:receipt-matched:1"
        ]
        graph = DeepSeekTaskGraphPlanner(FakeProvider(initial)).plan(
            "目标", device_id="phone-1"
        )
        provider = FakeProvider(invalid, repaired)

        result = DeepSeekTaskGraphPlanner(provider).replan(
            graph,
            matched_controller_observation(graph),
            trigger="action_result_matched",
            reason="一次性导航动作已由控制器验证",
        )

        self.assertEqual("completed", result.subgoals[0].status)
        self.assertEqual(
            ("controller_transition:receipt-matched:1",),
            result.subgoals[0].completion_evidence,
        )
        replan_prompts = [
            call[0]["content"]
            for call in provider.messages
            if "高层任务图重规划器" in call[0]["content"]
        ]
        self.assertEqual(2, len(replan_prompts))
        self.assertIn("subgoal_id、condition_id", replan_prompts[-1])

    def test_typed_controller_transition_cannot_complete_external_state(self):
        initial = active_external_payload()
        completed = copy.deepcopy(initial)
        completed["status"] = "completed"
        completed["active_subgoal_id"] = None
        completed["subgoals"][1]["status"] = "completed"
        completed["subgoals"][1]["completion_evidence"] = [
            "controller_transition:receipt-matched:1"
        ]
        completed["completion_conditions"][0]["satisfied"] = True
        completed["completion_conditions"][0]["evidence"] = [
            "页面显示已收藏状态"
        ]
        graph = DeepSeekTaskGraphPlanner(FakeProvider(initial)).plan(
            "目标", device_id="phone-1"
        )
        observed = matched_controller_observation(graph)
        observed = ObservedState(
            **{
                **observed.__dict__,
                "visible_evidence": ("页面显示已收藏状态",),
            }
        )

        with self.assertRaisesRegex(TaskGraphError, "navigation_only"):
            DeepSeekTaskGraphPlanner(FakeProvider(completed)).replan(
                graph,
                observed,
                trigger="action_result_matched",
                reason="外部动作返回 matched",
            )

    def test_consumed_action_transition_receipt_cannot_be_replayed(self):
        initial = base_payload()
        revised = copy.deepcopy(initial)
        ref_id = "controller_transition:receipt-reuse:1"
        revised["status"] = "awaiting_confirmation"
        revised["subgoals"][0]["status"] = "completed"
        revised["subgoals"][0]["completion_evidence"] = [ref_id]
        revised["subgoals"][1]["status"] = "active"
        revised["active_subgoal_id"] = "save_target"
        graph = DeepSeekTaskGraphPlanner(FakeProvider(initial)).plan(
            "目标", device_id="phone-1"
        )
        revision_two = DeepSeekTaskGraphPlanner(FakeProvider(revised)).replan(
            graph,
            matched_controller_observation(graph, receipt_id="receipt-reuse"),
            trigger="action_result_matched",
            reason="导航完成",
        )
        replay_observed = matched_controller_observation(
            revision_two,
            receipt_id="receipt-reuse",
        )

        with self.assertRaisesRegex(TaskGraphError, "已经消费"):
            DeepSeekTaskGraphPlanner(FakeProvider(active_external_payload())).replan(
                revision_two,
                replay_observed,
                trigger="action_result_matched",
                reason="重放旧回执",
            )

    def test_replan_repairs_one_low_level_protocol_violation(self):
        graph = DeepSeekTaskGraphPlanner(FakeProvider(base_payload())).plan(
            "目标", device_id="phone-1"
        )
        invalid = copy.deepcopy(base_payload())
        invalid["clarification_questions"] = ["请点击浏览器图标后继续。"]
        repaired = copy.deepcopy(base_payload())
        provider = FakeProvider(invalid, repaired)

        result = DeepSeekTaskGraphPlanner(provider).replan(
            graph,
            mismatch_observation(graph),
            trigger="action_result_mismatch",
            reason="动作已执行但预期结果没有出现",
        )

        self.assertEqual(2, result.revision)
        self.assertEqual("action_result_mismatch", result.replan_history[-1].trigger)
        self.assertEqual(3, len(provider.messages))
        self.assertIn("clarification_questions", provider.messages[1][0]["content"])
        self.assertIn("唯一一次", provider.messages[1][0]["content"])

    def test_replan_second_low_level_protocol_violation_stays_blocked(self):
        graph = DeepSeekTaskGraphPlanner(FakeProvider(base_payload())).plan(
            "目标", device_id="phone-1"
        )
        first = copy.deepcopy(base_payload())
        first["clarification_questions"] = ["请点击浏览器图标后继续。"]
        second = copy.deepcopy(base_payload())
        second["clarification_questions"] = ["请再次点击同一位置。"]
        provider = FakeProvider(first, second)

        with self.assertRaisesRegex(TaskGraphError, "包含低层动作表达"):
            DeepSeekTaskGraphPlanner(provider).replan(
                graph,
                mismatch_observation(graph),
                trigger="action_result_mismatch",
                reason="动作已执行但预期结果没有出现",
            )

        self.assertEqual(2, len(provider.messages))

    def test_replan_blocked_retry_request_becomes_high_level_clarification(self):
        graph = DeepSeekTaskGraphPlanner(FakeProvider(base_payload())).plan(
            "目标", device_id="phone-1"
        )

        def blocked_retry_payload():
            payload = copy.deepcopy(base_payload())
            payload["status"] = "blocked"
            payload["subgoals"][0]["status"] = "blocked"
            payload["active_subgoal_id"] = None
            payload["clarification_questions"] = [
                "请确认是否允许再次点击同一图标？"
            ]
            return payload

        provider = FakeProvider(blocked_retry_payload(), blocked_retry_payload())
        result = DeepSeekTaskGraphPlanner(provider).replan(
            graph,
            mismatch_observation(graph),
            trigger="action_result_mismatch",
            reason="动作已执行但预期结果没有出现",
        )

        self.assertEqual("blocked", result.status)
        self.assertIsNone(result.active_subgoal_id)
        self.assertEqual(2, result.revision)
        self.assertEqual(
            (
                "动作后的新画面未证明预期结果，且当前没有可验证的安全替代路径；"
                "请说明希望继续原目标还是停止任务。",
            ),
            result.clarification_questions,
        )
        self.assertEqual(3, len(provider.messages))

    def test_replan_cannot_mutate_goal(self):
        initial = base_payload()
        revised = copy.deepcopy(initial)
        revised["goal"]["objective"] = "另一个目标"
        graph = DeepSeekTaskGraphPlanner(FakeProvider(initial)).plan("目标", device_id="phone-1")
        with self.assertRaisesRegex(TaskGraphError, "不能改写用户目标"):
            DeepSeekTaskGraphPlanner(FakeProvider(revised)).replan(
                graph,
                observation(),
                trigger="observation_changed",
                reason="页面发生变化",
            )

    def test_replan_cannot_delete_user_constraint(self):
        initial = base_payload()
        revised = copy.deepcopy(initial)
        revised["constraints"] = []
        graph = DeepSeekTaskGraphPlanner(FakeProvider(initial)).plan("目标", device_id="phone-1")
        with self.assertRaisesRegex(TaskGraphError, "不能删除已有全局约束"):
            DeepSeekTaskGraphPlanner(FakeProvider(revised)).replan(
                graph,
                observation(),
                trigger="constraint_discovered",
                reason="发现新约束",
            )

    def test_replan_cannot_detach_existing_risk_from_subgoal(self):
        initial = base_payload()
        revised = copy.deepcopy(initial)
        revised["risk_actions"][0]["subgoal_ids"] = ["locate_target"]
        replacement_risk = copy.deepcopy(revised["risk_actions"][0])
        replacement_risk["risk_id"] = "replacement_risk"
        replacement_risk["subgoal_ids"] = ["save_target"]
        revised["risk_actions"].append(replacement_risk)
        revised["subgoals"][0]["risk_action_ids"] = ["save_place"]
        revised["subgoals"][0]["external_impact"] = "external_state"
        revised["subgoals"][1]["risk_action_ids"] = ["replacement_risk"]
        revised["status"] = "awaiting_confirmation"
        graph = DeepSeekTaskGraphPlanner(FakeProvider(initial)).plan("目标", device_id="phone-1")
        with self.assertRaisesRegex(TaskGraphError, "不能改写或降低既有风险"):
            DeepSeekTaskGraphPlanner(FakeProvider(revised)).replan(
                graph,
                observation(),
                trigger="risk_detected",
                reason="重新检查风险",
            )

    def test_replan_cannot_delete_existing_risk(self):
        initial = base_payload()
        revised = copy.deepcopy(initial)
        revised["risk_actions"] = []
        revised["subgoals"][1]["risk_action_ids"] = []
        graph = DeepSeekTaskGraphPlanner(FakeProvider(initial)).plan(
            "目标",
            device_id="phone-1",
        )
        with self.assertRaisesRegex(TaskGraphError, "不能删除既有风险"):
            DeepSeekTaskGraphPlanner(FakeProvider(revised)).replan(
                graph,
                observation(),
                trigger="observation_changed",
                reason="模型删除了风险",
            )

    def test_replan_cannot_change_existing_risk_type(self):
        initial = base_payload()
        revised = copy.deepcopy(initial)
        revised["risk_actions"][0]["risk_type"] = "data_deletion"
        graph = DeepSeekTaskGraphPlanner(FakeProvider(initial)).plan(
            "目标",
            device_id="phone-1",
        )
        with self.assertRaisesRegex(TaskGraphError, "不能改换既有风险类别"):
            DeepSeekTaskGraphPlanner(FakeProvider(revised)).replan(
                graph,
                observation(),
                trigger="risk_detected",
                reason="模型改换了风险类别",
            )

    def test_replan_cannot_downgrade_external_state_to_read_only(self):
        initial = base_payload()
        revised = copy.deepcopy(initial)
        revised["subgoals"][1]["external_impact"] = "read_only"
        graph = DeepSeekTaskGraphPlanner(FakeProvider(initial)).plan(
            "目标",
            device_id="phone-1",
        )
        with self.assertRaisesRegex(TaskGraphError, "不能降低既有 external_state"):
            DeepSeekTaskGraphPlanner(FakeProvider(revised)).replan(
                graph,
                observation(),
                trigger="observation_changed",
                reason="模型错误地降低影响分类",
            )

    def test_completed_subgoal_cannot_be_resurrected(self):
        completed_payload = base_payload()
        completed_payload["subgoals"][0]["status"] = "completed"
        completed_payload["subgoals"][0]["completion_evidence"] = ["历史可见证据"]
        completed_payload["subgoals"][1]["status"] = "active"
        completed_payload["active_subgoal_id"] = "save_target"
        completed_payload["status"] = "awaiting_confirmation"
        graph = DeepSeekTaskGraphPlanner(FakeProvider(base_payload())).plan("目标", device_id="phone-1")
        first = DeepSeekTaskGraphPlanner(FakeProvider(completed_payload)).replan(
            graph,
            ObservedState("scene-1", "详情已显示", ("历史可见证据",)),
            trigger="subgoal_completed",
            reason="第一子目标完成",
        )
        resurrected = copy.deepcopy(completed_payload)
        resurrected["subgoals"][0]["status"] = "active"
        resurrected["subgoals"][0]["completion_evidence"] = []
        resurrected["subgoals"][1]["status"] = "pending"
        resurrected["active_subgoal_id"] = "locate_target"
        resurrected["status"] = "running"
        with self.assertRaisesRegex(TaskGraphError, "不能复活或改写"):
            DeepSeekTaskGraphPlanner(FakeProvider(resurrected)).replan(
                first,
                observation(),
                trigger="observation_changed",
                reason="页面变化",
            )

    def test_replan_restores_omitted_completed_history_evidence(self):
        completed_payload = base_payload()
        completed_payload["subgoals"][0]["status"] = "completed"
        completed_payload["subgoals"][0]["completion_evidence"] = ["历史可见证据"]
        completed_payload["subgoals"][1]["status"] = "active"
        completed_payload["active_subgoal_id"] = "save_target"
        completed_payload["status"] = "awaiting_confirmation"
        graph = DeepSeekTaskGraphPlanner(FakeProvider(base_payload())).plan(
            "目标",
            device_id="phone-1",
        )
        first = DeepSeekTaskGraphPlanner(FakeProvider(completed_payload)).replan(
            graph,
            ObservedState("scene-1", "详情已显示", ("历史可见证据",)),
            trigger="subgoal_completed",
            reason="第一子目标完成",
        )
        omitted = copy.deepcopy(completed_payload)
        omitted["subgoals"][0]["completion_evidence"] = []

        revised = DeepSeekTaskGraphPlanner(FakeProvider(omitted)).replan(
            first,
            observation(),
            trigger="observation_changed",
            reason="页面出现轻微观察差异",
        )

        self.assertEqual(
            ("历史可见证据",),
            revised.subgoals[0].completion_evidence,
        )

    def test_new_completion_claim_requires_current_observation_evidence(self):
        initial = base_payload()
        revised = copy.deepcopy(initial)
        revised["subgoals"][0]["status"] = "completed"
        revised["subgoals"][0]["completion_evidence"] = ["模型虚构的证据"]
        revised["subgoals"][1]["status"] = "active"
        revised["active_subgoal_id"] = "save_target"
        revised["status"] = "awaiting_confirmation"
        graph = DeepSeekTaskGraphPlanner(FakeProvider(initial)).plan("目标", device_id="phone-1")
        with self.assertRaisesRegex(TaskGraphError, "当前观察之外"):
            DeepSeekTaskGraphPlanner(
                FakeProvider(revised, copy.deepcopy(revised))
            ).replan(
                graph,
                observation(),
                trigger="subgoal_completed",
                reason="模型认为已经完成",
            )

    def test_replan_canonicalizes_one_complete_literal_visible_clause(self):
        initial = base_payload()
        initial["goal"]["objective"] = "软键盘最终不可见"
        initial["goal"]["entities"] = {}
        initial["constraints"] = ["不得提交或发送"]
        initial["risk_actions"] = []
        initial["completion_conditions"] = [
            {
                "condition_id": "keyboard_hidden",
                "description": "软键盘不可见",
                "evidence_required": ["软键盘未显示"],
                "satisfied": False,
                "evidence": [],
            }
        ]
        initial["subgoals"] = [
            {
                "subgoal_id": "hide_keyboard",
                "objective": "软键盘不可见",
                "status": "active",
                "depends_on": [],
                "constraints": ["不得提交或发送"],
                "completion_conditions": ["软键盘未显示"],
                "completion_evidence": [],
                "risk_action_ids": [],
                "external_impact": "navigation_only",
            }
        ]
        initial["active_subgoal_id"] = "hide_keyboard"
        completed = copy.deepcopy(initial)
        completed["status"] = "completed"
        completed["active_subgoal_id"] = None
        completed["completion_conditions"][0]["satisfied"] = True
        completed["completion_conditions"][0]["evidence"] = ["软键盘未显示"]
        completed["subgoals"][0]["status"] = "completed"
        completed["subgoals"][0]["completion_evidence"] = ["软键盘未显示"]
        graph = DeepSeekTaskGraphPlanner(FakeProvider(initial)).plan(
            "让软键盘最终不可见",
            device_id="phone-1",
        )
        observed = matched_controller_observation(graph)
        full_fact = "页面显示输入结果，唯一输入框为 agent，软键盘未显示。"
        observed = ObservedState(
            **{
                **observed.__dict__,
                "summary": full_fact,
                "visible_evidence": (full_fact, "应用输入框当前文字：agent"),
            }
        )
        provider = FakeProvider(completed)

        result = DeepSeekTaskGraphPlanner(provider).replan(
            graph,
            observed,
            trigger="action_result_matched",
            reason="系统返回动作后键盘已经收起",
        )

        self.assertEqual("completed", result.status)
        self.assertEqual((full_fact,), result.completion_conditions[0].evidence)
        self.assertEqual((full_fact,), result.subgoals[0].completion_evidence)
        graph_messages = [
            messages
            for messages in provider.messages
            if "semantic-risk-audit-v1" not in messages[0]["content"]
        ]
        self.assertEqual(1, len(graph_messages))

    def test_replan_does_not_canonicalize_identifier_or_partial_phrase(self):
        initial = base_payload()
        revised = copy.deepcopy(initial)
        revised["subgoals"][0]["status"] = "completed"
        revised["subgoals"][0]["completion_evidence"] = ["locate_target"]
        revised["subgoals"][1]["status"] = "active"
        revised["active_subgoal_id"] = "save_target"
        revised["status"] = "awaiting_confirmation"
        graph = DeepSeekTaskGraphPlanner(FakeProvider(initial)).plan(
            "目标",
            device_id="phone-1",
        )
        observed = ObservedState(
            "scene-2",
            "页面显示目标地点详情",
            ("页面显示目标地点详情，软键盘未显示。",),
        )

        with self.assertRaisesRegex(TaskGraphError, "当前观察之外"):
            DeepSeekTaskGraphPlanner(
                FakeProvider(revised, copy.deepcopy(revised))
            ).replan(
                graph,
                observed,
                trigger="subgoal_completed",
                reason="模型使用标识符而非证据",
            )

        partial = copy.deepcopy(revised)
        partial["subgoals"][0]["completion_evidence"] = ["目标地点"]
        with self.assertRaisesRegex(TaskGraphError, "当前观察之外"):
            DeepSeekTaskGraphPlanner(
                FakeProvider(partial, copy.deepcopy(partial))
            ).replan(
                graph,
                observed,
                trigger="subgoal_completed",
                reason="模型只复制了证据片段",
            )

    def test_completed_graph_requires_all_global_evidence(self):
        payload = base_payload()
        payload["status"] = "completed"
        payload["active_subgoal_id"] = None
        payload["subgoals"][0]["status"] = "skipped"
        payload["subgoals"][1]["status"] = "skipped"
        with self.assertRaisesRegex(TaskGraphError, "满足全部全局完成条件"):
            DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan("目标", device_id="phone-1")

    def test_unconfigured_provider_stops_safely(self):
        provider = FakeProvider(base_payload())
        provider.configured = False
        with self.assertRaisesRegex(TaskGraphError, "尚未配置"):
            DeepSeekTaskGraphPlanner(provider).plan("目标", device_id="phone-1")

    def test_invalid_json_is_reported_as_task_graph_error(self):
        provider = FakeProvider(base_payload())
        provider.payloads = []

        def invalid_json(messages, max_tokens=2000):
            return "not-json"

        provider.chat_json = invalid_json
        with self.assertRaisesRegex(TaskGraphError, "有效 JSON"):
            DeepSeekTaskGraphPlanner(provider).plan("目标", device_id="phone-1")


if __name__ == "__main__":
    unittest.main()
