import copy
import json
import unittest

from deepseek_task_graph import (
    DEEPSEEK_TASK_GRAPH_PROTOCOL_VERSION,
    DeepSeekTaskGraphPlanner,
    ObservedState,
    TaskGraphError,
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

    def test_initial_plan_repairs_one_structural_status_error(self):
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
        self.assertEqual(len(provider.messages), 3)
        self.assertIn("可推进任务图必须且只能有一个活动子目标", provider.messages[1][0]["content"])

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

    def test_active_external_state_subgoal_must_await_confirmation(self):
        payload = base_payload()
        payload["status"] = "running"
        payload["subgoals"][0]["status"] = "skipped"
        payload["subgoals"][1]["status"] = "active"
        payload["subgoals"][1]["depends_on"] = []
        payload["active_subgoal_id"] = "save_target"
        with self.assertRaisesRegex(TaskGraphError, "必须等待用户确认"):
            DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
                "目标",
                device_id="phone-1",
            )

    def test_unknown_impact_cannot_auto_advance(self):
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
        with self.assertRaisesRegex(TaskGraphError, "必须等待用户确认"):
            DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
                "目标",
                device_id="phone-1",
            )

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
            observation(),
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
            observation(),
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
                observation(),
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
            observation(),
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
            DeepSeekTaskGraphPlanner(FakeProvider(revised)).replan(
                graph,
                observation(),
                trigger="subgoal_completed",
                reason="模型认为已经完成",
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
