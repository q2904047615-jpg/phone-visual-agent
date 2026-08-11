import copy
import json
import unittest

from deepseek_task_graph import (
    DeepSeekTaskGraphPlanner,
    ObservedState,
    TaskGraphError,
)


class FakeProvider:
    configured = True

    def __init__(self, *payloads):
        self.payloads = list(payloads)
        self.messages = []

    def chat_json(self, messages, max_tokens=2000):
        self.messages.append(messages)
        return json.dumps(self.payloads.pop(0), ensure_ascii=False)


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


class DeepSeekTaskGraphTests(unittest.TestCase):
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

    def test_rejects_low_level_action_fields(self):
        payload = base_payload()
        payload["subgoals"][0]["steps"] = [{"tap": [10, 20]}]
        with self.assertRaisesRegex(TaskGraphError, "协议外字段"):
            DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan("目标", device_id="phone-1")

    def test_rejects_low_level_instruction_in_subgoal_text(self):
        payload = base_payload()
        payload["subgoals"][0]["objective"] = "点击搜索结果中的图书馆"
        with self.assertRaisesRegex(TaskGraphError, "包含低层动作表达"):
            DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
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
            "运行 PowerShell 系统命令",
        )
        for objective in forbidden_objectives:
            with self.subTest(objective=objective):
                payload = base_payload()
                payload["subgoals"][0]["objective"] = objective
                with self.assertRaisesRegex(TaskGraphError, "包含低层动作表达"):
                    DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
                        "目标",
                        device_id="phone-1",
                    )

    def test_rejects_low_level_instruction_in_completion_condition(self):
        payload = base_payload()
        payload["subgoals"][0]["completion_conditions"] = ["点击收藏按钮后完成"]
        with self.assertRaisesRegex(TaskGraphError, "包含低层动作表达"):
            DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
                "目标",
                device_id="phone-1",
            )

    def test_rejects_low_level_instruction_hidden_as_constraint(self):
        payload = base_payload()
        payload["subgoals"][0]["constraints"] = ["点击第一个搜索结果"]
        with self.assertRaisesRegex(TaskGraphError, "包含低层动作表达"):
            DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan(
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

    def test_risk_action_cannot_disable_confirmation(self):
        payload = base_payload()
        payload["risk_actions"][0]["confirmation_required"] = False
        with self.assertRaisesRegex(TaskGraphError, "必须等待用户确认"):
            DeepSeekTaskGraphPlanner(FakeProvider(payload)).plan("目标", device_id="phone-1")

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
        self.assertFalse(gate["external_state_action_allowed"])

        confirmed_gate = graph.to_qwen_context(
            confirmed_risk_ids=("save_place",)
        )["confirmation_gate"]
        self.assertEqual(confirmed_gate["state"], "confirmed")
        self.assertTrue(confirmed_gate["external_state_action_allowed"])

    def test_confirmation_cannot_be_reused_for_an_unrelated_risk(self):
        graph = DeepSeekTaskGraphPlanner(FakeProvider(base_payload())).plan(
            "目标",
            device_id="phone-1",
        )
        with self.assertRaisesRegex(TaskGraphError, "不属于 current_subgoal"):
            graph.to_qwen_context(confirmed_risk_ids=("save_place",))

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
