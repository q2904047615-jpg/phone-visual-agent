import json
import unittest

from agent.application.deepseek_task_graph import DeepSeekTaskGraphPlanner
from agent.domain.generic_goal import ActiveVisualGoal
from agent.domain.qwen_task_context import QwenTaskContext
from agent.domain.task_graph import TaskGraphError


class FakeProvider:
    configured = True

    def __init__(self, payload):
        self.payload = payload
        self.messages = []

    def chat_json(self, messages, max_tokens=2000):
        self.messages.append(messages)
        return json.dumps(self.payload, ensure_ascii=False)


def subgoal(subgoal_id, objective, *, depends_on=(), result="目标状态可见",
    input_field_id="", input_operation="", required_action_kind=""):
    value = {
        "subgoal_id": subgoal_id,
        "objective": objective,
        "depends_on": list(depends_on),
        "constraints": [],
        "completion_conditions": [result],
    }
    if input_field_id:
        value["input_field_id"] = input_field_id
        value["input_operation"] = input_operation
    if required_action_kind:
        value["required_action_kind"] = required_action_kind
    return value


def initial_plan(*, subgoals=None, effects=(), entities=None, objective="完成普通手机任务"):
    return {
        "goal": {
            "objective": objective,
            "target_apps": [{"app_id": "current_foreground", "app_name": "当前前台应用"}],
            "entities": dict(entities or {}),
        },
        "constraints": [],
        "completion_conditions": [{
            "condition_id": "done",
            "description": "用户目标已经完成",
            "evidence_required": ["动作后新画面或结果证明目标完成"],
        }],
        "effect_intents": list(effects),
        "subgoals": list(subgoals or [subgoal("step", "完成当前目标")]),
        "clarification_questions": [],
    }


class DeepSeekOneShotTransportTests(unittest.TestCase):
    def plan(self, payload, goal="完成普通手机任务"):
        provider = FakeProvider(payload)
        graph = DeepSeekTaskGraphPlanner(provider).plan(goal, device_id="phone-1")
        self.assertEqual(1, len(provider.messages))
        return graph, provider.messages[0][0]["content"]

    def test_initial_plan_derives_one_active_subgoal_without_replan_api(self):
        graph, _ = self.plan(initial_plan(subgoals=[
            subgoal("open_app", "打开目标应用"),
            subgoal("open_page", "进入目标页面", depends_on=("open_app",)),
        ]))
        self.assertEqual("open_app", graph.active_subgoal_id)
        self.assertEqual(["active", "pending"], [item.status for item in graph.subgoals])
        self.assertEqual("ready", graph.status)
        self.assertFalse(hasattr(DeepSeekTaskGraphPlanner, "replan"))

    def test_initial_prompt_requires_home_recents_clear_all_sequence(self):
        _, prompt = self.plan(initial_plan(), goal="清理全部后台应用")

        self.assertIn("先回到 Android Launcher/主屏幕", prompt)
        self.assertIn("再打开系统最近任务页", prompt)
        self.assertIn("最后点击当前截图中的系统一键清理全部按钮", prompt)
        self.assertIn('required_action_kind="home"', prompt)
        self.assertIn('required_action_kind="open_recent_apps"', prompt)
        self.assertIn('required_action_kind="tap_semantic"', prompt)
        self.assertIn("其它任务不得因此套用这个固定顺序", prompt)

    def test_required_action_kind_is_structural_and_visible_to_qwen(self):
        raw = initial_plan(subgoals=[subgoal(
            "open-recents",
            "打开系统最近任务页",
            required_action_kind="open_recent_apps",
        )])

        graph, _ = self.plan(raw, goal="清理后台卡片")
        context = QwenTaskContext.from_dict(graph.to_qwen_context())

        self.assertEqual("open_recent_apps", graph.active_subgoal().required_action_kind)
        self.assertEqual("open_recent_apps", context.current_subgoal["required_action_kind"])

    def test_ordinary_subgoal_does_not_gain_required_action_kind(self):
        graph, _ = self.plan(initial_plan(subgoals=[subgoal("browse", "查看当前页面详情")]))

        self.assertEqual("", graph.active_subgoal().required_action_kind)
        self.assertNotIn("required_action_kind", graph.to_qwen_context()["current_subgoal"])

    def test_required_action_kind_cannot_duplicate_typed_operation_authority(self):
        raw = initial_plan(entities={"input_text": "hello"}, subgoals=[subgoal(
            "type",
            "输入正文",
            input_field_id="primary_input",
            input_operation="input_verified_text",
            required_action_kind="tap_semantic",
        )])

        with self.assertRaisesRegex(TaskGraphError, "不能同时声明 required_action_kind"):
            self.plan(raw)

    def test_retired_runtime_copies_have_no_authority(self):
        raw = initial_plan(subgoals=[subgoal("first", "先处理目标"), subgoal("second", "再处理目标")])
        raw.update({"status": "completed", "active_subgoal_id": "second", "replan_history": ["stale"]})
        raw["subgoals"][0].update({"status": "completed", "completion_evidence": ["模型自写证据"]})
        graph, _ = self.plan(raw)
        self.assertEqual("ready", graph.status)
        self.assertEqual("first", graph.active_subgoal_id)
        self.assertFalse(graph.completion_conditions[0].satisfied)

    def test_effect_source_is_the_only_runtime_effect_link(self):
        effect = {
            "effect_id": "send_effect", "kind": "send_message",
            "target_entity_roles": ["recipient"], "payload_entity_roles": ["input_text"],
            "source_subgoal_ids": ["send"], "expected_results": ["retired model copy"],
        }
        raw = initial_plan(entities={"recipient": "文件传输助手", "input_text": "你好"}, subgoals=[
            subgoal("type", "输入授权文字", input_field_id="primary_input",
                input_operation="input_verified_text", result="输入框显示授权文字"),
            subgoal("send", "发送消息", depends_on=("type",), result="消息已经发送"),
        ], effects=[effect])
        graph, _ = self.plan(raw)
        by_id = {item.subgoal_id: item for item in graph.subgoals}
        self.assertEqual((), by_id["type"].risk_action_ids)
        self.assertEqual(("send_effect",), by_id["send"].risk_action_ids)
        self.assertEqual(("消息已经发送",), graph.risk_actions[0].expected_result_texts)
        self.assertFalse(graph.risk_actions[0].confirmation_required)

    def test_only_authentication_and_financial_transaction_require_confirmation(self):
        cases = (
            ("authentication", True), ("financial_transaction", True), ("send_message", False),
            ("publish_content", False), ("relationship_change", False), ("membership_change", False),
            ("data_mutation", False), ("sensitive_permission_change", False),
            ("irreversible_account_deletion", False), ("irreversible_data_deletion", False),
        )
        for kind, confirmation_required in cases:
            with self.subTest(kind=kind):
                raw = initial_plan(entities={"target": "当前对象"}, effects=[{
                    "effect_id": "effect", "kind": kind, "target_entity_roles": ["target"],
                    "payload_entity_roles": [], "source_subgoal_ids": ["step"],
                }])
                graph, _ = self.plan(raw)
                self.assertEqual(confirmation_required, graph.risk_actions[0].confirmation_required)
                self.assertEqual("awaiting_confirmation" if confirmation_required else "ready", graph.status)

    def test_single_input_has_explicit_typed_binding_in_qwen_context(self):
        raw = initial_plan(entities={"input_text": "aaazjie？你好"}, subgoals=[
            subgoal("type", "填写正文", input_field_id="primary_input", input_operation="input_verified_text")])
        graph, prompt = self.plan(raw)
        context = QwenTaskContext.from_dict(graph.to_qwen_context())
        self.assertEqual("primary_input", context.current_subgoal["input_field_id"])
        self.assertEqual("input_verified_text", context.current_subgoal["input_operation"])
        self.assertEqual("aaazjie？你好", context.requested_input_text)
        self.assertIn('input_field_id="primary_input"', prompt)
        self.assertNotIn("deepseek_replan", prompt)

    def test_single_step_single_field_gets_bounded_typed_default_without_prose(self):
        raw = initial_plan(entities={"input_text": "alpha"}, subgoals=[
            subgoal("step", "任意不参与本地解释的描述")])
        graph, _ = self.plan(raw)
        context = QwenTaskContext.from_dict(graph.to_qwen_context())

        self.assertEqual("primary_input", context.current_subgoal["input_field_id"])
        self.assertEqual("input_verified_text", context.current_subgoal["input_operation"])
        self.assertEqual("alpha", context.requested_input_text)

    def test_multi_field_uses_current_typed_field_without_prose_matching(self):
        raw = initial_plan(entities={"input_fields": [
            {"field_id": "email", "field_label": "邮箱", "text": "a@example.com"},
            {"field_id": "note", "field_label": "备注", "text": "你好"},
        ]}, subgoals=[
            subgoal("email", "填写第一个字段", input_field_id="email", input_operation="input_verified_text"),
            subgoal("note", "填写第二个字段", depends_on=("email",), input_field_id="note",
                input_operation="input_verified_text"),
        ])
        graph, _ = self.plan(raw)
        context = QwenTaskContext.from_dict(graph.to_qwen_context())
        self.assertEqual("email", context.current_subgoal["input_field_id"])
        self.assertEqual("a@example.com", context.requested_input_text)
        self.assertEqual(2, len(context.goal["entities"]["input_fields"]))

    def test_clear_only_field_has_typed_target_without_inventing_body_text(self):
        raw = initial_plan(entities={"input_fields": [{
            "field_id": "search", "field_label": "搜索框", "text": "", "target_only": True,
        }]}, subgoals=[subgoal("clear", "清空当前字段", input_field_id="search",
            input_operation="clear_verified_text")])
        graph, _ = self.plan(raw)
        context = QwenTaskContext.from_dict(graph.to_qwen_context())

        self.assertEqual("search", context.current_subgoal["input_field_id"])
        self.assertEqual("clear_verified_text", context.current_subgoal["input_operation"])
        self.assertIsNone(context.requested_input_text)
        self.assertTrue(context.goal["entities"]["input_fields"][0]["target_only"])

    def test_unique_typed_field_does_not_require_visible_label_text(self):
        context = {
            "entities": {
                "input_fields": [{"field_id": "message", "field_label": "", "text": "你好"}],
                "active_subgoal_visual_context": {
                    "subgoal_id": "type",
                    "objective": "填写当前字段",
                    "constraints": [],
                    "completion_conditions": ["输入完成"],
                    "execution_class": "physical_action",
                    "goal_entities": {
                        "active_input_field_id": "message",
                        "active_input_field_label": "",
                        "active_input_multiline": False,
                        "active_input_operation": "input_verified_text",
                        "active_input_transaction_text": "你好",
                    },
                },
            }
        }

        self.assertTrue(ActiveVisualGoal.from_context(context).unique_typed_field)

    def test_unknown_typed_field_is_rejected_structurally(self):
        raw = initial_plan(entities={"input_fields": [{"field_id": "email", "text": "a@example.com"}]},
            subgoals=[subgoal("type", "填写字段", input_field_id="missing",
                input_operation="input_verified_text")])
        with self.assertRaisesRegex(TaskGraphError, "未声明的 typed input field"):
            self.plan(raw)

    def test_ordinary_text_may_mention_control_words_without_semantic_blacklist(self):
        raw = initial_plan(objective="在备注输入框逐字输入 adb shell x=20 main.exe",
            entities={"input_text": "adb shell x=20 main.exe"}, subgoals=[
                subgoal("type", "填写备注", input_field_id="primary_input",
                    input_operation="input_verified_text")])
        graph, _ = self.plan(raw, goal=raw["goal"]["objective"])
        self.assertEqual("adb shell x=20 main.exe", graph.goal.entities["input_text"])

    def test_actual_low_level_control_fields_are_still_rejected(self):
        raw = initial_plan(entities={"action": {"x": 100, "y": 200}})
        with self.assertRaisesRegex(TaskGraphError, "低层控制字段"):
            self.plan(raw)

    def test_dependency_cycle_and_missing_effect_reference_remain_errors(self):
        cycle = initial_plan(subgoals=[subgoal("a", "步骤A", depends_on=("b",)),
            subgoal("b", "步骤B", depends_on=("a",))])
        with self.assertRaisesRegex(TaskGraphError, "依赖形成环"):
            self.plan(cycle)
        missing = initial_plan(effects=[{
            "effect_id": "effect", "kind": "send_message", "target_entity_roles": [],
            "payload_entity_roles": [], "source_subgoal_ids": ["missing"],
        }])
        with self.assertRaisesRegex(TaskGraphError, "引用不存在子目标"):
            self.plan(missing)

    def test_clarification_is_diagnostic_when_an_executable_plan_exists(self):
        raw = initial_plan()
        raw["clarification_questions"] = ["可选的补充信息"]
        graph, _ = self.plan(raw)
        self.assertEqual("ready", graph.status)
        self.assertEqual("step", graph.active_subgoal_id)

    def test_empty_subgoals_derive_one_executable_goal_instead_of_blocking(self):
        raw = initial_plan()
        raw["subgoals"] = []
        raw["clarification_questions"] = ["可选的补充信息"]

        graph, _ = self.plan(raw)

        self.assertEqual("ready", graph.status)
        self.assertEqual("goal", graph.active_subgoal_id)
        self.assertEqual(1, len(graph.subgoals))
        self.assertEqual(graph.goal.objective, graph.subgoals[0].objective)
        self.assertEqual(("用户目标已经完成",), graph.subgoals[0].completion_conditions)
        self.assertEqual(("可选的补充信息",), graph.clarification_questions)

    def test_empty_subgoals_keep_unique_input_transport_binding(self):
        raw = initial_plan(entities={"input_text": "aaazjie？你好"})
        raw["subgoals"] = []
        raw["clarification_questions"] = ["可选说明"]

        graph, _ = self.plan(raw)
        context = QwenTaskContext.from_dict(graph.to_qwen_context())

        self.assertEqual("goal", graph.active_subgoal_id)
        self.assertEqual("primary_input", context.current_subgoal["input_field_id"])
        self.assertEqual("input_verified_text", context.current_subgoal["input_operation"])
        self.assertEqual("aaazjie？你好", context.requested_input_text)

    def test_empty_subgoals_do_not_bypass_missing_target_scope(self):
        raw = initial_plan()
        raw["goal"]["target_apps"] = []
        raw["goal"]["entities"] = {}
        raw["subgoals"] = []
        raw["clarification_questions"] = ["请说明要操作哪台设备或当前表面"]

        with self.assertRaisesRegex(TaskGraphError, "必须声明目标 App"):
            self.plan(raw)


if __name__ == "__main__":
    unittest.main()
