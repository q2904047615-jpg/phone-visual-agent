import copy
import json
import unittest

from deepseek_task_graph import (
    DeepSeekTaskGraphPlanner,
    ObservedState,
    TaskGraphError,
    _named_visual_identity_anchor,
    named_visual_identity_is_grounded,
)


class FakeProvider:
    configured = True

    def __init__(self, *payloads):
        self.payloads = list(payloads)
        self.messages = []

    def chat_json(self, messages, max_tokens=2000):
        self.messages.append(messages)
        return json.dumps(self.payloads.pop(0), ensure_ascii=False)


def payload(*, objective="进入普通会话并聚焦空输入框", effects=()):
    effect_ids = [item["effect_id"] for item in effects]
    execution_class = "effect" if effect_ids else "navigate"
    return {
        "status": "ready",
        "goal": {
            "objective": objective,
            "target_apps": [{"app_id": "chat", "app_name": "聊天应用"}],
            "entities": {},
        },
        "constraints": ["不得发送、删除或清空任何内容"],
        "completion_conditions": [
            {
                "condition_id": "done",
                "description": "目标状态可见",
                "evidence_required": ["目标状态可见"],
                "satisfied": False,
                "evidence": [],
            }
        ],
        "effect_intents": list(effects),
        "subgoals": [
            {
                "subgoal_id": "step",
                "objective": objective,
                "status": "active",
                "depends_on": [],
                "constraints": ["不得发送、删除或清空任何内容"],
                "completion_conditions": ["目标状态可见"],
                "completion_evidence": [],
                "effect_ids": effect_ids,
                "execution_class": execution_class,
            }
        ],
        "active_subgoal_id": "step",
        "clarification_questions": [],
    }


class TypedPlannerTransportTests(unittest.TestCase):
    def test_functional_page_modifiers_are_not_treated_as_page_names(self):
        for text in (
            "可输入搜索内容的页面可见",
            "用于填写订单编号的界面可见",
            "能够编辑草稿的视图出现",
            "editable input page is visible",
        ):
            with self.subTest(text=text):
                self.assertEqual("", _named_visual_identity_anchor((text,)))
                self.assertTrue(named_visual_identity_is_grounded((text,), ()))

    def test_real_named_pages_still_require_structured_identity(self):
        unrelated_facts = ('{"app_id":"settings","screen_id":"main"}',)
        for text in (
            "系统设置搜索页面可见",
            "订单详情页面可见",
            "张三的聊天页面可见",
        ):
            with self.subTest(text=text):
                self.assertNotEqual("", _named_visual_identity_anchor((text,)))
                self.assertFalse(
                    named_visual_identity_is_grounded((text,), unrelated_facts)
                )

    def test_forbidden_effect_words_do_not_create_or_block_navigation(self):
        raw = payload()
        graph = DeepSeekTaskGraphPlanner(FakeProvider(raw)).plan(
            raw["goal"]["objective"],
            device_id="phone-1",
        )
        self.assertEqual((), graph.risk_actions)
        self.assertEqual("navigation_only", graph.subgoals[0].external_impact)

    def test_retired_risk_fields_are_rejected_at_transport_boundary(self):
        raw = payload()
        raw["risk_actions"] = []
        with self.assertRaisesRegex(TaskGraphError, "退役协议字段|协议外字段"):
            DeepSeekTaskGraphPlanner(FakeProvider(raw)).plan(
                raw["goal"]["objective"],
                device_id="phone-1",
            )

    def test_send_effect_is_typed_and_automatic(self):
        raw = payload(
            objective="向文件传输助手发送你好",
            effects=(
                {
                    "effect_id": "send_message",
                    "kind": "send_message",
                    "target_entity_roles": ["recipient"],
                    "payload_entity_roles": ["input_text"],
                    "source_subgoal_ids": ["step"],
                    "expected_results": ["消息你好已发送给文件传输助手"],
                },
            ),
        )
        raw["goal"]["entities"] = {
            "recipient": "文件传输助手",
            "input_text": "你好",
        }
        raw["subgoals"][0]["completion_conditions"] = [
            "消息你好已发送给文件传输助手"
        ]
        graph = DeepSeekTaskGraphPlanner(FakeProvider(raw)).plan(
            raw["goal"]["objective"],
            device_id="phone-1",
        )
        self.assertEqual("send_message", graph.risk_actions[0].effect_kind)
        self.assertFalse(graph.risk_actions[0].confirmation_required)
        self.assertEqual("ready", graph.status)

    def test_financial_effect_uses_local_confirmation_policy(self):
        raw = payload(
            objective="向商户付款20元",
            effects=(
                {
                    "effect_id": "pay_merchant",
                    "kind": "financial_transaction",
                    "target_entity_roles": ["merchant"],
                    "payload_entity_roles": ["amount", "currency"],
                    "source_subgoal_ids": ["step"],
                    "expected_results": ["已向商户付款20元"],
                },
            ),
        )
        raw["goal"]["entities"] = {
            "merchant": "商户",
            "amount": "20",
            "currency": "CNY",
        }
        raw["subgoals"][0]["completion_conditions"] = ["已向商户付款20元"]
        graph = DeepSeekTaskGraphPlanner(FakeProvider(raw)).plan(
            raw["goal"]["objective"],
            device_id="phone-1",
        )
        self.assertTrue(graph.risk_actions[0].confirmation_required)
        self.assertEqual("awaiting_confirmation", graph.status)

    def test_effect_expected_result_cannot_be_a_forbidden_state(self):
        raw = payload(
            effects=(
                {
                    "effect_id": "send_message",
                    "kind": "send_message",
                    "target_entity_roles": [],
                    "payload_entity_roles": [],
                    "source_subgoal_ids": ["step"],
                    "expected_results": ["保持原草稿不变且不得发送"],
                },
            )
        )
        raw["subgoals"][0]["completion_conditions"] = [
            "保持原草稿不变且不得发送"
        ]
        with self.assertRaisesRegex(TaskGraphError, "没有证明效果类型|不能声明"):
            DeepSeekTaskGraphPlanner(FakeProvider(raw)).plan(
                raw["goal"]["objective"],
                device_id="phone-1",
            )

    def test_effect_role_must_exist_in_user_entities(self):
        raw = payload(
            objective="发送消息",
            effects=(
                {
                    "effect_id": "send_message",
                    "kind": "send_message",
                    "target_entity_roles": ["recipient"],
                    "payload_entity_roles": ["input_text"],
                    "source_subgoal_ids": ["step"],
                    "expected_results": ["消息已发送"],
                },
            ),
        )
        raw["subgoals"][0]["completion_conditions"] = ["消息已发送"]
        with self.assertRaisesRegex(TaskGraphError, "不存在的 goal.entities"):
            DeepSeekTaskGraphPlanner(FakeProvider(raw)).plan(
                raw["goal"]["objective"],
                device_id="phone-1",
            )

    def test_model_cannot_choose_confirmation_status(self):
        raw = payload()
        raw["status"] = "awaiting_confirmation"
        with self.assertRaisesRegex(TaskGraphError, "模型不得决定 awaiting_confirmation"):
            DeepSeekTaskGraphPlanner(FakeProvider(raw)).plan(
                raw["goal"]["objective"],
                device_id="phone-1",
            )

    def test_prompt_contains_only_new_effect_transport(self):
        raw = payload()
        provider = FakeProvider(copy.deepcopy(raw))
        DeepSeekTaskGraphPlanner(provider).plan(
            raw["goal"]["objective"],
            device_id="phone-1",
        )
        prompt = provider.messages[0][0]["content"]
        self.assertIn('"effect_intents"', prompt)
        self.assertNotIn('"risk_actions"', prompt)
        self.assertNotIn('"risk_action_ids"', prompt)
        self.assertNotIn('"external_impact"', prompt)
        self.assertNotIn('"confirmation_required"', prompt)

    def test_natural_action_words_are_valid_but_direct_control_is_not(self):
        natural = payload(objective="点击设置入口后滑动列表并返回首页")
        graph = DeepSeekTaskGraphPlanner(FakeProvider(natural)).plan(
            natural["goal"]["objective"],
            device_id="phone-1",
        )
        self.assertEqual(natural["goal"]["objective"], graph.goal.objective)

        for objective in (
            "点击归一化坐标0.5,0.5",
            "执行 adb shell input tap 10 20",
            "运行 PowerShell 调用机械臂",
        ):
            with self.subTest(objective=objective):
                raw = payload(objective=objective)
                with self.assertRaisesRegex(TaskGraphError, "越权执行细节"):
                    DeepSeekTaskGraphPlanner(FakeProvider(raw)).plan(
                        objective,
                        device_id="phone-1",
                    )

    def test_cross_app_goal_preserves_all_typed_surfaces(self):
        raw = payload(objective="从首页打开工具甲，再查看工具乙中的结果")
        raw["goal"]["target_apps"] = [
            {"app_id": "tool-a", "app_name": "工具甲"},
            {"app_id": "tool-b", "app_name": "工具乙"},
        ]
        graph = DeepSeekTaskGraphPlanner(FakeProvider(raw)).plan(
            raw["goal"]["objective"],
            device_id="phone-1",
        )
        self.assertEqual(
            ("tool-a", "tool-b"),
            tuple(item.app_id for item in graph.goal.target_apps),
        )

    def test_dependency_cycle_and_ambiguous_frontier_fail_closed(self):
        cycle = payload()
        first = cycle["subgoals"][0]
        cycle["subgoals"] = [
            {**copy.deepcopy(first), "subgoal_id": "a", "depends_on": ["b"]},
            {**copy.deepcopy(first), "subgoal_id": "b", "depends_on": ["a"], "status": "pending"},
        ]
        cycle["active_subgoal_id"] = "a"
        with self.assertRaisesRegex(TaskGraphError, "依赖.*环|循环"):
            DeepSeekTaskGraphPlanner(FakeProvider(cycle)).plan(
                cycle["goal"]["objective"],
                device_id="phone-1",
            )

        ambiguous = payload()
        first = ambiguous["subgoals"][0]
        ambiguous["subgoals"] = [
            {**copy.deepcopy(first), "subgoal_id": "a", "status": "pending"},
            {**copy.deepcopy(first), "subgoal_id": "b", "status": "pending"},
        ]
        ambiguous["active_subgoal_id"] = None
        with self.assertRaisesRegex(TaskGraphError, "唯一|活动"):
            DeepSeekTaskGraphPlanner(FakeProvider(ambiguous)).plan(
                ambiguous["goal"]["objective"],
                device_id="phone-1",
            )

    def test_replan_preserves_identity_goal_constraints_and_exact_revision(self):
        initial = payload()
        candidate = copy.deepcopy(initial)
        provider = FakeProvider(copy.deepcopy(initial), candidate)
        planner = DeepSeekTaskGraphPlanner(provider)
        graph = planner.plan(
            initial["goal"]["objective"],
            device_id="phone-1",
            task_id="task-fixed",
        )
        revised = planner.replan(
            graph,
            ObservedState(
                scene_id="scene-2",
                summary="页面发生变化",
                visible_evidence=("目标仍未完成",),
            ),
            trigger="observation_changed",
            reason="当前画面已变化",
        )
        self.assertEqual("task-fixed", revised.task_id)
        self.assertEqual("phone-1", revised.device_id)
        self.assertEqual(2, revised.revision)
        self.assertEqual(graph.goal, revised.goal)
        self.assertEqual(graph.constraints, revised.constraints)

    def test_replan_cannot_mutate_goal_or_delete_constraint(self):
        for mutation, expected in (
            (lambda item: item["goal"].update(objective="替换后的目标"), "目标"),
            (lambda item: item.update(constraints=[]), "约束"),
        ):
            with self.subTest(expected=expected):
                initial = payload()
                candidate = copy.deepcopy(initial)
                mutation(candidate)
                planner = DeepSeekTaskGraphPlanner(
                    FakeProvider(copy.deepcopy(initial), candidate)
                )
                graph = planner.plan(
                    initial["goal"]["objective"],
                    device_id="phone-1",
                )
                observation = ObservedState(
                    scene_id="scene-2",
                    summary="页面发生变化",
                    visible_evidence=("目标仍未完成",),
                )
                with self.assertRaisesRegex(TaskGraphError, expected):
                    planner.replan(
                        graph,
                        observation,
                        trigger="observation_changed",
                        reason="当前画面已变化",
                    )

    def test_replan_cannot_delete_or_retype_existing_effect(self):
        effect = {
            "effect_id": "send_message",
            "kind": "send_message",
            "target_entity_roles": ["recipient"],
            "payload_entity_roles": ["input_text"],
            "source_subgoal_ids": ["step"],
            "expected_results": ["消息已发送"],
        }
        initial = payload(objective="向收件人发送正文", effects=(effect,))
        initial["goal"]["entities"] = {"recipient": "收件人", "input_text": "正文"}
        initial["subgoals"][0]["completion_conditions"] = ["消息已发送"]
        for candidate in (
            {**copy.deepcopy(initial), "effect_intents": []},
            copy.deepcopy(initial),
        ):
            if candidate["effect_intents"]:
                candidate["effect_intents"][0]["kind"] = "publish_content"
            with self.subTest(candidate=candidate["effect_intents"]):
                planner = DeepSeekTaskGraphPlanner(
                    FakeProvider(copy.deepcopy(initial), candidate)
                )
                graph = planner.plan(
                    initial["goal"]["objective"],
                    device_id="phone-1",
                )
                with self.assertRaisesRegex(TaskGraphError, "[Ee]ffect|效果"):
                    planner.replan(
                        graph,
                        ObservedState(
                            scene_id="scene-2",
                            summary="页面发生变化",
                            visible_evidence=("目标仍未完成",),
                        ),
                        trigger="observation_changed",
                        reason="当前画面已变化",
                    )

    def test_invalid_json_is_not_retried_remotely(self):
        class InvalidProvider:
            configured = True

            def __init__(self):
                self.calls = 0

            def chat_json(self, messages, max_tokens=2000):
                del messages, max_tokens
                self.calls += 1
                return "{invalid"

        provider = InvalidProvider()
        with self.assertRaises(TaskGraphError):
            DeepSeekTaskGraphPlanner(provider).plan(
                "查看当前页面",
                device_id="phone-1",
            )
        self.assertEqual(1, provider.calls)


if __name__ == "__main__":
    unittest.main()
