import json
import unittest
from pathlib import Path

from agent.application.deepseek_task_graph import DeepSeekTaskGraphPlanner
from agent.domain.task_graph import (
    ControllerTransitionEvidenceRef,
    ObservedState,
    TaskGraphError,
    VerifiedActionTransition,
)


class FakeProvider:
    configured = True

    def __init__(self, *payloads):
        self.payloads = list(payloads)
        self.messages = []

    def chat_json(self, messages, max_tokens=2000):
        self.messages.append(messages)
        return json.dumps(self.payloads.pop(0), ensure_ascii=False)


class RawProvider:
    configured = True

    def __init__(self, raw):
        self.raw = raw

    def chat_json(self, messages, max_tokens=2000):
        return self.raw


def subgoal(subgoal_id, objective, *, depends_on=(), execution_class="navigate", result="目标状态可见"):
    return {
        "subgoal_id": subgoal_id,
        "objective": objective,
        "depends_on": list(depends_on),
        "constraints": [],
        "completion_conditions": [result],
        "execution_class": execution_class,
    }


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


def remaining(*subgoals, questions=()):
    return {"subgoals": list(subgoals), "clarification_questions": list(questions)}


def matched_observation(graph, *, outcome="matched", scene_id="obs-after"):
    receipt = VerifiedActionTransition(
        receipt_id="receipt-step",
        session_id="session-step",
        task_id=graph.task_id,
        device_id=graph.device_id,
        prior_revision=graph.revision,
        subgoal_id=graph.active_subgoal_id,
        decision_node_id="decision-step",
        action_digest="a" * 64,
        rebound_action_digest="b" * 64,
        resolved_action_digest="c" * 64,
        action_kind="tap_semantic",
        before_observation_id="obs-before",
        before_fingerprint="before-fingerprint",
        after_observation_id=scene_id,
        after_fingerprint="after-fingerprint",
        physical_actions=1,
        outcome=outcome,
        errors=() if outcome == "matched" else ("目标状态未出现",),
        controller_transition_evidence=("本轮动作与新画面已经绑定",) if outcome == "matched" else (),
    )
    refs = ()
    visible = ("当前新画面已取得",)
    if outcome == "matched":
        ref = ControllerTransitionEvidenceRef(
            ref_id=f"controller_transition:{receipt.receipt_id}:1",
            receipt_id=receipt.receipt_id,
            subgoal_id=receipt.subgoal_id,
            text=receipt.controller_transition_evidence[0],
        )
        refs = (ref,)
        visible = (ref.ref_id,)
    return ObservedState(
        scene_id=scene_id,
        summary="动作后新画面",
        visible_evidence=visible,
        last_action_outcome=outcome,
        verified_action_transition=receipt,
        controller_transition_evidence_refs=refs,
    )


class SimplePlannerTransportTests(unittest.TestCase):
    def test_latest_all_pending_failure_replays_as_one_local_frontier(self):
        artifact = Path("output/web/generic_supervised_20260831_140234_e497d7d7/") / (
            "deepseek_initial_task_graph_step_1_deepseek_failure.json"
        )
        diagnostic = json.loads(artifact.read_text(encoding="utf-8"))

        graph = DeepSeekTaskGraphPlanner(RawProvider(diagnostic["redacted_raw_response"])).plan(
            "打开微信，给文件传输助手输入aaazjie？你好，然后发送", device_id="phone-1"
        )

        self.assertEqual("open_wechat", graph.active_subgoal_id)
        self.assertEqual(1, sum(item.status == "active" for item in graph.subgoals))
        self.assertEqual("ready", graph.status)

    def test_initial_plan_derives_one_active_subgoal(self):
        raw = initial_plan(subgoals=[
            subgoal("open_app", "打开目标应用"),
            subgoal("open_page", "进入目标页面", depends_on=("open_app",)),
        ])
        graph = DeepSeekTaskGraphPlanner(FakeProvider(raw)).plan("打开目标应用并进入目标页面", device_id="phone-1")

        self.assertEqual("open_app", graph.active_subgoal_id)
        self.assertEqual(["active", "pending"], [item.status for item in graph.subgoals])
        self.assertEqual("ready", graph.status)

    def test_two_runnable_roots_choose_first_plan_item(self):
        raw = initial_plan(subgoals=[
            subgoal("first", "先处理第一个可见目标"),
            subgoal("second", "随后处理第二个可见目标"),
        ])
        graph = DeepSeekTaskGraphPlanner(FakeProvider(raw)).plan("依次处理两个目标", device_id="phone-1")

        self.assertEqual("first", graph.active_subgoal_id)
        self.assertEqual(1, sum(item.status == "active" for item in graph.subgoals))

    def test_retired_runtime_copies_have_no_authority(self):
        raw = initial_plan(subgoals=[
            subgoal("open_app", "打开目标应用"),
            subgoal("open_page", "进入目标页面", depends_on=("open_app",)),
        ])
        raw.update({"status": "completed", "active_subgoal_id": "open_page"})
        raw["completion_conditions"][0].update({"satisfied": True, "evidence": ["模型自写证据"]})
        raw["subgoals"][0].update({"status": "pending", "completion_evidence": ["模型自写证据"],
            "effect_ids": ["模型自写关联"]})
        raw["subgoals"][1].update({"status": "active", "completion_evidence": [], "effect_ids": []})

        graph = DeepSeekTaskGraphPlanner(FakeProvider(raw)).plan("打开目标应用并进入页面", device_id="phone-1")

        self.assertEqual("ready", graph.status)
        self.assertEqual("open_app", graph.active_subgoal_id)
        self.assertFalse(graph.completion_conditions[0].satisfied)
        self.assertEqual((), graph.subgoals[0].completion_evidence)

    def test_effect_reverse_links_and_results_are_derived_once(self):
        effect = {
            "effect_id": "send_message",
            "kind": "send_message",
            "target_entity_roles": ["recipient"],
            "payload_entity_roles": ["input_text"],
            "source_subgoal_ids": ["send"],
            "expected_results": ["模型不再拥有的错误副本"],
        }
        raw = initial_plan(
            entities={"recipient": "文件传输助手", "input_text": "你好"},
            subgoals=[subgoal("send", "向文件传输助手发送你好", execution_class="effect", result="消息你好已发送")],
            effects=[effect],
            objective="向文件传输助手发送你好",
        )
        graph = DeepSeekTaskGraphPlanner(FakeProvider(raw)).plan(raw["goal"]["objective"], device_id="phone-1")

        self.assertEqual(("send_message",), graph.subgoals[0].risk_action_ids)
        self.assertEqual(("消息你好已发送",), graph.risk_actions[0].expected_result_texts)
        self.assertFalse(graph.risk_actions[0].confirmation_required)
        self.assertEqual("ready", graph.status)

    def test_only_authentication_and_payment_require_confirmation(self):
        cases = (
            ("authentication", {"account": "当前账号"}, "account"),
            ("financial_transaction", {"merchant": "商户", "amount": "20", "currency": "CNY"}, "merchant"),
        )
        for kind, entities, target_role in cases:
            with self.subTest(kind=kind):
                effect = {"effect_id": "effect", "kind": kind, "target_entity_roles": [target_role],
                    "payload_entity_roles": [], "source_subgoal_ids": ["step"]}
                raw = initial_plan(entities=entities,
                    subgoals=[subgoal("step", "执行用户明确要求的操作", execution_class="effect",
                        result="用户要求的结果已经出现")], effects=[effect])
                graph = DeepSeekTaskGraphPlanner(FakeProvider(raw)).plan("执行该操作", device_id="phone-1")
                self.assertEqual("awaiting_confirmation", graph.status)
                self.assertTrue(graph.risk_actions[0].confirmation_required)

    def test_matched_action_advances_to_remaining_subgoal(self):
        first = subgoal("open_app", "打开目标应用")
        second = subgoal("open_page", "进入目标页面", depends_on=("open_app",))
        provider = FakeProvider(initial_plan(subgoals=[first, second]), remaining(second))
        planner = DeepSeekTaskGraphPlanner(provider)
        graph = planner.plan("打开目标应用并进入目标页面", device_id="phone-1")

        revised = planner.replan(graph, matched_observation(graph), trigger="action_result_matched",
            reason="打开应用动作已经匹配")

        self.assertEqual("open_page", revised.active_subgoal_id)
        self.assertEqual("completed", next(item for item in revised.subgoals
            if item.subgoal_id == "open_app").status)
        self.assertEqual(2, revised.revision)
        self.assertEqual(1, len(provider.messages))

    def test_matched_final_action_finishes_without_model_status(self):
        provider = FakeProvider(initial_plan(), remaining())
        planner = DeepSeekTaskGraphPlanner(provider)
        graph = planner.plan("完成当前目标", device_id="phone-1")

        revised = planner.replan(graph, matched_observation(graph), trigger="action_result_matched",
            reason="本轮动作已经匹配")

        self.assertEqual("completed", revised.status)
        self.assertIsNone(revised.active_subgoal_id)
        self.assertTrue(revised.completion_conditions[0].satisfied)
        self.assertEqual(1, len(provider.messages))

    def test_mismatch_keeps_runtime_uncompleted_and_uses_new_path(self):
        provider = FakeProvider(initial_plan(), remaining(subgoal("recover", "根据当前新画面换一条路径")))
        planner = DeepSeekTaskGraphPlanner(provider)
        graph = planner.plan("完成当前目标", device_id="phone-1")

        revised = planner.replan(graph, matched_observation(graph, outcome="mismatched"),
            trigger="action_result_mismatch", reason="原动作没有产生预期变化")

        self.assertEqual("recover", revised.active_subgoal_id)
        self.assertFalse(any(item.status == "completed" for item in revised.subgoals))

    def test_visible_completion_uses_current_observation_and_advances(self):
        first = subgoal("inspect", "读取当前可见结果", execution_class="observe")
        second = subgoal("open_next", "进入下一页面", depends_on=("inspect",))
        provider = FakeProvider(initial_plan(subgoals=[first, second]), remaining(second))
        planner = DeepSeekTaskGraphPlanner(provider)
        graph = planner.plan("读取当前结果后进入下一页", device_id="phone-1")
        observation = ObservedState(scene_id="obs-visible", summary="结果已经可见",
            visible_evidence=("当前画面中结果文字完整可见",))

        revised = planner.replan(graph, observation, trigger="subgoal_completed", reason="当前结果已经可见")

        self.assertEqual("open_next", revised.active_subgoal_id)
        self.assertEqual("completed", next(item for item in revised.subgoals
            if item.subgoal_id == "inspect").status)

    def test_wrong_receipt_scope_is_rejected_before_replan_call(self):
        provider = FakeProvider(initial_plan())
        planner = DeepSeekTaskGraphPlanner(provider)
        graph = planner.plan("完成当前目标", device_id="phone-1")
        observation = matched_observation(graph)
        observation = ObservedState(
            scene_id=observation.scene_id,
            summary=observation.summary,
            visible_evidence=observation.visible_evidence,
            last_action_outcome=observation.last_action_outcome,
            verified_action_transition=VerifiedActionTransition(
                **{**observation.verified_action_transition.to_dict(), "task_id": "wrong-task"}),
            controller_transition_evidence_refs=observation.controller_transition_evidence_refs,
        )

        with self.assertRaisesRegex(TaskGraphError, "绑定"):
            planner.replan(graph, observation, trigger="action_result_matched", reason="错误作用域")
        self.assertEqual(0, len(provider.payloads))

    def test_dependency_cycle_is_still_a_structural_error(self):
        raw = initial_plan(subgoals=[
            subgoal("a", "步骤A", depends_on=("b",)),
            subgoal("b", "步骤B", depends_on=("a",)),
        ])
        with self.assertRaisesRegex(TaskGraphError, "依赖形成环"):
            DeepSeekTaskGraphPlanner(FakeProvider(raw)).plan("完成循环计划", device_id="phone-1")

    def test_clarification_blocks_only_when_model_reports_missing_user_information(self):
        raw = initial_plan()
        raw["clarification_questions"] = ["请提供无法推断的唯一收件人"]
        graph = DeepSeekTaskGraphPlanner(FakeProvider(raw)).plan("发送消息", device_id="phone-1")

        self.assertEqual("blocked", graph.status)
        self.assertIsNone(graph.active_subgoal_id)


if __name__ == "__main__":
    unittest.main()
