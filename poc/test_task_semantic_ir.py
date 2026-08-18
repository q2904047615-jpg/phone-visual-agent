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
    SemanticShadowReport,
    SourceSpan,
    TaskSemanticIR,
    TaskSemanticIRError,
    compile_legacy_graph_shadow,
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
        "risk_actions": [
            {
                "risk_id": "send_message",
                "description": "向文件传输助手发送消息“你好”",
                "external_effect": "向指定收件人发送一条消息",
                "risk_type": "message_or_communication",
                "risk_level": "medium",
                "subgoal_ids": ["send_message"],
                "confirmation_required": True,
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
                "risk_action_ids": [],
                "external_impact": "navigation_only",
            },
            {
                "subgoal_id": "send_message",
                "objective": "输入并发送消息“你好”",
                "status": "pending",
                "depends_on": ["open_wechat"],
                "constraints": [constraint],
                "completion_conditions": ["消息“你好”已发送"],
                "completion_evidence": [],
                "risk_action_ids": ["send_message"],
                "external_impact": "external_state",
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
    def test_current_send_failure_projects_to_automatic_typed_effect(self):
        report = compile_legacy_graph_shadow(graph_from_payload())

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

    def test_constraint_wording_never_changes_shadow_risk(self):
        variants = (
            "确认发送对象为文件传输助手",
            "聊天对象必须为文件传输助手",
            "仅向文件传输助手发送指定文字",
        )
        reports = [
            compile_legacy_graph_shadow(
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
            compile_legacy_graph_shadow(graph_from_payload(in_subgoal)),
            compile_legacy_graph_shadow(graph_from_payload(at_graph)),
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
        report = compile_legacy_graph_shadow(graph_from_payload())

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

    def test_shadow_capture_preserves_existing_formal_rejection(self):
        provider = OneResponseProvider(current_send_failure_payload())
        planner = DeepSeekTaskGraphPlanner(provider)

        with self.assertRaises(TaskGraphError):
            planner.plan(RAW_GOAL, device_id="device-local-01", task_id="9abc")

        self.assertEqual(provider.calls, 1)
        self.assertIsNotNone(planner.last_semantic_shadow)
        self.assertEqual(planner.last_semantic_shadow_error, "")
        assert planner.last_semantic_shadow is not None
        self.assertEqual(
            planner.last_semantic_shadow.risk_decisions[0].policy,
            AUTOMATIC,
        )
        self.assertFalse(planner.last_semantic_shadow.execution_allowed)

    def test_current_failure_artifact_contains_non_authoritative_shadow(self):
        planner = DeepSeekTaskGraphPlanner(
            OneResponseProvider(current_send_failure_payload())
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

        shadow = artifact["semantic_shadow"]
        self.assertFalse(shadow["authoritative"])
        self.assertFalse(shadow["execution_allowed"])
        self.assertEqual(
            shadow["risk_decisions"][0]["policy"],
            AUTOMATIC,
        )

    def test_shadow_compiler_error_cannot_replace_formal_error(self):
        provider = OneResponseProvider(current_send_failure_payload())
        planner = DeepSeekTaskGraphPlanner(provider)
        original_compiler = __import__("deepseek_task_graph").compile_legacy_graph_shadow

        def broken_compiler(graph):
            raise RuntimeError("shadow-only failure")

        module = __import__("deepseek_task_graph")
        module.compile_legacy_graph_shadow = broken_compiler
        try:
            with self.assertRaises(TaskGraphError) as caught:
                planner.plan(RAW_GOAL, device_id="device-local-01", task_id="9abc")
        finally:
            module.compile_legacy_graph_shadow = original_compiler

        self.assertNotEqual(str(caught.exception), "shadow-only failure")
        self.assertIsNone(planner.last_semantic_shadow)
        self.assertEqual(planner.last_semantic_shadow_error, "shadow-only failure")

    def test_new_request_clears_stale_shadow_before_json_parse(self):
        planner = DeepSeekTaskGraphPlanner(RawResponseProvider("{"))
        planner.last_semantic_shadow = compile_legacy_graph_shadow(graph_from_payload())
        planner.last_semantic_shadow_error = "stale"

        with self.assertRaises(TaskGraphError):
            planner.plan("新的目标", device_id="device-local-01", task_id="9abc")

        self.assertIsNone(planner.last_semantic_shadow)
        self.assertEqual(planner.last_semantic_shadow_error, "")


if __name__ == "__main__":
    unittest.main()
