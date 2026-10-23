from __future__ import annotations

from types import SimpleNamespace
import unittest

from agent.application.verifier import PlatformVerifier
from agent.domain.canonical_action_protocol import GenericStepProposal
from agent.domain.semantic_action import SemanticAction
from agent.domain.task_plan import ConditionResult, TaskPlan
from agent.domain.tool_intent import ToolIntent


class _Scene:
    def validate(self) -> None:
        return None


def _observation(suffix: str = "1"):
    return SimpleNamespace(
        device_id="phone-1", observation_id=f"obs_{suffix * 16}",
        fingerprint=f"frame-{suffix}", scene=_Scene(),
    )


_MET = (ConditionResult("c1", "met", "设置页蓝牙开关为开"), ConditionResult("c2", "met", "标题为蓝牙"))


def _plan() -> TaskPlan:
    return TaskPlan.commit(task_id="task-1", device_id="phone-1", goal="打开蓝牙设置并开启蓝牙", revision=1,
        proposed_conditions=("蓝牙开关显示为开启", "当前页面标题为蓝牙"))


def _intent(observation, status: str = "finish", results=_MET) -> ToolIntent:
    action = None if status == "finish" else SemanticAction(node_id="node-1", action="home", params={})
    return ToolIntent(
        task_id="task-1", device_id=observation.device_id, revision=1,
        observation_id=observation.observation_id, fingerprint=observation.fingerprint,
        proposal=GenericStepProposal(status=status, action=action, reason="当前观察完成事实"),
        model_decision=SimpleNamespace(decision_source="model_gateway"),
        condition_results=results if status == "finish" else None,
    )


class PlatformVerifierTests(unittest.TestCase):
    def test_finish_requires_observation_bound_intent_and_returns_completed(self):
        observation = _observation()
        plan = _plan()
        result = PlatformVerifier().verify_finish(intent=_intent(observation), observation=observation, history=(),
            plan=plan)
        self.assertEqual("completed", result.state)
        self.assertEqual("succeeded", result.terminal_status)
        self.assertEqual("plan_conditions_met", result.evidence_kind)
        self.assertEqual((plan.plan_id, 1), (result.plan_id, result.plan_version))
        self.assertEqual(observation.observation_id, result.observation_id)

    def test_finish_without_condition_report_is_not_completion(self):
        observation = _observation()
        result = PlatformVerifier().verify_finish(intent=_intent(observation, results=None),
            observation=observation, plan=_plan())
        self.assertEqual("needs_reobservation", result.state)
        self.assertIsNone(result.terminal_status)
        self.assertEqual("plan_conditions_unmet", result.evidence_kind)

    def test_each_committed_condition_must_be_met_with_visible_evidence(self):
        observation = _observation()
        variants = {
            "missing": (ConditionResult("c1", "met", "开关为开"),),
            "unknown": (ConditionResult("c1", "met", "开关为开"), ConditionResult("c2", "unknown", "看不清标题")),
            "not_met": (ConditionResult("c1", "not_met", "开关为关"), ConditionResult("c2", "met", "标题为蓝牙")),
            "no_evidence": (ConditionResult("c1", "met", ""), ConditionResult("c2", "met", "标题为蓝牙")),
            "foreign_id_only": (ConditionResult("c9", "met", "无关"), ConditionResult("c2", "met", "标题")),
        }
        for name, results in variants.items():
            with self.subTest(name):
                result = PlatformVerifier().verify_finish(intent=_intent(observation, results=results),
                    observation=observation, plan=_plan())
                self.assertEqual("needs_reobservation", result.state)
                self.assertTrue(result.unmet)

    def test_finish_requires_a_committed_plan(self):
        observation = _observation()
        with self.assertRaises(Exception):
            PlatformVerifier().verify_finish(intent=_intent(observation), observation=observation, plan=None)

    def test_uncertain_tail_defers_terminal_transition(self):
        observation = _observation()
        result = PlatformVerifier().verify_finish(intent=_intent(observation), observation=observation,
            history=({"execution": {"action_outcome": "uncertain"}, "visual_outcome": "uncertain"},), plan=_plan())
        self.assertEqual("needs_reobservation", result.state)
        self.assertIsNone(result.terminal_status)

    def test_executed_action_is_verified_before_next_plan(self):
        before = _observation("1")
        after = _observation("2")
        result = SimpleNamespace(action_outcome="executed", verification_errors=(), physical_actions=1)
        verification = PlatformVerifier().verify_action_execution(
            intent=_intent(before, status="action"), before=before, after=after, result=result,
        )
        self.assertEqual("continue", verification.state)
        self.assertEqual("tool_result_observed", verification.evidence_kind)


if __name__ == "__main__":
    unittest.main()
