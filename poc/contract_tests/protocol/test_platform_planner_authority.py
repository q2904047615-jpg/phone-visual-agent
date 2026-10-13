from __future__ import annotations

from types import SimpleNamespace
import unittest

from agent.application.planner import ModelAssistedPlanner
from agent.domain.canonical_action_protocol import GenericStepProposal
from agent.domain.semantic_action import SemanticAction
from agent.domain.tool_intent import ToolIntent, ToolIntentError


class _Scene:
    def validate(self) -> None:
        return None


class _Gateway:
    def __init__(self, decision):
        self.decision = decision
        self.calls = []

    def decide(self, **kwargs):
        self.calls.append(kwargs)
        return self.decision


def _observation():
    return SimpleNamespace(
        device_id="phone-1", observation_id="obs_1234567890123456",
        fingerprint="frame-1", scene=_Scene(),
    )


def _model_decision(status="action"):
    action = None if status == "finish" else SemanticAction(
        node_id="node-1", action="home", params={}
    )
    return SimpleNamespace(
        task_id="task-1", device_id="phone-1", revision=1,
        observation_id="obs_1234567890123456", fingerprint="frame-1",
        proposal=GenericStepProposal(status=status, action=action, reason="当前观察建议"),
        decision_source="model_gateway",
    )


class PlatformPlannerAuthorityTests(unittest.TestCase):
    def test_gateway_suggestion_is_wrapped_before_runtime_consumes_it(self):
        gateway = _Gateway(_model_decision())
        intent = ModelAssistedPlanner(gateway).plan(
            frames=(), task_context=SimpleNamespace(),
            trusted_observation=_observation(), decision_number=1,
            available_action_kinds={"home"}, model_decision={"status": "action"},
        )

        self.assertIsInstance(intent, ToolIntent)
        self.assertEqual("platform_planner", intent.to_dict()["authority"])
        self.assertEqual("home", intent.action.action)
        self.assertEqual("model_gateway", intent.decision_source)
        self.assertEqual("model_gateway", gateway.calls[0]["decision_source"])

    def test_gateway_finish_is_only_a_planner_intent_until_runtime_verifies_it(self):
        intent = ModelAssistedPlanner(_Gateway(_model_decision("finish"))).plan(
            frames=(), task_context=SimpleNamespace(),
            trusted_observation=_observation(), decision_number=1,
            available_action_kinds={"home"}, model_decision={"status": "finish"},
        )

        self.assertEqual("finish", intent.status)
        self.assertIsNone(intent.action)
        self.assertEqual("platform_planner", intent.to_dict()["authority"])

    def test_stale_gateway_suggestion_cannot_be_signed(self):
        decision = _model_decision()
        decision = SimpleNamespace(**{**vars(decision), "fingerprint": "stale-frame"})
        with self.assertRaises(ToolIntentError):
            ModelAssistedPlanner(_Gateway(decision)).plan(
                frames=(), task_context=SimpleNamespace(),
                trusted_observation=_observation(), decision_number=1,
                available_action_kinds={"home"}, model_decision={"status": "action"},
            )


if __name__ == "__main__":
    unittest.main()
