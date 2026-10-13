from __future__ import annotations

from types import SimpleNamespace
import unittest

from agent.application.policy import PlatformPolicy
from agent.domain.canonical_action_protocol import GenericStepProposal
from agent.domain.semantic_action import SemanticAction
from agent.domain.tool_intent import ToolIntent


class _Scene:
    def validate(self) -> None:
        return None


def _observation():
    return SimpleNamespace(
        device_id="phone-1", observation_id="obs_1234567890123456",
        fingerprint="frame-1", scene=_Scene(),
    )


def _intent(action: SemanticAction | None, status: str = "action") -> ToolIntent:
    return ToolIntent(
        task_id="task-1", device_id="phone-1", revision=1,
        observation_id="obs_1234567890123456", fingerprint="frame-1",
        proposal=GenericStepProposal(status=status, action=action, reason="当前观察"),
        model_decision=SimpleNamespace(decision_source="model_gateway"),
    )


class PlatformPolicyTests(unittest.TestCase):
    def test_ordinary_action_is_automatically_allowed(self):
        result = PlatformPolicy().evaluate(
            intent=_intent(SemanticAction(node_id="node-1", action="home", params={})),
            observation=_observation(), history=(),
        )
        self.assertEqual("auto_execute", result.status)
        self.assertTrue(result.allowed)
        self.assertFalse(result.requires_user_confirmation)

    def test_authentication_requires_existing_user_confirmation(self):
        result = PlatformPolicy().evaluate(
            intent=_intent(SemanticAction(node_id="node-1", action="tap_semantic",
                params={"effect_kind": "authentication", "tap_point": (0.5, 0.5)})),
            observation=_observation(), history=(),
        )
        self.assertEqual("requires_user_confirmation", result.status)
        self.assertTrue(result.requires_user_confirmation)
        self.assertEqual(("authentication",), result.effect_ids)

    def test_finish_is_a_terminal_candidate_for_verifier(self):
        result = PlatformPolicy().evaluate(
            intent=_intent(None, status="finish"), observation=_observation(), history=(),
        )
        self.assertEqual("terminal_candidate", result.status)
        self.assertFalse(result.allowed)
        self.assertFalse(result.requires_user_confirmation)


if __name__ == "__main__":
    unittest.main()
