from __future__ import annotations

from types import SimpleNamespace
import unittest

from agent.application.verifier import PlatformVerifier
from agent.domain.canonical_action_protocol import GenericStepProposal
from agent.domain.semantic_action import SemanticAction
from agent.domain.tool_intent import ToolIntent


class _Scene:
    def validate(self) -> None:
        return None


def _observation(suffix: str = "1"):
    return SimpleNamespace(
        device_id="phone-1", observation_id=f"obs_{suffix * 16}",
        fingerprint=f"frame-{suffix}", scene=_Scene(),
    )


def _intent(observation, status: str = "finish") -> ToolIntent:
    action = None if status == "finish" else SemanticAction(node_id="node-1", action="home", params={})
    return ToolIntent(
        task_id="task-1", device_id=observation.device_id, revision=1,
        observation_id=observation.observation_id, fingerprint=observation.fingerprint,
        proposal=GenericStepProposal(status=status, action=action, reason="当前观察完成事实"),
        model_decision=SimpleNamespace(decision_source="model_gateway"),
    )


class PlatformVerifierTests(unittest.TestCase):
    def test_finish_requires_observation_bound_intent_and_returns_completed(self):
        observation = _observation()
        result = PlatformVerifier().verify_finish(intent=_intent(observation), observation=observation, history=())
        self.assertEqual("completed", result.state)
        self.assertEqual("succeeded", result.terminal_status)
        self.assertEqual("observation_bound_finish", result.evidence_kind)

    def test_uncertain_tail_defers_terminal_transition(self):
        observation = _observation()
        result = PlatformVerifier().verify_finish(intent=_intent(observation), observation=observation,
            history=({"execution": {"action_outcome": "uncertain"}, "visual_outcome": "uncertain"},))
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
