"""Platform-owned intent produced after model-assisted planning.

The model gateway may suggest an action or a terminal observation.  This
domain object is the only value that the runtime is allowed to hand to Policy
and Executor.  Keeping the model response alongside the signed intent makes
the migration auditable without granting the model a second execution path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent.domain.canonical_action_protocol import GenericStepProposal
from agent.domain.validation import reject_if


class ToolIntentError(ValueError):
    """Raised when a platform intent is stale, unbound, or malformed."""


@dataclass(frozen=True)
class ToolIntent:
    """A Planner-issued action or terminal intent bound to one observation."""

    task_id: str
    device_id: str
    revision: int
    observation_id: str
    fingerprint: str
    proposal: GenericStepProposal
    model_decision: Any
    planner_id: str = "model_assisted_planner_v1"
    trusted_observation: Any = field(default=None, repr=False, compare=False)

    @classmethod
    def from_model_decision(cls, decision: Any) -> "ToolIntent":
        proposal = getattr(decision, "proposal", None)
        reject_if(proposal is None, ToolIntentError("Planner输入缺少模型观察建议。"))
        return cls(
            task_id=str(decision.task_id),
            device_id=str(decision.device_id),
            revision=int(decision.revision),
            observation_id=str(decision.observation_id),
            fingerprint=str(decision.fingerprint),
            proposal=proposal,
            model_decision=decision,
            trusted_observation=getattr(decision, "trusted_observation", None),
        )

    @property
    def status(self) -> str:
        return self.proposal.status

    @property
    def action(self) -> Any:
        return self.proposal.action

    @property
    def reason(self) -> str:
        return self.proposal.reason

    @property
    def previous_action_outcome(self) -> str | None:
        return getattr(self.model_decision, "previous_action_outcome", None)

    @property
    def decision_source(self) -> str:
        return str(getattr(self.model_decision, "decision_source", "model_gateway"))

    def validate(self, observation: Any) -> None:
        expected = (
            str(getattr(observation, "task_id", self.task_id)),
            self.device_id,
            self.revision,
            self.observation_id,
            self.fingerprint,
        )
        actual = (
            self.task_id,
            str(getattr(observation, "device_id", "")),
            self.revision,
            str(getattr(observation, "observation_id", "")),
            str(getattr(observation, "fingerprint", "")),
        )
        reject_if(actual != expected, ToolIntentError("ToolIntent没有绑定当前task/device/observation。"))
        self.proposal.validate(observation.scene)

    def to_dict(self) -> dict[str, Any]:
        return {
            "authority": "platform_planner",
            "planner_id": self.planner_id,
            "task_id": self.task_id,
            "device_id": self.device_id,
            "revision": self.revision,
            "observation_id": self.observation_id,
            "fingerprint": self.fingerprint,
            "status": self.status,
            "proposal": self.proposal.to_dict(),
            "decision_source": self.decision_source,
        }


__all__ = ["ToolIntent", "ToolIntentError"]
