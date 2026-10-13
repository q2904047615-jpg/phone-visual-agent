"""Platform verification boundary for post-action and terminal transitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from agent.domain.tool_intent import ToolIntent
from agent.domain.validation import dataclass_wire, reject_if


VERIFIER_PROTOCOL_VERSION = "2026-10-13-platform-verifier-v1"
VerificationState = Literal["continue", "completed", "needs_reobservation"]


class VerificationError(ValueError):
    """Raised when a tool result cannot be bound to its signed intent."""


@dataclass(frozen=True)
class VerificationResult:
    state: VerificationState
    reason: str
    observation_id: str
    fingerprint: str
    evidence_kind: str
    terminal_status: str | None = None
    verifier_version: str = VERIFIER_PROTOCOL_VERSION

    def to_dict(self) -> dict[str, Any]:
        return dataclass_wire(self)


class PlatformVerifier:
    """Own the only transition from an intent candidate to runtime state."""

    @staticmethod
    def _validate_observation(observation: Any) -> None:
        reject_if(observation is None, VerificationError("Verifier缺少当前Observation。"))
        scene = getattr(observation, "scene", None)
        reject_if(scene is None, VerificationError("Verifier缺少当前Observation的scene。"))
        validator = getattr(scene, "validate", None)
        reject_if(not callable(validator), VerificationError("Verifier收到不可验证的scene。"))
        validator()

    def verify_finish(self, *, intent: ToolIntent, observation: Any,
        history: list[dict[str, Any]] | tuple[dict[str, Any], ...] = ()) -> VerificationResult:
        """Accept a terminal candidate only when it is bound to fresh evidence."""

        self._validate_observation(observation)
        intent.validate(observation)
        reject_if(intent.status != "finish", VerificationError("Verifier只接受finish ToolIntent作为终态候选。"))
        reject_if(not intent.reason.strip(), VerificationError("终态候选缺少当前Observation完成事实。"))
        if (history and self._has_uncertain_tail(history[-1])
                and intent.previous_action_outcome != "matched"):
            return VerificationResult(
                state="needs_reobservation", terminal_status=None,
                observation_id=observation.observation_id, fingerprint=observation.fingerprint,
                evidence_kind="terminal_candidate_deferred",
                reason="历史末步效果仍不确定，Verifier要求重新观察后再判断终态。",
            )
        return VerificationResult(
            state="completed", terminal_status="succeeded",
            observation_id=observation.observation_id, fingerprint=observation.fingerprint,
            evidence_kind="observation_bound_finish", reason="当前Observation与终态ToolIntent已绑定。",
        )

    def verify_action_execution(self, *, intent: ToolIntent, before: Any, after: Any,
        result: Any) -> VerificationResult:
        """Verify one executed tool result before the next planning cycle."""

        self._validate_observation(before)
        self._validate_observation(after)
        intent.validate(before)
        reject_if(intent.status != "action" or intent.action is None,
            VerificationError("动作结果没有绑定action ToolIntent。"))
        reject_if(str(getattr(result, "action_outcome", "")) != "executed",
            VerificationError("执行结果没有executed回执。"))
        errors = tuple(str(item) for item in getattr(result, "verification_errors", ()) or () if str(item).strip())
        reject_if(errors, VerificationError("执行结果包含Verifier错误。"))
        reject_if(int(getattr(result, "physical_actions", 0)) not in {0, 1},
            VerificationError("执行结果的物理动作数无效。"))
        reject_if(str(getattr(after, "device_id", "")) != intent.device_id,
            VerificationError("动作后Observation不属于当前设备。"))
        return VerificationResult(
            state="continue", terminal_status=None,
            observation_id=after.observation_id, fingerprint=after.fingerprint,
            evidence_kind="tool_result_observed", reason="工具结果与动作后Observation已绑定。",
        )

    @staticmethod
    def _has_uncertain_tail(entry: dict[str, Any]) -> bool:
        execution = entry.get("execution") if isinstance(entry, dict) else None
        return (str((execution or {}).get("action_outcome") or "") == "uncertain"
            or str(entry.get("visual_outcome") or "") == "uncertain")


__all__ = ["VERIFIER_PROTOCOL_VERSION", "PlatformVerifier", "VerificationError", "VerificationResult"]
