"""Platform policy boundary for signed tool intents.

The policy engine does not invent product restrictions.  It projects the
approved risk contract onto one Planner-issued intent: ordinary phone actions
may proceed automatically, while authentication and financial transactions
require the existing user confirmation path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from agent.domain.qwen_task_context import CONFIRMATION_EFFECT_KINDS, action_effect_kind
from agent.domain.tool_intent import ToolIntent
from agent.domain.validation import dataclass_wire, reject_if


POLICY_PROTOCOL_VERSION = "2026-10-13-platform-policy-v1"
PolicyStatus = Literal["auto_execute", "requires_user_confirmation", "terminal_candidate"]


class PolicyViolation(ValueError):
    """Raised when a signed intent cannot enter the platform execution path."""


@dataclass(frozen=True)
class PolicyDecision:
    """The sole policy result consumed by the runtime for one ToolIntent."""

    status: PolicyStatus
    action_kind: str | None
    effect_ids: tuple[str, ...]
    reason: str
    policy_version: str = POLICY_PROTOCOL_VERSION

    @property
    def allowed(self) -> bool:
        return self.status in {"auto_execute", "requires_user_confirmation"}

    @property
    def requires_user_confirmation(self) -> bool:
        return self.status == "requires_user_confirmation"

    def to_dict(self) -> dict[str, Any]:
        return dataclass_wire(self) | {
            "allowed": self.allowed,
            "requires_user_confirmation": self.requires_user_confirmation,
        }


class PlatformPolicy:
    """Apply the approved risk contract to one current-observation intent."""

    def evaluate(self, *, intent: ToolIntent, observation: Any, history: list[dict[str, Any]] | tuple[dict[str, Any], ...] = ()) -> PolicyDecision:
        del history  # History is evidence for Verifier; Policy only gates this intent.
        intent.validate(observation)
        if intent.status == "finish":
            return PolicyDecision(
                status="terminal_candidate", action_kind=None, effect_ids=(),
                reason="终态候选已绑定当前 ToolIntent，交由 Verifier 判断。",
            )

        action = intent.action
        reject_if(action is None, PolicyViolation("Policy收到没有动作载荷的非终态ToolIntent。"))
        effect = action_effect_kind(action)
        effect_ids = (effect,) if effect else ()
        if effect in CONFIRMATION_EFFECT_KINDS:
            return PolicyDecision(
                status="requires_user_confirmation", action_kind=action.action,
                effect_ids=effect_ids, reason="当前动作属于已批准的登录或付款确认范围。",
            )
        return PolicyDecision(
            status="auto_execute", action_kind=action.action, effect_ids=effect_ids,
            reason="当前动作属于用户已授权的普通手机操作。",
        )


__all__ = ["POLICY_PROTOCOL_VERSION", "PlatformPolicy", "PolicyDecision", "PolicyViolation"]
