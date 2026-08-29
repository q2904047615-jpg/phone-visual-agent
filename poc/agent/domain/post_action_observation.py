"""Typed domain context for the first observation after one action."""

from __future__ import annotations

from .validation import ValidatedDataclassWire, reject_if
from dataclasses import dataclass
from typing import Any

from agent.domain.canonical_action_kinds import CANONICAL_ACTION_KINDS
from agent.domain.canonical_action_protocol import CanonicalActionProtocolError, StateExpectation
from agent.domain.vision_model import VisionAgentError


POST_ACTION_VISUAL_CONTEXT_VERSION = '2026-08-25-local-post-action-visual-context-v1'
POST_NAVIGATION_RESULT_OBSERVATION_PHASE = "verified_navigation_result_v1"
POST_NAVIGATION_RESULT_OBJECTIVE = "观察本次导航后的当前稳定画面"
POST_NAVIGATION_RESULT_COMPLETION_CONDITIONS = ["当前稳定结果画面已被重新观察"]


@dataclass(frozen=True)
class PostActionVisualContext(ValidatedDataclassWire):
    """One executed canonical action awaiting fresh visual verification."""

    canonical_action_kind: str
    expected_postconditions: tuple[StateExpectation, ...]
    protocol_version: str = POST_ACTION_VISUAL_CONTEXT_VERSION
    execution_state: str = "physical_action_executed"
    outcome: str = "pending_visual_verification"

    def validate(self) -> None:
        reject_if(self.protocol_version != POST_ACTION_VISUAL_CONTEXT_VERSION, VisionAgentError("动作后视觉上下文协议版本无效。"))
        reject_if(self.execution_state != 'physical_action_executed', VisionAgentError("动作后视觉上下文没有证明物理动作已执行。"))
        reject_if(self.outcome != 'pending_visual_verification', VisionAgentError("动作后视觉上下文不得提前声明动作匹配结果。"))
        reject_if(self.canonical_action_kind not in CANONICAL_ACTION_KINDS, VisionAgentError("动作后视觉上下文包含非canonical动作。"))
        reject_if(not 1 <= len(self.expected_postconditions) <= 16, VisionAgentError("动作后视觉上下文必须包含1..16个typed后置条件。"))
        for expectation in self.expected_postconditions:
            expectation.validate()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> 'PostActionVisualContext':
        reject_if(
            not isinstance(value, dict) or set(value) != {'protocol_version', 'execution_state', 'outcome',
            'canonical_action_kind', 'expected_postconditions'},
            VisionAgentError("动作后视觉上下文结构无效。"),
        )
        raw_expectations = value.get("expected_postconditions")
        reject_if(not isinstance(raw_expectations, list), VisionAgentError("动作后视觉上下文的typed后置条件必须是数组。"))
        expectations: list[StateExpectation] = []
        for item in raw_expectations:
            try:
                expectations.append(StateExpectation.from_dict(item))
            except CanonicalActionProtocolError as exc:
                raise VisionAgentError(f"动作后视觉上下文包含无效typed后置条件：{exc}") from exc
        context = cls(protocol_version=str(value.get('protocol_version') or ''),
            execution_state=str(value.get('execution_state') or ''), outcome=str(value.get('outcome') or ''),
            canonical_action_kind=str(value.get('canonical_action_kind') or ''),
            expected_postconditions=tuple(expectations))
        context.validate()
        return context


__all__ = ['POST_ACTION_VISUAL_CONTEXT_VERSION', 'POST_NAVIGATION_RESULT_COMPLETION_CONDITIONS',
    'POST_NAVIGATION_RESULT_OBJECTIVE', 'POST_NAVIGATION_RESULT_OBSERVATION_PHASE', 'PostActionVisualContext']
