"""Low-level shared helpers for the universal agent orchestration layer."""

from __future__ import annotations

from typing import Any

from agent.application.action_adapter import GenericActionAdapterError
from agent.domain.validation import canonical_digest, reject_if


class UniversalAgentOrchestratorError(RuntimeError):
    pass


_STALE_FRAME_FAILURE_PREFIXES = (
    '确认时前台 App 已变化',
    '确认时页面已变化',
    '当前新截图不再包含 Qwen 已选',
    '确认时目标区域已明显移动',
)


def _is_zero_action_stale_frame_fault(exc: GenericActionAdapterError) -> bool:
    return exc.physical_actions == 0 and str(exc).startswith(_STALE_FRAME_FAILURE_PREFIXES)


def _is_zero_action_reobservation_fault(exc: GenericActionAdapterError) -> bool:
    """Return whether the failed attempt can be retried with a fresh observation."""

    return _is_zero_action_stale_frame_fault(exc) or bool(exc.observation_errors)


def _action_digest(action: Any) -> str:
    reject_if(action is None, UniversalAgentOrchestratorError("动作摘要缺少语义动作。"))
    payload = action.to_dict() if callable(getattr(action, 'to_dict', None)) else action
    return canonical_digest(payload)
