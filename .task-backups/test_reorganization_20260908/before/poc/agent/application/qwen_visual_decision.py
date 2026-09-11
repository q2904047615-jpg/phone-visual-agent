"""Bind the action or finish emitted by the sole Qwen scene response."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import json
import time
from typing import Any, Callable, Iterable

from PIL import Image

from agent.domain.canonical_action_kinds import CANONICAL_ACTION_KINDS
from agent.domain.canonical_action_protocol import (
    CanonicalActionProtocolError,
    GenericStepProposal,
    bind_same_response_action,
    normalize_model_step_decision,
)
import agent.domain.qwen_task_context as qwen_task_context_domain
from agent.domain.semantic_action import SemanticAction
from agent.domain.recent_navigation import RECENT_NAVIGATION_PROTOCOL, LOCAL_NAVIGATION_SOURCE
from agent.domain.text_transport import TextTransportProfile
import agent.domain.trusted_observation as trusted_observation_domain
from agent.domain.validation import reject_if
from agent.domain.vision_model import VisionAgentError, public_model_identity


QWEN_VISUAL_DECISION_PROTOCOL_VERSION = "2026-09-06-qwen-whole-task-v19"
QWEN_VISUAL_DECISION_MODEL_ROLE = "single_response_scene_action_or_finish"
QWEN_PROTOCOL_ACTIONS = frozenset(CANONICAL_ACTION_KINDS)


@dataclass(frozen=True)
class QwenVisualDecision:
    task_id: str
    device_id: str
    revision: int
    observation_id: str
    fingerprint: str
    trusted_observation: trusted_observation_domain.TrustedObservation
    proposal: GenericStepProposal
    previous_action_outcome: str | None = None
    protocol_version: str = QWEN_VISUAL_DECISION_PROTOCOL_VERSION
    decision_source: str = 'qwen_same_response_decision'

    def validate(self, context: qwen_task_context_domain.QwenTaskContext) -> None:
        expected_scope = (context.task_id, context.device_id, context.revision,
            self.trusted_observation.observation_id, self.trusted_observation.fingerprint)
        reject_if((self.task_id, self.device_id, self.revision, self.observation_id,
            self.fingerprint) != expected_scope,
            CanonicalActionProtocolError("task/device/revision/observation/fingerprint 已过期或不匹配。"))
        self.proposal.validate(self.trusted_observation.scene)
        reject_if(self.protocol_version != QWEN_VISUAL_DECISION_PROTOCOL_VERSION,
            CanonicalActionProtocolError("Qwen视觉决策协议版本无效。"))
    def to_dict(self) -> dict[str, Any]:
        return {'protocol_version': self.protocol_version, 'task_id': self.task_id, 'device_id': self.device_id,
            'revision': self.revision, 'observation_id': self.observation_id, 'fingerprint': self.fingerprint,
            'decision_source': self.decision_source,
            'status': self.proposal.status, 'previous_action_outcome': self.previous_action_outcome, 'next_action': self.proposal.action.to_dict()
            if self.proposal.action else None, 'reason': self.proposal.reason}


class QwenVisualDecisionObserver:
    """Bind Qwen's one current-frame decision; never build or choose another candidate."""

    def __init__(self, provider: Any, *, trusted_observation_frame_validator: Callable[..., None]) -> None:
        self.provider = provider
        self.trusted_observation_frame_validator = trusted_observation_frame_validator
        self.last_raw_response = ""
        self.last_diagnostics: dict[str, Any] = {}
        self._metrics = {"decision_count": 0, "model_action_count": 0, "model_finish_count": 0,
            "canonical_mapping_failure_count": 0, "local_navigation_action_count": 0}

    def status(self) -> dict[str, Any]:
        value = dict(self.provider.status())
        value.update({"visual_decision_protocol": QWEN_VISUAL_DECISION_PROTOCOL_VERSION,
            "task_context_protocol": qwen_task_context_domain.SUPPORTED_TASK_CONTEXT_PROTOCOL,
            "model_role": QWEN_VISUAL_DECISION_MODEL_ROLE, "hardware_actions_enabled": False,
            "selection_authority": "qwen_same_response_decision", **self._metrics})
        value['recent_navigation_protocol'] = RECENT_NAVIGATION_PROTOCOL
        return value

    def decide(self, *, frames: list[Image.Image], task_context: qwen_task_context_domain.QwenTaskContext | dict[str,
        Any], trusted_observation: trusted_observation_domain.TrustedObservation, decision_number: int=1,
        available_action_kinds: Iterable[str] | None=None, launch_target: Mapping[str, str] | None=None,
        text_transport_profile: TextTransportProfile | None=None,
        model_decision: Mapping[str, Any], decision_source: str='qwen_same_response_decision') -> QwenVisualDecision:
        del decision_number
        started = time.perf_counter()
        reject_if(decision_source not in {'qwen_same_response_decision', LOCAL_NAVIGATION_SOURCE},
            VisionAgentError('未知执行决策来源。'))
        context = task_context if isinstance(task_context,
            qwen_task_context_domain.QwenTaskContext) else qwen_task_context_domain.QwenTaskContext.from_dict(
            task_context)
        context.validate()
        available_actions = _normalize_available_action_kinds(available_action_kinds)
        self.trusted_observation_frame_validator(trusted_observation, frames, allow_leading_outlier=True)
        reject_if(context.device_id != trusted_observation.device_id,
            VisionAgentError("任务 device_id 与可信观察不一致。"))
        self._metrics["decision_count"] += 1
        try:
            payload = normalize_model_step_decision(model_decision)
        except CanonicalActionProtocolError as exc:
            raise VisionAgentError(str(exc)) from exc
        self.last_raw_response = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        self.last_diagnostics = {"visual_decision_protocol": QWEN_VISUAL_DECISION_PROTOCOL_VERSION,
            "vision_model": public_model_identity(self.provider.status()), "task_id": context.task_id,
            "device_id": context.device_id, "revision": context.revision,
            "observation_id": trusted_observation.observation_id, "fingerprint": trusted_observation.fingerprint,
            "model_calls": 0, "decision_from_same_observation_response": decision_source == 'qwen_same_response_decision',
            "decision_source": decision_source,
            "model_decision_status": payload["status"], "canonical_binding": "direct_current_frame",
            "device_action_kinds": sorted(available_actions)}

        if payload["status"] == "finish":
            decision = _finish_decision(payload, context=context, observation=trusted_observation)
            self._metrics["model_finish_count"] += 1
        else:
            try:
                action = bind_same_response_action(payload, context=context, observation=trusted_observation,
                    available_action_kinds=available_actions, launch_target=launch_target,
                    text_transport_profile=text_transport_profile)
            except CanonicalActionProtocolError as exc:
                reason = f"Qwen 当前帧动作无法绑定可信执行参数：{exc}"
                self._metrics["canonical_mapping_failure_count"] += 1
                self.last_diagnostics["canonical_binding_error"] = reason
                self.last_diagnostics.update({"decision_status": "invalid",
                    "elapsed_seconds": round(time.perf_counter() - started, 3)})
                raise VisionAgentError(reason) from exc
            decision = _action_decision(payload, context=context, observation=trusted_observation, action=action)
            self._metrics['local_navigation_action_count' if decision_source == LOCAL_NAVIGATION_SOURCE
                else 'model_action_count'] += 1
        decision = replace(decision, decision_source=decision_source)
        try:
            decision.validate(context)
        except CanonicalActionProtocolError as exc:
            raise VisionAgentError(str(exc)) from exc
        self.last_diagnostics.update({"decision_status": decision.proposal.status,
            "elapsed_seconds": round(time.perf_counter() - started, 3)})
        return decision

def _action_decision(payload: Mapping[str, Any], *, context: qwen_task_context_domain.QwenTaskContext,
    observation: trusted_observation_domain.TrustedObservation, action: SemanticAction) -> QwenVisualDecision:
    reason = str(payload.get("reason") or "").strip()[:500]
    return QwenVisualDecision(task_id=context.task_id, device_id=context.device_id,
        revision=context.revision, observation_id=observation.observation_id,
        fingerprint=observation.fingerprint, trusted_observation=observation,
        previous_action_outcome=payload.get("previous_action_outcome"),
        proposal=GenericStepProposal(status="action", action=action, reason=reason))


def _finish_decision(payload: Mapping[str, Any], *, context: qwen_task_context_domain.QwenTaskContext,
    observation: trusted_observation_domain.TrustedObservation) -> QwenVisualDecision:
    """Accept finish only from evidence bound to this exact current observation."""

    reason = str(payload.get("reason") or observation.scene.summary or "").strip()[:500]
    reject_if(not reason, VisionAgentError("finish缺少当前截图完成事实。"))
    return QwenVisualDecision(task_id=context.task_id, device_id=context.device_id,
        revision=context.revision, observation_id=observation.observation_id,
        fingerprint=observation.fingerprint, trusted_observation=observation,
        previous_action_outcome=payload.get("previous_action_outcome"),
        proposal=GenericStepProposal(status="finish", reason=reason))


def _normalize_available_action_kinds(value: Iterable[str] | None) -> frozenset[str]:
    if value is None:
        return QWEN_PROTOCOL_ACTIONS
    try:
        normalized = frozenset(str(item or "").strip() for item in value)
    except TypeError as exc:
        raise VisionAgentError("设备动作能力必须是可迭代字符串集合。") from exc
    reject_if("" in normalized, VisionAgentError("设备动作能力不能包含空值。"))
    reject_if(normalized - QWEN_PROTOCOL_ACTIONS,
        VisionAgentError("设备动作能力包含协议外动作：" + ", ".join(sorted(normalized - QWEN_PROTOCOL_ACTIONS))))
    reject_if(not normalized, VisionAgentError("设备没有任何可执行canonical动作。"))
    return normalized
