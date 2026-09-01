"""Bind the action or finish emitted by the sole Qwen scene response."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
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
from agent.domain.text_transport import TextTransportProfile
import agent.domain.trusted_observation as trusted_observation_domain
from agent.domain.validation import NormalizedBounds, dataclass_wire, reject_if
from agent.domain.vision_model import VisionAgentError, public_model_identity


QWEN_VISUAL_DECISION_PROTOCOL_VERSION = "2026-09-01-qwen-same-response-action-finish-v9"
QWEN_VISUAL_DECISION_MODEL_ROLE = "single_response_scene_action_or_finish"
SINGLE_ELEMENT_ACTIONS = frozenset({"tap_semantic", "dismiss_overlay", "input_verified_text", "press_enter",
    "clear_verified_text", "double_tap", "long_press"})
QWEN_PROTOCOL_ACTIONS = frozenset(CANONICAL_ACTION_KINDS)


def _targets_single_element(kind: str, params: Mapping[str, Any]) -> bool:
    return kind in SINGLE_ELEMENT_ACTIONS or (kind == "swipe" and bool(str(params.get("element_id") or "").strip()))


@dataclass(frozen=True)
class VisualTargetRegion:
    kind: str
    bounds: NormalizedBounds
    description: str
    element_id: str = ""
    destination_element_id: str = ""
    destination_bounds: NormalizedBounds | None = None

    def validate(self, observation: trusted_observation_domain.TrustedObservation, action: SemanticAction) -> None:
        reject_if(self != _canonical_target_region(action, observation),
            CanonicalActionProtocolError("目标区域没有逐项复用 Qwen 当前帧所选元素。"))

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "element_id": self.element_id or None, "bounds": list(self.bounds),
            "destination_element_id": self.destination_element_id or None,
            "destination_bounds": list(self.destination_bounds) if self.destination_bounds is not None else None,
            "description": self.description}


@dataclass(frozen=True)
class QwenVisualDecision:
    task_id: str
    device_id: str
    revision: int
    observation_id: str
    fingerprint: str
    page_state: dict[str, Any]
    trusted_observation: trusted_observation_domain.TrustedObservation
    proposal: GenericStepProposal
    target_region: VisualTargetRegion | None
    confidence: float
    reason: str
    completion_evidence: tuple[str, ...] = ()
    protocol_version: str = QWEN_VISUAL_DECISION_PROTOCOL_VERSION

    def validate(self, context: qwen_task_context_domain.QwenTaskContext) -> None:
        expected_scope = (context.task_id, context.device_id, context.revision,
            self.trusted_observation.observation_id, self.trusted_observation.fingerprint)
        reject_if((self.task_id, self.device_id, self.revision, self.observation_id,
            self.fingerprint) != expected_scope,
            CanonicalActionProtocolError("task/device/revision/observation/fingerprint 已过期或不匹配。"))
        self.proposal.validate(self.trusted_observation.scene)
        reject_if(self.protocol_version != QWEN_VISUAL_DECISION_PROTOCOL_VERSION,
            CanonicalActionProtocolError("Qwen视觉决策协议版本无效。"))
        reject_if(not 0.0 <= float(self.confidence) <= 1.0,
            CanonicalActionProtocolError("Qwen视觉决策置信度必须在0到1之间。"))

        action = self.proposal.action
        if self.proposal.status == "action":
            reject_if(action is None or self.target_region is None or self.completion_evidence,
                CanonicalActionProtocolError("唯一下一动作缺少当前帧目标或夹带完成证据。"))
            reject_if(not context.effect_action_allowed and context.current_execution_class == "effect",
                CanonicalActionProtocolError("登录或付款确认门未满足，禁止产生效果动作。"))
            self.target_region.validate(self.trusted_observation, action)
        else:
            reject_if(self.target_region is not None or not self.completion_evidence,
                CanonicalActionProtocolError("finish 必须只携带同一 scene 的完成证据。"))

    def to_dict(self) -> dict[str, Any]:
        value = dataclass_wire(self, omit=("proposal",))
        value.update(status=self.proposal.status,
            next_action=self.proposal.action.to_dict() if self.proposal.action else None,
            confidence=float(self.confidence), completion_evidence=list(self.completion_evidence))
        return value


class QwenVisualDecisionObserver:
    """Bind Qwen's one current-frame decision; never build or choose another candidate."""

    def __init__(self, provider: Any, *, trusted_observation_frame_validator: Callable[..., None]) -> None:
        self.provider = provider
        self.trusted_observation_frame_validator = trusted_observation_frame_validator
        self.last_raw_response = ""
        self.last_diagnostics: dict[str, Any] = {}
        self._metrics = {"decision_count": 0, "model_action_count": 0, "model_finish_count": 0,
            "canonical_mapping_failure_count": 0}

    def status(self) -> dict[str, Any]:
        value = dict(self.provider.status())
        value.update({"visual_decision_protocol": QWEN_VISUAL_DECISION_PROTOCOL_VERSION,
            "task_context_protocol": qwen_task_context_domain.SUPPORTED_TASK_CONTEXT_PROTOCOL,
            "model_role": QWEN_VISUAL_DECISION_MODEL_ROLE, "hardware_actions_enabled": False,
            "selection_authority": "qwen_same_response_decision", **self._metrics})
        return value

    def decide(self, *, frames: list[Image.Image], task_context: qwen_task_context_domain.QwenTaskContext | dict[str,
        Any], trusted_observation: trusted_observation_domain.TrustedObservation, decision_number: int=1,
        available_action_kinds: Iterable[str] | None=None, launch_target: Mapping[str, str] | None=None,
        text_transport_profile: TextTransportProfile | None=None,
        model_decision: Mapping[str, Any]) -> QwenVisualDecision:
        del decision_number
        started = time.perf_counter()
        context = task_context if isinstance(task_context,
            qwen_task_context_domain.QwenTaskContext) else qwen_task_context_domain.QwenTaskContext.from_dict(
            task_context)
        context.validate()
        available_actions = _normalize_available_action_kinds(available_action_kinds)
        self.trusted_observation_frame_validator(trusted_observation, frames, allow_leading_outlier=True)
        reject_if(context.device_id != trusted_observation.device_id,
            VisionAgentError("任务 device_id 与可信观察不一致。"))
        self._metrics["decision_count"] += 1
        reject_if(context.current_execution_class == "effect" and not context.effect_action_allowed,
            VisionAgentError("登录或付款确认尚未完成，Qwen动作循环不得开始。"))

        try:
            payload = normalize_model_step_decision(model_decision)
        except CanonicalActionProtocolError as exc:
            raise VisionAgentError(str(exc)) from exc
        self.last_raw_response = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        self.last_diagnostics = {"visual_decision_protocol": QWEN_VISUAL_DECISION_PROTOCOL_VERSION,
            "vision_model": public_model_identity(self.provider.status()), "task_id": context.task_id,
            "device_id": context.device_id, "revision": context.revision,
            "observation_id": trusted_observation.observation_id, "fingerprint": trusted_observation.fingerprint,
            "model_calls": 0, "decision_from_same_observation_response": True,
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
            self._metrics["model_action_count"] += 1
        decision.validate(context)
        self.last_diagnostics.update({"decision_status": decision.proposal.status,
            "elapsed_seconds": round(time.perf_counter() - started, 3)})
        return decision

def _action_decision(payload: Mapping[str, Any], *, context: qwen_task_context_domain.QwenTaskContext,
    observation: trusted_observation_domain.TrustedObservation, action: SemanticAction) -> QwenVisualDecision:
    element_ids = tuple(str(action.params.get(key) or "").strip()
        for key in ("element_id", "source_element_id", "destination_element_id")
        if str(action.params.get(key) or "").strip())
    confidence_parts = [float(payload.get("confidence", 1.0)), float(observation.scene.confidence)]
    confidence_parts.extend(float(observation.get_candidate(item).confidence) for item in element_ids)
    reason = str(payload.get("reason") or "").strip()[:500]
    return QwenVisualDecision(task_id=context.task_id, device_id=context.device_id,
        revision=context.revision, observation_id=observation.observation_id,
        fingerprint=observation.fingerprint, page_state=_page_state(observation.scene),
        trusted_observation=observation, proposal=GenericStepProposal(status="action", action=action, reason=reason),
        target_region=_canonical_target_region(action, observation), confidence=min(confidence_parts), reason=reason)


def _finish_decision(payload: Mapping[str, Any], *, context: qwen_task_context_domain.QwenTaskContext,
    observation: trusted_observation_domain.TrustedObservation) -> QwenVisualDecision:
    """Accept finish only from evidence bound to this exact current observation."""

    evidence: list[str] = []
    for ref in payload.get("evidence_refs") or ():
        if ref == "scene.summary":
            reject_if(not observation.scene.summary.strip(), VisionAgentError("finish引用了空scene.summary。"))
            evidence.append(observation.scene.summary.strip())
            continue
        reject_if(not str(ref).startswith("element:"), VisionAgentError("finish包含未知scene证据引用。"))
        element_id = str(ref).split(":", 1)[1]
        element = observation.get_candidate(element_id)
        visible = next((str(item).strip() for item in element.evidence if str(item).strip()), "")
        evidence.append(visible or element.label or element.meaning)
    reject_if(not evidence, VisionAgentError("finish没有绑定当前scene证据。"))
    reason = str(payload.get("reason") or "").strip()[:500]
    return QwenVisualDecision(task_id=context.task_id, device_id=context.device_id,
        revision=context.revision, observation_id=observation.observation_id,
        fingerprint=observation.fingerprint, page_state=_page_state(observation.scene),
        trusted_observation=observation, proposal=GenericStepProposal(status="finish", reason=reason),
        target_region=None, confidence=float(payload["confidence"]), reason=reason,
        completion_evidence=tuple(dict.fromkeys(evidence)))


def _canonical_target_region(action: SemanticAction,
    observation: trusted_observation_domain.TrustedObservation) -> VisualTargetRegion:
    if _targets_single_element(action.action, action.params):
        element = observation.get_candidate(str(action.params.get("element_id") or "").strip())
        return VisualTargetRegion(kind="element", element_id=element.element_id, bounds=element.bounds,
            description=element.label or element.meaning)
    if action.action == "drag":
        source = observation.get_candidate(str(action.params.get("source_element_id") or "").strip())
        destination = observation.get_candidate(str(action.params.get("destination_element_id") or "").strip())
        return VisualTargetRegion(kind="element_path", element_id=source.element_id, bounds=source.bounds,
            destination_element_id=destination.element_id, destination_bounds=destination.bounds,
            description=f"{source.label or source.meaning} 到 {destination.label or destination.meaning}")
    descriptions = {"back": "系统返回区域", "home": "Android系统Home键",
        "open_recent_apps": "Android系统最近任务键", "launch_app": "受信任的 App 启动通道",
        "reveal_system_navigation": "Android系统导航栏"}
    return VisualTargetRegion(kind="system_navigation" if action.action in descriptions else "screen",
        bounds=(0.0, 0.0, 1.0, 1.0), description=descriptions.get(action.action, "当前屏幕"))


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


def _page_state(scene: Any) -> dict[str, Any]:
    return {"foreground_app_id": scene.foreground_app_id, "screen_id": scene.screen_id,
        "summary": scene.summary, "overlays": list(scene.overlays)}
