"""Bind the action or finish emitted by the sole Qwen scene response."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import re
import time
from typing import Any, Callable, Iterable

from PIL import Image

from agent.domain.validation import NormalizedBounds, dataclass_wire, reject_if
from agent.domain.canonical_action_kinds import CANONICAL_ACTION_KINDS
from agent.domain.canonical_action_protocol import (
    CanonicalActionProtocolError as GenericStepPlanningError,
    GenericStepProposal,
)
import agent.domain.qwen_task_context as qwen_task_context_domain
import agent.domain.trusted_observation as trusted_observation_domain
from agent.domain.semantic_action import SemanticAction
from agent.domain.text_transport import TextTransportProfile
from agent.domain.ui_scene import UIElement
from agent.domain.vision_model import VisionAgentError, public_model_identity


QWEN_VISUAL_DECISION_PROTOCOL_VERSION = "2026-09-01-qwen-same-response-action-finish-v8"
QWEN_VISUAL_DECISION_MODEL_ROLE = "single_response_scene_action_or_finish"
SINGLE_ELEMENT_ACTIONS = frozenset({'tap_semantic', 'dismiss_overlay', 'input_verified_text', 'press_enter',
    'clear_verified_text', 'double_tap', 'long_press'})
QWEN_PROTOCOL_ACTIONS = frozenset(CANONICAL_ACTION_KINDS)
_ANDROID_PACKAGE = re.compile(r"[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)+")


def _targets_single_element(kind: str, params: Mapping[str, Any]) -> bool:
    return kind in SINGLE_ELEMENT_ACTIONS or (kind == 'swipe' and bool(str(params.get('element_id') or '').strip()))


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
            GenericStepPlanningError("目标区域没有逐项复用 canonical 候选。"))

    def to_dict(self) -> dict[str, Any]:
        return {'kind': self.kind, 'element_id': self.element_id or None, 'bounds': list(self.bounds),
            'destination_element_id': self.destination_element_id or None,
            'destination_bounds': list(self.destination_bounds) if self.destination_bounds is not None else None,
            'description': self.description}


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
    expected_result: dict[str, Any]
    confidence: float
    reason: str
    completion_evidence: tuple[str, ...] = ()
    protocol_version: str = QWEN_VISUAL_DECISION_PROTOCOL_VERSION

    def validate(self, context: qwen_task_context_domain.QwenTaskContext) -> None:
        expected_scope = (context.task_id, context.device_id, context.revision,
            self.trusted_observation.observation_id, self.trusted_observation.fingerprint)
        reject_if((self.task_id, self.device_id, self.revision, self.observation_id,
            self.fingerprint) != expected_scope,
            GenericStepPlanningError('task/device/revision/observation/fingerprint 已过期或不匹配。'))
        self.proposal.validate(self.trusted_observation.scene)
        reject_if(self.protocol_version != QWEN_VISUAL_DECISION_PROTOCOL_VERSION,
            GenericStepPlanningError("Qwen视觉决策协议版本无效。"))
        reject_if(not 0.0 <= float(self.confidence) <= 1.0,
            GenericStepPlanningError("Qwen视觉决策置信度必须在0到1之间。"))
        reject_if(not isinstance(self.expected_result, dict), GenericStepPlanningError("expected_result 必须是JSON对象。"))

        action = self.proposal.action
        if self.proposal.status == 'action':
            reject_if(action is None or self.target_region is None,
                GenericStepPlanningError("唯一下一动作缺少可信目标区域。"))
            reject_if(not self.expected_result or self.completion_evidence,
                GenericStepPlanningError("动作决策的后置条件或完成证据无效。"))
            reject_if(not context.effect_action_allowed and context.current_execution_class == 'effect',
                GenericStepPlanningError("登录或付款确认门未满足，禁止产生效果动作。"))
            self.target_region.validate(self.trusted_observation, action)
            reject_if(dict(action.params.get('expected_effect') or {}) != self.expected_result,
                GenericStepPlanningError("动作 expected_effect 与顶层预期不一致。"))
        elif self.proposal.status == 'finish':
            reject_if(self.target_region is not None or self.expected_result or not self.completion_evidence,
                GenericStepPlanningError("finish 必须只携带同一scene的完成证据。"))

    def to_dict(self) -> dict[str, Any]:
        value = dataclass_wire(self, omit=('proposal',))
        value.update(status=self.proposal.status,
            next_action=self.proposal.action.to_dict() if self.proposal.action else None,
            confidence=float(self.confidence), completion_evidence=list(self.completion_evidence))
        return value


class QwenVisualDecisionObserver:
    """Validate Qwen's same-response choice against the canonical catalog; never choose for it."""

    def __init__(self, provider: Any, *, trusted_observation_frame_validator: Callable[..., None],
        decision_source: Any | None=None) -> None:
        self.provider = provider
        self.decision_source = decision_source
        self.trusted_observation_frame_validator = trusted_observation_frame_validator
        self.last_raw_response = ""
        self.last_diagnostics: dict[str, Any] = {}
        self._metrics = {'decision_count': 0, 'model_action_count': 0, 'model_finish_count': 0,
            'canonical_mapping_failure_count': 0}

    def status(self) -> dict[str, Any]:
        value = dict(self.provider.status())
        value.update({'visual_decision_protocol': QWEN_VISUAL_DECISION_PROTOCOL_VERSION,
            'task_context_protocol': qwen_task_context_domain.SUPPORTED_TASK_CONTEXT_PROTOCOL,
            'model_role': QWEN_VISUAL_DECISION_MODEL_ROLE, 'hardware_actions_enabled': False,
            'selection_authority': 'qwen_same_response_decision', **self._metrics})
        return value

    def decide(self, *, frames: list[Image.Image], task_context: qwen_task_context_domain.QwenTaskContext | dict[str,
        Any], trusted_observation: trusted_observation_domain.TrustedObservation, decision_number: int=1,
        available_action_kinds: Iterable[str] | None=None, launch_target: Mapping[str, str] | None=None,
        text_transport_profile: TextTransportProfile | None=None) -> QwenVisualDecision:
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
        self._metrics['decision_count'] += 1
        reject_if(context.current_execution_class == 'effect' and not context.effect_action_allowed,
            VisionAgentError("登录或付款确认尚未完成，Qwen动作循环不得开始。"))

        choices = _selection_choices(context, trusted_observation, available_actions,
            launch_target=launch_target, text_transport_profile=text_transport_profile)
        source = self.decision_source if self.decision_source is not None else self.provider
        getter = getattr(source, 'decision_for', None)
        reject_if(not callable(getter), VisionAgentError("Qwen观察器没有提供同响应action/finish决策。"))
        payload = _normalize_model_decision_payload(getter(trusted_observation.fingerprint))
        self.last_raw_response = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
        self.last_diagnostics = {'visual_decision_protocol': QWEN_VISUAL_DECISION_PROTOCOL_VERSION,
            'vision_model': public_model_identity(self.provider.status()), 'task_id': context.task_id,
            'device_id': context.device_id, 'revision': context.revision,
            'observation_id': trusted_observation.observation_id, 'fingerprint': trusted_observation.fingerprint,
            'model_calls': 0, 'decision_from_same_observation_response': True,
            'model_decision_status': payload['status'], 'canonical_choice_count': len(choices),
            'canonical_choices': [{'action': str(item.get('action') or ''),
            'direction': str(item.get('direction') or ''), 'element_id': str(item.get('element_id') or '')}
            for item in choices], 'device_action_kinds': sorted(available_actions)}

        if payload['status'] == 'finish':
            decision = _finish_decision(payload, context=context, observation=trusted_observation,
                launch_target=launch_target)
            self._metrics['model_finish_count'] += 1
        else:
            matches = tuple(item for item in choices if _choice_matches_model_decision(item, payload))
            if len(matches) != 1:
                reason = (f"Qwen选择未精确映射唯一canonical candidate：action={payload['action']}，"
                    f"matches={len(matches)}。本地没有改选其它动作。")
                self._metrics['canonical_mapping_failure_count'] += 1
                self.last_diagnostics['canonical_mapping_error'] = reason
                self.last_diagnostics.update({'decision_status': 'invalid',
                    'elapsed_seconds': round(time.perf_counter() - started, 3)})
                raise VisionAgentError(reason)
            else:
                decision = _hydrate_canonical_selection(payload, context=context,
                    observation=trusted_observation, choice=matches[0])
                self._metrics['model_action_count'] += 1
        decision.validate(context)
        self.last_diagnostics.update({'decision_status': decision.proposal.status,
            'elapsed_seconds': round(time.perf_counter() - started, 3)})
        return decision


def _selection_choices(context: qwen_task_context_domain.QwenTaskContext,
    observation: trusted_observation_domain.TrustedObservation, available_action_kinds: frozenset[str], *,
    launch_target: Mapping[str, str] | None=None,
    text_transport_profile: TextTransportProfile | None=None) -> tuple[dict[str, Any], ...]:
    reject_if(context.semantic_ir is None, VisionAgentError("视觉选择缺少 canonical TaskSemanticIR。"))
    try:
        from agent.domain.canonical_action_protocol import (
            canonical_candidate_expected_result,
            compile_canonical_action_catalog,
        )
        report = compile_canonical_action_catalog(observation.scene, context.semantic_ir, available_action_kinds,
            launch_target=launch_target, text_transport_profile=text_transport_profile)
    except Exception as exc:
        raise VisionAgentError(f'canonical action catalog 构建失败：{exc}') from exc
    return tuple({'action': candidate.action_kind, **dict(candidate.parameters),
        'expected_result': canonical_candidate_expected_result(candidate, observation.scene),
        'formal_candidate_id': candidate.candidate_id, 'formal_transition': candidate.transition.to_dict()}
        for candidate in report.candidates)


def _normalize_model_decision_payload(value: Any) -> dict[str, Any]:
    required = {'status', 'action', 'element_id', 'source_element_id', 'destination_element_id', 'direction',
        'evidence_refs', 'confidence', 'reason'}
    reject_if(not isinstance(value, Mapping) or set(value) != required,
        VisionAgentError("同响应decision字段不完整或包含协议外字段。"))
    reject_if(value.get('status') not in {'action', 'finish'},
        VisionAgentError("同响应decision只允许action或finish。"))
    return dict(value)


def _choice_matches_model_decision(choice: Mapping[str, Any], payload: Mapping[str, Any]) -> bool:
    kind = str(payload.get('action') or '')
    if choice.get('action') != kind:
        return False
    if kind in SINGLE_ELEMENT_ACTIONS:
        return choice.get('element_id') == payload.get('element_id')
    if kind == 'drag':
        return (choice.get('source_element_id') == payload.get('source_element_id')
            and choice.get('destination_element_id') == payload.get('destination_element_id'))
    if kind == 'swipe':
        if choice.get('direction') != payload.get('direction'):
            return False
        selected_element = payload.get('element_id')
        return selected_element is None or choice.get('element_id') == selected_element
    return True


def _hydrate_canonical_selection(payload: Mapping[str, Any], *,
    context: qwen_task_context_domain.QwenTaskContext,
    observation: trusted_observation_domain.TrustedObservation,
    choice: Mapping[str, Any] | None=None, choices: tuple[dict[str, Any], ...] | None=None) -> QwenVisualDecision:
    """Hydrate exactly the canonical candidate named by Qwen's same-response decision."""

    if choice is None:
        matches = tuple(item for item in choices or () if _choice_matches_model_decision(item, payload))
        reject_if(len(matches) != 1, VisionAgentError("Qwen选择没有唯一canonical candidate。"))
        choice = matches[0]
    kind = str(choice.get('action') or '').strip()
    expected_result = dict(choice.get('expected_result') or {})
    reject_if(kind not in CANONICAL_ACTION_KINDS or not expected_result,
        VisionAgentError("canonical candidate动作或expected_result无效。"))
    params = {key: value for key, value in choice.items()
        if key not in {'action', 'expected_result', 'selection_context'}}
    bound_ids: list[str] = []
    if _targets_single_element(kind, params):
        element = observation.get_candidate(str(params.get('element_id') or ''))
        _bind_element_params(params, element)
        bound_ids.append(element.element_id)
        if kind == 'input_verified_text':
            params['text'] = context.requested_input_text
        elif kind == 'long_press':
            params['duration_ms'] = 800
    elif kind == 'drag':
        for prefix in ('source_', 'destination_'):
            element = observation.get_candidate(str(params.get(f'{prefix}element_id') or ''))
            _bind_element_params(params, element, prefix=prefix)
            bound_ids.append(element.element_id)
    params['expected_effect'] = expected_result
    action = SemanticAction(node_id=f'qwen_visual_revision_{context.revision}', action=kind, params=params)
    confidence = min(float(payload.get('confidence', 0.0)), float(observation.scene.confidence),
        *(float(observation.get_candidate(item).confidence) for item in bound_ids))
    reason = str(payload.get('reason') or '').strip()[:500]
    decision = QwenVisualDecision(task_id=context.task_id, device_id=context.device_id,
        revision=context.revision, observation_id=observation.observation_id,
        fingerprint=observation.fingerprint, page_state=_page_state(observation.scene),
        trusted_observation=observation, proposal=GenericStepProposal(status='action', action=action, reason=reason),
        target_region=_canonical_target_region(action, observation), expected_result=expected_result,
        confidence=confidence, reason=reason)
    decision.validate(context)
    return decision


def _finish_decision(payload: Mapping[str, Any], *, context: qwen_task_context_domain.QwenTaskContext,
    observation: trusted_observation_domain.TrustedObservation,
    launch_target: Mapping[str, str] | None=None) -> QwenVisualDecision:
    expected_app_id = str((launch_target or {}).get('expected_app_id') or '').strip()
    current_app_id = str(observation.scene.foreground_app_id or '').strip()
    reject_if(bool(_ANDROID_PACKAGE.fullmatch(expected_app_id) and _ANDROID_PACKAGE.fullmatch(current_app_id)
        and expected_app_id.casefold() != current_app_id.casefold()),
        VisionAgentError("当前Android系统前台包名与正式目标App不一致，Qwen不能把本轮判为finish："
            f"current={current_app_id}，expected={expected_app_id}。"))
    evidence: list[str] = []
    for ref in payload.get('evidence_refs') or ():
        if ref == 'scene.summary':
            reject_if(not observation.scene.summary.strip(), VisionAgentError("finish引用了空scene.summary。"))
            evidence.append(observation.scene.summary.strip())
            continue
        reject_if(not str(ref).startswith('element:'), VisionAgentError("finish包含未知scene证据引用。"))
        element_id = str(ref).split(':', 1)[1]
        element = observation.get_candidate(element_id)
        visible = next((str(item).strip() for item in element.evidence if str(item).strip()), '')
        evidence.append(visible or element.label or element.meaning)
    reject_if(not evidence, VisionAgentError("finish没有绑定当前scene证据。"))
    reason = str(payload.get('reason') or '').strip()[:500]
    return QwenVisualDecision(task_id=context.task_id, device_id=context.device_id,
        revision=context.revision, observation_id=observation.observation_id,
        fingerprint=observation.fingerprint, page_state=_page_state(observation.scene),
        trusted_observation=observation, proposal=GenericStepProposal(status='finish', reason=reason),
        target_region=None, expected_result={}, confidence=float(payload['confidence']), reason=reason,
        completion_evidence=tuple(dict.fromkeys(evidence)))


def _canonical_target_region(action: SemanticAction,
    observation: trusted_observation_domain.TrustedObservation) -> VisualTargetRegion:
    if _targets_single_element(action.action, action.params):
        element = observation.get_candidate(str(action.params.get('element_id') or '').strip())
        return VisualTargetRegion(kind='element', element_id=element.element_id, bounds=element.bounds,
            description=element.label or element.meaning)
    if action.action == 'drag':
        source = observation.get_candidate(str(action.params.get('source_element_id') or '').strip())
        destination = observation.get_candidate(str(action.params.get('destination_element_id') or '').strip())
        return VisualTargetRegion(kind='element_path', element_id=source.element_id, bounds=source.bounds,
            destination_element_id=destination.element_id, destination_bounds=destination.bounds,
            description=f'{source.label or source.meaning} 到 {destination.label or destination.meaning}')
    descriptions = {'back': '系统返回区域', 'home': 'Android系统Home键',
        'open_recent_apps': 'Android系统最近任务键', 'launch_app': '受信任的 App 启动通道',
        'reveal_system_navigation': 'Android系统导航栏'}
    return VisualTargetRegion(kind='system_navigation' if action.action in descriptions else 'screen',
        bounds=(0.0, 0.0, 1.0, 1.0), description=descriptions.get(action.action, '当前屏幕'))


def _normalize_available_action_kinds(value: Iterable[str] | None) -> frozenset[str]:
    if value is None:
        return QWEN_PROTOCOL_ACTIONS
    try:
        normalized = frozenset(str(item or '').strip() for item in value)
    except TypeError as exc:
        raise VisionAgentError("设备动作能力必须是可迭代字符串集合。") from exc
    reject_if('' in normalized, VisionAgentError("设备动作能力不能包含空值。"))
    reject_if(normalized - QWEN_PROTOCOL_ACTIONS,
        VisionAgentError('设备动作能力包含协议外动作：' + ', '.join(sorted(normalized - QWEN_PROTOCOL_ACTIONS))))
    reject_if(not normalized, VisionAgentError("设备没有任何可执行canonical动作。"))
    return normalized


def _bind_element_params(params: dict[str, Any], element: UIElement, *, prefix: str='') -> None:
    params.update({f'{prefix}element_id': element.element_id, f'{prefix}target': element.meaning,
        f'{prefix}role': element.role, f'{prefix}label': element.label,
        f'{prefix}states': dict(element.states)})


def _page_state(scene: Any) -> dict[str, Any]:
    return {'foreground_app_id': scene.foreground_app_id, 'screen_id': scene.screen_id,
        'summary': scene.summary, 'overlays': list(scene.overlays)}
