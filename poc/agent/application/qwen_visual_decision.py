"""Single-step Qwen visual decision application service."""

from __future__ import annotations

from agent.domain.validation import NormalizedBounds, dataclass_wire, reject_if
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from PIL import Image

from agent.domain.canonical_action_protocol import (
    CanonicalActionProtocolError as GenericStepPlanningError,
    GenericStepProposal,
)
from agent.domain.canonical_action_kinds import CANONICAL_ACTION_KINDS
import agent.domain.qwen_task_context as qwen_task_context_domain
import agent.domain.trusted_observation as trusted_observation_domain
from agent.domain.semantic_action import SemanticAction
from agent.domain.ui_scene import UIElement, UIScene
from agent.domain.vision_model import VisionAgentError, public_model_identity


QWEN_VISUAL_DECISION_PROTOCOL_VERSION = "2026-08-14-qwen-visual-decision-v5"
QWEN_VISUAL_DECISION_MODEL_ROLE = "trusted_observation_single_step_selector"
MIN_DECISION_CONFIDENCE = 0.72
SINGLE_ELEMENT_ACTIONS = frozenset({'tap_semantic', 'dismiss_overlay', 'input_verified_text', 'press_enter',
    'clear_verified_text', 'double_tap', 'long_press'})


def _targets_single_element(kind: str, params: Mapping[str, Any]) -> bool:
    """Return whether this canonical action binds one observed element, including an anchored swipe."""

    return kind in SINGLE_ELEMENT_ACTIONS or (kind == 'swipe' and bool(str(params.get('element_id') or '').strip()))


QWEN_PROTOCOL_ACTIONS = frozenset(CANONICAL_ACTION_KINDS)

@dataclass(frozen=True)
class VisualTargetRegion:
    kind: str
    bounds: NormalizedBounds
    description: str
    element_id: str = ""
    destination_element_id: str = ""
    destination_bounds: NormalizedBounds | None = None

    def validate(self, observation: trusted_observation_domain.TrustedObservation, action: SemanticAction) -> None:
        expected = _canonical_target_region(action, observation)
        reject_if(self != expected, GenericStepPlanningError("目标区域没有逐项复用 canonical 候选。"))

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
    protocol_version: str = QWEN_VISUAL_DECISION_PROTOCOL_VERSION

    def validate(self, context: qwen_task_context_domain.QwenTaskContext) -> None:
        reject_if((self.task_id, self.device_id, self.revision, self.observation_id, self.fingerprint) != (context.task_id, context.device_id, context.revision, self.trusted_observation.observation_id, self.trusted_observation.fingerprint), GenericStepPlanningError('task/device/revision/observation/fingerprint 已过期或不匹配。'))
        self.proposal.validate(self.trusted_observation.scene)
        reject_if(self.protocol_version != QWEN_VISUAL_DECISION_PROTOCOL_VERSION, GenericStepPlanningError("Qwen视觉决策协议版本无效。"))
        reject_if(not 0.0 <= float(self.confidence) <= 1.0, GenericStepPlanningError("Qwen视觉决策置信度必须在0到1之间。"))
        reject_if(not isinstance(self.expected_result, dict), GenericStepPlanningError("expected_result 必须是JSON对象。"))

        action = self.proposal.action
        if self.proposal.status == 'action':
            reject_if(action is None or self.target_region is None, GenericStepPlanningError("唯一下一动作缺少可信目标区域。"))
            reject_if(not self.expected_result, GenericStepPlanningError("唯一下一动作缺少可验证预期结果。"))
            reject_if(not context.effect_action_allowed and context.current_execution_class == 'effect', GenericStepPlanningError("风险确认门未满足，禁止产生外部状态动作。"))
            self.target_region.validate(self.trusted_observation, action)
            reject_if(dict(action.params.get('expected_effect') or {}) != self.expected_result, GenericStepPlanningError("动作 expected_effect 与顶层预期不一致。"))
            reject_if(float(self.confidence) < MIN_DECISION_CONFIDENCE, GenericStepPlanningError("动作置信度不足，必须 blocked。"))
        elif self.target_region is not None or self.expected_result:
            raise GenericStepPlanningError("blocked 不能携带动作目标区域。")

    def to_dict(self) -> dict[str, Any]:
        value = dataclass_wire(self, omit=('proposal',))
        value.update(status=self.proposal.status, next_action=self.proposal.action.to_dict(
            ) if self.proposal.action else None, confidence=float(self.confidence))
        return value


class QwenVisualDecisionObserver:
    """Select one canonical action locally from one trusted observation."""

    def __init__(self, provider: Any, *, trusted_observation_frame_validator: Callable[..., None]) -> None:
        self.provider = provider
        self.trusted_observation_frame_validator = trusted_observation_frame_validator
        self.last_raw_response = ""
        self.last_diagnostics: dict[str, Any] = {}
        self._metrics = {'decision_count': 0, 'deterministic_action_count': 0, 'final_blocked_count': 0}

    def status(self) -> dict[str, Any]:
        value = dict(self.provider.status())
        decisions = self._metrics["decision_count"]
        value.update({'visual_decision_protocol': QWEN_VISUAL_DECISION_PROTOCOL_VERSION,
            'task_context_protocol': qwen_task_context_domain.SUPPORTED_TASK_CONTEXT_PROTOCOL,
            'model_role': QWEN_VISUAL_DECISION_MODEL_ROLE, 'hardware_actions_enabled': False,
            'selection_authority': 'canonical_action_catalog_local', **self._metrics,
            'deterministic_action_rate': _ratio(self._metrics['deterministic_action_count'], decisions),
            'final_blocked_rate': _ratio(self._metrics['final_blocked_count'], decisions)})
        return value

    def decide(self, *, frames: list[Image.Image], task_context: qwen_task_context_domain.QwenTaskContext | dict[str,
        Any], trusted_observation: trusted_observation_domain.TrustedObservation, decision_number: int=1,
        available_action_kinds: Iterable[str] | None=None) -> QwenVisualDecision:
        started = time.perf_counter()
        self.last_raw_response = ""
        self.last_diagnostics = {}
        context = task_context if isinstance(task_context,
            qwen_task_context_domain.QwenTaskContext) else qwen_task_context_domain.QwenTaskContext.from_dict(
            task_context)
        context.validate()
        available_actions = _normalize_available_action_kinds(available_action_kinds)
        # These are the same read-only frames that established the trusted
        # observation, so apply the observer's one-leading-frame tolerance.
        # Confirmation-time recapture and post-action verification use their
        # own stricter full-window stability checks.
        self.trusted_observation_frame_validator(trusted_observation, frames, allow_leading_outlier=True)
        reject_if(context.device_id != trusted_observation.device_id, VisionAgentError("任务 device_id 与可信观察不一致。"))
        self._metrics["decision_count"] += 1
        canonical_choices = _selection_choices(context, trusted_observation, available_actions)
        canonical_action_kinds = sorted({str(item['action']) for item in canonical_choices})

        model_identity = public_model_identity(self.provider.status())
        base_diagnostics = {'visual_decision_protocol': QWEN_VISUAL_DECISION_PROTOCOL_VERSION,
            'vision_model': model_identity, 'task_id': context.task_id, 'device_id': context.device_id,
            'revision': context.revision, 'observation_id': trusted_observation.observation_id,
            'fingerprint': trusted_observation.fingerprint, 'model_calls': 0, 'hardware_actions_enabled': False,
            'available_action_kinds': canonical_action_kinds, 'canonical_choice_count': len(canonical_choices),
            'canonical_choices': [{'action': str(item.get('action') or ''),
            'direction': str(item.get('direction') or ''), 'element_id': str(item.get('element_id') or '')} for item
            in canonical_choices], 'device_action_kinds': sorted(available_actions)}
        self.last_diagnostics = dict(base_diagnostics)

        if context.current_execution_class == 'effect' and not context.effect_action_allowed:
            reason = '风险确认门未满足，本轮禁止提出外部状态动作。'
            decision = _local_blocked_decision(context, trusted_observation, reason=reason)
            self._metrics["final_blocked_count"] += 1
            self.last_diagnostics.update({'local_safety_block': 'effect_gate', 'decision_status': 'blocked',
                'elapsed_seconds': round(time.perf_counter() - started, 3)})
            return decision

        deterministic_selection = _deterministic_exact_selection_payload(context, canonical_choices,
            observation=trusted_observation)
        if deterministic_selection is not None:
            raw = json.dumps(deterministic_selection, ensure_ascii=False, separators=(',', ':'))
            decision = _hydrate_canonical_selection(deterministic_selection, context=context,
                observation=trusted_observation, choices=canonical_choices)
            self.last_raw_response = raw
            self._metrics["deterministic_action_count"] += 1
            self.last_diagnostics.update({'local_deterministic_selection': True,
                'decision_status': decision.proposal.status, 'elapsed_seconds': round(time.perf_counter() - started,
                3)})
            return decision

        decision = _local_blocked_decision(
            context,
            trusted_observation,
            reason=(
            "当前canonical目录未能依据本步当前截图中的唯一视觉目标确定单一动作；"
            "本地选择器停止，不沿用历史页面动作，也不发起重复模型请求。"
            ),
        )
        self._metrics["final_blocked_count"] += 1
        self.last_diagnostics.update({'local_safety_block': 'single_step_candidate_not_unique',
            'local_deterministic_selection': False, 'model_calls': 0, 'decision_status': 'blocked',
            'elapsed_seconds': round(time.perf_counter() - started, 3)})
        return decision


def _selection_choices(context: qwen_task_context_domain.QwenTaskContext,
    observation: trusted_observation_domain.TrustedObservation,
    available_action_kinds: frozenset[str]) -> tuple[dict[str, Any], ...]:
    """Build generic action choices from the trusted scene, never app steps."""

    choices: list[dict[str, Any]] = []
    reject_if(context.semantic_ir is None, VisionAgentError("typed v4 视觉选择缺少 canonical TaskSemanticIR。"))
    try:
        from agent.domain.canonical_action_protocol import (
            canonical_candidate_expected_result,
            compile_canonical_action_catalog,
        )

        formal_report = compile_canonical_action_catalog(observation.scene, context.semantic_ir, available_action_kinds)
    except Exception as exc:
        raise VisionAgentError(f'canonical action catalog 构建失败：{exc}') from exc

    # Qwen receives a presentation of the canonical catalog, not a separately
    # rebuilt action list.  Every action parameter and postcondition below is a
    # deterministic projection of the same immutable candidate that Policy
    # later selects by ID.
    for candidate in formal_report.candidates:
        choices.append({'choice_id': f'choice_{len(choices) + 1}', 'action': candidate.action_kind,
            **dict(candidate.parameters), 'expected_result': canonical_candidate_expected_result(candidate,
            observation.scene), 'formal_candidate_id': candidate.candidate_id,
            'formal_transition': candidate.transition.to_dict()})
    return tuple(choices)


def _deterministic_exact_selection_payload(context: qwen_task_context_domain.QwenTaskContext, choices: tuple[dict[str,
    Any], ...] | list[dict[str, Any]], *, observation: trusted_observation_domain.TrustedObservation |
    None=None) -> dict[str, Any] | None:
    """Choose only a unique candidate proved by the current observation."""

    target = observation.target_local_candidate() if observation is not None else None
    active_id = str(context.current_subgoal.get("subgoal_id") or "").strip()
    if active_id == 'input_exact_text':
        if target is None:
            return None
        current_value = getattr(target, "states", {}).get("value")
        authorized_text = getattr(context, "requested_input_text", None)
        allowed = {'clear_verified_text'} if isinstance(current_value, str) and isinstance(authorized_text,
            str) and (not authorized_text.startswith(current_value)) else {'tap_semantic', 'input_verified_text',
            'press_enter', 'clear_verified_text'}
        matching_choices = tuple((item for item in choices if str(item.get('element_id') or '') == target.element_id
            and str(item.get('action') or '') in allowed))
    else:
        expected_action = {'exact_back': 'back', 'exact_home': 'home', 'exact_open_recent_apps': 'open_recent_apps',
            'exact_tap_semantic': 'tap_semantic'}.get(active_id)
        matching_choices = tuple((item for item in choices if item.get('action') ==
            expected_action)) if expected_action is not None else ()
        if active_id == 'exact_tap_semantic' and target is not None:
            matching_choices = tuple((item for item in matching_choices if item.get('element_id') == target.element_id))
    if len(matching_choices) == 1:
        return _selection_payload(matching_choices[0], '结构化直推目录只有一个合法 canonical candidate。')

    if (getattr(context, 'current_execution_class', '') == 'effect' and bool(getattr(context, 'effect_action_allowed',
        False))):
        effect_choices = tuple((item for item in choices if _choice_applies_effect(item)))
        if len(effect_choices) == 1:
            return _selection_payload(effect_choices[0], 'canonical目录只有一个绑定当前EffectIntent的动作。')
        if effect_choices:
            return None

    if target is not None:
        same_target = tuple((item for item in choices if item.get('element_id') == target.element_id))
        if len(same_target) == 1:
            return _selection_payload(same_target[0], '单次Qwen画面的唯一目标与canonical目录唯一候选一致。')

    if (len(choices) == 1 and str(choices[0].get('action') or '') in {'back', 'home', 'open_recent_apps',
        'reveal_system_navigation', 'swipe', 'wait_for_change'}):
        return _selection_payload(choices[0], 'canonical目录只有一个坐标无关或容器级合法动作。')
    return None


def _selection_payload(choice: Mapping[str, Any], reason: str) -> dict[str, Any] | None:
    choice_id = str(choice.get("choice_id") or "").strip()
    return {'status': 'action', 'choice_id': choice_id, 'confidence': 1.0, 'reason': reason} if choice_id else None


def _choice_applies_effect(choice: Mapping[str, Any]) -> bool:
    transition = choice.get("formal_transition")
    if not isinstance(transition, Mapping):
        return False
    expectations = transition.get("expectations")
    if not isinstance(expectations, list):
        return False
    return any((isinstance(item, Mapping) and item.get('predicate') == 'effect.applied'
        and (item.get('operator') == 'equals') and (item.get('value') is True) for item in expectations))


def _hydrate_canonical_selection(payload: Mapping[str, Any], *, context: qwen_task_context_domain.QwenTaskContext,
    observation: trusted_observation_domain.TrustedObservation, choices: tuple[dict[str, Any],
    ...]) -> QwenVisualDecision:
    """Hydrate the already-selected immutable canonical candidate."""

    reject_if(set(payload) != {'status', 'choice_id', 'confidence', 'reason'}, VisionAgentError("本地 canonical 选择结构字段无效。"))
    reject_if(payload.get('status') != 'action', VisionAgentError("本地 canonical 选择结果必须是 action。"))
    matches = [item for item in choices if item.get('choice_id') == payload.get('choice_id')]
    reject_if(len(matches) != 1, VisionAgentError("本地选择引用了不存在或不唯一的 choice_id。"))
    choice = matches[0]
    kind = str(choice.get("action") or "").strip()
    reject_if(kind not in CANONICAL_ACTION_KINDS, VisionAgentError("canonical candidate 包含未知动作。"))
    reject_if(not isinstance(choice.get('expected_result'), Mapping) or not choice['expected_result'], VisionAgentError("canonical candidate 缺少可验证 expected_result。"))
    expected_result = dict(choice["expected_result"])
    params = {key: value for key, value in choice.items() if key not in {'choice_id', 'action', 'expected_result',
        'selection_context'}}
    bound_ids: list[str] = []
    if _targets_single_element(kind, params):
        element = observation.get_candidate(str(params.get("element_id") or ""))
        _bind_element_params(params, element)
        bound_ids.append(element.element_id)
        if kind == 'input_verified_text':
            params["text"] = context.requested_input_text
        elif kind == 'long_press':
            params["duration_ms"] = 800
    elif kind == 'drag':
        for prefix in ('source_', 'destination_'):
            element = observation.get_candidate(str(params.get(f"{prefix}element_id") or ""))
            _bind_element_params(params, element, prefix=prefix)
            bound_ids.append(element.element_id)
    params["expected_effect"] = expected_result
    action = SemanticAction(node_id=f'qwen_visual_revision_{context.revision}', action=kind, params=params)
    raw_confidence = payload.get("confidence")
    reject_if(isinstance(raw_confidence, bool) or not isinstance(raw_confidence, (int, float)), VisionAgentError("本地 canonical 选择 confidence 无效。"))
    confidence = min(float(raw_confidence), float(observation.scene.confidence),
        *(float(observation.get_candidate(item).confidence) for item in bound_ids))
    reason = str(payload.get("reason") or "").strip()[:500]
    decision = QwenVisualDecision(task_id=context.task_id, device_id=context.device_id, revision=context.revision,
        observation_id=observation.observation_id, fingerprint=observation.fingerprint,
        page_state=_page_state(observation.scene), trusted_observation=observation,
        proposal=GenericStepProposal(status='action', action=action, reason=reason),
        target_region=_canonical_target_region(action, observation), expected_result=expected_result,
        confidence=confidence, reason=reason)
    decision.validate(context)
    return decision


def _canonical_target_region(action: SemanticAction,
    observation: trusted_observation_domain.TrustedObservation) -> VisualTargetRegion:
    if _targets_single_element(action.action, action.params):
        element = observation.get_candidate(str(action.params.get('element_id') or '').strip())
        return VisualTargetRegion(kind='element', element_id=element.element_id, bounds=element.bounds,
            description=element.label or element.meaning)
    if action.action == 'drag':
        source = observation.get_candidate(str(action.params.get('source_element_id') or '').strip())
        destination = observation.get_candidate(str(action.params.get('destination_element_id') or '').strip())
        return VisualTargetRegion(
            kind="element_path",
            element_id=source.element_id,
            bounds=source.bounds,
            destination_element_id=destination.element_id,
            destination_bounds=destination.bounds,
            description=(
                f"{source.label or source.meaning} 到 "
                f"{destination.label or destination.meaning}"
            ),
        )
    system_descriptions = {'back': '系统返回区域', 'home': 'Android系统Home键', 'open_recent_apps': 'Android系统最近任务键',
        'reveal_system_navigation': 'Android系统导航栏'}
    return VisualTargetRegion(kind='system_navigation' if action.action in system_descriptions else 'screen',
        bounds=(0.0, 0.0, 1.0, 1.0), description=system_descriptions.get(action.action, '当前屏幕'))


def _normalize_available_action_kinds(value: Iterable[str] | None) -> frozenset[str]:
    if value is None:
        return QWEN_PROTOCOL_ACTIONS
    try:
        normalized = frozenset(str(item or "").strip() for item in value)
    except TypeError as exc:
        raise VisionAgentError("设备动作能力必须是可迭代字符串集合。") from exc
    reject_if('' in normalized, VisionAgentError("设备动作能力不能包含空值。"))
    unexpected = normalized - QWEN_PROTOCOL_ACTIONS
    reject_if(unexpected, VisionAgentError('设备动作能力包含协议外动作：' + ', '.join(sorted(unexpected))))
    reject_if(not normalized, VisionAgentError("设备没有任何可供本地选择的 canonical 动作。"))
    return normalized


def _local_blocked_decision(context: qwen_task_context_domain.QwenTaskContext,
    observation: trusted_observation_domain.TrustedObservation, *, reason: str) -> QwenVisualDecision:
    decision = QwenVisualDecision(task_id=context.task_id, device_id=context.device_id, revision=context.revision,
        observation_id=observation.observation_id, fingerprint=observation.fingerprint,
        page_state=_page_state(observation.scene), trusted_observation=observation,
        proposal=GenericStepProposal(status='blocked', reason=reason), target_region=None, expected_result={},
        confidence=min(float(observation.scene.confidence), 1.0), reason=reason)
    decision.validate(context)
    return decision


def _bind_element_params(params: dict[str, Any], element: UIElement, *, prefix: str='') -> None:
    params.update({f'{prefix}element_id': element.element_id, f'{prefix}target': element.meaning, f'{
        prefix}role': element.role, f'{prefix}label': element.label, f'{prefix}states': dict(element.states)})


def _page_state(scene: UIScene) -> dict[str, Any]:
    return {'foreground_app_id': scene.foreground_app_id, 'screen_id': scene.screen_id, 'summary': scene.summary,
        'overlays': list(scene.overlays)}


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0
