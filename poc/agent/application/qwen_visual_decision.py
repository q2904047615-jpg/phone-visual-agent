"""Single-step Qwen visual decision application service."""

from __future__ import annotations

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
from agent.domain.task_semantic_ir import TaskSemanticIR
from agent.domain.ui_scene import MIN_TARGET_CONFIDENCE, UIElement, UIScene
from agent.domain.vision_model import VisionAgentError, public_model_identity


QWEN_VISUAL_DECISION_PROTOCOL_VERSION = "2026-08-14-qwen-visual-decision-v5"
QWEN_VISUAL_DECISION_MODEL_ROLE = "trusted_observation_single_step_selector"
MIN_DECISION_CONFIDENCE = 0.72
SINGLE_ELEMENT_ACTIONS = frozenset({
    "tap_semantic", "dismiss_overlay", "input_verified_text", "press_enter",
    "clear_verified_text", "double_tap", "long_press",
})


def _targets_single_element( action_or_kind: SemanticAction | str, params: Mapping[str, Any] | None = None, ) -> bool:
    """Return whether this exact canonical action binds one observed element.

    Ordinary viewport swipes remain screen actions.  A swipe becomes an
    element action only when the canonical catalog binds its immutable
    ``element_id``; Qwen never invents that identity or any coordinates.
    """

    if isinstance(action_or_kind, SemanticAction):
        kind = action_or_kind.action
        values = action_or_kind.params
    else:
        kind = str(action_or_kind or "").strip()
        values = params or {}
    return kind in SINGLE_ELEMENT_ACTIONS or (kind == 'swipe' and bool(str(values.get('element_id') or '').strip()))


QWEN_PROTOCOL_ACTIONS = frozenset(CANONICAL_ACTION_KINDS)

ACTIONABLE_EXACT_TEXT_ROLES = frozenset({'button', 'icon', 'input', 'tab', 'toggle', 'list_item', 'keyboard_key'})
@dataclass(frozen=True)
class VisualTargetRegion:
    kind: str
    bounds: tuple[float, float, float, float]
    description: str
    element_id: str = ""
    destination_element_id: str = ""
    destination_bounds: tuple[float, float, float, float] | None = None

    def validate(self, observation: trusted_observation_domain.TrustedObservation, action: SemanticAction) -> None:
        expected = _canonical_target_region(action, observation)
        if self != expected:
            raise GenericStepPlanningError("目标区域没有逐项复用 canonical 候选。")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "element_id": self.element_id or None,
            "bounds": list(self.bounds),
            "destination_element_id": self.destination_element_id or None,
            "destination_bounds": (
                list(self.destination_bounds)
                if self.destination_bounds is not None
                else None
            ),
            "description": self.description,
        }


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
        if (
            self.task_id,
            self.device_id,
            self.revision,
            self.observation_id,
            self.fingerprint,
        ) != (
            context.task_id,
            context.device_id,
            context.revision,
            self.trusted_observation.observation_id,
            self.trusted_observation.fingerprint,
        ):
            raise GenericStepPlanningError('task/device/revision/observation/fingerprint 已过期或不匹配。')
        self.proposal.validate(self.trusted_observation.scene)
        if self.protocol_version != QWEN_VISUAL_DECISION_PROTOCOL_VERSION:
            raise GenericStepPlanningError("Qwen视觉决策协议版本无效。")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise GenericStepPlanningError("Qwen视觉决策置信度必须在0到1之间。")
        if not isinstance(self.expected_result, dict):
            raise GenericStepPlanningError("expected_result 必须是JSON对象。")

        action = self.proposal.action
        if self.proposal.status == "action":
            if action is None or self.target_region is None:
                raise GenericStepPlanningError("唯一下一动作缺少可信目标区域。")
            if not self.expected_result:
                raise GenericStepPlanningError("唯一下一动作缺少可验证预期结果。")
            if not context.effect_action_allowed and context.current_execution_class == "effect":
                raise GenericStepPlanningError("风险确认门未满足，禁止产生外部状态动作。")
            self.target_region.validate(self.trusted_observation, action)
            if dict(action.params.get("expected_effect") or {}) != self.expected_result:
                raise GenericStepPlanningError("动作 expected_effect 与顶层预期不一致。")
            if float(self.confidence) < MIN_DECISION_CONFIDENCE:
                raise GenericStepPlanningError("动作置信度不足，必须 blocked。")
        elif self.target_region is not None or self.expected_result:
            raise GenericStepPlanningError("blocked 不能携带动作目标区域。")

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "task_id": self.task_id,
            "device_id": self.device_id,
            "revision": self.revision,
            "observation_id": self.observation_id,
            "fingerprint": self.fingerprint,
            "page_state": dict(self.page_state),
            "trusted_observation": self.trusted_observation.to_dict(),
            "status": self.proposal.status,
            "next_action": self.proposal.action.to_dict() if self.proposal.action else None,
            "target_region": self.target_region.to_dict() if self.target_region else None,
            "expected_result": dict(self.expected_result),
            "confidence": float(self.confidence),
            "reason": self.reason,
        }


class QwenVisualDecisionObserver:
    """Select one canonical action locally from one trusted observation."""

    def __init__( self, provider: Any, *, trusted_observation_frame_validator: Callable[..., None], ) -> None:
        self.provider = provider
        self.trusted_observation_frame_validator = trusted_observation_frame_validator
        self.last_raw_response = ""
        self.last_diagnostics: dict[str, Any] = {}
        self._metrics = {'decision_count': 0, 'deterministic_action_count': 0, 'final_blocked_count': 0}

    def status(self) -> dict[str, Any]:
        value = dict(self.provider.status())
        decisions = self._metrics["decision_count"]
        value.update(
            {
                "visual_decision_protocol": QWEN_VISUAL_DECISION_PROTOCOL_VERSION,
                "task_context_protocol": qwen_task_context_domain.SUPPORTED_TASK_CONTEXT_PROTOCOL,
                "model_role": QWEN_VISUAL_DECISION_MODEL_ROLE,
                "hardware_actions_enabled": False,
                "selection_authority": "canonical_action_catalog_local",
                **self._metrics,
                "deterministic_action_rate": _ratio(
                    self._metrics["deterministic_action_count"], decisions
                ),
                "final_blocked_rate": _ratio(
                    self._metrics["final_blocked_count"], decisions
                ),
            }
        )
        return value

    def decide(
        self,
        *,
        frames: list[Image.Image],
        task_context: qwen_task_context_domain.QwenTaskContext | dict[str, Any],
        trusted_observation: trusted_observation_domain.TrustedObservation,
        decision_number: int = 1,
        available_action_kinds: Iterable[str] | None = None,
    ) -> QwenVisualDecision:
        started = time.perf_counter()
        self.last_raw_response = ""
        self.last_diagnostics = {}
        context = (
            task_context
            if isinstance(task_context, qwen_task_context_domain.QwenTaskContext)
            else qwen_task_context_domain.QwenTaskContext.from_dict(task_context)
        )
        context.validate()
        available_actions = _normalize_available_action_kinds(available_action_kinds)
        # These are the same read-only frames that established the trusted
        # observation, so apply the observer's one-leading-frame tolerance.
        # Confirmation-time recapture and post-action verification use their
        # own stricter full-window stability checks.
        self.trusted_observation_frame_validator(trusted_observation, frames, allow_leading_outlier=True)
        if context.device_id != trusted_observation.device_id:
            raise VisionAgentError("任务 device_id 与可信观察不一致。")
        self._metrics["decision_count"] += 1
        canonical_choices = _selection_choices(context, trusted_observation, available_actions)
        canonical_action_kinds = sorted({str(item['action']) for item in canonical_choices})

        model_identity = public_model_identity(self.provider.status())
        base_diagnostics = {
            "visual_decision_protocol": QWEN_VISUAL_DECISION_PROTOCOL_VERSION,
            "vision_model": model_identity,
            "task_id": context.task_id,
            "device_id": context.device_id,
            "revision": context.revision,
            "observation_id": trusted_observation.observation_id,
            "fingerprint": trusted_observation.fingerprint,
            "model_calls": 0,
            "hardware_actions_enabled": False,
            "available_action_kinds": canonical_action_kinds,
            "canonical_choice_count": len(canonical_choices),
            "canonical_choices": [
                {
                    "action": str(item.get("action") or ""),
                    "direction": str(item.get("direction") or ""),
                    "element_id": str(item.get("element_id") or ""),
                }
                for item in canonical_choices
            ],
            "device_action_kinds": sorted(available_actions),
        }
        self.last_diagnostics = dict(base_diagnostics)

        block = (
            (
                "风险确认门未满足，本轮禁止提出外部状态动作。",
                "effect_gate",
            )
            if context.current_execution_class == "effect"
            and not context.effect_action_allowed
            else _exact_text_candidate_block(context, trusted_observation)
            or _identity_text_candidate_block(context, trusted_observation)
        )
        if block is not None:
            reason, block_code = block
            decision = _local_blocked_decision(context, trusted_observation, reason=reason)
            self._metrics["final_blocked_count"] += 1
            self.last_diagnostics.update(
                {
                    "local_safety_block": block_code,
                    "decision_status": "blocked",
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                }
            )
            return decision

        deterministic_selection = _deterministic_exact_selection_payload(
            context,
            canonical_choices,
            observation=trusted_observation,
        )
        if deterministic_selection is not None:
            raw = json.dumps(deterministic_selection, ensure_ascii=False, separators=(',', ':'))
            decision = _hydrate_canonical_selection(
                deterministic_selection,
                context=context,
                observation=trusted_observation,
                choices=canonical_choices,
            )
            self.last_raw_response = raw
            self._metrics["deterministic_action_count"] += 1
            self.last_diagnostics.update(
                {
                    "local_deterministic_selection": True,
                    "decision_status": decision.proposal.status,
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                }
            )
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
        self.last_diagnostics.update(
            {
                "local_safety_block": "single_step_candidate_not_unique",
                "local_deterministic_selection": False,
                "model_calls": 0,
                "decision_status": "blocked",
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        )
        return decision


def _selection_choices(
    context: qwen_task_context_domain.QwenTaskContext,
    observation: trusted_observation_domain.TrustedObservation,
    available_action_kinds: frozenset[str],
) -> tuple[dict[str, Any], ...]:
    """Build generic action choices from the trusted scene, never app steps."""

    choices: list[dict[str, Any]] = []
    if context.semantic_ir is None:
        raise VisionAgentError("typed v4 视觉选择缺少 canonical TaskSemanticIR。")
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
    # later selects by digest and ID.
    for candidate in formal_report.candidates:
        choices.append(
            {
                "choice_id": f"choice_{len(choices) + 1}",
                "action": candidate.action_kind,
                **dict(candidate.parameters),
                "expected_result": canonical_candidate_expected_result(
                    candidate,
                    observation.scene,
                ),
                "formal_candidate_id": candidate.candidate_id,
                "formal_report_digest": formal_report.report_digest,
                "formal_transition": candidate.transition.to_dict(),
            }
        )
    return tuple(choices)


def _deterministic_exact_selection_payload(
    context: qwen_task_context_domain.QwenTaskContext,
    choices: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    *,
    observation: trusted_observation_domain.TrustedObservation | None = None,
) -> dict[str, Any] | None:
    """Choose only a unique candidate proved by the current observation."""

    target = observation.target_local_candidate() if observation is not None else None
    active_id = str(context.current_subgoal.get("subgoal_id") or "").strip()
    if active_id == "input_exact_text":
        if target is None:
            return None
        current_value = getattr(target, "states", {}).get("value")
        authorized_text = getattr(context, "requested_input_text", None)
        allowed = {"clear_verified_text"} if (
            isinstance(current_value, str)
            and isinstance(authorized_text, str)
            and not authorized_text.startswith(current_value)
        ) else {
            "tap_semantic", "input_verified_text", "press_enter", "clear_verified_text",
        }
        matching_choices = tuple(
            item for item in choices
            if str(item.get("element_id") or "") == target.element_id
            and str(item.get("action") or "") in allowed
        )
    else:
        expected_action = {
            "exact_back": "back",
            "exact_home": "home",
            "exact_open_recent_apps": "open_recent_apps",
            "exact_tap_semantic": "tap_semantic",
        }.get(active_id)
        matching_choices = (
            tuple(item for item in choices if item.get("action") == expected_action)
            if expected_action is not None
            else ()
        )
        if active_id == "exact_tap_semantic" and target is not None:
            matching_choices = tuple((item for item in matching_choices if item.get('element_id') == target.element_id))
    if len(matching_choices) == 1:
        return _selection_payload(matching_choices[0], '结构化直推目录只有一个合法 canonical candidate。')

    if (
        getattr(context, "current_execution_class", "") == "effect"
        and bool(getattr(context, "effect_action_allowed", False))
    ):
        effect_choices = tuple((item for item in choices if _choice_applies_effect(item)))
        if len(effect_choices) == 1:
            return _selection_payload(effect_choices[0], 'canonical目录只有一个绑定当前EffectIntent的动作。')
        if effect_choices:
            return None

    if target is not None:
        same_target = tuple((item for item in choices if item.get('element_id') == target.element_id))
        if len(same_target) == 1:
            return _selection_payload(same_target[0], '单次Qwen画面的唯一目标与canonical目录唯一候选一致。')

    if len(choices) == 1 and str(choices[0].get("action") or "") in {
        "back",
        "home",
        "open_recent_apps",
        "reveal_system_navigation",
        "swipe",
        "wait_for_change",
    }:
        return _selection_payload(choices[0], 'canonical目录只有一个坐标无关或容器级合法动作。')
    return None


def _selection_payload( choice: Mapping[str, Any], reason: str ) -> dict[str, Any] | None:
    choice_id = str(choice.get("choice_id") or "").strip()
    return {'status': 'action', 'choice_id': choice_id, 'confidence': 1.0, 'reason': reason} if choice_id else None


def _choice_applies_effect(choice: Mapping[str, Any]) -> bool:
    transition = choice.get("formal_transition")
    if not isinstance(transition, Mapping):
        return False
    expectations = transition.get("expectations")
    if not isinstance(expectations, list):
        return False
    return any(
        isinstance(item, Mapping)
        and item.get("predicate") == "effect.applied"
        and item.get("operator") == "equals"
        and item.get("value") is True
        for item in expectations
    )


def _hydrate_canonical_selection(
    payload: Mapping[str, Any],
    *,
    context: qwen_task_context_domain.QwenTaskContext,
    observation: trusted_observation_domain.TrustedObservation,
    choices: tuple[dict[str, Any], ...],
) -> QwenVisualDecision:
    """Hydrate the already-selected immutable canonical candidate."""

    if set(payload) != {"status", "choice_id", "confidence", "reason"}:
        raise VisionAgentError("本地 canonical 选择结构字段无效。")
    if payload.get("status") != "action":
        raise VisionAgentError("本地 canonical 选择结果必须是 action。")
    matches = [item for item in choices if item.get('choice_id') == payload.get('choice_id')]
    if len(matches) != 1:
        raise VisionAgentError("本地选择引用了不存在或不唯一的 choice_id。")
    choice = matches[0]
    kind = str(choice.get("action") or "").strip()
    if kind not in CANONICAL_ACTION_KINDS:
        raise VisionAgentError("canonical candidate 包含未知动作。")
    if not isinstance(choice.get("expected_result"), Mapping) or not choice["expected_result"]:
        raise VisionAgentError("canonical candidate 缺少可验证 expected_result。")
    expected_result = dict(choice["expected_result"])
    params = {
        key: value
        for key, value in choice.items()
        if key not in {"choice_id", "action", "expected_result", "selection_context"}
    }
    bound_ids: list[str] = []
    if _targets_single_element(kind, params):
        element = observation.get_candidate(str(params.get("element_id") or ""))
        _bind_element_params(params, element)
        bound_ids.append(element.element_id)
        if kind == "input_verified_text":
            params["text"] = context.requested_input_text
        elif kind == "long_press":
            params["duration_ms"] = 800
    elif kind == "drag":
        for prefix in ("source_", "destination_"):
            element = observation.get_candidate(str(params.get(f"{prefix}element_id") or ""))
            _bind_element_params(params, element, prefix=prefix)
            bound_ids.append(element.element_id)
    params["expected_effect"] = expected_result
    action = SemanticAction(node_id=f'qwen_visual_revision_{context.revision}', action=kind, params=params)
    raw_confidence = payload.get("confidence")
    if isinstance(raw_confidence, bool) or not isinstance( raw_confidence, (int, float) ):
        raise VisionAgentError("本地 canonical 选择 confidence 无效。")
    confidence = min(
        float(raw_confidence), float(observation.scene.confidence),
        *(float(observation.get_candidate(item).confidence) for item in bound_ids),
    )
    reason = str(payload.get("reason") or "").strip()[:500]
    decision = QwenVisualDecision(
        task_id=context.task_id,
        device_id=context.device_id,
        revision=context.revision,
        observation_id=observation.observation_id,
        fingerprint=observation.fingerprint,
        page_state=_page_state(observation.scene),
        trusted_observation=observation,
        proposal=GenericStepProposal(
            status="action",
            action=action,
            reason=reason,
        ),
        target_region=_canonical_target_region(action, observation),
        expected_result=expected_result,
        confidence=confidence,
        reason=reason,
    )
    decision.validate(context)
    return decision


def _canonical_target_region(
    action: SemanticAction,
    observation: trusted_observation_domain.TrustedObservation,
) -> VisualTargetRegion:
    if _targets_single_element(action):
        element = observation.get_candidate(str(action.params.get('element_id') or '').strip())
        return VisualTargetRegion(
            kind="element",
            element_id=element.element_id,
            bounds=element.bounds,
            description=element.label or element.meaning,
        )
    if action.action == "drag":
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
    system_descriptions = {
        "back": "系统返回区域",
        "home": "Android系统Home键",
        "open_recent_apps": "Android系统最近任务键",
        "reveal_system_navigation": "Android系统导航栏",
    }
    return VisualTargetRegion(
        kind=(
            "system_navigation"
            if action.action in system_descriptions
            else "screen"
        ),
        bounds=(0.0, 0.0, 1.0, 1.0),
        description=system_descriptions.get(action.action, "当前屏幕"),
    )


def _normalize_available_action_kinds( value: Iterable[str] | None, ) -> frozenset[str]:
    if value is None:
        return QWEN_PROTOCOL_ACTIONS
    try:
        normalized = frozenset(str(item or "").strip() for item in value)
    except TypeError as exc:
        raise VisionAgentError("设备动作能力必须是可迭代字符串集合。") from exc
    if "" in normalized:
        raise VisionAgentError("设备动作能力不能包含空值。")
    unexpected = normalized - QWEN_PROTOCOL_ACTIONS
    if unexpected:
        raise VisionAgentError('设备动作能力包含协议外动作：' + ', '.join(sorted(unexpected)))
    if not normalized:
        raise VisionAgentError("设备没有任何可供本地选择的 canonical 动作。")
    return normalized


def _typed_required_action_kinds(context: qwen_task_context_domain.QwenTaskContext) -> frozenset[str]:
    """Read typed action requirements for identity/evidence checks only."""

    semantic_ir = context.semantic_ir
    if semantic_ir is None:
        return frozenset()
    active_id = str(context.current_subgoal.get("subgoal_id") or "")
    subgoal = next((item for item in semantic_ir.subgoals if item.subgoal_id == active_id), None)
    if subgoal is None:
        return frozenset()
    constraints = {item.constraint_id: item for item in semantic_ir.constraints}
    return frozenset(
        str(constraints[constraint_ref].value)
        for constraint_ref in subgoal.constraint_refs
        if constraint_ref in constraints
        and constraints[constraint_ref].kind == "required_action"
    )


def _launcher_app_entry_candidate_ids(
    context: qwen_task_context_domain.QwenTaskContext,
    observation: trusted_observation_domain.TrustedObservation,
) -> tuple[str, ...]:
    """Return one typed App entry before applying inner-page text gates."""

    semantic_ir = context.semantic_ir
    scene = observation.scene
    if semantic_ir is None:
        return ()
    current_identity = f"{scene.foreground_app_id} {scene.screen_id}".casefold()
    if not any(token in current_identity for token in ("launcher", "home_screen", "desktop")):
        return ()
    active_id = str(context.current_subgoal.get("subgoal_id") or "")
    typed_subgoal = next((item for item in semantic_ir.subgoals if item.subgoal_id == active_id), None)
    surfaces = {item.surface_id: item for item in semantic_ir.surfaces}
    target_surface = surfaces.get(typed_subgoal.surface_ref) if typed_subgoal is not None else None
    if target_surface is None or target_surface.kind != "app":
        return ()
    app_name = str(target_surface.app_name or "").strip()
    if not app_name:
        return ()
    matches = tuple(
        element.element_id
        for element in scene.elements
        if element.role in {"button", "icon", "list_item"}
        and element.label == app_name
        and float(element.confidence) >= MIN_TARGET_CONFIDENCE
        and element.states.get("goal_relevant") is True
        and element.states.get("fully_visible") is True
    )
    return matches if len(matches) == 1 else ()


def _exact_text_candidate_block(
    context: qwen_task_context_domain.QwenTaskContext,
    observation: trusted_observation_domain.TrustedObservation,
) -> tuple[str, str] | None:
    """Reject missing or ambiguous structured exact-text targets locally."""

    if _launcher_app_entry_candidate_ids(context, observation):
        return None
    for required_text in context.exact_text_requirements:
        identity_matches = _identity_scoped_exact_text_matches(context, observation, required_text)
        if identity_matches is not None:
            if not identity_matches:
                return (f'当前可信页面身份中不存在逐字一致文字：{required_text}', 'exact_text_missing')
            if len(identity_matches) != 1:
                return (f'逐字一致页面身份不唯一：{required_text}，共{len(identity_matches)}个', 'exact_text_ambiguous')
            continue
        matches = _matching_exact_text_candidates(context, observation, required_text)
        if not matches:
            return (f'当前可信候选中不存在逐字一致文字：{required_text}', 'exact_text_missing')
        if len(matches) != 1:
            return (f'逐字一致文字目标不唯一：{required_text}，共{len(matches)}个', 'exact_text_ambiguous')
    return None


def _identity_text_candidate_block(
    context: qwen_task_context_domain.QwenTaskContext,
    observation: trusted_observation_domain.TrustedObservation,
) -> tuple[str, str] | None:
    if _launcher_app_entry_candidate_ids(context, observation):
        return None
    for required_text in context.identity_text_requirements:
        matches = _matching_identity_text_candidates(observation, required_text)
        if not matches:
            return (f"当前画面不存在收件人逐字身份：{required_text}", "identity_missing")
        if len(matches) != 1:
            return (f"当前画面收件人身份不唯一：{required_text}", "identity_ambiguous")
    return None


def _matching_identity_text_candidates(
    observation: trusted_observation_domain.TrustedObservation,
    required_text: str,
) -> list[str]:
    return _identity_candidate_ids(
        observation.scene,
        meanings={
            "recipient_identity", "conversation_identity",
            "conversation_title", "page_title",
        },
        text_matches=lambda item: required_text in (item.label, *item.evidence),
    )


_IDENTITY_SCOPED_EXACT_TEXT_ACTIONS = frozenset({
    "swipe", "back", "home", "open_recent_apps",
    "reveal_system_navigation", "wait_for_change",
})

_SURFACE_IDENTITY_TYPE_SUFFIXES = frozenset({
    "页", "页面", "界面", "屏幕", "窗口", "主页", "首页", "聊天页",
    "聊天页面", "对话页", "对话页面", "详情页", "详情页面", "列表页",
    "列表页面", "设置页", "设置页面", " page", " screen", " window",
    " chat page", " conversation page", " detail page", " list page",
    " settings page",
})


def _surface_identity_text_matches(label: str, required_text: str) -> bool:
    """Match one literal title plus a bounded generic surface-type suffix."""

    literal = label.strip()
    required = required_text.strip()
    if not literal or not required:
        return False
    if literal == required:
        return True
    if required.startswith(literal):
        return required[len(literal) :].casefold() in _SURFACE_IDENTITY_TYPE_SUFFIXES
    if literal.startswith(required):
        return literal[len(required) :].casefold() in _SURFACE_IDENTITY_TYPE_SUFFIXES
    return False


def _matching_surface_identity_candidates(
    observation: trusted_observation_domain.TrustedObservation,
    required_text: str,
) -> list[str]:
    return _identity_candidate_ids(
        observation.scene,
        meanings={"conversation_title", "page_title"},
        text_matches=lambda item: _surface_identity_text_matches(
            item.label, required_text
        ),
    )


def _identity_candidate_ids(
    scene: UIScene,
    *,
    meanings: set[str],
    text_matches: Callable[[UIElement], bool],
) -> list[str]:
    return [
        item.element_id for item in scene.elements
        if float(item.confidence) >= MIN_TARGET_CONFIDENCE
        and item.states.get("visible") is not False
        and item.role != "input"
        and (
            item.states.get("identity_anchor") is True
            or item.states.get("goal_relevant") is True
            or item.meaning.strip().casefold() in meanings
        )
        and text_matches(item)
    ]


def _surface_descriptor_identity_candidate_ids( scene: UIScene, required_text: str, ) -> tuple[str, ...]:
    """Bind a generic page descriptor to its visible literal title.

    A typed ``target_ui_label`` can name the current page while the physical
    target is a separate input or button.  Only a strict generic type suffix
    plus a shorter visible title establishes this relation.  Exact labels stay
    element targets; zero or multiple titles never grant action authority.
    """

    required = str(required_text or "").strip()
    if not required:
        return ()
    matches = tuple(_identity_candidate_ids(
        scene,
        meanings={"conversation_title", "page_title"},
        text_matches=lambda item: _surface_identity_text_matches(
            item.label, required
        ),
    ))
    has_descriptor_title = any(
        element.element_id in matches and element.label.strip() != required
        for element in scene.elements
    )
    return matches if has_descriptor_title else ()


def _identity_scoped_exact_text_matches(
    context: qwen_task_context_domain.QwenTaskContext,
    observation: trusted_observation_domain.TrustedObservation,
    required_text: str,
) -> list[str] | None:
    """Resolve exact text as surface identity for non-element actions.

    A literal carried by the typed goal may name the current page or
    container (for example a conversation title) while the typed active
    action is a viewport gesture or a coordinate-free system action.  In that
    case the literal must still be uniquely visible, but binding the physical
    action to that title would invert the entity relation.  Element-bound
    actions deliberately keep the existing strict target requirement.
    """

    descriptor_matches = _surface_descriptor_identity_candidate_ids(observation.scene, required_text)
    if descriptor_matches:
        return list(descriptor_matches)
    required_actions = _typed_required_action_kinds(context)
    if ( len(required_actions) != 1 or not required_actions.issubset(_IDENTITY_SCOPED_EXACT_TEXT_ACTIONS) ):
        return None
    return _matching_surface_identity_candidates(observation, required_text)


def _matching_exact_text_candidates(
    context: qwen_task_context_domain.QwenTaskContext,
    observation: trusted_observation_domain.TrustedObservation,
    required_text: str,
) -> list[str]:
    active_input_matches = _active_input_transaction_exact_candidate_ids(context, observation, required_text)
    if active_input_matches is not None:
        return active_input_matches
    roles = set(context.exact_text_target_roles)
    meanings = set(context.exact_text_target_meanings)
    matches: list[str] = []
    for element in observation.scene.elements:
        if float(element.confidence) < MIN_TARGET_CONFIDENCE:
            continue
        literal_match = required_text in (element.label, *element.evidence)
        if not literal_match:
            continue
        if roles and element.role not in roles:
            continue
        if meanings and element.meaning.strip().casefold() not in meanings:
            continue
        if context.current_execution_class != "observe" and not roles:
            if element.role not in ACTIONABLE_EXACT_TEXT_ROLES:
                continue
        matches.append(element.element_id)
    return matches


def _active_input_transaction_exact_candidate_ids(
    context: qwen_task_context_domain.QwenTaskContext,
    observation: trusted_observation_domain.TrustedObservation,
    required_text: str,
) -> list[str] | None:
    """Bind a non-empty visible prefix to its sole typed input field."""
    semantic_ir = context.semantic_ir
    if semantic_ir is None:
        return None
    active_id = str(context.current_subgoal.get("subgoal_id") or "")
    fields = tuple(
        field
        for field in semantic_ir.input_fields
        if active_id in field.source_subgoal_ids
        and field.field_label in {"", required_text}
    )
    if len(fields) != 1:
        return None
    field = fields[0]
    payload = next((item for item in semantic_ir.entities if item.entity_id == field.payload_ref), None)
    if ( payload is None or payload.role != "input_text" or not isinstance(payload.value, str) or not payload.value ):
        return None

    field_elements = [
        item for item in observation.scene.elements
        if item.states.get("input_field_id") == field.field_id
    ]
    if not any( isinstance(item.states.get("value"), str) and item.states["value"] for item in field_elements ):
        return None
    return [
        item.element_id for item in field_elements
        if item.role == "input"
        and item.meaning == "application_text_input"
        and float(item.confidence) >= MIN_TARGET_CONFIDENCE
        and item.states.get("goal_relevant") is True
        and item.states.get("fully_visible") is True
        and (
            not field.field_label
            or item.states.get("input_field_label") == field.field_label
        )
        and isinstance(item.states.get("value"), str)
        and bool(item.states["value"])
        and payload.value.startswith(item.states["value"])
        and item.label == item.states["value"]
    ]


def _local_blocked_decision(
    context: qwen_task_context_domain.QwenTaskContext,
    observation: trusted_observation_domain.TrustedObservation,
    *,
    reason: str,
) -> QwenVisualDecision:
    decision = QwenVisualDecision(
        task_id=context.task_id,
        device_id=context.device_id,
        revision=context.revision,
        observation_id=observation.observation_id,
        fingerprint=observation.fingerprint,
        page_state=_page_state(observation.scene),
        trusted_observation=observation,
        proposal=GenericStepProposal(status="blocked", reason=reason),
        target_region=None,
        expected_result={},
        confidence=min(float(observation.scene.confidence), 1.0),
        reason=reason,
    )
    decision.validate(context)
    return decision


def _bind_element_params( params: dict[str, Any], element: UIElement, *, prefix: str = "" ) -> None:
    params.update({
        f"{prefix}element_id": element.element_id,
        f"{prefix}target": element.meaning,
        f"{prefix}role": element.role,
        f"{prefix}label": element.label,
        f"{prefix}states": dict(element.states),
    })


def _page_state(scene: UIScene) -> dict[str, Any]:
    return {
        "foreground_app_id": scene.foreground_app_id,
        "screen_id": scene.screen_id,
        "summary": scene.summary,
        "overlays": list(scene.overlays),
    }


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0
