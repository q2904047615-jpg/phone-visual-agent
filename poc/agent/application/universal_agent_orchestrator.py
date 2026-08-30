"""Application orchestration for the generic one-action visual loop."""

from __future__ import annotations

from agent.domain.validation import DataclassWire, canonical_digest, reject_if
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any, Callable, NoReturn
import uuid

from agent.domain.task_graph import (
    ControllerTransitionEvidenceRef,
    TaskGraphError,
    DynamicTaskGraph,
    ObservedState,
    VerifiedActionTransition,
    VisualClaimEvidenceRef,
    build_exact_action_task_graph,
    build_exact_input_task_graph,
)
from agent.application.runtime_session import (
    CORRECTIVE_RETRY_PROTOCOL_VERSION,
    POST_ACTION_TRANSITION_PROTOCOL_VERSION,
    UniversalAgentSessionState,
)
from agent.domain.action_capabilities import build_device_capability_snapshot
from agent.domain.canonical_action_kinds import CANONICAL_ACTION_KINDS
from agent.domain import (
    AgentEvidenceStoreFactory,
    AppSurfaceLineageAuthority,
    AppSurfaceLineageError,
    CANONICAL_SELECTION_RECEIPT_VERSION,
    CanonicalSelectionReceipt,
    ConfirmationAuthority,
    DeviceTaskRegistryPort,
    EffectConfirmationAuthority,
    EvidenceStoreError,
    VerifiedAppSurfaceLineage,
)
from agent.application.action_adapter import GenericActionAdapterError, GenericSingleActionAdapterPort
from agent.domain.generic_goal import GenericIntentDraft, VisibleGoalEvidence
from agent.domain.canonical_action_protocol import CanonicalActionProtocolError, GenericStepProposal
from agent.domain.qwen_task_context import QwenTaskContext
from agent.domain.universal_action_controller import CONTROLLER_INPUT_PREEDIT_PENDING
from agent.domain.task_semantic_ir import TaskSemanticIRError, compile_formal_semantic_authority, effect_preview_digest
from agent.domain.verified_text_transaction import (
    VerifiedTextTransactionError,
    plan_next_verified_input,
)
from agent.application.vision_usage import VisionSessionUsageLedger


POST_ACTION_OUTCOMES = frozenset({"matched", "mismatched"})
CORRECTIVE_RETRY_IMPACTS = frozenset({"read_only", "navigation_only"})
CORRECTIVE_RETRY_ACTION_KINDS = frozenset({'back', 'dismiss_overlay', 'double_tap', 'drag', 'home', 'open_recent_apps',
    'long_press', 'swipe', 'tap_semantic'})
_TRANSIENT_ACTION_KEYS = frozenset({'node_id', 'element_id', 'source_element_id', 'destination_element_id', 'bounds',
    'source_bounds', 'destination_bounds', 'point', 'normalized_point', 'before_fingerprint', 'observation_id',
    'fingerprint'})


def _allows_fresh_observation_corrective_retry(*, impact: str, action_kind: str) -> bool:
    """Return whether one freshly replanned physical correction is allowed."""

    return str(impact or '').strip() in CORRECTIVE_RETRY_IMPACTS and str(action_kind
        or '').strip() in CORRECTIVE_RETRY_ACTION_KINDS


def _validate_visible_completion_condition_progress(previous: DynamicTaskGraph, revised: DynamicTaskGraph,
    observed: ObservedState) -> None:
    """Allow evidence-backed condition state progress without semantic rewrites."""

    old_conditions = tuple(previous.completion_conditions)
    new_conditions = tuple(revised.completion_conditions)
    old_ids = tuple(item.condition_id for item in old_conditions)
    new_ids = tuple(item.condition_id for item in new_conditions)
    reject_if(old_ids != new_ids, UniversalAgentOrchestratorError('可见状态证据推进不得增加、删除或重排全局完成条件。'))
    typed_refs = {item.ref_id for item in observed.visual_claim_evidence_refs}
    current_evidence = typed_refs if typed_refs else set(observed.visible_evidence).union(
        observed.grounded_visual_facts)
    for (old, new) in zip(old_conditions, new_conditions):
        reject_if(new.description != old.description or new.evidence_required != old.evidence_required, UniversalAgentOrchestratorError(f'可见状态证据推进不得改写全局完成条件定义：{old.condition_id}。'))
        reject_if(old.satisfied and (not new.satisfied), UniversalAgentOrchestratorError(f'可见状态证据推进不得撤销已满足的全局完成条件：{old.condition_id}。'))
        old_evidence = set(old.evidence)
        new_evidence = set(new.evidence)
        reject_if(not old_evidence.issubset(new_evidence), UniversalAgentOrchestratorError(f'可见状态证据推进不得删除既有全局完成证据：{old.condition_id}。'))
        reject_if(not (new_evidence - old_evidence).issubset(current_evidence), UniversalAgentOrchestratorError(f'可见状态证据推进使用了当前观察之外的全局完成证据：{old.condition_id}。'))


def _stable_action_payload(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _stable_action_payload(item) for key, item in sorted(value.items(),
            key=lambda pair: str(pair[0])) if str(key) not in _TRANSIENT_ACTION_KEYS}
    if isinstance(value, (list, tuple)):
        return [_stable_action_payload(item) for item in value]
    return value


def _action_digest(action: Any) -> str:
    reject_if(action is None, UniversalAgentOrchestratorError("动作摘要缺少语义动作。"))
    payload = action.to_dict() if callable(getattr(action, "to_dict", None)) else action
    return canonical_digest(payload)


def _action_equivalence_digest(action: Any) -> str:
    reject_if(action is None, UniversalAgentOrchestratorError("动作等价摘要缺少语义动作。"))
    payload = action.to_dict() if callable(getattr(action, "to_dict", None)) else action
    return canonical_digest(_stable_action_payload(payload))


def _subgoal_progress_signature(subgoal: Any) -> str:
    if subgoal is None:
        return ""
    payload = {'subgoal_id': str(getattr(subgoal, 'subgoal_id', '')), 'objective': str(getattr(subgoal, 'objective',
        '')), 'completion_conditions': list(getattr(subgoal, 'completion_conditions', ())),
        'external_impact': str(getattr(subgoal, 'external_impact', ''))}
    return canonical_digest(payload)


def _confirmation_effect_ids(graph: DynamicTaskGraph, current: Any | None) -> tuple[str, ...]:
    """Return only locally-policy-bound confirmation risks for one subgoal."""

    if current is None:
        return ()
    active_ids = set(tuple(getattr(current, "risk_action_ids", ()) or ()))
    return tuple(sorted((risk.risk_id for risk in graph.risk_actions if risk.risk_id in active_ids
        and risk.confirmation_required)))


def _requires_effect_confirmation(graph: DynamicTaskGraph, current: Any | None) -> bool:
    return bool(_confirmation_effect_ids(graph, current))


def _effect_confirmation_material(graph: DynamicTaskGraph, current: Any) -> tuple[str, dict[str, Any]]:
    effect_ids = _confirmation_effect_ids(graph, current)
    risks = [risk for risk in graph.risk_actions if risk.risk_id in set(effect_ids)]
    reject_if({risk.risk_id for risk in risks} != set(effect_ids), UniversalAgentOrchestratorError('效果确认引用了任务图中不存在的 EffectIntent。'))
    serialized_effects = {item['effect_id']: item for item in graph.to_dict()['effect_intents']}
    selected_effects = [serialized_effects[effect_id] for effect_id in effect_ids]
    preview: dict[str, Any] = {'kind': 'typed_effects', 'effect_ids': list(effect_ids),
        'effects': [{'effect_id': item['effect_id'], 'kind': item['kind'],
        'expected_results': list(item['expected_results']),
        'policy_level': item['local_policy']['policy_level']} for item in selected_effects]}
    payload = {'protocol_version': '2026-08-20-typed-effect-confirmation-v1', 'task_id': graph.task_id,
        'device_id': graph.device_id, 'revision': graph.revision, 'subgoal_id': current.subgoal_id,
        'effect_ids': list(effect_ids), 'effect_intents': selected_effects,
        'goal': {'target_apps': [{'app_id': app.app_id, 'app_name': app.app_name} for app in graph.goal.target_apps],
        'entities': graph.goal.entities}, 'preview': preview}
    return canonical_digest(payload), preview


class UniversalAgentOrchestratorError(RuntimeError):
    pass


class ObservationBridge:
    """Translate protocol objects without inventing actions or business flow."""

    _APP_REFERENCE_IDS = frozenset({'unknown', 'current_foreground', 'current_app', 'foreground_app', 'target_app',
        'active_app'})
    _SURFACE_OBSERVATION_IDENTITIES = {'device': ('device', '设备界面'), 'system': ('system', '系统界面'),
        'current_surface': ('current_surface', '当前界面')}

    @classmethod
    def _active_app_entry_target_label(cls, graph: DynamicTaskGraph, active: Any) -> str:
        """Project one unambiguous typed App name into the current observation node without action authority."""

        if active is None or active.external_impact != 'navigation_only':
            return ""
        objective = str(active.objective or "").strip()
        if not objective:
            return ""

        eligible_apps = [app for app in graph.goal.target_apps if str(app.app_name or '').strip() and str(app.app_id
            or '').strip().casefold() not in cls._APP_REFERENCE_IDS]
        mentioned_apps = [app for app in eligible_apps if str(app.app_name or '').strip().casefold()
            in objective.casefold()]
        if len(mentioned_apps) != 1:
            return ""

        matches: list[str] = []
        for app in mentioned_apps:
            app_id = str(app.app_id or "").strip().casefold()
            app_name = str(app.app_name or "").strip()
            assert app_name and app_id not in cls._APP_REFERENCE_IDS
            literal = re.escape(app_name)
            patterns = (
                rf"(?:打开|进入|启动|切换到|切至|前往)\s*(?:应用|app)?\s*{literal}",
                rf"{literal}\s*(?:应用)?\s*(?:已打开|已启动|主界面可见|首页可见)",
                rf"(?<![a-z0-9_])(?:open|launch|enter|go\s+to|switch\s+to)\s+"
                rf"(?:the\s+)?(?:app\s+)?{literal}(?![a-z0-9_])",
            )
            if any((re.search(pattern, objective, flags=re.IGNORECASE) for pattern in patterns)):
                matches.append(app_name)
        unique = tuple(dict.fromkeys(matches))
        return unique[0] if len(unique) == 1 else ""

    @staticmethod
    def _typed_input_views(graph: DynamicTaskGraph, active: Any) -> tuple[dict[str, Any], dict[str, Any],
        dict[str, Any]]:
        """Compile active, predecessor and verification input views from one formal authority."""

        try:
            semantic_ir = compile_formal_semantic_authority(graph).semantic_ir
        except TaskSemanticIRError:
            return {}, {}, {}
        typed = next((item for item in semantic_ir.subgoals if item.subgoal_id == getattr(active, 'subgoal_id', '')),
            None)
        if typed is None:
            return {}, {}, {}
        entities = {item.entity_id: item for item in semantic_ir.entities}
        fields = tuple(item for item in semantic_ir.input_fields if typed.subgoal_id in item.source_subgoal_ids)
        constraints = {item.constraint_id: item for item in semantic_ir.constraints}
        actions = {str(constraints[ref].value) for ref in typed.constraint_refs if ref in constraints
            and constraints[ref].kind == 'required_action'}
        active_input: dict[str, Any] = {}
        if getattr(active, 'external_impact', '') in {'navigation_only', 'read_only'}:
            if not fields and 'clear_verified_text' in actions and 'input_verified_text' not in actions:
                active_input = {'text': '', 'field_id': 'input_field_clear_target', 'field_label': '',
                    'multiline': False, 'target_only': True}
            elif len(fields) == 1:
                field = fields[0]
                payload = entities.get(field.payload_ref)
                if payload is not None and payload.role == 'input_text' and isinstance(payload.value,
                    str) and payload.value:
                    active_input = {'text': payload.value, 'field_id': field.field_id,
                        'field_label': field.field_label, 'multiline': field.multiline}
        predecessor: dict[str, Any] = {}
        if active_input:
            preceding = tuple(item for item in semantic_ir.input_fields if set(item.source_subgoal_ids).intersection(
                typed.depends_on))
            if len(preceding) == 1:
                field = preceding[0]
                payload = entities.get(field.payload_ref)
                if payload is not None and payload.role == 'input_text' and payload.value:
                    predecessor = {'field_id': field.field_id, 'field_label': field.field_label,
                        'text': payload.value}
        verification: dict[str, Any] = {}
        if getattr(active, 'external_impact', '') == 'read_only' and not active_input:
            desired = {item.state_id: item for item in semantic_ir.desired_states}
            by_payload = {item.payload_ref: item for item in semantic_ir.input_fields}
            values: dict[str, str] = {}
            labels: dict[str, str] = {}
            for ref in typed.desired_state_refs:
                state = desired.get(ref)
                field = by_payload.get(getattr(state, 'subject_ref', ''))
                if state is None or state.predicate != 'input.value_equals' or not isinstance(state.value,
                    str) or field is None:
                    continue
                values[field.field_id] = state.value
                if field.field_label:
                    labels[field.field_id] = field.field_label
            if values:
                verification = {'desired_input_values': values,
                    **({'desired_input_labels': labels} if labels else {})}
        return active_input, predecessor, verification

    @classmethod
    def _active_input_transaction_text(cls, graph: DynamicTaskGraph, active: Any) -> str:
        transaction = cls._typed_input_views(graph, active)[0]
        value = transaction.get("text")
        return value if isinstance(value, str) else ""

    @classmethod
    def _subgoal_visual_context(cls, graph: DynamicTaskGraph, subgoal: Any) -> dict[str, Any]:
        """Project one typed subgoal into the bounded visual prompt shape."""

        goal_entities = {key: value for key, value in graph.goal.entities.items() if key != 'input_fields'}
        for local_marker in ('active_input_transaction_text', 'active_input_field_id', 'active_input_field_label',
            'active_input_multiline'):
            goal_entities.pop(local_marker, None)
        active_app_label = cls._active_app_entry_target_label(graph, subgoal)
        if active_app_label:
            goal_entities["target_ui_label"] = active_app_label
        active_input, predecessor, verification = cls._typed_input_views(graph, subgoal)
        active_input_text = active_input.get("text")
        target_only_input = active_input.get("target_only") is True
        if isinstance(active_input_text, str) and (bool(active_input_text) or target_only_input):
            if active_input_text:
                goal_entities["active_input_transaction_text"] = active_input_text
            goal_entities["active_input_field_id"] = active_input["field_id"]
            if target_only_input:
                goal_entities["active_input_target_only"] = True
            if active_input.get('field_label'):
                goal_entities['active_input_field_label'] = active_input['field_label']
            goal_entities['active_input_multiline'] = bool(active_input.get('multiline'))
            if predecessor:
                goal_entities.update({'active_input_predecessor_field_id': predecessor['field_id'],
                    'active_input_predecessor_field_label': predecessor['field_label'],
                    'active_input_predecessor_text': predecessor['text']})
        else:
            goal_entities.pop("input_text", None)
            goal_entities.update(verification)
        return {'subgoal_id': subgoal.subgoal_id, 'objective': subgoal.objective,
            'constraints': list(subgoal.constraints), 'completion_conditions': list(subgoal.completion_conditions),
            'execution_class': {'read_only': 'observe', 'navigation_only': 'navigate', 'external_state': 'effect',
            'unknown': 'unknown'}.get(subgoal.external_impact, 'unknown'), 'goal_entities': goal_entities}

    def goal_draft(self, graph: DynamicTaskGraph) -> GenericIntentDraft:
        graph.validate()
        target_surface = str(graph.goal.entities.get('target_surface') or '').strip()
        if graph.goal.target_apps:
            primary_app_id = graph.goal.target_apps[0].app_id
            primary_app_name = graph.goal.target_apps[0].app_name
        elif target_surface in self._SURFACE_OBSERVATION_IDENTITIES:
            primary_app_id, primary_app_name = self._SURFACE_OBSERVATION_IDENTITIES[target_surface]
        else:
            raise UniversalAgentOrchestratorError('任务图没有目标 App 或正式目标 surface，不能建立通用观察上下文。')
        active = graph.active_subgoal()
        constraints = list(graph.constraints)
        if active is not None:
            constraints.extend(active.constraints)
        entities = dict(graph.goal.entities)
        # Preserve user visual descriptors only for read-only observation; Qwen authority stays in the typed graph.
        if graph.raw_user_goal.strip():
            entities["original_goal_visual_context"] = graph.raw_user_goal.strip()
        if active is not None:
            # Visual requests see only the active subgoal; successors require activation and a new observation.
            entities['active_subgoal_visual_context'] = self._subgoal_visual_context(graph, active)
        entities['target_apps'] = [{'app_id': item.app_id,
            'app_name': item.app_name} for item in graph.goal.target_apps]
        success_criteria = {item.condition_id: {'description': item.description,
            'evidence_required': list(item.evidence_required),
            'satisfied': item.satisfied} for item in graph.completion_conditions}
        account_effects = tuple(dict.fromkeys((item.effect_kind for item in graph.risk_actions if item.effect_kind)))
        draft = GenericIntentDraft(understood=True, app_id=primary_app_id, app_name=primary_app_name,
            objective=graph.goal.objective, entities=entities, constraints=tuple(dict.fromkeys(constraints)),
            success_criteria=success_criteria, account_effects=account_effects, needs_confirmation=True)
        draft.validate()
        return draft

    @staticmethod
    def _text_items(value: Any) -> tuple[str, ...]:
        if not isinstance(value, (list, tuple)):
            return ()
        return tuple((text for item in value if (text := str(item or '').strip())))

    def observed_state(self, *, graph: DynamicTaskGraph, trusted_observation: Any, action_outcome: str,
        verification: dict[str, Any], verified_action_transition: VerifiedActionTransition | None=None,
        controller_transition_evidence_refs: tuple[ControllerTransitionEvidenceRef, ...]=()) -> ObservedState:
        graph.validate()
        observation_device = str(getattr(trusted_observation, 'device_id', '')).strip()
        reject_if(observation_device != graph.device_id, UniversalAgentOrchestratorError('任务图与可信观察 device_id 不一致。'))
        scene = getattr(trusted_observation, "scene", None)
        reject_if(scene is None, UniversalAgentOrchestratorError("可信观察缺少 UIScene。"))
        scene.validate()
        fingerprint = str(getattr(trusted_observation, 'fingerprint', '')).strip()
        reject_if(not fingerprint or scene.fingerprint != fingerprint, UniversalAgentOrchestratorError('可信观察与 UIScene fingerprint 不一致。'))
        reject_if(not isinstance(verification, dict), UniversalAgentOrchestratorError("控制器验证结果必须是对象。"))

        evidence: list[str] = []
        grounded_visual_facts: list[str] = []

        def add(items: Any) -> None:
            for item in self._text_items(items):
                if item not in evidence:
                    evidence.append(item)

        def add_grounded(item: str) -> None:
            value = str(item or "").strip()
            if value and value not in grounded_visual_facts:
                grounded_visual_facts.append(value)

        if scene.summary.strip():
            add((scene.summary.strip(),))
        add(scene.overlays)
        add_grounded(json.dumps({'app_id': scene.app_id, 'screen_id': scene.screen_id, 'overlays': list(scene.overlays),
            'stable': scene.stable}, ensure_ascii=False, sort_keys=True, separators=(',', ':')))
        for element in scene.elements:
            add(element.evidence)
            visible = ' / '.join((item for item in (element.label.strip(), element.meaning.strip()) if item))
            if visible:
                add((visible,))
            add_grounded(json.dumps({'element_id': element.element_id, 'role': element.role, 'label': element.label,
                'meaning': element.meaning, 'states': dict(element.states)}, ensure_ascii=False, sort_keys=True,
                separators=(',', ':')))
        add(verification.get("visible_evidence"))
        if action_outcome == 'matched':
            add(verification.get("completion_evidence"))
        reject_if(not evidence, UniversalAgentOrchestratorError('当前观察没有可交给 DeepSeek 的可见证据。'))

        scene_id = str(getattr(trusted_observation, 'observation_id', '')).strip() or f'{scene.screen_id}:{fingerprint[
            :16]}'
        visual_claim_evidence_refs: list[VisualClaimEvidenceRef] = []
        typed_fact_sources = [(fact, 'scene.visible_literal', '') for fact in evidence] + [(fact, 'grounded.snapshot',
            'grounded') for fact in grounded_visual_facts]
        for (fact, default_predicate, source_kind) in typed_fact_sources:
            try:
                payload = json.loads(fact)
            except json.JSONDecodeError:
                payload = {}
            element_id = str(payload.get("element_id") or "").strip()
            subject_ref = f'element:{element_id}' if element_id else f'scene:{scene_id}'
            predicate = (
                "element.snapshot"
                if element_id
                else "scene.snapshot"
                if source_kind == "grounded"
                else default_predicate
            )
            claim_id = hashlib.sha256(json.dumps({'scene_id': scene_id, 'subject_ref': subject_ref,
                'predicate': predicate, 'fact': fact}, ensure_ascii=False, sort_keys=True, separators=(',',
                ':')).encode('utf-8')).hexdigest()
            visual_claim_evidence_refs.append(VisualClaimEvidenceRef(ref_id=f'visual_claim:{scene_id}:{claim_id}',
                claim_id=claim_id, scene_id=scene_id, subject_ref=subject_ref, predicate=predicate, fact=fact))
        observed = ObservedState(scene_id=scene_id, summary=scene.summary.strip() or '当前可信页面观察',
            visible_evidence=tuple(evidence), grounded_visual_facts=tuple(grounded_visual_facts),
            last_action_outcome=str(action_outcome or 'not_applicable'),
            blocked_reasons=self._text_items(verification.get('blocked_reasons')),
            verified_action_transition=verified_action_transition,
            controller_transition_evidence_refs=controller_transition_evidence_refs,
            visual_claim_evidence_refs=tuple(visual_claim_evidence_refs))
        observed.validate()
        return observed


@dataclass(frozen=True)
class TaskGraphTransitionReport(DataclassWire):
    """Non-action progress/completion report from the validated task graph."""

    status: str
    reason: str
    completion_evidence: tuple[str, ...]
    action: None = field(default=None, init=False)
    authority: str = field(default="deepseek_task_graph", init=False)

    def __post_init__(self) -> None:
        reject_if(self.status not in {'progressed', 'completed'}, UniversalAgentOrchestratorError("任务图报告状态无效。"))
        reject_if(not self.reason.strip(), UniversalAgentOrchestratorError("任务图报告缺少原因。"))
        reject_if(not self.completion_evidence, UniversalAgentOrchestratorError("任务图报告缺少可见证据。"))

class UniversalAgentOrchestrator:
    """Coordinate the generic one-action visual loop without App workflows."""

    def __init__(self, *, deepseek_planner: Any, qwen_observer: Any, adapter_factory: Callable[[str],
        GenericSingleActionAdapterPort], evidence_store_factory: AgentEvidenceStoreFactory,
        trusted_observation_factory: Callable[..., Any], bridge: ObservationBridge | None=None,
        device_registry: DeviceTaskRegistryPort, deepseek_failure_diagnostic_writer: Callable[..., tuple[str,
        ...]] | None=None) -> None:
        self.deepseek_planner = deepseek_planner
        self.qwen_observer = qwen_observer
        self.adapter_factory = adapter_factory
        self.trusted_observation_factory = trusted_observation_factory
        self.evidence_store_factory = evidence_store_factory
        self.bridge = bridge or ObservationBridge()
        self.device_registry = device_registry
        self.deepseek_failure_diagnostic_writer = deepseek_failure_diagnostic_writer or (lambda *_args,
            **_kwargs: ())

    def _vision_usage_scope(self, ledger: VisionSessionUsageLedger | None):
        provider = getattr(self.qwen_observer, "provider", None)
        scope_factory = getattr(provider, "session_usage_scope", None)
        return scope_factory(ledger) if callable(scope_factory) else nullcontext()

    def _release_if_terminal(self, session: UniversalAgentSessionState) -> None:
        if session.status in self.device_registry.TERMINAL_STATUSES:
            self.device_registry.release(session.device_id, session.session_id)

    @staticmethod
    def _set_status(session: UniversalAgentSessionState, status: str, reason: str='') -> None:
        session.status = status
        session.failed_reason = reason

    def _finish_session(self, session: UniversalAgentSessionState, status: str,
        reason: str='') -> UniversalAgentSessionState:
        self._set_status(session, status, reason)
        self._write_terminal_snapshot(session)
        return session

    @staticmethod
    def _invalidate_authorities(session: UniversalAgentSessionState, reason: str, *, clear: bool=False) -> None:
        for authority in (session.confirmation_authority, session.effect_confirmation_authority):
            if authority is not None:
                authority.consumed = True
                authority.invalid_reason = reason
        session.confirmed_effect_ids = ()
        if clear:
            session.confirmation_authority = None
            session.effect_confirmation_authority = None

    @staticmethod
    def _clear_action_decision(session: UniversalAgentSessionState, *, effects: bool=False) -> None:
        session.qwen_decision = None
        session.controller_decision = None
        session.confirmation_authority = None
        if effects:
            session.effect_confirmation_authority = None
            session.confirmed_effect_ids = ()

    @classmethod
    def _require_reobservation(cls, session: UniversalAgentSessionState) -> None:
        cls._set_status(session, "needs_reobservation")
        cls._clear_action_decision(session)

    @staticmethod
    def _blocked_decision(reason: str) -> SimpleNamespace:
        return SimpleNamespace(proposal=GenericStepProposal(status="blocked", reason=reason))

    @staticmethod
    def _transition_decision(status: str, reason: str, evidence: tuple[str, ...]) -> SimpleNamespace:
        return SimpleNamespace(proposal=TaskGraphTransitionReport(status=status, reason=reason,
            completion_evidence=evidence))

    @staticmethod
    def _available_action_kinds(session: UniversalAgentSessionState) -> frozenset[str]:
        provider = getattr(session.adapter, "supported_action_kinds", None)
        if not callable(provider):
            return CANONICAL_ACTION_KINDS
        actions = frozenset(str(item or "").strip() for item in provider())
        reject_if(not actions or '' in actions, UniversalAgentOrchestratorError('设备动作能力为空或包含无效动作。'))
        unexpected = actions - CANONICAL_ACTION_KINDS
        reject_if(unexpected, UniversalAgentOrchestratorError('设备报告了协议外动作：' + ', '.join(sorted(unexpected))))
        return actions

    @staticmethod
    def _lineage_matches_observed_foreground(lineage: VerifiedAppSurfaceLineage, foreground_app_id: str) -> bool:
        return lineage.matches_foreground(foreground_app_id)

    @classmethod
    def _bind_verified_lineage_to_qwen_context(cls, session: UniversalAgentSessionState, context: QwenTaskContext,
        trusted_observation: Any) -> QwenTaskContext:
        """Rebind a typed App surface to its same-scope, receipt-proven runtime package."""

        lineage = session.verified_app_surface_lineage
        graph = session.task_graph
        semantic_ir = context.semantic_ir
        scene = getattr(trusted_observation, "scene", None)
        if lineage is None or graph is None or semantic_ir is None or (scene is None):
            return context
        if (lineage.session_id != session.session_id or lineage.task_id != graph.task_id
            or lineage.task_id != context.task_id or (lineage.device_id != session.device_id)
            or (lineage.device_id != graph.device_id) or (lineage.device_id != context.device_id)
            or (lineage.physical_actions != session.physical_actions)
            or (not cls._lineage_matches_observed_foreground(lineage, str(getattr(scene, 'foreground_app_id', ''))))):
            return context

        by_id = {item.subgoal_id: item for item in graph.subgoals}
        source = by_id.get(lineage.source_subgoal_id)
        current = graph.active_subgoal()
        if source is None or source.status != 'completed' or current is None:
            return context
        pending = list(current.depends_on)
        visited: set[str] = set()
        lineage_is_ancestor = False
        while pending:
            dependency_id = pending.pop()
            if dependency_id == lineage.source_subgoal_id:
                lineage_is_ancestor = True
                break
            if dependency_id in visited:
                continue
            visited.add(dependency_id)
            dependency = by_id.get(dependency_id)
            if dependency is not None:
                pending.extend(dependency.depends_on)
        if not lineage_is_ancestor:
            return context

        matching_surfaces = tuple((surface for surface in semantic_ir.surfaces if surface.surface_id ==
            lineage.surface_id and surface.kind == 'app' and (surface.app_id.casefold() == lineage.app_id.casefold())
            and (surface.app_name.casefold() == lineage.app_name.casefold())))
        if len(matching_surfaces) != 1:
            return context
        target_surface = matching_surfaces[0]
        # Preserve a receipt-proven runtime package unless the typed surface already has that identity.
        if (str(getattr(scene, 'foreground_app_id', '') or '').strip().casefold() == str(target_surface.app_id
            or '').strip().casefold()):
            return context
        rebound_surface = replace(target_surface, app_id=lineage.functional_foreground_app_id)
        rebound_ir = replace(semantic_ir, surfaces=tuple((rebound_surface if item.surface_id ==
            target_surface.surface_id else item for item in semantic_ir.surfaces)))
        rebound_context = replace(context, semantic_ir=rebound_ir)
        try:
            rebound_context.validate()
        except Exception:
            return context
        return rebound_context

    def _decide_next_action(self, session: UniversalAgentSessionState, *, frames: list[Any], task_context: Any,
        trusted_observation: Any) -> Any:
        context = task_context if isinstance(task_context,
            QwenTaskContext) else QwenTaskContext.from_dict(dict(task_context))
        if context.semantic_ir is None:
            graph = session.task_graph
            semantic_authority = None
            if graph is not None:
                try:
                    semantic_authority = compile_formal_semantic_authority(graph)
                except TaskSemanticIRError as exc:
                    raise UniversalAgentOrchestratorError(f'正式 TaskSemanticIR authority 拒绝：{exc}') from exc
                semantic_ir = semantic_authority.semantic_ir
            else:
                semantic_authority = getattr(self.deepseek_planner, 'last_semantic_authority', None)
                semantic_ir = getattr(semantic_authority, "semantic_ir", None)
                reject_if(semantic_ir is None, UniversalAgentOrchestratorError('正式视觉决策缺少当前任务图。'))
            context = replace(context, semantic_ir=semantic_ir)
            context.validate()
            if semantic_authority is not None and hasattr(semantic_authority, 'effect_previews'):
                session.effect_previews = tuple(({**preview, 'preview_digest': effect_preview_digest(preview)}
                    for preview in semantic_authority.effect_previews))
        context = self._bind_verified_lineage_to_qwen_context(session, context, trusted_observation)
        session.semantic_task_context = context
        available_actions = self._available_action_kinds(session)
        semantic_ir = context.semantic_ir
        assert semantic_ir is not None
        active_id = str(context.current_subgoal.get("subgoal_id") or "")
        typed_subgoal = next((item for item in semantic_ir.subgoals if item.subgoal_id == active_id), None)
        launch_parameters: dict[str, str] | None = None
        target_surface = next((surface for surface in semantic_ir.surfaces if typed_subgoal is not None
            and surface.surface_id == typed_subgoal.surface_ref and surface.kind == 'app'), None)
        launch_resolver = getattr(session.adapter, 'resolve_app_launch_target', None)
        launch_target = (launch_resolver(target_surface.app_id, target_surface.app_name)
            if target_surface is not None and callable(launch_resolver) else None)
        if launch_target is None:
            available_actions = available_actions - {'launch_app'}
        else:
            reject_if(not str(getattr(launch_target, 'launch_ref', '') or '').strip()
                or not str(getattr(launch_target, 'expected_app_id', '') or '').strip(),
                UniversalAgentOrchestratorError('App 直启能力缺少受信任引用或前台包。'))
            launch_parameters = {'launch_ref': launch_target.launch_ref,
                'expected_app_id': launch_target.expected_app_id}
        constraints = {item.constraint_id: item for item in semantic_ir.constraints}
        required_actions = tuple(dict.fromkeys((str(constraints[ref].value) for ref
            in (typed_subgoal.constraint_refs if typed_subgoal
            is not None else ()) if constraints[ref].kind == 'required_action')))
        unsupported = tuple((action for action in required_actions if action not in available_actions))
        if unsupported:
            provider = getattr(session.adapter, "capability_snapshot", None)
            capability = provider() if callable(provider) else build_device_capability_snapshot(
                device_id=context.device_id, supported_actions=available_actions)
            gap = capability.gap(unsupported[0])
            assert gap is not None
            session.capability_gap = gap.to_dict()
            reason = "当前设备能力不支持 typed required_action：" + unsupported[0]
            proposal = GenericStepProposal(status="blocked", reason=reason)
            decision = SimpleNamespace(task_id=context.task_id, device_id=context.device_id, revision=context.revision,
                observation_id=trusted_observation.observation_id, fingerprint=trusted_observation.fingerprint,
                trusted_observation=trusted_observation, proposal=proposal, target_region=None, expected_result={},
                confidence=1.0, reason=reason, block_stage='required_action_capability')
            decision.to_dict = lambda: {'task_id': decision.task_id, 'device_id': decision.device_id,
                'revision': decision.revision, 'observation_id': decision.observation_id,
                'fingerprint': decision.fingerprint, 'status': 'blocked', 'next_action': None, 'reason': reason,
                'capability_gap': dict(session.capability_gap)}
            return decision
        decision_args = {'frames': frames, 'task_context': context, 'trusted_observation': trusted_observation,
            'decision_number': session.step_number, 'available_action_kinds': available_actions}
        if launch_parameters:
            decision_args['launch_target'] = launch_parameters
        return self.qwen_observer.decide(**decision_args)

    def _build_and_record_current_observation(self, session: UniversalAgentSessionState, *, scene: Any,
        frames: list[Any], observation_id: str | None=None) -> Any:
        """Adopt one current trusted observation through a single evidence path."""

        factory_args: dict[str, Any] = {'frames': frames, 'device_id': session.device_id, 'scene': scene}
        if observation_id is not None:
            factory_args["observation_id"] = observation_id
        observation = self.trusted_observation_factory(**factory_args)
        session.trusted_observation = observation
        session.trusted_frames = tuple(frames)
        self._remember(session, session.evidence_store.write_trusted_observation(session.step_number, observation))
        return observation

    def _stage_current_observation_decision(self, session: UniversalAgentSessionState, *, graph: DynamicTaskGraph,
        frames: list[Any], task_context: Any, trusted_observation: Any, unsupported_status_reason: Callable[[str],
        str] | None=None, before_selection: Callable[[Any], str | None] | None=None,
        stage_capability_block: bool=True) -> Any:
        """Stage one canonical action and its one-shot confirmation from the current observation."""

        decision = self._decide_next_action(session, frames=frames, task_context=task_context,
            trusted_observation=trusted_observation)
        self._validate_decision_binding(graph, trusted_observation, decision)
        if not stage_capability_block and str(getattr(decision, 'block_stage', '')) == 'required_action_capability':
            session.status = "blocked"
            session.failed_reason = decision.proposal.reason
            session.qwen_decision = None
            session.controller_decision = None
            session.confirmation_authority = None
            return decision

        session.qwen_decision = decision
        self._remember(session, session.evidence_store.write_qwen_decision(session.step_number, decision))
        proposal = decision.proposal
        if proposal.status != 'action':
            if proposal.status == 'blocked':
                reason = proposal.reason
            elif unsupported_status_reason is None:
                raise UniversalAgentOrchestratorError(f'不支持的 Qwen 状态：{proposal.status}')
            else:
                reason = unsupported_status_reason(proposal.status)
            session.status = "blocked"
            session.failed_reason = reason
            session.controller_decision = CanonicalSelectionReceipt(allowed=False, reason=reason)
            session.confirmation_authority = None
            return decision

        guard_reason = before_selection(decision) if before_selection else None
        if guard_reason:
            session.status = "blocked"
            session.failed_reason = guard_reason
            session.controller_decision = CanonicalSelectionReceipt(allowed=False, reason=guard_reason)
            session.confirmation_authority = None
            return decision

        selection_receipt = self._selection_receipt(session, decision)
        session.controller_decision = selection_receipt
        self._remember(session, session.evidence_store.write_controller_decision(session.step_number,
            self._selection_receipt_payload(selection_receipt)))
        if selection_receipt.allowed:
            session.status = "awaiting_confirmation"
            session.failed_reason = ""
            self._bind_confirmation(session)
        else:
            session.status = "blocked"
            session.failed_reason = selection_receipt.reason
            session.confirmation_authority = None
        return decision

    @staticmethod
    def _validate_visible_replan_shape(previous: DynamicTaskGraph, revised: DynamicTaskGraph,
        observed: ObservedState) -> None:
        reject_if(revised.revision != previous.revision + 1, UniversalAgentOrchestratorError('可见状态证据推进必须且只能产生一个新 revision。'))
        reject_if(tuple((item.subgoal_id for item in previous.subgoals)) != tuple((item.subgoal_id for item in revised.subgoals)), UniversalAgentOrchestratorError('可见状态证据推进不得增加、删除或重排子目标。'))
        reject_if(revised.goal != previous.goal or revised.constraints != previous.constraints or revised.risk_actions != previous.risk_actions, UniversalAgentOrchestratorError('可见状态证据推进不得修改目标、约束或效果定义。'))
        immutable = ('objective', 'depends_on', 'constraints', 'completion_conditions', 'risk_action_ids',
            'external_impact')
        for (old, new) in zip(previous.subgoals, revised.subgoals):
            reject_if(any((getattr(old, name) != getattr(new, name) for name in immutable)), UniversalAgentOrchestratorError('可见状态证据推进只能改变子目标状态和完成证据。'))
        _validate_visible_completion_condition_progress(previous, revised, observed)

    def _validated_visible_step(self, *, previous: DynamicTaskGraph, revised: DynamicTaskGraph, current: Any,
        trusted_observation: Any) -> DynamicTaskGraph:
        old = {item.subgoal_id: item for item in previous.subgoals}
        new = {item.subgoal_id: item for item in revised.subgoals}
        newly_completed = tuple((item.subgoal_id for item in previous.subgoals if item.status != 'completed'
            and new[item.subgoal_id].status == 'completed'))
        reject_if(not newly_completed or newly_completed[0] != current.subgoal_id,
            UniversalAgentOrchestratorError('可见状态证据只能完成当前活动子目标。'))
        if len(newly_completed) > 1:
            narrowed = self._narrow_unproven_visible_successor(previous=previous, revised=revised,
                current_subgoal_id=current.subgoal_id, unsupported_subgoal_id=newly_completed[1])
            reject_if(narrowed is None,
                UniversalAgentOrchestratorError('当前截图不能完成后继子目标；活动目标变化后必须重新观察。'))
            revised = narrowed
            new = {item.subgoal_id: item for item in revised.subgoals}
            newly_completed = tuple((subgoal_id for subgoal_id,
                item in old.items() if item.status != 'completed' and new[subgoal_id].status == 'completed'))
        reject_if(newly_completed != (current.subgoal_id,),
            UniversalAgentOrchestratorError('当前截图必须且只能推进一个活动子目标。'))
        for (subgoal_id, source) in old.items():
            status = new[subgoal_id].status
            reject_if(source.status == 'completed' and status != 'completed', UniversalAgentOrchestratorError('可见状态证据推进不得回退已完成子目标。'))
            reject_if(source.status == 'pending' and subgoal_id not in newly_completed and (status not in {'pending', 'active'}), UniversalAgentOrchestratorError('可见状态证据推进不得越过后续子目标。'))
        newly_active = tuple((subgoal_id for subgoal_id, source in old.items() if source.status == 'pending'
            and new[subgoal_id].status == 'active'))
        reject_if(len(newly_active) > 1 or (revised.status != 'completed' and (len(newly_active) != 1 or revised.active_subgoal_id != newly_active[0])), UniversalAgentOrchestratorError('可见状态证据推进后必须精确激活一个后续子目标。'))
        return revised

    def _try_advance_visible_subgoal(self, session: UniversalAgentSessionState, *, graph: DynamicTaskGraph,
        trusted_observation: Any) -> DynamicTaskGraph | None:
        """Advance one current-frame read-only checkpoint with one authority."""

        current = graph.active_subgoal()
        if (current is None or current.external_impact not in {'read_only',
            'navigation_only'} or not (VisibleGoalEvidence.presence_only(current)
            or VisibleGoalEvidence.visible_text_read(current))):
            return None
        visible_evidence = VisibleGoalEvidence.evidence(graph, current, trusted_observation,
            app_surface_lineage=session.verified_app_surface_lineage)
        if not visible_evidence:
            return None
        observed = self.bridge.observed_state(graph=graph, trusted_observation=trusted_observation,
            action_outcome='not_applicable', verification={'visible_evidence': list(visible_evidence)})
        text_read = VisibleGoalEvidence.visible_text_read(current)
        if text_read and session.verified_app_surface_lineage is not None:
            lineage = session.verified_app_surface_lineage
            fact = json.dumps({'source': 'verified_app_surface_lineage', 'app_id': lineage.app_id,
                'app_name': lineage.app_name, 'surface_id': lineage.surface_id,
                'functional_foreground_app_id': lineage.functional_foreground_app_id}, ensure_ascii=False,
                sort_keys=True, separators=(',', ':'))
            observed = replace(observed, grounded_visual_facts=(*observed.grounded_visual_facts, fact))
        revised = self.deepseek_planner.replan(
            graph,
            observed,
            trigger="subgoal_completed",
            reason=(
                "当前可信画面已逐项证明当前正向可见状态。只完成从当前节点开始、"
                "依赖连续且 completion_evidence 逐字引用 visible_evidence 的安全前缀；"
                "不得推断元素值、外部效果、缺失状态或执行动作。"
            ),
        )
        self._validate_graph_identity(revised, device_id=session.device_id, previous=graph,
            trusted_observation=trusted_observation, session_id=session.session_id,
            verified_app_surface_lineage=session.verified_app_surface_lineage,
            physical_actions=session.physical_actions)
        if text_read:
            visible_fact = next((item for item in visible_evidence if item.startswith('当前可信画面读取结果：')), '')
            completed = next((item for item in revised.subgoals if item.subgoal_id == current.subgoal_id), None)
            reject_if(completed is None or completed.status != 'completed'
                or visible_fact not in completed.completion_evidence,
                UniversalAgentOrchestratorError('DeepSeek 未使用唯一可信文字结果完成当前 read_only 子目标。'))
            return revised
        self._validate_visible_replan_shape(graph, revised, observed)
        return self._validated_visible_step(previous=graph, revised=revised, current=current,
            trusted_observation=trusted_observation)

    @staticmethod
    def _narrow_unproven_visible_successor(*, previous: DynamicTaskGraph, revised: DynamicTaskGraph,
        current_subgoal_id: str, unsupported_subgoal_id: str) -> DynamicTaskGraph | None:
        """Revoke unproven completion without adding evidence or widening action authority."""

        if not current_subgoal_id or not unsupported_subgoal_id:
            return None
        old_by_id = {item.subgoal_id: item for item in previous.subgoals}
        new_by_id = {item.subgoal_id: item for item in revised.subgoals}
        unsupported = old_by_id.get(unsupported_subgoal_id)
        accepted = {current_subgoal_id}
        previously_completed = {item.subgoal_id for item in previous.subgoals if item.status == 'completed'}
        if (unsupported is None or unsupported_subgoal_id not in new_by_id or unsupported.status != 'pending'
            or (new_by_id[unsupported_subgoal_id].status != 'completed') or (unsupported.external_impact not
            in {'read_only', 'navigation_only'}) or (current_subgoal_id not in unsupported.depends_on)
            or (not all((dependency in previously_completed or dependency in accepted for dependency
            in unsupported.depends_on)))):
            return None

        normalized = []
        for old_item in previous.subgoals:
            subgoal_id = old_item.subgoal_id
            if subgoal_id in accepted or old_item.status == 'completed':
                normalized.append(new_by_id[subgoal_id])
            elif subgoal_id == unsupported_subgoal_id:
                normalized.append(replace(old_item, status='active', completion_evidence=()))
            else:
                normalized.append(replace(old_item, completion_evidence=()))
        narrowed = replace(revised, status='ready', subgoals=tuple(normalized),
            active_subgoal_id=unsupported_subgoal_id, clarification_questions=previous.clarification_questions)
        narrowed.validate()
        return narrowed

    def _store_revised_graph(self, session: UniversalAgentSessionState, revised: DynamicTaskGraph) -> None:
        session.task_graph = revised
        session.goal_draft = self.bridge.goal_draft(revised)
        session.confirmation_authority = None
        session.effect_confirmation_authority = None
        session.confirmed_effect_ids = ()
        self._remember(session, session.evidence_store.write_task_graph(revised),
            session.evidence_store.write_effect_policy_snapshot(revised))

    def _replan_current_observation(self, session: UniversalAgentSessionState, graph: DynamicTaskGraph,
        observation: Any, *, trigger: str, reason: str, observed: Any | None=None,
        store: bool=False) -> tuple[DynamicTaskGraph, Any]:
        observed = observed or self.bridge.observed_state(graph=graph, trusted_observation=observation,
            action_outcome='not_applicable', verification={'visible_evidence': [observation.scene.summary],
            'blocked_reasons': []})
        revised = self.deepseek_planner.replan(graph, observed, trigger=trigger, reason=reason)
        self._validate_graph_identity(revised, device_id=session.device_id, previous=graph,
            trusted_observation=observation)
        if store:
            self._store_revised_graph(session, revised)
        return revised, observed

    def _finish_visible_advancement(self, session: UniversalAgentSessionState, revised: DynamicTaskGraph, *,
        evidence: str, reason: str, missing_reason: str) -> SimpleNamespace:
        if revised.status == 'completed':
            self._set_status(session, "succeeded")
        else:
            current = revised.active_subgoal()
            if current is None:
                self._set_status(session, "blocked", missing_reason)
            elif _requires_effect_confirmation(revised, current):
                self._set_status(session, "awaiting_effect_confirmation")
                self._bind_effect_confirmation(session)
            else:
                self._require_reobservation(session)
        self._clear_action_decision(session)
        self._write_terminal_snapshot(session)
        return self._transition_decision('completed' if revised.status == 'completed' else 'progressed', reason,
            (evidence,))

    @staticmethod
    def _selection_receipt_payload(decision: CanonicalSelectionReceipt) -> dict[str, Any]:
        return decision.to_dict()

    @classmethod
    def _selection_receipt(cls, session: UniversalAgentSessionState, decision: Any) -> CanonicalSelectionReceipt:
        """Record the already-validated canonical choice without judging it again."""

        proposal = getattr(decision, "proposal", None)
        action = getattr(proposal, "action", None)
        if proposal is None or getattr(proposal, 'status', '') != 'action':
            return CanonicalSelectionReceipt(allowed=False, reason='当前单步决策没有唯一 canonical 动作。')
        action_kind = str(getattr(action, "action", "") or "").strip()
        available = cls._available_action_kinds(session)
        if action_kind not in available:
            return CanonicalSelectionReceipt(allowed=False, reason=f'当前设备没有 canonical 动作能力：{action_kind or 'missing'}。')
        return CanonicalSelectionReceipt(
            allowed=True,
            reason=(
                "单步 Qwen 决策已绑定当前 observation、canonical candidate "
                "和 typed transition；后续只消费同一 scope。"
            ),
            canonical_class=action_kind,
        )

    @classmethod
    def _validate_newly_completed_named_app_surfaces(cls, *, previous: DynamicTaskGraph, revised: DynamicTaskGraph,
        trusted_observation: Any, session_id: str='', verified_transition: VerifiedActionTransition | None=None,
        controller_transition_evidence_refs: tuple[ControllerTransitionEvidenceRef, ...]=(),
        before_observation: Any | None=None, previous_decision: Any | None=None, execution_result: Any | None=None,
        verified_app_surface_lineage: VerifiedAppSurfaceLineage | None=None, physical_actions: int=0) -> None:
        try:
            AppSurfaceLineageAuthority.validate_newly_completed(previous=previous, revised=revised,
                trusted_observation=trusted_observation, session_id=session_id, verified_transition=verified_transition,
                controller_transition_evidence_refs=controller_transition_evidence_refs,
                before_observation=before_observation, previous_decision=previous_decision,
                execution_result=execution_result, verified_app_surface_lineage=verified_app_surface_lineage,
                physical_actions=physical_actions)
        except AppSurfaceLineageError as exc:
            raise UniversalAgentOrchestratorError(str(exc)) from exc


    @classmethod
    def _validate_graph_identity(cls, graph: DynamicTaskGraph, *, device_id: str,
        previous: DynamicTaskGraph | None=None, trusted_observation: Any | None=None, session_id: str='',
        verified_transition: VerifiedActionTransition | None=None,
        controller_transition_evidence_refs: tuple[ControllerTransitionEvidenceRef, ...]=(),
        before_observation: Any | None=None, previous_decision: Any | None=None, execution_result: Any | None=None,
        verified_app_surface_lineage: VerifiedAppSurfaceLineage | None=None, physical_actions: int=0) -> None:
        graph.validate()
        reject_if(graph.device_id != device_id, UniversalAgentOrchestratorError('DeepSeek 任务图 device_id 与会话设备不一致。'))
        if previous is not None:
            reject_if(graph.task_id != previous.task_id or graph.device_id != previous.device_id, UniversalAgentOrchestratorError('DeepSeek 重规划改变了 task_id 或 device_id。'))
            reject_if(graph.revision != previous.revision + 1, UniversalAgentOrchestratorError('DeepSeek 重规划 revision 必须严格等于上一 revision + 1。'))
            if trusted_observation is not None:
                cls._validate_newly_completed_named_app_surfaces(previous=previous, revised=graph,
                    trusted_observation=trusted_observation, session_id=session_id,
                    verified_transition=verified_transition,
                    controller_transition_evidence_refs=controller_transition_evidence_refs,
                    before_observation=before_observation, previous_decision=previous_decision,
                    execution_result=execution_result, verified_app_surface_lineage=verified_app_surface_lineage,
                    physical_actions=physical_actions)

    @staticmethod
    def _validate_decision_binding(graph: DynamicTaskGraph, observation: Any, decision: Any) -> None:
        expected = {'task_id': graph.task_id, 'device_id': graph.device_id, 'revision': graph.revision,
            'observation_id': str(getattr(observation, 'observation_id', '')), 'fingerprint': str(getattr(observation,
            'fingerprint', ''))}
        for (field_name, expected_value) in expected.items():
            actual = getattr(decision, field_name, None)
            reject_if(actual != expected_value, UniversalAgentOrchestratorError(f'Qwen 决策 {field_name} 与当前权威状态不一致。'))
        bound = getattr(decision, "trusted_observation", None)
        reject_if(bound is None or (str(getattr(bound, 'observation_id', '')) != expected['observation_id'] or str(getattr(bound, 'fingerprint', '')) != expected['fingerprint']), UniversalAgentOrchestratorError('Qwen 决策没有绑定当前可信观察。'))
        proposal = getattr(decision, "proposal", None)
        reject_if(proposal is None, UniversalAgentOrchestratorError("Qwen 决策缺少 proposal。"))
        try:
            proposal.validate(observation.scene)
        except (CanonicalActionProtocolError, AttributeError, TypeError) as exc:
            raise UniversalAgentOrchestratorError(f'Qwen 决策 proposal 不符合 canonical 合同：{exc}') from exc

    @staticmethod
    def _remember(session: UniversalAgentSessionState, *paths: Any) -> None:
        for path in paths:
            if path is None:
                continue
            values = path if isinstance(path, (list, tuple)) else (path,)
            for item in values:
                text = str(item or "").strip()
                if text and text not in session.evidence_paths:
                    session.evidence_paths.append(text)

    def _record_deepseek_failure(self, session: UniversalAgentSessionState, error: Exception, *, stage: str,
        previous_graph: DynamicTaskGraph | None=None) -> None:
        if not isinstance(error, TaskGraphError):
            return
        try:
            paths = self.deepseek_failure_diagnostic_writer(self.deepseek_planner, evidence_dir=session.run_dir,
                prefix=f'deepseek_{stage}_step_{session.step_number}', failed_stage=stage, error=error,
                previous_graph=previous_graph)
        except Exception:
            return
        self._remember(session, paths)

    def _write_terminal_snapshot(self, session: UniversalAgentSessionState) -> None:
        if session.vision_usage is not None:
            self._remember(session, session.evidence_store.write_json('qwen_usage.json',
                session.vision_usage.to_dict()))
        session_path = session.evidence_store.write_session(session)
        self._remember(session, session_path)
        report_path = session.evidence_store.write_report({'mode': 'universal_agent_safe_live_loop',
            'policy_version': CANONICAL_SELECTION_RECEIPT_VERSION, 'session': session.snapshot()})
        self._remember(session, report_path)

    def _ensure_terminal_snapshot(self, session: UniversalAgentSessionState) -> None:
        """Write a terminal report only when the persisted one is missing/stale."""

        try:
            report = session.evidence_store.read_report()
            reject_if(report is None, KeyError("report"))
            persisted = report["session"]
            current = session.snapshot()
            compared_fields = ('status', 'failed_reason', 'confirm_stage', 'physical_actions', 'qwen_usage',
                'last_post_action_transition', 'last_confirmation_failure')
            if all((persisted.get(key) == current.get(key) for key in compared_fields)):
                return
        except (EvidenceStoreError, KeyError, TypeError):
            pass
        self._write_terminal_snapshot(session)

    def _best_effort_terminal_snapshot(self, session: UniversalAgentSessionState, *, ensure: bool=False) -> None:
        """Persist failure state without replacing the authoritative exception."""

        try:
            (self._ensure_terminal_snapshot if ensure else self._write_terminal_snapshot)(session)
        except Exception:
            pass

    def _current_confirmation_scope(self, session: UniversalAgentSessionState) -> dict[str, Any]:
        graph = session.task_graph
        observation = session.trusted_observation
        decision = session.qwen_decision
        reject_if(graph is None or observation is None or decision is None, UniversalAgentOrchestratorError('当前会话没有完整的确认权威状态。'))
        self._validate_graph_identity(graph, device_id=session.device_id)
        self._validate_decision_binding(graph, observation, decision)
        reject_if(decision.proposal.status != 'action', UniversalAgentOrchestratorError('非 action 决策没有可确认动作。'))
        current = graph.active_subgoal()
        reject_if(current is None, UniversalAgentOrchestratorError("当前任务没有活动子目标。"))
        return {'session_id': session.session_id, 'task_id': graph.task_id, 'device_id': graph.device_id,
            'revision': graph.revision, 'subgoal_id': current.subgoal_id, 'effect_ids': sorted(current.risk_action_ids),
            'observation_id': str(observation.observation_id), 'fingerprint': str(observation.fingerprint),
            'decision_node_id': str(decision.proposal.action.node_id),
            'action_digest': _action_digest(decision.proposal.action)}

    def _bind_confirmation(self, session: UniversalAgentSessionState) -> None:
        scope = self._current_confirmation_scope(session)
        session.confirmation_authority = ConfirmationAuthority(session_id=scope['session_id'], task_id=scope['task_id'],
            device_id=scope['device_id'], revision=scope['revision'], subgoal_id=scope['subgoal_id'],
            effect_ids=tuple(scope['effect_ids']), observation_id=scope['observation_id'],
            fingerprint=scope['fingerprint'], decision_node_id=scope['decision_node_id'],
            action_digest=scope['action_digest'])

    @staticmethod
    def _normalize_confirmation_scope(value: Mapping[str, Any], *, required: set[str], digest_key: str,
        label: str) -> dict[str, Any]:
        shape_label = '效果确认作用域' if label == '效果确认' else label
        reject_if(not isinstance(value, Mapping) or set(value) != required,
            UniversalAgentOrchestratorError(f'{shape_label}字段缺失或包含额外字段。'))
        effect_ids = value.get("effect_ids")
        reject_if(not isinstance(effect_ids, list), UniversalAgentOrchestratorError(f"{label} effect_ids 必须是数组。"))
        revision = value.get("revision")
        reject_if(isinstance(revision, bool) or not isinstance(revision, int),
            UniversalAgentOrchestratorError(f"{label} revision 格式无效。"))
        digest = str(value.get(digest_key) or "").strip()
        reject_if(not re.fullmatch('[0-9a-f]{64}', digest),
            UniversalAgentOrchestratorError(f'{label} {digest_key} 必须是 64 位小写 SHA-256。'))
        result = {'session_id': str(value.get('session_id') or ''), 'task_id': str(value.get('task_id') or ''),
            'device_id': str(value.get('device_id') or ''), 'revision': revision,
            'subgoal_id': str(value.get('subgoal_id') or ''), 'effect_ids': sorted((str(item) for item in effect_ids)),
            digest_key: digest}
        return result

    @classmethod
    def _normalize_confirmation(cls, value: Mapping[str, Any]) -> dict[str, Any]:
        required = {'session_id', 'task_id', 'device_id', 'revision', 'subgoal_id', 'effect_ids', 'observation_id',
            'fingerprint', 'decision_node_id', 'action_digest'}
        result = cls._normalize_confirmation_scope(value, required=required, digest_key='action_digest', label='确认作用域')
        decision_node_id = str(value.get("decision_node_id") or "").strip()
        reject_if(not decision_node_id, UniversalAgentOrchestratorError('确认作用域 decision_node_id 不能为空。'))
        result.update({'observation_id': str(value.get('observation_id') or ''),
            'fingerprint': str(value.get('fingerprint') or ''), 'decision_node_id': decision_node_id})
        return result

    def _bind_effect_confirmation(self, session: UniversalAgentSessionState) -> None:
        graph = session.task_graph
        reject_if(graph is None, UniversalAgentOrchestratorError("效果确认缺少任务图。"))
        current = graph.active_subgoal()
        effect_ids = _confirmation_effect_ids(graph, current)
        reject_if(current is None or not effect_ids, UniversalAgentOrchestratorError("当前子目标没有可确认风险。"))
        intent_digest, intent_preview = _effect_confirmation_material(graph, current)
        session.effect_confirmation_authority = EffectConfirmationAuthority(session_id=session.session_id,
            task_id=graph.task_id, device_id=graph.device_id, revision=graph.revision, subgoal_id=current.subgoal_id,
            effect_ids=effect_ids, intent_digest=intent_digest, intent_preview=intent_preview)

    @classmethod
    def _normalize_effect_confirmation(cls, value: Mapping[str, Any]) -> dict[str, Any]:
        required = {'session_id', 'task_id', 'device_id', 'revision', 'subgoal_id', 'effect_ids', 'intent_digest'}
        return cls._normalize_confirmation_scope(value, required=required, digest_key='intent_digest', label='效果确认')

    def _validate_and_consume_confirmation(self, session: UniversalAgentSessionState, confirmation: Mapping[str,
        Any]) -> None:
        authority = session.confirmation_authority
        reject_if(session.status != 'awaiting_confirmation', UniversalAgentOrchestratorError(f'会话已推进，当前状态不能确认：{session.status}。'))
        reject_if(authority is None, UniversalAgentOrchestratorError("当前会话没有可用确认作用域。"))
        reject_if(authority.consumed, UniversalAgentOrchestratorError("当前确认已使用，禁止重放。"))
        current_scope = self._current_confirmation_scope(session)
        if current_scope != authority.scope():
            authority.consumed = True
            authority.invalid_reason = "authoritative_state_changed"
            raise UniversalAgentOrchestratorError('任务、画面或视觉决策已经变化，当前确认已失效。')
        try:
            requested = self._normalize_confirmation(confirmation)
        except UniversalAgentOrchestratorError:
            authority.consumed = True
            authority.invalid_reason = "invalid_confirmation_shape"
            raise
        if requested != current_scope:
            authority.consumed = True
            authority.invalid_reason = "confirmation_scope_mismatch"
            raise UniversalAgentOrchestratorError('确认作用域与当前 task/device/revision/subgoal/risk/observation 不一致。')

    @staticmethod
    def _verified_target_app_home_reset_microstep(*, graph: DynamicTaskGraph, previous_decision: Any, result: Any,
        before_observation: Any, new_observation: Any) -> bool:
        """Keep the active App goal after a verified intermediate Home reset to Launcher."""

        current = graph.active_subgoal()
        action = getattr(getattr(previous_decision, 'proposal', None), 'action', None)
        resolved = getattr(result, "resolved_action", None)
        before_scene = getattr(result, "before_scene", None)
        after_scene = getattr(result, "after_scene", None)
        if (current is None or current.external_impact != 'navigation_only' or action is None or (str(getattr(action,
            'action', '')) != 'home') or (resolved is None) or (str(getattr(resolved, 'kind',
            '')) != 'home') or (before_scene is None) or (after_scene is None) or (str(getattr(result, 'action_outcome',
            '')) != 'matched') or (int(getattr(result, 'physical_actions', 0)) != 1) or tuple(getattr(result,
            'verification_errors', ())) or (str(getattr(before_scene, 'foreground_app_id',
            '')).casefold() == 'launcher') or (str(getattr(after_scene, 'foreground_app_id',
            '')).casefold() != 'launcher') or (str(getattr(new_observation, 'fingerprint',
            '')) != str(getattr(after_scene, 'fingerprint', ''))) or (str(getattr(before_observation, 'fingerprint',
            '')) != str(getattr(before_scene, 'fingerprint', '')))):
            return False
        try:
            semantic_ir = compile_formal_semantic_authority(graph).semantic_ir
        except TaskSemanticIRError:
            return False
        active = next((item for item in semantic_ir.subgoals if item.subgoal_id == current.subgoal_id), None)
        surfaces = {item.surface_id: item for item in semantic_ir.surfaces}
        target_surface = surfaces.get(active.surface_ref) if active is not None else None
        if target_surface is None or target_surface.kind != 'app':
            return False
        params = getattr(action, "params", None)
        transition = params.get('formal_transition') if isinstance(params, Mapping) else None
        expectations = transition.get('expectations') if isinstance(transition, Mapping) else None
        return bool(transition.get('exploratory') is False and isinstance(expectations,
            list) and (len(expectations) == 1) and (expectations[0] == {'subject_ref': 'surface_current',
            'predicate': 'surface.kind', 'operator': 'equals', 'value': 'launcher'}))

    @staticmethod
    def _resolved_input_effect(resolved: Any) -> tuple[str, dict[str, Any] | None]:
        effect = getattr(resolved, "expected_effect", None)
        element = effect.get("element_state") if isinstance(effect, Mapping) else None
        states = element.get("states") if isinstance(element, Mapping) else None
        meaning = str(element.get("meaning") or "").strip() if isinstance(element, Mapping) else ""
        return meaning, dict(states) if isinstance(states, Mapping) else None

    @classmethod
    def _verified_input_transaction_microstep(cls, *, graph: DynamicTaskGraph, previous_decision: Any, result: Any,
        before_observation: Any, new_observation: Any, allow_terminal: bool=False) -> bool:
        current = graph.active_subgoal()
        canonical = graph.goal.entities.get("input_text")
        if not isinstance(canonical, str) or not canonical:
            canonical = ObservationBridge._active_input_transaction_text(graph, current)
        resolved = getattr(result, "resolved_action", None)
        before_scene = getattr(result, "before_scene", None)
        after_scene = getattr(result, "after_scene", None)
        if (current is None or current.external_impact != 'navigation_only' or (not isinstance(canonical,
            str)) or (not canonical) or (resolved is None) or (before_scene is None) or (after_scene is None)
            or (str(getattr(result, 'action_outcome', '')) != 'matched') or (int(getattr(result, 'physical_actions',
            0)) != 1) or tuple(getattr(result, 'verification_errors', ())) or (str(getattr(resolved,
            'before_fingerprint', '')) != str(getattr(before_scene, 'fingerprint',
            ''))) or (str(getattr(new_observation, 'fingerprint', '')) != str(getattr(after_scene, 'fingerprint',
            '')))):
            return False
        proposal_action = getattr(getattr(previous_decision, "proposal", None), "action", None)
        evidence = tuple((str(item) for item in getattr(result, 'controller_transition_evidence', ()) if str(item)))
        if proposal_action is None or not evidence:
            return False
        kind = str(getattr(resolved, "kind", ""))
        if kind not in {'input_verified_text', 'tap_semantic', 'press_enter'} or str(getattr(proposal_action,
            'action', '')) != kind or (kind == 'input_verified_text' and str(getattr(resolved, 'text',
            '')) != canonical):
            return False
        expected_meaning, expected_states = cls._resolved_input_effect(resolved)
        if expected_meaning != 'application_text_input' or not isinstance(expected_states, dict):
            return False
        if kind == 'tap_semantic' and expected_states == {'focused': True}:
            return True
        expected_value = expected_states.get("value")
        if not isinstance(expected_value, str) or not canonical.startswith(expected_value):
            return False
        if CONTROLLER_INPUT_PREEDIT_PENDING in evidence:
            return True
        if expected_value == canonical:
            return bool(allow_terminal)
        if (cls._input_step_reaches_formal_successor(graph=graph, current_subgoal_id=current.subgoal_id,
            canonical=canonical, expected_value=expected_value)):
            return False
        return True

    @staticmethod
    def _input_transaction_reached_canonical(graph: DynamicTaskGraph, result: Any) -> bool:
        canonical = graph.goal.entities.get("input_text")
        resolved = getattr(result, "resolved_action", None)
        expected_effect = getattr(resolved, "expected_effect", None)
        expected_element = expected_effect.get('element_state') if isinstance(expected_effect, Mapping) else None
        expected_states = expected_element.get('states') if isinstance(expected_element, Mapping) else None
        evidence = tuple((str(item) for item in getattr(result, 'controller_transition_evidence', ()) if str(item)))
        return bool(isinstance(canonical, str) and canonical and isinstance(expected_states, Mapping)
            and expected_states.get('value') == canonical and evidence
            and CONTROLLER_INPUT_PREEDIT_PENDING not in evidence)

    @staticmethod
    def _complete_local_exact_input_graph(graph: DynamicTaskGraph, *, new_observation: Any) -> DynamicTaskGraph:
        current = graph.active_subgoal()
        reject_if(graph.status not in {'ready', 'running', 'awaiting_confirmation'} or current is None or len(graph.subgoals) != 1 or (graph.subgoals[0].subgoal_id != current.subgoal_id) or (current.external_impact != 'navigation_only') or graph.risk_actions or (not isinstance(graph.goal.entities.get('input_text'), str)), UniversalAgentOrchestratorError('本地 exact_input_text 完成只允许单一、无效果的输入子目标。'))
        observation_id = str(getattr(new_observation, 'observation_id', '') or '').strip()
        fingerprint = str(getattr(new_observation, 'fingerprint', '') or '').strip()
        reject_if(not observation_id or not fingerprint, UniversalAgentOrchestratorError('本地 exact_input_text 完成缺少新 observation/fingerprint。'))
        evidence = (f'动作后观察 {observation_id} 已验证输入框精确值，fingerprint={fingerprint}',)
        completed = replace(graph, revision=graph.revision + 1, status='completed',
            completion_conditions=tuple((replace(condition, satisfied=True,
            evidence=evidence) for condition in graph.completion_conditions)), subgoals=(replace(current,
            status='completed', completion_evidence=evidence),), active_subgoal_id=None)
        completed.validate()
        return completed

    @staticmethod
    def _input_step_reaches_formal_successor(*, graph: DynamicTaskGraph, current_subgoal_id: str, canonical: str,
        expected_value: str) -> bool:
        """Detect a formal direct successor whose required action is the next deterministic input step."""

        try:
            semantic_ir = compile_formal_semantic_authority(graph).semantic_ir
            next_step = plan_next_verified_input(canonical, expected_value)
        except (TaskSemanticIRError, ValueError, VerifiedTextTransactionError):
            return False
        if next_step is None:
            return False
        next_required_action = (
            "press_enter"
            if next_step.kind == "literal_key" and next_step.segment == "\n"
            else "input_verified_text"
        )
        constraints = {item.constraint_id: item for item in semantic_ir.constraints}

        def required_actions(subgoal: Any) -> frozenset[str]:
            return frozenset((str(constraints[ref].value) for ref in subgoal.constraint_refs if ref in constraints
                and constraints[ref].kind == 'required_action'))

        current = next((item for item in semantic_ir.subgoals if item.subgoal_id == current_subgoal_id), None)
        successors = tuple((item for item in semantic_ir.subgoals if item.status == 'pending' and current_subgoal_id
            in item.depends_on))
        if current is None or len(successors) != 1:
            return False
        current_actions = required_actions(current)
        successor_actions = required_actions(successors[0])
        return bool('input_verified_text' in current_actions and next_required_action not in current_actions
            and (next_required_action in successor_actions))

    @staticmethod
    def _build_effect_verification(session: UniversalAgentSessionState, *, graph: DynamicTaskGraph, subgoal: Any,
        receipt: VerifiedActionTransition, consumed_revision: int) -> dict[str, Any]:
        """Bind one matched external effect receipt to a fresh read-only result check."""

        receipt.validate()
        semantic_ir = getattr(session.semantic_task_context, "semantic_ir", None)
        effects = tuple((effect for effect in tuple(getattr(semantic_ir, 'effects',
            ()) or ()) if subgoal.subgoal_id in tuple(effect.source_subgoal_ids)))
        reject_if(len(effects) != 1, UniversalAgentOrchestratorError('外部效果结果复核要求当前子目标唯一绑定一个 EffectIntent。'))
        effect = effects[0]
        previews = tuple((item for item in session.effect_previews if str(item.get('effect_id')
            or '') == effect.effect_id))
        reject_if(len(previews) != 1, UniversalAgentOrchestratorError('外部效果结果复核缺少唯一 EffectPreview。'))
        preview = previews[0]
        preview_digest = str(preview.get("preview_digest") or "")
        reject_if(subgoal.external_impact != 'external_state' or receipt.outcome != 'matched' or receipt.physical_actions != 1 or (receipt.session_id != session.session_id) or (receipt.task_id != graph.task_id) or (receipt.device_id != graph.device_id) or (receipt.prior_revision != graph.revision) or (receipt.subgoal_id != subgoal.subgoal_id) or (str(preview.get('task_id') or '') != graph.task_id) or (str(preview.get('device_id') or '') != graph.device_id) or (int(preview.get('revision') or 0) != graph.revision) or (str(preview.get('effect_kind') or '') != effect.kind) or (not re.fullmatch('[0-9a-f]{64}', preview_digest)), UniversalAgentOrchestratorError('外部效果结果复核的 receipt/EffectIntent/preview 绑定不一致。'))
        return {'protocol_version': '2026-08-19-effect-result-verification-v1', 'status': 'pending',
            'session_id': session.session_id, 'task_id': graph.task_id, 'device_id': graph.device_id,
            'effect_id': effect.effect_id, 'effect_kind': effect.kind, 'effect_preview_digest': preview_digest,
            'subgoal_id': subgoal.subgoal_id, 'receipt_id': receipt.receipt_id,
            'receipt_prior_revision': receipt.prior_revision,
            'receipt_after_observation_id': receipt.after_observation_id,
            'receipt_after_fingerprint': receipt.after_fingerprint, 'consumed_revision': consumed_revision,
            'verification_attempts': 0}

    @staticmethod
    def _validate_pending_effect_verification(session: UniversalAgentSessionState, graph: DynamicTaskGraph) -> dict[str,
        Any]:
        pending = session.effect_verification
        reject_if(not isinstance(pending, dict) or pending.get('status') != 'pending', UniversalAgentOrchestratorError("当前没有待处理的外部效果只读复核。"))
        previews = tuple((item for item in session.effect_previews if str(item.get('effect_id')
            or '') == pending.get('effect_id')))
        subgoal = next((item for item in graph.subgoals if item.subgoal_id == pending.get('subgoal_id')), None)
        transition = session.last_post_action_transition or {}
        receipt = transition.get("receipt") or {}
        reject_if(pending.get('protocol_version') != '2026-08-19-effect-result-verification-v1' or pending.get('session_id') != session.session_id or pending.get('task_id') != graph.task_id or (pending.get('device_id') != graph.device_id) or (pending.get('consumed_revision') != graph.revision) or (pending.get('verification_attempts') != 0) or (subgoal is None) or (subgoal.external_impact != 'external_state') or (len(previews) != 1) or (previews[0].get('preview_digest') != pending.get('effect_preview_digest')) or (receipt.get('receipt_id') != pending.get('receipt_id')) or (receipt.get('outcome') != 'matched') or (receipt.get('physical_actions') != 1) or (receipt.get('subgoal_id') != pending.get('subgoal_id')) or (receipt.get('after_observation_id') != pending.get('receipt_after_observation_id')) or (receipt.get('after_fingerprint') != pending.get('receipt_after_fingerprint')), UniversalAgentOrchestratorError('待复核外部效果与当前 session/graph/receipt/preview 不一致。'))
        return dict(pending)

    def _advance_after_observation(self, session: UniversalAgentSessionState, *, result: Any, before_observation: Any,
        new_observation: Any, prior_verified_app_surface_lineage: VerifiedAppSurfaceLineage | None=None,
        prior_physical_actions: int | None=None) -> None:
        previous_graph = session.task_graph
        assert previous_graph is not None
        action_outcome = str(getattr(result, "action_outcome", "matched"))
        reject_if(action_outcome not in POST_ACTION_OUTCOMES, UniversalAgentOrchestratorError(f'动作后验证返回了不支持的 outcome：{action_outcome}。'))
        matched = action_outcome == "matched"
        verification_errors = tuple((str(item) for item in getattr(result, 'verification_errors',
            ()) if str(item).strip()))
        reject_if(matched and verification_errors, UniversalAgentOrchestratorError('动作后验证同时返回 matched 与 verification_errors。'))
        reject_if(not matched and (not verification_errors), UniversalAgentOrchestratorError('动作后验证返回 mismatched 但没有结构化错误。'))
        previous_current = previous_graph.active_subgoal()
        previous_decision = session.qwen_decision
        authority = session.confirmation_authority
        reject_if(previous_current is None or previous_decision is None or previous_decision.proposal.action is None or (authority is None), UniversalAgentOrchestratorError('动作后重规划缺少上一子目标、决策或确认权威。'))
        input_transaction_microstep = self._verified_input_transaction_microstep(graph=previous_graph,
            previous_decision=previous_decision, result=result, before_observation=before_observation,
            new_observation=new_observation, allow_terminal=session.local_exact_input_authority)
        input_transaction_terminal = bool(input_transaction_microstep and session.local_exact_input_authority
            and self._input_transaction_reached_canonical(previous_graph, result))
        target_app_home_reset_microstep = self._verified_target_app_home_reset_microstep(graph=previous_graph,
            previous_decision=previous_decision, result=result, before_observation=before_observation,
            new_observation=new_observation)
        wait_transition = result.resolved_action.kind == 'wait_for_change' and int(result.physical_actions) == 0
        receipt: VerifiedActionTransition | None = None
        controller_refs: tuple[ControllerTransitionEvidenceRef, ...] = ()
        if not wait_transition:
            receipt = VerifiedActionTransition(receipt_id=f'receipt_{uuid.uuid4().hex}', session_id=session.session_id,
                task_id=previous_graph.task_id, device_id=previous_graph.device_id,
                prior_revision=previous_graph.revision, subgoal_id=previous_current.subgoal_id,
                decision_node_id=authority.decision_node_id, action_digest=authority.action_digest,
                rebound_action_digest=_action_digest(result.rebound_action),
                resolved_action_digest=_action_digest(result.resolved_action),
                action_kind=str(previous_decision.proposal.action.action),
                before_observation_id=str(before_observation.observation_id),
                before_fingerprint=str(before_observation.fingerprint),
                after_observation_id=str(new_observation.observation_id),
                after_fingerprint=str(new_observation.fingerprint), physical_actions=int(result.physical_actions),
                outcome=action_outcome, errors=verification_errors,
                controller_transition_evidence=tuple((str(item) for item in getattr(result,
                'controller_transition_evidence', ()) if str(item).strip())))
            receipt.validate()
            if (previous_current.external_impact == 'navigation_only' and (not input_transaction_microstep)
                and (not target_app_home_reset_microstep)):
                controller_refs = tuple((ControllerTransitionEvidenceRef(ref_id=f'controller_transition:{
                    receipt.receipt_id}:{index}', receipt_id=receipt.receipt_id, subgoal_id=receipt.subgoal_id,
                    text=text) for index, text in enumerate(receipt.controller_transition_evidence, start=1)))
        transition_kind = "wait_observation" if wait_transition else "physical_action"
        receipt_payload = receipt.to_dict() if receipt is not None else None
        controller_evidence = receipt.controller_transition_evidence if receipt is not None else ()
        verification = {'matched': matched, 'action_outcome': action_outcome,
            'physical_actions': result.physical_actions, 'before_fingerprint': before_observation.fingerprint,
            'execution_before_fingerprint': result.before_scene.fingerprint,
            'after_fingerprint': result.after_scene.fingerprint, 'visible_evidence': [result.after_scene.summary],
            'blocked_reasons': list(verification_errors), 'after_frame_paths': list(result.after_frame_paths),
            'execution_metadata': dict(getattr(result, 'execution_metadata', {}) or {}),
            'controller_transition_evidence': list(controller_evidence), 'verified_action_transition': receipt_payload,
            'transition_kind': transition_kind}
        self._remember(session, session.evidence_store.write_verification(max(1, session.step_number - 1),
            verification))
        observed = self.bridge.observed_state(graph=previous_graph, trusted_observation=new_observation,
            action_outcome='not_applicable' if wait_transition else action_outcome, verification=verification,
            verified_action_transition=receipt, controller_transition_evidence_refs=controller_refs)
        previous_signature = _subgoal_progress_signature(previous_current)
        transition_record = {'protocol_version': POST_ACTION_TRANSITION_PROTOCOL_VERSION, 'receipt': receipt_payload,
            'transition_kind': transition_kind, 'prior_subgoal_signature': previous_signature,
            'prior_action_equivalence_digest': _action_equivalence_digest(previous_decision.proposal.action),
            'disposition': 'replanning'}
        if input_transaction_terminal:
            transition_record["input_transaction_completed"] = True
        elif input_transaction_microstep:
            transition_record["input_transaction_progress"] = True
        if target_app_home_reset_microstep:
            transition_record["target_app_home_reset_progress"] = True

        def persist_transition() -> None:
            session.last_post_action_transition = dict(transition_record)
            self._remember(session, session.evidence_store.write_post_action_transition(max(1, session.step_number - 1),
                transition_record))

        def finish_transition(disposition: str, *, status: str | None=None, reason: str | None=None) -> None:
            transition_record["disposition"] = disposition
            if status is not None:
                self._set_status(session, status, reason or "")
            if reason:
                transition_record["diagnostic"] = reason
            persist_transition()

        persist_transition()
        self._clear_action_decision(session, effects=True)
        try:
            if target_app_home_reset_microstep:
                revised = previous_graph
            elif input_transaction_microstep:
                revised = self._complete_local_exact_input_graph(previous_graph,
                    new_observation=new_observation) if input_transaction_terminal else previous_graph
            else:
                revised = self.deepseek_planner.replan(
                    previous_graph,
                    observed,
                    trigger=(
                        "observation_changed"
                        if wait_transition
                        else "action_result_matched"
                        if matched
                        else "action_result_mismatch"
                    ),
                    reason=(
                        "wait_for_change 未产生物理动作；仅依据新的可信画面重规划。"
                        if wait_transition
                        else "一个动作已经执行并由新的可信画面验证。"
                        if matched
                        else "动作已执行，但新画面没有证明预期语义变化，必须重规划。"
                    ),
                )
            next_lineage = None
            if not input_transaction_microstep and (not target_app_home_reset_microstep):
                next_lineage = AppSurfaceLineageAuthority.build(session=session, previous=previous_graph,
                    revised=revised, trusted_observation=new_observation, receipt=receipt,
                    controller_refs=controller_refs, before_observation=before_observation,
                    previous_decision=previous_decision, execution_result=result)
            if next_lineage is None and prior_physical_actions is not None:
                next_lineage = AppSurfaceLineageAuthority.carry(session=session, previous=previous_graph,
                    revised=revised, trusted_observation=new_observation,
                    prior_lineage=prior_verified_app_surface_lineage, prior_physical_actions=prior_physical_actions,
                    execution_result=result)
            if not input_transaction_microstep and (not target_app_home_reset_microstep):
                self._validate_graph_identity(revised, device_id=session.device_id, previous=previous_graph,
                    trusted_observation=new_observation, session_id=session.session_id, verified_transition=receipt,
                    controller_transition_evidence_refs=controller_refs, before_observation=before_observation,
                    previous_decision=previous_decision, execution_result=result,
                    verified_app_surface_lineage=next_lineage, physical_actions=session.physical_actions)
            session.verified_app_surface_lineage = next_lineage
        except Exception as exc:
            reason = f"DeepSeek 重规划失败：{exc}"
            self._record_deepseek_failure(session, exc, stage='post_action_replan', previous_graph=previous_graph)
            finish_transition("blocked_replan_failure", status="blocked", reason=reason)
            return

        session.task_graph = revised
        session.goal_draft = self.bridge.goal_draft(revised)
        revised_current = revised.active_subgoal()
        transition_record["revised_revision"] = revised.revision
        transition_record['revised_subgoal_id'] = revised_current.subgoal_id if revised_current is not None else None
        transition_record['revised_subgoal_signature'] = _subgoal_progress_signature(revised_current)
        if receipt is not None:
            transition_record["receipt_consumed_revision"] = revised.revision
        if not input_transaction_microstep and (not target_app_home_reset_microstep) or input_transaction_terminal:
            self._remember(session, session.evidence_store.write_task_graph(revised),
                session.evidence_store.write_effect_policy_snapshot(revised))
        if revised.status == 'completed':
            finish_transition("task_completed", status="succeeded")
            return
        if matched and receipt is not None and (previous_current.external_impact == 'external_state'):
            try:
                session.effect_verification = self._build_effect_verification(session, graph=previous_graph,
                    subgoal=previous_current, receipt=receipt, consumed_revision=revised.revision)
            except Exception as exc:
                finish_transition('blocked_effect_verification_binding', status='blocked', reason=f'外部效果只读复核绑定失败：{exc}')
                return
            self._set_status(session, "needs_effect_verification")
            self._clear_action_decision(session, effects=True)
            transition_record['disposition'] = 'pending_read_only_effect_result_verification'
            transition_record['effect_verification'] = dict(session.effect_verification)
            persist_transition()
            self._complete_pending_effect_verification(session, graph=revised, observation=new_observation,
                before_actions=session.physical_actions)
            transition_record["disposition"] = (
                "effect_verified_from_same_post_action_observation"
                if session.status == "succeeded"
                else "blocked_same_post_action_effect_verification"
            )
            transition_record['effect_verification'] = dict(session.effect_verification or {})
            if session.task_graph is not None:
                verified_current = session.task_graph.active_subgoal()
                transition_record['effect_verification_revision'] = session.task_graph.revision
                transition_record["effect_verification_subgoal_id"] = (
                    verified_current.subgoal_id
                    if verified_current is not None
                    else None
                )
            finish_transition(str(transition_record['disposition']), reason=session.failed_reason or None)
            return
        if controller_refs:
            prior_in_revised = next((item for item in revised.subgoals if item.subgoal_id ==
                previous_current.subgoal_id), None)
            if prior_in_revised is None or prior_in_revised.status != 'completed':
                reason = '本地一次性 controller_transition 完成证据已满足，但重规划未完成其绑定的 navigation_only 子目标；禁止继续产生动作或第二确认。'
                finish_transition("blocked_unconsumed_controller_completion", status="blocked", reason=reason)
                return
        current = revised.active_subgoal()
        impact = current.external_impact if current is not None else "unknown"
        if current is None:
            self._set_status(session, "blocked", "重规划后的任务图没有活动子目标。")
            finish_transition("blocked_missing_active_subgoal")
            return
        current_focus_changed = _subgoal_progress_signature(current) != previous_signature
        if _requires_effect_confirmation(revised, current):
            self._set_status(session, "awaiting_effect_confirmation")
            self._bind_effect_confirmation(session)
            finish_transition("advanced_to_effect_confirmation")
            return
        if current_focus_changed:
            self._require_reobservation(session)
            transition_record["reobservation_subgoal_id"] = current.subgoal_id
            finish_transition("advanced_to_current_subgoal_reobservation")
            return
        if (not matched and _allows_fresh_observation_corrective_retry(impact=impact,
            action_kind=previous_decision.proposal.action.action)):
            self._require_reobservation(session)
            transition_record["reobservation_subgoal_id"] = current.subgoal_id
            finish_transition("navigation_mismatch_needs_fresh_observation")
            return
        if impact == 'read_only':
            try:
                read_only_observed = replace(observed, last_action_outcome='not_applicable',
                    verified_action_transition=None, controller_transition_evidence_refs=())
                reviewed, _ = self._replan_current_observation(session, revised, new_observation,
                    observed=read_only_observed, trigger='subgoal_completed', store=True,
                    reason='当前 read_only 子目标只能用已经采集的当前可信画面完成或阻塞；不得请求任何新的物理动作。')
            except Exception as exc:
                finish_transition('blocked_read_only_review', status='blocked', reason=f'只读完成复核失败：{exc}')
                return
            if reviewed.status == 'completed':
                finish_transition("task_completed_after_read_only_review", status="succeeded")
                return
            reviewed_current = reviewed.active_subgoal()
            reviewed_impact = reviewed_current.external_impact if reviewed_current is not None else 'unknown'
            if reviewed_current is None:
                finish_transition('blocked_read_only_no_active', status='blocked', reason='只读复核后的任务图没有活动子目标。')
                return
            if _requires_effect_confirmation(reviewed, reviewed_current):
                self._set_status(session, "awaiting_effect_confirmation")
                self._bind_effect_confirmation(session)
                finish_transition("advanced_to_effect_confirmation_after_read_only")
                return
            if _subgoal_progress_signature(reviewed_current) != _subgoal_progress_signature(current):
                self._require_reobservation(session)
                transition_record['reobservation_subgoal_id'] = reviewed_current.subgoal_id
                finish_transition("read_only_advanced_to_current_subgoal_reobservation")
                return
            if reviewed_impact == 'read_only':
                reason = '当前可信画面没有让 DeepSeek 完成 read_only 子目标；禁止为只读验证请求物理动作。'
                finish_transition("blocked_read_only_incomplete", status="blocked", reason=reason)
                return
            revised = reviewed
            transition_record["revised_revision"] = revised.revision
            transition_record["revised_subgoal_id"] = revised.active_subgoal_id
            transition_record['revised_subgoal_signature'] = _subgoal_progress_signature(revised.active_subgoal())

        frames = list(result.after_frames)
        context = revised.to_qwen_context()
        def block_equivalent_repeat(next_decision: Any) -> str | None:
            new_equivalence_digest = _action_equivalence_digest(next_decision.proposal.action)
            transition_record['next_action_equivalence_digest'] = new_equivalence_digest
            if (not (bool(controller_refs) and matched and (transition_record.get('revised_subgoal_signature') ==
                previous_signature) and (new_equivalence_digest ==
                transition_record['prior_action_equivalence_digest']))):
                return None
            reason = '一次性 controller_transition 完成证据已满足，但同一活动子目标仍提出等价动作；禁止生成第二确认。'
            transition_record["disposition"] = "blocked_equivalent_repeat"
            transition_record["diagnostic"] = reason
            return reason

        decision = self._stage_current_observation_decision(session, graph=revised, frames=frames, task_context=context,
            trusted_observation=new_observation, unsupported_status_reason=lambda status: '本地动作选择器返回了不支持的状态：' + status,
            before_selection=block_equivalent_repeat)
        if session.status == 'blocked':
            if decision.proposal.status != 'action':
                transition_record["disposition"] = "blocked_qwen_no_action"
            elif transition_record.get('disposition') != 'blocked_equivalent_repeat':
                transition_record["disposition"] = "blocked_canonical_selection"
            finish_transition(str(transition_record['disposition']), reason=session.failed_reason)
            return
        transition_record['next_confirmation_scope'] = session.confirmation_authority.scope()
        finish_transition("advanced_to_new_confirmation")

    def _complete_pending_effect_verification(self, session: UniversalAgentSessionState, *, graph: DynamicTaskGraph,
        observation: Any, before_actions: int) -> Any:
        pending = self._validate_pending_effect_verification(session, graph)
        observed = self.bridge.observed_state(graph=graph, trusted_observation=observation,
            action_outcome='not_applicable', verification={'visible_evidence': [observation.scene.summary],
            'blocked_reasons': []})
        try:
            revised, _ = self._replan_current_observation(session, graph, observation, observed=observed,
                trigger='observation_changed',
                reason='外部效果动作已有严格一次性 matched receipt；本轮仅用新鲜 typed visual claim 复核结果，禁止规划或重复任何效果动作。')
        except Exception as exc:
            failed = {**pending, 'status': 'failed', 'verification_attempts': 1,
                'verification_observation_id': observation.observation_id,
                'verification_fingerprint': observation.fingerprint, 'reason': f'外部效果只读结果复核失败：{exc}'}
            session.effect_verification = failed
            session.status = "blocked"
            session.failed_reason = failed["reason"]
            session.qwen_decision = None
            session.controller_decision = CanonicalSelectionReceipt(allowed=False, reason=session.failed_reason)
            reject_if(session.physical_actions != before_actions, UniversalAgentOrchestratorError('外部效果只读复核失败路径错误地改变了物理动作计数。'))
            decision = SimpleNamespace(proposal=GenericStepProposal(status='blocked', reason=session.failed_reason))
            self._write_terminal_snapshot(session)
            return decision

        session.qwen_decision = None
        session.controller_decision = None
        session.confirmation_authority = None
        session.effect_confirmation_authority = None
        session.confirmed_effect_ids = ()
        result_subgoal = next((item for item in revised.subgoals if item.subgoal_id == pending['subgoal_id']), None)
        current_visual_refs = {item.ref_id for item in observed.visual_claim_evidence_refs}
        visual_result_proven = bool(revised.status == 'completed' and result_subgoal is not None
            and (result_subgoal.status == 'completed') and set(result_subgoal.completion_evidence).intersection(
            current_visual_refs))
        if visual_result_proven or revised.status != 'completed':
            session.task_graph = revised
            session.goal_draft = self.bridge.goal_draft(revised)
            self._remember(session, session.evidence_store.write_task_graph(revised),
                session.evidence_store.write_effect_policy_snapshot(revised))
        final = {**pending, 'status': 'verified' if visual_result_proven else 'failed', 'verification_attempts': 1,
            'verification_observation_id': observation.observation_id,
            'verification_fingerprint': observation.fingerprint,
            'visual_claim_refs': sorted(set(result_subgoal.completion_evidence).intersection(
            current_visual_refs) if result_subgoal is not None else ())}
        session.effect_verification = final
        if visual_result_proven:
            session.status = "succeeded"
            session.failed_reason = ""
            proposal = TaskGraphTransitionReport(status='completed', reason='新的可信画面已证明一次性外部效果结果。',
                completion_evidence=tuple(final['visual_claim_refs'][:3]))
        else:
            session.status = "blocked"
            session.failed_reason = '新的只读观察仍未以当前 typed visual claim 证明外部效果结果；效果动作不会重试。'
            final["reason"] = session.failed_reason
            session.effect_verification = final
            proposal = GenericStepProposal(status='blocked', reason=session.failed_reason)
        reject_if(session.physical_actions != before_actions, UniversalAgentOrchestratorError('外部效果只读复核错误地改变了物理动作计数。'))
        decision = SimpleNamespace(proposal=proposal)
        self._write_terminal_snapshot(session)
        return decision

    def refresh_decision(self, session: UniversalAgentSessionState) -> Any:
        """Replace the pending decision from a fresh read-only scene after invalidating its old scope."""

        reject_if(self.device_registry.active_session(session.device_id) != session.session_id, UniversalAgentOrchestratorError('当前会话已不再拥有该设备，禁止重新观察。'))
        try:
            with self._vision_usage_scope(session.vision_usage), self.device_registry.device_lock(session.device_id):
                return self._refresh_decision_locked(session)
        finally:
            self._release_if_terminal(session)

    def _refresh_decision_locked(self, session: UniversalAgentSessionState) -> Any:
        graph = session.task_graph
        goal = session.goal_draft
        reject_if(graph is None or goal is None, UniversalAgentOrchestratorError("当前会话缺少任务图或目标投影。"))
        self._validate_graph_identity(graph, device_id=session.device_id)
        pending_effect = None
        if session.status == 'needs_effect_verification':
            pending_effect = self._validate_pending_effect_verification(session, graph)
        current = graph.active_subgoal()
        observed_subgoal_signature = _subgoal_progress_signature(current)
        impact = current.external_impact if current is not None else "unknown"
        reject_if(pending_effect is None and session.status == 'awaiting_effect_confirmation', UniversalAgentOrchestratorError('当前子目标必须先满足本地效果策略，禁止提前调用 Qwen。'))
        reject_if(pending_effect is None and (current is None or (_requires_effect_confirmation(graph, current) and (not session.confirmed_effect_ids))), UniversalAgentOrchestratorError(f'当前 {impact} 子目标缺少有效效果确认。'))

        prior_observation = session.trusted_observation
        authority = session.confirmation_authority
        if authority is not None:
            authority.consumed = True
            authority.invalid_reason = "fresh_observation_requested"
        session.confirmation_authority = None
        before_actions = session.physical_actions
        try:
            self._set_status(session, "observing")
            observation_id = f"obs_{uuid.uuid4().hex}"
            scene, frames, frame_paths = session.adapter.capture_scene(
                goal,
                evidence_dir=session.run_dir,
                prefix=(
                    f"refresh_step_{session.step_number}_"
                    f"{observation_id[-8:]}_frame"
                ),
            )
            self._remember(session, frame_paths)
            try:
                observation = self._build_and_record_current_observation(session, scene=scene, frames=frames,
                    observation_id=observation_id)
            except EvidenceStoreError:
                raise
            except Exception as exc:
                self._set_status(session, "blocked", f"重新观察证据不足：{exc}")
                self._clear_action_decision(session)
                session.controller_decision = CanonicalSelectionReceipt(allowed=False, reason=session.failed_reason)
                reject_if(session.physical_actions != before_actions, UniversalAgentOrchestratorError('重新观察证据失败路径错误地改变了物理动作计数。'))
                self._write_terminal_snapshot(session)
                return self._blocked_decision(session.failed_reason)
            lineage = session.verified_app_surface_lineage
            if (lineage is not None and (lineage.physical_actions != session.physical_actions
                or not self._lineage_matches_observed_foreground(lineage, str(scene.foreground_app_id)))):
                session.verified_app_surface_lineage = AppSurfaceLineageAuthority.refresh(session=session,
                    graph=graph, prior_observation=prior_observation, new_observation=observation)

            if pending_effect is not None:
                return self._complete_pending_effect_verification(session, graph=graph, observation=observation,
                    before_actions=before_actions)

            current = graph.active_subgoal()
            if current is not None and current.external_impact in {'read_only', 'navigation_only'}:
                visible_revised = self._try_advance_visible_subgoal(session, graph=graph,
                    trusted_observation=observation)
                if visible_revised is not None:
                    self._store_revised_graph(session, visible_revised)
                    graph = visible_revised
                    return self._finish_visible_advancement(session, visible_revised, evidence=scene.summary,
                        reason='当前可见状态已由 DeepSeek 任务图和本地完成条件共同复核。', missing_reason='可见状态证据推进后没有活动子目标。')

            if (prior_observation is not None and str(getattr(prior_observation, 'fingerprint',
                '')) != str(observation.fingerprint)):
                observed = self.bridge.observed_state(graph=graph, trusted_observation=observation,
                    action_outcome='not_applicable', verification={'visible_evidence': [scene.summary],
                    'blocked_reasons': []})
                try:
                    revised, _ = self._replan_current_observation(session, graph, observation, observed=observed,
                        trigger='observation_changed', reason='只读重新观察发现页面指纹变化；必须先修订高层状态，再允许 Qwen 规划下一动作。',
                        store=True)
                except Exception as exc:
                    self._set_status(session, "blocked", f"页面变化重规划失败：{exc}")
                    self._clear_action_decision(session, effects=True)
                    session.controller_decision = CanonicalSelectionReceipt(allowed=False, reason=session.failed_reason)
                    reject_if(session.physical_actions != before_actions, UniversalAgentOrchestratorError('页面变化重规划失败路径错误地改变了物理动作计数。'))
                    self._write_terminal_snapshot(session)
                    return self._blocked_decision(session.failed_reason)
                self._clear_action_decision(session, effects=True)
                graph = revised
                goal = session.goal_draft
                if revised.status == 'completed':
                    self._set_status(session, "succeeded")
                    terminal_decision = self._transition_decision('completed', 'DeepSeek 已依据新的可信画面确认任务完成。',
                        observed.visible_evidence[:3])
                    reject_if(session.physical_actions != before_actions, UniversalAgentOrchestratorError('重新观察路径错误地改变了物理动作计数。'))
                    self._write_terminal_snapshot(session)
                    return terminal_decision
                current = revised.active_subgoal()
                impact = current.external_impact if current is not None else 'unknown'
                if current is None:
                    self._set_status(session, "blocked", "页面变化重规划后没有活动子目标。")
                    self._write_terminal_snapshot(session)
                    return self._blocked_decision(session.failed_reason)
                if _requires_effect_confirmation(graph, current):
                    self._set_status(session, "awaiting_effect_confirmation")
                    self._bind_effect_confirmation(session)
                    self._write_terminal_snapshot(session)
                    return self._blocked_decision("页面变化后必须重新确认当前效果作用域。")
                if _subgoal_progress_signature(current) != observed_subgoal_signature:
                    self._require_reobservation(session)
                    progressed_decision = self._transition_decision('progressed', '页面变化已推进活动子目标；必须按新子目标重新截图后再选择动作。',
                        (scene.summary,))
                    self._write_terminal_snapshot(session)
                    return progressed_decision

            if session.confirmed_effect_ids:
                context = graph.to_qwen_context(confirmed_effect_ids=session.confirmed_effect_ids,
                    confirmed_task_id=graph.task_id, confirmed_device_id=graph.device_id,
                    confirmed_subgoal_id=graph.active_subgoal_id, confirmed_revision=graph.revision)
            else:
                context = graph.to_qwen_context()
            decision = self._stage_current_observation_decision(session, graph=graph, frames=frames,
                task_context=context, trusted_observation=observation,
                unsupported_status_reason=lambda status: f'不支持的 Qwen 状态：{status}')
            reject_if(session.physical_actions != before_actions, UniversalAgentOrchestratorError('重新观察路径错误地改变了物理动作计数。'))
            self._write_terminal_snapshot(session)
            return decision
        except Exception as exc:
            self._set_status(session, "failed", str(exc))
            self._record_deepseek_failure(session, exc, stage='refresh_decision')
            self._best_effort_terminal_snapshot(session)
            raise

    def confirm_one(self, session: UniversalAgentSessionState, confirmation: Mapping[str, Any]) -> Any:
        reject_if(self.device_registry.active_session(session.device_id) != session.session_id, UniversalAgentOrchestratorError('当前会话已不再拥有该设备，禁止执行。'))
        before_actions = session.physical_actions
        authority_before = session.confirmation_authority
        post_transition_before = session.last_post_action_transition
        try:
            with self._vision_usage_scope(session.vision_usage), self.device_registry.device_lock(session.device_id):
                try:
                    return self._confirm_one_locked(session, confirmation)
                except Exception as exc:
                    self._finalize_confirm_failure(session, exc, before_actions=before_actions,
                        authority_before=authority_before, post_transition_before=post_transition_before)
                    raise
        finally:
            self._release_if_terminal(session)

    def _finalize_confirm_failure(self, session: UniversalAgentSessionState, error: Exception, *, before_actions: int,
        authority_before: Any, post_transition_before: Any) -> None:
        """Converge artifacts after a consumed confirm without observing, executing, or masking its error."""

        authority = session.confirmation_authority or authority_before
        authority_consumed = bool(authority is not None and getattr(authority, 'consumed', False))
        request_actions = max(0, session.physical_actions - int(before_actions))
        if not (authority_consumed or request_actions):
            return

        # Never replace a more specific transition already recorded by the after-observation path.
        if session.last_post_action_transition is not post_transition_before:
            self._best_effort_terminal_snapshot(session, ensure=True)
            return

        failed_stage = session.confirm_stage or "confirmation"
        reason = str(error).strip() or error.__class__.__name__
        if request_actions == 0 and session.status == "blocked":
            # A policy-recheck rejection remains blocked after its one-shot token is consumed.
            self._best_effort_terminal_snapshot(session, ensure=True)
            return
        recoverable_reobservation = bool(request_actions == 0 and session.status == 'needs_reobservation')
        if not recoverable_reobservation:
            session.status = "failed"
        session.failed_reason = reason

        def failure_transition(transition_kind: str, disposition: str) -> dict[str, Any]:
            transition: dict[str, Any] = {'protocol_version': POST_ACTION_TRANSITION_PROTOCOL_VERSION,
                'transition_kind': transition_kind, 'disposition': disposition, 'failed_stage': failed_stage,
                'error_type': error.__class__.__name__, 'error': reason,
                'authority_consumed': authority_consumed, 'physical_actions_before': int(before_actions),
                'physical_actions': int(session.physical_actions), 'request_physical_actions': request_actions,
                'evidence': list(dict.fromkeys(session.evidence_paths))}
            if authority is not None and callable(getattr(authority, 'scope', None)):
                try:
                    transition["authority_scope"] = authority.scope()
                except Exception:
                    pass
            execution_metadata = dict(getattr(error, 'execution_metadata', {}) or {})
            if execution_metadata:
                transition['execution_metadata'] = execution_metadata
            return transition

        pre_action_failure = request_actions == 0
        requested_kind = str(getattr(getattr(getattr(session.qwen_decision, 'proposal', None), 'action', None),
            'action', '') or '') if pre_action_failure else ''
        transition_kind = ('post_action_failure' if not pre_action_failure else 'confirmation_failure'
            if failed_stage == 'validating_confirmation' else 'wait_observation_failure'
            if requested_kind == 'wait_for_change' else 'pre_action_failure')
        transition = failure_transition(transition_kind,
            'needs_reobservation' if recoverable_reobservation else 'failed')
        transition_field, writer = (('last_confirmation_failure', session.evidence_store.write_confirmation_failure)
            if pre_action_failure else ('last_post_action_transition',
            session.evidence_store.write_post_action_transition))
        setattr(session, transition_field, transition)
        failure_step = max(1, session.step_number)
        if (not pre_action_failure and session.history and authority is not None and (latest := session.history[-1]
            ).get('task_revision') == getattr(authority, 'revision', None)):
            failure_step = max(1, int(latest.get("step_number") or failure_step))
        try:
            self._remember(session, writer(failure_step, transition))
        except Exception:
            pass
        self._best_effort_terminal_snapshot(session)

    def _confirm_one_locked(self, session: UniversalAgentSessionState, confirmation: Mapping[str, Any]) -> Any:
        session.confirm_stage = "validating_confirmation"
        self._validate_and_consume_confirmation(session, confirmation)
        authority = session.confirmation_authority
        assert authority is not None
        graph = session.task_graph
        observation = session.trusted_observation
        decision = session.qwen_decision
        assert graph is not None and observation is not None and decision is not None

        def reject(reason: str, *, snapshot: bool=False) -> NoReturn:
            self._set_status(session, "failed", reason)
            if snapshot:
                self._best_effort_terminal_snapshot(session)
            raise UniversalAgentOrchestratorError(reason)

        session.confirm_stage = "scope_consumed"
        selection_receipt = session.controller_decision
        reject_if(selection_receipt is None or not selection_receipt.allowed, UniversalAgentOrchestratorError('确认作用域缺少已验证的 canonical selection receipt。'))
        authority.consumed = True
        authority.invalid_reason = "consumed_before_execution"

        session.confirm_stage = "pre_execute_evidence"
        try:
            self._remember(session, session.evidence_store.write_controller_decision(session.step_number,
                {**self._selection_receipt_payload(selection_receipt), 'phase': 'pre_execute_scope_consume'}))
        except Exception:
            self._set_status(session, "failed", "执行前控制器证据写入失败。")
            raise

        session.status = "executing_one_action"
        session.confirm_stage = "executing"
        prior_verified_app_surface_lineage = session.verified_app_surface_lineage
        prior_physical_actions = session.physical_actions
        if decision.proposal.action.action != 'wait_for_change':
            session.verified_app_surface_lineage = None
        try:
            result = session.adapter.execute(requested_action=decision.proposal.action, planned_scene=observation.scene,
                goal=session.goal_draft, confirmed=True, evidence_dir=session.run_dir,
                planned_frames=session.trusted_frames)
        except GenericActionAdapterError as exc:
            failed_physical_actions = max(0, int(exc.physical_actions))
            if failed_physical_actions == 0:
                session.verified_app_surface_lineage = prior_verified_app_surface_lineage
            session.physical_actions += failed_physical_actions
            self._remember(session, exc.evidence)
            self._set_status(session, 'needs_reobservation' if failed_physical_actions == 0 else 'failed', str(exc))
            self._best_effort_terminal_snapshot(session)
            raise

        self._remember(session, getattr(result, 'evidence', ()), getattr(result, 'after_frame_paths', ()))
        session.confirm_stage = "validating_execution_result"
        physical_actions = int(result.physical_actions)
        wait_transition = result.resolved_action.kind == "wait_for_change"
        if physical_actions != 1 and (not (wait_transition and physical_actions == 0)):
            session.physical_actions += max(0, physical_actions)
            reject(f'已确认动作必须产生一次物理动作，或仅 wait_for_change 产生零动作；实际返回：{physical_actions}。')
        session.physical_actions += physical_actions

        requested_action = decision.proposal.action
        rebound_params = dict(getattr(result.rebound_action, "params", {}) or {})
        resolved_kind = str(getattr(result.resolved_action, "kind", ""))
        target_fields = (('target_element_id', 'element_id'),) if resolved_kind in {'tap_semantic', 'dismiss_overlay',
            'input_verified_text', 'press_enter', 'clear_verified_text', 'long_press'} else (('target_element_id',
            'source_element_id'), ('destination_element_id',
            'destination_element_id')) if resolved_kind == 'drag' else ()
        resolved_target_binding_ok = all((str(getattr(result.resolved_action, resolved_field,
            '') or '') == str(rebound_params.get(rebound_field) or '') for resolved_field,
            rebound_field in target_fields))
        if (requested_action is None or _action_digest(requested_action) != authority.action_digest
            or _action_digest(result.requested_action) != authority.action_digest
            or (str(getattr(result.requested_action, 'node_id',
            '')) != authority.decision_node_id) or (str(getattr(result.rebound_action, 'node_id',
            '')) != authority.decision_node_id) or (str(getattr(result.resolved_action, 'node_id',
            '')) != authority.decision_node_id) or (str(getattr(result.resolved_action, 'kind',
            '')) != str(getattr(result.rebound_action, 'action',
            ''))) or (not resolved_target_binding_ok) or (dict(getattr(result.resolved_action, 'expected_effect',
            {}) or {}) != dict(getattr(result.rebound_action, 'params', {}).get('expected_effect', {}) or {}))):
            reject('执行结果没有严格绑定 confirmed/requested/rebound/resolved 动作链。')
        session.status = "verifying"
        session.confirm_stage = "validating_post_action_evidence"
        action_outcome = str(getattr(result, "action_outcome", ""))
        verification_errors = tuple((str(item) for item in getattr(result, 'verification_errors',
            ()) if str(item).strip()))
        if action_outcome not in POST_ACTION_OUTCOMES:
            reject(f"动作结果 outcome 无效：{action_outcome}。")
        if (action_outcome == 'matched') == bool(verification_errors):
            reject("动作结果 outcome 与 verification_errors 不一致。")
        if (result.planned_scene_fingerprint != observation.fingerprint or result.confirmation_frame_identity_verified
            is not True):
            reject("动作结果没有绑定确认 scope 的规划画面。")
        if result.resolved_action.before_fingerprint != result.before_scene.fingerprint:
            reject("动作结果没有绑定复核后的执行前画面。")
        after_frames = tuple(getattr(result, "after_frames", ()))
        after_paths = tuple((str(item).strip() for item in getattr(result, 'after_frame_paths', ())))
        if (len(after_frames) < 4 or len(after_paths) != len(after_frames) or any((not item for item in after_paths))
            or (len(set(after_paths)) != len(after_paths))):
            reject("动作后可信观察缺少完整且唯一的原始帧证据。")
        if (action_outcome == 'matched' and result.resolved_action.kind != 'wait_for_change'
            and (result.after_scene.fingerprint == observation.fingerprint)):
            reject("动作后 fingerprint 没有变化，禁止继续。", snapshot=True)

        session.confirm_stage = "building_trusted_observation"
        new_observation = self.trusted_observation_factory(frames=list(after_frames), device_id=session.device_id,
            scene=result.after_scene, observation_id=f'obs_{uuid.uuid4().hex}')
        if (str(getattr(new_observation, 'device_id', '')) != session.device_id or str(getattr(new_observation,
            'fingerprint', '')) != result.after_scene.fingerprint or str(getattr(getattr(new_observation, 'scene',
            None), 'fingerprint', '')) != result.after_scene.fingerprint):
            reject("动作后可信观察未严格绑定 device/after scene fingerprint。")
        if (new_observation.observation_id == observation.observation_id or (action_outcome == 'matched'
            and result.resolved_action.kind != 'wait_for_change'
            and (new_observation.fingerprint == observation.fingerprint))):
            reject("动作后可信观察 observation/fingerprint 未更新。")
        session.confirm_stage = "persisting_post_observation"
        session.trusted_observation = new_observation
        session.trusted_frames = after_frames
        session.step_number += 1
        self._remember(session, session.evidence_store.write_trusted_observation(session.step_number, new_observation))
        session.history.append({'step_number': session.step_number - 1, 'task_revision': graph.revision,
            'qwen_decision': UniversalAgentSessionState._serialize(decision), 'execution': result.to_dict(),
            'before_observation_id': observation.observation_id, 'before_fingerprint': observation.fingerprint,
            'after_observation_id': new_observation.observation_id, 'after_fingerprint': new_observation.fingerprint})
        try:
            session.status = "replanning"
            session.confirm_stage = "replanning"
            self._advance_after_observation(session, result=result, before_observation=observation,
                new_observation=new_observation, prior_verified_app_surface_lineage=prior_verified_app_surface_lineage,
                prior_physical_actions=prior_physical_actions)
            session.confirm_stage = "completed"
            self._write_terminal_snapshot(session)
            return result
        except EvidenceStoreError as exc:
            self._set_status(session, "failed", str(exc))
            raise
        except Exception as exc:
            self._set_status(session, "failed", str(exc))
            self._record_deepseek_failure(session, exc, stage='post_action_replan')
            self._best_effort_terminal_snapshot(session)
            raise

    def approve_effects(self, session: UniversalAgentSessionState, confirmation: Mapping[str, Any]) -> Any:
        """Consume one typed-effect approval and at most its separately scoped canonical action."""

        reject_if(self.device_registry.active_session(session.device_id) != session.session_id, UniversalAgentOrchestratorError('当前会话已不再拥有该设备，禁止确认风险。'))
        try:
            with self._vision_usage_scope(session.vision_usage), self.device_registry.device_lock(session.device_id):
                reject_if(session.status != 'awaiting_effect_confirmation', UniversalAgentOrchestratorError(f'当前状态不能确认风险：{session.status}。'))
                authority = session.effect_confirmation_authority
                reject_if(authority is None or authority.consumed, UniversalAgentOrchestratorError("当前效果确认已失效或已使用。"))
                requested = self._normalize_effect_confirmation(confirmation)
                if requested != authority.scope():
                    authority.consumed = True
                    authority.invalid_reason = "effect_scope_mismatch"
                    raise UniversalAgentOrchestratorError('效果确认与当前 task/device/revision/subgoal/effect 不一致。')
                authority.consumed = True
                authority.invalid_reason = "consumed_before_observation"
                session.confirmed_effect_ids = tuple(authority.effect_ids)
                graph = session.task_graph
                assert graph is not None
                context = graph.to_qwen_context(confirmed_effect_ids=session.confirmed_effect_ids,
                    confirmed_task_id=graph.task_id, confirmed_device_id=graph.device_id,
                    confirmed_subgoal_id=graph.active_subgoal_id, confirmed_revision=graph.revision)
                result = self._observe_after_effect_confirmation(session, context)
                if session.status == 'awaiting_confirmation':
                    action_authority = session.confirmation_authority
                    reject_if(action_authority is None or action_authority.consumed, UniversalAgentOrchestratorError('效果确认后没有形成一次性精确动作作用域。'))
                    result = self._confirm_one_locked(session, action_authority.scope())
                self._write_terminal_snapshot(session)
                return result
        finally:
            self._release_if_terminal(session)

    def run_autonomous_safe_loop(self, session: UniversalAgentSessionState, *, max_physical_actions: int=12,
        max_iterations: int=24) -> dict[str, Any]:
        reject_if(self.device_registry.active_session(session.device_id) != session.session_id, UniversalAgentOrchestratorError('当前会话已不再拥有该设备，禁止自动推进。'))
        for (value, maximum, message) in ((max_physical_actions, 20, '安全动作预算必须是1～20。'), (max_iterations, 40,
            '安全迭代预算必须是1～40。')):
            reject_if(isinstance(value, bool) or not isinstance(value, int) or (not 1 <= value <= maximum), UniversalAgentOrchestratorError(message))

        start_actions = session.physical_actions
        iterations = 0
        pending_corrective_retry: dict[str, Any] | None = None

        def observation_identity() -> tuple[str, str]:
            observation = session.trusted_observation
            return (str(getattr(observation, 'observation_id', '') or ''), str(getattr(observation, 'fingerprint',
                '') or ''))

        def action_kind() -> str:
            proposal = getattr(session.qwen_decision, "proposal", None)
            action = getattr(proposal, "action", None)
            return str(getattr(action, "action", "") or "")

        def finish_retry(status: str, reason: str='', *, fail_session: bool=False, **fields: Any) -> None:
            nonlocal pending_corrective_retry
            assert pending_corrective_retry is not None
            pending_corrective_retry.update(status=status, **fields)
            if reason:
                pending_corrective_retry["stop_reason"] = reason
            if fail_session:
                self._set_status(session, "failed", reason)
                session.auto_pause_reason = reason
            pending_corrective_retry = None

        session.automatic_loop_enabled = True
        session.auto_pause_reason = ""
        try:
            with self._vision_usage_scope(session.vision_usage), self.device_registry.device_lock(session.device_id):
                while iterations < max_iterations:
                    if (session.status in {'succeeded', 'blocked', 'failed', 'cancelled',
                        'awaiting_effect_confirmation'}):
                        break
                    graph = session.task_graph
                    current = graph.active_subgoal() if graph is not None else None
                    impact = current.external_impact if current is not None else "unknown"
                    if impact not in {'read_only', 'navigation_only'}:
                        if pending_corrective_retry is not None:
                            finish_retry('stopped_before_corrective_action', '重新规划后的子目标不再属于普通只读或导航动作。')
                        session.auto_pause_reason = '下一子目标可能产生外部影响或仍未知，已在物理动作前停止。'
                        break
                    if session.status == 'needs_reobservation':
                        prior_observation_id, _ = observation_identity()
                        self._refresh_decision_locked(session)
                        iterations += 1
                        if pending_corrective_retry is not None:
                            refreshed_observation_id, refreshed_fingerprint = observation_identity()
                            pending_corrective_retry.update(refresh_observation_id=refreshed_observation_id,
                                refresh_fingerprint=refreshed_fingerprint)
                            if not refreshed_observation_id or refreshed_observation_id == prior_observation_id:
                                finish_retry('stopped_before_corrective_action', '纠正重观察没有形成新的 observation。',
                                    fail_session=True)
                            else:
                                refreshed_graph = session.task_graph
                                refreshed_current = refreshed_graph.active_subgoal(
                                    ) if refreshed_graph is not None else None
                                refreshed_subgoal_id = str(getattr(refreshed_current, 'subgoal_id', '') or '')
                                if session.status == 'succeeded':
                                    finish_retry("resolved_by_reobservation")
                                elif (refreshed_subgoal_id and refreshed_subgoal_id !=
                                    pending_corrective_retry.get('source_subgoal_id')):
                                    finish_retry('resolved_by_replan', replanned_subgoal_id=refreshed_subgoal_id)
                                elif session.status == 'awaiting_confirmation':
                                    pending_corrective_retry["status"] = "ready_for_corrective_action"
                                elif session.status == 'needs_reobservation':
                                    finish_retry('stopped_before_corrective_action', '一次新观察仍未形成唯一可执行动作。',
                                        fail_session=True)
                                else:
                                    finish_retry('stopped_before_corrective_action',
                                        session.failed_reason or f'重新观察后状态为 {session.status}。')
                        continue
                    if session.status != 'awaiting_confirmation':
                        session.auto_pause_reason = f'会话状态 {session.status} 没有可执行的安全动作。'
                        break
                    authority = session.confirmation_authority
                    reject_if(authority is None or authority.consumed, UniversalAgentOrchestratorError('安全自动推进缺少当前一次性动作作用域。'))
                    before = session.physical_actions
                    current_action_kind = action_kind()
                    is_corrective_action = bool(pending_corrective_retry is not None
                        and pending_corrective_retry.get('status') == 'ready_for_corrective_action')
                    if (is_corrective_action and (not _allows_fresh_observation_corrective_retry(impact=impact,
                        action_kind=current_action_kind))):
                        reason = "新计划不再是允许自动纠正的普通导航动作。"
                        finish_retry('stopped_before_corrective_action', reason)
                        session.auto_pause_reason = reason
                        break
                    if is_corrective_action:
                        assert pending_corrective_retry is not None
                        corrective_observation_id, corrective_fingerprint = observation_identity()
                        pending_corrective_retry.update(corrective_action_kind=current_action_kind,
                            corrective_observation_id=corrective_observation_id,
                            corrective_fingerprint=corrective_fingerprint)
                    result = self._confirm_one_locked(session, authority.scope())
                    iterations += 1
                    delta = session.physical_actions - before
                    reject_if(delta not in {0, 1}, UniversalAgentOrchestratorError('单轮安全自动推进产生了超过一个物理动作。'))
                    if getattr(result, 'action_outcome', 'matched') != 'matched':
                        if is_corrective_action:
                            assert pending_corrective_retry is not None
                            reason = '新观察重新规划后的唯一纠正动作仍未产生预期语义变化。'
                            finish_retry('exhausted', reason, fail_session=True, corrective_outcome='mismatched')
                            self._clear_action_decision(session)
                            break
                        if (not (session.status in {'needs_reobservation',
                            'awaiting_confirmation'} and _allows_fresh_observation_corrective_retry(impact=impact,
                            action_kind=current_action_kind))):
                            session.auto_pause_reason = '当前动作不属于一次新观察纠正范围，已按具体结果停止。'
                            break
                        if session.physical_actions - start_actions >= max_physical_actions:
                            reason = '动作未产生预期变化，但本次物理动作预算不足以执行一次纠正。'
                            self._set_status(session, "failed", reason)
                            session.auto_pause_reason = reason
                            break
                        session.status = "needs_reobservation"
                        self._clear_action_decision(session)
                        transition = dict(session.last_post_action_transition or {})
                        receipt = dict(transition.get("receipt") or {})
                        pending_corrective_retry = {'protocol_version': CORRECTIVE_RETRY_PROTOCOL_VERSION,
                            'correction_id': f'correction_{uuid.uuid4().hex}', 'status': 'needs_reobservation',
                            'source_receipt_id': str(receipt.get('receipt_id') or ''),
                            'source_subgoal_id': str(receipt.get('subgoal_id') or getattr(current, 'subgoal_id',
                            '') or ''), 'source_action_kind': str(receipt.get('action_kind') or current_action_kind),
                            'source_before_observation_id': str(receipt.get('before_observation_id') or ''),
                            'source_after_observation_id': str(receipt.get('after_observation_id') or ''),
                            'source_action_digest': str(receipt.get('action_digest') or ''),
                            'scheduled_after_physical_action': session.physical_actions}
                        session.corrective_retry_history.append(pending_corrective_retry)
                        session.auto_pause_reason = ""
                        continue
                    if is_corrective_action:
                        assert pending_corrective_retry is not None
                        finish_retry("matched", corrective_outcome="matched")
                    if session.physical_actions - start_actions >= max_physical_actions:
                        session.auto_pause_reason = "已达到本次安全物理动作预算。"
                        break
                if pending_corrective_retry is not None:
                    finish_retry('stopped_before_corrective_action', '自动循环迭代预算耗尽，纠正动作未执行。', fail_session=True)
                if not session.auto_pause_reason:
                    session.auto_pause_reason = {'awaiting_effect_confirmation': '下一子目标需要一次效果确认。',
                        'succeeded': '目标已由新观察和任务图修订证明完成。', 'blocked': '当前视觉或本地门禁已阻止继续。',
                        'failed': '当前执行或验证失败，禁止自动重试。'}.get(session.status, '已达到本次安全迭代预算。')
        finally:
            session.automatic_loop_enabled = False
            self._write_terminal_snapshot(session)
            self._release_if_terminal(session)
        return {'physical_actions': session.physical_actions - start_actions, 'iterations': iterations,
            'status': session.status, 'pause_reason': session.auto_pause_reason}

    def _observe_after_effect_confirmation(self, session: UniversalAgentSessionState, task_context: Mapping[str,
        Any]) -> Any:
        graph = session.task_graph
        assert graph is not None and session.goal_draft is not None
        before_actions = session.physical_actions
        session.status = "observing"
        scene, frames, frame_paths = session.adapter.capture_scene(session.goal_draft, evidence_dir=session.run_dir,
            prefix=f'before_step_{session.step_number}_frame')
        self._remember(session, frame_paths)
        observation = self._build_and_record_current_observation(session, scene=scene, frames=frames)
        decision = self._stage_current_observation_decision(session, graph=graph, frames=frames,
            task_context=task_context, trusted_observation=observation,
            unsupported_status_reason=lambda _status: '外部状态目标的完成候选必须由 DeepSeek 新 revision 复核。')
        reject_if(session.physical_actions != before_actions, UniversalAgentOrchestratorError("效果确认路径错误地产生了额外物理动作。"))
        return decision

    def start(self, *, session_id: str, raw_goal: str, exact_input_text: str | None=None,
        exact_action_kind: str | None=None, exact_target_label: str='', device_id: str,
        run_dir: Path) -> UniversalAgentSessionState:
        resolved_session = str(session_id or "").strip()
        resolved_device = str(device_id or "").strip()
        self.device_registry.reserve(resolved_device, resolved_session)
        try:
            vision_usage = VisionSessionUsageLedger(session_id=resolved_session)
            with self.device_registry.device_lock(resolved_device):
                with self._vision_usage_scope(vision_usage):
                    session = self._start_reserved(session_id=resolved_session, raw_goal=raw_goal,
                        exact_input_text=exact_input_text, exact_action_kind=exact_action_kind,
                        exact_target_label=exact_target_label, device_id=resolved_device, run_dir=run_dir,
                        vision_usage=vision_usage)
        except Exception:
            self.device_registry.release(resolved_device, resolved_session)
            raise
        self._release_if_terminal(session)
        return session

    def _start_reserved(self, *, session_id: str, raw_goal: str, exact_input_text: str | None,
        exact_action_kind: str | None, exact_target_label: str, device_id: str, run_dir: Path,
        vision_usage: VisionSessionUsageLedger | None=None) -> UniversalAgentSessionState:
        adapter = self.adapter_factory(device_id)
        store = self.evidence_store_factory(Path(run_dir))
        session = UniversalAgentSessionState(
            session_id=str(session_id or "").strip(),
            # Preserve literal payload whitespace; the typed planner validates derived input_text.
            raw_goal=str(raw_goal or "").strip(),
            device_id=str(device_id or "").strip(),
            run_dir=Path(run_dir),
            adapter=adapter,
            evidence_store=store,
            vision_usage=vision_usage,
            local_exact_input_authority=exact_input_text is not None,
        )
        reject_if(not session.session_id or not session.raw_goal or (not session.device_id), UniversalAgentOrchestratorError('启动通用 Agent 需要 session_id、目标和 device_id。'))
        try:
            session.status = "planning"
            reject_if(exact_input_text is not None and exact_action_kind is not None, UniversalAgentOrchestratorError('exact_input_text 与 exact_action_kind 不能同时使用。'))
            graph = build_exact_input_task_graph(session.raw_goal, exact_input_text=exact_input_text,
                device_id=session.device_id) if exact_input_text is not None else build_exact_action_task_graph(
                session.raw_goal, action_kind=exact_action_kind, target_label=exact_target_label,
                device_id=session.device_id) if exact_action_kind is not None else self.deepseek_planner.plan(
                session.raw_goal, device_id=session.device_id)
            self._validate_graph_identity(graph, device_id=session.device_id)
            session.task_graph = graph
            session.goal_draft = self.bridge.goal_draft(graph)
            self._remember(session, store.write_task_graph(graph), store.write_effect_policy_snapshot(graph))

            current = graph.active_subgoal()
            impact = current.external_impact if current is not None else "unknown"
            if current is None:
                return self._finish_session(session, 'blocked', '任务图没有活动子目标。')
            session.status = "observing"
            scene, frames, frame_paths = adapter.capture_scene(session.goal_draft, evidence_dir=session.run_dir,
                prefix=f'before_step_{session.step_number}_frame')
            self._remember(session, frame_paths)
            observation = self._build_and_record_current_observation(session, scene=scene, frames=frames)

            if impact == 'unknown':
                revised, _ = self._replan_current_observation(session, graph, observation,
                    trigger='observation_changed', store=True,
                    reason='初始只读观察已经可用；请仅依据当前结构化画面事实重新分类 unknown 子目标。不能因此宣称动作已执行。')
                graph = revised
                if graph.status == 'completed':
                    return self._finish_session(session, 'succeeded')
                current = graph.active_subgoal()
                if current is None:
                    return self._finish_session(session, 'blocked', 'unknown 子目标重分类后没有活动子目标。')
                impact = current.external_impact if current is not None else "unknown"

            if _requires_effect_confirmation(graph, current):
                self._set_status(session, 'awaiting_effect_confirmation')
                session.confirmed_effect_ids = ()
                session.confirmation_authority = None
                self._bind_effect_confirmation(session)
                self._write_terminal_snapshot(session)
                return session

            if impact in {'read_only', 'navigation_only'}:
                initial_safe = graph.active_subgoal()
                revised = self._try_advance_visible_subgoal(session, graph=graph,
                    trusted_observation=observation)
                visible_advanced = revised is not None
                if visible_advanced:
                    self._store_revised_graph(session, revised)
                    graph = revised
                if (impact == 'read_only' and (not visible_advanced) and (initial_safe is not None)
                    and (not VisibleGoalEvidence.presence_only(initial_safe))):
                    revised, _ = self._replan_current_observation(session, graph, observation,
                        trigger='observation_changed', store=True,
                        reason='初始可信画面不能直接证明当前 read_only 结果。如果目标页面或区域尚未出现，必须先修订为一个 navigation_only 中间状态；不得请求低层动作、不得直接宣称结果完成。')
                    graph = revised
                    visible_advanced = True
                if not visible_advanced and impact == 'read_only':
                    return self._finish_session(session, 'blocked',
                        '当前 read_only 子目标不是可由唯一完整可见元素证明的定位目标，或当前画面证据不唯一；未请求物理动作。')
                if visible_advanced:
                    if revised.status == 'completed':
                        return self._finish_session(session, 'succeeded')
                    current = revised.active_subgoal()
                    impact = current.external_impact if current is not None else 'unknown'
                    if current is None:
                        return self._finish_session(session, 'blocked', '可见状态证据推进后没有活动子目标。')
                    if _requires_effect_confirmation(revised, current):
                        self._set_status(session, 'awaiting_effect_confirmation')
                        self._bind_effect_confirmation(session)
                        self._write_terminal_snapshot(session)
                        return session
                    self._set_status(session, 'needs_reobservation',
                        '可见状态证据已切换活动子目标；必须按新子目标重新观察，不得复用旧目标条件下的候选清单。')
                    session.controller_decision = None
                    session.confirmation_authority = None
                    self._write_terminal_snapshot(session)
                    return session

            task_context_payload = graph.to_qwen_context()
            task_context = QwenTaskContext.from_dict(task_context_payload)
            try:
                semantic_authority = compile_formal_semantic_authority(graph)
                semantic_ir = semantic_authority.semantic_ir
            except TaskSemanticIRError as exc:
                return self._finish_session(session, 'blocked', f'正式 TaskSemanticIR authority 拒绝：{exc}')
            task_context = replace(task_context, semantic_ir=semantic_ir)
            task_context.validate()
            session.effect_previews = tuple(({**preview, 'preview_digest': effect_preview_digest(preview)}
                for preview in semantic_authority.effect_previews))
            self._stage_current_observation_decision(session, graph=graph, frames=frames, task_context=task_context,
                trusted_observation=observation, stage_capability_block=False)

            reject_if(session.physical_actions != 0, UniversalAgentOrchestratorError('start 路径错误地触发了物理动作。'))
            self._write_terminal_snapshot(session)
            return session
        except Exception as exc:
            self._set_status(session, 'failed', str(exc))
            self._record_deepseek_failure(session, exc, stage='initial_task_graph' if session.task_graph
                is None else 'start_replan')
            self._best_effort_terminal_snapshot(session)
            raise

    def _terminate(self, session: UniversalAgentSessionState, *, status: str, message: str) -> None:
        try:
            with self.device_registry.device_lock(session.device_id):
                self._invalidate_authorities(session, status)
                self._set_status(session, status, message)
                self._write_terminal_snapshot(session)
        finally:
            self.device_registry.release(session.device_id, session.session_id)

    def pause(self, session: UniversalAgentSessionState) -> None:
        self._terminate(session, status='paused', message='用户已暂停；旧确认和旧观察不可复用。')

    def invalidate_confirmation(self, session: UniversalAgentSessionState, *, reason: str) -> None:
        """Invalidate a pending scope while retaining the device for re-observation."""

        if self.device_registry.active_session(session.device_id) != session.session_id:
            return
        with self.device_registry.device_lock(session.device_id):
            self._invalidate_authorities(session, str(reason or 'invalidated'), clear=True)
            self._set_status(session, 'needs_reobservation', '设备或摄像头状态变化；旧确认已失效，必须重新观察后再确认。')
            self._write_terminal_snapshot(session)

    def cancel(self, session: UniversalAgentSessionState) -> None:
        self._terminate(session, status='cancelled', message='用户已取消任务。')
