"""The production screenshot -> Qwen decision -> one action -> screenshot loop."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any, Callable
import uuid

from agent.domain.validation import canonical_digest, reject_if
from agent.application.action_adapter import GenericActionAdapterError, GenericSingleActionAdapterPort
from agent.application.runtime_session import POST_ACTION_TRANSITION_PROTOCOL_VERSION, UniversalAgentSessionState
from agent.application.vision_usage import VisionSessionUsageLedger
from agent.domain import (
    AgentEvidenceStoreFactory,
    CANONICAL_SELECTION_RECEIPT_VERSION,
    CanonicalSelectionReceipt,
    ConfirmationAuthority,
    DeviceTaskRegistryPort,
    EffectConfirmationAuthority,
    EvidenceStoreError,
)
from agent.domain.canonical_action_kinds import CANONICAL_ACTION_KINDS
from agent.domain.generic_goal import GenericIntentDraft
from agent.domain.qwen_task_context import QwenTaskContext
from agent.domain.task_graph import (
    DynamicTaskGraph,
    TaskGraphError,
    build_exact_action_task_graph,
    build_exact_input_task_graph,
    complete_active_subgoal,
)
from agent.domain.task_semantic_ir import TaskSemanticIRError, compile_formal_semantic_authority, effect_preview_digest


POST_ACTION_OUTCOMES = frozenset({'matched', 'mismatched'})
CORRECTIVE_RETRY_IMPACTS = frozenset({'read_only', 'navigation_only'})
CORRECTIVE_RETRY_ACTIONS = frozenset({'back', 'dismiss_overlay', 'double_tap', 'drag', 'home',
    'open_recent_apps', 'long_press', 'swipe', 'tap_semantic'})


class UniversalAgentOrchestratorError(RuntimeError):
    pass


def _action_digest(action: Any) -> str:
    reject_if(action is None, UniversalAgentOrchestratorError("动作摘要缺少语义动作。"))
    payload = action.to_dict() if callable(getattr(action, 'to_dict', None)) else action
    return canonical_digest(payload)


def _confirmation_effect_ids(graph: DynamicTaskGraph, current: Any | None) -> tuple[str, ...]:
    if current is None:
        return ()
    active_ids = set(tuple(getattr(current, 'risk_action_ids', ()) or ()))
    return tuple(sorted(risk.risk_id for risk in graph.risk_actions
        if risk.risk_id in active_ids and risk.confirmation_required))


def _requires_effect_confirmation(graph: DynamicTaskGraph, current: Any | None) -> bool:
    return bool(_confirmation_effect_ids(graph, current))


def _effect_confirmation_material(graph: DynamicTaskGraph, current: Any) -> tuple[str, dict[str, Any]]:
    effect_ids = _confirmation_effect_ids(graph, current)
    serialized = {item['effect_id']: item for item in graph.to_dict()['effect_intents']}
    selected = [serialized[item] for item in effect_ids]
    preview = {'kind': 'typed_effects', 'effect_ids': list(effect_ids),
        'effects': [{'effect_id': item['effect_id'], 'kind': item['kind'],
        'expected_results': list(item['expected_results']),
        'policy_level': item['local_policy']['policy_level']} for item in selected]}
    payload = {'protocol_version': '2026-08-20-typed-effect-confirmation-v1', 'task_id': graph.task_id,
        'device_id': graph.device_id, 'revision': graph.revision, 'subgoal_id': current.subgoal_id,
        'effect_ids': list(effect_ids), 'effect_intents': selected, 'preview': preview}
    return canonical_digest(payload), preview


class ObservationBridge:
    """Project the current local plan item into the one Qwen observation prompt."""

    _APP_REFERENCE_IDS = frozenset({'unknown', 'current_foreground', 'current_app', 'foreground_app',
        'target_app', 'active_app'})
    _SURFACE_IDENTITIES = {'device': ('device', '设备界面'), 'system': ('system', '系统界面'),
        'current_surface': ('current_surface', '当前界面')}

    @classmethod
    def _active_app_entry_target_label(cls, graph: DynamicTaskGraph, active: Any) -> str:
        if active is None or active.external_impact != 'navigation_only':
            return ''
        objective = str(active.objective or '').strip()
        eligible = [app for app in graph.goal.target_apps if str(app.app_name or '').strip()
            and str(app.app_id or '').strip().casefold() not in cls._APP_REFERENCE_IDS]
        mentioned = [app for app in eligible if app.app_name.casefold() in objective.casefold()]
        if len(mentioned) != 1:
            return ''
        app_name = mentioned[0].app_name.strip()
        literal = re.escape(app_name)
        patterns = (rf'(?:打开|进入|启动|切换到|切至|前往)\s*(?:应用|app)?\s*{literal}',
            rf'{literal}\s*(?:应用)?\s*(?:已打开|已启动|主界面可见|首页可见)',
            rf'(?<![a-z0-9_])(?:open|launch|enter|go\s+to|switch\s+to)\s+(?:the\s+)?'
            rf'(?:app\s+)?{literal}(?![a-z0-9_])')
        return app_name if any(re.search(item, objective, flags=re.IGNORECASE) for item in patterns) else ''

    @staticmethod
    def _typed_input_views(graph: DynamicTaskGraph, active: Any) -> tuple[dict[str, Any], dict[str, Any],
        dict[str, Any]]:
        try:
            ir = compile_formal_semantic_authority(graph).semantic_ir
        except TaskSemanticIRError:
            return {}, {}, {}
        typed = next((item for item in ir.subgoals if item.subgoal_id == getattr(active, 'subgoal_id', '')), None)
        if typed is None:
            return {}, {}, {}
        entities = {item.entity_id: item for item in ir.entities}
        fields = tuple(item for item in ir.input_fields if typed.subgoal_id in item.source_subgoal_ids)
        constraints = {item.constraint_id: item for item in ir.constraints}
        actions = {str(constraints[ref].value) for ref in typed.constraint_refs
            if ref in constraints and constraints[ref].kind == 'required_action'}
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
            preceding = tuple(item for item in ir.input_fields if set(item.source_subgoal_ids).intersection(
                typed.depends_on))
            if len(preceding) == 1:
                field = preceding[0]
                payload = entities.get(field.payload_ref)
                if payload is not None and payload.role == 'input_text' and payload.value:
                    predecessor = {'field_id': field.field_id, 'field_label': field.field_label,
                        'text': payload.value}
        verification: dict[str, Any] = {}
        if getattr(active, 'external_impact', '') == 'read_only' and not active_input:
            desired = {item.state_id: item for item in ir.desired_states}
            by_payload = {item.payload_ref: item for item in ir.input_fields}
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
    def _subgoal_visual_context(cls, graph: DynamicTaskGraph, subgoal: Any) -> dict[str, Any]:
        entities = {key: value for key, value in graph.goal.entities.items() if key != 'input_fields'}
        for key in ('active_input_transaction_text', 'active_input_field_id', 'active_input_field_label',
            'active_input_multiline'):
            entities.pop(key, None)
        app_label = cls._active_app_entry_target_label(graph, subgoal)
        if app_label:
            entities['target_ui_label'] = app_label
        active_input, predecessor, verification = cls._typed_input_views(graph, subgoal)
        text = active_input.get('text')
        target_only = active_input.get('target_only') is True
        if isinstance(text, str) and (text or target_only):
            if text:
                entities['active_input_transaction_text'] = text
            entities['active_input_field_id'] = active_input['field_id']
            if target_only:
                entities['active_input_target_only'] = True
            if active_input.get('field_label'):
                entities['active_input_field_label'] = active_input['field_label']
            entities['active_input_multiline'] = bool(active_input.get('multiline'))
            if predecessor:
                entities.update({'active_input_predecessor_field_id': predecessor['field_id'],
                    'active_input_predecessor_field_label': predecessor['field_label'],
                    'active_input_predecessor_text': predecessor['text']})
        else:
            entities.pop('input_text', None)
            entities.update(verification)
        return {'subgoal_id': subgoal.subgoal_id, 'objective': subgoal.objective,
            'constraints': list(subgoal.constraints), 'completion_conditions': list(subgoal.completion_conditions),
            'execution_class': {'read_only': 'observe', 'navigation_only': 'navigate',
            'external_state': 'effect', 'unknown': 'unknown'}.get(subgoal.external_impact, 'unknown'),
            'goal_entities': entities}

    def goal_draft(self, graph: DynamicTaskGraph) -> GenericIntentDraft:
        graph.validate()
        target_surface = str(graph.goal.entities.get('target_surface') or '').strip()
        if graph.goal.target_apps:
            app_id, app_name = graph.goal.target_apps[0].app_id, graph.goal.target_apps[0].app_name
        elif target_surface in self._SURFACE_IDENTITIES:
            app_id, app_name = self._SURFACE_IDENTITIES[target_surface]
        else:
            raise UniversalAgentOrchestratorError('计划没有目标App或目标surface。')
        active = graph.active_subgoal()
        constraints = list(graph.constraints)
        if active is not None:
            constraints.extend(active.constraints)
        entities = dict(graph.goal.entities)
        if graph.raw_user_goal.strip():
            entities['original_goal_visual_context'] = graph.raw_user_goal.strip()
        if active is not None:
            entities['active_subgoal_visual_context'] = self._subgoal_visual_context(graph, active)
        entities['target_apps'] = [{'app_id': item.app_id, 'app_name': item.app_name}
            for item in graph.goal.target_apps]
        criteria = {item.condition_id: {'description': item.description,
            'evidence_required': list(item.evidence_required), 'satisfied': item.satisfied}
            for item in graph.completion_conditions}
        effects = tuple(dict.fromkeys(item.effect_kind for item in graph.risk_actions if item.effect_kind))
        draft = GenericIntentDraft(understood=True, app_id=app_id, app_name=app_name,
            objective=graph.goal.objective, entities=entities, constraints=tuple(dict.fromkeys(constraints)),
            success_criteria=criteria, account_effects=effects, needs_confirmation=False)
        draft.validate()
        return draft


class UniversalAgentOrchestrator:
    """Run one authoritative Qwen action/finish decision at a time."""

    def __init__(self, *, deepseek_planner: Any, qwen_observer: Any,
        adapter_factory: Callable[[str], GenericSingleActionAdapterPort],
        evidence_store_factory: AgentEvidenceStoreFactory, trusted_observation_factory: Callable[..., Any],
        bridge: ObservationBridge | None=None, device_registry: DeviceTaskRegistryPort,
        deepseek_failure_diagnostic_writer: Callable[..., tuple[str, ...]] | None=None) -> None:
        self.deepseek_planner = deepseek_planner
        self.qwen_observer = qwen_observer
        self.adapter_factory = adapter_factory
        self.evidence_store_factory = evidence_store_factory
        self.trusted_observation_factory = trusted_observation_factory
        self.bridge = bridge or ObservationBridge()
        self.device_registry = device_registry
        self.deepseek_failure_diagnostic_writer = deepseek_failure_diagnostic_writer or (lambda *_args,
            **_kwargs: ())

    def _vision_usage_scope(self, ledger: VisionSessionUsageLedger | None):
        provider = getattr(self.qwen_observer, 'provider', None)
        factory = getattr(provider, 'session_usage_scope', None)
        return factory(ledger) if callable(factory) else nullcontext()

    def _release_if_terminal(self, session: UniversalAgentSessionState) -> None:
        if session.status in self.device_registry.TERMINAL_STATUSES:
            self.device_registry.release(session.device_id, session.session_id)

    @staticmethod
    def _set_status(session: UniversalAgentSessionState, status: str, reason: str='') -> None:
        session.status = status
        session.failed_reason = reason

    @staticmethod
    def _remember(session: UniversalAgentSessionState, *paths: Any) -> None:
        for path in paths:
            values = path if isinstance(path, (list, tuple)) else (path,)
            for item in values:
                text = str(item or '').strip()
                if text and text not in session.evidence_paths:
                    session.evidence_paths.append(text)

    @staticmethod
    def _clear_action(session: UniversalAgentSessionState) -> None:
        if session.confirmation_authority is not None:
            session.confirmation_authority.consumed = True
            session.confirmation_authority.invalid_reason = 'reobserve'
        session.qwen_decision = None
        session.controller_decision = None
        session.confirmation_authority = None

    @staticmethod
    def _available_action_kinds(session: UniversalAgentSessionState) -> frozenset[str]:
        provider = getattr(session.adapter, 'supported_action_kinds', None)
        actions = CANONICAL_ACTION_KINDS if not callable(provider) else frozenset(str(item or '').strip()
            for item in provider())
        reject_if(not actions or '' in actions or actions - CANONICAL_ACTION_KINDS,
            UniversalAgentOrchestratorError('设备canonical动作能力无效。'))
        return actions

    def _record_deepseek_failure(self, session: UniversalAgentSessionState, error: Exception) -> None:
        if not isinstance(error, TaskGraphError):
            return
        try:
            self._remember(session, self.deepseek_failure_diagnostic_writer(self.deepseek_planner,
                evidence_dir=session.run_dir, prefix='deepseek_initial', failed_stage='initial_plan', error=error,
                previous_graph=None))
        except Exception:
            pass

    def _write_snapshot(self, session: UniversalAgentSessionState) -> None:
        if session.vision_usage is not None:
            self._remember(session, session.evidence_store.write_json('qwen_usage.json',
                session.vision_usage.to_dict()))
        self._remember(session, session.evidence_store.write_session(session))
        self._remember(session, session.evidence_store.write_report({'mode': 'qwen_same_response_single_loop',
            'policy_version': CANONICAL_SELECTION_RECEIPT_VERSION, 'session': session.snapshot()}))

    def _best_effort_snapshot(self, session: UniversalAgentSessionState) -> None:
        try:
            self._write_snapshot(session)
        except Exception:
            pass

    def _build_observation(self, session: UniversalAgentSessionState, *, scene: Any, frames: list[Any] | tuple[Any,
        ...], observation_id: str | None=None) -> Any:
        values: dict[str, Any] = {'frames': list(frames), 'device_id': session.device_id, 'scene': scene}
        if observation_id is not None:
            values['observation_id'] = observation_id
        observation = self.trusted_observation_factory(**values)
        session.trusted_observation = observation
        session.trusted_frames = tuple(frames)
        self._remember(session, session.evidence_store.write_trusted_observation(session.step_number, observation))
        return observation

    def _task_context(self, session: UniversalAgentSessionState) -> QwenTaskContext:
        graph = session.task_graph
        reject_if(graph is None, UniversalAgentOrchestratorError('会话缺少当前计划。'))
        kwargs: dict[str, Any] = {}
        if session.confirmed_effect_ids:
            kwargs = {'confirmed_effect_ids': session.confirmed_effect_ids, 'confirmed_task_id': graph.task_id,
                'confirmed_device_id': graph.device_id, 'confirmed_subgoal_id': graph.active_subgoal_id,
                'confirmed_revision': graph.revision}
        context = QwenTaskContext.from_dict(graph.to_qwen_context(**kwargs))
        try:
            authority = compile_formal_semantic_authority(graph)
        except TaskSemanticIRError as exc:
            raise UniversalAgentOrchestratorError(f'正式TaskSemanticIR拒绝：{exc}') from exc
        session.effect_previews = tuple({**item, 'preview_digest': effect_preview_digest(item)}
            for item in authority.effect_previews)
        context = replace(context, semantic_ir=authority.semantic_ir)
        context.validate()
        session.semantic_task_context = context
        return context

    def _decide(self, session: UniversalAgentSessionState, *, frames: list[Any] | tuple[Any, ...],
        observation: Any) -> Any:
        context = self._task_context(session)
        available = self._available_action_kinds(session)
        ir = context.semantic_ir
        assert ir is not None
        active_id = str(context.current_subgoal.get('subgoal_id') or '')
        typed_subgoal = next((item for item in ir.subgoals if item.subgoal_id == active_id), None)
        surface = next((item for item in ir.surfaces if typed_subgoal is not None
            and item.surface_id == typed_subgoal.surface_ref and item.kind == 'app'), None)
        resolver = getattr(session.adapter, 'resolve_app_launch_target', None)
        launch = resolver(surface.app_id, surface.app_name) if surface is not None and callable(resolver) else None
        launch_parameters = None
        if launch is None:
            available = available - {'launch_app'}
        else:
            launch_parameters = {'launch_ref': str(launch.launch_ref),
                'expected_app_id': str(launch.expected_app_id)}
        args: dict[str, Any] = {'frames': list(frames), 'task_context': context,
            'trusted_observation': observation, 'decision_number': session.step_number,
            'available_action_kinds': available}
        if launch_parameters:
            args['launch_target'] = launch_parameters
        profile_provider = getattr(session.adapter, 'text_transport_profile', None)
        profile = profile_provider() if callable(profile_provider) else None
        if profile is not None:
            profile.validate()
            reject_if(profile.device_id != context.device_id,
                UniversalAgentOrchestratorError('Companion IME profile与当前设备不一致。'))
            args['text_transport_profile'] = profile
        return self.qwen_observer.decide(**args)

    @staticmethod
    def _validate_decision_binding(graph: DynamicTaskGraph, observation: Any, decision: Any) -> None:
        expected = (graph.task_id, graph.device_id, graph.revision, observation.observation_id,
            observation.fingerprint)
        actual = (decision.task_id, decision.device_id, decision.revision, decision.observation_id,
            decision.fingerprint)
        reject_if(actual != expected or decision.trusted_observation is not observation,
            UniversalAgentOrchestratorError('Qwen决策没有绑定当前task/device/revision/observation。'))
        decision.proposal.validate(observation.scene)

    @classmethod
    def _selection_receipt(cls, session: UniversalAgentSessionState, decision: Any) -> CanonicalSelectionReceipt:
        action = getattr(getattr(decision, 'proposal', None), 'action', None)
        kind = str(getattr(action, 'action', '') or '')
        if kind not in cls._available_action_kinds(session):
            return CanonicalSelectionReceipt(allowed=False, reason=f'当前设备不支持canonical动作：{kind or "missing"}。')
        return CanonicalSelectionReceipt(allowed=True,
            reason='Qwen同响应动作已精确绑定当前canonical candidate。', canonical_class=kind)

    def _complete_from_finish(self, session: UniversalAgentSessionState, decision: Any) -> None:
        graph = session.task_graph
        assert graph is not None
        revised = complete_active_subgoal(graph, evidence=tuple(decision.completion_evidence))
        session.task_graph = revised
        session.goal_draft = self.bridge.goal_draft(revised) if revised.status != 'completed' else session.goal_draft
        session.confirmed_effect_ids = ()
        session.effect_confirmation_authority = None
        session.controller_decision = None
        session.confirmation_authority = None
        self._remember(session, session.evidence_store.write_task_graph(revised),
            session.evidence_store.write_effect_policy_snapshot(revised))
        if revised.status == 'completed':
            self._set_status(session, 'succeeded')
            return
        current = revised.active_subgoal()
        if _requires_effect_confirmation(revised, current):
            self._set_status(session, 'awaiting_effect_confirmation')
            self._bind_effect_confirmation(session)
        else:
            self._set_status(session, 'needs_reobservation',
                'Qwen已完成当前高层目标；下一目标必须取得新截图。')

    def _stage_decision(self, session: UniversalAgentSessionState, *, decision: Any,
        allow_action: bool=True, action_block_reason: str='') -> Any:
        graph = session.task_graph
        observation = session.trusted_observation
        assert graph is not None and observation is not None
        self._validate_decision_binding(graph, observation, decision)
        session.qwen_decision = decision
        self._remember(session, session.evidence_store.write_qwen_decision(session.step_number, decision))
        if decision.proposal.status == 'finish':
            self._complete_from_finish(session, decision)
            return decision
        if decision.proposal.status == 'blocked' or not allow_action:
            reason = decision.proposal.reason if decision.proposal.status == 'blocked' else action_block_reason
            session.controller_decision = CanonicalSelectionReceipt(allowed=False, reason=reason)
            session.confirmation_authority = None
            self._set_status(session, 'blocked', reason)
            return decision
        receipt = self._selection_receipt(session, decision)
        session.controller_decision = receipt
        self._remember(session, session.evidence_store.write_controller_decision(session.step_number,
            receipt.to_dict()))
        if not receipt.allowed:
            self._set_status(session, 'blocked', receipt.reason)
            return decision
        self._set_status(session, 'awaiting_confirmation')
        self._bind_confirmation(session)
        return decision

    def _observe_and_decide(self, session: UniversalAgentSessionState) -> Any:
        graph = session.task_graph
        reject_if(graph is None or graph.active_subgoal() is None,
            UniversalAgentOrchestratorError('当前计划没有可观察高层目标。'))
        current = graph.active_subgoal()
        if _requires_effect_confirmation(graph, current) and not session.confirmed_effect_ids:
            self._set_status(session, 'awaiting_effect_confirmation')
            self._bind_effect_confirmation(session)
            self._write_snapshot(session)
            return SimpleNamespace(proposal=SimpleNamespace(status='blocked',
                reason='登录或付款目标等待用户确认。'))
        self._clear_action(session)
        session.goal_draft = self.bridge.goal_draft(graph)
        self._set_status(session, 'observing')
        scene, frames, paths = session.adapter.capture_scene(session.goal_draft,
            evidence_dir=session.run_dir, prefix=f'before_step_{session.step_number}_frame')
        self._remember(session, paths)
        observation = self._build_observation(session, scene=scene, frames=frames)
        decision = self._decide(session, frames=frames, observation=observation)
        self._stage_decision(session, decision=decision)
        self._write_snapshot(session)
        return decision

    def _current_confirmation_scope(self, session: UniversalAgentSessionState) -> dict[str, Any]:
        graph, observation, decision = session.task_graph, session.trusted_observation, session.qwen_decision
        reject_if(graph is None or observation is None or decision is None
            or decision.proposal.status != 'action', UniversalAgentOrchestratorError('当前没有可执行动作scope。'))
        self._validate_decision_binding(graph, observation, decision)
        current = graph.active_subgoal()
        assert current is not None
        return {'session_id': session.session_id, 'task_id': graph.task_id, 'device_id': graph.device_id,
            'revision': graph.revision, 'subgoal_id': current.subgoal_id,
            'effect_ids': sorted(current.risk_action_ids), 'observation_id': observation.observation_id,
            'fingerprint': observation.fingerprint, 'decision_node_id': decision.proposal.action.node_id,
            'action_digest': _action_digest(decision.proposal.action)}

    def _bind_confirmation(self, session: UniversalAgentSessionState) -> None:
        scope = self._current_confirmation_scope(session)
        session.confirmation_authority = ConfirmationAuthority(session_id=scope['session_id'],
            task_id=scope['task_id'], device_id=scope['device_id'], revision=scope['revision'],
            subgoal_id=scope['subgoal_id'], effect_ids=tuple(scope['effect_ids']),
            observation_id=scope['observation_id'], fingerprint=scope['fingerprint'],
            decision_node_id=scope['decision_node_id'], action_digest=scope['action_digest'])

    @staticmethod
    def _normalize_scope(value: Mapping[str, Any], *, required: set[str], digest_key: str,
        label: str) -> dict[str, Any]:
        reject_if(not isinstance(value, Mapping) or set(value) != required,
            UniversalAgentOrchestratorError(f'{label}字段缺失或包含额外字段。'))
        effect_ids = value.get('effect_ids')
        revision = value.get('revision')
        digest = str(value.get(digest_key) or '')
        reject_if(not isinstance(effect_ids, list) or isinstance(revision, bool) or not isinstance(revision, int)
            or not re.fullmatch('[0-9a-f]{64}', digest), UniversalAgentOrchestratorError(f'{label}格式无效。'))
        return {'session_id': str(value.get('session_id') or ''), 'task_id': str(value.get('task_id') or ''),
            'device_id': str(value.get('device_id') or ''), 'revision': revision,
            'subgoal_id': str(value.get('subgoal_id') or ''),
            'effect_ids': sorted(str(item) for item in effect_ids), digest_key: digest}

    @classmethod
    def _normalize_confirmation(cls, value: Mapping[str, Any]) -> dict[str, Any]:
        required = {'session_id', 'task_id', 'device_id', 'revision', 'subgoal_id', 'effect_ids',
            'observation_id', 'fingerprint', 'decision_node_id', 'action_digest'}
        result = cls._normalize_scope(value, required=required, digest_key='action_digest', label='动作scope')
        result.update({'observation_id': str(value.get('observation_id') or ''),
            'fingerprint': str(value.get('fingerprint') or ''),
            'decision_node_id': str(value.get('decision_node_id') or '')})
        reject_if(not all(result[key] for key in ('observation_id', 'fingerprint', 'decision_node_id')),
            UniversalAgentOrchestratorError('动作scope缺少观察或决策标识。'))
        return result

    def _consume_confirmation(self, session: UniversalAgentSessionState, value: Mapping[str, Any]) -> ConfirmationAuthority:
        authority = session.confirmation_authority
        reject_if(session.status != 'awaiting_confirmation' or authority is None or authority.consumed,
            UniversalAgentOrchestratorError(f'当前状态不能执行动作：{session.status}。'))
        current = self._current_confirmation_scope(session)
        requested = self._normalize_confirmation(value)
        if current != authority.scope() or requested != current:
            authority.consumed = True
            authority.invalid_reason = 'scope_mismatch'
            raise UniversalAgentOrchestratorError('动作scope与当前截图或canonical动作不一致。')
        authority.consumed = True
        authority.invalid_reason = 'consumed_before_execution'
        return authority

    def _bind_effect_confirmation(self, session: UniversalAgentSessionState) -> None:
        graph = session.task_graph
        assert graph is not None
        current = graph.active_subgoal()
        effect_ids = _confirmation_effect_ids(graph, current)
        reject_if(current is None or not effect_ids,
            UniversalAgentOrchestratorError('当前目标没有登录或付款确认项。'))
        digest, preview = _effect_confirmation_material(graph, current)
        session.effect_confirmation_authority = EffectConfirmationAuthority(session_id=session.session_id,
            task_id=graph.task_id, device_id=graph.device_id, revision=graph.revision,
            subgoal_id=current.subgoal_id, effect_ids=effect_ids, intent_digest=digest,
            intent_preview=preview)

    @classmethod
    def _normalize_effect_confirmation(cls, value: Mapping[str, Any]) -> dict[str, Any]:
        required = {'session_id', 'task_id', 'device_id', 'revision', 'subgoal_id', 'effect_ids', 'intent_digest'}
        return cls._normalize_scope(value, required=required, digest_key='intent_digest', label='效果确认scope')

    def _may_correct_after_mismatch(self, session: UniversalAgentSessionState, *, impact: str,
        action_kind: str) -> bool:
        if impact not in CORRECTIVE_RETRY_IMPACTS or action_kind not in CORRECTIVE_RETRY_ACTIONS:
            return False
        graph = session.task_graph
        assert graph is not None
        current = graph.active_subgoal()
        key = (graph.revision, current.subgoal_id if current else '')
        return not any((item.get('revision'), item.get('subgoal_id')) == key
            for item in session.corrective_retry_history)

    def _confirm_one_locked(self, session: UniversalAgentSessionState, confirmation: Mapping[str, Any]) -> Any:
        authority = self._consume_confirmation(session, confirmation)
        graph, observation, decision = session.task_graph, session.trusted_observation, session.qwen_decision
        assert graph is not None and observation is not None and decision is not None
        receipt = session.controller_decision
        reject_if(receipt is None or not receipt.allowed,
            UniversalAgentOrchestratorError('动作缺少canonical映射回执。'))
        session.confirm_stage = 'executing'
        self._set_status(session, 'executing_one_action')
        before_actions = session.physical_actions
        try:
            result = session.adapter.execute(requested_action=decision.proposal.action,
                planned_scene=observation.scene, goal=session.goal_draft, confirmed=True,
                evidence_dir=session.run_dir, planned_frames=session.trusted_frames,
                action_authority=authority)
        except GenericActionAdapterError as exc:
            session.physical_actions += max(0, int(exc.physical_actions))
            self._remember(session, exc.evidence)
            self._set_status(session, 'needs_reobservation' if exc.physical_actions == 0 else 'failed', str(exc))
            self._best_effort_snapshot(session)
            raise

        self._remember(session, result.evidence, result.after_frame_paths)
        physical = int(result.physical_actions)
        wait = result.resolved_action.kind == 'wait_for_change'
        reject_if(physical != 1 and not (wait and physical == 0),
            UniversalAgentOrchestratorError(f'一次动作返回了无效物理动作数：{physical}。'))
        session.physical_actions += physical
        reject_if(_action_digest(result.requested_action) != authority.action_digest
            or result.requested_action.node_id != authority.decision_node_id
            or result.rebound_action.node_id != authority.decision_node_id
            or result.resolved_action.node_id != authority.decision_node_id
            or result.resolved_action.kind != result.rebound_action.action,
            UniversalAgentOrchestratorError('执行结果没有绑定已消费的canonical动作。'))
        outcome = str(result.action_outcome or '')
        errors = tuple(str(item) for item in result.verification_errors if str(item).strip())
        reject_if(outcome not in POST_ACTION_OUTCOMES or ((outcome == 'matched') == bool(errors)),
            UniversalAgentOrchestratorError('动作outcome与verification_errors不一致。'))
        after_frames = tuple(result.after_frames)
        after_paths = tuple(str(item).strip() for item in result.after_frame_paths)
        reject_if(len(after_frames) < 4 or len(after_paths) != len(after_frames) or any(not item for item in after_paths),
            UniversalAgentOrchestratorError('动作后缺少完整四帧新观察。'))

        session.step_number += 1
        new_observation = self._build_observation(session, scene=result.after_scene, frames=after_frames,
            observation_id=f'obs_{uuid.uuid4().hex}')
        session.history.append({'step_number': session.step_number - 1, 'task_revision': graph.revision,
            'qwen_decision': UniversalAgentSessionState._serialize(decision), 'execution': result.to_dict(),
            'before_observation_id': observation.observation_id, 'before_fingerprint': observation.fingerprint,
            'after_observation_id': new_observation.observation_id,
            'after_fingerprint': new_observation.fingerprint})
        self._remember(session, session.evidence_store.write_verification(session.step_number - 1,
            {'outcome': outcome, 'verification_errors': list(errors),
            'after_observation_id': new_observation.observation_id,
            'after_fingerprint': new_observation.fingerprint}))

        current = graph.active_subgoal()
        impact = current.external_impact if current is not None else 'unknown'
        allow_action = outcome == 'matched'
        block_reason = ''
        if outcome == 'mismatched':
            allow_action = self._may_correct_after_mismatch(session, impact=impact,
                action_kind=decision.proposal.action.action)
            if allow_action:
                session.corrective_retry_history.append({'revision': graph.revision,
                    'subgoal_id': current.subgoal_id if current else '',
                    'after_observation_id': new_observation.observation_id})
            else:
                block_reason = ('动作后新截图未证明预期结果；输入、效果或第二次导航失败均不自动重试。')

        next_decision = self._decide(session, frames=after_frames, observation=new_observation)
        self._stage_decision(session, decision=next_decision, allow_action=allow_action,
            action_block_reason=block_reason)
        transition = {'protocol_version': POST_ACTION_TRANSITION_PROTOCOL_VERSION,
            'transition_kind': 'new_screenshot_decision', 'outcome': outcome,
            'physical_actions_before': before_actions, 'physical_actions': session.physical_actions,
            'next_decision_status': next_decision.proposal.status,
            'after_observation_id': new_observation.observation_id,
            'after_fingerprint': new_observation.fingerprint}
        session.last_post_action_transition = transition
        self._remember(session, session.evidence_store.write_post_action_transition(session.step_number - 1,
            transition))
        session.confirm_stage = 'completed'
        self._write_snapshot(session)
        return result

    def confirm_one(self, session: UniversalAgentSessionState, confirmation: Mapping[str, Any]) -> Any:
        reject_if(self.device_registry.active_session(session.device_id) != session.session_id,
            UniversalAgentOrchestratorError('当前会话不再拥有设备。'))
        try:
            with self._vision_usage_scope(session.vision_usage), self.device_registry.device_lock(session.device_id):
                return self._confirm_one_locked(session, confirmation)
        except Exception as exc:
            if session.status not in {'needs_reobservation', 'failed'}:
                self._set_status(session, 'failed', str(exc))
                self._best_effort_snapshot(session)
            raise
        finally:
            self._release_if_terminal(session)

    def approve_effects(self, session: UniversalAgentSessionState, confirmation: Mapping[str, Any]) -> Any:
        reject_if(self.device_registry.active_session(session.device_id) != session.session_id,
            UniversalAgentOrchestratorError('当前会话不再拥有设备。'))
        try:
            with self._vision_usage_scope(session.vision_usage), self.device_registry.device_lock(session.device_id):
                authority = session.effect_confirmation_authority
                reject_if(session.status != 'awaiting_effect_confirmation' or authority is None
                    or authority.consumed, UniversalAgentOrchestratorError('当前没有可用登录或付款确认。'))
                requested = self._normalize_effect_confirmation(confirmation)
                if requested != authority.scope():
                    authority.consumed = True
                    authority.invalid_reason = 'scope_mismatch'
                    raise UniversalAgentOrchestratorError('效果确认scope不一致。')
                authority.consumed = True
                authority.invalid_reason = 'confirmed'
                session.confirmed_effect_ids = tuple(authority.effect_ids)
                result = self._observe_and_decide(session)
                if session.status == 'awaiting_confirmation':
                    assert session.confirmation_authority is not None
                    result = self._confirm_one_locked(session, session.confirmation_authority.scope())
                return result
        finally:
            self._release_if_terminal(session)

    def refresh_decision(self, session: UniversalAgentSessionState) -> Any:
        reject_if(self.device_registry.active_session(session.device_id) != session.session_id,
            UniversalAgentOrchestratorError('当前会话不再拥有设备。'))
        with self._vision_usage_scope(session.vision_usage), self.device_registry.device_lock(session.device_id):
            return self._observe_and_decide(session)

    def run_autonomous_safe_loop(self, session: UniversalAgentSessionState, *, max_physical_actions: int=12,
        max_iterations: int=24) -> dict[str, Any]:
        reject_if(isinstance(max_physical_actions, bool) or max_physical_actions < 1
            or isinstance(max_iterations, bool) or max_iterations < 1,
            UniversalAgentOrchestratorError('自动循环上限必须为正整数。'))
        reject_if(self.device_registry.active_session(session.device_id) != session.session_id,
            UniversalAgentOrchestratorError('当前会话不再拥有设备。'))
        start_actions = session.physical_actions
        iterations = 0
        session.automatic_loop_enabled = True
        session.auto_pause_reason = ''
        try:
            with self._vision_usage_scope(session.vision_usage), self.device_registry.device_lock(session.device_id):
                while iterations < max_iterations and session.physical_actions - start_actions < max_physical_actions:
                    if session.status in self.device_registry.TERMINAL_STATUSES:
                        break
                    if session.status == 'awaiting_effect_confirmation':
                        session.auto_pause_reason = '登录或付款目标等待用户确认。'
                        break
                    if session.status == 'needs_reobservation':
                        self._observe_and_decide(session)
                    elif session.status == 'awaiting_confirmation':
                        authority = session.confirmation_authority
                        reject_if(authority is None or authority.consumed,
                            UniversalAgentOrchestratorError('待执行动作缺少一次性scope。'))
                        self._confirm_one_locked(session, authority.scope())
                    else:
                        session.auto_pause_reason = f'当前状态不能自动推进：{session.status}。'
                        break
                    iterations += 1
                if iterations >= max_iterations:
                    session.auto_pause_reason = '达到本次自动循环迭代上限。'
                elif session.physical_actions - start_actions >= max_physical_actions:
                    session.auto_pause_reason = '达到本次物理动作上限。'
                self._write_snapshot(session)
        finally:
            session.automatic_loop_enabled = False
            self._release_if_terminal(session)
        return {'physical_actions': session.physical_actions - start_actions, 'iterations': iterations,
            'status': session.status, 'pause_reason': session.auto_pause_reason}

    def start(self, *, session_id: str, raw_goal: str, exact_input_text: str | None=None,
        exact_action_kind: str | None=None, exact_target_label: str='', device_id: str,
        run_dir: Path) -> UniversalAgentSessionState:
        resolved_session = str(session_id or '').strip()
        resolved_device = str(device_id or '').strip()
        self.device_registry.reserve(resolved_device, resolved_session)
        try:
            ledger = VisionSessionUsageLedger(session_id=resolved_session)
            with self.device_registry.device_lock(resolved_device), self._vision_usage_scope(ledger):
                adapter = self.adapter_factory(resolved_device)
                store = self.evidence_store_factory(Path(run_dir))
                session = UniversalAgentSessionState(session_id=resolved_session,
                    raw_goal=str(raw_goal or '').strip(), device_id=resolved_device, run_dir=Path(run_dir),
                    adapter=adapter, evidence_store=store, vision_usage=ledger,
                    local_exact_input_authority=exact_input_text is not None)
                reject_if(not session.session_id or not session.raw_goal or not session.device_id,
                    UniversalAgentOrchestratorError('启动Agent需要session_id、目标和device_id。'))
                reject_if(exact_input_text is not None and exact_action_kind is not None,
                    UniversalAgentOrchestratorError('exact_input_text与exact_action_kind不能同时使用。'))
                try:
                    self._set_status(session, 'planning')
                    graph = build_exact_input_task_graph(session.raw_goal, exact_input_text=exact_input_text,
                        device_id=session.device_id) if exact_input_text is not None else build_exact_action_task_graph(
                        session.raw_goal, action_kind=exact_action_kind, target_label=exact_target_label,
                        device_id=session.device_id) if exact_action_kind is not None else self.deepseek_planner.plan(
                        session.raw_goal, device_id=session.device_id)
                    graph.validate()
                    session.task_graph = graph
                    self._remember(session, store.write_task_graph(graph),
                        store.write_effect_policy_snapshot(graph))
                    if graph.status == 'blocked':
                        self._set_status(session, 'blocked', '\n'.join(graph.clarification_questions)
                            or '计划缺少可执行目标。')
                    elif _requires_effect_confirmation(graph, graph.active_subgoal()):
                        self._set_status(session, 'awaiting_effect_confirmation')
                        self._bind_effect_confirmation(session)
                    else:
                        self._observe_and_decide(session)
                    self._write_snapshot(session)
                except Exception as exc:
                    self._set_status(session, 'failed', str(exc))
                    self._record_deepseek_failure(session, exc)
                    self._best_effort_snapshot(session)
                    raise
        except Exception:
            self.device_registry.release(resolved_device, resolved_session)
            raise
        self._release_if_terminal(session)
        return session

    def _terminate(self, session: UniversalAgentSessionState, *, status: str, message: str) -> None:
        try:
            with self.device_registry.device_lock(session.device_id):
                self._clear_action(session)
                if session.effect_confirmation_authority is not None:
                    session.effect_confirmation_authority.consumed = True
                    session.effect_confirmation_authority.invalid_reason = status
                self._set_status(session, status, message)
                self._write_snapshot(session)
        finally:
            self.device_registry.release(session.device_id, session.session_id)

    def pause(self, session: UniversalAgentSessionState) -> None:
        self._terminate(session, status='paused', message='用户已暂停。')

    def cancel(self, session: UniversalAgentSessionState) -> None:
        self._terminate(session, status='cancelled', message='用户已取消任务。')

    def invalidate_confirmation(self, session: UniversalAgentSessionState, *, reason: str) -> None:
        if self.device_registry.active_session(session.device_id) != session.session_id:
            return
        with self.device_registry.device_lock(session.device_id):
            self._clear_action(session)
            self._set_status(session, 'needs_reobservation',
                f'设备状态变化，旧动作已失效：{str(reason or "unknown")}。')
            self._write_snapshot(session)
