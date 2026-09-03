"""The production screenshot -> Qwen decision -> one action -> screenshot loop."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
import re
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
from agent.domain.canonical_action_kinds import (
    CANONICAL_ACTION_KINDS,
    expected_idempotent_system_surface_kind,
)
from agent.domain.generic_goal import GenericIntentDraft
from agent.domain.qwen_task_context import QwenTaskContext
from agent.domain.task_graph import (
    DynamicTaskGraph,
    TaskGraphError,
    build_exact_action_task_graph,
    build_exact_input_task_graph,
    complete_active_subgoal,
)


class UniversalAgentOrchestratorError(RuntimeError):
    pass


_STALE_FRAME_FAILURE_PREFIXES = (
    '确认时本地真实画面已变化',
    '确认时前台 App 已变化',
    '确认时页面已变化',
    '当前新截图不再包含 Qwen 已选',
    '确认时目标区域已明显移动',
    '确认前本地多帧稳定性检查未通过',
)

_TYPED_INPUT_ACTION_KINDS = frozenset({'input_verified_text', 'clear_verified_text', 'press_enter'})
_TYPED_ACTIONS_BY_OPERATION = {
    '': frozenset(),
    'focus': frozenset(),
    'input_verified_text': frozenset({'input_verified_text', 'clear_verified_text'}),
    'clear_verified_text': frozenset({'clear_verified_text'}),
    'press_enter': frozenset({'press_enter'}),
}
_ELEMENT_GESTURE_TRAJECTORY_EPSILON = 0.08


def _is_zero_action_stale_frame_fault(exc: GenericActionAdapterError) -> bool:
    return exc.physical_actions == 0 and str(exc).startswith(_STALE_FRAME_FAILURE_PREFIXES)


def _action_digest(action: Any) -> str:
    reject_if(action is None, UniversalAgentOrchestratorError("动作摘要缺少语义动作。"))
    payload = action.to_dict() if callable(getattr(action, 'to_dict', None)) else action
    return canonical_digest(payload)


def _element_gesture_signature(graph: DynamicTaskGraph, action: Any) -> dict[str, Any] | None:
    if str(getattr(action, 'action', '') or '') != 'swipe_element':
        return None
    current = graph.active_subgoal()
    params = getattr(action, 'params', None)
    if current is None or not isinstance(params, Mapping):
        return None
    start = params.get('start')
    end = params.get('end')
    if (not isinstance(start, (list, tuple)) or len(start) != 2
        or not isinstance(end, (list, tuple)) or len(end) != 2):
        return None
    try:
        start_point = tuple(float(part) for part in start)
        end_point = tuple(float(part) for part in end)
    except (TypeError, ValueError):
        return None
    target = {'target': str(params.get('target') or '').strip().casefold(),
        'role': str(params.get('role') or '').strip().casefold(),
        'label': str(params.get('label') or '').strip().casefold()}
    if not any(target.values()):
        target['element_id'] = str(params.get('element_id') or '').strip()
    delta_x = end_point[0] - start_point[0]
    delta_y = end_point[1] - start_point[1]
    direction = ('right' if delta_x > 0 else 'left') if abs(delta_x) > abs(delta_y) else (
        'down' if delta_y > 0 else 'up')
    target_key = canonical_digest({'subgoal_id': current.subgoal_id, **target})
    return {'subgoal_id': current.subgoal_id, 'target_key': target_key, 'target': target,
        'start': start_point, 'end': end_point, 'direction': direction}


def _same_element_gesture_target(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    return (first.get('subgoal_id'), first.get('target_key')) == (
        second.get('subgoal_id'), second.get('target_key'))


def _near_same_element_gesture(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    if not _same_element_gesture_target(first, second) or first.get('direction') != second.get('direction'):
        return False
    return max(abs(float(a) - float(b)) for name in ('start', 'end')
        for a, b in zip(first[name], second[name])) <= _ELEMENT_GESTURE_TRAJECTORY_EPSILON


def _input_focus_action_facts(graph: DynamicTaskGraph, decision: Any) -> tuple[str, bool] | None:
    """Identify one typed input focus tap without depending on model element IDs or bounds jitter."""

    proposal = getattr(decision, 'proposal', None)
    action = getattr(proposal, 'action', None)
    if getattr(proposal, 'status', None) != 'action' or getattr(action, 'action', None) != 'tap_semantic':
        return None
    current = graph.active_subgoal()
    if current is None or not current.input_field_id:
        return None
    params = getattr(action, 'params', None)
    if not isinstance(params, Mapping) or str(params.get('role') or '') != 'input':
        return None
    states = params.get('states')
    if not isinstance(states, Mapping):
        return None
    field_id = str(states.get('input_field_id') or '').strip()
    if not field_id or field_id != current.input_field_id:
        return None
    key = canonical_digest({'subgoal_id': current.subgoal_id, 'action': 'tap_semantic',
        'role': 'input', 'input_field_id': field_id})
    return key, states.get('focused') is True


def _matched_current_input_receipt(session: UniversalAgentSessionState, *,
    allow_one_correction_observation: bool=False) -> bool:
    """Prove this session just produced the exact value for the still-active typed field."""

    graph = session.task_graph
    observation = session.trusted_observation
    current = graph.active_subgoal() if graph is not None else None
    if (current is None or current.input_operation != 'input_verified_text'
        or not current.input_field_id or observation is None or not session.history):
        return False
    latest = session.history[-1]
    if (not isinstance(latest, Mapping) or latest.get('task_revision') != graph.revision
        or (not allow_one_correction_observation
            and str(latest.get('after_observation_id') or '') != observation.observation_id)):
        return False
    execution = latest.get('execution')
    resolved = execution.get('resolved_action') if isinstance(execution, Mapping) else None
    expected = str(resolved.get('expected_input_value') or '') if isinstance(resolved, Mapping) else ''
    if (not expected or resolved.get('kind') != 'input_verified_text'
        or str(resolved.get('input_field_id') or '') != current.input_field_id
        or execution.get('physical_actions') != 1 or execution.get('action_outcome') != 'matched'):
        return False
    inputs = [element for element in observation.scene.elements
        if element.role == 'input'
        and str(element.states.get('input_field_id') or '') == current.input_field_id]
    return len(inputs) == 1 and inputs[0].states.get('value') == expected


def _matched_current_input_focus_receipt(session: UniversalAgentSessionState) -> bool:
    """Require one matched same-subgoal focus tap before exposing typed transports."""

    graph = session.task_graph
    observation = session.trusted_observation
    current = graph.active_subgoal() if graph is not None else None
    if (current is None or current.input_operation not in _TYPED_ACTIONS_BY_OPERATION
        or not current.input_field_id or observation is None or not session.history):
        return False
    latest = session.history[-1]
    if not isinstance(latest, Mapping) or latest.get('task_revision') != graph.revision:
        return False
    execution = latest.get('execution')
    action = execution.get('requested_action') if isinstance(execution, Mapping) else None
    params = action.get('params') if isinstance(action, Mapping) else None
    states = params.get('states') if isinstance(params, Mapping) else None
    resolved = execution.get('resolved_action') if isinstance(execution, Mapping) else None
    return bool(isinstance(action, Mapping) and action.get('action') == 'tap_semantic'
        and isinstance(states, Mapping)
        and str(states.get('input_field_id') or '') == current.input_field_id
        and isinstance(resolved, Mapping) and resolved.get('kind') == 'tap_semantic'
        and execution.get('physical_actions') == 1 and execution.get('action_outcome') == 'matched')


def _redundant_idempotent_system_action(decision: Any, observation: Any) -> tuple[str, str] | None:
    """Detect a system action whose destination is already the current visual surface."""

    proposal = getattr(decision, 'proposal', None)
    action = getattr(proposal, 'action', None)
    if getattr(proposal, 'status', None) != 'action':
        return None
    action_kind = str(getattr(action, 'action', '') or '').strip()
    expected_surface = expected_idempotent_system_surface_kind(action_kind)
    if expected_surface is None:
        return None
    scene = getattr(observation, 'scene', None)
    if scene is None:
        return None
    foreground_app_id = str(getattr(scene, 'foreground_app_id', '') or '').strip().lower()
    screen_id = str(getattr(scene, 'screen_id', '') or '').strip().lower()
    actual_surface = foreground_app_id if expected_surface == 'launcher' else (
        'recent_tasks' if screen_id == 'system_recent_tasks' else screen_id
    )
    return (action_kind, expected_surface) if actual_surface == expected_surface else None


def _repeated_idempotent_system_action_without_transition(
    session: UniversalAgentSessionState,
    decision: Any,
) -> tuple[str, str] | None:
    """Stop a consecutive Home/Recents retry after its first physical attempt."""

    proposal = getattr(decision, 'proposal', None)
    action = getattr(proposal, 'action', None)
    if getattr(proposal, 'status', None) != 'action':
        return None
    action_kind = str(getattr(action, 'action', '') or '').strip()
    expected_surface = expected_idempotent_system_surface_kind(action_kind)
    if expected_surface is None:
        return None
    for item in reversed(session.history):
        execution = item.get('execution') if isinstance(item, Mapping) else None
        if not isinstance(execution, Mapping) or int(execution.get('physical_actions') or 0) < 1:
            continue
        resolved = execution.get('resolved_action')
        last_kind = str(resolved.get('kind') or '').strip() if isinstance(resolved, Mapping) else ''
        return (action_kind, expected_surface) if last_kind == action_kind else None
    return None


def _future_effect_target(graph: DynamicTaskGraph, decision: Any) -> tuple[str, str] | None:
    """Reject an element action that names an effect owned by a later subgoal."""

    current = graph.active_subgoal()
    proposal = getattr(decision, 'proposal', None)
    action = getattr(proposal, 'action', None)
    params = getattr(action, 'params', None)
    if current is None or getattr(proposal, 'status', None) != 'action' or not isinstance(params, Mapping):
        return None
    action_kind = str(getattr(action, 'action', '') or '').strip()
    target = str(params.get('target') or '').strip()
    if not action_kind or not target:
        return None
    effect_kind = next((effect.effect_kind for effect in graph.risk_actions
        if effect.effect_kind == target and current.subgoal_id not in effect.subgoal_ids), '')
    return (action_kind, effect_kind) if effect_kind else None


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

    _SURFACE_IDENTITIES = {'device': ('device', '设备界面'), 'system': ('system', '系统界面'),
        'current_surface': ('current_surface', '当前界面')}

    @staticmethod
    def _typed_input_view(graph: DynamicTaskGraph, active: Any) -> dict[str, Any]:
        field_id = str(getattr(active, 'input_field_id', '') or '').strip()
        operation = str(getattr(active, 'input_operation', '') or '').strip()
        if not field_id or not operation:
            return {}
        entities = graph.goal.entities
        label = ''
        text: Any = None
        if field_id == 'primary_input':
            text = entities.get('input_text')
            label = str(entities.get('target_ui_label') or '')
        else:
            fields = entities.get('input_fields')
            matches = [item for item in fields or () if isinstance(item, dict)
                and str(item.get('field_id') or '') == field_id]
            reject_if(len(matches) != 1,
                UniversalAgentOrchestratorError(f'当前子目标引用的 typed input field 不唯一：{field_id}'))
            text = matches[0].get('text')
            label = str(matches[0].get('field_label') or '')
        if operation == 'clear_verified_text':
            text = ''
        reject_if(operation in {'focus', 'input_verified_text', 'press_enter'} and not isinstance(text, str),
            UniversalAgentOrchestratorError(f'当前 typed input field 缺少逐字正文：{field_id}'))
        return {'text': text if isinstance(text, str) else '', 'field_id': field_id, 'field_label': label,
            'multiline': bool(isinstance(text, str) and '\n' in text), 'operation': operation,
            'target_only': operation == 'clear_verified_text'}

    @classmethod
    def _subgoal_visual_context(cls, graph: DynamicTaskGraph, subgoal: Any) -> dict[str, Any]:
        entities = {key: value for key, value in graph.goal.entities.items() if key != 'input_fields'}
        for key in ('active_input_transaction_text', 'active_input_field_id', 'active_input_field_label',
            'active_input_multiline', 'active_input_operation', 'active_input_target_only'):
            entities.pop(key, None)
        active_input = cls._typed_input_view(graph, subgoal)
        text = active_input.get('text')
        target_only = active_input.get('target_only') is True
        if isinstance(text, str) and (text or target_only):
            if text:
                entities['active_input_transaction_text'] = text
            entities['active_input_field_id'] = active_input['field_id']
            entities['active_input_operation'] = active_input['operation']
            if target_only:
                entities['active_input_target_only'] = True
            if active_input.get('field_label'):
                entities['active_input_field_label'] = active_input['field_label']
            entities['active_input_multiline'] = bool(active_input.get('multiline'))
            predecessors = [item for item in graph.subgoals if item.subgoal_id in subgoal.depends_on
                and item.input_field_id]
            if len(predecessors) == 1:
                predecessor = cls._typed_input_view(graph, predecessors[0])
                if predecessor.get('text'):
                    entities.update({'active_input_predecessor_field_id': predecessor['field_id'],
                        'active_input_predecessor_field_label': predecessor['field_label'],
                        'active_input_predecessor_text': predecessor['text']})
        else:
            entities.pop('input_text', None)
        transition_receipt = None
        required_operation = str(subgoal.input_operation or '').strip()
        effect_ids = tuple(sorted(subgoal.risk_action_ids))
        if required_operation or effect_ids:
            transition_receipt = {'state': 'pending', 'subgoal_id': subgoal.subgoal_id,
                'required_operation': required_operation or 'external_effect',
                'executed_operation': '', 'effect_ids': list(effect_ids)}
        result = {'subgoal_id': subgoal.subgoal_id, 'objective': subgoal.objective,
            'constraints': list(subgoal.constraints), 'completion_conditions': list(subgoal.completion_conditions),
            'execution_class': {'read_only': 'observe', 'navigation_only': 'navigate',
            'external_state': 'effect', 'unknown': 'unknown'}.get(subgoal.external_impact, 'unknown'),
            'goal_entities': entities,
            'current_effect_kinds': list(dict.fromkeys(effect.effect_kind for effect in graph.risk_actions
                if effect.effect_kind and effect.risk_id in effect_ids)),
            'forbidden_future_effect_kinds': list(dict.fromkeys(effect.effect_kind for effect in graph.risk_actions
                if effect.effect_kind and effect.risk_id not in effect_ids))}
        if subgoal.required_action_kind:
            result['required_action_kind'] = subgoal.required_action_kind
        if transition_receipt is not None:
            result['transition_receipt'] = transition_receipt
        return result

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
    def _available_action_kinds(session: UniversalAgentSessionState, *,
        assume_current_input_focus_receipt: bool=False) -> frozenset[str]:
        provider = getattr(session.adapter, 'supported_action_kinds', None)
        actions = CANONICAL_ACTION_KINDS if not callable(provider) else frozenset(str(item or '').strip()
            for item in provider())
        reject_if(not actions or '' in actions or actions - CANONICAL_ACTION_KINDS,
            UniversalAgentOrchestratorError('设备canonical动作能力无效。'))
        graph = session.task_graph
        current = graph.active_subgoal() if graph is not None else None
        reject_if(current is None, UniversalAgentOrchestratorError('当前任务没有可裁剪动作集合的active子目标。'))
        operation = str(current.input_operation or '').strip()
        reject_if(operation not in _TYPED_ACTIONS_BY_OPERATION,
            UniversalAgentOrchestratorError(f'当前子目标input_operation无对应动作合同：{operation}。'))
        allowed_typed = _TYPED_ACTIONS_BY_OPERATION[operation]
        scoped = (actions - _TYPED_INPUT_ACTION_KINDS) | (actions & allowed_typed)
        focus_ready = bool(assume_current_input_focus_receipt or session.input_focus_retry_key
            or _matched_current_input_focus_receipt(session))
        if allowed_typed and not focus_ready:
            # A typed transport can affect Android's current editor connection without
            # observing which field owns it.  Expose only focus/navigation actions until
            # this same subgoal has one matched tap on its typed field and a new screenshot.
            scoped = scoped - _TYPED_INPUT_ACTION_KINDS
        if session.input_focus_retry_key and allowed_typed:
            # The previous same-frame Qwen decision selected the active input field even though
            # its audited scene already proved that field focused.  The one permitted fresh
            # correction observation must not reopen that now-proven-redundant action; Qwen still
            # chooses from every other device-supported action, including the required typed one.
            scoped = scoped - {'tap_semantic'}
            if _matched_current_input_receipt(session):
                # The same session has already executed and visually verified this exact typed
                # transaction.  Keep Qwen as the sole completion authority, but do not offer it
                # a second write or clear while it corrects its redundant focus decision.
                scoped = scoped - {'input_verified_text', 'clear_verified_text'}
        if session.future_effect_retry is not None:
            scoped = scoped - {session.future_effect_retry[0]}
        required_action_kind = str(current.required_action_kind or '').strip()
        if required_action_kind:
            scoped = scoped & {required_action_kind}
        reject_if(not scoped, UniversalAgentOrchestratorError('当前子目标没有可用canonical动作。'))
        return frozenset(scoped)

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
        session.effect_previews = tuple(dict(item) for item in graph.to_dict()['effect_intents'])
        context.validate()
        return context

    def _decide(self, session: UniversalAgentSessionState, *, frames: list[Any] | tuple[Any, ...],
        observation: Any, model_decision: Mapping[str, Any]) -> Any:
        context = self._task_context(session)
        available = self._available_action_kinds(session)
        graph = session.task_graph
        assert graph is not None
        target = graph.goal.target_apps[0] if len(graph.goal.target_apps) == 1 else None
        resolver = getattr(session.adapter, 'resolve_app_launch_target', None)
        launch = resolver(target.app_id, target.app_name) if target is not None and callable(resolver) else None
        launch_parameters = None
        if launch is None:
            available = available - {'launch_app'}
        else:
            launch_parameters = {'launch_ref': str(launch.launch_ref),
                'expected_app_id': str(launch.expected_app_id),
                'target_app_id': target.app_id, 'target_app_name': target.app_name}
        args: dict[str, Any] = {'frames': list(frames), 'task_context': context,
            'trusted_observation': observation, 'decision_number': session.step_number,
            'available_action_kinds': available, 'model_decision': dict(model_decision)}
        if launch_parameters:
            args['launch_target'] = launch_parameters
        profile_provider = getattr(session.adapter, 'text_transport_profile', None)
        profile = profile_provider() if callable(profile_provider) else None
        if profile is not None:
            profile.validate()
            reject_if(profile.device_id != context.device_id,
                UniversalAgentOrchestratorError('ADB Keyboard profile与当前设备不一致。'))
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

    @staticmethod
    def _selection_receipt(session: UniversalAgentSessionState, decision: Any) -> CanonicalSelectionReceipt:
        del session
        action = getattr(getattr(decision, 'proposal', None), 'action', None)
        kind = str(getattr(action, 'action', '') or '')
        return CanonicalSelectionReceipt(allowed=True,
            reason='Qwen同响应动作已绑定当前截图；最终设备能力由执行器校验。', canonical_class=kind)

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
        session.gesture_correction = None
        session.element_gesture_attempts.clear()
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
        executed_input_focus: tuple[str, bool] | None=None,
        executed_effect_scope: tuple[str, tuple[str, ...]] | None=None,
        executed_transition: tuple[str, str, tuple[str, ...]] | None=None,
        executed_action: Any=None) -> Any:
        graph = session.task_graph
        observation = session.trusted_observation
        assert graph is not None and observation is not None
        self._validate_decision_binding(graph, observation, decision)
        executed_gesture = _element_gesture_signature(graph, executed_action)
        if executed_gesture is not None:
            key = str(executed_gesture['target_key'])
            session.element_gesture_attempts[key] = session.element_gesture_attempts.get(key, 0) + 1
        session.qwen_decision = decision
        self._remember(session, session.evidence_store.write_qwen_decision(session.step_number, decision))
        if decision.proposal.status == 'finish':
            correcting_redundant_focus = bool(session.input_focus_retry_key)
            session.future_effect_retry = None
            session.gesture_correction = None
            current = graph.active_subgoal()
            assert current is not None
            if (executed_transition is None and correcting_redundant_focus
                and _matched_current_input_receipt(session, allow_one_correction_observation=True)):
                # A redundant-focus correction is a read-only observation after the exact input
                # action.  Carry forward only that tightly bound same-session receipt; an old
                # pre-existing field value has no history entry and can never use this path.
                executed_transition = (current.subgoal_id, 'input_verified_text', ())
            session.input_focus_retry_key = ''
            transition_error = self._finish_transition_error(current, executed_transition)
            if transition_error:
                session.controller_decision = None
                session.confirmation_authority = None
                self._set_status(session, 'failed', transition_error)
                return decision
            self._complete_from_finish(session, decision)
            return decision
        reject_if(decision.proposal.status != 'action',
            UniversalAgentOrchestratorError('Qwen单步决策只允许action或finish。'))
        proposed_gesture = _element_gesture_signature(graph, decision.proposal.action)
        pending_correction = session.gesture_correction
        if pending_correction is not None:
            if proposed_gesture is not None and _same_element_gesture_target(pending_correction, proposed_gesture):
                if _near_same_element_gesture(pending_correction, proposed_gesture):
                    session.gesture_correction_history.append({'state': 'rejected_after_reobservation',
                        'target_key': proposed_gesture['target_key'], 'direction': proposed_gesture['direction'],
                        'physical_actions': session.physical_actions})
                    session.gesture_correction = None
                    session.controller_decision = None
                    session.confirmation_authority = None
                    self._set_status(session, 'failed',
                        '同一目标元素在一次新观察纠正后仍选择近似滑动轨迹；已在下一次物理动作前停止。')
                    return decision
                session.gesture_correction_history.append({'state': 'accepted_changed_trajectory',
                    'target_key': proposed_gesture['target_key'], 'from_direction': pending_correction['direction'],
                    'to_direction': proposed_gesture['direction'], 'physical_actions': session.physical_actions})
            else:
                session.gesture_correction_history.append({'state': 'accepted_alternative_action',
                    'target_key': pending_correction['target_key'], 'physical_actions': session.physical_actions})
            session.gesture_correction = None
        if proposed_gesture is not None:
            attempts = session.element_gesture_attempts.get(str(proposed_gesture['target_key']), 0)
            if attempts >= 2:
                session.controller_decision = None
                session.confirmation_authority = None
                self._set_status(session, 'failed',
                    '同一子目标和目标元素已经执行两次滑动但当前截图仍未证明完成；已停止且未执行第三次。')
                return decision
        if (executed_gesture is not None and proposed_gesture is not None
            and _near_same_element_gesture(executed_gesture, proposed_gesture)):
            session.gesture_correction = dict(executed_gesture)
            session.gesture_correction_history.append({'state': 'needs_reobservation',
                'target_key': executed_gesture['target_key'], 'direction': executed_gesture['direction'],
                'physical_actions': session.physical_actions})
            session.controller_decision = None
            session.confirmation_authority = None
            self._set_status(session, 'needs_reobservation',
                '当前截图仍建议对同一目标重复近似元素滑动；先取得一次带纠正提示的新截图，不重复旧轨迹。')
            return decision
        redundant_system_action = _redundant_idempotent_system_action(decision, observation)
        if redundant_system_action is not None:
            action_kind, surface_kind = redundant_system_action
            session.controller_decision = None
            session.confirmation_authority = None
            self._set_status(session, 'failed',
                f'当前截图已经处于 {surface_kind} 系统页面；{action_kind} 不会推进当前目标，'
                '已在下一次物理动作前停止。')
            return decision
        repeated_system_action = _repeated_idempotent_system_action_without_transition(session, decision)
        if repeated_system_action is not None:
            action_kind, surface_kind = repeated_system_action
            session.controller_decision = None
            session.confirmation_authority = None
            self._set_status(session, 'failed',
                f'{action_kind} 上一次物理动作后没有到达 {surface_kind} 系统页面；'
                '同一系统动作不得重复，已在下一次物理动作前停止。')
            return decision
        future_effect = _future_effect_target(graph, decision)
        if future_effect:
            action_kind, effect_kind = future_effect
            session.controller_decision = None
            session.confirmation_authority = None
            if session.future_effect_retry is not None:
                session.future_effect_retry = None
                self._set_status(session, 'failed',
                    f'Qwen在一次新观察纠正后仍选择后继子目标效果 {effect_kind}；已停止且未执行。')
            else:
                session.future_effect_retry = (action_kind, effect_kind)
                self._set_status(session, 'needs_reobservation',
                    f'Qwen选择了后继子目标效果 {effect_kind}；当前子目标无权执行，先取得一次新截图纠正。')
            return decision
        session.future_effect_retry = None
        if executed_effect_scope is not None:
            effect_subgoal_id, effect_ids = executed_effect_scope
            current = graph.active_subgoal()
            reject_if(current is None or current.subgoal_id != effect_subgoal_id
                or tuple(sorted(current.risk_action_ids)) != effect_ids,
                UniversalAgentOrchestratorError('已执行效果scope与当前任务图不一致。'))
            session.controller_decision = None
            session.confirmation_authority = None
            self._set_status(session, 'failed',
                '外部效果子目标已执行一次物理动作；新截图未证明完成，已停止且未自动重复。')
            return decision
        focus = _input_focus_action_facts(graph, decision)
        if session.input_focus_retry_key:
            repeated_after_correction = bool(focus and focus[0] == session.input_focus_retry_key)
            session.input_focus_retry_key = ''
            reject_if(repeated_after_correction, UniversalAgentOrchestratorError(
                '同一输入框聚焦动作在一次新观察纠正后仍无推进；已停止且未重复点击。'))
        repeated_without_progress = bool(focus and executed_input_focus
            and focus[0] == executed_input_focus[0])
        redundant_focused_tap = bool(focus and focus[1])
        if focus and (repeated_without_progress or redundant_focused_tap):
            session.input_focus_retry_key = focus[0]
            session.controller_decision = None
            session.confirmation_authority = None
            self._set_status(session, 'needs_reobservation',
                '当前截图仍建议重复聚焦同一输入框；先取得一次新截图让Qwen纠正，不重复旧点击。')
            return decision
        receipt = self._selection_receipt(session, decision)
        session.controller_decision = receipt
        self._remember(session, session.evidence_store.write_controller_decision(session.step_number,
            receipt.to_dict()))
        self._set_status(session, 'awaiting_confirmation')
        self._bind_confirmation(session)
        return decision

    @staticmethod
    def _goal_with_gesture_correction(goal: GenericIntentDraft,
        correction: Mapping[str, Any] | None) -> GenericIntentDraft:
        if not isinstance(correction, Mapping):
            return goal
        entities = dict(goal.entities)
        active = entities.get('active_subgoal_visual_context')
        if not isinstance(active, Mapping) or str(active.get('subgoal_id') or '') != str(
            correction.get('subgoal_id') or ''):
            return goal
        active = dict(active)
        active['gesture_correction'] = {'state': 'previous_trajectory_did_not_complete_target',
            'target': dict(correction.get('target') or {}),
            'start': list(correction.get('start') or ()), 'end': list(correction.get('end') or ()),
            'direction': str(correction.get('direction') or ''),
            'required_change': 'choose_materially_different_trajectory_or_other_action_or_finish_from_current_scene'}
        entities['active_subgoal_visual_context'] = active
        projected = replace(goal, entities=entities)
        projected.validate()
        return projected

    @staticmethod
    def _finish_transition_error(current: Any,
        executed_transition: tuple[str, str, tuple[str, ...]] | None) -> str:
        """Reject only impossible attribution; Qwen still owns visual completion."""

        required_operation = str(current.input_operation or '').strip()
        required_effect_ids = tuple(sorted(current.risk_action_ids))
        if not required_operation and not required_effect_ids:
            return ''
        if executed_transition is None:
            return ('当前输入/外部效果子目标尚无本会话动作回执；当前截图中的既有状态不能证明'
                '本次任务刚完成。')
        subgoal_id, operation, effect_ids = executed_transition
        if subgoal_id != current.subgoal_id:
            return '当前动作回执不属于当前子目标，不能据此完成。'
        expected_operation = {'focus': 'tap_semantic'}.get(required_operation, required_operation)
        if required_operation and operation != expected_operation:
            return (f'当前输入子目标需要本会话执行 {expected_operation}；刚执行的是 {operation}，'
                '不能据此完成。')
        if required_effect_ids and effect_ids != required_effect_ids:
            return '当前外部效果动作回执与当前 effect scope 不一致，不能据此完成。'
        return ''

    def _observe_and_decide(self, session: UniversalAgentSessionState) -> Any:
        graph = session.task_graph
        reject_if(graph is None or graph.active_subgoal() is None,
            UniversalAgentOrchestratorError('当前计划没有可观察高层目标。'))
        current = graph.active_subgoal()
        if _requires_effect_confirmation(graph, current) and not session.confirmed_effect_ids:
            self._set_status(session, 'awaiting_effect_confirmation')
            self._bind_effect_confirmation(session)
            self._write_snapshot(session)
            return None
        self._clear_action(session)
        session.goal_draft = self._goal_with_gesture_correction(
            self.bridge.goal_draft(graph), session.gesture_correction)
        self._set_status(session, 'observing')
        available = self._available_action_kinds(session)
        scene, frames, paths, model_decision = session.adapter.capture_scene(session.goal_draft,
            evidence_dir=session.run_dir, prefix=f'before_step_{session.step_number}_frame',
            available_action_kinds=available)
        reject_if(not isinstance(model_decision, Mapping),
            UniversalAgentOrchestratorError('当前Qwen观察没有直接返回同响应action/finish。'))
        self._remember(session, paths)
        observation = self._build_observation(session, scene=scene, frames=frames)
        decision = self._decide(session, frames=frames, observation=observation,
            model_decision=model_decision)
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

    def _confirm_one_locked(self, session: UniversalAgentSessionState, confirmation: Mapping[str, Any]) -> Any:
        authority = self._consume_confirmation(session, confirmation)
        graph, observation, decision = session.task_graph, session.trusted_observation, session.qwen_decision
        assert graph is not None and observation is not None and decision is not None
        executed_input_focus = _input_focus_action_facts(graph, decision)
        receipt = session.controller_decision
        reject_if(receipt is None,
            UniversalAgentOrchestratorError('动作缺少canonical映射回执。'))
        self._set_status(session, 'executing_one_action')
        before_actions = session.physical_actions
        try:
            result = session.adapter.execute(requested_action=decision.proposal.action,
                planned_scene=observation.scene, goal=session.goal_draft, confirmed=True,
                evidence_dir=session.run_dir, planned_frames=session.trusted_frames,
                action_authority=authority,
                available_action_kinds=self._available_action_kinds(session),
                post_action_available_action_kinds=self._available_action_kinds(session,
                    assume_current_input_focus_receipt=executed_input_focus is not None))
        except GenericActionAdapterError as exc:
            session.physical_actions += max(0, int(exc.physical_actions))
            self._remember(session, exc.evidence)
            self._set_status(session,
                'needs_reobservation' if _is_zero_action_stale_frame_fault(exc) else 'failed', str(exc))
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
        reject_if(outcome != 'matched' or errors,
            UniversalAgentOrchestratorError('执行器没有确认本次物理动作及必要硬校验。'))
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

        reject_if(not isinstance(result.after_model_decision, Mapping),
            UniversalAgentOrchestratorError('动作后Qwen观察没有直接返回同响应action/finish。'))
        next_decision = self._decide(session, frames=after_frames, observation=new_observation,
            model_decision=result.after_model_decision)
        executed_effect_scope = ((authority.subgoal_id, tuple(sorted(authority.effect_ids)))
            if physical == 1 and authority.effect_ids else None)
        executed_transition = ((authority.subgoal_id, result.resolved_action.kind,
            tuple(sorted(authority.effect_ids))) if physical == 1 else None)
        self._stage_decision(session, decision=next_decision, executed_input_focus=executed_input_focus,
            executed_effect_scope=executed_effect_scope, executed_transition=executed_transition,
            executed_action=decision.proposal.action)
        transition = {'protocol_version': POST_ACTION_TRANSITION_PROTOCOL_VERSION,
            'transition_kind': 'new_screenshot_decision', 'outcome': outcome,
            'physical_actions_before': before_actions, 'physical_actions': session.physical_actions,
            'next_decision_status': next_decision.proposal.status,
            'after_observation_id': new_observation.observation_id,
            'after_fingerprint': new_observation.fingerprint}
        session.last_post_action_transition = transition
        self._remember(session, session.evidence_store.write_post_action_transition(session.step_number - 1,
            transition))
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
        try:
            with self._vision_usage_scope(session.vision_usage), self.device_registry.device_lock(session.device_id):
                return self._observe_and_decide(session)
        finally:
            self._release_if_terminal(session)

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
                        try:
                            self._confirm_one_locked(session, authority.scope())
                        except GenericActionAdapterError as exc:
                            # No device action occurred and the adapter already classified
                            # the old screenshot as stale.  Discard that action and let the
                            # next iteration obtain one new screenshot/Qwen decision.
                            if exc.physical_actions != 0 or session.status != 'needs_reobservation':
                                raise
                            self._clear_action(session)
                    else:
                        session.auto_pause_reason = f'当前状态不能自动推进：{session.status}。'
                        break
                    iterations += 1
                if iterations >= max_iterations:
                    session.auto_pause_reason = '达到本次自动循环迭代上限。'
                elif session.physical_actions - start_actions >= max_physical_actions:
                    session.auto_pause_reason = '达到本次物理动作上限。'
                session.automatic_loop_enabled = False
                self._write_snapshot(session)
        except Exception as exc:
            session.automatic_loop_enabled = False
            self._clear_action(session)
            self._set_status(session, 'failed', str(exc).strip() or type(exc).__name__)
            self._best_effort_snapshot(session)
            raise
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
