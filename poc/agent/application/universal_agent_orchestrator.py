"""The production screenshot -> Qwen decision -> one action -> screenshot loop."""

from __future__ import annotations

from agent.domain.confirmation_authority import normalize_action_scope

from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
import re
from typing import Any, Callable
import uuid

from agent.domain.validation import reject_if
from agent.application.action_adapter import GenericSingleActionAdapterPort
from agent.application.orchestration_support import (
    UniversalAgentOrchestratorError,
    _action_digest,
    _is_zero_action_stale_frame_fault,
    _is_zero_action_reobservation_fault,
)
from agent.application.runtime_session import POST_ACTION_TRANSITION_PROTOCOL_VERSION, UniversalAgentSessionState
from agent.application.vision_usage import VisionSessionUsageLedger
from agent.domain import (
    AgentEvidenceStoreFactory,
    CANONICAL_SELECTION_RECEIPT_VERSION,
    ConfirmationAuthority,
    DeviceTaskRegistryPort,
    EffectConfirmationAuthority,
)
from agent.domain.action_catalog import (
    CANONICAL_ACTION_KINDS,
)
from agent.domain.execution_budget import (
    DEFAULT_DEVICE_ACTION_BUDGET, DEFAULT_OBSERVATION_BUDGET,
    ExecutionBudgetExhausted, TaskExecutionBudget,
)
from agent.domain.generic_goal import GenericIntentDraft
from agent.domain.recent_navigation import LOCAL_NAVIGATION_SOURCE
from agent.domain.qwen_task_context import (
    QwenTaskContext, action_effect_kind, CONFIRMATION_EFFECT_KINDS, execution_history_entry,
)
from agent.domain.session import TERMINAL_SESSION_STATUSES


def _model_history(session: UniversalAgentSessionState) -> list[dict[str, Any]]:
    return [execution_history_entry(step=item["step_number"],
        requested_action=item["execution"]["requested_action"],
        resolved_action=item["execution"]["resolved_action"],
        physical_actions=item["execution"]["physical_actions"],
        transport_outcome=item["execution"]["action_outcome"],
        execution_metadata=item["execution"].get("execution_metadata"),
        after_scene=item["execution"].get("after_scene", {}).get("summary", ""),
        visual_outcome=item.get("visual_outcome"))
        for item in session.history]


class ObservationBridge:
    """Supply the whole task and only actually executed actions; no hidden plan."""
    def goal_draft(self, session: UniversalAgentSessionState) -> GenericIntentDraft:
        aliases = getattr(session.adapter, "supported_app_aliases", None)
        return GenericIntentDraft(understood=True, app_id="current_surface", app_name="当前设备",
            objective=session.raw_goal, entities={
                "task_id": session.session_id, "history": _model_history(session),
                "conversation": [dict(item) for item in session.conversation],
                "exact_input_text": session.exact_input_text,
                "required_action_kind": session.exact_action_kind,
                "exact_target_label": session.exact_target_label,
                "launch_app_aliases": list(aliases()) if callable(aliases) else [],
            })


class UniversalAgentOrchestrator:
    """Run one authoritative Qwen action/finish decision at a time."""

    TERMINAL_STATUSES = TERMINAL_SESSION_STATUSES

    def __init__(self, *, qwen_observer: Any,
        adapter_factory: Callable[[str], GenericSingleActionAdapterPort],
        evidence_store_factory: AgentEvidenceStoreFactory, trusted_observation_factory: Callable[..., Any],
        bridge: ObservationBridge | None=None, device_registry: DeviceTaskRegistryPort,
        required_action_kind: str | None=None) -> None:
        self.required_action_kind = required_action_kind
        self.qwen_observer = qwen_observer
        self.adapter_factory = adapter_factory
        self.evidence_store_factory = evidence_store_factory
        self.trusted_observation_factory = trusted_observation_factory
        self.bridge = bridge or ObservationBridge()
        self.device_registry = device_registry
        self._observation_coordinator = ObservationDecisionCoordinator()
        self._confirmation_coordinator = ConfirmationExecutionCoordinator()
        self._execution_coordinator = AutonomousExecutionCoordinator()
        self._lifecycle_coordinator = SessionLifecycleCoordinator()

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
        session.confirmation_authority = None

    @staticmethod
    def _record_qwen_reply(session: UniversalAgentSessionState, model_decision: Mapping[str, Any]) -> dict[str, Any]:
        """Keep the same-response natural reply separate from canonical action fields."""
        decision = dict(model_decision)
        reply = str(decision.pop("_qwen_reply", "") or "").strip()
        if reply:
            session.qwen_reply = reply
            session.conversation.append({"role": "assistant", "content": reply})
        return decision

    def _record_uncertain_execution(self, session: UniversalAgentSessionState, *, authority: ConfirmationAuthority,
        decision: Any, physical_actions: int, error: Exception) -> None:
        """Persist a dispatched action whose post-action result is unavailable.

        The consumed authority is deliberately not recreated.  Recording the
        dispatch before asking for a fresh observation gives Qwen the factual
        history while preventing the local loop from replaying a potentially
        effective action (send, submit, delete, etc.).
        """

        action = decision.proposal.action
        error_text = str(error).strip() or type(error).__name__
        metadata = dict(getattr(error, 'execution_metadata', {}) or {})
        metadata.update({'uncertain_effect': True, 'failure_stage': 'post_action_observation',
            'error': error_text})
        executed_revision = session.step_number
        session.step_number += 1
        execution = {
            'requested_action': action.to_dict(),
            'resolved_action': action.to_dict(),
            'physical_actions': int(physical_actions),
            'action_outcome': 'uncertain',
            'execution_metadata': metadata,
            'verification_errors': list(getattr(error, 'verification_errors', ()) or ()),
            'after_scene': {'summary': f'动作已派发，但动作后观察失败：{error_text}'},
            'evidence': list(getattr(error, 'evidence', ()) or ()),
            'after_frame_paths': [],
        }
        session.history.append({
            'step_number': executed_revision,
            'task_revision': authority.revision,
            'step_id': authority.step_id,
            'effect_ids': list(authority.effect_ids),
            'confirmation_receipt': {'authoritative': True, 'consumed': True, 'scope': authority.scope()},
            'qwen_decision': UniversalAgentSessionState._serialize(decision),
            'execution': execution,
            'before_observation_id': getattr(session.trusted_observation, 'observation_id', None),
            'before_fingerprint': getattr(session.trusted_observation, 'fingerprint', None),
            'after_observation_id': None,
            'after_fingerprint': None,
            'visual_outcome': 'uncertain',
        })
        self._remember(session, getattr(error, 'evidence', ()))
        session.last_post_action_transition = {
            'protocol_version': POST_ACTION_TRANSITION_PROTOCOL_VERSION,
            'transition_kind': 'uncertain_action_reobservation',
            'execution_outcome': 'uncertain',
            'visual_outcome': 'uncertain',
            'physical_actions': session.physical_actions,
            'error': error_text,
        }

    def _mark_last_execution_uncertain(self, session: UniversalAgentSessionState, error: Exception) -> None:
        """Downgrade a dispatched result when its local Qwen binding fails."""

        if not session.history:
            return
        entry = session.history[-1]
        execution = dict(entry.get('execution') or {})
        metadata = dict(execution.get('execution_metadata') or {})
        error_text = str(error).strip() or type(error).__name__
        metadata.update({'uncertain_effect': True, 'failure_stage': 'post_action_decision_binding',
            'error': error_text})
        execution.update({'action_outcome': 'uncertain', 'execution_metadata': metadata,
            'after_scene': {'summary': f'动作已派发，但动作后决策绑定失败：{error_text}'}})
        entry['execution'] = execution
        entry['visual_outcome'] = 'uncertain'
        session.last_post_action_transition = {
            'protocol_version': POST_ACTION_TRANSITION_PROTOCOL_VERSION,
            'transition_kind': 'uncertain_action_reobservation',
            'execution_outcome': 'uncertain', 'visual_outcome': 'uncertain',
            'physical_actions': session.physical_actions, 'error': error_text,
        }

    @staticmethod
    def _available_action_kinds(session: UniversalAgentSessionState) -> frozenset[str]:
        provider = getattr(session.adapter, 'supported_action_kinds', None)
        actions = CANONICAL_ACTION_KINDS if not callable(provider) else frozenset(str(item or '').strip()
            for item in provider())
        reject_if(not actions or '' in actions or actions - CANONICAL_ACTION_KINDS,
            UniversalAgentOrchestratorError('设备canonical动作能力无效。'))
        if session.exact_action_kind:
            reject_if(session.exact_action_kind not in actions,
                UniversalAgentOrchestratorError('设备没有当前显式动作能力。'))
            permitted = ({'open_recent_apps', 'home'} if session.exact_action_kind == 'open_recent_apps'
                else {session.exact_action_kind})
            actions = actions & permitted
        reject_if(not actions, UniversalAgentOrchestratorError("设备没有当前显式动作能力。"))
        return frozenset(actions)

    def _write_snapshot(self, session: UniversalAgentSessionState) -> None:
        return self._lifecycle_coordinator.write_snapshot(self, session)

    def _best_effort_snapshot(self, session: UniversalAgentSessionState) -> None:
        return self._lifecycle_coordinator.best_effort_snapshot(self, session)

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
        return QwenTaskContext(task_id=session.session_id, device_id=session.device_id,
            revision=session.step_number, raw_goal=session.raw_goal,
            history=tuple(_model_history(session)), exact_input_text=session.exact_input_text)

    def _prepare_observation_actions(self, session: UniversalAgentSessionState) -> frozenset[str]:
        session.observation_action_kinds = self._available_action_kinds(session)
        return session.observation_action_kinds

    def _decide(self, session: UniversalAgentSessionState, *, frames: list[Any] | tuple[Any, ...],
        observation: Any, model_decision: Mapping[str, Any]) -> Any:
        context = self._task_context(session)
        available = session.observation_action_kinds
        reject_if(not available, UniversalAgentOrchestratorError('当前观察缺少已签发动作集合。'))
        self._remember(session, session.evidence_store.write_json(
            f'model_binding_step_{session.step_number}.json', {
                'observation_id': observation.observation_id, 'fingerprint': observation.fingerprint,
                'task_id': context.task_id,
                'available_action_kinds': sorted(available), 'model_decision': dict(model_decision)}))
        navigation = session.recent_navigation.select(model_decision,
            foreground_app_id=observation.scene.foreground_app_id,
            available_action_kinds=available)
        if navigation is not None:
            model_decision = navigation
        args: dict[str, Any] = {'frames': list(frames), 'task_context': context,
            'trusted_observation': observation, 'decision_number': session.step_number,
            'available_action_kinds': available, 'model_decision': dict(model_decision)}
        if model_decision.get("action") == "launch_app":
            app = model_decision.get("app")
            resolver = getattr(session.adapter, "resolve_app_launch_target", None)
            launch = resolver(app, app) if isinstance(app, str) and callable(resolver) else None
            reject_if(launch is None, UniversalAgentOrchestratorError("Qwen所选App没有本地可信启动映射。"))
            args["launch_target"] = {"launch_ref": launch.launch_ref, "expected_app_id": launch.expected_app_id,
                "target_app_id": app, "target_app_name": app}
        profile_provider = getattr(session.adapter, 'text_transport_profile', None)
        profile = profile_provider() if callable(profile_provider) else None
        if profile is not None:
            profile.validate()
            reject_if(profile.device_id != context.device_id,
                UniversalAgentOrchestratorError('ADB Keyboard profile与当前设备不一致。'))
            args['text_transport_profile'] = profile
        if navigation is not None:
            args['decision_source'] = LOCAL_NAVIGATION_SOURCE
        return self.qwen_observer.decide(**args)

    @staticmethod
    def _validate_decision_binding(session: UniversalAgentSessionState, observation: Any, decision: Any) -> None:
        expected = (session.session_id, session.device_id, session.step_number,
            observation.observation_id, observation.fingerprint)
        actual = (decision.task_id, decision.device_id, decision.revision, decision.observation_id,
            decision.fingerprint)
        reject_if(actual != expected or decision.trusted_observation is not observation,
            UniversalAgentOrchestratorError("Qwen决策没有绑定当前task/device/revision/observation。"))
        decision.proposal.validate(observation.scene)

    def _stage_decision(self, session: UniversalAgentSessionState, *, decision: Any,
        executed_effect: bool=False) -> Any:
        self._validate_decision_binding(session, session.trusted_observation, decision)
        session.qwen_decision = decision
        self._remember(session, session.evidence_store.write_qwen_decision(session.step_number, decision))
        uncertain_history = bool(session.history and (
            str(session.history[-1].get('execution', {}).get('action_outcome') or '') == 'uncertain'
            or str(session.history[-1].get('visual_outcome') or '') == 'uncertain'
        ))
        previous_effect = ""
        if uncertain_history:
            effect_ids = session.history[-1].get('effect_ids') or []
            if isinstance(effect_ids, (list, tuple)) and effect_ids:
                previous_effect = str(effect_ids[0] or '').strip()
        unmatched_effect = bool(session.history
            and session.history[-1].get('effect_ids')
            and str(session.history[-1].get('visual_outcome') or '') == 'unmatched')
        current_effect = action_effect_kind(decision.proposal.action)
        if (unmatched_effect or uncertain_history) and decision.proposal.status == 'finish':
            self._clear_action(session)
            self._set_status(session, 'failed',
                '本次外部效果未得到当前截图确认，不能以finish结束任务。')
            return decision
        if (uncertain_history and previous_effect and current_effect == previous_effect
                and decision.previous_action_outcome != "matched"):
            self._clear_action(session)
            session.auto_pause_reason = (
                "上一次外部效果的结果仍不确定，Qwen再次选择相同效果；已暂停，避免重复发送或提交。"
            )
            self._set_status(session, "failed", session.auto_pause_reason)
            return decision
        if ((executed_effect or uncertain_history) and decision.previous_action_outcome != "matched"
                and (decision.proposal.action is None
                    or current_effect != "")):
            self._clear_action(session)
            self._set_status(session, "needs_reobservation",
                "本次效果已执行，但新图未确认结果；保留不确定效果并重新观察，不自动重复效果。")
            return decision
        if decision.proposal.status == "finish":
            self._clear_action(session)
            session.qwen_decision = decision
            self._set_status(session, "succeeded")
            return decision
        self._bind_confirmation(session)
        kind = action_effect_kind(decision.proposal.action)
        if kind in CONFIRMATION_EFFECT_KINDS:
            self._set_status(session, "awaiting_effect_confirmation")
            self._bind_effect_confirmation(session)
        else:
            self._set_status(session, "awaiting_confirmation")
        self._remember(session, session.evidence_store.write_controller_decision(session.step_number,
            session.controller_decision.to_dict()))
        return decision

    def _reserve_observation(self, session: UniversalAgentSessionState, *, will_execute: bool=False) -> bool:
        try:
            session.execution_budget.request_observation(
                physical_actions=session.physical_actions, will_execute=will_execute)
        except ExecutionBudgetExhausted as exc:
            self._clear_action(session)
            session.auto_pause_reason = str(exc)
            self._set_status(session, 'budget_paused')
            self._write_snapshot(session)
            return False
        return True

    def _observe_and_decide(self, session: UniversalAgentSessionState) -> Any:
        return self._observation_coordinator.observe_and_decide(self, session)

    def _current_confirmation_scope(self, session: UniversalAgentSessionState) -> dict[str, Any]:
        observation, decision = session.trusted_observation, session.qwen_decision
        reject_if(observation is None or decision is None or decision.proposal.status != "action",
            UniversalAgentOrchestratorError("当前没有可执行动作scope。"))
        self._validate_decision_binding(session, observation, decision)
        effect = action_effect_kind(decision.proposal.action)
        return {"session_id": session.session_id, "task_id": session.session_id, "device_id": session.device_id,
            "revision": session.step_number, "step_id": f"step_{session.step_number}",
            "effect_ids": [effect] if effect else [], "observation_id": observation.observation_id,
            "fingerprint": observation.fingerprint, "decision_node_id": decision.proposal.action.node_id,
            "action_digest": _action_digest(decision.proposal.action)}

    def _bind_confirmation(self, session: UniversalAgentSessionState) -> None:
        scope = self._current_confirmation_scope(session)
        session.confirmation_authority = ConfirmationAuthority(session_id=scope['session_id'],
            task_id=scope['task_id'], device_id=scope['device_id'], revision=scope['revision'],
            step_id=scope['step_id'], effect_ids=tuple(scope['effect_ids']),
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
            'step_id': str(value.get('step_id') or ''),
            'effect_ids': sorted(str(item) for item in effect_ids), digest_key: digest}

    @staticmethod
    def _normalize_confirmation(value: Mapping[str, Any]) -> dict[str, Any]:
        try:
            return normalize_action_scope(value)
        except ValueError as exc:
            raise UniversalAgentOrchestratorError(str(exc)) from exc

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
        scope = self._current_confirmation_scope(session)
        preview = session.qwen_decision.proposal.action.to_dict()
        session.effect_confirmation_authority = EffectConfirmationAuthority(session_id=session.session_id,
            task_id=session.session_id, device_id=session.device_id, revision=session.step_number,
            step_id=scope["step_id"], effect_ids=tuple(scope["effect_ids"]),
            intent_digest=scope["action_digest"], intent_preview=preview)

    @classmethod
    def _normalize_effect_confirmation(cls, value: Mapping[str, Any]) -> dict[str, Any]:
        required = {'session_id', 'task_id', 'device_id', 'revision', 'step_id', 'effect_ids', 'intent_digest'}
        return cls._normalize_scope(value, required=required, digest_key='intent_digest', label='效果确认scope')

    def _confirm_one_locked(self, session: UniversalAgentSessionState, confirmation: Mapping[str, Any]) -> Any:
        return self._confirmation_coordinator.confirm_one_locked(self, session, confirmation)

    def confirm_one(self, session: UniversalAgentSessionState, confirmation: Mapping[str, Any]) -> Any:
        reject_if(self.device_registry.active_session(session.device_id) != session.session_id,
            UniversalAgentOrchestratorError('当前会话不再拥有设备。'))
        try:
            with self._vision_usage_scope(session.vision_usage), self.device_registry.device_lock(session.device_id):
                if self._apply_pause_request(session):
                    return None
                result = self._confirm_one_locked(session, confirmation)
                self._apply_pause_request(session)
                return result
        except Exception as exc:
            self._handle_operation_failure(session, exc)
            raise
        finally:
            self._release_if_terminal(session)

    def _handle_operation_failure(self, session: UniversalAgentSessionState, error: Exception) -> None:
        # Never overwrite a terminal result (or a user pause/cancel) with a
        # secondary exception raised while unwinding the operation.
        if session.status in self.device_registry.TERMINAL_STATUSES:
            return
        if session.pause_requested.is_set():
            self._apply_pause_request(session)
            return
        # Re-observation is a resumable state.  The caller may have already
        # recorded evidence and consumed the action authority; do not clear it
        # into a terminal failure here.
        if session.status in {'needs_reobservation', 'paused', 'budget_paused'}:
            self._best_effort_snapshot(session)
            return
        self._clear_action(session)
        self._set_status(session, 'failed', str(error).strip() or type(error).__name__)
        self._best_effort_snapshot(session)

    def approve_effects(self, session: UniversalAgentSessionState, confirmation: Mapping[str, Any]) -> Any:
        reject_if(self.device_registry.active_session(session.device_id) != session.session_id,
            UniversalAgentOrchestratorError('当前会话不再拥有设备。'))
        try:
            with self._vision_usage_scope(session.vision_usage), self.device_registry.device_lock(session.device_id):
                if self._apply_pause_request(session):
                    return None
                self._approve_effects_locked(session, confirmation)
                result = self._confirm_one_locked(session, session.confirmation_authority.scope())
                self._apply_pause_request(session)
                return result
        except Exception as exc:
            self._handle_operation_failure(session, exc)
            raise
        finally:
            self._release_if_terminal(session)

    def _approve_effects_locked(self, session: UniversalAgentSessionState,
        confirmation: Mapping[str, Any]) -> None:
        """Validate and consume the effect grant without executing the action."""
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
        reject_if(session.confirmation_authority is None
            or session.confirmation_authority.action_digest != authority.intent_digest,
            UniversalAgentOrchestratorError("登录/付款确认不属于当前动作。"))
        self._set_status(session, "awaiting_confirmation")

    def approve_effects_without_execution(self, session: UniversalAgentSessionState,
        confirmation: Mapping[str, Any]) -> None:
        """Approve only the effect scope; the next explicit confirmation executes once."""
        reject_if(self.device_registry.active_session(session.device_id) != session.session_id,
            UniversalAgentOrchestratorError('当前会话不再拥有设备。'))
        try:
            with self._vision_usage_scope(session.vision_usage), self.device_registry.device_lock(session.device_id):
                if self._apply_pause_request(session):
                    return None
                self._approve_effects_locked(session, confirmation)
                self._write_snapshot(session)
                return None
        except Exception as exc:
            self._handle_operation_failure(session, exc)
            raise
        finally:
            self._release_if_terminal(session)

    def refresh_decision(self, session: UniversalAgentSessionState) -> Any:
        reject_if(self.device_registry.active_session(session.device_id) != session.session_id,
            UniversalAgentOrchestratorError('当前会话不再拥有设备。'))
        try:
            with self._vision_usage_scope(session.vision_usage), self.device_registry.device_lock(session.device_id):
                if session.status == 'paused':
                    session.pause_requested.clear()
                result = self._observe_and_decide(session)
                self._apply_pause_request(session)
                return result
        except Exception as exc:
            self._handle_operation_failure(session, exc)
            raise
        finally:
            self._release_if_terminal(session)

    def run_autonomous_safe_loop(self, session: UniversalAgentSessionState, *, max_physical_actions: int | None=None, max_observations: int | None=None) -> dict[str, Any]:
        return self._execution_coordinator.run(self, session, max_physical_actions=max_physical_actions, max_observations=max_observations)

    def start(self, *, session_id: str, raw_goal: str, exact_input_text: str | None=None,
        exact_action_kind: str | None=None, exact_target_label: str='', device_id: str,
        run_dir: Path, visual_reference_paths: tuple[Path, ...] = (),
        conversation: tuple[dict[str, str], ...] = (),
        max_physical_actions: int=DEFAULT_DEVICE_ACTION_BUDGET,
        max_observations: int=DEFAULT_OBSERVATION_BUDGET) -> UniversalAgentSessionState:
        return self._lifecycle_coordinator.start(self, session_id=session_id, raw_goal=raw_goal,
            exact_input_text=exact_input_text, exact_action_kind=exact_action_kind,
            exact_target_label=exact_target_label, device_id=device_id, run_dir=run_dir,
            visual_reference_paths=visual_reference_paths, conversation=conversation,
            max_physical_actions=max_physical_actions, max_observations=max_observations)

    def _terminate(self, session: UniversalAgentSessionState, *, status: str, message: str) -> None:
        return self._lifecycle_coordinator.terminate(self, session, status=status, message=message)

    def _apply_pause_request(self, session: UniversalAgentSessionState) -> bool:
        return self._lifecycle_coordinator.apply_pause_request(self, session)

    def pause(self, session: UniversalAgentSessionState) -> None:
        return self._lifecycle_coordinator.pause(self, session)

    def cancel(self, session: UniversalAgentSessionState) -> None:
        return self._lifecycle_coordinator.cancel(self, session)

    def invalidate_confirmation(self, session: UniversalAgentSessionState, *, reason: str) -> None:
        return self._lifecycle_coordinator.invalidate_confirmation(self, session, reason=reason)

from agent.application.orchestration_components import (
    ObservationDecisionCoordinator, ConfirmationExecutionCoordinator,
    AutonomousExecutionCoordinator, SessionLifecycleCoordinator,
)
