"""The production screenshot -> Qwen decision -> one action -> screenshot loop."""

from __future__ import annotations

from agent.domain.confirmation_authority import normalize_action_scope

from collections.abc import Mapping
from contextlib import nullcontext
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
    ConfirmationAuthority,
    DeviceTaskRegistryPort,
    EffectConfirmationAuthority,
)
from agent.domain.canonical_action_kinds import (
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


def _action_digest(action: Any) -> str:
    reject_if(action is None, UniversalAgentOrchestratorError("动作摘要缺少语义动作。"))
    payload = action.to_dict() if callable(getattr(action, 'to_dict', None)) else action
    return canonical_digest(payload)


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
        if (executed_effect and decision.previous_action_outcome != "matched"
                and (decision.proposal.action is None
                    or action_effect_kind(decision.proposal.action) != "")):
            self._clear_action(session)
            self._set_status(session, "failed", "本次效果已执行，但新图未确认结果；不自动重复效果。")
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
        self._clear_action(session)
        if not self._reserve_observation(session):
            return None
        session.goal_draft = self.bridge.goal_draft(session)
        self._set_status(session, "observing")
        available = self._prepare_observation_actions(session)
        scene, frames, paths, model_decision = session.adapter.capture_scene(session.goal_draft,
            evidence_dir=session.run_dir, prefix=f"before_step_{session.step_number}_frame",
            available_action_kinds=available)
        reject_if(not isinstance(model_decision, Mapping),
            UniversalAgentOrchestratorError("Qwen当前观察缺少同响应action/finish。"))
        self._remember(session, paths)
        observation = self._build_observation(session, scene=scene, frames=frames)
        decision = self._decide(session, frames=frames, observation=observation, model_decision=model_decision)
        self._stage_decision(session, decision=decision)
        self._write_snapshot(session)
        return decision

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
        authority = self._consume_confirmation(session, confirmation)
        observation, decision = session.trusted_observation, session.qwen_decision
        assert observation is not None and decision is not None
        if not self._reserve_observation(session,
            will_execute=decision.proposal.action.action != 'wait_for_change'):
            return None
        self._set_status(session, 'executing_one_action')
        before_actions = session.physical_actions
        session.goal_draft = self.bridge.goal_draft(session)
        try:
            result = session.adapter.execute(requested_action=decision.proposal.action,
                planned_scene=observation.scene, goal=session.goal_draft, confirmed=True,
                evidence_dir=session.run_dir, planned_frames=session.trusted_frames,
                action_authority=authority,
                available_action_kinds=session.observation_action_kinds,
                post_action_available_action_kinds=session.observation_action_kinds)
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
        reject_if(outcome != 'executed' or errors,
            UniversalAgentOrchestratorError('执行器没有确认本次物理动作及必要硬校验。'))
        after_frames = tuple(result.after_frames)
        after_paths = tuple(str(item).strip() for item in result.after_frame_paths)
        reject_if(len(after_frames) < 4 or len(after_paths) != len(after_frames) or any(not item for item in after_paths),
            UniversalAgentOrchestratorError('动作后缺少完整四帧新观察。'))

        session.step_number += 1
        new_observation = self._build_observation(session, scene=result.after_scene, frames=after_frames,
            observation_id=f'obs_{uuid.uuid4().hex}')
        session.history.append({'step_number': session.step_number - 1, 'task_revision': authority.revision,
            'step_id': authority.step_id, 'effect_ids': list(authority.effect_ids),
            'confirmation_receipt': {'authoritative': True, 'consumed': True, 'scope': authority.scope()},
            'qwen_decision': UniversalAgentSessionState._serialize(decision), 'execution': result.to_dict(),
            'before_observation_id': observation.observation_id, 'before_fingerprint': observation.fingerprint,
            'after_observation_id': new_observation.observation_id,
            'after_fingerprint': new_observation.fingerprint})
        reject_if(not isinstance(result.after_model_decision, Mapping),
            UniversalAgentOrchestratorError('动作后Qwen观察没有直接返回同响应action/finish。'))
        session.recent_navigation.record_execution(result.resolved_action.kind)
        next_decision = self._decide(session, frames=after_frames, observation=new_observation,
            model_decision=result.after_model_decision)
        session.history[-1]["visual_outcome"] = next_decision.previous_action_outcome
        self._remember(session, session.evidence_store.write_verification(session.step_number - 1,
            {'execution_outcome': outcome, 'visual_outcome': next_decision.previous_action_outcome,
            'verification_errors': list(errors), 'after_observation_id': new_observation.observation_id,
            'after_fingerprint': new_observation.fingerprint}))
        self._stage_decision(session, decision=next_decision,
            executed_effect=bool(physical == 1 and authority.effect_ids))
        transition = {'protocol_version': POST_ACTION_TRANSITION_PROTOCOL_VERSION,
            'transition_kind': 'new_screenshot_decision', 'execution_outcome': outcome,
            'visual_outcome': next_decision.previous_action_outcome,
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
        # Only the classified zero-action stale frame fault retains a resumable lease.
        if isinstance(error, GenericActionAdapterError) and _is_zero_action_stale_frame_fault(error):
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

    def run_autonomous_safe_loop(self, session: UniversalAgentSessionState, *, max_physical_actions: int | None=None,
        max_observations: int | None=None) -> dict[str, Any]:
        # Bad caller configuration is not a task failure and must not consume its scope.
        TaskExecutionBudget(
            max_physical_actions=session.execution_budget.max_physical_actions if max_physical_actions is None else max_physical_actions,
            max_observations=session.execution_budget.max_observations if max_observations is None else max_observations)
        reject_if(self.device_registry.active_session(session.device_id) != session.session_id,
            UniversalAgentOrchestratorError('当前会话不再拥有设备。'))
        start_actions = session.physical_actions
        iterations = 0
        session.automatic_loop_enabled = True
        session.auto_pause_reason = ''
        try:
            with self._vision_usage_scope(session.vision_usage), self.device_registry.device_lock(session.device_id):
                session.execution_budget.configure(max_physical_actions=max_physical_actions,
                    max_observations=max_observations)
                if session.status in {'budget_paused', 'paused'}:
                    session.pause_requested.clear()
                    self._set_status(session, 'needs_reobservation')
                while session.status != 'budget_paused':
                    if self._apply_pause_request(session):
                        break
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
                self._apply_pause_request(session)
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
        run_dir: Path, max_physical_actions: int=DEFAULT_DEVICE_ACTION_BUDGET,
        max_observations: int=DEFAULT_OBSERVATION_BUDGET) -> UniversalAgentSessionState:
        budget = TaskExecutionBudget(max_physical_actions=max_physical_actions, max_observations=max_observations)
        resolved_session = str(session_id or '').strip()
        resolved_device = str(device_id or '').strip()
        self.device_registry.reserve(resolved_device, resolved_session)
        try:
            ledger = VisionSessionUsageLedger(session_id=resolved_session)
            with self.device_registry.device_lock(resolved_device), self._vision_usage_scope(ledger):
                adapter = self.adapter_factory(resolved_device)
                store = self.evidence_store_factory(Path(run_dir))
                session = UniversalAgentSessionState(session_id=resolved_session,
                    raw_goal=str(raw_goal or ''), device_id=resolved_device, run_dir=Path(run_dir),
                    adapter=adapter, evidence_store=store, vision_usage=ledger, execution_budget=budget,
                    local_exact_input_authority=exact_input_text is not None, exact_input_text=exact_input_text,
                    exact_action_kind=exact_action_kind or self.required_action_kind, exact_target_label=exact_target_label)
                reject_if(not session.session_id or not session.raw_goal or not session.device_id,
                    UniversalAgentOrchestratorError('启动Agent需要session_id、目标和device_id。'))
                reject_if(exact_input_text is not None and exact_action_kind is not None,
                    UniversalAgentOrchestratorError('exact_input_text与exact_action_kind不能同时使用。'))
                try:
                    self._observe_and_decide(session)
                    self._write_snapshot(session)
                except Exception as exc:
                    self._set_status(session, 'failed', str(exc))
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

    def _apply_pause_request(self, session: UniversalAgentSessionState) -> bool:
        if not session.pause_requested.is_set() or session.status in self.device_registry.TERMINAL_STATUSES:
            return False
        self._clear_action(session)
        if session.effect_confirmation_authority is not None:
            session.effect_confirmation_authority.consumed = True
            session.effect_confirmation_authority.invalid_reason = 'paused'
            session.effect_confirmation_authority = None
        session.auto_pause_reason = '用户已暂停；恢复时重新观察。'
        self._set_status(session, 'paused')
        self._write_snapshot(session)
        return True

    def pause(self, session: UniversalAgentSessionState) -> None:
        if session.status in self.device_registry.TERMINAL_STATUSES:
            return
        session.pause_requested.set()
        # Signal without waiting behind the lock held by an in-flight action/loop.
        # That operation applies the pause only after recording its real result.
        if session.automatic_loop_enabled or session.status in {'observing', 'executing_one_action'}:
            return
        with self.device_registry.device_lock(session.device_id):
            self._apply_pause_request(session)

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
