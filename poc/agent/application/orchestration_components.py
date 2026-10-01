"""P0 orchestration collaborators.

The public orchestrator remains the compatibility facade.  These collaborators
own the four lifecycle responsibilities while calling the facade's shared
protocol helpers, so canonical action semantics and persistence stay unchanged.
"""

from collections.abc import Mapping
from pathlib import Path
import uuid
from typing import Any

from agent.application.action_adapter import GenericActionAdapterError
from agent.application.orchestration_support import (
    UniversalAgentOrchestratorError,
    _action_digest,
    _is_zero_action_reobservation_fault,
)
from agent.application.runtime_session import POST_ACTION_TRANSITION_PROTOCOL_VERSION, UniversalAgentSessionState
from agent.application.vision_usage import VisionSessionUsageLedger
from agent.domain import CANONICAL_SELECTION_RECEIPT_VERSION, ConfirmationAuthority, EffectConfirmationAuthority
from agent.domain.execution_budget import (
    DEFAULT_DEVICE_ACTION_BUDGET, DEFAULT_OBSERVATION_BUDGET,
    TaskExecutionBudget,
)
from agent.domain.validation import reject_if
class ObservationDecisionCoordinator:
    def observe_and_decide(self, host, session: UniversalAgentSessionState) -> Any:
        host._clear_action(session)
        if not host._reserve_observation(session):
            return None
        session.goal_draft = host.bridge.goal_draft(session)
        host._set_status(session, "observing")
        available = host._prepare_observation_actions(session)
        try:
            scene, frames, paths, model_decision = session.adapter.capture_scene(session.goal_draft,
                evidence_dir=session.run_dir, prefix=f"before_step_{session.step_number}_frame",
                available_action_kinds=available)
        except GenericActionAdapterError as exc:
            host._remember(session, exc.evidence)
            if _is_zero_action_reobservation_fault(exc):
                host._clear_action(session)
                host._set_status(session, 'needs_reobservation', str(exc).strip() or type(exc).__name__)
            host._best_effort_snapshot(session)
            raise
        reject_if(not isinstance(model_decision, Mapping),
            UniversalAgentOrchestratorError("Qwen当前观察缺少同响应action/finish。"))
        host._remember(session, paths)
        model_decision = host._record_qwen_reply(session, model_decision)
        observation = host._build_observation(session, scene=scene, frames=frames)
        decision = host._decide(session, frames=frames, observation=observation, model_decision=model_decision)
        host._stage_decision(session, decision=decision)
        host._write_snapshot(session)
        return decision

class ConfirmationExecutionCoordinator:
    def confirm_one_locked(self, host, session: UniversalAgentSessionState, confirmation: Mapping[str, Any]) -> Any:
        authority = host._consume_confirmation(session, confirmation)
        observation, decision = session.trusted_observation, session.qwen_decision
        assert observation is not None and decision is not None
        if not host._reserve_observation(session,
            will_execute=decision.proposal.action.action != 'wait_for_change'):
            return None
        host._set_status(session, 'executing_one_action')
        before_actions = session.physical_actions
        session.goal_draft = host.bridge.goal_draft(session)
        try:
            result = session.adapter.execute(requested_action=decision.proposal.action,
                planned_scene=observation.scene, goal=session.goal_draft, confirmed=True,
                evidence_dir=session.run_dir, planned_frames=session.trusted_frames,
                action_authority=authority,
                available_action_kinds=session.observation_action_kinds,
                post_action_available_action_kinds=session.observation_action_kinds)
        except GenericActionAdapterError as exc:
            physical_actions = max(0, int(exc.physical_actions))
            session.physical_actions += physical_actions
            host._remember(session, exc.evidence)
            if physical_actions > 0:
                # The device has already received the action.  The consumed
                # authority is never recreated; save an uncertain execution
                # fact and continue with a new screenshot instead of replaying.
                host._record_uncertain_execution(session, authority=authority, decision=decision,
                    physical_actions=physical_actions, error=exc)
                host._clear_action(session)
                host._set_status(session, 'failed', str(exc))
            elif _is_zero_action_reobservation_fault(exc):
                host._clear_action(session)
                host._set_status(session, 'needs_reobservation', str(exc).strip() or type(exc).__name__)
            else:
                host._set_status(session, 'failed', str(exc))
            host._best_effort_snapshot(session)
            raise

        def decide_after_action(frames, new_observation, model_decision):
            return host._decide(session, frames=frames, observation=new_observation,
                model_decision=model_decision)

        return self._record_executed_action_result(
            host, session, authority, decision, observation, result, before_actions,
            decide_after_action)

    def _record_executed_action_result(self, host, session, authority, decision, observation,
        result, before_actions, decide_after_action):
        host._remember(session, result.evidence, result.after_frame_paths)
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
        new_observation = host._build_observation(session, scene=result.after_scene, frames=after_frames,
            observation_id=f'obs_{uuid.uuid4().hex}')
        session.history.append({'step_number': session.step_number - 1, 'task_revision': authority.revision,
            'step_id': authority.step_id, 'effect_ids': list(authority.effect_ids),
            'confirmation_receipt': {'authoritative': True, 'consumed': True, 'scope': authority.scope()},
            'qwen_decision': UniversalAgentSessionState._serialize(decision), 'execution': result.to_dict(),
            'before_observation_id': observation.observation_id, 'before_fingerprint': observation.fingerprint,
            'after_observation_id': new_observation.observation_id,
            'after_fingerprint': new_observation.fingerprint})
        return self._record_post_action_decision(
            host, session, authority, decision, observation, result, before_actions,
            physical, outcome, errors, after_frames, new_observation, decide_after_action)

    def _record_post_action_decision(self, host, session, authority, decision, observation,
        result, before_actions, physical, outcome, errors, after_frames, new_observation,
        decide_after_action):
        reject_if(not isinstance(result.after_model_decision, Mapping),
            UniversalAgentOrchestratorError('动作后Qwen观察没有直接返回同响应action/finish。'))
        session.recent_navigation.record_execution(result.resolved_action.kind)
        try:
            next_model_decision = host._record_qwen_reply(session, result.after_model_decision)
            next_decision = decide_after_action(after_frames, new_observation, next_model_decision)
            session.history[-1]["visual_outcome"] = next_decision.previous_action_outcome
            host._remember(session, session.evidence_store.write_verification(session.step_number - 1,
                {'execution_outcome': outcome, 'visual_outcome': next_decision.previous_action_outcome,
                'verification_errors': list(errors), 'after_observation_id': new_observation.observation_id,
                'after_fingerprint': new_observation.fingerprint}))
            host._stage_decision(session, decision=next_decision,
                executed_effect=bool(physical == 1 and authority.effect_ids))
        except Exception as exc:
            # The device action is already dispatched.  A malformed or
            # unbindable post-action Qwen response cannot safely be repaired by
            # replaying the action; retain the frame/evidence and reobserve.
            host._mark_last_execution_uncertain(session, exc)
            host._clear_action(session)
            host._set_status(session, 'failed', str(exc).strip() or type(exc).__name__)
            host._best_effort_snapshot(session)
            raise
        transition = {'protocol_version': POST_ACTION_TRANSITION_PROTOCOL_VERSION,
            'transition_kind': 'new_screenshot_decision', 'execution_outcome': outcome,
            'visual_outcome': next_decision.previous_action_outcome,
            'physical_actions_before': before_actions, 'physical_actions': session.physical_actions,
            'next_decision_status': next_decision.proposal.status,
            'after_observation_id': new_observation.observation_id,
            'after_fingerprint': new_observation.fingerprint}
        session.last_post_action_transition = transition
        host._remember(session, session.evidence_store.write_post_action_transition(session.step_number - 1,
            transition))
        host._write_snapshot(session)
        return result

class AutonomousExecutionCoordinator:
    def run(self, host, session: UniversalAgentSessionState, *, max_physical_actions: int | None=None,
        max_observations: int | None=None) -> dict[str, Any]:
        # Bad caller configuration is not a task failure and must not consume its scope.
        TaskExecutionBudget(
            max_physical_actions=session.execution_budget.max_physical_actions if max_physical_actions is None else max_physical_actions,
            max_observations=session.execution_budget.max_observations if max_observations is None else max_observations)
        reject_if(host.device_registry.active_session(session.device_id) != session.session_id,
            UniversalAgentOrchestratorError('当前会话不再拥有设备。'))
        start_actions = session.physical_actions
        iterations = 0
        session.automatic_loop_enabled = True
        session.auto_pause_reason = ''
        try:
            with host._vision_usage_scope(session.vision_usage), host.device_registry.device_lock(session.device_id):
                session.execution_budget.configure(max_physical_actions=max_physical_actions,
                    max_observations=max_observations)
                if session.status in {'budget_paused', 'paused'}:
                    session.pause_requested.clear()
                    host._set_status(session, 'needs_reobservation')
                while session.status != 'budget_paused':
                    if host._apply_pause_request(session):
                        break
                    if session.status in host.device_registry.TERMINAL_STATUSES:
                        break
                    if session.status == 'awaiting_effect_confirmation':
                        session.auto_pause_reason = '登录或付款目标等待用户确认。'
                        break
                    if session.status == 'paused':
                        # Preserve the reason that caused the recoverable pause;
                        # do not replace it with a generic loop-state message.
                        break
                    if not self._run_active_state_step(host, session):
                        break
                    iterations += 1
                host._apply_pause_request(session)
                session.automatic_loop_enabled = False
                host._write_snapshot(session)
        except Exception as exc:
            session.automatic_loop_enabled = False
            if session.pause_requested.is_set():
                host._apply_pause_request(session)
            elif session.status not in host.device_registry.TERMINAL_STATUSES | {
                'needs_reobservation', 'paused', 'budget_paused'}:
                host._clear_action(session)
                host._set_status(session, 'failed', str(exc).strip() or type(exc).__name__)
            host._best_effort_snapshot(session)
            raise
        finally:
            session.automatic_loop_enabled = False
            host._release_if_terminal(session)
        return {'physical_actions': session.physical_actions - start_actions, 'iterations': iterations,
            'status': session.status, 'pause_reason': session.auto_pause_reason}

    def _run_active_state_step(self, host, session: UniversalAgentSessionState) -> bool:
        if session.status == 'needs_reobservation':
            try:
                host._observe_and_decide(session)
            except GenericActionAdapterError as exc:
                # A transient model/observation failure consumed no device
                # action. Keep the lease and spend the next observation budget
                # on a fresh screenshot.
                if session.status != 'needs_reobservation':
                    raise
                host._remember(session, exc.evidence)
                return True
            return True
        if session.status == 'awaiting_confirmation':
            authority = session.confirmation_authority
            reject_if(authority is None or authority.consumed,
                UniversalAgentOrchestratorError('待执行动作缺少一次性scope。'))
            try:
                host._confirm_one_locked(session, authority.scope())
            except Exception as exc:
                # The action authority is consumed. For both a stale
                # pre-action frame and an uncertain post-action result, the
                # next iteration must only obtain a fresh screenshot; never
                # replay here.
                if session.status != 'needs_reobservation':
                    raise
                host._clear_action(session)
                host._remember(session, getattr(exc, 'evidence', ()))
                return True
            return True
        session.auto_pause_reason = f'当前状态不能自动推进：{session.status}。'
        return False

class SessionLifecycleCoordinator:
    def write_snapshot(self, host, session: UniversalAgentSessionState) -> None:
        if session.vision_usage is not None:
            host._remember(session, session.evidence_store.write_json('qwen_usage.json',
                session.vision_usage.to_dict()))
        host._remember(session, session.evidence_store.write_session(session))
        host._remember(session, session.evidence_store.write_report({'mode': 'qwen_same_response_single_loop',
            'policy_version': CANONICAL_SELECTION_RECEIPT_VERSION, 'session': session.snapshot()}))
    def best_effort_snapshot(self, host, session: UniversalAgentSessionState) -> None:
        try:
            host._write_snapshot(session)
        except Exception:
            pass
    def _build_session(self, host, *, session_id: str, raw_goal: str,
        exact_input_text: str | None, exact_action_kind: str | None,
        exact_target_label: str, device_id: str, run_dir: Path,
        visual_reference_paths: tuple[Path, ...], conversation: tuple[dict[str, str], ...],
        adapter: Any, evidence_store: Any, vision_usage: VisionSessionUsageLedger,
        execution_budget: TaskExecutionBudget) -> UniversalAgentSessionState:
        """Assemble and validate the session before its first observation."""
        visual_paths = tuple(Path(item) for item in visual_reference_paths)
        # Reference assets are an internal visual-prompt input, never goal text or action authority.
        setattr(adapter, "visual_reference_paths", visual_paths)
        session = UniversalAgentSessionState(session_id=session_id,
            raw_goal=str(raw_goal or ''), device_id=device_id, run_dir=Path(run_dir),
            visual_reference_paths=visual_paths,
            conversation=[dict(item) for item in conversation],
            adapter=adapter, evidence_store=evidence_store, vision_usage=vision_usage,
            execution_budget=execution_budget,
            local_exact_input_authority=exact_input_text is not None,
            exact_input_text=exact_input_text,
            exact_action_kind=exact_action_kind or host.required_action_kind,
            exact_target_label=exact_target_label)
        reject_if(not session.session_id or not session.raw_goal or not session.device_id,
            UniversalAgentOrchestratorError('启动Agent需要session_id、目标和device_id。'))
        reject_if(exact_input_text is not None and exact_action_kind is not None,
            UniversalAgentOrchestratorError('exact_input_text与exact_action_kind不能同时使用。'))
        return session
    def start(self, host, *, session_id: str, raw_goal: str, exact_input_text: str | None=None,
        exact_action_kind: str | None=None, exact_target_label: str='', device_id: str,
        run_dir: Path, visual_reference_paths: tuple[Path, ...] = (),
        conversation: tuple[dict[str, str], ...] = (),
        max_physical_actions: int=DEFAULT_DEVICE_ACTION_BUDGET,
        max_observations: int=DEFAULT_OBSERVATION_BUDGET) -> UniversalAgentSessionState:
        budget = TaskExecutionBudget(max_physical_actions=max_physical_actions, max_observations=max_observations)
        resolved_session = str(session_id or '').strip()
        resolved_device = str(device_id or '').strip()
        host.device_registry.reserve(resolved_device, resolved_session)
        session: UniversalAgentSessionState | None = None
        try:
            ledger = VisionSessionUsageLedger(session_id=resolved_session)
            with host.device_registry.device_lock(resolved_device), host._vision_usage_scope(ledger):
                adapter = host.adapter_factory(resolved_device)
                store = host.evidence_store_factory(Path(run_dir))
                session = self._build_session(host, session_id=resolved_session, raw_goal=raw_goal,
                    exact_input_text=exact_input_text, exact_action_kind=exact_action_kind,
                    exact_target_label=exact_target_label, device_id=resolved_device,
                    run_dir=run_dir, visual_reference_paths=visual_reference_paths,
                    conversation=conversation, adapter=adapter, evidence_store=store,
                    vision_usage=ledger, execution_budget=budget)
                try:
                    host._observe_and_decide(session)
                    host._write_snapshot(session)
                except Exception as exc:
                    if session.status not in {'needs_reobservation', 'paused', 'budget_paused'}:
                        host._set_status(session, 'failed', str(exc))
                    host._best_effort_snapshot(session)
                    # Preserve the failed/recoverable session for the API
                    # boundary.  Without this attachment an initial Qwen
                    # contract error disappears before the repository can
                    # expose its evidence and resume state.
                    try:
                        setattr(exc, 'session', session)
                    except Exception:
                        pass
                    raise
        except Exception:
            # A zero-action observation/provider failure is resumable. Keep the
            # device lease with the attached session so the caller can issue a
            # fresh observation instead of receiving a dead session id.
            if session is None or session.status not in {'needs_reobservation', 'paused', 'budget_paused'}:
                host.device_registry.release(resolved_device, resolved_session)
            raise
        host._release_if_terminal(session)
        return session
    def terminate(self, host, session: UniversalAgentSessionState, *, status: str, message: str) -> None:
        try:
            with host.device_registry.device_lock(session.device_id):
                host._clear_action(session)
                if session.effect_confirmation_authority is not None:
                    session.effect_confirmation_authority.consumed = True
                    session.effect_confirmation_authority.invalid_reason = status
                host._set_status(session, status, message)
                host._write_snapshot(session)
        finally:
            host.device_registry.release(session.device_id, session.session_id)
    def apply_pause_request(self, host, session: UniversalAgentSessionState) -> bool:
        if not session.pause_requested.is_set() or session.status in host.device_registry.TERMINAL_STATUSES:
            return False
        host._clear_action(session)
        if session.effect_confirmation_authority is not None:
            session.effect_confirmation_authority.consumed = True
            session.effect_confirmation_authority.invalid_reason = 'paused'
            session.effect_confirmation_authority = None
        session.auto_pause_reason = '用户已暂停；恢复时重新观察。'
        host._set_status(session, 'paused')
        host._write_snapshot(session)
        return True
    def pause(self, host, session: UniversalAgentSessionState) -> None:
        if session.status in host.device_registry.TERMINAL_STATUSES:
            return
        session.pause_requested.set()
        # Signal without waiting behind the lock held by an in-flight action/loop.
        # That operation applies the pause only after recording its real result.
        if session.automatic_loop_enabled or session.status in {'observing', 'executing_one_action'}:
            return
        with host.device_registry.device_lock(session.device_id):
            host._apply_pause_request(session)
    def cancel(self, host, session: UniversalAgentSessionState) -> None:
        host._terminate(session, status='cancelled', message='用户已取消任务。')
    def invalidate_confirmation(self, host, session: UniversalAgentSessionState, *, reason: str) -> None:
        if host.device_registry.active_session(session.device_id) != session.session_id:
            return
        with host.device_registry.device_lock(session.device_id):
            host._clear_action(session)
            host._set_status(session, 'needs_reobservation',
                f'设备状态变化，旧动作已失效：{str(reason or "unknown")}。')
            host._write_snapshot(session)
