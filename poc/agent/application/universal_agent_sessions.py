from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from agent.domain import (
    AgentSession,
    AgentSessionConflictError,
    AgentSessionRepository,
    DeviceTaskRegistryPort,
    require_session_device,
)


class UniversalAgentOrchestratorPort(Protocol):
    device_registry: DeviceTaskRegistryPort

    def start(self, **kwargs: Any) -> AgentSession: ...

    def approve_effects( self, session: AgentSession, confirmation: Mapping[str, Any],
    ) -> Any: ...

    def confirm_one( self, session: AgentSession, confirmation: Mapping[str, Any],
    ) -> Any: ...

    def refresh_decision(self, session: AgentSession) -> Any: ...

    def run_autonomous_safe_loop( self, session: AgentSession, *, max_physical_actions: int, max_iterations: int,
    ) -> dict[str, Any]: ...

    def invalidate_confirmation( self, session: AgentSession, *, reason: str,
    ) -> None: ...

    def cancel(self, session: AgentSession) -> None: ...

    def pause(self, session: AgentSession) -> None: ...


class AgentSessionCommandError(RuntimeError):
    """Application-level rejection that does not execute a physical action."""


class AgentDeviceRuntimeError(RuntimeError):
    """Transport-neutral device readiness or exclusive-control failure."""

    def __init__(self, detail: Any, *, status_code: int=409) -> None:
        super().__init__(str(detail))
        self.detail = detail
        self.status_code = status_code


@dataclass(frozen=True)
class StartUniversalAgentSessionCommand:
    session_id: str
    raw_goal: str
    exact_input_text: str | None
    exact_action_kind: str | None
    exact_target_label: str
    device_id: str
    run_dir: Path
    auto_advance: bool


@dataclass(frozen=True)
class StartUniversalAgentSessionResult:
    session: AgentSession
    automatic_progress: dict[str, Any]


@dataclass(frozen=True)
class AgentSessionOperationResult:
    session: AgentSession
    operation: Any = None
    physical_actions: int = 0


class UniversalAgentSessionApplicationService:
    """One application boundary for the universal-agent session lifecycle."""

    def __init__(self, *, orchestrator_provider: Callable[[], UniversalAgentOrchestratorPort],
        sessions: AgentSessionRepository, ensure_device_ready: Callable[[str], None],
        exclusive_device_session: Callable[[str], AbstractContextManager[None]], begin_new_task: Callable[[str],
        None]) -> None:
        self._orchestrator_provider = orchestrator_provider
        self._sessions = sessions
        self._ensure_device_ready = ensure_device_ready
        self._exclusive_device_session = exclusive_device_session
        self._begin_new_task = begin_new_task

    def _orchestrator(self) -> UniversalAgentOrchestratorPort:
        return self._orchestrator_provider()

    @staticmethod
    def _require_start_available(orchestrator: UniversalAgentOrchestratorPort, device_id: str) -> None:
        active_session_id = orchestrator.device_registry.active_session(device_id)
        if active_session_id is not None:
            raise AgentSessionConflictError(f'设备 {device_id} 已有活动任务：{active_session_id}。')

    def require(self, session_id: str) -> AgentSession:
        return self._sessions.require(session_id)

    def active_snapshots(self) -> list[dict[str, Any]]:
        return self._sessions.active_snapshots()

    def start(self, command: StartUniversalAgentSessionCommand) -> StartUniversalAgentSessionResult:
        orchestrator = self._orchestrator()
        self._require_start_available(orchestrator, command.device_id)
        self._ensure_device_ready(command.device_id)
        command.run_dir.mkdir(parents=True, exist_ok=True)
        with self._exclusive_device_session(command.device_id):
            self._require_start_available(orchestrator, command.device_id)
            self._begin_new_task(command.device_id)
            session = orchestrator.start(session_id=command.session_id, raw_goal=command.raw_goal,
                exact_input_text=command.exact_input_text, exact_action_kind=command.exact_action_kind,
                exact_target_label=command.exact_target_label, device_id=command.device_id, run_dir=command.run_dir)
        self._sessions.add(session)
        automatic_progress = {'physical_actions': 0, 'iterations': 0, 'status': session.status,
            'pause_reason': '当前没有可自动推进的安全动作。'}
        if command.auto_advance and session.status in {'awaiting_confirmation', 'needs_reobservation'}:
            with self._exclusive_device_session(command.device_id):
                automatic_progress = orchestrator.run_autonomous_safe_loop(session, max_physical_actions=12,
                    max_iterations=24)
        return StartUniversalAgentSessionResult(session=session, automatic_progress=automatic_progress)

    def approve_effects(self, session: AgentSession, *, confirmed: bool, confirmation: Mapping[str,
        Any] | None) -> AgentSessionOperationResult:
        self._ensure_device_ready(session.device_id)
        before_actions = session.physical_actions
        if confirmed is not True or confirmation is None:
            raise AgentSessionCommandError('调用 Qwen 处理受限效果前必须确认完整效果作用域。')
        require_session_device(session, str(confirmation.get('device_id') or ''))
        orchestrator = self._orchestrator()
        with self._exclusive_device_session(session.device_id):
            result = orchestrator.approve_effects(session, confirmation)
        physical_actions = session.physical_actions - before_actions
        if physical_actions not in {0, 1}:
            raise AgentSessionCommandError("一次效果确认产生了超过一个物理动作。")
        return AgentSessionOperationResult(session=session, operation=result, physical_actions=physical_actions)

    def confirm(self, session: AgentSession, *, confirmed: bool, confirmation: Mapping[str,
        Any] | None) -> AgentSessionOperationResult:
        orchestrator = self._orchestrator()
        try:
            self._ensure_device_ready(session.device_id)
        except AgentDeviceRuntimeError:
            try:
                orchestrator.invalidate_confirmation(session, reason='device_readiness_failed')
            except Exception:
                pass
            raise
        before_actions = session.physical_actions
        if confirmed is not True or confirmation is None:
            raise AgentSessionCommandError('执行一个动作前必须提交完整且明确的确认作用域。')
        with self._exclusive_device_session(session.device_id):
            result = orchestrator.confirm_one(session, confirmation)
        return AgentSessionOperationResult(session=session, operation=result, physical_actions=max(0,
            session.physical_actions - before_actions))

    def refresh(self, session: AgentSession, *, requested_device_id: str) -> AgentSessionOperationResult:
        require_session_device(session, requested_device_id)
        self._ensure_device_ready(session.device_id)
        before_actions = session.physical_actions
        with self._exclusive_device_session(session.device_id):
            decision = self._orchestrator().refresh_decision(session)
        return AgentSessionOperationResult(session=session, operation=decision, physical_actions=max(0,
            session.physical_actions - before_actions))

    def run_automatic(self, session: AgentSession, *, requested_device_id: str, confirmed: bool,
        confirmation: Mapping[str, Any] | None, max_physical_actions: int,
        max_iterations: int) -> AgentSessionOperationResult:
        require_session_device(session, requested_device_id)
        orchestrator = self._orchestrator()
        try:
            self._ensure_device_ready(session.device_id)
        except AgentDeviceRuntimeError:
            try:
                orchestrator.invalidate_confirmation(session, reason='device_readiness_failed')
            except Exception:
                pass
            raise
        before_actions = session.physical_actions
        if confirmed is True or confirmation is not None:
            raise AgentSessionCommandError('安全自动推进不接收用户动作确认；外部影响请使用风险确认接口。')
        with self._exclusive_device_session(session.device_id):
            result = orchestrator.run_autonomous_safe_loop(session, max_physical_actions=max_physical_actions,
                max_iterations=max_iterations)
        return AgentSessionOperationResult(session=session, operation=result, physical_actions=max(0,
            session.physical_actions - before_actions))

    def cancel(self, session_id: str, *, requested_device_id: str) -> AgentSessionOperationResult:
        with self._sessions.locked(session_id) as session:
            require_session_device(session, requested_device_id)
            self._orchestrator().cancel(session)
            return AgentSessionOperationResult(session=session)

    def pause(self, session_id: str, *, requested_device_id: str) -> AgentSessionOperationResult:
        with self._sessions.locked(session_id) as session:
            require_session_device(session, requested_device_id)
            self._orchestrator().pause(session)
            return AgentSessionOperationResult(session=session)
