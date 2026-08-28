"""Domain port for authoritative universal-agent session evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Protocol


class EvidenceStoreError(RuntimeError):
    """Raised when authoritative session evidence cannot be persisted."""


class AgentEvidenceStorePort(Protocol):
    """Persist the authoritative artifacts emitted by one agent session."""

    run_dir: Path

    def write_json(self, name: str, payload: Any) -> Path: ...

    def write_session(self, session: Any) -> Path: ...

    def write_task_graph(self, graph: Any) -> Path: ...

    def write_effect_policy_snapshot(self, graph: Any) -> Path: ...

    def write_trusted_observation( self, step_number: int, observation: Any,
    ) -> Path: ...

    def write_qwen_decision(self, step_number: int, decision: Any) -> Path: ...

    def write_controller_decision( self, step_number: int, decision: Any,
    ) -> Path: ...

    def write_verification( self, step_number: int, verification: Any,
    ) -> Path: ...

    def write_post_action_transition( self, step_number: int, transition: Any,
    ) -> Path: ...

    def write_confirmation_failure( self, step_number: int, transition: Any,
    ) -> Path: ...

    def read_report(self) -> dict[str, Any] | None: ...

    def write_report(self, report: Any) -> Path: ...


AgentEvidenceStoreFactory = Callable[[Path], AgentEvidenceStorePort]
