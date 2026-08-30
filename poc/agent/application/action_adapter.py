"""Application-facing port for one observed device action."""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple, Protocol


class AppLaunchTarget(NamedTuple):
    """Opaque local launch binding exposed to the application layer."""

    launch_ref: str
    expected_app_id: str


class GenericActionAdapterError(RuntimeError):
    def __init__(self, message: str, *, physical_actions: int=0, evidence: tuple[str, ...]=(),
        observation_errors: tuple[str, ...]=(), verification_errors: tuple[str, ...]=(),
        execution_metadata: Mapping[str, Any] | None=None) -> None:
        super().__init__(message)
        self.physical_actions = int(physical_actions)
        self.evidence = tuple(evidence)
        self.observation_errors = tuple(observation_errors)
        self.verification_errors = tuple(verification_errors)
        self.execution_metadata = dict(execution_metadata or {})


class GenericSingleActionAdapterPort(Protocol):
    """Minimal application contract implemented by the device adapter."""

    def capture_scene(self, *args: Any, **kwargs: Any) -> tuple[Any, ...]: ...

    def resolve_app_launch_target(self, app_id: str, app_name: str) -> AppLaunchTarget | None: ...

    def execute(self, *args: Any, **kwargs: Any) -> Any: ...


__all__ = ["AppLaunchTarget", "GenericActionAdapterError", "GenericSingleActionAdapterPort"]
