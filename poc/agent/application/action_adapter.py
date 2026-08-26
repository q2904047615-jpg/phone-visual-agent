"""Application-facing port for one observed device action."""

from __future__ import annotations

from typing import Any, Protocol


class GenericActionAdapterError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        physical_actions: int = 0,
        evidence: tuple[str, ...] = (),
        observation_errors: tuple[str, ...] = (),
        verification_errors: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.physical_actions = int(physical_actions)
        self.evidence = tuple(evidence)
        self.observation_errors = tuple(observation_errors)
        self.verification_errors = tuple(verification_errors)


class GenericSingleActionAdapterPort(Protocol):
    """Minimal application contract implemented by the device adapter."""

    def capture_scene(self, *args: Any, **kwargs: Any) -> tuple[Any, ...]: ...

    def execute(self, *args: Any, **kwargs: Any) -> Any: ...


__all__ = ["GenericActionAdapterError", "GenericSingleActionAdapterPort"]
