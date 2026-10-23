"""Planner boundary between Model Gateway suggestions and runtime authority."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Protocol

from agent.domain.tool_intent import ToolIntent


class PlannerPort(Protocol):
    def plan(
        self,
        *,
        frames: list[Any] | tuple[Any, ...],
        task_context: Any,
        trusted_observation: Any,
        decision_number: int,
        available_action_kinds: Iterable[str],
        model_decision: Mapping[str, Any],
        launch_target: Mapping[str, str] | None = None,
        text_transport_profile: Any = None,
        decision_source: str = "model_gateway",
    ) -> ToolIntent: ...


class ModelAssistedPlanner:
    """Use one model gateway suggestion, then issue the platform intent.

    The wrapped observer is treated as a Model Gateway adapter.  Its result is
    recorded as evidence, while only the returned ``ToolIntent`` is exposed to
    the runtime and executor.
    """

    def __init__(self, model_gateway: Any) -> None:
        self.model_gateway = model_gateway

    def plan(
        self,
        *,
        frames: list[Any] | tuple[Any, ...],
        task_context: Any,
        trusted_observation: Any,
        decision_number: int,
        available_action_kinds: Iterable[str],
        model_decision: Mapping[str, Any],
        launch_target: Mapping[str, str] | None = None,
        text_transport_profile: Any = None,
        decision_source: str = "model_gateway",
    ) -> ToolIntent:
        gateway_decision = self.model_gateway.decide(
            frames=frames,
            task_context=task_context,
            trusted_observation=trusted_observation,
            decision_number=decision_number,
            available_action_kinds=available_action_kinds,
            model_decision=dict(model_decision),
            launch_target=launch_target,
            text_transport_profile=text_transport_profile,
            decision_source=decision_source,
        )
        intent = ToolIntent.from_model_decision(gateway_decision, model_fields=dict(model_decision))
        intent.validate(trusted_observation)
        return intent


__all__ = ["ModelAssistedPlanner", "PlannerPort"]
