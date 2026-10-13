"""Generic Tool/Connector Registry contract.

Canonical action names remain defined by ``action_catalog``.  This registry
adds operational metadata and is the single admission check before execution.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .action_catalog import CANONICAL_ACTION_KINDS
from .validation import reject_if


@dataclass(frozen=True)
class ToolRegistration:
    name: str
    version: str
    effect_category: str
    permissions: frozenset[str]
    idempotency: str
    rollback: str
    simulation_supported: bool
    real_supported: bool
    enabled: bool = True
    health: str = "unknown"

    def validate(self) -> None:
        reject_if(not self.name.strip(), ValueError("Tool Registration 缺少 name。"))
        reject_if(not self.version.strip(), ValueError("Tool Registration 缺少 version。"))
        reject_if(not self.effect_category.strip(), ValueError("Tool Registration 缺少 effect_category。"))
        reject_if(not self.permissions or any(not item.strip() for item in self.permissions),
                  ValueError("Tool Registration permissions 不能为空。"))
        reject_if(not self.idempotency.strip(), ValueError("Tool Registration 缺少 idempotency。"))
        reject_if(not self.rollback.strip(), ValueError("Tool Registration 缺少 rollback。"))


class ToolRegistry:
    """Reject unknown or disabled tools before a device transport is called."""

    def __init__(self, registrations: Iterable[ToolRegistration] = ()) -> None:
        self._registrations: dict[str, ToolRegistration] = {}
        for registration in registrations:
            self.register(registration)

    def register(self, registration: ToolRegistration) -> None:
        registration.validate()
        reject_if(registration.name in self._registrations,
                  ValueError(f"Tool 已注册：{registration.name}"))
        self._registrations[registration.name] = registration

    def require(self, name: str) -> ToolRegistration:
        registration = self._registrations.get(str(name).strip())
        reject_if(registration is None, KeyError(f"未知 Tool：{name}"))
        reject_if(not registration.enabled, ValueError(f"Tool 已禁用：{name}"))
        return registration

    def names(self) -> frozenset[str]:
        return frozenset(self._registrations)

    def registrations(self) -> tuple[ToolRegistration, ...]:
        return tuple(self._registrations[key] for key in sorted(self._registrations))

    def status(self) -> dict[str, Any]:
        return {"tools": [
            {
                "name": item.name,
                "version": item.version,
                "effect_category": item.effect_category,
                "permissions": sorted(item.permissions),
                "idempotency": item.idempotency,
                "rollback": item.rollback,
                "simulation_supported": item.simulation_supported,
                "real_supported": item.real_supported,
                "enabled": item.enabled,
                "health": item.health,
            }
            for item in self.registrations()
        ]}


def build_canonical_tool_registry() -> ToolRegistry:
    """Build registrations from the one canonical action catalogue."""

    registrations = []
    for name in sorted(CANONICAL_ACTION_KINDS):
        physical = name not in {"launch_app", "wait_for_change"}
        registrations.append(ToolRegistration(
            name=name,
            version="canonical-v1",
            effect_category="device_physical" if physical else "device_transport",
            permissions=frozenset({"device.execute"}),
            idempotency="single_step_no_replay",
            rollback="reobserve_or_stop",
            simulation_supported=True,
            real_supported=True,
            health="registered",
        ))
    return ToolRegistry(registrations)


__all__ = ["ToolRegistration", "ToolRegistry", "build_canonical_tool_registry"]
