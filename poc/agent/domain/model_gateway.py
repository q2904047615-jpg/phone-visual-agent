"""Generic Model Gateway contracts and registry.

The registry owns provider selection.  A provider can suggest observations or
plans, but it never owns canonical actions or task terminal state.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from .validation import reject_if


class ModelGateway(Protocol):
    """Minimal provider adapter consumed by Planner."""

    def decide(self, **kwargs: Any) -> Any: ...

    def status(self) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class ModelGatewayRegistration:
    gateway_id: str
    model: str
    capabilities: frozenset[str]
    version: str
    gateway: ModelGateway
    enabled: bool = True

    def validate(self) -> None:
        reject_if(not self.gateway_id.strip(), ValueError("Model Gateway 缺少 gateway_id。"))
        reject_if(not self.model.strip(), ValueError("Model Gateway 缺少 model。"))
        reject_if(not self.version.strip(), ValueError("Model Gateway 缺少 version。"))
        reject_if(not self.capabilities or any(not item.strip() for item in self.capabilities),
                  ValueError("Model Gateway capabilities 不能为空。"))
        reject_if(not callable(getattr(self.gateway, "decide", None)),
                  ValueError("Model Gateway 缺少 decide 合同。"))
        reject_if(not callable(getattr(self.gateway, "status", None)),
                  ValueError("Model Gateway 缺少 status 合同。"))


class ModelGatewayRegistry:
    """Deterministic provider registry used by the Runtime composition root."""

    def __init__(self, registrations: Iterable[ModelGatewayRegistration] = ()) -> None:
        self._registrations: dict[str, ModelGatewayRegistration] = {}
        self._default_id: str | None = None
        for registration in registrations:
            self.register(registration)

    def register(self, registration: ModelGatewayRegistration, *, default: bool = False) -> None:
        registration.validate()
        reject_if(registration.gateway_id in self._registrations,
                  ValueError(f"Model Gateway 已注册：{registration.gateway_id}"))
        self._registrations[registration.gateway_id] = registration
        if default:
            self.set_default(registration.gateway_id)

    def set_default(self, gateway_id: str) -> None:
        registration = self.require_registration(gateway_id)
        reject_if(not registration.enabled, ValueError(f"Model Gateway 已禁用：{gateway_id}"))
        self._default_id = gateway_id

    def require_registration(self, gateway_id: str) -> ModelGatewayRegistration:
        registration = self._registrations.get(str(gateway_id).strip())
        reject_if(registration is None, KeyError(f"未知 Model Gateway：{gateway_id}"))
        return registration

    def default_registration(self) -> ModelGatewayRegistration:
        reject_if(self._default_id is None, RuntimeError("尚未配置默认 Model Gateway。"))
        return self.require_registration(self._default_id)

    def default(self) -> ModelGateway:
        return self.default_registration().gateway

    def registrations(self) -> tuple[ModelGatewayRegistration, ...]:
        return tuple(self._registrations[key] for key in sorted(self._registrations))

    def status(self) -> dict[str, Any]:
        return {
            "default_gateway_id": self._default_id,
            "gateways": [
                {
                    "gateway_id": item.gateway_id,
                    "model": item.model,
                    "version": item.version,
                    "capabilities": sorted(item.capabilities),
                    "enabled": item.enabled,
                    "provider_status": dict(item.gateway.status()),
                }
                for item in self.registrations()
            ],
        }


__all__ = ["ModelGateway", "ModelGatewayRegistration", "ModelGatewayRegistry"]
