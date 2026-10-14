"""External identity-provider port for the local Control Plane API."""

from __future__ import annotations

from typing import Any, Mapping

from agent.domain.control_plane import ExternalIdentity, ExternalIdentityProvider


class ExternalIdentityProviderRegistry:
    """Explicit provider registry; no provider is enabled implicitly."""

    def __init__(self) -> None:
        self._providers: dict[str, ExternalIdentityProvider] = {}

    def register(self, provider: ExternalIdentityProvider) -> None:
        provider_id = str(provider.provider_id or "").strip()
        if not provider_id:
            raise ValueError("外部身份 provider_id 不能为空。")
        if provider_id in self._providers:
            raise ValueError(f"外部身份 provider 已注册：{provider_id}")
        self._providers[provider_id] = provider

    def authenticate(self, provider_id: str, assertion: Mapping[str, Any]) -> ExternalIdentity:
        provider = self._providers.get(str(provider_id or "").strip())
        if provider is None:
            raise KeyError("external_provider_not_configured")
        identity = provider.authenticate(assertion)
        if not isinstance(identity, ExternalIdentity):
            raise ValueError("外部身份 provider 必须返回 ExternalIdentity。")
        identity.validate()
        if identity.provider_id != provider.provider_id:
            raise ValueError("外部身份 provider_id 与返回身份不一致。")
        return identity

    def provider_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._providers))


__all__ = ["ExternalIdentityProviderRegistry"]
