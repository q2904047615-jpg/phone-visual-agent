"""GitHub OAuth token verification and trusted local identity mapping.

The Control Plane receives an OAuth access token, verifies it server-to-server
against GitHub's fixed Check a token endpoint, and maps GitHub's stable numeric
user id to an already provisioned local principal.  No provider claims or
client-supplied organization/role fields are trusted for authorization.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Mapping, MutableMapping
from urllib.parse import quote, urlencode
import base64
import hashlib

import httpx

from agent.domain.control_plane import ExternalIdentity


GITHUB_PROVIDER_ID = "github"
GITHUB_API_BASE_URL = "https://api.github.com"
GITHUB_API_VERSION = "2026-03-10"
GITHUB_AUTHORIZE_URL = "https://github.com/login/oauth/authorize"
GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
DEFAULT_GITHUB_REDIRECT_URI = "http://127.0.0.1:8765/api/control-plane/auth/github/callback"
GITHUB_OAUTH_SCOPE = "read:user"
_MAX_ACCESS_TOKEN_LENGTH = 512
_MAX_AUTHORIZATION_CODE_LENGTH = 1024


class GitHubOAuthConfigurationError(ValueError):
    """Raised when the provider is partially or incorrectly configured."""


class ExternalIdentityProviderUnavailable(RuntimeError):
    """Raised when GitHub cannot be reached or returns an upstream failure."""


class ExternalIdentityMappingNotFound(LookupError):
    """Raised when a verified provider subject has no trusted local mapping."""


@dataclass(frozen=True)
class GitHubIdentityMapping:
    organization_id: str
    user_id: str


def _read_mapping_file(path: Path) -> dict[str, GitHubIdentityMapping]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GitHubOAuthConfigurationError("GitHub OAuth identity mapping file 无法读取。") from exc
    if not isinstance(raw, Mapping):
        raise GitHubOAuthConfigurationError("GitHub OAuth identity mapping 必须是 JSON 对象。")
    # Keep the file intentionally small and explicit: {"github_user_id":
    # {"organization_id": "...", "user_id": "..."}}.  Claims from GitHub
    # never populate this table automatically.
    mappings = raw.get("mappings", raw)
    if not isinstance(mappings, Mapping):
        raise GitHubOAuthConfigurationError("GitHub OAuth mappings 必须是对象。")
    result: dict[str, GitHubIdentityMapping] = {}
    for subject, value in mappings.items():
        subject_text = str(subject).strip()
        if not subject_text or not subject_text.isdigit():
            raise GitHubOAuthConfigurationError("GitHub OAuth mapping subject 必须是数字用户 ID。")
        if not isinstance(value, Mapping):
            raise GitHubOAuthConfigurationError("GitHub OAuth mapping 条目必须是对象。")
        try:
            organization_id = str(value["organization_id"]).strip()
            user_id = str(value["user_id"]).strip()
        except KeyError as exc:
            raise GitHubOAuthConfigurationError("GitHub OAuth mapping 缺少 organization_id 或 user_id。") from exc
        result[subject_text] = GitHubIdentityMapping(organization_id, user_id)
    if not result:
        raise GitHubOAuthConfigurationError("GitHub OAuth mapping 不能为空。")
    return result


class GitHubOAuthProvider:
    """Verify GitHub OAuth tokens and map verified subjects to local users."""

    provider_id = GITHUB_PROVIDER_ID

    def __init__(
        self,
        *,
        client_id: str,
        client_secret: str,
        mappings: Mapping[str, GitHubIdentityMapping],
        transport: httpx.BaseTransport | None = None,
        timeout_seconds: float = 10.0,
        base_url: str = GITHUB_API_BASE_URL,
        redirect_uri: str = DEFAULT_GITHUB_REDIRECT_URI,
    ) -> None:
        self.client_id = str(client_id or "").strip()
        self.client_secret = str(client_secret or "")
        if not self.client_id or not self.client_secret:
            raise GitHubOAuthConfigurationError("GitHub OAuth client_id 和 client_secret 必须同时配置。")
        if base_url != GITHUB_API_BASE_URL and transport is None:
            raise GitHubOAuthConfigurationError("GitHub OAuth endpoint 必须使用固定 api.github.com。")
        self.redirect_uri = str(redirect_uri or "").strip()
        if not self.redirect_uri or not self.redirect_uri.startswith(("http://", "https://")) \
                or "?" in self.redirect_uri or "#" in self.redirect_uri:
            raise GitHubOAuthConfigurationError("GitHub OAuth redirect_uri 必须是无查询参数的 HTTP(S) URL。")
        self._mappings = dict(mappings)
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout_seconds, connect=min(5.0, timeout_seconds)),
            follow_redirects=False,
            transport=transport,
        )
        self._oauth_client = httpx.Client(
            base_url="https://github.com",
            timeout=httpx.Timeout(timeout_seconds, connect=min(5.0, timeout_seconds)),
            follow_redirects=False,
            transport=transport,
        )

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> "GitHubOAuthProvider | None":
        values = os.environ if environ is None else environ
        client_id = str(values.get("GITHUB_OAUTH_CLIENT_ID", "")).strip()
        client_secret = str(values.get("GITHUB_OAUTH_CLIENT_SECRET", ""))
        mapping_path = str(values.get("GITHUB_OAUTH_IDENTITY_MAP_FILE", "")).strip()
        redirect_uri = str(values.get("GITHUB_OAUTH_REDIRECT_URI", DEFAULT_GITHUB_REDIRECT_URI)).strip()
        if not client_id and not client_secret and not mapping_path:
            return None
        if not client_id or not client_secret or not mapping_path:
            raise GitHubOAuthConfigurationError(
                "GitHub OAuth 需要同时配置 GITHUB_OAUTH_CLIENT_ID、GITHUB_OAUTH_CLIENT_SECRET 和 GITHUB_OAUTH_IDENTITY_MAP_FILE。"
            )
        return cls(
            client_id=client_id,
            client_secret=client_secret,
            mappings=_read_mapping_file(Path(mapping_path)),
            transport=transport,
            redirect_uri=redirect_uri,
        )

    def close(self) -> None:
        self._client.close()
        self._oauth_client.close()

    def authorization_url(self, *, state: str, code_challenge: str) -> str:
        if not state or len(state) > 256:
            raise ValueError("GitHub OAuth state 无效。")
        if not code_challenge or len(code_challenge) < 43 or len(code_challenge) > 128:
            raise ValueError("GitHub OAuth code_challenge 无效。")
        return GITHUB_AUTHORIZE_URL + "?" + urlencode({
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": GITHUB_OAUTH_SCOPE,
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "allow_signup": "false",
        })

    @staticmethod
    def pkce_challenge(code_verifier: str) -> str:
        if not isinstance(code_verifier, str) or len(code_verifier) < 43 or len(code_verifier) > 128:
            raise ValueError("GitHub OAuth code_verifier 无效。")
        return base64.urlsafe_b64encode(hashlib.sha256(code_verifier.encode("ascii")).digest()).rstrip(b"=").decode("ascii")

    def exchange_code(self, *, code: str, code_verifier: str) -> str:
        code = str(code or "").strip()
        if not code or len(code) > _MAX_AUTHORIZATION_CODE_LENGTH:
            raise ValueError("GitHub OAuth authorization code 无效。")
        if not isinstance(code_verifier, str) or len(code_verifier) < 43 or len(code_verifier) > 128:
            raise ValueError("GitHub OAuth code_verifier 无效。")
        try:
            response = self._oauth_client.post(
                "/login/oauth/access_token",
                headers={"Accept": "application/json"},
                json={
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "code": code,
                    "redirect_uri": self.redirect_uri,
                    "code_verifier": code_verifier,
                },
            )
        except httpx.TransportError as exc:
            raise ExternalIdentityProviderUnavailable("GitHub OAuth provider 不可达。") from exc
        if response.status_code >= 500:
            raise ExternalIdentityProviderUnavailable("GitHub OAuth provider 返回服务故障。")
        if response.status_code != 200:
            raise ValueError("GitHub OAuth authorization code 未通过验证。")
        try:
            payload = response.json()
        except ValueError as exc:
            raise ExternalIdentityProviderUnavailable("GitHub OAuth provider 返回了无效响应。") from exc
        if not isinstance(payload, Mapping):
            raise ExternalIdentityProviderUnavailable("GitHub OAuth provider 返回了无效响应。")
        token = payload.get("access_token")
        if not isinstance(token, str) or not token.strip() or len(token) > _MAX_ACCESS_TOKEN_LENGTH:
            raise ValueError("GitHub OAuth authorization code 未换取有效 token。")
        return token.strip()

    @staticmethod
    def _access_token(assertion: Mapping[str, Any]) -> str:
        if not isinstance(assertion, Mapping):
            raise ValueError("GitHub OAuth assertion 必须是对象。")
        unknown = set(assertion) - {"access_token"}
        if unknown:
            raise ValueError("GitHub OAuth assertion 含有未支持字段。")
        token = assertion.get("access_token")
        if not isinstance(token, str) or not token.strip() or len(token) > _MAX_ACCESS_TOKEN_LENGTH:
            raise ValueError("GitHub OAuth access_token 无效。")
        return token.strip()

    def _check_token(self, access_token: str) -> Mapping[str, Any]:
        path = f"/applications/{quote(self.client_id, safe='')}/token"
        try:
            response = self._client.post(
                path,
                auth=(self.client_id, self.client_secret),
                headers={
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": GITHUB_API_VERSION,
                },
                json={"access_token": access_token},
            )
        except httpx.TransportError as exc:
            raise ExternalIdentityProviderUnavailable("GitHub OAuth provider 不可达。") from exc
        if response.status_code in {404, 422}:
            raise ValueError("GitHub OAuth access_token 未通过验证。")
        if response.status_code >= 500:
            raise ExternalIdentityProviderUnavailable("GitHub OAuth provider 返回服务故障。")
        if response.status_code != 200:
            raise ValueError("GitHub OAuth access_token 未通过验证。")
        try:
            payload = response.json()
        except ValueError as exc:
            raise ExternalIdentityProviderUnavailable("GitHub OAuth provider 返回了无效响应。") from exc
        if not isinstance(payload, Mapping):
            raise ExternalIdentityProviderUnavailable("GitHub OAuth provider 返回了无效响应。")
        return payload

    @staticmethod
    def _check_expiry(payload: Mapping[str, Any]) -> None:
        expires_at = payload.get("expires_at")
        if not expires_at:
            return
        if not isinstance(expires_at, str):
            raise ValueError("GitHub OAuth access_token 过期字段无效。")
        try:
            expiry = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("GitHub OAuth access_token 过期字段无效。") from exc
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)
        if expiry <= datetime.now(timezone.utc):
            raise ValueError("GitHub OAuth access_token 已过期。")

    def authenticate(self, assertion: Mapping[str, Any]) -> ExternalIdentity:
        access_token = self._access_token(assertion)
        payload = self._check_token(access_token)
        app = payload.get("app")
        user = payload.get("user")
        if not isinstance(app, Mapping) or app.get("client_id") != self.client_id:
            raise ValueError("GitHub OAuth token 不属于已配置应用。")
        if not isinstance(user, Mapping) or isinstance(user.get("id"), bool) or not isinstance(user.get("id"), int):
            raise ValueError("GitHub OAuth provider 未返回稳定用户 ID。")
        self._check_expiry(payload)
        subject = str(user["id"])
        mapping = self._mappings.get(subject)
        if mapping is None:
            raise ExternalIdentityMappingNotFound("GitHub OAuth 用户尚未完成身份映射。")
        # Keep claims limited to non-authoritative display/audit facts.  The
        # local role, organization and permissions always come from SQLite.
        claims: MutableMapping[str, Any] = {"login": str(user.get("login", ""))[:128]}
        scopes = payload.get("scopes")
        if isinstance(scopes, list) and all(isinstance(item, str) for item in scopes):
            claims["scopes"] = list(scopes)
        return ExternalIdentity(
            provider_id=self.provider_id,
            subject=subject,
            organization_id=mapping.organization_id,
            user_id=mapping.user_id,
            claims=claims,
        )


__all__ = [
    "ExternalIdentityMappingNotFound",
    "ExternalIdentityProviderUnavailable",
    "GitHubIdentityMapping",
    "GitHubOAuthConfigurationError",
    "GitHubOAuthProvider",
    "GITHUB_API_BASE_URL",
    "GITHUB_API_VERSION",
    "GITHUB_AUTHORIZE_URL",
    "GITHUB_TOKEN_URL",
    "DEFAULT_GITHUB_REDIRECT_URI",
    "GITHUB_OAUTH_SCOPE",
    "GITHUB_PROVIDER_ID",
]
