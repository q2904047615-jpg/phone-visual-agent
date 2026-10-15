"""Control Plane contracts for the GA transition.

These contracts define organization, team, permission, quota, API and SDK
semantics.  The current online integration is a read-only local projection;
they are intentionally not Runtime admission or Policy authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Mapping, Protocol


CONTROL_PLANE_PROTOCOL = "2026-10-14-control-plane-v1"
API_CONTRACT_VERSION = "2026-10-14-api-v1"
SDK_CONTRACT_VERSION = "2026-10-14-sdk-v1"
EXTERNAL_IDENTITY_PROTOCOL = "2026-10-14-external-identity-v1"
GITHUB_OAUTH_BROWSER_PROTOCOL = "2026-10-14-github-oauth-browser-v1"
_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_ROLES = frozenset({"viewer", "operator", "admin", "owner"})
_PERMISSIONS = frozenset({
    "task.read", "task.create", "task.cancel", "device.read", "device.execute",
    "audit.read", "team.manage", "identity.manage", "device.manage", "quota.read", "quota.manage", "billing.read",
})
ROLE_PERMISSIONS: Mapping[str, frozenset[str]] = {
    "viewer": frozenset({"task.read", "device.read", "audit.read"}),
    "operator": frozenset({"task.read", "task.create", "task.cancel", "device.read", "device.execute", "audit.read"}),
    "admin": frozenset({"task.read", "task.create", "task.cancel", "device.read", "device.execute", "device.manage", "audit.read", "team.manage", "identity.manage", "quota.read", "quota.manage"}),
    "owner": _PERMISSIONS,
}
QUOTA_PERIODS = frozenset({"daily", "monthly"})


def manageable_member_roles(actor_role: str, target_role: str | None = None) -> tuple[str, ...]:
    """The single authority for member administration (approved 2026-10-15)."""
    if actor_role == "owner":
        return ("viewer", "operator", "admin", "owner")
    if actor_role == "admin" and target_role in {None, "viewer", "operator"}:
        return ("viewer", "operator")
    return ()


def validate_github_subject(subject: str) -> str:
    if not isinstance(subject, str) or not re.fullmatch(r"[1-9][0-9]{0,19}", subject):
        raise ValueError("github_subject_invalid")
    return subject


MUTATING_OPERATIONS = frozenset({
    "task.create", "task.cancel", "team.create", "team.update", "principal.create", "principal.update", "quota.update", "device.assign", "github_identity.update",
})


def _require_id(value: str, field_name: str) -> str:
    resolved = str(value or "").strip()
    if not _ID_PATTERN.fullmatch(resolved):
        raise ValueError(f"{field_name} 必须是 1-128 位稳定 ID。")
    return resolved


def _require_nonnegative(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} 必须是非负整数。")
    return value


@dataclass(frozen=True)
class ExternalIdentity:
    """Identity returned only after an external provider verifies an assertion."""

    provider_id: str
    subject: str
    organization_id: str
    user_id: str
    claims: Mapping[str, Any] = field(default_factory=dict)
    protocol_version: str = EXTERNAL_IDENTITY_PROTOCOL

    def validate(self) -> None:
        if self.protocol_version != EXTERNAL_IDENTITY_PROTOCOL:
            raise ValueError("外部身份协议版本不受支持。")
        _require_id(self.provider_id, "provider_id")
        _require_id(self.subject, "subject")
        _require_id(self.organization_id, "organization_id")
        _require_id(self.user_id, "user_id")
        if not isinstance(self.claims, Mapping):
            raise ValueError("外部身份 claims 必须是对象。")


class ExternalIdentityProvider(Protocol):
    provider_id: str

    def authenticate(self, assertion: Mapping[str, Any]) -> ExternalIdentity:
        """Verify an external assertion and return a normalized identity."""


@dataclass(frozen=True)
class ControlPlanePrincipal:
    user_id: str
    organization_id: str
    role: str
    team_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_id(self.user_id, "user_id")
        _require_id(self.organization_id, "organization_id")
        if self.role not in _ROLES:
            raise ValueError(f"未知角色：{self.role}")
        normalized = tuple(_require_id(item, "team_id") for item in self.team_ids)
        if len(set(normalized)) != len(normalized):
            raise ValueError("team_ids 不能重复。")
        object.__setattr__(self, "team_ids", normalized)

    def permissions(self) -> frozenset[str]:
        return ROLE_PERMISSIONS[self.role]

    def can(self, permission: str) -> bool:
        resolved = str(permission or "").strip()
        if resolved not in _PERMISSIONS:
            raise ValueError(f"未知权限：{permission}")
        return resolved in self.permissions()


@dataclass(frozen=True)
class ControlPlaneTeam:
    team_id: str
    organization_id: str
    name: str
    member_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_id(self.team_id, "team_id")
        _require_id(self.organization_id, "organization_id")
        if not str(self.name or "").strip():
            raise ValueError("团队名称不能为空。")
        normalized = tuple(_require_id(item, "member_id") for item in self.member_ids)
        if len(set(normalized)) != len(normalized):
            raise ValueError("member_ids 不能重复。")
        object.__setattr__(self, "member_ids", normalized)


@dataclass(frozen=True)
class ControlPlaneOrganization:
    organization_id: str
    name: str
    owner_user_id: str
    team_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_id(self.organization_id, "organization_id")
        _require_id(self.owner_user_id, "owner_user_id")
        if not str(self.name or "").strip():
            raise ValueError("组织名称不能为空。")
        normalized = tuple(_require_id(item, "team_id") for item in self.team_ids)
        if len(set(normalized)) != len(normalized):
            raise ValueError("组织 team_ids 不能重复。")
        object.__setattr__(self, "team_ids", normalized)


@dataclass(frozen=True)
class QuotaLimit:
    period: str
    max_runs: int | None = None
    max_physical_actions: int | None = None
    max_model_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.period not in QUOTA_PERIODS:
            raise ValueError(f"quota period 必须是 daily 或 monthly：{self.period}")
        for field_name in ("max_runs", "max_physical_actions", "max_model_tokens"):
            value = getattr(self, field_name)
            if value is not None:
                _require_nonnegative(value, field_name)
        if all(getattr(self, name) is None for name in ("max_runs", "max_physical_actions", "max_model_tokens")):
            raise ValueError("配额至少需要一个上限。")


@dataclass(frozen=True)
class QuotaUsage:
    runs: int = 0
    physical_actions: int = 0
    model_tokens: int = 0

    def __post_init__(self) -> None:
        for name in ("runs", "physical_actions", "model_tokens"):
            _require_nonnegative(getattr(self, name), name)


@dataclass(frozen=True)
class QuotaDecision:
    allowed: bool
    reason: str
    remaining: Mapping[str, int | None]


def evaluate_quota(limit: QuotaLimit, usage: QuotaUsage, *, requested_runs: int = 0,
                   requested_physical_actions: int = 0, requested_model_tokens: int = 0) -> QuotaDecision:
    """Return a pure offline quota decision; it does not mutate usage or block Runtime."""

    requested = QuotaUsage(
        runs=_require_nonnegative(requested_runs, "requested_runs"),
        physical_actions=_require_nonnegative(requested_physical_actions, "requested_physical_actions"),
        model_tokens=_require_nonnegative(requested_model_tokens, "requested_model_tokens"),
    )
    values = {
        "runs": (limit.max_runs, usage.runs + requested.runs),
        "physical_actions": (limit.max_physical_actions, usage.physical_actions + requested.physical_actions),
        "model_tokens": (limit.max_model_tokens, usage.model_tokens + requested.model_tokens),
    }
    remaining = {name: None if ceiling is None else max(0, ceiling - current) for name, (ceiling, current) in values.items()}
    exceeded = [name for name, (ceiling, current) in values.items() if ceiling is not None and current > ceiling]
    return QuotaDecision(not exceeded, "within_quota" if not exceeded else f"quota_exceeded:{','.join(exceeded)}", remaining)


@dataclass(frozen=True)
class ControlPlaneSnapshot:
    organization: ControlPlaneOrganization
    teams: tuple[ControlPlaneTeam, ...]
    principals: tuple[ControlPlanePrincipal, ...]
    quotas: Mapping[str, QuotaLimit] = field(default_factory=dict)

    def validate(self) -> None:
        organization_id = self.organization.organization_id
        teams_by_id = {team.team_id: team for team in self.teams}
        if len(teams_by_id) != len(self.teams):
            raise ValueError("团队 ID 不能重复。")
        principals_by_id = {principal.user_id: principal for principal in self.principals}
        if len(principals_by_id) != len(self.principals):
            raise ValueError("用户 ID 不能重复。")
        if self.organization.owner_user_id not in principals_by_id:
            raise ValueError("组织 owner_user_id 必须对应一个用户。")
        if principals_by_id[self.organization.owner_user_id].role != "owner":
            raise ValueError("组织 owner 必须拥有 owner 角色。")
        if any(team.organization_id != organization_id for team in self.teams):
            raise ValueError("团队不能跨组织归属。")
        if any(principal.organization_id != organization_id for principal in self.principals):
            raise ValueError("用户不能跨组织归属。")
        if set(self.organization.team_ids) != set(teams_by_id):
            raise ValueError("组织 team_ids 必须与团队集合一致。")
        for principal in self.principals:
            if any(team_id not in teams_by_id for team_id in principal.team_ids):
                raise ValueError(f"用户 {principal.user_id} 引用了未知团队。")
        for quota_key, quota in self.quotas.items():
            _require_id(quota_key, "quota_key")
            if not isinstance(quota, QuotaLimit):
                raise ValueError("quotas 的值必须是 QuotaLimit。")


@dataclass(frozen=True)
class ApiRequestEnvelope:
    request_id: str
    idempotency_key: str
    organization_id: str
    user_id: str
    operation: str
    body: Mapping[str, Any]
    api_version: str = API_CONTRACT_VERSION

    def validate(self) -> None:
        if self.api_version != API_CONTRACT_VERSION:
            raise ValueError("API version 不受支持。")
        for value, name in ((self.request_id, "request_id"), (self.idempotency_key, "idempotency_key"),
                            (self.organization_id, "organization_id"), (self.user_id, "user_id")):
            _require_id(value, name)
        if not str(self.operation or "").strip():
            raise ValueError("operation 不能为空。")
        if not isinstance(self.body, Mapping):
            raise ValueError("API body 必须是对象。")
        if self.operation in MUTATING_OPERATIONS and not self.idempotency_key:
            raise ValueError("写操作必须带 idempotency_key。")


@dataclass(frozen=True)
class ApiError:
    code: str
    message: str
    retryable: bool = False

    def __post_init__(self) -> None:
        _require_id(self.code, "error.code")
        if not str(self.message or "").strip():
            raise ValueError("error.message 不能为空。")
        if not isinstance(self.retryable, bool):
            raise ValueError("error.retryable 必须是布尔值。")


@dataclass(frozen=True)
class ApiResponseEnvelope:
    request_id: str
    status_code: int
    data: Any = None
    error: ApiError | None = None
    api_version: str = API_CONTRACT_VERSION

    def validate(self) -> None:
        if self.api_version != API_CONTRACT_VERSION:
            raise ValueError("API version 不受支持。")
        _require_id(self.request_id, "request_id")
        if isinstance(self.status_code, bool) or not isinstance(self.status_code, int) or not 200 <= self.status_code <= 599:
            raise ValueError("status_code 必须在 200-599。")
        success = 200 <= self.status_code < 300
        if success and self.error is not None:
            raise ValueError("成功响应不能同时携带 error。")
        if not success and self.error is None:
            raise ValueError("失败响应必须携带 error。")
        if self.error is not None and not isinstance(self.error, ApiError):
            raise ValueError("error 必须是 ApiError。")


@dataclass(frozen=True)
class SdkContractManifest:
    sdk_name: str
    sdk_version: str
    api_version: str
    supported_operations: tuple[str, ...]
    error_codes: tuple[str, ...]
    idempotency_for_writes: bool = True
    protocol_version: str = SDK_CONTRACT_VERSION

    def validate(self) -> None:
        if self.protocol_version != SDK_CONTRACT_VERSION:
            raise ValueError("SDK protocol_version 不受支持。")
        if not str(self.sdk_name or "").strip() or not str(self.sdk_version or "").strip():
            raise ValueError("SDK name/version 不能为空。")
        if self.api_version != API_CONTRACT_VERSION:
            raise ValueError("SDK 必须声明当前 API 版本。")
        if not self.supported_operations or any(not str(item or "").strip() for item in self.supported_operations):
            raise ValueError("SDK supported_operations 不能为空。")
        if len(set(self.error_codes)) != len(self.error_codes):
            raise ValueError("SDK error_codes 不能重复。")
        if not isinstance(self.idempotency_for_writes, bool):
            raise ValueError("SDK idempotency_for_writes 必须是布尔值。")


__all__ = [
    "API_CONTRACT_VERSION", "CONTROL_PLANE_PROTOCOL", "EXTERNAL_IDENTITY_PROTOCOL", "SDK_CONTRACT_VERSION", "ROLE_PERMISSIONS",
    "ApiError", "ApiRequestEnvelope", "ApiResponseEnvelope", "ControlPlaneOrganization",
    "ControlPlanePrincipal", "ControlPlaneSnapshot", "ControlPlaneTeam", "QuotaDecision", "QuotaLimit",
    "QuotaUsage", "SdkContractManifest", "ExternalIdentity", "ExternalIdentityProvider", "GITHUB_OAUTH_BROWSER_PROTOCOL", "evaluate_quota",
]
