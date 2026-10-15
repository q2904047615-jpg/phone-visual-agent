"""Typed request bodies owned by the HTTP boundary.

Keeping transport validation here leaves the composition root focused on
wiring and route behavior.  These models intentionally contain no runtime
or device logic.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr

from agent.domain.execution_budget import (
    DEFAULT_DEVICE_ACTION_BUDGET,
    DEFAULT_OBSERVATION_BUDGET,
)


class GenericSceneRequest(BaseModel):
    goal: dict[str, Any] = Field(default_factory=dict)


class StrictAgentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ControlPlaneIdentitySessionRequest(StrictAgentRequest):
    organization_id: StrictStr = Field(min_length=1, max_length=128)
    user_id: StrictStr = Field(min_length=1, max_length=128)


class ControlPlaneExternalIdentityRequest(StrictAgentRequest):
    provider_id: StrictStr = Field(min_length=1, max_length=128)
    assertion: dict[str, Any] = Field(default_factory=dict)


class ControlPlaneTeamWriteRequest(StrictAgentRequest):
    team_id: StrictStr = Field(min_length=1, max_length=128)
    name: StrictStr = Field(min_length=1, max_length=256)
    member_ids: list[StrictStr] = Field(default_factory=list, max_length=500)


class ControlPlaneTeamUpdateRequest(StrictAgentRequest):
    expected_revision: StrictInt | None = Field(default=None, ge=0)
    name: StrictStr | None = Field(default=None, min_length=1, max_length=256)
    member_ids: list[StrictStr] | None = Field(default=None, max_length=500)


class ControlPlanePrincipalWriteRequest(StrictAgentRequest):
    user_id: StrictStr = Field(min_length=1, max_length=128)
    role: Literal["viewer", "operator", "admin", "owner"]
    team_ids: list[StrictStr] = Field(default_factory=list, max_length=500)


class ControlPlanePrincipalUpdateRequest(StrictAgentRequest):
    role: Literal["viewer", "operator", "admin", "owner"]
    status: Literal["active", "disabled"]
    expected_revision: StrictInt = Field(ge=0)


class ControlPlaneQuotaWriteRequest(StrictAgentRequest):
    quota_key: StrictStr = Field(min_length=1, max_length=128)
    period: Literal["daily", "monthly"]
    max_runs: StrictInt | None = Field(default=None, ge=0)
    max_physical_actions: StrictInt | None = Field(default=None, ge=0)
    max_model_tokens: StrictInt | None = Field(default=None, ge=0)


class DeviceAssignmentRequest(StrictAgentRequest):
    user_id: StrictStr | None = Field(..., min_length=1, max_length=128)
    expected_revision: StrictInt = Field(ge=0)


class GitHubIdentityWriteRequest(StrictAgentRequest):
    user_id: StrictStr | None = Field(..., min_length=1, max_length=128)
    expected_revision: StrictInt = Field(ge=0)


class QwenConversationMessage(StrictAgentRequest):
    role: Literal["user", "assistant"]
    content: StrictStr = Field(min_length=1)


class GenericSupervisedStartRequest(StrictAgentRequest):
    text: StrictStr = Field(min_length=1)
    exact_input_text: StrictStr | None = Field(default=None, min_length=1)
    exact_action_kind: StrictStr | None = Field(default=None, max_length=32)
    exact_target_label: StrictStr = Field(default="")
    device_id: StrictStr = Field(min_length=1, max_length=128)
    auto_advance: StrictBool = True
    conversation: list[QwenConversationMessage] = Field(default_factory=list)
    max_physical_actions: StrictInt = Field(default=DEFAULT_DEVICE_ACTION_BUDGET, ge=1)
    max_observations: StrictInt = Field(default=DEFAULT_OBSERVATION_BUDGET, ge=1)


class GenericSupervisedDeviceRequest(StrictAgentRequest):
    device_id: StrictStr = Field(min_length=1, max_length=128)


class GenericRecoveryReconciliationRequest(StrictAgentRequest):
    device_id: StrictStr = Field(min_length=1, max_length=128)
    external_effect: Literal["not_dispatched", "unknown"] = "unknown"
    confirmed: StrictBool = False
    resume: StrictBool = False


class QwenChatRequest(StrictAgentRequest):
    """A natural-language message added to an existing Qwen task session."""

    device_id: StrictStr = Field(min_length=1, max_length=128)
    text: StrictStr = Field(min_length=1)


class QwenDraftChatRequest(StrictAgentRequest):
    """One multimodal Qwen message; Qwen returns natural reply and optional action together."""

    text: StrictStr = Field(min_length=1)
    device_id: StrictStr = Field(default="device-local-01", min_length=1, max_length=128)
    conversation: list[QwenConversationMessage] = Field(default_factory=list)
    max_physical_actions: StrictInt = Field(default=DEFAULT_DEVICE_ACTION_BUDGET, ge=1)
    max_observations: StrictInt = Field(default=DEFAULT_OBSERVATION_BUDGET, ge=1)


class MachinePositionRequest(StrictAgentRequest):
    machine_position: StrictInt = Field(ge=1, le=10)


class DirectionalSwipeRequest(StrictAgentRequest):
    direction: Literal["left", "right", "up", "down"]


class ManualMotionStartRequest(StrictAgentRequest):
    kind: Literal["click_pair", "swipe"]
    device_id: StrictStr = Field(min_length=1, max_length=128)
    point_a: tuple[int, int]
    point_b: tuple[int, int]
    interval_seconds: float = Field(gt=0)
    duration_seconds: float = Field(gt=0)

    @classmethod
    def _validate_finite(cls, value: float, name: str) -> float:
        import math
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} 必须是有限数字。")
        return float(value)

    def model_post_init(self, __context: object) -> None:
        from agent.infrastructure.manual_motion import MotionProfile
        MotionProfile(
            device_id=self.device_id,
            kind=self.kind,
            point_a=self.point_a,
            point_b=self.point_b,
            interval_seconds=self._validate_finite(self.interval_seconds, "interval_seconds"),
            duration_seconds=self._validate_finite(self.duration_seconds, "duration_seconds"),
        )


class DevicePairRequest(StrictAgentRequest):
    """User-visible Android wireless-debugging pairing form."""

    pairing_host: StrictStr = Field(min_length=1, max_length=255)
    pairing_port: StrictInt = Field(ge=1, le=65535)
    pairing_code: StrictStr = Field(pattern=r"^[0-9]{6}$")


class BaseActionConfirmationScopeRequest(StrictAgentRequest):
    session_id: StrictStr = Field(min_length=1, max_length=128)
    task_id: StrictStr = Field(min_length=1, max_length=128)
    device_id: StrictStr = Field(min_length=1, max_length=128)
    revision: StrictInt = Field(ge=1)
    step_id: StrictStr = Field(min_length=1, max_length=128)
    effect_ids: list[StrictStr] = Field(default_factory=list)
    observation_id: StrictStr = Field(min_length=1, max_length=128)
    fingerprint: StrictStr = Field(min_length=1, max_length=256)


class GenericConfirmationScopeRequest(BaseActionConfirmationScopeRequest):
    decision_node_id: StrictStr = Field(min_length=1, max_length=128)
    action_digest: StrictStr = Field(min_length=64, max_length=64)


class GenericEffectConfirmationScopeRequest(StrictAgentRequest):
    session_id: StrictStr = Field(min_length=1, max_length=128)
    task_id: StrictStr = Field(min_length=1, max_length=128)
    device_id: StrictStr = Field(min_length=1, max_length=128)
    revision: StrictInt = Field(ge=1)
    step_id: StrictStr = Field(min_length=1, max_length=128)
    effect_ids: list[StrictStr] = Field(min_length=1)
    intent_digest: StrictStr = Field(min_length=64, max_length=64)


class GenericEffectApprovalRequest(StrictAgentRequest):
    confirmed: StrictBool = False
    confirmation: GenericEffectConfirmationScopeRequest | None = None


class GenericSupervisedStepRequest(StrictAgentRequest):
    confirmed: StrictBool = False
    confirmation: GenericConfirmationScopeRequest | None = None


class GenericSupervisedAutoRequest(StrictAgentRequest):
    device_id: StrictStr = Field(min_length=1, max_length=128)
    confirmed: StrictBool = False
    confirmation: GenericConfirmationScopeRequest | None = None
    max_physical_actions: StrictInt | None = Field(default=None, ge=1)
    max_observations: StrictInt | None = Field(default=None, ge=1)


class CapabilityAcceptanceStartRequest(StrictAgentRequest):
    device_id: StrictStr = Field(min_length=1, max_length=128)
    action: StrictStr = Field(min_length=1, max_length=64)
    text: StrictStr = Field(min_length=1)


class CapabilityActionConfirmationScopeRequest(GenericConfirmationScopeRequest):
    trial_id: StrictStr = Field(min_length=1, max_length=128)
    action: StrictStr = Field(min_length=1, max_length=64)


class CapabilityEffectConfirmationScopeRequest(GenericEffectConfirmationScopeRequest):
    trial_id: StrictStr = Field(min_length=1, max_length=128)
    action: StrictStr = Field(min_length=1, max_length=64)


class CapabilityActionConfirmationRequest(StrictAgentRequest):
    confirmed: StrictBool = False
    confirmation: CapabilityActionConfirmationScopeRequest | None = None


class CapabilityEffectApprovalRequest(StrictAgentRequest):
    confirmed: StrictBool = False
    confirmation: CapabilityEffectConfirmationScopeRequest | None = None


class CapabilityPromotionRequest(StrictAgentRequest):
    confirmed: StrictBool = False
    trial_id: StrictStr = Field(min_length=1, max_length=128)
    device_id: StrictStr = Field(min_length=1, max_length=128)
    action: StrictStr = Field(min_length=1, max_length=64)
    report_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    registry_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")


class CapabilityCancelRequest(StrictAgentRequest):
    device_id: StrictStr = Field(min_length=1, max_length=128)
    action: StrictStr = Field(min_length=1, max_length=64)


__all__ = [
    "CapabilityAcceptanceStartRequest",
    "CapabilityActionConfirmationRequest",
    "CapabilityActionConfirmationScopeRequest",
    "CapabilityCancelRequest",
    "CapabilityEffectApprovalRequest",
    "CapabilityEffectConfirmationScopeRequest",
    "CapabilityPromotionRequest",
    "DevicePairRequest",
    "DirectionalSwipeRequest",
    "GenericConfirmationScopeRequest",
    "GenericEffectApprovalRequest",
    "GenericEffectConfirmationScopeRequest",
    "GenericSceneRequest",
    "GenericSupervisedAutoRequest",
    "GenericSupervisedDeviceRequest",
    "GenericSupervisedStartRequest",
    "GenericSupervisedStepRequest",
    "MachinePositionRequest",
    "ManualMotionStartRequest",
]
