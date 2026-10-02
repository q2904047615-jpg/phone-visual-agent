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
]
