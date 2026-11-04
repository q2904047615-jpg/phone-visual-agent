"""HTTP request models for the manual Soda Music page test."""

from __future__ import annotations

from pydantic import Field, StrictBool, StrictInt, StrictStr

from agent.interfaces.http_models import (
    GenericConfirmationScopeRequest,
    StrictAgentRequest,
)


class QishuiAdTestStartRequest(StrictAgentRequest):
    device_id: StrictStr = Field(min_length=1, max_length=128)
    duration_seconds: StrictInt = Field(default=180, ge=1, le=3600)


class QishuiAdTestDeviceRequest(StrictAgentRequest):
    device_id: StrictStr = Field(min_length=1, max_length=128)


class QishuiAdTestConfirmationRequest(StrictAgentRequest):
    device_id: StrictStr = Field(min_length=1, max_length=128)
    confirmed: StrictBool = False
    mode: StrictStr = Field(default="end", pattern=r"^(back|end)$")
    confirmation: GenericConfirmationScopeRequest | None = None


__all__ = [
    "QishuiAdTestConfirmationRequest",
    "QishuiAdTestDeviceRequest",
    "QishuiAdTestStartRequest",
]
