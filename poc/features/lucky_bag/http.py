"""HTTP request models for the additive lucky-bag feature."""

from __future__ import annotations

from pydantic import Field, StrictInt, StrictStr

from agent.interfaces.http_models import StrictAgentRequest


class LuckyBagStartRequest(StrictAgentRequest):
    device_id: StrictStr = Field(default="device-local-01", min_length=1, max_length=128)
    recipient: StrictStr = Field(default="q2904047615@gmail.com", min_length=3, max_length=320)
    duration_seconds: StrictInt = Field(default=24 * 60 * 60, ge=1)


class LuckyBagDeviceRequest(StrictAgentRequest):
    device_id: StrictStr = Field(min_length=1, max_length=128)
