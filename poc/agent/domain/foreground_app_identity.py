"""Trusted current-foreground App identity reported by the paired Android companion."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any


FOREGROUND_APP_IDENTITY_PROTOCOL = "2026-09-01-foreground-app-identity-v1"
FOREGROUND_APP_IDENTITY_SOURCES = frozenset({"editor_info", "usage_stats"})
FOREGROUND_APP_IDENTITY_TTL_SECONDS = 6.0

_DEVICE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}")
_ANDROID_PACKAGE = re.compile(r"[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)+")


class ForegroundAppIdentityError(ValueError):
    """A companion foreground identity is malformed, stale, or out of scope."""


@dataclass(frozen=True)
class ForegroundAppIdentity:
    """One signed system fact; it is App identity, never page semantics."""

    device_id: str
    package_name: str
    source: str
    event_at_epoch: float
    observed_at_epoch: float
    protocol_version: str = FOREGROUND_APP_IDENTITY_PROTOCOL

    def validate(self) -> None:
        if self.protocol_version != FOREGROUND_APP_IDENTITY_PROTOCOL:
            raise ForegroundAppIdentityError("前台 App 身份协议版本不匹配。")
        if not isinstance(self.device_id, str) or not _DEVICE_ID.fullmatch(self.device_id):
            raise ForegroundAppIdentityError("前台 App 身份 device_id 无效。")
        if not isinstance(self.package_name, str) or not _ANDROID_PACKAGE.fullmatch(self.package_name):
            raise ForegroundAppIdentityError("前台 App 包名无效。")
        if self.source not in FOREGROUND_APP_IDENTITY_SOURCES:
            raise ForegroundAppIdentityError("前台 App 身份来源无效。")
        for label, value in (("event_at_epoch", self.event_at_epoch),
            ("observed_at_epoch", self.observed_at_epoch)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ForegroundAppIdentityError(f"前台 App 身份 {label} 无效。")
        if float(self.event_at_epoch) > float(self.observed_at_epoch):
            raise ForegroundAppIdentityError("前台 App 事件时间晚于观察时间。")

    def is_fresh(self, now_epoch: float, *, ttl_seconds: float=FOREGROUND_APP_IDENTITY_TTL_SECONDS) -> bool:
        self.validate()
        if (isinstance(now_epoch, bool) or not isinstance(now_epoch, (int, float))
            or not math.isfinite(float(now_epoch)) or not math.isfinite(float(ttl_seconds))
            or float(ttl_seconds) <= 0):
            raise ForegroundAppIdentityError("前台 App 身份新鲜度参数无效。")
        age = float(now_epoch) - float(self.observed_at_epoch)
        return 0.0 <= age <= float(ttl_seconds)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {"protocol_version": self.protocol_version, "device_id": self.device_id,
            "package_name": self.package_name, "source": self.source,
            "event_at_epoch": float(self.event_at_epoch), "observed_at_epoch": float(self.observed_at_epoch)}


__all__ = ["FOREGROUND_APP_IDENTITY_PROTOCOL", "FOREGROUND_APP_IDENTITY_SOURCES",
    "FOREGROUND_APP_IDENTITY_TTL_SECONDS", "ForegroundAppIdentity", "ForegroundAppIdentityError"]
