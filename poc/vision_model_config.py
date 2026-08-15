from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Mapping


VISION_MODEL_CONFIG_VERSION = "2026-08-15-vision-model-config-v1"
DEFAULT_VISION_MODEL = "qwen3.7-plus"
DEFAULT_VISION_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_VISION_PROVIDER = "aliyun_model_studio"
VISION_COORDINATE_SCALE = 1000

_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")


@dataclass(frozen=True)
class VisionModelConfig:
    """Immutable runtime identity and request policy for the visual model.

    The model may be replaced through configuration, while the observation,
    safety and mechanical-action contracts remain local and unchanged.
    """

    model: str
    base_url: str
    provider: str = DEFAULT_VISION_PROVIDER
    enable_thinking: bool = False
    coordinate_scale: int = VISION_COORDINATE_SCALE
    config_version: str = VISION_MODEL_CONFIG_VERSION

    def __post_init__(self) -> None:
        model = self.model.strip()
        base_url = self.base_url.strip().rstrip("/")
        if not _MODEL_ID_RE.fullmatch(model):
            raise ValueError("视觉模型 ID 格式无效。")
        if not base_url.startswith(("https://", "http://127.0.0.1", "http://localhost")):
            raise ValueError("视觉模型地址必须使用 HTTPS 或本机回环地址。")
        if self.coordinate_scale != VISION_COORDINATE_SCALE:
            raise ValueError("视觉模型坐标必须使用项目统一的 0..1000 归一化尺度。")
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "base_url", base_url)

    def request_options(self) -> dict[str, bool]:
        return {"enable_thinking": self.enable_thinking}

    def public_identity(self) -> dict[str, object]:
        return {
            "config_version": self.config_version,
            "provider": self.provider,
            "model": self.model,
            "thinking_enabled": self.enable_thinking,
            "coordinate_scale": self.coordinate_scale,
        }


def load_vision_model_config(
    *,
    model: str | None = None,
    base_url: str | None = None,
    enable_thinking: bool = False,
    environ: Mapping[str, str] | None = None,
) -> VisionModelConfig:
    """Resolve the visual model without coupling callers to one model name.

    New generic environment names take precedence.  The former Qwen-specific
    names remain read-only compatibility inputs so existing launch setups keep
    working during migration.
    """

    values = os.environ if environ is None else environ
    resolved_model = (
        model
        or values.get("VISION_MODEL")
        or values.get("QWEN_VL_MODEL")
        or DEFAULT_VISION_MODEL
    )
    resolved_base_url = (
        base_url
        or values.get("VISION_MODEL_BASE_URL")
        or values.get("DASHSCOPE_BASE_URL")
        or DEFAULT_VISION_BASE_URL
    )
    return VisionModelConfig(
        model=resolved_model,
        base_url=resolved_base_url,
        enable_thinking=bool(enable_thinking),
    )


def public_model_identity(status: Mapping[str, object]) -> dict[str, object]:
    """Extract stable, secret-free model provenance for reports."""

    keys = (
        "model_config_version",
        "provider",
        "model",
        "thinking_enabled",
        "coordinate_scale",
        "response_model",
    )
    return {key: status[key] for key in keys if key in status}
