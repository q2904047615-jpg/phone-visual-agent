from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping


VISION_MODEL_CONFIG_VERSION = "2026-08-15-vision-model-config-v1"
DEFAULT_VISION_MODEL = "qwen3.7-plus"
DEFAULT_VISION_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_VISION_PROVIDER = "aliyun_model_studio"
VISION_COORDINATE_SCALE = 1000

_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")


class VisionAgentError(RuntimeError):
    """The configured visual model or its response cannot be used."""


@dataclass(frozen=True)
class VisionModelConfig:
    """Immutable runtime identity and request policy for the visual model."""

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
        if not base_url.startswith( ("https://", "http://127.0.0.1", "http://localhost") ):
            raise ValueError("视觉模型地址必须使用 HTTPS 或本机回环地址。")
        if self.coordinate_scale != VISION_COORDINATE_SCALE:
            raise ValueError("视觉模型坐标必须使用项目统一的 0..1000 归一化尺度。")
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "base_url", base_url)

    def request_options(self) -> dict[str, bool]:
        return {"enable_thinking": self.enable_thinking}


def public_model_identity(status: Mapping[str, object]) -> dict[str, object]:
    """Extract stable, secret-free model provenance for reports."""

    keys = ('model_config_version', 'provider', 'model', 'thinking_enabled', 'coordinate_scale', 'response_model')
    return {key: status[key] for key in keys if key in status}
