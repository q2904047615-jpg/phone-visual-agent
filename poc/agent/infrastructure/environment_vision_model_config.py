"""Environment-backed loader for the visual-model domain configuration."""

from __future__ import annotations

import os
from typing import Mapping

from agent.domain.vision_model import (
    DEFAULT_VISION_BASE_URL,
    DEFAULT_VISION_MODEL,
    VisionModelConfig,
)


def load_vision_model_config(
    *,
    model: str | None = None,
    base_url: str | None = None,
    enable_thinking: bool = False,
    environ: Mapping[str, str] | None = None,
) -> VisionModelConfig:
    """Resolve the visual model from the single current configuration surface."""

    values = os.environ if environ is None else environ
    resolved_model = model or values.get("VISION_MODEL") or DEFAULT_VISION_MODEL
    resolved_base_url = (
        base_url or values.get("VISION_MODEL_BASE_URL") or DEFAULT_VISION_BASE_URL
    )
    return VisionModelConfig(
        model=resolved_model,
        base_url=resolved_base_url,
        enable_thinking=bool(enable_thinking),
    )
