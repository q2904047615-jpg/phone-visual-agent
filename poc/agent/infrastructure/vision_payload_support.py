"""Supplier-neutral JSON and JPEG helpers for visual-model requests."""

from __future__ import annotations

import base64
import json
import re
from io import BytesIO
from typing import Any

from PIL import Image

from agent.domain.validation import reject_if
from agent.domain.vision_model import VisionAgentError


class _DuplicateJSONKeyError(ValueError):
    pass


def _reject_duplicate_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        reject_if(key in value, _DuplicateJSONKeyError(key))
        value[key] = item
    return value


def extract_json_object(raw: str, *, reject_duplicate_keys: bool = False,
    unwrap_singleton_object_array: bool = False) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith('```'):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    load_options = {
        'object_pairs_hook': _reject_duplicate_json_pairs,
    } if reject_duplicate_keys else {}
    try:
        value = json.loads(text, **load_options)
    except _DuplicateJSONKeyError as exc:
        raise VisionAgentError(f"模型返回的 JSON 包含重复字段：{exc}") from exc
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        reject_if(start < 0 or end <= start, VisionAgentError("模型没有返回 JSON 对象。"))
        try:
            value = json.loads(text[start : end + 1], **load_options)
        except _DuplicateJSONKeyError as exc:
            raise VisionAgentError(f'模型返回的 JSON 包含重复字段：{exc}') from exc
        except json.JSONDecodeError as exc:
            raise VisionAgentError(f"模型返回的 JSON 无法解析：{exc}") from exc
    if unwrap_singleton_object_array and isinstance(value, list):
        reject_if(len(value) != 1 or not isinstance(value[0], dict),
            VisionAgentError("模型返回的单步观察数组必须恰好包含一个 JSON 对象。"))
        value = value[0]
    reject_if(not isinstance(value, dict), VisionAgentError("模型返回值必须是 JSON 对象。"))
    return value


def image_request_size(image: Image.Image) -> tuple[int, int]:
    """Return the exact JPEG dimensions sent to the visual model."""
    return image.width, image.height


def image_data_url(image: Image.Image) -> str:
    """Encode a readable, bounded JPEG for visual-model requests."""

    result = image.convert("RGB")
    request_size = image_request_size(result)
    if result.size != request_size:
        result = result.resize(request_size, Image.Resampling.LANCZOS)
    buffer = BytesIO()
    result.save(buffer, format="JPEG", quality=82, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"
