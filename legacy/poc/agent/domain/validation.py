from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
import hashlib
import json
from typing import Any


NormalizedPoint = tuple[float, float]
NormalizedBounds = tuple[float, float, float, float]


class DataclassWire:
    """Serialize a domain dataclass through the shared wire vocabulary."""

    def to_dict(self) -> dict[str, Any]:
        return dataclass_wire(self)


class ValidatedDataclassWire(DataclassWire):
    """Validate a domain dataclass before using the shared wire vocabulary."""

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return super().to_dict()


def reject_if(condition: object, error: Exception) -> None:
    """Raise a prepared contract error when an invariant is violated."""

    if condition:
        raise error


def bounds_overlap(left: NormalizedBounds, right: NormalizedBounds) -> dict[str, float]:
    """Return the shared deterministic overlap measures for two rectangles."""

    intersection_width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    intersection_height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    smaller = min(left_area, right_area)
    return {'iou': intersection / union if union > 0 else 0.0,
        'intersection_over_smaller': intersection / smaller if smaller > 0 else 0.0}


def wire_value(value: Any) -> Any:
    serializer = getattr(value, 'to_dict', None)
    if callable(serializer):
        return serializer()
    if isinstance(value, Mapping):
        return {str(key): wire_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [wire_value(item) for item in value]
    return value


def dataclass_wire(value: Any, *, omit: tuple[str, ...]=()) -> dict[str, Any]:
    return {item.name: wire_value(getattr(value, item.name)) for item in fields(value) if item.name not in omit}


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()
