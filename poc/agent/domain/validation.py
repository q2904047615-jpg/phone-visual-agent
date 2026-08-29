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
