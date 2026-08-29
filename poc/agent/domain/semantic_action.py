from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .validation import DataclassWire


@dataclass(frozen=True)
class SemanticAction(DataclassWire):
    """One device-independent action selected from the canonical catalog."""

    node_id: str
    action: str
    params: dict[str, Any] = field(default_factory=dict)
