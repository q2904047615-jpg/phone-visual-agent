from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .validation import dataclass_wire


CANONICAL_SELECTION_RECEIPT_VERSION = '2026-08-26-canonical-selection-receipt-v1'


@dataclass(frozen=True)
class CanonicalSelectionReceipt:
    """Evidence that the sole step selector emitted one executable candidate."""

    allowed: bool
    reason: str
    canonical_class: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {**dataclass_wire(self), 'policy_version': CANONICAL_SELECTION_RECEIPT_VERSION}
