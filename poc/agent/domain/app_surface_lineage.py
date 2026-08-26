from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VerifiedAppSurfaceLineage:
    session_id: str
    task_id: str
    device_id: str
    app_id: str
    app_name: str
    surface_id: str
    source_receipt_id: str
    source_subgoal_id: str
    functional_foreground_app_id: str
    physical_actions: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "task_id": self.task_id,
            "device_id": self.device_id,
            "app_id": self.app_id,
            "app_name": self.app_name,
            "surface_id": self.surface_id,
            "source_receipt_id": self.source_receipt_id,
            "source_subgoal_id": self.source_subgoal_id,
            "functional_foreground_app_id": self.functional_foreground_app_id,
            "physical_actions": self.physical_actions,
        }
