"""Pure trusted-observation snapshot for one current Qwen scene."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from .ui_scene import UIScene
from .validation import dataclass_wire
from .visual_evidence import LocalFrameStability


OBSERVATION_ID_PATTERN = re.compile(r"^obs_[A-Za-z0-9]{16,64}$")
@dataclass(frozen=True)
class TrustedObservation:
    observation_id: str
    device_id: str
    fingerprint: str
    scene: UIScene
    local_stability: LocalFrameStability
    selected_frame_index: int
    frame_sharpness_scores: tuple[float, ...]

    def get_candidate(self, element_id: str):
        return self.scene.get_element(element_id)

    def to_dict(self) -> dict[str, Any]:
        value = dataclass_wire(self)
        scene = value['scene']
        system_ui = structured_system_ui(self.scene)
        if system_ui is not None:
            scene["system_ui"] = system_ui
        value.update(frame_sharpness_scores=[round(score, 3) for score in self.frame_sharpness_scores],
            candidate_aliases={}, candidate_conflicts=[])
        return value


def structured_system_ui(scene: UIScene) -> dict[str, Any] | None:
    facts = getattr(scene, "system_ui", None)
    return facts.to_dict() if facts is not None else None
