from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ConfirmationAuthority:
    session_id: str
    task_id: str
    device_id: str
    revision: int
    subgoal_id: str
    effect_ids: tuple[str, ...]
    observation_id: str
    fingerprint: str
    decision_node_id: str
    action_digest: str
    consumed: bool = False
    invalid_reason: str = ""

    def scope(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "task_id": self.task_id,
            "device_id": self.device_id,
            "revision": self.revision,
            "subgoal_id": self.subgoal_id,
            "effect_ids": sorted(self.effect_ids),
            "observation_id": self.observation_id,
            "fingerprint": self.fingerprint,
            "decision_node_id": self.decision_node_id,
            "action_digest": self.action_digest,
        }


@dataclass
class EffectConfirmationAuthority:
    session_id: str
    task_id: str
    device_id: str
    revision: int
    subgoal_id: str
    effect_ids: tuple[str, ...]
    intent_digest: str
    intent_preview: dict[str, Any]
    consumed: bool = False
    invalid_reason: str = ""

    def scope(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "task_id": self.task_id,
            "device_id": self.device_id,
            "revision": self.revision,
            "subgoal_id": self.subgoal_id,
            "effect_ids": sorted(self.effect_ids),
            "intent_digest": self.intent_digest,
        }
