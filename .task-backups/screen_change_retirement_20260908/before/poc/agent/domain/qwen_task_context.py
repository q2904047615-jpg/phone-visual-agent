"""The whole user request and factual execution history supplied to one visual model."""
from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any
from .validation import dataclass_wire, reject_if
from .vision_model import VisionAgentError

DEVICE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")

SUPPORTED_TASK_CONTEXT_PROTOCOL = "2026-09-06-single-visual-task-v1"
EFFECT_KINDS = frozenset({"send_message", "publish_content", "relationship_change", "membership_change",
    "data_mutation", "authentication", "financial_transaction", "sensitive_permission_change",
    "irreversible_account_deletion", "irreversible_data_deletion"})
CONFIRMATION_EFFECT_KINDS = frozenset({"authentication", "financial_transaction"})

@dataclass(frozen=True)
class QwenTaskContext:
    task_id: str
    device_id: str
    revision: int
    raw_goal: str
    history: tuple[dict[str, Any], ...] = ()
    exact_input_text: str | None = None
    protocol_version: str = SUPPORTED_TASK_CONTEXT_PROTOCOL

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "QwenTaskContext":
        result = cls(**value)
        result.validate()
        return result

    def validate(self) -> None:
        reject_if(self.protocol_version != SUPPORTED_TASK_CONTEXT_PROTOCOL,
            VisionAgentError("不支持的整任务视觉协议。"))
        reject_if(not self.task_id or not self.device_id or not self.raw_goal.strip(),
            VisionAgentError("整任务缺少 task/device/用户目标。"))
        reject_if(isinstance(self.revision, bool) or not isinstance(self.revision, int) or self.revision < 1,
            VisionAgentError("观察版本必须为正整数。"))
        reject_if(self.exact_input_text is not None and not isinstance(self.exact_input_text, str),
            VisionAgentError("逐字输入正文必须为字符串。"))

    def to_dict(self) -> dict[str, Any]:
        return dataclass_wire(self)

def action_effect_kind(action: Any) -> str:
    if action.action in {"input_verified_text", "clear_verified_text", "press_enter"}:
        return "data_mutation"
    meaning = str(action.params.get("target") or "")
    return meaning if meaning in EFFECT_KINDS else ""


def execution_history_entry(*, step: int, requested_action: dict[str, Any],
    resolved_action: dict[str, Any], physical_actions: int, transport_outcome: str,
    visual_outcome: Any = None, after_scene: str = "") -> dict[str, Any]:
    """Project executed facts without losing semantic targets or sharing mutable params."""
    return deepcopy({"step": step, "canonical_action": requested_action, "action": resolved_action,
        "physical_actions": physical_actions, "transport_outcome": transport_outcome,
        "visual_outcome": visual_outcome, "after_scene": after_scene})
