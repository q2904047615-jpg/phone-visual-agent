"""Runtime-owned task plan and its observable completion conditions.

The Planner may propose conditions (model-assisted); only the Runtime commits a
``TaskPlan`` version, and the Verifier judges a finish candidate against exactly
those conditions. Condition reports stay plain ``{"id","status","evidence"}``
dicts because they are model evidence, not authority.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
import uuid

from .validation import reject_if

PLAN_PROTOCOL_VERSION = "2026-10-24-task-plan-v1"
CONDITION_STATUSES = frozenset({"met", "not_met", "unknown"})
GOAL_FALLBACK_SOURCE = "goal_fallback"
MODEL_PROPOSAL_SOURCE = "model_gateway_proposal"


class TaskPlanError(ValueError):
    """A plan violates the Runtime contract."""


@dataclass(frozen=True)
class TaskPlan:
    """One committed plan version; a revision creates a new object."""

    plan_id: str
    version: int
    task_id: str
    device_id: str
    goal: str
    conditions: tuple[str, ...]
    source: str
    committed_revision: int

    @property
    def condition_ids(self) -> tuple[str, ...]:
        return tuple(f"c{index}" for index in range(1, len(self.conditions) + 1))

    def prompt_view(self) -> dict[str, Any]:
        """What the model sees: the frozen conditions it must report on."""
        return {"version": self.version, "source": self.source,
            "success_conditions": [{"id": cid, "description": text}
                for cid, text in zip(self.condition_ids, self.conditions)]}

    def to_dict(self) -> dict[str, Any]:
        return {"protocol_version": PLAN_PROTOCOL_VERSION, "plan_id": self.plan_id, "task_id": self.task_id,
            "device_id": self.device_id, "goal": self.goal, "committed_revision": self.committed_revision,
            **self.prompt_view()}

    @classmethod
    def commit(cls, *, task_id: str, device_id: str, goal: str, revision: int,
        proposed_conditions: tuple[str, ...] | None, previous: "TaskPlan | None" = None) -> "TaskPlan":
        """Runtime commit: the model's proposal when present, otherwise the goal itself."""
        conditions, source = proposed_conditions, MODEL_PROPOSAL_SOURCE
        if not conditions:
            conditions, source = (f"用户任务要求的全部内容已在手机画面上完成：{goal.strip()}",), GOAL_FALLBACK_SOURCE
        reject_if(not task_id or not device_id, TaskPlanError("计划缺少 task/device 绑定。"))
        return cls(plan_id=previous.plan_id if previous else f"plan_{uuid.uuid4().hex}",
            version=previous.version + 1 if previous else 1, task_id=task_id, device_id=device_id, goal=goal,
            conditions=tuple(conditions), source=source, committed_revision=int(revision))

    def unmet(self, results: tuple[dict[str, str], ...] | None) -> list[str]:
        """Why this plan is not yet satisfied by a finish report (empty when it is)."""
        if results is None:
            return ["终态候选没有逐条报告计划完成条件。"]
        by_id: dict[str, dict[str, str]] = {}
        for item in results:
            by_id.setdefault(item["id"], item)
        reasons = []
        for cid, text in zip(self.condition_ids, self.conditions):
            item = by_id.get(cid)
            if item is None:
                reasons.append(f"{cid} 未报告：{text}")
            elif item["status"] != "met":
                reasons.append(f"{cid} 为 {item['status']}：{text}")
            elif not item["evidence"]:
                reasons.append(f"{cid} 缺少可见证据：{text}")
        return reasons


def parse_proposed_conditions(value: Any) -> tuple[str, ...] | None:
    """Lenient wire parse; malformed input degrades to ``None`` instead of vetoing the step."""
    if not isinstance(value, (list, tuple)):
        return None
    texts = tuple(dict.fromkeys(text.strip() for text in value if isinstance(text, str) and text.strip()))
    return texts or None


def parse_condition_results(value: Any) -> tuple[dict[str, str], ...] | None:
    if not isinstance(value, (list, tuple)):
        return None
    results = []
    for item in value:
        if not isinstance(item, Mapping):
            return None
        cid = str(item.get("id") or "").strip()
        status = str(item.get("status") or "").strip().lower()
        if not cid or status not in CONDITION_STATUSES:
            return None
        results.append({"id": cid, "status": status, "evidence": str(item.get("evidence") or "").strip()})
    return tuple(results)


__all__ = ["CONDITION_STATUSES", "GOAL_FALLBACK_SOURCE", "MODEL_PROPOSAL_SOURCE", "PLAN_PROTOCOL_VERSION",
    "TaskPlan", "TaskPlanError", "parse_condition_results", "parse_proposed_conditions"]
