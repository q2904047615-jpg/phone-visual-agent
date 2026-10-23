"""Runtime-owned task plan, completion conditions and one-step execution permits.

The Planner may propose observable completion conditions (model-assisted), but
only the Runtime commits a ``TaskPlan`` version. The Verifier judges a finish
candidate against exactly those committed conditions, and the Executor only
runs an action that holds an unconsumed ``ExecutionPermit`` bound to the
current plan version, observation, intent and policy decision.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import uuid

from .validation import canonical_digest, reject_if

PLAN_PROTOCOL_VERSION = "2026-10-24-task-plan-v1"
PERMIT_PROTOCOL_VERSION = "2026-10-24-execution-permit-v1"
CONDITION_STATUSES = frozenset({"met", "not_met", "unknown"})
GOAL_FALLBACK_SOURCE = "goal_fallback"
MODEL_PROPOSAL_SOURCE = "model_gateway_proposal"


class TaskPlanError(ValueError):
    """A plan, condition result or permit violates the Runtime contract."""


@dataclass(frozen=True)
class SuccessCondition:
    condition_id: str
    description: str

    def to_dict(self) -> dict[str, str]:
        return {"id": self.condition_id, "description": self.description}


@dataclass(frozen=True)
class TaskPlan:
    """One committed plan version; future versions never mutate this object."""

    plan_id: str
    version: int
    task_id: str
    device_id: str
    goal: str
    success_conditions: tuple[SuccessCondition, ...]
    source: str
    committed_revision: int
    protocol_version: str = PLAN_PROTOCOL_VERSION

    def validate(self) -> None:
        reject_if(not self.plan_id or not self.task_id or not self.device_id,
            TaskPlanError("计划缺少 plan/task/device 绑定。"))
        reject_if(isinstance(self.version, bool) or not isinstance(self.version, int) or self.version < 1,
            TaskPlanError("计划版本无效。"))
        reject_if(not self.success_conditions, TaskPlanError("计划至少需要一个可观察完成条件。"))
        ids = [item.condition_id for item in self.success_conditions]
        reject_if(len(ids) != len(set(ids)), TaskPlanError("完成条件编号重复。"))
        reject_if(any(not item.description.strip() for item in self.success_conditions),
            TaskPlanError("完成条件不能为空。"))
        reject_if(self.source not in {GOAL_FALLBACK_SOURCE, MODEL_PROPOSAL_SOURCE}, TaskPlanError("计划来源无效。"))

    @property
    def condition_ids(self) -> tuple[str, ...]:
        return tuple(item.condition_id for item in self.success_conditions)

    def to_dict(self) -> dict[str, Any]:
        data = {"protocol_version": self.protocol_version, "plan_id": self.plan_id, "version": self.version,
            "task_id": self.task_id, "device_id": self.device_id, "goal": self.goal,
            "success_conditions": [item.to_dict() for item in self.success_conditions],
            "source": self.source, "committed_revision": self.committed_revision}
        data["digest"] = canonical_digest(data)
        return data

    def prompt_view(self) -> dict[str, Any]:
        """What the model sees: the frozen conditions it must report on."""
        return {"version": self.version, "source": self.source,
            "success_conditions": [item.to_dict() for item in self.success_conditions]}

    @classmethod
    def commit(cls, *, task_id: str, device_id: str, goal: str, revision: int,
        proposed_conditions: Iterable[str] | None, previous: "TaskPlan | None" = None) -> "TaskPlan":
        """Runtime commit: model proposal when usable, otherwise the goal itself."""
        descriptions = tuple(dict.fromkeys(text.strip() for text in (proposed_conditions or ())
            if isinstance(text, str) and text.strip()))
        source = MODEL_PROPOSAL_SOURCE
        if not descriptions:
            descriptions = (f"用户任务要求的全部内容已在手机画面上完成：{goal.strip()}",)
            source = GOAL_FALLBACK_SOURCE
        plan = cls(plan_id=previous.plan_id if previous else f"plan_{uuid.uuid4().hex}",
            version=previous.version + 1 if previous else 1, task_id=task_id, device_id=device_id,
            goal=goal, success_conditions=tuple(SuccessCondition(f"c{index}", text)
                for index, text in enumerate(descriptions, 1)),
            source=source, committed_revision=int(revision))
        plan.validate()
        return plan


@dataclass(frozen=True)
class ConditionResult:
    condition_id: str
    status: str
    evidence: str

    def to_dict(self) -> dict[str, str]:
        return {"id": self.condition_id, "status": self.status, "evidence": self.evidence}


def parse_condition_results(value: Any) -> tuple[ConditionResult, ...] | None:
    """Lenient wire parse: malformed results become ``None`` (not verifiable), never a crash."""
    if not isinstance(value, (list, tuple)):
        return None
    results: list[ConditionResult] = []
    for item in value:
        if not isinstance(item, Mapping):
            return None
        condition_id = str(item.get("id") or item.get("condition_id") or "").strip()
        status = str(item.get("status") or "").strip().lower()
        if not condition_id or status not in CONDITION_STATUSES:
            return None
        results.append(ConditionResult(condition_id, status, str(item.get("evidence") or "").strip()))
    return tuple(results)


def parse_proposed_conditions(value: Any) -> tuple[str, ...] | None:
    if not isinstance(value, (list, tuple)):
        return None
    texts = tuple(text.strip() for text in value if isinstance(text, str) and text.strip())
    return texts or None


def unmet_conditions(plan: TaskPlan, results: tuple[ConditionResult, ...] | None) -> list[str]:
    """Return human-readable reasons why the plan is not yet satisfied (empty when it is)."""
    if results is None:
        return ["终态候选没有逐条报告计划完成条件。"]
    by_id: dict[str, ConditionResult] = {}
    for result in results:
        by_id.setdefault(result.condition_id, result)
    reasons = []
    for condition in plan.success_conditions:
        result = by_id.get(condition.condition_id)
        if result is None:
            reasons.append(f"{condition.condition_id} 未报告：{condition.description}")
        elif result.status != "met":
            reasons.append(f"{condition.condition_id} 为 {result.status}：{condition.description}")
        elif not result.evidence:
            reasons.append(f"{condition.condition_id} 缺少可见证据：{condition.description}")
    return reasons


@dataclass
class ExecutionPermit:
    """Server-side one-shot permit issued by the Runtime for exactly one action.

    It is never sent to the model or clients; invalidation happens whenever the
    plan version, observation, intent, policy decision or device lease changes.
    There is deliberately no wall-clock expiry: approved login/payment
    confirmations may legitimately wait for the user.
    """

    permit_id: str
    task_id: str
    device_id: str
    lease_epoch: int
    revision: int
    plan_id: str
    plan_version: int
    observation_id: str
    fingerprint: str
    intent_digest: str
    action_digest: str
    policy_version: str
    policy_status: str
    effect_ids: tuple[str, ...]
    issued_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    consumed: bool = False
    invalid_reason: str = ""
    protocol_version: str = PERMIT_PROTOCOL_VERSION

    @classmethod
    def issue(cls, *, task_id: str, device_id: str, lease_epoch: int, revision: int, plan: TaskPlan,
        observation_id: str, fingerprint: str, intent_digest: str, action_digest: str,
        policy_version: str, policy_status: str, effect_ids: Iterable[str]) -> "ExecutionPermit":
        reject_if(policy_status not in {"auto_execute", "requires_user_confirmation"},
            TaskPlanError("只有Policy允许的动作才能签发执行许可。"))
        return cls(permit_id=f"permit_{uuid.uuid4().hex}", task_id=task_id, device_id=device_id,
            lease_epoch=int(lease_epoch), revision=int(revision), plan_id=plan.plan_id, plan_version=plan.version,
            observation_id=observation_id, fingerprint=fingerprint, intent_digest=intent_digest,
            action_digest=action_digest, policy_version=policy_version, policy_status=policy_status,
            effect_ids=tuple(sorted(effect_ids)))

    def consume(self, *, task_id: str, device_id: str, lease_epoch: int, revision: int, plan: TaskPlan | None,
        observation_id: str, fingerprint: str, intent_digest: str, action_digest: str,
        confirmed_effect_ids: Iterable[str]) -> None:
        """Validate every binding, then mark the permit used (even when invalid)."""
        problems = []
        if self.consumed:
            problems.append("许可已经使用")
        if plan is None or (plan.plan_id, plan.version) != (self.plan_id, self.plan_version):
            problems.append("计划版本已变化")
        expected = (self.task_id, self.device_id, self.lease_epoch, self.revision, self.observation_id,
            self.fingerprint, self.intent_digest, self.action_digest)
        actual = (task_id, device_id, int(lease_epoch), int(revision), observation_id, fingerprint,
            intent_digest, action_digest)
        if expected != actual:
            problems.append("观察、意图、设备或租约已变化")
        if self.policy_status == "requires_user_confirmation" and not set(self.effect_ids) <= set(confirmed_effect_ids):
            problems.append("登录/付款尚未获得用户确认")
        self.consumed = True
        self.invalid_reason = "；".join(problems) or "consumed_before_execution"
        reject_if(problems, TaskPlanError("执行许可无效：" + "；".join(problems) + "。"))

    def to_dict(self) -> dict[str, Any]:
        return {"protocol_version": self.protocol_version, "permit_id": self.permit_id, "task_id": self.task_id,
            "device_id": self.device_id, "lease_epoch": self.lease_epoch, "revision": self.revision,
            "plan_id": self.plan_id, "plan_version": self.plan_version, "observation_id": self.observation_id,
            "fingerprint": self.fingerprint, "intent_digest": self.intent_digest,
            "action_digest": self.action_digest, "policy_version": self.policy_version,
            "policy_status": self.policy_status, "effect_ids": list(self.effect_ids),
            "issued_at": self.issued_at, "consumed": self.consumed, "invalid_reason": self.invalid_reason}


__all__ = ["CONDITION_STATUSES", "ConditionResult", "ExecutionPermit", "GOAL_FALLBACK_SOURCE",
    "MODEL_PROPOSAL_SOURCE", "PERMIT_PROTOCOL_VERSION", "PLAN_PROTOCOL_VERSION", "SuccessCondition", "TaskPlan",
    "TaskPlanError", "parse_condition_results", "parse_proposed_conditions", "unmet_conditions"]
