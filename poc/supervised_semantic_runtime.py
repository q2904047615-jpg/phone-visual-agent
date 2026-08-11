from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from semantic_action_adapter import (
    PhysicalActionVerificationError,
    SemanticActionAdapterError,
    SemanticActionRouter,
)
from semantic_executor import (
    ActionResult,
    PageObservation,
    SingleStepSemanticExecutor,
    StepDecision,
)
from task_orchestrator import GoalSpec, TaskPlan


INTERNAL_ZERO_ACTIONS = frozenset({"record_verified_result", "finish"})


class SupervisedSemanticSessionError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        physical_actions: int = 0,
        evidence: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.physical_actions = int(physical_actions)
        self.evidence = tuple(evidence)


@dataclass
class SupervisedSemanticSession:
    session_id: str
    goal: GoalSpec
    plan: TaskPlan
    executor: SingleStepSemanticExecutor
    router: SemanticActionRouter
    decision: StepDecision
    created_at: str
    history: list[dict[str, Any]] = field(default_factory=list)
    cancelled: bool = False
    failed_reason: str = ""

    @classmethod
    def start(
        cls,
        *,
        session_id: str,
        goal: GoalSpec,
        plan: TaskPlan,
        initial_observation: PageObservation,
        router: SemanticActionRouter,
    ) -> "SupervisedSemanticSession":
        executor = SingleStepSemanticExecutor(plan)
        decision = executor.start(initial_observation)
        return cls(
            session_id=session_id,
            goal=goal,
            plan=plan,
            executor=executor,
            router=router,
            decision=decision,
            created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        )

    def step(
        self,
        *,
        confirmed: bool,
        evidence_dir: Path | None = None,
    ) -> dict[str, Any]:
        if self.cancelled:
            raise SupervisedSemanticSessionError("本次人工监督会话已经取消。")
        if confirmed is not True:
            raise SupervisedSemanticSessionError("必须明确确认当前这一个语义动作。")
        if self.decision.status != "action" or self.decision.action is None:
            raise SupervisedSemanticSessionError("当前会话没有等待执行的语义动作。")

        action = self.decision.action
        before = self.decision.to_dict()
        if action.action in INTERNAL_ZERO_ACTIONS:
            observation = self.executor.current_observation
            if observation is None:
                raise SupervisedSemanticSessionError("本地控制节点缺少最新页面证据。")
            result = ActionResult(
                node_id=action.node_id,
                success=True,
                observation=observation,
                details={
                    "reason": "本地控制器节点已确认，不操作机械臂。",
                    "physical_actions": 0,
                    "internal_controller_action": True,
                },
            )
            adapter_payload: dict[str, Any] = {
                "internal_controller_action": True,
                "robot_action_called": False,
            }
        else:
            if not self.router.supports(action):
                raise SupervisedSemanticSessionError(
                    f"动作 {action.action} 尚未接入人工监督白名单；会话停在原节点。"
                )
            try:
                adapter_result = self.router.execute(
                    action,
                    self.goal,
                    evidence_dir=evidence_dir,
                )
            except PhysicalActionVerificationError as exc:
                self.fail(str(exc))
                raise SupervisedSemanticSessionError(
                    str(exc),
                    physical_actions=exc.physical_actions,
                    evidence=exc.evidence,
                ) from exc
            except SemanticActionAdapterError as exc:
                raise SupervisedSemanticSessionError(str(exc)) from exc
            except RuntimeError as exc:
                self.fail(str(exc))
                raise SupervisedSemanticSessionError(
                    f"语义动作执行异常，已安全停止：{exc}"
                ) from exc
            result = adapter_result.action_result
            adapter_payload = adapter_result.to_dict()

        after = self.executor.advance(result)
        self.decision = after
        record = {
            "step": len(self.history) + 1,
            "decision_before": before,
            "result": {
                "node_id": result.node_id,
                "success": result.success,
                "observation": result.observation.to_dict(),
                "details": dict(result.details),
            },
            "adapter": adapter_payload,
            "decision_after": after.to_dict(),
        }
        self.history.append(record)
        return record

    def cancel(self) -> None:
        self.cancelled = True

    def fail(self, reason: str) -> None:
        self.failed_reason = str(reason)
        self.decision = StepDecision(
            status="failed",
            reason=self.failed_reason,
            counters=dict(self.executor.counters),
        )

    def snapshot(self) -> dict[str, Any]:
        action = self.decision.action
        action_supported = bool(
            action
            and (
                action.action in INTERNAL_ZERO_ACTIONS
                or self.router.supports(action)
            )
        )
        physical_possible = bool(
            action and action.action in {"ensure_app", "tap_semantic", "swipe"}
        )
        account_effect = bool(
            action
            and action.action == "tap_semantic"
            and str(action.params.get("target") or "") == "heart"
        )
        status = "cancelled" if self.cancelled else self.decision.status
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "status": status,
            "goal": self.goal.to_dict(),
            "decision": self.decision.to_dict(),
            "step_count": len(self.history),
            "failed_reason": self.failed_reason,
            "history": list(self.history),
            "current_action": {
                "supported": action_supported,
                "physical_action_possible": physical_possible,
                "account_effect_possible": account_effect,
                "requires_confirmation": status == "action",
            },
        }
