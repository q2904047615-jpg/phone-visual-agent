from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from generic_action_adapter import (
    GenericActionAdapterError,
    GenericActionExecutionResult,
    GenericSingleActionAdapter,
)
from generic_intent import GenericIntentDraft
from generic_step_planner import GenericStepPlanner, GenericStepProposal
from ui_scene import UIScene
from universal_action_controller import (
    UniversalActionController,
    UniversalActionError,
    action_has_account_effect,
)


@dataclass
class GenericSupervisedSession:
    session_id: str
    goal: GenericIntentDraft
    current_scene: UIScene
    proposal: GenericStepProposal | None
    planner: GenericStepPlanner
    adapter: GenericSingleActionAdapter
    run_dir: Path
    created_at: str = field(
        default_factory=lambda: datetime.now().astimezone().isoformat(timespec="seconds")
    )
    status: str = "awaiting_confirmation"
    step_number: int = 1
    history: list[dict[str, Any]] = field(default_factory=list)
    failed_reason: str = ""
    automatic_loop_enabled: bool = False
    auto_pause_reason: str = ""

    @classmethod
    def start(
        cls,
        *,
        session_id: str,
        goal: GenericIntentDraft,
        scene: UIScene,
        proposal: GenericStepProposal,
        planner: GenericStepPlanner,
        adapter: GenericSingleActionAdapter,
        run_dir: Path,
    ) -> "GenericSupervisedSession":
        proposal.validate(scene)
        status = {
            "action": "awaiting_confirmation",
            "finished": "succeeded",
            "blocked": "blocked",
        }[proposal.status]
        return cls(
            session_id=session_id,
            goal=goal,
            current_scene=scene,
            proposal=proposal,
            planner=planner,
            adapter=adapter,
            run_dir=run_dir,
            status=status,
        )

    def confirm(self, *, confirmed: bool) -> GenericActionExecutionResult:
        if confirmed is not True:
            raise GenericActionAdapterError("必须明确确认当前这一个语义动作。")
        if self.status != "awaiting_confirmation":
            raise GenericActionAdapterError(
                f"当前会话状态不能执行动作：{self.status}"
            )
        if self.proposal is None or self.proposal.action is None:
            raise GenericActionAdapterError("当前会话没有等待确认的动作。")
        try:
            result = self.adapter.execute(
                requested_action=self.proposal.action,
                planned_scene=self.current_scene,
                goal=self.goal,
                confirmed=True,
                evidence_dir=self.run_dir,
            )
        except GenericActionAdapterError as exc:
            self.status = "failed"
            self.failed_reason = str(exc)
            raise
        self.history.append(
            {
                "step_number": self.step_number,
                "proposal": self.proposal.to_dict(),
                "execution": result.to_dict(),
            }
        )
        self.current_scene = result.after_scene
        completion_evidence = self.adapter.controller.completion_evidence_after_action(
            result.resolved_action,
            result.before_scene,
            result.after_scene,
        )
        if completion_evidence:
            self.history[-1]["completion_evidence"] = list(completion_evidence)
            self.proposal = GenericStepProposal(
                status="finished",
                action=None,
                reason="控制器已验证本步预期效果，目标完成。",
                completion_evidence=completion_evidence,
            )
            self.status = "succeeded"
        else:
            self.proposal = None
            self.status = "paused_after_action"
        return result

    def run_safe_loop(
        self,
        *,
        confirmed: bool,
        max_physical_actions: int = 4,
        max_iterations: int = 8,
    ) -> dict[str, Any]:
        """Advance safe navigation steps until completion or a safety boundary.

        The existing single-action adapter remains the only component allowed to
        touch the robot.  This method merely repeats its already verified
        observe -> execute one action -> reobserve contract.
        """

        if confirmed is not True:
            raise GenericActionAdapterError("必须明确确认启动安全自动推进。")
        if self.status not in {"awaiting_confirmation", "paused_after_action"}:
            raise GenericActionAdapterError(
                f"当前会话状态不能自动推进：{self.status}"
            )
        action_limit = max(1, min(8, int(max_physical_actions)))
        iteration_limit = max(action_limit, min(16, int(max_iterations)))
        self.automatic_loop_enabled = True
        self.auto_pause_reason = ""
        physical_actions = 0
        iterations = 0
        seen_steps: set[str] = set()

        if self.status == "paused_after_action":
            self.plan_next(self.current_scene)

        while self.status == "awaiting_confirmation":
            if self.proposal is None or self.proposal.action is None:
                raise GenericActionAdapterError("自动推进缺少待执行动作。")
            if self.current_action_has_account_effect():
                self.auto_pause_reason = "下一步会改变账号状态，等待单独确认。"
                break
            if iterations >= iteration_limit:
                self.status = "paused_after_action"
                self.auto_pause_reason = "达到单次自动推进迭代上限。"
                break
            signature = self._step_signature()
            if signature in seen_steps:
                self.status = "blocked"
                self.failed_reason = "检测到相同页面和相同动作重复出现，已停止循环。"
                self.auto_pause_reason = self.failed_reason
                break
            seen_steps.add(signature)
            iterations += 1

            result = self.confirm(confirmed=True)
            physical_actions += result.physical_actions
            if physical_actions >= action_limit:
                self.auto_pause_reason = "达到单次自动推进物理动作上限。"
                break
            self.plan_next(self.current_scene)

        return {
            "physical_actions": physical_actions,
            "iterations": iterations,
            "status": self.status,
            "pause_reason": self.auto_pause_reason,
        }

    def plan_next(self, scene: UIScene) -> GenericStepProposal:
        if self.status != "paused_after_action":
            raise GenericActionAdapterError(
                f"当前会话状态不能规划下一步：{self.status}"
            )
        self.step_number += 1
        proposal = self.planner.propose(
            self.goal,
            scene,
            step_number=self.step_number,
        )
        UniversalActionController().resolve_one(
            proposal.action,
            scene,
            confirmed=True,
        ) if proposal.action is not None else None
        self.current_scene = scene
        self.proposal = proposal
        self.status = {
            "action": "awaiting_confirmation",
            "finished": "succeeded",
            "blocked": "blocked",
        }[proposal.status]
        return proposal

    def cancel(self) -> None:
        if self.status not in {"succeeded", "cancelled"}:
            self.status = "cancelled"

    def current_action_has_account_effect(self) -> bool:
        action = self.proposal.action if self.proposal else None
        if action is None:
            return False
        return action_has_account_effect(action)

    def _step_signature(self) -> str:
        action = self.proposal.action if self.proposal else None
        scene_value = {
            "foreground_app_id": self.current_scene.foreground_app_id,
            "screen_id": self.current_scene.screen_id,
            "overlays": sorted(self.current_scene.overlays),
            "elements": sorted(
                (
                    item.role,
                    item.meaning.casefold(),
                    item.label.casefold(),
                    tuple(sorted((str(k), str(v)) for k, v in item.states.items())),
                )
                for item in self.current_scene.elements
            ),
            "action": action.to_dict() if action else None,
        }
        import json

        return json.dumps(scene_value, ensure_ascii=False, sort_keys=True)

    def snapshot(self) -> dict[str, Any]:
        action = self.proposal.action if self.proposal else None
        target = str(action.params.get("target") or "") if action else ""
        account_effect = self.current_action_has_account_effect()
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "status": self.status,
            "step_number": self.step_number,
            "goal": self.goal.to_dict(),
            "scene": self.current_scene.to_dict(),
            "proposal": self.proposal.to_dict() if self.proposal else None,
            "history": list(self.history),
            "failed_reason": self.failed_reason,
            "automatic_loop_enabled": self.automatic_loop_enabled,
            "auto_pause_reason": self.auto_pause_reason,
            "current_action": {
                "requires_confirmation": self.status == "awaiting_confirmation",
                "account_effect_possible": account_effect,
                "max_physical_actions": 1,
                "physical_action_possible": bool(
                    action
                    and action.action
                    in {"tap_semantic", "dismiss_overlay", "swipe", "back"}
                ),
            },
        }
