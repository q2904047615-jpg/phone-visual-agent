from __future__ import annotations

import copy
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


DEEPSEEK_TASK_GRAPH_V3 = "2026-08-11-deepseek-task-graph-v3"
QWEN_VISUAL_DECISION_V3 = "2026-08-12-qwen-visual-decision-v3"
QWEN_VISUAL_DECISION_V4 = "2026-08-14-qwen-visual-decision-v4"
QWEN_VISUAL_DECISION_V5 = "2026-08-14-qwen-visual-decision-v5"
SUPPORTED_QWEN_VISUAL_DECISION_PROTOCOLS = frozenset(
    {
        QWEN_VISUAL_DECISION_V3,
        QWEN_VISUAL_DECISION_V4,
        QWEN_VISUAL_DECISION_V5,
    }
)
_SCOPED_CONFIRMATION_CAPABILITY = object()


@dataclass
class V3ConfirmationAuthority:
    scope: dict[str, Any]
    observation_id: str
    fingerprint: str
    consumed: bool = False
    invalid_reason: str = ""


@dataclass
class GenericSupervisedSession:
    session_id: str
    goal: GenericIntentDraft
    current_scene: UIScene
    proposal: GenericStepProposal | None
    planner: GenericStepPlanner
    adapter: GenericSingleActionAdapter
    run_dir: Path
    device_id: str = "default-device"
    created_at: str = field(
        default_factory=lambda: datetime.now().astimezone().isoformat(timespec="seconds")
    )
    status: str = "awaiting_confirmation"
    step_number: int = 1
    history: list[dict[str, Any]] = field(default_factory=list)
    failed_reason: str = ""
    automatic_loop_enabled: bool = False
    auto_pause_reason: str = ""
    task_graph: dict[str, Any] | None = None
    qwen_decision: dict[str, Any] | None = None
    _v3_confirmation: V3ConfirmationAuthority | None = field(
        default=None,
        repr=False,
    )

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
        device_id: str = "default-device",
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
            device_id=str(device_id or "").strip(),
            status=status,
        )

    def confirm(self, *, confirmed: bool) -> GenericActionExecutionResult:
        """Reject the retired boolean-only confirmation path."""

        raise GenericActionAdapterError(
            "旧裸布尔确认入口已关闭；动作必须使用服务端完整确认作用域。"
        )

    def _execute_current_action(self, capability: object) -> GenericActionExecutionResult:
        if capability is not _SCOPED_CONFIRMATION_CAPABILITY:
            raise GenericActionAdapterError(
                "动作执行缺少已消费的服务端完整确认作用域。"
            )
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

    def bind_v3_confirmation_context(
        self,
        *,
        task_graph: dict[str, Any],
        qwen_decision: dict[str, Any],
    ) -> None:
        """Install the authoritative v3 state produced by the future agent loop.

        The current web backend does not create this state itself.  Until its
        DeepSeek/Qwen loop calls this method, scoped confirmation fails closed.
        """

        self.invalidate_v3_confirmation("authority_replaced")
        self.task_graph = copy.deepcopy(task_graph)
        self.qwen_decision = copy.deepcopy(qwen_decision)
        if str(self.qwen_decision.get("status") or "") != "action":
            return
        scope, observation_id, fingerprint = self._current_v3_authority()
        self._v3_confirmation = V3ConfirmationAuthority(
            scope=scope,
            observation_id=observation_id,
            fingerprint=fingerprint,
        )

    def confirm_v3(
        self,
        *,
        confirmed: bool,
        confirmation: dict[str, Any] | None,
    ) -> GenericActionExecutionResult:
        """Atomically validate and consume one exact v3 confirmation."""

        if confirmed is not True or not isinstance(confirmation, dict):
            raise GenericActionAdapterError(
                "缺少当前一步的完整v3确认作用域。"
            )
        authority = self._v3_confirmation
        if authority is None:
            raise GenericActionAdapterError(
                "当前会话没有权威v3确认作用域，拒绝执行。"
            )
        if authority.consumed:
            raise GenericActionAdapterError(
                "当前v3确认已使用或会话已推进，拒绝重放。"
            )
        if self.status != "awaiting_confirmation":
            self.invalidate_v3_confirmation("session_advanced")
            raise GenericActionAdapterError(
                "会话已推进，当前确认不能复用。"
            )

        try:
            current_scope, observation_id, fingerprint = self._current_v3_authority()
        except GenericActionAdapterError:
            self.invalidate_v3_confirmation("authoritative_state_invalid")
            raise
        if (
            current_scope != authority.scope
            or observation_id != authority.observation_id
            or fingerprint != authority.fingerprint
        ):
            self.invalidate_v3_confirmation("authoritative_state_changed")
            raise GenericActionAdapterError(
                "任务、画面或视觉决策已经变化，当前确认已失效。"
            )

        requested_scope = self._normalize_requested_confirmation(confirmation)
        expected_confirmation = {
            **current_scope,
            "observation_id": observation_id,
            "fingerprint": fingerprint,
        }
        if requested_scope != expected_confirmation:
            self.invalidate_v3_confirmation("request_scope_mismatch")
            raise GenericActionAdapterError(
                "确认的任务、设备、revision、子目标、risk_ids或画面身份与当前权威作用域不一致。"
            )

        # Consume before entering the adapter.  Failure and partial failure are
        # deliberately non-retryable with the same user confirmation.
        authority.consumed = True
        authority.invalid_reason = "consumed_before_execution"
        try:
            return self._execute_current_action(_SCOPED_CONFIRMATION_CAPABILITY)
        finally:
            authority.consumed = True
            if not authority.invalid_reason:
                authority.invalid_reason = "execution_finished"

    def invalidate_v3_confirmation(self, reason: str) -> None:
        authority = self._v3_confirmation
        if authority is not None:
            authority.consumed = True
            authority.invalid_reason = str(reason or "invalidated")

    def _current_v3_authority(
        self,
    ) -> tuple[dict[str, Any], str, str]:
        graph = self.task_graph if isinstance(self.task_graph, dict) else None
        decision = self.qwen_decision if isinstance(self.qwen_decision, dict) else None
        if graph is None or decision is None:
            raise GenericActionAdapterError(
                "当前会话没有权威v3确认作用域，拒绝执行。"
            )
        if graph.get("protocol_version") != DEEPSEEK_TASK_GRAPH_V3:
            raise GenericActionAdapterError(
                "当前会话没有权威v3确认作用域，拒绝旧协议确认。"
            )
        gate = graph.get("confirmation_gate")
        current_subgoal = graph.get("current_subgoal")
        if not isinstance(gate, dict) or not isinstance(current_subgoal, dict):
            raise GenericActionAdapterError("v3确认门或当前子目标缺失，拒绝执行。")
        gate_scope = gate.get("scope")
        if not isinstance(gate_scope, dict):
            raise GenericActionAdapterError("v3确认门缺少权威scope，拒绝执行。")
        external_impact = str(
            graph.get("current_external_impact")
            or current_subgoal.get("external_impact")
            or ""
        )
        if (
            gate.get("required") is not True
            or gate.get("state") != "awaiting_confirmation"
            or gate.get("external_state_action_allowed") is not False
            or str(graph.get("task_status") or graph.get("status") or "")
            != "awaiting_confirmation"
            or external_impact not in {"external_state", "unknown"}
        ):
            raise GenericActionAdapterError("当前v3状态没有等待确认的外部动作。")

        raw_risk_ids = gate.get("risk_ids")
        subgoal_risk_ids = current_subgoal.get("risk_action_ids")
        if not isinstance(raw_risk_ids, list) or not isinstance(subgoal_risk_ids, list):
            raise GenericActionAdapterError("v3确认门risk_ids格式无效。")
        risk_ids = sorted(str(item) for item in raw_risk_ids)
        if (
            not risk_ids
            or any(not item for item in risk_ids)
            or len(risk_ids) != len(set(risk_ids))
        ):
            raise GenericActionAdapterError("v3确认门risk_ids缺失或重复。")
        if risk_ids != sorted(str(item) for item in subgoal_risk_ids):
            raise GenericActionAdapterError("v3风险与当前子目标不一致。")
        risk_actions = graph.get("risk_actions")
        if not isinstance(risk_actions, list) or sorted(
            str(item.get("risk_id") or "")
            for item in risk_actions
            if isinstance(item, dict)
        ) != risk_ids:
            raise GenericActionAdapterError("v3确认门与当前风险动作不一致。")

        graph_task_id = str(graph.get("task_id") or "")
        graph_device_id = str(graph.get("device_id") or "")
        graph_revision = graph.get("revision")
        subgoal_id = str(current_subgoal.get("subgoal_id") or "")
        if (
            not graph_task_id
            or not graph_device_id
            or not subgoal_id
            or current_subgoal.get("status") != "active"
            or isinstance(graph_revision, bool)
            or not isinstance(graph_revision, int)
            or graph_revision < 1
        ):
            raise GenericActionAdapterError("v3权威任务身份字段无效。")
        expected_scope = {
            "session_id": self.session_id,
            "task_id": graph_task_id,
            "device_id": graph_device_id,
            "revision": graph_revision,
            "subgoal_id": subgoal_id,
            "risk_ids": risk_ids,
        }
        gate_identity = {
            "task_id": gate_scope.get("task_id"),
            "device_id": gate_scope.get("device_id"),
            "revision": gate_scope.get("revision"),
            "subgoal_id": gate_scope.get("subgoal_id"),
        }
        if gate_identity != {
            "task_id": graph_task_id,
            "device_id": graph_device_id,
            "revision": graph_revision,
            "subgoal_id": subgoal_id,
        }:
            raise GenericActionAdapterError("v3确认门scope与当前任务状态不一致。")
        if graph_device_id != self.device_id:
            raise GenericActionAdapterError("v3任务设备与会话锁定设备不一致。")

        if (
            decision.get("protocol_version")
            not in SUPPORTED_QWEN_VISUAL_DECISION_PROTOCOLS
            or str(decision.get("status") or "") != "action"
        ):
            raise GenericActionAdapterError("Qwen blocked/finished决策不可执行。")
        identity = {
            "task_id": decision.get("task_id"),
            "device_id": decision.get("device_id"),
            "revision": decision.get("revision"),
        }
        if identity != {
            "task_id": graph_task_id,
            "device_id": graph_device_id,
            "revision": graph_revision,
        }:
            raise GenericActionAdapterError("Qwen决策身份已经失效。")
        observation_id = str(decision.get("observation_id") or "")
        fingerprint = str(decision.get("fingerprint") or "")
        trusted = decision.get("trusted_observation")
        trusted_scene = trusted.get("scene") if isinstance(trusted, dict) else None
        if (
            not observation_id
            or not fingerprint
            or fingerprint != self.current_scene.fingerprint
            or not isinstance(trusted, dict)
            or trusted.get("observation_id") != observation_id
            or trusted.get("device_id") != graph_device_id
            or trusted.get("fingerprint") != fingerprint
            or not isinstance(trusted_scene, dict)
            or trusted_scene.get("fingerprint") != fingerprint
        ):
            raise GenericActionAdapterError("当前画面或Qwen决策已经失效。")
        next_action = decision.get("next_action")
        proposal_action = self.proposal.action if self.proposal else None
        if (
            not isinstance(next_action, dict)
            or proposal_action is None
            or next_action != proposal_action.to_dict()
        ):
            raise GenericActionAdapterError("Qwen决策与待执行唯一动作不一致。")
        return expected_scope, observation_id, fingerprint

    @staticmethod
    def _normalize_requested_confirmation(value: dict[str, Any]) -> dict[str, Any]:
        raw_risk_ids = value.get("risk_ids")
        if not isinstance(raw_risk_ids, list):
            raw_risk_ids = []
        return {
            "session_id": str(value.get("session_id") or ""),
            "task_id": str(value.get("task_id") or ""),
            "device_id": str(value.get("device_id") or ""),
            "revision": value.get("revision"),
            "subgoal_id": str(value.get("subgoal_id") or ""),
            "risk_ids": sorted(str(item) for item in raw_risk_ids),
            "observation_id": str(value.get("observation_id") or ""),
            "fingerprint": str(value.get("fingerprint") or ""),
        }

    def run_safe_loop(
        self,
        *,
        confirmed: bool,
        confirmation: dict[str, Any] | None = None,
        max_physical_actions: int = 1,
        max_iterations: int = 1,
    ) -> dict[str, Any]:
        """Fail closed: the retired compatibility loop has no execution grant."""

        if confirmed is not True or not isinstance(confirmation, dict):
            raise GenericActionAdapterError(
                "旧自动推进入口缺少服务端完整确认作用域，拒绝执行。"
            )
        if (
            isinstance(max_physical_actions, bool)
            or not isinstance(max_physical_actions, int)
            or max_physical_actions != 1
        ):
            raise GenericActionAdapterError("每次确认只允许一个物理动作。")
        if (
            isinstance(max_iterations, bool)
            or not isinstance(max_iterations, int)
            or max_iterations != 1
        ):
            raise GenericActionAdapterError("每次确认只允许一个动作轮次。")
        self.invalidate_v3_confirmation("retired_compatibility_loop")
        raise GenericActionAdapterError(
            "旧自动推进执行入口已关闭；请使用通用编排器的精确单步确认接口。"
        )

    def _v3_automatic_block_reason(self) -> str:
        graph = self.task_graph if isinstance(self.task_graph, dict) else None
        if graph is None or graph.get("protocol_version") != DEEPSEEK_TASK_GRAPH_V3:
            return ""
        current = graph.get("current_subgoal")
        gate = graph.get("confirmation_gate")
        if not isinstance(current, dict) or not isinstance(gate, dict):
            return "v3任务缺少权威当前子目标或确认门，自动推进失败关闭。"
        impact = str(
            graph.get("current_external_impact")
            or current.get("external_impact")
            or "unknown"
        )
        task_status = str(graph.get("task_status") or graph.get("status") or "")
        if (
            impact in {"external_state", "unknown"}
            or gate.get("required") is True
            or gate.get("state") == "awaiting_confirmation"
            or task_status == "awaiting_confirmation"
            or bool(gate.get("risk_ids"))
        ):
            return "外部状态、未知影响或等待确认的v3步骤禁止自动推进。"
        decision = self.qwen_decision if isinstance(self.qwen_decision, dict) else None
        if decision is None or decision.get("status") in {"blocked", "finished"}:
            return "Qwen blocked/finished决策没有自动执行入口。"
        if decision.get("status") != "action":
            return "v3自动推进缺少唯一Qwen动作，失败关闭。"
        return ""

    def plan_next(self, scene: UIScene) -> GenericStepProposal:
        self.invalidate_v3_confirmation("replan")
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
        self.invalidate_v3_confirmation("session_cancelled")
        if self.status not in {"succeeded", "cancelled"}:
            self.status = "cancelled"

    def pause(self) -> None:
        self.invalidate_v3_confirmation("user_paused")
        self.automatic_loop_enabled = False
        self.auto_pause_reason = "用户已暂停；旧确认已失效。"

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
            "device_id": self.device_id,
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
                    in {
                        "tap_semantic",
                        "dismiss_overlay",
                        "swipe",
                        "back",
                        "home",
                        "input_verified_text",
                        "long_press",
                        "drag",
                    }
                ),
            },
            "task_graph": copy.deepcopy(self.task_graph),
            "qwen_decision": copy.deepcopy(self.qwen_decision),
            "confirmation_authority": {
                "protocol_version": DEEPSEEK_TASK_GRAPH_V3,
                "available": bool(
                    self._v3_confirmation and not self._v3_confirmation.consumed
                ),
                "consumed": bool(
                    self._v3_confirmation and self._v3_confirmation.consumed
                ),
            },
        }
