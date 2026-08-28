"""Deterministic application planner for one capability acceptance action."""

from __future__ import annotations

from dataclasses import replace
from typing import Final
import uuid

from agent.domain.action_capabilities import PROMOTABLE_ACTIONS
from agent.domain.task_graph import (
    CompletionCondition,
    DynamicTaskGraph,
    GraphGoal,
    ObservedState,
    ReplanRecord,
    Subgoal,
    TargetApp,
    TaskGraphError,
)


class CapabilityAcceptancePlannerError(TaskGraphError):
    pass


_SEMANTIC_CONTRACTS: Final[dict[str, tuple[str, str]]] = {
    "tap_semantic": (
        "目标视觉控件触发后的页面状态可见",
        "新画面显示目标视觉控件预期的结构化状态变化",
    ),
    "dismiss_overlay": (
        "当前遮挡层关闭后的底层页面可见",
        "新画面显示遮挡层消失且底层页面保持可见",
    ),
    "swipe": (
        "当前页面沿目标方向移动后的新内容可见",
        "新画面显示与目标方向一致的结构化页面位移",
    ),
    "back": (
        "当前页面的上一级状态可见",
        "新画面显示可验证的上一级页面状态",
    ),
    "home": (
        "设备系统主屏幕可见",
        "新画面显示可验证的系统主屏幕状态",
    ),
    "reveal_system_navigation": (
        "设备系统导航区域可见",
        "新画面显示结构化系统导航区域",
    ),
    "input_verified_text": (
        "目标输入区域显示指定的本地临时文字",
        "新画面逐字显示指定文字且没有提交或发送",
    ),
    "long_press": (
        "目标视觉对象显示持续触发后的可见状态",
        "新画面显示仅由持续触发产生的结构化状态变化",
    ),
    "drag": (
        "源视觉对象稳定到达目标视觉区域",
        "新画面显示源对象相对目标区域发生可验证的位置变化",
    ),
}


class CapabilityAcceptanceTaskGraphPlanner:
    """Deterministic graph contract used only to certify one primitive.

    Production sessions continue to use ``DeepSeekTaskGraphPlanner``.  A
    capability trial has a narrower job: prove that one already selected
    primitive can pass the normal Qwen, policy, controller and re-observation
    gates.  Re-running stochastic task decomposition here adds no safety and
    makes the same physical primitive depend on incidental graph wording.
    """

    def __init__(self, candidate_action: str) -> None:
        action = str(candidate_action or "").strip()
        if action not in PROMOTABLE_ACTIONS or action not in _SEMANTIC_CONTRACTS:
            raise CapabilityAcceptancePlannerError(f'动作 {action or 'missing'} 没有确定性的能力验收合同。')
        self.candidate_action = action

    def plan( self, raw_goal: str, *, device_id: str, task_id: str | None = None, ) -> DynamicTaskGraph:
        text = " ".join(str(raw_goal or "").split())
        if not text:
            raise CapabilityAcceptancePlannerError("能力验收目标不能为空。")
        objective, completion = _SEMANTIC_CONTRACTS[self.candidate_action]
        graph = DynamicTaskGraph(
            task_id=str(task_id or uuid.uuid4().hex),
            device_id=str(device_id or "").strip(),
            revision=1,
            status="ready",
            goal=GraphGoal(
                objective=objective,
                target_apps=(
                    TargetApp(
                        app_id="current_foreground_surface",
                        app_name="当前前台手机界面",
                    ),
                ),
                entities={
                    "capability_trial_kind": self.candidate_action,
                },
            ),
            constraints=(
                "仅允许当前本地验收页面内可由新画面验证的可逆变化",
                "不得提交、发送、删除、支付、修改账号或产生其他外部状态影响",
                "每次确认只允许一个物理动作且禁止自动重试",
            ),
            completion_conditions=(
                CompletionCondition(
                    condition_id="primitive_result_visible",
                    description=completion,
                    evidence_required=(
                        "动作后重新采集的可信画面",
                        "结构化动作结果与页面变化证据",
                    ),
                ),
            ),
            risk_actions=(),
            subgoals=(
                Subgoal(
                    subgoal_id="certify_primitive",
                    objective=objective,
                    status="active",
                    depends_on=(),
                    constraints=(
                        "只使用当前可信画面中的唯一视觉对象",
                        "动作后必须重新观察并验证结构化结果",
                    ),
                    completion_conditions=(completion,),
                    completion_evidence=(),
                    risk_action_ids=(),
                    external_impact="navigation_only",
                ),
            ),
            active_subgoal_id="certify_primitive",
            raw_user_goal=text,
        )
        graph.validate()
        return graph

    def replan(
        self,
        graph: DynamicTaskGraph,
        observation: ObservedState,
        *,
        trigger: str,
        reason: str,
    ) -> DynamicTaskGraph:
        graph.validate()
        observation.validate()
        if graph.raw_user_goal == "" or graph.active_subgoal_id != "certify_primitive":
            raise CapabilityAcceptancePlannerError('能力验收任务图身份或活动节点已变化。')
        evidence = tuple(
            dict.fromkeys(
                (
                    observation.summary,
                    *observation.visible_evidence,
                    *observation.grounded_visual_facts,
                )
            )
        )
        record = ReplanRecord(
            revision=graph.revision + 1,
            trigger=str(trigger or "").strip(),
            reason=str(reason or "").strip(),
            scene_id=observation.scene_id,
            evidence=evidence,
            retained_completed_subgoal_ids=(),
            added_subgoal_ids=(),
            skipped_subgoal_ids=(),
        )
        if observation.last_action_outcome == "matched" and not observation.blocked_reasons:
            condition = replace(graph.completion_conditions[0], satisfied=True, evidence=evidence)
            subgoal = replace(graph.subgoals[0], status='completed', completion_evidence=evidence)
            revised = replace(
                graph,
                revision=graph.revision + 1,
                status="completed",
                completion_conditions=(condition,),
                subgoals=(subgoal,),
                active_subgoal_id=None,
                replan_history=graph.replan_history + (record,),
            )
        else:
            blocked_evidence = observation.blocked_reasons or (f'动作结果为 {observation.last_action_outcome}',)
            subgoal = replace(graph.subgoals[0], status="blocked")
            revised = replace(
                graph,
                revision=graph.revision + 1,
                status="blocked",
                subgoals=(subgoal,),
                active_subgoal_id=None,
                clarification_questions=(
                    "当前真机证据未证明候选动作成功；本次验收必须停止。",
                ),
                replan_history=(
                    graph.replan_history
                    + (
                        replace(
                            record,
                            evidence=tuple(
                                dict.fromkeys((*evidence, *blocked_evidence))
                            ),
                        ),
                    )
                ),
            )
        revised.validate()
        return revised
