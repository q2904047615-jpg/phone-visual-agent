from __future__ import annotations

from dataclasses import replace
import unittest

from capability_acceptance_planner import (
    CapabilityAcceptancePlannerError,
    CapabilityAcceptanceTaskGraphPlanner,
)
from deepseek_task_graph import ObservedState


class CapabilityAcceptanceTaskGraphPlannerTests(unittest.TestCase):
    def test_drag_plan_is_deterministic_high_level_and_navigation_only(self) -> None:
        planner = CapabilityAcceptanceTaskGraphPlanner("drag")
        first = planner.plan(
            "把唯一紫色起点拖到唯一绿色终点",
            device_id="device-local-01",
            task_id="trial-task",
        )
        second = planner.plan(
            "另一种自然语言描述",
            device_id="device-local-01",
            task_id="trial-task",
        )

        self.assertEqual(first.goal, second.goal)
        self.assertEqual(first.subgoals, second.subgoals)
        self.assertEqual("navigation_only", first.subgoals[0].external_impact)
        self.assertNotIn("拖", first.goal.objective)
        self.assertNotIn("drag", first.goal.objective.casefold())
        self.assertEqual("drag", first.goal.entities["capability_trial_kind"])
        self.assertEqual(1, first.revision)
        self.assertEqual("ready", first.status)

    def test_matched_reobservation_completes_without_model_call(self) -> None:
        planner = CapabilityAcceptanceTaskGraphPlanner("drag")
        graph = planner.plan(
            "把唯一紫色起点拖到唯一绿色终点",
            device_id="device-local-01",
            task_id="trial-task",
        )
        observation = ObservedState(
            scene_id="after-scene",
            summary="紫色对象已进入绿色目标区域",
            visible_evidence=("源对象中心位于目标区域内",),
            grounded_visual_facts=("源对象相对位置已变化",),
            last_action_outcome="matched",
        )

        revised = planner.replan(
            graph,
            observation,
            trigger="observation_changed",
            reason="一个动作已经执行并重新观察。",
        )

        self.assertEqual("completed", revised.status)
        self.assertIsNone(revised.active_subgoal_id)
        self.assertTrue(revised.completion_conditions[0].satisfied)
        self.assertEqual("completed", revised.subgoals[0].status)
        self.assertEqual(2, revised.revision)
        revised.validate()

    def test_mismatch_blocks_and_never_retries(self) -> None:
        planner = CapabilityAcceptanceTaskGraphPlanner("drag")
        graph = planner.plan(
            "把唯一紫色起点拖到唯一绿色终点",
            device_id="device-local-01",
            task_id="trial-task",
        )
        observation = ObservedState(
            scene_id="after-scene",
            summary="源对象位置没有变化",
            visible_evidence=("源对象仍在原位置",),
            last_action_outcome="mismatched",
            blocked_reasons=("未证明源对象向目标区域移动",),
        )

        revised = planner.replan(
            graph,
            observation,
            trigger="action_result_mismatch",
            reason="动作结果不匹配。",
        )

        self.assertEqual("blocked", revised.status)
        self.assertIsNone(revised.active_subgoal_id)
        self.assertEqual("blocked", revised.subgoals[0].status)
        self.assertFalse(revised.completion_conditions[0].satisfied)
        revised.validate()

    def test_rejects_unknown_or_empty_candidate(self) -> None:
        for value in ("", "wait_for_change", "unknown"):
            with self.subTest(value=value):
                with self.assertRaises(CapabilityAcceptancePlannerError):
                    CapabilityAcceptanceTaskGraphPlanner(value)


if __name__ == "__main__":
    unittest.main()
