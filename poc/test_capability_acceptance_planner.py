from __future__ import annotations

import unittest

from agent.application.capability_acceptance_planner import (
    CapabilityAcceptancePlannerError,
    CapabilityAcceptanceTaskGraphPlanner,
)


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

    def test_rejects_unknown_or_empty_candidate(self) -> None:
        for value in ("", "wait_for_change", "unknown"):
            with self.subTest(value=value):
                with self.assertRaises(CapabilityAcceptancePlannerError):
                    CapabilityAcceptanceTaskGraphPlanner(value)


if __name__ == "__main__":
    unittest.main()
