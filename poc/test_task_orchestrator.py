import unittest

from task_orchestrator import (
    ALLOWED_ACTIONS,
    GenericTaskOrchestrator,
    GoalSpec,
    PlanNode,
    TaskPlan,
    TaskPlanError,
)


def flatten(node: PlanNode):
    yield node
    for child in (*node.children, *node.otherwise):
        yield from flatten(child)


class GenericTaskOrchestratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.orchestrator = GenericTaskOrchestrator()

    def test_is_dormant_and_has_no_hardware_execution(self) -> None:
        status = self.orchestrator.status()
        self.assertTrue(status["available"])
        self.assertFalse(status["execution_enabled"])

    def test_ten_likes_compile_to_generic_bounded_loop(self) -> None:
        plan = self.orchestrator.compile(
            "douyin.batch_interact",
            {
                "target_count": 10,
                "like": True,
                "comment": False,
            },
        )
        plan.validate()
        nodes = list(flatten(plan.root))
        loop = next(node for node in nodes if node.node_id == "video_loop")
        self.assertEqual(loop.max_iterations, 15)
        self.assertEqual(loop.condition.params["value"], 10)
        actions = {node.action for node in nodes if node.kind == "action"}
        self.assertIn("ensure_app", actions)
        self.assertIn("observe", actions)
        self.assertIn("tap_semantic", actions)
        self.assertIn("swipe", actions)
        self.assertTrue(actions.issubset(ALLOWED_ACTIONS))
        self.assertIn("record_verified_result", actions)
        continue_branch = next(
            node for node in nodes if node.node_id == "continue_if_needed"
        )
        self.assertEqual(continue_branch.children[0].action, "swipe")

    def test_goal_spec_separates_goal_from_execution_plan(self) -> None:
        goal = GoalSpec.from_operation(
            "douyin.batch_interact",
            {"target_count": 3, "like": True, "comment": False},
            objective="给接下来三个普通视频点赞",
        )
        self.assertEqual(goal.app_id, "douyin")
        self.assertEqual(goal.success_criteria["target_count"], 3)
        self.assertEqual(goal.limits["max_pages"], 8)
        plan = self.orchestrator.compile_goal(goal)
        self.assertEqual(plan.goal, goal)
        self.assertFalse(self.orchestrator.execution_enabled)

    def test_album_image_uses_same_generic_primitives(self) -> None:
        plan = self.orchestrator.compile(
            "wechat.send_album_image",
            {"chat_name": "文件传输助手", "image_index": 2},
        )
        nodes = list(flatten(plan.root))
        select_image = next(node for node in nodes if node.node_id == "select_image")
        self.assertEqual(select_image.action, "tap_semantic")
        self.assertEqual(select_image.params["index"], 2)

    def test_unsafe_goal_parameter_is_rejected_before_planning(self) -> None:
        with self.assertRaisesRegex(TaskPlanError, "禁止字段"):
            GoalSpec.from_operation(
                "douyin.search",
                {"keyword": "机械臂", "x": 100},
            )

    def test_plan_contains_no_coordinates_or_vendor_commands(self) -> None:
        plan = self.orchestrator.compile(
            "douyin.search",
            {"keyword": "机械臂"},
        ).to_dict()
        serialized = repr(plan).lower()
        self.assertNotIn("coordinate", serialized)
        self.assertNotIn("main_exe", serialized)
        self.assertNotIn("powershell", serialized)

    def test_wechat_text_is_composed_from_same_generic_primitives(self) -> None:
        plan = self.orchestrator.compile(
            "wechat.send_text",
            {"chat_name": "文件传输助手", "text": "你好"},
        )
        nodes = list(flatten(plan.root))
        actions = [node.action for node in nodes if node.kind == "action"]
        self.assertEqual(actions[0], "ensure_app")
        self.assertIn("input_verified_text", actions)
        self.assertEqual(actions[-1], "finish")

    def test_raw_coordinate_is_rejected(self) -> None:
        plan = TaskPlan(
            app_id="douyin",
            objective="非法坐标测试",
            source_operation="test",
            root=PlanNode(
                kind="action",
                node_id="bad",
                action="tap_semantic",
                params={"target": "heart", "x": 100, "y": 200},
            ),
        )
        with self.assertRaisesRegex(TaskPlanError, "禁止字段"):
            plan.validate()

    def test_shell_action_is_rejected(self) -> None:
        plan = TaskPlan(
            app_id="douyin",
            objective="非法命令测试",
            source_operation="test",
            root=PlanNode(
                kind="action",
                node_id="bad",
                action="shell",
                params={"value": "whoami"},
            ),
        )
        with self.assertRaisesRegex(TaskPlanError, "不允许的通用动作"):
            plan.validate()

    def test_duplicate_node_ids_are_rejected(self) -> None:
        plan = TaskPlan(
            app_id="douyin",
            objective="重复节点测试",
            source_operation="test",
            root=PlanNode(
                kind="sequence",
                node_id="root",
                children=(
                    PlanNode(kind="action", node_id="same", action="observe"),
                    PlanNode(kind="action", node_id="same", action="observe"),
                ),
            ),
        )
        with self.assertRaisesRegex(TaskPlanError, "重复"):
            plan.validate()


if __name__ == "__main__":
    unittest.main()
