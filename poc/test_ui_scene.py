import unittest

from semantic_executor import SemanticAction
from task_orchestrator import (
    GenericTaskOrchestrator,
    GoalSpec,
    PlanNode,
    TaskPlanError,
)
from ui_scene import UIElement, UIScene, UISceneError
from universal_action_controller import (
    UniversalActionController,
    UniversalActionError,
)


def element(
    element_id: str,
    meaning: str,
    *,
    role: str = "button",
    confidence: float = 0.95,
    states=None,
) -> UIElement:
    return UIElement(
        element_id=element_id,
        role=role,
        meaning=meaning,
        label=meaning,
        bounds=(0.2, 0.3, 0.4, 0.5),
        confidence=confidence,
        states=states or {},
        evidence=("visible",),
    )


def scene(*elements: UIElement, app_id="calculator", screen_id="home", fingerprint="a"):
    return UIScene(
        app_id=app_id,
        screen_id=screen_id,
        summary="test",
        elements=tuple(elements),
        confidence=0.95,
        stable=True,
        fingerprint=fingerprint,
    )


class UISceneTests(unittest.TestCase):
    def test_scene_protocol_separates_foreground_app_from_target_goal(self) -> None:
        current = UIScene.from_dict(
            {
                "foreground_app_id": "douyin",
                "screen_id": "android_home",
                "summary": "安卓桌面",
                "elements": [],
                "stable": True,
                "confidence": 0.95,
            }
        )
        self.assertEqual(current.foreground_app_id, "launcher")
        self.assertEqual(current.to_dict()["foreground_app_id"], "launcher")

    def test_arbitrary_app_goal_and_plan_are_valid(self) -> None:
        goal = GoalSpec.from_dynamic(
            app_id="calculator",
            objective="打开计算器并点击数字5",
            success_criteria={"screen_contains": "5"},
        )
        plan = GenericTaskOrchestrator().compile_dynamic(
            goal,
            {
                "root": {
                    "kind": "sequence",
                    "node_id": "root",
                    "children": [
                        {
                            "kind": "action",
                            "node_id": "open",
                            "action": "ensure_app",
                            "params": {"app_id": "calculator"},
                        },
                        {
                            "kind": "action",
                            "node_id": "tap_five",
                            "action": "tap_semantic",
                            "params": {"target": "digit_5"},
                        },
                        {
                            "kind": "action",
                            "node_id": "verify",
                            "action": "verify",
                            "params": {"expected": "display_5"},
                        },
                    ],
                }
            },
        )
        self.assertEqual(plan.app_id, "calculator")
        self.assertEqual(plan.goal, goal)

    def test_dynamic_plan_rejects_raw_coordinates(self) -> None:
        goal = GoalSpec.from_dynamic(app_id="settings", objective="打开设置")
        with self.assertRaisesRegex(TaskPlanError, "禁止字段"):
            GenericTaskOrchestrator().compile_dynamic(
                goal,
                {
                    "root": {
                        "kind": "action",
                        "node_id": "bad",
                        "action": "tap_semantic",
                        "params": {"target": "settings", "x": 0.5},
                    }
                },
            )

    def test_scene_rejects_model_action_fields(self) -> None:
        with self.assertRaisesRegex(UISceneError, "动作字段"):
            element("one", "confirm", states={"next_action": "tap"}).validate()

    def test_unique_semantic_target_is_resolved(self) -> None:
        current = scene(element("five", "digit_5"))
        action = SemanticAction(
            node_id="tap_five",
            action="tap_semantic",
            params={"target": "digit_5"},
        )
        resolved = UniversalActionController().resolve_one(action, current)
        self.assertEqual(resolved.target_element_id, "five")
        self.assertEqual(resolved.normalized_point, (0.30000000000000004, 0.4))

    def test_container_is_valid_scene_structure_but_not_clickable(self) -> None:
        content = element("content", "video_content", role="container")
        current = scene(content)
        current.validate()
        action = SemanticAction(
            node_id="tap_content",
            action="tap_semantic",
            params={"target": "video_content", "role": "container"},
        )
        with self.assertRaisesRegex(UniversalActionError, "页面容器不是可点击控件"):
            UniversalActionController().resolve_one(action, current)

    def test_ambiguous_target_stops_without_guessing(self) -> None:
        current = scene(element("one", "search"), element("two", "search"))
        action = SemanticAction(
            node_id="tap_search",
            action="tap_semantic",
            params={"target": "search"},
        )
        with self.assertRaisesRegex(UniversalActionError, "不唯一"):
            UniversalActionController().resolve_one(action, current)

    def test_account_action_requires_confirmation(self) -> None:
        current = scene(element("send", "send"))
        action = SemanticAction(
            node_id="send",
            action="tap_semantic",
            params={"target": "send"},
        )
        with self.assertRaisesRegex(UniversalActionError, "尚未确认"):
            UniversalActionController().resolve_one(action, current)

    def test_natural_language_like_button_requires_confirmation(self) -> None:
        current = scene(element("heart", "点赞按钮", role="button"))
        action = SemanticAction(
            node_id="like",
            action="tap_semantic",
            params={"target": "点赞按钮", "element_id": "heart"},
        )
        with self.assertRaisesRegex(UniversalActionError, "尚未确认"):
            UniversalActionController().resolve_one(action, current)

    def test_expected_liked_state_requires_confirmation(self) -> None:
        current = scene(element("heart", "reaction_button", role="button"))
        action = SemanticAction(
            node_id="like",
            action="tap_semantic",
            params={
                "target": "reaction_button",
                "element_id": "heart",
                "expected_effect": {
                    "element_state": {
                        "meaning": "reaction_button",
                        "states": {"is_liked": True},
                    }
                },
            },
        )
        with self.assertRaisesRegex(UniversalActionError, "尚未确认"):
            UniversalActionController().resolve_one(action, current)

    def test_action_requires_fresh_verified_scene(self) -> None:
        controller = UniversalActionController()
        before = scene(element("five", "digit_5"), fingerprint="before")
        action = SemanticAction(
            node_id="tap_five",
            action="tap_semantic",
            params={"target": "digit_5"},
        )
        resolved = controller.resolve_one(action, before)
        unchanged = scene(element("five", "digit_5"), fingerprint="before")
        with self.assertRaisesRegex(UniversalActionError, "没有可验证变化"):
            controller.verify_after_action(resolved, before, unchanged)
        changed = scene(element("five", "digit_5"), fingerprint="after")
        controller.verify_after_action(resolved, before, changed)

    def test_verified_input_requires_focused_input_and_preserves_exact_text(self) -> None:
        current = scene(
            element(
                "search-field",
                "搜索输入框",
                role="input",
                states={"focused": True},
            )
        )
        action = SemanticAction(
            node_id="type-query",
            action="input_verified_text",
            params={
                "element_id": "search-field",
                "target": "搜索输入框",
                "role": "input",
                "label": "搜索输入框",
                "states": {"focused": True},
                "text": "蓝牙设置",
            },
        )

        resolved = UniversalActionController().resolve_one(action, current)

        self.assertEqual("input_verified_text", resolved.kind)
        self.assertEqual("蓝牙设置", resolved.text)
        self.assertEqual("search-field", resolved.target_element_id)

    def test_verified_input_rejects_unfocused_field(self) -> None:
        current = scene(element("field", "查询框", role="input"))
        action = SemanticAction(
            node_id="type",
            action="input_verified_text",
            params={"element_id": "field", "target": "查询框", "text": "测试"},
        )

        with self.assertRaisesRegex(UniversalActionError, "已聚焦"):
            UniversalActionController().resolve_one(action, current)

    def test_long_press_has_bounded_duration(self) -> None:
        current = scene(element("item", "列表项目", role="list_item"))
        action = SemanticAction(
            node_id="hold",
            action="long_press",
            params={
                "element_id": "item",
                "target": "列表项目",
                "duration_ms": 900,
            },
        )

        resolved = UniversalActionController().resolve_one(action, current)

        self.assertEqual("long_press", resolved.kind)
        self.assertEqual(0.9, resolved.hold_seconds)

    def test_drag_resolves_two_distinct_semantic_elements(self) -> None:
        source = element("source", "待移动项目", role="list_item")
        destination = UIElement(
            element_id="destination",
            role="container",
            meaning="目标区域",
            label="目标区域",
            bounds=(0.6, 0.6, 0.9, 0.9),
            confidence=0.95,
            evidence=("visible",),
        )
        current = scene(source, destination)
        action = SemanticAction(
            node_id="drag",
            action="drag",
            params={
                "source_element_id": "source",
                "source_target": "待移动项目",
                "destination_element_id": "destination",
                "destination_target": "目标区域",
            },
        )

        resolved = UniversalActionController().resolve_one(action, current)

        self.assertEqual(source.center, resolved.normalized_point)
        self.assertEqual(destination.center, resolved.normalized_end_point)


if __name__ == "__main__":
    unittest.main()
