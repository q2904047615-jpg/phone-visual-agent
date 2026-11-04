from __future__ import annotations
from agent.domain.semantic_action import SemanticAction
from agent.domain.ui_scene import UIElement
from agent.domain.ui_scene import UIScene
from agent.domain.ui_scene import UISceneError
from agent.domain.universal_action_controller import UniversalActionController
from agent.domain.universal_action_controller import UniversalActionError
from copy import deepcopy
from dataclasses import replace
import unittest
from test_support.ui_scene import (
    element,
    scene,
)


class UISceneTests(unittest.TestCase):
    def test_scene_rejects_action_plan_or_raw_coordinate_injection_at_any_depth(self) -> None:
        base = {
            "foreground_app_id": "sample.app",
            "screen_id": "main",
            "summary": "页面显示唯一目标按钮",
            "system_ui": {
                "immersive_or_fullscreen": False,
                "navigation_bar_visible": True,
            },
            "camera_alignment": {
                "camera_layout_orientation": "portrait",
                "phone_content_rotation": "upright",
                "confidence": 0.95,
                "evidence": ["手机页面文字正向显示"],
            },
            "elements": [
                {
                    "element_id": "target",
                    "role": "button",
                    "meaning": "open_target",
                    "label": "打开",
                    "bounds": [0.1, 0.2, 0.4, 0.3],
                    "confidence": 0.95,
                    "states": {"goal_relevant": True},
                    "evidence": ["打开"],
                }
            ],
            "stable": True,
            "confidence": 0.95,
        }
        targets = {
            "scene": lambda payload: payload,
            "element": lambda payload: payload["elements"][0],
            "system_ui": lambda payload: payload["system_ui"],
            "camera_alignment": lambda payload: payload["camera_alignment"],
        }
        injections = {
            "action": "tap_semantic",
            "plan": ["tap target", "tap another target"],
            "coordinates": [120, 240],
        }
        for scope, select in targets.items():
            for field, value in injections.items():
                with self.subTest(scope=scope, field=field), self.assertRaisesRegex(
                    UISceneError, "动作|计划|坐标"
                ):
                    payload = deepcopy(base)
                    select(payload)[field] = value
                    UIScene.from_dict(payload)

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

    def test_home_screen_preserves_trusted_concrete_launcher_package(self) -> None:
        current = UIScene.from_dict({"foreground_app_id": "com.miui.home", "screen_id": "android_home",
            "summary": "安卓桌面", "elements": [], "stable": True, "confidence": 0.95})

        self.assertEqual("com.miui.home", current.foreground_app_id)
        self.assertEqual("com.miui.home", current.foreground_app_id)

    def test_scene_rejects_model_action_fields(self) -> None:
        with self.assertRaisesRegex(UISceneError, "动作字段"):
            element("one", "confirm", states={"next_action": "tap"}).validate()

    def test_qwen_direct_tap_point_is_the_only_point_authority(self) -> None:
        target = replace(
            element("target-row", "open_target", role="list_item"),
            label="目标条目",
            bounds=(0.07, 0.29, 0.93, 0.39),
        )
        current = scene(target, fingerprint="fresh-frame")
        action = SemanticAction(
            node_id="open-target",
            action="tap_semantic",
            params={
                "element_id": target.element_id,
                "target": target.meaning,
                "role": target.role,
                "label": target.label,
                "tap_point": (0.50, 0.22),
            },
        )
        resolved = UniversalActionController().resolve_one(action, current)

        self.assertEqual((0.50, 0.22), resolved.normalized_point)
        self.assertNotEqual(target.center, resolved.normalized_point)
        self.assertEqual("target-row", resolved.target_element_id)
        self.assertNotIn("point_grounding", resolved.to_dict())
        self.assertNotIn("proposed_normalized_point", resolved.to_dict())

    def test_input_tap_point_must_land_inside_same_frame_input_bounds(self) -> None:
        target = replace(
            element("input", "chat_message_input", role="input"),
            bounds=(0.12, 0.922, 0.78, 0.969),
            states={"input_field_id": "current_input", "focused": False},
        )
        current = scene(target, fingerprint="fresh-input-frame")
        action = SemanticAction(
            node_id="focus-input",
            action="tap_semantic",
            params={
                "element_id": target.element_id,
                "target": target.meaning,
                "role": target.role,
                "tap_point": (0.0, 0.001),
            },
        )
        with self.assertRaisesRegex(UniversalActionError, "输入框聚焦落点"):
            UniversalActionController().resolve_one(action, current)

    def test_input_tap_point_inside_bounds_is_accepted(self) -> None:
        target = replace(
            element("input", "chat_message_input", role="input"),
            bounds=(0.12, 0.922, 0.78, 0.969),
            states={"input_field_id": "current_input", "focused": False},
        )
        current = scene(target, fingerprint="fresh-input-frame")
        action = SemanticAction(
            node_id="focus-input",
            action="tap_semantic",
            params={
                "element_id": target.element_id,
                "target": target.meaning,
                "role": target.role,
                "tap_point": (0.38, 0.945),
            },
        )
        resolved = UniversalActionController().resolve_one(action, current)
        self.assertEqual((0.38, 0.945), resolved.normalized_point)

    def test_low_scene_confidence_does_not_veto_screen_wide_action(self) -> None:
        current = scene(
            element("tab-list", "open_tab_list", states={"goal_relevant": True}),
            confidence=0.01,
        )
        action = SemanticAction(
            node_id="scroll",
            action="scroll",
            params={"direction": "up"},
        )

        resolved = UniversalActionController().resolve_one(action, current)

        self.assertEqual("scroll", resolved.kind)

    def test_malformed_unselected_scene_fact_and_surplus_count_do_not_veto_unique_target(self) -> None:
        raw_elements = [
            {
                "element_id": "target",
                "role": "button",
                "meaning": "open_target",
                "bounds": [0.1, 0.2, 0.4, 0.3],
                "states": {"goal_relevant": True, "visible": True, "enabled": True},
            },
            {
                "element_id": "bad-optional",
                "role": "button",
                "meaning": "decorative_suggestion",
                "bounds": [0.8, 0.2, 0.7, 0.3],
                # A redundant goal hint is not the same-envelope selected ID.
                "states": {"goal_relevant": True},
            },
        ]
        raw_elements.extend(
            {
                "element_id": f"optional-{index}",
                "role": "text",
                "meaning": "optional_context",
                "bounds": [0.01, 0.01, 0.02, 0.02],
                "states": {"goal_relevant": False},
            }
            for index in range(65)
        )

        current = UIScene.from_dict({
            "foreground_app_id": "sample.app",
            "screen_id": "main",
            "summary": "页面显示唯一可执行目标",
            "elements": raw_elements,
        })

        self.assertEqual("target", current.get_element("target").element_id)
        self.assertNotIn("bad-optional", {item.element_id for item in current.elements})
        self.assertEqual(66, len(current.elements))

    def test_non_actionable_goal_context_does_not_veto_unique_action_target(self) -> None:
        target = element(
            "wechat",
            "launch_wechat",
            states={"goal_relevant": True, "fully_visible": True},
        )
        page_context = element(
            "pages",
            "paged_viewport",
            role="container",
            states={
                "goal_relevant": True,
                "fully_visible": True,
                "scrollable": True,
                "scroll_axis": "horizontal",
                "page_index": 2,
                "page_count": 4,
            },
        )

        self.assertEqual(target, scene(target, page_context).get_element("wechat"))

    def test_completion_evidence_does_not_veto_screen_action(self) -> None:
        evidence = element(
            "visible-count",
            "four_tabs_visible",
            role="container",
            states={"goal_relevant": True},
        )
        current = scene(evidence, confidence=0.6)
        self.assertEqual(evidence, current.get_element("visible-count"))

        resolved = UniversalActionController().resolve_one(
            SemanticAction(
                node_id="scroll",
                action="scroll",
                params={"direction": "up"},
            ),
            current,
        )

        self.assertEqual("scroll", resolved.kind)

    def test_recent_page_does_not_forbid_non_cleanup_element_gestures(self) -> None:
        target = element("resize-handle", "resize_handle", role="container",
            bounds=(0.1, 0.2, 0.6, 0.75))
        current = scene(target, app_id="system", screen_id="system_recent_tasks")
        trajectories = {
            "left": ((0.4, 0.5), (0.08, 0.5)),
            "right": ((0.3, 0.5), (0.7, 0.5)),
            "up": ((0.4, 0.5), (0.4, 0.1)),
            "down": ((0.4, 0.5), (0.4, 0.9)),
        }
        for direction, (start, end) in trajectories.items():
            with self.subTest(direction=direction):
                resolved = UniversalActionController().resolve_one(
                    SemanticAction(node_id="resize", action="swipe_element", params={
                        "element_id": "resize-handle", "start": start, "end": end,
                    }), current)
                self.assertEqual(direction, resolved.direction)

    def test_recent_page_does_not_restrict_navigation_to_vertical_scroll(self) -> None:
        current = scene(app_id="system", screen_id="system_recent_tasks")
        controller = UniversalActionController()
        for direction in ("up", "down", "left", "right"):
            with self.subTest(direction=direction):
                resolved = controller.resolve_one(SemanticAction(
                    node_id="find-card", action="scroll", params={"direction": direction}), current)
                self.assertEqual(direction, resolved.direction)

    def test_element_swipe_rejects_start_outside_target(self) -> None:
        target = element("recent-card", "recent_app_card", role="container",
            bounds=(0.1, 0.2, 0.6, 0.75))
        with self.assertRaisesRegex(UniversalActionError, "目标元素内"):
            UniversalActionController().resolve_one(
                SemanticAction(node_id="dismiss-card", action="swipe_element", params={
                    "element_id": "recent-card", "start": (0.7, 0.5), "end": (0.08, 0.5),
                }),
                scene(target),
            )

    def test_container_is_valid_current_frame_point_target(self) -> None:
        content = element("content", "video_content", role="container")
        current = scene(content)
        current.validate()
        action = SemanticAction(
            node_id="tap_content",
            action="tap_semantic",
            params={"target": "video_content", "role": "container", "element_id": "content",
                "tap_point": content.center},
        )
        resolved = UniversalActionController().resolve_one(action, current)
        self.assertEqual("content", resolved.target_element_id)
        self.assertEqual(content.center, resolved.normalized_point)

    def test_ordinary_send_does_not_require_controller_confirmation(self) -> None:
        current = scene(element("send", "send"))
        action = SemanticAction(
            node_id="send",
            action="tap_semantic",
            params={"target": "send", "element_id": "send", "tap_point": (0.3, 0.4)},
        )
        resolved = UniversalActionController().resolve_one(action, current)
        self.assertEqual("tap_semantic", resolved.kind)
        self.assertEqual("send", resolved.target_element_id)

    def test_projected_target_role_matches_while_optional_display_states_do_not_veto(self) -> None:
        selected = element(
            "current-target",
            "current_meaning",
            states={
                "visible": False,
                "fully_visible": False,
                "enabled": False,
                "occluded": True,
            },
        )
        resolved = UniversalActionController().resolve_one(
            SemanticAction(
                node_id="tap-current",
                action="tap_semantic",
                params={
                    "element_id": "current-target",
                    "target": "stale copied meaning",
                    "role": "button",
                    "label": "stale copied label",
                    "states": "stale copied states",
                    "tap_point": (0.3, 0.4),
                },
            ),
            scene(selected),
        )

        self.assertEqual("current-target", resolved.target_element_id)
        self.assertEqual((0.3, 0.4), resolved.normalized_point)

    def test_ordinary_account_mutation_target_does_not_require_controller_confirmation(self) -> None:
        current = scene(element("heart", "reaction_button", role="button"))
        action = SemanticAction(
            node_id="like",
            action="tap_semantic",
            params={
                "target": "reaction_button",
                "element_id": "heart",
                "tap_point": (0.3, 0.4),
            },
        )
        resolved = UniversalActionController().resolve_one(action, current)
        self.assertEqual("tap_semantic", resolved.kind)
        self.assertEqual("heart", resolved.target_element_id)

    def test_action_requires_fresh_verified_scene(self) -> None:
        controller = UniversalActionController()
        before = scene(element("five", "digit_5"), fingerprint="before")
        action = SemanticAction(
            node_id="tap_five",
            action="tap_semantic",
            params={"target": "digit_5", "element_id": "five", "tap_point": (0.3, 0.4)},
        )
        resolved = controller.resolve_one(action, before)
        unchanged = scene(element("five", "digit_5"), fingerprint="before")
        self.assertEqual((), controller.verify_after_action(resolved, before, unchanged))
        camera_noise_only = scene(element("five", "digit_5"), fingerprint="noise")
        self.assertEqual((), controller.verify_after_action(resolved, before, camera_noise_only))
        changed = scene(element("result", "result_page"), fingerprint="after")
        self.assertEqual((), controller.verify_after_action(resolved, before, changed))

    def test_android_home_resolves_as_independent_system_action(self) -> None:
        current = scene(element("title", "设置", role="text"), app_id="settings")
        action = SemanticAction(
            node_id="return-to-launcher",
            action="home",
            params={},
        )

        resolved = UniversalActionController().resolve_one(action, current)

        self.assertEqual("home", resolved.kind)
        self.assertIsNone(resolved.normalized_point)
        self.assertNotIn("expected_effect", resolved.to_dict())

    def test_long_press_has_bounded_duration(self) -> None:
        current = scene(element("item", "列表项目", role="list_item"))
        action = SemanticAction(
            node_id="hold",
            action="long_press",
            params={
                "element_id": "item",
                "target": "列表项目",
                "duration_ms": 900,
                "tap_point": (0.3, 0.4),
            },
        )

        resolved = UniversalActionController().resolve_one(action, current)

        self.assertEqual("long_press", resolved.kind)
        self.assertEqual(0.9, resolved.hold_seconds)

    def test_double_tap_resolves_one_current_frame_target(self) -> None:
        current = scene(element("preview", "预览图", role="list_item"))
        action = SemanticAction(
            node_id="double",
            action="double_tap",
            params={
                "element_id": "preview",
                "target": "预览图",
                "tap_point": (0.3, 0.4),
            },
        )

        resolved = UniversalActionController().resolve_one(action, current)

        self.assertEqual("double_tap", resolved.kind)
        self.assertAlmostEqual(0.3, resolved.normalized_point[0])
        self.assertAlmostEqual(0.4, resolved.normalized_point[1])
        without_postcondition = UniversalActionController().resolve_one(
            SemanticAction(
                node_id="double",
                action="double_tap",
                params={"element_id": "preview", "target": "预览图", "tap_point": (0.3, 0.4)},
            ),
            current,
        )
        self.assertEqual("preview", without_postcondition.target_element_id)

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
        self.assertEqual(0.8, resolved.hold_seconds)
        self.assertGreaterEqual(resolved.path_distance, 0.08)

    def test_long_press_allows_edge_points_without_visual_postcondition(self) -> None:
        edge = UIElement(
            element_id="edge",
            role="button",
            meaning="边缘控件",
            label="边缘控件",
            bounds=(0.0, 0.0, 0.02, 0.02),
            confidence=0.95,
        )
        ordinary = scene(element("item", "列表项目"))
        resolved = UniversalActionController().resolve_one(
            SemanticAction(
                node_id="hold", action="long_press",
                params={"element_id": "item", "target": "列表项目", "duration_ms": 800,
                    "tap_point": ordinary.elements[0].center},
            ),
            ordinary,
        )
        self.assertEqual("long_press", resolved.kind)
        edge_result = UniversalActionController().resolve_one(
                SemanticAction(
                    node_id="hold", action="long_press",
                    params={"element_id": "edge", "target": "边缘控件", "duration_ms": 800,
                        "tap_point": edge.center},
                ),
                scene(edge),
            )

        self.assertEqual(edge.center, edge_result.normalized_point)

    def test_drag_allows_short_long_and_edge_paths(self) -> None:
        cases = (
            ((0.20, 0.20, 0.30, 0.30), (0.21, 0.21, 0.31, 0.31), "中心距离"),
            ((0.02, 0.02, 0.08, 0.08), (0.90, 0.90, 0.98, 0.98), "中心距离"),
            ((0.0, 0.0, 0.02, 0.02), (0.20, 0.20, 0.30, 0.30), "画面边缘"),
        )
        for source_bounds, destination_bounds, message in cases:
            source = UIElement("source", "button", "源", source_bounds, 0.95, label="源")
            destination = UIElement(
                "destination",
                "container",
                "目标",
                destination_bounds,
                0.95,
                label="目标",
            )
            with self.subTest(former_rejection=message):
                result = UniversalActionController().resolve_one(
                    SemanticAction(
                        node_id="drag",
                        action="drag",
                        params={
                            "source_element_id": "source",
                            "source_target": "源",
                            "destination_element_id": "destination",
                            "destination_target": "目标",
                        },
                    ),
                    scene(source, destination),
                )

                self.assertEqual(source.center, result.normalized_point)
                self.assertEqual(destination.center, result.normalized_end_point)

    def test_drag_result_is_left_to_next_qwen_observation(self) -> None:
        source = UIElement(
            "source", "button", "源", (0.10, 0.20, 0.20, 0.30), 0.95, label="源"
        )
        destination = UIElement(
            "destination",
            "container",
            "目标",
            (0.70, 0.20, 0.90, 0.40),
            0.95,
            label="目标",
        )
        before = scene(source, destination, fingerprint="before")
        resolved = UniversalActionController().resolve_one(
            SemanticAction(
                node_id="drag",
                action="drag",
                params={
                    "source_element_id": "source",
                    "source_target": "源",
                    "destination_element_id": "destination",
                    "destination_target": "目标",
                },
            ),
            before,
        )
        unmoved = scene(source, destination, fingerprint="after")
        self.assertEqual((), UniversalActionController().verify_after_action(resolved, before, unmoved))

        moved_source = UIElement(
            "source", "button", "源", (0.55, 0.20, 0.65, 0.30), 0.95, label="源"
        )
        moved = scene(moved_source, destination, fingerprint="after-moved")
        UniversalActionController().verify_after_action(resolved, before, moved)

    def test_drag_controller_accepts_current_frame_container_source(self) -> None:
        source = UIElement(
            "source",
            "container",
            "可移动源物体",
            (0.18, 0.68, 0.38, 0.82),
            0.98,
            label="起点",
            states={"goal_relevant": True, "fully_visible": True},
        )
        destination = UIElement(
            "destination",
            "container",
            "目标区域",
            (0.55, 0.65, 0.85, 0.88),
            0.98,
            label="绿色终点",
            states={"goal_relevant": True, "fully_visible": True},
        )
        before = scene(source, destination, fingerprint="compact-before")
        action = SemanticAction(
            node_id="drag-container",
            action="drag",
            params={
                "source_element_id": source.element_id,
                "source_target": source.meaning,
                "source_role": source.role,
                "source_label": source.label,
                "source_states": dict(source.states),
                "destination_element_id": destination.element_id,
                "destination_target": destination.meaning,
                "destination_role": destination.role,
                "destination_label": destination.label,
                "destination_states": dict(destination.states),
            },
        )

        resolved = UniversalActionController().resolve_one(action, before)
        self.assertEqual("drag", resolved.kind)

        moved_source = replace(source, bounds=(0.60, 0.68, 0.78, 0.82))
        after = scene(moved_source, destination, fingerprint="compact-after")
        UniversalActionController().verify_after_action(resolved, before, after)

        oversized_source = replace(source, bounds=(0.02, 0.05, 0.98, 0.90))
        oversized = UniversalActionController().resolve_one(
            action,
            scene(oversized_source, destination, fingerprint="oversized"),
        )
        self.assertEqual("source", oversized.target_element_id)

    def test_drag_preexisting_unrelated_state_does_not_create_controller_authority(self) -> None:
        source = UIElement(
            "source", "button", "源", (0.10, 0.20, 0.20, 0.30), 0.95, label="源"
        )
        destination = UIElement(
            "destination",
            "container",
            "目标",
            (0.70, 0.20, 0.90, 0.40),
            0.95,
            label="目标",
        )
        existing = element("status", "完成状态", states={"done": True})
        before = scene(source, destination, existing, fingerprint="before")
        resolved = UniversalActionController().resolve_one(
            SemanticAction(
                node_id="drag",
                action="drag",
                params={
                    "source_element_id": "source",
                    "source_target": "源",
                    "destination_element_id": "destination",
                    "destination_target": "目标",
                },
            ),
            before,
        )
        after = scene(source, destination, existing, fingerprint="after")

        self.assertEqual((), UniversalActionController().verify_after_action(resolved, before, after))

    def test_long_press_result_is_left_to_next_qwen_observation(self) -> None:
        item = element("item", "列表项目", role="list_item")
        before = scene(item, fingerprint="before")
        resolved = UniversalActionController().resolve_one(
            SemanticAction(
                node_id="hold",
                action="long_press",
                params={
                    "element_id": "item",
                    "target": "列表项目",
                    "duration_ms": 800,
                    "tap_point": (0.3, 0.4),
                },
            ),
            before,
        )
        unrelated = scene(element("other", "其他变化"), fingerprint="after")
        self.assertEqual((), UniversalActionController().verify_after_action(resolved, before, unrelated))

        structured_result = scene(
            replace(
                item,
                role="container",
                bounds=(0.1, 0.5, 0.9, 0.75),
            ),
            element(
                "status",
                "verification_status",
                role="text",
                states={"goal_relevant": True, "fully_visible": True},
            ),
            fingerprint="after-result",
        )
        UniversalActionController().verify_after_action(
            resolved,
            before,
            structured_result,
        )

        unrelated_goal_text = scene(
            element(
                "other",
                "help_copy",
                role="text",
                states={"goal_relevant": True, "fully_visible": True},
            ),
            fingerprint="after-unrelated-text",
        )
        self.assertEqual(
            (),
            UniversalActionController().verify_after_action(resolved, before, unrelated_goal_text),
        )

        overlay = UIScene(
            app_id="calculator",
            screen_id="home",
            summary="context menu",
            elements=(item,),
            overlays=("context_menu",),
            confidence=0.95,
            stable=True,
            fingerprint="after-overlay",
        )
        UniversalActionController().verify_after_action(resolved, before, overlay)

    def test_action_verification_rejects_stale_before_fingerprint(self) -> None:
        current = scene(element("item", "列表项目"), fingerprint="before")
        resolved = UniversalActionController().resolve_one(
            SemanticAction(
                node_id="hold",
                action="long_press",
                params={
                    "element_id": "item",
                    "target": "列表项目",
                    "tap_point": (0.3, 0.4),
                },
            ),
            current,
        )
        stale = scene(element("item", "列表项目"), fingerprint="new-before")
        changed = scene(element("menu", "菜单"), fingerprint="after")
        with self.assertRaisesRegex(UniversalActionError, "fingerprint 已过期"):
            UniversalActionController().verify_after_action(resolved, stale, changed)


if __name__ == "__main__":
    unittest.main()
