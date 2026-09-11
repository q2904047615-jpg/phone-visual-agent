from __future__ import annotations
from agent.domain.semantic_action import SemanticAction
from agent.domain.ui_scene import SystemUIFacts
from agent.domain.ui_scene import UIElement
from agent.domain.ui_scene import UIScene
from agent.domain.ui_scene import UISceneError
from agent.domain.ui_scene import UI_SCENE_PROTOCOL_VERSION
from agent.domain.universal_action_controller import UniversalActionController
from dataclasses import replace
import unittest
from test_support.ui_scene import (
    _BaseUISceneTests,
    element,
    scene,
)


class UISceneTests(_BaseUISceneTests):
    def test_system_ui_boolean_facts_round_trip(self) -> None:
        current = UIScene.from_dict(
            {
                "foreground_app_id": "browser",
                "screen_id": "page",
                "summary": "导航栏清晰可见",
                "system_ui": {
                    "immersive_or_fullscreen": False,
                    "navigation_bar_visible": True,
                },
                "elements": [],
                "stable": True,
                "confidence": 0.95,
            }
        )

        self.assertIs(False, current.system_ui.immersive_or_fullscreen)
        self.assertIs(True, current.system_ui.navigation_bar_visible)
        self.assertEqual(
            {
                "immersive_or_fullscreen": False,
                "navigation_bar_visible": True,
            },
            current.to_dict()["system_ui"],
        )

    def test_legacy_scene_without_system_ui_fails_closed_to_unknown(self) -> None:
        current = UIScene.from_dict(
            {
                "foreground_app_id": "unknown",
                "screen_id": "unknown",
                "summary": "旧观察",
                "elements": [],
                "stable": True,
                "confidence": 0.8,
            }
        )

        self.assertEqual(SystemUIFacts(), current.system_ui)
        self.assertEqual(
            {
                "immersive_or_fullscreen": "unknown",
                "navigation_bar_visible": "unknown",
            },
            current.to_dict()["system_ui"],
        )

    def test_system_ui_missing_or_ambiguous_values_normalize_to_unknown(self) -> None:
        cases = (
            (
                {"immersive_or_fullscreen": True},
                (True, "unknown"),
            ),
            (
                {
                    "immersive_or_fullscreen": "yes",
                    "navigation_bar_visible": False,
                },
                ("unknown", False),
            ),
            (
                {
                    "immersive_or_fullscreen": False,
                    "navigation_bar_visible": None,
                },
                (False, "unknown"),
            ),
        )
        for system_ui, expected in cases:
            with self.subTest(system_ui=system_ui):
                parsed = SystemUIFacts.from_dict(system_ui)
                self.assertEqual(expected[0], parsed.immersive_or_fullscreen)
                self.assertEqual(expected[1], parsed.navigation_bar_visible)

    def test_scene_element_system_ui_and_camera_ignore_harmless_metadata(self) -> None:
        current = UIScene.from_dict(
            {
                "foreground_app_id": "sample.app",
                "screen_id": "main",
                "summary": "页面显示唯一目标按钮",
                "system_ui": {
                    "immersive_or_fullscreen": False,
                    "navigation_bar_visible": True,
                    "status_bar_visible": True,
                },
                "camera_alignment": {
                    "camera_layout_orientation": "portrait",
                    "phone_content_rotation": "upright",
                    "confidence": 0.01,
                    "evidence": ["手机页面文字正向显示"],
                    "diagnostic_note": "optional model metadata",
                },
                "elements": [
                    {
                        "element_id": "target",
                        "role": "button",
                        "meaning": "open_target",
                        "label": "打开",
                        "bounds": [0.1, 0.2, 0.4, 0.3],
                        "states": {"goal_relevant": True},
                        "evidence": [],
                        "visual_note": "optional model metadata",
                    }
                ],
                "stable": True,
                "diagnostic_note": "optional model metadata",
            }
        )

        self.assertEqual("target", current.get_element("target").element_id)
        self.assertIs(True, current.system_ui.navigation_bar_visible)
        self.assertEqual("upright", current.camera_alignment.phone_content_rotation)

    def test_navigation_bar_is_not_a_scene_element(self) -> None:
        with self.assertRaisesRegex(UISceneError, "只能写入 scene.system_ui"):
            element("system-bar", "system_navigation_bar", role="container").validate()

    def test_non_string_overlay_metadata_is_ignored_instead_of_stringified(self) -> None:
        for optional_overlays in ([{
                "overlay_id": "add-new",
                "role": "button",
                "bounds": [0.4, 0.8, 0.6, 0.9],
            }], "unstructured optional overlay note"):
            with self.subTest(optional_overlays=optional_overlays):
                current = UIScene.from_dict({
                "protocol_version": UI_SCENE_PROTOCOL_VERSION,
                "foreground_app_id": "unknown",
                "screen_id": "window_manager",
                "summary": "窗口管理",
                "elements": [],
                "overlays": optional_overlays,
                "stable": True,
                "confidence": 0.95,
                "fingerprint": "frame-1",
                })

                self.assertEqual((), current.overlays)

    def test_qwen_selected_element_id_is_resolved_directly(self) -> None:
        current = scene(element("five", "digit_5"))
        action = SemanticAction(
            node_id="tap_five",
            action="tap_semantic",
            params={"target": "digit_5", "element_id": "five", "tap_point": (0.3, 0.4)},
        )
        resolved = UniversalActionController().resolve_one(action, current)
        self.assertEqual(resolved.target_element_id, "five")
        self.assertEqual(resolved.normalized_point, (0.3, 0.4))

    def test_low_scene_confidence_allows_exact_unique_goal_element_only(self) -> None:
        target = replace(
            element(
                "tab-list",
                "open_tab_list",
                states={"goal_relevant": True},
            ),
            confidence=0.01,
        )
        current = scene(target, confidence=0.01)
        action = SemanticAction(
            node_id="open-tabs",
            action="tap_semantic",
            params={"element_id": "tab-list", "target": "open_tab_list", "tap_point": (0.3, 0.4)},
        )

        resolved = UniversalActionController().resolve_one(action, current)

        self.assertEqual("tab-list", resolved.target_element_id)

    def test_invalid_bounds_and_duplicate_element_ids_still_fail(self) -> None:
        with self.assertRaisesRegex(UISceneError, "bounds"):
            UIElement.from_dict(
                {
                    "element_id": "invalid",
                    "role": "button",
                    "meaning": "invalid_bounds",
                    "bounds": [0.1, 0.2, 1.1, 0.3],
                }
            )

        first = element("duplicate", "first")
        with self.assertRaisesRegex(UISceneError, "元素ID重复"):
            scene(first, replace(first, meaning="second")).validate()

    def test_unrelated_overlapping_element_does_not_veto_unique_goal(self) -> None:
        target = element(
            "tab-list",
            "open_tab_list",
            states={"goal_relevant": True},
        )
        conflicting = element("other", "close_tab")

        self.assertEqual(target, scene(target, conflicting, confidence=0.6).get_element("tab-list"))

    def test_qwen_selected_element_id_does_not_trigger_local_semantic_reselection(self) -> None:
        current = scene(element("one", "search"), element("two", "search"))
        action = SemanticAction(
            node_id="tap_search",
            action="tap_semantic",
            params={"target": "search", "element_id": "two", "tap_point": (0.3, 0.4)},
        )
        resolved = UniversalActionController().resolve_one(action, current)
        self.assertEqual("two", resolved.target_element_id)

    def test_natural_language_like_button_does_not_require_confirmation(self) -> None:
        current = scene(element("heart", "点赞按钮", role="button"))
        action = SemanticAction(
            node_id="like",
            action="tap_semantic",
            params={"target": "点赞按钮", "element_id": "heart", "tap_point": (0.3, 0.4)},
        )
        resolved = UniversalActionController().resolve_one(action, current)
        self.assertEqual("tap_semantic", resolved.kind)
        self.assertEqual("heart", resolved.target_element_id)

    def test_labeled_element_wording_drift_is_left_to_next_qwen_observation(self) -> None:
        controller = UniversalActionController()
        before_element = UIElement(
            element_id="home",
            role="tab",
            meaning="当前激活标签页",
            label="主页",
            bounds=(0.2, 0.3, 0.4, 0.5),
            confidence=0.95,
        )
        after_element = UIElement(
            element_id="home",
            role="tab",
            meaning="当前活跃标签页",
            label="主页",
            bounds=(0.2, 0.3, 0.4, 0.5),
            confidence=0.95,
        )
        before = scene(before_element, fingerprint="before")
        after = scene(after_element, fingerprint="after")
        resolved = controller.resolve_one(
            SemanticAction(
                node_id="switch-tab",
                action="tap_semantic",
                params={"element_id": "home", "target": "当前激活标签页", "tap_point": (0.3, 0.4)},
            ),
            before,
        )

        self.assertEqual((), controller.verify_after_action(resolved, before, after))

    def test_unlabeled_element_change_is_left_to_next_qwen_observation(self) -> None:
        controller = UniversalActionController()
        before_icon = UIElement(
            element_id="icon",
            role="icon",
            meaning="open_menu",
            label="",
            bounds=(0.2, 0.3, 0.4, 0.5),
            confidence=0.95,
        )
        after_icon = UIElement(
            element_id="icon",
            role="icon",
            meaning="close_menu",
            label="",
            bounds=(0.2, 0.3, 0.4, 0.5),
            confidence=0.95,
        )

        before = scene(before_icon, fingerprint="before")
        resolved = controller.resolve_one(
            SemanticAction(node_id="open", action="tap_semantic",
                params={"element_id": "icon", "tap_point": before_icon.center}),
            before,
        )
        self.assertEqual(
            (),
            controller.verify_after_action(resolved, before, scene(after_icon, fingerprint="after")),
        )


if __name__ == "__main__":
    unittest.main()
