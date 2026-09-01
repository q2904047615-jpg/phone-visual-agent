import unittest
from copy import deepcopy
from dataclasses import replace

from agent.domain.semantic_action import SemanticAction
from agent.domain.ui_scene import (
    UI_SCENE_PROTOCOL_VERSION,
    SystemUIFacts,
    UIElement,
    UIScene,
    UISceneError,
)
from agent.domain.universal_action_controller import (
    LOCAL_POINT_GROUNDING_SOURCE,
    LocalPointGrounding,
    ResolvedSemanticAction,
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


def scene(
    *elements: UIElement,
    app_id="calculator",
    screen_id="home",
    fingerprint="a",
    confidence=0.95,
):
    return UIScene(
        app_id=app_id,
        screen_id=screen_id,
        summary="test",
        elements=tuple(elements),
        confidence=confidence,
        stable=True,
        fingerprint=fingerprint,
    )


class UISceneTests(unittest.TestCase):
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

    def test_qwen_selected_element_id_is_resolved_directly(self) -> None:
        current = scene(element("five", "digit_5"))
        action = SemanticAction(
            node_id="tap_five",
            action="tap_semantic",
            params={"target": "digit_5", "element_id": "five"},
        )
        resolved = UniversalActionController().resolve_one(action, current)
        self.assertEqual(resolved.target_element_id, "five")
        self.assertEqual(resolved.normalized_point, (0.30000000000000004, 0.4))

    def test_stable_local_text_grounding_refines_only_selected_target(self) -> None:
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
            },
        )
        grounding = LocalPointGrounding(
            source=LOCAL_POINT_GROUNDING_SOURCE,
            scene_fingerprint=current.fingerprint,
            element_id=target.element_id,
            label=target.label,
            model_bounds=target.bounds,
            proposed_point=target.center,
            grounded_bounds=(0.23, 0.245, 0.46, 0.267),
            grounded_point=(0.345, 0.256),
            matched_frames=2,
            inspected_frames=2,
        )

        resolved = UniversalActionController().resolve_one(
            action,
            current,
            local_point_grounding=grounding,
        )

        self.assertEqual((0.345, 0.256), resolved.normalized_point)
        self.assertEqual(target.center, resolved.proposed_normalized_point)
        self.assertEqual(
            [0.345, 0.256],
            resolved.to_dict()["point_grounding"]["grounded_point"],
        )

    def test_local_text_grounding_rejects_stale_scene_binding(self) -> None:
        target = replace(
            element("target-row", "open_target", role="list_item"),
            label="目标条目",
            bounds=(0.07, 0.29, 0.93, 0.39),
        )
        current = scene(target, fingerprint="fresh-frame")
        grounding = LocalPointGrounding(
            source=LOCAL_POINT_GROUNDING_SOURCE,
            scene_fingerprint="stale-frame",
            element_id=target.element_id,
            label=target.label,
            model_bounds=target.bounds,
            proposed_point=target.center,
            grounded_bounds=(0.23, 0.245, 0.46, 0.267),
            grounded_point=(0.345, 0.256),
            matched_frames=2,
            inspected_frames=2,
        )

        with self.assertRaisesRegex(UniversalActionError, "不属于当前"):
            UniversalActionController().resolve_one(
                SemanticAction(
                    node_id="open-target",
                    action="tap_semantic",
                    params={"element_id": target.element_id},
                ),
                current,
                local_point_grounding=grounding,
            )

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
            params={"element_id": "tab-list", "target": "open_tab_list"},
        )

        resolved = UniversalActionController().resolve_one(action, current)

        self.assertEqual("tab-list", resolved.target_element_id)

    def test_low_scene_confidence_does_not_veto_screen_wide_action(self) -> None:
        current = scene(
            element("tab-list", "open_tab_list", states={"goal_relevant": True}),
            confidence=0.01,
        )
        action = SemanticAction(
            node_id="scroll",
            action="swipe",
            params={"direction": "up"},
        )

        resolved = UniversalActionController().resolve_one(action, current)

        self.assertEqual("swipe", resolved.kind)

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

    def test_unrelated_overlapping_element_does_not_veto_unique_goal(self) -> None:
        target = element(
            "tab-list",
            "open_tab_list",
            states={"goal_relevant": True},
        )
        conflicting = element("other", "close_tab")

        self.assertEqual(target, scene(target, conflicting, confidence=0.6).get_element("tab-list"))

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
                action="swipe",
                params={"direction": "up"},
            ),
            current,
        )

        self.assertEqual("swipe", resolved.kind)

    def test_container_is_valid_current_frame_point_target(self) -> None:
        content = element("content", "video_content", role="container")
        current = scene(content)
        current.validate()
        action = SemanticAction(
            node_id="tap_content",
            action="tap_semantic",
            params={"target": "video_content", "role": "container", "element_id": "content"},
        )
        resolved = UniversalActionController().resolve_one(action, current)
        self.assertEqual("content", resolved.target_element_id)
        self.assertEqual(content.center, resolved.normalized_point)

    def test_qwen_selected_element_id_does_not_trigger_local_semantic_reselection(self) -> None:
        current = scene(element("one", "search"), element("two", "search"))
        action = SemanticAction(
            node_id="tap_search",
            action="tap_semantic",
            params={"target": "search", "element_id": "two"},
        )
        resolved = UniversalActionController().resolve_one(action, current)
        self.assertEqual("two", resolved.target_element_id)

    def test_ordinary_send_does_not_require_controller_confirmation(self) -> None:
        current = scene(element("send", "send"))
        action = SemanticAction(
            node_id="send",
            action="tap_semantic",
            params={"target": "send", "element_id": "send"},
        )
        resolved = UniversalActionController().resolve_one(action, current)
        self.assertEqual("tap_semantic", resolved.kind)
        self.assertEqual("send", resolved.target_element_id)

    def test_natural_language_like_button_does_not_require_confirmation(self) -> None:
        current = scene(element("heart", "点赞按钮", role="button"))
        action = SemanticAction(
            node_id="like",
            action="tap_semantic",
            params={"target": "点赞按钮", "element_id": "heart"},
        )
        resolved = UniversalActionController().resolve_one(action, current)
        self.assertEqual("tap_semantic", resolved.kind)
        self.assertEqual("heart", resolved.target_element_id)

    def test_selected_element_id_does_not_recheck_copied_semantics_or_display_states(self) -> None:
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
                    "role": "dialog",
                    "label": "stale copied label",
                    "states": "stale copied states",
                },
            ),
            scene(selected),
        )

        self.assertEqual("current-target", resolved.target_element_id)
        self.assertEqual(selected.center, resolved.normalized_point)

    def test_typed_input_still_requires_actual_input_role(self) -> None:
        not_an_input = element("selected-button", "current_control", role="button")
        with self.assertRaisesRegex(UniversalActionError, "角色必须为 input"):
            UniversalActionController().resolve_one(
                SemanticAction(
                    node_id="type",
                    action="input_verified_text",
                    params={
                        "element_id": "selected-button",
                        "role": "input",
                        "text": "hello",
                    },
                ),
                scene(not_an_input),
            )

    def test_ordinary_account_mutation_target_does_not_require_controller_confirmation(self) -> None:
        current = scene(element("heart", "reaction_button", role="button"))
        action = SemanticAction(
            node_id="like",
            action="tap_semantic",
            params={
                "target": "reaction_button",
                "element_id": "heart",
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
            params={"target": "digit_5", "element_id": "five"},
        )
        resolved = controller.resolve_one(action, before)
        unchanged = scene(element("five", "digit_5"), fingerprint="before")
        self.assertEqual((), controller.verify_after_action(resolved, before, unchanged))
        camera_noise_only = scene(element("five", "digit_5"), fingerprint="noise")
        self.assertEqual((), controller.verify_after_action(resolved, before, camera_noise_only))
        changed = scene(element("result", "result_page"), fingerprint="after")
        self.assertEqual((), controller.verify_after_action(resolved, before, changed))

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
                params={"element_id": "home", "target": "当前激活标签页"},
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
            SemanticAction(node_id="open", action="tap_semantic", params={"element_id": "icon"}),
            before,
        )
        self.assertEqual(
            (),
            controller.verify_after_action(resolved, before, scene(after_icon, fingerprint="after")),
        )

    def test_verified_input_requires_focused_input_and_preserves_exact_text(self) -> None:
        current = scene(
            element(
                "search-field",
                "搜索输入框",
                role="input",
                states={
                    "focused": True,
                    "value": "",
                    "keyboard_layout": "qwerty",
                    "keyboard_input_mode": "direct_latin",
                    "goal_relevant": True,
                },
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
                "states": {
                    "focused": True,
                    "value": "",
                    "keyboard_layout": "qwerty",
                    "keyboard_input_mode": "direct_latin",
                    "goal_relevant": True,
                },
                "text": "agent",
            },
        )

        resolved = UniversalActionController().resolve_one(action, current)

        self.assertEqual("input_verified_text", resolved.kind)
        self.assertEqual("agent", resolved.text)
        self.assertEqual("search-field", resolved.target_element_id)

        exact_after = scene(
            element(
                "search-field-after",
                "搜索输入框",
                role="input",
                states={"focused": True, "value": "agent"},
            ),
            fingerprint="after",
        )
        UniversalActionController().verify_after_action(resolved, current, exact_after)

    def test_verified_input_rejects_wrong_or_missing_exact_value(self) -> None:
        current = scene(
            element(
                "search-field",
                "搜索输入框",
                role="input",
                states={
                    "focused": True,
                    "value": "",
                    "keyboard_layout": "qwerty",
                    "keyboard_input_mode": "direct_latin",
                    "goal_relevant": True,
                },
            ),
            fingerprint="before",
        )
        resolved = UniversalActionController().resolve_one(
            SemanticAction(
                node_id="type-query",
                action="input_verified_text",
                params={
                    "element_id": "search-field",
                    "target": "搜索输入框",
                    "text": "agent",
                },
            ),
            current,
        )
        cases = (
            ({"focused": True, "value": "agent.com"}, "文字不匹配"),
            ({"focused": True}, "states.value"),
        )
        for states, message in cases:
            with self.subTest(states=states), self.assertRaisesRegex(
                UniversalActionError,
                message,
            ):
                UniversalActionController().verify_after_action(
                    resolved,
                    current,
                    scene(
                        element(
                            "search-field-after",
                            "搜索输入框",
                            role="input",
                            states=states,
                        ),
                        fingerprint="after",
                    ),
                )

    def test_verified_clear_binds_exact_nonempty_value_and_verifies_empty(self) -> None:
        states = {
            "focused": True,
            "value": "lxs,",
            "keyboard_layout": "qwerty",
            "keyboard_input_mode": "direct_latin",
            "goal_relevant": True,
        }
        before = scene(
            element("field", "draft_input", role="input", states=states),
            fingerprint="before",
        )
        resolved = UniversalActionController().resolve_one(
            SemanticAction(
                node_id="clear-draft",
                action="clear_verified_text",
                params={
                    "element_id": "field",
                    "target": "draft_input",
                },
            ),
            before,
        )

        self.assertEqual("clear_verified_text", resolved.kind)
        self.assertEqual(4, resolved.delete_count)
        after = scene(
            element(
                "field-after",
                "draft_input",
                role="input",
                states={**states, "value": ""},
            ),
            fingerprint="after",
        )
        UniversalActionController().verify_after_action(resolved, before, after)

        with self.assertRaisesRegex(UniversalActionError, "状态证据|文字不匹配"):
            UniversalActionController().verify_after_action(
                resolved,
                before,
                scene(
                    element(
                        "field-after",
                        "draft_input",
                        role="input",
                        states={**states, "value": "lxs"},
                    ),
                    fingerprint="wrong-after",
                ),
            )

    def test_verified_clear_uses_bound_ime_preedit_when_app_value_is_empty(self) -> None:
        states = {
            "focused": True,
            "value": "",
            "ime_preedit_text": "longinp",
            "input_field_id": "input_field_1",
            "input_multiline": False,
            "keyboard_layout": "qwerty",
            "keyboard_input_mode": "chinese_pinyin",
            "goal_relevant": True,
        }
        before = scene(
            element(
                "field",
                "application_text_input",
                role="input",
                states=states,
            ),
            app_id="微信",
            screen_id="chat_window",
            fingerprint="before",
        )
        resolved = UniversalActionController().resolve_one(
            SemanticAction(
                node_id="clear-draft",
                action="clear_verified_text",
                params={
                    "element_id": "field",
                    "target": "application_text_input",
                },
            ),
            before,
        )

        self.assertEqual(7, resolved.delete_count)
        after = scene(
            element(
                "field-after",
                "application_text_input",
                role="input",
                states={
                    **states,
                    "value": "",
                    "ime_preedit_text": "",
                    "keyboard_input_mode": "direct_latin",
                },
            ),
            app_id="wechat",
            screen_id="chat_conversation",
            fingerprint="after",
        )
        UniversalActionController().verify_after_action(resolved, before, after)

        with self.assertRaisesRegex(UniversalActionError, "预编辑"):
            UniversalActionController().verify_after_action(
                resolved,
                before,
                replace(
                    after,
                    elements=(
                        replace(
                            after.elements[0],
                            states={**states, "value": ""},
                        ),
                    ),
                ),
            )

    def test_verified_clear_counts_audited_extra_visual_row_units(self) -> None:
        states = {
            "focused": True,
            "value": "first",
            "clear_extra_delete_units": 1,
            "keyboard_layout": "qwerty",
            "keyboard_input_mode": "direct_latin",
            "goal_relevant": True,
        }
        before = scene(
            element(
                "field",
                "application_text_input",
                role="input",
                states=states,
            ),
            fingerprint="before-extra-row",
        )
        resolved = UniversalActionController().resolve_one(
            SemanticAction(
                node_id="clear-extra-row",
                action="clear_verified_text",
                params={
                    "element_id": "field",
                    "target": "application_text_input",
                },
            ),
            before,
        )
        self.assertEqual(6, resolved.delete_count)

        for invalid in (-1, 31, True, "1"):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                UniversalActionError,
                "额外视觉行",
            ):
                bad = replace(
                    before,
                    elements=(
                        replace(
                            before.elements[0],
                            states={
                                **states,
                                "clear_extra_delete_units": invalid,
                            },
                        ),
                    ),
                )
                UniversalActionController().resolve_one(
                    SemanticAction(
                        node_id="clear-extra-row-invalid",
                        action="clear_verified_text",
                        params={
                            "element_id": "field",
                            "target": "application_text_input",
                        },
                    ),
                    bad,
                )

    def test_verified_clear_uses_exact_current_field_value_not_optional_app_or_screen_wording(self) -> None:
        states = {
            "focused": True,
            "value": "lxs,",
            "keyboard_layout": "qwerty",
            "keyboard_input_mode": "direct_latin",
            "goal_relevant": True,
        }
        before_input = element(
            "field",
            "application_text_input",
            role="input",
            states=states,
        )
        before = UIScene(
            app_id="com.example.real",
            screen_id="chat_window",
            summary="chat with draft",
            elements=(before_input,),
            stable=True,
            confidence=0.98,
            fingerprint="before",
        )
        resolved = UniversalActionController().resolve_one(
            SemanticAction(
                node_id="clear-draft",
                action="clear_verified_text",
                params={
                    "element_id": "field",
                    "target": "application_text_input",
                },
            ),
            before,
        )
        after_input = replace(
            before_input,
            element_id="field-after",
            bounds=before_input.bounds,
            states={**states, "value": ""},
        )
        after = UIScene(
            app_id="current_foreground",
            screen_id="chat_conversation",
            summary="chat with empty draft",
            elements=(after_input,),
            stable=True,
            confidence=0.98,
            fingerprint="after",
        )

        UniversalActionController().verify_after_action(resolved, before, after)

        different_app = replace(after, app_id="com.example.other")
        UniversalActionController().verify_after_action(resolved, before, different_app)

        different_family = replace(
            after,
            app_id="com.example.real",
            screen_id="settings_page",
        )
        UniversalActionController().verify_after_action(resolved, before, different_family)

    def test_verified_clear_rejects_empty_but_uses_selected_current_element(self) -> None:
        base = {
            "focused": True,
            "value": "",
            "keyboard_layout": "qwerty",
            "goal_relevant": True,
        }
        action = SemanticAction(
            node_id="clear-draft",
            action="clear_verified_text",
            params={
                "element_id": "field-a",
                "target": "draft_input",
            },
        )
        with self.assertRaisesRegex(UniversalActionError, "至少一项非空"):
            UniversalActionController().resolve_one(
                action,
                scene(element("field-a", "draft_input", role="input", states=base)),
            )
        nonempty = {**base, "value": "wrong"}
        resolved = UniversalActionController().resolve_one(
            action,
            scene(
                element("field-a", "draft_input", role="input", states=nonempty),
                element("field-b", "other_input", role="input", states=nonempty),
            ),
        )
        self.assertEqual("field-a", resolved.target_element_id)

    def test_focus_tap_accepts_unique_post_action_input_semantic_alias(self) -> None:
        before = scene(
            element(
                "rough-input",
                "target_text_input",
                role="input",
                states={"goal_relevant": True, "value": ""},
            ),
            app_id="browser",
            screen_id="local-page",
            fingerprint="before",
        )
        resolved = UniversalActionController().resolve_one(
            SemanticAction(
                node_id="focus-input",
                action="tap_semantic",
                params={
                    "element_id": "rough-input",
                    "target": "target_text_input",
                    "role": "input",
                    "states": {"goal_relevant": True, "value": ""},
                },
            ),
            before,
        )
        after = scene(
            element(
                "audited-input",
                "application_text_input",
                role="input",
                states={"goal_relevant": True, "value": "", "focused": True},
            ),
            app_id="browser",
            screen_id="local-page",
            fingerprint="after",
        )

        UniversalActionController().verify_after_action(resolved, before, after)

    def test_focus_only_input_surface_cannot_carry_typed_input_authority(self) -> None:
        states = {
            "goal_relevant": True,
            "fully_visible": True,
            "focus_only_input_surface": True,
        }
        focus_only = element(
            "coarse-input",
            "message_input_field",
            role="input",
            states=states,
        )
        focus_only.validate()

        for forbidden_key, forbidden_value in (
            ("value", "draft"),
            ("input_field_id", "message_field"),
            ("focused", True),
            ("primary_input_geometry_verified", True),
        ):
            with self.subTest(forbidden_key=forbidden_key):
                with self.assertRaisesRegex(
                    UISceneError,
                    "不得携带正文、typed字段身份、键盘状态或本地审计权威",
                ):
                    replace(
                        focus_only,
                        states={**states, forbidden_key: forbidden_value},
                    ).validate()

    def test_non_input_tap_result_is_not_reinterpreted_as_input_semantics(self) -> None:
        before = scene(
            element("rough-button", "target_control", role="button"),
            fingerprint="before",
        )
        resolved = UniversalActionController().resolve_one(
            SemanticAction(
                node_id="tap-button",
                action="tap_semantic",
                params={
                    "element_id": "rough-button",
                    "target": "target_control",
                },
            ),
            before,
        )
        after = scene(
            element(
                "unrelated-input",
                "application_text_input",
                role="input",
                states={"focused": True},
            ),
            fingerprint="after",
        )

        self.assertEqual((), UniversalActionController().verify_after_action(resolved, before, after))

    def test_verified_input_rejects_nonempty_value_before_resolution(self) -> None:
        current = scene(
            element(
                "search-field",
                "搜索输入框",
                role="input",
                states={
                    "focused": True,
                    "value": "alreadythere",
                    "keyboard_layout": "qwerty",
                    "keyboard_input_mode": "direct_latin",
                    "goal_relevant": True,
                },
            ),
            fingerprint="before",
        )

        with self.assertRaisesRegex(UniversalActionError, "精确前缀"):
            UniversalActionController().resolve_one(
                SemanticAction(
                    node_id="type-query",
                    action="input_verified_text",
                    params={
                        "element_id": "search-field",
                        "target": "搜索输入框",
                        "text": "agent",
                    },
                ),
                current,
            )

    def test_verified_input_rejects_chinese_pinyin_qwerty_before_resolution(self) -> None:
        current = scene(
            element(
                "search-field",
                "搜索输入框",
                role="input",
                states={
                    "focused": True,
                    "value": "",
                    "keyboard_layout": "qwerty",
                    "keyboard_input_mode": "chinese_pinyin",
                    "goal_relevant": True,
                },
            ),
            fingerprint="before",
        )
        action = SemanticAction(
            node_id="type-query",
            action="input_verified_text",
            params={
                "element_id": "search-field",
                "target": "搜索输入框",
                "text": "agent",
            },
        )

        with self.assertRaisesRegex(UniversalActionError, "输入模式"):
            UniversalActionController().resolve_one(action, current)

    def test_input_state_value_and_keyboard_layout_are_typed(self) -> None:
        parsed = UIElement.from_dict(
            {
                "element_id": "field",
                "role": "input",
                "meaning": "search_field",
                "bounds": [100, 100, 900, 200],
                "confidence": 0.95,
                "states": {"value": "", "keyboard_layout": "qwerty"},
            },
            coordinate_scale=1000,
        )
        self.assertEqual("", parsed.states["value"])
        max_length = UIElement.from_dict(
            {
                "element_id": "long-field",
                "role": "input",
                "meaning": "message_field",
                "bounds": [100, 100, 900, 200],
                "states": {"value": "x" * 4000},
            },
            coordinate_scale=1000,
        )
        self.assertEqual(4000, len(max_length.states["value"]))
        with self.assertRaisesRegex(UISceneError, "4000"):
            UIElement.from_dict(
                {
                    "element_id": "too-long-field",
                    "role": "input",
                    "meaning": "message_field",
                    "bounds": [100, 100, 900, 200],
                    "states": {"value": "x" * 4001},
                },
                coordinate_scale=1000,
            )
        with self.assertRaisesRegex(UISceneError, "keyboard_layout"):
            UIElement.from_dict(
                {
                    "element_id": "field",
                    "role": "input",
                    "meaning": "search_field",
                    "bounds": [100, 100, 900, 200],
                    "confidence": 0.95,
                    "states": {"keyboard_layout": "t9"},
                },
                coordinate_scale=1000,
            )
        with self.assertRaisesRegex(
            UISceneError,
            r"keyboard-key.*role=keyboard_key.*keyboard_layout",
        ):
            UIElement.from_dict(
                {
                    "element_id": "keyboard-key",
                    "role": "keyboard_key",
                    "meaning": "letter_key",
                    "bounds": [100, 700, 200, 800],
                    "confidence": 0.95,
                    "states": {"keyboard_layout": "qwerty"},
                },
                coordinate_scale=1000,
            )

        clear = UIElement.from_dict(
            {
                "element_id": "clear",
                "role": "icon",
                "meaning": "clear_local_text",
                "label": "×",
                "bounds": [760, 110, 820, 170],
                "confidence": 0.95,
                "states": {"local_text_clear": True},
            },
            coordinate_scale=1000,
        )
        self.assertTrue(clear.states["local_text_clear"])
        with self.assertRaisesRegex(UISceneError, "local_text_clear"):
            UIElement.from_dict(
                {
                    "element_id": "bad-clear",
                    "role": "keyboard_key",
                    "meaning": "delete",
                    "label": "×",
                    "bounds": [760, 700, 820, 760],
                    "confidence": 0.95,
                    "states": {"local_text_clear": True},
                },
                coordinate_scale=1000,
            )

        with self.assertRaisesRegex(UISceneError, "真实可见"):
            UIElement.from_dict(
                {
                    "element_id": "hallucinated-clear",
                    "role": "icon",
                    "meaning": "clear_local_text",
                    "label": "",
                    "bounds": [760, 110, 820, 170],
                    "confidence": 0.95,
                    "states": {"local_text_clear": True},
                    "evidence": ["模型自由描述为圆形叉号"],
                },
                coordinate_scale=1000,
            )

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

    def test_companion_input_binds_typed_field_and_exact_unicode_receipt(self) -> None:
        before_input = element(
            "field", "application_text_input", role="input",
            states={
                "focused": True, "fully_visible": True, "value": "first",
                "input_field_id": "body_field", "input_field_label": "正文",
                "input_multiline": True, "ime_preedit_text": "",
            },
        )
        before = scene(before_input, app_id="sample.app", screen_id="editor", fingerprint="before-companion")
        expected = "first\nsecond🙂"
        action = SemanticAction(
            node_id="companion-input", action="input_verified_text",
            params={
                "element_id": "field", "target": "application_text_input", "role": "input",
                "text": expected, "text_transport": "companion_ime", "input_field_id": "body_field",
                "prior_input_value": "first", "input_fragment": "\nsecond🙂",
                "expected_input_value": expected,
            },
        )
        controller = UniversalActionController()
        resolved = controller.resolve_one(action, before)
        self.assertEqual("unicode_commit", resolved.input_method)
        after_input = replace(
            before_input,
            element_id="field-after",
            label="",
            states={**before_input.states, "value": expected},
        )
        after = scene(after_input, app_id="sample.app", screen_id="editor-result", fingerprint="after-companion")
        self.assertEqual(
            (f"控制器确认 typed 输入值：{expected!r}",),
            controller.verify_after_action(resolved, before, after),
        )

    def test_companion_clear_binds_typed_field_and_rejects_stale_preedit(self) -> None:
        before_input = element(
            "field", "application_text_input", role="input",
            states={
                "focused": True, "fully_visible": True, "value": "待清除🙂",
                "input_field_id": "body_field", "input_field_label": "正文",
                "input_multiline": True, "ime_preedit_text": "",
            },
        )
        before = scene(before_input, fingerprint="before-companion-clear")
        action = SemanticAction(
            node_id="companion-clear", action="clear_verified_text",
            params={
                "element_id": "field", "target": "application_text_input", "role": "input",
                "text_transport": "companion_ime", "input_field_id": "body_field",
                "prior_input_value": "待清除🙂", "expected_input_value": "",
            },
        )
        controller = UniversalActionController()
        resolved = controller.resolve_one(action, before)
        self.assertIsNone(resolved.normalized_point)
        after_input = replace(before_input, label="", states={**before_input.states, "value": ""})
        after = scene(after_input, fingerprint="after-companion-clear")
        controller.verify_after_action(resolved, before, after)
        stale = replace(
            after,
            fingerprint="stale-companion-clear",
            elements=(replace(after_input, states={**after_input.states, "ime_preedit_text": "stale"}),),
        )
        with self.assertRaisesRegex(UniversalActionError, "预编辑"):
            controller.verify_after_action(resolved, before, stale)

    def test_verified_input_rejects_unfocused_field(self) -> None:
        current = scene(element("field", "查询框", role="input"))
        action = SemanticAction(
            node_id="type",
            action="input_verified_text",
            params={"element_id": "field", "target": "查询框", "text": "agent"},
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

    def test_double_tap_resolves_one_current_frame_target(self) -> None:
        current = scene(element("preview", "预览图", role="list_item"))
        action = SemanticAction(
            node_id="double",
            action="double_tap",
            params={
                "element_id": "preview",
                "target": "预览图",
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
                params={"element_id": "preview", "target": "预览图"},
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

    def test_verified_input_segments_mixed_text_without_broadening_key_profile(self) -> None:
        current = scene(
            element(
                "field",
                "查询框",
                role="input",
                states={
                    "focused": True,
                    "value": "",
                    "keyboard_layout": "qwerty",
                    "keyboard_input_mode": "direct_latin",
                    "goal_relevant": True,
                },
            )
        )
        for text, error in (
            ("Agent", "大小写状态"),
            ("中文", "输入模式"),
        ):
            with self.subTest(text=text), self.assertRaisesRegex(UniversalActionError, error):
                UniversalActionController().resolve_one(
                    SemanticAction(
                        node_id="type",
                        action="input_verified_text",
                        params={"element_id": "field", "target": "查询框", "text": text},
                    ),
                    current,
                )

        digit_pending = UniversalActionController().resolve_one(
            SemanticAction(
                node_id="type-digit",
                action="input_verified_text",
                params={"element_id": "field", "target": "查询框", "text": "agent1"},
            ),
            current,
        )
        self.assertEqual("agent", digit_pending.input_fragment)
        long_text = UniversalActionController().resolve_one(
            SemanticAction(
                node_id="type-long",
                action="input_verified_text",
                params={"element_id": "field", "target": "查询框", "text": "a" * 31},
            ),
            current,
        )
        self.assertEqual("a" * 20, long_text.input_fragment)

    def test_verified_input_uses_unique_typed_field_identity_after_layout_shift(self) -> None:
        before_states = {
            "focused": True,
            "value": "",
            "keyboard_layout": "qwerty",
            "keyboard_input_mode": "direct_latin",
            "goal_relevant": True,
            "fully_visible": True,
            "input_field_id": "body_field",
            "input_field_label": "正文",
            "input_multiline": False,
        }
        before_input = element(
            "local_audited_input_1",
            "application_text_input",
            role="input",
            states=before_states,
        )
        before = scene(
            before_input,
            app_id="browser",
            screen_id="input_form",
            fingerprint="before-typed-shift",
        )
        resolved = UniversalActionController().resolve_one(
            SemanticAction(
                node_id="type-body",
                action="input_verified_text",
                params={
                    "element_id": before_input.element_id,
                    "target": "application_text_input",
                    "text": "agent",
                },
            ),
            before,
        )
        after_input = replace(
            before_input,
            element_id="fresh_local_input_id",
            label="agent",
            bounds=(0.55, 0.65, 0.85, 0.78),
            states={**before_states, "value": "agent"},
        )
        after = scene(
            after_input,
            app_id="browser",
            screen_id="input_form_result",
            fingerprint="after-typed-shift",
        )

        UniversalActionController().verify_after_action(resolved, before, after)

        wrong_field = replace(
            after_input,
            element_id=before_input.element_id,
            states={**after_input.states, "input_field_id": "other_field"},
        )
        with self.assertRaisesRegex(
            UniversalActionError,
            "无法唯一绑定原目标输入框",
        ):
            UniversalActionController().verify_after_action(
                resolved,
                before,
                replace(after, elements=(wrong_field,)),
            )

        duplicate_field = replace(
            after_input,
            element_id="duplicate_field",
            bounds=(0.1, 0.15, 0.4, 0.25),
        )
        with self.assertRaisesRegex(
            UniversalActionError,
            "无法唯一绑定原目标输入框",
        ):
            UniversalActionController().verify_after_action(
                resolved,
                before,
                replace(after, elements=(after_input, duplicate_field)),
            )

        conflicting_label = replace(
            after_input,
            states={**after_input.states, "input_field_label": "标题"},
        )
        UniversalActionController().verify_after_action(
            resolved,
            before,
            replace(after, elements=(conflicting_label,)),
        )

    def test_next_field_tap_validates_current_binding_then_defers_focus_result(self) -> None:
        next_key = element(
            "next", "input_next_field_key",
            states={
                "goal_relevant": True, "fully_visible": True,
                "input_next_field_key": True, "key_action": "next",
                "source_input_field_id": "subject_field",
                "target_input_field_id": "body_field",
                "target_input_field_label": "正文",
            },
        )
        before = scene(next_key, fingerprint="before-next-field")
        action = SemanticAction(
            node_id="focus-body", action="tap_semantic",
            params={"element_id": "next"},
        )
        controller = UniversalActionController()
        resolved = controller.resolve_one(action, before)
        body = element(
            "body", "application_text_input", role="input",
            states={"focused": True, "input_field_id": "body_field",
                    "input_field_label": "正文", "value": ""},
        )
        after = scene(body, fingerprint="after-next-field")
        self.assertEqual((), controller.verify_after_action(resolved, before, after))

        for name, elements in (
            ("wrong field", (replace(body, states={**body.states, "input_field_id": "other"}),)),
            ("not focused", (replace(body, states={**body.states, "focused": False}),)),
            ("wrong label", (replace(body, states={**body.states, "input_field_label": "标题"}),)),
            ("duplicate", (body, replace(body, element_id="body-duplicate"))),
        ):
            with self.subTest(name=name):
                self.assertEqual(
                    (), controller.verify_after_action(resolved, before, replace(after, elements=elements))
                )

        malformed = replace(
            next_key,
            states={**next_key.states, "target_input_field_id": "subject_field"},
        )
        with self.assertRaisesRegex(UniversalActionError, "唯一 typed 字段切换"):
            controller.resolve_one(action, scene(malformed, fingerprint="bad-next-field"))

    def test_generic_keyboard_geometry_is_strictly_backspace_only(self) -> None:
        parsed = UIElement.from_dict(
            {
                "element_id": "field",
                "role": "input",
                "meaning": "message_input",
                "bounds": [100, 100, 900, 200],
                "confidence": 0.95,
                "states": {
                    "focused": True,
                    "keyboard_layout": "numeric",
                    "keyboard_geometry": {
                        "type": "generic",
                        "anchors": {"backspace": [900, 720]},
                        "source": "input_structure_audit",
                    },
                },
            },
            coordinate_scale=1000,
        )
        self.assertEqual(
            {"backspace": [900, 720]},
            parsed.states["keyboard_geometry"]["anchors"],
        )

        invalid_states = (
            {
                "focused": True,
                "keyboard_layout": "numeric",
                "keyboard_geometry": {
                    "type": "generic",
                    "anchors": {},
                    "source": "input_structure_audit",
                },
            },
            {
                "focused": True,
                "keyboard_layout": "symbol",
                "keyboard_geometry": {
                    "type": "generic",
                    "anchors": {"backspace": [900, 720], "a": [100, 600]},
                    "source": "input_structure_audit",
                },
            },
            {
                "focused": False,
                "keyboard_layout": "numeric",
                "keyboard_geometry": {
                    "type": "generic",
                    "anchors": {"backspace": [900, 720]},
                    "source": "input_structure_audit",
                },
            },
            {
                "focused": True,
                "keyboard_layout": "numeric",
                "keyboard_geometry": {
                    "type": "generic",
                    "anchors": {"backspace": [900, 720]},
                    "source": "vision_model",
                },
            },
        )
        for states in invalid_states:
            with self.subTest(states=states), self.assertRaises(UISceneError):
                UIElement.from_dict(
                    {
                        "element_id": "field",
                        "role": "input",
                        "meaning": "message_input",
                        "bounds": [100, 100, 900, 200],
                        "confidence": 0.95,
                        "states": states,
                    },
                    coordinate_scale=1000,
                )

    def test_verified_uppercase_segment_requires_visible_upper_case_mode(self) -> None:
        field = element(
            "field", "消息", role="input",
            states={
                "focused": True, "value": "", "keyboard_layout": "qwerty",
                "keyboard_input_mode": "direct_latin",
                "keyboard_case_mode": "upper", "goal_relevant": True,
            },
        )
        action = SemanticAction(
            node_id="type-upper", action="input_verified_text",
            params={
                "element_id": "field", "target": "消息", "text": "Meeting",
            },
        )
        resolved = UniversalActionController().resolve_one(action, scene(field))
        self.assertEqual("M", resolved.input_fragment)
        self.assertEqual("direct_latin", resolved.input_method)

    def test_exact_literal_key_is_bound_to_input_prefix_and_explicit_receipt(self) -> None:
        field = element(
            "field", "application_text_input", role="input",
            states={
                "focused": True, "value": "draft", "keyboard_layout": "qwerty",
                "keyboard_input_mode": "direct_latin", "goal_relevant": False,
            },
        )
        key = element(
            "literal-key", "input_exact_literal_key", role="button",
            states={
                "goal_relevant": True, "fully_visible": True,
                "input_literal_key": True, "key_value": " ",
                "prior_input_value": "draft", "expected_input_value": "draft ",
                "input_element_id": "field",
            },
        )
        action = SemanticAction(
            node_id="space", action="tap_semantic",
            params={
                "element_id": "literal-key", "target": "input_exact_literal_key",
                "role": "button", "label": "input_exact_literal_key",
            },
        )
        resolved = UniversalActionController().resolve_one(action, scene(field, key))
        self.assertEqual(key.center, resolved.normalized_point)

    def test_keyboard_input_mode_switch_uses_explicit_bound_input_state(self) -> None:
        field = element(
            "field",
            "application_text_input",
            role="input",
            states={
                "focused": True,
                "value": "draft",
                "keyboard_layout": "qwerty",
                "keyboard_input_mode": "direct_latin",
                "goal_relevant": False,
            },
        )
        mode_switch = element(
            "mode-switch",
            "switch_keyboard_input_mode",
            states={
                "goal_relevant": True,
                "fully_visible": True,
                "keyboard_input_mode_switch": True,
                "current_mode": "direct_latin",
                "target_mode": "chinese_pinyin",
                "prior_input_value": "draft",
                "input_element_id": "field",
            },
        )
        action = SemanticAction(
            node_id="switch-input-mode",
            action="tap_semantic",
            params={
                "element_id": "mode-switch",
                "target": "switch_keyboard_input_mode",
                "role": "button",
                "label": "switch_keyboard_input_mode",
            },
        )

        resolved = UniversalActionController().resolve_one(
            action,
            scene(field, mode_switch),
        )

        self.assertEqual(mode_switch.center, resolved.normalized_point)
        self.assertEqual("draft", resolved.prior_input_value)
        self.assertEqual("draft", resolved.expected_input_value)
        self.assertEqual({"keyboard_input_mode": "chinese_pinyin"}, resolved.expected_input_state)
        self.assertEqual("field", resolved.input_element_id)
        mismatched_field = replace(
            field,
            states={**field.states, "keyboard_input_mode": "chinese_pinyin"},
        )
        with self.assertRaisesRegex(UniversalActionError, "输入模式切换方向"):
            UniversalActionController().resolve_one(
                action,
                scene(mismatched_field, mode_switch),
            )

        reverse_switch = replace(
            mode_switch,
            states={
                **mode_switch.states,
                "current_mode": "chinese_pinyin",
                "target_mode": "direct_latin",
            },
        )
        UniversalActionController().resolve_one(
            action,
            scene(mismatched_field, reverse_switch),
        )
        before = scene(field, mode_switch, fingerprint="before")
        resolved = UniversalActionController().resolve_one(action, before)
        switched_field = replace(
            field,
            states={**field.states, "keyboard_input_mode": "chinese_pinyin"},
        )
        UniversalActionController().verify_after_action(
            resolved,
            before,
            scene(switched_field, fingerprint="after"),
        )
        with self.assertRaisesRegex(
            UniversalActionError,
            "keyboard_input_mode 不匹配",
        ):
            UniversalActionController().verify_after_action(
                resolved,
                before,
                scene(field, fingerprint="after"),
            )

    def test_explicit_keyboard_state_receipts_share_one_bound_field_rule(
        self,
    ) -> None:
        for state_key, prior, value in (
            ("keyboard_layout", "qwerty", "numeric"),
            ("keyboard_input_mode", "chinese_pinyin", "direct_latin"),
            ("keyboard_case_mode", "lower", "upper"),
        ):
            with self.subTest(state_key=state_key):
                action = ResolvedSemanticAction(
                    node_id="switch",
                    kind="tap_semantic",
                    target_element_id="switch",
                    input_element_id="field",
                    prior_input_value="draft",
                    expected_input_value="draft",
                    expected_input_state={state_key: value},
                    before_fingerprint="before",
                )
                before_field = element(
                    "field", "application_text_input", role="input",
                    states={"focused": True, "value": "draft", state_key: prior},
                )
                after_field = element(
                    "field",
                    "application_text_input",
                    role="input",
                    states={"focused": True, "value": "draft", state_key: value},
                )
                UniversalActionController().verify_after_action(
                    action,
                    scene(
                        before_field,
                        fingerprint="before",
                    ),
                    scene(after_field, fingerprint="after"),
                )

    def test_long_press_requires_safe_bounds_but_not_visual_postcondition(self) -> None:
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
                params={"element_id": "item", "target": "列表项目", "duration_ms": 800},
            ),
            ordinary,
        )
        self.assertEqual("long_press", resolved.kind)
        with self.assertRaisesRegex(UniversalActionError, "画面边缘"):
            UniversalActionController().resolve_one(
                SemanticAction(
                    node_id="hold", action="long_press",
                    params={"element_id": "edge", "target": "边缘控件", "duration_ms": 800},
                ),
                scene(edge),
            )

    def test_drag_rejects_too_short_too_long_and_edge_paths(self) -> None:
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
            with self.subTest(message=message), self.assertRaisesRegex(
                UniversalActionError,
                message,
            ):
                UniversalActionController().resolve_one(
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

    def test_verified_input_uses_qwen_selected_current_element(self) -> None:
        states = {
            "focused": True,
            "value": "",
            "keyboard_layout": "qwerty",
            "keyboard_input_mode": "direct_latin",
            "goal_relevant": True,
        }
        current = scene(
            element("field-a", "查询框", role="input", states=states),
            element("field-b", "备用查询框", role="input", states=states),
        )

        resolved = UniversalActionController().resolve_one(
            SemanticAction(
                node_id="type",
                action="input_verified_text",
                params={"element_id": "field-a", "target": "查询框", "text": "agent"},
            ),
            current,
        )
        self.assertEqual("field-a", resolved.target_element_id)
        self.assertEqual("agent", resolved.expected_input_value)

    def test_action_verification_rejects_stale_before_fingerprint(self) -> None:
        current = scene(element("item", "列表项目"), fingerprint="before")
        resolved = UniversalActionController().resolve_one(
            SemanticAction(
                node_id="hold",
                action="long_press",
                params={
                    "element_id": "item",
                    "target": "列表项目",
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
