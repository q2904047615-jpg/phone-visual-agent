from __future__ import annotations
from agent.domain.semantic_action import SemanticAction
from agent.domain.ui_scene import UIElement
from agent.domain.ui_scene import UIScene
from agent.domain.ui_scene import UISceneError
from agent.domain.universal_action_controller import UniversalActionController
from agent.domain.universal_action_controller import UniversalActionError
from dataclasses import replace
import unittest
from test_support.ui_scene import (
    _BaseUISceneTests,
    element,
    scene,
)


class UISceneTests(_BaseUISceneTests):
    def test_recent_tasks_clear_all_uses_current_frame_button_tap(self) -> None:
        target = element("clear-all", "clear_all_recent_tasks", role="button",
            bounds=(0.45, 0.82, 0.57, 0.92))
        resolved = UniversalActionController().resolve_one(
            SemanticAction(node_id="clear-all-recents", action="tap_semantic", params={
                "element_id": "clear-all", "target": "clear_all_recent_tasks",
                "role": "button", "label": "×", "states": {}, "tap_point": target.center,
            }),
            scene(target, app_id="system", screen_id="system_recent_tasks"),
        )

        self.assertEqual("tap_semantic", resolved.kind)
        self.assertEqual("clear-all", resolved.target_element_id)
        self.assertEqual(target.center, resolved.normalized_point)

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

    def test_verified_input_requires_focused_input_and_preserves_exact_text(self) -> None:
        current = scene(
            element(
                "search-field",
                "搜索输入框",
                role="input",
                states={"input_field_id": "field_primary", **({
                    "focused": True,
                    "value": "",
                    "keyboard_layout": "qwerty",
                    "keyboard_input_mode": "direct_latin",
                    "goal_relevant": True,
                })},
            )
        )
        action = SemanticAction(
            node_id="type-query",
            action="input_verified_text",
            params={"text_transport": "adb_keyboard", "input_field_id": 'field_primary', "prior_input_value": '', "expected_input_value": "agent", "input_fragment": ("agent")[len(''):],
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
                states={"input_field_id": "field_primary", **({"focused": True, "value": "agent"})},
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
                states={"input_field_id": "field_primary", **({
                    "focused": True,
                    "value": "",
                    "keyboard_layout": "qwerty",
                    "keyboard_input_mode": "direct_latin",
                    "goal_relevant": True,
                })},
            ),
            fingerprint="before",
        )
        resolved = UniversalActionController().resolve_one(
            SemanticAction(
                node_id="type-query",
                action="input_verified_text",
                params={"text_transport": "adb_keyboard", "input_field_id": 'field_primary', "prior_input_value": '', "expected_input_value": "agent", "input_fragment": ("agent")[len(''):],
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
                            states={"input_field_id": "field_primary", **(states)},
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
            element("field", "draft_input", role="input", states={"input_field_id": "field_primary", **(states)}),
            fingerprint="before",
        )
        resolved = UniversalActionController().resolve_one(
            SemanticAction(
                node_id="clear-draft",
                action="clear_verified_text",
                params={"text_transport": "adb_keyboard", "input_field_id": 'field_primary', "prior_input_value": 'lxs,', "expected_input_value": '',
                    "element_id": "field",
                    "target": "draft_input",
                },
            ),
            before,
        )

        self.assertEqual("clear_verified_text", resolved.kind)
        self.assertNotIn("delete_count", resolved.to_dict())
        after = scene(
            element(
                "field-after",
                "draft_input",
                role="input",
                states={"input_field_id": "field_primary", **({**states, "value": ""})},
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
                        states={"input_field_id": "field_primary", **({**states, "value": "lxs"})},
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
                states={"input_field_id": "field_primary", **(states)},
            ),
            app_id="微信",
            screen_id="chat_window",
            fingerprint="before",
        )
        resolved = UniversalActionController().resolve_one(
            SemanticAction(
                node_id="clear-draft",
                action="clear_verified_text",
                params={"text_transport": "adb_keyboard", "input_field_id": 'input_field_1', "prior_input_value": '', "expected_input_value": '',
                    "element_id": "field",
                    "target": "application_text_input",
                },
            ),
            before,
        )

        self.assertEqual("", resolved.expected_input_value)
        after = scene(
            element(
                "field-after",
                "application_text_input",
                role="input",
                states={"input_field_id": "field_primary", **({
                    **states,
                    "value": "",
                    "ime_preedit_text": "",
                    "keyboard_input_mode": "direct_latin",
                })},
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

    def test_verified_clear_uses_exact_current_field_value_not_optional_app_or_screen_wording(self) -> None:
        states = {
            "input_field_id": "field_primary",
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
            states={"input_field_id": "field_primary", **(states)},
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
                params={"text_transport": "adb_keyboard", "input_field_id": 'field_primary', "prior_input_value": 'lxs,', "expected_input_value": '',
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
            params={"text_transport": "adb_keyboard", "input_field_id": 'field_primary', "prior_input_value": '', "expected_input_value": '',
                "element_id": "field-a",
                "target": "draft_input",
            },
        )
        with self.assertRaisesRegex(UniversalActionError, "至少一项非空"):
            UniversalActionController().resolve_one(
                action,
                scene(element("field-a", "draft_input", role="input", states={"input_field_id": "field_primary", **(base)})),
            )
        nonempty = {**base, "value": "wrong"}
        action = replace(action, params={**action.params, "prior_input_value": "wrong"})
        resolved = UniversalActionController().resolve_one(
            action,
            scene(
                element("field-a", "draft_input", role="input", states={"input_field_id": "field_primary", **(nonempty)}),
                element("field-b", "other_input", role="input", states={"input_field_id": "other_field", **(nonempty)}),
            ),
        )
        self.assertEqual("field-a", resolved.target_element_id)

    def test_focus_tap_accepts_unique_post_action_input_semantic_alias(self) -> None:
        before = scene(
            element(
                "rough-input",
                "target_text_input",
                role="input",
                states={"input_field_id": "field_primary", **({"goal_relevant": True, "value": ""})},
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
                    "tap_point": (0.3, 0.4),
                },
            ),
            before,
        )
        after = scene(
            element(
                "audited-input",
                "application_text_input",
                role="input",
                states={"input_field_id": "field_primary", **({"goal_relevant": True, "value": "", "focused": True})},
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
                    "tap_point": (0.3, 0.4),
                },
            ),
            before,
        )
        after = scene(
            element(
                "unrelated-input",
                "application_text_input",
                role="input",
                states={"input_field_id": "field_primary", **({"focused": True})},
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
                states={"input_field_id": "field_primary", **({
                    "focused": True,
                    "value": "alreadythere",
                    "keyboard_layout": "qwerty",
                    "keyboard_input_mode": "direct_latin",
                    "goal_relevant": True,
                })},
            ),
            fingerprint="before",
        )

        with self.assertRaisesRegex(UniversalActionError, "prior/fragment/expected"):
            UniversalActionController().resolve_one(
                SemanticAction(
                    node_id="type-query",
                    action="input_verified_text",
                    params={"text_transport": "adb_keyboard", "input_field_id": 'field_primary', "prior_input_value": 'already', "expected_input_value": "agent", "input_fragment": ("agent")[len('already'):],
                        "element_id": "search-field",
                        "target": "搜索输入框",
                        "text": "agent",
                    },
                ),
                current,
            )

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
        longer = UIElement.from_dict(
                {
                    "element_id": "too-long-field",
                    "role": "input",
                    "meaning": "message_field",
                    "bounds": [100, 100, 900, 200],
                    "states": {"value": "x" * 4001},
                },
                coordinate_scale=1000,
            )
        self.assertEqual(4001, len(longer.states["value"]))

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

    def test_adb_input_binds_typed_field_and_exact_unicode_receipt(self) -> None:
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
                "text": expected, "text_transport": "adb_keyboard", "input_field_id": "body_field",
                "prior_input_value": "first", "input_fragment": "\nsecond🙂",
                "expected_input_value": expected,
            },
        )
        controller = UniversalActionController()
        resolved = controller.resolve_one(action, before)
        self.assertEqual("adb_keyboard", resolved.text_transport)
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

    def test_adb_keyboard_input_requires_focus_even_without_visible_soft_keyboard(self) -> None:
        before_input = element(
            "field", "application_text_input", role="input",
            states={
                "fully_visible": True, "value": "", "input_field_id": "body_field",
                "input_multiline": False, "soft_keyboard_visible": False,
                "ime_preedit_text": "",
            },
        )
        before = scene(before_input, app_id="sample.app", screen_id="editor", fingerprint="before-adb")
        action = SemanticAction(
            node_id="adb-input", action="input_verified_text",
            params={
                "element_id": "field", "target": "application_text_input", "role": "input",
                "text": "ADB测试？你好", "text_transport": "adb_keyboard",
                "input_field_id": "body_field", "prior_input_value": "",
                "input_fragment": "ADB测试？你好", "expected_input_value": "ADB测试？你好",
            },
        )

        with self.assertRaisesRegex(UniversalActionError, "已聚焦"):
            UniversalActionController().resolve_one(action, before)

    def test_adb_keyboard_clear_requires_focus_before_broadcast(self) -> None:
        before_input = element(
            "field", "application_text_input", role="input",
            states={
                "fully_visible": True, "value": "aaazjie？你好", "input_field_id": "body_field",
                "input_multiline": False, "soft_keyboard_visible": False,
                "ime_preedit_text": "",
            },
        )
        before = scene(before_input, app_id="sample.app", screen_id="editor", fingerprint="before-adb-clear")
        action = SemanticAction(
            node_id="adb-clear", action="clear_verified_text",
            params={
                "element_id": "field", "target": "application_text_input", "role": "input",
                "text_transport": "adb_keyboard", "input_field_id": "body_field",
                "prior_input_value": "aaazjie？你好", "expected_input_value": "",
            },
        )

        with self.assertRaisesRegex(UniversalActionError, "已聚焦"):
            UniversalActionController().resolve_one(action, before)

    def test_adb_clear_binds_typed_field_and_rejects_stale_preedit(self) -> None:
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
                "text_transport": "adb_keyboard", "input_field_id": "body_field",
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
            params={"text_transport": "adb_keyboard", "input_field_id": 'field_primary', "prior_input_value": '', "expected_input_value": "agent", "input_fragment": ("agent")[len(''):], "element_id": "field", "target": "查询框", "text": "agent"},
        )

        with self.assertRaisesRegex(UniversalActionError, "已聚焦"):
            UniversalActionController().resolve_one(action, current)

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
            states={"input_field_id": "field_primary", **(before_states)},
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
                params={"text_transport": "adb_keyboard", "input_field_id": 'body_field', "prior_input_value": '', "expected_input_value": "agent", "input_fragment": "agent",
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

    def test_exact_literal_key_is_bound_to_input_prefix_and_explicit_receipt(self) -> None:
        field = element(
            "field", "application_text_input", role="input",
            states={"input_field_id": "field_primary", **({
                "focused": True, "value": "draft", "keyboard_layout": "qwerty",
                "keyboard_input_mode": "direct_latin", "goal_relevant": False,
            })},
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
                "tap_point": key.center,
            },
        )
        resolved = UniversalActionController().resolve_one(action, scene(field, key))
        self.assertEqual(key.center, resolved.normalized_point)

    def test_verified_input_uses_qwen_selected_current_element(self) -> None:
        states = {
            "focused": True,
            "value": "",
            "keyboard_layout": "qwerty",
            "keyboard_input_mode": "direct_latin",
            "goal_relevant": True,
        }
        current = scene(
            element("field-a", "查询框", role="input", states={"input_field_id": "field_primary", **(states)}),
            element("field-b", "备用查询框", role="input", states={"input_field_id": "other_field", **(states)}),
        )

        resolved = UniversalActionController().resolve_one(
            SemanticAction(
                node_id="type",
                action="input_verified_text",
                params={"text_transport": "adb_keyboard", "input_field_id": 'field_primary', "prior_input_value": '', "expected_input_value": "agent", "input_fragment": ("agent")[len(''):], "element_id": "field-a", "target": "查询框", "text": "agent"},
            ),
            current,
        )
        self.assertEqual("field-a", resolved.target_element_id)
        self.assertEqual("agent", resolved.expected_input_value)


if __name__ == "__main__":
    unittest.main()
