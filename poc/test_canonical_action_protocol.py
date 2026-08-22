import unittest
from dataclasses import replace

from task_semantic_ir import (
    ConstraintIntent,
    EffectIntent,
    InputFieldIntent,
    SemanticEntity,
    SemanticSubgoal,
    SurfaceRef,
    TaskSemanticIR,
)
from ui_scene import UIElement, UIScene
from verified_text_transaction import plan_from_input_states
from canonical_action_protocol import (
    CANONICAL_ACTION_PROTOCOL,
    CanonicalActionProtocolError,
    canonical_candidate_expected_result,
    compile_canonical_action_catalog,
    select_canonical_action_candidate,
)


ALL_ACTIONS = frozenset(
    {
        "tap_semantic",
        "dismiss_overlay",
        "swipe",
        "back",
        "home",
        "reveal_system_navigation",
        "input_verified_text",
        "press_enter",
        "clear_verified_text",
        "long_press",
        "drag",
        "wait_for_change",
    }
)


def element(
    element_id: str,
    *,
    label: str,
    meaning: str,
    role: str = "button",
    states: dict | None = None,
    bounds: tuple[float, float, float, float] = (0.1, 0.2, 0.4, 0.3),
) -> UIElement:
    return UIElement(
        element_id=element_id,
        role=role,
        meaning=meaning,
        bounds=bounds,
        confidence=0.98,
        label=label,
        states={
            "visible": True,
            "enabled": True,
            "fully_visible": True,
            **(states or {}),
        },
        evidence=(f"可见 {label}",),
    )


def scene(*elements: UIElement, app_id: str = "sample.app") -> UIScene:
    return UIScene(
        app_id=app_id,
        screen_id="main",
        summary="当前稳定真实画面",
        elements=tuple(elements),
        stable=True,
        confidence=0.98,
        fingerprint="f" * 64,
    )


def input_ir(*, active: str) -> TaskSemanticIR:
    target = SemanticEntity(
        entity_id="entity_input_target",
        entity_type="text",
        role="target_ui_label",
        value="当前唯一输入框",
    )
    payload = SemanticEntity(
        entity_id="entity_input_text",
        entity_type="text",
        role="input_text",
        value="longinputvalidation2026:12",
    )
    return TaskSemanticIR(
        task_id="task-input",
        device_id="device-1",
        revision=1,
        raw_goal=(
            "让当前唯一输入框内容变为 longinputvalidation2026:12，"
            "只输入最后一个字符2"
        ),
        surfaces=(SurfaceRef("surface_current", "current_surface"),),
        entities=(target, payload),
        effects=(),
        subgoals=(
            SemanticSubgoal(
                subgoal_id="navigate_to_input",
                surface_ref="surface_current",
                status="active" if active == "navigate_to_input" else "completed",
                external_impact="navigation_only",
                entity_refs=(target.entity_id,),
            ),
            SemanticSubgoal(
                subgoal_id="type_last_char",
                surface_ref="surface_current",
                status="active" if active == "type_last_char" else "pending",
                external_impact="navigation_only",
                depends_on=("navigate_to_input",),
                entity_refs=(payload.entity_id,),
            ),
        ),
        input_fields=(
            InputFieldIntent(
                field_id="field_primary",
                payload_ref=payload.entity_id,
                source_subgoal_ids=("type_last_char",),
            ),
        ),
    )


def exact_tap_ir(*, target_label: str) -> TaskSemanticIR:
    target = SemanticEntity(
        entity_id="entity_exact_tap_target",
        entity_type="ui_literal",
        role="target_ui_label",
        value=target_label,
    )
    required_tap = ConstraintIntent(
        constraint_id="constraint.exact_tap",
        kind="required_action",
        value="tap_semantic",
        source_text="点击当前画面中的目标控件",
        authoritative=True,
    )
    return TaskSemanticIR(
        task_id="task-exact-tap",
        device_id="device-1",
        revision=1,
        raw_goal=f"点击{target_label}",
        surfaces=(SurfaceRef("surface_current", "current_surface"),),
        entities=(target,),
        effects=(),
        constraints=(required_tap,),
        subgoals=(
            SemanticSubgoal(
                subgoal_id="exact_tap_semantic",
                surface_ref="surface_current",
                status="active",
                external_impact="navigation_only",
                constraint_refs=(required_tap.constraint_id,),
                # The locally built exact-action graph keeps the target label
                # as typed goal authority rather than a model-owned subgoal ref.
                entity_refs=(),
            ),
        ),
    )


def input_scene() -> UIScene:
    return scene(
        element(
            "input",
            label="longinputvalidation2026:1",
            meaning="application_text_input",
            role="input",
            states={
                "focused": True,
                "value": "longinputvalidation2026:1",
                "keyboard_layout": "numeric",
                "keyboard_input_mode": "direct_latin",
            },
        ),
        element(
            "key-2",
            label="2",
            meaning="input_exact_literal_key",
            states={
                "input_literal_key": True,
                "key_value": "2",
                "prior_input_value": "longinputvalidation2026:1",
                "expected_input_value": "longinputvalidation2026:12",
                "input_element_id": "input",
            },
            bounds=(0.15, 0.7, 0.25, 0.8),
        ),
    )


class CanonicalActionProtocolTests(unittest.TestCase):
    def test_exact_tap_admits_unique_visible_text_target(self) -> None:
        current_scene = scene(
            element(
                "target-link",
                label="返回验收模式选择",
                meaning="return_to_mode_selection",
                role="text",
                states={"goal_relevant": True},
            ),
            element(
                "other-button",
                label="等待动作",
                meaning="wait_for_action",
            ),
        )

        report = compile_canonical_action_catalog(
            current_scene,
            exact_tap_ir(target_label="返回验收模式选择"),
            {"tap_semantic"},
        )

        self.assertTrue(
            any(
                candidate.action_kind == "tap_semantic"
                and candidate.parameters.get("element_id") == "target-link"
                for candidate in report.candidates
            )
        )

    def test_exact_tap_visible_text_target_is_label_and_language_agnostic(self) -> None:
        current_scene = scene(
            element(
                "advanced-link",
                label="Advanced options",
                meaning="open_advanced_options",
                role="text",
                states={"goal_relevant": True},
            ),
            element(
                "cancel-button",
                label="Cancel",
                meaning="cancel",
            ),
        )

        report = compile_canonical_action_catalog(
            current_scene,
            exact_tap_ir(target_label="Advanced options"),
            {"tap_semantic"},
        )

        self.assertTrue(
            any(
                candidate.parameters.get("element_id") == "advanced-link"
                for candidate in report.candidates
            )
        )

    def test_exact_tap_rejects_duplicate_visible_text_targets(self) -> None:
        current_scene = scene(
            element(
                "duplicate-a",
                label="More",
                meaning="more_information",
                role="text",
                states={"goal_relevant": True},
            ),
            element(
                "duplicate-b",
                label="More",
                meaning="more_information",
                role="text",
                states={"goal_relevant": True},
                bounds=(0.5, 0.2, 0.8, 0.3),
            ),
        )

        report = compile_canonical_action_catalog(
            current_scene,
            exact_tap_ir(target_label="More"),
            {"tap_semantic"},
        )

        self.assertFalse(report.candidates)
        self.assertEqual("blocked", report.status)

    def test_plain_navigation_does_not_make_visible_text_actionable(self) -> None:
        exact_ir = exact_tap_ir(target_label="Documentation")
        semantic_ir = replace(
            exact_ir,
            subgoals=(
                SemanticSubgoal(
                    subgoal_id="navigate",
                    surface_ref="surface_current",
                    status="active",
                    external_impact="navigation_only",
                    constraint_refs=(exact_ir.constraints[0].constraint_id,),
                ),
            ),
        )
        current_scene = scene(
            element(
                "documentation-text",
                label="Documentation",
                meaning="documentation",
                role="text",
            )
        )

        report = compile_canonical_action_catalog(
            current_scene,
            semantic_ir,
            {"tap_semantic"},
        )

        self.assertFalse(report.candidates)

    def test_active_clear_step_exposes_only_owned_clear_candidate(self) -> None:
        semantic_ir = input_ir(active="type_last_char")
        semantic_ir = replace(
            semantic_ir,
            constraints=(
                ConstraintIntent(
                    constraint_id="constraint.clear",
                    kind="required_action",
                    value="clear_verified_text",
                    source_text="clear current input draft",
                    authoritative=True,
                ),
            ),
            subgoals=tuple(
                replace(item, constraint_refs=("constraint.clear",))
                if item.subgoal_id == "type_last_char"
                else item
                for item in semantic_ir.subgoals
            ),
        )
        current_scene = scene(
            element(
                "input",
                label="draft",
                meaning="application_text_input",
                role="input",
                states={
                    "focused": True,
                    "value": "draft",
                    "keyboard_layout": "qwerty",
                    "keyboard_input_mode": "direct_latin",
                    "keyboard_case_mode": "lower",
                },
            ),
            element(
                "key-x",
                label="x",
                meaning="input_exact_literal_key",
                states={
                    "input_literal_key": True,
                    "key_value": "x",
                    "prior_input_value": "draft",
                    "expected_input_value": "draftx",
                    "input_element_id": "input",
                },
            ),
        )

        report = compile_canonical_action_catalog(
            current_scene,
            semantic_ir,
            {"tap_semantic", "input_verified_text", "clear_verified_text"},
        )

        self.assertEqual(
            ["clear_verified_text"],
            [item.action_kind for item in report.candidates],
        )

    def test_active_exact_input_exposes_clear_for_nonprefix_current_value(self) -> None:
        semantic_ir = input_ir(active="type_last_char")
        current_scene = scene(
            element(
                "input",
                label="wrong draft",
                meaning="application_text_input",
                role="input",
                states={
                    "focused": True,
                    "value": "wrong draft",
                    "keyboard_layout": "qwerty",
                    "keyboard_input_mode": "direct_latin",
                    "keyboard_case_mode": "lower",
                },
            )
        )

        report = compile_canonical_action_catalog(
            current_scene,
            semantic_ir,
            {"input_verified_text", "clear_verified_text"},
        )

        self.assertEqual(
            ["clear_verified_text"],
            [item.action_kind for item in report.candidates],
        )

    def test_clear_only_goal_does_not_require_a_new_input_payload(self) -> None:
        target = SemanticEntity(
            entity_id="entity_input_target",
            entity_type="text",
            role="target_ui_label",
            value="唯一已聚焦输入框",
        )
        required_clear = ConstraintIntent(
            constraint_id="constraint.clear",
            kind="required_action",
            value="clear_verified_text",
            source_text="删除唯一已聚焦输入框中的全部现有文字",
            authoritative=True,
        )
        semantic_ir = TaskSemanticIR(
            task_id="task-clear-only",
            device_id="device-1",
            revision=1,
            raw_goal="删除唯一已聚焦输入框中的全部现有文字",
            surfaces=(SurfaceRef("surface_current", "current_surface"),),
            entities=(target,),
            effects=(),
            constraints=(required_clear,),
            subgoals=(
                SemanticSubgoal(
                    subgoal_id="clear_input",
                    surface_ref="surface_current",
                    status="active",
                    external_impact="navigation_only",
                    constraint_refs=(required_clear.constraint_id,),
                    entity_refs=(target.entity_id,),
                ),
            ),
            input_fields=(),
        )
        current_scene = scene(
            element(
                "input",
                label="existing draft",
                meaning="application_text_input",
                role="input",
                states={
                    "focused": True,
                    "value": "existing draft",
                    "keyboard_layout": "qwerty",
                    "keyboard_input_mode": "direct_latin",
                },
            )
        )

        report = compile_canonical_action_catalog(
            current_scene,
            semantic_ir,
            {"back", "clear_verified_text", "wait_for_change"},
        )

        self.assertEqual(
            ["clear_verified_text"],
            [item.action_kind for item in report.candidates],
        )

    def test_catalog_is_the_only_action_protocol(self) -> None:
        report = compile_canonical_action_catalog(
            input_scene(),
            input_ir(active="type_last_char"),
            ALL_ACTIONS,
        )
        payload = report.to_dict()
        self.assertEqual(CANONICAL_ACTION_PROTOCOL, payload["protocol_version"])
        self.assertNotIn("authoritative", payload)
        self.assertNotIn("execution_allowed", payload)
        self.assertTrue(report.candidates)

    def test_pending_input_action_is_absent_from_active_navigation_catalog(self) -> None:
        report = compile_canonical_action_catalog(
            input_scene(),
            input_ir(active="navigate_to_input"),
            {"tap_semantic", "input_verified_text", "clear_verified_text"},
        )
        self.assertFalse(
            any(
                candidate.parameters.get("element_id") == "key-2"
                for candidate in report.candidates
            )
        )

    def test_same_input_action_appears_when_its_subgoal_becomes_active(self) -> None:
        report = compile_canonical_action_catalog(
            input_scene(),
            input_ir(active="type_last_char"),
            {"tap_semantic", "input_verified_text", "clear_verified_text"},
        )
        matches = [
            candidate
            for candidate in report.candidates
            if candidate.action_kind == "tap_semantic"
            and candidate.parameters.get("element_id") == "key-2"
        ]
        self.assertEqual(1, len(matches))
        self.assertEqual(
            "longinputvalidation2026:12",
            matches[0].transition.expectations[0].value,
        )
        self.assertFalse(
            any(
                candidate.action_kind == "input_verified_text"
                for candidate in report.candidates
            ),
            "numeric literal-key steps must not advertise an impossible batch input",
        )

    def test_qwerty_direct_latin_continuation_keeps_batch_input(self) -> None:
        semantic_ir = input_ir(active="type_last_char")
        payload = next(
            item for item in semantic_ir.entities if item.role == "input_text"
        )
        semantic_ir = replace(
            semantic_ir,
            entities=tuple(
                replace(item, value="draftmore") if item is payload else item
                for item in semantic_ir.entities
            ),
        )
        direct_scene = scene(
            element(
                "input",
                label="draft",
                meaning="application_text_input",
                role="input",
                states={
                    "focused": True,
                    "value": "draft",
                    "keyboard_layout": "qwerty",
                    "keyboard_input_mode": "direct_latin",
                    "keyboard_case_mode": "lower",
                },
            )
        )
        report = compile_canonical_action_catalog(
            direct_scene,
            semantic_ir,
            {"tap_semantic", "input_verified_text"},
        )
        matches = [
            candidate
            for candidate in report.candidates
            if candidate.action_kind == "input_verified_text"
        ]
        self.assertEqual(1, len(matches))
        self.assertEqual(
            "draftmore",
            matches[0].transition.expectations[0].value,
        )

    def test_chinese_input_first_commits_preedit_then_unique_candidate(self) -> None:
        semantic_ir = input_ir(active="type_last_char")
        semantic_ir = replace(
            semantic_ir,
            entities=tuple(
                replace(item, value="你好")
                if item.role == "input_text"
                else item
                for item in semantic_ir.entities
            ),
        )
        current_scene = scene(
            element(
                "input",
                label="消息",
                meaning="application_text_input",
                role="input",
                states={
                    "focused": True,
                    "value": "",
                    "keyboard_layout": "qwerty",
                    "keyboard_input_mode": "chinese_pinyin",
                    "keyboard_case_mode": "lower",
                    "input_field_id": "field_primary",
                },
            )
        )

        report = compile_canonical_action_catalog(
            current_scene,
            semantic_ir,
            {"tap_semantic", "input_verified_text"},
        )
        candidate = next(
            item
            for item in report.candidates
            if item.action_kind == "input_verified_text"
        )

        self.assertEqual(
            {
                "element_state": {
                    "meaning": "application_text_input",
                    "states": {
                        "value": "",
                        "ime_preedit_text": "nihao",
                        "ime_exact_candidate_text": "你好",
                    },
                }
            },
            canonical_candidate_expected_result(candidate, current_scene),
        )

        candidate_scene = replace(
            current_scene,
            elements=(
                replace(
                    current_scene.elements[0],
                    states={
                        **current_scene.elements[0].states,
                        "goal_relevant": False,
                        "ime_preedit_text": "nihao",
                        "ime_exact_candidate_text": "你好",
                    },
                ),
                element(
                    "local_audited_ime_candidate_1",
                    label="你好",
                    meaning="ime_exact_candidate",
                    states={
                        "goal_relevant": True,
                        "ime_candidate": True,
                        "input_element_id": "input",
                        "prior_input_value": "",
                        "expected_input_value": "你好",
                        "pinyin": "nihao",
                    },
                    bounds=(0.1, 0.6, 0.25, 0.66),
                ),
            ),
        )
        candidate_report = compile_canonical_action_catalog(
            candidate_scene,
            semantic_ir,
            {"tap_semantic", "input_verified_text"},
        )
        candidate_tap = next(
            item
            for item in candidate_report.candidates
            if item.action_kind == "tap_semantic"
            and item.parameters.get("element_id")
            == "local_audited_ime_candidate_1"
        )
        self.assertEqual(
            {
                "element_state": {
                    "meaning": "application_text_input",
                    "states": {"value": "你好"},
                }
            },
            canonical_candidate_expected_result(candidate_tap, candidate_scene),
        )

    def test_chinese_preedit_projection_varies_without_changing_latin_steps(self) -> None:
        semantic_ir = input_ir(active="type_last_char")
        semantic_ir = replace(
            semantic_ir,
            entities=tuple(
                replace(item, value="验收")
                if item.role == "input_text"
                else item
                for item in semantic_ir.entities
            ),
        )
        current_scene = scene(
            element(
                "input",
                label="备注",
                meaning="application_text_input",
                role="input",
                states={
                    "focused": True,
                    "value": "",
                    "keyboard_layout": "qwerty",
                    "keyboard_input_mode": "chinese_pinyin",
                    "keyboard_case_mode": "lower",
                    "input_field_id": "field_primary",
                },
            )
        )

        report = compile_canonical_action_catalog(
            current_scene,
            semantic_ir,
            {"tap_semantic", "input_verified_text"},
        )
        candidate = next(
            item
            for item in report.candidates
            if item.action_kind == "input_verified_text"
        )
        states = canonical_candidate_expected_result(candidate, current_scene)[
            "element_state"
        ]["states"]
        self.assertEqual("yanshou", states["ime_preedit_text"])
        self.assertEqual("验收", states["ime_exact_candidate_text"])
        self.assertEqual("", states["value"])

        latin_ir = replace(
            semantic_ir,
            entities=tuple(
                replace(item, value="agent")
                if item.role == "input_text"
                else item
                for item in semantic_ir.entities
            ),
        )
        latin_scene = replace(
            current_scene,
            elements=(
                replace(
                    current_scene.elements[0],
                    states={
                        **current_scene.elements[0].states,
                        "keyboard_input_mode": "direct_latin",
                    },
                ),
            ),
        )
        latin_report = compile_canonical_action_catalog(
            latin_scene,
            latin_ir,
            {"tap_semantic", "input_verified_text"},
        )
        latin_candidate = next(
            item
            for item in latin_report.candidates
            if item.action_kind == "input_verified_text"
        )
        self.assertEqual(
            {
                "element_state": {
                    "meaning": "application_text_input",
                    "states": {"value": "agent"},
                }
            },
            canonical_candidate_expected_result(latin_candidate, latin_scene),
        )

    def test_long_qwerty_candidate_binds_only_next_deterministic_segment(self) -> None:
        semantic_ir = input_ir(active="type_last_char")
        payload = next(
            item for item in semantic_ir.entities if item.role == "input_text"
        )
        target = "abcdefghijklmnopqrstuvwxyzabcdefghijk"
        semantic_ir = replace(
            semantic_ir,
            entities=tuple(
                replace(item, value=target) if item is payload else item
                for item in semantic_ir.entities
            ),
        )
        current_scene = scene(
            element(
                "input",
                label="长文本",
                meaning="application_text_input",
                role="input",
                states={
                    "focused": True,
                    "value": "",
                    "keyboard_layout": "qwerty",
                    "keyboard_input_mode": "direct_latin",
                    "keyboard_case_mode": "lower",
                },
            )
        )
        report = compile_canonical_action_catalog(
            current_scene,
            semantic_ir,
            {"tap_semantic", "input_verified_text"},
        )
        candidate = next(
            item
            for item in report.candidates
            if item.action_kind == "input_verified_text"
        )
        expected = plan_from_input_states(
            target,
            current_scene.get_element("input").states,
        ).expected_value
        self.assertEqual(expected, candidate.transition.expectations[0].value)
        self.assertNotEqual(target, candidate.transition.expectations[0].value)
        self.assertEqual(20, len(candidate.transition.expectations[0].value))

    def test_multiline_enter_is_a_distinct_canonical_action(self) -> None:
        semantic_ir = input_ir(active="type_last_char")
        semantic_ir = replace(
            semantic_ir,
            entities=tuple(
                replace(item, value="first\nsecond")
                if item.role == "input_text"
                else item
                for item in semantic_ir.entities
            ),
            input_fields=(
                replace(
                    semantic_ir.input_fields[0],
                    field_label="正文",
                    multiline=True,
                ),
            ),
            constraints=(
                ConstraintIntent(
                    constraint_id="constraint.input",
                    kind="required_action",
                    value="input_verified_text",
                    source_text="typed input",
                    authoritative=True,
                ),
                ConstraintIntent(
                    constraint_id="constraint.enter",
                    kind="required_action",
                    value="press_enter",
                    source_text="multiline input",
                    authoritative=True,
                ),
            ),
            subgoals=tuple(
                replace(
                    item,
                    constraint_refs=("constraint.input", "constraint.enter"),
                )
                if item.subgoal_id == "type_last_char"
                else item
                for item in semantic_ir.subgoals
            ),
        )
        current_scene = scene(
            element(
                "input",
                label="正文",
                meaning="application_text_input",
                role="input",
                states={
                    "focused": True,
                    "value": "first",
                    "input_field_id": "field_primary",
                    "input_field_label": "正文",
                    "input_multiline": True,
                    "keyboard_layout": "qwerty",
                    "keyboard_input_mode": "direct_latin",
                },
            ),
            element(
                "enter",
                label="↵",
                meaning="input_exact_enter_key",
                states={
                    "input_enter_key": True,
                    "key_action": "newline",
                    "key_value": "\n",
                    "prior_input_value": "first",
                    "expected_input_value": "first\n",
                    "input_element_id": "input",
                    "input_field_id": "field_primary",
                },
                bounds=(0.8, 0.8, 0.94, 0.92),
            ),
        )
        report = compile_canonical_action_catalog(
            current_scene,
            semantic_ir,
            {"tap_semantic", "input_verified_text", "press_enter"},
        )
        matches = [
            candidate
            for candidate in report.candidates
            if candidate.action_kind == "press_enter"
        ]
        self.assertEqual(1, len(matches))
        self.assertEqual("enter", matches[0].parameters["element_id"])
        self.assertEqual("first\n", matches[0].transition.expectations[0].value)

        self.assertFalse(
            any(
                item.action_kind == "tap_semantic"
                and item.parameters.get("element_id") == "enter"
                for item in report.candidates
            )
        )

    def test_split_newline_subgoal_inherits_unique_adjacent_input_field(self) -> None:
        semantic_ir = input_ir(active="type_last_char")
        semantic_ir = replace(
            semantic_ir,
            entities=tuple(
                replace(item, value="first\nsecond")
                if item.role == "input_text"
                else item
                for item in semantic_ir.entities
            ),
            constraints=(
                ConstraintIntent(
                    constraint_id="constraint.enter",
                    kind="required_action",
                    value="press_enter",
                    source_text="click newline key",
                    authoritative=True,
                ),
            ),
            subgoals=(
                replace(
                    semantic_ir.subgoals[0],
                    subgoal_id="input_first_line",
                    status="completed",
                ),
                replace(
                    semantic_ir.subgoals[1],
                    subgoal_id="press_enter",
                    status="active",
                    depends_on=("input_first_line",),
                    constraint_refs=("constraint.enter",),
                    entity_refs=(),
                ),
                replace(
                    semantic_ir.subgoals[1],
                    subgoal_id="input_second_line",
                    status="pending",
                    depends_on=("press_enter",),
                    constraint_refs=(),
                    entity_refs=(),
                ),
            ),
            input_fields=(
                replace(
                    semantic_ir.input_fields[0],
                    multiline=True,
                    source_subgoal_ids=(
                        "input_first_line",
                        "input_second_line",
                    ),
                ),
            ),
        )
        current_scene = scene(
            element(
                "input",
                label="first",
                meaning="application_text_input",
                role="input",
                states={
                    "focused": True,
                    "value": "first",
                    "input_field_id": "field_primary",
                    "input_multiline": True,
                    "keyboard_layout": "qwerty",
                    "keyboard_input_mode": "direct_latin",
                },
            ),
            element(
                "enter",
                label="↵",
                meaning="input_exact_enter_key",
                states={
                    "input_enter_key": True,
                    "key_action": "newline",
                    "key_value": "\n",
                    "prior_input_value": "first",
                    "expected_input_value": "first\n",
                    "input_element_id": "input",
                    "input_field_id": "field_primary",
                },
                bounds=(0.8, 0.8, 0.94, 0.92),
            ),
        )

        report = compile_canonical_action_catalog(
            current_scene,
            semantic_ir,
            {"tap_semantic", "input_verified_text", "press_enter"},
        )

        matches = [
            candidate
            for candidate in report.candidates
            if candidate.action_kind == "press_enter"
        ]
        self.assertEqual(1, len(matches))
        self.assertEqual("enter", matches[0].parameters["element_id"])
        self.assertEqual("first\n", matches[0].transition.expectations[0].value)

        wrong_field_scene = replace(
            current_scene,
            elements=tuple(
                replace(
                    item,
                    states={**item.states, "input_field_id": "field_other"},
                )
                if item.meaning == "input_exact_enter_key"
                else item
                for item in current_scene.elements
            ),
        )
        wrong_field_report = compile_canonical_action_catalog(
            wrong_field_scene,
            semantic_ir,
            {"tap_semantic", "input_verified_text", "press_enter"},
        )
        self.assertFalse(
            any(
                candidate.action_kind == "press_enter"
                for candidate in wrong_field_report.candidates
            )
        )

    def test_batch_input_is_absent_when_typed_step_is_not_executable(self) -> None:
        cases = (
            ("already complete", "draft", "draft", "qwerty", "direct_latin", "lower", ""),
            ("wrong prefix", "draft", "other", "qwerty", "direct_latin", "lower", ""),
            ("non qwerty", "draftmore", "draft", "numeric", "direct_latin", "unknown", ""),
            ("wrong mode", "draftmore", "draft", "qwerty", "chinese_pinyin", "lower", ""),
            ("pending preedit", "draftmore", "draft", "qwerty", "direct_latin", "lower", "draft"),
            ("wrong case", "DRAFT", "", "qwerty", "direct_latin", "lower", ""),
        )
        for name, target, current, layout, mode, case_mode, preedit in cases:
            with self.subTest(name=name):
                semantic_ir = input_ir(active="type_last_char")
                semantic_ir = replace(
                    semantic_ir,
                    entities=tuple(
                        replace(item, value=target)
                        if item.role == "input_text"
                        else item
                        for item in semantic_ir.entities
                    ),
                )
                current_scene = scene(
                    element(
                        "input",
                        label=current,
                        meaning="application_text_input",
                        role="input",
                        states={
                            "focused": True,
                            "value": current,
                            "keyboard_layout": layout,
                            "keyboard_input_mode": mode,
                            "keyboard_case_mode": case_mode,
                            "ime_preedit_text": preedit,
                        },
                    )
                )
                report = compile_canonical_action_catalog(
                    current_scene,
                    semantic_ir,
                    {"tap_semantic", "input_verified_text"},
                )
                self.assertFalse(
                    any(
                        candidate.action_kind == "input_verified_text"
                        for candidate in report.candidates
                    )
                )

    def test_selection_only_returns_existing_candidate_from_same_digest(self) -> None:
        report = compile_canonical_action_catalog(
            input_scene(),
            input_ir(active="type_last_char"),
            {"tap_semantic"},
        )
        candidate = report.candidates[0]
        selected = select_canonical_action_candidate(
            report,
            report_digest=report.report_digest,
            candidate_id=candidate.candidate_id,
        )
        self.assertIs(candidate, selected)
        with self.assertRaisesRegex(CanonicalActionProtocolError, "digest"):
            select_canonical_action_candidate(
                report,
                report_digest="0" * 64,
                candidate_id=candidate.candidate_id,
            )

    def test_effect_candidate_only_exists_for_active_effect_subgoal(self) -> None:
        recipient = SemanticEntity(
            entity_id="entity_recipient",
            entity_type="text",
            role="recipient",
            value="文件传输助手",
        )
        effect = EffectIntent(
            effect_id="effect_send",
            kind="send_message",
            target_refs=(recipient.entity_id,),
            source_subgoal_ids=("send_message",),
        )
        base = TaskSemanticIR(
            task_id="task-effect",
            device_id="device-1",
            revision=1,
            raw_goal="向文件传输助手发送消息",
            surfaces=(SurfaceRef("surface_current", "current_surface"),),
            entities=(recipient,),
            effects=(effect,),
            subgoals=(
                SemanticSubgoal(
                    subgoal_id="prepare_message",
                    surface_ref="surface_current",
                    status="active",
                    external_impact="navigation_only",
                    entity_refs=(recipient.entity_id,),
                ),
                SemanticSubgoal(
                    subgoal_id="send_message",
                    surface_ref="surface_current",
                    status="pending",
                    external_impact="external_state",
                    effect_refs=(effect.effect_id,),
                    entity_refs=(recipient.entity_id,),
                ),
            ),
        )
        current_scene = scene(
            element("send", label="发送", meaning="send_message")
        )
        prepare = compile_canonical_action_catalog(
            current_scene,
            base,
            {"tap_semantic"},
        )
        self.assertFalse(any(item.effect_ref for item in prepare.candidates))
        active_effect = replace(
            base,
            subgoals=(
                replace(base.subgoals[0], status="completed"),
                replace(base.subgoals[1], status="active"),
            ),
        )
        send = compile_canonical_action_catalog(
            current_scene,
            active_effect,
            {"tap_semantic"},
        )
        self.assertEqual(["effect_send"], [item.effect_ref for item in send.candidates])

    def test_required_long_press_and_drag_are_scoped_by_active_subgoal(self) -> None:
        source = SemanticEntity("entity_source", "text", "drag_source", "起点")
        destination = SemanticEntity(
            "entity_destination", "text", "drag_destination", "终点"
        )
        long_press = ConstraintIntent(
            "constraint_long_press",
            "required_action",
            value="long_press",
            source_text="typed required action",
            authoritative=True,
        )
        drag = ConstraintIntent(
            "constraint_drag",
            "required_action",
            value="drag",
            source_text="typed required action",
            authoritative=True,
        )
        ir = TaskSemanticIR(
            task_id="task-gesture",
            device_id="device-1",
            revision=1,
            raw_goal="先长按起点，再将起点拖到终点",
            surfaces=(SurfaceRef("surface_current", "current_surface"),),
            entities=(source, destination),
            effects=(),
            constraints=(long_press, drag),
            subgoals=(
                SemanticSubgoal(
                    "press_source",
                    "surface_current",
                    "active",
                    "navigation_only",
                    constraint_refs=(long_press.constraint_id,),
                    entity_refs=(source.entity_id,),
                ),
                SemanticSubgoal(
                    "drag_source",
                    "surface_current",
                    "pending",
                    "navigation_only",
                    depends_on=("press_source",),
                    constraint_refs=(drag.constraint_id,),
                    entity_refs=(source.entity_id, destination.entity_id),
                ),
            ),
        )
        current_scene = scene(
            element("source", label="起点", meaning="drag_source_block"),
            element("destination", label="终点", meaning="drop_target_zone"),
        )
        first = compile_canonical_action_catalog(
            current_scene, ir, {"long_press", "drag"}
        )
        self.assertEqual({"long_press"}, {item.action_kind for item in first.candidates})
        second_ir = replace(
            ir,
            subgoals=(
                replace(ir.subgoals[0], status="completed"),
                replace(ir.subgoals[1], status="active"),
            ),
        )
        second = compile_canonical_action_catalog(
            current_scene, second_ir, {"long_press", "drag"}
        )
        self.assertEqual({"drag"}, {item.action_kind for item in second.candidates})

    def test_target_app_navigation_exposes_ordinary_controls_without_literal_patch(self) -> None:
        target = SurfaceRef(
            "surface_target",
            "app",
            app_id="sample.app",
            app_name="示例应用",
        )
        ir = TaskSemanticIR(
            task_id="task-navigation",
            device_id="device-1",
            revision=1,
            raw_goal="在示例应用中进入下一页",
            surfaces=(target,),
            entities=(),
            effects=(),
            subgoals=(
                SemanticSubgoal(
                    "open_next_page",
                    target.surface_id,
                    "active",
                    "navigation_only",
                ),
            ),
        )
        current = scene(
            element("next", label="继续", meaning="open_next_page"),
            app_id="sample.app",
        )
        report = compile_canonical_action_catalog(current, ir, {"tap_semantic"})
        self.assertEqual(
            ["next"],
            [item.parameters.get("element_id") for item in report.candidates],
        )

    def test_navigation_never_promotes_unbound_external_effect_control(self) -> None:
        ir = TaskSemanticIR(
            task_id="task-safe-navigation",
            device_id="device-1",
            revision=1,
            raw_goal="浏览当前页面",
            surfaces=(SurfaceRef("surface_current", "current_surface"),),
            entities=(),
            effects=(),
            subgoals=(
                SemanticSubgoal(
                    "browse_page",
                    "surface_current",
                    "active",
                    "navigation_only",
                ),
            ),
        )
        current = scene(
            element("details", label="详情", meaning="open_details"),
            element("send", label="发送", meaning="send_message"),
        )
        report = compile_canonical_action_catalog(current, ir, {"tap_semantic"})
        self.assertEqual(
            ["details"],
            [item.parameters.get("element_id") for item in report.candidates],
        )

    def test_cross_app_navigation_returns_home_before_page_specific_taps(self) -> None:
        target = SurfaceRef(
            "surface_target",
            "app",
            app_id="sample.app",
            app_name="示例应用",
        )
        ir = TaskSemanticIR(
            task_id="task-cross-app",
            device_id="device-1",
            revision=1,
            raw_goal="打开示例应用",
            surfaces=(target,),
            entities=(),
            effects=(),
            subgoals=(
                SemanticSubgoal(
                    "open_target_app",
                    target.surface_id,
                    "active",
                    "navigation_only",
                ),
            ),
        )
        unrelated = scene(
            element("unrelated", label="任意控件", meaning="open_details"),
            app_id="other.app",
        )
        report = compile_canonical_action_catalog(
            unrelated,
            ir,
            {"tap_semantic", "home"},
        )
        self.assertEqual(["home"], [item.action_kind for item in report.candidates])


if __name__ == "__main__":
    unittest.main()
