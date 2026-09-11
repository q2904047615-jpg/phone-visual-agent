from __future__ import annotations
from agent.domain.confirmation_authority import ConfirmationAuthority
from agent.application.action_adapter import GenericActionAdapterError
from agent.domain.generic_goal import GenericIntentDraft
from PIL import Image
from PIL import ImageFilter
from pathlib import Path
from agent.domain.universal_action_controller import ResolvedSemanticAction
from agent.domain.semantic_action import SemanticAction
from agent.domain.ui_scene import UIElement
from agent.domain.ui_scene import UIScene
from agent.domain.universal_action_controller import UniversalActionController
from agent.domain.universal_action_controller import UniversalActionError
from agent.domain.validation import canonical_digest
import json
from agent.infrastructure.observation_images import local_frame_fingerprint
from agent.infrastructure.observation_images import measure_frame_sharpness
from unittest.mock import patch
from dataclasses import replace
import unittest
from test_support.generic_action_adapter import (
    FakeAdbKeyboardTextTransport,
    FakeRobot,
    FakeSceneObserver,
    GenericSingleActionAdapter,
    SequenceCapture,
    TEST_QWERTY_GEOMETRY,
    _BaseGenericActionAdapterTests,
    aligned_camera_facts,
    consumed_authority,
    goal,
    navigation_goal,
    scene,
    textured_phone_frame,
)


class GenericActionAdapterTests(_BaseGenericActionAdapterTests):
    def test_retired_confirmation_input_recovery_helpers_stay_absent(self) -> None:
        source = (
            Path(__file__).resolve().parent
            / "agent"
            / "infrastructure"
            / "generic_action_adapter.py"
        ).read_text(encoding="utf-8")
        for retired_name in (
            "_local_input_auxiliary_recovery_target",
            "_confirmation_allows_omitted_local_input_auxiliary",
            "_recover_omitted_verified_input_scene",
            "_recover_conflicting_clear_input_scene",
        ):
            self.assertNotIn(retired_name, source)

    def test_strict_input_execute_uses_zero_duplicate_pre_action_model_audits(self):
        gray = Image.new("RGB", (540, 960), "gray")
        planned = self._strict_primary_input_scene(
            local_frame_fingerprint(gray)
        )
        field = planned.elements[0]
        after_field = replace(
            field,
            label="agent",
            states={**field.states, "value": "agent"},
            evidence=("正文输入框逐字显示 agent",),
        )
        after = replace(
            planned,
            summary="正文输入框逐字显示 agent",
            elements=(after_field,),
            fingerprint="after-agent",
        )
        observer = FakeSceneObserver([after])
        robot = FakeRobot()
        anchors = {
            key: list(value)
            for key, value in TEST_QWERTY_GEOMETRY["anchors"].items()
        }
        adapter = GenericSingleActionAdapter(
            capture=SequenceCapture(["gray"] * 4 + ["white"] * 4),
            observer=observer,
            robot=robot,
            frame_interval=0,
            post_action_settle=0,
        )
        text_action_1 = SemanticAction(
                node_id="type-body",
                action="input_verified_text",
                params={"text_transport": "adb_keyboard", "input_field_id": (dict(field.states)).get("input_field_id", "field_primary"), "prior_input_value": (dict(field.states)).get("value", ""), "expected_input_value": "agent", "input_fragment": ("agent")[len((dict(field.states)).get("value", "")):],


                    "element_id": field.element_id,
                    "target": field.meaning,
                    "role": field.role,
                    "label": field.label,
                    "states": dict(field.states),
                    "text": "agent",

                },
            )
        result = adapter.execute(action_authority=consumed_authority(text_action_1, planned),
            requested_action=text_action_1,
            planned_scene=planned,
            planned_frames=(gray, gray.copy(), gray.copy(), gray.copy()),
            goal=GenericIntentDraft(
                understood=True,
                app_id="generic_app",
                app_name="当前应用",
                objective="正文逐字等于 agent 且不发送",
                entities={"input_text": "agent"},
                success_criteria={"input": "agent"},
            ),
            confirmed=True,
        )

        self.assertEqual([("input", "agent")], robot.actions)
        self.assertEqual(1, observer.calls)
        self.assertFalse(result.primary_input_confirmation_reused)
        self.assertEqual("executed", result.action_outcome)
        self.assertFalse(hasattr(observer, "last_orientation_audit_diagnostics"))

    def test_placeholder_loss_recovery_executes_only_authorized_suffix(self):
        gray = Image.new("RGB", (540, 960), "gray")

        def typed_scene(fingerprint, value, *, bounds=(0.15, 0.53, 0.70, 0.59)):
            source = self._literal_input_scene(
                fingerprint,
                value=value,
                include_key=False,
            )
            field = source.elements[0]
            return replace(
                source,
                elements=(
                    replace(
                        field,
                        label="first",
                        bounds=bounds,
                        states={
                            **field.states,
                            "input_field_id": "input_field_1",
                            "input_multiline": True,
                            "keyboard_geometry": TEST_QWERTY_GEOMETRY,
                        },
                    ),
                ),
            )

        planned = typed_scene("planned", "first\n")
        fresh_missing = replace(planned, fingerprint="fresh", elements=())
        planned_audited = typed_scene("planned-audited", "first\n")
        fresh_audited = typed_scene(
            "fresh-audited",
            "first\n",
            bounds=(0.14, 0.52, 0.71, 0.60),
        )
        after = typed_scene("after", "first\nsecond")
        observer = FakeSceneObserver([after])
        robot = FakeRobot()
        adapter = GenericSingleActionAdapter(
            capture=SequenceCapture(["gray"] * 4 + ["white"] * 4),
            observer=observer,
            robot=robot,
            frame_interval=0,
            post_action_settle=0,
        )
        field = planned.elements[0]
        expected = "first\nsecond"

        text_action_2 = SemanticAction(
                node_id="append-second",
                action="input_verified_text",
                params={"text_transport": "adb_keyboard", "input_field_id": (dict(field.states)).get("input_field_id", "field_primary"), "prior_input_value": (dict(field.states)).get("value", ""), "expected_input_value": expected, "input_fragment": (expected)[len((dict(field.states)).get("value", "")):],


                    "element_id": field.element_id,
                    "target": field.meaning,
                    "role": field.role,
                    "label": field.label,
                    "states": dict(field.states),
                    "text": expected,

                },
            )
        result = adapter.execute(action_authority=consumed_authority(text_action_2, planned),
            requested_action=text_action_2,
            planned_scene=planned,
            planned_frames=tuple(gray.copy() for _ in range(4)),
            goal=goal(),
            confirmed=True,
        )

        self.assertEqual([("input", "second")], robot.actions)
        self.assertEqual(1, result.physical_actions)
        self.assertEqual("executed", result.action_outcome)
        self.assertEqual("input_field_1", result.before_scene.elements[0].states["input_field_id"])

    def test_typed_field_identity_bridges_optional_visual_wording_and_mode_drift(self):
        states = {
            "focused": True,
            "fully_visible": True,
            "value": "first\n",
            "input_field_id": "input_field_1",
            "input_multiline": True,
            "keyboard_layout": "qwerty",
            "keyboard_input_mode": "direct_latin",
            "goal_relevant": True,
        }
        before = UIScene(
            app_id="unknown",
            screen_id="通用动作具机验收页",
            summary="同一多行输入框",
            elements=(
                UIElement(
                    element_id="before-field",
                    role="input",
                    meaning="application_text_input",
                    label="first",
                    bounds=(0.14, 0.52, 0.71, 0.60),
                    confidence=1.0,
                    states={"input_field_id": "field_primary", **(states)},
                ),
            ),
            stable=True,
            confidence=1.0,
            fingerprint="before",
            camera_alignment=aligned_camera_facts(),
        )
        after = replace(
            before,
            screen_id="通用动作真机验收页",
            fingerprint="after",
            elements=(
                replace(
                    before.elements[0],
                    element_id="after-field",
                    bounds=(0.13, 0.51, 0.72, 0.61),
                    label="first\nsecond",
                    states={**states, "value": "first\nsecond"},
                ),
            ),
        )
        resolved = ResolvedSemanticAction(
            node_id="append-second",
            kind="input_verified_text",
            text="first\nsecond",
            input_fragment="second",
            prior_input_value="first\n",
            expected_input_value="first\nsecond",
            input_field_id="input_field_1",
            target_element_id="before-field",
            before_fingerprint=before.fingerprint,
        )

        UniversalActionController().verify_after_action(resolved, before, after)

        different_field = replace(
            after,
            elements=(
                replace(
                    after.elements[0],
                    states={
                        **after.elements[0].states,
                        "input_field_id": "input_field_2",
                    },
                ),
            ),
        )
        changed_mode = replace(
            after,
            elements=(
                replace(
                    after.elements[0],
                    states={
                        **after.elements[0].states,
                        "keyboard_input_mode": "chinese_pinyin",
                    },
                ),
            ),
        )
        for unsafe_before, unsafe_after in ((before, different_field),):
            with self.subTest(
                before_app_id=unsafe_before.app_id,
                app_id=unsafe_after.app_id,
                states=unsafe_after.elements[0].states,
            ):
                with self.assertRaises(UniversalActionError):
                    UniversalActionController().verify_after_action(
                        resolved,
                        unsafe_before,
                        unsafe_after,
                    )

        UniversalActionController().verify_after_action(resolved, before, changed_mode)

    def test_direct_input_and_clear_use_current_exact_value_without_persistent_lineage(self):
        def input_scene(fingerprint, value, *, goal_relevant=True):
            return UIScene(
                app_id="generic_app",
                screen_id="editor",
                summary="唯一聚焦输入框",
                elements=(
                    UIElement(
                        element_id="field",
                        role="input",
                        meaning="application_text_input",
                        label=value,
                        bounds=(0.13, 0.54, 0.69, 0.61),
                        confidence=1.0,
                        states={
                            "focused": True,
                            "fully_visible": True,
                            "value": value,
                            "keyboard_layout": "qwerty",
                            "keyboard_input_mode": "direct_latin",
                            "keyboard_geometry": TEST_QWERTY_GEOMETRY,
                            "input_field_id": "field",
                            "goal_relevant": goal_relevant,
                        },
                        evidence=(f"应用输入框当前文字：{value}",),
                    ),
                ),
                stable=True,
                confidence=1.0,
                fingerprint=fingerprint,
                camera_alignment=aligned_camera_facts(),
            )

        before = input_scene("before", "")
        after = input_scene("after", "longinput")
        robot = FakeRobot()
        input_action = SemanticAction(
            node_id="type-segment",
            action="input_verified_text",
            params={"text_transport": "adb_keyboard", "input_field_id": (dict(before.elements[0].states)).get("input_field_id", "field_primary"), "prior_input_value": (dict(before.elements[0].states)).get("value", ""), "expected_input_value": "longinput", "input_fragment": ("longinput")[len((dict(before.elements[0].states)).get("value", "")):],
                "element_id": "field",
                "target": "application_text_input",
                "role": "input",
                "label": "",
                "states": dict(before.elements[0].states),
                "text": "longinput",
            },
        )
        prepared_4_adapter = self._adapter(FakeSceneObserver([before, after]), robot)
        prepared_4_goal = goal()
        prepared_4_scene, prepared_4_frames, _, _ = prepared_4_adapter.capture_scene(
            prepared_4_goal, prefix="test_orchestrator_before", evidence_dir=None)
        text_action_3 = input_action
        result = prepared_4_adapter.execute(action_authority=consumed_authority(text_action_3, prepared_4_scene),
            requested_action=text_action_3,
            planned_scene=prepared_4_scene, planned_frames=prepared_4_frames,
            goal=prepared_4_goal,
            confirmed=True,
        )
        self.assertEqual("executed", result.action_outcome)

        clear_after = input_scene("cleared", "", goal_relevant=False)
        prepared_5_adapter = self._adapter(FakeSceneObserver([after, clear_after]), FakeRobot())
        prepared_5_goal = goal()
        prepared_5_scene, prepared_5_frames, _, _ = prepared_5_adapter.capture_scene(
            prepared_5_goal, prefix="test_orchestrator_before", evidence_dir=None)
        text_action_4 = SemanticAction(
                node_id="clear-segment",
                action="clear_verified_text",
                params={"text_transport": "adb_keyboard", "input_field_id": (dict(after.elements[0].states)).get("input_field_id", "field_primary"), "prior_input_value": (dict(after.elements[0].states)).get("value", ""), "expected_input_value": '',
                    "element_id": "field",
                    "target": "application_text_input",
                    "role": "input",
                    "label": "longinput",
                    "states": dict(after.elements[0].states),
                },
            )
        clear_result = prepared_5_adapter.execute(action_authority=consumed_authority(text_action_4, prepared_5_scene),
            requested_action=text_action_4,
            planned_scene=prepared_5_scene, planned_frames=prepared_5_frames,
            goal=prepared_5_goal,
            confirmed=True,
        )
        self.assertEqual("executed", clear_result.action_outcome)

        wrong_after = input_scene("wrong-after", "longinpuw")
        mismatch_robot = FakeRobot()
        with self.assertRaisesRegex(GenericActionAdapterError, "文字不匹配") as caught:
            prepared_18_adapter = self._adapter(FakeSceneObserver([before, wrong_after]), mismatch_robot)
            prepared_18_goal = goal()
            prepared_18_scene, prepared_18_frames, _, _ = prepared_18_adapter.capture_scene(
                prepared_18_goal, prefix="test_orchestrator_before", evidence_dir=None)
            text_action_10 = input_action
            prepared_18_adapter.execute(action_authority=consumed_authority(text_action_10, prepared_18_scene),
                requested_action=text_action_10,
                planned_scene=prepared_18_scene, planned_frames=prepared_18_frames,
                goal=prepared_18_goal,
                confirmed=True,
            )
        self.assertEqual(1, caught.exception.physical_actions)
        self.assertEqual(1, len(mismatch_robot.actions))

    def test_confirmed_input_uses_qwen_selected_id_before_exact_post_receipt(self):
        def input_scene(
            fingerprint,
            element_id,
            meaning,
            label,
            value,
            *,
            app_id="browser",
            screen_id="search",
            bounds=(0.1, 0.1, 0.9, 0.2),
            keyboard_input_mode="direct_latin",
        ):
            return UIScene(
                app_id=app_id,
                screen_id=screen_id,
                summary="唯一输入框",
                elements=(
                    UIElement(
                        element_id=element_id,
                        role="input",
                        meaning=meaning,
                        label=label,
                        bounds=bounds,
                        confidence=0.98,
                        states={"input_field_id": "field_primary", **({
                            "focused": True,
                            "value": value,
                            "keyboard_layout": "qwerty",
                            "keyboard_input_mode": keyboard_input_mode,
                            "keyboard_geometry": TEST_QWERTY_GEOMETRY,
                            "goal_relevant": True,
                        })},
                    ),
                ),
                stable=True,
                confidence=0.98,
                fingerprint=fingerprint,
            )

        planned = input_scene(
            "planned", "planned-input", "target_text_input", "", ""
        )
        fresh = input_scene(
            "before", "planned-input", "target_text_input", "", ""
        )
        after = input_scene(
            "after", "audited-input", "application_text_input", "agent", "agent"
        )
        robot = FakeRobot()
        prepared_6_adapter = self._adapter(FakeSceneObserver([fresh, after]), robot)
        prepared_6_goal = goal()
        prepared_6_scene, prepared_6_frames, _, _ = prepared_6_adapter.capture_scene(
            prepared_6_goal, prefix="test_orchestrator_before", evidence_dir=None)
        text_action_5 = SemanticAction(
                node_id="input-alias",
                action="input_verified_text",
                params={"text_transport": "adb_keyboard", "input_field_id": ({
                        "focused": True,
                        "value": "",
                        "keyboard_layout": "qwerty",
                        "keyboard_input_mode": "direct_latin",
                        "keyboard_geometry": TEST_QWERTY_GEOMETRY,
                        "goal_relevant": True,
                    }).get("input_field_id", "field_primary"), "prior_input_value": ({
                        "focused": True,
                        "value": "",
                        "keyboard_layout": "qwerty",
                        "keyboard_input_mode": "direct_latin",
                        "keyboard_geometry": TEST_QWERTY_GEOMETRY,
                        "goal_relevant": True,
                    }).get("value", ""), "expected_input_value": "agent", "input_fragment": ("agent")[len(({
                        "focused": True,
                        "value": "",
                        "keyboard_layout": "qwerty",
                        "keyboard_input_mode": "direct_latin",
                        "keyboard_geometry": TEST_QWERTY_GEOMETRY,
                        "goal_relevant": True,
                    }).get("value", "")):],
                    "element_id": "planned-input",
                    "target": "target_text_input",
                    "role": "input",
                    "label": "",
                    "states": {
                        "focused": True,
                        "value": "",
                        "keyboard_layout": "qwerty",
                        "keyboard_input_mode": "direct_latin",
                        "keyboard_geometry": TEST_QWERTY_GEOMETRY,
                        "goal_relevant": True,
                    },
                    "text": "agent",

                },
            )
        result = prepared_6_adapter.execute(action_authority=consumed_authority(text_action_5, prepared_6_scene),
            requested_action=text_action_5,
            planned_scene=prepared_6_scene, planned_frames=prepared_6_frames,
            goal=prepared_6_goal,
            confirmed=True,
        )

        self.assertEqual([("input", "agent")], robot.actions)
        self.assertEqual("executed", result.action_outcome)

    def test_confirmed_input_accepts_identity_degradation_to_unknown_for_same_input(self):
        def input_scene(fingerprint, app_id, screen_id, value):
            return UIScene(
                app_id=app_id,
                screen_id=screen_id,
                summary="本地输入页",
                elements=(
                    UIElement(
                        element_id="audited-input",
                        role="input",
                        meaning="target_text_input",
                        label="验收输入框",
                        bounds=(0.12, 0.38, 0.87, 0.47),
                        confidence=0.98,
                        states={"input_field_id": "field_primary", **({
                            "focused": True,
                            "value": value,
                            "keyboard_layout": "qwerty",
                            "keyboard_input_mode": "direct_latin",
                            "keyboard_geometry": TEST_QWERTY_GEOMETRY,
                            "goal_relevant": True,
                        })},
                    ),
                ),
                stable=True,
                confidence=0.98,
                fingerprint=fingerprint,
                camera_alignment=aligned_camera_facts(),
            )

        planned = input_scene("planned", "current_foreground", "通用动作真机验收页", "")
        fresh = input_scene("before", "current_foreground", "通用动作真机验收页", "")
        after = input_scene("after", "unknown", "unknown", "agent")
        robot = FakeRobot()

        prepared_7_adapter = self._adapter(FakeSceneObserver([fresh, after]), robot)
        prepared_7_goal = goal()
        prepared_7_scene, prepared_7_frames, _, _ = prepared_7_adapter.capture_scene(
            prepared_7_goal, prefix="test_orchestrator_before", evidence_dir=None)
        text_action_6 = SemanticAction(
                node_id="input-identity-degradation",
                action="input_verified_text",
                params={"text_transport": "adb_keyboard", "input_field_id": ({
                        "focused": True,
                        "value": "",
                        "keyboard_layout": "qwerty",
                        "keyboard_input_mode": "direct_latin",
                        "keyboard_geometry": TEST_QWERTY_GEOMETRY,
                        "goal_relevant": True,
                    }).get("input_field_id", "field_primary"), "prior_input_value": ({
                        "focused": True,
                        "value": "",
                        "keyboard_layout": "qwerty",
                        "keyboard_input_mode": "direct_latin",
                        "keyboard_geometry": TEST_QWERTY_GEOMETRY,
                        "goal_relevant": True,
                    }).get("value", ""), "expected_input_value": "agent", "input_fragment": ("agent")[len(({
                        "focused": True,
                        "value": "",
                        "keyboard_layout": "qwerty",
                        "keyboard_input_mode": "direct_latin",
                        "keyboard_geometry": TEST_QWERTY_GEOMETRY,
                        "goal_relevant": True,
                    }).get("value", "")):],
                    "element_id": "audited-input",
                    "target": "target_text_input",
                    "role": "input",
                    "label": "验收输入框",
                    "states": {
                        "focused": True,
                        "value": "",
                        "keyboard_layout": "qwerty",
                        "keyboard_input_mode": "direct_latin",
                        "keyboard_geometry": TEST_QWERTY_GEOMETRY,
                        "goal_relevant": True,
                    },
                    "text": "agent",

                },
            )
        result = prepared_7_adapter.execute(action_authority=consumed_authority(text_action_6, prepared_7_scene),
            requested_action=text_action_6,
            planned_scene=prepared_7_scene, planned_frames=prepared_7_frames,
            goal=prepared_7_goal,
            confirmed=True,
        )

        self.assertEqual([("input", "agent")], robot.actions)
        self.assertEqual("executed", result.action_outcome)

    def test_adb_unicode_input_uses_authorized_transport_once_then_visual_verifies(self):
        before = UIScene(
            app_id="sample.app",
            screen_id="editor",
            summary="唯一 typed 输入框已聚焦",
            elements=(UIElement(element_id="field", role="input", meaning="application_text_input",
                label="前缀", bounds=(0.1, 0.2, 0.9, 0.3), confidence=0.99, states={
                    "focused": True, "goal_relevant": True, "fully_visible": True,
                    "value": "前缀", "input_field_id": "field_primary", "ime_preedit_text": "",
                }, evidence=("应用输入框当前文字：前缀",)),),
            stable=True,
            confidence=0.99,
            fingerprint="before-companion",
            camera_alignment=aligned_camera_facts(),
        )
        expected = "前缀🙂\nsecond@例"
        after = replace(before, fingerprint="after-companion", elements=(replace(before.elements[0],
            label=expected, states={**before.elements[0].states, "value": expected},
            evidence=(f"应用输入框当前文字：{expected}",)),))
        action = SemanticAction(node_id="companion-unicode", action="input_verified_text", params={
            "element_id": "field", "target": "application_text_input", "role": "input", "label": "前缀",
            "states": before.elements[0].states, "text": expected, "text_transport": "adb_keyboard",
            "input_field_id": "field_primary", "prior_input_value": "前缀",
            "input_fragment": "🙂\nsecond@例", "expected_input_value": expected,

        })
        authority = ConfirmationAuthority(session_id="session-1", task_id="task-1", device_id="test-device",
            revision=3, step_id="subgoal-1", effect_ids=(), observation_id="observation-1",
            fingerprint=before.fingerprint, decision_node_id=action.node_id,
            action_digest=canonical_digest(action.to_dict()), consumed=True)
        transport = FakeAdbKeyboardTextTransport()
        robot = FakeRobot()

        prepared_8_adapter = self._adapter(FakeSceneObserver([before, after, after]), robot,
            text_transport=transport)
        prepared_8_goal = goal()
        prepared_8_scene, prepared_8_frames, _, _ = prepared_8_adapter.capture_scene(
            prepared_8_goal, prefix="test_orchestrator_before", evidence_dir=None)
        result = prepared_8_adapter.execute(requested_action=action, planned_scene=prepared_8_scene, planned_frames=prepared_8_frames, goal=prepared_8_goal,
            confirmed=True, action_authority=authority)

        self.assertEqual("executed", result.action_outcome)
        self.assertEqual(1, result.physical_actions)
        self.assertEqual([], robot.actions)
        self.assertEqual(1, len(transport.calls))
        self.assertEqual(("append_text", "🙂\nsecond@例"), (transport.calls[0][0], transport.calls[0][2]))
        minted = transport.minted[0]
        self.assertEqual("field_primary", minted["input_field_id"])
        self.assertEqual(before.fingerprint, minted["observation_fingerprint"])
        self.assertNotIn(expected, json.dumps(result.execution_metadata, ensure_ascii=False))
        self.assertEqual("accepted", result.execution_metadata["transport_status"])

    def test_adb_keyboard_transport_accepts_focused_typed_field_without_visible_keyboard(self):
        before = UIScene(
            app_id="sample.app", screen_id="editor", summary="唯一 typed 输入框可见",
            elements=(UIElement(element_id="field", role="input", meaning="application_text_input",
                label="", bounds=(0.1, 0.2, 0.9, 0.3), confidence=0.99, states={
                    "goal_relevant": True, "fully_visible": True, "focused": True,
                    "soft_keyboard_visible": False,
                    "value": "", "input_field_id": "field_primary", "ime_preedit_text": "",
                }),), stable=True, confidence=0.99, fingerprint="before-adb-no-keyboard",
            camera_alignment=aligned_camera_facts(),
        )
        expected = "ADB测试？你好"
        after = replace(before, fingerprint="after-adb-no-keyboard", elements=(replace(before.elements[0],
            label=expected, states={**before.elements[0].states, "value": expected}),))
        action = SemanticAction(node_id="adb-unicode", action="input_verified_text", params={
            "element_id": "field", "target": "application_text_input", "role": "input", "label": "",
            "states": before.elements[0].states, "text": expected, "text_transport": "adb_keyboard",
            "input_field_id": "field_primary", "prior_input_value": "",
            "input_fragment": expected, "expected_input_value": expected,
        })
        authority = ConfirmationAuthority(session_id="session-1", task_id="task-1", device_id="test-device",
            revision=3, step_id="subgoal-1", effect_ids=(), observation_id="observation-1",
            fingerprint=before.fingerprint, decision_node_id=action.node_id,
            action_digest=canonical_digest(action.to_dict()), consumed=True)
        transport = FakeAdbKeyboardTextTransport()
        robot = FakeRobot()

        prepared_9_adapter = self._adapter(FakeSceneObserver([before, after, after]), robot,
            text_transport=transport)
        prepared_9_goal = goal()
        prepared_9_scene, prepared_9_frames, _, _ = prepared_9_adapter.capture_scene(
            prepared_9_goal, prefix="test_orchestrator_before", evidence_dir=None)
        result = prepared_9_adapter.execute(requested_action=action, planned_scene=prepared_9_scene, planned_frames=prepared_9_frames, goal=prepared_9_goal,
            confirmed=True, action_authority=authority)

        self.assertEqual("executed", result.action_outcome)
        self.assertEqual(("append_text", expected), (transport.calls[0][0], transport.calls[0][2]))

    def test_confirmed_input_accepts_optional_page_wording_change_with_exact_value(self):
        before = UIScene(
            app_id="browser",
            screen_id="page-a",
            summary="输入页",
            elements=(
                UIElement(
                    element_id="field",
                    role="input",
                    meaning="target_text_input",
                    label="输入框",
                    bounds=(0.1, 0.1, 0.9, 0.2),
                    confidence=0.98,
                    states={"input_field_id": "field_primary", **({
                        "focused": True,
                        "value": "",
                        "keyboard_layout": "qwerty",
                        "keyboard_input_mode": "direct_latin",
                        "keyboard_geometry": TEST_QWERTY_GEOMETRY,
                        "goal_relevant": True,
                    })},
                ),
            ),
            stable=True,
            confidence=0.98,
            fingerprint="before",
            camera_alignment=aligned_camera_facts(),
        )
        after = replace(
            before,
            screen_id="page-b",
            fingerprint="after",
            elements=(replace(before.elements[0], states={**before.elements[0].states, "value": "agent"}),),
        )
        robot = FakeRobot()

        prepared_10_adapter = self._adapter(FakeSceneObserver([before, after]), robot)
        prepared_10_goal = goal()
        prepared_10_scene, prepared_10_frames, _, _ = prepared_10_adapter.capture_scene(
            prepared_10_goal, prefix="test_orchestrator_before", evidence_dir=None)
        text_action_7 = SemanticAction(
                node_id="accept-page-wording-change",
                action="input_verified_text",
                params={"text_transport": "adb_keyboard", "input_field_id": (before.elements[0].states).get("input_field_id", "field_primary"), "prior_input_value": (before.elements[0].states).get("value", ""), "expected_input_value": "agent", "input_fragment": ("agent")[len((before.elements[0].states).get("value", "")):],
                    "element_id": "field",
                    "target": "target_text_input",
                    "role": "input",
                    "label": "输入框",
                    "states": before.elements[0].states,
                    "text": "agent",
                },
            )
        result = prepared_10_adapter.execute(action_authority=consumed_authority(text_action_7, prepared_10_scene),
            requested_action=text_action_7,
            planned_scene=prepared_10_scene, planned_frames=prepared_10_frames,
            goal=prepared_10_goal,
            confirmed=True,
        )
        self.assertEqual("executed", result.action_outcome)
        self.assertEqual(1, result.physical_actions)

    def test_stable_frames_allow_input_field_bounds_drift_with_fresh_keyboard_geometry(self):
        sharp_frame = textured_phone_frame()
        blurred_frame = sharp_frame.filter(ImageFilter.GaussianBlur(2))
        states = {
            "goal_relevant": True,
            "fully_visible": True,
            "focused": True,
            "value": "",
            "keyboard_layout": "qwerty",
            "keyboard_input_mode": "direct_latin",
            "keyboard_case_mode": "lower",
            "keyboard_geometry": TEST_QWERTY_GEOMETRY,
        }

        def input_scene(fingerprint, bounds, *, value=""):
            return UIScene(
                app_id="wechat",
                screen_id="chat_file_transfer_helper",
                summary="同一会话中的唯一聚焦输入框和 QWERTY 键盘",
                elements=(
                    UIElement(
                        element_id="local_audited_input_1",
                        role="input",
                        meaning="application_text_input",
                        label="",
                        bounds=bounds,
                        confidence=1.0,
                        states={"input_field_id": "field_primary", **({**states, "value": value})},
                    ),
                ),
                stable=True,
                confidence=1.0,
                fingerprint=fingerprint,
                camera_alignment=aligned_camera_facts(),
            )

        planned = input_scene("planned", (0.14, 0.53, 0.69, 0.58))
        fresh = input_scene("fresh", (0.24, 0.623, 0.80, 0.68))
        target_text = "longinput2026abcdefghijklmnopqrstuvwxyz"
        first_segment = target_text
        after = input_scene(
            "after",
            (0.15, 0.53, 0.70, 0.60),
            value=first_segment,
        )
        observer = FakeSceneObserver([after])
        robot = FakeRobot()
        snapped_anchors = {
            key: list(value)
            for key, value in TEST_QWERTY_GEOMETRY["anchors"].items()
        }
        adapter = GenericSingleActionAdapter(
            capture=SequenceCapture(
                [sharp_frame] * 4 + [blurred_frame] * 4 + [sharp_frame] * 4
            ),
            observer=observer,
            robot=robot,
            frame_interval=0,
            post_action_settle=0,
        )

        text_action_8 = SemanticAction(
                node_id="input-long-text",
                action="input_verified_text",
                params={"text_transport": "adb_keyboard", "input_field_id": (states).get("input_field_id", "field_primary"), "prior_input_value": (states).get("value", ""), "expected_input_value": target_text, "input_fragment": (target_text)[len((states).get("value", "")):],
                    "element_id": "local_audited_input_1",
                    "target": "application_text_input",
                    "role": "input",
                    "label": "",
                    "states": states,
                    "text": target_text,

                },
            )
        result = adapter.execute(action_authority=consumed_authority(text_action_8, planned),
            requested_action=text_action_8,
            planned_scene=planned,
            planned_frames=tuple(
                sharp_frame.copy() for _ in range(4)
            ),
            goal=goal(),
            confirmed=True,
        )

        self.assertEqual([("input", first_segment)], robot.actions)
        self.assertEqual(1, result.physical_actions)
        self.assertEqual("executed", result.action_outcome)
        self.assertGreater(adapter.capture.calls, 8)
        self.assertEqual(1, observer.calls)
        self.assertEqual("planned", result.before_scene.fingerprint)
        self.assertFalse(hasattr(robot, "vision_type_text_with_layout"))

    def test_matching_frames_do_not_accept_second_model_input_mode_veto(self):
        planned_states = {
            "goal_relevant": True,
            "fully_visible": True,
            "focused": True,
            "value": "",
            "keyboard_layout": "qwerty",
            "keyboard_input_mode": "direct_latin",
            "keyboard_case_mode": "lower",
            "keyboard_geometry": TEST_QWERTY_GEOMETRY,
        }
        planned = UIScene(
            app_id="generic_app",
            screen_id="editor",
            summary="唯一聚焦输入框",
            elements=(
                UIElement(
                    element_id="input",
                    role="input",
                    meaning="application_text_input",
                    label="",
                    bounds=(0.14, 0.53, 0.69, 0.58),
                    confidence=1.0,
                    states={"input_field_id": "field_primary", **(planned_states)},
                ),
            ),
            stable=True,
            confidence=1.0,
            fingerprint="planned",
            camera_alignment=aligned_camera_facts(),
        )
        fresh = replace(
            planned,
            elements=(
                replace(
                    planned.elements[0],
                    bounds=(0.24, 0.623, 0.80, 0.68),
                    states={
                        **planned_states,
                        "keyboard_input_mode": "chinese_pinyin",
                    },
                ),
            ),
            fingerprint="fresh",
        )
        after = replace(
            planned,
            elements=(
                replace(
                    planned.elements[0],
                    states={**planned_states, "value": "agent"},
                ),
            ),
            fingerprint="after",
        )
        observer = FakeSceneObserver([after])
        robot = FakeRobot()
        adapter = GenericSingleActionAdapter(
            capture=SequenceCapture(["gray"] * 4),
            observer=observer,
            robot=robot,
            frame_interval=0,
            post_action_settle=0,
        )

        text_action_9 = SemanticAction(
                    node_id="input-mode-changed",
                    action="input_verified_text",
                    params={"text_transport": "adb_keyboard", "input_field_id": (planned_states).get("input_field_id", "field_primary"), "prior_input_value": (planned_states).get("value", ""), "expected_input_value": "agent", "input_fragment": ("agent")[len((planned_states).get("value", "")):],
                        "element_id": "input",
                        "target": "application_text_input",
                        "role": "input",
                        "label": "",
                        "states": planned_states,
                        "text": "agent",
                    },
                )
        result = adapter.execute(action_authority=consumed_authority(text_action_9, planned),
                requested_action=text_action_9,
                planned_scene=planned,
                planned_frames=tuple(
                    Image.new("RGB", (540, 960), "gray") for _ in range(4)
                ),
                goal=goal(),
                confirmed=True,
            )

        self.assertEqual(1, result.physical_actions)
        self.assertEqual([("input", "agent")], robot.actions)

    def test_navigation_receipt_does_not_depend_on_next_input_readiness(self):
        planned = scene("before")
        wrong_conversation = scene(
            "wrong-after",
            screen_id="wrong_named_conversation",
            app_id="com.example.messaging",
        )
        current_goal = navigation_goal()
        current_goal.entities["next_subgoal_visual_context"] = {
            "step_id": "input_message",
            "objective": "在消息输入框输入abc",
            "constraints": [],
            "completion_conditions": ["消息输入框内容为abc"],
            "execution_class": "navigate",
            "goal_entities": {
                "active_input_transaction_text": "abc",
                "active_input_field_id": "message_field",
                "active_input_field_label": "消息",
                "active_input_multiline": False,
            },
        }
        robot = FakeRobot()
        adapter = GenericSingleActionAdapter(
            capture=SequenceCapture(["gray"] * 8),
            observer=FakeSceneObserver([wrong_conversation]),
            robot=robot,
            frame_interval=0,
            post_action_settle=0,
        )

        result = adapter.execute(
            requested_action=SemanticAction(
                node_id="open-target",
                action="tap_semantic",
                params={"tap_point": (0.3, 0.4),
                    "element_id": "e1",
                    "target": "app_icon",

                },
            ),
            planned_scene=planned,
            planned_frames=tuple(
                Image.new("RGB", (540, 960), "gray") for _ in range(4)
            ),
            goal=current_goal,
            confirmed=True,
        )

        self.assertEqual(1, result.physical_actions)
        self.assertEqual("executed", result.action_outcome)
        self.assertEqual((), result.verification_errors)
        self.assertEqual(1, len(robot.actions))

    def test_input_post_action_waits_past_stable_blur_until_clarity_recovers(self):
        sharp = textured_phone_frame()
        blurred = sharp.filter(ImageFilter.GaussianBlur(2))
        capture = SequenceCapture([blurred] * 4 + [sharp] * 4)
        adapter = GenericSingleActionAdapter(
            capture=capture,
            observer=FakeSceneObserver([]),
            robot=FakeRobot(),
            frame_interval=0,
            post_action_settle=0,
            post_action_timeout=1,
        )

        frames, _paths = adapter._capture_stable_post_action_frames(
            deadline=10**9,
            evidence_dir=None,
            prefix="clarity_recovers",
            clarity_reference_frames=tuple(sharp.copy() for _ in range(4)),
            require_relative_clarity=True,
        )

        reference = measure_frame_sharpness(sharp)
        candidate = sorted(measure_frame_sharpness(frame) for frame in frames)
        self.assertGreater(capture.calls, 4)
        self.assertGreaterEqual(
            (candidate[1] + candidate[2]) / 2 / reference,
            0.80,
        )

    def test_relative_clarity_threshold_separates_observed_input_samples(self):
        adapter = GenericSingleActionAdapter(
            capture=SequenceCapture(["gray"]),
            observer=FakeSceneObserver([]),
            robot=FakeRobot(),
        )
        observed_success_ratios = (
            0.934,
            1.007,
            0.985,
            0.945,
            1.005,
            1.018,
            1.276,
        )

        self.assertLess(0.568, adapter.post_action_min_relative_sharpness)
        self.assertTrue(
            all(
                ratio >= adapter.post_action_min_relative_sharpness
                for ratio in observed_success_ratios
            )
        )

    def test_relative_clarity_is_scoped_to_comparable_input_mutations(self):
        input_focus = ResolvedSemanticAction(
            node_id="focus-input",
            kind="tap_semantic",
        )
        navigation = ResolvedSemanticAction(
            node_id="open-page",
            kind="tap_semantic",
        )

        self.assertFalse(
            GenericSingleActionAdapter._requires_post_action_relative_clarity(
                input_focus
            )
        )
        self.assertFalse(
            GenericSingleActionAdapter._requires_post_action_relative_clarity(
                navigation
            )
        )
        for kind in (
            "input_verified_text",
            "press_enter",
            "clear_verified_text",
        ):
            with self.subTest(kind=kind):
                mutation = ResolvedSemanticAction(
                    node_id=kind,
                    kind=kind,
                )
                self.assertTrue(
                    GenericSingleActionAdapter._requires_post_action_relative_clarity(
                        mutation
                    )
                )

    def test_input_post_action_stable_blur_times_out_before_model_call(self):
        sharp = textured_phone_frame()
        blurred = sharp.filter(ImageFilter.GaussianBlur(2))
        observer = FakeSceneObserver([])
        adapter = GenericSingleActionAdapter(
            capture=SequenceCapture([blurred] * 4),
            observer=observer,
            robot=FakeRobot(),
            frame_interval=0,
            post_action_settle=0,
            post_action_timeout=1,
        )

        with patch(
            "agent.infrastructure.generic_action_adapter.time.monotonic",
            side_effect=[0.0, 0.0, 0.0, 1.0],
        ):
            with self.assertRaisesRegex(
                GenericActionAdapterError,
                "仍不够清晰.*相对值.*要求至少0.800",
            ):
                adapter._capture_stable_post_action_frames(
                    deadline=0.5,
                    evidence_dir=None,
                    prefix="clarity_timeout",
                    clarity_reference_frames=tuple(
                        sharp.copy() for _ in range(4)
                    ),
                    require_relative_clarity=True,
                )

        self.assertEqual(observer.calls, 0)

    def test_low_texture_input_reference_keeps_existing_stability_contract(self):
        adapter = GenericSingleActionAdapter(
            capture=SequenceCapture(["white"] * 4),
            observer=FakeSceneObserver([]),
            robot=FakeRobot(),
            frame_interval=0,
            post_action_settle=0,
            post_action_timeout=1,
        )

        frames, _paths = adapter._capture_stable_post_action_frames(
            deadline=10**9,
            evidence_dir=None,
            prefix="low_texture_reference",
            clarity_reference_frames=tuple(
                Image.new("RGB", (540, 960), "gray") for _ in range(4)
            ),
            require_relative_clarity=True,
        )

        self.assertEqual(len(frames), 4)


if __name__ == "__main__":
    unittest.main()
