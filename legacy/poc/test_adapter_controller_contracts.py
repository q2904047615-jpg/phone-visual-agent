from __future__ import annotations
from agent.domain import DeviceActionRequest
from agent.domain import DeviceExecutionError
from agent.infrastructure import RobotDeviceExecutor
from agent.domain.semantic_action import SemanticAction
from agent.domain.ui_scene import SystemUIFacts
from agent.domain.ui_scene import UIElement
from agent.domain.ui_scene import UIScene
from agent.domain.universal_action_controller import UniversalActionController
from agent.domain.universal_action_controller import UniversalActionError
from dataclasses import replace
import unittest
from test_support.generic_action_adapter import (
    FakeRobot,
    _BaseElementBoundSwipeControllerTests,
    _BaseRevealSystemNavigationControllerTests,
    scene,
)


class RevealSystemNavigationControllerTests(_BaseRevealSystemNavigationControllerTests):
    def test_requires_pre_facts_but_does_not_rejudge_post_navigation(self):
        controller = UniversalActionController()
        before = scene(
            "same",
            system_ui=SystemUIFacts(
                immersive_or_fullscreen=True,
                navigation_bar_visible=False,
            ),
        )
        resolved = controller.resolve_one(self.action(), before, confirmed=True)

        controller.verify_after_action(
            resolved,
            before,
            scene(
                "same",
                system_ui=SystemUIFacts(
                    immersive_or_fullscreen=True,
                    navigation_bar_visible=True,
                ),
            ),
        )

        self.assertEqual((), controller.verify_after_action(
                resolved,
                before,
                scene("different", screen_id="navigation_bar_visible_summary_only"),
            ))

    def test_rejects_unknown_or_already_visible_system_ui_precondition(self):
        controller = UniversalActionController()
        for facts in (
            SystemUIFacts(
                immersive_or_fullscreen="unknown",
                navigation_bar_visible="unknown",
            ),
            SystemUIFacts(
                immersive_or_fullscreen=True,
                navigation_bar_visible=True,
            ),
        ):
            with self.subTest(facts=facts), self.assertRaisesRegex(
                UniversalActionError,
                "沉浸态且导航栏隐藏",
            ):
                controller.resolve_one(
                    self.action(),
                    scene("before", system_ui=facts),
                    confirmed=True,
                )


class ElementBoundSwipeControllerTests(_BaseElementBoundSwipeControllerTests):
    def test_controller_derives_path_and_accepts_fresh_stable_receipt(self):
        controller = UniversalActionController()
        before = self.before_scene()
        resolved = controller.resolve_one(self.action(), before, confirmed=True)

        self.assertAlmostEqual(0.5, resolved.normalized_point[0])
        self.assertAlmostEqual(0.68, resolved.normalized_point[1])
        self.assertEqual((0.5, 0.08), resolved.normalized_end_point)
        self.assertEqual("preview-card", resolved.target_element_id)
        self.assertEqual("up", resolved.direction)

        unchanged = replace(before, fingerprint="fresh-stable-receipt")
        self.assertEqual((), controller.verify_after_action(resolved, before, unchanged))

    def test_targeted_transport_uses_relative_path_but_viewport_keeps_preset(self):
        robot = FakeRobot()
        executor = RobotDeviceExecutor(robot)

        robot._armed = "swipe"
        targeted = executor.execute(
            DeviceActionRequest(
                kind="swipe_element",
                point=(500, 680),
                end_point=(500, 80),
                direction="up",
            )
        )
        self.assertEqual(1, targeted.physical_actions)
        self.assertEqual(
            [("swipe_relative", "up", 500, 680, 500, 80)],
            robot.actions,
        )

        robot._armed = "swipe"
        viewport = executor.execute(
            DeviceActionRequest(kind="scroll", direction="up")
        )
        self.assertEqual(1, viewport.physical_actions)
        self.assertEqual(("swipe", "up"), robot.actions[-1])

        with self.assertRaisesRegex(DeviceExecutionError, "请求方向不一致"):
            DeviceActionRequest(
                kind="swipe_element",
                point=(500, 80),
                end_point=(500, 680),
                direction="up",
            ).validate()


class ExactTypedInputControllerTests(unittest.TestCase):
    def test_adb_keyboard_clear_binds_typed_field_without_keyboard_geometry(self):
        controller = UniversalActionController()
        before = UIScene(app_id="generic_app", screen_id="editor", summary="唯一聚焦输入框",
            elements=(UIElement(element_id="field", role="input", meaning="application_text_input",
                label="草稿", bounds=(0.1, 0.1, 0.9, 0.2), confidence=0.98, states={
                    "focused": True, "goal_relevant": True, "fully_visible": True,
                    "value": "草稿🙂", "input_field_id": "field_primary", "ime_preedit_text": "",
                }),), stable=True, confidence=0.98, fingerprint="before-companion-clear")
        action = SemanticAction(node_id="companion-clear", action="clear_verified_text", params={
            "element_id": "field", "target": "application_text_input", "role": "input", "label": "草稿",
            "states": before.elements[0].states, "text_transport": "adb_keyboard",
            "input_field_id": "field_primary", "prior_input_value": "草稿🙂", "expected_input_value": "",
        })

        resolved = controller.resolve_one(action, before, confirmed=True)
        self.assertEqual("adb_keyboard", resolved.text_transport)
        self.assertIsNone(resolved.normalized_point)
        self.assertNotIn("delete_count", resolved.to_dict())
        after = replace(before, fingerprint="after-companion-clear", elements=(replace(before.elements[0],
            label="", states={**before.elements[0].states, "value": ""}),))
        controller.verify_after_action(resolved, before, after)

        stale_preedit = replace(after, fingerprint="after-companion-stale", elements=(replace(after.elements[0],
            states={**after.elements[0].states, "ime_preedit_text": "stale"}),))
        with self.assertRaisesRegex(UniversalActionError, "预编辑"):
            controller.verify_after_action(resolved, before, stale_preedit)


if __name__ == "__main__":
    unittest.main()
