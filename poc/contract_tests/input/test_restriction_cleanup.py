"""Offline counterexamples for unnecessary restrictions; no device or model I/O."""
from dataclasses import replace
from pathlib import Path
import unittest
from PIL import Image
from agent.domain.confirmation_authority import ConfirmationAuthority
from agent.domain.semantic_action import SemanticAction
from agent.domain.ui_scene import UIElement, UIScene
from agent.domain.universal_action_controller import UniversalActionController, UniversalActionError
from agent.domain.validation import canonical_digest
import test_support.generic_action_adapter as fixtures


class ClearResultRestrictionTests(unittest.TestCase):
    @staticmethod
    def sample(app_id="notes", value="草稿", preedit="cao"):
        before = UIScene(app_id=app_id, screen_id="editor", summary="当前输入框",
            elements=(UIElement(element_id="field", role="input", meaning="application_text_input",
                label="正文", bounds=(.1, .1, .9, .3), confidence=.99,
                states={"focused": True, "goal_relevant": True, "fully_visible": True,
                    "value": value, "input_field_id": "body", "ime_preedit_text": preedit}),),
            stable=True, confidence=.99, fingerprint="before")
        action = SemanticAction("clear", "clear_verified_text", {
            "element_id": "field", "text_transport": "adb_keyboard", "input_field_id": "body",
            "prior_input_value": value, "expected_input_value": ""})
        controller = UniversalActionController()
        resolved = controller.resolve_one(action, before, confirmed=True)
        after = replace(before, fingerprint="after", elements=(replace(before.elements[0],
            states={**before.elements[0].states, "value": "", "ime_preedit_text": "", "focused": False}),))
        return controller, action, resolved, before, after

    def test_empty_result_does_not_require_continued_focus(self):
        for app_id, value, preedit in (("notes", "草稿", "cao"), ("mail", "", "draft")):
            for focus in (False, "unknown", True):
                with self.subTest(app=app_id, focus=focus):
                    controller, _, resolved, before, after = self.sample(app_id, value, preedit)
                    after = replace(after, elements=(replace(after.elements[0],
                        states={**after.elements[0].states, "focused": focus}),))
                    self.assertTrue(controller.verify_after_action(resolved, before, after))

    def test_residue_and_wrong_field_still_rejected(self):
        controller, _, resolved, before, after = self.sample()
        for change in ({"value": "残留"}, {"ime_preedit_text": "cao"}, {"input_field_id": "other"}):
            with self.subTest(change=change):
                bad = replace(after, elements=(replace(after.elements[0],
                    states={**after.elements[0].states, **change}),))
                with self.assertRaises(UniversalActionError):
                    controller.verify_after_action(resolved, before, bad)

    def test_clear_before_action_still_requires_focus(self):
        controller, action, _, before, _ = self.sample()
        for focus in (False, "unknown"):
            with self.subTest(focus=focus):
                before = replace(before, elements=(replace(before.elements[0],
                    states={**before.elements[0].states, "focused": focus}),))
                with self.assertRaisesRegex(UniversalActionError, "清空文字前.*聚焦"):
                    controller.resolve_one(action, before, confirmed=True)

    def test_adapter_clears_once_and_preserves_new_frame_finish_after_focus_loss(self):
        _, action, _, before, after = self.sample()
        before = replace(before, camera_alignment=fixtures.aligned_camera_facts())
        after = replace(after, camera_alignment=fixtures.aligned_camera_facts())
        authority = ConfirmationAuthority(session_id="audit", task_id="task", device_id="test-device",
            revision=1, step_id="clear", effect_ids=(), observation_id="obs",
            fingerprint=before.fingerprint, decision_node_id=action.node_id,
            action_digest=canonical_digest(action.to_dict()), consumed=True)
        transport = fixtures.FakeAdbKeyboardTextTransport()
        robot = fixtures.FakeRobot()
        observer = fixtures.FakeSceneObserver([after])
        adapter = fixtures.GenericSingleActionAdapter(
            capture=fixtures.SequenceCapture(["gray"] * 4 + ["white"] * 4), observer=observer,
            robot=robot, text_transport=transport, frame_interval=0, post_action_settle=0)
        result = adapter.execute(requested_action=action, planned_scene=before,
            planned_frames=tuple(Image.new("RGB", (540, 960), "gray") for _ in range(4)),
            goal=fixtures.goal(), confirmed=True, action_authority=authority)
        self.assertEqual("executed", result.action_outcome)
        self.assertEqual("finish", result.after_model_decision["status"])
        self.assertEqual(1, result.physical_actions)
        self.assertEqual(1, observer.calls)
        self.assertEqual(["clear_text"], [call[0] for call in transport.calls])
        self.assertEqual([], robot.actions)


class InputEffectOrderingPromptTests(unittest.TestCase):
    def test_explicit_text_effects_keep_input_as_a_separate_step(self):
        root = Path(__file__).resolve().parents[2] / "agent"
        prompt = (root / "infrastructure/prompts/single_step_observation.txt").read_text(encoding="utf-8")
        self.assertIn("需要输入并提交明确文字时，输入必须作为独立动作完成", prompt)
        self.assertIn("只有实际历史已经记录该input_verified_text", prompt)
        self.assertIn("不能跳过输入", prompt)


class RecentPageRestrictionTests(unittest.TestCase):
    def test_navigation_is_general_but_cleanup_button_remains_scoped(self):
        root = Path(__file__).resolve().parents[2] / "agent"
        step = (root / "infrastructure/prompts/single_step_observation.txt").read_text(encoding="utf-8")
        self.assertIn("打开后台统一选择 open_recent_apps", step)
        self.assertIn("本地负责上述 Home→新图确认 Launcher→打开后台", step)
        self.assertIn("不得用 swipe_element 划卡片", step)
        self.assertIn("系统清理按钮规则不适用于普通查看后台", step)
        self.assertNotIn("此顺序不适用于普通查看后台", step)

    def test_page_name_alone_does_not_forbid_navigation(self):
        controller, _, _, before, _ = ClearResultRestrictionTests.sample()
        for screen in ("system_recent_tasks", "ordinary_list"):
            current = replace(before, screen_id=screen, elements=())
            for direction in ("left", "right", "up", "down"):
                with self.subTest(screen=screen, direction=direction):
                    resolved = controller.resolve_one(SemanticAction("browse", "scroll",
                        {"direction": direction}), current)
                    self.assertEqual(direction, resolved.direction)

    def test_non_cleanup_element_gesture_is_not_forbidden_by_page_name(self):
        controller, _, _, before, _ = ClearResultRestrictionTests.sample()
        target = UIElement(element_id="split-handle", role="container", meaning="resize_handle",
            label="调整区域", bounds=(.2, .2, .6, .7), confidence=.99,
            states={"goal_relevant": True, "fully_visible": True})
        action = SemanticAction("resize", "swipe_element", {
            "element_id": target.element_id, "start": (.4, .5), "end": (.7, .5)})
        for screen in ("system_recent_tasks", "ordinary_list"):
            with self.subTest(screen=screen):
                current = replace(before, screen_id=screen, elements=(target,))
                self.assertEqual("swipe_element", controller.resolve_one(action, current).kind)


if __name__ == "__main__":
    unittest.main()
