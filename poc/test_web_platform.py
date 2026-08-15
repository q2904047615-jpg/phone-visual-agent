from __future__ import annotations

import ctypes
from contextlib import nullcontext
import json
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

import robot_gui_poc
import web_app
from device_exclusivity import InterProcessLease
from generic_intent import GenericIntentDraft
from generic_step_planner import GenericStepProposal
from ocr_runtime import find_text
from robot_core import (
    DEFAULT_CONFIG,
    MockRobotController as _MockRobotController,
    RobotController as _RobotController,
    classify_obscured_wechat_title,
    classify_wechat_page,
    controller_client_has_camera,
    oriented_navigation_ratio,
    qwerty_keyboard_config_from_anchors,
    qwerty_key_point,
)
from orientation_safety import _mint_audited_credential


class _TestDirectionCredentialMixin:
    """Keep legacy no-hardware tests behind the same one-shot gate."""

    def _consume_physical_execution(self, action, frame):
        credential = _mint_audited_credential(
            device_id=self.device_id,
            scene_fingerprint="test-scene",
            frame=frame,
            phone_content_rotation="upright",
            confidence=0.99,
            evidence=("合成手机界面轴线",),
        )
        self._physical_execution_gate.arm(
            credential,
            action=action,
            scene_fingerprint="test-scene",
        )
        return super()._consume_physical_execution(action, frame)


class RobotController(_TestDirectionCredentialMixin, _RobotController):
    def __init__(self, *args, device_id="test-device", **kwargs):
        super().__init__(*args, device_id=device_id, **kwargs)


class MockRobotController(_TestDirectionCredentialMixin, _MockRobotController):
    def __init__(self, *args, device_id="test-device", **kwargs):
        super().__init__(*args, device_id=device_id, **kwargs)


web_app.RobotController = RobotController
web_app.MockRobotController = MockRobotController
from web_app import (
    HybridAgent,
    RuleAgent,
    TaskStore,
    build_generic_plan_preview,
)
from task_orchestrator import GenericTaskOrchestrator
from semantic_executor import SemanticAction
from ui_scene import UIElement, UIScene
from state_controller import PageObservation as StatePageObservation
from target_locator import TargetResolution
from intent_provider import DeepSeekIntentProvider, IntentProviderError
from vision_agent import (
    DashScopeVisionProvider,
    ScriptedVisionProvider,
    VisionAgentError,
    VisionAgentRunner,
    VisionDecision,
    classify_douyin_heart_roi,
    douyin_heart_became_liked,
    goal_is_forbidden,
    infer_safe_plan_resume_cursor,
    locate_douyin_heart,
    numeric_grid_key_coordinate,
    requested_social_interaction_count,
    sent_message_transition_metrics,
    validate_decision,
    validate_execution_plan,
    plan_checkpoint_is_proven,
)
from vision_replay import (
    controller_owned_replay_decision,
    load_manifest,
    normalize_replay_decision,
)


TEST_QWERTY_LAYOUT = {
    "type": "qwerty",
    "anchors": {
        "q": [115, 704],
        "p": [875, 704],
        "a": [157, 773],
        "l": [832, 773],
        "z": [241, 844],
        "m": [747, 844],
        "backspace": [862, 844],
    },
}


class PhysicalNavigationSafetyTests(unittest.TestCase):
    def test_live_preview_uses_passive_capture_without_active_capture_path(self):
        controller = RobotController(title="test")
        frame = Image.new("RGB", (540, 1038), "white")

        with (
            patch("robot_core.legacy.find_window", return_value=(123, "test")),
            patch(
                "robot_core.legacy.capture_client_passive",
                return_value=frame,
            ) as passive,
            patch("robot_core.legacy.capture_client") as active,
        ):
            payload = controller.capture_preview()

        self.assertTrue(payload.startswith(b"\xff\xd8"))
        passive.assert_called_once_with(123)
        active.assert_not_called()

    def test_passive_capture_never_invokes_window_activation(self):
        frame = Image.new("RGB", (540, 1038), "white")

        with (
            patch("robot_gui_poc._window_is_minimized", return_value=False),
            patch("robot_gui_poc._validate_camera_region_unoccluded") as validate,
            patch(
                "robot_gui_poc.client_geometry",
                return_value=(10, 20, 540, 1038),
            ),
            patch("robot_gui_poc.ImageGrab.grab", return_value=frame) as grab,
            patch("robot_gui_poc.ensure_camera_region_unoccluded") as activate,
        ):
            result = robot_gui_poc.capture_client_passive(123)

        self.assertEqual((540, 1038), result.size)
        validate.assert_called_once_with(123)
        grab.assert_called_once_with(
            bbox=(10, 20, 550, 1058),
            all_screens=True,
        )
        activate.assert_not_called()

    def test_passive_capture_keeps_minimized_window_minimized(self):
        with (
            patch("robot_gui_poc._window_is_minimized", return_value=True),
            patch("robot_gui_poc.ImageGrab.grab") as grab,
            patch("robot_gui_poc.ensure_camera_region_unoccluded") as activate,
        ):
            with self.assertRaisesRegex(RuntimeError, "已最小化"):
                robot_gui_poc.capture_client_passive(123)

        grab.assert_not_called()
        activate.assert_not_called()

    def test_controller_client_rejects_small_landscape_error_dialog(self):
        self.assertFalse(controller_client_has_camera(379, 169))
        self.assertFalse(controller_client_has_camera(540, 400))
        self.assertTrue(controller_client_has_camera(540, 1038))
        self.assertTrue(controller_client_has_camera(1440, 810))
        self.assertTrue(controller_client_has_camera(1440, 885))

    def test_navigation_ratio_rotates_counter_clockwise_landscape_feed(self):
        landscape = oriented_navigation_ratio(0.685, 0.976, landscape=True)
        self.assertAlmostEqual(landscape[0], 0.976)
        self.assertAlmostEqual(landscape[1], 0.315)
        self.assertEqual(
            oriented_navigation_ratio(0.685, 0.976, landscape=False),
            (0.685, 0.976),
        )

    def test_navigation_tap_forces_single_click_and_returns_exact_pixel(self):
        controller = RobotController(title="test")
        frame = Image.new("RGB", (540, 960), "white")

        with (
            patch("robot_core.legacy.find_window", return_value=(123, "test")),
            patch.object(controller, "_capture_phone", return_value=frame),
            patch.object(controller, "_checkpoint"),
            patch("robot_core.legacy.configure_single_click_count") as configure,
            patch("robot_core.legacy.click_client_point") as click,
            patch("robot_core.legacy.move_cursor_outside_camera"),
            patch(
                "robot_core.load_workflow_config",
                return_value={"vision_agent": {"tap_hold": 0.35}},
            ),
        ):
            point = controller._vision_nav_tap(0.685, 0.976, action="back")

        self.assertEqual(point, (370, 937))
        configure.assert_called_once_with(123)
        click.assert_called_once_with(
            123,
            370,
            937,
            countdown=0,
            hold_seconds=0.35,
        )

    def test_drag_is_fail_closed_until_device_marks_it_verified(self):
        controller = RobotController(title="test")

        self.assertFalse(controller.hardware_capabilities()["drag"])
        with self.assertRaisesRegex(Exception, "尚未完成任意两点拖动真机验收"):
            controller.vision_drag_relative(100, 200, 700, 800)

    def test_unverified_input_and_long_press_fail_before_hardware_access(self):
        controller = RobotController(
            title="test",
            verified_actions={
                "tap_semantic",
                "dismiss_overlay",
                "swipe",
                "back",
                "wait_for_change",
            },
        )

        with patch("robot_core.legacy.find_window") as find_window:
            with self.assertRaisesRegex(Exception, "输入.*真机验收"):
                controller.vision_type_text("通用Agent验收草稿")
            with self.assertRaisesRegex(Exception, "长按.*真机验收"):
                controller.vision_long_press_relative(500, 500)

        find_window.assert_not_called()

    def test_verified_text_profile_rejects_unverified_characters_before_hardware(self):
        controller = RobotController(
            title="test",
            verified_actions={"input_verified_text"},
        )

        with patch("robot_core.legacy.find_window") as find_window:
            for text in ("Agent123", "中文", ".com"):
                with self.subTest(text=text), self.assertRaisesRegex(
                    Exception,
                    "小写英文字母",
                ):
                    controller.vision_type_text(text)

        find_window.assert_not_called()

    def test_verified_text_profile_uses_calibrated_qwerty_letter_path(self):
        controller = RobotController(
            title="test",
            verified_actions={"input_verified_text"},
        )

        with patch.object(controller, "vision_type_pinyin") as type_pinyin:
            controller.vision_type_text("agent")

        type_pinyin.assert_called_once_with("agent", "agent")

    def test_verified_text_profile_requires_empty_focused_qwerty_scene(self):
        controller = RobotController(
            title="test",
            verified_actions={"input_verified_text"},
        )
        valid = {
            "focused": True,
            "value": "",
            "keyboard_layout": "qwerty",
            "keyboard_input_mode": "direct_latin",
        }
        controller.validate_verified_text("agent", valid)
        cases = (
            ({**valid, "value": "old"}, "空输入框"),
            ({**valid, "keyboard_layout": "symbol"}, "QWERTY"),
            ({**valid, "keyboard_input_mode": "chinese_pinyin"}, "direct_latin"),
            ({key: value for key, value in valid.items() if key != "keyboard_input_mode"}, "direct_latin"),
            ({**valid, "focused": False}, "聚焦"),
        )
        for states, message in cases:
            with self.subTest(states=states), self.assertRaisesRegex(
                Exception,
                message,
            ):
                controller.validate_verified_text("agent", states)

    def test_every_unverified_physical_primitive_fails_before_hardware_access(self):
        controller = RobotController(title="test", verified_actions=set())

        operations = (
            ("点击", lambda: controller.vision_tap_relative(500, 500)),
            ("主页导航", controller.vision_android_home),
            ("滑动", controller.vision_swipe_up),
            ("返回", controller.vision_android_back),
            ("输入", lambda: controller.vision_type_text("草稿")),
            ("拼音输入", lambda: controller.vision_type_pinyin("草稿", "caogao")),
            ("退格清空", lambda: controller.vision_clear_text(delete_count=2)),
            ("长按", lambda: controller.vision_long_press_relative(500, 500)),
            ("拖动", lambda: controller.vision_drag_relative(100, 100, 900, 900)),
        )
        with patch("robot_core.legacy.find_window") as find_window:
            for label, operation in operations:
                with self.subTest(label=label), self.assertRaisesRegex(
                    Exception,
                    "尚未完成|尚未验证",
                ):
                    operation()

        find_window.assert_not_called()

    def test_dismiss_capability_cannot_authorize_an_arbitrary_tap(self):
        controller = RobotController(
            title="test",
            verified_actions={"dismiss_overlay"},
        )

        with patch("robot_core.legacy.find_window") as find_window:
            with self.assertRaisesRegex(Exception, "点击.*真机验收"):
                controller.vision_tap_relative(500, 500)

        find_window.assert_not_called()

    def test_dismiss_uses_its_own_verified_physical_entry(self):
        controller = RobotController(
            title="test",
            verified_actions={"dismiss_overlay"},
        )

        with patch.object(
            controller,
            "_vision_press_relative",
            return_value=(270, 480),
        ) as press:
            point = controller.vision_dismiss_overlay_relative(500, 500)

        self.assertEqual(point, (270, 480))
        press.assert_called_once()

    def test_deprecated_text_workflows_fail_before_hardware(self):
        controller = RobotController(
            title="test",
            verified_actions={"tap_semantic"},
        )

        with patch("robot_core.legacy.find_window") as find_window:
            with self.assertRaisesRegex(Exception, "已废弃的 App 多步流程入口已禁用"):
                controller.comment_current_douyin({"text": "草稿"})
            with self.assertRaisesRegex(Exception, "已废弃的 App 多步流程入口已禁用"):
                controller.send_wechat_text({"text": "草稿"})

        find_window.assert_not_called()

    def test_verified_drag_uses_two_calibrated_points_once(self):
        controller = RobotController(
            title="test",
            verified_actions={"drag"},
        )
        frame = Image.new("RGB", (540, 960), "white")

        with (
            patch("robot_core.legacy.find_window", return_value=(123, "test")),
            patch.object(controller, "_capture_phone", return_value=frame),
            patch.object(controller, "_checkpoint"),
            patch(
                "tap_calibration.corrected_grid_point",
                side_effect=[(100.0, 200.0), (700.0, 800.0)],
            ),
            patch("robot_core.legacy.drag_client_path") as drag,
            patch("robot_core.legacy.move_cursor_outside_camera") as move_out,
        ):
            result = controller.vision_drag_relative(100, 200, 700, 800)

        self.assertEqual(result, ((54, 192), (377, 767)))
        drag.assert_called_once_with(123, (54, 192), (377, 767))
        move_out.assert_called_once_with(123)

    def test_system_navigation_reveal_is_independent_and_default_disabled(self):
        controller = RobotController(title="test")
        self.assertFalse(
            controller.hardware_capabilities()["reveal_system_navigation"]
        )
        controller = RobotController(
            title="test",
            verified_actions={"swipe", "drag"},
        )
        with (
            patch("robot_core.legacy.find_window") as find_window,
            patch("robot_core.legacy.drag_client_path") as drag,
        ):
            with self.assertRaisesRegex(Exception, "系统边缘唤出导航栏.*真机验收"):
                controller.vision_reveal_system_navigation()
        find_window.assert_not_called()
        drag.assert_not_called()

    def test_verified_system_navigation_reveal_uses_one_local_path(self):
        controller = RobotController(
            title="test",
            verified_actions={"reveal_system_navigation"},
        )
        frame = Image.new("RGB", (810, 1440), "white")
        derived = {
            "action": "reveal_system_navigation",
            "edge": "bottom",
            "frame_size": [810, 1440],
            "dom_path": [[0.504, 0.986], [0.505, 0.700]],
            "requested_grid": [[118, 500], [350, 500]],
            "corrected_grid": [[92, 495], [333, 495]],
        }
        with (
            patch("robot_core.legacy.find_window", return_value=(123, "test")),
            patch.object(controller, "_capture_phone", return_value=frame),
            patch.object(controller, "_checkpoint"),
            patch(
                "tap_calibration.reveal_system_navigation_path",
                return_value=derived,
            ) as derive,
            patch("robot_core.legacy.drag_client_path") as drag,
            patch("robot_core.legacy.move_cursor_outside_camera") as move_out,
        ):
            result = controller.vision_reveal_system_navigation()

        derive.assert_called_once_with((810, 1440), controller.calibration_path)
        drag.assert_called_once_with(123, (74, 712), (269, 712))
        move_out.assert_called_once_with(123)
        self.assertEqual([[74, 712], [269, 712]], result["client_path"])
        self.assertEqual(derived["dom_path"], result["dom_path"])

    def test_system_navigation_reveal_never_retries_failed_drag(self):
        controller = RobotController(
            title="test",
            verified_actions={"reveal_system_navigation"},
        )
        frame = Image.new("RGB", (810, 1440), "white")
        with (
            patch("robot_core.legacy.find_window", return_value=(123, "test")),
            patch.object(controller, "_capture_phone", return_value=frame),
            patch.object(controller, "_checkpoint"),
            patch(
                "tap_calibration.reveal_system_navigation_path",
                return_value={"corrected_grid": [[92, 495], [333, 495]]},
            ),
            patch(
                "robot_core.legacy.drag_client_path",
                side_effect=RuntimeError("drag failed"),
            ) as drag,
        ):
            with self.assertRaisesRegex(RuntimeError, "drag failed"):
                controller.vision_reveal_system_navigation()
        drag.assert_called_once()

    def test_system_navigation_reveal_calibration_failure_is_zero_action(self):
        controller = RobotController(
            title="test",
            verified_actions={"reveal_system_navigation"},
        )
        frame = Image.new("RGB", (810, 1440), "white")
        with (
            patch("robot_core.legacy.find_window", return_value=(123, "test")),
            patch.object(controller, "_capture_phone", return_value=frame),
            patch(
                "tap_calibration.reveal_system_navigation_path",
                side_effect=RuntimeError("invalid calibration"),
            ),
            patch("robot_core.legacy.drag_client_path") as drag,
        ):
            with self.assertRaisesRegex(RuntimeError, "invalid calibration"):
                controller.vision_reveal_system_navigation()
        drag.assert_not_called()

    def test_mock_system_navigation_reveal_records_semantic_evidence(self):
        controller = MockRobotController(
            verified_actions={"reveal_system_navigation"}
        )
        result = controller.vision_reveal_system_navigation()

        self.assertEqual("reveal_system_navigation", result["action"])
        self.assertEqual("bottom", result["edge"])
        self.assertEqual(1, len(controller.executions))

TEST_NUMERIC_GRID_LAYOUT = {
    "type": "numeric_grid",
    "anchors": {
        "1": [304, 744],
        "3": [685, 744],
        "7": [304, 839],
        "9": [685, 839],
        "backspace": [850, 698],
    },
}


class LowLevelInputTests(unittest.TestCase):
    @staticmethod
    def _synthetic_douyin_heart(color: str) -> Image.Image:
        image = Image.new("RGB", (540, 960), "black")
        draw = ImageDraw.Draw(image)
        center_x = round((image.width - 1) * 0.86)
        center_y = round((image.height - 1) * 0.49)
        draw.ellipse(
            (
                center_x - 23,
                center_y - 23,
                center_x + 23,
                center_y + 23,
            ),
            fill=color,
        )
        return image

    def test_fixed_douyin_heart_roi_distinguishes_red_and_white(self) -> None:
        red = self._synthetic_douyin_heart((235, 45, 100))
        white = self._synthetic_douyin_heart((235, 235, 235))
        self.assertEqual(classify_douyin_heart_roi(red)["state"], "liked")
        self.assertEqual(classify_douyin_heart_roi(white)["state"], "unliked")

    def test_douyin_heart_transition_survives_bright_video_background(self) -> None:
        before = Image.new("RGB", (540, 960), "white")
        after = before.copy()
        draw = ImageDraw.Draw(after)
        center_x = round((after.width - 1) * 0.86)
        center_y = round((after.height - 1) * 0.514)
        draw.ellipse(
            (
                center_x - 17,
                center_y - 22,
                center_x + 17,
                center_y + 22,
            ),
            fill=(235, 45, 100),
        )
        before_metrics = classify_douyin_heart_roi(before)
        after_metrics = classify_douyin_heart_roi(after)
        self.assertEqual(before_metrics["state"], "unliked")
        self.assertEqual(after_metrics["state"], "liked")
        self.assertTrue(
            douyin_heart_became_liked(before_metrics, after_metrics)
        )

    def test_douyin_heart_location_tracks_shifted_layout(self) -> None:
        image = Image.new("RGB", (540, 960), "black")
        draw = ImageDraw.Draw(image)
        draw.ellipse((446, 424, 484, 459), fill=(235, 235, 235))
        located = locate_douyin_heart(image)
        self.assertTrue(located["trusted"])
        self.assertEqual(located["state"], "unliked")
        self.assertAlmostEqual(located["center"][0], 863, delta=4)
        self.assertAlmostEqual(located["center"][1], 460, delta=4)

    def test_continuous_like_uses_local_heart_and_swipes_after_red(self) -> None:
        class HeartSequenceController(MockRobotController):
            def __init__(self, output_dir: Path, frames: list[Image.Image]) -> None:
                super().__init__()
                self.output_dir = output_dir
                self.frames = frames
                self.frame_index = 0

            def _sleep(self, seconds: float) -> None:
                del seconds
                self._checkpoint()

            def _new_run_dir(self, operation: str) -> Path:
                del operation
                path = self.output_dir / "agent_run"
                path.mkdir()
                return path

            def vision_capture(self) -> Image.Image:
                return self.frames[self.frame_index].copy()

            def vision_tap_relative(self, x: int, y: int) -> tuple[int, int]:
                result = super().vision_tap_relative(x, y)
                self.frame_index += 1
                return result

            def vision_swipe_up(self) -> None:
                super().vision_swipe_up()
                self.frame_index += 1

        white = self._synthetic_douyin_heart((235, 235, 235))
        red = self._synthetic_douyin_heart((235, 45, 100))
        # Page 1: white -> tap -> red. Page 2 is already red and must be
        # skipped without a tap. Page 3: white -> tap -> red.
        frames = [white, red, red, white, red]
        provider = ScriptedVisionProvider(
            [
                VisionDecision(
                    screen_type="video",
                    action="wait",
                    confidence=0.98,
                    reason="普通视频页",
                )
                for _ in frames
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            controller = HeartSequenceController(Path(directory), frames)
            result = VisionAgentRunner(controller, provider).execute(
                {
                    "goal": "给接下来的三条视频点赞",
                    "allowed_texts": [],
                    "task_mode": "operate",
                }
            )
            self.assertEqual(
                [item["action"] for item in controller.executions],
                ["tap", "swipe_up", "swipe_up", "tap"],
            )
            self.assertEqual(result["controller_metrics"]["completed_pages"], 3)
            self.assertEqual(result["controller_metrics"]["verified_likes"], 2)
            self.assertEqual(
                result["controller_metrics"]["skipped_already_liked"],
                1,
            )
            self.assertEqual(result["outcome"], "completed")

    def test_continuous_like_allows_three_dynamic_attempts_then_stops(self) -> None:
        class NoTransitionController(MockRobotController):
            def __init__(self, output_dir: Path, frame: Image.Image) -> None:
                super().__init__()
                self.output_dir = output_dir
                self.frame = frame

            def _sleep(self, seconds: float) -> None:
                del seconds
                self._checkpoint()

            def _new_run_dir(self, operation: str) -> Path:
                del operation
                path = self.output_dir / "agent_run"
                path.mkdir()
                return path

            def vision_capture(self) -> Image.Image:
                return self.frame.copy()

        white = self._synthetic_douyin_heart((235, 235, 235))
        provider = ScriptedVisionProvider(
            [
                VisionDecision("video", "wait", 0.98, "普通视频页"),
                VisionDecision("video", "wait", 0.98, "普通视频页"),
                VisionDecision("video", "wait", 0.98, "普通视频页"),
                VisionDecision("video", "wait", 0.98, "普通视频页"),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            controller = NoTransitionController(Path(directory), white)
            with self.assertRaisesRegex(
                VisionAgentError,
                "补点后仍失败",
            ):
                VisionAgentRunner(controller, provider).execute(
                    {
                        "goal": "给接下来的一条视频点赞",
                        "allowed_texts": [],
                        "task_mode": "operate",
                    }
                )
            self.assertEqual(
                [item["action"] for item in controller.executions],
                ["tap", "tap", "tap"],
            )
            self.assertEqual(
                controller.executions[0]["coordinate"],
                controller.executions[1]["coordinate"],
            )
            self.assertEqual(
                controller.executions[1]["coordinate"],
                controller.executions[2]["coordinate"],
            )

    def test_input_structure_matches_windows_native_size(self) -> None:
        expected_size = 40 if ctypes.sizeof(ctypes.c_void_p) == 8 else 28
        self.assertEqual(ctypes.sizeof(robot_gui_poc.INPUT), expected_size)

    def test_unicode_digit_input_bypasses_keyboard_layout(self) -> None:
        class FakeUser32:
            def __init__(self) -> None:
                self.events: list[tuple[int, int, int]] = []

            def SendInput(self, count, events, _size) -> int:
                self.events = [
                    (
                        int(events[index].type),
                        int(events[index].ki.wScan),
                        int(events[index].ki.dwFlags),
                    )
                    for index in range(count)
                ]
                return count

        fake = FakeUser32()
        with (
            patch.object(robot_gui_poc, "user32", fake),
            patch("robot_gui_poc.time.sleep"),
        ):
            robot_gui_poc.type_unicode_digit("1")

        self.assertEqual(
            fake.events,
            [
                (
                    robot_gui_poc.INPUT_KEYBOARD,
                    ord("1"),
                    robot_gui_poc.KEYEVENTF_UNICODE,
                ),
                (
                    robot_gui_poc.INPUT_KEYBOARD,
                    ord("1"),
                    (
                        robot_gui_poc.KEYEVENTF_UNICODE
                        | robot_gui_poc.KEYEVENTF_KEYUP
                    ),
                ),
            ],
        )

    def test_nihao_maps_to_calibrated_qwerty_key_centers(self) -> None:
        keyboard = DEFAULT_CONFIG["vision_agent"]["pinyin_keyboard"]
        points = [
            qwerty_key_point(540, 960, key, keyboard)
            for key in "nihao"
        ]
        self.assertEqual(
            points,
            [
                (358, 810),
                (381, 676),
                (313, 742),
                (85, 742),
                (427, 676),
            ],
        )

    def test_dynamic_qwerty_anchors_rebuild_letter_centers(self) -> None:
        keyboard = qwerty_keyboard_config_from_anchors(
            TEST_QWERTY_LAYOUT["anchors"]
        )
        points = [
            qwerty_key_point(540, 960, key, keyboard)
            for key in "nihao"
        ]
        self.assertEqual(
            points,
            [
                (358, 810),
                (381, 676),
                (313, 742),
                (85, 742),
                (427, 676),
            ],
        )
        self.assertAlmostEqual(keyboard["backspace_x_ratio"], 0.862)
        self.assertAlmostEqual(keyboard["backspace_y_ratio"], 0.844)

    def test_dynamic_qwerty_normalizes_imprecise_row_endpoints(self) -> None:
        keyboard = qwerty_keyboard_config_from_anchors(
            {
                "q": [124, 698],
                "p": [874, 698],
                "a": [124, 768],
                "l": [874, 768],
                "z": [124, 838],
                "m": [749, 838],
                "backspace": [874, 838],
            }
        )
        points = [
            qwerty_key_point(540, 960, key, keyboard)
            for key in "nihao"
        ]
        self.assertEqual(
            points,
            [
                (359, 804),
                (382, 670),
                (314, 737),
                (89, 737),
                (427, 670),
            ],
        )
        self.assertEqual(keyboard["source"], "vision_anchors_normalized")

    def test_dynamic_qwerty_rejects_non_qwerty_geometry(self) -> None:
        invalid = {
            **TEST_QWERTY_LAYOUT["anchors"],
            "p": [90, 704],
        }
        with self.assertRaisesRegex(Exception, "左右顺序错误"):
            qwerty_keyboard_config_from_anchors(invalid)


class OcrRuntimeTests(unittest.TestCase):
    @staticmethod
    def _line(text: str, left: int, top: int) -> dict:
        words = []
        x = left
        for character in text:
            words.append(
                {
                    "text": character,
                    "left": x,
                    "top": top,
                    "width": 18,
                    "height": 22,
                }
            )
            x += 20
        return {"text": " ".join(text), "words": words}

    def test_split_chinese_words_are_joined_for_matching(self) -> None:
        payload = {
            "lines": [
                {
                    "text": "微 信",
                    "words": [
                        {
                            "text": "微",
                            "left": 100,
                            "top": 200,
                            "width": 20,
                            "height": 24,
                        },
                        {
                            "text": "信",
                            "left": 124,
                            "top": 200,
                            "width": 20,
                            "height": 24,
                        },
                    ],
                }
            ]
        }
        matches = find_text(payload, "微信")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].center, (122, 212))

    def test_file_transfer_list_item_is_not_treated_as_chat_title(self) -> None:
        payload = {
            "lines": [
                self._line("文件传输助手", 130, 114),
                self._line("通讯录", 195, 883),
                self._line("发现", 317, 883),
            ]
        }
        self.assertEqual(
            classify_wechat_page(payload, 540, 960),
            "conversation_list",
        )

    def test_file_transfer_title_without_bottom_navigation_is_target_chat(self) -> None:
        payload = {"lines": [self._line("文件传输助手", 190, 52)]}
        self.assertEqual(
            classify_wechat_page(payload, 540, 960),
            "target_chat",
        )

    def test_obscured_file_transfer_title_suffix_is_recognized(self) -> None:
        payload = {"lines": [{"text": "件 传 输 ' 手", "words": []}]}
        self.assertTrue(classify_obscured_wechat_title(payload))
        self.assertFalse(
            classify_obscured_wechat_title(
                {"lines": [{"text": "张三", "words": []}]}
            )
        )


class RuleAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = RuleAgent()

    def test_wechat_send_text(self) -> None:
        result = self.agent.parse("给文件传输助手发送：测试123")
        self.assertTrue(result["understood"])
        self.assertEqual(result["operation"], "wechat.send_text")
        self.assertEqual(
            result["params"],
            {"chat_name": "文件传输助手", "text": "测试123"},
        )
        self.assertTrue(result["needs_confirmation"])

    def test_douyin_like(self) -> None:
        result = self.agent.parse("给当前视频点赞")
        self.assertEqual(result["operation"], "douyin.batch_interact")
        self.assertEqual(result["params"]["target_count"], 1)
        self.assertTrue(result["params"]["like"])

    def test_douyin_comment(self) -> None:
        result = self.agent.parse("评论：这个视频很有意思")
        self.assertEqual(result["operation"], "douyin.batch_interact")
        self.assertEqual(result["params"]["comment_text"], "这个视频很有意思")

    def test_ambiguous_instruction_is_not_executable(self) -> None:
        result = self.agent.parse("帮我操作一下手机")
        self.assertFalse(result["understood"])
        self.assertTrue(result["needs_clarification"])

    def test_missing_comment_is_not_executable(self) -> None:
        result = self.agent.parse("评论当前视频")
        self.assertFalse(result["understood"])


class HybridAgentStateGraphTests(unittest.TestCase):
    class Provider:
        configured = True

        def __init__(self, response: str) -> None:
            self.response = response

        def chat_json(self, messages, max_tokens):
            self.messages = messages
            self.max_tokens = max_tokens
            return self.response

    def test_deepseek_fallback_can_only_return_structured_operation(self) -> None:
        provider = self.Provider(
            '{"understood":true,"operation":"douyin.search",'
            '"params":{"keyword":"机械臂"},"summary":"搜索机械臂","message":""}'
        )
        result = HybridAgent(provider).parse("帮我在短视频应用里找机械臂")
        self.assertEqual(result["operation"], "douyin.search")
        self.assertEqual(result["params"], {"keyword": "机械臂"})
        self.assertEqual(result["provider"], "deepseek-v4-flash_structured_intent")

    def test_configured_deepseek_parses_common_instruction_before_local_rule(self) -> None:
        provider = self.Provider(
            '{"understood":true,"operation":"douyin.batch_interact",'
            '"params":{"keyword":null,"target_count":1,"like":true,'
            '"comment":false},"summary":"点赞当前视频","message":""}'
        )
        result = HybridAgent(provider).parse("给当前视频点赞")
        self.assertEqual(result["operation"], "douyin.batch_interact")
        self.assertEqual(result["provider"], "deepseek-v4-flash_structured_intent")
        self.assertIn("用户原文", provider.messages[0]["content"])

    def test_unconfigured_deepseek_uses_local_rule_as_explicit_fallback(self) -> None:
        provider = self.Provider("")
        provider.configured = False
        result = HybridAgent(provider).parse("给当前视频点赞")
        self.assertTrue(result["understood"])
        self.assertEqual(result["provider"], "local_rule_fallback")
        self.assertTrue(result["fallback"])

    def test_deepseek_fallback_rejects_action_or_plan_fields(self) -> None:
        provider = self.Provider(
            '{"understood":true,"operation":"douyin.search",'
            '"params":{"keyword":"机械臂"},"summary":"搜索",'
            '"message":"","action":"tap"}'
        )
        result = HybridAgent(provider).parse("帮我在短视频应用里找机械臂")
        self.assertFalse(result["understood"])
        self.assertIn("动作或计划字段", result["message"])

    def test_structured_intent_can_compile_to_dormant_plan_preview(self) -> None:
        provider = self.Provider(
            '{"understood":true,"operation":"douyin.batch_interact",'
            '"params":{"keyword":null,"target_count":3,"like":true,'
            '"comment":false},"summary":"点赞三个视频","message":""}'
        )
        parsed = HybridAgent(provider).parse("打开抖音，点赞三个视频")
        preview = build_generic_plan_preview(parsed, GenericTaskOrchestrator())
        self.assertTrue(preview["compiled"])
        self.assertFalse(preview["execution_enabled"])
        self.assertEqual(preview["goal"]["success_criteria"]["target_count"], 3)
        self.assertEqual(preview["plan"]["goal"]["source_operation"], "douyin.batch_interact")


class DeepSeekIntentProviderTests(unittest.TestCase):
    @staticmethod
    def _response(content: str = '{"understood":false}') -> httpx.Response:
        request = httpx.Request(
            "POST",
            "https://api.deepseek.com/chat/completions",
        )
        return httpx.Response(
            200,
            request=request,
            json={
                "id": "deepseek-test-request",
                "choices": [{"message": {"content": content}}],
                "usage": {"total_tokens": 8},
            },
        )

    def test_uses_v4_flash_json_non_thinking_mode(self) -> None:
        provider = DeepSeekIntentProvider(
            api_key="test-key",
            max_attempts=1,
            retry_base_delay=0,
        )
        with patch(
            "intent_provider.httpx.post",
            return_value=self._response(),
        ) as mocked:
            result = provider.chat_json(
                [{"role": "user", "content": "json: test"}],
                max_tokens=123,
            )
        self.assertEqual(result, '{"understood":false}')
        body = mocked.call_args.kwargs["json"]
        self.assertEqual(body["model"], "deepseek-v4-flash")
        self.assertEqual(body["thinking"], {"type": "disabled"})
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertEqual(body["max_tokens"], 123)
        self.assertEqual(provider.status()["last_request_id"], "deepseek-test-request")

    def test_missing_key_fails_without_network_request(self) -> None:
        provider = DeepSeekIntentProvider(api_key="")
        with patch("intent_provider.httpx.post") as mocked:
            with self.assertRaisesRegex(IntentProviderError, "DEEPSEEK_API_KEY"):
                provider.chat_json([{"role": "user", "content": "json"}])
        mocked.assert_not_called()

    def test_uses_windows_user_environment_when_process_env_is_stale(self) -> None:
        with patch.dict("intent_provider.os.environ", {}, clear=True), patch(
            "intent_provider._read_windows_user_environment",
            return_value="registry-key",
        ):
            provider = DeepSeekIntentProvider()
        self.assertTrue(provider.configured)
        self.assertEqual(provider.api_key, "registry-key")


class FrozenExecutionPlanTests(unittest.TestCase):
    class FastPlanController(MockRobotController):
        def __init__(self, output_dir: Path) -> None:
            super().__init__()
            self.output_dir = output_dir

        def _sleep(self, seconds: float) -> None:
            del seconds
            self._checkpoint()

        def _new_run_dir(self, operation: str) -> Path:
            del operation
            path = self.output_dir / "agent_run"
            path.mkdir()
            return path

    @staticmethod
    def _wechat_plan() -> list[dict[str, object]]:
        return [
            {
                "id": "step_1",
                "intent": "open_app",
                "label": "打开微信",
                "app_id": "wechat",
                "target": "微信",
                "text_ref": None,
                "count": 1,
                "checkpoint": "画面显示微信",
            },
            {
                "id": "step_2",
                "intent": "focus_input",
                "label": "聚焦输入框",
                "app_id": "wechat",
                "target": "聊天输入框",
                "text_ref": None,
                "count": 1,
                "checkpoint": "键盘完整显示",
            },
            {
                "id": "step_3",
                "intent": "enter_text",
                "label": "输入你好",
                "app_id": "wechat",
                "target": "聊天输入框",
                "text_ref": 0,
                "count": 1,
                "checkpoint": "输入框逐字显示你好",
            },
        ]

    def test_valid_plan_is_normalized_and_frozen(self) -> None:
        plan = validate_execution_plan(
            self._wechat_plan(),
            goal="打开微信并输入你好，但不要发送",
            allowed_texts=["你好"],
        )
        self.assertEqual([step["id"] for step in plan], ["step_1", "step_2", "step_3"])
        self.assertEqual(plan[-1]["text_ref"], 0)

    def test_known_chinese_app_name_is_normalized_locally(self) -> None:
        raw = self._wechat_plan()
        for step in raw:
            step["app_id"] = "微信"
        plan = validate_execution_plan(
            raw,
            goal="打开微信并输入你好，但不要发送",
            allowed_texts=["你好"],
        )
        self.assertEqual({step["app_id"] for step in plan}, {"wechat"})

    def test_missing_navigation_targets_are_inferred_without_coordinates(self) -> None:
        raw = self._wechat_plan()[:1]
        raw[0]["target"] = None
        raw.append(
            {
                "id": "step_2",
                "intent": "open_target",
                "label": "进入文件传输助手",
                "app_id": "wechat",
                "target": None,
                "text_ref": None,
                "count": 1,
                "checkpoint": "聊天标题显示文件传输助手",
            }
        )
        plan = validate_execution_plan(
            raw,
            goal="打开微信并进入文件传输助手",
            allowed_texts=[],
        )
        self.assertEqual(plan[0]["target"], "微信")
        self.assertEqual(plan[1]["target"], "文件传输助手")

    def test_terminal_model_stop_marker_is_removed(self) -> None:
        raw = self._wechat_plan()[:1]
        raw.append(
            {
                "id": "step_2",
                "intent": "stop",
                "label": "停止",
                "app_id": "wechat",
                "target": None,
                "text_ref": None,
                "count": 1,
                "checkpoint": "任务停止",
            }
        )
        plan = validate_execution_plan(
            raw,
            goal="打开微信然后停止",
            allowed_texts=[],
        )
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0]["intent"], "open_app")

    def test_nonterminal_model_stop_marker_is_rejected(self) -> None:
        raw = self._wechat_plan()[:2]
        raw[0] = {
            "id": "step_1",
            "intent": "stop",
            "label": "停止",
            "app_id": "wechat",
            "target": None,
            "text_ref": None,
            "count": 1,
            "checkpoint": "任务停止",
        }
        with self.assertRaisesRegex(VisionAgentError, "未授权意图"):
            validate_execution_plan(
                raw,
                goal="停止后继续聚焦输入框",
                allowed_texts=[],
            )

    def test_submit_is_rejected_when_user_forbids_sending(self) -> None:
        raw = self._wechat_plan()
        raw.append(
            {
                "id": "step_4",
                "intent": "submit",
                "label": "发送消息",
                "app_id": "wechat",
                "target": "发送按钮",
                "text_ref": None,
                "count": 1,
                "checkpoint": "出现新的本人消息气泡",
            }
        )
        with self.assertRaisesRegex(VisionAgentError, "不得包含 submit"):
            validate_execution_plan(
                raw,
                goal="打开微信并输入你好，不要发送",
                allowed_texts=["你好"],
            )

    def test_text_reference_must_point_to_user_whitelist(self) -> None:
        raw = self._wechat_plan()
        raw[-1]["text_ref"] = 1
        with self.assertRaisesRegex(VisionAgentError, "文字白名单"):
            validate_execution_plan(
                raw,
                goal="输入你好",
                allowed_texts=["你好"],
            )

    def test_one_plan_cannot_cross_apps(self) -> None:
        raw = self._wechat_plan()
        raw[-1]["app_id"] = "douyin"
        with self.assertRaisesRegex(VisionAgentError, "只能操作一个 App"):
            validate_execution_plan(
                raw,
                goal="输入你好",
                allowed_texts=["你好"],
            )

    def test_social_count_cannot_be_changed_by_model(self) -> None:
        raw = [
            {
                "id": "step_1",
                "intent": "like_current",
                "label": "点赞接下来三个视频",
                "app_id": "douyin",
                "target": "当前视频爱心",
                "text_ref": None,
                "count": 2,
                "checkpoint": "三个视频都显示已点赞",
            }
        ]
        with self.assertRaisesRegex(VisionAgentError, "原文数量不一致"):
            validate_execution_plan(
                raw,
                goal="给接下来的三个视频点赞",
                allowed_texts=[],
            )

    def test_enter_text_checkpoint_needs_exact_input_roi_text(self) -> None:
        step = self._wechat_plan()[-1]
        exact = VisionDecision(
            screen_type="keyboard",
            action="wait",
            confidence=0.95,
            reason="底部输入框逐字显示你好",
            observed_input_text="你好",
            plan_step_id="step_3",
            checkpoint_met=True,
        )
        wrong = VisionDecision(
            screen_type="keyboard",
            action="wait",
            confidence=0.95,
            reason="聊天气泡中看见你好",
            observed_input_text="你号",
            plan_step_id="step_3",
            checkpoint_met=True,
        )
        self.assertTrue(plan_checkpoint_is_proven(step, exact, ["你好"]))
        self.assertFalse(plan_checkpoint_is_proven(step, wrong, ["你好"]))

    def test_deep_restored_chat_can_resume_at_enter_text(self) -> None:
        plan = [
            {
                "id": "step_1",
                "intent": "open_app",
                "label": "打开微信",
                "app_id": "wechat",
                "target": "微信",
                "text_ref": None,
                "count": 1,
                "checkpoint": "微信主界面清晰可见",
            },
            {
                "id": "step_2",
                "intent": "open_target",
                "label": "进入文件传输助手",
                "app_id": "wechat",
                "target": "文件传输助手",
                "text_ref": None,
                "count": 1,
                "checkpoint": "标题显示文件传输助手",
            },
            {
                "id": "step_3",
                "intent": "focus_input",
                "label": "聚焦输入框",
                "app_id": "wechat",
                "target": None,
                "text_ref": None,
                "count": 1,
                "checkpoint": "键盘清晰可见",
            },
            {
                "id": "step_4",
                "intent": "enter_text",
                "label": "输入你好",
                "app_id": "wechat",
                "target": None,
                "text_ref": 0,
                "count": 1,
                "checkpoint": "输入框显示你好",
            },
        ]
        restored = VisionDecision(
            screen_type="chat",
            action="type_pinyin",
            confidence=0.98,
            reason="底部输入框已聚焦且键盘清晰可见",
            page_title="文件传输助手",
            target="底部输入框",
            text="你好",
            pinyin="nihao",
            keyboard_layout=TEST_QWERTY_LAYOUT,
            plan_step_id="step_1",
        )
        self.assertEqual(
            infer_safe_plan_resume_cursor(plan, 0, restored, ["你好"]),
            3,
        )

    def test_deep_resume_requires_named_target_evidence(self) -> None:
        plan = self._wechat_plan()
        unknown_chat = VisionDecision(
            screen_type="chat",
            action="type_pinyin",
            confidence=0.98,
            reason="某个聊天页的底部输入框已聚焦且键盘清晰可见",
            target="底部输入框",
            text="你好",
            pinyin="nihao",
            keyboard_layout=TEST_QWERTY_LAYOUT,
            plan_step_id="step_1",
        )
        self.assertEqual(
            infer_safe_plan_resume_cursor(plan, 0, unknown_chat, ["你好"]),
            0,
        )

    def test_verify_result_checkpoint_requires_named_target_evidence(self) -> None:
        step = {
            "id": "step_1",
            "intent": "verify_result",
            "label": "确认文件传输助手聊天页",
            "app_id": "wechat",
            "target": "文件传输助手",
            "text_ref": None,
            "count": 1,
            "checkpoint": "聊天页顶部标题准确显示文件传输助手",
        }
        exact = VisionDecision(
            screen_type="chat",
            action="finish",
            confidence=0.98,
            reason="聊天页顶部标题准确显示文件传输助手",
            success=True,
            plan_step_id="step_1",
            checkpoint_met=True,
        )
        wrong_page = VisionDecision(
            screen_type="chat",
            action="finish",
            confidence=0.98,
            reason="当前是普通微信聊天页面",
            success=True,
            plan_step_id="step_1",
            checkpoint_met=True,
        )
        self.assertTrue(plan_checkpoint_is_proven(step, exact, []))
        self.assertFalse(plan_checkpoint_is_proven(step, wrong_page, []))

    def test_runner_advances_only_after_each_checkpoint(self) -> None:
        plan = [
            {
                "id": "step_1",
                "intent": "open_app",
                "label": "启动微信",
                "app_id": "wechat",
                "target": "微信",
                "text_ref": None,
                "count": 1,
                "checkpoint": "微信主界面可见",
            },
            {
                "id": "step_2",
                "intent": "open_target",
                "label": "进入文件传输助手",
                "app_id": "wechat",
                "target": "文件传输助手",
                "text_ref": None,
                "count": 1,
                "checkpoint": "聊天标题显示文件传输助手",
            },
        ]
        decisions = [
            VisionDecision(
                screen_type="chat_list",
                action="wait",
                confidence=0.98,
                reason="微信主界面已经显示",
                target="微信",
                plan_step_id="step_1",
                checkpoint_met=True,
            ),
            VisionDecision(
                screen_type="chat",
                action="finish",
                confidence=0.98,
                reason="聊天标题显示文件传输助手",
                target="文件传输助手",
                success=True,
                plan_step_id="step_2",
                checkpoint_met=True,
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            controller = self.FastPlanController(Path(directory))
            result = VisionAgentRunner(
                controller,
                ScriptedVisionProvider(decisions),
            ).execute(
                {
                    "goal": "打开微信并进入文件传输助手，不输入不发送",
                    "allowed_texts": [],
                    "execution_plan": plan,
                    "task_mode": "operate",
                }
            )
        self.assertEqual(result["outcome"], "completed")
        self.assertEqual(result["plan_cursor"], 1)
        self.assertEqual(
            [item["completed_plan_step"]["id"] for item in result["steps"]],
            ["step_1", "step_2"],
        )
        self.assertEqual(controller.executions, [])

    def test_runner_fast_forwards_restored_target_and_focus_only(self) -> None:
        plan = [
            {
                "id": "step_1",
                "intent": "open_app",
                "label": "打开微信",
                "app_id": "wechat",
                "target": "微信",
                "text_ref": None,
                "count": 1,
                "checkpoint": "微信主界面可见",
            },
            {
                "id": "step_2",
                "intent": "open_target",
                "label": "进入文件传输助手",
                "app_id": "wechat",
                "target": "文件传输助手",
                "text_ref": None,
                "count": 1,
                "checkpoint": "标题显示文件传输助手",
            },
            {
                "id": "step_3",
                "intent": "focus_input",
                "label": "聚焦输入框",
                "app_id": "wechat",
                "target": None,
                "text_ref": None,
                "count": 1,
                "checkpoint": "键盘可见",
            },
            {
                "id": "step_4",
                "intent": "enter_text",
                "label": "输入hello",
                "app_id": "wechat",
                "target": None,
                "text_ref": 0,
                "count": 1,
                "checkpoint": "输入框显示hello",
            },
        ]
        decisions = [
            VisionDecision(
                screen_type="chat",
                action="type_text",
                confidence=0.98,
                reason="输入框已聚焦且键盘清晰可见",
                page_title="文件传输助手",
                target="底部输入框",
                text="hello",
                plan_step_id="step_1",
            ),
            VisionDecision(
                screen_type="keyboard",
                action="wait",
                confidence=0.98,
                reason="只看底部输入框，逐字显示hello",
                observed_input_text="hello",
                plan_step_id="step_4",
                checkpoint_met=True,
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            controller = self.FastPlanController(Path(directory))
            result = VisionAgentRunner(
                controller,
                ScriptedVisionProvider(decisions),
            ).execute(
                {
                    "goal": "打开微信进入文件传输助手并输入hello但不要发送",
                    "allowed_texts": ["hello"],
                    "execution_plan": plan,
                    "task_mode": "operate",
                }
            )
        self.assertEqual(result["outcome"], "completed")
        self.assertEqual(result["plan_cursor"], 3)
        self.assertEqual(
            [item["id"] for item in result["steps"][0]["fast_forwarded_plan_steps"]],
            ["step_1", "step_2", "step_3"],
        )
        self.assertEqual(controller.executions[0]["action"], "type_text")

    def test_restored_named_chat_resumes_at_empty_check_before_typing(self) -> None:
        def step(
            index: int,
            intent: str,
            label: str,
            target: str | None = None,
            text_ref: int | None = None,
            checkpoint: str = "画面确认",
        ) -> dict[str, object]:
            return {
                "id": f"step_{index}",
                "intent": intent,
                "label": label,
                "app_id": "wechat",
                "target": target,
                "text_ref": text_ref,
                "count": 1,
                "checkpoint": checkpoint,
            }

        plan = [
            step(1, "open_app", "打开微信", "微信"),
            step(2, "open_target", "打开微信搜索", "微信搜索"),
            step(3, "focus_input", "聚焦搜索框", "微信搜索框"),
            step(4, "enter_text", "输入聊天名", "微信搜索框", 0),
            step(5, "submit", "提交搜索", "搜索"),
            step(6, "open_target", "打开文件传输助手", "文件传输助手"),
            step(
                7,
                "focus_input",
                "聚焦聊天输入框",
                "聊天输入框",
                checkpoint="底部输入框为空且键盘完整清晰",
            ),
            step(8, "enter_text", "输入你好", "聊天输入框", 1),
        ]
        decision = VisionDecision(
            screen_type="chat",
            action="type_pinyin",
            confidence=0.99,
            reason="已经恢复到文件传输助手，输入框聚焦且键盘清晰可见",
            page_title="文件传输助手",
            target="底部输入框",
            text="你好",
            pinyin="nihao",
            keyboard_layout=TEST_QWERTY_LAYOUT,
        )
        self.assertEqual(
            infer_safe_plan_resume_cursor(plan, 0, decision, ["文件传输", "你好"]),
            6,
        )
        decision = replace(
            decision,
            input_is_empty=True,
            reason=decision.reason + "，底部输入框为空",
        )
        self.assertEqual(
            infer_safe_plan_resume_cursor(plan, 0, decision, ["文件传输", "你好"]),
            7,
        )

    def test_runner_blocks_send_during_enter_text_step(self) -> None:
        plan = [self._wechat_plan()[-1]]
        plan[0]["id"] = "step_1"
        decision = VisionDecision(
            screen_type="keyboard",
            action="tap",
            confidence=0.98,
            reason="错误地尝试提前发送",
            target="发送按钮",
            coordinate=(900, 600),
            plan_step_id="step_1",
        )
        with tempfile.TemporaryDirectory() as directory:
            controller = self.FastPlanController(Path(directory))
            with self.assertRaisesRegex(VisionAgentError, "禁止提前点击发送"):
                VisionAgentRunner(
                    controller,
                    ScriptedVisionProvider([decision]),
                ).execute(
                    {
                        "goal": "输入你好但不要发送",
                        "allowed_texts": ["你好"],
                        "execution_plan": plan,
                        "task_mode": "operate",
                    }
                )
        self.assertEqual(controller.executions, [])


class VisionAgentTests(unittest.TestCase):
    @staticmethod
    def _successful_qwen_response() -> httpx.Response:
        request = httpx.Request(
            "POST",
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        )
        return httpx.Response(
            200,
            request=request,
            json={
                "id": "mock-request",
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"understood":true,"summary":"打开微信",'
                                '"task_mode":"operate","expected_result":null,'
                                '"allowed_texts":[],"message":"",'
                                '"execution_plan":[{"id":"step_1",'
                                '"intent":"open_app","label":"打开微信",'
                                '"app_id":"wechat","target":"微信",'
                                '"text_ref":null,"count":1,'
                                '"checkpoint":"画面显示微信"}]}'
                            )
                        }
                    }
                ],
                "usage": {"total_tokens": 12},
            },
        )

    def test_transient_remote_disconnect_is_retried(self) -> None:
        provider = DashScopeVisionProvider(
            api_key="test-key",
            max_attempts=3,
            retry_base_delay=0,
        )
        with patch(
            "vision_agent.httpx.post",
            side_effect=[
                httpx.RemoteProtocolError(
                    "Server disconnected without sending a response."
                ),
                self._successful_qwen_response(),
            ],
        ) as request:
            parsed = provider.parse_goal("打开微信")
        self.assertTrue(parsed["understood"])
        self.assertEqual(request.call_count, 2)
        self.assertEqual(provider.status()["last_network_attempts"], 2)

    def test_non_transient_http_error_is_not_retried(self) -> None:
        provider = DashScopeVisionProvider(
            api_key="test-key",
            max_attempts=3,
            retry_base_delay=0,
        )
        request = httpx.Request(
            "POST",
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        )
        response = httpx.Response(
            400,
            request=request,
            text='{"message":"bad request"}',
        )
        with patch(
            "vision_agent.httpx.post",
            return_value=response,
        ) as mocked:
            with self.assertRaisesRegex(VisionAgentError, "HTTP 400"):
                provider.parse_goal("打开微信")
        self.assertEqual(mocked.call_count, 1)

    def test_missing_tap_coordinate_gets_one_format_only_repair(self) -> None:
        provider = DashScopeVisionProvider(
            api_key="test-key",
            max_attempts=1,
            retry_base_delay=0,
        )
        frames = [Image.new("RGB", (540, 960), "white") for _ in range(4)]
        malformed = (
            '{"screen_type":"chat_list","action":"tap",'
            '"confidence":0.96,"reason":"搜索按钮清晰可见",'
            '"target":"微信搜索","coordinate":null,'
            '"wait_seconds":1,"success":false}'
        )
        repaired = (
            '{"screen_type":"chat_list","action":"tap",'
            '"confidence":0.96,"reason":"搜索按钮清晰可见",'
            '"target":"微信搜索","coordinate":[910,80],'
            '"wait_seconds":1,"success":false}'
        )
        with patch.object(provider, "_chat", side_effect=[malformed, repaired]) as chat:
            decision = provider.decide(
                goal="打开微信搜索",
                frames=frames,
                history=[],
                allowed_texts=[],
                task_mode="operate",
                expected_result=None,
            )
        self.assertEqual(chat.call_count, 2)
        self.assertEqual(decision.action, "tap")
        self.assertEqual(decision.coordinate, (910, 80))

    def test_missing_tap_coordinate_repair_may_safely_stop(self) -> None:
        provider = DashScopeVisionProvider(
            api_key="test-key",
            max_attempts=1,
            retry_base_delay=0,
        )
        frames = [Image.new("RGB", (540, 960), "white") for _ in range(4)]
        malformed = (
            '{"screen_type":"unknown","action":"tap",'
            '"confidence":0.80,"reason":"可能存在目标",'
            '"target":"未知","coordinate":null}'
        )
        repaired = (
            '{"screen_type":"unknown","action":"stop",'
            '"confidence":0.95,"reason":"无法安全定位目标",'
            '"target":"未知页面","coordinate":null,'
            '"wait_seconds":1,"success":false}'
        )
        with patch.object(provider, "_chat", side_effect=[malformed, repaired]) as chat:
            decision = provider.decide(
                goal="打开目标",
                frames=frames,
                history=[],
                allowed_texts=[],
                task_mode="operate",
                expected_result=None,
            )
        self.assertEqual(chat.call_count, 2)
        self.assertEqual(decision.action, "stop")
        self.assertIsNone(decision.coordinate)

    def test_risky_goals_are_blocked_but_likes_and_comments_are_allowed(self) -> None:
        self.assertFalse(goal_is_forbidden("连续给10个视频点赞"))
        self.assertFalse(goal_is_forbidden("给接下来的十条视频评论：很好"))
        self.assertTrue(goal_is_forbidden("帮我付款下单"))
        self.assertFalse(goal_is_forbidden("打开微信发送：你好"))
        self.assertEqual(
            requested_social_interaction_count("给接下来的十条视频点赞"),
            10,
        )
        self.assertEqual(
            requested_social_interaction_count("连续评论12个视频：很好"),
            12,
        )
        self.assertEqual(
            requested_social_interaction_count("给下二十五条视频点赞"),
            25,
        )

    def test_relative_tap_requires_coordinate_in_safe_range(self) -> None:
        decision = validate_decision(
            {
                "screen_type": "android_home",
                "action": "tap",
                "confidence": 0.91,
                "coordinate": [250, 750],
                "reason": "点击微信图标",
            }
        )
        self.assertEqual(decision.coordinate, (250, 750))
        with self.assertRaises(VisionAgentError):
            validate_decision(
                {
                    "screen_type": "android_home",
                    "action": "tap",
                    "confidence": 0.91,
                    "coordinate": [1200, 750],
                }
            )

    def test_digit_prefers_local_text_input(self) -> None:
        decision = validate_decision(
            {
                "screen_type": "keyboard",
                "action": "type_text",
                "confidence": 0.98,
                "reason": "输入框已聚焦，使用本地数字输入",
                "target": "评论输入框",
                "text": "1",
            }
        )
        self.assertEqual(decision.text, "1")
        with self.assertRaisesRegex(VisionAgentError, "numeric_grid"):
            validate_decision(
                {
                    "screen_type": "keyboard",
                    "action": "type_symbol",
                    "confidence": 0.98,
                    "reason": "只给了单点，没有数字键盘几何锚点",
                    "target": "数字键1",
                    "coordinate": [312, 745],
                    "text": "1",
                }
            )
        decision = validate_decision(
            {
                "screen_type": "keyboard",
                "action": "type_symbol",
                "confidence": 0.98,
                "reason": "数字键盘中真实看见1键",
                "target": "数字键1",
                "coordinate": [312, 745],
                "text": "1",
                "keyboard_layout": TEST_NUMERIC_GRID_LAYOUT,
            }
        )
        self.assertEqual(decision.text, "1")
        self.assertEqual(decision.coordinate, (312, 745))
        self.assertEqual(decision.keyboard_layout["type"], "numeric_grid")
        self.assertEqual(
            numeric_grid_key_coordinate(decision.keyboard_layout, "1"),
            (304, 698),
        )
        self.assertEqual(
            numeric_grid_key_coordinate(
                decision.keyboard_layout,
                "1",
                (540, 1010),
            ),
            (164, 672),
        )
        self.assertEqual(
            numeric_grid_key_coordinate(
                decision.keyboard_layout,
                "9",
                (540, 1010),
            ),
            (370, 810),
        )
        self.assertEqual(
            numeric_grid_key_coordinate(
                decision.keyboard_layout,
                "0",
                (540, 1010),
            ),
            (270, 876),
        )
        with self.assertRaisesRegex(VisionAgentError, "画面尺寸"):
            numeric_grid_key_coordinate(
                decision.keyboard_layout,
                "1",
                (1920, 1080),
            )
        clearing = validate_decision(
            {
                "screen_type": "keyboard",
                "action": "clear_text",
                "confidence": 0.99,
                "reason": "数字键盘仍可见，输入框中实际显示w",
                "target": "退格键",
                "observed_input_text": "w",
                "delete_count": 1,
                "keyboard_layout": {
                    "type": "generic",
                    "anchors": {"backspace": [865, 810]},
                },
            }
        )
        self.assertEqual(clearing.keyboard_layout["type"], "generic")
        self.assertEqual(
            clearing.keyboard_layout["anchors"]["backspace"],
            [865, 810],
        )

    def test_replay_normalizes_model_numeric_coordinate_to_local_profile(self) -> None:
        decision = VisionDecision(
            screen_type="keyboard",
            action="type_symbol",
            confidence=0.99,
            reason="数字键盘可见",
            target="数字键1",
            coordinate=(306, 698),
            text="1",
            keyboard_layout=TEST_NUMERIC_GRID_LAYOUT,
        )
        normalized = normalize_replay_decision(
            decision,
            {},
            (540, 1010),
        )
        self.assertEqual(normalized.coordinate, (164, 672))

    def test_second_stable_empty_check_finishes_in_controller(self) -> None:
        decision = controller_owned_replay_decision(
            {
                "request": {
                    "history": [
                        {
                            "controller_phase": "verify_empty_after_clear",
                            "stable_empty_checks": 2,
                        }
                    ]
                }
            }
        )
        self.assertIsNotNone(decision)
        self.assertEqual(decision.action, "finish")
        self.assertTrue(decision.success)
        self.assertTrue(decision.input_is_empty)

    def test_numeric_wrong_comment_uses_exact_visible_delete_count(self) -> None:
        decision = VisionDecision(
            screen_type="keyboard",
            action="wait",
            confidence=0.61,
            reason="数字键盘可见，但输入框内容错误",
            observed_input_text=".com",
            input_is_empty=False,
        )
        normalized = normalize_replay_decision(
            decision,
            {
                "request": {
                    "history": [
                        {
                            "controller_phase": "verify_comment_input",
                            "keyboard_profile": "installed_numeric_symbol",
                            "text": "1",
                        }
                    ]
                }
            },
            (540, 1010),
        )
        self.assertEqual(normalized.action, "clear_text")
        self.assertEqual(normalized.observed_input_text, ".com")
        self.assertEqual(normalized.delete_count, 4)
        self.assertEqual(
            normalized.keyboard_layout["anchors"]["backspace"],
            [862, 844],
        )

    def test_historical_send_transition_finishes_before_retry(self) -> None:
        manifest_path = (
            Path(__file__).resolve().parent
            / "evals"
            / "vision_replay"
            / "cases.json"
        )
        manifest = load_manifest(manifest_path)
        case = next(
            item
            for item in manifest["cases"]
            if item["id"] == "conflict_sent_bubble_visible_finish_early"
        )
        decision = controller_owned_replay_decision(
            case,
            manifest_path=manifest_path,
        )
        self.assertIsNotNone(decision)
        self.assertEqual(decision.action, "finish")
        self.assertTrue(decision.success)
        self.assertTrue(decision.sent_message_visible)
        self.assertTrue(decision.input_is_empty)

    def test_douyin_comment_placeholder_is_converted_to_empty_wait(self) -> None:
        decision = validate_decision(
            {
                "screen_type": "keyboard",
                "action": "clear_text",
                "confidence": 0.99,
                "reason": "误以为占位提示是输入内容",
                "target": "退格键",
                "observed_input_text": "爱评论的人，运气不会差",
                "delete_count": 12,
                "keyboard_layout": {
                    "type": "generic",
                    "anchors": {"backspace": [850, 698]},
                },
            }
        )
        self.assertEqual(decision.action, "wait")
        self.assertTrue(decision.input_is_empty)
        self.assertEqual(decision.target, "评论输入框为空")
        self.assertIsNone(decision.delete_count)

    def test_irrelevant_model_fields_are_safely_ignored(self) -> None:
        typing = validate_decision(
            {
                "screen_type": "keyboard",
                "action": "type_text",
                "confidence": 0.95,
                "coordinate": [420, 500],
                "text": "你好",
            }
        )
        self.assertIsNone(typing.coordinate)
        self.assertEqual(typing.text, "你好")

        pinyin_typing = validate_decision(
            {
                "screen_type": "keyboard",
                "action": "type_pinyin",
                "confidence": 0.95,
                "text": "你好",
                "pinyin": "nihao",
                "observed_input_text": "",
                "input_is_empty": True,
                "keyboard_layout": TEST_QWERTY_LAYOUT,
            }
        )
        self.assertIsNone(pinyin_typing.observed_input_text)
        self.assertEqual(pinyin_typing.pinyin, "nihao")

        tapping = validate_decision(
            {
                "screen_type": "chat",
                "action": "tap",
                "confidence": 0.95,
                "coordinate": [420, 500],
                "text": "模型误填的无关文字",
            }
        )
        self.assertEqual(tapping.coordinate, (420, 500))
        self.assertIsNone(tapping.text)

        clearing = validate_decision(
            {
                "screen_type": "keyboard",
                "action": "clear_text",
                "confidence": 0.96,
                "coordinate": [875, 845],
                "observed_input_text": "错误文字",
                "delete_count": 4,
                "keyboard_layout": TEST_QWERTY_LAYOUT,
            }
        )
        self.assertIsNone(clearing.coordinate)
        self.assertIsNone(clearing.text)
        self.assertEqual(clearing.observed_input_text, "错误文字")
        self.assertEqual(clearing.delete_count, 4)

    def test_clear_text_requires_exact_verified_character_count(self) -> None:
        with self.assertRaises(VisionAgentError):
            validate_decision(
                {
                    "screen_type": "keyboard",
                    "action": "clear_text",
                    "confidence": 0.96,
                    "observed_input_text": "你好",
                    "delete_count": 8,
                    "keyboard_layout": TEST_QWERTY_LAYOUT,
                }
            )

        pinyin_clear = validate_decision(
            {
                "screen_type": "keyboard",
                "action": "clear_text",
                "confidence": 0.96,
                "observed_input_text": "n'nihao",
                "delete_count": 6,
                "keyboard_layout": TEST_QWERTY_LAYOUT,
            }
        )
        self.assertEqual(pinyin_clear.delete_count, 6)

        with self.assertRaises(VisionAgentError):
            validate_decision(
                {
                    "screen_type": "keyboard",
                    "action": "clear_text",
                    "confidence": 0.96,
                    "observed_input_text": "n'nihao",
                    "delete_count": 7,
                    "keyboard_layout": TEST_QWERTY_LAYOUT,
                }
            )

        with self.assertRaises(VisionAgentError):
            validate_decision(
                {
                    "screen_type": "keyboard",
                    "action": "clear_text",
                    "confidence": 0.96,
                    "observed_input_text": "",
                    "delete_count": 0,
                    "keyboard_layout": TEST_QWERTY_LAYOUT,
                }
            )

    def test_clear_text_passes_exact_delete_count_to_controller(self) -> None:
        class FastMockRobotController(MockRobotController):
            def __init__(self, output_dir: Path) -> None:
                super().__init__()
                self.output_dir = output_dir

            def _sleep(self, seconds: float) -> None:
                del seconds
                self._checkpoint()

            def _new_run_dir(self, operation: str) -> Path:
                del operation
                path = self.output_dir / "agent_run"
                path.mkdir()
                return path

        with tempfile.TemporaryDirectory() as directory:
            controller = FastMockRobotController(Path(directory))
            provider = ScriptedVisionProvider(
                [
                    VisionDecision(
                        screen_type="keyboard",
                        action="clear_text",
                        confidence=0.98,
                        reason="底部输入框准确显示两个错误字符",
                        observed_input_text="你号",
                        delete_count=2,
                        keyboard_layout=TEST_QWERTY_LAYOUT,
                    ),
                    VisionDecision(
                        screen_type="keyboard",
                        action="finish",
                        confidence=0.98,
                        reason="底部输入框已经为空",
                        input_is_empty=True,
                        success=True,
                    ),
                ]
            )
            result = VisionAgentRunner(controller, provider).execute(
                {
                    "goal": "清空底部输入框，但不要发送",
                    "allowed_texts": [],
                    "task_mode": "test",
                    "expected_result": "底部输入框为空",
                }
            )
            clear_actions = [
                item
                for item in controller.executions
                if item.get("action") == "clear_text"
            ]
            self.assertEqual(len(clear_actions), 1)
            self.assertEqual(clear_actions[0]["delete_count"], 2)
            self.assertEqual(result["outcome"], "passed")

    def test_clear_text_must_confirm_empty_before_retyping(self) -> None:
        class FastMockRobotController(MockRobotController):
            def __init__(self, output_dir: Path) -> None:
                super().__init__()
                self.output_dir = output_dir

            def _sleep(self, seconds: float) -> None:
                del seconds
                self._checkpoint()

            def _new_run_dir(self, operation: str) -> Path:
                del operation
                path = self.output_dir / "agent_run"
                path.mkdir()
                return path

        with tempfile.TemporaryDirectory() as directory:
            controller = FastMockRobotController(Path(directory))
            provider = ScriptedVisionProvider(
                [
                    VisionDecision(
                        screen_type="keyboard",
                        action="clear_text",
                        confidence=0.98,
                        reason="底部输入框准确显示一个错误字符",
                        observed_input_text="错",
                        delete_count=1,
                        keyboard_layout=TEST_QWERTY_LAYOUT,
                    ),
                    VisionDecision(
                        screen_type="keyboard",
                        action="type_pinyin",
                        confidence=0.98,
                        reason="错误地跳过空框确认",
                        target="消息输入框",
                        text="你好",
                        pinyin="nihao",
                        keyboard_layout=TEST_QWERTY_LAYOUT,
                    ),
                ]
            )

            with self.assertRaisesRegex(
                VisionAgentError,
                "退格后尚未确认底部输入框为空",
            ):
                VisionAgentRunner(controller, provider).execute(
                    {
                        "goal": "先清空错误文字，再输入你好但不要发送",
                        "allowed_texts": ["你好"],
                        "task_mode": "test",
                        "expected_result": "输入框准确显示你好且没有发送",
                    }
                )

            self.assertEqual(
                [item["action"] for item in controller.executions],
                ["clear_text"],
            )

    def test_input_phases_append_enlarged_bottom_input_roi(self) -> None:
        class FastMockRobotController(MockRobotController):
            def __init__(self, output_dir: Path) -> None:
                super().__init__()
                self.output_dir = output_dir

            def _sleep(self, seconds: float) -> None:
                del seconds
                self._checkpoint()

            def _new_run_dir(self, operation: str) -> Path:
                del operation
                path = self.output_dir / "agent_run"
                path.mkdir()
                return path

        class RecordingProvider(ScriptedVisionProvider):
            def __init__(self, decisions: list[VisionDecision]) -> None:
                super().__init__(decisions)
                self.frame_sizes: list[list[tuple[int, int]]] = []
                self.histories: list[list[dict[str, object]]] = []

            def decide(self, **kwargs: object) -> VisionDecision:
                frames = kwargs["frames"]
                history = kwargs["history"]
                assert isinstance(frames, list)
                assert isinstance(history, list)
                self.frame_sizes.append([frame.size for frame in frames])
                self.histories.append(list(history))
                return super().decide(**kwargs)

        with tempfile.TemporaryDirectory() as directory:
            controller = FastMockRobotController(Path(directory))
            provider = RecordingProvider(
                [
                    VisionDecision(
                        screen_type="keyboard",
                        action="clear_text",
                        confidence=0.98,
                        reason="底部输入框准确显示一个错误字符",
                        observed_input_text="错",
                        delete_count=1,
                        keyboard_layout=TEST_QWERTY_LAYOUT,
                    ),
                    VisionDecision(
                        screen_type="keyboard",
                        action="wait",
                        confidence=0.98,
                        reason="放大输入框确认已经为空",
                        target="底部输入框为空",
                        input_is_empty=True,
                    ),
                    VisionDecision(
                        screen_type="keyboard",
                        action="type_pinyin",
                        confidence=0.98,
                        reason="标准拼音键盘可见",
                        target="消息输入框",
                        text="你好",
                        pinyin="nihao",
                        keyboard_layout=TEST_QWERTY_LAYOUT,
                    ),
                    VisionDecision(
                        screen_type="chat",
                        action="tap",
                        confidence=0.98,
                        reason="选择准确候选词",
                        target="候选词你好",
                        coordinate=(140, 635),
                        observed_input_text="ni'hao",
                    ),
                    VisionDecision(
                        screen_type="chat",
                        action="finish",
                        confidence=0.98,
                        reason="底部输入框准确显示你好",
                        success=True,
                    ),
                ]
            )

            result = VisionAgentRunner(controller, provider).execute(
                {
                    "goal": "先清空错误文字，再输入你好但不要发送",
                    "allowed_texts": ["你好"],
                    "task_mode": "test",
                    "expected_result": "输入框准确显示你好且没有发送",
                }
            )

            self.assertEqual(result["outcome"], "passed")
            self.assertEqual(provider.frame_sizes[0], [(540, 960)] * 4)
            self.assertEqual(len(provider.frame_sizes[1]), 5)
            self.assertEqual(provider.frame_sizes[1][-1], (1080, 154))
            self.assertEqual(len(provider.frame_sizes[3]), 5)
            self.assertEqual(provider.frame_sizes[3][-1], (1080, 404))
            self.assertEqual(len(provider.frame_sizes[4]), 5)
            self.assertEqual(provider.frame_sizes[4][-1], (1080, 154))
            self.assertTrue(
                any(
                    item.get("controller_phase") == "verify_empty_after_clear"
                    for item in provider.histories[1]
                )
            )

    def test_null_wait_seconds_uses_safe_default(self) -> None:
        decision = validate_decision(
            {
                "screen_type": "chat",
                "action": "wait",
                "confidence": 0.95,
                "reason": "等待画面稳定",
                "wait_seconds": None,
            }
        )
        self.assertEqual(decision.wait_seconds, 1.0)

    def test_observe_discards_transient_truncated_frame(self) -> None:
        class TransientFrameController(MockRobotController):
            def __init__(self) -> None:
                super().__init__()
                self.capture_calls = 0

            def vision_capture(self) -> Image.Image:
                self.capture_calls += 1
                if self.capture_calls == 1:
                    return Image.new("RGB", (280, 150), "black")
                return Image.new("RGB", (540, 960), "white")

            def _sleep(self, seconds: float) -> None:
                del seconds
                self._checkpoint()

        controller = TransientFrameController()
        runner = VisionAgentRunner(
            controller,
            ScriptedVisionProvider([]),
            observation_frames=4,
            observation_seconds=1.5,
        )
        frames = runner._observe()
        self.assertEqual(controller.capture_calls, 5)
        self.assertEqual(len(frames), 4)
        self.assertTrue(all(frame.size == (540, 960) for frame in frames))

    def test_pinyin_input_requires_lowercase_spelling(self) -> None:
        decision = validate_decision(
            {
                "screen_type": "keyboard",
                "action": "type_pinyin",
                "confidence": 0.98,
                "reason": "标准拼音键盘可见",
                "target": "消息输入框",
                "coordinate": None,
                "text": "你好",
                "pinyin": "nihao",
                "keyboard_layout": TEST_QWERTY_LAYOUT,
            }
        )
        self.assertEqual(decision.text, "你好")
        self.assertEqual(decision.pinyin, "nihao")
        with self.assertRaisesRegex(VisionAgentError, "小写英文字母"):
            validate_decision(
                {
                    "screen_type": "keyboard",
                    "action": "type_pinyin",
                    "confidence": 0.98,
                    "text": "你好",
                    "pinyin": "ni hao",
                    "keyboard_layout": TEST_QWERTY_LAYOUT,
                }
            )
        with self.assertRaisesRegex(VisionAgentError, "keyboard_layout"):
            validate_decision(
                {
                    "screen_type": "keyboard",
                    "action": "type_pinyin",
                    "confidence": 0.98,
                    "text": "你好",
                    "pinyin": "nihao",
                }
            )

    def test_type_text_must_equal_confirmed_user_text(self) -> None:
        class FastMockRobotController(MockRobotController):
            def __init__(self, output_dir: Path) -> None:
                super().__init__()
                self.output_dir = output_dir

            def _sleep(self, seconds: float) -> None:
                del seconds
                self._checkpoint()

            def _new_run_dir(self, operation: str) -> Path:
                del operation
                path = self.output_dir / "agent_run"
                path.mkdir()
                return path

        with tempfile.TemporaryDirectory() as directory:
            controller = FastMockRobotController(Path(directory))
            provider = ScriptedVisionProvider(
                [
                    VisionDecision(
                        screen_type="keyboard",
                        action="type_text",
                        confidence=0.95,
                        reason="输入框已聚焦",
                        text="被篡改的文字",
                    )
                ]
            )
            runner = VisionAgentRunner(
                controller,
                provider,
                observation_seconds=1.5,
            )
            with self.assertRaisesRegex(VisionAgentError, "不完全一致"):
                runner.execute(
                    {
                        "goal": "发送：测试123",
                        "allowed_texts": ["测试123"],
                        "task_mode": "operate",
                        "expected_result": None,
                    }
                )
            self.assertFalse(
                any(item.get("action") == "type_text" for item in controller.executions)
            )

    def test_scripted_agent_executes_one_action_per_observation(self) -> None:
        class FastMockRobotController(MockRobotController):
            def __init__(self, output_dir: Path) -> None:
                super().__init__()
                self.output_dir = output_dir

            def _sleep(self, seconds: float) -> None:
                del seconds
                self._checkpoint()

            def _new_run_dir(self, operation: str) -> Path:
                del operation
                path = self.output_dir / "agent_run"
                path.mkdir()
                return path

        with tempfile.TemporaryDirectory() as directory:
            controller = FastMockRobotController(Path(directory))
            provider = ScriptedVisionProvider(
                [
                    VisionDecision(
                        screen_type="android_home",
                        action="tap",
                        confidence=0.96,
                        reason="微信图标清晰",
                        target="微信",
                        coordinate=(300, 600),
                    ),
                    VisionDecision(
                        screen_type="chat",
                        action="finish",
                        confidence=0.93,
                        reason="目标消息已出现",
                        target="发送结果",
                        success=True,
                    ),
                ]
            )
            result = VisionAgentRunner(controller, provider).execute(
                {
                    "goal": "打开微信",
                    "allowed_texts": [],
                    "task_mode": "operate",
                    "expected_result": None,
                }
            )
            self.assertEqual(len(result["steps"]), 2)
            self.assertEqual(controller.executions[0]["action"], "tap")
            self.assertEqual(controller.executions[0]["coordinate"], [300, 600])
            self.assertEqual(len(result["evidence"]), 2)
            self.assertEqual(result["outcome"], "completed")
            self.assertTrue(Path(result["report"]).is_file())

    def test_identical_action_stops_when_screen_does_not_change(self) -> None:
        class FastMockRobotController(MockRobotController):
            def __init__(self, output_dir: Path) -> None:
                super().__init__()
                self.output_dir = output_dir

            def _sleep(self, seconds: float) -> None:
                del seconds
                self._checkpoint()

            def _new_run_dir(self, operation: str) -> Path:
                del operation
                path = self.output_dir / "agent_run"
                path.mkdir()
                return path

        repeated_back = VisionDecision(
            screen_type="app_page",
            action="android_back",
            confidence=0.98,
            reason="返回上一页",
            target="返回",
        )
        with tempfile.TemporaryDirectory() as directory:
            controller = FastMockRobotController(Path(directory))
            runner = VisionAgentRunner(
                controller,
                ScriptedVisionProvider([repeated_back, repeated_back]),
            )
            with self.assertRaisesRegex(VisionAgentError, "画面几乎未变化"):
                runner.execute(
                    {
                        "goal": "进入页面后返回",
                        "allowed_texts": [],
                        "task_mode": "test",
                        "expected_result": "返回上一页",
                    }
                )

            self.assertEqual(
                [
                    item["action"]
                    for item in controller.executions
                    if item.get("action") == "android_back"
                ],
                ["android_back"],
            )

    def test_safe_navigation_tap_allows_one_retry_then_stops(self) -> None:
        class FastMockRobotController(MockRobotController):
            def __init__(self, output_dir: Path) -> None:
                super().__init__()
                self.output_dir = output_dir

            def _sleep(self, seconds: float) -> None:
                del seconds
                self._checkpoint()

            def _new_run_dir(self, operation: str) -> Path:
                del operation
                path = self.output_dir / "agent_run"
                path.mkdir()
                return path

        attempts = [
            VisionDecision(
                screen_type="android_home",
                action="tap",
                confidence=0.98,
                reason="微信图标清晰可见",
                target="微信应用图标",
                coordinate=(498, 673),
            ),
            VisionDecision(
                screen_type="android_home",
                action="tap",
                confidence=0.98,
                reason="首次物理点击漏点，微信图标仍在原位",
                target="微信应用图标",
                coordinate=(505, 668),
            ),
            VisionDecision(
                screen_type="android_home",
                action="tap",
                confidence=0.98,
                reason="补点后仍未打开",
                target="微信应用图标",
                coordinate=(500, 672),
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            controller = FastMockRobotController(Path(directory))
            runner = VisionAgentRunner(
                controller,
                ScriptedVisionProvider(attempts),
            )
            with self.assertRaisesRegex(VisionAgentError, "画面几乎未变化"):
                runner.execute(
                    {
                        "goal": "打开微信",
                        "allowed_texts": [],
                        "task_mode": "test",
                        "expected_result": "微信已经打开",
                    }
                )

            taps = [
                item for item in controller.executions if item.get("action") == "tap"
            ]
            self.assertEqual(
                [item["coordinate"] for item in taps],
                [[498, 673], [505, 668]],
            )

    def test_unsafe_chat_list_tap_does_not_receive_navigation_retry(self) -> None:
        class FastMockRobotController(MockRobotController):
            def __init__(self, output_dir: Path) -> None:
                super().__init__()
                self.output_dir = output_dir

            def _sleep(self, seconds: float) -> None:
                del seconds
                self._checkpoint()

            def _new_run_dir(self, operation: str) -> Path:
                del operation
                path = self.output_dir / "agent_run"
                path.mkdir()
                return path

        repeated_delete = VisionDecision(
            screen_type="chat_list",
            action="tap",
            confidence=0.98,
            reason="请求取消聊天操作",
            target="取消聊天会话",
            coordinate=(900, 500),
        )
        with tempfile.TemporaryDirectory() as directory:
            controller = FastMockRobotController(Path(directory))
            runner = VisionAgentRunner(
                controller,
                ScriptedVisionProvider([repeated_delete, repeated_delete]),
            )
            with self.assertRaisesRegex(VisionAgentError, "画面几乎未变化"):
                runner.execute(
                    {
                        "goal": "返回聊天列表",
                        "allowed_texts": [],
                        "task_mode": "test",
                        "expected_result": "返回聊天列表",
                    }
                )

            taps = [
                item for item in controller.executions if item.get("action") == "tap"
            ]
            self.assertEqual(len(taps), 1)

    def test_multiple_confirmed_texts_are_typed_in_order(self) -> None:
        class FastMockRobotController(MockRobotController):
            def __init__(self, output_dir: Path) -> None:
                super().__init__()
                self.output_dir = output_dir

            def _sleep(self, seconds: float) -> None:
                del seconds
                self._checkpoint()

            def _new_run_dir(self, operation: str) -> Path:
                del operation
                path = self.output_dir / "agent_run"
                path.mkdir()
                return path

        with tempfile.TemporaryDirectory() as directory:
            controller = FastMockRobotController(Path(directory))
            provider = ScriptedVisionProvider(
                [
                    VisionDecision(
                        screen_type="keyboard",
                        action="type_text",
                        confidence=0.98,
                        reason="用户名输入框已聚焦",
                        text="张三",
                    ),
                    VisionDecision(
                        screen_type="keyboard",
                        action="type_text",
                        confidence=0.98,
                        reason="部门输入框已聚焦",
                        text="质检",
                    ),
                    VisionDecision(
                        screen_type="app_page",
                        action="finish",
                        confidence=0.96,
                        reason="两个字段均已显示正确",
                        success=True,
                    ),
                ]
            )
            result = VisionAgentRunner(controller, provider).execute(
                {
                    "goal": "在表单中填写用户名张三，部门质检",
                    "allowed_texts": ["张三", "质检"],
                    "task_mode": "test",
                    "expected_result": "用户名显示张三且部门显示质检",
                }
            )
            typed = [
                item["text"]
                for item in controller.executions
                if item.get("action") == "type_text"
            ]
            self.assertEqual(typed, ["张三", "质检"])
            self.assertEqual(result["outcome"], "passed")

    def test_comment_must_close_panel_before_swiping(self) -> None:
        class FastMockRobotController(MockRobotController):
            def __init__(self, output_dir: Path) -> None:
                super().__init__()
                self.output_dir = output_dir

            def _sleep(self, seconds: float) -> None:
                del seconds
                self._checkpoint()

            def _new_run_dir(self, operation: str) -> Path:
                del operation
                path = self.output_dir / "agent_run"
                path.mkdir()
                return path

        with tempfile.TemporaryDirectory() as directory:
            controller = FastMockRobotController(Path(directory))
            provider = ScriptedVisionProvider(
                [
                    VisionDecision(
                        screen_type="keyboard",
                        action="type_text",
                        confidence=0.99,
                        reason="输入框已聚焦，使用本地数字输入",
                        target="评论输入框",
                        text="1",
                    ),
                    VisionDecision(
                        screen_type="video",
                        action="tap",
                        confidence=0.99,
                        reason="输入框逐字显示1",
                        target="发送按钮",
                        coordinate=(850, 490),
                        observed_input_text="1",
                    ),
                    VisionDecision(
                        screen_type="video",
                        action="tap",
                        confidence=0.99,
                        reason="本人新评论逐字显示1",
                        target="关闭评论面板X",
                        coordinate=(870, 320),
                        observed_input_text="1",
                    ),
                    VisionDecision(
                        screen_type="video",
                        action="swipe_up",
                        confidence=0.99,
                        reason="评论面板已关闭，键盘已消失",
                        target="评论面板已关闭",
                    ),
                ]
            )
            result = VisionAgentRunner(controller, provider).execute(
                {
                    "goal": "给接下来一条视频评论1",
                    "allowed_texts": ["1"],
                    "task_mode": "test",
                    "expected_result": "完整评论1且评论面板已关闭",
                }
            )
            actions = [item["action"] for item in controller.executions]
            self.assertEqual(actions, ["type_text", "tap", "tap"])
            self.assertEqual(result["controller_metrics"]["completed_pages"], 1)

    def test_comment_digit_rejects_qwerty_corner_legend_before_switch(self) -> None:
        class FastMockRobotController(MockRobotController):
            def __init__(self, output_dir: Path) -> None:
                super().__init__()
                self.output_dir = output_dir

            def _sleep(self, seconds: float) -> None:
                del seconds
                self._checkpoint()

            def _new_run_dir(self, operation: str) -> Path:
                del operation
                path = self.output_dir / "agent_run"
                path.mkdir()
                return path

        with tempfile.TemporaryDirectory() as directory:
            controller = FastMockRobotController(Path(directory))
            provider = ScriptedVisionProvider(
                [
                    VisionDecision(
                        screen_type="video",
                        action="tap",
                        confidence=0.99,
                        reason="评论编辑框清晰可见",
                        target="评论输入框",
                        coordinate=(380, 910),
                    ),
                    VisionDecision(
                        screen_type="keyboard",
                        action="type_symbol",
                        confidence=0.99,
                        reason="错误地把W键角落的1副标当成数字键",
                        target="数字1键",
                        text="1",
                        coordinate=(218, 695),
                    ),
                ]
            )
            with self.assertRaisesRegex(
                VisionAgentError,
                "尚未完成至少一次真实的数字/符号键盘切换",
            ):
                VisionAgentRunner(controller, provider).execute(
                    {
                        "goal": "给接下来一条视频评论1",
                        "allowed_texts": ["1"],
                        "task_mode": "test",
                        "expected_result": "完整评论1且评论面板已关闭",
                    }
                )
            self.assertEqual(
                [item["action"] for item in controller.executions],
                ["tap"],
            )

    def test_comment_wrong_symbol_clears_confirms_empty_and_relocates(self) -> None:
        class FastMockRobotController(MockRobotController):
            def __init__(self, output_dir: Path) -> None:
                super().__init__()
                self.output_dir = output_dir

            def _sleep(self, seconds: float) -> None:
                del seconds
                self._checkpoint()

            def _new_run_dir(self, operation: str) -> Path:
                del operation
                path = self.output_dir / "agent_run"
                path.mkdir()
                return path

        with tempfile.TemporaryDirectory() as directory:
            controller = FastMockRobotController(Path(directory))
            provider = ScriptedVisionProvider(
                [
                    VisionDecision(
                        screen_type="keyboard",
                        action="tap",
                        confidence=0.99,
                        reason="先切换到数字键盘",
                        target="123数字键盘切换键",
                        coordinate=(250, 900),
                    ),
                    VisionDecision(
                        screen_type="keyboard",
                        action="type_symbol",
                        confidence=0.99,
                        reason="尝试点击独立的数字1键",
                        target="数字键1",
                        text="1",
                        coordinate=(312, 745),
                        keyboard_layout=TEST_NUMERIC_GRID_LAYOUT,
                    ),
                    VisionDecision(
                        screen_type="keyboard",
                        action="clear_text",
                        confidence=0.99,
                        reason="复核发现输入框实际为w，精确退格一次",
                        target="退格键",
                        observed_input_text="w",
                        delete_count=1,
                        keyboard_layout={
                            "type": "generic",
                            "anchors": {"backspace": [865, 810]},
                        },
                    ),
                    VisionDecision(
                        screen_type="keyboard",
                        action="wait",
                        confidence=0.99,
                        reason="退格后只显示爱评论的人，运气不会差占位提示",
                        target="等待复核",
                        input_is_empty=False,
                    ),
                    VisionDecision(
                        screen_type="keyboard",
                        action="type_symbol",
                        confidence=0.99,
                        reason="重新定位并点击独立的数字1主键帽",
                        target="数字键1",
                        text="1",
                        coordinate=(304, 698),
                        keyboard_layout=TEST_NUMERIC_GRID_LAYOUT,
                    ),
                    VisionDecision(
                        screen_type="video",
                        action="tap",
                        confidence=0.99,
                        reason="输入框逐字显示1",
                        target="发送按钮",
                        coordinate=(850, 490),
                        observed_input_text="1",
                    ),
                    VisionDecision(
                        screen_type="video",
                        action="tap",
                        confidence=0.99,
                        reason="本人新评论逐字显示1",
                        target="关闭评论面板X",
                        coordinate=(870, 320),
                        observed_input_text="1",
                    ),
                    VisionDecision(
                        screen_type="video",
                        action="swipe_up",
                        confidence=0.99,
                        reason="评论面板已关闭，键盘已消失",
                        target="评论面板已关闭",
                    ),
                ]
            )
            result = VisionAgentRunner(controller, provider).execute(
                {
                    "goal": "给接下来一条视频评论1",
                    "allowed_texts": ["1"],
                    "task_mode": "test",
                    "expected_result": "完整评论1且评论面板已关闭",
                }
            )
            self.assertEqual(
                [item["action"] for item in controller.executions],
                ["tap", "tap", "clear_text", "tap", "tap", "tap"],
            )
            self.assertEqual(
                controller.executions[1]["coordinate"],
                [164, 638],
            )
            self.assertEqual(
                controller.executions[3]["coordinate"],
                [164, 638],
            )
            self.assertEqual(result["controller_metrics"]["completed_pages"], 1)

    def test_comment_panel_open_rejects_swipe(self) -> None:
        class FastMockRobotController(MockRobotController):
            def __init__(self, output_dir: Path) -> None:
                super().__init__()
                self.output_dir = output_dir

            def _sleep(self, seconds: float) -> None:
                del seconds
                self._checkpoint()

            def _new_run_dir(self, operation: str) -> Path:
                del operation
                path = self.output_dir / "agent_run"
                path.mkdir()
                return path

        with tempfile.TemporaryDirectory() as directory:
            controller = FastMockRobotController(Path(directory))
            provider = ScriptedVisionProvider(
                [
                    VisionDecision(
                        screen_type="keyboard",
                        action="tap",
                        confidence=0.99,
                        reason="QWERTY键盘显示123切换键",
                        target="123数字键盘切换键",
                        coordinate=(250, 900),
                    ),
                    VisionDecision(
                        screen_type="keyboard",
                        action="type_symbol",
                        confidence=0.99,
                        reason="数字键盘中真实看见1键",
                        target="数字键1",
                        text="1",
                        coordinate=(105, 695),
                    ),
                    VisionDecision(
                        screen_type="video",
                        action="tap",
                        confidence=0.99,
                        reason="输入完整",
                        target="发送按钮",
                        coordinate=(850, 490),
                        observed_input_text="1",
                    ),
                    VisionDecision(
                        screen_type="video",
                        action="swipe_up",
                        confidence=0.99,
                        reason="尝试滑动",
                        target="下一条视频",
                    ),
                ]
            )
            with self.assertRaisesRegex(VisionAgentError, "评论面板尚未关闭"):
                VisionAgentRunner(controller, provider).execute(
                    {
                        "goal": "给接下来二条视频评论1",
                        "allowed_texts": ["1"],
                        "task_mode": "test",
                        "expected_result": "逐条完整评论1",
                    }
                )
            self.assertFalse(
                any(
                    item.get("action") == "swipe_up"
                    for item in controller.executions
                )
            )

    def test_confirmed_chinese_reaches_pinyin_controller(self) -> None:
        class FastMockRobotController(MockRobotController):
            def __init__(self, output_dir: Path) -> None:
                super().__init__()
                self.output_dir = output_dir

            def _sleep(self, seconds: float) -> None:
                del seconds
                self._checkpoint()

            def _new_run_dir(self, operation: str) -> Path:
                del operation
                path = self.output_dir / "agent_run"
                path.mkdir()
                return path

        with tempfile.TemporaryDirectory() as directory:
            controller = FastMockRobotController(Path(directory))
            provider = ScriptedVisionProvider(
                [
                    VisionDecision(
                        screen_type="keyboard",
                        action="type_pinyin",
                        confidence=0.98,
                        reason="标准拼音键盘可见",
                        target="消息输入框",
                        text="你好",
                        pinyin="nihao",
                        keyboard_layout=TEST_QWERTY_LAYOUT,
                    ),
                    VisionDecision(
                        screen_type="chat",
                        action="tap",
                        confidence=0.98,
                        reason="选择准确候选词",
                        target="候选词你好",
                        coordinate=(140, 635),
                        observed_input_text="ni'hao",
                    ),
                    VisionDecision(
                        screen_type="chat",
                        action="finish",
                        confidence=0.98,
                        reason="输入框已显示你好",
                        success=True,
                    ),
                ]
            )
            result = VisionAgentRunner(controller, provider).execute(
                {
                    "goal": "在输入框中输入你好，绝对不要点击发送按钮",
                    "allowed_texts": ["你好"],
                    "task_mode": "test",
                    "expected_result": "输入框显示你好且没有发送",
                }
            )

            self.assertIn(
                {
                    "action": "type_pinyin",
                    "text": "你好",
                    "pinyin": "nihao",
                    "keyboard_layout": TEST_QWERTY_LAYOUT,
                },
                controller.executions,
            )
            self.assertEqual(result["outcome"], "passed")

    def test_pinyin_candidate_can_only_be_tapped_once(self) -> None:
        class FastMockRobotController(MockRobotController):
            def __init__(self, output_dir: Path) -> None:
                super().__init__()
                self.output_dir = output_dir

            def _sleep(self, seconds: float) -> None:
                del seconds
                self._checkpoint()

            def _new_run_dir(self, operation: str) -> Path:
                del operation
                path = self.output_dir / "agent_run"
                path.mkdir()
                return path

        with tempfile.TemporaryDirectory() as directory:
            controller = FastMockRobotController(Path(directory))
            provider = ScriptedVisionProvider(
                [
                    VisionDecision(
                        screen_type="keyboard",
                        action="type_pinyin",
                        confidence=0.98,
                        reason="标准拼音键盘可见",
                        target="消息输入框",
                        text="你好",
                        pinyin="nihao",
                        keyboard_layout=TEST_QWERTY_LAYOUT,
                    ),
                    VisionDecision(
                        screen_type="chat",
                        action="tap",
                        confidence=0.98,
                        reason="选择准确候选词",
                        target="候选词你好",
                        coordinate=(140, 635),
                        observed_input_text="ni'hao",
                    ),
                    VisionDecision(
                        screen_type="chat",
                        action="tap",
                        confidence=0.98,
                        reason="错误地再次点击候选词",
                        target="候选词你好",
                        coordinate=(145, 635),
                    ),
                ]
            )

            with self.assertRaisesRegex(
                VisionAgentError,
                "核对候选词写入后的底部输入框",
            ):
                VisionAgentRunner(controller, provider).execute(
                    {
                        "goal": "在输入框中输入你好但不要发送",
                        "allowed_texts": ["你好"],
                        "task_mode": "test",
                        "expected_result": "输入框显示你好且没有发送",
                    }
                )

            taps = [
                item for item in controller.executions if item.get("action") == "tap"
            ]
            self.assertEqual(len(taps), 1)

    def test_pinyin_candidate_missed_tap_reuses_verified_coordinate_once(self) -> None:
        class CandidateRetryController(MockRobotController):
            def __init__(self, output_dir: Path) -> None:
                super().__init__()
                self.output_dir = output_dir
                self.candidate_taps = 0
                self.pinyin_visible = False
                self.candidate_selected = False

            def _sleep(self, seconds: float) -> None:
                del seconds
                self._checkpoint()

            def _new_run_dir(self, operation: str) -> Path:
                del operation
                path = self.output_dir / "agent_run"
                path.mkdir()
                return path

            def vision_capture(self) -> Image.Image:
                image = Image.new("RGB", (540, 960), "white")
                draw = ImageDraw.Draw(image)
                if self.pinyin_visible:
                    draw.rectangle((20, 470, 520, 680), fill=(225, 225, 225))
                    draw.text((75, 560), "ni'hao", fill="black")
                if self.candidate_selected:
                    draw.rectangle((20, 470, 520, 680), fill=(185, 235, 185))
                    draw.text((75, 560), "selected", fill="black")
                return image

            def vision_type_pinyin(
                self,
                text: str,
                pinyin: str,
                keyboard_layout: dict[str, object] | None = None,
            ) -> None:
                super().vision_type_pinyin(text, pinyin, keyboard_layout)
                self.pinyin_visible = True

            def vision_tap_relative(self, x: int, y: int) -> tuple[int, int]:
                result = super().vision_tap_relative(x, y)
                self.candidate_taps += 1
                if self.candidate_taps == 2:
                    self.candidate_selected = True
                return result

        class MissingCoordinateOnceProvider(ScriptedVisionProvider):
            def __init__(self) -> None:
                super().__init__([])
                self.calls = 0

            def decide(self, **kwargs: object) -> VisionDecision:
                del kwargs
                self.calls += 1
                if self.calls == 1:
                    return VisionDecision(
                        screen_type="keyboard",
                        action="type_pinyin",
                        confidence=0.98,
                        reason="标准拼音键盘可见",
                        target="消息输入框",
                        text="你好",
                        pinyin="nihao",
                        keyboard_layout=TEST_QWERTY_LAYOUT,
                    )
                if self.calls == 2:
                    return VisionDecision(
                        screen_type="chat",
                        action="tap",
                        confidence=0.98,
                        reason="选择准确候选词",
                        target="候选词你好",
                        coordinate=(140, 635),
                        observed_input_text="ni'hao",
                    )
                if self.calls == 3:
                    raise VisionAgentError("tap 动作缺少 coordinate。")
                return VisionDecision(
                    screen_type="chat",
                    action="finish",
                    confidence=0.98,
                    reason="输入框已显示你好",
                    success=True,
                )

        with tempfile.TemporaryDirectory() as directory:
            controller = CandidateRetryController(Path(directory))
            result = VisionAgentRunner(
                controller,
                MissingCoordinateOnceProvider(),
            ).execute(
                {
                    "goal": "在输入框中输入你好但不要发送",
                    "allowed_texts": ["你好"],
                    "task_mode": "test",
                    "expected_result": "输入框显示你好且没有发送",
                }
            )

            taps = [
                item for item in controller.executions if item.get("action") == "tap"
            ]
            self.assertEqual(len(taps), 2)
            self.assertEqual(taps[0]["coordinate"], taps[1]["coordinate"])
            self.assertTrue(result["steps"][2]["controller_owned"])
            self.assertEqual(result["outcome"], "passed")

    def test_pinyin_candidate_phase_blocks_retyping(self) -> None:
        class FastMockRobotController(MockRobotController):
            def __init__(self, output_dir: Path) -> None:
                super().__init__()
                self.output_dir = output_dir

            def _sleep(self, seconds: float) -> None:
                del seconds
                self._checkpoint()

            def _new_run_dir(self, operation: str) -> Path:
                del operation
                path = self.output_dir / "agent_run"
                path.mkdir()
                return path

        repeated_type = VisionDecision(
            screen_type="keyboard",
            action="type_pinyin",
            confidence=0.98,
            reason="错误地重复输入拼音",
            target="消息输入框",
            text="你好",
            pinyin="nihao",
            keyboard_layout=TEST_QWERTY_LAYOUT,
        )
        with tempfile.TemporaryDirectory() as directory:
            controller = FastMockRobotController(Path(directory))
            provider = ScriptedVisionProvider([repeated_type, repeated_type])

            with self.assertRaisesRegex(
                VisionAgentError,
                "正在核对拼音和候选词",
            ):
                VisionAgentRunner(controller, provider).execute(
                    {
                        "goal": "在输入框中输入你好但不要发送",
                        "allowed_texts": ["你好"],
                        "task_mode": "test",
                        "expected_result": "输入框显示你好且没有发送",
                    }
                )

            typed = [
                item
                for item in controller.executions
                if item.get("action") == "type_pinyin"
            ]
            self.assertEqual(len(typed), 1)

    def test_wrong_visible_pinyin_is_cleared_then_relocalized_and_retyped(
        self,
    ) -> None:
        class FastMockRobotController(MockRobotController):
            def __init__(self, output_dir: Path) -> None:
                super().__init__()
                self.output_dir = output_dir

            def _sleep(self, seconds: float) -> None:
                del seconds
                self._checkpoint()

            def _new_run_dir(self, operation: str) -> Path:
                del operation
                path = self.output_dir / "agent_run"
                path.mkdir()
                return path

        with tempfile.TemporaryDirectory() as directory:
            controller = FastMockRobotController(Path(directory))
            provider = ScriptedVisionProvider(
                [
                    VisionDecision(
                        screen_type="keyboard",
                        action="type_pinyin",
                        confidence=0.98,
                        reason="标准拼音键盘可见",
                        target="消息输入框",
                        text="你好",
                        pinyin="nihao",
                        keyboard_layout=TEST_QWERTY_LAYOUT,
                    ),
                    VisionDecision(
                        screen_type="chat",
                        action="tap",
                        confidence=0.98,
                        reason="错误地把多余拼音看成可选候选词",
                        target="候选词你好",
                        coordinate=(140, 635),
                        observed_input_text="n'nihao",
                    ),
                    VisionDecision(
                        screen_type="keyboard",
                        action="wait",
                        confidence=0.98,
                        reason="退格后底部输入框为空",
                        target="底部输入框",
                        input_is_empty=True,
                    ),
                    VisionDecision(
                        screen_type="keyboard",
                        action="type_pinyin",
                        confidence=0.98,
                        reason="重新观察并定位标准拼音键盘",
                        target="消息输入框",
                        text="你好",
                        pinyin="nihao",
                        keyboard_layout=TEST_QWERTY_LAYOUT,
                    ),
                    VisionDecision(
                        screen_type="chat",
                        action="tap",
                        confidence=0.98,
                        reason="拼音正确且候选词与原文完全一致",
                        target="候选词你好",
                        coordinate=(140, 635),
                        observed_input_text="ni'hao",
                    ),
                    VisionDecision(
                        screen_type="chat",
                        action="finish",
                        confidence=0.98,
                        reason="底部输入框准确显示你好且没有发送",
                        target="底部输入框你好",
                        success=True,
                    ),
                ]
            )

            result = VisionAgentRunner(controller, provider).execute(
                {
                    "goal": "在输入框中输入你好但不要发送",
                    "allowed_texts": ["你好"],
                    "task_mode": "test",
                    "expected_result": "输入框显示你好且没有发送",
                }
            )

            taps = [
                item for item in controller.executions if item.get("action") == "tap"
            ]
            clears = [
                item
                for item in controller.executions
                if item.get("action") == "clear_text"
            ]
            typed = [
                item
                for item in controller.executions
                if item.get("action") == "type_pinyin"
            ]
            self.assertEqual(result["outcome"], "passed")
            self.assertEqual(len(taps), 1)
            self.assertEqual(len(clears), 1)
            self.assertEqual(clears[0]["delete_count"], 6)
            self.assertEqual(len(typed), 2)

    def test_pinyin_send_is_allowed_once_after_input_verification(self) -> None:
        class FastMockRobotController(MockRobotController):
            def __init__(self, output_dir: Path) -> None:
                super().__init__()
                self.output_dir = output_dir

            def _sleep(self, seconds: float) -> None:
                del seconds
                self._checkpoint()

            def _new_run_dir(self, operation: str) -> Path:
                del operation
                path = self.output_dir / "agent_run"
                path.mkdir()
                return path

        with tempfile.TemporaryDirectory() as directory:
            controller = FastMockRobotController(Path(directory))
            provider = ScriptedVisionProvider(
                [
                    VisionDecision(
                        screen_type="keyboard",
                        action="type_pinyin",
                        confidence=0.98,
                        reason="标准拼音键盘可见",
                        target="消息输入框",
                        text="你好",
                        pinyin="nihao",
                        keyboard_layout=TEST_QWERTY_LAYOUT,
                    ),
                    VisionDecision(
                        screen_type="chat",
                        action="tap",
                        confidence=0.98,
                        reason="选择准确候选词",
                        target="候选词你好",
                        coordinate=(140, 635),
                        observed_input_text="ni'hao",
                    ),
                    VisionDecision(
                        screen_type="chat",
                        action="tap",
                        confidence=0.98,
                        reason="输入框逐字显示你好，发送按钮清晰可见",
                        target="发送按钮",
                        coordinate=(850, 540),
                    ),
                    VisionDecision(
                        screen_type="chat",
                        action="finish",
                        confidence=0.98,
                        reason="新的本人发送气泡可见且输入框已清空",
                        input_is_empty=True,
                        sent_message_visible=True,
                        success=True,
                    ),
                ]
            )

            result = VisionAgentRunner(controller, provider).execute(
                {
                    "goal": "打开微信，在文件传输助手里发送你好",
                    "allowed_texts": ["你好"],
                    "task_mode": "operate",
                }
            )

            taps = [
                item for item in controller.executions if item.get("action") == "tap"
            ]
            self.assertEqual(
                [item["coordinate"] for item in taps],
                [[140, 635], [850, 540]],
            )
            self.assertEqual(result["outcome"], "completed")

    def test_controller_confirms_send_from_pre_post_transition_and_stability(self) -> None:
        class TransitionController(MockRobotController):
            def __init__(self, output_dir: Path) -> None:
                super().__init__()
                self.output_dir = output_dir

            def _sleep(self, seconds: float) -> None:
                del seconds
                self._checkpoint()

            def _new_run_dir(self, operation: str) -> Path:
                del operation
                path = self.output_dir / "agent_run"
                path.mkdir()
                return path

            def vision_capture(self) -> Image.Image:
                image = Image.new("RGB", (540, 960), "white")
                draw = ImageDraw.Draw(image)
                tap_count = sum(
                    item.get("action") == "tap" for item in self.executions
                )
                if tap_count < 2:
                    # Before the send tap: text/send state occupies the input ROI.
                    draw.rectangle((20, 470, 510, 680), fill=(60, 60, 60))
                else:
                    # After send: input is empty and a new right-side green bubble
                    # appears. Repeated captures are intentionally identical.
                    draw.rectangle((330, 160, 500, 260), fill=(45, 185, 80))
                return image

        with tempfile.TemporaryDirectory() as directory:
            controller = TransitionController(Path(directory))
            provider = ScriptedVisionProvider(
                [
                    VisionDecision(
                        screen_type="keyboard",
                        action="type_pinyin",
                        confidence=0.98,
                        reason="标准拼音键盘可见",
                        target="消息输入框",
                        text="你好",
                        pinyin="nihao",
                        keyboard_layout=TEST_QWERTY_LAYOUT,
                    ),
                    VisionDecision(
                        screen_type="chat",
                        action="tap",
                        confidence=0.98,
                        reason="选择准确候选词",
                        target="候选词你好",
                        coordinate=(140, 635),
                        observed_input_text="ni'hao",
                    ),
                    VisionDecision(
                        screen_type="chat",
                        action="tap",
                        confidence=0.98,
                        reason="输入框逐字显示你好",
                        target="发送按钮",
                        coordinate=(850, 540),
                    ),
                    VisionDecision(
                        screen_type="chat",
                        action="wait",
                        confidence=0.98,
                        reason="输入框为空，但模型不确定气泡是否为本次新增",
                        input_is_empty=True,
                    ),
                    VisionDecision(
                        screen_type="chat",
                        action="wait",
                        confidence=0.98,
                        reason="画面稳定，输入框仍为空",
                        input_is_empty=True,
                    ),
                ]
            )

            result = VisionAgentRunner(controller, provider).execute(
                {
                    "goal": "打开微信，在文件传输助手里发送你好",
                    "allowed_texts": ["你好"],
                    "task_mode": "operate",
                }
            )

            self.assertEqual(result["outcome"], "completed")
            self.assertTrue(result["steps"][-1]["sent_message_visible"])
            self.assertTrue(result["steps"][-1]["input_is_empty"])
            self.assertIn("控制器比较发送前后画面", result["steps"][-1]["reason"])

    def test_old_bubble_without_input_transition_is_not_send_evidence(self) -> None:
        before = Image.new("RGB", (540, 960), "white")
        draw = ImageDraw.Draw(before)
        draw.rectangle((330, 160, 500, 260), fill=(45, 185, 80))
        unchanged = before.copy()

        metrics = sent_message_transition_metrics(before, unchanged)

        self.assertFalse(metrics["transition_visible"])
        self.assertEqual(metrics["input_delta"], 0.0)
        self.assertEqual(metrics["message_delta"], 0.0)

    def test_message_change_without_input_change_is_not_send_evidence(self) -> None:
        before = Image.new("RGB", (540, 960), "white")
        after = before.copy()
        ImageDraw.Draw(after).rectangle((330, 160, 500, 260), fill=(45, 185, 80))

        metrics = sent_message_transition_metrics(before, after)

        self.assertFalse(metrics["transition_visible"])
        self.assertEqual(metrics["input_delta"], 0.0)
        self.assertGreater(metrics["message_delta"], 0.0)

    def test_pinyin_send_allows_one_verified_retry_after_physical_miss(self) -> None:
        class FastMockRobotController(MockRobotController):
            def __init__(self, output_dir: Path) -> None:
                super().__init__()
                self.output_dir = output_dir

            def _sleep(self, seconds: float) -> None:
                del seconds
                self._checkpoint()

            def _new_run_dir(self, operation: str) -> Path:
                del operation
                path = self.output_dir / "agent_run"
                path.mkdir()
                return path

        with tempfile.TemporaryDirectory() as directory:
            controller = FastMockRobotController(Path(directory))
            provider = ScriptedVisionProvider(
                [
                    VisionDecision(
                        screen_type="keyboard",
                        action="type_pinyin",
                        confidence=0.98,
                        reason="标准拼音键盘可见",
                        target="消息输入框",
                        text="你好",
                        pinyin="nihao",
                        keyboard_layout=TEST_QWERTY_LAYOUT,
                    ),
                    VisionDecision(
                        screen_type="chat",
                        action="tap",
                        confidence=0.98,
                        reason="选择准确候选词",
                        target="候选词你好",
                        coordinate=(140, 635),
                        observed_input_text="ni'hao",
                    ),
                    VisionDecision(
                        screen_type="chat",
                        action="tap",
                        confidence=0.98,
                        reason="输入框逐字显示你好，发送按钮清晰可见",
                        target="发送按钮",
                        coordinate=(850, 540),
                    ),
                    VisionDecision(
                        screen_type="chat",
                        action="wait",
                        confidence=0.98,
                        reason="发送后第一轮先等待稳定复核",
                    ),
                    VisionDecision(
                        screen_type="chat",
                        action="tap",
                        confidence=0.98,
                        reason="第一次物理触控漏点，输入框仍逐字保留你好",
                        target="发送按钮",
                        coordinate=(850, 540),
                        observed_input_text="你好",
                    ),
                    VisionDecision(
                        screen_type="chat",
                        action="finish",
                        confidence=0.98,
                        reason="新的本人发送气泡可见且输入框已清空",
                        input_is_empty=True,
                        sent_message_visible=True,
                        success=True,
                    ),
                ]
            )

            result = VisionAgentRunner(controller, provider).execute(
                {
                    "goal": "打开微信，在文件传输助手里发送你好",
                    "allowed_texts": ["你好"],
                    "task_mode": "operate",
                }
            )

            taps = [
                item for item in controller.executions if item.get("action") == "tap"
            ]
            self.assertEqual(
                [item["coordinate"] for item in taps],
                [[140, 635], [850, 540], [850, 540]],
            )
            self.assertEqual(result["outcome"], "completed")

    def test_pinyin_send_rejects_unverified_retry(self) -> None:
        class FastMockRobotController(MockRobotController):
            def __init__(self, output_dir: Path) -> None:
                super().__init__()
                self.output_dir = output_dir

            def _sleep(self, seconds: float) -> None:
                del seconds
                self._checkpoint()

            def _new_run_dir(self, operation: str) -> Path:
                del operation
                path = self.output_dir / "agent_run"
                path.mkdir()
                return path

        with tempfile.TemporaryDirectory() as directory:
            controller = FastMockRobotController(Path(directory))
            provider = ScriptedVisionProvider(
                [
                    VisionDecision(
                        screen_type="keyboard",
                        action="type_pinyin",
                        confidence=0.98,
                        reason="标准拼音键盘可见",
                        target="消息输入框",
                        text="你好",
                        pinyin="nihao",
                        keyboard_layout=TEST_QWERTY_LAYOUT,
                    ),
                    VisionDecision(
                        screen_type="chat",
                        action="tap",
                        confidence=0.98,
                        reason="选择准确候选词",
                        target="候选词你好",
                        coordinate=(140, 635),
                        observed_input_text="ni'hao",
                    ),
                    VisionDecision(
                        screen_type="chat",
                        action="tap",
                        confidence=0.98,
                        reason="输入框逐字显示你好，发送按钮清晰可见",
                        target="发送按钮",
                        coordinate=(850, 540),
                    ),
                    VisionDecision(
                        screen_type="chat",
                        action="wait",
                        confidence=0.98,
                        reason="发送后第一轮先等待稳定复核",
                    ),
                    VisionDecision(
                        screen_type="chat",
                        action="tap",
                        confidence=0.98,
                        reason="未逐字报告输入框内容却请求重试",
                        target="发送按钮",
                        coordinate=(849, 540),
                    ),
                ]
            )

            with self.assertRaisesRegex(
                VisionAgentError,
                "不满足安全补点条件",
            ):
                VisionAgentRunner(controller, provider).execute(
                    {
                        "goal": "打开微信，在文件传输助手里发送你好",
                        "allowed_texts": ["你好"],
                        "task_mode": "operate",
                    }
                )

            taps = [
                item for item in controller.executions if item.get("action") == "tap"
            ]
            self.assertEqual(
                [item["coordinate"] for item in taps],
                [[140, 635], [850, 540]],
            )

    def test_pinyin_send_cannot_finish_before_send_tap(self) -> None:
        class FastMockRobotController(MockRobotController):
            def __init__(self, output_dir: Path) -> None:
                super().__init__()
                self.output_dir = output_dir

            def _sleep(self, seconds: float) -> None:
                del seconds
                self._checkpoint()

            def _new_run_dir(self, operation: str) -> Path:
                del operation
                path = self.output_dir / "agent_run"
                path.mkdir()
                return path

        with tempfile.TemporaryDirectory() as directory:
            controller = FastMockRobotController(Path(directory))
            provider = ScriptedVisionProvider(
                [
                    VisionDecision(
                        screen_type="keyboard",
                        action="type_pinyin",
                        confidence=0.98,
                        reason="标准拼音键盘可见",
                        target="消息输入框",
                        text="你好",
                        pinyin="nihao",
                        keyboard_layout=TEST_QWERTY_LAYOUT,
                    ),
                    VisionDecision(
                        screen_type="chat",
                        action="tap",
                        confidence=0.98,
                        reason="选择准确候选词",
                        target="候选词你好",
                        coordinate=(140, 635),
                        observed_input_text="ni'hao",
                    ),
                    VisionDecision(
                        screen_type="chat",
                        action="finish",
                        confidence=0.98,
                        reason="输入框逐字显示你好",
                        success=True,
                    ),
                ]
            )

            with self.assertRaisesRegex(
                VisionAgentError,
                "核对候选词写入后的底部输入框",
            ):
                VisionAgentRunner(controller, provider).execute(
                    {
                        "goal": "打开微信，在文件传输助手里发送你好",
                        "allowed_texts": ["你好"],
                        "task_mode": "operate",
                    }
                )

    def test_pinyin_send_finish_requires_bubble_and_empty_input(self) -> None:
        class FastMockRobotController(MockRobotController):
            def __init__(self, output_dir: Path) -> None:
                super().__init__()
                self.output_dir = output_dir

            def _sleep(self, seconds: float) -> None:
                del seconds
                self._checkpoint()

            def _new_run_dir(self, operation: str) -> Path:
                del operation
                path = self.output_dir / "agent_run"
                path.mkdir()
                return path

        with tempfile.TemporaryDirectory() as directory:
            controller = FastMockRobotController(Path(directory))
            provider = ScriptedVisionProvider(
                [
                    VisionDecision(
                        screen_type="keyboard",
                        action="type_pinyin",
                        confidence=0.98,
                        reason="标准拼音键盘可见",
                        target="消息输入框",
                        text="你好",
                        pinyin="nihao",
                        keyboard_layout=TEST_QWERTY_LAYOUT,
                    ),
                    VisionDecision(
                        screen_type="chat",
                        action="tap",
                        confidence=0.98,
                        reason="选择准确候选词",
                        target="候选词你好",
                        coordinate=(140, 635),
                        observed_input_text="ni'hao",
                    ),
                    VisionDecision(
                        screen_type="chat",
                        action="tap",
                        confidence=0.98,
                        reason="输入框逐字显示你好",
                        target="发送按钮",
                        coordinate=(850, 540),
                    ),
                    VisionDecision(
                        screen_type="chat",
                        action="finish",
                        confidence=0.98,
                        reason="只看见新的本人消息气泡，未确认输入框为空",
                        sent_message_visible=True,
                        success=True,
                    ),
                ]
            )

            with self.assertRaisesRegex(
                VisionAgentError,
                "尚未同时确认新的本人消息气泡和空输入框",
            ):
                VisionAgentRunner(controller, provider).execute(
                    {
                        "goal": "打开微信，在文件传输助手里发送你好",
                        "allowed_texts": ["你好"],
                        "task_mode": "operate",
                    }
                )

    def test_all_four_swipe_directions_reach_controller(self) -> None:
        class FastMockRobotController(MockRobotController):
            def __init__(self, output_dir: Path) -> None:
                super().__init__()
                self.output_dir = output_dir

            def _sleep(self, seconds: float) -> None:
                del seconds
                self._checkpoint()

            def _new_run_dir(self, operation: str) -> Path:
                del operation
                path = self.output_dir / "agent_run"
                path.mkdir()
                return path

        actions = ["swipe_up", "swipe_down", "swipe_left", "swipe_right"]
        decisions = [
            VisionDecision(
                screen_type="app_page",
                action=action,
                confidence=0.95,
                reason=f"执行{action}",
            )
            for action in actions
        ]
        decisions.append(
            VisionDecision(
                screen_type="app_page",
                action="finish",
                confidence=0.96,
                reason="滑动测试完成",
                success=True,
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            controller = FastMockRobotController(Path(directory))
            result = VisionAgentRunner(
                controller,
                ScriptedVisionProvider(decisions),
            ).execute(
                {
                    "goal": "依次测试上划、下划、左划和右划",
                    "allowed_texts": [],
                    "task_mode": "test",
                    "expected_result": "四个方向滑动均触发页面变化",
                }
            )
            self.assertEqual(
                [item["action"] for item in controller.executions],
                actions,
            )
            self.assertEqual(result["outcome"], "passed")


class StoreAndQueueTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = TaskStore(Path(self.temp_dir.name) / "tasks.sqlite3")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_unconfirmed_task_cannot_be_started(self) -> None:
        task = self.store.create("douyin", "douyin.like_current", {})
        with self.assertRaises(ValueError):
            self.store.transition(task["id"], {"queued"}, "running")

    def test_task_survives_store_reopen(self) -> None:
        task = self.store.create("douyin", "douyin.like_current", {})
        reopened = TaskStore(self.store.path)
        loaded = reopened.get(task["id"])
        self.assertEqual(loaded["status"], "awaiting_confirmation")

    def test_tasks_execute_serially(self) -> None:
        controller = MockRobotController()
        active = 0
        max_active = 0
        guard = threading.Lock()

        def run(index: int) -> None:
            nonlocal active, max_active
            with controller.operation_lock:
                with guard:
                    active += 1
                    max_active = max(max_active, active)
                time.sleep(0.03)
                controller.executions.append({"index": index})
                with guard:
                    active -= 1

        threads = [threading.Thread(target=run, args=(index,)) for index in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(max_active, 1)
        self.assertEqual(len(controller.executions), 2)

    def test_interrupted_running_task_is_failed_on_reopen(self) -> None:
        task = self.store.create("douyin", "douyin.like_current", {})
        self.store.transition(task["id"], {"awaiting_confirmation"}, "queued")
        self.store.transition(task["id"], {"queued"}, "running")
        reopened = TaskStore(self.store.path)
        loaded = reopened.get(task["id"])
        self.assertEqual(loaded["status"], "failed")
        self.assertIn("上次退出", loaded["error"])


class DeviceControllerRegistryTests(unittest.TestCase):
    def test_default_real_device_advertises_only_actions_with_live_evidence(self) -> None:
        registry = web_app.DeviceControllerRegistry(web_app.DEVICE_REGISTRY_PATH, mock=False)
        controller = registry.controller(registry.default_device_id)

        self.assertTrue(controller.hardware_capabilities()["input_verified_text"])
        self.assertFalse(controller.hardware_capabilities()["long_press"])
        self.assertFalse(controller.hardware_capabilities()["drag"])

    def test_two_devices_have_independent_controllers_and_calibrations(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "devices.json"
            path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "default_device_id": "phone-a",
                        "devices": [
                            {
                                "device_id": "phone-a",
                                "enabled": True,
                                "window_title": "controller-a",
                                "calibration_path": "calibration-a.json",
                            },
                            {
                                "device_id": "phone-b",
                                "enabled": True,
                                "window_title": "controller-b",
                                "calibration_path": "calibration-b.json",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            registry = web_app.DeviceControllerRegistry(path, mock=False)

            first = registry.controller("phone-a")
            second = registry.controller("phone-b")

        self.assertIsNot(first, second)
        self.assertEqual(first.title, "controller-a")
        self.assertEqual(second.title, "controller-b")
        self.assertTrue(str(first.calibration_path).endswith("calibration-a.json"))
        self.assertTrue(str(second.calibration_path).endswith("calibration-b.json"))
        with self.assertRaisesRegex(web_app.UniversalAgentOrchestratorError, "未登记"):
            registry.controller("phone-c")

    def test_duplicate_enabled_window_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "devices.json"
            path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "default_device_id": "phone-a",
                        "devices": [
                            {"device_id": "phone-a", "enabled": True, "window_title": "same"},
                            {"device_id": "phone-b", "enabled": True, "window_title": "same"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "同一个机械臂控制窗口"):
                web_app.DeviceControllerRegistry(path)


class ApiEndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.original_web_output_dir = web_app.WEB_OUTPUT_DIR
        web_app.WEB_OUTPUT_DIR = Path(cls.temp_dir.name) / "web_output"
        web_app.WEB_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        web_app.runtime.store = TaskStore(
            Path(cls.temp_dir.name) / "api_tasks.sqlite3"
        )
        web_app.runtime.controller = MockRobotController()
        cls.client_context = TestClient(web_app.app)
        cls.client = cls.client_context.__enter__()
        cls.headers = {"X-Control-Token": web_app.CONTROL_TOKEN}

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client_context.__exit__(None, None, None)
        web_app.WEB_OUTPUT_DIR = cls.original_web_output_dir
        cls.temp_dir.cleanup()

    def setUp(self) -> None:
        from universal_agent_orchestrator import DeviceTaskRegistry

        self.device_registry_patcher = patch.object(
            web_app.runtime,
            "device_task_registry",
            DeviceTaskRegistry(),
        )
        self.device_registry_patcher.start()
        with web_app.runtime.supervised_session_lock:
            web_app.runtime.supervised_sessions.clear()
            web_app.runtime.supervised_session_dirs.clear()
        with web_app.runtime.generic_supervised_session_lock:
            web_app.runtime.generic_supervised_sessions.clear()

    def tearDown(self) -> None:
        self.device_registry_patcher.stop()

    def _universal_api_orchestrator(self, *, device_id="phone-01", graph=None):
        from test_universal_agent_orchestrator import (
            FakeDeepSeekPlanner,
            FakeExecutingAdapter,
            FakeQwenObserver,
            _graph,
            _scene,
            _trusted_factory,
        )
        from universal_agent_orchestrator import (
            DeviceTaskRegistry,
            UniversalAgentOrchestrator,
        )

        initial = graph or _graph(device_id=device_id)
        planner = FakeDeepSeekPlanner(
            initial,
            replan_result=replace(initial, revision=initial.revision + 1),
        )
        qwen = FakeQwenObserver()
        adapter = FakeExecutingAdapter(
            _scene(),
            _scene(
                fingerprint="frame-after-api",
                meaning="open_more",
                label="查看更多",
            ),
        )
        orchestrator = UniversalAgentOrchestrator(
            deepseek_planner=planner,
            qwen_observer=qwen,
            adapter_factory=lambda _device_id: adapter,
            trusted_observation_factory=_trusted_factory,
            device_registry=DeviceTaskRegistry(),
        )
        return orchestrator, planner, qwen, adapter

    def _fake_capability_manager(self, *, device_id="capability-api-device"):
        calls = []

        class Session:
            def __init__(self):
                self.physical_actions = 0
                self.status = "awaiting_confirmation"

            def snapshot(self):
                return {
                    "session_id": "capability-session-001",
                    "status": self.status,
                    "physical_actions": self.physical_actions,
                    "confirmation_scope": {
                        "session_id": "capability-session-001",
                        "task_id": "task-001",
                        "device_id": device_id,
                        "revision": 1,
                        "subgoal_id": "subgoal-001",
                        "risk_ids": [],
                        "observation_id": "obs-001",
                        "fingerprint": "frame-001",
                    },
                }

        session = Session()
        trial = SimpleNamespace(
            trial_id="trial-api-001",
            device_id=device_id,
            candidate_action="drag",
            session=session,
        )
        trial.snapshot = lambda: {
            "trial_id": trial.trial_id,
            "device_id": trial.device_id,
            "candidate_action": trial.candidate_action,
            "session": session.snapshot(),
            "report": None,
            "promotion_scope": None,
            "promotion": None,
            "requires_restart": False,
        }

        class Manager:
            def __init__(self):
                self.promoted = False

            def start(self, *, device_id, candidate_action, text):
                calls.append(("start", device_id, candidate_action, text))
                return trial

            def get(self, trial_id):
                calls.append(("get", trial_id))
                if trial_id != trial.trial_id:
                    raise web_app.CapabilityAcceptanceError("不存在")
                return trial

            def confirm(self, trial_id, confirmation):
                calls.append(("confirm", trial_id, dict(confirmation)))
                if session.physical_actions:
                    raise web_app.CapabilityAcceptanceError("动作确认已使用")
                session.physical_actions = 1
                session.status = "paused"
                return {
                    "physical_actions": 1,
                    "action_outcome": "matched",
                }

            def promotion_scope(self, trial_id):
                calls.append(("promotion_scope", trial_id))
                return SimpleNamespace(
                    to_dict=lambda: {
                        "trial_id": trial.trial_id,
                        "device_id": trial.device_id,
                        "action": trial.candidate_action,
                        "report_sha256": "a" * 64,
                        "registry_sha256": "b" * 64,
                    }
                )

            def promote(self, trial_id, confirmation):
                calls.append(("promote", trial_id, dict(confirmation)))
                if self.promoted:
                    raise web_app.CapabilityAcceptanceError("晋级确认已使用")
                self.promoted = True
                return {
                    "device_id": trial.device_id,
                    "action": trial.candidate_action,
                    "requires_restart": True,
                }

            def cancel(self, trial_id):
                calls.append(("cancel", trial_id))
                session.status = "cancelled"

        return Manager(), trial, calls

    def test_home_and_device_are_available(self) -> None:
        self.assertEqual(self.client.get("/").status_code, 200)
        device = self.client.get("/api/device").json()
        self.assertTrue(device["controller_online"])
        self.assertTrue(device["camera_online"])
        self.assertEqual(device["default_device_id"], "device-local-01")
        self.assertEqual(device["devices"][0]["device_id"], "device-local-01")
        self.assertNotIn("legacy_free_agent", device)
        architecture = dict(device["execution_architecture"])
        universal = dict(architecture["universal_agent"])
        observer = universal.pop("observer")
        self.assertEqual(
            observer["observer_version"],
            "2026-08-15-generic-scene-observer-v28",
        )
        self.assertEqual(observer["supported_app_scope"], "dynamic")
        architecture["universal_agent"] = universal
        self.assertEqual(
            architecture,
            {
                "model_role": "observation_only",
                "controller": "single_state_controller",
                "legacy_free_agent_enabled": False,
                "active_orchestrator": "universal_agent",
                "background_compatibility_worker": {
                    "enabled": True,
                    "mode": "legacy",
                    "default_user_path": False,
                },
                "universal_agent": {
                    "goal_protocol": "2026-08-10-generic-intent-v1",
                    "scene_protocol": "2026-08-14-ui-scene-v3",
                    "action_protocol": "2026-08-14-universal-action-v10",
                    "goal_preview_enabled": True,
                    "scene_preview_enabled": True,
                    "hardware_execution_enabled": True,
                    "automatic_loop_enabled": False,
                    "automatic_loop_max_physical_actions": 1,
                    "supervised_single_step_enabled": True,
                    "enabled_physical_actions": [
                        "back",
                        "dismiss_overlay",
                        "drag",
                        "home",
                        "input_verified_text",
                        "long_press",
                        "reveal_system_navigation",
                        "swipe",
                        "tap_semantic",
                    ],
                    "protocol_physical_actions": [
                        "tap_semantic",
                        "dismiss_overlay",
                        "swipe",
                        "back",
                        "home",
                        "reveal_system_navigation",
                        "input_verified_text",
                        "long_press",
                        "drag",
                    ],
                    "hardware_capabilities": {
                        "tap_semantic": True,
                        "dismiss_overlay": True,
                        "swipe": True,
                        "back": True,
                        "home": True,
                        "wait_for_change": True,
                        "input_verified_text": True,
                        "long_press": True,
                        "drag": True,
                        "reveal_system_navigation": True,
                    },
                    "supported_app_scope": "dynamic",
                },
                "generic_orchestrator": {
                    "available": True,
                    "execution_enabled": False,
                    "protocol_version": "2026-08-06-generic-plan-v1",
                    "role": "compatibility_only",
                    "default_user_path": False,
                    "allowed_actions": [
                        "back",
                        "dismiss_overlay",
                        "ensure_app",
                        "finish",
                        "input_verified_text",
                        "observe",
                        "record_verified_result",
                        "recover_unknown",
                        "swipe",
                        "tap_semantic",
                        "verify",
                        "wait_for_change",
                    ],
                },
                "semantic_action_adapter": {
                    "role": "compatibility_only",
                    "default_user_path": False,
                    "execution_enabled": True,
                    "enabled_real_actions": [
                        "ensure_app",
                        "tap_semantic:heart",
                        "swipe:up_on_live_preview_or_ad",
                    ],
                    "enabled_read_only_actions": ["observe"],
                    "max_physical_actions_per_request": 1,
                },
            },
        )

    def test_home_uses_generic_supervised_single_step_endpoints(self) -> None:
        home = self.client.get("/")
        script = self.client.get("/assets/app.js")
        protocol_adapter = self.client.get("/assets/protocol_adapter.js")
        styles = self.client.get("/assets/styles.css")
        self.assertEqual(home.status_code, 200)
        self.assertEqual(script.status_code, 200)
        self.assertEqual(protocol_adapter.status_code, 200)
        self.assertEqual(styles.status_code, 200)
        self.assertIn("你希望手机完成什么", home.text)
        self.assertIn("动态计划", home.text)
        self.assertIn("当前画面", home.text)
        self.assertIn("步骤记录", home.text)
        self.assertIn('id="pauseButton"', home.text)
        self.assertIn('id="stopButton"', home.text)
        self.assertIn('id="riskDialog"', home.text)
        self.assertIn('id="capabilityAcceptancePanel"', home.text)
        self.assertIn('id="promotionDialog"', home.text)
        self.assertIn('id="deviceId"', home.text)
        self.assertNotIn("微信工作流", home.text)
        self.assertNotIn("抖音工作流", home.text)
        self.assertIn('/assets/protocol_adapter.js', home.text)
        self.assertIn("/api/agent/generic-supervised/start", script.text)
        self.assertIn("/api/capability-acceptance/start", script.text)
        self.assertIn("createPromotionGrant", protocol_adapter.text)
        self.assertNotIn("/api/agent/generic-supervised/${view.sessionId}/auto", script.text)
        self.assertNotIn('id="confirmSafeLoop"', home.text)
        self.assertIn("nextSupervisedAgent", script.text)
        self.assertIn("togglePause", script.text)
        self.assertNotIn('api("/api/agent/supervised/start"', script.text)
        self.assertNotIn("wechatView", script.text)
        self.assertNotIn("douyinView", script.text)

    def test_capability_revision_must_match_loaded_service_code(self) -> None:
        runtime = web_app.Runtime.__new__(web_app.Runtime)
        runtime.loaded_code_revision = "loaded-revision"

        with patch.object(web_app, "current_code_revision", return_value="loaded-revision"):
            self.assertEqual(runtime.capability_code_revision(), "loaded-revision")
        with (
            patch.object(web_app, "current_code_revision", return_value="new-revision"),
            self.assertRaisesRegex(
                web_app.CapabilityAcceptanceError,
                "服务启动后代码状态发生变化",
            ),
        ):
            runtime.capability_code_revision()

    def test_generic_supervised_auto_request_is_strict_and_bounded(self) -> None:
        request = web_app.GenericSupervisedAutoRequest(device_id="phone-01")
        self.assertFalse(request.confirmed)
        self.assertIsNone(request.confirmation)
        self.assertEqual(request.max_physical_actions, 1)
        self.assertEqual(request.max_iterations, 1)
        with self.assertRaises(ValueError):
            web_app.GenericSupervisedAutoRequest(
                device_id="phone-01",
                max_physical_actions=2,
            )
        with self.assertRaises(ValueError):
            web_app.GenericSupervisedAutoRequest(
                device_id="phone-01",
                max_physical_actions="1",
            )

    def test_capability_acceptance_requests_are_strict(self) -> None:
        with self.assertRaises(ValueError):
            web_app.CapabilityAcceptanceStartRequest(
                device_id="device-a",
                action="drag",
                text="拖动安全控件",
                unexpected="forbidden",
            )
        with self.assertRaises(ValueError):
            web_app.CapabilityActionConfirmationRequest(confirmed="true")
        with self.assertRaises(ValueError):
            web_app.CapabilityPromotionRequest(
                confirmed=True,
                trial_id="trial-a",
                device_id="device-a",
                action="drag",
                report_sha256="not-a-sha",
                registry_sha256="b" * 64,
            )

    def test_capability_acceptance_api_starts_at_zero_and_binds_action_scope(self) -> None:
        manager, trial, calls = self._fake_capability_manager()
        before_executions = list(web_app.runtime.controller.executions)
        with (
            patch.object(web_app.runtime, "capability_acceptance_manager", manager),
            patch.object(web_app, "_require_supervised_device_ready"),
            patch.object(
                web_app,
                "_supervised_hardware_lock",
                side_effect=lambda _device_id: nullcontext(),
            ),
        ):
            missing_token = self.client.post(
                "/api/capability-acceptance/start",
                json={
                    "device_id": trial.device_id,
                    "action": "drag",
                    "text": "拖动安全控件",
                },
            )
            started = self.client.post(
                "/api/capability-acceptance/start",
                headers=self.headers,
                json={
                    "device_id": trial.device_id,
                    "action": "drag",
                    "text": "拖动安全控件",
                },
            )

        self.assertEqual(missing_token.status_code, 403, missing_token.text)
        self.assertEqual(started.status_code, 200, started.text)
        payload = started.json()
        self.assertEqual(payload["physical_actions"], 0)
        scope = payload["trial"]["action_confirmation_scope"]
        self.assertEqual(scope["trial_id"], trial.trial_id)
        self.assertEqual(scope["action"], "drag")
        self.assertEqual([call[0] for call in calls], ["start"])
        self.assertEqual(before_executions, web_app.runtime.controller.executions)

    def test_capability_action_confirmation_is_exact_once(self) -> None:
        manager, trial, calls = self._fake_capability_manager()
        scope = {
            **trial.session.snapshot()["confirmation_scope"],
            "trial_id": trial.trial_id,
            "action": trial.candidate_action,
        }
        path = f"/api/capability-acceptance/{trial.trial_id}/confirm"
        before_executions = list(web_app.runtime.controller.executions)
        with (
            patch.object(web_app.runtime, "capability_acceptance_manager", manager),
            patch.object(web_app, "_require_supervised_device_ready"),
            patch.object(
                web_app,
                "_supervised_hardware_lock",
                side_effect=lambda _device_id: nullcontext(),
            ),
        ):
            first = self.client.post(
                path,
                headers=self.headers,
                json={"confirmed": True, "confirmation": scope},
            )
            replay = self.client.post(
                path,
                headers=self.headers,
                json={"confirmed": True, "confirmation": scope},
            )
            extra = self.client.post(
                path,
                headers=self.headers,
                json={
                    "confirmed": True,
                    "confirmation": {**scope, "unexpected": "forbidden"},
                },
            )

        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(first.json()["physical_actions"], 1)
        self.assertEqual(replay.status_code, 409, replay.text)
        self.assertEqual(extra.status_code, 422, extra.text)
        self.assertEqual([call[0] for call in calls].count("confirm"), 2)
        self.assertEqual(before_executions, web_app.runtime.controller.executions)

    def test_capability_outer_scope_mismatch_is_terminal_without_action(self) -> None:
        manager, trial, calls = self._fake_capability_manager()
        scope = {
            **trial.session.snapshot()["confirmation_scope"],
            "trial_id": trial.trial_id,
            "action": "long_press",
        }
        path = f"/api/capability-acceptance/{trial.trial_id}/confirm"
        with patch.object(web_app.runtime, "capability_acceptance_manager", manager):
            wrong = self.client.post(
                path,
                headers=self.headers,
                json={"confirmed": True, "confirmation": scope},
            )

        self.assertEqual(wrong.status_code, 409, wrong.text)
        self.assertEqual(wrong.json()["detail"]["physical_actions"], 0)
        self.assertEqual(trial.session.physical_actions, 0)
        self.assertEqual(trial.session.status, "cancelled")
        self.assertEqual([call[0] for call in calls].count("confirm"), 0)
        self.assertEqual([call[0] for call in calls].count("cancel"), 1)

    def test_capability_promotion_is_separate_zero_action_and_requires_restart(self) -> None:
        manager, trial, calls = self._fake_capability_manager()
        path = f"/api/capability-acceptance/{trial.trial_id}"
        before_executions = list(web_app.runtime.controller.executions)
        with (
            patch.object(web_app.runtime, "capability_acceptance_manager", manager),
            patch.object(
                web_app.runtime.vision_provider,
                "status",
                side_effect=AssertionError("promotion must not inspect Qwen"),
            ),
        ):
            preview = self.client.get(
                f"{path}/promotion-preview",
                headers=self.headers,
            )
            scope = preview.json()["promotion_scope"]
            refused = self.client.post(
                f"{path}/promote",
                headers=self.headers,
                json={"confirmed": False, **scope},
            )
            promoted = self.client.post(
                f"{path}/promote",
                headers=self.headers,
                json={"confirmed": True, **scope},
            )
            replay = self.client.post(
                f"{path}/promote",
                headers=self.headers,
                json={"confirmed": True, **scope},
            )

        self.assertEqual(preview.status_code, 200, preview.text)
        self.assertEqual(preview.json()["physical_actions"], 0)
        self.assertEqual(refused.status_code, 409, refused.text)
        self.assertEqual(promoted.status_code, 200, promoted.text)
        self.assertEqual(promoted.json()["physical_actions"], 0)
        self.assertTrue(promoted.json()["promotion"]["requires_restart"])
        self.assertEqual(replay.status_code, 409, replay.text)
        self.assertEqual([call[0] for call in calls].count("promote"), 2)
        self.assertEqual(before_executions, web_app.runtime.controller.executions)

    def test_capability_evidence_endpoint_is_token_and_trial_bound(self) -> None:
        manager, trial, _calls = self._fake_capability_manager()
        run_dir = Path(self.temp_dir.name) / "capability-evidence"
        run_dir.mkdir(exist_ok=True)
        frame = run_dir / "before-1.jpg"
        Image.new("RGB", (8, 8), "white").save(frame, format="JPEG")
        outside = Path(self.temp_dir.name) / "outside.jpg"
        Image.new("RGB", (8, 8), "black").save(outside, format="JPEG")
        trial.run_dir = run_dir
        trial.report_path = run_dir / "acceptance_report.json"
        trial.report_path.write_text(
            json.dumps(
                {
                    "before_frame_paths": [str(frame)],
                    "after_frame_paths": [],
                }
            ),
            encoding="utf-8",
        )
        path = f"/api/capability-acceptance/{trial.trial_id}/evidence/before/0"

        with patch.object(web_app.runtime, "capability_acceptance_manager", manager):
            forbidden = self.client.get(path)
            accepted = self.client.get(path, headers=self.headers)
            missing = self.client.get(
                f"/api/capability-acceptance/{trial.trial_id}/evidence/before/1",
                headers=self.headers,
            )
            trial.report_path.write_text(
                json.dumps(
                    {
                        "before_frame_paths": [str(outside)],
                        "after_frame_paths": [],
                    }
                ),
                encoding="utf-8",
            )
            escaped = self.client.get(path, headers=self.headers)

        self.assertEqual(forbidden.status_code, 403, forbidden.text)
        self.assertEqual(accepted.status_code, 200, accepted.text)
        self.assertEqual(accepted.headers["content-type"], "image/jpeg")
        self.assertEqual(missing.status_code, 404, missing.text)
        self.assertEqual(escaped.status_code, 404, escaped.text)

    def test_v3_confirm_request_requires_scope_and_forbids_extra_fields(self) -> None:
        orchestrator, _planner, _qwen, adapter = self._universal_api_orchestrator()
        session = orchestrator.start(
            session_id="api-scope",
            raw_goal="查看详情",
            device_id="phone-01",
            run_dir=web_app.WEB_OUTPUT_DIR / "api-scope",
        )
        with web_app.runtime.generic_supervised_session_lock:
            web_app.runtime.generic_supervised_sessions[session.session_id] = session
        path = f"/api/agent/generic-supervised/{session.session_id}/confirm"
        scope = session.snapshot()["confirmation_scope"]

        with (
            patch.object(web_app, "_require_supervised_device_ready"),
            patch.object(
                web_app.runtime,
                "universal_agent_orchestrator",
                orchestrator,
            ),
        ):
            missing = self.client.post(
                path,
                headers=self.headers,
                json={"confirmed": True},
            )
        self.assertEqual(missing.status_code, 409, missing.text)
        self.assertEqual(missing.json()["detail"]["physical_actions"], 0)
        self.assertEqual(adapter.execute_calls, 0)

        with self.assertRaises(ValueError):
            web_app.GenericSupervisedStepRequest(
                confirmed="true",
                confirmation=scope,
            )

        for payload in (
            {
                "confirmed": True,
                "confirmation": scope,
                "unexpected": "forbidden",
            },
            {
                "confirmed": True,
                "confirmation": {
                    **scope,
                    "unexpected": "forbidden",
                },
            },
        ):
            with self.subTest(payload=payload):
                with patch.object(web_app, "_require_supervised_device_ready"):
                    rejected = self.client.post(
                        path,
                        headers=self.headers,
                        json=payload,
                    )
                self.assertEqual(rejected.status_code, 422, rejected.text)
                self.assertEqual(adapter.execute_calls, 0)

    def test_v3_confirm_api_atomically_consumes_one_scope(self) -> None:
        orchestrator, _planner, _qwen, adapter = self._universal_api_orchestrator()
        session = orchestrator.start(
            session_id="api-confirm-once",
            raw_goal="查看详情",
            device_id="phone-01",
            run_dir=web_app.WEB_OUTPUT_DIR / "api-confirm-once",
        )
        with web_app.runtime.generic_supervised_session_lock:
            web_app.runtime.generic_supervised_sessions[session.session_id] = session
        path = f"/api/agent/generic-supervised/{session.session_id}/confirm"
        payload = {
            "confirmed": True,
            "confirmation": session.snapshot()["confirmation_scope"],
        }

        with (
            patch.object(web_app, "_require_supervised_device_ready"),
            patch.object(
                web_app.runtime,
                "universal_agent_orchestrator",
                orchestrator,
            ),
        ):
            first = self.client.post(path, headers=self.headers, json=payload)
            replay = self.client.post(path, headers=self.headers, json=payload)

        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(first.json()["execution"]["physical_actions"], 1)
        self.assertEqual(first.json()["session"]["task_graph"]["revision"], 2)
        self.assertEqual(len(first.json()["execution"]["after_frame_paths"]), 4)
        self.assertEqual(replay.status_code, 409, replay.text)
        self.assertEqual(replay.json()["detail"]["physical_actions"], 0)
        self.assertEqual(adapter.execute_calls, 1)

    def test_device_disconnect_invalidates_pending_confirmation(self) -> None:
        orchestrator, _planner, _qwen, adapter = self._universal_api_orchestrator()
        session = orchestrator.start(
            session_id="api-disconnect-invalidates",
            raw_goal="查看详情",
            device_id="phone-01",
            run_dir=web_app.WEB_OUTPUT_DIR / "api-disconnect-invalidates",
        )
        with web_app.runtime.generic_supervised_session_lock:
            web_app.runtime.generic_supervised_sessions[session.session_id] = session
        path = f"/api/agent/generic-supervised/{session.session_id}/confirm"
        payload = {
            "confirmed": True,
            "confirmation": session.snapshot()["confirmation_scope"],
        }

        def offline(_device_id=None) -> None:
            raise web_app.HTTPException(status_code=409, detail="控制端或摄像头离线。")

        with (
            patch.object(web_app, "_require_supervised_device_ready", offline),
            patch.object(
                web_app.runtime,
                "universal_agent_orchestrator",
                orchestrator,
            ),
        ):
            disconnected = self.client.post(
                path, headers=self.headers, json=payload
            )

        self.assertEqual(409, disconnected.status_code, disconnected.text)
        self.assertIsNone(session.snapshot()["confirmation_scope"])
        self.assertEqual("needs_reobservation", session.status)
        self.assertEqual(0, adapter.execute_calls)

        with (
            patch.object(web_app, "_require_supervised_device_ready"),
            patch.object(
                web_app.runtime,
                "universal_agent_orchestrator",
                orchestrator,
            ),
        ):
            stale = self.client.post(path, headers=self.headers, json=payload)
        self.assertEqual(409, stale.status_code, stale.text)
        self.assertEqual(0, adapter.execute_calls)

    def test_hardware_lock_rejects_another_process_lease(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            lease_dir = Path(temp)
            external = InterProcessLease(
                lease_dir / "physical_hardware_action.lease",
                owner_id="other-service",
                metadata={"purpose": "physical_hardware_action"},
            )
            self.assertTrue(external.acquire())
            try:
                with (
                    patch.object(web_app, "SHARED_DEVICE_LEASE_DIR", lease_dir),
                    self.assertRaisesRegex(
                        web_app.HTTPException, "另一进程已占用"
                    ),
                ):
                    with web_app._supervised_hardware_lock():
                        self.fail("cross-process lease must block the hardware lock")
            finally:
                external.release()

    def test_hardware_locks_allow_different_devices_but_reject_same_device(self) -> None:
        first_controller = SimpleNamespace(operation_lock=threading.Lock())
        second_controller = SimpleNamespace(operation_lock=threading.Lock())
        controllers = {
            "phone-a": first_controller,
            "phone-b": second_controller,
        }
        with tempfile.TemporaryDirectory() as temp:
            with (
                patch.object(web_app, "SHARED_DEVICE_LEASE_DIR", Path(temp)),
                patch.object(
                    web_app.runtime,
                    "controller_for_device",
                    side_effect=lambda device_id: controllers[device_id],
                ),
            ):
                with web_app._supervised_hardware_lock("phone-a"):
                    with web_app._supervised_hardware_lock("phone-b"):
                        self.assertTrue(first_controller.operation_lock.locked())
                        self.assertTrue(second_controller.operation_lock.locked())
                    with self.assertRaisesRegex(
                        web_app.HTTPException,
                        "占用|正在进行",
                    ):
                        with web_app._supervised_hardware_lock("phone-a"):
                            self.fail("同一设备不能取得第二个硬件锁")

        self.assertFalse(first_controller.operation_lock.locked())
        self.assertFalse(second_controller.operation_lock.locked())

    def test_compatibility_physical_endpoints_share_the_process_lease(self) -> None:
        lease = InterProcessLease(
            web_app.SHARED_DEVICE_LEASE_DIR / "physical_hardware_action.lease",
            owner_id="other-worktree-service",
            metadata={"purpose": "test_external_owner"},
        )
        self.assertTrue(lease.acquire())
        try:
            before_executions = list(web_app.runtime.controller.executions)
            for endpoint, text in (
                ("/api/agent/execute-ensure-app-step", "打开设置"),
                ("/api/agent/execute-observe-step", "查看设置"),
                ("/api/agent/execute-tap-heart-step", "点赞当前视频"),
            ):
                with self.subTest(endpoint=endpoint):
                    response = self.client.post(
                        endpoint,
                        headers=self.headers,
                        json={"confirmed": True, "text": text},
                    )
                    self.assertEqual(409, response.status_code, response.text)
                    self.assertIn("另一进程已占用", response.text)
            self.assertEqual(
                before_executions, web_app.runtime.controller.executions
            )
        finally:
            lease.release()

    def test_capability_trial_blocks_every_compatibility_hardware_entry(self) -> None:
        from universal_agent_orchestrator import DeviceTaskRegistry

        device_id = web_app.runtime.device_controllers.default_device_id
        isolated_registry = DeviceTaskRegistry()
        with patch.object(
            web_app.runtime,
            "device_task_registry",
            isolated_registry,
        ):
            isolated_registry.reserve(device_id, "capability-trial-test")
            try:
                before_executions = list(web_app.runtime.controller.executions)
                for endpoint, text in (
                    ("/api/agent/execute-ensure-app-step", "打开设置"),
                    ("/api/agent/execute-observe-step", "查看设置"),
                    ("/api/agent/execute-tap-heart-step", "点赞当前视频"),
                ):
                    with self.subTest(endpoint=endpoint):
                        response = self.client.post(
                            endpoint,
                            headers=self.headers,
                            json={"confirmed": True, "text": text},
                        )
                        self.assertEqual(409, response.status_code, response.text)
                        self.assertIn("已有活动任务", response.text)
                self.assertEqual(before_executions, web_app.runtime.controller.executions)
            finally:
                isolated_registry.release(device_id, "capability-trial-test")

    def test_legacy_worker_declares_the_same_cross_process_action_lease(self) -> None:
        import inspect

        source = inspect.getsource(web_app.Runtime._worker_loop)
        self.assertIn("physical_hardware_action.lease", source)
        self.assertIn("process_lease.acquire()", source)
        self.assertIn("process_lease.release()", source)
        self.assertIn("self.device_task_registry.reserve", source)
        self.assertIn("self.device_task_registry.release", source)

    def test_v3_confirm_api_rejects_cross_device_scope(self) -> None:
        orchestrator, _planner, _qwen, adapter = self._universal_api_orchestrator()
        session = orchestrator.start(
            session_id="api-cross-device",
            raw_goal="查看详情",
            device_id="phone-01",
            run_dir=web_app.WEB_OUTPUT_DIR / "api-cross-device",
        )
        with web_app.runtime.generic_supervised_session_lock:
            web_app.runtime.generic_supervised_sessions[session.session_id] = session
        scope = session.snapshot()["confirmation_scope"]
        scope["device_id"] = "phone-02"

        with (
            patch.object(web_app, "_require_supervised_device_ready"),
            patch.object(
                web_app.runtime,
                "universal_agent_orchestrator",
                orchestrator,
            ),
        ):
            response = self.client.post(
                f"/api/agent/generic-supervised/{session.session_id}/confirm",
                headers=self.headers,
                json={"confirmed": True, "confirmation": scope},
            )
        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(response.json()["detail"]["physical_actions"], 0)
        self.assertEqual(adapter.execute_calls, 0)

    def test_plan_preview_compiles_without_creating_or_running_task(self) -> None:
        before = len(web_app.runtime.store.list(100))
        before_executions = len(web_app.runtime.controller.executions)
        parsed = {
            "understood": True,
            "app_id": "douyin",
            "operation": "douyin.batch_interact",
            "params": {
                "keyword": None,
                "target_count": 3,
                "like": True,
                "comment": False,
            },
            "summary": "点赞三个普通视频",
            "needs_confirmation": True,
            "provider": "test",
        }
        with patch.object(web_app.runtime.agent, "parse", return_value=parsed):
            response = self.client.post(
                "/api/agent/plan-preview",
                json={"text": "打开抖音，给三个普通视频点赞"},
            )
        self.assertEqual(response.status_code, 200)
        preview = response.json()
        self.assertTrue(preview["compiled"])
        self.assertFalse(preview["execution_enabled"])
        self.assertEqual(preview["goal"]["success_criteria"]["target_count"], 3)
        self.assertEqual(len(web_app.runtime.store.list(100)), before)
        self.assertEqual(
            len(web_app.runtime.controller.executions),
            before_executions,
        )

    def test_real_observation_dry_run_never_executes_or_creates_task(self) -> None:
        before_tasks = len(web_app.runtime.store.list(100))
        before_executions = len(web_app.runtime.controller.executions)
        parsed = {
            "understood": True,
            "app_id": "douyin",
            "operation": "douyin.batch_interact",
            "params": {
                "keyword": None,
                "target_count": 3,
                "like": True,
                "comment": False,
            },
            "summary": "点赞三个普通视频",
            "needs_confirmation": True,
            "provider": "test",
        }

        class Observer:
            def observe(self, **_kwargs):
                return StatePageObservation(
                    state="android_home",
                    base_state="android_home",
                    confidence=0.96,
                    stable=True,
                    reason="桌面稳定",
                )

        with (
            patch.object(web_app.runtime.agent, "parse", return_value=parsed),
            patch.object(
                web_app.runtime.vision_provider,
                "status",
                return_value={"configured": True},
            ),
            patch.object(web_app.runtime, "state_observer", Observer()),
        ):
            response = self.client.post(
                "/api/agent/dry-run-step",
                headers=self.headers,
                json={"text": "打开抖音，给三个普通视频点赞"},
            )
        self.assertEqual(response.status_code, 200, response.text)
        preview = response.json()
        self.assertFalse(preview["executed"])
        self.assertFalse(preview["execution_enabled"])
        self.assertEqual(preview["captured_frames"], 4)
        self.assertEqual(preview["observation"]["page_state"], "android_home")
        self.assertEqual(preview["decision"]["status"], "action")
        self.assertEqual(
            preview["decision"]["action"]["action"],
            "ensure_app",
        )
        self.assertEqual(len(web_app.runtime.store.list(100)), before_tasks)
        self.assertEqual(
            len(web_app.runtime.controller.executions),
            before_executions,
        )

    def test_generic_scene_preview_is_read_only(self) -> None:
        scene = UIScene(
            app_id="calculator",
            screen_id="app_home",
            summary="计算器首页",
            stable=True,
            confidence=0.96,
            fingerprint="local-frame",
        )
        before_executions = len(web_app.runtime.controller.executions)
        with (
            patch.object(
                web_app.runtime.vision_provider,
                "status",
                return_value={"configured": True},
            ),
            patch.object(
                web_app.runtime.generic_scene_observer,
                "observe",
                return_value=scene,
            ),
        ):
            response = self.client.post(
                "/api/agent/generic-scene",
                headers=self.headers,
                json={"goal": {"objective": "在计算器输入7"}},
            )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertFalse(payload["executed"])
        self.assertFalse(payload["physical_action_requested"])
        self.assertEqual(payload["scene"]["app_id"], "calculator")
        self.assertEqual(
            len(web_app.runtime.controller.executions),
            before_executions,
        )

    def test_generic_supervised_api_starts_unexecuted_and_requires_confirmation(self) -> None:
        orchestrator, planner, qwen, adapter = self._universal_api_orchestrator()
        before_executions = len(web_app.runtime.controller.executions)
        with (
            patch.object(web_app, "_require_supervised_device_ready"),
            patch.object(
                web_app.runtime,
                "universal_agent_orchestrator",
                orchestrator,
            ),
            patch.object(
                web_app.runtime.generic_intent_parser,
                "parse",
                side_effect=AssertionError("legacy parser must not run"),
            ),
            patch.object(
                web_app.runtime.generic_step_planner,
                "propose",
                side_effect=AssertionError("legacy step planner must not run"),
            ),
        ):
            started = self.client.post(
                "/api/agent/generic-supervised/start",
                headers=self.headers,
                json={"text": "查看当前页面的详情", "device_id": "phone-01"},
            )
            self.assertEqual(started.status_code, 200, started.text)
            payload = started.json()
            session_id = payload["session"]["session_id"]
            self.assertEqual(payload["physical_actions"], 0)
            self.assertEqual(
                payload["session"]["status"], "awaiting_confirmation"
            )
            self.assertEqual(len(planner.plan_calls), 1)
            self.assertEqual(len(qwen.calls), 1)
            self.assertEqual(adapter.capture_calls, 1)
            self.assertEqual(adapter.execute_calls, 0)

            rejected = self.client.post(
                f"/api/agent/generic-supervised/{session_id}/confirm",
                headers=self.headers,
                json={"confirmed": False},
            )
            self.assertEqual(rejected.status_code, 409, rejected.text)
            self.assertEqual(rejected.json()["detail"]["physical_actions"], 0)

            cancelled = self.client.post(
                f"/api/agent/generic-supervised/{session_id}/cancel",
                headers=self.headers,
                json={"device_id": "phone-01"},
            )
        self.assertEqual(cancelled.status_code, 200, cancelled.text)
        self.assertEqual(cancelled.json()["session"]["status"], "cancelled")
        self.assertEqual(
            len(web_app.runtime.controller.executions),
            before_executions,
        )

    def test_external_state_requires_risk_approval_before_qwen_or_robot(self) -> None:
        from test_universal_agent_orchestrator import _external_graph

        orchestrator, _planner, qwen, adapter = self._universal_api_orchestrator(
            device_id="device-1",
            graph=_external_graph(),
        )
        with (
            patch.object(web_app, "_require_supervised_device_ready"),
            patch.object(
                web_app.runtime,
                "universal_agent_orchestrator",
                orchestrator,
            ),
        ):
            response = self.client.post(
                "/api/agent/generic-supervised/start",
                headers=self.headers,
                json={
                    "text": "向联系人发送一条消息",
                    "device_id": "device-1",
                },
            )
            self.assertEqual(response.status_code, 200, response.text)
            session = response.json()["session"]
            self.assertEqual(session["status"], "awaiting_risk_confirmation")
            self.assertEqual(response.json()["physical_actions"], 0)
            self.assertEqual(len(qwen.calls), 0)
            self.assertEqual(adapter.capture_calls, 0)
            self.assertEqual(adapter.execute_calls, 0)

            approved = self.client.post(
                f"/api/agent/generic-supervised/{session['session_id']}/approve-risk",
                headers=self.headers,
                json={
                    "confirmed": True,
                    "confirmation": session["risk_confirmation_scope"],
                },
            )

        self.assertEqual(approved.status_code, 200, approved.text)
        self.assertEqual(approved.json()["session"]["status"], "awaiting_confirmation")
        self.assertEqual(approved.json()["physical_actions"], 0)
        self.assertEqual(len(qwen.calls), 1)
        self.assertEqual(adapter.capture_calls, 1)
        self.assertEqual(adapter.execute_calls, 0)

    def test_same_device_second_generic_session_returns_409(self) -> None:
        orchestrator, planner, qwen, adapter = self._universal_api_orchestrator()
        with tempfile.TemporaryDirectory() as temp:
            with (
                patch.object(web_app, "WEB_OUTPUT_DIR", Path(temp)),
                patch.object(web_app, "_require_supervised_device_ready"),
                patch.object(
                    web_app.runtime,
                    "universal_agent_orchestrator",
                    orchestrator,
                ),
            ):
                first = self.client.post(
                    "/api/agent/generic-supervised/start",
                    headers=self.headers,
                    json={"text": "查看详情", "device_id": "phone-01"},
                )
                directories_after_first = sorted(Path(temp).iterdir())
                second = self.client.post(
                    "/api/agent/generic-supervised/start",
                    headers=self.headers,
                    json={"text": "返回上一页", "device_id": "phone-01"},
                )
                directories_after_second = sorted(Path(temp).iterdir())
                session_id = first.json()["session"]["session_id"]
                self.client.post(
                    f"/api/agent/generic-supervised/{session_id}/cancel",
                    headers=self.headers,
                    json={"device_id": "phone-01"},
                )

        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(second.status_code, 409, second.text)
        self.assertEqual(second.json()["detail"]["physical_actions"], 0)
        self.assertEqual(directories_after_second, directories_after_first)
        self.assertEqual(len(directories_after_first), 1)
        self.assertEqual(len(planner.plan_calls), 1)
        self.assertEqual(len(qwen.calls), 1)
        self.assertEqual(adapter.capture_calls, 1)
        self.assertEqual(adapter.execute_calls, 0)

    def test_next_reobserves_then_bounded_auto_executes_one_verified_action(self) -> None:
        orchestrator, _planner, _qwen, adapter = self._universal_api_orchestrator()
        with (
            patch.object(web_app, "_require_supervised_device_ready"),
            patch.object(
                web_app.runtime,
                "universal_agent_orchestrator",
                orchestrator,
            ),
        ):
            started = self.client.post(
                "/api/agent/generic-supervised/start",
                headers=self.headers,
                json={"text": "查看详情", "device_id": "phone-01"},
            )
            session_id = started.json()["session"]["session_id"]
            first_observation = started.json()["session"]["confirmation_scope"][
                "observation_id"
            ]
            refreshed = self.client.post(
                f"/api/agent/generic-supervised/{session_id}/next",
                headers=self.headers,
                json={"device_id": "phone-01"},
            )
            automatic = self.client.post(
                f"/api/agent/generic-supervised/{session_id}/auto",
                headers=self.headers,
                json={
                    "device_id": "phone-01",
                    "confirmed": True,
                    "confirmation": refreshed.json()["session"]["confirmation_scope"],
                    "max_physical_actions": 1,
                    "max_iterations": 1,
                },
            )
            self.client.post(
                f"/api/agent/generic-supervised/{session_id}/cancel",
                headers=self.headers,
                json={"device_id": "phone-01"},
            )

        self.assertEqual(refreshed.status_code, 200, refreshed.text)
        self.assertEqual(refreshed.json()["physical_actions"], 0)
        self.assertNotEqual(
            first_observation,
            refreshed.json()["session"]["confirmation_scope"]["observation_id"],
        )
        self.assertEqual(automatic.status_code, 200, automatic.text)
        self.assertEqual(automatic.json()["execution"]["physical_actions"], 1)
        self.assertEqual(automatic.json()["session"]["physical_actions"], 1)
        self.assertEqual(adapter.capture_calls, 2)
        self.assertEqual(adapter.execute_calls, 1)

    def test_supervised_session_starts_paused_and_requires_confirmation(self) -> None:
        before_tasks = len(web_app.runtime.store.list(100))
        before_executions = len(web_app.runtime.controller.executions)
        parsed = {
            "understood": True,
            "app_id": "douyin",
            "operation": "douyin.batch_interact",
            "params": {
                "keyword": None,
                "target_count": 1,
                "like": True,
                "comment": False,
            },
            "summary": "给当前普通视频点赞",
            "needs_confirmation": True,
            "provider": "test",
        }

        class Observer:
            def observe(self, **_kwargs):
                return StatePageObservation(
                    state="android_home",
                    base_state="android_home",
                    confidence=0.98,
                    stable=True,
                    reason="桌面稳定",
                )

        with (
            patch.object(web_app.runtime.agent, "parse", return_value=parsed),
            patch.object(
                web_app.runtime.vision_provider,
                "status",
                return_value={"configured": True},
            ),
            patch.object(web_app.runtime, "state_observer", Observer()),
        ):
            started = self.client.post(
                "/api/agent/supervised/start",
                headers=self.headers,
                json={"text": "给当前普通视频点赞"},
            )
        self.assertEqual(started.status_code, 200, started.text)
        payload = started.json()
        self.assertEqual(payload["physical_actions"], 0)
        self.assertEqual(payload["session"]["status"], "action")
        self.assertEqual(
            payload["session"]["decision"]["action"]["action"],
            "ensure_app",
        )
        self.assertEqual(len(web_app.runtime.store.list(100)), before_tasks)
        self.assertEqual(
            len(web_app.runtime.controller.executions),
            before_executions,
        )

        session_id = payload["session"]["session_id"]
        refused = self.client.post(
            f"/api/agent/supervised/{session_id}/step",
            headers=self.headers,
            json={"confirmed": False},
        )
        self.assertEqual(refused.status_code, 409, refused.text)
        self.assertEqual(
            len(web_app.runtime.controller.executions),
            before_executions,
        )

    def test_supervised_confirmation_executes_only_one_pending_node(self) -> None:
        before_executions = len(web_app.runtime.controller.executions)
        parsed = {
            "understood": True,
            "app_id": "douyin",
            "operation": "douyin.batch_interact",
            "params": {
                "keyword": None,
                "target_count": 1,
                "like": True,
                "comment": False,
            },
            "summary": "给当前普通视频点赞",
            "needs_confirmation": True,
            "provider": "test",
        }

        class Observer:
            def __init__(self):
                self.values = [
                    StatePageObservation(
                        state="android_home",
                        base_state="android_home",
                        confidence=0.99,
                        stable=True,
                        reason="初始桌面",
                    ),
                    StatePageObservation(
                        state="android_home",
                        base_state="android_home",
                        confidence=0.99,
                        stable=True,
                        reason="点击前桌面",
                        targets={"douyin_icon": (800, 700)},
                        target_bounds={"douyin_icon": (740, 640, 860, 760)},
                    ),
                    StatePageObservation(
                        state="douyin_home",
                        base_state="douyin_home",
                        confidence=0.98,
                        stable=True,
                        reason="已打开抖音",
                    ),
                ]

            def observe(self, **_kwargs):
                return self.values.pop(0)

        resolution = TargetResolution(
            target="douyin_icon",
            method="test_guarded_target",
            proposed_coordinate=(800, 700),
            resolved_coordinate=(800, 700),
            pixel_center=(432, 672),
            samples=4,
            spread_px=0.0,
            detail="test",
        )
        with (
            patch.object(web_app.runtime.agent, "parse", return_value=parsed),
            patch.object(
                web_app.runtime.vision_provider,
                "status",
                return_value={"configured": True},
            ),
            patch.object(web_app.runtime, "state_observer", Observer()),
            patch(
                "semantic_action_adapter.LocalTargetResolver.resolve",
                return_value=resolution,
            ),
            patch("semantic_action_adapter.time.sleep", return_value=None),
        ):
            started = self.client.post(
                "/api/agent/supervised/start",
                headers=self.headers,
                json={"text": "给当前普通视频点赞"},
            )
            self.assertEqual(started.status_code, 200, started.text)
            session_id = started.json()["session"]["session_id"]
            advanced = self.client.post(
                f"/api/agent/supervised/{session_id}/step",
                headers=self.headers,
                json={"confirmed": True},
            )

        self.assertEqual(advanced.status_code, 200, advanced.text)
        payload = advanced.json()
        self.assertEqual(payload["last_step"]["step"], 1)
        self.assertEqual(
            payload["last_step"]["decision_before"]["action"]["action"],
            "ensure_app",
        )
        self.assertEqual(
            payload["session"]["decision"]["action"]["action"],
            "observe",
        )
        self.assertEqual(payload["session"]["step_count"], 1)
        new_executions = web_app.runtime.controller.executions[before_executions:]
        self.assertEqual(
            new_executions,
            [{"action": "tap", "coordinate": [800, 700]}],
        )

    def test_confirmed_ensure_app_step_taps_once_and_never_creates_task(self) -> None:
        before_tasks = len(web_app.runtime.store.list(100))
        before_executions = len(web_app.runtime.controller.executions)
        parsed = {
            "understood": True,
            "app_id": "douyin",
            "operation": "douyin.search",
            "params": {"keyword": "机械臂"},
            "summary": "打开抖音并准备搜索机械臂",
            "needs_confirmation": True,
            "provider": "test",
        }

        class Observer:
            def __init__(self):
                self.index = 0

            def observe(self, **_kwargs):
                values = [
                    StatePageObservation(
                        state="android_home",
                        base_state="android_home",
                        confidence=0.99,
                        stable=True,
                        reason="桌面稳定",
                        targets={"douyin_icon": (800, 700)},
                        target_bounds={"douyin_icon": (740, 640, 860, 760)},
                    ),
                    StatePageObservation(
                        state="douyin_home",
                        base_state="douyin_home",
                        confidence=0.97,
                        stable=True,
                        reason="抖音首页稳定",
                    ),
                ]
                value = values[self.index]
                self.index += 1
                return value

        resolution = TargetResolution(
            target="douyin_icon",
            method="test_guarded_target",
            proposed_coordinate=(800, 700),
            resolved_coordinate=(800, 700),
            pixel_center=(432, 672),
            samples=4,
            spread_px=0.0,
            detail="test",
        )
        with (
            patch.object(web_app.runtime.agent, "parse", return_value=parsed),
            patch.object(
                web_app.runtime.vision_provider,
                "status",
                return_value={"configured": True},
            ),
            patch.object(web_app.runtime, "state_observer", Observer()),
            patch(
                "semantic_action_adapter.LocalTargetResolver.resolve",
                return_value=resolution,
            ),
            patch("semantic_action_adapter.time.sleep", return_value=None),
        ):
            response = self.client.post(
                "/api/agent/execute-ensure-app-step",
                headers=self.headers,
                json={
                    "text": "打开抖音并搜索机械臂",
                    "confirmed": True,
                },
            )
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertTrue(result["execution_success"])
        self.assertTrue(result["executed"])
        self.assertEqual(result["result"]["before"]["state"], "android_home")
        self.assertEqual(result["result"]["after"]["state"], "douyin_home")
        self.assertEqual(result["next_decision"]["action"]["action"], "observe")
        self.assertEqual(len(web_app.runtime.store.list(100)), before_tasks)
        self.assertEqual(
            len(web_app.runtime.controller.executions),
            before_executions + 1,
        )

    def test_ensure_app_step_requires_explicit_confirmation(self) -> None:
        before_executions = len(web_app.runtime.controller.executions)
        response = self.client.post(
            "/api/agent/execute-ensure-app-step",
            headers=self.headers,
            json={"text": "打开抖音", "confirmed": False},
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(
            len(web_app.runtime.controller.executions),
            before_executions,
        )

    def test_confirmed_observe_step_classifies_without_physical_action(self) -> None:
        before_tasks = len(web_app.runtime.store.list(100))
        before_executions = len(web_app.runtime.controller.executions)
        parsed = {
            "understood": True,
            "app_id": "douyin",
            "operation": "douyin.batch_interact",
            "params": {
                "keyword": None,
                "target_count": 3,
                "like": True,
                "comment": False,
            },
            "summary": "给三个普通视频点赞",
            "needs_confirmation": True,
            "provider": "test",
        }

        class Observer:
            def observe(self, **_kwargs):
                return StatePageObservation(
                    state="douyin_video",
                    base_state="douyin_video",
                    confidence=0.99,
                    stable=True,
                    reason="普通视频稳定",
                    heart_state="unliked",
                )

        with (
            patch.object(web_app.runtime.agent, "parse", return_value=parsed),
            patch.object(
                web_app.runtime.vision_provider,
                "status",
                return_value={"configured": True},
            ),
            patch.object(web_app.runtime, "state_observer", Observer()),
        ):
            response = self.client.post(
                "/api/agent/execute-observe-step",
                headers=self.headers,
                json={
                    "text": "打开抖音，给接下来的三个视频点赞",
                    "confirmed": True,
                },
            )
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["physical_actions"], 0)
        self.assertEqual(result["result"]["classification"], "ordinary_video")
        self.assertTrue(result["result"]["safe_for_next_action"])
        self.assertFalse(result["result"]["robot_action_called"])
        self.assertEqual(
            result["next_decision"]["action"]["action"],
            "tap_semantic",
        )
        self.assertEqual(len(web_app.runtime.store.list(100)), before_tasks)
        self.assertEqual(
            len(web_app.runtime.controller.executions),
            before_executions,
        )

    def test_observe_step_requires_explicit_confirmation(self) -> None:
        before_executions = len(web_app.runtime.controller.executions)
        response = self.client.post(
            "/api/agent/execute-observe-step",
            headers=self.headers,
            json={"text": "观察当前抖音页面", "confirmed": False},
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(
            len(web_app.runtime.controller.executions),
            before_executions,
        )

    @staticmethod
    def _tap_heart_parsed_goal() -> dict:
        return {
            "understood": True,
            "app_id": "douyin",
            "operation": "douyin.batch_interact",
            "params": {
                "keyword": None,
                "target_count": 10,
                "like": True,
                "comment": False,
            },
            "summary": "给接下来的十个普通视频点赞",
            "needs_confirmation": True,
            "provider": "test",
        }

    @staticmethod
    def _heart_observation(state: str) -> StatePageObservation:
        return StatePageObservation(
            state="douyin_video",
            base_state="douyin_video",
            confidence=0.99,
            stable=True,
            reason=f"普通视频，爱心={state}",
            heart_state=state,
            targets={"heart": (470, 450)},
            target_bounds={"heart": (440, 420, 500, 480)},
        )

    def test_confirmed_tap_heart_step_clicks_once_and_stops_before_swipe(self) -> None:
        before_tasks = len(web_app.runtime.store.list(100))
        before_executions = len(web_app.runtime.controller.executions)

        class Observer:
            def __init__(self):
                self.values = [
                    ApiEndToEndTests._heart_observation("unliked"),
                    ApiEndToEndTests._heart_observation("liked"),
                ]

            def observe(self, **_kwargs):
                return self.values.pop(0)

        resolutions = [
            TargetResolution(
                target="heart",
                method="test_guarded_target",
                proposed_coordinate=(470, 450),
                resolved_coordinate=(470, 450),
                pixel_center=(470, 450),
                samples=4,
                spread_px=0.0,
                detail="white heart",
                verified_state="unliked",
            ),
            TargetResolution(
                target="heart",
                method="test_guarded_target",
                proposed_coordinate=(470, 450),
                resolved_coordinate=(470, 450),
                pixel_center=(470, 450),
                samples=4,
                spread_px=0.0,
                detail="red heart",
                verified_state="liked",
            ),
        ]
        with (
            patch.object(
                web_app.runtime.agent,
                "parse",
                return_value=self._tap_heart_parsed_goal(),
            ),
            patch.object(
                web_app.runtime.vision_provider,
                "status",
                return_value={"configured": True},
            ),
            patch.object(web_app.runtime, "state_observer", Observer()),
            patch(
                "semantic_action_adapter.LocalTargetResolver.resolve",
                side_effect=resolutions,
            ),
            patch("semantic_action_adapter.time.sleep", return_value=None),
        ):
            response = self.client.post(
                "/api/agent/execute-tap-heart-step",
                headers=self.headers,
                json={
                    "text": "打开抖音，给接下来的十个视频点赞",
                    "confirmed": True,
                },
            )
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertTrue(result["execution_success"])
        self.assertEqual(result["physical_actions"], 1)
        self.assertEqual(result["result"]["before"]["heart_state"], "unliked")
        self.assertEqual(result["result"]["after"]["heart_state"], "liked")
        self.assertEqual(
            result["next_decision"]["action"]["action"],
            "record_verified_result",
        )
        self.assertEqual(result["safety"]["retry_count"], 0)
        self.assertFalse(result["safety"]["swipe_performed"])
        self.assertEqual(len(web_app.runtime.store.list(100)), before_tasks)
        new_executions = web_app.runtime.controller.executions[before_executions:]
        self.assertEqual(new_executions, [{"action": "tap", "coordinate": [470, 450]}])

    def test_tap_heart_step_requires_explicit_confirmation(self) -> None:
        before_executions = len(web_app.runtime.controller.executions)
        response = self.client.post(
            "/api/agent/execute-tap-heart-step",
            headers=self.headers,
            json={"text": "给当前视频点赞", "confirmed": False},
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(len(web_app.runtime.controller.executions), before_executions)

    def test_tap_heart_step_already_liked_performs_zero_actions(self) -> None:
        before_executions = len(web_app.runtime.controller.executions)

        class Observer:
            def observe(self, **_kwargs):
                return ApiEndToEndTests._heart_observation("liked")

        with (
            patch.object(
                web_app.runtime.agent,
                "parse",
                return_value=self._tap_heart_parsed_goal(),
            ),
            patch.object(
                web_app.runtime.vision_provider,
                "status",
                return_value={"configured": True},
            ),
            patch.object(web_app.runtime, "state_observer", Observer()),
        ):
            response = self.client.post(
                "/api/agent/execute-tap-heart-step",
                headers=self.headers,
                json={"text": "给当前视频点赞", "confirmed": True},
            )
        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(response.json()["detail"]["physical_actions"], 0)
        self.assertEqual(len(web_app.runtime.controller.executions), before_executions)

    def test_tap_heart_post_failure_records_one_action_without_retry(self) -> None:
        before_executions = len(web_app.runtime.controller.executions)

        class Observer:
            def __init__(self):
                self.values = [
                    ApiEndToEndTests._heart_observation("unliked"),
                    ApiEndToEndTests._heart_observation("unliked"),
                ]

            def observe(self, **_kwargs):
                return self.values.pop(0)

        resolutions = [
            TargetResolution(
                target="heart",
                method="test_guarded_target",
                proposed_coordinate=(470, 450),
                resolved_coordinate=(470, 450),
                pixel_center=(470, 450),
                samples=4,
                spread_px=0.0,
                detail="white before",
                verified_state="unliked",
            ),
            TargetResolution(
                target="heart",
                method="test_guarded_target",
                proposed_coordinate=(470, 450),
                resolved_coordinate=(470, 450),
                pixel_center=(470, 450),
                samples=4,
                spread_px=0.0,
                detail="still white",
                verified_state="unliked",
            ),
        ]
        with (
            patch.object(
                web_app.runtime.agent,
                "parse",
                return_value=self._tap_heart_parsed_goal(),
            ),
            patch.object(
                web_app.runtime.vision_provider,
                "status",
                return_value={"configured": True},
            ),
            patch.object(web_app.runtime, "state_observer", Observer()),
            patch(
                "semantic_action_adapter.LocalTargetResolver.resolve",
                side_effect=resolutions,
            ),
            patch("semantic_action_adapter.time.sleep", return_value=None),
        ):
            response = self.client.post(
                "/api/agent/execute-tap-heart-step",
                headers=self.headers,
                json={"text": "给当前视频点赞", "confirmed": True},
            )
        self.assertEqual(response.status_code, 409, response.text)
        detail = response.json()["detail"]
        self.assertEqual(detail["physical_actions"], 1)
        self.assertEqual(detail["retry_count"], 0)
        new_executions = web_app.runtime.controller.executions[before_executions:]
        self.assertEqual(new_executions, [{"action": "tap", "coordinate": [470, 450]}])

    def test_mutation_without_token_is_rejected(self) -> None:
        response = self.client.post(
            "/api/tasks",
            json={
                "app_id": "douyin",
                "operation": "douyin.like_current",
                "params": {},
            },
        )
        self.assertEqual(response.status_code, 403)

    def test_confirmed_deprecated_task_fails_closed(self) -> None:
        created = self.client.post(
            "/api/tasks",
            headers=self.headers,
            json={
                "app_id": "douyin",
                "operation": "douyin.like_current",
                "params": {},
            },
        )
        self.assertEqual(created.status_code, 201)
        task = created.json()
        self.assertEqual(task["status"], "awaiting_confirmation")

        confirmed = self.client.post(
            f"/api/tasks/{task['id']}/confirm", headers=self.headers
        )
        self.assertEqual(confirmed.status_code, 200)
        self.assertEqual(confirmed.json()["status"], "queued")

        deadline = time.monotonic() + 3
        final = None
        while time.monotonic() < deadline:
            final = self.client.get(f"/api/tasks/{task['id']}").json()
            if final["status"] in {"succeeded", "failed", "cancelled"}:
                break
            time.sleep(0.05)
        self.assertEqual(final["status"], "failed")
        self.assertRegex(final["error"], "旧多步 workflow")

    def test_capability_trial_blocks_legacy_task_confirmation_before_queue(self) -> None:
        from universal_agent_orchestrator import DeviceTaskRegistry

        created = self.client.post(
            "/api/tasks",
            headers=self.headers,
            json={
                "app_id": "douyin",
                "operation": "douyin.like_current",
                "params": {},
            },
        )

        self.assertEqual(created.status_code, 201, created.text)
        task = created.json()
        device_id = web_app.runtime.device_controllers.default_device_id
        isolated_registry = DeviceTaskRegistry()
        with patch.object(
            web_app.runtime,
            "device_task_registry",
            isolated_registry,
        ):
            isolated_registry.reserve(device_id, "capability-trial-test")
            try:
                confirmed = self.client.post(
                    f"/api/tasks/{task['id']}/confirm",
                    headers=self.headers,
                )
            finally:
                isolated_registry.release(device_id, "capability-trial-test")

        self.assertEqual(confirmed.status_code, 409, confirmed.text)
        self.assertIn("已有活动任务", confirmed.text)
        self.assertEqual(
            web_app.runtime.store.get(task["id"])["status"],
            "awaiting_confirmation",
        )

    def test_preview_requires_and_uses_exact_registered_device(self) -> None:
        class PreviewController:
            def __init__(self, marker: bytes):
                self.marker = marker
                self.calls = []

            def capture_preview(self, quality=76):
                self.calls.append(quality)
                return b"jpeg-" + self.marker

        phone_a = PreviewController(b"phone-a")
        phone_b = PreviewController(b"phone-b")

        def controller_for_device(device_id):
            controllers = {"phone-a": phone_a, "phone-b": phone_b}
            if device_id not in controllers:
                raise web_app.UniversalAgentOrchestratorError(
                    f"device_id 未登记或未启用：{device_id}。"
                )
            return controllers[device_id]

        with patch.object(
            web_app.runtime,
            "controller_for_device",
            side_effect=controller_for_device,
        ):
            missing = self.client.get("/api/preview.jpg")
            unknown = self.client.get("/api/preview.jpg?device_id=phone-x")
            first = self.client.get("/api/preview.jpg?device_id=phone-a")
            second = self.client.get("/api/preview.jpg?device_id=phone-b")

        self.assertEqual(422, missing.status_code)
        self.assertEqual(404, unknown.status_code)
        self.assertEqual(b"jpeg-phone-a", first.content)
        self.assertEqual(b"jpeg-phone-b", second.content)
        self.assertEqual([76], phone_a.calls)
        self.assertEqual([76], phone_b.calls)

    def test_capability_trial_blocks_already_queued_legacy_worker_task(self) -> None:
        from universal_agent_orchestrator import DeviceTaskRegistry

        task = web_app.runtime.store.create(
            "douyin",
            "douyin.search",
            {"keyword": "机械臂"},
        )
        web_app.runtime.store.transition(
            task["id"],
            {"awaiting_confirmation"},
            "queued",
            message="测试验收会话占用设备后注入旧队列。",
        )
        device_id = web_app.runtime.device_controllers.default_device_id
        before_executions = list(web_app.runtime.controller.executions)
        isolated_registry = DeviceTaskRegistry()
        with patch.object(
            web_app.runtime,
            "device_task_registry",
            isolated_registry,
        ):
            isolated_registry.reserve(device_id, "capability-trial-test")
            try:
                web_app.runtime.jobs.put(task["id"])
                deadline = time.monotonic() + 3
                final = None
                while time.monotonic() < deadline:
                    final = web_app.runtime.store.get(task["id"])
                    if final["status"] == "failed":
                        break
                    time.sleep(0.05)
            finally:
                isolated_registry.release(device_id, "capability-trial-test")

        self.assertIsNotNone(final)
        self.assertEqual(final["status"], "failed")
        self.assertIn("已有活动任务", final["error"])
        self.assertEqual(before_executions, web_app.runtime.controller.executions)

    def test_legacy_free_agent_task_is_rejected(self) -> None:
        created = self.client.post(
            "/api/tasks",
            headers=self.headers,
            json={
                "app_id": "agent",
                "operation": "agent.execute_goal",
                "params": {"goal": "打开微信", "allowed_text": None},
            },
        )

        self.assertEqual(created.status_code, 400)
        self.assertIn("自由 Agent", created.json()["detail"])

    def test_device_status_identifies_each_active_generic_session_device(self) -> None:
        def active_session(session_id: str, device_id: str):
            payload = {
                "session_id": session_id,
                "device_id": device_id,
                "status": "awaiting_confirmation",
                "step_number": 1,
                "proposal": {"status": "action"},
            }
            return SimpleNamespace(
                status="awaiting_confirmation",
                snapshot=lambda payload=payload: dict(payload),
            )

        with web_app.runtime.generic_supervised_session_lock:
            web_app.runtime.generic_supervised_sessions.update(
                {
                    "session-phone-a": active_session("session-phone-a", "phone-a"),
                    "session-phone-b": active_session("session-phone-b", "phone-b"),
                }
            )

        active = self.client.get("/api/device").json()[
            "generic_supervised_execution"
        ]["active_sessions"]

        self.assertEqual(
            {
                (item["session_id"], item["device_id"])
                for item in active
            },
            {
                ("session-phone-a", "phone-a"),
                ("session-phone-b", "phone-b"),
            },
        )

    def test_worker_rejects_injected_legacy_task(self) -> None:
        task = web_app.runtime.store.create(
            "agent",
            "agent.execute_goal",
            {"goal": "打开微信"},
        )
        web_app.runtime.store.transition(
            task["id"],
            {"awaiting_confirmation"},
            "queued",
            message="测试绕过 API 注入旧任务。",
        )
        web_app.runtime.jobs.put(task["id"])

        deadline = time.monotonic() + 3
        final = None
        while time.monotonic() < deadline:
            final = web_app.runtime.store.get(task["id"])
            if final["status"] == "failed":
                break
            time.sleep(0.05)
        self.assertIsNotNone(final)
        self.assertEqual(final["status"], "failed")
        self.assertIn("唯一控制器拒绝执行", final["error"])

    def test_unknown_operation_is_rejected(self) -> None:
        response = self.client.post(
            "/api/tasks",
            headers=self.headers,
            json={
                "app_id": "wechat",
                "operation": "system.shell",
                "params": {"command": "whoami"},
            },
        )
        self.assertEqual(response.status_code, 400)

    def test_all_structured_operations_create_state_graph_confirmable_tasks(self) -> None:
        cases = [
            (
                "wechat",
                "wechat.send_text",
                {"chat_name": "文件传输助手", "text": "你好abc1"},
            ),
            (
                "wechat",
                "wechat.send_album_image",
                {"chat_name": "文件传输助手", "image_index": 3},
            ),
            ("douyin", "douyin.search", {"keyword": "机械臂"}),
            (
                "douyin",
                "douyin.batch_interact",
                {
                    "keyword": "机械臂",
                    "target_count": 2,
                    "like": True,
                    "comment": True,
                    "comment_text": "做得很好1",
                },
            ),
        ]
        for app_id, operation, params in cases:
            with self.subTest(operation=operation):
                response = self.client.post(
                    "/api/tasks",
                    headers=self.headers,
                    json={
                        "app_id": app_id,
                        "operation": operation,
                        "params": params,
                    },
                )
                self.assertEqual(response.status_code, 201, response.text)
                task = response.json()
                self.assertEqual(task["status"], "awaiting_confirmation")
                self.assertEqual(
                    task["params"]["workflow_version"],
                    "page_state_graph_v1",
                )
                self.assertEqual(task["params"]["model_role"], "observation_only")
                self.assertNotIn("execution_plan", task["params"])
                self.assertTrue(task["params"]["safety"]["one_action_per_observation"])
                self.assertTrue(task["params"]["safety"]["verify_after_every_action"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
