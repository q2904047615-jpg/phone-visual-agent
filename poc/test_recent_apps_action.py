from __future__ import annotations

import unittest

from agent.domain import DeviceActionRequest
from agent.domain.canonical_action_protocol import bind_same_response_action
from agent.domain.qwen_task_context import QwenTaskContext
from agent.domain.trusted_observation import TrustedObservation
from agent.domain.ui_scene import UIElement, UIScene
from agent.domain.universal_action_controller import UniversalActionController
from agent.domain.visual_evidence import LocalFrameStability
from agent.infrastructure import RobotDeviceExecutor
from agent.infrastructure.robot_controller import (
    DEFAULT_CONTROLLER_CONFIG,
    load_controller_config,
)


def scene(
    *,
    app_id: str,
    screen_id: str,
    fingerprint: str,
    elements: tuple[UIElement, ...] = (),
) -> UIScene:
    return UIScene(
        app_id=app_id,
        screen_id=screen_id,
        summary="当前稳定画面",
        elements=elements,
        stable=True,
        confidence=0.99,
        fingerprint=fingerprint,
    )


def recent_context(objective: str) -> QwenTaskContext:
    task_id = "task-recent-current-frame"
    subgoal_id = "recent-current-step"
    return QwenTaskContext(
        protocol_version="2026-08-20-deepseek-typed-task-graph-v4",
        task_id=task_id,
        device_id="device-local-01",
        revision=1,
        task_status="running",
        goal={"objective": objective, "target_apps": [], "entities": {}},
        global_constraints=(),
        goal_completion_conditions=(),
        current_subgoal={
            "subgoal_id": subgoal_id,
            "objective": objective,
            "status": "active",
            "depends_on": (),
            "constraints": (),
            "completion_conditions": ("当前导航步骤完成",),
            "completion_evidence": (),
            "execution_class": "navigate",
        },
        current_execution_class="navigate",
        effect_intents=(),
        effect_gate={
            "state": "not_required",
            "effect_action_allowed": False,
            "scope": {
                "task_id": task_id,
                "device_id": "device-local-01",
                "revision": 1,
                "subgoal_id": subgoal_id,
            },
        },
    )


def trusted(current: UIScene) -> TrustedObservation:
    return TrustedObservation(
        observation_id="obs_0123456789abcdef",
        device_id="device-local-01",
        fingerprint=current.fingerprint,
        scene=current,
        local_stability=LocalFrameStability(
            stable=True,
            mean_delta=0.0,
            max_delta=0.0,
            frame_count=1,
            threshold=1.0,
            reason="test",
        ),
        selected_frame_index=0,
        frame_sharpness_scores=(1.0,),
    )


class FakeRobot:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.receipt: dict | None = None

    def vision_android_recent_apps(self):
        self.calls.append("open_recent_apps")
        self.receipt = {
            "seller_event_barrier_confirmed": True,
            "round_trip_position_confirmed": True,
            "mechanical_contact_ack": False,
            "click_count": 1,
        }
        return 315, 976

    def consume_last_click_receipt(self):
        receipt = self.receipt
        self.receipt = None
        return receipt


class RecentAppsActionTests(unittest.TestCase):
    def test_launcher_frame_binds_qwen_recent_apps_action_directly(self) -> None:
        before = scene(
            app_id="launcher",
            screen_id="home_screen",
            fingerprint="a" * 64,
        )
        action = bind_same_response_action(
            {"action": "open_recent_apps"},
            context=recent_context("打开系统最近任务页面"),
            observation=trusted(before),
            available_action_kinds={"back", "home", "open_recent_apps"},
        )

        self.assertEqual("open_recent_apps", action.action)
        self.assertEqual({}, action.params)
        resolved = UniversalActionController().resolve_one(action, before)
        self.assertEqual("open_recent_apps", resolved.kind)
        self.assertEqual(before.fingerprint, resolved.before_fingerprint)

    def test_new_recent_tasks_frame_binds_only_the_qwen_selected_card_and_direction(self) -> None:
        card = UIElement(
            element_id="preview-card",
            role="list_item",
            meaning="sample_preview_card",
            label="示例应用",
            bounds=(0.25, 0.2, 0.75, 0.8),
            confidence=1.0,
            states={"visible": True, "enabled": True, "fully_visible": True},
            evidence=("唯一完整可见的应用预览卡片",),
        )
        current = scene(
            app_id="system",
            screen_id="system_recent_tasks",
            fingerprint="b" * 64,
            elements=(card,),
        )
        for direction in ("left", "right"):
            with self.subTest(direction=direction):
                action = bind_same_response_action(
                    {
                        "action": "swipe",
                        "direction": direction,
                        "element_id": card.element_id,
                    },
                    context=recent_context("清除当前截图中的示例应用预览卡片"),
                    observation=trusted(current),
                    available_action_kinds={"swipe", "back", "home", "open_recent_apps"},
                )
                self.assertEqual(
                    {"direction": direction, "element_id": card.element_id,
                     "target": card.meaning, "role": card.role, "label": card.label,
                     "states": dict(card.states)},
                    action.params,
                )

                resolved = UniversalActionController().resolve_one(action, current)
                self.assertEqual("swipe", resolved.kind)
                self.assertEqual(direction, resolved.direction)
                self.assertEqual(card.element_id, resolved.target_element_id)
                self.assertIsNotNone(resolved.normalized_point)
                self.assertIsNotNone(resolved.normalized_end_point)
                self.assertGreater(resolved.path_distance or 0.0, 0.0)
                for point in (resolved.normalized_point, resolved.normalized_end_point):
                    self.assertTrue(all(0.0 <= value <= 1.0 for value in point or ()))

    def test_device_recents_calibration_stays_inside_left_navigation_key(self) -> None:
        configured = load_controller_config()
        for source in (DEFAULT_CONTROLLER_CONFIG, configured):
            self.assertEqual(0.33, source["android_recents_x_ratio"])
            self.assertEqual(0.976, source["android_recents_y_ratio"])
            self.assertLess(
                source["android_recents_x_ratio"],
                source["android_home_x_ratio"],
            )

    def test_screen_ids_are_not_rewritten_by_local_aliases(self) -> None:
        recent = scene(
            app_id="system",
            screen_id="system_recent_tasks",
            fingerprint="c" * 64,
        )
        browser_page = scene(
            app_id="browser",
            screen_id="recent_tasks",
            fingerprint="d" * 64,
        )

        self.assertEqual("system_recent_tasks", recent.screen_id)
        self.assertEqual("recent_tasks", browser_page.screen_id)

    def test_controller_accepts_fresh_stable_frame_without_second_semantic_verdict(self) -> None:
        before = scene(
            app_id="launcher",
            screen_id="android_home",
            fingerprint="e" * 64,
        )
        action = bind_same_response_action(
            {"action": "open_recent_apps"},
            context=recent_context("打开系统最近任务页面"),
            observation=trusted(before),
            available_action_kinds={"open_recent_apps"},
        )
        resolved = UniversalActionController().resolve_one(action, before)
        after = scene(
            app_id="system",
            screen_id="system_recent_tasks",
            fingerprint="f" * 64,
        )

        self.assertEqual((), UniversalActionController().verify_after_action(
            resolved, before, after,
        ))

    def test_device_executor_dispatches_one_verified_system_click(self) -> None:
        robot = FakeRobot()
        result = RobotDeviceExecutor(robot).execute(
            DeviceActionRequest(kind="open_recent_apps")
        )

        self.assertEqual(["open_recent_apps"], robot.calls)
        self.assertEqual(1, result.physical_actions)
        self.assertEqual(1, result.hardware_receipt["click_count"])
        self.assertIsNone(robot.consume_last_click_receipt())


if __name__ == "__main__":
    unittest.main()
