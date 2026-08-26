from __future__ import annotations

import unittest
from types import SimpleNamespace

from canonical_action_protocol import compile_canonical_action_catalog
from deepseek_task_graph import build_exact_action_task_graph
from device_executor import DeviceActionRequest, RobotDeviceExecutor
from generic_action_adapter import _post_action_visual_context
from generic_scene_observer import POST_ACTION_VISUAL_CONTEXT_VERSION
from qwen_visual_decision import _deterministic_exact_selection_payload
from robot_core import DEFAULT_CONTROLLER_CONFIG, load_controller_config
from semantic_action import SemanticAction
from task_semantic_ir import compile_formal_semantic_authority
from test_task_semantic_ir import required_actions_for_objective
from ui_scene import UIScene, scene_surface_kind
from universal_action_controller import (
    ResolvedSemanticAction,
    UniversalActionController,
    UniversalActionError,
)


def scene(*, app_id: str, screen_id: str, fingerprint: str) -> UIScene:
    return UIScene(
        app_id=app_id,
        screen_id=screen_id,
        summary="当前稳定画面",
        stable=True,
        confidence=0.99,
        fingerprint=fingerprint,
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
    def test_device_recents_calibration_stays_inside_left_navigation_key(self) -> None:
        configured = load_controller_config()
        for source in (DEFAULT_CONTROLLER_CONFIG, configured):
            self.assertEqual(0.33, source["android_recents_x_ratio"])
            self.assertEqual(0.976, source["android_recents_y_ratio"])
            self.assertLess(
                source["android_recents_x_ratio"],
                source["android_home_x_ratio"],
            )

    def test_natural_goal_compiles_one_recent_apps_requirement(self) -> None:
        for objective in (
            "打开系统最近任务页面",
            "调出最近应用列表",
            "Show recent apps",
        ):
            with self.subTest(objective=objective):
                self.assertEqual(
                    {"open_recent_apps"},
                    required_actions_for_objective(objective),
                )

        self.assertNotIn(
            "open_recent_apps",
            required_actions_for_objective("打开浏览器多任务页面"),
        )

    def test_exact_graph_catalog_and_qwen_select_same_canonical_action(self) -> None:
        graph = build_exact_action_task_graph(
            "打开系统最近任务页面",
            action_kind="open_recent_apps",
            device_id="device-local-01",
            task_id="task-open-recent-apps",
        )
        authority = compile_formal_semantic_authority(graph)
        before = scene(
            app_id="launcher",
            screen_id="android_home",
            fingerprint="a" * 64,
        )
        report = compile_canonical_action_catalog(
            before,
            authority.semantic_ir,
            {"back", "home", "open_recent_apps"},
        )
        candidates = tuple(
            item for item in report.candidates if item.action_kind == "open_recent_apps"
        )
        self.assertEqual(1, len(candidates))
        self.assertEqual(
            [
                {
                    "subject_ref": candidates[0].transition.expectations[0].subject_ref,
                    "predicate": "surface.kind",
                    "operator": "equals",
                    "value": "recent_tasks",
                }
            ],
            [item.to_dict() for item in candidates[0].transition.expectations],
        )

        choice = {
            "choice_id": "choice_recent_apps",
            "action": "open_recent_apps",
        }
        selected = _deterministic_exact_selection_payload(
            SimpleNamespace(
                current_subgoal={"subgoal_id": "exact_open_recent_apps"},
                current_execution_class="navigate",
            ),
            (choice,),
        )
        self.assertEqual("choice_recent_apps", selected["choice_id"])

    def test_recent_tasks_surface_is_exact_and_not_an_app_page_alias(self) -> None:
        recent = scene(
            app_id="system",
            screen_id="system_recent_tasks",
            fingerprint="b" * 64,
        )
        browser_page = scene(
            app_id="browser",
            screen_id="recent_tasks",
            fingerprint="c" * 64,
        )

        self.assertEqual("recent_tasks", scene_surface_kind(recent))
        self.assertEqual("app", scene_surface_kind(browser_page))

    def test_controller_requires_typed_recent_tasks_postcondition(self) -> None:
        before = scene(
            app_id="launcher",
            screen_id="android_home",
            fingerprint="d" * 64,
        )
        action = SemanticAction(
            node_id="formal-open-recent-apps",
            action="open_recent_apps",
            params={
                "formal_candidate_id": "candidate.open_recent_apps",
                "formal_report_digest": "e" * 64,
                "formal_transition": {
                    "transition_id": "transition.open_recent_apps",
                    "precondition_claim_ids": ["claim.surface"],
                    "expectations": [
                        {
                            "subject_ref": "surface.current",
                            "predicate": "surface.kind",
                            "operator": "equals",
                            "value": "recent_tasks",
                        }
                    ],
                    "exploratory": False,
                },
            },
        )
        controller = UniversalActionController()
        resolved = controller.resolve_one(action, before, confirmed=True)

        controller.verify_after_action(
            resolved,
            before,
            scene(
                app_id="system",
                screen_id="system_recent_tasks",
                fingerprint="f" * 64,
            ),
        )
        with self.assertRaisesRegex(UniversalActionError, "typed surface.kind"):
            controller.verify_after_action(
                resolved,
                before,
                scene(
                    app_id="launcher",
                    screen_id="home_screen",
                    fingerprint="1" * 64,
                ),
            )

    def test_post_action_visual_context_carries_typed_expectation_without_verdict(
        self,
    ) -> None:
        samples = (
            (
                "open_recent_apps",
                {
                    "subject_ref": "surface_current",
                    "predicate": "surface.kind",
                    "operator": "equals",
                    "value": "recent_tasks",
                },
            ),
            (
                "back",
                {
                    "subject_ref": "surface_current",
                    "predicate": "surface.navigation_depth",
                    "operator": "changed",
                },
            ),
        )
        for kind, expectation in samples:
            with self.subTest(kind=kind):
                context = _post_action_visual_context(
                    ResolvedSemanticAction(
                        node_id=f"resolved-{kind}",
                        kind=kind,
                        before_fingerprint="a" * 64,
                        formal_candidate_id=f"candidate.{kind}",
                        formal_transition={
                            "transition_id": f"transition.{kind}",
                            "precondition_claim_ids": ["claim.surface"],
                            "expectations": [expectation],
                            "exploratory": False,
                        },
                    )
                )
                self.assertIsNotNone(context)
                payload = context.to_dict()
                self.assertEqual(
                    POST_ACTION_VISUAL_CONTEXT_VERSION,
                    payload["protocol_version"],
                )
                self.assertEqual(kind, payload["canonical_action_kind"])
                self.assertEqual([expectation], payload["expected_postconditions"])
                self.assertEqual(
                    "pending_visual_verification",
                    payload["outcome"],
                )
                self.assertNotIn("matched", payload)
                self.assertNotIn("coordinates", payload)

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
