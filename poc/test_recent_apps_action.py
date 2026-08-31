from __future__ import annotations

import unittest

from agent.domain.canonical_action_protocol import compile_canonical_action_catalog
from agent.domain.task_graph import (
    _graph_from_payload,
    build_exact_action_task_graph,
)
from agent.domain import DeviceActionRequest
from agent.infrastructure import RobotDeviceExecutor
from agent.infrastructure.generic_action_adapter import _post_action_visual_context
from agent.domain.post_action_observation import (
    POST_ACTION_VISUAL_CONTEXT_VERSION,
)
from agent.infrastructure.robot_controller import (
    DEFAULT_CONTROLLER_CONFIG,
    load_controller_config,
)
from agent.domain.semantic_action import SemanticAction
from agent.domain.task_semantic_ir import compile_formal_semantic_authority
from test_task_semantic_ir import required_actions_for_objective
from agent.domain.ui_scene import UIElement, UIScene, scene_surface_kind
from agent.domain.universal_action_controller import (
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
    @staticmethod
    def clear_card_graph(
        card_objective: str = "在最近任务界面中清除示例应用卡片",
    ):
        raw_goal = "先清除示例应用卡片，然后打开示例应用"
        return _graph_from_payload(
            {
                "status": "ready",
                "goal": {
                    "objective": raw_goal,
                    "target_apps": [
                        {"app_id": "sample_app", "app_name": "示例应用"}
                    ],
                    "entities": {},
                },
                "constraints": [],
                "completion_conditions": [
                    {
                        "condition_id": "opened",
                        "description": "示例应用主界面可见",
                        "evidence_required": ["示例应用页面"],
                        "satisfied": False,
                        "evidence": [],
                    }
                ],
                "effect_intents": [],
                "subgoals": [
                    {
                        "subgoal_id": "clear_card",
                        "objective": card_objective,
                        "status": "active",
                        "depends_on": [],
                        "constraints": ["仅清除应用预览卡片"],
                        "completion_conditions": [
                            "最近任务界面中示例应用卡片被移除"
                        ],
                        "completion_evidence": [],
                        "effect_ids": [],
                        "execution_class": "navigate",
                    },
                    {
                        "subgoal_id": "open_app",
                        "objective": "打开示例应用",
                        "status": "pending",
                        "depends_on": ["clear_card"],
                        "constraints": [],
                        "completion_conditions": ["示例应用主界面可见"],
                        "completion_evidence": [],
                        "effect_ids": [],
                        "execution_class": "navigate",
                    },
                ],
                "active_subgoal_id": "clear_card",
                "clarification_questions": [],
            },
            task_id="task-clear-card-current-frame",
            device_id="device-local-01",
            revision=1,
            raw_user_goal=raw_goal,
        )

    def test_clear_card_from_launcher_only_allows_entering_recent_tasks(self) -> None:
        authority = compile_formal_semantic_authority(self.clear_card_graph())
        active = next(
            item
            for item in authority.semantic_ir.subgoals
            if item.subgoal_id == "clear_card"
        )
        surfaces = {
            item.surface_id: item for item in authority.semantic_ir.surfaces
        }
        self.assertEqual("recent_tasks", surfaces[active.surface_ref].kind)

        launcher = UIScene(
            app_id="launcher",
            screen_id="home_screen",
            summary="桌面分页中可见示例应用图标",
            elements=(
                UIElement(
                    element_id="app-icon",
                    role="icon",
                    meaning="app_launcher_sample",
                    label="示例应用",
                    bounds=(0.1, 0.2, 0.3, 0.4),
                    confidence=1.0,
                    states={"goal_relevant": True, "fully_visible": True},
                ),
                UIElement(
                    element_id="pages",
                    role="container",
                    meaning="paged_viewport",
                    label="桌面分页区",
                    bounds=(0.0, 0.1, 1.0, 0.9),
                    confidence=1.0,
                    states={
                        "goal_relevant": True,
                        "fully_visible": True,
                        "scrollable": True,
                        "scroll_axis": "horizontal",
                        "page_index": 0,
                        "page_count": 2,
                    },
                    evidence=("两个桌面分页圆点",),
                ),
            ),
            stable=True,
            confidence=1.0,
            fingerprint="launcher-current-frame",
        )
        report = compile_canonical_action_catalog(
            launcher,
            authority.semantic_ir,
            {"tap_semantic", "swipe", "back", "home", "open_recent_apps"},
        )
        self.assertEqual(
            ["open_recent_apps"],
            [item.action_kind for item in report.candidates],
        )
    def test_new_recent_tasks_frame_only_allows_bound_card_swipe(self) -> None:
        recent_tasks = UIScene(
            app_id="system",
            screen_id="system_recent_tasks",
            summary="最近任务中显示唯一示例应用预览卡片",
            elements=(
                UIElement(
                    element_id="preview-card",
                    role="list_item",
                    meaning="sample_preview_card",
                    label="示例应用",
                    bounds=(0.25, 0.2, 0.75, 0.8),
                    confidence=1.0,
                    states={"goal_relevant": True, "fully_visible": True},
                    evidence=("唯一完整可见的应用预览卡片",),
                ),
            ),
            stable=True,
            confidence=1.0,
            fingerprint="new-recent-tasks-frame",
        )
        for objective, expected_direction in (
            ("在最近任务界面中清除示例应用卡片", "left"),
            ("在最近任务界面中向右划掉示例应用卡片", "right"),
        ):
            with self.subTest(objective=objective):
                authority = compile_formal_semantic_authority(
                    self.clear_card_graph(objective)
                )
                report = compile_canonical_action_catalog(
                    recent_tasks,
                    authority.semantic_ir,
                    {
                        "tap_semantic",
                        "swipe",
                        "back",
                        "home",
                        "open_recent_apps",
                    },
                )
                self.assertEqual(1, len(report.candidates))
                candidate = report.candidates[0]
                self.assertEqual("swipe", candidate.action_kind)
                self.assertEqual(
                    {
                        "direction": expected_direction,
                        "element_id": "preview-card",
                    },
                    candidate.parameters,
                )
                self.assertEqual(
                    [("element.exists", "absent")],
                    [
                        (item.predicate, item.operator)
                        for item in candidate.transition.expectations
                    ],
                )

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

    def test_exact_graph_catalog_exposes_one_canonical_action(self) -> None:
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
