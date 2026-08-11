import unittest
from types import SimpleNamespace

from semantic_executor import ActionResult, PageObservation
from supervised_semantic_runtime import (
    SupervisedSemanticSession,
    SupervisedSemanticSessionError,
)
from task_orchestrator import GenericTaskOrchestrator, GoalSpec


def obs(page: str, **properties) -> PageObservation:
    return PageObservation(
        page_state=page,
        properties=properties,
        confidence=0.96,
        stable=True,
    )


class FakeRouter:
    def __init__(self, observations):
        self.observations = list(observations)
        self.calls = []

    @staticmethod
    def supports(action):
        return action.action in {"ensure_app", "observe", "tap_semantic", "swipe"}

    def execute(self, action, goal, **kwargs):
        self.calls.append((action.action, dict(action.params)))
        observation = self.observations.pop(0)
        return SimpleNamespace(
            action_result=ActionResult(
                node_id=action.node_id,
                success=True,
                observation=observation,
                details={"physical_actions": int(action.action != "observe")},
            ),
            to_dict=lambda: {
                "robot_action_called": action.action != "observe",
            },
        )


class SupervisedSemanticSessionTests(unittest.TestCase):
    def make_session(self, target_count=1):
        goal = GoalSpec.from_operation(
            "douyin.batch_interact",
            {
                "target_count": target_count,
                "like": True,
                "comment": False,
            },
        )
        plan = GenericTaskOrchestrator().compile_goal(goal)
        router = FakeRouter(
            [
                obs("douyin_video", video_id="v1", heart_state="unliked"),
                obs("douyin_video", video_id="v1", heart_state="unliked"),
                obs("douyin_video", video_id="v1", heart_state="liked"),
                obs("douyin_video", video_id="v1", heart_state="liked"),
                obs("douyin_video", video_id="v2", heart_state="unliked"),
            ]
        )
        session = SupervisedSemanticSession.start(
            session_id="session-1",
            goal=goal,
            plan=plan,
            initial_observation=obs("android_home"),
            router=router,
        )
        return session, router

    def test_each_confirmation_advances_exactly_one_node(self):
        session, router = self.make_session()
        expected = [
            "ensure_app",
            "observe",
            "tap_semantic",
            "observe",
            "record_verified_result",
            "finish",
        ]
        seen = []
        for action_name in expected:
            seen.append(session.snapshot()["decision"]["action"]["action"])
            session.step(confirmed=True)
        self.assertEqual(seen, expected)
        self.assertEqual(session.snapshot()["status"], "finished")
        self.assertEqual(session.executor.counters["processed_videos"], 1)
        self.assertEqual(
            [name for name, _params in router.calls],
            ["ensure_app", "observe", "tap_semantic", "observe"],
        )

    def test_unconfirmed_step_changes_nothing(self):
        session, router = self.make_session()
        before = session.snapshot()
        with self.assertRaisesRegex(SupervisedSemanticSessionError, "明确确认"):
            session.step(confirmed=False)
        self.assertEqual(session.snapshot()["decision"], before["decision"])
        self.assertEqual(router.calls, [])

    def test_swipe_is_supervised_physical_navigation_without_account_effect(self):
        session, router = self.make_session(target_count=2)
        for _ in range(5):
            session.step(confirmed=True)
        snapshot = session.snapshot()
        self.assertEqual(snapshot["decision"]["action"]["action"], "swipe")
        self.assertTrue(snapshot["current_action"]["supported"])
        self.assertTrue(snapshot["current_action"]["physical_action_possible"])
        self.assertFalse(snapshot["current_action"]["account_effect_possible"])
        session.step(confirmed=True)
        self.assertEqual(router.calls[-1][0], "swipe")

    def test_cancel_blocks_future_steps(self):
        session, router = self.make_session()
        session.cancel()
        self.assertEqual(session.snapshot()["status"], "cancelled")
        with self.assertRaisesRegex(SupervisedSemanticSessionError, "已经取消"):
            session.step(confirmed=True)
        self.assertEqual(router.calls, [])


if __name__ == "__main__":
    unittest.main()
