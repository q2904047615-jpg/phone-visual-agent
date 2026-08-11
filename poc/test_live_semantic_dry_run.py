import unittest

from PIL import Image

from live_semantic_dry_run import (
    ReadOnlySemanticDryRunner,
    to_semantic_observation,
)
from state_controller import PageObservation as VisionPageObservation
from task_orchestrator import GenericTaskOrchestrator, GoalSpec


class FakeObserver:
    def __init__(self, observation: VisionPageObservation) -> None:
        self.observation = observation
        self.calls = []

    def observe(self, **kwargs):
        self.calls.append(kwargs)
        return self.observation


class LiveSemanticDryRunTests(unittest.TestCase):
    def test_mapping_strips_raw_coordinates_but_preserves_semantics(self) -> None:
        observed = VisionPageObservation(
            state="douyin_video",
            base_state="douyin_video",
            confidence=0.93,
            stable=True,
            reason="普通视频页",
            heart_state="unliked",
            page_fingerprint="author-title",
            targets={"heart": (880, 420)},
            target_bounds={"heart": (830, 370, 930, 470)},
        )
        semantic = to_semantic_observation(observed)
        self.assertEqual(semantic.page_state, "douyin_video")
        self.assertEqual(semantic.properties["heart_state"], "unliked")
        self.assertEqual(semantic.properties["visible_targets"], ["heart"])
        self.assertNotIn("targets", semantic.properties)
        self.assertNotIn("target_bounds", semantic.properties)

    def test_real_frame_preview_returns_one_action_without_execution(self) -> None:
        capture_calls = []

        def capture():
            capture_calls.append(True)
            return Image.new("RGB", (540, 960), "black")

        observer = FakeObserver(
            VisionPageObservation(
                state="android_home",
                base_state="android_home",
                confidence=0.95,
                stable=True,
                reason="桌面稳定",
            )
        )
        goal = GoalSpec.from_operation(
            "douyin.search",
            {"keyword": "机械臂"},
        )
        plan = GenericTaskOrchestrator().compile_goal(goal)
        runner = ReadOnlySemanticDryRunner(
            capture,
            observer,
            observation_seconds=0,
        )
        result = runner.preview(goal, plan).to_dict()

        self.assertEqual(len(capture_calls), 4)
        self.assertEqual(len(observer.calls), 1)
        self.assertEqual(result["decision"]["status"], "action")
        self.assertEqual(result["decision"]["action"]["action"], "ensure_app")
        self.assertFalse(result["executed"])
        self.assertEqual(
            result["safety"],
            {
                "robot_action_called": False,
                "task_created": False,
                "account_action_performed": False,
            },
        )
        self.assertEqual(
            observer.calls[0]["controller_context"]["executed_actions"],
            [],
        )

    def test_low_confidence_fails_before_exposing_action(self) -> None:
        observer = FakeObserver(
            VisionPageObservation(
                state="unknown",
                base_state="unknown",
                confidence=0.31,
                stable=True,
                reason="画面不清晰",
            )
        )
        goal = GoalSpec.from_operation(
            "douyin.search",
            {"keyword": "机械臂"},
        )
        plan = GenericTaskOrchestrator().compile_goal(goal)
        runner = ReadOnlySemanticDryRunner(
            lambda: Image.new("RGB", (540, 960), "black"),
            observer,
            observation_seconds=0,
        )
        result = runner.preview(goal, plan).to_dict()
        self.assertEqual(result["decision"]["status"], "failed")
        self.assertIsNone(result["decision"]["action"])
        self.assertIn("置信度不足", result["decision"]["reason"])


if __name__ == "__main__":
    unittest.main()
