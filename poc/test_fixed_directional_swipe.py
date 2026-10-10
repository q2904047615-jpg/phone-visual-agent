from __future__ import annotations

import unittest

from agent.infrastructure.robot_controller import MockRobotController, fixed_directional_swipe_points


class FixedDirectionalSwipeTests(unittest.TestCase):
    def test_default_paths_cover_four_directions(self) -> None:
        self.assertEqual(((800, 500), (200, 500)), fixed_directional_swipe_points("left"))
        self.assertEqual(((200, 500), (800, 500)), fixed_directional_swipe_points("right"))
        self.assertEqual(((500, 800), (500, 200)), fixed_directional_swipe_points("up"))
        self.assertEqual(((500, 200), (500, 800)), fixed_directional_swipe_points("down"))

    def test_mock_direct_execution_is_one_local_swipe_without_model(self) -> None:
        controller = MockRobotController(device_id="device-test")
        result = controller.vision_fixed_directional_swipe("left")
        self.assertEqual("left", result["direction"])
        self.assertEqual(1, len(controller.executions))
        self.assertEqual("swipe_relative", controller.executions[0]["action"])
        self.assertFalse("qwen_called" in result)


if __name__ == "__main__":
    unittest.main()
