"""Check dead-name removal without mistaking dynamic transports for dead code."""
import unittest

from agent.domain import canonical_action_kinds
from agent.domain.device_execution import DeviceActionRequest
from agent.infrastructure import generic_scene_observer
from agent.infrastructure.generic_action_adapter import GenericSingleActionAdapter
from agent.infrastructure.device_executor import RobotDeviceExecutor
from agent.infrastructure.robot_controller import RobotController


class UnusedRuntimeResidueTests(unittest.TestCase):
    def test_unreferenced_names_stay_retired(self):
        for owner, names in (
            (canonical_action_kinds, ('expected_idempotent_system_surface_kind', '_IDEMPOTENT_SYSTEM_SURFACES')),
            (GenericSingleActionAdapter, ('GEOMETRY_BOUND_KINDS', 'INDEPENDENT_GEOMETRY_AUDIT_KINDS')),
            (generic_scene_observer, ('_QWERTY_ANCHOR_KEYS',)),
            (RobotController, ('_sleep',)),
        ):
            for name in names:
                with self.subTest(name=name):
                    self.assertFalse(hasattr(owner, name))

    def test_all_four_dynamic_scroll_methods_remain_executable(self):
        class RecordingRobot(RobotController):
            def __init__(self):
                self.calls = []  # No real controller construction or window access.

            def _vision_swipe(self, direction):
                self.calls.append(direction)

        for direction in ('up', 'down', 'left', 'right'):
            with self.subTest(direction=direction):
                robot = RecordingRobot()
                result = RobotDeviceExecutor(robot).execute(DeviceActionRequest(kind='scroll', direction=direction))
                self.assertEqual([direction], robot.calls)
                self.assertEqual(1, result.physical_actions)

    def test_wait_uses_executor_dependency_not_retired_robot_sleep(self):
        calls = []
        result = RobotDeviceExecutor(object(), sleep=calls.append).execute(
            DeviceActionRequest(kind='wait_for_change', wait_seconds=0.75))
        self.assertEqual([0.75], calls)
        self.assertEqual(0, result.physical_actions)


if __name__ == '__main__':
    unittest.main()
