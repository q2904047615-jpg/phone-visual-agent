import unittest

from device_executor import (
    DeviceActionRequest,
    DeviceExecutionError,
    ReplayDeviceExecutor,
    RobotDeviceExecutor,
)


class FakeRobot:
    def __init__(self) -> None:
        self.calls = []
        self._click_receipt = None
        self._long_press_receipt = None

    def _record_click(self, count: int) -> None:
        self._click_receipt = {
            "seller_event_barrier_confirmed": True,
            "round_trip_position_confirmed": True,
            "mechanical_contact_ack": False,
            "click_count": count,
        }

    def vision_tap_relative(self, x, y):
        self.calls.append(("tap", x, y))
        self._record_click(1)
        return (x, y)

    def vision_double_tap_relative(self, x, y):
        self.calls.append(("double_tap", x, y))
        self._record_click(2)
        return (x, y)

    def vision_type_text_with_layout(self, text, geometry):
        self.calls.append(("type", text, geometry))

    def vision_type_pinyin(self, text, pinyin, geometry):
        self.calls.append(("pinyin", text, pinyin, geometry))

    def vision_clear_text(self, geometry, delete_count):
        self.calls.append(("clear", geometry, delete_count))

    def vision_long_press_relative(self, x, y, hold_seconds):
        self.calls.append(("long_press", x, y, hold_seconds))
        self._long_press_receipt = {"hold_started_after_barrier": True}
        return (x, y)

    def consume_last_click_receipt(self):
        receipt = self._click_receipt
        self._click_receipt = None
        return receipt

    def consume_last_long_press_receipt(self):
        receipt = self._long_press_receipt
        self._long_press_receipt = None
        return receipt


class DeviceExecutorTests(unittest.TestCase):
    def test_single_tap_dispatches_once_and_consumes_receipt(self):
        robot = FakeRobot()
        executor = RobotDeviceExecutor(robot)

        result = executor.execute(
            DeviceActionRequest(kind="tap_semantic", point=(250, 750))
        )

        self.assertEqual(robot.calls, [("tap", 250, 750)])
        self.assertEqual(result.physical_actions, 1)
        self.assertEqual(result.hardware_receipt["click_count"], 1)
        self.assertIsNone(robot.consume_last_click_receipt())

    def test_double_tap_is_one_canonical_transport_with_count_two(self):
        robot = FakeRobot()
        executor = RobotDeviceExecutor(robot)

        result = executor.execute(
            DeviceActionRequest(kind="double_tap", point=(500, 500))
        )

        self.assertEqual(robot.calls, [("double_tap", 500, 500)])
        self.assertEqual(result.physical_actions, 1)
        self.assertEqual(result.hardware_receipt["click_count"], 2)

    def test_click_without_valid_receipt_is_a_failed_physical_action(self):
        robot = FakeRobot()
        robot._record_click = lambda _count: None
        executor = RobotDeviceExecutor(robot)

        with self.assertRaises(DeviceExecutionError) as context:
            executor.execute(
                DeviceActionRequest(kind="tap_semantic", point=(100, 200))
            )

        self.assertEqual(context.exception.physical_actions, 1)
        self.assertEqual(robot.calls, [("tap", 100, 200)])

    def test_input_transports_are_selected_from_typed_method(self):
        robot = FakeRobot()
        executor = RobotDeviceExecutor(robot)
        geometry = {"type": "qwerty", "source": "input_structure_audit"}

        executor.execute(
            DeviceActionRequest(
                kind="input_verified_text",
                input_fragment="abc",
                input_method="direct_latin",
                keyboard_geometry=geometry,
            )
        )
        executor.execute(
            DeviceActionRequest(
                kind="input_verified_text",
                input_fragment="你好",
                input_method="chinese_pinyin",
                input_pinyin="nihao",
                keyboard_geometry=geometry,
            )
        )

        self.assertEqual(robot.calls[0][:2], ("type", "abc"))
        self.assertEqual(robot.calls[1][:3], ("pinyin", "你好", "nihao"))

    def test_wait_is_zero_action(self):
        slept = []
        executor = RobotDeviceExecutor(FakeRobot(), sleep=slept.append)

        result = executor.execute(
            DeviceActionRequest(kind="wait_for_change", wait_seconds=0.75)
        )

        self.assertEqual(slept, [0.75])
        self.assertEqual(result.physical_actions, 0)

    def test_replay_executor_never_calls_hardware(self):
        executor = ReplayDeviceExecutor(
            [
                {
                    "kind": "home",
                    "transport_result": "desktop",
                },
                {
                    "kind": "tap_semantic",
                    "request": {"point": [400, 300]},
                },
            ]
        )

        first = executor.execute(DeviceActionRequest(kind="home"))
        second = executor.execute(
            DeviceActionRequest(kind="tap_semantic", point=(400, 300))
        )

        self.assertEqual(first.physical_actions + second.physical_actions, 0)
        self.assertEqual(first.replayed_actions + second.replayed_actions, 2)
        self.assertTrue(executor.complete)

    def test_replay_detects_request_mismatch(self):
        executor = ReplayDeviceExecutor(
            [{"kind": "tap_semantic", "request": {"point": [1, 2]}}]
        )

        with self.assertRaises(DeviceExecutionError):
            executor.execute(
                DeviceActionRequest(kind="tap_semantic", point=(2, 1))
            )


if __name__ == "__main__":
    unittest.main()
