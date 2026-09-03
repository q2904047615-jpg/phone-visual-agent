import unittest

from agent.domain import (
    DeviceActionRequest,
    DeviceExecutionError,
)
from agent.domain.text_transport import (
    EMPTY_TEXT_DIGEST,
    TEXT_TRANSPORT_PROTOCOL,
    TextTransportActionScope,
    TextTransportResult,
    text_digest,
)
from agent.infrastructure import (
    ReplayDeviceExecutor,
    RobotDeviceExecutor,
)


class FakeRobot:
    def __init__(self) -> None:
        self.calls = []
        self._click_receipt = None
        self._long_press_receipt = None
        self._swipe_receipt = None

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

    def vision_swipe_relative(self, start_x, start_y, end_x, end_y, direction):
        self.calls.append(("swipe", start_x, start_y, end_x, end_y, direction))
        self._swipe_receipt = {
            "right_button_down_dispatched": True,
            "right_button_up_dispatched": True,
            "seller_position_barrier_confirmed": True,
            "round_trip_position_confirmed": True,
            "mechanical_contact_ack": False,
            "requested_direction": direction,
            "step_count": 6,
            "interpolation_steps_completed": 6,
        }
        return ((start_x, start_y), (end_x, end_y))

    def consume_last_click_receipt(self):
        receipt = self._click_receipt
        self._click_receipt = None
        return receipt

    def consume_last_long_press_receipt(self):
        receipt = self._long_press_receipt
        self._long_press_receipt = None
        return receipt

    def consume_last_swipe_receipt(self):
        receipt = self._swipe_receipt
        self._swipe_receipt = None
        return receipt


def adb_keyboard_scope(*, fragment: str, expected: str) -> TextTransportActionScope:
    return TextTransportActionScope(protocol_version=TEXT_TRANSPORT_PROTOCOL, device_id="device-1",
        session_id="session-1", task_id="task-1", revision=1, action_id="action-1",
        input_field_id="field-1",
        observation_fingerprint="observation-1", prior_text_digest=text_digest(""),
        fragment_text_digest=text_digest(fragment), expected_text_digest=text_digest(expected),
        issued_at_epoch=100.0, nonce="nonce-0000000000001")


class FakeAdbKeyboardTransport:
    def __init__(self, *, status: str = "accepted") -> None:
        self.status = status
        self.calls = []

    def _result(self, scope, operation):
        attempted = self.status != "unavailable"
        accepted = self.status == "accepted"
        result = TextTransportResult(protocol_version=TEXT_TRANSPORT_PROTOCOL, device_id=scope.device_id,
            action_id=scope.action_id, nonce=scope.nonce, operation=operation, status=self.status,
            attempted=attempted, accepted=accepted,
            reason_code=None if accepted else "transport_unavailable" if not attempted else "transport_unknown",
            command_digest=None if not attempted else "a" * 64,
            receipt_digest="b" * 64 if self.status in {"accepted", "rejected"} else None)
        result.validate()
        return result

    def append_text(self, scope, text):
        self.calls.append(("append_text", scope, text))
        return self._result(scope, "append_text")

    def clear_text(self, scope):
        self.calls.append(("clear_text", scope))
        return self._result(scope, "clear_text")


class DeviceExecutorTests(unittest.TestCase):
    def test_relative_swipe_dispatches_once_and_consumes_path_receipt(self):
        robot = FakeRobot()
        executor = RobotDeviceExecutor(robot)

        result = executor.execute(DeviceActionRequest(kind="swipe_element", point=(700, 500),
            end_point=(100, 500), direction="left"))

        self.assertEqual([("swipe", 700, 500, 100, 500, "left")], robot.calls)
        self.assertEqual(1, result.physical_actions)
        self.assertEqual("left", result.hardware_receipt["requested_direction"])
        self.assertEqual(6, result.hardware_receipt["interpolation_steps_completed"])
        self.assertIsNone(robot.consume_last_swipe_receipt())

    def test_relative_swipe_without_valid_receipt_fails_after_one_attempt(self):
        robot = FakeRobot()
        robot.consume_last_swipe_receipt = lambda: None
        executor = RobotDeviceExecutor(robot)

        with self.assertRaises(DeviceExecutionError) as context:
            executor.execute(DeviceActionRequest(kind="swipe_element", point=(700, 500),
                end_point=(100, 500), direction="left"))

        self.assertEqual(1, context.exception.physical_actions)
        self.assertEqual([("swipe", 700, 500, 100, 500, "left")], robot.calls)

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

    def test_adb_keyboard_append_and_clear_never_call_mechanical_keyboard(self):
        robot = FakeRobot()
        transport = FakeAdbKeyboardTransport()
        executor = RobotDeviceExecutor(robot, text_transport=transport)
        append_scope = adb_keyboard_scope(fragment="你好🙂\nsecond", expected="你好🙂\nsecond")
        clear_scope = adb_keyboard_scope(fragment="", expected="")
        self.assertEqual(EMPTY_TEXT_DIGEST, clear_scope.fragment_text_digest)

        append = executor.execute(DeviceActionRequest(kind="input_verified_text",
            input_fragment="你好🙂\nsecond", input_method="unicode_commit", text_transport="adb_keyboard",
            text_scope=append_scope))
        clear = executor.execute(DeviceActionRequest(kind="clear_verified_text", text_transport="adb_keyboard",
            text_scope=clear_scope))

        self.assertEqual([], robot.calls)
        self.assertEqual(["append_text", "clear_text"], [item[0] for item in transport.calls])
        self.assertEqual((1, 1), (append.physical_actions, clear.physical_actions))
        self.assertEqual("accepted", append.metadata["transport_status"])

    def test_adb_keyboard_pre_send_unavailable_is_zero_action_and_never_falls_back(self):
        robot = FakeRobot()
        transport = FakeAdbKeyboardTransport(status="unavailable")
        executor = RobotDeviceExecutor(robot, text_transport=transport)

        with self.assertRaises(DeviceExecutionError) as context:
            executor.execute(DeviceActionRequest(kind="input_verified_text", input_fragment="🙂",
                input_method="unicode_commit", text_transport="adb_keyboard",
                text_scope=adb_keyboard_scope(fragment="🙂", expected="🙂")))

        self.assertEqual(0, context.exception.physical_actions)
        self.assertEqual([], robot.calls)
        self.assertEqual(1, len(transport.calls))

    def test_adb_keyboard_rejected_input_is_one_failed_action_and_never_falls_back(self):
        robot = FakeRobot()
        transport = FakeAdbKeyboardTransport(status="rejected")
        executor = RobotDeviceExecutor(robot, text_transport=transport)

        with self.assertRaises(DeviceExecutionError) as context:
            executor.execute(DeviceActionRequest(kind="input_verified_text", input_fragment="你好",
                input_method="unicode_commit", text_transport="adb_keyboard",
                text_scope=adb_keyboard_scope(fragment="你好", expected="你好")))

        self.assertEqual(1, context.exception.physical_actions)
        self.assertEqual("rejected", context.exception.metadata["transport_status"])
        self.assertEqual([], robot.calls)
        self.assertEqual(1, len(transport.calls))

    def test_adb_keyboard_unknown_clear_is_one_failed_action_and_never_falls_back(self):
        robot = FakeRobot()
        transport = FakeAdbKeyboardTransport(status="unknown")
        executor = RobotDeviceExecutor(robot, text_transport=transport)

        with self.assertRaises(DeviceExecutionError) as context:
            executor.execute(DeviceActionRequest(kind="clear_verified_text", text_transport="adb_keyboard",
                text_scope=adb_keyboard_scope(fragment="", expected="")))

        self.assertEqual(1, context.exception.physical_actions)
        self.assertEqual("unknown", context.exception.metadata["transport_status"])
        self.assertEqual([], robot.calls)
        self.assertEqual(1, len(transport.calls))

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

    def test_invalid_swipe_trajectory_is_rejected_before_transport(self):
        robot = FakeRobot()
        executor = RobotDeviceExecutor(robot)

        with self.assertRaisesRegex(DeviceExecutionError, "请求方向不一致"):
            executor.execute(
                DeviceActionRequest(
                    kind="swipe_element",
                    direction="up",
                    point=(500, 100),
                    end_point=(500, 900),
                )
            )

        self.assertEqual([], robot.calls)


if __name__ == "__main__":
    unittest.main()
