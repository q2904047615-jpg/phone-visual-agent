from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock

from agent.infrastructure.device_task_registry import DeviceTaskRegistry
from agent.infrastructure.manual_motion import (
    BackgroundWindowTransport,
    ManualMotionMonitor,
    MotionProfile,
    map_baseline_point,
)


class FakeUser32:
    def __init__(self):
        self.messages = []

    def PostMessageW(self, hwnd, message, flags, lparam):
        self.messages.append((hwnd, message, flags, lparam))
        return 1


class ManualMotionTests(unittest.TestCase):
    def test_mapping_uses_seller_baseline_without_cursor_calls(self):
        self.assertEqual((539, 959), map_baseline_point((539, 959), 540, 960))
        self.assertEqual((808, 1918), map_baseline_point((539, 959), 810, 1920))

    def test_background_click_posts_two_pairs_without_activation(self):
        user32 = FakeUser32()
        controller = Mock(title="seller")
        transport = BackgroundWindowTransport(
            controller,
            user32=user32,
            find_window=lambda _: (7, "seller"),
            client_geometry=lambda _: (0, 0, 540, 1010),
            camera_height=lambda _w, _h: 960,
        )
        profile = MotionProfile("device-1", "click_pair", (10, 20), (30, 40), 1, 60)
        receipt = transport.click_pair(profile, threading.Event())
        self.assertEqual(4, receipt["input_events_dispatched"])
        self.assertFalse(receipt["cursor_moved"])
        self.assertEqual([0x0201, 0x0202, 0x0201, 0x0202], [item[1] for item in user32.messages])

    def test_swipe_releases_right_button_when_stopped(self):
        user32 = FakeUser32()
        controller = Mock(title="seller")
        transport = BackgroundWindowTransport(
            controller,
            user32=user32,
            find_window=lambda _: (7, "seller"),
            client_geometry=lambda _: (0, 0, 540, 1010),
            camera_height=lambda _w, _h: 960,
        )
        event = threading.Event()
        event.set()
        with self.assertRaises(InterruptedError):
            transport.swipe(MotionProfile("device-1", "swipe", (10, 20), (30, 40), 1, 60), event)
        self.assertEqual(0x0205, user32.messages[-1][1])

    def test_monitor_runs_bounded_mock_cycles_and_releases_registry(self):
        registry = DeviceTaskRegistry()
        receipts = []
        class FakeTransport:
            def swipe(self, profile, event):
                receipts.append(profile.point_a)
                return {"input_events_dispatched": 8, "cursor_moved": False}
        with tempfile.TemporaryDirectory() as directory:
            monitor = ManualMotionMonitor(
                registry=registry,
                transport_factory=lambda _device: FakeTransport(),
                output_dir=Path(directory),
            )
            record = monitor.start(MotionProfile("device-1", "swipe", (1, 1), (2, 2), .01, .06))
            deadline = time.time() + 2
            while monitor.get(record["motion_id"])["status"] in monitor.ACTIVE and time.time() < deadline:
                time.sleep(.01)
            finished = monitor.get(record["motion_id"])
            self.assertEqual("completed", finished["status"])
            self.assertGreaterEqual(finished["cycles_dispatched"], 1)
            self.assertIsNone(registry.active_session("device-1"))
            self.assertTrue(receipts)
            self.assertTrue(list(Path(directory).glob("*.jsonl")))

    def test_stop_wakes_wait_without_a_second_cycle(self):
        registry = DeviceTaskRegistry()
        calls = []
        class FakeTransport:
            def swipe(self, profile, event):
                calls.append(True)
                return {"input_events_dispatched": 8}
        with tempfile.TemporaryDirectory() as directory:
            monitor = ManualMotionMonitor(registry=registry, transport_factory=lambda _: FakeTransport(), output_dir=Path(directory))
            record = monitor.start(MotionProfile("device-1", "swipe", (1, 1), (2, 2), 10, 100))
            monitor.stop(record["motion_id"])
            deadline = time.time() + 2
            while monitor.get(record["motion_id"])["status"] in monitor.ACTIVE and time.time() < deadline:
                time.sleep(.01)
            self.assertEqual("stopped", monitor.get(record["motion_id"])["status"])
            self.assertFalse(calls)
            self.assertIsNone(registry.active_session("device-1"))


if __name__ == "__main__":
    unittest.main()
