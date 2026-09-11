from __future__ import annotations
from agent.infrastructure import CameraPreviewUnavailable
from agent.infrastructure import DeviceCameraCoordinator
from agent.infrastructure import DeviceControllerRegistry
from agent.infrastructure import DeviceControllerRegistryError
from agent.infrastructure import DeviceRuntimeResourceError
from agent.infrastructure import DeviceRuntimeResourceRegistry
from PIL import Image
from agent.domain.action_capabilities import PROMOTABLE_ACTIONS
from pathlib import Path
from agent.infrastructure.robot_controller import MockRobotController as _MockRobotController
import json
import tempfile
import threading
import time
import unittest
import web_app


class DeviceControllerRegistryTests(unittest.TestCase):
    def test_default_real_device_advertises_only_actions_with_live_evidence(self) -> None:
        registry = DeviceControllerRegistry(
            web_app.DEVICE_REGISTRY_PATH,
            promotable_actions=PROMOTABLE_ACTIONS,
            mock=False,
        )
        controller = registry.controller(registry.default_device_id)

        self.assertNotIn("input_verified_text", controller.hardware_capabilities())
        self.assertTrue(controller.hardware_capabilities()["long_press"])
        self.assertTrue(controller.hardware_capabilities()["drag"])

    def test_two_devices_have_independent_controllers_and_calibrations(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "devices.json"
            path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "default_device_id": "phone-a",
                        "devices": [
                            {
                                "device_id": "phone-a",
                                "enabled": True,
                                "window_title": "controller-a",
                                "calibration_path": "calibration-a.json",
                            },
                            {
                                "device_id": "phone-b",
                                "enabled": True,
                                "window_title": "controller-b",
                                "calibration_path": "calibration-b.json",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            registry = DeviceControllerRegistry(
                path,
                promotable_actions=PROMOTABLE_ACTIONS,
                mock=False,
            )

            first = registry.controller("phone-a")
            second = registry.controller("phone-b")

        self.assertIsNot(first, second)
        self.assertEqual(first.title, "controller-a")
        self.assertEqual(second.title, "controller-b")
        self.assertTrue(str(first.calibration_path).endswith("calibration-a.json"))
        self.assertTrue(str(second.calibration_path).endswith("calibration-b.json"))
        with self.assertRaisesRegex(DeviceControllerRegistryError, "未登记"):
            registry.controller("phone-c")

    def test_duplicate_enabled_window_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "devices.json"
            path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "default_device_id": "phone-a",
                        "devices": [
                            {"device_id": "phone-a", "enabled": True, "window_title": "same"},
                            {"device_id": "phone-b", "enabled": True, "window_title": "same"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "同一个机械臂控制窗口"):
                DeviceControllerRegistry(
                    path,
                    promotable_actions=PROMOTABLE_ACTIONS,
                )

    def test_mock_devices_remain_independent_without_mutating_descriptors(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "devices.json"
            path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "default_device_id": "phone-a",
                        "devices": [
                            {
                                "device_id": "phone-a",
                                "enabled": True,
                                "window_title": "controller-a",
                                "calibration_path": "a.json",
                            },
                            {
                                "device_id": "phone-b",
                                "enabled": True,
                                "window_title": "controller-b",
                                "calibration_path": "b.json",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            registry = DeviceControllerRegistry(
                path,
                promotable_actions=PROMOTABLE_ACTIONS,
                mock=True,
            )

            first = registry.controller("phone-a")
            second = registry.controller("phone-b")
            descriptors_before = registry.descriptors()

        self.assertIsInstance(first, _MockRobotController)
        self.assertIsInstance(second, _MockRobotController)
        self.assertIsNot(first, second)
        self.assertEqual(descriptors_before, registry.descriptors())

    def test_invalid_verified_actions_fails_before_controller_construction(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "devices.json"
            path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "default_device_id": "phone-a",
                        "devices": [
                            {
                                "device_id": "phone-a",
                                "enabled": True,
                                "verified_actions": ["back", ""],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                DeviceControllerRegistryError,
                "verified_actions 格式无效",
            ):
                DeviceControllerRegistry(
                    path,
                    promotable_actions=PROMOTABLE_ACTIONS,
                )


class DeviceCameraCoordinatorTests(unittest.TestCase):
    def test_task_lease_serves_cached_preview_without_another_capture(self) -> None:
        coordinator = DeviceCameraCoordinator()
        preview_calls = []

        def capture_preview(*, quality):
            preview_calls.append(quality)
            return b"live-preview"

        first, first_cached = coordinator.capture_preview(
            capture_preview,
            quality=72,
            cache_only=False,
        )
        with coordinator.serial_session():
            cached, cached_during_task = coordinator.capture_preview(
                capture_preview,
                quality=72,
                cache_only=True,
            )
            frame = coordinator.capture_agent_frame(
                lambda: Image.new("RGB", (540, 960), "#203040")
            )

        self.assertEqual(b"live-preview", first)
        self.assertFalse(first_cached)
        self.assertEqual(b"live-preview", cached)
        self.assertTrue(cached_during_task)
        self.assertEqual([72], preview_calls)
        self.assertEqual((540, 960), frame.size)

    def test_concurrent_preview_never_waits_or_captures_behind_task_lease(self) -> None:
        coordinator = DeviceCameraCoordinator()
        coordinator.capture_preview(
            lambda *, quality: b"primed-preview",
            quality=72,
            cache_only=False,
        )
        lease_started = threading.Event()
        release_lease = threading.Event()

        def hold_task_lease():
            with coordinator.serial_session():
                lease_started.set()
                release_lease.wait(2.0)

        worker = threading.Thread(target=hold_task_lease)
        worker.start()
        self.assertTrue(lease_started.wait(1.0))
        capture_calls = []
        started = time.monotonic()
        try:
            payload, cached = coordinator.capture_preview(
                lambda *, quality: capture_calls.append(quality) or b"wrong",
                quality=72,
                cache_only=False,
            )
        finally:
            release_lease.set()
            worker.join(2.0)

        self.assertLess(time.monotonic() - started, 0.5)
        self.assertEqual(b"primed-preview", payload)
        self.assertTrue(cached)
        self.assertEqual([], capture_calls)
        self.assertFalse(worker.is_alive())

    def test_cache_only_without_a_frame_fails_closed(self) -> None:
        coordinator = DeviceCameraCoordinator()

        with self.assertRaisesRegex(CameraPreviewUnavailable, "尚无可复用"):
            coordinator.capture_preview(
                lambda *, quality: b"must-not-run",
                quality=72,
                cache_only=True,
            )

    def test_two_device_coordinators_do_not_share_preview_cache(self) -> None:
        first = DeviceCameraCoordinator()
        second = DeviceCameraCoordinator()
        first.capture_preview(
            lambda *, quality: b"phone-a",
            quality=72,
            cache_only=False,
        )

        with self.assertRaises(CameraPreviewUnavailable):
            second.capture_preview(
                lambda *, quality: b"phone-b",
                quality=72,
                cache_only=True,
            )

        cached, is_cached = first.capture_preview(
            lambda *, quality: b"wrong",
            quality=72,
            cache_only=True,
        )
        self.assertEqual(b"phone-a", cached)
        self.assertTrue(is_cached)


class DeviceRuntimeResourceRegistryTests(unittest.TestCase):
    def test_same_device_reuses_resources_and_devices_remain_isolated(self) -> None:
        registry = DeviceRuntimeResourceRegistry(("phone-a",))

        self.assertIs(
            registry.coordination_lock("phone-a"),
            registry.coordination_lock("phone-a"),
        )
        self.assertIs(
            registry.camera_coordinator("phone-a"),
            registry.camera_coordinator("phone-a"),
        )
        self.assertIsNot(
            registry.coordination_lock("phone-a"),
            registry.coordination_lock("phone-b"),
        )
        self.assertIsNot(
            registry.camera_coordinator("phone-a"),
            registry.camera_coordinator("phone-b"),
        )

    def test_concurrent_first_lookup_returns_one_stable_resource(self) -> None:
        registry = DeviceRuntimeResourceRegistry()
        barrier = threading.Barrier(9)
        results = []

        def resolve() -> None:
            barrier.wait()
            results.append(registry.coordination_lock("phone-parallel"))

        workers = [threading.Thread(target=resolve) for _index in range(8)]
        for worker in workers:
            worker.start()
        barrier.wait()
        for worker in workers:
            worker.join(timeout=2)

        self.assertEqual(8, len(results))
        self.assertTrue(all(item is results[0] for item in results))

    def test_empty_device_fails_before_resource_creation(self) -> None:
        registry = DeviceRuntimeResourceRegistry()
        with self.assertRaisesRegex(DeviceRuntimeResourceError, "device_id 不能为空"):
            registry.coordination_lock("  ")
        with self.assertRaisesRegex(DeviceRuntimeResourceError, "device_id 不能为空"):
            registry.camera_coordinator("")


if __name__ == "__main__":
    unittest.main()
