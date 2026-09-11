from __future__ import annotations
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, patch
import unittest
import web_app
from agent.infrastructure.adb_keyboard_transport import (
    ADB_KEYBOARD_RUNTIME_REGISTRY_VERSION, AdbKeyboardRuntimeRegistry,
)
from agent.infrastructure.robot_controller import RobotController
from test_adb_keyboard_transport import profile
from test_support.web_platform import (
    MockRobotController,
)


class AdbKeyboardCompositionTests(unittest.TestCase):
    def test_optional_adb_preserves_mechanics_without_borrowing_another_device(self) -> None:
        # Exercise the real registries, composition factory, adapter and executor;
        # replace only hardware and external process I/O.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "adb.exe"
            executable.write_bytes(b"offline fixture")
            text_registry = root / "text.json"
            text_registry.write_text(json.dumps({
                "version": ADB_KEYBOARD_RUNTIME_REGISTRY_VERSION,
                "devices": [{"profile": profile().to_dict(), "adb_executable": str(executable)}],
            }), encoding="utf-8")
            app_registry = root / "apps.json"
            app_registry.write_text(json.dumps({"version": 1, "devices": [{
                "device_id": "device-local-01", "enabled": True,
                "adb_executable": str(executable), "adb_serial": "serial-01",
                "apps": [{"launch_ref": "fixture-app", "aliases": ["fixture"],
                    "package": "org.example.fixture"}],
            }]}), encoding="utf-8")
            runner = Mock(side_effect=AssertionError("must not contact a device"))
            registry = AdbKeyboardRuntimeRegistry(text_registry, runner=runner)
            controllers = {device_id: MockRobotController(device_id=device_id)
                for device_id in ("device-local-01", "device-local-02", "device-local-03")}
            with (
                patch.object(web_app.runtime, "adb_keyboard_runtime", registry),
                patch.object(web_app.runtime, "_app_launchers", {}),
                patch.object(web_app, "APP_PACKAGE_REGISTRY_PATH", app_registry),
                patch.object(web_app.runtime, "controller_for_device", side_effect=controllers.__getitem__),
                patch.object(web_app.runtime.vision_provider, "status", return_value={"configured": True}),
            ):
                for device_id, controller in controllers.items():
                    with self.subTest(device_id=device_id):
                        web_app._require_agent_device_ready(device_id)
                        adapters = (
                            web_app.runtime.universal_agent_orchestrator.adapter_factory(device_id),
                            web_app.runtime.capability_trial_orchestrator(controller, "long_press")
                                .adapter_factory(device_id),
                        )
                        for index, adapter in enumerate(adapters):
                            supported = adapter.supported_action_kinds()
                            self.assertTrue({"tap_semantic", "home", "back", "open_recent_apps",
                                "scroll", "swipe_element"} <= supported)
                            adb_actions = {"input_verified_text", "clear_verified_text", "press_enter"}
                            if device_id == "device-local-01":
                                self.assertTrue(adb_actions <= supported)
                                self.assertIs(registry.transport_for_device(device_id), adapter.text_transport)
                                self.assertEqual(index == 0, "launch_app" in supported)
                                self.assertEqual(("fixture",) if index == 0 else (), adapter.supported_app_aliases())
                            else:
                                self.assertFalse(adb_actions & supported)
                                self.assertNotIn("launch_app", supported)
                                self.assertIsNone(adapter.text_transport)
                                self.assertIsNone(adapter.device_executor.text_transport)
                                self.assertEqual((), adapter.supported_app_aliases())
                        self.assertEqual([], controller.executions)
            runner.assert_not_called()

    def test_begin_task_does_not_dispatch_an_unobserved_position_click(self) -> None:
        controller = RobotController(device_id="device-local-02")
        controller.request_stop()
        with (
            patch("agent.infrastructure.robot_controller.seller_gui.find_window") as find_window,
            patch("agent.infrastructure.robot_controller.seller_gui.user32.mouse_event") as mouse_event,
        ):
            controller.begin_new_task()
        self.assertFalse(controller.stop_event.is_set())
        find_window.assert_not_called()
        mouse_event.assert_not_called()

    def test_main_and_capability_adapters_receive_same_device_transport(self) -> None:
        device_id = web_app.runtime.device_controllers.default_device_id
        transport = object()
        controller = MockRobotController(device_id=device_id)
        main_adapter = object()
        capability_adapter = object()

        with (
            patch.object(
                web_app.runtime,
                "text_transport_for_device",
                return_value=transport,
            ) as transport_for_device,
            patch.object(
                web_app.runtime,
                "controller_for_device",
                return_value=controller,
            ),
            patch.object(
                web_app.runtime,
                "app_launcher_for_device",
                return_value=None,
            ),
            patch.object(
                web_app,
                "GenericSingleActionAdapter",
                return_value=main_adapter,
            ) as adapter_factory,
        ):
            built_main = web_app.runtime.universal_agent_orchestrator.adapter_factory(
                device_id
            )

        self.assertIs(main_adapter, built_main)
        self.assertIs(transport, adapter_factory.call_args.kwargs["text_transport"])
        self.assertNotIn("foreground_identity_provider", adapter_factory.call_args.kwargs)
        transport_for_device.assert_called_once_with(device_id)

        with (
            patch.object(
                web_app.runtime,
                "text_transport_for_device",
                return_value=transport,
            ) as capability_transport_for_device,
            patch.object(
                web_app,
                "GenericSingleActionAdapter",
                return_value=capability_adapter,
            ) as capability_adapter_factory,
        ):
            orchestrator = web_app.runtime.capability_trial_orchestrator(
                controller,
                "long_press",
            )
            built_capability = orchestrator.adapter_factory(device_id)

        self.assertIs(capability_adapter, built_capability)
        self.assertIs(
            transport,
            capability_adapter_factory.call_args.kwargs["text_transport"],
        )
        self.assertNotIn(
            "foreground_identity_provider",
            capability_adapter_factory.call_args.kwargs,
        )
        capability_transport_for_device.assert_called_once_with(device_id)

    def test_runtime_lifecycle_has_no_companion_bridge_and_stops_controller(self) -> None:
        events: list[str] = []
        runtime = web_app.Runtime.__new__(web_app.Runtime)
        runtime.controller = SimpleNamespace(
            request_stop=lambda: events.append("controller:stop")
        )

        runtime.start()
        runtime.shutdown()

        self.assertEqual(["controller:stop"], events)

    def test_project_api_exposes_no_raw_companion_text_route(self) -> None:
        paths = {
            str(getattr(route, "path", ""))
            for route in web_app.app.routes
        }
        self.assertFalse(any("companion-ime" in path for path in paths))
        self.assertFalse(any("text-transport" in path for path in paths))

    def test_runtime_has_no_adb_page_observation_authority(self) -> None:
        root = Path(__file__).resolve().parent
        sources = "\n".join((root / relative).read_text(encoding="utf-8") for relative in (
            "web_app.py",
            "agent/infrastructure/generic_action_adapter.py",
            "agent/infrastructure/generic_scene_observer.py",
        ))
        self.assertFalse((root / "agent" / "domain" / "foreground_app_identity.py").exists())
        for forbidden in ("foreground_identity_provider", "trusted_foreground_identity",
            "supports_trusted_foreground_identity", "adb_dumpsys"):
            self.assertNotIn(forbidden, sources)


if __name__ == "__main__":
    unittest.main()
