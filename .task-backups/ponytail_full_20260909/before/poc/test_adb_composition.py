from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import web_app
from test_support.web_platform import (
    MockRobotController,
    _BaseAdbKeyboardCompositionTests,
)


class AdbKeyboardCompositionTests(_BaseAdbKeyboardCompositionTests):
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
