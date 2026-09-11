"""Trusted-launch descriptive target tolerance; no real model/device calls."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from PIL import Image

from agent.domain.canonical_action_protocol import (
    CanonicalActionProtocolError, bind_same_response_action, normalize_model_step_decision,
)
from agent.domain.device_execution import DeviceActionRequest
from agent.domain.qwen_task_context import QwenTaskContext
from agent.domain.universal_action_controller import UniversalActionController
from agent.infrastructure.adb_package_launcher import AdbPackageLauncher
from agent.infrastructure.device_executor import RobotDeviceExecutor
from agent.infrastructure.generic_scene_observer import (
    SingleStepGenericSceneObserver, SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
)


def decision(app="示例应用"):
    return {"status": "action", "action": "launch_app", "app": app,
            "target": {"role": "icon", "meaning": "launch_app", "label": app,
                       "evidence": ["当前画面可见应用图标"]}}


class LaunchDescriptionTests(unittest.TestCase):
    def test_pure_description_is_projected_without_mutating_raw_or_app(self):
        for app in ("抖音", "微信", "便签"):
            for description in ({"role": "icon", "meaning": "launch_app"},
                                {"role": "input", "meaning": "page_context",
                                 "element_id": "diagnostic-only", "label": "无关说明", "evidence": []}):
                with self.subTest(app=app, description=description):
                    raw = {**decision(app), "target": description}
                    before = copy.deepcopy(raw)
                    result = normalize_model_step_decision(raw)
                    self.assertEqual(before, raw)
                    self.assertEqual("launch_app", result["action"])
                    self.assertEqual(app, result["app"])
                    self.assertIsNone(result["target"])
                    self.assertEqual(result, normalize_model_step_decision(result))

    def test_top_level_action_conflicts_still_rejected(self):
        conflicts = {"tap_point": [100, 200], "start": [100, 200], "end": [200, 300],
                     "direction": "up", "element_id": "e1", "source_element_id": "e1",
                     "destination_element_id": "e2", "text": "unexpected",
                     "actions": [{"action": "home"}], "shell": "example"}
        for field, value in conflicts.items():
            with self.subTest(field=field), self.assertRaises(CanonicalActionProtocolError):
                normalize_model_step_decision({**decision(), field: value})

    def test_target_is_prose_not_geometry_commands_or_nested_payload(self):
        for field, value in {"bounds": [0, 0, 100, 100], "tap_point": [10, 20],
                             "command": "example", "actions": [{"action": "home"}],
                             "states": {}, "label": {"shell": "example"},
                             "evidence": [{"command": "example"}], "element_id": ["e1"]}.items():
            raw = decision()
            raw["target"][field] = value
            with self.subTest(field=field), self.assertRaises(CanonicalActionProtocolError):
                normalize_model_step_decision(raw)
        for target in ("icon", []):
            with self.subTest(target=target), self.assertRaises(CanonicalActionProtocolError):
                normalize_model_step_decision({**decision(), "target": target})

    def test_text_point_and_finish_do_not_gain_system_description_tolerance(self):
        for action in ("clear_verified_text", "press_enter"):
            with self.subTest(action=action), self.assertRaises(CanonicalActionProtocolError):
                normalize_model_step_decision({**decision(), "action": action, "app": None})
        with self.assertRaises(CanonicalActionProtocolError):
            normalize_model_step_decision({**decision(), "status": "finish", "action": None, "app": None})
        for action in ("tap_semantic", "dismiss_overlay", "double_tap", "long_press"):
            raw = {**decision(), "action": action, "app": None, "tap_point": [321, 654]}
            result = normalize_model_step_decision(raw)
            self.assertEqual([321, 654], result["tap_point"])
            self.assertEqual(raw["target"]["meaning"], result["target"]["meaning"])
            raw["tap_point"] = None
            with self.assertRaises(CanonicalActionProtocolError):
                normalize_model_step_decision(raw)

    def test_missing_app_is_not_inferred_from_description(self):
        for app in (None, "", "  "):
            with self.subTest(app=app), self.assertRaises(CanonicalActionProtocolError):
                normalize_model_step_decision({**decision(), "app": app})

    def test_real_observer_registry_binder_controller_executor_keep_one_launch(self):
        for app in ("抖音", "微信"):
            with self.subTest(app=app), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                payload = {
                    "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
                    "coordinate_space": {"kind": "axis_grid", "width": 1000, "height": 240},
                    "scene": {"foreground_app_id": "launcher", "screen_id": "launcher_home",
                              "summary": "当前为桌面", "elements": [], "overlays": [],
                              "stable": True, "confidence": 1.0},
                    "decision": decision(app),
                }
                raw = json.dumps(payload, ensure_ascii=False)
                calls = []
                provider = SimpleNamespace(status=lambda: {"configured": True, "model": "offline"},
                    _chat=lambda *args, **kwargs: (calls.append((args, kwargs)) or raw))
                observer = SingleStepGenericSceneObserver(provider)
                scene, normalized = observer.observe_with_decision(
                    frames=[Image.new("RGB", (160, 240), "white") for _ in range(4)],
                    goal_context={}, device_id="device-test", available_action_kinds={"launch_app", "home"},
                    response_evidence_dir=root)
                self.assertEqual(1, len(calls))
                self.assertEqual(raw, observer.last_raw_response)
                saved = json.loads(Path(observer.last_response_evidence_path).read_text(encoding="utf-8"))
                self.assertEqual(raw, saved["redacted_raw_response"])
                self.assertIsNone(normalized["target"])
                adb = root / "adb.exe"
                adb.write_bytes(b"")
                registry = root / "registry.json"
                registry.write_text(json.dumps({"version": 1, "devices": [{"device_id": "device-test",
                    "enabled": True, "adb_executable": str(adb), "adb_serial": "TEST-SERIAL",
                    "apps": [{"launch_ref": "app.sample", "aliases": [app],
                              "package": "com.example.sample"}]}]}), encoding="utf-8")
                transport_calls = []
                launcher = AdbPackageLauncher(registry, "device-test", runner=lambda argv, **kwargs:
                    (transport_calls.append(argv) or SimpleNamespace(returncode=0)))
                target = launcher.resolve(normalized["app"], normalized["app"])
                self.assertIsNone(launcher.resolve("unknown_app", "unknown_app"))
                context = QwenTaskContext(task_id="offline-task", device_id="device-test", revision=1,
                                          raw_goal=f"打开{app}")
                args = {"context": context, "observation": SimpleNamespace(scene=scene),
                        "available_action_kinds": {"launch_app", "home"}}
                with self.assertRaises(CanonicalActionProtocolError):
                    bind_same_response_action(normalized, **args, launch_target=None)
                bound = bind_same_response_action(normalized, **args, launch_target={
                    "launch_ref": target.launch_ref, "expected_app_id": target.expected_app_id,
                    "target_app_id": app, "target_app_name": app})
                self.assertNotIn("target", bound.params)
                resolved = UniversalActionController().resolve_one(bound, scene, confirmed=True)
                result = RobotDeviceExecutor(SimpleNamespace(), app_launcher=launcher).execute(
                    DeviceActionRequest(kind=resolved.kind, launch_ref=resolved.launch_ref))
                self.assertEqual(1, result.physical_actions)
                self.assertEqual(1, len(transport_calls))
                self.assertEqual([str(adb), "-s", "TEST-SERIAL", "shell", "monkey", "-p",
                                  "com.example.sample", "-c", "android.intent.category.LAUNCHER", "1"],
                                 transport_calls[0])


if __name__ == "__main__":
    unittest.main()
