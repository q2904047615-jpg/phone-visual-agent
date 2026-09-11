from __future__ import annotations
import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from PIL import Image
from agent.domain.generic_goal import GenericIntentDraft
from agent.application.qwen_visual_decision import QwenVisualDecisionObserver
from agent.domain.canonical_action_protocol import (
    CanonicalActionProtocolError,
    bind_same_response_action,
)
from agent.domain.device_execution import DeviceActionRequest
from agent.domain.semantic_action import SemanticAction
from agent.domain.qwen_task_context import QwenTaskContext
from agent.domain.trusted_observation import TrustedObservation
from agent.domain.ui_scene import UIElement, UIScene
from agent.domain.universal_action_controller import (
    UniversalActionController,
    UniversalActionError,
)
from agent.domain.visual_evidence import LocalFrameStability
from agent.infrastructure.adb_package_launcher import AdbPackageLauncher
from agent.infrastructure.device_executor import RobotDeviceExecutor
from agent.infrastructure.generic_action_adapter import GenericSingleActionAdapter


def _scene(
    *,
    app_id: str,
    fingerprint: str,
    elements: tuple[UIElement, ...] = (),
) -> UIScene:
    return UIScene(
        app_id=app_id,
        screen_id="home",
        summary=f"当前前台为 {app_id}",
        elements=elements,
        stable=True,
        confidence=0.98,
        fingerprint=fingerprint,
    )


def _launcher_icon(app_name: str) -> UIElement:
    return UIElement(
        element_id="target-app-icon",
        role="icon",
        meaning="app_launcher_target",
        label=app_name,
        bounds=(0.15, 0.20, 0.35, 0.40),
        confidence=0.98,
        states={
            "visible": True,
            "enabled": True,
            "fully_visible": True,
            "goal_relevant": True,
        },
        evidence=(f"桌面可见 {app_name} 图标",),
    )


def _launch_context(*, app_id: str, app_name: str) -> QwenTaskContext:
    return QwenTaskContext(task_id=f"task-open-{app_id}",device_id="device-1",revision=1,
        raw_goal=f"打开{app_name}")


def _trusted(scene: UIScene) -> TrustedObservation:
    return TrustedObservation(
        observation_id="obs_0123456789abcdef",
        device_id="device-1",
        fingerprint=scene.fingerprint,
        scene=scene,
        local_stability=LocalFrameStability(
            stable=True,
            mean_delta=0.0,
            max_delta=0.0,
            frame_count=1,
            threshold=1.0,
            reason="test",
        ),
        selected_frame_index=0,
        frame_sharpness_scores=(1.0,),
    )


def _write_registry(root: Path, payload: dict) -> Path:
    path = root / "app_package_registry.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def _registry_payload(adb_executable: Path) -> dict:
    return {
        "version": 1,
        "devices": [
            {
                "device_id": "device-1",
                "enabled": True,
                "adb_executable": str(adb_executable),
                "adb_serial": "SERIAL-001",
                "apps": [
                    {
                        "launch_ref": "launch_ref.sample_app",
                        "aliases": ["sample_app", "示例应用"],
                        "package": "com.example.sample",
                    },
                    {
                        "launch_ref": "launch_ref.notes",
                        "aliases": ["notes", "便签"],
                        "package": "com.example.notes",
                    },
                ],
            },
        ],
    }


class _SequenceObserver:
    def __init__(self, scenes: tuple[UIScene, ...]) -> None:
        self.scenes = list(scenes)
        self.calls = 0

    def observe_with_decision(self, *, frames, goal_context=None, **_kwargs):
        self.calls += 1
        return self.scenes.pop(0), {"status": "finish", "evidence_refs": ["scene.summary"]}


def _launch_goal() -> GenericIntentDraft:
    return GenericIntentDraft(
        understood=True,
        app_id="sample_app",
        app_name="示例应用",
        objective="打开示例应用",
        success_criteria={"description": "示例应用已在前台"},
    )


class LaunchAppCanonicalTests(unittest.TestCase):
    def test_two_apps_bind_current_qwen_action_to_one_opaque_launch_target(self) -> None:
        samples = (
            ("sample_app", "示例应用", "launch_ref.sample_app", "com.example.sample"),
            ("notes", "便签", "launch_ref.notes", "com.example.notes"),
        )
        for app_id, app_name, launch_ref, expected_app_id in samples:
            with self.subTest(app_id=app_id):
                context = _launch_context(app_id=app_id, app_name=app_name)
                observation = _trusted(_scene(
                    app_id="launcher", fingerprint=f"before-{app_id}",
                ))
                action = bind_same_response_action(
                    {"action": "launch_app"},
                    context=context,
                    observation=observation,
                    available_action_kinds={"launch_app", "tap_semantic", "home"},
                    launch_target={
                        "launch_ref": launch_ref,
                        "expected_app_id": expected_app_id,
                        "target_app_id": app_id,
                        "target_app_name": app_name,
                    },
                )

                self.assertEqual("launch_app", action.action)
                self.assertEqual(
                    {
                        "target_surface_id": "target_app",
                        "target_app_id": app_id,
                        "target_app_name": app_name,
                        "launch_ref": launch_ref,
                        "expected_app_id": expected_app_id,
                    },
                    action.params,
                )
                self.assertNotIn("package_name", action.params)
                self.assertNotIn("shell", action.params)
                self.assertNotIn("command", action.params)

    def test_qwen_same_response_selects_launch_and_local_code_only_maps_it(self) -> None:
        context = _launch_context(app_id="sample_app", app_name="示例应用")
        observation = _trusted(_scene(app_id="launcher", fingerprint="launch-selector-frame"))

        class Provider:
            calls = 0

            @staticmethod
            def status() -> dict:
                return {"configured": True, "model": "fake-qwen"}

            def _chat(self, *_args, **_kwargs) -> str:
                self.calls += 1
                raise AssertionError("同一观察响应已含决策，不应发起第二次模型调用")

        provider = Provider()
        selector = QwenVisualDecisionObserver(
            provider,
            trusted_observation_frame_validator=lambda *_args, **_kwargs: None,
        )
        decision = selector.decide(
            frames=[Image.new("RGB", (8, 8), "white")],
            task_context=context,
            trusted_observation=observation,
            available_action_kinds={"launch_app", "tap_semantic", "home"},
            launch_target={
                "launch_ref": "launch_ref.sample_app",
                "expected_app_id": "com.example.sample",
                "target_app_id": "sample_app",
                "target_app_name": "示例应用",
            },
            model_decision={
                "status": "action",
                "action": "launch_app", "app": "设置",
                "element_id": None,
                "source_element_id": None,
                "destination_element_id": None,
                "direction": None,
                "evidence_refs": [],
                "confidence": 0.96,
                "reason": "当前目标是启动已登记应用",
            },
        )

        self.assertEqual(0, provider.calls)
        self.assertEqual(0, selector.last_diagnostics["model_calls"])
        self.assertEqual("action", decision.proposal.status)
        self.assertEqual("launch_app", decision.proposal.action.action)
        self.assertEqual("launch_ref.sample_app", decision.proposal.action.params["launch_ref"])

    def test_missing_launch_mapping_rejects_without_locally_selecting_a_fallback(self) -> None:
        context = _launch_context(app_id="sample_app", app_name="示例应用")
        observation = _trusted(_scene(
            app_id="launcher",
            fingerprint="launcher-before",
            elements=(_launcher_icon("示例应用"),),
        ))
        with self.assertRaisesRegex(CanonicalActionProtocolError, "可信包名映射"):
            bind_same_response_action(
                {"action": "launch_app"},
                context=context,
                observation=observation,
                available_action_kinds={"launch_app", "tap_semantic", "home"},
                launch_target=None,
            )


class AdbPackageLauncherTests(unittest.TestCase):
    def test_registry_resolves_opaque_ref_and_launcher_uses_fixed_argv(self) -> None:
        launcher_class = AdbPackageLauncher
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            adb_path = root / "adb.exe"
            adb_path.write_bytes(b"")
            registry_path = _write_registry(root, _registry_payload(adb_path))
            calls: list[tuple[list[str], dict]] = []

            def runner(argv, **kwargs):
                calls.append((list(argv), dict(kwargs)))
                return SimpleNamespace(returncode=0, stdout="Events injected: 1", stderr="")

            launcher = launcher_class(
                registry_path,
                "device-1",
                runner=runner,
                timeout_seconds=7.5,
            )
            target = launcher.resolve("sample_app", "示例应用")
            self.assertIsNotNone(target)
            self.assertEqual("launch_ref.sample_app", target.launch_ref)
            self.assertEqual("com.example.sample", target.expected_app_id)
            self.assertIsNone(launcher.resolve("unknown_app", "未知应用"))

            request = DeviceActionRequest(
                kind="launch_app",
                launch_ref=target.launch_ref,
            )
            request.validate()
            launcher.launch(request.launch_ref)

            self.assertEqual(1, len(calls))
            argv, kwargs = calls[0]
            self.assertEqual(
                [
                    str(adb_path),
                    "-s",
                    "SERIAL-001",
                    "shell",
                    "monkey",
                    "-p",
                    "com.example.sample",
                    "-c",
                    "android.intent.category.LAUNCHER",
                    "1",
                ],
                argv,
            )
            self.assertEqual(
                {
                    "check": False,
                    "capture_output": True,
                    "text": True,
                    "timeout": 7.5,
                },
                kwargs,
            )
            self.assertNotIn("shell", kwargs)

    def test_registry_rejects_invalid_package_and_control_fields(self) -> None:
        launcher_class = AdbPackageLauncher
        invalid_packages = (
            "",
            "wechat",
            "com.example/.Main",
            "com.example.app;rm",
            "$(whoami)",
            "com.example bad",
            "com.example\nbad",
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            adb_path = root / "adb.exe"
            adb_path.write_bytes(b"")
            for index, package in enumerate(invalid_packages):
                with self.subTest(package=package):
                    payload = _registry_payload(adb_path)
                    payload["devices"][0]["apps"][0]["package"] = package
                    path = root / f"invalid-package-{index}.json"
                    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                    with self.assertRaises((ValueError, RuntimeError)):
                        launcher_class(path, "device-1", runner=lambda *_args, **_kwargs: None)

            for field in ("command", "shell", "intent", "component"):
                with self.subTest(forbidden_field=field):
                    payload = _registry_payload(adb_path)
                    payload["devices"][0]["apps"][0][field] = "user-controlled"
                    path = root / f"forbidden-{field}.json"
                    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                    with self.assertRaises((ValueError, RuntimeError)):
                        launcher_class(path, "device-1", runner=lambda *_args, **_kwargs: None)

            payload = _registry_payload(adb_path)
            payload["devices"][0]["adb_executable"] = "python.exe"
            path = _write_registry(root, payload)
            with self.assertRaises((ValueError, RuntimeError)):
                launcher_class(path, "device-1", runner=lambda *_args, **_kwargs: None)

            payload = _registry_payload(adb_path)
            payload["devices"].append(copy.deepcopy(payload["devices"][0]))
            path = _write_registry(root, payload)
            with self.assertRaises((ValueError, RuntimeError)):
                launcher_class(path, "device-1", runner=lambda *_args, **_kwargs: None)

    def test_unavailable_registry_transport_resolves_none(self) -> None:
        launcher_class = AdbPackageLauncher
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cases = []

            missing_adb = _registry_payload(root / "missing" / "adb.exe")
            cases.append(("missing_adb", missing_adb))

            adb_path = root / "adb.exe"
            adb_path.write_bytes(b"")
            missing_serial = _registry_payload(adb_path)
            missing_serial["devices"][0]["adb_serial"] = ""
            cases.append(("missing_serial", missing_serial))

            disabled = _registry_payload(adb_path)
            disabled["devices"][0]["enabled"] = False
            cases.append(("disabled", disabled))

            for name, payload in cases:
                with self.subTest(name=name):
                    path = root / f"{name}.json"
                    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                    launcher = launcher_class(
                        path,
                        "device-1",
                        runner=lambda *_args, **_kwargs: None,
                    )
                    self.assertIsNone(launcher.resolve("sample_app", "示例应用"))

            launcher = launcher_class(
                _write_registry(root, _registry_payload(adb_path)),
                "unregistered-device",
                runner=lambda *_args, **_kwargs: None,
            )
            self.assertIsNone(launcher.resolve("sample_app", "示例应用"))

    def test_registry_conflicts_and_transport_failures_are_fail_closed(self) -> None:
        launcher_class = AdbPackageLauncher
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            adb_path = root / "adb.exe"
            adb_path.write_bytes(b"")

            alias_conflict = _registry_payload(adb_path)
            alias_conflict["devices"][0]["apps"][1]["aliases"] = ["sample_app"]
            with self.assertRaises(RuntimeError) as raised:
                launcher_class(_write_registry(root, alias_conflict), "device-1")
            self.assertEqual(
                "App 映射的 ref、alias 或 package 缺失/重复。",
                str(raised.exception),
            )

            profile_extra_field = _registry_payload(adb_path)
            profile_extra_field["devices"][0]["intent"] = "forbidden"
            with self.assertRaises(RuntimeError) as raised:
                launcher_class(_write_registry(root, profile_extra_field), "device-1")
            self.assertEqual("device profile 字段不符合合同。", str(raised.exception))

            def timeout_runner(*_args, **_kwargs):
                raise subprocess.TimeoutExpired(cmd="adb", timeout=1)

            launcher = launcher_class(
                _write_registry(root, _registry_payload(adb_path)),
                "device-1",
                runner=timeout_runner,
            )
            with self.assertRaises(RuntimeError) as raised:
                launcher.launch("launch_ref.sample_app")
            self.assertEqual("ADB 包名启动超时。", str(raised.exception))

            launcher = launcher_class(
                _write_registry(root, _registry_payload(adb_path)),
                "device-1",
                runner=lambda *_args, **_kwargs: SimpleNamespace(returncode=7),
            )
            with self.assertRaises(RuntimeError) as raised:
                launcher.launch("launch_ref.sample_app")
            self.assertEqual("ADB 包名启动失败：returncode=7", str(raised.exception))

    def test_device_executor_launches_exactly_once_without_robot_contact(self) -> None:
        calls: list[str] = []
        launcher = SimpleNamespace(launch=lambda launch_ref: calls.append(launch_ref))
        result = RobotDeviceExecutor(SimpleNamespace(), app_launcher=launcher).execute(
            DeviceActionRequest(kind='launch_app', launch_ref='launch_ref.sample_app'))
        self.assertEqual(['launch_ref.sample_app'], calls)
        self.assertEqual(1, result.physical_actions)
        self.assertEqual({'transport': 'adb_package_launch', 'mechanical_contact_ack': False,
            'transport_status': 'accepted'}, result.metadata)


class LaunchAppVerificationTests(unittest.TestCase):
    @staticmethod
    def _candidate_action(
        *,
        app_id: str = "sample_app",
        app_name: str = "示例应用",
        launch_ref: str = "launch_ref.sample_app",
        expected_app_id: str = "com.example.sample",
    ) -> tuple[UIScene, SemanticAction]:
        before = _scene(app_id="launcher", fingerprint="before-launch")
        action = bind_same_response_action(
            {"action": "launch_app"},
            context=_launch_context(app_id=app_id, app_name=app_name),
            observation=_trusted(before),
            available_action_kinds={"launch_app", "tap_semantic", "home"},
            launch_target={
                "launch_ref": launch_ref,
                "expected_app_id": expected_app_id,
                "target_app_id": app_id,
                "target_app_name": app_name,
            },
        )
        return before, action

    @classmethod
    def _execute_through_adapter(cls, runner) -> tuple[object, int, int, int]:
        before, action = cls._candidate_action()
        after = _scene(
            app_id="com.example.sample",
            fingerprint="after-launch",
        )
        observer = _SequenceObserver((after,))
        capture_calls = 0

        def capture() -> Image.Image:
            nonlocal capture_calls
            capture_calls += 1
            return Image.new("RGB", (540, 960), "gray")

        transport_calls = 0

        def recording_runner(argv, **kwargs):
            nonlocal transport_calls
            transport_calls += 1
            return runner(argv, **kwargs)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            adb_path = root / "adb.exe"
            adb_path.write_bytes(b"")
            launcher = AdbPackageLauncher(
                _write_registry(root, _registry_payload(adb_path)),
                "device-1",
                runner=recording_runner,
            )
            result = GenericSingleActionAdapter(
                capture=capture,
                observer=observer,
                robot=SimpleNamespace(device_id="device-1"),
                app_launcher=launcher,
                controller=UniversalActionController(),
                frame_interval=0,
                post_action_settle=0,
                post_action_timeout=1,
                device_id="device-1",
            ).execute(
                requested_action=action,
                planned_scene=before,
                planned_frames=tuple(Image.new("RGB", (540, 960), "gray") for _ in range(4)),
                goal=_launch_goal(),
                confirmed=True,
            )
        return result, transport_calls, capture_calls, observer.calls

    def test_adapter_launch_success_reobserves_and_exposes_transport_metadata(self) -> None:
        result, transport_calls, capture_calls, observer_calls = self._execute_through_adapter(
            lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout="Events injected: 1", stderr=""))

        self.assertEqual(1, transport_calls)
        self.assertEqual(8, capture_calls)
        self.assertEqual(1, observer_calls)
        self.assertEqual(1, result.physical_actions)
        self.assertEqual("executed", result.action_outcome)
        execution_metadata = result.to_dict()["execution_metadata"]
        self.assertEqual("adb_package_launch", execution_metadata["transport"])
        self.assertIs(False, execution_metadata["mechanical_contact_ack"])

    def test_attempted_transport_failure_is_not_repeated_and_still_reobserves(self) -> None:
        def timeout_runner(*_args, **_kwargs):
            raise subprocess.TimeoutExpired(cmd="adb", timeout=1)

        cases = (
            ("timeout", timeout_runner),
            ("nonzero", lambda *_args, **_kwargs: SimpleNamespace(returncode=7, stdout="", stderr="failed")),
        )
        for name, runner in cases:
            with self.subTest(name=name):
                result, transport_calls, capture_calls, observer_calls = self._execute_through_adapter(runner)

                self.assertEqual(1, transport_calls)
                self.assertEqual(8, capture_calls)
                self.assertEqual(1, observer_calls)
                self.assertEqual(1, result.physical_actions)
                self.assertEqual("executed", result.action_outcome)
                execution_metadata = result.to_dict()["execution_metadata"]
                self.assertEqual("adb_package_launch", execution_metadata["transport"])
                self.assertIs(False, execution_metadata["mechanical_contact_ack"])

    def test_controller_does_not_rejudge_post_launch_app_identity(self) -> None:
        before, action = self._candidate_action()
        controller = UniversalActionController()
        resolved = controller.resolve_one(action, before, confirmed=True)
        self.assertEqual("launch_app", resolved.kind)
        self.assertEqual("launch_ref.sample_app", resolved.launch_ref)

        after_scenes = (
            _scene(app_id="com.example.sample", fingerprint="after-package"),
            _scene(app_id="sample_app", fingerprint="after-semantic-id"),
            _scene(app_id="示例应用", fingerprint="after-semantic-name"),
            _scene(app_id="com.example.other", fingerprint="after-other"),
            _scene(app_id="unknown", fingerprint="after-unknown"),
            _scene(app_id="launcher", fingerprint="after-launcher"),
        )
        for after in after_scenes:
            with self.subTest(after_app=after.foreground_app_id, fingerprint=after.fingerprint):
                self.assertEqual((), controller.verify_after_action(resolved, before, after))

if __name__ == "__main__":
    unittest.main()
