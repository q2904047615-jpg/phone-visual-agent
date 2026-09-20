"""Runtime composition for the local Agent service.

The HTTP module owns routes; this module owns construction and lifecycle of
the application graph.  No route or HTTP type is imported here.
"""

from __future__ import annotations

import os
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from agent.application import UniversalAgentSessionApplicationService
from agent.application.qwen_visual_decision import QwenVisualDecisionObserver
from agent.application.universal_agent_orchestrator import UniversalAgentOrchestrator
from agent.domain.action_catalog import PROMOTABLE_ACTION_KINDS
from agent.domain.universal_action_controller import UniversalActionController
from agent.infrastructure import (
    DeviceControllerRegistry as InfrastructureDeviceControllerRegistry,
    DeviceRuntimeResourceRegistry,
    DeviceTaskRegistry,
    FileSystemAgentEvidenceStore,
    InMemoryAgentSessionRepository,
    SHARED_DEVICE_LEASE_DIR,
)
from agent.infrastructure.adb_keyboard_transport import AdbKeyboardRuntimeRegistry
from agent.infrastructure.adb_package_launcher import AdbPackageLauncher
from agent.infrastructure.capability_acceptance import CapabilityAcceptanceError
from agent.infrastructure.capability_acceptance_runtime import CapabilityAcceptanceManager
from agent.infrastructure.dashscope_vision_provider import DashScopeVisionProvider
from agent.infrastructure.generic_action_adapter import GenericSingleActionAdapter
from agent.infrastructure.generic_scene_observer import SingleStepGenericSceneObserver
from agent.infrastructure.robot_controller import MockRobotController, RobotController
from agent.infrastructure.task_screenshot_history import cleanup_evidence_runs
from agent.infrastructure.trusted_observation_frames import (
    build_trusted_observation,
    validate_trusted_observation_against_frames,
)
from agent.infrastructure import seller_window_adapter as seller_gui


def current_code_revision(repository: Path) -> str:
    """Return a reproducible revision; dirty worktrees are never promotable."""

    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        ).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return os.environ.get("ROBOT_CODE_REVISION", "unversioned")
    if not revision:
        return os.environ.get("ROBOT_CODE_REVISION", "unversioned")
    return revision + ("+dirty" if dirty else "")


class Runtime:
    def __init__(
        self,
        *,
        device_registry_path: Path,
        app_package_registry_path: Path,
        adb_keyboard_registry_path: Path,
        output_dir: Path,
        ensure_device_ready: Callable[[str], None],
        exclusive_device_session: Callable[[str], Any],
        code_revision_provider: Callable[[], str],
    ) -> None:
        self.app_package_registry_path = app_package_registry_path
        self.ensure_device_ready = ensure_device_ready
        self.exclusive_device_session = exclusive_device_session
        self.code_revision_provider = code_revision_provider
        self.mock_mode = os.environ.get("ROBOT_WEB_MOCK") == "1"
        self.hardware_mode = (not self.mock_mode) and seller_gui.IS_WINDOWS
        self.loaded_code_revision = code_revision_provider()
        cleanup_evidence_runs(output_dir)
        self.device_controllers = InfrastructureDeviceControllerRegistry(
            device_registry_path,
            promotable_actions=PROMOTABLE_ACTION_KINDS,
            mock=self.mock_mode,
        )
        self.controller: RobotController = self.device_controllers.controller(
            self.device_controllers.default_device_id
        )
        self._app_launchers: dict[str, AdbPackageLauncher | None] = {}
        self.vision_provider = DashScopeVisionProvider(enable_thinking=True)
        self.adb_keyboard_runtime = AdbKeyboardRuntimeRegistry(adb_keyboard_registry_path)
        self.generic_scene_observer = SingleStepGenericSceneObserver(self.vision_provider)
        self.qwen_visual_decision_observer = QwenVisualDecisionObserver(
            self.vision_provider,
            trusted_observation_frame_validator=validate_trusted_observation_against_frames,
        )
        self.device_task_registry = DeviceTaskRegistry(lease_directory=SHARED_DEVICE_LEASE_DIR)
        self.universal_agent_orchestrator = UniversalAgentOrchestrator(
            qwen_observer=self.qwen_visual_decision_observer,
            adapter_factory=lambda device_id: GenericSingleActionAdapter(
                capture=lambda: self.capture_agent_frame(device_id),
                observer=self.generic_scene_observer,
                robot=self.controller_for_device(device_id),
                app_launcher=self.app_launcher_for_device(device_id),
                controller=UniversalActionController(),
                device_id=device_id,
                text_transport=self.text_transport_for_device(device_id),
            ),
            trusted_observation_factory=build_trusted_observation,
            evidence_store_factory=FileSystemAgentEvidenceStore,
            device_registry=self.device_task_registry,
        )
        self.agent_session_repository = InMemoryAgentSessionRepository()
        self.universal_agent_session_service = UniversalAgentSessionApplicationService(
            orchestrator_provider=lambda: self.universal_agent_orchestrator,
            sessions=self.agent_session_repository,
            ensure_device_ready=self.ensure_device_ready,
            exclusive_device_session=self.exclusive_device_session,
            begin_new_task=self._begin_new_task_for_device,
        )
        self.capability_acceptance_manager = CapabilityAcceptanceManager(
            provisional_controller_factory=self.device_controllers.provisional_controller,
            orchestrator_factory=self.capability_trial_orchestrator,
            device_registry=self.device_task_registry,
            output_dir=output_dir,
            registry_path=device_registry_path,
            code_revision_provider=self.capability_code_revision,
        )
        self.device_runtime_resources = DeviceRuntimeResourceRegistry(
            tuple(item["device_id"] for item in self.device_controllers.descriptors())
        )

    def _begin_new_task_for_device(self, device_id: str) -> None:
        controller = self.controller_for_device(device_id)
        controller.begin_new_task()
        controller.prepare_machine_position()

    def controller_for_device(self, device_id: str) -> RobotController:
        if str(device_id or "").strip() == self.device_controllers.default_device_id:
            return self.controller
        if isinstance(self.controller, MockRobotController):
            return self.controller
        return self.device_controllers.controller(device_id)

    def app_launcher_for_device(self, device_id: str) -> AdbPackageLauncher | None:
        if not self.app_package_registry_path.exists():
            return None
        if device_id not in self._app_launchers:
            self._app_launchers[device_id] = AdbPackageLauncher(
                self.app_package_registry_path,
                device_id,
            )
        return self._app_launchers[device_id]

    def text_transport_for_device(self, device_id: str):
        return self.adb_keyboard_runtime.transport_for_device(device_id)

    def capability_code_revision(self) -> str:
        current = self.code_revision_provider()
        if (
            self.loaded_code_revision == "unversioned"
            or current == "unversioned"
            or self.loaded_code_revision.endswith("+dirty")
            or current.endswith("+dirty")
        ):
            raise CapabilityAcceptanceError(
                "当前运行环境没有干净、可验证的 Git 提交；离线、打包或脏工作树不能进行真机能力晋级。"
            )
        if current != self.loaded_code_revision:
            raise CapabilityAcceptanceError("服务启动后代码状态发生变化，必须安全重启后才能进行真机验收。")
        return self.loaded_code_revision

    def capability_trial_orchestrator(
        self,
        provisional_controller: RobotController,
        candidate_action: str,
    ) -> UniversalAgentOrchestrator:
        return UniversalAgentOrchestrator(
            required_action_kind=candidate_action,
            qwen_observer=self.qwen_visual_decision_observer,
            adapter_factory=lambda device_id: GenericSingleActionAdapter(
                capture=lambda: self.capture_agent_frame(device_id, controller=provisional_controller),
                observer=self.generic_scene_observer,
                robot=provisional_controller,
                controller=UniversalActionController(),
                device_id=device_id,
                text_transport=self.text_transport_for_device(device_id),
            ),
            trusted_observation_factory=build_trusted_observation,
            evidence_store_factory=FileSystemAgentEvidenceStore,
            device_registry=self.device_task_registry,
        )

    def capture_agent_frame(self, device_id: str, *, controller: RobotController | None = None) -> Any:
        resolved = controller or self.controller_for_device(device_id)
        return self.device_runtime_resources.camera_coordinator(device_id).capture_agent_frame(
            resolved.vision_capture
        )

    def capture_preview(self, device_id: str, *, quality: int = 72) -> tuple[bytes, bool]:
        controller = self.controller_for_device(device_id)
        resources = self.device_runtime_resources
        cache_only = resources.coordination_lock(device_id).locked()
        return resources.camera_coordinator(device_id).capture_preview(
            controller.capture_preview,
            quality=quality,
            cache_only=cache_only,
        )

    @contextmanager
    def serial_camera_session(self, device_id: str) -> Iterator[None]:
        with self.device_runtime_resources.camera_coordinator(device_id).serial_session():
            yield

    def start(self) -> None:
        return None

    def shutdown(self) -> None:
        registry = getattr(self, "device_controllers", None)
        request_stop_all = getattr(registry, "request_stop_all", None)
        if callable(request_stop_all):
            request_stop_all()
        controller = getattr(self, "controller", None)
        if controller is not None:
            controller.request_stop()


class RuntimeUnavailable:
    """Import-safe placeholder when local configuration cannot be loaded."""

    mock_mode = False
    hardware_mode = False

    def __init__(self, error: Exception) -> None:
        self.startup_error = str(error)

    def start(self) -> None:
        return None

    def shutdown(self) -> None:
        return None

    def __getattr__(self, name: str) -> Any:
        raise RuntimeError(f"运行时尚未就绪（{name}）：{self.startup_error}")


__all__ = ["Runtime", "RuntimeUnavailable", "current_code_revision"]
