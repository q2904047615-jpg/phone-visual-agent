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
from agent.application.runtime_quota_billing import RuntimeQuotaBilling
from agent.application.qwen_visual_decision import QwenVisualDecisionObserver
from agent.application.universal_agent_orchestrator import UniversalAgentOrchestrator
from agent.domain.action_catalog import PROMOTABLE_ACTION_KINDS
from agent.domain.model_gateway import ModelGatewayRegistration, ModelGatewayRegistry
from agent.domain.tool_registry import build_canonical_tool_registry
from agent.domain.universal_action_controller import UniversalActionController
from agent.infrastructure import (
    DeviceControllerRegistry as InfrastructureDeviceControllerRegistry,
    DeviceRuntimeResourceRegistry,
    DeviceTaskRegistry,
    FileSystemAgentEvidenceStore,
    SqliteAgentSessionRepository,
    SqliteControlPlaneRepository,
    ConnectorCommandStore,
    SHARED_DEVICE_LEASE_DIR,
)
from agent.infrastructure.adb_keyboard_transport import AdbKeyboardRuntimeRegistry
from agent.infrastructure.adb_pairing import AdbPairingService
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
        # Opt-in demo/evaluation mode: simulated ADB Keyboard and app launch on the mock phone.
        self.simulated_phone_tools = self.mock_mode and os.environ.get("ROBOT_WEB_SIMULATED_TOOLS") == "1"
        self.hardware_mode = (not self.mock_mode) and seller_gui.IS_WINDOWS
        self.loaded_code_revision = code_revision_provider()
        cleanup_evidence_runs(output_dir)
        runtime_database = Path(output_dir) / "runtime.sqlite3"
        self.connector_store = ConnectorCommandStore(
            Path(os.environ.get("PHONE_AGENT_CONNECTOR_DB", Path(output_dir) / "connector.sqlite3")),
        )
        self.device_controllers = InfrastructureDeviceControllerRegistry(
            device_registry_path,
            promotable_actions=PROMOTABLE_ACTION_KINDS,
            mock=self.mock_mode,
            connector_store=self.connector_store,
        )
        self.controller: RobotController = self.device_controllers.controller(
            self.device_controllers.default_device_id
        )
        self.hardware_mode = self.hardware_mode or (not self.mock_mode and any(
            descriptor.get("connector_id") for descriptor in self.device_controllers.descriptors()))
        self._app_launchers: dict[str, AdbPackageLauncher | None] = {}
        self._simulated_tools: dict[tuple[int, str], tuple[Any, Any]] = {}
        self.vision_provider = DashScopeVisionProvider(enable_thinking=True)
        self.adb_keyboard_runtime = AdbKeyboardRuntimeRegistry(adb_keyboard_registry_path)
        self._adb_pairing_services: dict[str, AdbPairingService | None] = {}
        self.qwen_visual_decision_observer = QwenVisualDecisionObserver(
            self.vision_provider,
            trusted_observation_frame_validator=validate_trusted_observation_against_frames,
        )
        provider_status = self.vision_provider.status()
        self.model_gateway_registry = ModelGatewayRegistry()
        self.model_gateway_registry.register(ModelGatewayRegistration(
            gateway_id="qwen",
            model=str(provider_status.get("model") or "qwen3-vl-plus"),
            capabilities=frozenset({"vision", "structured_output", "planning_suggestion"}),
            version="qwen-visual-decision-v1",
            gateway=self.qwen_visual_decision_observer,
        ), default=True)
        self.model_gateway = self.model_gateway_registry.default()
        self.tool_registry = build_canonical_tool_registry()
        self.generic_scene_observer = SingleStepGenericSceneObserver(
            self.vision_provider, gateway=self.model_gateway
        )
        self.device_task_registry = DeviceTaskRegistry(lease_directory=SHARED_DEVICE_LEASE_DIR)
        self.agent_session_repository = SqliteAgentSessionRepository(runtime_database)
        self.control_plane_repository = SqliteControlPlaneRepository(runtime_database)
        self.control_plane_repository.ensure_platform_release(self.loaded_code_revision)
        self.universal_agent_orchestrator = UniversalAgentOrchestrator(
            model_gateway=self.model_gateway,
            adapter_factory=lambda device_id: GenericSingleActionAdapter(
                capture=lambda: self.capture_agent_frame(device_id),
                observer=self.generic_scene_observer,
                robot=self.controller_for_device(device_id),
                app_launcher=self.app_launcher_for_device(device_id),
                controller=UniversalActionController(tool_registry=self.tool_registry),
                device_id=device_id,
                text_transport=self.text_transport_for_device(device_id),
                tool_registry=self.tool_registry,
            ),
            trusted_observation_factory=build_trusted_observation,
            evidence_store_factory=FileSystemAgentEvidenceStore,
            device_registry=self.device_task_registry,
            session_persister=lambda session: self.agent_session_repository.save(
                session, event_type=("dispatching" if session.status == "executing_one_action"
                                     else "runtime_snapshot")
            ),
        )
        self.universal_agent_session_service = UniversalAgentSessionApplicationService(
            orchestrator_provider=lambda: self.universal_agent_orchestrator,
            sessions=self.agent_session_repository,
            ensure_device_ready=self.ensure_device_ready,
            exclusive_device_session=self.exclusive_device_session,
            begin_new_task=self._begin_new_task_for_device,
            runtime_quota_billing=RuntimeQuotaBilling(self.control_plane_repository),
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
        from agent.infrastructure.remote_connector import RemoteRobotController
        from agent.infrastructure.remote_device_tools import RemoteAppLauncher, describe_remote_tools
        controller = self.controller_for_device(device_id)
        if isinstance(controller, RemoteRobotController):
            description = describe_remote_tools(controller)
            return RemoteAppLauncher(controller, description) if description and description["apps"] else None
        if self.simulated_phone_tools and isinstance(controller, MockRobotController):
            return self._simulated_tools_for(controller, device_id)[1]
        if not self.app_package_registry_path.exists():
            return None
        if device_id not in self._app_launchers:
            self._app_launchers[device_id] = AdbPackageLauncher(
                self.app_package_registry_path,
                device_id,
            )
        return self._app_launchers[device_id]

    def text_transport_for_device(self, device_id: str):
        from agent.infrastructure.remote_connector import RemoteRobotController
        from agent.infrastructure.remote_device_tools import RemoteTextTransport, describe_remote_tools
        controller = self.controller_for_device(device_id)
        if isinstance(controller, RemoteRobotController):
            description = describe_remote_tools(controller)
            return RemoteTextTransport(controller, description) if description and description["text_profile"] else None
        if self.simulated_phone_tools and isinstance(controller, MockRobotController):
            return self._simulated_tools_for(controller, device_id)[0]
        return self.adb_keyboard_runtime.transport_for_device(device_id)

    def _simulated_tools_for(self, controller: MockRobotController, device_id: str) -> tuple[Any, Any]:
        """Mock mode only: text/launch tools bound to the mock controller's simulated phone."""
        from agent.infrastructure.simulated_device_tools import SimulatedAppLauncher, SimulatedTextTransport
        key = (id(controller), str(device_id or "").strip())
        if key not in self._simulated_tools:
            self._simulated_tools[key] = (SimulatedTextTransport(controller.phone, device_id),
                                          SimulatedAppLauncher(controller.phone))
        return self._simulated_tools[key]

    def adb_pairing_service_for_device(self, device_id: str) -> AdbPairingService | None:
        resolved = str(device_id or '').strip()
        from agent.infrastructure.remote_connector import RemoteRobotController
        if isinstance(self.controller_for_device(resolved), RemoteRobotController):
            from agent.infrastructure.adb_pairing import AdbPairingError
            raise AdbPairingError("远程手机请在 Windows 连接器主机完成 ADB 配对；云端不会运行本机 ADB。", code="pair_on_connector_host")
        if resolved not in self._adb_pairing_services:
            transport = self.text_transport_for_device(resolved)
            self._adb_pairing_services[resolved] = (
                AdbPairingService(transport.adb_executable, transport.profile.adb_serial)
                if transport is not None else None
            )
        return self._adb_pairing_services[resolved]

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
            model_gateway=self.model_gateway,
            adapter_factory=lambda device_id: GenericSingleActionAdapter(
                capture=lambda: self.capture_agent_frame(device_id, controller=provisional_controller),
                observer=self.generic_scene_observer,
                robot=provisional_controller,
                controller=UniversalActionController(tool_registry=self.tool_registry),
                device_id=device_id,
                text_transport=self.text_transport_for_device(device_id),
                tool_registry=self.tool_registry,
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
