from __future__ import annotations

import json
import os
import re
import secrets
import subprocess
import threading
import time
import uuid
import webbrowser
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Literal

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr

from capability_acceptance import (
    CapabilityAcceptanceError,
    PROMOTABLE_ACTIONS,
)
from capability_acceptance_runtime import CapabilityAcceptanceManager
from capability_acceptance_planner import CapabilityAcceptanceTaskGraphPlanner
from intent_provider import DeepSeekIntentProvider, IntentProviderError
from generic_action_adapter import (
    GenericActionAdapterError,
    GenericSingleActionAdapter,
    persist_observer_failure_diagnostic,
    stable_qwerty_ocr_anchors,
    stable_text_ocr_grounding,
)
from generic_scene_observer import SingleStepGenericSceneObserver
from input_value_lineage import TypedInputLineageStore
from deepseek_task_graph import DeepSeekTaskGraphPlanner, TaskGraphError
from qwen_visual_decision import QwenVisualDecisionObserver
from device_exclusivity import InterProcessLease, SHARED_DEVICE_LEASE_DIR
from universal_agent_orchestrator import (
    DeviceTaskRegistry,
    POST_ACTION_TRANSITION_PROTOCOL_VERSION,
    UniversalAgentOrchestrator,
    UniversalAgentOrchestratorError,
    UniversalAgentSessionState,
)
from universal_action_controller import (
    UNIVERSAL_CONTROLLER_PROTOCOL_VERSION,
    UniversalActionController,
    UniversalActionError,
)
from ui_scene import UI_SCENE_PROTOCOL_VERSION
from task_semantic_ir import (
    AUTHORITY_REPORT_PROTOCOL,
    RISK_POLICY_PROTOCOL,
    TASK_SEMANTIC_IR_PROTOCOL,
)
from canonical_action_protocol import CANONICAL_ACTION_PROTOCOL
from runtime_doctor import run_runtime_doctor

from robot_core import (
    MockRobotController,
    RobotController,
    WEB_OUTPUT_DIR,
)
from vision_agent import (
    DashScopeVisionProvider,
    VisionAgentError,
)


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
CONTROL_TOKEN = secrets.token_urlsafe(24)
DEVICE_REGISTRY_PATH = Path(
    os.environ.get(
        "ROBOT_DEVICE_REGISTRY",
        Path(__file__).with_name("device_registry.json"),
    )
)
def current_code_revision() -> str:
    """Return a reproducible revision; dirty worktrees are never promotable."""

    repository = ROOT.parent
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
                timeout=3,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise CapabilityAcceptanceError(f"无法读取当前 Git 提交：{exc}") from exc
    if not revision:
        raise CapabilityAcceptanceError("当前 Git 提交为空。")
    return revision + ("+dirty" if dirty else "")

APP_CATALOG = [
    {
        "id": "universal-agent",
        "name": "通用视觉操作 Agent",
        "icon": "智",
        "route": "#/",
        "operations": [],
        "enabled": True,
        "note": "唯一默认入口；按当前画面逐步观察、执行和验证",
    },
]
class GenericSceneRequest(BaseModel):
    goal: dict[str, Any] = Field(default_factory=dict)


class StrictAgentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GenericSupervisedStartRequest(StrictAgentRequest):
    text: StrictStr = Field(min_length=1, max_length=500)
    exact_input_text: StrictStr | None = Field(
        default=None,
        min_length=1,
        max_length=4000,
    )
    exact_action_kind: StrictStr | None = Field(default=None, max_length=32)
    exact_target_label: StrictStr = Field(default="", max_length=120)
    device_id: StrictStr = Field(min_length=1, max_length=128)
    auto_advance: StrictBool = True


class GenericSupervisedDeviceRequest(StrictAgentRequest):
    device_id: StrictStr = Field(min_length=1, max_length=128)


class BaseActionConfirmationScopeRequest(StrictAgentRequest):
    session_id: StrictStr = Field(min_length=1, max_length=128)
    task_id: StrictStr = Field(min_length=1, max_length=128)
    device_id: StrictStr = Field(min_length=1, max_length=128)
    revision: StrictInt = Field(ge=1)
    subgoal_id: StrictStr = Field(min_length=1, max_length=128)
    effect_ids: list[StrictStr] = Field(default_factory=list)
    observation_id: StrictStr = Field(min_length=1, max_length=128)
    fingerprint: StrictStr = Field(min_length=1, max_length=256)


class GenericConfirmationScopeRequest(BaseActionConfirmationScopeRequest):
    decision_node_id: StrictStr = Field(min_length=1, max_length=128)
    action_digest: StrictStr = Field(min_length=64, max_length=64)


class GenericEffectConfirmationScopeRequest(StrictAgentRequest):
    session_id: StrictStr = Field(min_length=1, max_length=128)
    task_id: StrictStr = Field(min_length=1, max_length=128)
    device_id: StrictStr = Field(min_length=1, max_length=128)
    revision: StrictInt = Field(ge=1)
    subgoal_id: StrictStr = Field(min_length=1, max_length=128)
    effect_ids: list[StrictStr] = Field(min_length=1)
    intent_digest: StrictStr = Field(min_length=64, max_length=64)


class GenericEffectApprovalRequest(StrictAgentRequest):
    confirmed: StrictBool = False
    confirmation: GenericEffectConfirmationScopeRequest | None = None


class GenericSupervisedStepRequest(StrictAgentRequest):
    confirmed: StrictBool = False
    confirmation: GenericConfirmationScopeRequest | None = None


class GenericSupervisedAutoRequest(StrictAgentRequest):
    device_id: StrictStr = Field(min_length=1, max_length=128)
    confirmed: StrictBool = False
    confirmation: GenericConfirmationScopeRequest | None = None
    max_physical_actions: StrictInt = Field(default=12, ge=1, le=20)
    max_iterations: StrictInt = Field(default=24, ge=1, le=40)


class CapabilityAcceptanceStartRequest(StrictAgentRequest):
    device_id: StrictStr = Field(min_length=1, max_length=128)
    action: StrictStr = Field(min_length=1, max_length=64)
    text: StrictStr = Field(min_length=1, max_length=500)


class CapabilityActionConfirmationScopeRequest(BaseActionConfirmationScopeRequest):
    trial_id: StrictStr = Field(min_length=1, max_length=128)
    action: StrictStr = Field(min_length=1, max_length=64)


class CapabilityEffectConfirmationScopeRequest(GenericEffectConfirmationScopeRequest):
    trial_id: StrictStr = Field(min_length=1, max_length=128)
    action: StrictStr = Field(min_length=1, max_length=64)


class CapabilityActionConfirmationRequest(StrictAgentRequest):
    confirmed: StrictBool = False
    confirmation: CapabilityActionConfirmationScopeRequest | None = None


class CapabilityEffectApprovalRequest(StrictAgentRequest):
    confirmed: StrictBool = False
    confirmation: CapabilityEffectConfirmationScopeRequest | None = None


class CapabilityPromotionRequest(StrictAgentRequest):
    confirmed: StrictBool = False
    trial_id: StrictStr = Field(min_length=1, max_length=128)
    device_id: StrictStr = Field(min_length=1, max_length=128)
    action: StrictStr = Field(min_length=1, max_length=64)
    report_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    registry_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")


class CapabilityCancelRequest(StrictAgentRequest):
    device_id: StrictStr = Field(min_length=1, max_length=128)
    action: StrictStr = Field(min_length=1, max_length=64)


class DeviceControllerRegistry:
    """Resolve one controller and calibration per device_id."""

    def __init__(self, path: Path, *, mock: bool = False) -> None:
        self.path = Path(path)
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"设备注册表无法读取：{exc}") from exc
        if payload.get("version") != 1 or not isinstance(payload.get("devices"), list):
            raise RuntimeError("设备注册表版本或 devices 格式无效。")
        self.default_device_id = str(payload.get("default_device_id") or "").strip()
        self._controllers: dict[str, RobotController] = {}
        self._descriptors: dict[str, dict[str, Any]] = {}
        enabled_windows: set[str] = set()
        for raw in payload["devices"]:
            if not isinstance(raw, dict) or raw.get("enabled") is not True:
                continue
            device_id = str(raw.get("device_id") or "").strip()
            window_title = str(raw.get("window_title") or "").strip()
            calibration_value = str(raw.get("calibration_path") or "").strip()
            raw_verified_actions = raw.get("verified_actions")
            if raw_verified_actions is None:
                verified_actions = None
            elif not isinstance(raw_verified_actions, list) or not all(
                isinstance(item, str) and item.strip()
                for item in raw_verified_actions
            ):
                raise RuntimeError(
                    f"设备 {device_id or 'missing'} 的 verified_actions 格式无效。"
                )
            else:
                verified_actions = {
                    str(item).strip() for item in raw_verified_actions
                }
            if not device_id or device_id in self._controllers:
                raise RuntimeError("设备注册表存在空或重复的 device_id。")
            effective_window = window_title or "__default_window__"
            if effective_window in enabled_windows:
                raise RuntimeError("两台已启用设备不能绑定同一个机械臂控制窗口。")
            enabled_windows.add(effective_window)
            calibration_path = Path(calibration_value or "tap_calibration.json")
            if not calibration_path.is_absolute():
                calibration_path = self.path.parent / calibration_path
            controller: RobotController
            if mock:
                controller = MockRobotController(device_id=device_id)
            elif window_title:
                controller = RobotController(
                    window_title,
                    calibration_path=calibration_path,
                    verified_actions=verified_actions,
                    device_id=device_id,
                )
            else:
                controller = RobotController(
                    calibration_path=calibration_path,
                    verified_actions=verified_actions,
                    device_id=device_id,
                )
            self._controllers[device_id] = controller
            self._descriptors[device_id] = {
                "device_id": device_id,
                "window_title": window_title,
                "calibration_path": str(calibration_path),
                "verified_actions": sorted(controller.verified_actions),
            }
        if not self._controllers or self.default_device_id not in self._controllers:
            raise RuntimeError("设备注册表必须包含已启用的 default_device_id。")

    def controller(self, device_id: str) -> RobotController:
        resolved = str(device_id or "").strip()
        controller = self._controllers.get(resolved)
        if controller is None:
            raise UniversalAgentOrchestratorError(
                f"device_id 未登记或未启用：{resolved or 'missing'}。"
            )
        return controller

    def provisional_controller(
        self,
        device_id: str,
        candidate_action: str,
    ) -> RobotController:
        """Create an unregistered controller for one evidence-bound trial.

        The returned object is deliberately not stored in this registry.  It
        cannot change the capabilities of the product controller that owns the
        normal web path.
        """

        resolved_device = str(device_id or "").strip()
        action = str(candidate_action or "").strip()
        if action not in PROMOTABLE_ACTIONS:
            raise CapabilityAcceptanceError(
                f"动作 {action or 'missing'} 不能进入真机能力验收。"
            )
        try:
            original = self._controllers[resolved_device]
            descriptor = self._descriptors[resolved_device]
        except KeyError as exc:
            raise CapabilityAcceptanceError(
                f"device_id 未登记或未启用：{resolved_device or 'missing'}。"
            ) from exc
        if action in original.verified_actions:
            raise CapabilityAcceptanceError(
                f"设备能力 {action} 已经通过真机验收。"
            )
        verified_actions = set(original.verified_actions) | {action}
        if isinstance(original, MockRobotController):
            return MockRobotController(
                verified_actions=verified_actions,
                device_id=resolved_device,
            )
        return RobotController(
            descriptor["window_title"] or original.title,
            calibration_path=Path(descriptor["calibration_path"]),
            verified_actions=verified_actions,
            device_id=resolved_device,
        )

    def descriptors(self) -> list[dict[str, Any]]:
        return [dict(self._descriptors[key]) for key in sorted(self._descriptors)]


class Runtime:
    def __init__(self) -> None:
        self.loaded_code_revision = current_code_revision()
        self.device_controllers = DeviceControllerRegistry(
            DEVICE_REGISTRY_PATH,
            mock=os.environ.get("ROBOT_WEB_MOCK") == "1",
        )
        self.controller: RobotController = self.device_controllers.controller(
            self.device_controllers.default_device_id
        )
        self.vision_provider = DashScopeVisionProvider()
        self.intent_provider = DeepSeekIntentProvider()
        self.input_lineage_store = TypedInputLineageStore(
            WEB_OUTPUT_DIR / "state"
        )
        self.generic_scene_observer = SingleStepGenericSceneObserver(
            self.vision_provider,
            input_lineage_store=self.input_lineage_store,
            qwerty_row_snapper=stable_qwerty_ocr_anchors,
        )
        self.deepseek_task_graph_planner = DeepSeekTaskGraphPlanner(
            self.intent_provider,
        )
        self.qwen_visual_decision_observer = QwenVisualDecisionObserver(
            self.vision_provider
        )
        self.device_task_registry = DeviceTaskRegistry(
            lease_directory=SHARED_DEVICE_LEASE_DIR
        )
        self.universal_agent_orchestrator = UniversalAgentOrchestrator(
            deepseek_planner=self.deepseek_task_graph_planner,
            qwen_observer=self.qwen_visual_decision_observer,
            adapter_factory=lambda device_id: GenericSingleActionAdapter(
                capture=self.controller_for_device(device_id).vision_capture,
                observer=self.generic_scene_observer,
                robot=self.controller_for_device(device_id),
                controller=UniversalActionController(),
                qwerty_row_snapper=stable_qwerty_ocr_anchors,
                text_point_grounder=(
                    stable_text_ocr_grounding
                    if not isinstance(
                        self.controller_for_device(device_id),
                        MockRobotController,
                    )
                    else None
                ),
                require_local_qwerty_row_snap=not isinstance(
                    self.controller_for_device(device_id),
                    MockRobotController,
                ),
                device_id=device_id,
                input_lineage_store=self.input_lineage_store,
            ),
            device_registry=self.device_task_registry,
        )
        self.capability_acceptance_manager = CapabilityAcceptanceManager(
            provisional_controller_factory=(
                self.device_controllers.provisional_controller
            ),
            orchestrator_factory=self.capability_trial_orchestrator,
            device_registry=self.device_task_registry,
            output_dir=WEB_OUTPUT_DIR,
            registry_path=DEVICE_REGISTRY_PATH,
            code_revision_provider=self.capability_code_revision,
        )
        self.device_coordination_lock_guard = threading.RLock()
        self.device_coordination_locks: dict[str, threading.Lock] = {
            self.device_controllers.default_device_id: threading.Lock(),
        }
        self.generic_supervised_sessions: dict[str, UniversalAgentSessionState] = {}
        self.generic_supervised_session_lock = threading.RLock()

    def controller_for_device(self, device_id: str) -> RobotController:
        if str(device_id or "").strip() == self.device_controllers.default_device_id:
            return self.controller
        if isinstance(self.controller, MockRobotController):
            # Test and explicit mock mode accepts logical device IDs while the
            # production registry remains strict.
            return self.controller
        return self.device_controllers.controller(device_id)

    def capability_code_revision(self) -> str:
        current = current_code_revision()
        if current != self.loaded_code_revision:
            raise CapabilityAcceptanceError(
                "服务启动后代码状态发生变化，必须安全重启后才能进行真机验收。"
            )
        return self.loaded_code_revision

    def capability_trial_orchestrator(
        self,
        provisional_controller: RobotController,
        candidate_action: str,
    ) -> UniversalAgentOrchestrator:
        """Build one isolated primitive-certification orchestrator."""

        return UniversalAgentOrchestrator(
            deepseek_planner=CapabilityAcceptanceTaskGraphPlanner(
                candidate_action
            ),
            qwen_observer=self.qwen_visual_decision_observer,
            adapter_factory=lambda device_id: GenericSingleActionAdapter(
                capture=provisional_controller.vision_capture,
                observer=self.generic_scene_observer,
                robot=provisional_controller,
                controller=UniversalActionController(),
                qwerty_row_snapper=stable_qwerty_ocr_anchors,
                text_point_grounder=(
                    stable_text_ocr_grounding
                    if not isinstance(provisional_controller, MockRobotController)
                    else None
                ),
                require_local_qwerty_row_snap=not isinstance(
                    provisional_controller,
                    MockRobotController,
                ),
                device_id=device_id,
                input_lineage_store=self.input_lineage_store,
            ),
            device_registry=self.device_task_registry,
        )

    def coordination_lock_for_device(self, device_id: str) -> threading.Lock:
        """Serialize observation/action work per device, not across devices."""

        resolved = str(device_id or "").strip()
        if not resolved:
            raise UniversalAgentOrchestratorError("device_id 不能为空。")
        with self.device_coordination_lock_guard:
            return self.device_coordination_locks.setdefault(
                resolved,
                threading.Lock(),
            )

    def start(self) -> None:
        return None

    def shutdown(self) -> None:
        self.controller.request_stop()



runtime = Runtime()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> Iterator[None]:
    runtime.start()
    print(f"机械臂网页控制台：http://127.0.0.1:8765/")
    print("控制令牌已生成，仅通过本机受保护的页面初始化接口使用。")
    if os.environ.get("ROBOT_WEB_NO_BROWSER") != "1":
        threading.Timer(
            1.0, lambda: webbrowser.open("http://127.0.0.1:8765/")
        ).start()
    yield
    runtime.shutdown()


app = FastAPI(
    title="多 App 机械臂网页控制平台",
    version="0.2.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
)
app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")


def verify_local_request(request: Request, token: str | None) -> None:
    if token != CONTROL_TOKEN:
        raise HTTPException(status_code=403, detail="控制令牌无效。")
    origin = request.headers.get("origin")
    if origin:
        try:
            origin_host = origin.split("://", 1)[1].split("/", 1)[0].split(":", 1)[0]
        except IndexError:
            raise HTTPException(status_code=403, detail="来源地址无效。")
        if origin_host not in {"127.0.0.1", "localhost"}:
            raise HTTPException(status_code=403, detail="只允许本机网页请求。")



@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/session")
def session() -> dict[str, Any]:
    return {
        "token": CONTROL_TOKEN,
        "mock": isinstance(runtime.controller, MockRobotController),
        "version": app.version,
    }


@app.get("/api/apps")
def apps() -> dict[str, Any]:
    readiness = {"vision_agent": {
        "ready": bool(runtime.vision_provider.status().get("configured")),
        "mode": runtime.vision_provider.status().get("model", "unknown"),
        "thinking_enabled": runtime.vision_provider.status().get(
            "thinking_enabled", False
        ),
        "missing_templates": [],
        "missing_capabilities": (
            []
            if runtime.vision_provider.status().get("configured")
            else ["DASHSCOPE_API_KEY"]
        ),
    }}
    readiness["intent_agent"] = {
        "ready": bool(runtime.intent_provider.configured),
        "mode": runtime.intent_provider.model,
        "missing_templates": [],
        "missing_capabilities": (
            [] if runtime.intent_provider.configured else ["DEEPSEEK_API_KEY"]
        ),
    }
    return {"apps": APP_CATALOG, "readiness": readiness}


@app.get("/api/device")
def device() -> dict[str, Any]:
    status = dict(runtime.controller.device_status())
    status.pop("readiness", None)
    capability_provider = getattr(runtime.controller, "hardware_capabilities", None)
    hardware_capabilities = (
        capability_provider() if callable(capability_provider) else {}
    )
    capability_profile_provider = getattr(
        runtime.controller,
        "hardware_capability_profile",
        None,
    )
    hardware_capability_profile = (
        capability_profile_provider()
        if callable(capability_profile_provider)
        else None
    )
    profile_actions = (
        hardware_capability_profile.get("actions", {})
        if isinstance(hardware_capability_profile, dict)
        else {}
    )
    effective_hardware_capabilities = {
        str(action): bool(spec.get("enabled"))
        for action, spec in profile_actions.items()
        if isinstance(action, str) and isinstance(spec, dict)
    }
    status["default_device_id"] = runtime.device_controllers.default_device_id
    devices = []
    for descriptor in runtime.device_controllers.descriptors():
        public_status = dict(
            runtime.controller_for_device(descriptor["device_id"]).device_status()
        )
        public_status.pop("readiness", None)
        devices.append({**descriptor, **public_status})
    status["devices"] = devices
    status["vision_agent"] = runtime.vision_provider.status()
    status["intent_agent"] = runtime.intent_provider.status()
    status["execution_architecture"] = {
        "model_role": "observation_only",
        "controller": "universal_action_controller",
        "fixed_app_workflows_retired": True,
        "active_orchestrator": "universal_agent",
        "universal_agent": {
            "goal_protocol": "2026-08-20-deepseek-typed-task-graph-v4",
            "scene_protocol": UI_SCENE_PROTOCOL_VERSION,
            "action_protocol": CANONICAL_ACTION_PROTOCOL,
            "controller_protocol": UNIVERSAL_CONTROLLER_PROTOCOL_VERSION,
            "goal_preview_enabled": False,
            "scene_preview_enabled": True,
            "hardware_execution_enabled": True,
            "automatic_loop_enabled": True,
            "automatic_loop_max_physical_actions": 12,
            "automatic_loop_max_iterations": 24,
            "supervised_single_step_enabled": False,
            "enabled_physical_actions": sorted(
                action
                for action, enabled in (
                    effective_hardware_capabilities or hardware_capabilities
                ).items()
                if enabled and action != "wait_for_change"
            ),
            "protocol_physical_actions": [
                "tap_semantic",
                "dismiss_overlay",
                "swipe",
                "back",
                "home",
                "reveal_system_navigation",
                "input_verified_text",
                "press_enter",
                "clear_verified_text",
                "double_tap",
                "long_press",
                "drag",
            ],
            "hardware_capabilities": hardware_capabilities,
            "hardware_capability_profile": hardware_capability_profile,
            "supported_app_scope": "dynamic",
            "typed_effect_authority": {
                "semantic_ir_protocol": TASK_SEMANTIC_IR_PROTOCOL,
                "authority_protocol": AUTHORITY_REPORT_PROTOCOL,
                "effect_policy_protocol": RISK_POLICY_PROTOCOL,
                "authority_scope": "typed_task_and_effect_policy",
                "retired_remote_risk_diagnostics_enabled": False,
                "canonical_action_protocol": CANONICAL_ACTION_PROTOCOL,
            },
            "observer": runtime.generic_scene_observer.status(),
        },
    }
    status["active_tasks"] = []
    with runtime.generic_supervised_session_lock:
        generic_sessions = [
            item.snapshot()
            for item in runtime.generic_supervised_sessions.values()
            if item.status in {
                "awaiting_effect_confirmation",
                "awaiting_confirmation",
                "paused_after_action",
                "needs_reobservation",
                "needs_effect_verification",
            }
        ]
    status["generic_supervised_execution"] = {
        "enabled": True,
        "automatic_loop_enabled": True,
        "max_physical_actions_per_confirmation": 1,
        "max_safe_loop_physical_actions": 12,
        "max_safe_loop_iterations": 24,
        "external_effect_confirmation_count": 1,
        "post_action_transition_protocol": (
            POST_ACTION_TRANSITION_PROTOCOL_VERSION
        ),
        "active_sessions": [
            {
                "session_id": item["session_id"],
                "device_id": item["device_id"],
                "status": item["status"],
                "step_number": item["step_number"],
                "proposal": item["proposal"],
            }
            for item in generic_sessions
        ],
    }
    return status


@app.get("/api/doctor/{device_id}")
def runtime_doctor(device_id: str) -> dict[str, Any]:
    """Run the only current zero-action runtime readiness inspection."""

    try:
        controller = runtime.controller_for_device(device_id)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    active_session = runtime.device_task_registry.active_session(device_id)
    coordination_lock = runtime.coordination_lock_for_device(device_id)
    acquired = coordination_lock.acquire(blocking=False)
    if not acquired and not active_session:
        active_session = "device-coordination-busy"
    try:
        return run_runtime_doctor(
            device_id=device_id,
            controller=controller,
            deepseek_provider=runtime.intent_provider,
            qwen_provider=runtime.vision_provider,
            active_session=active_session,
            protocols={
                "goal": "2026-08-20-deepseek-typed-task-graph-v4",
                "scene": UI_SCENE_PROTOCOL_VERSION,
                "action": CANONICAL_ACTION_PROTOCOL,
                "controller": UNIVERSAL_CONTROLLER_PROTOCOL_VERSION,
                "semantic_ir": TASK_SEMANTIC_IR_PROTOCOL,
                "risk": RISK_POLICY_PROTOCOL,
            },
        )
    finally:
        if acquired:
            coordination_lock.release()




def _require_supervised_device_ready(device_id: str | None = None) -> None:
    controller = runtime.controller_for_device(
        device_id or runtime.device_controllers.default_device_id
    )
    status = controller.device_status()
    if not status.get("controller_online") or not status.get("camera_online"):
        raise HTTPException(status_code=409, detail="控制端或摄像头离线。")
    if status.get("busy"):
        raise HTTPException(status_code=409, detail="机械臂正在执行其他任务。")
    if not runtime.vision_provider.status().get("configured"):
        raise HTTPException(status_code=409, detail="千问视觉尚未配置。")


@contextmanager
def _supervised_hardware_lock(device_id: str | None = None) -> Iterator[None]:
    resolved_device = str(
        device_id or runtime.device_controllers.default_device_id
    ).strip()
    controller = runtime.controller_for_device(resolved_device)
    lease_path = (
        SHARED_DEVICE_LEASE_DIR / "physical_hardware_action.lease"
        if resolved_device == runtime.device_controllers.default_device_id
        else SHARED_DEVICE_LEASE_DIR
        / f"physical_hardware_action_{re.sub(r'[^A-Za-z0-9_.-]+', '_', resolved_device)}.lease"
    )
    process_lease = InterProcessLease(
        lease_path,
        owner_id=f"web-{os.getpid()}-{threading.get_ident()}",
        metadata={"purpose": "physical_hardware_action", "device_id": resolved_device},
    )
    if not process_lease.acquire():
        raise HTTPException(status_code=409, detail="另一进程已占用机械臂物理控制权。")
    coordination_lock = runtime.coordination_lock_for_device(resolved_device)
    if not coordination_lock.acquire(blocking=False):
        process_lease.release()
        raise HTTPException(status_code=409, detail="已有语义观察或动作正在进行。")
    if not controller.operation_lock.acquire(blocking=False):
        coordination_lock.release()
        process_lease.release()
        raise HTTPException(status_code=409, detail="机械臂物理控制权已被占用。")
    try:
        yield
    finally:
        controller.operation_lock.release()
        coordination_lock.release()
        process_lease.release()




def _require_generic_session_device(
    session: UniversalAgentSessionState,
    requested_device_id: str,
) -> None:
    if str(requested_device_id or "") != session.device_id:
        raise HTTPException(
            status_code=409,
            detail={
                "success": False,
                "physical_actions": 0,
                "error": "请求device_id与会话锁定设备不一致。",
            },
        )


@app.post("/api/agent/generic-scene")
def observe_generic_scene(
    body: GenericSceneRequest,
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    """Capture a stable scene graph without moving or clicking the robot."""

    verify_local_request(request, x_control_token)
    _require_supervised_device_ready()
    frames = []
    with _supervised_hardware_lock():
        for index in range(4):
            frame = runtime.controller.vision_capture()
            if frame.width < 400 or frame.height < 700:
                raise HTTPException(
                    status_code=409,
                    detail="摄像头返回残缺画面，未调用通用观察器。",
                )
            frames.append(frame)
            if index < 3:
                time.sleep(0.5)
        try:
            scene = runtime.generic_scene_observer.observe(
                frames=frames,
                goal_context=body.goal,
                device_id=runtime.device_controllers.default_device_id,
            )
        except VisionAgentError as exc:
            failure_dir = WEB_OUTPUT_DIR / (
                "generic_scene_failure_"
                + datetime.now().strftime("%Y%m%d_%H%M%S_")
                + uuid.uuid4().hex[:8]
            )
            try:
                persist_observer_failure_diagnostic(
                    runtime.generic_scene_observer,
                    evidence_dir=failure_dir,
                    prefix="generic_scene_preview",
                    error=exc,
                )
            except Exception:
                pass
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "mode": "generic_scene_preview",
        "executed": False,
        "physical_action_requested": False,
        "scene": scene.to_dict(),
        "diagnostics": dict(runtime.generic_scene_observer.last_diagnostics),
    }


def _new_generic_action_adapter() -> GenericSingleActionAdapter:
    return GenericSingleActionAdapter(
        capture=runtime.controller.vision_capture,
        observer=runtime.generic_scene_observer,
        robot=runtime.controller,
        controller=UniversalActionController(),
        qwerty_row_snapper=stable_qwerty_ocr_anchors,
        text_point_grounder=(
            stable_text_ocr_grounding
            if not isinstance(runtime.controller, MockRobotController)
            else None
        ),
        require_local_qwerty_row_snap=not isinstance(
            runtime.controller,
            MockRobotController,
        ),
        device_id=runtime.device_controllers.default_device_id,
        input_lineage_store=runtime.input_lineage_store,
    )


def _write_generic_supervised_report(session: UniversalAgentSessionState) -> str:
    """Return the atomic report already maintained by the orchestrator."""

    return str(session.run_dir / "report.json")


def _generic_supervised_failure(
    session: UniversalAgentSessionState | None,
    exc: Exception,
    *,
    request_action_count: int = 0,
) -> dict[str, Any]:
    physical_actions = max(0, int(request_action_count))
    if isinstance(exc, GenericActionAdapterError):
        physical_actions = max(physical_actions, int(exc.physical_actions))
    return {
        "success": False,
        "physical_actions": physical_actions,
        "error": str(exc),
        "evidence": (
            list(session.snapshot().get("evidence", [])) if session else []
        ),
        "session": session.snapshot() if session else None,
        "report": _write_generic_supervised_report(session) if session else None,
    }


def _capability_trial_payload(trial: Any) -> dict[str, Any]:
    snapshot = trial.snapshot()
    session = snapshot.get("session")
    if not isinstance(session, dict):
        session = {}
    action_scope = session.get("confirmation_scope")
    effect_scope = session.get("effect_confirmation_scope")
    snapshot["action_confirmation_scope"] = (
        {
            **dict(action_scope),
            "trial_id": trial.trial_id,
            "action": trial.candidate_action,
        }
        if isinstance(action_scope, dict)
        else None
    )
    snapshot["effect_confirmation_scope"] = (
        {
            **dict(effect_scope),
            "trial_id": trial.trial_id,
            "action": trial.candidate_action,
        }
        if isinstance(effect_scope, dict)
        else None
    )
    return snapshot


def _capability_execution_payload(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return dict(result)
    method = getattr(result, "to_dict", None)
    if callable(method):
        payload = method()
        if isinstance(payload, dict):
            return payload
    raise CapabilityAcceptanceError("验收执行结果不是 JSON 对象。")


def _capability_failure(
    trial: Any | None,
    exc: Exception,
    *,
    request_action_count: int = 0,
) -> dict[str, Any]:
    physical_actions = max(
        0,
        int(request_action_count),
        int(getattr(exc, "physical_actions", 0) or 0),
    )
    return {
        "success": False,
        "physical_actions": physical_actions,
        "error": str(exc),
        "trial": _capability_trial_payload(trial) if trial is not None else None,
        "evidence": [str(path) for path in getattr(exc, "evidence", ())],
    }


def _require_capability_trial_binding(
    trial: Any,
    *,
    trial_id: str,
    device_id: str,
    action: str,
) -> None:
    if (
        trial.trial_id != trial_id
        or trial.device_id != device_id
        or trial.candidate_action != action
    ):
        raise CapabilityAcceptanceError("真机验收确认范围与当前会话不匹配。")


CAPABILITY_ACCEPTANCE_ERRORS = (
    CapabilityAcceptanceError,
    GenericActionAdapterError,
    IntentProviderError,
    TaskGraphError,
    UniversalActionError,
    UniversalAgentOrchestratorError,
    VisionAgentError,
)


@app.post("/api/capability-acceptance/start")
def start_capability_acceptance(
    body: CapabilityAcceptanceStartRequest,
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    """Plan and observe one unverified generic action with zero execution."""

    verify_local_request(request, x_control_token)
    if body.action not in PROMOTABLE_ACTIONS:
        exc = CapabilityAcceptanceError(
            f"动作 {body.action} 不能进入真机能力验收。"
        )
        raise HTTPException(
            status_code=409,
            detail=_capability_failure(None, exc),
        )
    active_session = runtime.device_task_registry.active_session(body.device_id)
    if active_session is not None:
        exc = CapabilityAcceptanceError(
            f"设备 {body.device_id} 已有活动任务：{active_session}。"
        )
        raise HTTPException(
            status_code=409,
            detail=_capability_failure(None, exc),
        )
    _require_supervised_device_ready(body.device_id)
    try:
        with _supervised_hardware_lock(body.device_id):
            trial = runtime.capability_acceptance_manager.start(
                device_id=body.device_id,
                candidate_action=body.action,
                text=body.text,
            )
        return {
            "mode": "capability_acceptance_single_action",
            "physical_actions": 0,
            "automatic_loop_enabled": False,
            "trial": _capability_trial_payload(trial),
        }
    except CAPABILITY_ACCEPTANCE_ERRORS as exc:
        raise HTTPException(
            status_code=409,
            detail=_capability_failure(None, exc),
        ) from exc


@app.get("/api/capability-acceptance")
def list_capability_acceptance(
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    verify_local_request(request, x_control_token)
    return {"trials": runtime.capability_acceptance_manager.snapshots()}


@app.get("/api/capability-acceptance/{trial_id}")
def get_capability_acceptance(
    trial_id: str,
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    verify_local_request(request, x_control_token)
    try:
        trial = runtime.capability_acceptance_manager.get(trial_id)
    except CapabilityAcceptanceError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "mode": "capability_acceptance_single_action",
        "trial": _capability_trial_payload(trial),
    }


@app.get("/api/capability-acceptance/{trial_id}/evidence/{phase}/{index}")
def get_capability_acceptance_evidence(
    trial_id: str,
    phase: Literal["before", "after"],
    index: int,
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> FileResponse:
    verify_local_request(request, x_control_token)
    try:
        trial = runtime.capability_acceptance_manager.get(trial_id)
        report = json.loads(trial.report_path.read_text(encoding="utf-8"))
        paths = report.get(f"{phase}_frame_paths")
        if (
            not isinstance(paths, list)
            or isinstance(index, bool)
            or index < 0
            or index >= len(paths)
            or not isinstance(paths[index], str)
        ):
            raise CapabilityAcceptanceError("验收证据索引不存在。")
        evidence_path = Path(paths[index]).resolve(strict=True)
        trial_root = trial.run_dir.resolve(strict=True)
        if (
            evidence_path.suffix.lower() not in {".jpg", ".jpeg"}
            or evidence_path == trial_root
            or trial_root not in evidence_path.parents
        ):
            raise CapabilityAcceptanceError("验收证据路径越出当前 trial。")
    except (
        CapabilityAcceptanceError,
        OSError,
        UnicodeError,
        ValueError,
        TypeError,
        json.JSONDecodeError,
    ) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(evidence_path, media_type="image/jpeg")


@app.post("/api/capability-acceptance/{trial_id}/approve-effect")
def approve_capability_acceptance_effect(
    trial_id: str,
    body: CapabilityEffectApprovalRequest,
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    verify_local_request(request, x_control_token)
    trial = None
    try:
        trial = runtime.capability_acceptance_manager.get(trial_id)
        if body.confirmed is not True or body.confirmation is None:
            raise CapabilityAcceptanceError("验收效果确认必须提交完整精确作用域。")
        try:
            _require_capability_trial_binding(
                trial,
                trial_id=body.confirmation.trial_id,
                device_id=body.confirmation.device_id,
                action=body.confirmation.action,
            )
        except CapabilityAcceptanceError:
            runtime.capability_acceptance_manager.cancel(trial_id)
            raise
        _require_supervised_device_ready(trial.device_id)
        before_actions = int(getattr(trial.session, "physical_actions", 0))
        with _supervised_hardware_lock(trial.device_id):
            runtime.capability_acceptance_manager.approve_effects(
                trial_id,
                body.confirmation.model_dump(exclude={"trial_id", "action"}),
            )
        request_actions = int(getattr(trial.session, "physical_actions", 0)) - before_actions
        if request_actions != 0:
            raise CapabilityAcceptanceError("验收效果确认错误地产生了物理动作。")
        return {
            "mode": "capability_acceptance_single_action",
            "physical_actions": 0,
            "trial": _capability_trial_payload(trial),
        }
    except CAPABILITY_ACCEPTANCE_ERRORS as exc:
        raise HTTPException(
            status_code=409,
            detail=_capability_failure(trial, exc),
        ) from exc


@app.post("/api/capability-acceptance/{trial_id}/confirm")
def confirm_capability_acceptance(
    trial_id: str,
    body: CapabilityActionConfirmationRequest,
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    """Consume one trial-bound action confirmation and execute exactly once."""

    verify_local_request(request, x_control_token)
    trial = None
    before_actions = 0
    try:
        trial = runtime.capability_acceptance_manager.get(trial_id)
        if body.confirmed is not True or body.confirmation is None:
            raise CapabilityAcceptanceError("执行验收动作前必须提交完整精确作用域。")
        try:
            _require_capability_trial_binding(
                trial,
                trial_id=body.confirmation.trial_id,
                device_id=body.confirmation.device_id,
                action=body.confirmation.action,
            )
        except CapabilityAcceptanceError:
            runtime.capability_acceptance_manager.cancel(trial_id)
            raise
        _require_supervised_device_ready(trial.device_id)
        before_actions = int(getattr(trial.session, "physical_actions", 0))
        with _supervised_hardware_lock(trial.device_id):
            result = runtime.capability_acceptance_manager.confirm(
                trial_id,
                body.confirmation.model_dump(exclude={"trial_id", "action"}),
            )
        request_actions = int(getattr(trial.session, "physical_actions", 0)) - before_actions
        if request_actions != 1:
            raise CapabilityAcceptanceError(
                f"验收动作确认必须恰好执行一次，实际为 {request_actions}。"
            )
        return {
            "mode": "capability_acceptance_single_action",
            "physical_actions": 1,
            "execution": _capability_execution_payload(result),
            "trial": _capability_trial_payload(trial),
        }
    except CAPABILITY_ACCEPTANCE_ERRORS as exc:
        request_actions = (
            max(
                0,
                int(getattr(trial.session, "physical_actions", 0)) - before_actions,
            )
            if trial is not None
            else 0
        )
        raise HTTPException(
            status_code=409,
            detail=_capability_failure(
                trial,
                exc,
                request_action_count=request_actions,
            ),
        ) from exc


@app.get("/api/capability-acceptance/{trial_id}/promotion-preview")
def preview_capability_promotion(
    trial_id: str,
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    verify_local_request(request, x_control_token)
    try:
        scope = runtime.capability_acceptance_manager.promotion_scope(trial_id)
        return {
            "physical_actions": 0,
            "promotion_scope": scope.to_dict(),
            "requires_separate_confirmation": True,
        }
    except CapabilityAcceptanceError as exc:
        raise HTTPException(
            status_code=409,
            detail=_capability_failure(None, exc),
        ) from exc


@app.post("/api/capability-acceptance/{trial_id}/promote")
def promote_capability_acceptance(
    trial_id: str,
    body: CapabilityPromotionRequest,
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    """Atomically update disk configuration; never touch camera or hardware."""

    verify_local_request(request, x_control_token)
    trial = None
    try:
        trial = runtime.capability_acceptance_manager.get(trial_id)
        if body.confirmed is not True:
            raise CapabilityAcceptanceError("能力晋级需要单独明确确认。")
        confirmation = body.model_dump(exclude={"confirmed"})
        result = runtime.capability_acceptance_manager.promote(
            trial_id,
            confirmation,
        )
        return {
            "physical_actions": 0,
            "promotion": result,
            "trial": _capability_trial_payload(trial),
        }
    except CapabilityAcceptanceError as exc:
        raise HTTPException(
            status_code=409,
            detail=_capability_failure(trial, exc),
        ) from exc


@app.post("/api/capability-acceptance/{trial_id}/cancel")
def cancel_capability_acceptance(
    trial_id: str,
    body: CapabilityCancelRequest,
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    verify_local_request(request, x_control_token)
    trial = None
    try:
        trial = runtime.capability_acceptance_manager.get(trial_id)
        _require_capability_trial_binding(
            trial,
            trial_id=trial_id,
            device_id=body.device_id,
            action=body.action,
        )
        runtime.capability_acceptance_manager.cancel(trial_id)
        return {
            "physical_actions": 0,
            "trial": _capability_trial_payload(trial),
        }
    except CAPABILITY_ACCEPTANCE_ERRORS as exc:
        raise HTTPException(
            status_code=409,
            detail=_capability_failure(trial, exc),
        ) from exc


@app.post("/api/agent/generic-supervised/start")
def start_generic_supervised_session(
    body: GenericSupervisedStartRequest,
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    """Plan once, then autonomously advance only safe read/navigation actions."""

    verify_local_request(request, x_control_token)
    active_session_id = (
        runtime.universal_agent_orchestrator.device_registry.active_session(
            body.device_id
        )
    )
    if active_session_id is not None:
        exc = UniversalAgentOrchestratorError(
            f"设备 {body.device_id} 已有活动任务：{active_session_id}。"
        )
        raise HTTPException(
            status_code=409,
            detail=_generic_supervised_failure(None, exc),
        )
    _require_supervised_device_ready(body.device_id)
    session_id = uuid.uuid4().hex
    run_dir = WEB_OUTPUT_DIR / (
        "generic_supervised_"
        + datetime.now().strftime("%Y%m%d_%H%M%S_")
        + session_id[:8]
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    session = None
    try:
        with _supervised_hardware_lock(body.device_id):
            active_session_id = (
                runtime.universal_agent_orchestrator.device_registry.active_session(
                    body.device_id
                )
            )
            if active_session_id is not None:
                raise UniversalAgentOrchestratorError(
                    f"设备 {body.device_id} 已有活动任务：{active_session_id}。"
                )
            runtime.controller_for_device(body.device_id).begin_new_task()
            session = runtime.universal_agent_orchestrator.start(
                session_id=session_id,
                raw_goal=body.text,
                exact_input_text=body.exact_input_text,
                exact_action_kind=body.exact_action_kind,
                exact_target_label=body.exact_target_label,
                device_id=body.device_id,
                run_dir=run_dir,
            )
        with runtime.generic_supervised_session_lock:
            runtime.generic_supervised_sessions[session_id] = session
        auto_result = {
            "physical_actions": 0,
            "iterations": 0,
            "status": session.status,
            "pause_reason": "当前没有可自动推进的安全动作。",
        }
        if (
            body.auto_advance is True
            and session.status in {"awaiting_confirmation", "needs_reobservation"}
        ):
            with _supervised_hardware_lock(body.device_id):
                auto_result = (
                    runtime.universal_agent_orchestrator.run_autonomous_safe_loop(
                        session,
                        max_physical_actions=12,
                        max_iterations=24,
                    )
                )
        report = _write_generic_supervised_report(session)
        return {
            "mode": (
                "generic_supervised_autonomous_safe_loop"
                if body.auto_advance is True
                else "generic_supervised_single_step"
            ),
            "physical_actions": auto_result["physical_actions"],
            "automatic_loop_supported": True,
            "automatic_loop_enabled": body.auto_advance,
            "automatic_progress": auto_result,
            "session": session.snapshot(),
            "report": report,
        }
    except (
        IntentProviderError,
        GenericActionAdapterError,
        UniversalActionError,
        UniversalAgentOrchestratorError,
        TaskGraphError,
        VisionAgentError,
    ) as exc:
        failure = _generic_supervised_failure(session, exc)
        failure["evidence"] = [
            str(path) for path in sorted(run_dir.glob("*.jpg"))
        ]
        raise HTTPException(status_code=409, detail=failure) from exc


@app.get("/api/agent/generic-supervised/{session_id}")
def get_generic_supervised_session(session_id: str) -> dict[str, Any]:
    with runtime.generic_supervised_session_lock:
        session = runtime.generic_supervised_sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="通用单步会话不存在。")
    return {
        "mode": "generic_supervised_single_step",
        "session": session.snapshot(),
        "report": str(session.run_dir / "report.json"),
    }


@app.post("/api/agent/generic-supervised/{session_id}/approve-effect")
def approve_generic_supervised_effect(
    session_id: str,
    body: GenericEffectApprovalRequest,
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    """Approve one canonical local-effect policy scope and execute at most one action."""

    verify_local_request(request, x_control_token)
    with runtime.generic_supervised_session_lock:
        session = runtime.generic_supervised_sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="通用单步会话不存在。")
    _require_supervised_device_ready(session.device_id)
    before_actions = session.physical_actions
    try:
        if body.confirmed is not True or body.confirmation is None:
            raise UniversalAgentOrchestratorError(
                "调用 Qwen 处理受限效果前必须确认完整效果作用域。"
            )
        _require_generic_session_device(session, body.confirmation.device_id)
        with _supervised_hardware_lock(session.device_id):
            result = runtime.universal_agent_orchestrator.approve_effects(
                session,
                body.confirmation.model_dump(),
            )
        request_actions = session.physical_actions - before_actions
        if request_actions not in {0, 1}:
            raise UniversalAgentOrchestratorError(
                "一次效果确认产生了超过一个物理动作。"
            )
        report = _write_generic_supervised_report(session)
        response = {
            "mode": "generic_supervised_single_step",
            "physical_actions": request_actions,
            "automatic_loop_enabled": False,
            "session": session.snapshot(),
            "report": report,
        }
        if hasattr(result, "action_outcome"):
            response["execution"] = result.to_dict()
        else:
            response["proposal"] = result.proposal.to_dict()
        return response
    except (
        GenericActionAdapterError,
        UniversalActionError,
        UniversalAgentOrchestratorError,
        IntentProviderError,
        TaskGraphError,
        VisionAgentError,
    ) as exc:
        request_actions = max(0, session.physical_actions - before_actions)
        raise HTTPException(
            status_code=409,
            detail=_generic_supervised_failure(
                session,
                exc,
                request_action_count=request_actions,
            ),
        ) from exc


@app.post("/api/agent/generic-supervised/{session_id}/confirm")
def confirm_generic_supervised_session(
    session_id: str,
    body: GenericSupervisedStepRequest,
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    """Consume one exact authority scope and execute at most one action."""

    verify_local_request(request, x_control_token)
    with runtime.generic_supervised_session_lock:
        session = runtime.generic_supervised_sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="通用单步会话不存在。")
    try:
        _require_supervised_device_ready(session.device_id)
    except HTTPException:
        try:
            runtime.universal_agent_orchestrator.invalidate_confirmation(
                session,
                reason="device_readiness_failed",
            )
        except Exception:
            pass
        raise
    before_actions = session.physical_actions
    try:
        if body.confirmed is not True or body.confirmation is None:
            raise UniversalAgentOrchestratorError(
                "执行一个动作前必须提交完整且明确的确认作用域。"
            )
        with _supervised_hardware_lock(session.device_id):
            result = runtime.universal_agent_orchestrator.confirm_one(
                session,
                body.confirmation.model_dump(),
            )
        report = _write_generic_supervised_report(session)
        return {
            "mode": "generic_supervised_single_step",
            "automatic_loop_enabled": False,
            "execution": result.to_dict(),
            "session": session.snapshot(),
            "report": report,
        }
    except (
        GenericActionAdapterError,
        UniversalActionError,
        UniversalAgentOrchestratorError,
        IntentProviderError,
        TaskGraphError,
        VisionAgentError,
    ) as exc:
        request_actions = max(0, session.physical_actions - before_actions)
        raise HTTPException(
            status_code=409,
            detail=_generic_supervised_failure(
                session,
                exc,
                request_action_count=request_actions,
            ),
        ) from exc


@app.post("/api/agent/generic-supervised/{session_id}/next")
def plan_next_generic_supervised_step(
    session_id: str,
    body: GenericSupervisedDeviceRequest,
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    """Reobserve and propose the next action without touching the robot."""

    verify_local_request(request, x_control_token)
    with runtime.generic_supervised_session_lock:
        session = runtime.generic_supervised_sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="通用单步会话不存在。")
    _require_generic_session_device(session, body.device_id)
    _require_supervised_device_ready(session.device_id)
    before_actions = session.physical_actions
    try:
        with _supervised_hardware_lock(session.device_id):
            decision = runtime.universal_agent_orchestrator.refresh_decision(session)
        report = _write_generic_supervised_report(session)
        return {
            "mode": "generic_supervised_single_step",
            "physical_actions": 0,
            "automatic_loop_enabled": False,
            "proposal": decision.proposal.to_dict(),
            "session": session.snapshot(),
            "report": report,
        }
    except (
        GenericActionAdapterError,
        UniversalActionError,
        IntentProviderError,
        UniversalAgentOrchestratorError,
        TaskGraphError,
        VisionAgentError,
    ) as exc:
        raise HTTPException(
            status_code=409,
            detail=_generic_supervised_failure(
                session,
                exc,
                request_action_count=max(
                    0, session.physical_actions - before_actions
                ),
            ),
        ) from exc


@app.post("/api/agent/generic-supervised/{session_id}/auto")
def run_generic_supervised_safe_loop(
    session_id: str,
    body: GenericSupervisedAutoRequest,
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    """Continue only safe read/navigation work without user action confirmation."""

    verify_local_request(request, x_control_token)
    with runtime.generic_supervised_session_lock:
        session = runtime.generic_supervised_sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="通用单步会话不存在。")
    _require_generic_session_device(session, body.device_id)
    try:
        _require_supervised_device_ready(session.device_id)
    except HTTPException:
        try:
            runtime.universal_agent_orchestrator.invalidate_confirmation(
                session,
                reason="device_readiness_failed",
            )
        except Exception:
            pass
        raise
    before_actions = session.physical_actions
    try:
        if body.confirmed is True or body.confirmation is not None:
            raise UniversalAgentOrchestratorError(
                "安全自动推进不接收用户动作确认；外部影响请使用风险确认接口。"
            )
        with _supervised_hardware_lock(session.device_id):
            result = runtime.universal_agent_orchestrator.run_autonomous_safe_loop(
                session,
                max_physical_actions=body.max_physical_actions,
                max_iterations=body.max_iterations,
            )
        report = _write_generic_supervised_report(session)
        return {
            "mode": "generic_supervised_safe_loop",
            "automatic_loop_enabled": True,
            "execution": result,
            "session": session.snapshot(),
            "report": report,
        }
    except (
        GenericActionAdapterError,
        UniversalActionError,
        UniversalAgentOrchestratorError,
        IntentProviderError,
        TaskGraphError,
        VisionAgentError,
    ) as exc:
        raise HTTPException(
            status_code=409,
            detail=_generic_supervised_failure(
                session,
                exc,
                request_action_count=max(
                    0, session.physical_actions - before_actions
                ),
            ),
        ) from exc


@app.post("/api/agent/generic-supervised/{session_id}/cancel")
def cancel_generic_supervised_session(
    session_id: str,
    body: GenericSupervisedDeviceRequest,
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    verify_local_request(request, x_control_token)
    with runtime.generic_supervised_session_lock:
        session = runtime.generic_supervised_sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="通用单步会话不存在。")
        _require_generic_session_device(session, body.device_id)
        runtime.universal_agent_orchestrator.cancel(session)
    report = _write_generic_supervised_report(session)
    return {"session": session.snapshot(), "report": report}


@app.post("/api/agent/generic-supervised/{session_id}/pause")
def pause_generic_supervised_session(
    session_id: str,
    body: GenericSupervisedDeviceRequest,
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    """Invalidate any pending confirmation without touching hardware."""

    verify_local_request(request, x_control_token)
    with runtime.generic_supervised_session_lock:
        session = runtime.generic_supervised_sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="通用单步会话不存在。")
        _require_generic_session_device(session, body.device_id)
        runtime.universal_agent_orchestrator.pause(session)
    report = _write_generic_supervised_report(session)
    return {
        "physical_actions": 0,
        "session": session.snapshot(),
        "report": report,
    }


@app.post("/api/stop")
def stop_all(
    request: Request,
    x_control_token: str | None = Header(default=None),
) -> dict[str, Any]:
    verify_local_request(request, x_control_token)
    runtime.controller.request_stop()
    capability_stop_requested = (
        runtime.capability_acceptance_manager.request_stop_all()
    )
    return {
        "stop_requested": True,
        "queued_cancelled": [],
        "capability_stop_requested": capability_stop_requested,
        "note": "正在执行的任务会在当前最小动作结束后停止。",
    }


@app.get("/api/preview.jpg")
def preview_jpg(device_id: str) -> Response:
    try:
        controller = runtime.controller_for_device(device_id)
    except UniversalAgentOrchestratorError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    try:
        content = controller.capture_preview()
    except Exception:
        content = MockRobotController(device_id="mock-preview").capture_preview()
    return Response(content, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/preview.mjpg")
def preview_mjpg(device_id: str) -> StreamingResponse:
    try:
        controller = runtime.controller_for_device(device_id)
    except UniversalAgentOrchestratorError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    def generate() -> Iterator[bytes]:
        while True:
            try:
                frame = controller.capture_preview(quality=68)
            except Exception:
                frame = MockRobotController(
                    device_id="mock-preview"
                ).capture_preview(quality=68)
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n"
                + f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii")
                + frame
                + b"\r\n"
            )
            time.sleep(0.35)

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"},
    )
