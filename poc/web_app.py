from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import threading
import time
import uuid
import webbrowser
from contextlib import asynccontextmanager, contextmanager, nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Literal

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from agent.application import (
    AgentDeviceRuntimeError,
    AgentSessionCommandError,
    StartUniversalAgentSessionCommand,
)
from agent.application.async_task_registry import AsyncTaskRegistry
from agent.domain import (
    AgentSession,
    AgentSessionConflictError,
    AgentSessionDeviceMismatchError,
    AgentSessionNotFoundError,
    DeviceTaskRegistryError,
    EvidenceStoreError,
)
from agent.infrastructure import (
    DeviceControllerRegistryError,
    DeviceRuntimeResourceError,
    InterProcessLease,
    SHARED_DEVICE_LEASE_DIR,
)
from agent.infrastructure.capability_acceptance import (
    CapabilityAcceptanceError,
)
from agent.domain.action_capabilities import physical_capability_for_action, unverified_promotable_actions
from agent.domain.action_catalog import PROMOTABLE_ACTION_KINDS
from agent.infrastructure.generic_action_adapter import persist_observer_failure_diagnostic
from agent.application.action_adapter import GenericActionAdapterError
from agent.application.universal_agent_orchestrator import (
    POST_ACTION_TRANSITION_PROTOCOL_VERSION,
    UniversalAgentOrchestratorError,
)
from agent.domain.universal_action_controller import (
    UNIVERSAL_CONTROLLER_PROTOCOL_VERSION,
    UniversalActionError,
)
from agent.domain.ui_scene import UI_SCENE_PROTOCOL_VERSION
from agent.domain.session import TERMINAL_SESSION_STATUSES
from agent.domain.action_catalog import CANONICAL_ACTION_KINDS
from agent.domain.canonical_action_protocol import CANONICAL_ACTION_PROTOCOL
from agent.domain.recent_navigation import RECENT_NAVIGATION_PROTOCOL
from agent.infrastructure.runtime_doctor import run_runtime_doctor
from agent.infrastructure.robot_controller import (
    MockRobotController,
    WEB_OUTPUT_DIR,
)
from agent.domain.vision_model import VisionAgentError
from agent.domain.execution_budget import (
    DEFAULT_DEVICE_ACTION_BUDGET,
    DEFAULT_OBSERVATION_BUDGET,
)
from agent.interfaces.http_models import (
    CapabilityAcceptanceStartRequest,
    CapabilityActionConfirmationRequest,
    CapabilityCancelRequest,
    CapabilityEffectApprovalRequest,
    CapabilityPromotionRequest,
    GenericEffectApprovalRequest,
    GenericSceneRequest,
    GenericSupervisedAutoRequest,
    GenericSupervisedDeviceRequest,
    GenericSupervisedStartRequest,
    GenericSupervisedStepRequest,
    MachinePositionRequest,
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
APP_PACKAGE_REGISTRY_PATH = Path(os.environ.get("ROBOT_APP_PACKAGE_REGISTRY",
    Path(__file__).with_name("app_package_registry.json")))
ADB_KEYBOARD_REGISTRY_PATH = Path(os.environ.get(
    "ROBOT_ADB_KEYBOARD_REGISTRY",
    Path(__file__).with_name("adb_keyboard_registry.json"),
))
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
from agent.bootstrap.runtime import (
    Runtime,
    RuntimeUnavailable,
    current_code_revision as _current_code_revision,
)


try:
    runtime: Runtime | RuntimeUnavailable = Runtime(
        device_registry_path=DEVICE_REGISTRY_PATH,
        app_package_registry_path=APP_PACKAGE_REGISTRY_PATH,
        adb_keyboard_registry_path=ADB_KEYBOARD_REGISTRY_PATH,
        output_dir=WEB_OUTPUT_DIR,
        ensure_device_ready=lambda device_id: _require_agent_device_ready(device_id),
        exclusive_device_session=lambda device_id: _agent_device_execution(device_id),
        code_revision_provider=lambda: _current_code_revision(ROOT.parent),
    )
except Exception as exc:
    runtime = RuntimeUnavailable(exc)


_GENERIC_START_TASKS = AsyncTaskRegistry(
    ttl_seconds=float(os.environ.get("ROBOT_START_TASK_TTL_SECONDS", "3600")),
    max_tasks=int(os.environ.get("ROBOT_START_TASK_MAX", "256")),
)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> Iterator[None]:
    runtime.start()
    print("机械臂网页控制台：http://127.0.0.1:8765/")
    print("控制令牌已生成，仅通过本机受保护的页面初始化接口使用。")
    if os.environ.get("ROBOT_WEB_NO_BROWSER") != "1":
        threading.Timer(
            1.0, lambda: webbrowser.open("http://127.0.0.1:8765/")
        ).start()
    yield
    runtime.shutdown()
    _GENERIC_START_TASKS.shutdown()


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
    ready = not isinstance(runtime, RuntimeUnavailable)
    return {
        "token": CONTROL_TOKEN,
        "mock": bool(ready and isinstance(runtime.controller, MockRobotController)),
        "ready": ready,
        "startup_error": runtime.startup_error if not ready else None,
        "version": app.version,
    }


@app.get("/api/apps")
def apps() -> dict[str, Any]:
    if isinstance(runtime, RuntimeUnavailable):
        return {"apps": APP_CATALOG, "readiness": {
            "runtime": {"ready": False, "error": runtime.startup_error}
        }}
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
    return {"apps": APP_CATALOG, "readiness": readiness}


@app.post("/api/device/{device_id}/machine-position")
def select_device_machine_position(
    device_id: str,
    body: MachinePositionRequest,
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    """Select one seller-control position from the web console."""

    verify_local_request(request, x_control_token)
    resolved_device = str(device_id or "").strip()
    try:
        controller = runtime.controller_for_device(resolved_device)
    except DeviceControllerRegistryError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if runtime.device_task_registry.active_session(resolved_device):
        raise HTTPException(status_code=409, detail="该设备正在执行任务，暂不能切换机位。")
    try:
        with _supervised_hardware_lock(resolved_device):
            controller.select_machine_position(body.machine_position)
            status = controller.device_status()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=409, detail=f"卖家控制端机位切换失败：{exc}") from exc
    return {
        "device_id": resolved_device,
        "machine_position": body.machine_position,
        "status": status,
    }


def _device_capability_snapshot(
    active_runtime: Runtime,
) -> tuple[str, dict[str, Any], dict[str, Any] | None, dict[str, Any] | None, set[str]]:
    default_device_id = active_runtime.device_controllers.default_device_id
    capability_provider = getattr(active_runtime.controller, "hardware_capabilities", None)
    hardware_capabilities = (
        capability_provider() if callable(capability_provider) else {}
    )
    capability_profile_provider = getattr(
        active_runtime.controller,
        "hardware_capability_profile",
        None,
    )
    hardware_capability_profile = (
        capability_profile_provider()
        if callable(capability_profile_provider)
        else None
    )
    default_text_transport = active_runtime.text_transport_for_device(default_device_id)
    text_transport_status = (
        default_text_transport.status()
        if default_text_transport is not None
        and callable(getattr(default_text_transport, "status", None))
        else None
    )
    if isinstance(hardware_capability_profile, dict) and default_text_transport is not None:
        hardware_capability_profile = dict(hardware_capability_profile)
        copied_actions = {
            str(name): dict(spec)
            for name, spec in dict(hardware_capability_profile.get("actions") or {}).items()
            if isinstance(spec, dict)
        }
        for action_name, operation in (
            ("input_verified_text", "append_text"),
            ("clear_verified_text", "clear_text"),
        ):
            if action_name in copied_actions:
                copied_actions[action_name]["text_transport"] = "adb_keyboard"
                copied_actions[action_name]["operation"] = operation
                copied_actions[action_name]["implicit_clear"] = False
                copied_actions[action_name]["retry_on_failure"] = False
        hardware_capability_profile["actions"] = copied_actions
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
    device_capabilities = effective_hardware_capabilities or hardware_capabilities
    enabled_physical_actions = {
        action
        for action in CANONICAL_ACTION_KINDS
        if action != "wait_for_change"
        and bool(device_capabilities.get(physical_capability_for_action(action), False))
    }
    default_app_launcher = active_runtime.app_launcher_for_device(default_device_id)
    if bool(getattr(default_app_launcher, "enabled", False)):
        enabled_physical_actions.add("launch_app")
    return (
        default_device_id,
        hardware_capabilities,
        hardware_capability_profile,
        text_transport_status,
        enabled_physical_actions,
    )


def _public_device_statuses(active_runtime: Runtime) -> list[dict[str, Any]]:
    devices = []
    for descriptor in active_runtime.device_controllers.descriptors():
        public_status = dict(
            active_runtime.controller_for_device(descriptor["device_id"]).device_status()
        )
        public_status.pop("readiness", None)
        devices.append(
            {
                **descriptor,
                **public_status,
                "capability_acceptance_actions": unverified_promotable_actions(
                    descriptor.get("verified_actions", [])
                ),
            }
        )
    return devices


def _execution_architecture_status(
    active_runtime: Runtime,
    *,
    hardware_capabilities: dict[str, Any],
    hardware_capability_profile: dict[str, Any] | None,
    text_transport_status: dict[str, Any] | None,
    enabled_physical_actions: set[str],
) -> dict[str, Any]:
    return {
        "model_role": "whole_task_visual_agent",
        "controller": "universal_action_controller",
        "fixed_app_workflows_retired": True,
        "active_orchestrator": "universal_agent",
        "universal_agent": {
            "goal_protocol": "2026-09-06-single-visual-task-v1",
            "scene_protocol": UI_SCENE_PROTOCOL_VERSION,
            "action_protocol": CANONICAL_ACTION_PROTOCOL,
            "recent_navigation_protocol": RECENT_NAVIGATION_PROTOCOL,
            "controller_protocol": UNIVERSAL_CONTROLLER_PROTOCOL_VERSION,
            "goal_preview_enabled": False,
            "scene_preview_enabled": True,
            "hardware_execution_enabled": active_runtime.hardware_mode,
            "execution_mode": (
                "mock"
                if active_runtime.mock_mode or isinstance(active_runtime.controller, MockRobotController)
                else ("hardware" if active_runtime.hardware_mode else "offline")
            ),
            "physical_execution": active_runtime.hardware_mode,
            "automatic_loop_enabled": True,
            "task_budget_default_actions": DEFAULT_DEVICE_ACTION_BUDGET,
            "task_budget_default_observations": DEFAULT_OBSERVATION_BUDGET,
            "supervised_single_step_enabled": False,
            "enabled_physical_actions": sorted(enabled_physical_actions),
            "protocol_physical_actions": sorted(CANONICAL_ACTION_KINDS - {"wait_for_change"}),
            "hardware_capabilities": hardware_capabilities,
            "hardware_capability_profile": hardware_capability_profile,
            "text_transport": text_transport_status,
            "supported_app_scope": "dynamic",
            "typed_effect_authority": {
                "authority_protocol": "2026-09-02-typed-effect-kind-v1",
                "effect_policy_protocol": "2026-09-02-auth-payment-only-v1",
                "authority_scope": "qwen_current_action_effect_kind",
                "retired_remote_risk_diagnostics_enabled": False,
                "canonical_action_protocol": CANONICAL_ACTION_PROTOCOL,
            },
            "observer": active_runtime.generic_scene_observer.status(),
        },
    }


def _active_session_status(active_runtime: Runtime) -> list[dict[str, Any]]:
    return [
        {
            "session_id": item["session_id"],
            "device_id": item["device_id"],
            "status": item["status"],
            "step_number": item["step_number"],
            "proposal": item["proposal"],
        }
        for item in active_runtime.universal_agent_session_service.active_snapshots()
    ]


@app.get("/api/device")
def device() -> dict[str, Any]:
    if isinstance(runtime, RuntimeUnavailable):
        raise HTTPException(status_code=503, detail={
            "code": "runtime_not_ready", "error": runtime.startup_error,
        })
    status = dict(runtime.controller.device_status())
    status.pop("readiness", None)
    (
        default_device_id,
        hardware_capabilities,
        hardware_capability_profile,
        text_transport_status,
        enabled_physical_actions,
    ) = _device_capability_snapshot(runtime)
    status["default_device_id"] = default_device_id
    status["devices"] = _public_device_statuses(runtime)
    status["vision_agent"] = runtime.vision_provider.status()
    status["execution_architecture"] = _execution_architecture_status(
        runtime,
        hardware_capabilities=hardware_capabilities,
        hardware_capability_profile=hardware_capability_profile,
        text_transport_status=text_transport_status,
        enabled_physical_actions=enabled_physical_actions,
    )
    active_sessions = _active_session_status(runtime)
    status["active_tasks"] = [dict(item) for item in active_sessions]
    status["generic_supervised_execution"] = {
        "enabled": True,
        "automatic_loop_enabled": True,
        "max_physical_actions_per_confirmation": 1,
        "post_action_transition_protocol": (
            POST_ACTION_TRANSITION_PROTOCOL_VERSION
        ),
        "active_sessions": active_sessions,
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
    coordination_lock = runtime.device_runtime_resources.coordination_lock(
        device_id
    )
    acquired = coordination_lock.acquire(blocking=False)
    if not acquired and not active_session:
        active_session = "device-coordination-busy"
    try:
        camera_session = (
            runtime.serial_camera_session(device_id)
            if acquired
            else nullcontext()
        )
        with camera_session:
            return run_runtime_doctor(
                device_id=device_id,
                controller=controller,
                qwen_provider=runtime.vision_provider,
                text_transport=runtime.text_transport_for_device(device_id),
                active_session=active_session,
                protocols={
                    "goal": "2026-09-06-single-visual-task-v1",
                    "scene": UI_SCENE_PROTOCOL_VERSION,
                    "action": CANONICAL_ACTION_PROTOCOL,
                    "controller": UNIVERSAL_CONTROLLER_PROTOCOL_VERSION,
                    "risk": "2026-09-02-auth-payment-only-v1",
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
    coordination_lock = runtime.device_runtime_resources.coordination_lock(
        resolved_device
    )
    if not coordination_lock.acquire(blocking=False):
        process_lease.release()
        raise HTTPException(status_code=409, detail="已有语义观察或动作正在进行。")
    if not controller.operation_lock.acquire(blocking=False):
        coordination_lock.release()
        process_lease.release()
        raise HTTPException(status_code=409, detail="机械臂物理控制权已被占用。")
    try:
        with runtime.serial_camera_session(resolved_device):
            yield
    finally:
        controller.operation_lock.release()
        coordination_lock.release()
        process_lease.release()


def _require_agent_device_ready(device_id: str) -> None:
    try:
        _require_supervised_device_ready(device_id)
    except HTTPException as exc:
        raise AgentDeviceRuntimeError(
            exc.detail,
            status_code=exc.status_code,
        ) from exc


@contextmanager
def _agent_device_execution(device_id: str) -> Iterator[None]:
    try:
        with _supervised_hardware_lock(device_id):
            yield
    except HTTPException as exc:
        raise AgentDeviceRuntimeError(
            exc.detail,
            status_code=exc.status_code,
        ) from exc


def _require_generic_supervised_session(session_id: str) -> AgentSession:
    try:
        return runtime.universal_agent_session_service.require(session_id)
    except AgentSessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _raise_agent_device_runtime_error(exc: AgentDeviceRuntimeError) -> None:
    raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


def _raise_agent_session_device_mismatch(
    exc: AgentSessionDeviceMismatchError,
) -> None:
    raise HTTPException(
        status_code=409,
        detail={
            "success": False,
            "physical_actions": 0,
            "error": str(exc),
        },
    ) from exc


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


def _write_generic_supervised_report(session: AgentSession) -> str:
    """Return the atomic report already maintained by the orchestrator."""
    if session.status in TERMINAL_SESSION_STATUSES:
        try:
            (session.run_dir / ".active").unlink(missing_ok=True)
        except OSError:
            pass
    return str(session.run_dir / "report.json")


def _generic_supervised_failure(
    session: AgentSession | None,
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


COMMON_OPERATION_ERRORS = (
    DeviceControllerRegistryError,
    DeviceRuntimeResourceError,
    DeviceTaskRegistryError,
    GenericActionAdapterError,
    UniversalActionError,
    EvidenceStoreError,
    UniversalAgentOrchestratorError,
    VisionAgentError,
)

CAPABILITY_ACCEPTANCE_ERRORS = (CapabilityAcceptanceError, *COMMON_OPERATION_ERRORS)

GENERIC_SESSION_ERRORS = (
    AgentSessionCommandError,
    AgentSessionConflictError,
    *COMMON_OPERATION_ERRORS,
)


def _confirmation_payload(body: Any) -> dict[str, Any] | None:
    confirmation = getattr(body, "confirmation", None)
    return confirmation.model_dump() if confirmation is not None else None


@app.post("/api/capability-acceptance/start")
def start_capability_acceptance(
    body: CapabilityAcceptanceStartRequest,
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    """Plan and observe one unverified generic action with zero execution."""

    verify_local_request(request, x_control_token)
    if body.action not in PROMOTABLE_ACTION_KINDS:
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


@app.post("/api/agent/generic-supervised/start-async")
def start_generic_supervised_async(body: GenericSupervisedStartRequest, request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token")) -> dict[str, Any]:
    verify_local_request(request, x_control_token)
    task_id = uuid.uuid4().hex
    try:
        _GENERIC_START_TASKS.reserve(task_id)
        _GENERIC_START_TASKS.submit(
            task_id,
            lambda: start_generic_supervised_session(body, request, x_control_token),
        )
    except OverflowError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail="异步启动执行器当前不可用。") from exc
    return {"task_id": task_id, "status": "running"}


@app.get("/api/agent/generic-supervised/start-async/{task_id}")
def get_generic_supervised_start_task(
    task_id: str,
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    verify_local_request(request, x_control_token)
    task = _GENERIC_START_TASKS.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="启动任务不存在或已过期。")
    return {"task_id": task_id, **{key: value for key, value in task.items() if key != "created_monotonic"}}


@app.post("/api/agent/generic-supervised/start")
def start_generic_supervised_session(
    body: GenericSupervisedStartRequest,
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    """Start one canonical whole-task visual session.

    Automatic mode advances the same Qwen action/finish loop; it does not build
    a local plan or restrict execution to a read-only action subset.
    """

    verify_local_request(request, x_control_token)
    session_id = uuid.uuid4().hex
    run_dir = WEB_OUTPUT_DIR / (
        "generic_supervised_"
        + datetime.now().strftime("%Y%m%d_%H%M%S_")
        + session_id[:8]
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / ".evidence-run").touch()
    (run_dir / ".active").touch()
    session = None
    try:
        started = runtime.universal_agent_session_service.start(
            StartUniversalAgentSessionCommand(
                session_id=session_id,
                raw_goal=body.text,
                exact_input_text=body.exact_input_text,
                exact_action_kind=body.exact_action_kind,
                exact_target_label=body.exact_target_label,
                device_id=body.device_id,
                run_dir=run_dir,
                auto_advance=body.auto_advance,
                max_physical_actions=body.max_physical_actions,
                max_observations=body.max_observations,
            )
        )
        session = started.session
        auto_result = started.automatic_progress
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
    except AgentDeviceRuntimeError as exc:
        try:
            if session is None:
                shutil.rmtree(run_dir, ignore_errors=True)
            else:
                (run_dir / ".active").unlink(missing_ok=True)
        except OSError:
            pass
        _raise_agent_device_runtime_error(exc)
    except GENERIC_SESSION_ERRORS as exc:
        if session is None:
            try:
                session = runtime.universal_agent_session_service.require(session_id)
            except AgentSessionNotFoundError:
                try:
                    shutil.rmtree(run_dir, ignore_errors=True)
                except OSError:
                    pass
        failure = _generic_supervised_failure(session, exc)
        failure["evidence"] = [
            str(path) for path in sorted(run_dir.glob("*.jpg"))
        ]
        raise HTTPException(status_code=409, detail=failure) from exc


@app.get("/api/agent/generic-supervised/{session_id}")
def get_generic_supervised_session(
    session_id: str,
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    verify_local_request(request, x_control_token)
    session = _require_generic_supervised_session(session_id)
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
    session = _require_generic_supervised_session(session_id)
    before_actions = session.physical_actions
    try:
        approved = runtime.universal_agent_session_service.approve_effects(
            session,
            confirmed=body.confirmed,
            confirmation=(
                _confirmation_payload(body)
            ),
        )
        result = approved.operation
        request_actions = approved.physical_actions
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
    except AgentDeviceRuntimeError as exc:
        _raise_agent_device_runtime_error(exc)
    except AgentSessionDeviceMismatchError as exc:
        _raise_agent_session_device_mismatch(exc)
    except GENERIC_SESSION_ERRORS as exc:
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
    session = _require_generic_supervised_session(session_id)
    before_actions = session.physical_actions
    try:
        confirmed = runtime.universal_agent_session_service.confirm(
            session,
            confirmed=body.confirmed,
            confirmation=(
                _confirmation_payload(body)
            ),
        )
        result = confirmed.operation
        report = _write_generic_supervised_report(session)
        return {
            "mode": "generic_supervised_single_step",
            "automatic_loop_enabled": False,
            "execution": result.to_dict(),
            "session": session.snapshot(),
            "report": report,
        }
    except AgentDeviceRuntimeError as exc:
        _raise_agent_device_runtime_error(exc)
    except GENERIC_SESSION_ERRORS as exc:
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
    session = _require_generic_supervised_session(session_id)
    before_actions = session.physical_actions
    try:
        refreshed = runtime.universal_agent_session_service.refresh(
            session,
            requested_device_id=body.device_id,
        )
        decision = refreshed.operation
        report = _write_generic_supervised_report(session)
        return {
            "mode": "generic_supervised_single_step",
            "physical_actions": 0,
            "automatic_loop_enabled": False,
            "proposal": decision.proposal.to_dict(),
            "session": session.snapshot(),
            "report": report,
        }
    except AgentDeviceRuntimeError as exc:
        _raise_agent_device_runtime_error(exc)
    except AgentSessionDeviceMismatchError as exc:
        _raise_agent_session_device_mismatch(exc)
    except GENERIC_SESSION_ERRORS as exc:
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
    """Continue the task within cumulative limits; only login/payment require confirmation."""

    verify_local_request(request, x_control_token)
    session = _require_generic_supervised_session(session_id)
    before_actions = session.physical_actions
    try:
        automatic = runtime.universal_agent_session_service.run_automatic(
            session,
            requested_device_id=body.device_id,
            confirmed=body.confirmed,
            confirmation=(
                _confirmation_payload(body)
            ),
            max_physical_actions=body.max_physical_actions,
            max_observations=body.max_observations,
        )
        result = automatic.operation
        report = _write_generic_supervised_report(session)
        return {
            "mode": "generic_supervised_safe_loop",
            "automatic_loop_enabled": True,
            "execution": result,
            "session": session.snapshot(),
            "report": report,
        }
    except AgentDeviceRuntimeError as exc:
        _raise_agent_device_runtime_error(exc)
    except AgentSessionDeviceMismatchError as exc:
        _raise_agent_session_device_mismatch(exc)
    except GENERIC_SESSION_ERRORS as exc:
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
    try:
        cancelled = runtime.universal_agent_session_service.cancel(
            session_id,
            requested_device_id=body.device_id,
        )
    except AgentSessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except AgentSessionDeviceMismatchError as exc:
        _raise_agent_session_device_mismatch(exc)
    session = cancelled.session
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
    try:
        paused = runtime.universal_agent_session_service.pause(
            session_id,
            requested_device_id=body.device_id,
        )
    except AgentSessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except AgentSessionDeviceMismatchError as exc:
        _raise_agent_session_device_mismatch(exc)
    session = paused.session
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
    stopped_devices = runtime.device_controllers.request_stop_all()
    # Tests and embedding callers may replace the default controller after the
    # registry was built; keep that explicit runtime handle covered as well.
    runtime.controller.request_stop()
    capability_stop_requested = (
        runtime.capability_acceptance_manager.request_stop_all()
    )
    return {
        "stop_requested": True,
        "stopped_device_ids": list(stopped_devices),
        "queued_cancelled": [],
        "capability_stop_requested": capability_stop_requested,
        "note": "正在执行的任务会在当前最小动作结束后停止。",
    }


@app.get("/api/preview.jpg")
def preview_jpg(device_id: str) -> Response:
    try:
        content, cached = runtime.capture_preview(device_id)
    except (
        DeviceControllerRegistryError,
        DeviceRuntimeResourceError,
        UniversalAgentOrchestratorError,
    ) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"实时相机预览不可用：{exc}",
        ) from exc
    return Response(
        content,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-store",
            "X-Camera-Source": "cache" if cached else "live",
        },
    )


@app.get("/api/preview.mjpg")
def preview_mjpg(device_id: str) -> StreamingResponse:
    try:
        runtime.controller_for_device(device_id)
    except (
        DeviceControllerRegistryError,
        DeviceRuntimeResourceError,
        UniversalAgentOrchestratorError,
    ) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    def generate() -> Iterator[bytes]:
        while True:
            try:
                frame, _cached = runtime.capture_preview(device_id, quality=68)
            except Exception:
                return
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
