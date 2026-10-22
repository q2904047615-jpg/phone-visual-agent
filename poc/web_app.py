from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import secrets
import shutil
import sys
import threading
import time
import uuid
import webbrowser
from contextlib import asynccontextmanager, contextmanager, nullcontext
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Literal, Mapping
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from agent.interfaces import website_access
from fastapi.responses import FileResponse, Response, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from agent.application import (
    AgentEntryRouteError,
    AgentEntryRouter,
    AgentDeviceRuntimeError,
    AgentSessionCommandError,
    AgentSessionRecoveryRequiredError,
    RuntimeQuotaExceeded,
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
    SqliteControlPlaneRepository,
    ExternalIdentityProviderRegistry,
    ExternalIdentityMappingNotFound,
    ExternalIdentityProviderUnavailable,
    GitHubOAuthConfigurationError,
    GitHubOAuthProvider,
    ConnectorProtocolError,
    validate_result as validate_connector_result,
)
from agent.domain.control_plane import (
    manageable_member_roles,
    API_CONTRACT_VERSION,
    CONTROL_PLANE_PROTOCOL,
    EXTERNAL_IDENTITY_PROTOCOL,
    GITHUB_OAUTH_BROWSER_PROTOCOL,
)
from agent.infrastructure.capability_acceptance import (
    CapabilityAcceptanceError,
)
from agent.domain.action_capabilities import physical_capability_for_action, unverified_promotable_actions
from agent.domain.action_catalog import PROMOTABLE_ACTION_KINDS
from agent.infrastructure.generic_action_adapter import persist_observer_failure_diagnostic
from agent.infrastructure.adb_pairing import AdbPairingError
from agent.infrastructure.manual_motion import (
    BackgroundWindowTransport,
    ManualMotionMonitor,
    MotionProfile,
)
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
from agent.infrastructure.sqlite_control_plane_repository import LOCAL_ORGANIZATION_ID
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
    ControlPlaneExternalIdentityRequest,
    ControlPlaneIdentitySessionRequest,
    ControlPlanePrincipalWriteRequest,
    ControlPlanePrincipalUpdateRequest,
    GitHubIdentityWriteRequest,
    ControlPlaneQuotaWriteRequest,
    ControlPlanePlatformRollbackRequest,
    DeviceAssignmentRequest,
    ControlPlaneTeamUpdateRequest,
    ControlPlaneTeamWriteRequest,
    ControlPlaneProjectUpdateRequest,
    ControlPlaneProjectWriteRequest,
    DevicePairRequest,
    DirectionalSwipeRequest,
    GenericEffectApprovalRequest,
    GenericSceneRequest,
    GenericSupervisedAutoRequest,
    GenericSupervisedDeviceRequest,
    GenericRecoveryReconciliationRequest,
    GenericSupervisedStartRequest,
    GenericSupervisedStepRequest,
    AgentChatRequest,
    AgentEntryRequest,
    MachinePositionRequest,
    ManualMotionStartRequest,
)
from features.lucky_bag import (
    DedicatedLuckyBagMonitor,
    FastLuckyBagSessionGateway,
    LuckyBagMonitor,
    LuckyBagProfile,
    build_lucky_bag_goal,
)


_LOGGER = logging.getLogger(__name__)
_SENSITIVE_DIAGNOSTIC_RE = re.compile(
    r"(?i)(authorization|api[-_ ]?key|token|password|secret|cookie)\s*[:=]\s*[^\s,;]+"
)


def _diagnostic_error_text(error: BaseException, *, limit: int = 200) -> str:
    """Return a short, single-line diagnostic without a traceback or payload."""
    text = " ".join(str(error).split())
    if not text:
        return type(error).__name__
    return _SENSITIVE_DIAGNOSTIC_RE.sub(r"\1=[REDACTED]", text)[:limit]

from features.lucky_bag.http import LuckyBagDeviceRequest, LuckyBagStartRequest
from features.lucky_bag.gmail import DurableNotificationRouter, configured_gmail_sink, gmail_configuration_status
from features.notifications import JsonlNotificationOutbox
from features.qishui_ad_test import (
    QishuiAdTestMonitor,
    QishuiAdTestProfile,
    build_qishui_ad_test_goal,
)
from features.qishui_ad_test.http import (
    QishuiAdTestConfirmationRequest,
    QishuiAdTestDeviceRequest,
    QishuiAdTestStartRequest,
)


ROOT = Path(__file__).resolve().parent
_TRIAL_CONFIG_PATH = ROOT.parent / "trial.json"
try:
    _TRIAL_CONFIG = json.loads(_TRIAL_CONFIG_PATH.read_text(encoding="utf-8"))
except (OSError, ValueError, TypeError):
    _TRIAL_CONFIG = {}
_TRIAL_MODE = str(_TRIAL_CONFIG.get("mode") or "")
_TRIAL_PORT = int(_TRIAL_CONFIG.get("port") or 8765)
_SERVICE_VERSION = f"0.2.0-trial-{_TRIAL_MODE}" if _TRIAL_MODE else "0.2.0"
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

control_plane_identity_registry = ExternalIdentityProviderRegistry()
_github_oauth_provider = None
_github_oauth_pending_verifiers: dict[str, tuple[str, float, str]] = {}
_BROWSER_SESSION_COOKIE = website_access.PRINCIPAL_COOKIE
_OAUTH_BROWSER_COOKIE = "agent_oauth_browser"
_AUTH_COOKIE_PATH = "/api/control-plane/auth"
_GITHUB_OAUTH_STATE_TTL_SECONDS = 600
try:
    _github_oauth_provider = GitHubOAuthProvider.from_environment(identity_store=getattr(runtime, "control_plane_repository", None))
except GitHubOAuthConfigurationError as exc:
    # A partial deployment configuration must fail closed for external login
    # without preventing the local phone Runtime from starting.
    _LOGGER.warning("GitHub OAuth provider disabled: %s", _diagnostic_error_text(exc))
if _github_oauth_provider is not None:
    control_plane_identity_registry.register(_github_oauth_provider)


class _QishuiAdTestGateway:
    """Expose only start, observe, one confirmed back, and cancel."""

    def start(self, *, goal: str, device_id: str, run_dir: Path) -> dict[str, Any]:
        if isinstance(runtime, RuntimeUnavailable):
            raise RuntimeError(runtime.startup_error)
        session_id = uuid.uuid4().hex
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / ".evidence-run").touch()
        (run_dir / ".active").touch()
        started = runtime.universal_agent_session_service.start(
            StartUniversalAgentSessionCommand(
                session_id=session_id,
                raw_goal=goal,
                exact_input_text=None,
                exact_action_kind=None,
                exact_target_label="",
                device_id=device_id,
                run_dir=run_dir,
                auto_advance=False,
                max_physical_actions=1,
                max_observations=20,
            )
        )
        return {"session": started.session.snapshot()}

    def get(self, session_id: str) -> dict[str, Any]:
        if isinstance(runtime, RuntimeUnavailable):
            raise RuntimeError(runtime.startup_error)
        return {"session": runtime.universal_agent_session_service.require(session_id).snapshot()}

    def observe(self, session_id: str, *, device_id: str) -> dict[str, Any]:
        if isinstance(runtime, RuntimeUnavailable):
            raise RuntimeError(runtime.startup_error)
        session = runtime.universal_agent_session_service.require(session_id)
        refreshed = runtime.universal_agent_session_service.refresh(
            session, requested_device_id=device_id,
        )
        return {"session": refreshed.session.snapshot()}

    def confirm(self, session_id: str, *, device_id: str,
        confirmation: dict[str, Any]) -> dict[str, Any]:
        if isinstance(runtime, RuntimeUnavailable):
            raise RuntimeError(runtime.startup_error)
        session = runtime.universal_agent_session_service.require(session_id)
        confirmed = runtime.universal_agent_session_service.confirm(
            session, confirmed=True, confirmation=confirmation,
        )
        return {"session": confirmed.session.snapshot()}

    def cancel(self, session_id: str, *, device_id: str) -> dict[str, Any]:
        if isinstance(runtime, RuntimeUnavailable):
            raise RuntimeError(runtime.startup_error)
        session = runtime.universal_agent_session_service.require(session_id)
        cancelled = runtime.universal_agent_session_service.cancel(
            session_id, requested_device_id=device_id,
        )
        return {"session": cancelled.session.snapshot()}


class _LuckyBagGateway:
    """Adapter from the monitor to the isolated two-observation flow."""

    def _fast(self) -> FastLuckyBagSessionGateway:
        if isinstance(runtime, RuntimeUnavailable):
            raise RuntimeError(runtime.startup_error)
        value = getattr(self, "_runner", None)
        if value is None:
            value = self._runner = FastLuckyBagSessionGateway(
                controller_for_device=runtime.controller_for_device,
                capture_frame=runtime.capture_agent_frame,
                observer=runtime.generic_scene_observer,
                ensure_device_ready=lambda device_id: _require_agent_device_ready(device_id),
                exclusive_device_session=lambda device_id: _agent_device_execution(device_id),
            )
        return value

    def start(self, *, goal: str, device_id: str, run_dir: Path, visual_reference_paths: tuple[Path, ...] = ()) -> dict[str, Any]:
        if isinstance(runtime, RuntimeUnavailable):
            raise RuntimeError(runtime.startup_error)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / ".evidence-run").touch()
        (run_dir / ".active").touch()
        return self._fast().start(
            goal=goal, device_id=device_id, run_dir=run_dir,
            visual_reference_paths=tuple(visual_reference_paths),
        )

    def get(self, session_id: str) -> dict[str, Any]:
        return self._fast().get(session_id)

    def auto(self, session_id: str, *, max_physical_actions: int, max_observations: int) -> dict[str, Any]:
        return self._fast().auto(
            session_id, max_physical_actions=max_physical_actions,
            max_observations=max_observations,
        )

    def pause(self, session_id: str) -> dict[str, Any]:
        return self._fast().pause(session_id)

    def cancel(self, session_id: str) -> dict[str, Any]:
        return self._fast().cancel(session_id)


_LUCKY_BAG_OUTPUT_DIR = WEB_OUTPUT_DIR / "lucky_bag"
_LUCKY_BAG_OUTBOX = JsonlNotificationOutbox(
    _LUCKY_BAG_OUTPUT_DIR / "lucky_bag_notifications.jsonl"
)
_LUCKY_BAG_NOTIFICATIONS = DurableNotificationRouter(
    outbox=_LUCKY_BAG_OUTBOX,
    remote=configured_gmail_sink(),
)
_LUCKY_BAG_MONITOR = DedicatedLuckyBagMonitor(
    runtime=runtime,
    hardware_lock=lambda device_id: _supervised_hardware_lock(device_id),
    output_root=_LUCKY_BAG_OUTPUT_DIR,
    notification_sink=_LUCKY_BAG_NOTIFICATIONS,
)

_QISHUI_AD_TEST_OUTPUT_DIR = WEB_OUTPUT_DIR / "qishui_ad_test"
_QISHUI_AD_TEST_MONITOR = QishuiAdTestMonitor(
    gateway=_QishuiAdTestGateway(),
    output_root=_QISHUI_AD_TEST_OUTPUT_DIR,
)

_GENERIC_START_TASKS = AsyncTaskRegistry(
    ttl_seconds=float(os.environ.get("ROBOT_START_TASK_TTL_SECONDS", "3600")),
    max_tasks=int(os.environ.get("ROBOT_START_TASK_MAX", "256")),
)


def _manual_transport_factory(device_id: str) -> BackgroundWindowTransport:
    controller = runtime.controller_for_device(device_id)
    return BackgroundWindowTransport(
        controller,
        mock=isinstance(controller, MockRobotController),
    )


_MANUAL_MOTION_OUTPUT_DIR = WEB_OUTPUT_DIR / "manual_motion"
_MANUAL_MOTION_MONITOR = ManualMotionMonitor(
    registry=(runtime.device_task_registry if not isinstance(runtime, RuntimeUnavailable) else None),
    transport_factory=_manual_transport_factory,
    output_dir=_MANUAL_MOTION_OUTPUT_DIR,
)

_API_RELOAD_LOCK = threading.Lock()
_API_RELOAD_SCHEDULED = False


def _api_reload_blockers() -> list[dict[str, Any]]:
    """Return active work that must settle before replacing this API process."""

    blockers: list[dict[str, Any]] = []
    queued_starts = getattr(_GENERIC_START_TASKS, "active_tasks", None)
    if callable(queued_starts):
        blockers.extend({"kind": "async_start", "task_id": item.get("task_id"),
                         "status": item.get("status")} for item in queued_starts())
    if not isinstance(runtime, RuntimeUnavailable):
        blockers.extend(
            {
                "kind": "generic_session",
                "session_id": item["session_id"],
                "device_id": item["device_id"],
                "status": item["status"],
            }
            for item in _active_session_status(runtime)
        )
    for record in _LUCKY_BAG_MONITOR.list():
        if record.status in {"starting", "running", "waiting_confirmation"}:
            blockers.append({
                "kind": "lucky_bag_monitor",
                "monitor_id": record.monitor_id,
                "device_id": record.profile.device_id,
                "status": record.status,
            })
    for record in _MANUAL_MOTION_MONITOR.list():
        if record["status"] in ManualMotionMonitor.ACTIVE:
            blockers.append({
                "kind": "manual_motion",
                "motion_id": record["motion_id"],
                "device_id": record["profile"]["device_id"],
                "status": record["status"],
            })
    return blockers


def _replace_api_process() -> None:
    """Replace this checkout's API process after the HTTP response is sent."""

    global _API_RELOAD_SCHEDULED
    time.sleep(0.2)
    os.chdir(ROOT)
    os.execv(
        sys.executable,
        [
            sys.executable,
            "-m",
            "uvicorn",
            *(["single_executor:create_app", "--factory"]
              if os.environ.get("PHONE_AGENT_EXECUTOR_MODE") == "1" else ["web_app:app"]),
            "--host",
            os.environ.get("ROBOT_API_HOST", "127.0.0.1"),
            "--port",
            os.environ.get("ROBOT_API_PORT", "8765"),
        ],
    )


def _schedule_api_reload() -> bool:
    global _API_RELOAD_SCHEDULED
    with _API_RELOAD_LOCK:
        if _API_RELOAD_SCHEDULED:
            return False
        _API_RELOAD_SCHEDULED = True
        threading.Thread(
            target=_replace_api_process,
            name="api-self-reload",
            daemon=True,
        ).start()
        return True


@asynccontextmanager
async def lifespan(_app: FastAPI) -> Iterator[None]:
    runtime.start()
    _MANUAL_MOTION_MONITOR.registry = runtime.device_task_registry
    # Keep startup notices ASCII-only so a Windows cp1252 stdout cannot fail
    # the lifespan before the application is ready.
    print(f"Phone Visual Agent web console: http://127.0.0.1:{_TRIAL_PORT}/", flush=True)
    print("Local control token generated for the protected bootstrap page.", flush=True)
    if os.environ.get("ROBOT_WEB_NO_BROWSER") != "1":
        threading.Timer(
            1.0, lambda: webbrowser.open(f"http://127.0.0.1:{_TRIAL_PORT}/")
        ).start()
    yield
    # In single-executor mode this joins running starts before their Runtime
    # repositories/controllers are closed; the OS owner lease remains held.
    _GENERIC_START_TASKS.shutdown()
    if _github_oauth_provider is not None:
        _github_oauth_provider.close()
    _MANUAL_MOTION_MONITOR.shutdown()
    runtime.shutdown()
    _LUCKY_BAG_MONITOR.shutdown()
    _QISHUI_AD_TEST_MONITOR.shutdown()


async def _authorize_website_request(request: Request):
    if request.url.path in {"/", "/api/session"}:
        yield
        return
    store = _control_plane_store()
    await website_access.authorize(request, store, verify_local_origin,
                                   runtime.device_controllers.default_device_id)
    devices = getattr(request.state, "website_devices", set())
    # A stream checks each frame; holding a lease for the entire stream would
    # make merely watching a phone prevent reassignment forever.
    with website_access.member_access(store, request):
        if devices and request.url.path != "/api/preview.mjpg":
            with website_access.device_access(store, request, devices):
                yield
        else:
            yield


app = FastAPI(
    title="多 App 机械臂网页控制平台",
    version=_SERVICE_VERSION,
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    dependencies=[Depends(_authorize_website_request)],
)
app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")


@app.middleware("http")
async def disable_static_cache(request: Request, call_next):
    """Always serve the current local UI files when the console is refreshed."""

    response = await call_next(request)
    path = request.url.path
    if path == "/" or path.startswith("/assets/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    if path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
        response.headers["Referrer-Policy"] = "no-referrer"
    return response


def _configured_public_origin() -> str | None:
    """Return the one explicitly configured public HTTPS origin, if valid."""

    raw = os.environ.get("PUBLIC_BASE_URL", "").strip().rstrip("/")
    if not raw:
        return None
    parsed = urlsplit(raw)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
        or parsed.hostname.lower() in {"localhost", "127.0.0.1", "::1"}
    ):
        return None
    return f"https://{parsed.netloc}".rstrip("/")


def verify_local_origin(request: Request) -> None:
    origin = request.headers.get("origin")
    if not origin:
        return
    normalized = origin.strip().rstrip("/")
    parsed = urlsplit(normalized)
    if (
        not parsed.scheme
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise HTTPException(status_code=403, detail="来源地址无效。")

    request_origin = str(request.base_url).rstrip("/")
    local_origin = parsed.hostname.lower() in {"127.0.0.1", "localhost", "::1"}
    if local_origin:
        if normalized != request_origin:
            raise HTTPException(status_code=403, detail="只允许本机同源网页请求。")
        return

    public_origin = _configured_public_origin()
    if public_origin and normalized == public_origin:
        return
    if public_origin:
        raise HTTPException(status_code=403, detail="只允许配置的公网同源网页请求。")
    raise HTTPException(status_code=403, detail="只允许本机网页请求。")


def verify_local_request(request: Request, token: str | None) -> None:
    verify_local_origin(request)
    if token != CONTROL_TOKEN:
        raise HTTPException(status_code=403, detail="控制令牌无效。")


def _control_plane_store() -> SqliteControlPlaneRepository:
    if isinstance(runtime, RuntimeUnavailable):
        raise HTTPException(status_code=503, detail="Control Plane 持久化尚未就绪。")
    store = getattr(runtime, "control_plane_repository", None)
    if not isinstance(store, SqliteControlPlaneRepository):
        raise HTTPException(status_code=503, detail="Control Plane 持久化尚未就绪。")
    if not store.website_ownership_initialized():
        legacy = [("trial", item["trial_id"], item["device_id"])
                  for item in runtime.capability_acceptance_manager.snapshots()]
        legacy += [("monitor", item.monitor_id, item.profile.device_id) for item in _LUCKY_BAG_MONITOR.list()]
        legacy += [("ad_test", item.test_id, item.profile.device_id) for item in _QISHUI_AD_TEST_MONITOR.list()]
        legacy += [("motion", item["motion_id"], item["profile"]["device_id"]) for item in _MANUAL_MOTION_MONITOR.list()]
        store.initialize_website_ownership(runtime.device_controllers.default_device_id, legacy_resources=legacy)
    return store


def _control_plane_request_id(request: Request) -> str:
    value = str(request.headers.get("x-request-id") or "").strip()
    return value[:128] if value else uuid.uuid4().hex


def _claim_website_resource(request: Request, kind: str, resource_id: str, device_id: str):
    if getattr(getattr(request, "state", None), "principal", None):
        website_access.claim(_control_plane_store(), request, kind, resource_id, device_id)


def _website_visible(request: Request | None, kind: str, resource_id: str) -> bool:
    if not getattr(getattr(request, "state", None), "principal", None):
        return True  # Trusted in-process callers, never an HTTP identity fallback.
    return website_access.visible(_control_plane_store(), request, kind, resource_id)


def _website_model_status(value: Mapping[str, Any]) -> dict[str, Any]:
    # Process-wide last-request diagnostics have no user ownership. Per-run
    # model usage and context remain available through the owned task/ledger.
    return {key: item for key, item in value.items()
            if not key.startswith("last_") and key not in {"usage_totals", "successful_call_count"}}


def _control_plane_envelope(data: Any) -> dict[str, Any]:
    return {
        "control_plane_protocol": CONTROL_PLANE_PROTOCOL,
        "api_version": API_CONTRACT_VERSION,
        "data": data,
    }


def _validate_control_plane_page(after_sequence: int, limit: int) -> tuple[int, int]:
    if isinstance(after_sequence, bool) or after_sequence < 0:
        raise HTTPException(status_code=422, detail="after_sequence 必须是非负整数。")
    if isinstance(limit, bool) or limit < 1 or limit > 100:
        raise HTTPException(status_code=422, detail="limit 必须在 1-100。")
    return int(after_sequence), int(limit)


def _control_plane_write_context(
    request: Request,
    organization_id: str,
    operation: str,
    permission: str,
    x_control_token: str | None,
    x_control_plane_token: str | None,
    idempotency_key: str | None,
) -> tuple[SqliteControlPlaneRepository, dict[str, Any], str]:
    verify_local_request(request, x_control_token)
    store = _control_plane_store()
    request_id = _control_plane_request_id(request)
    x_control_plane_token = x_control_plane_token or website_access.identity_token(request)
    if not x_control_plane_token:
        store.record_audit(organization_id=organization_id, resource=operation, outcome="identity_required",
                           request_id=request_id, metadata={"permission": permission}, actor="anonymous")
        raise HTTPException(status_code=401, detail={"code": "identity_required", "message": "需要 Control Plane 主体会话。"})
    principal = store.authenticate_identity_session(x_control_plane_token)
    if principal is None:
        store.record_audit(organization_id=organization_id, resource=operation, outcome="identity_invalid",
                           request_id=request_id, metadata={"permission": permission}, actor="anonymous")
        raise HTTPException(status_code=401, detail={"code": "identity_invalid", "message": "Control Plane 主体会话无效或已过期。"})
    if principal["organization_id"] != organization_id:
        store.record_audit(organization_id=organization_id, resource=operation, outcome="organization_forbidden",
                           request_id=request_id, metadata={"permission": permission, "user_id": principal["user_id"]}, actor=principal["user_id"])
        raise HTTPException(status_code=403, detail={"code": "organization_forbidden", "message": "主体不属于请求组织。"})
    if permission not in principal["permissions"]:
        store.record_audit(organization_id=organization_id, resource=operation, outcome="permission_denied",
                           request_id=request_id, metadata={"permission": permission, "user_id": principal["user_id"]}, actor=principal["user_id"])
        raise HTTPException(status_code=403, detail={"code": "permission_denied", "message": f"缺少权限：{permission}。"})
    key = str(idempotency_key or "").strip()
    if not key or len(key) > 128:
        store.record_audit(organization_id=organization_id, resource=operation, outcome="idempotency_required",
                           request_id=request_id, metadata={"permission": permission, "user_id": principal["user_id"]}, actor=principal["user_id"])
        raise HTTPException(status_code=422, detail={"code": "idempotency_required", "message": "写操作必须提供 1-128 位 Idempotency-Key。"})
    store.record_audit(organization_id=organization_id, resource=operation, outcome="permission_allowed",
                       request_id=request_id, metadata={"permission": permission, "user_id": principal["user_id"]}, actor=principal["user_id"])
    return store, principal, key


def _control_plane_write_envelope(
    request: Request,
    data: dict[str, Any],
    *,
    idempotent_replay: bool,
) -> dict[str, Any]:
    result = _control_plane_envelope(data)
    result.update({
        "request_id": _control_plane_request_id(request),
        "idempotent_replay": bool(idempotent_replay),
    })
    return result


def _control_plane_write_request_hash(operation: str, organization_id: str, body: Mapping[str, Any]) -> str:
    payload = json.dumps(
        {"operation": operation, "organization_id": organization_id, "body": dict(body)},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _control_plane_write_http_error(error: BaseException) -> HTTPException:
    code = str(error.args[0]) if isinstance(error, KeyError) and error.args else str(error)
    member_errors = {
        "team_conflict": (409, "团队已被其他操作修改，请刷新后重新选择成员。"),
        "team_permission_denied": (403, "当前角色不能管理团队。"),
        "project_conflict": (409, "项目已被其他操作修改，请刷新后重试。"),
        "project_permission_denied": (403, "当前角色不能管理项目。"),
        "project_member_not_found": (422, "项目成员必须是同一组织内已存在主体。"),
        "duplicate_project_member_ids": (422, "项目成员不能重复。"),
        "project_name_required": (422, "项目名称不能为空。"),
        "project_status_invalid": (422, "项目状态无效。"),
        "github_identity_in_use": (409, "此 GitHub 身份已有其他关联，不能直接覆盖。"),
        "github_identity_conflict": (409, "身份关联已变化，请重新查询或刷新。"),
        "github_subject_invalid": (422, "GitHub 用户 ID 必须是有效的稳定数字 ID。"),
        "github_login_invalid": (422, "请输入有效的 GitHub 用户名。"),
        "github_user_not_found": (404, "GitHub 账号不存在或不可公开查询。"),
        "member_permission_denied": (403, "当前角色不能管理此成员或授予该角色。"),
        "member_conflict": (409, "成员已被其他操作修改，请刷新后重试。"),
        "last_owner_required": (409, "组织必须保留至少一名启用的所有者。"),
        "last_linked_owner_required": (409, "组织必须保留至少一名已关联 GitHub 的启用所有者。"),
        "member_has_active_work": (409, "成员有未结束任务或正在处理的请求，请先处理后再停用。"),
    }
    if code in member_errors:
        status, message = member_errors[code]
        return HTTPException(status, detail={"code": code, "message": message})
    if code in {"organization_not_found", "team_not_found", "principal_not_found", "project_not_found"}:
        return HTTPException(status_code=404, detail={"code": code, "message": "Control Plane 资源不存在。"})
    if code in {"team_already_exists", "principal_already_exists", "idempotency_conflict"}:
        return HTTPException(status_code=409, detail={"code": code, "message": "Control Plane 写入冲突。"})
    if code in {"member_not_found", "duplicate_member_ids"}:
        return HTTPException(status_code=422, detail={"code": code, "message": "团队成员必须是同一组织内已存在主体且不可重复。"})
    if code == "team_not_found":
        return HTTPException(status_code=404, detail={"code": code, "message": "团队不存在。"})
    return HTTPException(status_code=422, detail={"code": "control_plane_validation", "message": str(error)})


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
        "reload_supported": True,
        "identity_required": True,
        "website_access_protocol": "2026-10-15-private-user-v1",
    }


@app.get("/api/control-plane/organizations/{organization_id}")
def control_plane_organization(
    organization_id: str,
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    verify_local_request(request, x_control_token)
    store = _control_plane_store()
    organization = store.get_organization(organization_id)
    store.record_audit(
        organization_id=organization_id, resource="organization", outcome="found" if organization else "not_found",
        request_id=_control_plane_request_id(request), actor=request.state.principal["user_id"], metadata={"method": "GET"},
    )
    if organization is None:
        raise HTTPException(status_code=404, detail="组织不存在。")
    return _control_plane_envelope({"organization": organization})


@app.get("/api/control-plane/organizations/{organization_id}/teams")
def control_plane_teams(
    organization_id: str,
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    verify_local_request(request, x_control_token)
    store = _control_plane_store()
    if store.get_organization(organization_id) is None:
        store.record_audit(organization_id=organization_id, resource="teams", outcome="not_found",
                           request_id=_control_plane_request_id(request), actor=request.state.principal["user_id"], metadata={"method": "GET"})
        raise HTTPException(status_code=404, detail="组织不存在。")
    teams = store.list_teams(organization_id)
    store.record_audit(organization_id=organization_id, resource="teams", outcome="ok",
                       request_id=_control_plane_request_id(request), actor=request.state.principal["user_id"], metadata={"method": "GET", "count": len(teams)})
    return _control_plane_envelope({"organization_id": organization_id, "teams": teams})


@app.get("/api/control-plane/organizations/{organization_id}/principals")
def control_plane_principals(
    organization_id: str,
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    verify_local_request(request, x_control_token)
    store = _control_plane_store()
    if store.get_organization(organization_id) is None:
        store.record_audit(organization_id=organization_id, resource="principals", outcome="not_found",
                           request_id=_control_plane_request_id(request), actor=request.state.principal["user_id"], metadata={"method": "GET"})
        raise HTTPException(status_code=404, detail="组织不存在。")
    principals = store.list_principals(organization_id)
    store.record_audit(organization_id=organization_id, resource="principals", outcome="ok",
                       request_id=_control_plane_request_id(request), actor=request.state.principal["user_id"], metadata={"method": "GET", "count": len(principals)})
    role = request.state.principal["role"]
    return _control_plane_envelope({"organization_id": organization_id,
        "create_roles": manageable_member_roles(role),
        "principals": [{**p, "editable_roles": manageable_member_roles(role, p["role"])} for p in principals]})


@app.get("/api/control-plane/organizations/{organization_id}/quotas")
def control_plane_quotas(
    organization_id: str,
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    verify_local_request(request, x_control_token)
    store = _control_plane_store()
    if store.get_organization(organization_id) is None:
        store.record_audit(organization_id=organization_id, resource="quotas", outcome="not_found",
                           request_id=_control_plane_request_id(request), actor=request.state.principal["user_id"], metadata={"method": "GET"})
        raise HTTPException(status_code=404, detail="组织不存在。")
    quotas = store.list_quotas(organization_id)
    store.record_audit(organization_id=organization_id, resource="quotas", outcome="ok",
                       request_id=_control_plane_request_id(request), actor=request.state.principal["user_id"], metadata={"method": "GET", "count": len(quotas)})
    return _control_plane_envelope({"organization_id": organization_id, "quotas": quotas})


@app.get("/api/control-plane/organizations/{organization_id}/audit")
def control_plane_audit(
    organization_id: str,
    request: Request,
    after_sequence: int = 0,
    limit: int = 50,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    verify_local_request(request, x_control_token)
    after_sequence, limit = _validate_control_plane_page(after_sequence, limit)
    store = _control_plane_store()
    if store.get_organization(organization_id) is None:
        raise HTTPException(status_code=404, detail="组织不存在。")
    request_id = _control_plane_request_id(request)
    store.record_audit(organization_id=organization_id, resource="audit", outcome="ok",
                       request_id=request_id, actor=request.state.principal["user_id"], metadata={"method": "GET", "after_sequence": after_sequence, "limit": limit})
    page = store.read_audit(organization_id, after_sequence=after_sequence, limit=limit,
                            actor=request.state.principal["user_id"])
    return _control_plane_envelope({"organization_id": organization_id, **page})


@app.get("/api/control-plane/organizations/{organization_id}/billing")
def control_plane_billing(
    organization_id: str,
    request: Request,
    limit: int = 100,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    verify_local_request(request, x_control_token)
    if isinstance(limit, bool) or limit < 1 or limit > 100:
        raise HTTPException(status_code=422, detail="limit 必须在 1-100。")
    store = _control_plane_store()
    if store.get_organization(organization_id) is None:
        raise HTTPException(status_code=404, detail="组织不存在。")
    entries = store.list_billing(organization_id, limit=limit)
    store.record_audit(organization_id=organization_id, resource="billing", outcome="ok",
                       request_id=_control_plane_request_id(request), actor=request.state.principal["user_id"],
                       metadata={"method": "GET", "count": len(entries)})
    return _control_plane_envelope({"organization_id": organization_id, "entries": entries})


@app.get("/api/control-plane/organizations/{organization_id}/platform/releases")
def control_plane_platform_releases(
    organization_id: str,
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    verify_local_request(request, x_control_token)
    store = _control_plane_store()
    if store.get_organization(organization_id) is None:
        raise HTTPException(status_code=404, detail="组织不存在。")
    releases = store.list_platform_releases()
    return _control_plane_envelope({"organization_id": organization_id, "releases": releases})


@app.post("/api/control-plane/organizations/{organization_id}/platform/rollback")
def control_plane_platform_rollback(
    organization_id: str,
    body: ControlPlanePlatformRollbackRequest,
    request: Request,
    response: Response,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
    x_control_plane_token: str | None = Header(default=None, alias="X-Control-Plane-Token"),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> dict[str, Any]:
    store, principal, key = _control_plane_write_context(
        request, organization_id, "platform.rollback", "identity.manage",
        x_control_token, x_control_plane_token, idempotency_key,
    )
    if principal["role"] != "owner":
        raise HTTPException(status_code=403, detail={"code": "owner_required", "message": "平台版本回滚仅允许组织所有者。"})
    request_hash = _control_plane_write_request_hash("platform.rollback", organization_id, body.model_dump())
    try:
        replay = store.idempotent_result(organization_id=organization_id, principal_user_id=principal["user_id"],
                                         idempotency_key=key, operation="platform.rollback", request_hash=request_hash)
    except ValueError as error:
        raise _control_plane_write_http_error(error) from error
    if replay:
        response.status_code = 200
        return _control_plane_write_envelope(request, replay[1], idempotent_replay=True)
    blockers = _api_reload_blockers()
    active_ids = [str(item.get("session_id") or item.get("task_id") or "") for item in blockers]
    try:
        plan = store.plan_platform_rollback(target_revision=body.target_revision, active_session_ids=active_ids)
    except (KeyError, ValueError) as error:
        raise _control_plane_write_http_error(error) from error
    if plan["active_revision"] != body.expected_active_revision:
        raise HTTPException(status_code=409, detail={"code": "platform_revision_conflict", "message": "当前活动版本已变化，请重新读取发布记录。", "plan": plan})
    if not plan["ready"]:
        store.record_audit(organization_id=organization_id, resource="platform.rollback", outcome="blocked",
                           request_id=_control_plane_request_id(request), actor=principal["user_id"], metadata={"plan": plan})
        return _control_plane_envelope({"plan": plan, "applied": False})
    if body.confirm is not True:
        return _control_plane_envelope({"plan": plan, "applied": False, "requires_confirmation": True})
    loaded_revision = _current_code_revision(ROOT.parent)
    if loaded_revision != body.target_revision:
        raise HTTPException(status_code=409, detail={"code": "rollback_target_not_loaded", "message": "目标版本尚未由受控重载加载；先完成停机检查与重载，再提交回滚确认。", "loaded_revision": loaded_revision, "plan": plan})
    result = store.commit_platform_rollback(target_revision=body.target_revision, reason=body.reason)
    store.record_audit(organization_id=organization_id, resource="platform.rollback", outcome="committed",
                       request_id=_control_plane_request_id(request), actor=principal["user_id"], metadata={"revision": body.target_revision})
    response.status_code = 200
    payload = {"plan": plan, "release": result, "applied": True}
    store.save_idempotent_result(organization_id=organization_id, principal_user_id=principal["user_id"],
                                 idempotency_key=key, operation="platform.rollback", request_hash=request_hash,
                                 status_code=200, response=payload)
    return _control_plane_write_envelope(request, payload, idempotent_replay=False)


@app.get("/api/control-plane/organizations/{organization_id}/notifications")
def control_plane_notifications(
    organization_id: str,
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    verify_local_request(request, x_control_token)
    store = _control_plane_store()
    principal = request.state.principal
    if store.get_organization(organization_id) is None:
        raise HTTPException(status_code=404, detail="组织不存在。")
    notifications = store.list_notifications(organization_id, principal["user_id"])
    store.record_audit(organization_id=organization_id, resource="notifications", outcome="ok",
                       request_id=_control_plane_request_id(request), actor=principal["user_id"],
                       metadata={"method": "GET", "count": len(notifications)})
    return _control_plane_envelope({"organization_id": organization_id, "notifications": notifications})


@app.post("/api/control-plane/organizations/{organization_id}/notifications/{notification_id}/read")
def control_plane_mark_notification_read(
    organization_id: str,
    notification_id: str,
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    verify_local_request(request, x_control_token)
    store = _control_plane_store()
    principal = request.state.principal
    if store.get_organization(organization_id) is None:
        raise HTTPException(status_code=404, detail="组织不存在。")
    if not store.mark_notification_read(organization_id, principal["user_id"], notification_id):
        raise HTTPException(status_code=404, detail="通知不存在或不属于当前账号。")
    return _control_plane_envelope({"organization_id": organization_id, "notification_id": notification_id, "read": True})


@app.post("/api/control-plane/auth/external-sessions")
def control_plane_external_identity_session(
    body: ControlPlaneExternalIdentityRequest,
    request: Request,
) -> dict[str, Any]:
    """Exchange a provider-verified assertion through the explicit provider port."""
    verify_local_origin(request)
    store = _control_plane_store()
    actor = f"external:{body.provider_id}"[:128]
    try:
        identity = control_plane_identity_registry.authenticate(body.provider_id, body.assertion)
    except KeyError as error:
        store.record_audit(organization_id="unknown", resource="identity.external", outcome="provider_unavailable",
                           request_id=_control_plane_request_id(request), metadata={"provider_id": body.provider_id}, actor=actor)
        raise HTTPException(status_code=503, detail={"code": str(error.args[0]), "message": "外部身份 provider 尚未配置。"}) from error
    except ExternalIdentityProviderUnavailable as error:
        store.record_audit(organization_id="unknown", resource="identity.external", outcome="provider_unavailable",
                           request_id=_control_plane_request_id(request), metadata={"provider_id": body.provider_id}, actor=actor)
        raise HTTPException(status_code=503, detail={"code": "external_provider_unavailable", "message": "外部身份 provider 暂时不可用。"}) from error
    except ExternalIdentityMappingNotFound as error:
        store.record_audit(organization_id="unknown", resource="identity.external", outcome="mapping_not_found",
                           request_id=_control_plane_request_id(request), metadata={"provider_id": body.provider_id}, actor=actor)
        raise HTTPException(status_code=404, detail={"code": "external_identity_mapping_not_found", "message": "外部身份尚未完成主体映射。"}) from error
    except (ValueError, TypeError) as error:
        store.record_audit(organization_id="unknown", resource="identity.external", outcome="assertion_rejected",
                           request_id=_control_plane_request_id(request), metadata={"provider_id": body.provider_id}, actor=actor)
        raise HTTPException(status_code=401, detail={"code": "external_identity_rejected", "message": "外部身份断言未通过验证。"}) from error
    try:
        session = store.issue_external_identity_session(identity)
    except KeyError as error:
        if error.args[0] == "external_identity_mapping_changed":
            store.record_audit(organization_id=identity.organization_id, resource="identity.external", outcome="mapping_changed", actor=actor)
            raise HTTPException(401, detail={"code":"external_identity_mapping_changed", "message":"身份关联已变更，请重新登录。"}) from error
        if error.args[0] == "principal_disabled":
            store.record_audit(organization_id=identity.organization_id, resource="identity.external", outcome="principal_disabled",
                actor=actor, metadata={"user_id": identity.user_id})
            raise HTTPException(403, detail={"code": "principal_disabled", "message": "此组织成员已停用，请联系管理员。"}) from error
        store.record_audit(organization_id=identity.organization_id, resource="identity.external", outcome="principal_not_found",
                           request_id=_control_plane_request_id(request), metadata={"provider_id": identity.provider_id, "user_id": identity.user_id}, actor=actor)
        raise HTTPException(status_code=404, detail={"code": "principal_not_found", "message": "外部身份映射的主体不存在。"}) from error
    store.record_audit(organization_id=identity.organization_id, resource="identity.external", outcome="issued",
                       request_id=_control_plane_request_id(request),
                       metadata={"provider_id": identity.provider_id, "subject": identity.subject, "user_id": identity.user_id},
                       actor=actor)
    return _control_plane_envelope({
        "external_identity_protocol": EXTERNAL_IDENTITY_PROTOCOL,
        "session": session,
    })


def _github_browser_provider() -> GitHubOAuthProvider:
    provider = control_plane_identity_registry.get("github")
    if not isinstance(provider, GitHubOAuthProvider):
        raise HTTPException(
            status_code=503,
            detail={"code": "external_provider_not_configured", "message": "GitHub OAuth provider 尚未配置。"},
        )
    return provider


@app.get("/api/control-plane/auth/github/start")
def control_plane_github_oauth_start(
    request: Request,
    response: Response,
    browser: bool = False,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    """Start GitHub's browser authorization-code flow with state and PKCE."""
    verify_local_request(request, x_control_token)
    provider = _github_browser_provider()
    store = _control_plane_store()
    now = time.time()
    for key, (_, expires_at, _) in list(_github_oauth_pending_verifiers.items()):
        if expires_at <= now:
            _github_oauth_pending_verifiers.pop(key, None)
    state = secrets.token_urlsafe(32)
    code_verifier = secrets.token_urlsafe(64)
    code_challenge = provider.pkce_challenge(code_verifier)
    state_hash = hashlib.sha256(state.encode("utf-8")).hexdigest()
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=_GITHUB_OAUTH_STATE_TTL_SECONDS)
    expires_text = expires_at.isoformat(timespec="milliseconds")
    store.create_oauth_state(
        state_hash=state_hash,
        provider_id=provider.provider_id,
        redirect_uri=provider.redirect_uri,
        expires_at=expires_text,
    )
    binding = secrets.token_urlsafe(32) if browser else ""
    _github_oauth_pending_verifiers[state_hash] = (
        code_verifier, expires_at.timestamp(),
        hashlib.sha256(binding.encode()).hexdigest() if binding else "",
    )
    if browser:
        response.set_cookie(_OAUTH_BROWSER_COOKIE, binding, max_age=_GITHUB_OAUTH_STATE_TTL_SECONDS,
                            path=_AUTH_COOKIE_PATH, httponly=True, samesite="lax",
                            secure=request.url.scheme == "https")
    authorization_url = provider.authorization_url(state=state, code_challenge=code_challenge)
    store.record_audit(
        organization_id="unknown",
        resource="identity.external.oauth",
        outcome="authorization_started",
        request_id=_control_plane_request_id(request),
        metadata={"provider_id": provider.provider_id, "redirect_uri": provider.redirect_uri},
        actor="local-control-token",
    )
    return _control_plane_envelope({
        "github_oauth_browser_protocol": GITHUB_OAUTH_BROWSER_PROTOCOL,
        "oauth_flow": "authorization_code",
        "provider_id": provider.provider_id,
        "authorization_url": authorization_url,
        "redirect_uri": provider.redirect_uri,
        "expires_at": expires_text,
        "state_ttl_seconds": _GITHUB_OAUTH_STATE_TTL_SECONDS,
    })


def _complete_github_oauth_callback(
    request: Request,
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
    error_description: str | None = None,
) -> dict[str, Any]:
    """Consume GitHub's callback, exchange the code, then issue a local session."""
    provider = _github_browser_provider()
    store = _control_plane_store()
    state_value = str(state or "").strip()
    state_hash = hashlib.sha256(state_value.encode("utf-8")).hexdigest() if state_value else ""
    pending = _github_oauth_pending_verifiers.get(state_hash)
    if pending and pending[2]:
        binding = request.cookies.get(_OAUTH_BROWSER_COOKIE, "")
        if not binding or not secrets.compare_digest(pending[2], hashlib.sha256(binding.encode()).hexdigest()):
            store.record_audit(organization_id="unknown", resource="identity.external.oauth",
                               outcome="browser_binding_rejected", metadata={"provider_id": "github"})
            raise HTTPException(status_code=400, detail={"code": "oauth_browser_mismatch", "message": "请在发起登录的浏览器中完成授权。"})
    record = store.consume_oauth_state(
        state_hash=state_hash,
        provider_id=provider.provider_id,
    ) if state_hash else None
    pending = _github_oauth_pending_verifiers.pop(state_hash, None) if state_hash else None
    if record is None or pending is None:
        store.record_audit(
            organization_id="unknown",
            resource="identity.external.oauth",
            outcome="state_rejected",
            request_id=_control_plane_request_id(request),
            metadata={"provider_id": provider.provider_id},
            actor="external:github",
        )
        raise HTTPException(status_code=400, detail={"code": "oauth_state_invalid", "message": "OAuth state 无效、已过期或已使用。"})
    code_verifier, expires_at, _ = pending
    if expires_at <= time.time() or record["redirect_uri"] != provider.redirect_uri:
        raise HTTPException(status_code=400, detail={"code": "oauth_state_invalid", "message": "OAuth state 无效或已过期。"})
    if error:
        store.record_audit(
            organization_id="unknown",
            resource="identity.external.oauth",
            outcome="authorization_denied",
            request_id=_control_plane_request_id(request),
            metadata={"provider_id": provider.provider_id, "error": str(error)[:64]},
            actor="external:github",
        )
        raise HTTPException(status_code=400, detail={"code": "oauth_authorization_denied", "message": "GitHub OAuth 授权未完成。"})
    code_value = str(code or "").strip()
    if not code_value:
        raise HTTPException(status_code=400, detail={"code": "oauth_code_missing", "message": "GitHub OAuth 回调缺少授权码。"})
    try:
        access_token = provider.exchange_code(code=code_value, code_verifier=code_verifier)
        identity = control_plane_identity_registry.authenticate(
            provider.provider_id, {"access_token": access_token}
        )
    except ExternalIdentityProviderUnavailable as exc:
        store.record_audit(organization_id="unknown", resource="identity.external.oauth", outcome="provider_unavailable",
                           request_id=_control_plane_request_id(request), metadata={"provider_id": provider.provider_id}, actor="external:github")
        raise HTTPException(status_code=503, detail={"code": "external_provider_unavailable", "message": "GitHub OAuth provider 暂时不可用。"}) from exc
    except ExternalIdentityMappingNotFound as exc:
        store.record_audit(organization_id="unknown", resource="identity.external.oauth", outcome="mapping_not_found",
                           request_id=_control_plane_request_id(request), metadata={"provider_id": provider.provider_id}, actor="external:github")
        raise HTTPException(status_code=404, detail={"code": "external_identity_mapping_not_found", "message": "GitHub 身份尚未完成主体映射。"}) from exc
    except (ValueError, TypeError) as exc:
        store.record_audit(organization_id="unknown", resource="identity.external.oauth", outcome="assertion_rejected",
                           request_id=_control_plane_request_id(request), metadata={"provider_id": provider.provider_id}, actor="external:github")
        raise HTTPException(status_code=401, detail={"code": "external_identity_rejected", "message": "GitHub OAuth 身份未通过验证。"}) from exc
    try:
        session = store.issue_external_identity_session(identity)
    except KeyError as exc:
        if exc.args[0] == "external_identity_mapping_changed":
            store.record_audit(organization_id=identity.organization_id, resource="identity.external.oauth", outcome="mapping_changed", actor="external:github")
            raise HTTPException(401, detail={"code":"external_identity_mapping_changed", "message":"身份关联已变更，请重新登录。"}) from exc
        if exc.args[0] == "principal_disabled":
            store.record_audit(organization_id=identity.organization_id, resource="identity.external.oauth", outcome="principal_disabled",
                actor="external:github", metadata={"user_id": identity.user_id})
            raise HTTPException(403, detail={"code": "principal_disabled", "message": "此组织成员已停用，请联系管理员。"}) from exc
        store.record_audit(organization_id=identity.organization_id, resource="identity.external.oauth", outcome="principal_not_found",
                           request_id=_control_plane_request_id(request), metadata={"provider_id": identity.provider_id, "user_id": identity.user_id}, actor="external:github")
        raise HTTPException(status_code=404, detail={"code": "principal_not_found", "message": "GitHub 身份映射的主体不存在。"}) from exc
    store.record_audit(
        organization_id=identity.organization_id,
        resource="identity.external.oauth",
        outcome="issued",
        request_id=_control_plane_request_id(request),
        metadata={"provider_id": identity.provider_id, "subject": identity.subject, "user_id": identity.user_id},
        actor="external:github",
    )
    return _control_plane_envelope({
        "github_oauth_browser_protocol": GITHUB_OAUTH_BROWSER_PROTOCOL,
        "oauth_flow": "authorization_code",
        "external_identity_protocol": EXTERNAL_IDENTITY_PROTOCOL,
        "session": session,
    })


@app.get("/api/control-plane/auth/github/callback", response_model=None)
def control_plane_github_oauth_callback(
    request: Request, code: str | None = None, state: str | None = None,
    error: str | None = None, error_description: str | None = None,
) -> Response | dict[str, Any]:
    state_hash = hashlib.sha256(str(state or "").strip().encode()).hexdigest()
    pending = _github_oauth_pending_verifiers.get(state_hash)
    browser = bool((pending and pending[2]) or request.cookies.get(_OAUTH_BROWSER_COOKIE))
    try:
        result = _complete_github_oauth_callback(request, code, state, error, error_description)
    except HTTPException as exc:
        if not browser:
            raise
        error_code = exc.detail.get("code", "login_failed") if isinstance(exc.detail, dict) else "login_failed"
        response = RedirectResponse("/?login_error=" + error_code, status_code=303)
        response.delete_cookie(_OAUTH_BROWSER_COOKIE, path=_AUTH_COOKIE_PATH)
        response.delete_cookie("agent_principal", path=_AUTH_COOKIE_PATH)
        return response
    if not browser:
        return result
    # Reuse the same durable principal session; the website never sees its token.
    session = result["data"]["session"]
    store = _control_plane_store()
    previous = store.authenticate_identity_session(request.cookies.get(_BROWSER_SESSION_COOKIE, ""))
    if previous:
        store.revoke_identity_session(previous["session_id"])
        store.record_audit(organization_id=previous["organization_id"], resource="identity.session.revoke",
                           outcome="replaced", metadata={"session_id": previous["session_id"]}, actor=previous["user_id"])
    response = RedirectResponse("/?login=success", status_code=303)
    ttl = max(0, int((datetime.fromisoformat(session["expires_at"]) - datetime.now(timezone.utc)).total_seconds()))
    response.set_cookie(_BROWSER_SESSION_COOKIE, session["principal_token"], max_age=ttl,
                        path="/api", httponly=True, samesite="lax", secure=request.url.scheme == "https")
    response.delete_cookie(_OAUTH_BROWSER_COOKIE, path=_AUTH_COOKIE_PATH)
    response.delete_cookie("agent_principal", path=_AUTH_COOKIE_PATH)
    return response


@app.get("/api/control-plane/auth/browser-session")
def control_plane_browser_session(request: Request, response: Response,
                                  x_control_token: str | None = Header(default=None, alias="X-Control-Token")) -> dict[str, Any]:
    verify_local_request(request, x_control_token)
    token = request.cookies.get(_BROWSER_SESSION_COOKIE, "")
    principal = _control_plane_store().authenticate_identity_session(token)
    if token and not principal:
        response.delete_cookie(_BROWSER_SESSION_COOKIE, path="/api")
    return _control_plane_envelope({
        "authenticated": principal is not None,
        "github_available": isinstance(control_plane_identity_registry.get("github"), GitHubOAuthProvider),
        "session": ({key: principal[key] for key in ("user_id", "organization_id", "role", "expires_at", "permissions")}
                    if principal else None),
    })


@app.post("/api/control-plane/auth/browser-logout")
def control_plane_browser_logout(request: Request, response: Response,
                                 x_control_token: str | None = Header(default=None, alias="X-Control-Token")) -> dict[str, Any]:
    verify_local_request(request, x_control_token)
    token = request.cookies.get(_BROWSER_SESSION_COOKIE, "")
    principal = _control_plane_store().authenticate_identity_session(token)
    if principal:
        control_plane_revoke_identity_session(
            principal["session_id"], request, response, x_control_token, token,
            "browser-logout-" + principal["session_id"],
        )
    response.delete_cookie(_BROWSER_SESSION_COOKIE, path="/api")
    response.delete_cookie(_OAUTH_BROWSER_COOKIE, path=_AUTH_COOKIE_PATH)
    response.delete_cookie("agent_principal", path=_AUTH_COOKIE_PATH)
    return _control_plane_envelope({"authenticated": False})


@app.delete("/api/control-plane/auth/sessions/{session_id}")
def control_plane_revoke_identity_session(
    session_id: str,
    request: Request,
    response: Response,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
    x_control_plane_token: str | None = Header(default=None, alias="X-Control-Plane-Token"),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> dict[str, Any]:
    verify_local_request(request, x_control_token)
    store = _control_plane_store()
    request_id = _control_plane_request_id(request)
    x_control_plane_token = x_control_plane_token or website_access.identity_token(request)
    if not x_control_plane_token:
        store.record_audit(organization_id="unknown", resource="identity.session.revoke", outcome="identity_required",
                           request_id=request_id, metadata={"session_id": session_id}, actor="anonymous")
        raise HTTPException(status_code=401, detail={"code": "identity_required", "message": "需要 Control Plane 主体会话。"})
    principal = store.authenticate_identity_session(x_control_plane_token)
    if principal is None:
        store.record_audit(organization_id="unknown", resource="identity.session.revoke", outcome="identity_invalid",
                           request_id=request_id, metadata={"session_id": session_id}, actor="anonymous")
        raise HTTPException(status_code=401, detail={"code": "identity_invalid", "message": "Control Plane 主体会话无效或已过期。"})
    target = store.get_identity_session(session_id)
    if target is None:
        raise HTTPException(status_code=404, detail={"code": "identity_session_not_found", "message": "主体会话不存在。"})
    if target["organization_id"] != principal["organization_id"]:
        store.record_audit(organization_id=target["organization_id"], resource="identity.session.revoke", outcome="organization_forbidden",
                           request_id=request_id, metadata={"session_id": session_id, "user_id": principal["user_id"]}, actor=principal["user_id"])
        raise HTTPException(status_code=403, detail={"code": "organization_forbidden", "message": "主体不属于目标会话组织。"})
    is_self = target["user_id"] == principal["user_id"] and target["session_id"] == principal["session_id"]
    if not is_self and "identity.manage" not in principal["permissions"]:
        store.record_audit(organization_id=target["organization_id"], resource="identity.session.revoke", outcome="permission_denied",
                           request_id=request_id, metadata={"permission": "identity.manage", "session_id": session_id, "user_id": principal["user_id"]}, actor=principal["user_id"])
        raise HTTPException(status_code=403, detail={"code": "permission_denied", "message": "撤销他人会话需要 identity.manage。"})
    key = str(idempotency_key or "").strip()
    if not key or len(key) > 128:
        raise HTTPException(status_code=422, detail={"code": "idempotency_required", "message": "会话撤销必须提供 Idempotency-Key。"})
    store.record_audit(organization_id=target["organization_id"], resource="identity.session.revoke", outcome="permission_allowed",
                       request_id=request_id, metadata={"session_id": session_id, "user_id": principal["user_id"], "self": is_self}, actor=principal["user_id"])
    try:
        status_code, data, replay = store.revoke_identity_session_idempotent(
            organization_id=target["organization_id"], principal_user_id=principal["user_id"],
            idempotency_key=key,
            request_hash=_control_plane_write_request_hash("identity.session.revoke", target["organization_id"], {"session_id": session_id}),
            session_id=session_id,
        )
    except (KeyError, ValueError) as error:
        raise _control_plane_write_http_error(error) from error
    store.record_audit(organization_id=target["organization_id"], resource="identity.session.revoke",
                       outcome="replay" if replay else "revoked", request_id=request_id,
                       metadata={"session_id": session_id, "status_code": status_code}, actor=principal["user_id"])
    response.status_code = 200
    return _control_plane_write_envelope(request, data, idempotent_replay=replay)


@app.get("/api/control-plane/organizations/{organization_id}/projects")
def control_plane_projects(organization_id: str, request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token")) -> dict[str, Any]:
    verify_local_request(request, x_control_token)
    store = _control_plane_store()
    if store.get_organization(organization_id) is None:
        raise HTTPException(404, detail={"code": "organization_not_found", "message": "组织不存在。"})
    projects = store.list_projects(organization_id)
    store.record_audit(organization_id=organization_id, resource='projects', outcome='ok', actor=request.state.principal['user_id'],
                       request_id=_control_plane_request_id(request), metadata={'count': len(projects)})
    return _control_plane_envelope({'organization_id': organization_id, 'projects': projects})


@app.post("/api/control-plane/organizations/{organization_id}/projects")
def control_plane_create_project(organization_id: str, body: ControlPlaneProjectWriteRequest, request: Request, response: Response,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
    x_control_plane_token: str | None = Header(default=None, alias="X-Control-Plane-Token"),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")) -> dict[str, Any]:
    store, principal, key = _control_plane_write_context(request, organization_id, 'project.create', 'team.manage', x_control_token, x_control_plane_token, idempotency_key)
    try:
        status_code, data, replay = store.create_project(organization_id=organization_id, principal_user_id=principal['user_id'],
            idempotency_key=key, request_hash=_control_plane_write_request_hash('project.create', organization_id, body.model_dump()),
            project_id=body.project_id, name=body.name, description=body.description, member_ids=tuple(body.member_ids))
    except (KeyError, ValueError) as error:
        raise _control_plane_write_http_error(error) from error
    if replay:
        store.record_audit(organization_id=organization_id, resource='project.create', outcome='replay', actor=principal['user_id'],
                           request_id=_control_plane_request_id(request), metadata={'project_id': body.project_id})
    response.status_code = 200 if replay else status_code
    return _control_plane_write_envelope(request, data, idempotent_replay=replay)


@app.patch("/api/control-plane/organizations/{organization_id}/projects/{project_id}")
def control_plane_update_project(organization_id: str, project_id: str, body: ControlPlaneProjectUpdateRequest,
    request: Request, response: Response,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
    x_control_plane_token: str | None = Header(default=None, alias="X-Control-Plane-Token"),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")) -> dict[str, Any]:
    store, principal, key = _control_plane_write_context(request, organization_id, 'project.update', 'team.manage', x_control_token, x_control_plane_token, idempotency_key)
    if all(getattr(body, name) is None for name in ('name', 'description', 'status', 'member_ids')):
        raise HTTPException(422, detail={'code': 'empty_update', 'message': '至少提供一个项目字段。'})
    try:
        status_code, data, replay = store.update_project(organization_id=organization_id, principal_user_id=principal['user_id'],
            idempotency_key=key, request_hash=_control_plane_write_request_hash('project.update', organization_id, {'project_id': project_id, **body.model_dump()}),
            project_id=project_id, name=body.name, description=body.description, status=body.status,
            member_ids=tuple(body.member_ids) if body.member_ids is not None else None, expected_revision=body.expected_revision)
    except (KeyError, ValueError) as error:
        raise _control_plane_write_http_error(error) from error
    if replay:
        store.record_audit(organization_id=organization_id, resource='project.update', outcome='replay', actor=principal['user_id'],
                           request_id=_control_plane_request_id(request), metadata={'project_id': project_id})
    response.status_code = 200 if replay else status_code
    return _control_plane_write_envelope(request, data, idempotent_replay=replay)


@app.post("/api/control-plane/organizations/{organization_id}/teams")
def control_plane_create_team(
    organization_id: str,
    body: ControlPlaneTeamWriteRequest,
    request: Request,
    response: Response,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
    x_control_plane_token: str | None = Header(default=None, alias="X-Control-Plane-Token"),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> dict[str, Any]:
    store, principal, key = _control_plane_write_context(
        request, organization_id, "team.create", "team.manage", x_control_token, x_control_plane_token, idempotency_key,
    )
    try:
        status_code, data, replay = store.create_team(
            organization_id=organization_id,
            principal_user_id=principal["user_id"],
            idempotency_key=key,
            request_hash=_control_plane_write_request_hash("team.create", organization_id, body.model_dump()),
            team_id=body.team_id,
            name=body.name,
            member_ids=tuple(body.member_ids),
        )
    except (KeyError, ValueError) as error:
        raise _control_plane_write_http_error(error) from error
    if replay:
        store.record_audit(organization_id=organization_id, resource="team.create", outcome="replay",
                           actor=principal['user_id'], request_id=_control_plane_request_id(request), metadata={"team_id": body.team_id})
    response.status_code = 200 if replay else status_code
    return _control_plane_write_envelope(request, data, idempotent_replay=replay)


@app.patch("/api/control-plane/organizations/{organization_id}/teams/{team_id}")
def control_plane_update_team(
    organization_id: str,
    team_id: str,
    body: ControlPlaneTeamUpdateRequest,
    request: Request,
    response: Response,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
    x_control_plane_token: str | None = Header(default=None, alias="X-Control-Plane-Token"),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> dict[str, Any]:
    store, principal, key = _control_plane_write_context(
        request, organization_id, "team.update", "team.manage", x_control_token, x_control_plane_token, idempotency_key,
    )
    if body.name is None and body.member_ids is None:
        raise HTTPException(status_code=422, detail={"code": "empty_update", "message": "至少提供一个团队字段。"})
    try:
        status_code, data, replay = store.update_team(
            organization_id=organization_id,
            principal_user_id=principal["user_id"],
            idempotency_key=key,
            request_hash=_control_plane_write_request_hash("team.update", organization_id, {"team_id": team_id, **body.model_dump()}),
            team_id=team_id,
            name=body.name,
            member_ids=tuple(body.member_ids) if body.member_ids is not None else None,
            expected_revision=body.expected_revision,
        )
    except (KeyError, ValueError) as error:
        raise _control_plane_write_http_error(error) from error
    if replay:
        store.record_audit(organization_id=organization_id, resource="team.update", outcome="replay",
                           actor=principal['user_id'], request_id=_control_plane_request_id(request), metadata={"team_id": team_id})
    response.status_code = 200 if replay else status_code
    return _control_plane_write_envelope(request, data, idempotent_replay=replay)


@app.post("/api/control-plane/organizations/{organization_id}/principals")
def control_plane_create_principal(
    organization_id: str,
    body: ControlPlanePrincipalWriteRequest,
    request: Request,
    response: Response,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
    x_control_plane_token: str | None = Header(default=None, alias="X-Control-Plane-Token"),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> dict[str, Any]:
    store, principal, key = _control_plane_write_context(
        request, organization_id, "principal.create", "identity.manage", x_control_token, x_control_plane_token, idempotency_key,
    )
    try:
        status_code, data, replay = store.create_principal(
            organization_id=organization_id,
            principal_user_id=principal["user_id"],
            idempotency_key=key,
            request_hash=_control_plane_write_request_hash("principal.create", organization_id, body.model_dump()),
            user_id=body.user_id,
            role=body.role,
            team_ids=tuple(body.team_ids),
        )
    except (KeyError, ValueError) as error:
        store.record_audit(organization_id=organization_id, resource="principal.create", outcome=str(error.args[0]),
            actor=principal["user_id"], request_id=_control_plane_request_id(request), metadata={"user_id": body.user_id})
        raise _control_plane_write_http_error(error) from error
    if replay:
        store.record_audit(organization_id=organization_id, resource="principal.create", outcome="replay",
            request_id=_control_plane_request_id(request), actor=principal["user_id"], metadata={"user_id": body.user_id})
    response.status_code = 200 if replay else status_code
    return _control_plane_write_envelope(request, data, idempotent_replay=replay)


def _member_unavailable(store, organization_id, user_id, actor_id):
    # Own update counts once; other in-flight writes must settle before disabling.
    if website_access.inflight_member_writes[(organization_id, user_id)] > int(user_id == actor_id):
        return "member_inflight"
    principal = {"organization_id": organization_id, "user_id": user_id}
    for item in _active_session_status(runtime):
        if store.owned_resource("session", item["session_id"], principal):
            return "member_has_active_work"
    for item in store.device_assignments(organization_id):
        if item["user_id"] == user_id and _device_assignment_unavailable(item["device_id"]):
            return "member_has_active_work"
    return ""


@app.put("/api/control-plane/organizations/{organization_id}/principals/{user_id}")
def control_plane_update_principal(organization_id: str, user_id: str, body: ControlPlanePrincipalUpdateRequest,
    request: Request, x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
    x_control_plane_token: str | None = Header(default=None, alias="X-Control-Plane-Token"),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
    store, principal, key = _control_plane_write_context(request, organization_id, "principal.update", "identity.manage",
        x_control_token, x_control_plane_token, idempotency_key)
    try:
        with website_access.assignment_lock:
            reason = _member_unavailable(store, organization_id, user_id, principal["user_id"]) if body.status == "disabled" else ""
            _, data, replay = store.update_principal(organization_id=organization_id, principal_user_id=principal["user_id"],
                user_id=user_id, role=body.role, status=body.status, expected_revision=body.expected_revision,
                idempotency_key=key, unavailable_reason=reason)
    except (KeyError, ValueError) as error:
        store.record_audit(organization_id=organization_id, resource="principal.update", outcome=str(error.args[0]),
            actor=principal["user_id"], request_id=_control_plane_request_id(request), metadata={"user_id": user_id})
        raise _control_plane_write_http_error(error) from error
    return _control_plane_write_envelope(request, data, idempotent_replay=replay)


@app.get("/api/control-plane/organizations/{organization_id}/github-identities")
def control_plane_github_identities(organization_id: str, request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token")):
    verify_local_request(request, x_control_token)
    store = _control_plane_store()
    identities = store.list_github_identities(organization_id)
    role = request.state.principal["role"]
    store.record_audit(organization_id=organization_id, resource="github_identity.list", outcome="ok",
        actor=request.state.principal["user_id"], metadata={"count":len(identities)})
    return _control_plane_envelope({"identities":[{**i, "manageable":bool(manageable_member_roles(role, i["role"]))} for i in identities],
        "github_available": isinstance(control_plane_identity_registry.get("github"), GitHubOAuthProvider)})


@app.get("/api/control-plane/organizations/{organization_id}/github-identities/lookup")
def control_plane_github_lookup(organization_id: str, request: Request, login: str | None = None, subject: str | None = None,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token")):
    verify_local_request(request, x_control_token)
    if (login is None) == (subject is None):
        raise HTTPException(422, detail={"code":"github_lookup_input", "message":"请提供用户名或稳定用户 ID 中的一项。"})
    try:
        profile = _github_browser_provider().public_user(login=login, subject=subject)
    except ExternalIdentityProviderUnavailable as error:
        raise HTTPException(503, detail={"code":"github_lookup_unavailable", "message":str(error)}) from error
    except (ValueError, KeyError) as error:
        raise _control_plane_write_http_error(error) from error
    binding = _control_plane_store().github_identity(profile["subject"])
    foreign = bool(binding and binding["user_id"] and binding["organization_id"] != organization_id)
    return _control_plane_envelope({"profile":profile, "linked":bool(binding and binding["user_id"]),
        "user_id":binding["user_id"] if binding and not foreign else None,
        "revision":binding["revision"] if binding and not foreign else 0,
        "external_binding":foreign})


@app.put("/api/control-plane/organizations/{organization_id}/github-identities/{subject}")
def control_plane_update_github_identity(organization_id: str, subject: str, body: GitHubIdentityWriteRequest,
    request: Request, x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
    x_control_plane_token: str | None = Header(default=None, alias="X-Control-Plane-Token"),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
    store, principal, key = _control_plane_write_context(request, organization_id, "github_identity.update", "identity.manage",
        x_control_token, x_control_plane_token, idempotency_key)
    try:
        replay = store.idempotent_result(organization_id=organization_id, principal_user_id=principal["user_id"],
            idempotency_key=key, operation="github_identity.update", request_hash=store.request_hash({"subject":subject, **body.model_dump()}))
        if replay:
            return _control_plane_write_envelope(request, replay[1], idempotent_replay=True)
        # Verify account facts outside SQLite's write transaction. Identity and
        # target-role authority are rechecked atomically by the repository.
        profile = _github_browser_provider().public_user(subject=subject) if body.user_id is not None else {}
        with website_access.assignment_lock:
            store, principal, key = _control_plane_write_context(request, organization_id, "github_identity.update", "identity.manage",
                x_control_token, x_control_plane_token, idempotency_key)
            _, data, repeated = store.update_github_identity(organization_id=organization_id, principal_user_id=principal["user_id"],
                subject=subject, user_id=body.user_id, expected_revision=body.expected_revision, idempotency_key=key,
                login=profile.get("login", ""))
    except ExternalIdentityProviderUnavailable as error:
        raise HTTPException(503, detail={"code":"github_lookup_unavailable", "message":str(error)}) from error
    except (ValueError, KeyError) as error:
        store.record_audit(organization_id=organization_id, resource="github_identity.update", outcome=str(error.args[0]),
            actor=principal["user_id"], metadata={"subject":subject})
        raise _control_plane_write_http_error(error) from error
    return _control_plane_write_envelope(request, data, idempotent_replay=repeated)


@app.put("/api/control-plane/organizations/{organization_id}/quotas/{quota_key}")
def control_plane_update_quota(
    organization_id: str,
    quota_key: str,
    body: ControlPlaneQuotaWriteRequest,
    request: Request,
    response: Response,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
    x_control_plane_token: str | None = Header(default=None, alias="X-Control-Plane-Token"),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> dict[str, Any]:
    store, principal, key = _control_plane_write_context(
        request, organization_id, "quota.update", "quota.manage", x_control_token, x_control_plane_token, idempotency_key,
    )
    if body.quota_key != quota_key:
        raise HTTPException(status_code=422, detail={"code": "quota_key_mismatch", "message": "路径 quota_key 与请求体不一致。"})
    if body.max_runs is None and body.max_physical_actions is None and body.max_model_tokens is None:
        raise HTTPException(status_code=422, detail={"code": "quota_limit_required", "message": "至少提供一个配额上限。"})
    try:
        status_code, data, replay = store.upsert_quota(
            organization_id=organization_id,
            principal_user_id=principal["user_id"],
            idempotency_key=key,
            request_hash=_control_plane_write_request_hash("quota.update", organization_id, body.model_dump()),
            quota_key=body.quota_key,
            period=body.period,
            max_runs=body.max_runs,
            max_physical_actions=body.max_physical_actions,
            max_model_tokens=body.max_model_tokens,
        )
    except (KeyError, ValueError) as error:
        raise _control_plane_write_http_error(error) from error
    store.record_audit(organization_id=organization_id, resource="quota.update",
                       outcome="replay" if replay else "updated", request_id=_control_plane_request_id(request),
                       metadata={"quota_key": quota_key, "status_code": status_code})
    response.status_code = 200 if replay else status_code
    return _control_plane_write_envelope(request, data, idempotent_replay=replay)


@app.post("/api/session/reload")
def reload_api(
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    """Reload this checkout's API after a browser refresh settles old work."""

    verify_local_request(request, x_control_token)
    blockers = _api_reload_blockers()
    if blockers:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "active_tasks",
                "error": "仍有活动任务，API 未重载。请先停止任务后再刷新。",
                "blockers": [{"kind": item.get("kind", "task")} for item in blockers],
            },
        )
    if not _schedule_api_reload():
        return {"reloading": True, "already_scheduled": True}
    return {
        "reloading": True,
        "already_scheduled": False,
        "revision": _current_code_revision(ROOT.parent),
    }


def _device_assignment_unavailable(device_id: str) -> str:
    if website_access.inflight_devices[device_id]:
        return "device_inflight"
    if _GENERIC_START_TASKS.active_for("device_id", device_id):
        return "device_has_active_work"
    if runtime.device_task_registry.active_session(device_id):
        return "device_has_active_work"
    if any(item["device_id"] == device_id for item in _active_session_status(runtime)):
        return "device_has_active_work"
    terminal = {"cancelled", "completed", "succeeded", "failed", "blocked", "expired", "stopped"}
    for monitor in (*_LUCKY_BAG_MONITOR.list(), *_QISHUI_AD_TEST_MONITOR.list()):
        if monitor.profile.device_id == device_id and monitor.status not in terminal:
            return "device_has_active_work"
    for trial in runtime.capability_acceptance_manager.snapshots():
        if trial["device_id"] == device_id and trial["session"].get("status") not in terminal:
            return "device_has_active_work"
    if any(item["profile"]["device_id"] == device_id and item["status"] in ManualMotionMonitor.ACTIVE
           for item in _MANUAL_MOTION_MONITOR.list()):
        return "device_has_active_work"
    if runtime.device_runtime_resources.coordination_lock(device_id).locked():
        return "device_inflight"
    if runtime.controller_for_device(device_id).device_status().get("busy"):
        return "device_inflight"
    return ""


@app.get("/api/control-plane/organizations/{organization_id}/devices")
def control_plane_devices(organization_id: str, request: Request,
                         x_control_token: str | None = Header(default=None, alias="X-Control-Token")):
    verify_local_request(request, x_control_token)
    # Inventory metadata only: never include camera, task or model content.
    known = {item["device_id"]: item for item in runtime.device_controllers.descriptors()}
    devices = [{**item, "registered": item["device_id"] in known}
               for item in _control_plane_store().device_assignments(organization_id)]
    return _control_plane_envelope({"devices": devices})


@app.put("/api/control-plane/organizations/{organization_id}/devices/{device_id}/assignment")
def control_plane_assign_device(organization_id: str, device_id: str, body: DeviceAssignmentRequest,
    request: Request, x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
    x_control_plane_token: str | None = Header(default=None, alias="X-Control-Plane-Token"),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
    store, principal, key = _control_plane_write_context(request, organization_id, "device.assign", "device.manage",
        x_control_token, x_control_plane_token, idempotency_key)
    try:
        with website_access.assignment_lock:
            own_inventory = {x["device_id"] for x in store.device_assignments(organization_id)}
            registered = {x["device_id"] for x in runtime.device_controllers.descriptors()}
            if device_id not in own_inventory or device_id not in registered:
                raise KeyError("device_not_found")
            _status, data, replay = store.assign_device(organization_id=organization_id,
                principal_user_id=principal["user_id"], device_id=device_id, user_id=body.user_id,
                expected_revision=body.expected_revision, idempotency_key=key,
                unavailable_reason=_device_assignment_unavailable(device_id))
    except (KeyError, ValueError) as error:
        code = str(error.args[0])
        messages = {"device_not_found": (404, "设备未登记在当前组织。"),
                    "assignee_not_found": (422, "请选择当前组织中已存在的成员。"),
                    "assignment_conflict": (409, "设备分配已变化，请刷新后重试。"),
                    "device_has_active_work": (409, "设备有未结束任务，请先结束或取消任务后再分配。"),
                    "device_inflight": (409, "设备请求正在处理，请稍后重试。"),
                    "idempotency_conflict": (409, "此请求标识已用于其他操作，请刷新后重试。")}
        if code not in messages:
            raise
        status, message = messages[code]
        store.record_audit(organization_id=organization_id, resource="device.assign", outcome=code,
            request_id=_control_plane_request_id(request), actor=principal["user_id"], metadata={"device_id": device_id})
        raise HTTPException(status, detail={"code": code, "message": message}) from error
    return _control_plane_write_envelope(request, data, idempotent_replay=replay)


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


def _connector_store():
    store = getattr(runtime, "connector_store", None)
    if store is None or not store.configured:
        raise HTTPException(status_code=503, detail="本机设备连接器尚未配置。")
    return store


def _connector_device_allowed(connector_id: str, device_id: str) -> None:
    descriptors = getattr(getattr(runtime, "device_controllers", None), "descriptors", lambda: [])()
    match = next((item for item in descriptors if item.get("device_id") == device_id), None)
    if not match or match.get("connector_id") != connector_id:
        raise HTTPException(status_code=403, detail="连接器未获准操作该设备。")


@app.get("/api/device-connector/v1/poll")
async def device_connector_poll(
    request: Request,
    device_id: str,
    x_connector_id: str | None = Header(default=None, alias="X-Connector-Id"),
    x_connector_timestamp: str | None = Header(default=None, alias="X-Connector-Timestamp"),
    x_connector_signature: str | None = Header(default=None, alias="X-Connector-Signature"),
    x_connector_nonce: str | None = Header(default=None, alias="X-Connector-Nonce"),
) -> Response:
    store = _connector_store()
    connector_id = str(x_connector_id or "").strip()
    _connector_device_allowed(connector_id, device_id)
    try:
        store.authenticate(connector_id=connector_id, method="GET",
                           target=request.url.path + ("?" + request.url.query if request.url.query else ""),
                           timestamp=x_connector_timestamp or "", nonce=x_connector_nonce or "",
                           body=b"", signature=x_connector_signature or "")
    except ConnectorProtocolError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    command = store.claim(connector_id=connector_id, device_id=device_id)
    if command is None:
        return Response(status_code=204)
    return Response(content=json.dumps(command, ensure_ascii=False, separators=(",", ":")),
                    media_type="application/json")


@app.post("/api/device-connector/v1/result")
async def device_connector_result(
    request: Request,
    x_connector_id: str | None = Header(default=None, alias="X-Connector-Id"),
    x_connector_timestamp: str | None = Header(default=None, alias="X-Connector-Timestamp"),
    x_connector_signature: str | None = Header(default=None, alias="X-Connector-Signature"),
    x_connector_nonce: str | None = Header(default=None, alias="X-Connector-Nonce"),
) -> dict[str, Any]:
    store = _connector_store()
    body = await request.body()
    try:
        store.authenticate(connector_id=x_connector_id or "", method="POST",
                           target=request.url.path + ("?" + request.url.query if request.url.query else ""),
                           timestamp=x_connector_timestamp or "", nonce=x_connector_nonce or "",
                           body=body, signature=x_connector_signature or "")
        payload = json.loads(body.decode("utf-8"))
    except (ConnectorProtocolError, ValueError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    try:
        validate_connector_result(payload, connector_id=x_connector_id or "")
    except ConnectorProtocolError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    _connector_device_allowed(x_connector_id or "", payload["device_id"])
    command_id = payload["command_id"]
    if not store.complete(connector_id=x_connector_id or "", result=payload):
        raise HTTPException(status_code=409, detail="连接器结果不匹配当前命令。")
    return {"accepted": True, "command_id": command_id}


@app.post("/api/device/{device_id}/directional-swipe")
def execute_directional_swipe(
    device_id: str,
    body: DirectionalSwipeRequest,
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    """Execute one configured directional swipe locally, without Qwen."""

    verify_local_request(request, x_control_token)
    resolved_device = str(device_id or "").strip()
    try:
        controller = runtime.controller_for_device(resolved_device)
    except (DeviceControllerRegistryError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if runtime.device_task_registry.active_session(resolved_device):
        raise HTTPException(status_code=409, detail="该设备正在执行任务，暂不能直接滑动。")
    status = controller.device_status()
    if not status.get("controller_online") or not status.get("camera_online"):
        raise HTTPException(status_code=409, detail="控制端或摄像头离线。")
    if status.get("busy"):
        raise HTTPException(status_code=409, detail="机械臂正在执行其他任务。")
    try:
        with _supervised_hardware_lock(resolved_device):
            controller.begin_new_task()
            execution = controller.vision_fixed_directional_swipe(body.direction)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=409, detail=f"固定方向滑动执行失败：{exc}") from exc
    return {
        "device_id": resolved_device,
        "direction": body.direction,
        "physical_actions": 1,
        "qwen_called": False,
        "execution": execution,
    }


def _effective_hardware_capabilities(
    hardware_capabilities: dict[str, Any],
    hardware_capability_profile: dict[str, Any] | None,
    default_text_transport: Any,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    effective_profile = hardware_capability_profile
    if isinstance(hardware_capability_profile, dict) and default_text_transport is not None:
        effective_profile = dict(hardware_capability_profile)
        copied_actions = {
            str(name): dict(spec)
            for name, spec in dict(effective_profile.get("actions") or {}).items()
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
        effective_profile["actions"] = copied_actions
    profile_actions = (
        effective_profile.get("actions", {})
        if isinstance(effective_profile, dict)
        else {}
    )
    profile_capabilities = {
        str(action): bool(spec.get("enabled"))
        for action, spec in profile_actions.items()
        if isinstance(action, str) and isinstance(spec, dict)
    }
    return effective_profile, profile_capabilities or hardware_capabilities


def _enabled_physical_actions(
    device_capabilities: dict[str, Any],
    app_launcher: Any,
) -> set[str]:
    enabled_physical_actions = {
        action
        for action in CANONICAL_ACTION_KINDS
        if action != "wait_for_change"
        and bool(device_capabilities.get(physical_capability_for_action(action), False))
    }
    if bool(getattr(app_launcher, "enabled", False)):
        enabled_physical_actions.add("launch_app")
    return enabled_physical_actions


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
    hardware_capability_profile, device_capabilities = _effective_hardware_capabilities(
        hardware_capabilities, hardware_capability_profile, default_text_transport)
    default_app_launcher = active_runtime.app_launcher_for_device(default_device_id)
    enabled_physical_actions = _enabled_physical_actions(device_capabilities, default_app_launcher)
    return (
        default_device_id,
        hardware_capabilities,
        hardware_capability_profile,
        text_transport_status,
        enabled_physical_actions,
    )


def _public_device_statuses(active_runtime: Runtime, request: Request | None = None) -> list[dict[str, Any]]:
    devices = []
    for descriptor in active_runtime.device_controllers.descriptors():
        if not _website_visible(request, "device", descriptor["device_id"]):
            continue
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
        "model_gateway": active_runtime.model_gateway_registry.status(),
        "tool_registry": active_runtime.tool_registry.status(),
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


@app.get("/api/features/lucky-bag")
def lucky_bag_feature(
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    """Return the additive lucky-bag goal preset for the existing Agent entry."""

    verify_local_request(request, x_control_token)
    profile = LuckyBagProfile(
        device_id="device-local-01",
        recipient=os.environ.get(
            "LUCKY_BAG_NOTIFICATION_RECIPIENT",
            "q2904047615@gmail.com",
        ),
    )
    return {
        "feature_id": "lucky_bag",
        "stage": "goal_preset",
        "trial": _LUCKY_BAG_MONITOR.config,
        "notification": gmail_configuration_status(),
        "profile": {
            "device_id": profile.device_id,
            "recipient": profile.recipient,
            "duration_seconds": profile.duration_seconds,
            "app_alias": profile.app_alias,
            "subject": profile.subject,
            "body": profile.body,
        },
        "goal": build_lucky_bag_goal(profile),
        "notification": gmail_configuration_status(),
        "notice": "此入口可填充通用Agent目标并启动分段长期监控；启动前请确认 Gmail 通知状态。",
    }


@app.get("/api/features/qishui-ad-test")
def qishui_ad_test_feature(
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    """Return the explicit manual, single-ad test entry point."""

    verify_local_request(request, x_control_token)
    profile = QishuiAdTestProfile(device_id="device-local-01")
    return {
        "feature_id": "qishui_ad_test",
        "stage": "manual_start",
        "profile": {
            "device_id": profile.device_id,
            "app_alias": profile.app_alias,
            "duration_seconds": profile.duration_seconds,
        },
        "goal": build_qishui_ad_test_goal(profile),
        "notice": "用户手动打开一次广告；功能只观察页面并在奖励/结束页等待人工确认，不会自动领取金币。",
        "automatic_loop_enabled": False,
        "max_ads_per_round": 1,
    }


def _qishui_ad_test_profile_payload(profile: QishuiAdTestProfile) -> dict[str, Any]:
    return {
        "device_id": profile.device_id,
        "app_alias": profile.app_alias,
        "duration_seconds": profile.duration_seconds,
    }


def _qishui_ad_test_record_or_404(test_id: str):
    record = _QISHUI_AD_TEST_MONITOR.get(test_id)
    if record is None:
        raise HTTPException(status_code=404, detail="汽水广告测试不存在。")
    return record


@app.post("/api/features/qishui-ad-test/start")
def start_qishui_ad_test(
    body: QishuiAdTestStartRequest,
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    verify_local_request(request, x_control_token)
    profile = QishuiAdTestProfile(
        device_id=body.device_id,
        duration_seconds=body.duration_seconds,
    )
    goal = build_qishui_ad_test_goal(profile)
    try:
        record = _QISHUI_AD_TEST_MONITOR.start(profile=profile, goal=goal)
        _claim_website_resource(request, "ad_test", record.test_id, body.device_id)
        if record.session_id:
            _claim_website_resource(request, "session", record.session_id, body.device_id)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {
        "feature_id": "qishui_ad_test",
        "stage": "manual_observe",
        "test": record.snapshot(),
        "profile": _qishui_ad_test_profile_payload(profile),
        "goal": goal,
        "notice": "请手动打开一次广告，然后按需请求observe；测试不会自动点击广告或领取金币。",
    }


@app.get("/api/features/qishui-ad-test/{test_id}")
def get_qishui_ad_test(
    test_id: str,
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    verify_local_request(request, x_control_token)
    record = _qishui_ad_test_record_or_404(test_id)
    return {
        "feature_id": "qishui_ad_test",
        "stage": record.phase,
        "test": record.snapshot(),
        "profile": _qishui_ad_test_profile_payload(record.profile),
        "goal": record.goal,
    }


@app.post("/api/features/qishui-ad-test/{test_id}/observe")
def observe_qishui_ad_test(
    test_id: str,
    body: QishuiAdTestDeviceRequest,
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    verify_local_request(request, x_control_token)
    _qishui_ad_test_record_or_404(test_id)
    try:
        updated = _QISHUI_AD_TEST_MONITOR.observe(test_id, device_id=body.device_id)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"feature_id": "qishui_ad_test", "stage": updated.phase, "test": updated.snapshot()}


@app.post("/api/features/qishui-ad-test/{test_id}/confirm")
def confirm_qishui_ad_test(
    test_id: str,
    body: QishuiAdTestConfirmationRequest,
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    verify_local_request(request, x_control_token)
    _qishui_ad_test_record_or_404(test_id)
    confirmation = body.confirmation.model_dump() if body.confirmation is not None else None
    try:
        updated = _QISHUI_AD_TEST_MONITOR.confirm(
            test_id,
            device_id=body.device_id,
            confirmed=body.confirmed,
            mode=body.mode,
            confirmation=confirmation,
        )
    except (KeyError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"feature_id": "qishui_ad_test", "stage": updated.phase, "test": updated.snapshot()}


@app.post("/api/features/qishui-ad-test/{test_id}/cancel")
def cancel_qishui_ad_test(
    test_id: str,
    body: QishuiAdTestDeviceRequest,
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    verify_local_request(request, x_control_token)
    _qishui_ad_test_record_or_404(test_id)
    try:
        updated = _QISHUI_AD_TEST_MONITOR.cancel(test_id, device_id=body.device_id)
    except (KeyError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"feature_id": "qishui_ad_test", "stage": updated.phase, "test": updated.snapshot()}


def _lucky_bag_profile_payload(profile: LuckyBagProfile) -> dict[str, Any]:
    return {
        "device_id": profile.device_id,
        "recipient": profile.recipient,
        "duration_seconds": profile.duration_seconds,
        "app_alias": profile.app_alias,
        "subject": profile.subject,
        "body": profile.body,
    }


def _lucky_bag_record_or_404(monitor_id: str):
    record = _LUCKY_BAG_MONITOR.get(monitor_id)
    if record is None:
        raise HTTPException(status_code=404, detail="福袋监控不存在。")
    return record


@app.post("/api/features/lucky-bag/start")
def start_lucky_bag_monitor(
    body: LuckyBagStartRequest,
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    verify_local_request(request, x_control_token)
    profile = LuckyBagProfile(
        device_id=body.device_id,
        recipient=body.recipient,
        duration_seconds=body.duration_seconds,
    )
    goal = build_lucky_bag_goal(profile)
    try:
        record = _LUCKY_BAG_MONITOR.start(profile=profile, goal=goal)
        _claim_website_resource(request, "monitor", record.monitor_id, body.device_id)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {
        "feature_id": "lucky_bag",
        "stage": "monitoring",
        "monitor": record.snapshot(),
        "profile": _lucky_bag_profile_payload(profile),
        "goal": goal,
        "notification": gmail_configuration_status(),
        "notice": "监控已排队启动；用户仍需手动打开直播间。",
    }


@app.get("/api/features/lucky-bag/monitors")
def list_lucky_bag_monitors(
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    verify_local_request(request, x_control_token)
    records = _LUCKY_BAG_MONITOR.list()
    return {
        "feature_id": "lucky_bag",
        "monitors": [
            {
                "monitor": record.snapshot(),
                "profile": _lucky_bag_profile_payload(record.profile),
            }
            for record in records if _website_visible(request, "monitor", record.monitor_id)
        ],
    }


@app.get("/api/features/lucky-bag/{monitor_id}")
def get_lucky_bag_monitor(
    monitor_id: str,
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    verify_local_request(request, x_control_token)
    record = _lucky_bag_record_or_404(monitor_id)
    return {
        "feature_id": "lucky_bag",
        "stage": "monitoring",
        "monitor": record.snapshot(),
        "profile": _lucky_bag_profile_payload(record.profile),
        "goal": record.goal,
    }


@app.post("/api/features/lucky-bag/{monitor_id}/pause")
def pause_lucky_bag_monitor(
    monitor_id: str,
    body: LuckyBagDeviceRequest,
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    verify_local_request(request, x_control_token)
    record = _lucky_bag_record_or_404(monitor_id)
    if body.device_id != record.profile.device_id:
        raise HTTPException(status_code=409, detail="请求device_id与福袋监控设备不一致。")
    try:
        updated = _LUCKY_BAG_MONITOR.pause(monitor_id)
    except Exception as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"feature_id": "lucky_bag", "monitor": updated.snapshot()}


@app.post("/api/features/lucky-bag/{monitor_id}/resume")
def resume_lucky_bag_monitor(
    monitor_id: str,
    body: LuckyBagDeviceRequest,
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    verify_local_request(request, x_control_token)
    record = _lucky_bag_record_or_404(monitor_id)
    if body.device_id != record.profile.device_id:
        raise HTTPException(status_code=409, detail="请求device_id与福袋监控设备不一致。")
    try:
        updated = _LUCKY_BAG_MONITOR.resume(monitor_id)
    except Exception as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"feature_id": "lucky_bag", "monitor": updated.snapshot()}


@app.post("/api/features/lucky-bag/{monitor_id}/cancel")
def cancel_lucky_bag_monitor(
    monitor_id: str,
    body: LuckyBagDeviceRequest,
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    verify_local_request(request, x_control_token)
    record = _lucky_bag_record_or_404(monitor_id)
    if body.device_id != record.profile.device_id:
        raise HTTPException(status_code=409, detail="请求device_id与福袋监控设备不一致。")
    try:
        updated = _LUCKY_BAG_MONITOR.cancel(monitor_id)
    except Exception as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"feature_id": "lucky_bag", "monitor": updated.snapshot()}

@app.post("/api/device/{device_id}/pair")
def pair_device(
    device_id: str,
    body: DevicePairRequest,
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    """Pair and reconnect one configured Android device from the local page."""

    verify_local_request(request, x_control_token)
    resolved_device = str(device_id or "").strip()
    if runtime.device_task_registry.active_session(resolved_device):
        raise HTTPException(status_code=409, detail="该设备正在执行任务，暂不能配对。")
    try:
        service = runtime.adb_pairing_service_for_device(resolved_device)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if service is None:
        raise HTTPException(status_code=409, detail="该设备没有配置可用的 ADB 连接。")
    try:
        result = service.pair_and_connect(
            pairing_host=body.pairing_host,
            pairing_port=body.pairing_port,
            pairing_code=body.pairing_code,
        )
    except AdbPairingError as exc:
        raise HTTPException(status_code=409, detail={"code": exc.code, "error": str(exc)}) from exc
    return {"device_id": resolved_device, "pairing": result}


@app.post("/api/manual-motion/start")
def start_manual_motion(
    body: ManualMotionStartRequest,
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    verify_local_request(request, x_control_token)
    if isinstance(runtime, RuntimeUnavailable):
        raise HTTPException(status_code=503, detail=runtime.startup_error)
    _MANUAL_MOTION_MONITOR.registry = runtime.device_task_registry
    # Keep startup notices ASCII-only so a Windows cp1252 stdout cannot fail
    # the lifespan before the application is ready.
    print(f"Phone Visual Agent web console: http://127.0.0.1:{_TRIAL_PORT}/", flush=True)
    print("Local control token generated for the protected bootstrap page.", flush=True)
    if os.environ.get("ROBOT_WEB_NO_BROWSER") != "1":
        threading.Timer(
            1.0, lambda: webbrowser.open(f"http://127.0.0.1:{_TRIAL_PORT}/")
        ).start()
    try:
        runtime.controller_for_device(body.device_id)
        profile = MotionProfile(
            device_id=body.device_id,
            kind=body.kind,
            point_a=tuple(body.point_a),
            point_b=tuple(body.point_b),
            interval_seconds=body.interval_seconds,
            duration_seconds=body.duration_seconds,
        )
        record = _MANUAL_MOTION_MONITOR.start(profile)
        _claim_website_resource(request, "motion", record["motion_id"], body.device_id)
    except (ValueError, DeviceTaskRegistryError, RuntimeError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"motion": record, "physical_actions": 0, "qwen_called": False}


@app.get("/api/manual-motion/{motion_id}")
def get_manual_motion(
    motion_id: str,
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    verify_local_request(request, x_control_token)
    record = _MANUAL_MOTION_MONITOR.get(motion_id)
    if record is None:
        raise HTTPException(status_code=404, detail="固定测试不存在。")
    return {"motion": record}


@app.post("/api/manual-motion/{motion_id}/stop")
def stop_manual_motion(
    motion_id: str,
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    verify_local_request(request, x_control_token)
    try:
        return {"motion": _MANUAL_MOTION_MONITOR.stop(motion_id)}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="固定测试不存在。") from exc


@app.get("/api/device")
def device(request: Request = None) -> dict[str, Any]:
    if isinstance(runtime, RuntimeUnavailable):
        raise HTTPException(status_code=503, detail={
            "code": "runtime_not_ready", "error": runtime.startup_error,
        })
    devices = _public_device_statuses(runtime, request)
    if not devices:
        return {"devices": [], "default_device_id": "", "active_tasks": [], "manual_motion_tests": [],
                "generic_supervised_execution": {"active_sessions": []}}
    default_owned = _website_visible(request, "device", runtime.device_controllers.default_device_id)
    status = dict(runtime.controller.device_status()) if default_owned else dict(devices[0])
    if not default_owned:
        sessions = [item for item in _active_session_status(runtime)
                    if _website_visible(request, "session", item["session_id"])]
        return {**status, "devices": devices, "default_device_id": devices[0]["device_id"],
                "active_tasks": sessions, "manual_motion_tests": [item for item in _MANUAL_MOTION_MONITOR.list()
                if _website_visible(request, "motion", item["motion_id"])],
                "generic_supervised_execution": {"active_sessions": sessions}}
    status.pop("readiness", None)
    (
        default_device_id,
        hardware_capabilities,
        hardware_capability_profile,
        text_transport_status,
        enabled_physical_actions,
    ) = _device_capability_snapshot(runtime)
    status["default_device_id"] = default_device_id if default_owned else devices[0]["device_id"]
    status["devices"] = devices
    status["vision_agent"] = _website_model_status(runtime.vision_provider.status())
    status["execution_architecture"] = _execution_architecture_status(
        runtime,
        hardware_capabilities=hardware_capabilities,
        hardware_capability_profile=hardware_capability_profile,
        text_transport_status=text_transport_status,
        enabled_physical_actions=enabled_physical_actions,
    )
    status["execution_architecture"]["universal_agent"]["observer"] = _website_model_status(
        status["execution_architecture"]["universal_agent"]["observer"])
    for gateway in status["execution_architecture"]["model_gateway"].get("gateways", []):
        gateway["provider_status"] = _website_model_status(gateway.get("provider_status", {}))
    active_sessions = [item for item in _active_session_status(runtime)
                       if _website_visible(request, "session", item["session_id"])]
    status["active_tasks"] = [dict(item) for item in active_sessions]
    status["manual_motion_tests"] = [item for item in _MANUAL_MOTION_MONITOR.list()
                                     if _website_visible(request, "motion", item["motion_id"])]
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
    RuntimeQuotaExceeded,
)

CAPABILITY_ACCEPTANCE_ERRORS = (CapabilityAcceptanceError, *COMMON_OPERATION_ERRORS)

GENERIC_SESSION_ERRORS = (
    AgentSessionCommandError,
    AgentSessionRecoveryRequiredError,
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
        _claim_website_resource(request, "trial", trial.trial_id, body.device_id)
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
    return {"trials": [item for item in runtime.capability_acceptance_manager.snapshots()
                       if _website_visible(request, "trial", item["trial_id"])]}


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


def _execute_queued_generic_start(body, request, control_token):
    # A queued closure can outlive the request's identity/device leases. Reuse
    # the same authority at dispatch, rather than trusting cached principal data.
    import asyncio
    store = _control_plane_store()
    asyncio.run(website_access.authorize(request, store, verify_local_origin,
                                        runtime.device_controllers.default_device_id))
    with website_access.member_access(store, request):
        with website_access.device_access(store, request, request.state.website_devices):
            return start_generic_supervised_session(body, request, control_token)


@app.post("/api/agent/generic-supervised/start-async")
def start_generic_supervised_async(body: GenericSupervisedStartRequest, request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token")) -> dict[str, Any]:
    verify_local_request(request, x_control_token)
    task_id = uuid.uuid4().hex
    request.state.start_session_id = uuid.uuid4().hex
    _claim_website_resource(request, "start", task_id, body.device_id)
    try:
        active = _GENERIC_START_TASKS.active_for("device_id", body.device_id)
        if active is not None:
            raise HTTPException(
                status_code=409,
                detail={
                    "success": False,
                    "code": "device_start_already_queued",
                    "error": f"设备 {body.device_id} 已有待启动任务。",
                    "task_id": active.get("task_id"),
                },
            )
        _GENERIC_START_TASKS.reserve(
            task_id,
            metadata={"task_id": task_id, "device_id": body.device_id,
                      "session_id": request.state.start_session_id},
        )
        _GENERIC_START_TASKS.submit(
            task_id,
            lambda: _execute_queued_generic_start(body, request, x_control_token),
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
    payload = {"task_id": task_id, **{key: value for key, value in task.items() if key != "created_monotonic"}}
    error_detail = payload.get("error_detail")
    if error_detail is not None:
        payload["detail"] = error_detail
        if isinstance(error_detail, dict):
            for key in ("code", "phase", "recoverable"):
                if key in error_detail:
                    payload[key if key != "code" else "error_code"] = error_detail[key]
    return payload


@app.post("/api/agent/generic-supervised/start-async/{task_id}/cancel")
def cancel_generic_supervised_start_task(
    task_id: str,
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    """Cancel a queued asynchronous start before it creates a device session."""

    verify_local_request(request, x_control_token)
    task = _GENERIC_START_TASKS.cancel(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="启动任务不存在或已过期。")
    payload = {"task_id": task_id, **{key: value for key, value in task.items() if key != "created_monotonic"}}
    error_detail = payload.get("error_detail")
    if error_detail is not None:
        payload["detail"] = error_detail
    return payload


@app.post("/api/agent/generic-supervised/start")
def start_generic_supervised_session(
    body: GenericSupervisedStartRequest,
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    """Start one canonical whole-task visual session.

    Automatic mode advances the same Planner ToolIntent loop with Qwen as the
    current Model Gateway; it does not build
    a local plan or restrict execution to a read-only action subset.
    """

    verify_local_request(request, x_control_token)
    session_id = getattr(request.state, "start_session_id", None) or uuid.uuid4().hex
    _claim_website_resource(request, "session", session_id, body.device_id)
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
                conversation=tuple(item.model_dump() for item in body.conversation),
                max_physical_actions=body.max_physical_actions,
                max_observations=body.max_observations,
                organization_id=str((getattr(request.state, "principal", None) or {}).get("organization_id") or LOCAL_ORGANIZATION_ID),
                principal_user_id=str((getattr(request.state, "principal", None) or {}).get("user_id") or "local-owner"),
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


@app.get("/api/agent/generic-supervised/{session_id}/events")
def get_generic_supervised_events(
    session_id: str,
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    """Return the durable Runtime event cursor for the console Trace view."""

    verify_local_request(request, x_control_token)
    _require_generic_supervised_session(session_id)
    after_raw = request.query_params.get("after_sequence", "0")
    limit_raw = request.query_params.get("limit", "200")
    try:
        after_sequence = int(after_raw)
        limit = int(limit_raw)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="after_sequence和limit必须是整数。") from exc
    if after_sequence < 0 or not 1 <= limit <= 200:
        raise HTTPException(status_code=422, detail="after_sequence必须非负，limit必须在1～200之间。")
    repository = getattr(runtime, "agent_session_repository", None)
    read_events = getattr(repository, "read_events", None)
    if not callable(read_events):
        raise HTTPException(status_code=503, detail="当前Runtime没有可查询的持久事件账本。")
    try:
        page = read_events(session_id, after_sequence=after_sequence, limit=limit)
    except AgentSessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "protocol_version": page.to_dict()["protocol_version"],
        "session_id": session_id,
        "after_sequence": after_sequence,
        "events": [event.to_dict() for event in page.events],
        "next_sequence": page.next_sequence,
        "has_more": page.has_more,
    }


@app.post("/api/agent/chat")
def unified_agent_chat(
    body: AgentEntryRequest,
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    """Let the Model Gateway suggest chat-only or Runtime task routing."""

    verify_local_request(request, x_control_token)
    if isinstance(runtime, RuntimeUnavailable):
        raise HTTPException(status_code=503, detail=runtime.startup_error)
    text = body.text.strip()
    if not text:
        raise HTTPException(
            status_code=422,
            detail={"success": False, "code": "empty_message", "error": "消息不能为空。"},
        )
    try:
        conversation = [
            {"role": item.role, "content": item.content}
            for item in body.conversation
        ]
        conversation.append({"role": "user", "content": text})
        gateway = runtime.model_gateway_registry.default()
        route = AgentEntryRouter(gateway).route(text=text, conversation=conversation[:-1])
        route_body = route.to_dict()
        if route.route == "chat_only":
            return {
                "mode": "agent_text_conversation",
                "reply": route.reply,
                "conversation": [*conversation, {"role": "assistant", "content": route.reply}],
                "phone_task_started": False,
                "route": route.route,
                "route_contract": route_body,
            }
        start_body = GenericSupervisedStartRequest(
            text=text,
            device_id=body.device_id,
            auto_advance=True,
            conversation=conversation,
            max_physical_actions=body.max_physical_actions,
            max_observations=body.max_observations,
        )
        started = start_generic_supervised_async(start_body, request, x_control_token)
        return {
            "mode": "agent_multimodal_conversation",
            "task_id": started["task_id"],
            "status": started["status"],
            "reply": route.reply,
            "conversation": conversation,
            "phone_task_started": True,
            "route": route.route,
            "route_contract": route_body,
        }
    except HTTPException:
        raise
    except AgentEntryRouteError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "success": False,
                "code": "agent_entry_route_failed",
                "phase": "agent_entry_route",
                "error": str(exc),
            },
        ) from exc
    except GENERIC_SESSION_ERRORS as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/agent/generic-supervised/{session_id}/chat")
def unified_agent_chat_in_session(
    session_id: str,
    body: AgentChatRequest,
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    """Continue the multimodal conversation; Model Gateway suggestions are signed by Planner."""

    verify_local_request(request, x_control_token)
    if isinstance(runtime, RuntimeUnavailable):
        raise HTTPException(status_code=503, detail=runtime.startup_error)
    try:
        with runtime.agent_session_repository.locked(session_id) as session:
            if session.device_id != body.device_id:
                raise AgentSessionDeviceMismatchError(session.device_id, body.device_id)
            session.conversation.append({"role": "user", "content": body.text.strip()})
            refreshed = runtime.universal_agent_session_service.refresh(
                session, requested_device_id=body.device_id)
            snapshot = refreshed.session.snapshot()
        result = {
            "mode": "agent_multimodal_conversation",
            "reply": snapshot.get("qwen_reply", ""),
            "session": snapshot,
        }
        if snapshot.get("status") not in TERMINAL_SESSION_STATUSES:
            automatic = runtime.universal_agent_session_service.run_automatic(
                runtime.universal_agent_session_service.require(session_id),
                requested_device_id=body.device_id,
                confirmed=False,
                confirmation=None,
            )
            result["session"] = automatic.session.snapshot()
            result["automatic_progress"] = automatic.operation
            result["reply"] = result["session"].get("qwen_reply", "")
        return result
    except AgentSessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except AgentSessionDeviceMismatchError as exc:
        _raise_agent_session_device_mismatch(exc)
    except GENERIC_SESSION_ERRORS as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


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
    session = _require_generic_supervised_session(session_id)
    try:
        cancelled = runtime.universal_agent_session_service.cancel(
            session_id,
            requested_device_id=body.device_id,
        )
    except AgentSessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except AgentSessionDeviceMismatchError as exc:
        _raise_agent_session_device_mismatch(exc)
    except GENERIC_SESSION_ERRORS as exc:
        raise HTTPException(
            status_code=409,
            detail=_generic_supervised_failure(
                session,
                exc,
                request_action_count=0,
            ),
        ) from exc
    session = cancelled.session
    report = _write_generic_supervised_report(session)
    return {"session": session.snapshot(), "report": report}


@app.post("/api/agent/generic-supervised/{session_id}/reconcile")
def reconcile_generic_supervised_session(
    session_id: str,
    body: GenericRecoveryReconciliationRequest,
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    verify_local_request(request, x_control_token)
    session = _require_generic_supervised_session(session_id)
    try:
        reconciled = runtime.universal_agent_session_service.reconcile_recovered(
            session_id,
            requested_device_id=body.device_id,
            external_effect=body.external_effect,
            confirmed=body.confirmed,
            resume=body.resume,
        )
    except AgentSessionDeviceMismatchError as exc:
        _raise_agent_session_device_mismatch(exc)
    except GENERIC_SESSION_ERRORS as exc:
        raise HTTPException(
            status_code=409,
            detail=_generic_supervised_failure(
                session,
                exc,
                request_action_count=0,
            ),
        ) from exc
    report = _write_generic_supervised_report(reconciled.session)
    replacement = reconciled.replacement
    return {
        "mode": "generic_supervised_recovery",
        "session": reconciled.session.snapshot(),
        "replacement_session": replacement.snapshot() if replacement is not None else None,
        "report": report,
    }


@app.post("/api/agent/generic-supervised/{session_id}/pause")
def pause_generic_supervised_session(
    session_id: str,
    body: GenericSupervisedDeviceRequest,
    request: Request,
    x_control_token: str | None = Header(default=None, alias="X-Control-Token"),
) -> dict[str, Any]:
    """Invalidate any pending confirmation without touching hardware."""

    verify_local_request(request, x_control_token)
    session = _require_generic_supervised_session(session_id)
    try:
        paused = runtime.universal_agent_session_service.pause(
            session_id,
            requested_device_id=body.device_id,
        )
    except AgentSessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except AgentSessionDeviceMismatchError as exc:
        _raise_agent_session_device_mismatch(exc)
    except GENERIC_SESSION_ERRORS as exc:
        raise HTTPException(
            status_code=409,
            detail=_generic_supervised_failure(
                session,
                exc,
                request_action_count=0,
            ),
        ) from exc
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
    if getattr(request.state, "principal", None):
        stopped = []
        motions = []
        for descriptor in runtime.device_controllers.descriptors():
            device_id = descriptor["device_id"]
            if _website_visible(request, "device", device_id):
                runtime.controller_for_device(device_id).request_stop()
                stopped.append(device_id)
                motions.extend(_MANUAL_MOTION_MONITOR.stop_all(device_id=device_id))
        trials = []
        for item in runtime.capability_acceptance_manager.snapshots():
            if _website_visible(request, "trial", item["trial_id"]):
                trial = runtime.capability_acceptance_manager.get(item["trial_id"])
                if not trial.read_only_recovered and not trial.report_path.exists():
                    trial.controller.request_stop()
                    trials.append(trial.trial_id)
        return {"stop_requested": True, "stopped_device_ids": stopped, "queued_cancelled": [],
                "capability_stop_requested": trials, "manual_motion_stop_requested": motions,
                "note": "已请求停止当前账号的设备任务。"}
    stopped_devices = runtime.device_controllers.request_stop_all()
    # Tests and embedding callers may replace the default controller after the
    # registry was built; keep that explicit runtime handle covered as well.
    runtime.controller.request_stop()
    manual_motion_stop_requested = _MANUAL_MOTION_MONITOR.stop_all()
    capability_stop_requested = (
        runtime.capability_acceptance_manager.request_stop_all()
    )
    return {
        "stop_requested": True,
        "stopped_device_ids": list(stopped_devices),
        "queued_cancelled": [],
        "capability_stop_requested": capability_stop_requested,
        "manual_motion_stop_requested": manual_motion_stop_requested,
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
def preview_mjpg(device_id: str, request: Request) -> StreamingResponse:
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
            principal = _control_plane_store().authenticate_identity_session(website_access.identity_token(request))
            if principal is None or _control_plane_store().owned_resource("device", device_id, principal) is None:
                return
            try:
                with website_access.device_access(_control_plane_store(), request, {device_id}):
                    frame, _cached = runtime.capture_preview(device_id, quality=68)
            except Exception as exc:
                error_type = type(exc).__name__
                error_text = _diagnostic_error_text(exc)
                _LOGGER.warning(
                    "mjpg_preview_stopped device_id=%s error_type=%s error=%s",
                    device_id,
                    error_type,
                    error_text,
                    extra={
                        "event": "mjpg_preview_stopped",
                        "device_id": device_id,
                        "error_type": error_type,
                        "error_message": error_text,
                    },
                )
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
