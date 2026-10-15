"""HTTP authorization adapter. Roles come from the Control Plane domain only.

The control token is CSRF protection, never an identity. All non-public routes
authenticate a durable identity session before reading data or dispatching work.
"""
from fastapi import HTTPException, Request
from collections import Counter
from contextlib import contextmanager
import threading

# Short critical sections serialize ownership changes with admission, without
# serializing ordinary requests or execution on independent devices.
assignment_lock = threading.RLock()
inflight_devices = Counter()


@contextmanager
def device_access(store, request, device_ids):
    devices = set(device_ids)
    with assignment_lock:
        principal = store.authenticate_identity_session(identity_token(request))
        if principal is None:
            reject(store, request, None, "identity_required", 401)
        for device_id in devices:
            require_device(store, request, principal, device_id)
        inflight_devices.update(devices)
    try:
        yield
    finally:
        with assignment_lock:
            inflight_devices.subtract(devices)
            for device_id in devices:
                if not inflight_devices[device_id]:
                    del inflight_devices[device_id]

PRINCIPAL_COOKIE = "agent_principal_v2"
RESOURCE_PARAMETERS = {
    "session_id": "session", "task_id": "start", "trial_id": "trial",
    "monitor_id": "monitor", "test_id": "ad_test", "motion_id": "motion",
}


def identity_token(request):
    # An explicitly supplied invalid token never falls back to a valid cookie.
    return request.headers.get("x-control-plane-token", request.cookies.get(PRINCIPAL_COOKIE, ""))


def reject(store, request, principal, code, status=403):
    store.record_audit(organization_id=principal["organization_id"] if principal else "unknown",
                       resource="website.access", outcome=code,
                       request_id=request.headers.get("x-request-id", "")[:128],
                       actor=principal["user_id"] if principal else "anonymous",
                       metadata={"route": request.scope["route"].path, "method": request.method})
    raise HTTPException(status, detail={"code": code, "message": {
        "identity_required": "请先登录，登录状态可能已失效。",
        "organization_forbidden": "无权访问此组织。",
        "resource_not_found": "资源不存在或不属于当前账号。",
        "device_forbidden": "此设备尚未分配给当前账号。",
        "permission_denied": "当前账号没有此操作权限。",
    }.get(code, "当前请求未获授权。")})


def permission_for(path, method):
    if path.startswith("/api/control-plane/organizations/"):
        tail = path.split("/{organization_id}", 1)[1]
        if tail.startswith("/devices"): return "device.manage"
        if tail.startswith("/quotas"): return "quota.read" if method == "GET" else "quota.manage"
        if tail == "/audit": return "audit.read"
        if tail.startswith("/principals"): return "identity.manage"
        return "task.read" if method == "GET" else "team.manage"
    if path == "/api/session/reload": return "identity.manage"
    if path.startswith("/api/control-plane/auth/sessions/"): return None  # self/admin check in handler
    if method == "GET":
        return "device.read" if path.startswith(("/api/device", "/api/doctor", "/api/preview", "/api/apps")) else "task.read"
    if path.endswith(("/cancel", "/stop", "/pause")) or path == "/api/stop": return "task.cancel"
    if path.startswith("/api/device/"): return "device.execute"
    return "task.create"


async def authorize(request: Request, store, verify_origin, default_device_id):
    path = request.scope["route"].path
    if path in {"/", "/api/session", "/api/control-plane/auth/github/start",
                "/api/control-plane/auth/github/callback", "/api/control-plane/auth/external-sessions",
                "/api/control-plane/auth/browser-session", "/api/control-plane/auth/browser-logout"}:
        return
    verify_origin(request)
    principal = store.authenticate_identity_session(identity_token(request))
    if principal is None:
        reject(store, request, None, "identity_required", 401)
    request.state.principal = principal
    request.state.website_devices = set()
    if path == "/api/control-plane/auth/sessions":
        # Retired bootstrap impersonation must not become available to a
        # logged-in operator while its route is being removed.
        reject(store, request, principal, "permission_denied")
    organization = request.path_params.get("organization_id")
    if organization and organization != principal["organization_id"]:
        reject(store, request, principal, "organization_forbidden")
    permission = permission_for(path, request.method)
    if permission and permission not in principal["permissions"]:
        reject(store, request, principal, "permission_denied")
    if path.startswith("/api/control-plane/"):
        return
    if path in {"/api/device", "/api/stop"}:
        for item in store.device_assignments(principal["organization_id"]):
            if item["user_id"] == principal["user_id"]:
                require_device(store, request, principal, item["device_id"])
    for param, kind in RESOURCE_PARAMETERS.items():
        if param in request.path_params:
            resource = store.owned_resource(kind, request.path_params[param], principal)
            if resource is None:
                reject(store, request, principal, "resource_not_found", 404)
            if request.method != "GET" and resource["device_id"]:
                require_device(store, request, principal, resource["device_id"])
    try:
        body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    except ValueError as error:
        raise HTTPException(422, detail="请求 JSON 无效。") from error
    body = body if isinstance(body, dict) else {}
    device_id = request.path_params.get("device_id") or request.query_params.get("device_id") or body.get("device_id")
    if path in {"/api/agent/generic-scene", "/api/features/lucky-bag", "/api/features/qishui-ad-test"}:
        device_id = default_device_id
    if path == "/api/qwen/chat" and not device_id:
        device_id = "device-local-01"  # Same default as QwenDraftChatRequest.
    if device_id:
        require_device(store, request, principal, str(device_id))
        if request.method != "GET" and "device.execute" not in principal["permissions"]:
            reject(store, request, principal, "permission_denied")
    # The reload operation restarts a shared process; only the installation
    # owner may operate it. Organization admin does not grant host ownership.
    if path == "/api/session/reload" and (principal["organization_id"], principal["user_id"]) != ("local-org", "local-owner"):
        reject(store, request, principal, "permission_denied")


def require_device(store, request, principal, device_id):
    if store.owned_resource("device", device_id, principal) is None:
        reject(store, request, principal, "device_forbidden")
    if hasattr(request.state, "website_devices"):
        request.state.website_devices.add(device_id)


def claim(store, request, kind, resource_id, device_id):
    principal = getattr(request.state, "principal", None)
    if principal:  # In-process application tests do not pass through HTTP.
        store.claim_resource(kind, resource_id, principal, device_id=device_id)


def visible(store, request, kind, resource_id):
    principal = getattr(request.state, "principal", None)
    return principal is None or store.owned_resource(kind, str(resource_id), principal) is not None
