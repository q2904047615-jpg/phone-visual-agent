"""Typed local HTTP gateway for the universal phone agent.

The gateway deliberately exposes named operations instead of a generic request
method.  Every operation is checked against the OpenAPI document served by the
currently running process before a request is sent.  The in-memory control
token is obtained from the same process and is never returned or persisted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
from urllib.parse import quote

import httpx


JsonObject = dict[str, Any]


@dataclass(frozen=True)
class ApiErrorDetails:
    category: str
    message: str
    method: str = ""
    path: str = ""
    status_code: int | None = None
    retryable: bool = False
    detail: Any = None

    def to_dict(self) -> JsonObject:
        return {
            "category": self.category,
            "message": self.message,
            "method": self.method,
            "path": self.path,
            "status_code": self.status_code,
            "retryable": self.retryable,
            "detail": self.detail,
        }


class LocalAgentApiError(RuntimeError):
    def __init__(self, details: ApiErrorDetails) -> None:
        super().__init__(details.message)
        self.details = details


class LocalAgentApiClient:
    """OpenAPI-checked client for the current universal-agent HTTP surface."""

    DEVICE_ROUTE = "/api/device"
    DOCTOR_ROUTE = "/api/doctor/{device_id}"
    START_ROUTE = "/api/agent/generic-supervised/start"
    SESSION_ROUTE = "/api/agent/generic-supervised/{session_id}"
    CONFIRM_ROUTE = SESSION_ROUTE + "/confirm"
    NEXT_ROUTE = SESSION_ROUTE + "/next"
    AUTO_ROUTE = SESSION_ROUTE + "/auto"
    CANCEL_ROUTE = SESSION_ROUTE + "/cancel"
    RECONCILE_ROUTE = SESSION_ROUTE + "/reconcile"
    PAUSE_ROUTE = SESSION_ROUTE + "/pause"
    CONTROL_PLANE_ORGANIZATION_ROUTE = "/api/control-plane/organizations/{organization_id}"
    CONTROL_PLANE_TEAMS_ROUTE = CONTROL_PLANE_ORGANIZATION_ROUTE + "/teams"
    CONTROL_PLANE_PRINCIPALS_ROUTE = CONTROL_PLANE_ORGANIZATION_ROUTE + "/principals"
    CONTROL_PLANE_QUOTAS_ROUTE = CONTROL_PLANE_ORGANIZATION_ROUTE + "/quotas"
    CONTROL_PLANE_AUDIT_ROUTE = CONTROL_PLANE_ORGANIZATION_ROUTE + "/audit"
    CONTROL_PLANE_IDENTITY_SESSION_ROUTE = "/api/control-plane/auth/sessions"
    CONTROL_PLANE_EXTERNAL_SESSION_ROUTE = "/api/control-plane/auth/external-sessions"
    CONTROL_PLANE_GITHUB_OAUTH_START_ROUTE = "/api/control-plane/auth/github/start"
    CONTROL_PLANE_GITHUB_OAUTH_CALLBACK_ROUTE = "/api/control-plane/auth/github/callback"
    CONTROL_PLANE_REVOKE_SESSION_ROUTE = CONTROL_PLANE_IDENTITY_SESSION_ROUTE + "/{session_id}"
    CONTROL_PLANE_TEAM_ITEM_ROUTE = CONTROL_PLANE_ORGANIZATION_ROUTE + "/teams/{team_id}"
    CONTROL_PLANE_QUOTA_ITEM_ROUTE = CONTROL_PLANE_ORGANIZATION_ROUTE + "/quotas/{quota_key}"

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8765",
        timeout_seconds: float = 180.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        normalized = base_url.rstrip("/")
        if normalized not in {"http://127.0.0.1:8765", "http://localhost:8765"}:
            raise LocalAgentApiError(
                ApiErrorDetails(
                    category="client_contract_error",
                    message="本地 Agent API 只允许连接 127.0.0.1:8765。",
                )
            )
        if timeout_seconds <= 0:
            raise LocalAgentApiError(
                ApiErrorDetails(
                    category="client_contract_error",
                    message="接口超时必须是正数。",
                )
            )
        self._client = httpx.Client(
            base_url=normalized,
            timeout=httpx.Timeout(timeout_seconds, connect=min(5.0, timeout_seconds)),
            transport=transport,
        )
        self._token = ""
        self._openapi: JsonObject | None = None
        self._service_version = ""

    def __enter__(self) -> "LocalAgentApiClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def bootstrap(self) -> JsonObject:
        if self._openapi is not None:
            return {
                "service_version": self._service_version,
                "openapi_verified": True,
                "control_token_loaded": bool(self._token),
            }
        session_payload = self._raw_json("GET", "/api/session", read_only=True)
        openapi = self._raw_json("GET", "/openapi.json", read_only=True)
        token = session_payload.get("token")
        service_version = session_payload.get("version")
        openapi_version = (openapi.get("info") or {}).get("version")
        if not isinstance(token, str) or not token:
            self._raise_contract("当前服务没有提供有效控制令牌。")
        if not isinstance(service_version, str) or not service_version:
            self._raise_contract("当前服务没有提供版本号。")
        if service_version != openapi_version:
            raise LocalAgentApiError(
                ApiErrorDetails(
                    category="service_contract_error",
                    message="服务版本与 OpenAPI 版本不一致。",
                    detail={
                        "service_version": service_version,
                        "openapi_version": openapi_version,
                    },
                )
            )
        if not isinstance(openapi.get("paths"), dict):
            self._raise_contract("当前 OpenAPI 缺少 paths。")
        self._token = token
        self._openapi = openapi
        self._service_version = service_version
        return {
            "service_version": service_version,
            "openapi_verified": True,
            "control_token_loaded": True,
        }

    def device_status(self) -> JsonObject:
        return self._request("GET", self.DEVICE_ROUTE, read_only=True)

    def control_plane_organization(self, organization_id: str) -> JsonObject:
        return self._request(
            "GET", self.CONTROL_PLANE_ORGANIZATION_ROUTE,
            path_params={"organization_id": self._resource_id(organization_id)}, read_only=True,
        )

    def control_plane_teams(self, organization_id: str) -> JsonObject:
        return self._request(
            "GET", self.CONTROL_PLANE_TEAMS_ROUTE,
            path_params={"organization_id": self._resource_id(organization_id)}, read_only=True,
        )

    def control_plane_principals(self, organization_id: str) -> JsonObject:
        return self._request(
            "GET", self.CONTROL_PLANE_PRINCIPALS_ROUTE,
            path_params={"organization_id": self._resource_id(organization_id)}, read_only=True,
        )

    def control_plane_quotas(self, organization_id: str) -> JsonObject:
        return self._request(
            "GET", self.CONTROL_PLANE_QUOTAS_ROUTE,
            path_params={"organization_id": self._resource_id(organization_id)}, read_only=True,
        )

    def control_plane_audit(self, organization_id: str, *, after_sequence: int = 0,
                            limit: int = 50) -> JsonObject:
        if isinstance(after_sequence, bool) or after_sequence < 0:
            self._raise_contract("after_sequence 必须是非负整数。")
        if isinstance(limit, bool) or limit < 1 or limit > 100:
            self._raise_contract("limit 必须在 1-100。")
        return self._request(
            "GET", self.CONTROL_PLANE_AUDIT_ROUTE,
            path_params={"organization_id": self._resource_id(organization_id)},
            query_params={"after_sequence": str(after_sequence), "limit": str(limit)},
            read_only=True,
        )

    def control_plane_identity_session(self, organization_id: str, user_id: str) -> JsonObject:
        return self._request(
            "POST", self.CONTROL_PLANE_IDENTITY_SESSION_ROUTE,
            payload={"organization_id": self._resource_id(organization_id), "user_id": self._resource_id(user_id)},
            read_only=False,
        )

    def control_plane_external_identity_session(self, provider_id: str, assertion: Mapping[str, Any]) -> JsonObject:
        return self._request(
            "POST", self.CONTROL_PLANE_EXTERNAL_SESSION_ROUTE,
            payload={"provider_id": provider_id, "assertion": dict(assertion)}, read_only=False,
            extra_headers={},
        )

    def control_plane_github_oauth_start(self, *, browser: bool = False) -> JsonObject:
        return self._request(
            "GET", self.CONTROL_PLANE_GITHUB_OAUTH_START_ROUTE,
            query_params={"browser": "true"} if browser else None,
            read_only=False,
        )

    def control_plane_browser_session(self) -> JsonObject:
        return self._request("GET", "/api/control-plane/auth/browser-session", read_only=True)

    def control_plane_browser_logout(self) -> JsonObject:
        return self._request("POST", "/api/control-plane/auth/browser-logout", read_only=False)

    def control_plane_revoke_identity_session(self, session_id: str, *, principal_token: str,
                                              idempotency_key: str) -> JsonObject:
        if not principal_token or not idempotency_key:
            self._raise_contract("会话撤销必须提供主体令牌和幂等键。")
        return self._request(
            "DELETE", self.CONTROL_PLANE_REVOKE_SESSION_ROUTE,
            path_params={"session_id": self._resource_id(session_id)}, read_only=False,
            extra_headers={"X-Control-Plane-Token": principal_token, "Idempotency-Key": idempotency_key},
        )

    def control_plane_create_team(self, organization_id: str, *, team_id: str, name: str,
                                  member_ids: list[str], principal_token: str,
                                  idempotency_key: str) -> JsonObject:
        return self._control_plane_write(
            "POST", self.CONTROL_PLANE_TEAMS_ROUTE,
            organization_id=organization_id, principal_token=principal_token,
            idempotency_key=idempotency_key,
            payload={"team_id": team_id, "name": name, "member_ids": list(member_ids)},
        )

    def control_plane_update_team(self, organization_id: str, team_id: str, *,
                                  principal_token: str, idempotency_key: str,
                                  name: str | None = None, member_ids: list[str] | None = None) -> JsonObject:
        payload: JsonObject = {}
        if name is not None:
            payload["name"] = name
        if member_ids is not None:
            payload["member_ids"] = list(member_ids)
        return self._control_plane_write(
            "PATCH", self.CONTROL_PLANE_TEAM_ITEM_ROUTE,
            organization_id=organization_id, team_id=team_id,
            principal_token=principal_token, idempotency_key=idempotency_key, payload=payload,
        )

    def control_plane_create_principal(self, organization_id: str, *, user_id: str, role: str,
                                       team_ids: list[str], principal_token: str,
                                       idempotency_key: str) -> JsonObject:
        return self._control_plane_write(
            "POST", self.CONTROL_PLANE_PRINCIPALS_ROUTE,
            organization_id=organization_id, principal_token=principal_token,
            idempotency_key=idempotency_key,
            payload={"user_id": user_id, "role": role, "team_ids": list(team_ids)},
        )

    def control_plane_update_quota(self, organization_id: str, quota_key: str, *, period: str,
                                   principal_token: str, idempotency_key: str,
                                   max_runs: int | None = None,
                                   max_physical_actions: int | None = None,
                                   max_model_tokens: int | None = None) -> JsonObject:
        return self._control_plane_write(
            "PUT", self.CONTROL_PLANE_QUOTA_ITEM_ROUTE,
            organization_id=organization_id, quota_key=quota_key,
            principal_token=principal_token, idempotency_key=idempotency_key,
            payload={"quota_key": quota_key, "period": period, "max_runs": max_runs,
                     "max_physical_actions": max_physical_actions, "max_model_tokens": max_model_tokens},
        )

    def _control_plane_write(self, method: str, route_template: str, *, organization_id: str,
                             principal_token: str, idempotency_key: str, payload: JsonObject,
                             team_id: str | None = None, quota_key: str | None = None) -> JsonObject:
        if not principal_token or not idempotency_key:
            self._raise_contract("Control Plane 写操作必须提供主体令牌和幂等键。")
        params = {"organization_id": self._resource_id(organization_id)}
        if team_id is not None:
            params["team_id"] = self._resource_id(team_id)
        if quota_key is not None:
            params["quota_key"] = self._resource_id(quota_key)
        return self._request(
            method, route_template, path_params=params, payload=payload, read_only=False,
            extra_headers={"X-Control-Plane-Token": principal_token, "Idempotency-Key": idempotency_key},
        )

    def doctor(self, device_id: str) -> JsonObject:
        return self._request(
            "GET",
            self.DOCTOR_ROUTE,
            path_params={"device_id": self._resource_id(device_id)},
            read_only=True,
        )

    def start_session(
        self,
        *,
        text: str,
        exact_input_text: str | None = None,
        exact_action_kind: str | None = None,
        exact_target_label: str = "",
        device_id: str,
        auto_advance: bool = False,
        max_physical_actions: int | None = None,
        max_observations: int | None = None,
    ) -> JsonObject:
        payload: JsonObject = {
            "text": text,
            "device_id": device_id,
            "auto_advance": auto_advance,
        }
        if max_physical_actions is not None:
            payload["max_physical_actions"] = max_physical_actions
        if max_observations is not None:
            payload["max_observations"] = max_observations
        if exact_input_text is not None:
            payload["exact_input_text"] = exact_input_text
        if exact_action_kind is not None:
            payload["exact_action_kind"] = exact_action_kind
        if exact_target_label:
            payload["exact_target_label"] = exact_target_label
        return self._request(
            "POST",
            self.START_ROUTE,
            payload=payload,
            read_only=False,
        )

    def get_session(self, session_id: str) -> JsonObject:
        return self._request(
            "GET",
            self.SESSION_ROUTE,
            path_params={"session_id": self._resource_id(session_id)},
            read_only=True,
        )

    def confirm_once(self, session_id: str) -> JsonObject:
        current = self.get_session(session_id)
        session = self._session_object(current)
        if session.get("status") != "awaiting_confirmation":
            self._raise_session(
                "当前会话不在 awaiting_confirmation 状态。",
                detail={"status": session.get("status")},
            )
        if session.get("confirmation_ready") is not True:
            self._raise_session("当前会话没有可消费的动作确认作用域。")
        scope = session.get("confirmation_scope")
        if not isinstance(scope, dict):
            self._raise_session("当前会话的动作确认作用域缺失或格式无效。")
        if scope.get("session_id") != session_id:
            self._raise_session("动作确认作用域与当前 session_id 不一致。")
        return self._request(
            "POST",
            self.CONFIRM_ROUTE,
            path_params={"session_id": self._resource_id(session_id)},
            payload={"confirmed": True, "confirmation": dict(scope)},
            read_only=False,
        )

    def plan_next(self, session_id: str) -> JsonObject:
        current = self.get_session(session_id)
        session = self._session_object(current)
        device_id = self._required_string(session, "device_id", "会话缺少 device_id。")
        return self._request(
            "POST",
            self.NEXT_ROUTE,
            path_params={"session_id": self._resource_id(session_id)},
            payload={"device_id": device_id},
            read_only=False,
        )

    def continue_automatic(self, session_id: str, *, max_physical_actions: int | None = None,
        max_observations: int | None = None) -> JsonObject:
        current = self.get_session(session_id)
        session = self._session_object(current)
        payload: JsonObject = {"device_id": self._required_string(session, "device_id", "会话缺少 device_id。")}
        if max_physical_actions is not None:
            payload["max_physical_actions"] = max_physical_actions
        if max_observations is not None:
            payload["max_observations"] = max_observations
        return self._request("POST", self.AUTO_ROUTE,
            path_params={"session_id": self._resource_id(session_id)}, payload=payload, read_only=False)

    def cancel_session(self, session_id: str) -> JsonObject:
        return self._device_session_post(self.CANCEL_ROUTE, session_id)

    def reconcile_session(
        self,
        session_id: str,
        *,
        resume: bool = False,
        confirmed: bool = True,
        external_effect: str = "not_dispatched",
    ) -> JsonObject:
        current = self.get_session(session_id)
        session = self._session_object(current)
        device_id = self._required_string(session, "device_id", "会话缺少 device_id。")
        return self._request(
            "POST",
            self.RECONCILE_ROUTE,
            path_params={"session_id": self._resource_id(session_id)},
            payload={
                "device_id": device_id,
                "external_effect": external_effect,
                "confirmed": confirmed,
                "resume": resume,
            },
            read_only=False,
        )

    def pause_session(self, session_id: str) -> JsonObject:
        return self._device_session_post(self.PAUSE_ROUTE, session_id)

    def _device_session_post(self, route: str, session_id: str) -> JsonObject:
        current = self.get_session(session_id)
        session = self._session_object(current)
        device_id = self._required_string(session, "device_id", "会话缺少 device_id。")
        return self._request(
            "POST",
            route,
            path_params={"session_id": self._resource_id(session_id)},
            payload={"device_id": device_id},
            read_only=False,
        )

    def _request(
        self,
        method: str,
        route_template: str,
        *,
        path_params: Mapping[str, str] | None = None,
        query_params: Mapping[str, str] | None = None,
        extra_headers: Mapping[str, str] | None = None,
        payload: JsonObject | None = None,
        read_only: bool,
    ) -> JsonObject:
        self.bootstrap()
        operation = self._operation(method, route_template)
        self._validate_request_body(operation, payload)
        concrete_path = route_template
        for name, value in (path_params or {}).items():
            concrete_path = concrete_path.replace("{" + name + "}", quote(value, safe=""))
        if "{" in concrete_path or "}" in concrete_path:
            self._raise_contract(
                "接口路径参数没有完整绑定。",
                method=method,
                path=route_template,
            )
        return self._raw_json(
            method,
            concrete_path,
            payload=payload,
            query_params=query_params,
            extra_headers=extra_headers,
            token=self._token,
            read_only=read_only,
        )

    def _operation(self, method: str, route_template: str) -> JsonObject:
        assert self._openapi is not None
        paths = self._openapi["paths"]
        path_item = paths.get(route_template)
        operation = path_item.get(method.lower()) if isinstance(path_item, dict) else None
        if not isinstance(operation, dict):
            self._raise_contract(
                "当前运行服务未公开该方法和路径。",
                method=method,
                path=route_template,
            )
        return operation

    def _validate_request_body(
        self, operation: Mapping[str, Any], payload: JsonObject | None
    ) -> None:
        request_body = operation.get("requestBody")
        if request_body is None:
            if payload is not None:
                self._raise_contract("该接口不接受 JSON 请求体。")
            return
        if payload is None:
            self._raise_contract("该接口要求 JSON 请求体。")
        content = request_body.get("content") if isinstance(request_body, dict) else None
        media = content.get("application/json") if isinstance(content, dict) else None
        schema = media.get("schema") if isinstance(media, dict) else None
        if not isinstance(schema, dict):
            self._raise_contract("OpenAPI 未提供 JSON 请求 schema。")
        self._validate_schema(payload, schema, "request")

    def _validate_schema(self, value: Any, schema: Mapping[str, Any], path: str) -> None:
        schema = self._resolve_schema(schema)
        any_of = schema.get("anyOf")
        if isinstance(any_of, list):
            for candidate in any_of:
                try:
                    self._validate_schema(value, candidate, path)
                    return
                except LocalAgentApiError:
                    continue
            self._raise_contract(f"{path} 不符合任何允许的 schema。")
        expected = schema.get("type")
        if expected == "null":
            if value is not None:
                self._raise_contract(f"{path} 必须为 null。")
            return
        if expected == "object":
            if not isinstance(value, dict):
                self._raise_contract(f"{path} 必须是对象。")
            properties = schema.get("properties") or {}
            required = schema.get("required") or []
            missing = [name for name in required if name not in value]
            if missing:
                self._raise_contract(f"{path} 缺少字段：{', '.join(missing)}。")
            if schema.get("additionalProperties") is False:
                extras = sorted(set(value) - set(properties))
                if extras:
                    self._raise_contract(f"{path} 包含额外字段：{', '.join(extras)}。")
            for name, item in value.items():
                child = properties.get(name)
                if isinstance(child, dict):
                    self._validate_schema(item, child, f"{path}.{name}")
            return
        if expected == "array":
            if not isinstance(value, list):
                self._raise_contract(f"{path} 必须是数组。")
            item_schema = schema.get("items")
            if isinstance(item_schema, dict):
                for index, item in enumerate(value):
                    self._validate_schema(item, item_schema, f"{path}[{index}]")
            return
        if expected == "string":
            if not isinstance(value, str):
                self._raise_contract(f"{path} 必须是字符串。")
            minimum = schema.get("minLength")
            maximum = schema.get("maxLength")
            if isinstance(minimum, int) and len(value) < minimum:
                self._raise_contract(f"{path} 长度小于 {minimum}。")
            if isinstance(maximum, int) and len(value) > maximum:
                self._raise_contract(f"{path} 长度超过 {maximum}。")
            return
        if expected == "boolean":
            if type(value) is not bool:
                self._raise_contract(f"{path} 必须是布尔值。")
            return
        if expected == "integer":
            if type(value) is not int:
                self._raise_contract(f"{path} 必须是整数。")
            minimum = schema.get("minimum")
            maximum = schema.get("maximum")
            if isinstance(minimum, (int, float)) and value < minimum:
                self._raise_contract(f"{path} 小于允许值 {minimum}。")
            if isinstance(maximum, (int, float)) and value > maximum:
                self._raise_contract(f"{path} 大于允许值 {maximum}。")

    def _resolve_schema(self, schema: Mapping[str, Any]) -> Mapping[str, Any]:
        ref = schema.get("$ref")
        if ref is None:
            return schema
        if not isinstance(ref, str) or not ref.startswith("#/components/schemas/"):
            self._raise_contract("OpenAPI 包含不支持的 schema 引用。")
        assert self._openapi is not None
        name = ref.rsplit("/", 1)[-1]
        schemas = ((self._openapi.get("components") or {}).get("schemas") or {})
        resolved = schemas.get(name)
        if not isinstance(resolved, dict):
            self._raise_contract(f"OpenAPI 缺少 schema：{name}。")
        return resolved

    def _raw_json(
        self,
        method: str,
        path: str,
        *,
        payload: JsonObject | None = None,
        query_params: Mapping[str, str] | None = None,
        extra_headers: Mapping[str, str] | None = None,
        token: str = "",
        read_only: bool,
    ) -> JsonObject:
        headers = {"X-Control-Token": token} if token else {}
        headers.update(dict(extra_headers or {}))
        try:
            kwargs: dict[str, Any] = {"headers": headers}
            if payload is not None:
                kwargs["json"] = payload
            if query_params:
                kwargs["params"] = dict(query_params)
            response = self._client.request(method, path, **kwargs)
        except httpx.TimeoutException as exc:
            raise LocalAgentApiError(
                ApiErrorDetails(
                    category="network_timeout",
                    message="本地 Agent API 请求超时。",
                    method=method,
                    path=path,
                    retryable=read_only,
                )
            ) from exc
        except httpx.TransportError as exc:
            raise LocalAgentApiError(
                ApiErrorDetails(
                    category="network_error",
                    message="无法连接本地 Agent API。",
                    method=method,
                    path=path,
                    retryable=read_only,
                    detail=type(exc).__name__,
                )
            ) from exc
        if response.status_code >= 400:
            try:
                error_payload: Any = response.json()
            except ValueError:
                error_payload = response.text[:500]
            category = self._classify_http_error(response.status_code, error_payload)
            raise LocalAgentApiError(
                ApiErrorDetails(
                    category=category,
                    message=f"本地 Agent API 返回 HTTP {response.status_code}。",
                    method=method,
                    path=path,
                    status_code=response.status_code,
                    retryable=False,
                    detail=error_payload,
                )
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise LocalAgentApiError(
                ApiErrorDetails(
                    category="service_error",
                    message="本地 Agent API 返回了非 JSON 响应。",
                    method=method,
                    path=path,
                    status_code=response.status_code,
                )
            ) from exc
        if not isinstance(data, dict):
            raise LocalAgentApiError(
                ApiErrorDetails(
                    category="service_error",
                    message="本地 Agent API 响应不是对象。",
                    method=method,
                    path=path,
                    status_code=response.status_code,
                )
            )
        return data

    @staticmethod
    def _classify_http_error(status_code: int, payload: Any) -> str:
        text = str(payload).lower()
        if status_code in {403, 405, 422}:
            return "client_contract_error"
        if status_code == 404:
            return "resource_not_found"
        if any(word in text for word in ("deepseek", "qwen", "模型", "视觉")):
            return "model_error"
        if any(word in text for word in ("设备", "相机", "控制器", "机械臂")):
            return "device_error"
        if any(word in text for word in ("验证", "不匹配", "mismatch")):
            return "verification_error"
        return "service_error"

    def _session_object(self, payload: Mapping[str, Any]) -> JsonObject:
        session = payload.get("session")
        if not isinstance(session, dict):
            raise LocalAgentApiError(
                ApiErrorDetails(
                    category="service_contract_error",
                    message="会话接口响应缺少 session 对象。",
                )
            )
        return session

    @staticmethod
    def _required_string(source: Mapping[str, Any], key: str, message: str) -> str:
        value = source.get(key)
        if not isinstance(value, str) or not value:
            raise LocalAgentApiError(
                ApiErrorDetails(category="service_contract_error", message=message)
            )
        return value

    def _resource_id(self, value: str) -> str:
        if not isinstance(value, str) or not value or len(value) > 128:
            self._raise_contract("session_id 必须是 1 到 128 个字符。")
        return value

    def _raise_contract(
        self,
        message: str,
        *,
        method: str = "",
        path: str = "",
    ) -> None:
        raise LocalAgentApiError(
            ApiErrorDetails(
                category="client_contract_error",
                message=message,
                method=method,
                path=path,
            )
        )

    @staticmethod
    def _raise_session(message: str, *, detail: Any = None) -> None:
        raise LocalAgentApiError(
            ApiErrorDetails(
                category="session_state_error",
                message=message,
                detail=detail,
            )
        )
