from __future__ import annotations

import json
import unittest

import httpx

try:
    from .local_agent_api_client import LocalAgentApiClient, LocalAgentApiError
except ImportError:  # Direct execution from the poc directory.
    from local_agent_api_client import LocalAgentApiClient, LocalAgentApiError


def _schema_object(properties, required=()):
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


def _openapi(*, version="0.2.0", include_start=True):
    string = {"type": "string", "minLength": 1, "maxLength": 128}
    scope = _schema_object(
        {
            "session_id": string,
            "task_id": string,
            "device_id": string,
            "revision": {"type": "integer", "minimum": 1},
            "subgoal_id": string,
            "effect_ids": {"type": "array", "items": {"type": "string"}},
            "observation_id": string,
            "fingerprint": {"type": "string", "minLength": 1, "maxLength": 256},
            "decision_node_id": string,
            "action_digest": {"type": "string", "minLength": 64, "maxLength": 64},
        },
        (
            "session_id",
            "task_id",
            "device_id",
            "revision",
            "subgoal_id",
            "observation_id",
            "fingerprint",
            "decision_node_id",
            "action_digest",
        ),
    )
    schemas = {
        "Start": _schema_object(
            {
                "text": {"type": "string", "minLength": 1, "maxLength": 500},
                "exact_input_text": {
                    "anyOf": [
                        {"type": "string", "minLength": 1, "maxLength": 4000},
                        {"type": "null"},
                    ]
                },
                "device_id": string,
                "auto_advance": {"type": "boolean"},
            },
            ("text", "device_id"),
        ),
        "Device": _schema_object({"device_id": string}, ("device_id",)),
        "Scope": scope,
        "Confirm": _schema_object(
            {
                "confirmed": {"type": "boolean"},
                "confirmation": {
                    "anyOf": [
                        {"$ref": "#/components/schemas/Scope"},
                        {"type": "null"},
                    ]
                },
            }
        ),
    }

    def body(ref):
        return {
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": {"$ref": ref}}},
            }
        }

    session_path = "/api/agent/generic-supervised/{session_id}"
    paths = {
        "/api/device": {"get": {}},
        session_path: {"get": {}},
        session_path + "/confirm": {
            "post": body("#/components/schemas/Confirm")
        },
        session_path + "/next": {"post": body("#/components/schemas/Device")},
        session_path + "/cancel": {"post": body("#/components/schemas/Device")},
        session_path + "/pause": {"post": body("#/components/schemas/Device")},
    }
    if include_start:
        paths["/api/agent/generic-supervised/start"] = {
            "post": body("#/components/schemas/Start")
        }
    return {
        "info": {"version": version},
        "paths": paths,
        "components": {"schemas": schemas},
    }


def _scope(**changes):
    value = {
        "session_id": "abc",
        "task_id": "task",
        "device_id": "device-local-01",
        "revision": 1,
        "subgoal_id": "append",
        "effect_ids": [],
        "observation_id": "obs",
        "fingerprint": "fp",
        "decision_node_id": "decision",
        "action_digest": "a" * 64,
    }
    value.update(changes)
    return value


class LocalAgentApiClientTests(unittest.TestCase):
    def _client(self, handler):
        return LocalAgentApiClient(transport=httpx.MockTransport(handler))

    def test_bootstrap_loads_token_without_exposing_it(self):
        def handler(request):
            if request.url.path == "/api/session":
                return httpx.Response(200, json={"token": "top-secret", "version": "0.2.0"})
            return httpx.Response(200, json=_openapi())

        with self._client(handler) as client:
            result = client.bootstrap()
        self.assertEqual(result["service_version"], "0.2.0")
        self.assertTrue(result["control_token_loaded"])
        self.assertNotIn("top-secret", str(result))

    def test_device_status_uses_runtime_token_and_verified_route(self):
        calls = []

        def handler(request):
            calls.append((request.method, request.url.path, request.headers.get("X-Control-Token")))
            if request.url.path == "/api/session":
                return httpx.Response(200, json={"token": "secret", "version": "0.2.0"})
            if request.url.path == "/openapi.json":
                return httpx.Response(200, json=_openapi())
            return httpx.Response(200, json={"controller_online": True})

        with self._client(handler) as client:
            result = client.device_status()
        self.assertTrue(result["controller_online"])
        self.assertEqual(calls[-1], ("GET", "/api/device", "secret"))

    def test_missing_openapi_route_blocks_before_target_request(self):
        target_calls = 0

        def handler(request):
            nonlocal target_calls
            if request.url.path == "/api/session":
                return httpx.Response(200, json={"token": "secret", "version": "0.2.0"})
            if request.url.path == "/openapi.json":
                return httpx.Response(200, json=_openapi(include_start=False))
            target_calls += 1
            return httpx.Response(500)

        with self._client(handler) as client:
            with self.assertRaises(LocalAgentApiError) as caught:
                client.start_session(text="目标", device_id="device-local-01")
        self.assertEqual(caught.exception.details.category, "client_contract_error")
        self.assertEqual(target_calls, 0)

    def test_version_mismatch_blocks_all_project_requests(self):
        def handler(request):
            if request.url.path == "/api/session":
                return httpx.Response(200, json={"token": "secret", "version": "0.2.0"})
            return httpx.Response(200, json=_openapi(version="0.3.0"))

        with self._client(handler) as client:
            with self.assertRaises(LocalAgentApiError) as caught:
                client.device_status()
        self.assertEqual(caught.exception.details.category, "service_contract_error")

    def test_start_validates_and_sends_exact_body_once(self):
        posts = []

        def handler(request):
            if request.url.path == "/api/session":
                return httpx.Response(200, json={"token": "secret", "version": "0.2.0"})
            if request.url.path == "/openapi.json":
                return httpx.Response(200, json=_openapi())
            posts.append((request.headers.get("X-Control-Token"), request.read().decode()))
            return httpx.Response(200, json={"session": {"session_id": "abc"}})

        with self._client(handler) as client:
            result = client.start_session(
                text="输入两个字符",
                exact_input_text="first\nsecond",
                device_id="device-local-01",
                auto_advance=False,
            )
        self.assertEqual(result["session"]["session_id"], "abc")
        self.assertEqual(len(posts), 1)
        self.assertEqual(posts[0][0], "secret")
        self.assertIn('"auto_advance":false', posts[0][1])
        self.assertIn('"exact_input_text":"first\\nsecond"', posts[0][1])

    def test_confirm_fetches_latest_scope_and_posts_it_once(self):
        confirms = []

        def handler(request):
            if request.url.path == "/api/session":
                return httpx.Response(200, json={"token": "secret", "version": "0.2.0"})
            if request.url.path == "/openapi.json":
                return httpx.Response(200, json=_openapi())
            if request.method == "GET":
                return httpx.Response(
                    200,
                    json={
                        "session": {
                            "session_id": "abc",
                            "device_id": "device-local-01",
                            "status": "awaiting_confirmation",
                            "confirmation_ready": True,
                            "confirmation_scope": _scope(),
                        }
                    },
                )
            confirms.append(json.loads(request.content.decode("utf-8")))
            return httpx.Response(200, json={"session": {"session_id": "abc"}})

        with self._client(handler) as client:
            client.confirm_once("abc")
        self.assertEqual(len(confirms), 1)
        self.assertTrue(confirms[0]["confirmed"])
        self.assertEqual(confirms[0]["confirmation"], _scope())

    def test_missing_scope_never_sends_confirmation(self):
        confirms = 0

        def handler(request):
            nonlocal confirms
            if request.url.path == "/api/session":
                return httpx.Response(200, json={"token": "secret", "version": "0.2.0"})
            if request.url.path == "/openapi.json":
                return httpx.Response(200, json=_openapi())
            if request.method == "GET":
                return httpx.Response(
                    200,
                    json={
                        "session": {
                            "session_id": "abc",
                            "status": "awaiting_confirmation",
                            "confirmation_ready": False,
                            "confirmation_scope": None,
                        }
                    },
                )
            confirms += 1
            return httpx.Response(500)

        with self._client(handler) as client:
            with self.assertRaises(LocalAgentApiError) as caught:
                client.confirm_once("abc")
        self.assertEqual(caught.exception.details.category, "session_state_error")
        self.assertEqual(confirms, 0)

    def test_scope_extra_field_is_rejected_before_confirmation(self):
        confirms = 0

        def handler(request):
            nonlocal confirms
            if request.url.path == "/api/session":
                return httpx.Response(200, json={"token": "secret", "version": "0.2.0"})
            if request.url.path == "/openapi.json":
                return httpx.Response(200, json=_openapi())
            if request.method == "GET":
                return httpx.Response(
                    200,
                    json={
                        "session": {
                            "session_id": "abc",
                            "status": "awaiting_confirmation",
                            "confirmation_ready": True,
                            "confirmation_scope": _scope(unexpected="bad"),
                        }
                    },
                )
            confirms += 1
            return httpx.Response(500)

        with self._client(handler) as client:
            with self.assertRaises(LocalAgentApiError) as caught:
                client.confirm_once("abc")
        self.assertEqual(caught.exception.details.category, "client_contract_error")
        self.assertEqual(confirms, 0)

    def test_mutating_timeout_is_not_retried(self):
        start_attempts = 0

        def handler(request):
            nonlocal start_attempts
            if request.url.path == "/api/session":
                return httpx.Response(200, json={"token": "secret", "version": "0.2.0"})
            if request.url.path == "/openapi.json":
                return httpx.Response(200, json=_openapi())
            start_attempts += 1
            raise httpx.ReadTimeout("timeout", request=request)

        with self._client(handler) as client:
            with self.assertRaises(LocalAgentApiError) as caught:
                client.start_session(text="目标", device_id="device-local-01")
        self.assertEqual(caught.exception.details.category, "network_timeout")
        self.assertFalse(caught.exception.details.retryable)
        self.assertEqual(start_attempts, 1)

    def test_token_error_is_classified_without_leaking_token(self):
        def handler(request):
            if request.url.path == "/api/session":
                return httpx.Response(200, json={"token": "secret", "version": "0.2.0"})
            if request.url.path == "/openapi.json":
                return httpx.Response(200, json=_openapi())
            return httpx.Response(403, json={"detail": "控制令牌无效。"})

        with self._client(handler) as client:
            with self.assertRaises(LocalAgentApiError) as caught:
                client.start_session(text="目标", device_id="device-local-01")
        details = caught.exception.details.to_dict()
        self.assertEqual(details["category"], "client_contract_error")
        self.assertNotIn("secret", str(details))


if __name__ == "__main__":
    unittest.main()
