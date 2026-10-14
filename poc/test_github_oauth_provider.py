from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import tempfile
import unittest
from urllib.parse import parse_qs, urlparse
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

import web_app
from agent.infrastructure.github_oauth_provider import (
    ExternalIdentityMappingNotFound,
    ExternalIdentityProviderUnavailable,
    GitHubIdentityMapping,
    GitHubOAuthConfigurationError,
    GitHubOAuthProvider,
)
from agent.infrastructure.sqlite_control_plane_repository import SqliteControlPlaneRepository


def _valid_payload(*, client_id: str = "client-123", user_id: int = 42, expires_at: str | None = None) -> dict:
    return {
        "app": {"client_id": client_id, "name": "test app"},
        "user": {"id": user_id, "login": "octocat"},
        "scopes": ["read:user"],
        **({"expires_at": expires_at} if expires_at is not None else {}),
    }


class GitHubOAuthProviderTests(unittest.TestCase):
    def test_check_token_uses_fixed_endpoint_and_maps_stable_user_id(self) -> None:
        seen: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["method"] = request.method
            seen["path"] = request.url.path
            seen["auth"] = request.headers.get("authorization")
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=_valid_payload(), request=request)

        provider = GitHubOAuthProvider(
            client_id="client-123",
            client_secret="secret-value",
            mappings={"42": GitHubIdentityMapping("local-org", "local-owner")},
            transport=httpx.MockTransport(handler),
        )
        try:
            identity = provider.authenticate({"access_token": "gho-test-token"})
        finally:
            provider.close()
        self.assertEqual("github", identity.provider_id)
        self.assertEqual("42", identity.subject)
        self.assertEqual("local-org", identity.organization_id)
        self.assertEqual("local-owner", identity.user_id)
        self.assertEqual("POST", seen["method"])
        self.assertEqual("/applications/client-123/token", seen["path"])
        expected_auth = "Basic " + base64.b64encode(b"client-123:secret-value").decode("ascii")
        self.assertEqual(expected_auth, seen["auth"])
        self.assertEqual({"access_token": "gho-test-token"}, seen["body"])

    def test_invalid_expired_wrong_app_and_unmapped_subject_are_rejected(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            if body["access_token"] == "expired":
                return httpx.Response(
                    200,
                    json=_valid_payload(expires_at=(datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()),
                    request=request,
                )
            if body["access_token"] == "wrong-app":
                return httpx.Response(200, json=_valid_payload(client_id="other-client"), request=request)
            if body["access_token"] == "unmapped":
                return httpx.Response(200, json=_valid_payload(user_id=99), request=request)
            return httpx.Response(404, json={"message": "Not Found"}, request=request)

        provider = GitHubOAuthProvider(
            client_id="client-123",
            client_secret="secret-value",
            mappings={"42": GitHubIdentityMapping("local-org", "local-owner")},
            transport=httpx.MockTransport(handler),
        )
        try:
            with self.assertRaises(ValueError):
                provider.authenticate({"access_token": "expired"})
            with self.assertRaises(ValueError):
                provider.authenticate({"access_token": "wrong-app"})
            with self.assertRaises(ExternalIdentityMappingNotFound):
                provider.authenticate({"access_token": "unmapped"})
            with self.assertRaises(ValueError):
                provider.authenticate({"access_token": "invalid"})
            with self.assertRaises(ValueError):
                provider.authenticate({"access_token": "valid", "organization_id": "forged"})
        finally:
            provider.close()

    def test_upstream_failures_and_environment_configuration_fail_closed(self) -> None:
        def outage(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, json={"message": "busy"})

        provider = GitHubOAuthProvider(
            client_id="client-123",
            client_secret="secret-value",
            mappings={"42": GitHubIdentityMapping("local-org", "local-owner")},
            transport=httpx.MockTransport(outage),
        )
        try:
            with self.assertRaises(ExternalIdentityProviderUnavailable):
                provider.authenticate({"access_token": "token"})
        finally:
            provider.close()
        with self.assertRaises(GitHubOAuthConfigurationError):
            GitHubOAuthProvider.from_environment({"GITHUB_OAUTH_CLIENT_ID": "client-123"})

        with tempfile.TemporaryDirectory() as directory:
            mapping_path = Path(directory) / "github.json"
            mapping_path.write_text(json.dumps({"42": {"organization_id": "local-org", "user_id": "local-owner"}}), encoding="utf-8")
            provider = GitHubOAuthProvider.from_environment({
                "GITHUB_OAUTH_CLIENT_ID": "client-123",
                "GITHUB_OAUTH_CLIENT_SECRET": "secret-value",
                "GITHUB_OAUTH_IDENTITY_MAP_FILE": str(mapping_path),
            }, transport=httpx.MockTransport(lambda request: httpx.Response(404, request=request)))
            self.assertIsNotNone(provider)
            provider.close()

    def test_authorization_url_and_code_exchange_use_pkce_and_fixed_redirect(self) -> None:
        seen: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["path"] = request.url.path
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json={"access_token": "gho-exchanged"}, request=request)

        provider = GitHubOAuthProvider(
            client_id="client-123",
            client_secret="secret-value",
            mappings={"42": GitHubIdentityMapping("local-org", "local-owner")},
            transport=httpx.MockTransport(handler),
        )
        verifier = "v" * 64
        state = "state-value"
        try:
            url = provider.authorization_url(state=state, code_challenge=provider.pkce_challenge(verifier))
            query = parse_qs(urlparse(url).query)
            self.assertEqual(["client-123"], query["client_id"])
            self.assertEqual(["S256"], query["code_challenge_method"])
            self.assertEqual([state], query["state"])
            self.assertEqual([provider.redirect_uri], query["redirect_uri"])
            self.assertEqual("gho-exchanged", provider.exchange_code(code="authorization-code", code_verifier=verifier))
        finally:
            provider.close()
        self.assertEqual("/login/oauth/access_token", seen["path"])
        self.assertEqual("authorization-code", seen["body"]["code"])
        self.assertEqual(verifier, seen["body"]["code_verifier"])
        self.assertEqual("secret-value", seen["body"]["client_secret"])


class GitHubOAuthControlPlaneRegressionTests(unittest.TestCase):
    def test_mapped_identity_issues_session_and_mapping_miss_is_audited(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = SqliteControlPlaneRepository(Path(directory) / "runtime.sqlite3")

            def handler(request: httpx.Request) -> httpx.Response:
                token = json.loads(request.content)["access_token"]
                if token == "unmapped":
                    return httpx.Response(200, json=_valid_payload(user_id=99), request=request)
                return httpx.Response(200, json=_valid_payload(), request=request)

            provider = GitHubOAuthProvider(
                client_id="client-123",
                client_secret="secret-value",
                mappings={"42": GitHubIdentityMapping("local-org", "local-owner")},
                transport=httpx.MockTransport(handler),
            )
            registry = web_app.ExternalIdentityProviderRegistry()
            registry.register(provider)
            with patch.object(web_app.runtime, "control_plane_repository", repository), \
                 patch.object(web_app, "control_plane_identity_registry", registry), \
                 patch.dict(os.environ, {"ROBOT_WEB_NO_BROWSER": "1"}):
                with TestClient(web_app.app) as client:
                    issued = client.post(
                        "/api/control-plane/auth/external-sessions",
                        headers={"Origin": "http://127.0.0.1"},
                        json={"provider_id": "github", "assertion": {"access_token": "valid"}},
                    )
                    self.assertEqual(200, issued.status_code, issued.text)
                    session = issued.json()["data"]["session"]
                    self.assertEqual("local-owner", session["user_id"])
                    self.assertNotIn("access_token", issued.text)
                    unmapped = client.post(
                        "/api/control-plane/auth/external-sessions",
                        headers={"Origin": "http://127.0.0.1"},
                        json={"provider_id": "github", "assertion": {"access_token": "unmapped"}},
                    )
                    self.assertEqual(404, unmapped.status_code, unmapped.text)
            try:
                audit = repository.read_audit("local-org", limit=100)["entries"]
                self.assertTrue(any(item["resource"] == "identity.external" and item["outcome"] == "issued" for item in audit))
                unknown_audit = repository.read_audit("unknown", limit=100)["entries"]
                self.assertTrue(any(item["outcome"] == "mapping_not_found" for item in unknown_audit))
            finally:
                repository.close()

    def test_browser_authorization_code_flow_consumes_state_and_issues_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = SqliteControlPlaneRepository(Path(directory) / "runtime.sqlite3")
            seen: dict[str, object] = {}

            def handler(request: httpx.Request) -> httpx.Response:
                payload = json.loads(request.content)
                if request.url.host == "github.com":
                    seen["exchange"] = payload
                    return httpx.Response(200, json={"access_token": "gho-browser-token"}, request=request)
                seen["check"] = payload
                return httpx.Response(200, json=_valid_payload(), request=request)

            provider = GitHubOAuthProvider(
                client_id="client-123",
                client_secret="secret-value",
                mappings={"42": GitHubIdentityMapping("local-org", "local-owner")},
                transport=httpx.MockTransport(handler),
            )
            registry = web_app.ExternalIdentityProviderRegistry()
            registry.register(provider)
            with patch.object(web_app.runtime, "control_plane_repository", repository), \
                 patch.object(web_app, "control_plane_identity_registry", registry), \
                 patch.dict(os.environ, {"ROBOT_WEB_NO_BROWSER": "1"}):
                with TestClient(web_app.app) as client:
                    start = client.get(
                        "/api/control-plane/auth/github/start",
                        headers={"Origin": "http://127.0.0.1", "X-Control-Token": web_app.CONTROL_TOKEN},
                    )
                    self.assertEqual(200, start.status_code, start.text)
                    start_data = start.json()["data"]
                    query = parse_qs(urlparse(start_data["authorization_url"]).query)
                    self.assertEqual([provider.redirect_uri], query["redirect_uri"])
                    state = query["state"][0]
                    callback = client.get(
                        "/api/control-plane/auth/github/callback",
                        params={"code": "authorization-code", "state": state},
                    )
                    self.assertEqual(200, callback.status_code, callback.text)
                    self.assertEqual("local-owner", callback.json()["data"]["session"]["user_id"])
                    self.assertNotIn("access_token", callback.text)
                    replay = client.get(
                        "/api/control-plane/auth/github/callback",
                        params={"code": "authorization-code", "state": state},
                    )
                    self.assertEqual(400, replay.status_code, replay.text)
            try:
                exchange = seen["exchange"]
                query_challenge = query["code_challenge"][0]
                self.assertEqual(query_challenge, provider.pkce_challenge(exchange["code_verifier"]))
                self.assertEqual({"access_token": "gho-browser-token"}, seen["check"])
                audit = repository.read_audit("local-org", limit=100)["entries"]
                self.assertTrue(any(item["resource"] == "identity.external.oauth" and item["outcome"] == "issued" for item in audit))
            finally:
                repository.close()
                provider.close()


if __name__ == "__main__":
    unittest.main()
