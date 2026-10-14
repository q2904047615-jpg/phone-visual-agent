from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import tempfile
import unittest
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


if __name__ == "__main__":
    unittest.main()
