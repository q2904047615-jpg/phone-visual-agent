from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import web_app
from agent.infrastructure.sqlite_control_plane_repository import SqliteControlPlaneRepository


class ControlPlaneApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.no_browser = patch.dict("os.environ", {"ROBOT_WEB_NO_BROWSER": "1"})
        cls.no_browser.start()
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.repository = SqliteControlPlaneRepository(Path(cls.temp_dir.name) / "runtime.sqlite3")
        cls.repository_patch = patch.object(web_app.runtime, "control_plane_repository", cls.repository)
        cls.repository_patch.start()
        cls.client_context = TestClient(web_app.app)
        cls.client = cls.client_context.__enter__()
        cls.headers = {"X-Control-Token": web_app.CONTROL_TOKEN}

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client_context.__exit__(None, None, None)
        cls.repository_patch.stop()
        cls.repository.close()
        cls.temp_dir.cleanup()
        cls.no_browser.stop()

    def test_projection_routes_are_local_token_protected_and_read_only(self) -> None:
        paths = {
            "/api/control-plane/organizations/local-org",
            "/api/control-plane/organizations/local-org/teams",
            "/api/control-plane/organizations/local-org/principals",
            "/api/control-plane/organizations/local-org/quotas",
            "/api/control-plane/organizations/local-org/audit?limit=20",
        }
        for path in paths:
            with self.subTest(path=path):
                response = self.client.get(path, headers=self.headers)
                self.assertEqual(200, response.status_code, response.text)
                body = response.json()
                self.assertEqual("2026-10-14-control-plane-v1", body["control_plane_protocol"])
                self.assertEqual("2026-10-14-api-v1", body["api_version"])
        self.assertEqual(403, self.client.get("/api/control-plane/organizations/local-org").status_code)
        spec = self.client.get("/openapi.json").json()["paths"]
        control_plane_paths = {path: methods for path, methods in spec.items() if path.startswith("/api/control-plane/")}
        self.assertTrue(control_plane_paths)
        read_paths = {
            "/api/control-plane/organizations/{organization_id}",
            "/api/control-plane/organizations/{organization_id}/teams",
            "/api/control-plane/organizations/{organization_id}/principals",
            "/api/control-plane/organizations/{organization_id}/quotas",
            "/api/control-plane/organizations/{organization_id}/audit",
        }
        self.assertTrue(all("get" in set(control_plane_paths[path]) for path in read_paths))
        self.assertEqual({"post"}, set(control_plane_paths["/api/control-plane/auth/sessions"]))

    def test_audit_is_durable_and_redacts_sensitive_metadata(self) -> None:
        response = self.client.get(
            "/api/control-plane/organizations/local-org/teams",
            headers={**self.headers, "X-Request-ID": "cp-test-001"},
        )
        self.assertEqual(200, response.status_code)
        page = self.repository.read_audit("local-org", limit=20)
        self.assertTrue(page["entries"])
        entry = page["entries"][-1]
        self.assertEqual("cp-test-001", entry["request_id"])
        self.assertEqual("teams", entry["resource"])
        self.assertEqual("local-control-token", entry["actor"])
        self.assertNotIn("token", entry["metadata"])

    def test_unknown_organization_is_not_projected(self) -> None:
        response = self.client.get(
            "/api/control-plane/organizations/missing-org/teams",
            headers=self.headers,
        )
        self.assertEqual(404, response.status_code)

    def test_identity_session_and_idempotent_writes(self) -> None:
        auth = self.client.post(
            "/api/control-plane/auth/sessions",
            headers=self.headers,
            json={"organization_id": "local-org", "user_id": "local-owner"},
        )
        self.assertEqual(200, auth.status_code, auth.text)
        session = auth.json()["data"]["session"]
        principal_token = session["principal_token"]
        self.assertNotIn(principal_token, self.repository.database_path.read_bytes().decode("utf-8", errors="ignore"))
        base_headers = {
            **self.headers,
            "X-Control-Plane-Token": principal_token,
            "Idempotency-Key": "team-write-001",
        }
        payload = {"team_id": "team-write-001", "name": "Write Team", "member_ids": ["local-owner"]}
        created = self.client.post(
            "/api/control-plane/organizations/local-org/teams",
            headers=base_headers,
            json=payload,
        )
        self.assertEqual(201, created.status_code, created.text)
        replay = self.client.post(
            "/api/control-plane/organizations/local-org/teams",
            headers=base_headers,
            json=payload,
        )
        self.assertEqual(200, replay.status_code, replay.text)
        self.assertTrue(replay.json()["idempotent_replay"])
        conflict = self.client.post(
            "/api/control-plane/organizations/local-org/teams",
            headers=base_headers,
            json={"team_id": "team-write-002", "name": "Different", "member_ids": ["local-owner"]},
        )
        self.assertEqual(409, conflict.status_code, conflict.text)
        no_identity = self.client.post(
            "/api/control-plane/organizations/local-org/teams",
            headers={"X-Control-Token": web_app.CONTROL_TOKEN, "Idempotency-Key": "team-write-unauth"},
            json={"team_id": "team-write-003", "name": "No Identity", "member_ids": []},
        )
        self.assertEqual(401, no_identity.status_code)

        principal_headers = {**self.headers, "X-Control-Plane-Token": principal_token, "Idempotency-Key": "principal-write-001"}
        principal = self.client.post(
            "/api/control-plane/organizations/local-org/principals",
            headers=principal_headers,
            json={"user_id": "write-user-001", "role": "operator", "team_ids": ["team-write-001"]},
        )
        self.assertEqual(201, principal.status_code, principal.text)
        quota_headers = {**self.headers, "X-Control-Plane-Token": principal_token, "Idempotency-Key": "quota-write-001"}
        quota = self.client.put(
            "/api/control-plane/organizations/local-org/quotas/daily",
            headers=quota_headers,
            json={"quota_key": "daily", "period": "daily", "max_runs": 88},
        )
        self.assertEqual(200, quota.status_code, quota.text)


if __name__ == "__main__":
    unittest.main()
