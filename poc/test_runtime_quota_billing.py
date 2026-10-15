from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from agent.infrastructure.sqlite_control_plane_repository import SqliteControlPlaneRepository


class _Session:
    session_id = "run-1"
    status = "cancelled"
    physical_actions = 2

    def snapshot(self):
        return {
            "qwen_usage": {
                "model": "qwen3-vl-plus",
                "pricing": {"version": "pricing-test"},
                "totals": {
                    "prompt_tokens": 3,
                    "completion_tokens": 4,
                    "total_tokens": 7,
                    "estimated_list_cost_cny": 0.00001,
                },
            }
        }


class RuntimeQuotaBillingTests(unittest.TestCase):
    def setUp(self):
        self.file = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False)
        self.file.close()
        self.path = Path(self.file.name)
        self.repo = SqliteControlPlaneRepository(self.path)

    def tearDown(self):
        self.repo.close()
        try:
            os.unlink(self.path)
        except OSError:
            pass

    def test_reserve_settle_and_billing_are_durable_and_idempotent(self):
        first = self.repo.reserve_runtime_quota(
            organization_id="local-org", user_id="local-owner", session_id="run-1",
            requested_physical_actions=3,
        )
        replay = self.repo.reserve_runtime_quota(
            organization_id="local-org", user_id="local-owner", session_id="run-1",
            requested_physical_actions=3,
        )
        self.assertEqual(first["reservation_id"], replay["reservation_id"])
        settled = self.repo.settle_runtime_usage(
            session_id="run-1", status="cancelled", physical_actions=2,
            prompt_tokens=3, completion_tokens=4, total_tokens=7,
            model="qwen3-vl-plus", cost_micros=10, pricing_version="pricing-test",
        )
        second = self.repo.settle_runtime_usage(
            session_id="run-1", status="cancelled", physical_actions=2,
            prompt_tokens=3, completion_tokens=4, total_tokens=7,
            model="qwen3-vl-plus", cost_micros=10, pricing_version="pricing-test",
        )
        self.assertFalse(settled["idempotent"])
        self.assertTrue(second["idempotent"])
        quota = next(item for item in self.repo.list_quotas("local-org") if item["quota_key"] == "daily")
        self.assertEqual({"runs": 1, "physical_actions": 2, "model_tokens": 7}, quota["usage"])
        self.assertEqual(1, len(self.repo.list_billing("local-org")))

    def test_reservation_rejects_concurrent_over_quota_and_release_restores_capacity(self):
        self.repo.upsert_quota(
            organization_id="local-org", principal_user_id="local-owner", idempotency_key="small-1",
            request_hash="small-1", quota_key="daily", period="daily", max_runs=1,
            max_physical_actions=1, max_model_tokens=100,
        )
        self.repo.reserve_runtime_quota(
            organization_id="local-org", user_id="local-owner", session_id="run-a",
            requested_physical_actions=1,
        )
        with self.assertRaisesRegex(RuntimeError, "quota_exceeded"):
            self.repo.reserve_runtime_quota(
                organization_id="local-org", user_id="local-owner", session_id="run-b",
                requested_physical_actions=1,
            )
        self.repo.release_runtime_quota("run-a", reason="test")
        self.repo.reserve_runtime_quota(
            organization_id="local-org", user_id="local-owner", session_id="run-c",
            requested_physical_actions=1,
        )

    def test_platform_release_plan_requires_no_active_sessions_and_commit_is_explicit(self):
        self.repo.ensure_platform_release("revision-a")
        self.repo.ensure_platform_release("revision-b")
        blocked = self.repo.plan_platform_rollback(target_revision="revision-a", active_session_ids=["run-1"])
        self.assertFalse(blocked["ready"])
        ready = self.repo.plan_platform_rollback(target_revision="revision-a")
        self.assertTrue(ready["ready"])
        applied = self.repo.commit_platform_rollback(target_revision="revision-a", reason="test")
        self.assertEqual("revision-a", applied["revision"])
        self.assertEqual("active", applied["status"])


if __name__ == "__main__":
    unittest.main()
