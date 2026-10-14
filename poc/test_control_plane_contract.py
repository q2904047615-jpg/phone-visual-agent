from __future__ import annotations

import unittest

from agent.domain import (
    API_CONTRACT_VERSION, ApiError, ApiRequestEnvelope, ApiResponseEnvelope,
    ControlPlaneOrganization, ControlPlanePrincipal, ControlPlaneSnapshot, ControlPlaneTeam,
    QuotaLimit, QuotaUsage, SdkContractManifest, evaluate_quota,
)


def _snapshot() -> ControlPlaneSnapshot:
    owner = ControlPlanePrincipal("user-owner", "org-demo", "owner", ("team-core",))
    operator = ControlPlanePrincipal("user-operator", "org-demo", "operator", ("team-core",))
    team = ControlPlaneTeam("team-core", "org-demo", "Core", ("user-owner", "user-operator"))
    organization = ControlPlaneOrganization("org-demo", "Demo", "user-owner", ("team-core",))
    return ControlPlaneSnapshot(organization, (team,), (owner, operator), {
        "org-demo": QuotaLimit("monthly", max_runs=10, max_physical_actions=100, max_model_tokens=10000),
    })


class ControlPlaneContractTests(unittest.TestCase):
    def test_organization_team_and_role_permissions_validate(self) -> None:
        snapshot = _snapshot()
        snapshot.validate()
        owner = snapshot.principals[0]
        operator = snapshot.principals[1]
        self.assertTrue(owner.can("billing.read"))
        self.assertTrue(operator.can("task.create"))
        self.assertFalse(operator.can("team.manage"))

    def test_snapshot_rejects_cross_organization_membership(self) -> None:
        snapshot = _snapshot()
        outsider = ControlPlanePrincipal("user-outsider", "org-other", "viewer")
        invalid = ControlPlaneSnapshot(snapshot.organization, snapshot.teams, snapshot.principals + (outsider,), {})
        with self.assertRaisesRegex(ValueError, "跨组织"):
            invalid.validate()

    def test_quota_decision_is_pure_and_reports_remaining(self) -> None:
        limit = QuotaLimit("daily", max_runs=2, max_physical_actions=10, max_model_tokens=100)
        usage = QuotaUsage(runs=1, physical_actions=8, model_tokens=90)
        allowed = evaluate_quota(limit, usage, requested_runs=1, requested_physical_actions=2)
        self.assertTrue(allowed.allowed)
        self.assertEqual(0, allowed.remaining["runs"])
        self.assertEqual(0, allowed.remaining["physical_actions"])
        self.assertEqual(10, allowed.remaining["model_tokens"])
        rejected = evaluate_quota(limit, usage, requested_model_tokens=11)
        self.assertFalse(rejected.allowed)
        self.assertIn("model_tokens", rejected.reason)
        self.assertEqual(usage, QuotaUsage(runs=1, physical_actions=8, model_tokens=90))

    def test_api_write_envelope_and_response_error_are_versioned(self) -> None:
        request = ApiRequestEnvelope("request-1", "idem-1", "org-demo", "user-owner", "task.create", {"goal": "read"})
        request.validate()
        response = ApiResponseEnvelope("request-1", 202, data={"task_id": "task-1"})
        response.validate()
        failed = ApiResponseEnvelope("request-1", 409, error=ApiError("device_busy", "device is busy", retryable=True))
        failed.validate()
        with self.assertRaisesRegex(ValueError, "失败响应必须携带 error"):
            ApiResponseEnvelope("request-1", 409).validate()
        self.assertEqual(API_CONTRACT_VERSION, request.api_version)

    def test_sdk_manifest_declares_api_and_idempotency_contract(self) -> None:
        manifest = SdkContractManifest(
            "phone-agent-python", "0.1.0", API_CONTRACT_VERSION,
            ("task.create", "task.read", "task.cancel"), ("device_busy", "quota_exceeded"),
        )
        manifest.validate()

    def test_contract_rejects_duplicate_team_members_and_unknown_role(self) -> None:
        with self.assertRaisesRegex(ValueError, "member_ids"):
            ControlPlaneTeam("team", "org", "Team", ("user", "user"))
        with self.assertRaisesRegex(ValueError, "未知角色"):
            ControlPlanePrincipal("user", "org", "superadmin")


if __name__ == "__main__":
    unittest.main()
