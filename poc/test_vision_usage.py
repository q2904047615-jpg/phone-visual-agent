from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from qwen_runtime_errors import classify_qwen_error
from vision_agent import DashScopeVisionProvider
import vision_usage
from vision_usage import (
    VisionModelIdentityMismatch,
    VisionSessionUsageLedger,
    VisionStepContractViolation,
)


class VisionSessionUsageLedgerTests(unittest.TestCase):
    def test_six_action_topology_targets_initial_plus_one_call_per_action(self) -> None:
        planned_actions = 6
        planned_calls = 1 + planned_actions
        self.assertEqual(7, planned_calls)

        ledger = VisionSessionUsageLedger(session_id="six-action-topology")
        for index in range(planned_calls):
            local_id = ledger.reserve_request(
                model="qwen3.7-plus",
                stage="single_step_observation",
                fingerprint=f"frame-{index + 1}",
                max_completion_tokens=2600,
            )
            ledger.record_success(
                local_id,
                provider_request_id=f"provider-{index + 1}",
                response_model="qwen3.7-plus",
                network_attempts=1,
                usage={
                    "prompt_tokens": 2500,
                    "completion_tokens": 300,
                    "total_tokens": 2800,
                },
                finish_reason="stop",
            )

        payload = ledger.to_dict()
        self.assertEqual(
            "2026-08-25-single-step-qwen-usage-v4",
            payload["version"],
        )
        self.assertFalse(payload["session_limits_enforced"])
        self.assertNotIn("budget", payload)
        self.assertEqual(7, payload["totals"]["model_requests"])
        self.assertEqual(19_600, payload["totals"]["total_tokens"])
        self.assertEqual(
            {"single_step_observation"},
            {
                event["stage"]
                for event in payload["events"]
                if event.get("event") == "model_request"
            },
        )
        self.assertNotIn("remaining_model_requests", payload["totals"])
        self.assertNotIn("remaining_tokens", payload["totals"])
        self.assertEqual("qwen3.7-plus", payload["model"])
        self.assertFalse(payload["downgrade_allowed"])

    def test_request_tokens_cost_and_cache_hit_are_persisted(self) -> None:
        ledger = VisionSessionUsageLedger(session_id="session-usage")
        local_id = ledger.reserve_request(
            model="qwen3.7-plus",
            stage="single_step_observation",
            fingerprint="frame-a",
            max_completion_tokens=2600,
        )
        ledger.record_success(
            local_id,
            provider_request_id="provider-1",
            response_model="qwen3.7-plus",
            network_attempts=1,
            usage={
                "prompt_tokens": 1000,
                "completion_tokens": 100,
                "total_tokens": 1100,
                "prompt_tokens_details": {"cached_tokens": 250},
            },
            finish_reason="stop",
            elapsed_seconds=1.25,
        )
        ledger.record_cache_hit(stage="same_fingerprint", fingerprint="frame-a")

        payload = ledger.to_dict()
        self.assertEqual("qwen3.7-plus", payload["model"])
        self.assertFalse(payload["downgrade_allowed"])
        self.assertEqual(1, payload["totals"]["model_requests"])
        self.assertEqual(1, payload["totals"]["successful_requests"])
        self.assertEqual(1100, payload["totals"]["total_tokens"])
        self.assertEqual(1, payload["totals"]["observation_cache_hits"])
        self.assertEqual(0.0028, payload["totals"]["estimated_list_cost_cny"])
        self.assertEqual(
            0.00224,
            payload["totals"]["estimated_promotional_cost_cny"],
        )
        request = payload["events"][0]
        self.assertEqual("single_step_observation", request["stage"])
        self.assertEqual("provider-1", request["provider_request_id"])
        self.assertEqual(250, request["cached_prompt_tokens"])
        self.assertEqual(1.25, request["elapsed_seconds"])
        self.assertEqual(1, payload["totals"]["timed_requests"])
        self.assertEqual(1.25, payload["totals"]["average_elapsed_seconds"])
        self.assertEqual(1.25, payload["totals"]["max_elapsed_seconds"])

    def test_old_request_and_token_thresholds_never_block_next_step(self) -> None:
        ledger = VisionSessionUsageLedger(session_id="unbounded-observation")
        for index in range(17):
            local_id = ledger.reserve_request(
                model="qwen3.7-plus",
                stage="single_step_observation",
                fingerprint=f"frame-{index + 1}",
                max_completion_tokens=5200,
            )
            ledger.record_success(
                local_id,
                provider_request_id=f"provider-{index + 1}",
                response_model="qwen3.7-plus",
                network_attempts=1,
                usage={
                    "prompt_tokens": 2500,
                    "completion_tokens": 500,
                    "total_tokens": 3000,
                },
                finish_reason="stop",
            )
        next_request = ledger.reserve_request(
            model="qwen3.7-plus",
            stage="single_step_observation",
            fingerprint="frame-18",
            max_completion_tokens=5200,
        )
        ledger.record_failure(
            next_request,
            network_attempts=0,
            error="offline fixture after reservation",
        )

        payload = ledger.to_dict()
        self.assertEqual(18, payload["totals"]["model_requests"])
        self.assertEqual(51_000, payload["totals"]["total_tokens"])
        self.assertFalse(payload["session_limits_enforced"])

    def test_old_session_budget_gate_is_absent_from_runtime(self) -> None:
        source = Path(vision_usage.__file__).read_text(encoding="utf-8")
        old_error_code = "model_" + "budget_exhausted"
        self.assertNotIn(old_error_code, source)
        self.assertFalse(hasattr(vision_usage, "VisionModelBudgetExceeded"))

    def test_non_plus_model_is_rejected_without_fallback(self) -> None:
        ledger = VisionSessionUsageLedger(session_id="fixed-plus")
        with self.assertRaisesRegex(
            VisionModelIdentityMismatch,
            "vision_model_identity_mismatch",
        ):
            ledger.reserve_request(
                model="qwen3.6-flash",
                stage="compact",
                fingerprint="frame",
                max_completion_tokens=100,
            )
        payload = ledger.to_dict()
        self.assertEqual(0, payload["totals"]["model_requests"])
        self.assertEqual(1, payload["totals"]["identity_rejections"])

    def test_legacy_online_stage_is_rejected_before_any_model_request(self) -> None:
        ledger = VisionSessionUsageLedger(session_id="single-step-only")

        with self.assertRaisesRegex(
            VisionStepContractViolation,
            "vision_step_contract_violation",
        ):
            ledger.reserve_request(
                model="qwen3.7-plus",
                stage="visual_action_selection",
                fingerprint="frame-one",
                max_completion_tokens=100,
            )

        payload = ledger.to_dict()
        self.assertEqual(0, payload["totals"]["model_requests"])
        self.assertEqual(0, payload["totals"]["network_attempts"])
        self.assertEqual(1, payload["totals"]["contract_rejections"])

    def test_runtime_error_classifier_keeps_model_and_stage_errors_typed(self) -> None:
        self.assertEqual(
            "vision_model_identity_mismatch",
            classify_qwen_error("vision_model_identity_mismatch: fixed plus"),
        )
        self.assertEqual(
            "vision_step_contract_violation",
            classify_qwen_error(
                "vision_step_contract_violation: legacy stage rejected"
            ),
        )


class DashScopeUsageIntegrationTests(unittest.TestCase):
    def test_provider_records_stage_request_id_and_tokens(self) -> None:
        provider = DashScopeVisionProvider(
            api_key="test-key",
            max_attempts=1,
        )
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "id": "dashscope-request-1",
            "model": "qwen3.7-plus",
            "usage": {
                "prompt_tokens": 200,
                "completion_tokens": 20,
                "total_tokens": 220,
            },
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": '{"ok":true}'},
                }
            ],
        }
        ledger = VisionSessionUsageLedger(session_id="provider-session")
        with patch("vision_agent.httpx.post", return_value=response) as post:
            with provider.session_usage_scope(ledger):
                with provider.call_scope(
                    stage="single_step_observation",
                    fingerprint="frame-provider",
                ):
                    raw = provider._chat(
                        [{"role": "user", "content": "observe"}],
                        max_tokens=50,
                    )

        self.assertEqual('{"ok":true}', raw)
        self.assertEqual(1, post.call_count)
        event = ledger.to_dict()["events"][0]
        self.assertEqual("single_step_observation", event["stage"])
        self.assertEqual("frame-provider", event["fingerprint"])
        self.assertEqual("dashscope-request-1", event["provider_request_id"])
        self.assertEqual(220, event["total_tokens"])


if __name__ == "__main__":
    unittest.main()
