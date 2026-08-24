from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from qwen_runtime_errors import classify_qwen_error
from vision_agent import DashScopeVisionProvider
from vision_usage import (
    VisionModelBudgetExceeded,
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
        self.assertEqual(16, payload["budget"]["max_model_requests"])
        self.assertEqual(50_000, payload["budget"]["max_total_tokens"])
        self.assertEqual(7, payload["budget"]["target_model_requests"])
        self.assertEqual(30_000, payload["budget"]["target_total_tokens"])
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
        self.assertFalse(payload["totals"]["target_request_count_exceeded"])
        self.assertFalse(payload["totals"]["target_token_count_exceeded"])
        self.assertFalse(payload["totals"]["budget_exhausted"])
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

    def test_request_and_token_budget_stop_before_network(self) -> None:
        request_limited = VisionSessionUsageLedger(
            session_id="request-limited",
            max_model_requests=2,
            target_model_requests=1,
        )
        for index in range(2):
            local_id = request_limited.reserve_request(
                model="qwen3.7-plus",
                stage="single_step_observation",
                fingerprint="same",
                max_completion_tokens=100,
            )
            request_limited.record_failure(
                local_id,
                network_attempts=0,
                error="offline fixture",
            )
        with self.assertRaisesRegex(
            VisionModelBudgetExceeded,
            "model_budget_exhausted",
        ):
            request_limited.reserve_request(
                model="qwen3.7-plus",
                stage="single_step_observation",
                fingerprint="same",
                max_completion_tokens=100,
            )
        self.assertEqual(
            2,
            request_limited.to_dict()["totals"]["model_requests"],
        )

        token_limited = VisionSessionUsageLedger(
            session_id="token-limited",
            max_total_tokens=10,
            target_total_tokens=5,
        )
        local_id = token_limited.reserve_request(
            model="qwen3.7-plus",
            stage="single_step_observation",
            fingerprint="frame",
            max_completion_tokens=5,
        )
        token_limited.record_success(
            local_id,
            provider_request_id="provider-token",
            response_model="qwen3.7-plus",
            network_attempts=1,
            usage={"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10},
            finish_reason="stop",
        )
        with self.assertRaisesRegex(
            VisionModelBudgetExceeded,
            "model_budget_exhausted",
        ):
            token_limited.reserve_request(
                model="qwen3.7-plus",
                stage="single_step_observation",
                fingerprint="frame-b",
                max_completion_tokens=5,
            )

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
        self.assertEqual(1, payload["totals"]["budget_rejections"])

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

    def test_runtime_error_classifier_keeps_budget_separate_from_policy(self) -> None:
        self.assertEqual(
            "model_budget_exhausted",
            classify_qwen_error("model_budget_exhausted: stop before request"),
        )
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
