from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

from eval_qwen_visual_decision import (
    _evaluate_case,
    _build_report,
    _format_model_usage,
    _format_metrics,
    _prior_reference,
    _timeout_result,
    _write_report,
)
from agent.infrastructure.qwen_runtime_errors import (
    classify_qwen_error,
    looks_like_truncated_json,
)
from agent.domain.vision_model import VisionAgentError


ROOT = Path(__file__).resolve().parent


class EvalProvider:
    configured = True
    model = "fake-qwen"

    def __init__(self, *, failure: BaseException | None = None) -> None:
        self.failure = failure
        self.calls = 0

    def status(self) -> dict:
        return {"configured": True, "model": self.model}

    def _chat(self, messages, max_tokens, **kwargs) -> str:
        self.calls += 1
        if self.failure:
            raise self.failure
        return json.dumps(
            {
                "protocol_version": "2026-08-10-ui-scene-v2",
                "foreground_app_id": "generic_surface",
                "screen_id": "input_overlay",
                "summary": "输入弹层清晰可见",
                "elements": [],
                "overlays": ["input_overlay"],
                "stable": True,
                "confidence": 0.95,
                "fingerprint": "ignored_model_value",
            },
            ensure_ascii=False,
        )


def case(case_id: str = "case_1") -> dict:
    return {
        "id": case_id,
        "page_type": "generic_page",
        "accepted_statuses": ["blocked"],
        "task_context": {"task_id": f"task_{case_id}"},
    }


class QwenRuntimeErrorTests(unittest.TestCase):
    def test_disconnect_timeout_invalid_and_truncated_json_are_distinct(self) -> None:
        self.assertEqual(
            classify_qwen_error("Server disconnected without sending a response"),
            "service_disconnect",
        )
        self.assertEqual(classify_qwen_error("千问视觉请求连续1次超时"), "service_timeout")
        self.assertEqual(
            classify_qwen_error("模型返回的 JSON 无法解析", raw_response="not-json"),
            "invalid_json",
        )
        self.assertEqual(
            classify_qwen_error("模型返回的 JSON 无法解析", raw_response='{"status":'),
            "truncated_json",
        )
        self.assertTrue(looks_like_truncated_json('{"status":"action"'))
        self.assertFalse(looks_like_truncated_json('{"status":"action"}'))


class QwenOfflineReportTests(unittest.TestCase):
    def test_unconfirmed_risk_blocks_before_observation_or_decision_model(self) -> None:
        manifest_path = ROOT / "evals" / "qwen_visual_decision" / "cases.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        risk_case = next(
            item
            for item in manifest["cases"]
            if item["id"] == "text_submit_without_confirmation_blocked"
        )
        success = _evaluate_case(
            risk_case,
            str(manifest_path),
            1,
            "run_test",
            provider_factory=EvalProvider,
        )
        self.assertEqual(success["status"], "blocked")
        self.assertTrue(success["score"]["passed"])
        self.assertEqual(success["observation_diagnostics"]["model_calls"], 0)
        self.assertEqual(success["decision_diagnostics"]["model_calls"], 0)
        self.assertEqual(success["failure"]["stage"], "pre_observation_risk_gate")

        still_blocked = _evaluate_case(
            risk_case,
            str(manifest_path),
            1,
            "run_test",
            provider_factory=lambda: EvalProvider(
                failure=VisionAgentError(
                    "千问视觉连接连续1次中断：Server disconnected"
                )
            ),
        )
        self.assertTrue(still_blocked["score"]["passed"])
        self.assertEqual(still_blocked["observation_diagnostics"]["model_calls"], 0)
        self.assertEqual(still_blocked["decision_diagnostics"]["model_calls"], 0)

    def test_case_evaluation_exception_path_always_sets_score(self) -> None:
        manifest_path = ROOT / "evals" / "qwen_visual_decision" / "cases.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        launcher_case = next(
            item
            for item in manifest["cases"]
            if item["id"] == "launcher_text_icon_single_action"
        )
        failed = _evaluate_case(
            launcher_case,
            str(manifest_path),
            1,
            "run_test",
            provider_factory=lambda: EvalProvider(
                failure=VisionAgentError(
                    "千问视觉连接连续1次中断：Server disconnected"
                )
            ),
        )
        self.assertFalse(failed["score"]["passed"])
        self.assertEqual(failed["failure"]["error_type"], "service_disconnect")

    def test_timeout_results_produce_complete_report_for_every_case(self) -> None:
        cases = [case("case_1"), case("case_2")]
        results = [
            _timeout_result(
                cases[0],
                run_id="run_current",
                timeout_seconds=3.0,
                error_type="case_timeout",
            ),
            _timeout_result(
                cases[1],
                run_id="run_current",
                timeout_seconds=5.0,
                error_type="suite_timeout",
            ),
        ]
        report = _build_report(
            run_id="run_current",
            manifest_path=Path("cases.json"),
            model="fake-qwen",
            results=results,
            selected_case_ids=["case_1", "case_2"],
            current_target_case_ids=["case_1", "case_2"],
            started_at="2026-08-11T00:00:00+08:00",
            started_monotonic=time.perf_counter(),
            case_timeout_seconds=3.0,
            suite_timeout_seconds=5.0,
            resume_report=None,
        )
        self.assertEqual(report["report_status"], "complete")
        self.assertTrue(report["full_case_report_complete"])
        self.assertEqual(len(report["results"]), 2)
        self.assertEqual(report["passed"], 0)
        self.assertEqual(report["failed"], 2)
        self.assertEqual(
            {item["error_type"] for item in report["failures"]},
            {"case_timeout", "suite_timeout"},
        )
        self.assertEqual(report["case_outcome_rates"]["final_blocked_rate"], 1.0)

    def test_resume_reference_is_explicit_and_excluded_from_current_metrics(self) -> None:
        old = _timeout_result(
            case("case_old"),
            run_id="run_old",
            timeout_seconds=1.0,
            error_type="case_timeout",
        )
        old["score"] = {"passed": True, "reasons": []}
        referenced = _prior_reference(
            old,
            source_report=Path("old/report.json"),
            source_run_id="run_old",
        )
        self.assertEqual(referenced["result_origin"], "prior_run_reference")
        self.assertFalse(referenced["counted_in_current_metrics"])
        self.assertEqual(referenced["source_run_id"], "run_old")
        self.assertEqual(_format_metrics([referenced])["combined"]["attempted_count"], 0)

    def test_format_metrics_separate_observation_and_decision(self) -> None:
        result = {
            "result_origin": "current_run",
            "observation_diagnostics": {
                "model_calls": 2,
                "first_pass_success": False,
                "repair_retry_success": True,
            },
            "decision_diagnostics": {
                "model_calls": 1,
                "first_pass_success": True,
                "repair_retry_success": False,
            },
        }
        metrics = _format_metrics([result])
        self.assertEqual(metrics["combined"]["attempted_count"], 2)
        self.assertEqual(metrics["combined"]["first_pass_rate"], 0.5)
        self.assertEqual(metrics["combined"]["repair_retry_rate"], 0.5)

    def test_model_usage_sums_current_results_only(self) -> None:
        current = {
            "result_origin": "current_run",
            "model_usage": {
                "successful_model_calls": 2,
                "prompt_tokens": 300,
                "completion_tokens": 40,
                "total_tokens": 340,
            },
        }
        prior = {
            "result_origin": "prior_run_reference",
            "model_usage": {
                "successful_model_calls": 99,
                "prompt_tokens": 999,
                "completion_tokens": 999,
                "total_tokens": 1998,
            },
        }
        self.assertEqual(
            _format_model_usage([current, prior]),
            {
                "successful_model_calls": 2,
                "prompt_tokens": 300,
                "completion_tokens": 40,
                "total_tokens": 340,
            },
        )

    def test_report_write_is_atomic_and_valid_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            _write_report(path, {"report_status": "running", "results": []})
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8"))["report_status"],
                "running",
            )
            self.assertFalse(path.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
