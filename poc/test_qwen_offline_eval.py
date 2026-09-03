from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

from eval_qwen_visual_decision import (
    _evaluate_case,
    _build_report,
    _format_capability_metrics,
    _format_model_usage,
    _format_metrics,
    _load_frames,
    _load_manifest,
    _prior_reference,
    _score,
    _timeout_result,
    _write_report,
)
from agent.infrastructure.qwen_runtime_errors import (
    classify_qwen_error,
    looks_like_truncated_json,
)
from agent.domain.vision_model import VisionAgentError
from agent.domain.qwen_task_context import QwenTaskContext


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


class SuccessfulEvalProvider(EvalProvider):
    def _chat(self, messages, max_tokens, **kwargs) -> str:
        del messages, max_tokens, kwargs
        self.calls += 1
        return json.dumps(
            {
                "protocol_version": "2026-09-03-single-step-required-action-v10",
                "coordinate_space": {"kind": "axis_grid", "width": 1000, "height": 960},
                "scene": {
                    "protocol_version": "2026-08-10-ui-scene-v2",
                    "foreground_app_id": "launcher",
                    "screen_id": "home_screen",
                    "summary": "手机桌面清晰可见",
                    "system_ui": {
                        "immersive_or_fullscreen": "unknown",
                        "navigation_bar_visible": "unknown",
                    },
                    "camera_alignment": {
                        "camera_layout_orientation": "portrait",
                        "phone_content_rotation": "unknown",
                        "confidence": 0.9,
                        "evidence": [],
                    },
                    "elements": [
                        {
                            "element_id": "settings-icon",
                            "role": "icon",
                            "meaning": "settings_app_icon",
                            "label": "设置",
                            "bounds": [650, 240, 830, 430],
                            "confidence": 0.98,
                            "states": {
                                "enabled": True,
                                "fully_visible": True,
                                "goal_relevant": True,
                            },
                            "evidence": ["设置"],
                        }
                    ],
                    "overlays": [],
                    "stable": True,
                    "confidence": 0.97,
                    "fingerprint": "ignored_model_value",
                },
                "input_structure": None,
                "decision": {
                    "status": "action",
                    "action": "tap_semantic",
                    "element_id": "settings-icon",
                    "source_element_id": None,
                    "destination_element_id": None,
                    "direction": None,
                    "evidence_refs": [],
                    "confidence": 0.96,
                    "reason": "设置图标与当前目标逐字对应",
                },
            },
            ensure_ascii=False,
        )


def case(case_id: str = "case_1") -> dict:
    return {
        "id": case_id,
        "page_type": "generic_page",
        "accepted_statuses": ["action"],
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
    def test_manifest_contains_only_action_or_finish_expectations(self) -> None:
        manifest_path = ROOT / "evals" / "qwen_visual_decision" / "cases.json"
        manifest = _load_manifest(manifest_path)
        accepted = {
            status
            for item in manifest["cases"]
            for status in item.get("accepted_statuses", [])
        }
        ids = {item["id"] for item in manifest["cases"]}

        self.assertLessEqual(accepted, {"action", "finish"})
        self.assertNotIn("text_submit_without_confirmation_blocked", ids)
        self.assertNotIn("exact_candidate_missing_blocked", ids)

    def test_manifest_covers_visual_capabilities_and_all_frames_load(self) -> None:
        manifest_path = ROOT / "evals" / "qwen_visual_decision" / "cases.json"
        manifest = _load_manifest(manifest_path)
        cases = manifest["cases"]
        dimensions = {
            name: sum(name in item["expectations"] for item in cases)
            for name in ("page", "target", "input", "finish")
        }

        self.assertGreaterEqual(len(cases), 20)
        self.assertEqual(len({item["id"] for item in cases}), len(cases))
        self.assertEqual(dimensions["page"], len(cases))
        self.assertGreaterEqual(dimensions["target"], 8)
        self.assertGreaterEqual(dimensions["input"], 6)
        self.assertGreaterEqual(dimensions["finish"], 8)
        for item in cases:
            QwenTaskContext.from_dict(item["task_context"])
            frames, paths = _load_frames(item, manifest_path)
            self.assertEqual(len(frames), 4)
            self.assertEqual(len(paths), 4)
            for frame in frames:
                frame.close()

    def test_case_evaluation_exception_path_always_sets_score(self) -> None:
        manifest_path = ROOT / "evals" / "qwen_visual_decision" / "cases.json"
        manifest = _load_manifest(manifest_path)
        launcher_case = next(
            item
            for item in manifest["cases"]
            if item["id"] == "launcher_settings_action"
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
        self.assertFalse(failed["score"]["dimensions"]["page"]["passed"])
        self.assertFalse(failed["score"]["dimensions"]["decision"]["passed"])

    def test_case_evaluation_uses_formal_semantic_authority_and_one_model_call(self) -> None:
        manifest_path = ROOT / "evals" / "qwen_visual_decision" / "cases.json"
        manifest = _load_manifest(manifest_path)
        launcher_case = next(
            item for item in manifest["cases"] if item["id"] == "launcher_settings_action"
        )
        providers = []

        def provider_factory():
            provider = SuccessfulEvalProvider()
            providers.append(provider)
            return provider

        result = _evaluate_case(
            launcher_case,
            str(manifest_path),
            1,
            "run_test",
            provider_factory=provider_factory,
        )

        self.assertTrue(result["score"]["passed"])
        self.assertEqual(result["status"], "action")
        self.assertEqual(result["observation_diagnostics"]["model_calls"], 1)
        self.assertTrue(result["decision_diagnostics"]["decision_from_same_observation_response"])
        self.assertEqual(providers[0].calls, 1)

    def test_score_page_and_bound_target_from_same_observation(self) -> None:
        score = _score(
            {
                "accepted_statuses": ["action"],
                "accepted_actions": ["tap_semantic"],
                "expectations": {
                    "page": {
                        "foreground_app_ids": ["launcher"],
                        "semantic_contains_any": ["桌面"],
                    },
                    "target": {
                        "label_any": ["设置"],
                        "role_any": ["icon"],
                        "meaning_contains_any": ["settings"],
                    },
                },
            },
            status="action",
            decision={
                "next_action": {
                    "action": "tap_semantic",
                    "params": {"element_id": "settings-icon"},
                }
            },
            observation={
                "scene": {
                    "foreground_app_id": "launcher",
                    "screen_id": "home_screen",
                    "summary": "手机桌面可见",
                    "overlays": [],
                    "elements": [
                        {
                            "element_id": "settings-icon",
                            "label": "设置",
                            "role": "icon",
                            "meaning": "settings_app_icon",
                            "states": {},
                        }
                    ],
                }
            },
        )

        self.assertTrue(score["passed"])
        self.assertTrue(score["dimensions"]["page"]["passed"])
        self.assertTrue(score["dimensions"]["target"]["passed"])
        self.assertTrue(score["dimensions"]["decision"]["passed"])
        self.assertFalse(score["dimensions"]["input"]["applicable"])

    def test_score_input_and_finish_from_same_observation(self) -> None:
        score = _score(
            {
                "accepted_statuses": ["finish"],
                "expectations": {
                    "page": {"semantic_contains_any": ["输入"]},
                    "input": {
                        "present": True,
                        "text_exact": ".com",
                        "focused": True,
                        "keyboard_layouts": ["symbol"],
                    },
                    "finish": {"evidence_required": True},
                },
            },
            status="finish",
            decision={"completion_evidence": ["输入框逐字显示 .com"]},
            observation={
                "scene": {
                    "foreground_app_id": "generic_surface",
                    "screen_id": "input_overlay",
                    "summary": "输入弹层和符号键盘可见",
                    "overlays": ["input_overlay"],
                    "elements": [
                        {
                            "element_id": "comment-input",
                            "label": "评论输入框",
                            "role": "input",
                            "meaning": "application_text_input",
                            "states": {
                                "value": ".com",
                                "focused": True,
                                "keyboard_layout": "symbol",
                            },
                        }
                    ],
                }
            },
        )

        self.assertTrue(score["passed"])
        self.assertTrue(score["dimensions"]["input"]["passed"])
        self.assertTrue(score["dimensions"]["finish"]["passed"])

    def test_capability_metrics_use_current_applicable_dimensions_only(self) -> None:
        current = {
            "result_origin": "current_run",
            "score": {
                "dimensions": {
                    "page": {"applicable": True, "passed": True},
                    "target": {"applicable": True, "passed": False},
                    "input": {"applicable": False, "passed": True},
                    "finish": {"applicable": False, "passed": True},
                    "decision": {"applicable": True, "passed": True},
                }
            },
        }
        prior = {
            "result_origin": "prior_run_reference",
            "score": current["score"],
        }
        metrics = _format_capability_metrics([current, prior])

        self.assertEqual(metrics["page"]["accuracy"], 1.0)
        self.assertEqual(metrics["target"]["accuracy"], 0.0)
        self.assertEqual(metrics["input"]["attempted_count"], 0)
        self.assertEqual(metrics["combined"]["attempted_count"], 3)
        self.assertEqual(metrics["combined"]["passed_count"], 2)

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
        self.assertEqual(report["case_outcome_rates"]["failure_rate"], 1.0)

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
