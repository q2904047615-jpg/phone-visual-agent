from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import queue
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

from agent.infrastructure.generic_scene_observer import SingleStepGenericSceneObserver
from agent.infrastructure.qwen_runtime_errors import (
    classify_qwen_error,
    failure_diagnostics,
)
from agent.application.qwen_visual_decision import QwenVisualDecisionObserver
from agent.domain.qwen_task_context import QwenTaskContext
from agent.domain.trusted_observation import TrustedObservation
from agent.infrastructure.trusted_observation_frames import (
    build_trusted_observation,
    validate_trusted_observation_against_frames,
)
from agent.infrastructure.dashscope_vision_provider import (
    DashScopeVisionProvider,
)
from agent.domain.vision_model import VisionAgentError


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "evals" / "qwen_visual_decision" / "cases.json"
DEFAULT_OUTPUT_ROOT = ROOT / "output" / "offline_qwen_visual_decision"
DEFAULT_CASE_TIMEOUT_SECONDS = 150.0
DEFAULT_SUITE_TIMEOUT_SECONDS = 720.0
EVAL_PROTOCOL_VERSION = "2026-09-01-qwen-visual-capability-benchmark-v5"
CAPABILITY_DIMENSIONS = ("page", "target", "input", "finish", "decision")


def _load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("cases"), list):
        raise ValueError("离线用例清单缺少 cases 数组。")
    if value.get("protocol_version") != EVAL_PROTOCOL_VERSION:
        raise ValueError("离线用例清单协议版本无效。")
    materialized = [
        _materialize_case(item, index=index)
        for index, item in enumerate(value["cases"], start=1)
    ]
    case_ids = [str(item.get("id") or "") for item in materialized]
    if any(not item for item in case_ids) or len(case_ids) != len(set(case_ids)):
        raise ValueError("离线用例 ID 为空或重复。")
    result = dict(value)
    result["cases"] = materialized
    return result


def _materialize_case(raw_case: Any, *, index: int) -> dict[str, Any]:
    if not isinstance(raw_case, dict):
        raise ValueError("离线用例必须是 JSON 对象。")
    case = copy.deepcopy(raw_case)
    case_id = str(case.get("id") or "").strip()
    task = case.get("task")
    task_context = case.get("task_context")
    if (task is None) == (task_context is None):
        raise ValueError(f"离线用例 {case_id or index} 必须且只能提供 task 或 task_context。")
    if task_context is None:
        if not isinstance(task, dict):
            raise ValueError(f"离线用例 {case_id or index} 的 task 必须是对象。")
        objective = str(task.get("objective") or "").strip()
        if not objective:
            raise ValueError("离线用例缺少整任务目标。")
        task_context = QwenTaskContext(task_id=f"task_{case_id}",
            device_id=str(task.get("device_id") or "offline_phone_01"),
            revision=task.get("revision", index), raw_goal=objective,
            history=tuple(task.get("history") or ()),
            exact_input_text=(task.get("entities") or {}).get("expected_text")).to_dict()
        case["task_context"] = task_context
    expectations = case.get("expectations")
    if not isinstance(expectations, dict) or not expectations:
        raise ValueError(f"离线用例 {case_id or index} 缺少 expectations。")
    unexpected = set(expectations) - {"page", "target", "input", "finish"}
    if unexpected or any(not isinstance(item, dict) for item in expectations.values()):
        raise ValueError(f"离线用例 {case_id or index} 的 expectations 无效。")
    statuses = case.get("accepted_statuses")
    if (not isinstance(statuses, list) or not statuses
        or any(item not in {"action", "finish"} for item in statuses)):
        raise ValueError(f"离线用例 {case_id or index} 的 accepted_statuses 无效。")
    QwenTaskContext.from_dict(dict(case["task_context"]))
    return case


def _load_frames(case: dict[str, Any], manifest_path: Path) -> tuple[list[Image.Image], list[str]]:
    raw_paths = case.get("frames")
    if raw_paths is None:
        raw_paths = [case.get("image")] * 4
    if not isinstance(raw_paths, list) or len(raw_paths) < 4:
        raise ValueError("每个离线用例必须提供至少4帧。")
    if any(not isinstance(item, str) or not item.strip() for item in raw_paths):
        raise ValueError("离线帧路径格式无效。")
    paths = [(manifest_path.parent / item).resolve() for item in raw_paths]
    frames: list[Image.Image] = []
    for path in paths:
        with Image.open(path) as source:
            frames.append(source.convert("RGB"))
    return frames, [str(path) for path in paths]


def _normalized_text(value: Any) -> str:
    return str(value or "").strip().casefold()


def _contains_any(value: Any, expected: Any) -> bool:
    haystack = _normalized_text(value)
    return bool(isinstance(expected, list) and expected
        and any(_normalized_text(item) in haystack for item in expected if _normalized_text(item)))


def _dimension(*, applicable: bool, reasons: list[str] | None=None) -> dict[str, Any]:
    resolved = list(reasons or [])
    return {"applicable": applicable, "passed": not resolved if applicable else True, "reasons": resolved}


def _failed_score(case: dict[str, Any], reason: str) -> dict[str, Any]:
    expectations = case.get("expectations") or {}
    dimensions = {
        name: _dimension(applicable=(name == "decision" or name in expectations), reasons=[reason])
        if name == "decision" or name in expectations else _dimension(applicable=False)
        for name in CAPABILITY_DIMENSIONS
    }
    return {"passed": False, "reasons": [reason], "dimensions": dimensions}


def _score(case: dict[str, Any], *, status: str, decision: dict[str, Any] | None,
    observation: dict[str, Any] | None) -> dict[str, Any]:
    expectations = case.get("expectations") or {}
    scene = (observation or {}).get("scene") or {}
    elements = scene.get("elements") or []
    if not isinstance(elements, list):
        elements = []
    dimensions: dict[str, dict[str, Any]] = {}

    accepted_statuses = [str(item) for item in case.get("accepted_statuses") or []]
    decision_reasons: list[str] = []
    if accepted_statuses and status not in accepted_statuses:
        decision_reasons.append(f"status={status!r} 不在 {accepted_statuses!r}")
    accepted_actions = [str(item) for item in case.get("accepted_actions") or []]
    next_action = (decision or {}).get("next_action") or {}
    action = str(next_action.get("action") or "") if isinstance(next_action, dict) else ""
    if status == "action" and accepted_actions and action not in accepted_actions:
        decision_reasons.append(f"action={action!r} 不在 {accepted_actions!r}")
    if status == "action" and isinstance(next_action, list):
        decision_reasons.append("next_action 不能是动作列表")
    dimensions["decision"] = _dimension(applicable=True, reasons=decision_reasons)

    page_reasons: list[str] = []
    page = expectations.get("page")
    if isinstance(page, dict):
        app_ids = page.get("foreground_app_ids")
        if isinstance(app_ids, list) and _normalized_text(scene.get("foreground_app_id")) not in {
            _normalized_text(item) for item in app_ids}:
            page_reasons.append(f"foreground_app_id={scene.get('foreground_app_id')!r} 不在 {app_ids!r}")
        semantic = " ".join([
            str(scene.get("screen_id") or ""),
            str(scene.get("summary") or ""),
            *[str(item) for item in scene.get("overlays") or []],
            *[
                str(value)
                for item in elements
                if isinstance(item, dict)
                for value in (item.get("label") or "", item.get("meaning") or "")
            ],
        ])
        semantic_terms = page.get("semantic_contains_any")
        if isinstance(semantic_terms, list) and not _contains_any(semantic, semantic_terms):
            page_reasons.append(f"页面语义未包含任一预期词：{semantic_terms!r}")
    dimensions["page"] = _dimension(applicable=isinstance(page, dict), reasons=page_reasons)

    target_reasons: list[str] = []
    target = expectations.get("target")
    if isinstance(target, dict):
        target_element = None
        params = next_action.get("params") if isinstance(next_action, dict) else None
        element_id = str((params or {}).get("element_id") or "") if isinstance(params, dict) else ""
        if status != "action":
            target_reasons.append("目标控件期望只适用于 action，模型没有返回 action")
        elif not element_id:
            target_reasons.append("动作没有绑定当前 scene 的 element_id")
        else:
            target_element = next((item for item in elements
                if isinstance(item, dict) and str(item.get("element_id") or "") == element_id), None)
            if target_element is None and isinstance(params, dict) and isinstance(params.get("tap_point"), list):
                target_element = {
                    "element_id": element_id,
                    "role": params.get("role"),
                    "meaning": params.get("target"),
                    "label": params.get("label"),
                    "evidence": params.get("target_evidence") or [],
                }
            if target_element is None:
                target_reasons.append(f"当前响应找不到动作绑定目标：{element_id}")
        if target_element is not None:
            labels = target.get("label_any")
            if isinstance(labels, list) and _normalized_text(target_element.get("label")) not in {
                _normalized_text(item) for item in labels}:
                target_reasons.append(f"目标标签={target_element.get('label')!r} 不在 {labels!r}")
            roles = target.get("role_any")
            if isinstance(roles, list) and _normalized_text(target_element.get("role")) not in {
                _normalized_text(item) for item in roles}:
                target_reasons.append(f"目标角色={target_element.get('role')!r} 不在 {roles!r}")
            meanings = target.get("meaning_contains_any")
            if isinstance(meanings, list) and not _contains_any(target_element.get("meaning"), meanings):
                target_reasons.append(f"目标语义={target_element.get('meaning')!r} 未包含 {meanings!r}")
    dimensions["target"] = _dimension(applicable=isinstance(target, dict), reasons=target_reasons)

    input_reasons: list[str] = []
    input_expectation = expectations.get("input")
    if isinstance(input_expectation, dict):
        inputs = [item for item in elements if isinstance(item, dict) and item.get("role") == "input"]
        present = input_expectation.get("present", True)
        expected_count = input_expectation.get("count")
        if isinstance(expected_count, int) and not isinstance(expected_count, bool) and len(inputs) != expected_count:
            input_reasons.append(f"输入框数量={len(inputs)}，预期={expected_count}")
        if present is False:
            if inputs:
                input_reasons.append(f"预期没有输入框，但识别到{len(inputs)}个")
        elif not inputs:
            input_reasons.append("没有识别到应用输入框")
        else:
            matches = []
            for item in inputs:
                states = item.get("states") or {}
                if not isinstance(states, dict):
                    continue
                if "text_exact" in input_expectation and states.get("value") != input_expectation["text_exact"]:
                    continue
                if "focused" in input_expectation and states.get("focused") is not input_expectation["focused"]:
                    continue
                if "preedit_exact" in input_expectation and states.get("ime_preedit_text") != input_expectation[
                    "preedit_exact"]:
                    continue
                preedit_terms = input_expectation.get("preedit_contains_any")
                if isinstance(preedit_terms, list) and not _contains_any(
                    states.get("ime_preedit_text"), preedit_terms
                ):
                    continue
                matches.append(item)
            if not matches:
                observed = [{"value": (item.get("states") or {}).get("value"),
                    "focused": (item.get("states") or {}).get("focused"),
                    "ime_preedit_text": (item.get("states") or {}).get("ime_preedit_text")}
                    for item in inputs]
                input_reasons.append(f"输入状态未匹配预期；observed={observed!r}")
    dimensions["input"] = _dimension(applicable=isinstance(input_expectation, dict), reasons=input_reasons)

    finish_reasons: list[str] = []
    finish = expectations.get("finish")
    if isinstance(finish, dict):
        if status != "finish":
            finish_reasons.append(f"预期 finish，但模型返回 {status!r}")
        if finish.get("evidence_required") is True:
            reason = str((decision or {}).get("reason") or "").strip()
            if not reason:
                finish_reasons.append("finish 没有陈述当前截图完成事实")
    dimensions["finish"] = _dimension(applicable=isinstance(finish, dict), reasons=finish_reasons)

    reasons = [f"{name}: {reason}" for name in CAPABILITY_DIMENSIONS
        for reason in dimensions[name]["reasons"]]
    return {"passed": not reasons, "reasons": reasons, "dimensions": dimensions}


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _provider_model_usage(provider: Any) -> dict[str, int]:
    try:
        status = provider.status()
    except (AttributeError, TypeError, ValueError):
        status = {}
    totals = status.get("usage_totals") if isinstance(status, dict) else {}
    if not isinstance(totals, dict):
        totals = {}

    def safe_count(key: str) -> int:
        value = totals.get(key)
        return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0

    call_count = status.get("successful_call_count") if isinstance(status, dict) else 0
    return {
        "successful_model_calls": (
            call_count
            if isinstance(call_count, int) and not isinstance(call_count, bool) and call_count >= 0
            else 0
        ),
        "prompt_tokens": safe_count("prompt_tokens"),
        "completion_tokens": safe_count("completion_tokens"),
        "total_tokens": safe_count("total_tokens"),
    }


def _evaluate_case(
    case: dict[str, Any],
    manifest_path: str,
    decision_number: int,
    run_id: str,
    provider_factory: Any = DashScopeVisionProvider,
) -> dict[str, Any]:
    started = time.perf_counter()
    path = Path(manifest_path)
    provider = provider_factory()
    scene_observer = SingleStepGenericSceneObserver(provider)
    decision_observer = QwenVisualDecisionObserver(
        provider,
        trusted_observation_frame_validator=(
            validate_trusted_observation_against_frames
        ),
    )
    frames: list[Image.Image] = []
    frame_paths: list[str] = []
    context: QwenTaskContext | None = None
    observation: TrustedObservation | None = None
    try:
        frames, frame_paths = _load_frames(case, path)
        base_context = QwenTaskContext.from_dict(dict(case["task_context"]))
        context = base_context
        context.validate()
        scene, model_decision = scene_observer.observe_with_decision(
            frames=frames,
            goal_context={"objective": context.raw_goal, "entities": {
                "history": list(context.history), "exact_input_text": context.exact_input_text}},
        )
        observation = build_trusted_observation(
            frames=frames,
            device_id=context.device_id,
            scene=scene,
            observation_id=f"obs_{uuid.uuid4().hex}",
        )
        decision = decision_observer.decide(
            frames=frames,
            task_context=context,
            trusted_observation=observation,
            decision_number=decision_number,
            model_decision=model_decision,
        )
        value = decision.to_dict()
        status = decision.proposal.status
        decision_diagnostics = dict(decision_observer.last_diagnostics)
        failure = None
        if decision_diagnostics.get("error_type"):
            failure = {
                "stage": "decision",
                "error_type": decision_diagnostics["error_type"],
                "error": decision_diagnostics.get("error"),
                "safe_stop_reason": decision_diagnostics.get("safe_stop_reason"),
            }
        elif decision_diagnostics.get("local_safety_block"):
            failure = {
                "stage": "local_decision_gate",
                "error_type": "local_safety_block",
                "error": decision_diagnostics["local_safety_block"],
                "safe_stop_reason": decision.proposal.reason,
            }
        score = _score(
            case,
            status=status,
            decision=value,
            observation=observation.to_dict(),
        )
        if failure and failure.get("error_type") != "local_safety_block":
            score = _failed_score(
                case,
                f"运行时失败：{failure.get('error_type')}；{failure.get('error') or ''}",
            )
        if not score["passed"] and failure is None:
            failure = {
                "stage": "scoring",
                "error_type": "model_outcome_mismatch",
                "error": " | ".join(score["reasons"]),
                "safe_stop_reason": decision.proposal.reason,
            }
        return {
            "case_id": case["id"],
            "page_type": case.get("page_type"),
            "result_origin": "current_run",
            "run_id": run_id,
            "frame_paths": frame_paths,
            "task_context": context.to_dict(),
            "observation": observation.to_dict(),
            "observation_diagnostics": dict(scene_observer.last_diagnostics),
            "decision": value,
            "decision_diagnostics": decision_diagnostics,
            "status": status,
            "failure": failure,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "hardware_actions_enabled": False,
            "model_usage": _provider_model_usage(provider),
            "score": score,
        }
    except (VisionAgentError, ValueError, TypeError, OSError) as exc:
        observation_diagnostics = dict(scene_observer.last_diagnostics)
        decision_diagnostics = dict(decision_observer.last_diagnostics)
        stage = "observation" if observation is None else "decision"
        raw_response = (
            scene_observer.last_raw_response
            if stage == "observation"
            else decision_observer.last_raw_response
        )
        error_type = classify_qwen_error(exc, raw_response=raw_response)
        safe_stop = (
            "观察阶段失败，未建立可信候选，决策模型、控制器和机械臂均未执行。"
            if observation is None
            else "决策阶段失败，未形成可执行动作，控制器和机械臂均未执行。"
        )
        if not observation_diagnostics and stage == "observation":
            observation_diagnostics = failure_diagnostics(
                exc,
                raw_response=raw_response,
                stage="loading_or_observation",
                model_calls=0,
                elapsed_seconds=time.perf_counter() - started,
                safe_stop_reason=safe_stop,
            )
        score = _failed_score(case, f"运行时失败：{error_type}；{exc}")
        return {
            "case_id": case["id"],
            "page_type": case.get("page_type"),
            "result_origin": "current_run",
            "run_id": run_id,
            "frame_paths": frame_paths,
            "task_context": context.to_dict() if context else case.get("task_context"),
            "observation": observation.to_dict() if observation else None,
            "observation_diagnostics": observation_diagnostics,
            "decision": None,
            "decision_diagnostics": decision_diagnostics,
            "status": "failed",
            "failure": {
                "stage": stage,
                "error_type": error_type,
                "error": str(exc),
                "safe_stop_reason": safe_stop,
            },
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "hardware_actions_enabled": False,
            "model_usage": _provider_model_usage(provider),
            "score": score,
        }


def _case_worker(
    case: dict[str, Any],
    manifest_path: str,
    decision_number: int,
    run_id: str,
    output_queue: Any,
) -> None:
    try:
        output_queue.put(
            _evaluate_case(case, manifest_path, decision_number, run_id)
        )
    except BaseException as exc:  # pragma: no cover - process boundary guard
        output_queue.put(
            _worker_crash_result(case, run_id, exc)
        )


def _worker_crash_result(
    case: dict[str, Any],
    run_id: str,
    error: BaseException,
) -> dict[str, Any]:
    return {
        "case_id": case["id"],
        "page_type": case.get("page_type"),
        "result_origin": "current_run",
        "run_id": run_id,
        "frame_paths": [],
        "task_context": case.get("task_context"),
        "observation": None,
        "observation_diagnostics": {},
        "decision": None,
        "decision_diagnostics": {},
        "status": "failed",
        "failure": {
            "stage": "case_worker",
            "error_type": "worker_crash",
            "error": str(error),
            "safe_stop_reason": "用例子进程异常退出，未形成可执行动作。",
        },
        "elapsed_seconds": 0.0,
        "hardware_actions_enabled": False,
        "score": _failed_score(case, f"用例子进程异常：{error}"),
    }


def _timeout_result(
    case: dict[str, Any],
    *,
    run_id: str,
    timeout_seconds: float,
    error_type: str,
) -> dict[str, Any]:
    label = "单用例" if error_type == "case_timeout" else "整套"
    return {
        "case_id": case["id"],
        "page_type": case.get("page_type"),
        "result_origin": "current_run",
        "run_id": run_id,
        "frame_paths": [],
        "task_context": case.get("task_context"),
        "observation": None,
        "observation_diagnostics": {},
        "decision": None,
        "decision_diagnostics": {},
        "status": "failed",
        "failure": {
            "stage": "case_process",
            "error_type": error_type,
            "error": f"{label}超时：{timeout_seconds:.1f}秒",
            "safe_stop_reason": f"{label}超时后终止隔离子进程，未形成可执行动作。",
        },
        "elapsed_seconds": round(timeout_seconds, 3),
        "hardware_actions_enabled": False,
        "score": _failed_score(case, f"运行时失败：{error_type}；{label}超时"),
    }


def _pending_result(case: dict[str, Any], run_id: str) -> dict[str, Any]:
    return {
        "case_id": case["id"],
        "page_type": case.get("page_type"),
        "result_origin": "current_run_pending",
        "run_id": run_id,
        "status": "pending",
        "hardware_actions_enabled": False,
        "score": _failed_score(case, "本轮尚未执行"),
    }


def _configuration_result(case: dict[str, Any], run_id: str) -> dict[str, Any]:
    return {
        "case_id": case["id"],
        "page_type": case.get("page_type"),
        "result_origin": "current_run",
        "run_id": run_id,
        "status": "failed",
        "failure": {
            "stage": "configuration",
            "error_type": "configuration_error",
            "error": "未配置 DASHSCOPE_API_KEY",
            "safe_stop_reason": "在线模型未配置，未调用观察、决策、控制器或机械臂。",
        },
        "observation_diagnostics": {"model_calls": 0},
        "decision_diagnostics": {"model_calls": 0},
        "hardware_actions_enabled": False,
        "score": _failed_score(
            case,
            "运行时失败：configuration_error；未配置 DASHSCOPE_API_KEY",
        ),
    }


def _run_case_with_timeout(
    case: dict[str, Any],
    *,
    manifest_path: Path,
    decision_number: int,
    run_id: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    context = mp.get_context("spawn")
    output_queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_case_worker,
        args=(case, str(manifest_path), decision_number, run_id, output_queue),
        daemon=True,
    )
    process.start()
    process.join(timeout=max(0.1, timeout_seconds))
    if process.is_alive():
        process.terminate()
        process.join(timeout=5.0)
        output_queue.close()
        return _timeout_result(
            case,
            run_id=run_id,
            timeout_seconds=timeout_seconds,
            error_type="case_timeout",
        )
    try:
        result = output_queue.get(timeout=2.0)
    except queue.Empty:
        result = _worker_crash_result(
            case,
            run_id,
            RuntimeError(f"用例子进程退出码 {process.exitcode}，未返回结果。"),
        )
    finally:
        output_queue.close()
    return result


def _load_resume_results(path: Path) -> tuple[str, dict[str, dict[str, Any]]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or not isinstance(report.get("results"), list):
        raise ValueError("恢复报告缺少 results 数组。")
    source_run_id = str(report.get("run_id") or path.parent.name)
    return source_run_id, {
        str(item.get("case_id")): dict(item)
        for item in report["results"]
        if isinstance(item, dict) and item.get("case_id")
    }


def _prior_reference(
    result: dict[str, Any],
    *,
    source_report: Path,
    source_run_id: str,
) -> dict[str, Any]:
    copied = copy.deepcopy(result)
    copied.update(
        {
            "result_origin": "prior_run_reference",
            "source_report": str(source_report),
            "source_run_id": source_run_id,
            "counted_in_current_metrics": False,
        }
    )
    return copied


def _format_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    current = [item for item in results if item.get("result_origin") == "current_run"]

    def stage_metrics(key: str) -> dict[str, Any]:
        attempted = [
            item.get(key) or {}
            for item in current
            if int((item.get(key) or {}).get("model_calls") or 0) > 0
        ]
        first = sum(bool(item.get("first_pass_success")) for item in attempted)
        repaired = sum(bool(item.get("repair_retry_success")) for item in attempted)
        return {
            "attempted_count": len(attempted),
            "first_pass_success_count": first,
            "repair_retry_success_count": repaired,
            "first_pass_rate": _rate(first, len(attempted)),
            "repair_retry_rate": _rate(repaired, len(attempted)),
        }

    observation = stage_metrics("observation_diagnostics")
    decision = stage_metrics("decision_diagnostics")
    attempted = observation["attempted_count"] + decision["attempted_count"]
    first = observation["first_pass_success_count"] + decision["first_pass_success_count"]
    repaired = observation["repair_retry_success_count"] + decision["repair_retry_success_count"]
    return {
        "observation": observation,
        "decision": decision,
        "combined": {
            "attempted_count": attempted,
            "first_pass_success_count": first,
            "repair_retry_success_count": repaired,
            "first_pass_rate": _rate(first, attempted),
            "repair_retry_rate": _rate(repaired, attempted),
        },
    }


def _format_capability_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    current = [item for item in results if item.get("result_origin") == "current_run"]
    metrics: dict[str, Any] = {}
    total_attempted = 0
    total_passed = 0
    for name in CAPABILITY_DIMENSIONS:
        attempted = []
        for item in current:
            dimensions = (item.get("score") or {}).get("dimensions") or {}
            value = dimensions.get(name) if isinstance(dimensions, dict) else None
            if isinstance(value, dict) and value.get("applicable") is True:
                attempted.append(value)
        passed = sum(item.get("passed") is True for item in attempted)
        metrics[name] = {
            "attempted_count": len(attempted),
            "passed_count": passed,
            "failed_count": len(attempted) - passed,
            "accuracy": _rate(passed, len(attempted)),
        }
        total_attempted += len(attempted)
        total_passed += passed
    metrics["combined"] = {
        "attempted_count": total_attempted,
        "passed_count": total_passed,
        "failed_count": total_attempted - total_passed,
        "accuracy": _rate(total_passed, total_attempted),
    }
    return metrics


def _format_model_usage(results: list[dict[str, Any]]) -> dict[str, int]:
    current = [item for item in results if item.get("result_origin") == "current_run"]
    keys = (
        "successful_model_calls",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    )
    totals = {key: 0 for key in keys}
    for item in current:
        usage = item.get("model_usage") or {}
        if not isinstance(usage, dict):
            continue
        for key in keys:
            value = usage.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                totals[key] += value
    return totals


def _build_report(
    *,
    run_id: str,
    manifest_path: Path,
    model: str,
    results: list[dict[str, Any]],
    selected_case_ids: list[str],
    current_target_case_ids: list[str],
    started_at: str,
    started_monotonic: float,
    case_timeout_seconds: float,
    suite_timeout_seconds: float,
    resume_report: Path | None,
) -> dict[str, Any]:
    statuses = [str(item.get("status") or "") for item in results]
    current = [
        item
        for item in results
        if item.get("result_origin") in {"current_run", "current_run_pending"}
    ]
    pending = [item for item in results if item.get("status") == "pending"]
    completed_current = [item for item in current if item.get("status") != "pending"]
    failures = [
        {
            "case_id": item.get("case_id"),
            **dict(item.get("failure") or {}),
        }
        for item in results
        if item.get("failure")
    ]
    return {
        "protocol_version": EVAL_PROTOCOL_VERSION,
        "run_id": run_id,
        "mode": "online_qwen_offline_existing_screenshots_only",
        "started_at": started_at,
        "elapsed_seconds": round(time.perf_counter() - started_monotonic, 3),
        "report_status": "running" if pending else "complete",
        "full_case_report_complete": not pending and len(results) == len(selected_case_ids),
        "manifest": str(manifest_path),
        "model": model,
        "hardware_actions_enabled": False,
        "camera_enabled": False,
        "web_console_enabled": False,
        "robot_enabled": False,
        "timeouts": {
            "case_timeout_seconds": case_timeout_seconds,
            "suite_timeout_seconds": suite_timeout_seconds,
        },
        "resume": {
            "enabled": resume_report is not None,
            "source_report": str(resume_report) if resume_report else None,
            "current_target_case_ids": current_target_case_ids,
            "prior_results_are_references_only": True,
        },
        "case_count": len(results),
        "current_run_case_count": len(current),
        "current_run_completed_count": len(completed_current),
        "reused_prior_reference_count": sum(
            item.get("result_origin") == "prior_run_reference" for item in results
        ),
        "passed": sum(bool((item.get("score") or {}).get("passed")) for item in results),
        "failed": sum(not bool((item.get("score") or {}).get("passed")) for item in results),
        "case_outcome_rates": {
            "action_rate": _rate(statuses.count("action"), len(statuses)),
            "finish_rate": _rate(statuses.count("finish"), len(statuses)),
            "failure_rate": _rate(statuses.count("failed"), len(statuses)),
        },
        "format_metrics": _format_metrics(results),
        "capability_metrics": _format_capability_metrics(results),
        "model_usage": _format_model_usage(results),
        "failures": failures,
        "results": results,
    }


def _write_report(path: Path, report: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="仅使用已有截图调用Qwen，隔离评估可信观察和唯一下一视觉动作。"
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--case", action="append", dest="case_ids")
    parser.add_argument("--resume-report", type=Path)
    parser.add_argument(
        "--case-timeout-seconds",
        type=float,
        default=DEFAULT_CASE_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--suite-timeout-seconds",
        type=float,
        default=DEFAULT_SUITE_TIMEOUT_SECONDS,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    if args.case_timeout_seconds <= 0 or args.suite_timeout_seconds <= 0:
        raise ValueError("单用例和整套超时必须大于0。")

    manifest_path = args.manifest.resolve()
    manifest = _load_manifest(manifest_path)
    all_cases = list(manifest["cases"])
    case_by_id = {str(item["id"]): item for item in all_cases}
    unknown = set(args.case_ids or []) - set(case_by_id)
    if unknown:
        raise ValueError("未知离线用例：" + ", ".join(sorted(unknown)))

    resume_path = args.resume_report.resolve() if args.resume_report else None
    prior_results: dict[str, dict[str, Any]] = {}
    source_run_id = ""
    if resume_path:
        source_run_id, prior_results = _load_resume_results(resume_path)

    if args.case_ids:
        current_target_ids = list(dict.fromkeys(args.case_ids))
    elif resume_path:
        current_target_ids = [
            str(case["id"])
            for case in all_cases
            if not bool((prior_results.get(str(case["id"]), {}).get("score") or {}).get("passed"))
        ]
    else:
        current_target_ids = [str(case["id"]) for case in all_cases]
    if not current_target_ids and resume_path:
        raise ValueError("恢复报告中没有失败或未完成用例。")

    selected_case_ids = (
        [str(case["id"]) for case in all_cases]
        if resume_path and not args.case_ids
        else current_target_ids
    )
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = args.output_root.resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    report_path = run_dir / "report.json"
    started_at = datetime.now().astimezone().isoformat()
    started_monotonic = time.perf_counter()
    results_by_id: dict[str, dict[str, Any]] = {}
    for case_id in selected_case_ids:
        if case_id not in current_target_ids and case_id in prior_results and resume_path:
            results_by_id[case_id] = _prior_reference(
                prior_results[case_id],
                source_report=resume_path,
                source_run_id=source_run_id,
            )
        else:
            results_by_id[case_id] = _pending_result(case_by_id[case_id], run_id)

    provider = DashScopeVisionProvider()

    def checkpoint() -> dict[str, Any]:
        ordered = [results_by_id[item] for item in selected_case_ids]
        report = _build_report(
            run_id=run_id,
            manifest_path=manifest_path,
            model=provider.model,
            results=ordered,
            selected_case_ids=selected_case_ids,
            current_target_case_ids=current_target_ids,
            started_at=started_at,
            started_monotonic=started_monotonic,
            case_timeout_seconds=args.case_timeout_seconds,
            suite_timeout_seconds=args.suite_timeout_seconds,
            resume_report=resume_path,
        )
        _write_report(report_path, report)
        return report

    checkpoint()
    if not provider.configured:
        for case_id in current_target_ids:
            results_by_id[case_id] = _configuration_result(case_by_id[case_id], run_id)
        report = checkpoint()
        print(
            json.dumps(
                {
                    "report": str(report_path),
                    "report_status": report["report_status"],
                    "full_case_report_complete": report["full_case_report_complete"],
                    "case_count": report["case_count"],
                    "passed": report["passed"],
                    "failed": report["failed"],
                    "capability_metrics": report["capability_metrics"],
                    "configuration_error": "未配置 DASHSCOPE_API_KEY",
                    "hardware_actions_enabled": False,
                },
                ensure_ascii=False,
            )
        )
        return 1
    for decision_number, case_id in enumerate(current_target_ids, start=1):
        elapsed = time.perf_counter() - started_monotonic
        remaining = args.suite_timeout_seconds - elapsed
        if remaining <= 0:
            for remaining_id in current_target_ids[decision_number - 1 :]:
                results_by_id[remaining_id] = _timeout_result(
                    case_by_id[remaining_id],
                    run_id=run_id,
                    timeout_seconds=args.suite_timeout_seconds,
                    error_type="suite_timeout",
                )
            checkpoint()
            break
        timeout_seconds = min(args.case_timeout_seconds, remaining)
        results_by_id[case_id] = _run_case_with_timeout(
            case_by_id[case_id],
            manifest_path=manifest_path,
            decision_number=decision_number,
            run_id=run_id,
            timeout_seconds=timeout_seconds,
        )
        case_path = run_dir / f"{case_id}.json"
        _write_report(case_path, results_by_id[case_id])
        checkpoint()

    report = checkpoint()
    print(
        json.dumps(
            {
                "report": str(report_path),
                "report_status": report["report_status"],
                "full_case_report_complete": report["full_case_report_complete"],
                "case_count": report["case_count"],
                "passed": report["passed"],
                "failed": report["failed"],
                "format_metrics": report["format_metrics"],
                "capability_metrics": report["capability_metrics"],
                "failure_rate": report["case_outcome_rates"]["failure_rate"],
                "hardware_actions_enabled": False,
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["failed"] == 0 and report["full_case_report_complete"] else 1


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
