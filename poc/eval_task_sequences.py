from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Mapping

from device_executor import (
    DeviceActionRequest,
    DeviceExecutionError,
    ReplayDeviceExecutor,
)


TASK_SEQUENCE_BENCHMARK_PROTOCOL = "2026-08-25-task-sequence-replay-v1"
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


class TaskSequenceBenchmarkError(ValueError):
    pass


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _action_request(value: Mapping[str, Any]) -> DeviceActionRequest:
    raw = dict(value)
    kind = str(raw.pop("kind", "")).strip()

    def point(name: str) -> tuple[int, int] | None:
        item = raw.pop(name, None)
        if item is None:
            return None
        if (
            not isinstance(item, list)
            or len(item) != 2
            or any(
                isinstance(part, bool) or not isinstance(part, int)
                for part in item
            )
        ):
            raise TaskSequenceBenchmarkError(f"{name} 必须是两个整数。")
        return item[0], item[1]

    request = DeviceActionRequest(
        kind=kind,
        point=point("point"),
        end_point=point("end_point"),
        direction=raw.pop("direction", None),
        hold_seconds=raw.pop("hold_seconds", None),
        input_fragment=raw.pop("input_fragment", None),
        input_method=raw.pop("input_method", None),
        input_pinyin=raw.pop("input_pinyin", None),
        keyboard_geometry=raw.pop("keyboard_geometry", None),
        delete_count=raw.pop("delete_count", None),
        wait_seconds=raw.pop("wait_seconds", None),
    )
    if raw:
        raise TaskSequenceBenchmarkError(
            "回放动作包含未知字段：" + ", ".join(sorted(raw))
        )
    request.validate()
    return request


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TaskSequenceBenchmarkError(f"无法读取任务级回放清单：{exc}") from exc
    if not isinstance(value, dict):
        raise TaskSequenceBenchmarkError("任务级回放清单必须是对象。")
    if value.get("protocol_version") != TASK_SEQUENCE_BENCHMARK_PROTOCOL:
        raise TaskSequenceBenchmarkError("任务级回放清单协议版本无效。")
    if value.get("mode") != "offline_read_only":
        raise TaskSequenceBenchmarkError("任务级回放只允许 offline_read_only。")
    cases = value.get("cases")
    if not isinstance(cases, list) or not cases:
        raise TaskSequenceBenchmarkError("任务级回放清单没有 cases。")
    return value


def evaluate_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise TaskSequenceBenchmarkError("任务级回放清单没有 cases。")
    started = time.perf_counter()
    case_ids: set[str] = set()
    app_scopes: set[str] = set()
    results: list[dict[str, Any]] = []
    total_expected = 0
    total_matched = 0

    for raw_case in cases:
        if not isinstance(raw_case, Mapping):
            raise TaskSequenceBenchmarkError("任务级回放 case 必须是对象。")
        case_id = str(raw_case.get("case_id") or "").strip()
        app_scope = str(raw_case.get("app_scope") or "").strip()
        goal = str(raw_case.get("goal") or "").strip()
        expected_terminal = str(
            raw_case.get("expected_terminal_status") or ""
        ).strip()
        recorded_terminal = str(
            raw_case.get("recorded_terminal_status") or ""
        ).strip()
        if not _ID.fullmatch(case_id) or case_id in case_ids:
            raise TaskSequenceBenchmarkError("case_id 无效或重复。")
        if not app_scope or not goal:
            raise TaskSequenceBenchmarkError(f"{case_id} 缺少 app_scope/goal。")
        if expected_terminal != "succeeded":
            raise TaskSequenceBenchmarkError(
                f"{case_id} 只接受 expected_terminal_status=succeeded。"
            )
        expected_actions = raw_case.get("expected_actions")
        recorded_actions = raw_case.get("recorded_actions")
        if (
            not isinstance(expected_actions, list)
            or not expected_actions
            or not isinstance(recorded_actions, list)
            or not recorded_actions
        ):
            raise TaskSequenceBenchmarkError(f"{case_id} 缺少动作序列。")
        if len(expected_actions) > 24 or len(recorded_actions) > 24:
            raise TaskSequenceBenchmarkError(f"{case_id} 动作序列超过24步。")

        case_ids.add(case_id)
        app_scopes.add(app_scope)
        expected_requests = [_action_request(item) for item in expected_actions]
        recorded_requests = [_action_request(item) for item in recorded_actions]
        replay = ReplayDeviceExecutor(
            [
                {"kind": request.kind, "request": request.to_dict()}
                for request in expected_requests
            ]
        )
        case_started = time.perf_counter()
        failure = ""
        matched = 0
        try:
            for request in recorded_requests:
                replay.execute(request)
                matched += 1
            if not replay.complete:
                failure = "录制序列提前结束。"
            elif recorded_terminal != expected_terminal:
                failure = (
                    "终态不匹配："
                    f"{recorded_terminal or 'missing'} != {expected_terminal}"
                )
        except (DeviceExecutionError, TaskSequenceBenchmarkError) as exc:
            failure = str(exc)
        total_expected += len(expected_requests)
        total_matched += matched
        results.append(
            {
                "case_id": case_id,
                "app_scope": app_scope,
                "goal": goal,
                "passed": not failure,
                "failure": failure,
                "expected_actions": len(expected_requests),
                "matched_actions": matched,
                "terminal_status": recorded_terminal,
                "physical_actions": 0,
                "remote_model_calls": 0,
                "elapsed_ms": round(
                    (time.perf_counter() - case_started) * 1000.0,
                    3,
                ),
            }
        )

    passed_cases = sum(1 for item in results if item["passed"])
    return {
        "protocol_version": TASK_SEQUENCE_BENCHMARK_PROTOCOL,
        "mode": "offline_read_only",
        "manifest_digest": _digest(dict(manifest)),
        "case_count": len(results),
        "distinct_app_scopes": len(app_scopes),
        "passed_cases": passed_cases,
        "failed_cases": len(results) - passed_cases,
        "task_success_rate": passed_cases / len(results),
        "expected_actions": total_expected,
        "matched_actions": total_matched,
        "action_match_rate": (
            total_matched / total_expected if total_expected else 0.0
        ),
        "physical_actions": 0,
        "remote_model_calls": 0,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "cases": results,
        "evidence_boundary": (
            "离线只读动作序列回放；不等于视觉模型、真实 App 或机械臂真机验收。"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="离线任务级动作序列回放")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate_manifest(load_manifest(args.manifest))
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if report["failed_cases"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
