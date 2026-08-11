from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def _iter_reports(root: Path) -> Iterable[Path]:
    for path in root.rglob("report.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if payload.get("architecture") == "page_state_graph_v1":
            yield path


def _merge_counts(target: Counter[str], values: dict[str, Any]) -> None:
    for key, value in values.items():
        try:
            target[str(key)] += int(value)
        except (TypeError, ValueError):
            continue


def summarize_reports(root: Path) -> dict[str, Any]:
    reports: list[tuple[Path, dict[str, Any]]] = []
    for path in _iter_reports(root):
        reports.append((path, json.loads(path.read_text(encoding="utf-8"))))

    instrumented = [item for item in reports if isinstance(item[1].get("reliability_metrics"), dict)]
    successes = sum(bool(payload.get("success")) for _, payload in instrumented)
    actions_issued = 0
    actions_confirmed = 0
    recovery_issued = 0
    recovery_confirmed = 0
    operation_totals: Counter[str] = Counter()
    operation_successes: Counter[str] = Counter()
    failure_stages: Counter[str] = Counter()
    target_methods: Counter[str] = Counter()
    action_kinds: Counter[str] = Counter()
    states_seen: Counter[str] = Counter()
    failed_reports: list[str] = []

    for path, payload in instrumented:
        metrics = dict(payload["reliability_metrics"])
        operation = str(payload.get("operation") or "unknown")
        operation_totals[operation] += 1
        if payload.get("success"):
            operation_successes[operation] += 1
        else:
            failed_reports.append(str(path))
            failure_stages[str(metrics.get("failure_stage") or "unknown")] += 1
        actions_issued += int(metrics.get("actions_issued", 0))
        actions_confirmed += int(metrics.get("actions_confirmed", 0))
        recovery_issued += int(metrics.get("recovery_actions_issued", 0))
        recovery_confirmed += int(metrics.get("recovery_actions_confirmed", 0))
        _merge_counts(target_methods, dict(metrics.get("target_resolution_methods") or {}))
        _merge_counts(action_kinds, dict(metrics.get("action_kinds") or {}))
        _merge_counts(states_seen, dict(metrics.get("states_seen") or {}))

    operation_summary = {
        operation: {
            "runs": count,
            "successes": operation_successes[operation],
            "success_rate": round(operation_successes[operation] / count, 4),
        }
        for operation, count in sorted(operation_totals.items())
    }
    total = len(instrumented)
    return {
        "root": str(root),
        "state_graph_reports_found": len(reports),
        "instrumented_runs": total,
        "uninstrumented_runs": len(reports) - total,
        "successes": successes,
        "failures": total - successes,
        "success_rate": round(successes / total, 4) if total else None,
        "actions_issued": actions_issued,
        "actions_confirmed": actions_confirmed,
        "action_confirmation_rate": (
            round(actions_confirmed / actions_issued, 4) if actions_issued else None
        ),
        "recovery_actions_issued": recovery_issued,
        "recovery_actions_confirmed": recovery_confirmed,
        "recovery_confirmation_rate": (
            round(recovery_confirmed / recovery_issued, 4) if recovery_issued else None
        ),
        "operations": operation_summary,
        "failure_stages": dict(failure_stages),
        "target_resolution_methods": dict(target_methods),
        "action_kinds": dict(action_kinds),
        "states_seen": dict(states_seen),
        "failed_reports": failed_reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="汇总页面状态图实机运行可靠性")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / "web",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = summarize_reports(args.root.resolve())
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
