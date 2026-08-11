from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "output" / "web"
REPLAY_ROOT = ROOT / "evals" / "vision_replay"
MANIFEST_PATH = REPLAY_ROOT / "cases.json"
HISTORY_IMAGE_ROOT = REPLAY_ROOT / "history" / "images"
HISTORY_INDEX_PATH = REPLAY_ROOT / "history" / "history_index.json"


def _golden_sources() -> dict[tuple[str, int], str]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    result: dict[tuple[str, int], str] = {}
    for case in manifest["cases"]:
        source = case.get("source") or {}
        run = source.get("run")
        step = source.get("step")
        if isinstance(run, str) and isinstance(step, int):
            result[(run, step)] = case["id"]
    return result


def build_history_index() -> dict[str, Any]:
    golden_sources = _golden_sources()
    items: list[dict[str, Any]] = []
    reports = sorted(OUTPUT_ROOT.rglob("report.json"))
    report_outcomes: dict[str, str] = {}
    for report_path in reports:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        run = report_path.parent.name
        outcome = str(report.get("outcome") or "unknown")
        report_outcomes[run] = outcome
        steps = report.get("steps") or []
        evidence = report.get("evidence") or []
        for index, evidence_value in enumerate(evidence, start=1):
            source_image = report_path.parent / Path(str(evidence_value)).name
            if not source_image.is_file():
                continue
            destination_dir = HISTORY_IMAGE_ROOT / run
            destination_dir.mkdir(parents=True, exist_ok=True)
            destination = destination_dir / source_image.name
            shutil.copy2(source_image, destination)
            step = steps[index - 1] if index <= len(steps) else {}
            step_number = int(step.get("step") or index)
            golden_case_id = golden_sources.get((run, step_number))
            if golden_case_id:
                review_status = "golden"
            elif outcome in {"passed", "completed"}:
                review_status = "success_candidate"
            else:
                review_status = "failure_candidate"
            items.append(
                {
                    "id": f"{run}_step_{step_number:03d}",
                    "run": run,
                    "step": step_number,
                    "outcome": outcome,
                    "goal": report.get("goal"),
                    "source_report": str(report_path.relative_to(ROOT)),
                    "source_image": str(source_image.relative_to(ROOT)),
                    "image": str(destination.relative_to(REPLAY_ROOT)),
                    "review_status": review_status,
                    "golden_case_id": golden_case_id,
                    "recorded_decision": step,
                    "task_error": (report.get("error") or {}).get("message"),
                }
            )

    stats = {
        "reports": len(reports),
        "screenshots": len(items),
        "successful_reports": sum(
            outcome in {"passed", "completed"}
            for outcome in report_outcomes.values()
        ),
        "failed_reports": sum(
            outcome == "failed" for outcome in report_outcomes.values()
        ),
        "indexed_successful_reports": len(
            {
                item["run"]
                for item in items
                if item["outcome"] in {"passed", "completed"}
            }
        ),
        "indexed_failed_reports": len(
            {item["run"] for item in items if item["outcome"] == "failed"}
        ),
        "reports_without_screenshots": len(reports)
        - len({item["run"] for item in items}),
        "golden": sum(item["review_status"] == "golden" for item in items),
        "success_candidates": sum(
            item["review_status"] == "success_candidate" for item in items
        ),
        "failure_candidates": sum(
            item["review_status"] == "failure_candidate" for item in items
        ),
    }
    payload = {
        "schema_version": 1,
        "description": (
            "Complete historical screenshot index. Only entries marked golden "
            "are allowed to participate in the model regression gate."
        ),
        "stats": stats,
        "items": items,
    }
    HISTORY_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_INDEX_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def main() -> int:
    payload = build_history_index()
    stats = payload["stats"]
    print(
        "历史索引完成："
        f"{stats['reports']} 份报告，{stats['screenshots']} 张截图；"
        f"成功报告 {stats['successful_reports']}，失败报告 {stats['failed_reports']}；"
        f"有图成功来源 {stats['indexed_successful_reports']}，"
        f"有图失败来源 {stats['indexed_failed_reports']}，"
        f"缺图报告 {stats['reports_without_screenshots']}；"
        f"金标 {stats['golden']}，待审成功 {stats['success_candidates']}，"
        f"待审失败 {stats['failure_candidates']}。"
    )
    print(f"索引：{HISTORY_INDEX_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
