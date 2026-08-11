from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

from qwen_visual_decision import QwenVisualDecisionObserver
from vision_agent import DashScopeVisionProvider, VisionAgentError


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "evals" / "qwen_visual_decision" / "cases.json"
DEFAULT_OUTPUT_ROOT = ROOT / "output" / "offline_qwen_visual_decision"


def _load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("cases"), list):
        raise ValueError("离线用例清单缺少 cases 数组。")
    return value


def _score(case: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    status = str(decision.get("status") or "")
    accepted_statuses = [str(item) for item in case.get("accepted_statuses") or []]
    reasons: list[str] = []
    if accepted_statuses and status not in accepted_statuses:
        reasons.append(f"status={status!r} 不在 {accepted_statuses!r}")
    accepted_actions = [str(item) for item in case.get("accepted_actions") or []]
    next_action = decision.get("next_action") or {}
    action = str(next_action.get("action") or "") if isinstance(next_action, dict) else ""
    if status == "action" and accepted_actions and action not in accepted_actions:
        reasons.append(f"action={action!r} 不在 {accepted_actions!r}")
    return {"passed": not reasons, "reasons": reasons}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="仅使用已有截图调用Qwen，离线评估页面状态和唯一下一视觉动作。"
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--case", action="append", dest="case_ids")
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = _load_manifest(manifest_path)
    selected = [
        item
        for item in manifest["cases"]
        if not args.case_ids or str(item.get("id")) in set(args.case_ids)
    ]
    if not selected:
        raise ValueError("没有匹配的离线Qwen视觉决策用例。")

    provider = DashScopeVisionProvider()
    if not provider.configured:
        raise RuntimeError("未配置 DASHSCOPE_API_KEY，不能运行真实Qwen离线截图测试。")
    observer = QwenVisualDecisionObserver(provider)
    run_dir = DEFAULT_OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    results: list[dict[str, Any]] = []

    for index, case in enumerate(selected, start=1):
        image_path = (manifest_path.parent / str(case["image"])).resolve()
        with Image.open(image_path) as source:
            frame = source.convert("RGB")
        try:
            decision = observer.decide(
                frames=[frame.copy() for _ in range(4)],
                device_id=str(case["device_id"]),
                current_subgoal=dict(case["current_subgoal"]),
                constraints=list(case.get("constraints") or []),
                decision_number=index,
            )
            value = decision.to_dict()
            score = _score(case, value)
            result = {
                "case_id": case["id"],
                "device_id": case["device_id"],
                "image": str(image_path),
                "current_subgoal": case["current_subgoal"],
                "decision": value,
                "diagnostics": dict(observer.last_diagnostics),
                "score": score,
            }
        except (VisionAgentError, ValueError, TypeError) as exc:
            result = {
                "case_id": case["id"],
                "device_id": case["device_id"],
                "image": str(image_path),
                "current_subgoal": case["current_subgoal"],
                "decision": None,
                "diagnostics": dict(observer.last_diagnostics),
                "raw_response": observer.last_raw_response,
                "score": {"passed": False, "reasons": [str(exc)]},
            }
        results.append(result)
        (run_dir / f"{case['id']}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    report = {
        "mode": "offline_existing_screenshots_only",
        "model": provider.model,
        "hardware_actions_enabled": False,
        "case_count": len(results),
        "passed": sum(1 for item in results if item["score"]["passed"]),
        "failed": sum(1 for item in results if not item["score"]["passed"]),
        "results": results,
    }
    report_path = run_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "report": str(report_path),
                "case_count": report["case_count"],
                "passed": report["passed"],
                "failed": report["failed"],
                "hardware_actions_enabled": False,
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
