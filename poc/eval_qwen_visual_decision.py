from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

from generic_scene_observer import GenericSceneObserver
from qwen_visual_decision import (
    QwenTaskContext,
    QwenVisualDecisionObserver,
    TrustedObservation,
)
from vision_agent import DashScopeVisionProvider, VisionAgentError


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "evals" / "qwen_visual_decision" / "cases.json"
DEFAULT_OUTPUT_ROOT = ROOT / "output" / "offline_qwen_visual_decision"


def _load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("cases"), list):
        raise ValueError("离线用例清单缺少 cases 数组。")
    return value


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


def _score(
    case: dict[str, Any],
    *,
    status: str,
    decision: dict[str, Any] | None,
) -> dict[str, Any]:
    accepted_statuses = [str(item) for item in case.get("accepted_statuses") or []]
    reasons: list[str] = []
    if accepted_statuses and status not in accepted_statuses:
        reasons.append(f"status={status!r} 不在 {accepted_statuses!r}")
    accepted_actions = [str(item) for item in case.get("accepted_actions") or []]
    next_action = (decision or {}).get("next_action") or {}
    action = str(next_action.get("action") or "") if isinstance(next_action, dict) else ""
    if status == "action" and accepted_actions and action not in accepted_actions:
        reasons.append(f"action={action!r} 不在 {accepted_actions!r}")
    if status == "action" and isinstance(next_action, list):
        reasons.append("next_action 不能是动作列表")
    return {"passed": not reasons, "reasons": reasons}


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="仅使用已有截图调用Qwen，离线评估可信观察和唯一下一视觉动作。"
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
    scene_observer = GenericSceneObserver(provider)
    decision_observer = QwenVisualDecisionObserver(provider)
    run_dir = DEFAULT_OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    results: list[dict[str, Any]] = []

    for index, case in enumerate(selected, start=1):
        decision_observer.last_diagnostics = {}
        frames, frame_paths = _load_frames(case, manifest_path)
        context = QwenTaskContext.from_dict(dict(case["task_context"]))
        observation: TrustedObservation | None = None
        try:
            scene = scene_observer.observe(
                frames=frames,
                goal_context=context.to_dict(),
            )
            observation = TrustedObservation.from_scene(
                frames=frames,
                device_id=context.device_id,
                scene=scene,
                observation_id=f"obs_{uuid.uuid4().hex}",
            )
            decision = decision_observer.decide(
                frames=frames,
                task_context=context,
                trusted_observation=observation,
                decision_number=index,
            )
            value = decision.to_dict()
            status = decision.proposal.status
            result = {
                "case_id": case["id"],
                "frame_paths": frame_paths,
                "task_context": context.to_dict(),
                "observation": observation.to_dict(),
                "observation_diagnostics": dict(scene_observer.last_diagnostics),
                "decision": value,
                "decision_diagnostics": dict(decision_observer.last_diagnostics),
                "status": status,
                "score": _score(case, status=status, decision=value),
            }
        except (VisionAgentError, ValueError, TypeError) as exc:
            # Observation failure is a local safety block. Crucially, the
            # decision selector has not received a candidate action.
            observation_diagnostics = dict(scene_observer.last_diagnostics)
            local_block = observation is None
            status = "blocked" if local_block else "error"
            result = {
                "case_id": case["id"],
                "frame_paths": frame_paths,
                "task_context": context.to_dict(),
                "observation": observation.to_dict() if observation else None,
                "observation_diagnostics": observation_diagnostics,
                "decision": None,
                "decision_diagnostics": dict(decision_observer.last_diagnostics),
                "status": status,
                "local_safety_block": local_block,
                "error": str(exc),
                "score": _score(case, status=status, decision=None),
            }
        results.append(result)
        (run_dir / f"{case['id']}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    statuses = [str(item["status"]) for item in results]
    decision_metrics = decision_observer.status()
    report = {
        "mode": "offline_existing_screenshots_only",
        "model": provider.model,
        "hardware_actions_enabled": False,
        "case_count": len(results),
        "passed": sum(1 for item in results if item["score"]["passed"]),
        "failed": sum(1 for item in results if not item["score"]["passed"]),
        "case_outcome_rates": {
            "action_rate": _rate(statuses.count("action"), len(statuses)),
            "finished_rate": _rate(statuses.count("finished"), len(statuses)),
            "final_blocked_rate": _rate(statuses.count("blocked"), len(statuses)),
        },
        "decision_format_metrics": {
            "first_pass_rate": decision_metrics["first_pass_rate"],
            "repair_retry_rate": decision_metrics["repair_retry_rate"],
            "final_blocked_rate": decision_metrics["final_blocked_rate"],
            "model_attempted_count": decision_metrics["model_attempted_count"],
            "first_pass_success_count": decision_metrics["first_pass_success_count"],
            "retry_success_count": decision_metrics["retry_success_count"],
            "final_blocked_count": decision_metrics["final_blocked_count"],
        },
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
                "decision_format_metrics": report["decision_format_metrics"],
                "hardware_actions_enabled": False,
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
