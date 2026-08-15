from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from PIL import Image

from vision_agent import (
    DashScopeVisionProvider,
    SEND_POST_STABLE_MAX_DELTA,
    VisionDecision,
    WECHAT_CANDIDATE_ROI_BOTTOM,
    WECHAT_CANDIDATE_ROI_TOP,
    WECHAT_INPUT_ROI_BOTTOM,
    WECHAT_INPUT_ROI_TOP,
    canonical_visible_pinyin,
    enlarged_vertical_roi,
    frame_mean_delta,
    numeric_grid_key_coordinate,
    sent_message_transition_metrics,
    validate_decision,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "evals" / "vision_replay" / "cases.json"
DEFAULT_CACHE_DIR = ROOT / "evals" / "vision_replay" / "cache"
DEFAULT_RESULTS_DIR = ROOT / "evals" / "vision_replay" / "results"


class ReplayValidationError(ValueError):
    pass


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ReplayValidationError("题库 schema_version 必须为 1。")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ReplayValidationError("题库必须包含非空 cases 数组。")

    seen: set[str] = set()
    for case in cases:
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ReplayValidationError("每道题必须有非空 id。")
        if case_id in seen:
            raise ReplayValidationError(f"题目 id 重复：{case_id}")
        seen.add(case_id)
        images = case.get("images")
        if not isinstance(images, list) or not images:
            raise ReplayValidationError(f"{case_id} 缺少 images。")
        for relative in images:
            image_path = path.parent / str(relative)
            if not image_path.is_file():
                raise ReplayValidationError(f"{case_id} 图片不存在：{image_path}")
            with Image.open(image_path) as image:
                image.verify()
        request = case.get("request")
        if not isinstance(request, dict) or not request.get("goal"):
            raise ReplayValidationError(f"{case_id} 缺少 request.goal。")
        expected = case.get("expected")
        if not isinstance(expected, dict) or not expected.get("accepted"):
            raise ReplayValidationError(f"{case_id} 缺少 expected.accepted。")
    return payload


def select_cases(
    manifest: dict[str, Any],
    *,
    case_ids: set[str] | None = None,
    tags: set[str] | None = None,
) -> list[dict[str, Any]]:
    selected = []
    for case in manifest["cases"]:
        if case_ids and case["id"] not in case_ids:
            continue
        case_tags = set(case.get("tags") or [])
        if tags and not tags.intersection(case_tags):
            continue
        selected.append(case)
    return selected


def load_case_frames(case: dict[str, Any], manifest_path: Path) -> list[Image.Image]:
    frames = [
        Image.open(manifest_path.parent / relative).convert("RGB")
        for relative in case["images"]
    ]
    if len(frames) < 4:
        frames = [frames[index % len(frames)].copy() for index in range(4)]
    zoom_kind = case.get("replay_zoom")
    if not zoom_kind:
        tags = set(case.get("tags") or [])
        history = (case.get("request") or {}).get("history") or []
        phases = {
            str(item.get("controller_phase"))
            for item in history
            if item.get("controller_phase")
        }
        if "awaiting_exact_pinyin_candidate" in phases:
            zoom_kind = "pinyin_candidate"
        elif tags.intersection(
            {
                "input-scope",
                "clear-verification",
                "input-verification",
                "send-verification",
            }
        ) or phases.intersection(
            {
                "verify_empty_after_clear",
                "awaiting_exact_pinyin_candidate",
                "verify_input_after_candidate",
                "verify_message_after_send",
            }
        ):
            zoom_kind = "wechat_bottom_input"
    if zoom_kind:
        source = frames[3]
        if zoom_kind == "pinyin_candidate":
            top = max(0, int(round(source.height * WECHAT_CANDIDATE_ROI_TOP)))
            bottom = min(
                source.height,
                max(
                    top + 1,
                    int(round(source.height * WECHAT_CANDIDATE_ROI_BOTTOM)),
                ),
            )
        elif zoom_kind == "comment_input":
            top = max(0, int(round(source.height * 0.22)))
            bottom = min(
                source.height,
                max(top + 1, int(round(source.height * 0.58))),
            )
        elif zoom_kind == "wechat_bottom_input":
            frames.append(
                enlarged_vertical_roi(
                    source,
                    top_ratio=WECHAT_INPUT_ROI_TOP,
                    bottom_ratio=WECHAT_INPUT_ROI_BOTTOM,
                )
            )
            return frames
        else:
            raise ReplayValidationError(
                f"{case['id']} replay_zoom 不受支持：{zoom_kind}"
            )
        zoom = source.crop((0, top, source.width, bottom))
        frames.append(
            zoom.resize(
                (zoom.width * 2, zoom.height * 2),
                Image.Resampling.LANCZOS,
            )
        )
    return frames


def _field(payload: dict[str, Any], dotted_name: str) -> Any:
    value: Any = payload
    for part in dotted_name.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _variant_matches(
    decision: dict[str, Any],
    variant: dict[str, Any],
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if "action" in variant and decision.get("action") != variant["action"]:
        errors.append(
            f"action={decision.get('action')!r}，期望 {variant['action']!r}"
        )
    if "action_in" in variant and decision.get("action") not in variant["action_in"]:
        errors.append(
            f"action={decision.get('action')!r}，允许 {variant['action_in']!r}"
        )
    for name, expected in (variant.get("fields") or {}).items():
        if decision.get(name) != expected:
            errors.append(
                f"{name}={decision.get(name)!r}，期望 {expected!r}"
            )
    for name, expected in (variant.get("normalized_fields") or {}).items():
        actual = decision.get(name)
        normalized = (
            canonical_visible_pinyin(actual) if isinstance(actual, str) else actual
        )
        if normalized != expected:
            errors.append(f"{name}规范化后={normalized!r}，期望 {expected!r}")
    for name, expected in (variant.get("nested_fields") or {}).items():
        actual = _field(decision, name)
        if actual != expected:
            errors.append(f"{name}={actual!r}，期望 {expected!r}")
    required_target = variant.get("target_contains") or []
    target = str(decision.get("target") or "")
    for token in required_target:
        if token not in target:
            errors.append(f"target 缺少 {token!r}")
    region = variant.get("coordinate_region")
    if region is not None:
        coordinate = decision.get("coordinate")
        if (
            not isinstance(coordinate, list)
            or len(coordinate) != 2
            or not (
                region[0] <= coordinate[0] <= region[2]
                and region[1] <= coordinate[1] <= region[3]
            )
        ):
            errors.append(f"coordinate={coordinate!r} 不在区域 {region!r}")
    return not errors, errors


def score_decision(
    decision: VisionDecision | dict[str, Any],
    expected: dict[str, Any],
) -> dict[str, Any]:
    payload = decision.to_dict() if isinstance(decision, VisionDecision) else decision
    forbidden = set(expected.get("forbidden_actions") or [])
    if payload.get("action") in forbidden:
        return {
            "passed": False,
            "errors": [f"命中禁止动作：{payload.get('action')}"],
        }

    variant_errors: list[list[str]] = []
    for variant in expected["accepted"]:
        matched, errors = _variant_matches(payload, variant)
        if matched:
            return {"passed": True, "errors": []}
        variant_errors.append(errors)
    best = min(variant_errors, key=len) if variant_errors else ["没有验收分支"]
    return {"passed": False, "errors": best}


def rescore_report(
    report_path: Path,
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Re-apply current golden scoring to an existing model report.

    No provider is constructed and no image is submitted to a model.  The raw
    decisions, usage and original report remain preserved for auditability.
    """

    manifest = load_manifest(manifest_path.resolve())
    cases_by_id = {case["id"]: case for case in manifest["cases"]}
    source = json.loads(report_path.read_text(encoding="utf-8"))
    results: list[dict[str, Any]] = []
    for original in source.get("results") or []:
        result = dict(original)
        case_id = result.get("case_id")
        case = cases_by_id.get(case_id)
        if case is None:
            raise ReplayValidationError(f"重评分报告含未知题目：{case_id}")
        controller_decision = controller_owned_replay_decision(
            case,
            manifest_path=manifest_path,
        )
        decision = result.get("decision")
        if controller_decision is not None:
            normalized_decision = controller_decision
            result["source"] = "controller-rescore"
        elif decision is None:
            normalized_decision = None
        else:
            normalized_decision = validate_decision(decision)
            with Image.open(manifest_path.parent / case["images"][0]) as source_frame:
                frame_size = source_frame.size
            normalized_decision = normalize_replay_decision(
                normalized_decision,
                case,
                frame_size,
            )
        if normalized_decision is None:
            result["passed"] = False
            result["errors"] = result.get("errors") or ["原报告没有有效模型决策。"]
        else:
            result["decision"] = normalized_decision.to_dict()
            score = score_decision(normalized_decision, case["expected"])
            result["passed"] = score["passed"]
            result["errors"] = score["errors"]
        results.append(result)

    payload = dict(source)
    payload["rescored_at"] = dt.datetime.now().astimezone().isoformat(
        timespec="seconds"
    )
    payload["rescored_from"] = str(report_path.resolve())
    payload["original_passed"] = source.get("passed")
    payload["passed"] = sum(1 for result in results if result["passed"])
    payload["total"] = len(results)
    payload["new_token_consumption"] = 0
    payload["results"] = results
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return payload


def _prompt_fingerprint(
    case: dict[str, Any],
    manifest_path: Path,
    model: str,
) -> str:
    digest = hashlib.sha256()
    digest.update((ROOT / "vision_agent.py").read_bytes())
    digest.update((ROOT / "vision_model_config.py").read_bytes())
    digest.update(json.dumps(case, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    digest.update(model.encode("utf-8"))
    for relative in case["images"]:
        digest.update((manifest_path.parent / relative).read_bytes())
    return digest.hexdigest()[:20]


def _cached_decision(path: Path) -> VisionDecision | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return validate_decision(payload["decision"])


def controller_owned_replay_decision(
    case: dict[str, Any],
    *,
    manifest_path: Path | None = None,
) -> VisionDecision | None:
    """Reproduce phases that never consult the model in the live runner."""

    history = (case.get("request") or {}).get("history") or []
    phase = next(
        (
            item
            for item in reversed(history)
            if item.get("controller_phase")
        ),
        None,
    )
    if not phase:
        return None
    if (
        phase.get("controller_phase") == "verify_message_after_send"
        and "stable_unchanged_checks" in phase
        and int(phase.get("stable_unchanged_checks") or 0) <= 1
    ):
        return VisionDecision(
            screen_type="chat",
            action="wait",
            confidence=1.0,
            reason="发送后的首次观察由本地控制器固定等待。",
            target="发送后首次观察",
            wait_seconds=1.0,
        )
    if phase.get("controller_phase") == "verify_message_after_send":
        if manifest_path is not None and len(case.get("images") or []) >= 3:
            raw_frames = [
                Image.open(manifest_path.parent / relative).convert("RGB")
                for relative in case["images"]
            ]
            transition = sent_message_transition_metrics(
                raw_frames[0],
                raw_frames[1],
            )
            stable_delta = frame_mean_delta(raw_frames[1], raw_frames[2])
            for frame in raw_frames:
                frame.close()
            if (
                transition["transition_visible"]
                and stable_delta <= SEND_POST_STABLE_MAX_DELTA
            ):
                return VisionDecision(
                    screen_type="chat",
                    action="finish",
                    confidence=1.0,
                    reason=(
                        "本地控制器检测到发送前后输入区和消息区同时变化，"
                        "且发送后画面再次观察保持稳定。"
                    ),
                    target=f"新的本人消息气泡：{phase.get('text') or ''}",
                    input_is_empty=True,
                    sent_message_visible=True,
                    success=True,
                )
        retry_coordinate = phase.get("send_coordinate")
        if (
            int(phase.get("stable_unchanged_checks") or 0) >= 4
            and phase.get("send_retry_used") is False
            and isinstance(phase.get("text"), str)
            and isinstance(retry_coordinate, list)
            and len(retry_coordinate) == 2
        ):
            return VisionDecision(
                screen_type="chat",
                action="tap",
                confidence=1.0,
                reason="本地控制器复用历史已验证的发送按钮坐标补点一次。",
                target="发送按钮（控制器单次补点）",
                coordinate=[int(retry_coordinate[0]), int(retry_coordinate[1])],
                observed_input_text=str(phase["text"]),
            )
    if (
        phase.get("controller_phase") == "verify_empty_after_clear"
        and int(phase.get("stable_empty_checks") or 0) >= 2
    ):
        return VisionDecision(
            screen_type="keyboard",
            action="finish",
            confidence=1.0,
            reason="本地控制器已连续两次确认底部输入框为空。",
            target="底部输入框为空",
            input_is_empty=True,
            success=True,
        )
    return None


def normalize_replay_decision(
    decision: VisionDecision,
    case: dict[str, Any],
    frame_size: tuple[int, int],
) -> VisionDecision:
    """Apply the same deterministic coordinate policy as the live runner."""

    history = (case.get("request") or {}).get("history") or []
    phase = next(
        (item for item in reversed(history) if item.get("controller_phase")),
        {},
    )
    if (
        phase.get("controller_phase") == "verify_comment_input"
        and phase.get("keyboard_profile") == "installed_numeric_symbol"
        and decision.action in {"wait", "stop"}
        and isinstance(decision.observed_input_text, str)
        and 1 <= len(decision.observed_input_text) <= 20
        and decision.observed_input_text != phase.get("text")
        and decision.input_is_empty is False
    ):
        return VisionDecision(
            screen_type="keyboard",
            action="clear_text",
            confidence=max(0.72, decision.confidence),
            reason="回放控制器按已标定键盘和实际可见错误字符数进行确定性清空。",
            target="已标定数字/符号键盘退格键",
            observed_input_text=decision.observed_input_text,
            delete_count=len(decision.observed_input_text),
            keyboard_layout={
                "type": "generic",
                "anchors": {"backspace": [862, 844]},
            },
        )
    if (
        decision.action == "type_symbol"
        and decision.text
        and decision.text.isdigit()
        and decision.keyboard_layout is not None
        and decision.keyboard_layout.get("type") == "numeric_grid"
    ):
        return replace(
            decision,
            coordinate=numeric_grid_key_coordinate(
                decision.keyboard_layout,
                decision.text,
                frame_size,
            ),
        )
    return decision


def run_case(
    case: dict[str, Any],
    *,
    manifest_path: Path,
    provider: DashScopeVisionProvider,
    use_cache: bool,
) -> dict[str, Any]:
    local_decision = controller_owned_replay_decision(
        case,
        manifest_path=manifest_path,
    )
    if local_decision is not None:
        score = score_decision(local_decision, case["expected"])
        return {
            "case_id": case["id"],
            "title": case["title"],
            "passed": score["passed"],
            "errors": score["errors"],
            "cached": False,
            "source": "controller",
            "decision": local_decision.to_dict(),
            "usage": {},
        }

    fingerprint = _prompt_fingerprint(case, manifest_path, provider.model)
    cache_path = DEFAULT_CACHE_DIR / case["id"] / f"{fingerprint}.json"
    decision = _cached_decision(cache_path) if use_cache else None
    cached = decision is not None
    if decision is None:
        request = case["request"]
        frames = load_case_frames(case, manifest_path)
        decision = provider.decide(
            goal=request["goal"],
            frames=frames,
            history=request.get("history") or [],
            allowed_texts=request.get("allowed_texts") or [],
            task_mode=request.get("task_mode") or "operate",
            expected_result=request.get("expected_result"),
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {
                    "case_id": case["id"],
                    "fingerprint": fingerprint,
                    "model": provider.model,
                    "decision": decision.to_dict(),
                    "usage": provider.last_usage,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    with Image.open(manifest_path.parent / case["images"][0]) as source_frame:
        frame_size = source_frame.size
    decision = normalize_replay_decision(decision, case, frame_size)
    score = score_decision(decision, case["expected"])
    return {
        "case_id": case["id"],
        "title": case["title"],
        "passed": score["passed"],
        "errors": score["errors"],
        "cached": cached,
        "source": "cache" if cached else "model",
        "decision": decision.to_dict(),
        "usage": {} if cached else provider.last_usage,
    }


def run_case_safely(
    case: dict[str, Any],
    *,
    manifest_path: Path,
    provider: DashScopeVisionProvider,
    use_cache: bool,
) -> dict[str, Any]:
    """Isolate one invalid model response from the rest of a baseline run.

    A malformed or internally inconsistent decision is itself a failed replay
    result.  It must not abort the remaining golden cases or hide their scores.
    """

    try:
        return run_case(
            case,
            manifest_path=manifest_path,
            provider=provider,
            use_cache=use_cache,
        )
    except Exception as exc:
        return {
            "case_id": case["id"],
            "title": case["title"],
            "passed": False,
            "errors": [f"模型输出无效或请求失败：{exc}"],
            "cached": False,
            "decision": None,
            "usage": provider.last_usage,
            "exception_type": type(exc).__name__,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="离线回放历史手机视觉题库。")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--case", action="append", dest="case_ids")
    parser.add_argument("--tag", action="append", dest="tags")
    parser.add_argument("--list", action="store_true", help="列出题目，不调用模型。")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="只校验题库、图片和评分规则，不调用模型。",
    )
    parser.add_argument(
        "--run-model",
        action="store_true",
        help="明确允许调用千问模型并产生 token。",
    )
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--rescore-report",
        type=Path,
        help="只用当前评分规则重算已有报告，不调用模型。",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest_path = args.manifest.resolve()
    if args.rescore_report:
        source = args.rescore_report.resolve()
        output = source.with_name(source.stem + "_rescored.json")
        payload = rescore_report(
            source,
            manifest_path=manifest_path,
            output_path=output,
        )
        print(
            f"重评分：{payload['original_passed']} -> "
            f"{payload['passed']}/{payload['total']}；新消耗 token：0"
        )
        print(f"报告：{output}")
        return 0
    manifest = load_manifest(manifest_path)
    cases = select_cases(
        manifest,
        case_ids=set(args.case_ids or []),
        tags=set(args.tags or []),
    )
    if not cases:
        raise ReplayValidationError("筛选后没有题目。")

    if args.list:
        for case in cases:
            print(f"{case['id']}: {case['title']} [{', '.join(case['tags'])}]")
        return 0
    if args.validate_only or not args.run_model:
        print(f"题库校验通过：{len(cases)} 题。未调用模型，未产生 token。")
        return 0

    provider = DashScopeVisionProvider()
    if not provider.configured:
        raise ReplayValidationError("未配置 DASHSCOPE_API_KEY，无法运行模型回放。")

    results = [
        run_case_safely(
            case,
            manifest_path=manifest_path,
            provider=provider,
            use_cache=not args.no_cache,
        )
        for case in cases
    ]
    passed = sum(1 for result in results if result["passed"])
    total_tokens = sum(
        int((result.get("usage") or {}).get("total_tokens") or 0)
        for result in results
    )
    payload = {
        "schema_version": 1,
        "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": provider.model,
        "passed": passed,
        "total": len(results),
        "total_tokens": total_tokens,
        "results": results,
    }
    DEFAULT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = DEFAULT_RESULTS_DIR / (
        dt.datetime.now().strftime("replay_%Y%m%d_%H%M%S") + ".json"
    )
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for result in results:
        status = "PASS" if result["passed"] else "FAIL"
        source = result.get("source") or ("cache" if result["cached"] else "model")
        print(f"[{status}] {result['case_id']} ({source})")
        for error in result["errors"]:
            print(f"  - {error}")
    print(f"结果：{passed}/{len(results)}；本次新消耗 token：{total_tokens}")
    print(f"报告：{output}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
