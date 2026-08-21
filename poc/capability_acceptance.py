from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import threading
from typing import Any, Callable, Mapping
import uuid

from PIL import Image, UnidentifiedImageError
from orientation_safety import (
    OrientationCredential,
    OrientationSafetyError,
    frame_fingerprint,
)

from device_exclusivity import InterProcessLease
from tap_calibration import (
    Affine2D,
    CALIBRATION_VERSION,
    MIN_COVERAGE_SPAN_X,
    MIN_COVERAGE_SPAN_Y,
    TapCalibrationError,
)
from ui_scene import UIScene, UISceneError
from universal_action_controller import (
    ResolvedSemanticAction,
    SAFE_VERIFIED_TEXT_RE,
    UniversalActionController,
    UniversalActionError,
)


PROMOTABLE_ACTIONS = frozenset(
    {
        "tap_semantic",
        "dismiss_overlay",
        "swipe",
        "back",
        "home",
        "reveal_system_navigation",
        "input_verified_text",
        "long_press",
        "drag",
    }
)
CALIBRATION_BOUND_ACTIONS = frozenset(
    {"long_press", "drag", "reveal_system_navigation"}
)
ACCEPTANCE_REPORT_VERSION = 3


def _normalized_coverage_bounds(value: Any, *, label: str) -> list[float]:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            for item in value
        )
    ):
        raise CapabilityAcceptanceError(f"{label}边界格式无效。")
    bounds = [float(item) for item in value]
    if (
        any(not 0.0 <= item <= 1.0 for item in bounds)
        or bounds[2] - bounds[0] < MIN_COVERAGE_SPAN_X
        or bounds[3] - bounds[1] < MIN_COVERAGE_SPAN_Y
    ):
        raise CapabilityAcceptanceError(f"{label}没有覆盖足够的归一化屏幕范围。")
    return bounds


def _validated_coverage(value: Any, *, label: str) -> list[float]:
    if not isinstance(value, dict) or value.get("sufficient") is not True:
        raise CapabilityAcceptanceError(f"{label}没有足够的实测屏幕覆盖。")
    bounds = _normalized_coverage_bounds(
        value.get("normalized_bounds"),
        label=label,
    )
    hull = value.get("normalized_hull")
    if (
        not isinstance(hull, list)
        or len(hull) < 3
        or any(
            not isinstance(point, list)
            or len(point) != 2
            or any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                or not 0.0 <= float(item) <= 1.0
                for item in point
            )
            for point in hull
        )
    ):
        raise CapabilityAcceptanceError(f"{label}凸包格式无效。")
    derived = [
        min(float(point[0]) for point in hull),
        min(float(point[1]) for point in hull),
        max(float(point[0]) for point in hull),
        max(float(point[1]) for point in hull),
    ]
    if any(abs(stored - computed) > 1e-6 for stored, computed in zip(bounds, derived)):
        raise CapabilityAcceptanceError(f"{label}边界与凸包不一致。")
    return bounds


def validated_calibration_evidence(path: Path) -> dict[str, Any]:
    """Load the exact active calibration state used by one gesture trial."""

    try:
        resolved = Path(path).resolve(strict=True)
        raw = resolved.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise CapabilityAcceptanceError(f"触控标定无法读取：{exc}") from exc
    if not isinstance(payload, dict):
        raise CapabilityAcceptanceError("触控标定必须是 JSON 对象。")
    version = payload.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version < CALIBRATION_VERSION:
        raise CapabilityAcceptanceError("触控标定版本过旧，不能用于正式手势验收。")
    if payload.get("enabled") is not True or payload.get("validated") is not True:
        raise CapabilityAcceptanceError("触控标定尚未启用并完成独立验证。")
    if payload.get("accepted_fit") is not True:
        raise CapabilityAcceptanceError("触控标定拟合尚未达到验收标准。")
    try:
        Affine2D.from_json(payload["target_to_command"])
    except (KeyError, TypeError, ValueError, TapCalibrationError) as exc:
        raise CapabilityAcceptanceError("触控标定变换矩阵无效。") from exc
    frame_size = payload.get("frame_size")
    if (
        not isinstance(frame_size, list)
        or len(frame_size) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, (int, float)) or item <= 0
            for item in frame_size
        )
    ):
        raise CapabilityAcceptanceError("触控标定 frame_size 无效。")
    bounds = _validated_coverage(payload.get("coverage"), label="触控标定采集覆盖")
    validation = payload.get("validation")
    if (
        not isinstance(validation, dict)
        or validation.get("passed") is not True
        or validation.get("coverage_passed") is not True
    ):
        raise CapabilityAcceptanceError("触控标定缺少通过的独立验证记录。")
    validation_bounds = _validated_coverage(
        validation.get("coverage"),
        label="触控标定独立验证覆盖",
    )
    return {
        "version": version,
        "sha256": _sha256_bytes(raw),
        "frame_size": [float(frame_size[0]), float(frame_size[1])],
        "coverage_bounds": bounds,
        "validation_coverage_bounds": validation_bounds,
    }


def _calibration_evidence(value: Any, *, action: str) -> dict[str, Any] | None:
    if action not in CALIBRATION_BOUND_ACTIONS:
        if value is not None:
            raise CapabilityAcceptanceError("非点位手势验收不能携带触控标定证据。")
        return None
    if not isinstance(value, dict) or set(value) != {
        "version",
        "sha256",
        "frame_size",
        "coverage_bounds",
        "validation_coverage_bounds",
    }:
        raise CapabilityAcceptanceError("手势验收缺少完整触控标定证据。")
    version = value.get("version")
    digest = value.get("sha256")
    frame_size = value.get("frame_size")
    bounds = _normalized_coverage_bounds(
        value.get("coverage_bounds"),
        label="手势验收采集覆盖",
    )
    validation_bounds = _normalized_coverage_bounds(
        value.get("validation_coverage_bounds"),
        label="手势验收独立验证覆盖",
    )
    if isinstance(version, bool) or not isinstance(version, int) or version < CALIBRATION_VERSION:
        raise CapabilityAcceptanceError("手势验收触控标定版本无效。")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise CapabilityAcceptanceError("手势验收触控标定摘要无效。")
    if (
        not isinstance(frame_size, list)
        or len(frame_size) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, (int, float)) or item <= 0
            for item in frame_size
        )
    ):
        raise CapabilityAcceptanceError("手势验收触控标定范围无效。")
    return {
        "version": version,
        "sha256": digest,
        "frame_size": [float(item) for item in frame_size],
        "coverage_bounds": bounds,
        "validation_coverage_bounds": validation_bounds,
    }


def exact_input_evidence_error(execution: Any) -> str:
    """Return a fail-closed error when an input report lacks exact target evidence."""

    if not isinstance(execution, dict):
        return "输入验收 execution 必须是对象。"
    resolved = execution.get("resolved_action")
    before_scene = execution.get("before_scene")
    after_scene = execution.get("after_scene")
    if not all(isinstance(value, dict) for value in (resolved, before_scene, after_scene)):
        return "输入验收缺少结构化 resolved_action/before_scene/after_scene。"
    expected = resolved.get("text")
    target_id = str(resolved.get("target_element_id") or "").strip()
    if (
        not isinstance(expected, str)
        or not SAFE_VERIFIED_TEXT_RE.fullmatch(expected)
        or not target_id
    ):
        return "输入验收缺少精确文字或目标输入框身份。"

    before_elements = before_scene.get("elements")
    after_elements = after_scene.get("elements")
    if not isinstance(before_elements, list) or not isinstance(after_elements, list):
        return "输入验收缺少动作前后元素证据。"
    before_matches = [
        item
        for item in before_elements
        if isinstance(item, dict)
        and item.get("element_id") == target_id
        and item.get("role") == "input"
        and isinstance(item.get("confidence"), (int, float))
        and not isinstance(item.get("confidence"), bool)
        and float(item["confidence"]) >= 0.72
    ]
    if len(before_matches) != 1:
        return "输入验收无法唯一绑定动作前目标输入框。"
    before_input = before_matches[0]
    before_states = before_input.get("states")
    if not isinstance(before_states, dict):
        return "输入验收缺少动作前输入框 states。"
    if before_states.get("focused") is not True:
        return "输入验收要求动作前输入框已聚焦。"
    if before_states.get("goal_relevant") is not True:
        return "输入验收要求动作前输入框与当前目标明确相关。"
    if before_states.get("value") != "":
        return "输入验收只允许从动作前确认的空输入框开始。"
    if before_states.get("keyboard_layout") != "qwerty":
        return "输入验收要求动作前画面确认 QWERTY 键盘。"
    if before_states.get("keyboard_input_mode") != "direct_latin":
        return "输入验收要求动作前画面确认 direct_latin 直输模式。"
    eligible_before = [
        item
        for item in before_elements
        if isinstance(item, dict)
        and item.get("role") == "input"
        and isinstance(item.get("confidence"), (int, float))
        and not isinstance(item.get("confidence"), bool)
        and float(item["confidence"]) >= 0.72
        and isinstance(item.get("states"), dict)
        and item["states"].get("visible") is not False
        and item["states"].get("goal_relevant") is True
        and item["states"].get("focused") is True
        and item["states"].get("value") == ""
        and item["states"].get("keyboard_layout") == "qwerty"
        and item["states"].get("keyboard_input_mode") == "direct_latin"
    ]
    if len(eligible_before) != 1 or eligible_before[0].get("element_id") != target_id:
        return "输入验收要求动作前只有一个符合安全条件的目标输入框。"
    if (
        str(before_scene.get("foreground_app_id") or "").strip()
        != str(after_scene.get("foreground_app_id") or "").strip()
        or str(before_scene.get("screen_id") or "").strip()
        != str(after_scene.get("screen_id") or "").strip()
    ):
        return "输入验收动作后 App 或页面身份发生变化。"

    def visible_states(item: dict[str, Any]) -> dict[str, Any] | None:
        states = item.get("states")
        return states if isinstance(states, dict) else None

    exact_id = [
        item
        for item in after_elements
        if isinstance(item, dict)
        and item.get("element_id") == target_id
        and item.get("role") == "input"
        and isinstance(item.get("confidence"), (int, float))
        and not isinstance(item.get("confidence"), bool)
        and float(item["confidence"]) >= 0.72
        and visible_states(item) is not None
        and visible_states(item).get("visible") is not False
    ]
    if exact_id:
        candidates = exact_id
    else:
        candidates = [
            item
            for item in after_elements
            if isinstance(item, dict)
            and item.get("role") == "input"
            and isinstance(item.get("confidence"), (int, float))
            and not isinstance(item.get("confidence"), bool)
            and float(item["confidence"]) >= 0.72
            and visible_states(item) is not None
            and visible_states(item).get("visible") is not False
            and str(item.get("meaning") or "").casefold()
            == str(before_input.get("meaning") or "").casefold()
            and str(item.get("label") or "").casefold()
            == str(before_input.get("label") or "").casefold()
        ]
    if len(candidates) != 1:
        return "输入验收无法唯一绑定动作后目标输入框。"
    states = candidates[0].get("states")
    if not isinstance(states, dict) or not isinstance(states.get("value"), str):
        return "输入验收缺少动作后 states.value 精确文字证据。"
    actual = states["value"]
    if actual != expected:
        return f"输入验收文字不匹配：实际 {actual!r}，预期 {expected!r}。"
    return ""


def _normalized_point(value: Any, *, field: str) -> tuple[float, float] | None:
    if value is None:
        return None
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, (int, float))
            for item in value
        )
    ):
        raise CapabilityAcceptanceError(f"验收执行字段 {field} 格式无效。")
    return float(value[0]), float(value[1])


def action_execution_evidence_error(action: str, execution: Any) -> str:
    """Independently replay controller verification from persisted scenes."""

    if not isinstance(execution, dict):
        return "验收 execution 必须是对象。"
    raw_resolved = execution.get("resolved_action")
    raw_before = execution.get("before_scene")
    raw_after = execution.get("after_scene")
    if not all(isinstance(value, dict) for value in (raw_resolved, raw_before, raw_after)):
        return "验收缺少结构化 resolved_action/before_scene/after_scene。"
    try:
        expected_effect = raw_resolved.get("expected_effect") or {}
        if not isinstance(expected_effect, dict):
            raise CapabilityAcceptanceError("验收执行字段 expected_effect 格式无效。")
        hold_seconds = raw_resolved.get("hold_seconds")
        path_distance = raw_resolved.get("path_distance")
        for field, value in (
            ("hold_seconds", hold_seconds),
            ("path_distance", path_distance),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float))
            ):
                raise CapabilityAcceptanceError(f"验收执行字段 {field} 格式无效。")
        resolved_kind = str(raw_resolved.get("kind") or "").strip()
        resolved_text = (
            raw_resolved.get("text")
            if isinstance(raw_resolved.get("text"), str)
            else None
        )
        resolved = ResolvedSemanticAction(
            node_id=str(raw_resolved.get("node_id") or "acceptance"),
            kind=resolved_kind,
            normalized_point=_normalized_point(
                raw_resolved.get("normalized_point"),
                field="normalized_point",
            ),
            normalized_end_point=_normalized_point(
                raw_resolved.get("normalized_end_point"),
                field="normalized_end_point",
            ),
            text=resolved_text,
            input_fragment=(
                str(raw_resolved.get("input_fragment") or "") or None
                if raw_resolved.get("input_fragment") is not None
                else None
            ),
            input_method=(
                str(raw_resolved.get("input_method") or "") or None
            ),
            input_pinyin=(
                str(raw_resolved.get("input_pinyin") or "") or None
            ),
            prior_input_value=(
                str(raw_resolved.get("prior_input_value") or "")
                if raw_resolved.get("prior_input_value") is not None
                else None
            ),
            expected_input_value=(
                str(raw_resolved.get("expected_input_value") or "")
                if raw_resolved.get("expected_input_value") is not None
                else None
            ),
            direction=(
                raw_resolved.get("direction")
                if isinstance(raw_resolved.get("direction"), str)
                else None
            ),
            hold_seconds=(float(hold_seconds) if hold_seconds is not None else None),
            path_distance=(float(path_distance) if path_distance is not None else None),
            target_element_id=(
                str(raw_resolved.get("target_element_id") or "").strip() or None
            ),
            destination_element_id=(
                str(raw_resolved.get("destination_element_id") or "").strip()
                or None
            ),
            before_fingerprint=str(
                raw_resolved.get("before_fingerprint") or ""
            ).strip(),
            expected_effect=dict(expected_effect),
        )
        if resolved.kind != action:
            return "执行动作类型与候选动作类型不一致。"
        before = UIScene.from_dict(raw_before)
        after = UIScene.from_dict(raw_after)
        UniversalActionController().verify_after_action(resolved, before, after)
    except (CapabilityAcceptanceError, UISceneError, UniversalActionError, ValueError) as exc:
        return f"验收动作证据无法通过控制器复核：{exc}"
    robot_result = execution.get("robot_result")

    def pixel_point(value: Any) -> bool:
        return bool(
            isinstance(value, (list, tuple))
            and len(value) == 2
            and all(
                isinstance(item, int) and not isinstance(item, bool) and item >= 0
                for item in value
            )
        )

    if action == "long_press":
        if not pixel_point(robot_result):
            return "长按验收缺少机械臂返回的实际像素落点。"
        receipt = execution.get("hardware_receipt")
        if not isinstance(receipt, dict):
            return "长按验收缺少控制端事件栅栏凭据。"
        required_true = (
            "seller_event_barrier_confirmed",
            "round_trip_position_confirmed",
            "hold_started_after_barrier",
        )
        if (
            receipt.get("version")
            != "2026-08-16-seller-gui-contact-barrier-v3"
            or receipt.get("channel") != "right_button_stationary_touch"
            or any(receipt.get(field) is not True for field in required_true)
        ):
            return "长按验收的控制端事件栅栏凭据无效。"
        receipt_hold = receipt.get("requested_hold_seconds")
        if (
            resolved.hold_seconds is None
            or isinstance(receipt_hold, bool)
            or not isinstance(receipt_hold, (int, float))
            or abs(float(receipt_hold) - float(resolved.hold_seconds)) > 1e-6
        ):
            return "长按验收的事件栅栏保压时长与已解析动作不一致。"
        offset = receipt.get("barrier_offset_pixels")
        changed = receipt.get("changed_pixels")
        return_changed = receipt.get("return_changed_pixels")
        elapsed = receipt.get("barrier_elapsed_ms")
        settle = receipt.get("post_barrier_settle_seconds")
        if (
            isinstance(offset, bool)
            or not isinstance(offset, int)
            or not 1 <= offset <= 3
            or isinstance(changed, bool)
            or not isinstance(changed, int)
            or changed < 120
            or isinstance(return_changed, bool)
            or not isinstance(return_changed, int)
            or return_changed < 120
            or isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not 0.0 <= float(elapsed) <= 5000.0
            or isinstance(settle, bool)
            or not isinstance(settle, (int, float))
            or not 0.4 <= float(settle) <= 0.6
        ):
            return "长按验收的控制端事件栅栏测量值无效。"
    if action == "drag":
        if (
            not isinstance(robot_result, (list, tuple))
            or len(robot_result) != 2
            or not all(pixel_point(point) for point in robot_result)
            or tuple(robot_result[0]) == tuple(robot_result[1])
        ):
            return "拖动验收缺少机械臂返回的两个不同实际像素端点。"
    if action == "reveal_system_navigation":
        geometry_fields = (
            "normalized_point",
            "normalized_end_point",
            "text",
            "direction",
            "hold_seconds",
            "path_distance",
            "target_element_id",
            "destination_element_id",
        )
        if any(raw_resolved.get(field) is not None for field in geometry_fields):
            return "系统边缘唤栏验收的已解析动作不能携带模型坐标、方向或距离。"
        client_path = (
            robot_result.get("client_path")
            if isinstance(robot_result, dict)
            else None
        )
        requested_grid = (
            robot_result.get("requested_grid")
            if isinstance(robot_result, dict)
            else None
        )
        corrected_grid = (
            robot_result.get("corrected_grid")
            if isinstance(robot_result, dict)
            else None
        )
        grid_path = lambda value: bool(
            isinstance(value, (list, tuple))
            and len(value) == 2
            and all(
                isinstance(point, (list, tuple))
                and len(point) == 2
                and all(
                    isinstance(item, int)
                    and not isinstance(item, bool)
                    and 0 <= item <= 1000
                    for item in point
                )
                for point in value
            )
        )
        if (
            not isinstance(robot_result, dict)
            or robot_result.get("action") != "reveal_system_navigation"
            or robot_result.get("edge") != "bottom"
            or not isinstance(client_path, (list, tuple))
            or len(client_path) != 2
            or not all(pixel_point(point) for point in client_path)
            or tuple(client_path[0]) == tuple(client_path[1])
            or not grid_path(requested_grid)
            or not grid_path(corrected_grid)
        ):
            return "系统边缘唤栏验收缺少受限语义和两个不同实际像素端点。"
    return ""


class CapabilityAcceptanceError(RuntimeError):
    pass


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    try:
        return _sha256_bytes(Path(path).read_bytes())
    except OSError as exc:
        raise CapabilityAcceptanceError(f"文件无法读取：{path}：{exc}") from exc


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise CapabilityAcceptanceError(f"{label}无法读取：{exc}") from exc
    if not isinstance(value, dict):
        raise CapabilityAcceptanceError(f"{label}必须是 JSON 对象。")
    return value


def _required_text(value: Any, *, field: str, max_length: int = 256) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CapabilityAcceptanceError(f"验收报告字段 {field} 不能为空。")
    clean = value.strip()
    if len(clean) > max_length:
        raise CapabilityAcceptanceError(f"验收报告字段 {field} 过长。")
    return clean


def _observation(value: Any, *, field: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise CapabilityAcceptanceError(f"验收报告字段 {field} 必须是对象。")
    return {
        "observation_id": _required_text(
            value.get("observation_id"), field=f"{field}.observation_id"
        ),
        "fingerprint": _required_text(
            value.get("fingerprint"), field=f"{field}.fingerprint"
        ),
    }


def _confirmation_scope(
    value: Any,
    *,
    session_id: str,
    task_id: str,
    device_id: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CapabilityAcceptanceError("验收报告 confirmation_scope 必须是对象。")
    required = {
        "session_id",
        "task_id",
        "device_id",
        "revision",
        "subgoal_id",
        "effect_ids",
        "observation_id",
        "fingerprint",
    }
    if set(value) != required:
        raise CapabilityAcceptanceError("验收报告 confirmation_scope 字段不完整。")
    normalized = dict(value)
    for field, expected in (
        ("session_id", session_id),
        ("task_id", task_id),
        ("device_id", device_id),
    ):
        if normalized.get(field) != expected:
            raise CapabilityAcceptanceError(
                f"验收报告 confirmation_scope.{field} 与报告范围不一致。"
            )
    revision = normalized.get("revision")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
        raise CapabilityAcceptanceError("验收报告 confirmation_scope.revision 无效。")
    _required_text(normalized.get("subgoal_id"), field="confirmation_scope.subgoal_id")
    _required_text(
        normalized.get("observation_id"),
        field="confirmation_scope.observation_id",
    )
    _required_text(
        normalized.get("fingerprint"),
        field="confirmation_scope.fingerprint",
    )
    effect_ids = normalized.get("effect_ids")
    if not isinstance(effect_ids, list) or any(
        not isinstance(item, str) or not item.strip() for item in effect_ids
    ):
        raise CapabilityAcceptanceError("验收报告 confirmation_scope.effect_ids 无效。")
    if effect_ids != sorted(set(effect_ids)):
        raise CapabilityAcceptanceError(
            "验收报告 confirmation_scope.effect_ids 必须去重并排序。"
        )
    return normalized


def _validate_frame_paths(
    value: Any,
    *,
    field: str,
    trial_root: Path,
    expected_sha256: Any,
) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) != 4:
        raise CapabilityAcceptanceError(f"{field} 必须恰好包含四张 JPEG 证据。")
    if (
        not isinstance(expected_sha256, list)
        or len(expected_sha256) != 4
        or any(
            not isinstance(item, str)
            or len(item) != 64
            or any(character not in "0123456789abcdef" for character in item)
            for item in expected_sha256
        )
    ):
        raise CapabilityAcceptanceError(f"{field} 的证据摘要格式无效。")
    resolved_root = trial_root.resolve(strict=True)
    result: list[str] = []
    seen: set[Path] = set()
    for index, item in enumerate(value, start=1):
        if not isinstance(item, str) or not item.strip():
            raise CapabilityAcceptanceError(f"{field}[{index}] 证据路径无效。")
        candidate = Path(item.strip())
        if not candidate.is_absolute():
            candidate = trial_root / candidate
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise CapabilityAcceptanceError(
                f"{field}[{index}] 证据文件不存在。"
            ) from exc
        try:
            resolved.relative_to(resolved_root)
        except ValueError as exc:
            raise CapabilityAcceptanceError(
                f"{field}[{index}] 证据必须位于本次 trial 目录。"
            ) from exc
        if resolved in seen:
            raise CapabilityAcceptanceError(f"{field} 不能重复引用同一张证据。")
        if resolved.suffix.lower() not in {".jpg", ".jpeg"}:
            raise CapabilityAcceptanceError(f"{field}[{index}] 证据必须是 JPEG。")
        try:
            with Image.open(resolved) as image:
                image.verify()
                if image.format != "JPEG":
                    raise CapabilityAcceptanceError(
                        f"{field}[{index}] 证据内容不是 JPEG。"
                    )
        except (OSError, UnidentifiedImageError) as exc:
            raise CapabilityAcceptanceError(
                f"{field}[{index}] 证据无法读取。"
            ) from exc
        if sha256_file(resolved) != expected_sha256[index - 1]:
            raise CapabilityAcceptanceError(
                f"{field}[{index}] 证据摘要与文件不一致。"
            )
        seen.add(resolved)
        result.append(str(resolved))
    return tuple(result)


def _consistent_frame_size(paths: tuple[str, ...], *, field: str) -> tuple[int, int]:
    sizes: set[tuple[int, int]] = set()
    for value in paths:
        try:
            with Image.open(value) as image:
                sizes.add(tuple(image.size))
        except (OSError, UnidentifiedImageError) as exc:
            raise CapabilityAcceptanceError(f"{field} 证据尺寸无法读取。") from exc
    if len(sizes) != 1:
        raise CapabilityAcceptanceError(f"{field} 四张证据尺寸不一致。")
    width, height = sizes.pop()
    if width <= 0 or height <= 0:
        raise CapabilityAcceptanceError(f"{field} 证据尺寸无效。")
    return width, height


def validate_acceptance_report(report_path: Path) -> dict[str, Any]:
    """Validate evidence strong enough to make one capability promotable."""

    resolved_report = Path(report_path).resolve(strict=True)
    report = _load_json_object(resolved_report, label="验收报告")
    if report.get("version") != ACCEPTANCE_REPORT_VERSION:
        raise CapabilityAcceptanceError("验收报告版本无效。")

    trial_id = _required_text(report.get("trial_id"), field="trial_id", max_length=128)
    session_id = _required_text(
        report.get("session_id"), field="session_id", max_length=128
    )
    task_id = _required_text(report.get("task_id"), field="task_id", max_length=128)
    device_id = _required_text(
        report.get("device_id"), field="device_id", max_length=128
    )
    action = _required_text(
        report.get("candidate_action"), field="candidate_action", max_length=64
    )
    if action not in PROMOTABLE_ACTIONS:
        raise CapabilityAcceptanceError(f"动作类型不能进入真机验收：{action}。")
    calibration_evidence = _calibration_evidence(
        report.get("calibration_evidence"),
        action=action,
    )
    code_revision = _required_text(
        report.get("code_revision"), field="code_revision", max_length=128
    )
    if code_revision.endswith("+dirty"):
        raise CapabilityAcceptanceError("验收报告来自未提交代码，不能晋级。")
    if report.get("status") != "passed":
        raise CapabilityAcceptanceError("验收报告结果不是 passed，不能晋级。")
    physical_actions = report.get("physical_actions")
    if isinstance(physical_actions, bool) or physical_actions != 1:
        raise CapabilityAcceptanceError("验收报告物理动作数必须严格等于 1。")
    if report.get("action_outcome") != "matched":
        raise CapabilityAcceptanceError("验收结果不是 matched，不能晋级。")

    before = _observation(report.get("before_observation"), field="before_observation")
    after = _observation(report.get("after_observation"), field="after_observation")
    confirmation_scope = _confirmation_scope(
        report.get("confirmation_scope"),
        session_id=session_id,
        task_id=task_id,
        device_id=device_id,
    )
    if (
        confirmation_scope["observation_id"] != before["observation_id"]
        or confirmation_scope["fingerprint"] != before["fingerprint"]
    ):
        raise CapabilityAcceptanceError(
            "动作前 observation/fingerprint 与确认作用域不一致。"
        )
    if after["observation_id"] == before["observation_id"]:
        raise CapabilityAcceptanceError("动作后 observation_id 未变化。")
    if after["fingerprint"] == before["fingerprint"]:
        raise CapabilityAcceptanceError("动作后 fingerprint 未变化。")

    execution = report.get("execution")
    if not isinstance(execution, dict):
        raise CapabilityAcceptanceError("验收报告 execution 必须是对象。")
    resolved_action = execution.get("resolved_action")
    if not isinstance(resolved_action, dict) or resolved_action.get("kind") != action:
        raise CapabilityAcceptanceError("执行动作类型与候选动作类型不一致。")
    observation_errors = execution.get("observation_errors")
    if not isinstance(observation_errors, list) or observation_errors:
        raise CapabilityAcceptanceError("验收报告包含观察错误，不能晋级。")
    verification_errors = execution.get("verification_errors")
    if not isinstance(verification_errors, list) or verification_errors:
        raise CapabilityAcceptanceError("验收报告包含验证错误，不能晋级。")
    evidence_error = action_execution_evidence_error(action, execution)
    if evidence_error:
        raise CapabilityAcceptanceError(evidence_error)
    if action == "input_verified_text":
        exact_error = exact_input_evidence_error(execution)
        if exact_error:
            raise CapabilityAcceptanceError(exact_error)
    before_scene = execution["before_scene"]
    after_scene = execution["after_scene"]
    if str(after_scene.get("fingerprint") or "") != after["fingerprint"]:
        raise CapabilityAcceptanceError("动作后场景 fingerprint 与验收观察不一致。")
    execution_before_fingerprint = str(before_scene.get("fingerprint") or "")
    if not execution_before_fingerprint:
        raise CapabilityAcceptanceError("执行前场景缺少 fingerprint。")
    if execution_before_fingerprint == after["fingerprint"]:
        raise CapabilityAcceptanceError("动作后 fingerprint 与执行前场景相同。")

    trial_root = resolved_report.parent
    before_paths = _validate_frame_paths(
        report.get("before_frame_paths"),
        field="before_frame_paths",
        trial_root=trial_root,
        expected_sha256=report.get("before_frame_sha256"),
    )
    after_paths = _validate_frame_paths(
        report.get("after_frame_paths"),
        field="after_frame_paths",
        trial_root=trial_root,
        expected_sha256=report.get("after_frame_sha256"),
    )
    if set(before_paths) & set(after_paths):
        raise CapabilityAcceptanceError("动作前后证据不能引用同一文件。")
    before_frame_size = _consistent_frame_size(
        before_paths,
        field="before_frame_paths",
    )
    after_frame_size = _consistent_frame_size(
        after_paths,
        field="after_frame_paths",
    )
    if before_frame_size != after_frame_size:
        raise CapabilityAcceptanceError("动作前后证据画面尺寸不一致。")
    try:
        orientation_credential = OrientationCredential.from_dict(
            execution.get("orientation_credential")
        )
        orientation_credential.assert_authorizes(
            device_id=device_id,
            scene_fingerprint=execution_before_fingerprint,
            frame_size=before_frame_size,
        )
    except OrientationSafetyError as exc:
        raise CapabilityAcceptanceError(
            f"独立方向凭据不能支持能力晋级：{exc}"
        ) from exc
    before_fingerprints: set[str] = set()
    for path in before_paths:
        with Image.open(path) as image:
            before_fingerprints.add(frame_fingerprint(image.convert("RGB")))
    if (
        not orientation_credential.evidence_frame_fingerprint
        or orientation_credential.evidence_frame_fingerprint
        not in before_fingerprints
    ):
        raise CapabilityAcceptanceError(
            "独立方向凭据未绑定动作前保存的稳定帧。"
        )
    raw_alignment = before_scene.get("camera_alignment")
    if isinstance(raw_alignment, dict):
        compact_layout = raw_alignment.get("camera_layout_orientation")
        compact_rotation = raw_alignment.get("phone_content_rotation")
        if compact_layout not in {
            "unknown", orientation_credential.camera_layout_orientation
        } or compact_rotation not in {
            "unknown", orientation_credential.phone_content_rotation
        }:
            raise CapabilityAcceptanceError(
                "主场景方向事实与独立方向凭据冲突，不能晋级。"
            )
    if calibration_evidence is not None:
        width, height = before_frame_size
        robot_result = execution.get("robot_result")
        if action == "long_press":
            points = [robot_result]
        elif action == "reveal_system_navigation":
            points = list(robot_result["client_path"])
        else:
            points = list(robot_result)
        if any(
            not 0 <= int(point[0]) < width or not 0 <= int(point[1]) < height
            for point in points
        ):
            raise CapabilityAcceptanceError(
                "手势验收机械臂实际像素端点超出动作前画面范围。"
            )
        if action == "reveal_system_navigation" and robot_result.get(
            "frame_size"
        ) != [width, height]:
            raise CapabilityAcceptanceError(
                "系统边缘唤栏回执的相机尺寸与动作前证据不一致。"
            )

    normalized = dict(report)
    normalized.update(
        {
            "trial_id": trial_id,
            "session_id": session_id,
            "task_id": task_id,
            "device_id": device_id,
            "candidate_action": action,
            "calibration_evidence": calibration_evidence,
            "before_observation": before,
            "after_observation": after,
            "confirmation_scope": confirmation_scope,
            "before_frame_paths": list(before_paths),
            "after_frame_paths": list(after_paths),
        }
    )
    return normalized


@dataclass(frozen=True)
class PromotionScope:
    trial_id: str
    device_id: str
    action: str
    report_sha256: str
    registry_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "device_id": self.device_id,
            "action": self.action,
            "report_sha256": self.report_sha256,
            "registry_sha256": self.registry_sha256,
        }


_PROMOTION_AUTHORITY_FACTORY_TOKEN = object()


def _resolved_execution_kind(execution_result: Any) -> str:
    resolved = getattr(execution_result, "resolved_action", None)
    if isinstance(resolved, Mapping):
        return str(resolved.get("kind") or "").strip()
    return str(getattr(resolved, "kind", "") or "").strip()


def _validate_live_promotion_source(
    *,
    report: Mapping[str, Any],
    orientation_credential: OrientationCredential,
    execution_result: Any,
) -> None:
    if not isinstance(orientation_credential, OrientationCredential):
        raise CapabilityAcceptanceError(
            "能力晋级必须接收本进程真实方向凭据对象。"
        )
    if getattr(execution_result, "orientation_credential", None) is not orientation_credential:
        raise CapabilityAcceptanceError("能力晋级方向凭据不是本次动作结果持有的同一对象。")
    physical_actions = getattr(execution_result, "physical_actions", None)
    if isinstance(physical_actions, bool) or physical_actions != 1:
        raise CapabilityAcceptanceError("能力晋级来源必须是恰好一次物理动作结果。")
    if str(getattr(execution_result, "action_outcome", "") or "") != "matched":
        raise CapabilityAcceptanceError("能力晋级来源动作结果未通过闭环验证。")
    action = str(report.get("candidate_action") or "")
    if _resolved_execution_kind(execution_result) != action:
        raise CapabilityAcceptanceError("能力晋级来源动作类型与报告不一致。")
    if orientation_credential.device_id != report.get("device_id"):
        raise CapabilityAcceptanceError("能力晋级 live 方向凭据与报告设备不一致。")
    before_scene = getattr(execution_result, "before_scene", None)
    before_fingerprint = str(getattr(before_scene, "fingerprint", "") or "")
    if before_fingerprint != orientation_credential.scene_fingerprint:
        raise CapabilityAcceptanceError("能力晋级 live 方向凭据与动作前场景不一致。")
    before_frames = getattr(execution_result, "before_frames", None)
    if not isinstance(before_frames, tuple) or not before_frames:
        raise CapabilityAcceptanceError("能力晋级来源缺少本进程动作前原始帧对象。")
    matching_frames = [
        frame
        for frame in before_frames
        if isinstance(frame, Image.Image)
        and tuple(frame.size) == orientation_credential.frame_size
        and frame_fingerprint(frame) == orientation_credential.frame_fingerprint
    ]
    if not matching_frames:
        raise CapabilityAcceptanceError("能力晋级 live 方向凭据未绑定动作前帧对象。")
    execution = report.get("execution")
    if not isinstance(execution, Mapping):
        raise CapabilityAcceptanceError("能力晋级报告缺少执行对象。")
    if execution.get("orientation_credential") != orientation_credential.to_dict():
        raise CapabilityAcceptanceError("能力晋级报告方向凭据不是 live 对象的序列化视图。")
    if report.get("physical_actions") != physical_actions:
        raise CapabilityAcceptanceError("能力晋级报告与 live 动作计数不一致。")
    if report.get("action_outcome") != getattr(execution_result, "action_outcome", None):
        raise CapabilityAcceptanceError("能力晋级报告与 live 动作结果不一致。")
    before_paths = tuple(str(value) for value in report.get("before_frame_paths", ()))
    result_paths = tuple(
        str(value) for value in getattr(execution_result, "before_frame_paths", ())
    )
    if before_paths != result_paths:
        raise CapabilityAcceptanceError("能力晋级报告与 live 动作前证据路径不一致。")


class PromotionAuthority:
    """One-shot local authority bound to one report and registry revision."""

    def __init__(
        self,
        scope: PromotionScope,
        *,
        orientation_credential: OrientationCredential,
        execution_result: Any,
        report: Mapping[str, Any],
        _factory_token: object | None = None,
    ) -> None:
        if _factory_token is not _PROMOTION_AUTHORITY_FACTORY_TOKEN:
            raise CapabilityAcceptanceError("PromotionAuthority 只能由 live preview 签发。")
        self.scope = scope
        self.consumed = False
        self._orientation_credential = orientation_credential
        self._execution_result = execution_result
        self._report = json.loads(json.dumps(dict(report), ensure_ascii=False))
        self._source_nonce = object()
        self._lifecycle_lock = threading.Lock()

    def begin_promotion(self) -> None:
        if not self._lifecycle_lock.acquire(blocking=False):
            raise CapabilityAcceptanceError("能力晋级 authority 正在使用。")

    def end_promotion(self) -> None:
        self._lifecycle_lock.release()

    def _live_source(self) -> tuple[OrientationCredential, Any, Mapping[str, Any]]:
        if (
            self._source_nonce is None
            or self._orientation_credential is None
            or self._execution_result is None
            or self._report is None
        ):
            raise CapabilityAcceptanceError("能力晋级 live 来源已失效。")
        return self._orientation_credential, self._execution_result, self._report

    def validate_and_consume(self, value: Mapping[str, Any]) -> None:
        if self.consumed:
            raise CapabilityAcceptanceError("能力晋级确认已使用，禁止重放。")
        self.consumed = True
        orientation_credential, execution_result, report = self._live_source()
        _validate_live_promotion_source(
            report=report,
            orientation_credential=orientation_credential,
            execution_result=execution_result,
        )
        if not isinstance(value, Mapping):
            raise CapabilityAcceptanceError("能力晋级确认范围必须是对象。")
        expected = self.scope.to_dict()
        requested = dict(value)
        if set(requested) != set(expected) or any(
            not isinstance(requested.get(key), str) or requested.get(key) != expected[key]
            for key in expected
        ):
            raise CapabilityAcceptanceError("能力晋级确认范围不匹配。")

    def assert_current_report(self, report: Mapping[str, Any]) -> None:
        orientation_credential, execution_result, _stored_report = self._live_source()
        _validate_live_promotion_source(
            report=report,
            orientation_credential=orientation_credential,
            execution_result=execution_result,
        )

    def release_source(self) -> None:
        self._source_nonce = None
        self._orientation_credential = None
        self._execution_result = None
        self._report = None

    def invalidate(self) -> None:
        with self._lifecycle_lock:
            self.consumed = True
            self.release_source()


class CapabilityRegistryPromoter:
    """Promote one report-bound action through an atomic registry replacement."""

    def __init__(
        self,
        registry_path: Path,
        *,
        lease_path: Path | None = None,
        replace_file: Callable[[Path, Path], None] | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.registry_path = Path(registry_path)
        self.lease_path = Path(
            lease_path
            if lease_path is not None
            else self.registry_path.with_suffix(".promotion.lease")
        )
        self._replace_file = replace_file or (
            lambda source, target: os.replace(source, target)
        )
        self._now = now or (lambda: datetime.now(timezone.utc))

    @staticmethod
    def _registry_device(
        payload: dict[str, Any], device_id: str
    ) -> dict[str, Any]:
        if payload.get("version") != 1 or not isinstance(payload.get("devices"), list):
            raise CapabilityAcceptanceError("设备注册表版本或 devices 格式无效。")
        matches = [
            item
            for item in payload["devices"]
            if isinstance(item, dict)
            and item.get("enabled") is True
            and item.get("device_id") == device_id
        ]
        if len(matches) != 1:
            raise CapabilityAcceptanceError(
                f"设备注册表没有唯一的已启用设备：{device_id}。"
            )
        device = matches[0]
        actions = device.get("verified_actions")
        if not isinstance(actions, list) or any(
            not isinstance(item, str) or not item.strip() for item in actions
        ):
            raise CapabilityAcceptanceError("设备 verified_actions 格式无效。")
        normalized = [item.strip() for item in actions]
        if len(normalized) != len(set(normalized)):
            raise CapabilityAcceptanceError("设备 verified_actions 存在重复动作。")
        device["verified_actions"] = normalized
        return device

    def _load_registry(self) -> tuple[dict[str, Any], bytes]:
        try:
            raw = self.registry_path.read_bytes()
            payload = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise CapabilityAcceptanceError(f"设备注册表无法读取：{exc}") from exc
        if not isinstance(payload, dict):
            raise CapabilityAcceptanceError("设备注册表必须是 JSON 对象。")
        return payload, raw

    def _require_matching_calibration(
        self,
        report: Mapping[str, Any],
        device: Mapping[str, Any],
    ) -> None:
        action = str(report.get("candidate_action") or "")
        if action not in CALIBRATION_BOUND_ACTIONS:
            return
        calibration_value = device.get("calibration_path")
        if not isinstance(calibration_value, str) or not calibration_value.strip():
            raise CapabilityAcceptanceError("设备注册表缺少触控标定路径。")
        calibration_path = Path(calibration_value.strip())
        if not calibration_path.is_absolute():
            calibration_path = self.registry_path.parent / calibration_path
        current = validated_calibration_evidence(calibration_path)
        if current != report.get("calibration_evidence"):
            raise CapabilityAcceptanceError(
                "触控标定与真机验收报告不一致；必须在当前标定上重新验收。"
            )

    def preview(
        self,
        report_path: Path,
        *,
        orientation_credential: OrientationCredential | None = None,
        execution_result: Any | None = None,
    ) -> PromotionAuthority:
        if orientation_credential is None or execution_result is None:
            raise CapabilityAcceptanceError(
                "能力晋级 preview 必须由 live manager 提供方向凭据和一次动作结果。"
            )
        report = validate_acceptance_report(report_path)
        _validate_live_promotion_source(
            report=report,
            orientation_credential=orientation_credential,
            execution_result=execution_result,
        )
        registry, raw = self._load_registry()
        device = self._registry_device(registry, report["device_id"])
        action = report["candidate_action"]
        self._require_matching_calibration(report, device)
        if action in device["verified_actions"]:
            raise CapabilityAcceptanceError(f"设备能力 {action} 已经启用。")
        scope = PromotionScope(
            trial_id=report["trial_id"],
            device_id=report["device_id"],
            action=action,
            report_sha256=sha256_file(Path(report_path)),
            registry_sha256=_sha256_bytes(raw),
        )
        try:
            orientation_credential.claim_live_execution_source()
        except OrientationSafetyError as exc:
            raise CapabilityAcceptanceError(
                f"能力晋级缺少 live-trial 方向来源：{exc}"
            ) from exc
        return PromotionAuthority(
            scope,
            orientation_credential=orientation_credential,
            execution_result=execution_result,
            report=report,
            _factory_token=_PROMOTION_AUTHORITY_FACTORY_TOKEN,
        )

    @staticmethod
    def _write_new_file(path: Path, payload: bytes) -> None:
        try:
            with path.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError as exc:
            raise CapabilityAcceptanceError(f"晋级证据文件已经存在：{path.name}。") from exc
        except OSError as exc:
            raise CapabilityAcceptanceError(f"晋级证据无法写入：{path.name}：{exc}") from exc

    def _replace_registry(self, payload: bytes) -> None:
        temporary = self.registry_path.parent / (
            f".{self.registry_path.name}.{uuid.uuid4().hex}.tmp"
        )
        try:
            with temporary.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            self._replace_file(temporary, self.registry_path)
        except OSError as exc:
            raise CapabilityAcceptanceError(f"设备注册表原子替换失败：{exc}") from exc
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    def promote(
        self,
        report_path: Path,
        *,
        confirmation: Mapping[str, Any],
        authority: PromotionAuthority,
    ) -> dict[str, Any]:
        authority.begin_promotion()
        try:
            return self._promote_bound(
                report_path,
                confirmation=confirmation,
                authority=authority,
            )
        finally:
            authority.release_source()
            authority.end_promotion()

    def _promote_bound(
        self,
        report_path: Path,
        *,
        confirmation: Mapping[str, Any],
        authority: PromotionAuthority,
    ) -> dict[str, Any]:
        authority.validate_and_consume(confirmation)
        scope = authority.scope
        lease = InterProcessLease(
            self.lease_path,
            owner_id=scope.trial_id,
            metadata={
                "kind": "capability_registry_promotion",
                "trial_id": scope.trial_id,
                "device_id": scope.device_id,
                "action": scope.action,
            },
        )
        if not lease.acquire():
            raise CapabilityAcceptanceError("设备注册表晋级锁已被其他进程占用。")
        try:
            if sha256_file(Path(report_path)) != scope.report_sha256:
                raise CapabilityAcceptanceError("验收报告在确认后发生变化。")
            report = validate_acceptance_report(report_path)
            authority.assert_current_report(report)
            if (
                report["trial_id"] != scope.trial_id
                or report["device_id"] != scope.device_id
                or report["candidate_action"] != scope.action
            ):
                raise CapabilityAcceptanceError("验收报告范围在确认后发生变化。")

            registry, registry_raw = self._load_registry()
            if _sha256_bytes(registry_raw) != scope.registry_sha256:
                raise CapabilityAcceptanceError("设备注册表在确认后发生变化。")
            device = self._registry_device(registry, scope.device_id)
            self._require_matching_calibration(report, device)
            if scope.action in device["verified_actions"]:
                raise CapabilityAcceptanceError(f"设备能力 {scope.action} 已经启用。")

            trial_dir = Path(report_path).resolve(strict=True).parent
            backup_path = trial_dir / "registry_before.json"
            promotion_path = trial_dir / "promotion.json"
            if backup_path.exists() or promotion_path.exists():
                raise CapabilityAcceptanceError("本次 trial 已存在晋级证据，禁止重复晋级。")

            device["verified_actions"] = sorted(
                {*device["verified_actions"], scope.action}
            )
            encoded_registry = (
                json.dumps(registry, ensure_ascii=False, indent=2) + "\n"
            ).encode("utf-8")
            after_sha256 = _sha256_bytes(encoded_registry)
            promoted_at = self._now().astimezone(timezone.utc).isoformat()
            result = {
                "version": 1,
                "trial_id": scope.trial_id,
                "device_id": scope.device_id,
                "action": scope.action,
                "report_sha256": scope.report_sha256,
                "registry_before_sha256": scope.registry_sha256,
                "registry_after_sha256": after_sha256,
                "promoted_at": promoted_at,
                "requires_restart": True,
            }
            encoded_promotion = (
                json.dumps(result, ensure_ascii=False, indent=2) + "\n"
            ).encode("utf-8")

            self._write_new_file(backup_path, registry_raw)
            try:
                self._replace_registry(encoded_registry)
            except CapabilityAcceptanceError:
                try:
                    backup_path.unlink(missing_ok=True)
                except OSError:
                    pass
                raise
            try:
                self._write_new_file(promotion_path, encoded_promotion)
            except CapabilityAcceptanceError:
                # The registry update is already durable.  Restore the exact
                # previous bytes so a missing audit record never leaves an
                # enabled capability behind.
                self._replace_registry(registry_raw)
                raise
            return result
        finally:
            lease.release()
