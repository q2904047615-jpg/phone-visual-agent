from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image, ImageDraw

import robot_gui_poc as legacy
from device_exclusivity import InterProcessLease, SHARED_DEVICE_LEASE_DIR
from robot_core import RobotController, load_workflow_config
from tap_calibration import (
    CALIBRATION_PATH,
    Affine2D,
    TapCalibrationError,
    build_calibration,
    build_coverage,
    save_calibration,
)
from universal_agent_orchestrator import (
    DeviceTaskRegistry,
    UniversalAgentOrchestratorError,
)


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "output" / "xy_calibration"
CALIBRATION_SESSION_VERSION = 1


def request_json(url: str, method: str = "GET", payload: object | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(url, data=data, method=method)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    with urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def locate_magenta_target(frame: Image.Image) -> tuple[int, int, tuple[int, int, int, int]]:
    pixels = np.asarray(frame.convert("RGB"))
    red = pixels[:, :, 0]
    green = pixels[:, :, 1]
    blue = pixels[:, :, 2]
    mask = (
        (red > 165)
        & (blue > 125)
        & (green < 150)
        & ((red.astype(np.int16) - green.astype(np.int16)) > 55)
        & ((blue.astype(np.int16) - green.astype(np.int16)) > 35)
    )
    ys, xs = np.nonzero(mask)
    if len(xs) < 24:
        raise TapCalibrationError("未在实时画面中找到清晰的紫红色校准靶点。")
    # Use the densest local cluster so unrelated pink UI pixels cannot pull the
    # center away from the high-contrast calibration crosshair.
    coarse_x = int(np.median(xs))
    coarse_y = int(np.median(ys))
    nearby = (np.abs(xs - coarse_x) <= 40) & (np.abs(ys - coarse_y) <= 40)
    xs = xs[nearby]
    ys = ys[nearby]
    if len(xs) < 24:
        raise TapCalibrationError("紫红色靶点像素不足，可能画面模糊或页面不正确。")
    box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    if box[2] - box[0] > 90 or box[3] - box[1] > 90:
        raise TapCalibrationError("检测到的校准靶点尺寸异常，已停止。")
    return int(round(float(xs.mean()))), int(round(float(ys.mean()))), box


def wait_for_stable_target(
    robot: RobotController, timeout: float = 5.0
) -> tuple[Image.Image, tuple[int, int, tuple[int, int, int, int]]]:
    """Require the same target in two frames, ignoring page-transition frames."""
    deadline = time.monotonic() + timeout
    previous: tuple[int, int] | None = None
    last_error = ""
    while time.monotonic() < deadline:
        frame = robot.vision_capture().convert("RGB")
        try:
            detected = locate_magenta_target(frame)
            center = detected[:2]
            if previous is not None and math.hypot(
                center[0] - previous[0], center[1] - previous[1]
            ) <= 5.0:
                return frame, detected
            previous = center
        except TapCalibrationError as exc:
            last_error = str(exc)
            previous = None
        time.sleep(0.22)
    raise TapCalibrationError(
        f"连续画面未检测到稳定校准靶点。{last_error}".rstrip()
    )


def click_raw_pixel(robot: RobotController, frame: Image.Image, point: tuple[int, int]) -> None:
    x, y = point
    if not (0 <= x < frame.width and 0 <= y < frame.height):
        raise TapCalibrationError(f"校准落点{x, y}超出相机画面。")
    hwnd, _title = legacy.find_window(robot.title)
    robot._checkpoint()
    legacy.configure_single_click_count(hwnd)
    legacy.click_client_point(
        hwnd,
        x,
        y,
        countdown=0,
        hold_seconds=float(load_workflow_config()["vision_agent"]["tap_hold"]),
    )
    legacy.move_cursor_outside_camera(hwnd)


def wait_for_new_sample(base_url: str, previous_count: int, timeout: float = 8.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = request_json(f"{base_url}/api/samples")
        samples = snapshot.get("samples", [])
        if len(samples) > previous_count:
            return samples[-1]
        time.sleep(0.2)
    raise TapCalibrationError("机械臂落笔后，手机校准页没有回传触点。")


def normalized_dom(record: dict[str, object], prefix: str) -> list[float]:
    width = float(record["viewport_width"])
    height = float(record["viewport_height"])
    return [float(record[f"{prefix}_x"]) / width, float(record[f"{prefix}_y"]) / height]


def wait_for_page_state(
    base_url: str,
    *,
    phases: set[str],
    timeout: float = 6.0,
    newer_than: str | None = None,
) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    latest: dict[str, object] = {}
    threshold = _parse_page_timestamp(newer_than) if newer_than else None
    while time.monotonic() < deadline:
        latest = request_json(f"{base_url}/api/page-state")
        timestamp = _parse_page_timestamp(str(latest.get("updated_at") or ""))
        age = (
            (datetime.now(timezone.utc) - timestamp).total_seconds()
            if timestamp is not None
            else float("inf")
        )
        heartbeat_advanced = threshold is None or (
            timestamp is not None and timestamp > threshold
        )
        if (
            str(latest.get("phase") or "") in phases
            and 0 <= age <= 3.0
            and heartbeat_advanced
        ):
            return latest
        time.sleep(0.2)
    raise TapCalibrationError(
        f"手机校准页没有进入预期阶段{sorted(phases)}；当前状态：{latest}"
    )


def _parse_page_timestamp(value: str | None) -> datetime | None:
    try:
        timestamp = datetime.fromisoformat(str(value or ""))
    except ValueError:
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def calibration_page_ready(state: dict[str, object]) -> bool:
    """Accept only measured fullscreen or the explicitly bounded viewport fallback."""

    mode = str(state.get("calibration_mode") or "")
    if mode == "fullscreen":
        return state.get("fullscreen") is True
    coverage = state.get("viewport_coverage")
    try:
        return bool(
            mode == "viewport_coverage"
            and state.get("fullscreen") is False
            and state.get("fullscreen_attempted") is True
            and isinstance(coverage, dict)
            and coverage.get("eligible") is True
            and math.isfinite(float(coverage.get("width_ratio") or 0))
            and math.isfinite(float(coverage.get("height_ratio") or 0))
            and 0.92 <= float(coverage.get("width_ratio") or 0) <= 1.08
            and 0.92 <= float(coverage.get("height_ratio") or 0) <= 1.08
        )
    except (TypeError, ValueError):
        return False


def _select_device_status(
    payload: dict[str, object], requested_device_id: str | None
) -> tuple[str, str, dict[str, object]]:
    default_device_id = str(payload.get("default_device_id") or "").strip()
    resolved = str(requested_device_id or default_device_id).strip()
    if not default_device_id or not resolved:
        raise TapCalibrationError("/api/device缺少默认device_id。")
    devices = payload.get("devices")
    if not isinstance(devices, list):
        raise TapCalibrationError("/api/device缺少设备列表。")
    selected = next(
        (
            item
            for item in devices
            if isinstance(item, dict)
            and str(item.get("device_id") or "").strip() == resolved
        ),
        None,
    )
    if selected is None:
        raise TapCalibrationError(f"/api/device中找不到目标设备：{resolved}。")
    return resolved, default_device_id, selected


def _require_device_ready(
    device_status_url: str,
    device_id: str | None,
) -> tuple[str, str, dict[str, object]]:
    payload = request_json(device_status_url)
    resolved, default_device_id, selected = _select_device_status(payload, device_id)
    if selected.get("controller_online") is not True:
        raise TapCalibrationError(f"设备 {resolved} 控制端离线，保持0动作。")
    if selected.get("camera_online") is not True:
        raise TapCalibrationError(f"设备 {resolved} 摄像头离线，保持0动作。")
    if selected.get("busy") is not False:
        raise TapCalibrationError(f"设备 {resolved} 非空闲，保持0动作。")
    return resolved, default_device_id, selected


@contextmanager
def _calibration_device_reservation(
    *,
    device_status_url: str,
    device_id: str | None,
) -> Iterator[tuple[str, str, str]]:
    initial_payload = request_json(device_status_url)
    resolved, default_device_id, _selected = _select_device_status(
        initial_payload, device_id
    )
    session_id = (
        f"calibration-step-{resolved}-{os.getpid()}-{uuid.uuid4().hex[:12]}"
    )
    registry = DeviceTaskRegistry(lease_directory=SHARED_DEVICE_LEASE_DIR)
    try:
        registry.reserve(resolved, session_id)
    except UniversalAgentOrchestratorError as exc:
        raise TapCalibrationError(str(exc)) from exc
    try:
        confirmed, current_default, _status = _require_device_ready(
            device_status_url, resolved
        )
        if confirmed != resolved or current_default != default_device_id:
            raise TapCalibrationError("设备注册表在校准准备期间发生变化，保持0动作。")
        yield resolved, default_device_id, session_id
    finally:
        registry.release(resolved, session_id)


@contextmanager
def _calibration_physical_lease(
    *,
    device_status_url: str,
    device_id: str,
    default_device_id: str,
    calibration_session_id: str,
    mode: str,
    page_session_id: str,
) -> Iterator[None]:
    lease_path = (
        SHARED_DEVICE_LEASE_DIR / "physical_hardware_action.lease"
        if device_id == default_device_id
        else SHARED_DEVICE_LEASE_DIR
        / f"physical_hardware_action_{re.sub(r'[^A-Za-z0-9_.-]+', '_', device_id)}.lease"
    )
    lease = InterProcessLease(
        lease_path,
        owner_id=calibration_session_id,
        metadata={
            "purpose": "calibration_step",
            "calibration_step": True,
            "device_id": device_id,
            "mode": mode,
            "page_session_id": page_session_id,
        },
    )
    if not lease.acquire():
        raise TapCalibrationError("另一进程已占用机械臂物理控制权，保持0动作。")
    try:
        confirmed, current_default, _status = _require_device_ready(
            device_status_url, device_id
        )
        if confirmed != device_id or current_default != default_device_id:
            raise TapCalibrationError("设备状态在物理锁获取后发生变化，保持0动作。")
        yield
    finally:
        lease.release()


def _safe_session_component(value: str) -> str:
    cleaned = "".join(
        character
        for character in value
        if character.isalnum() or character in "-_."
    )
    return cleaned[:80] or hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_progress(
    output_dir: Path,
    *,
    mode: str,
    page_session_id: str,
    base_url: str,
) -> dict[str, object]:
    path = output_dir / "progress.json"
    if not path.exists():
        progress: dict[str, object] = {
            "version": CALIBRATION_SESSION_VERSION,
            "mode": mode,
            "page_session_id": page_session_id,
            "base_url": base_url,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "frame_size": None,
            "samples": [],
            "attempts": [],
        }
        _write_json(path, progress)
        return progress
    try:
        progress = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise TapCalibrationError("校准进度文件损坏，禁止继续执行。") from exc
    if not isinstance(progress, dict):
        raise TapCalibrationError("校准进度文件不是JSON对象，禁止继续执行。")
    if (
        progress.get("version") != CALIBRATION_SESSION_VERSION
        or progress.get("mode") != mode
        or progress.get("page_session_id") != page_session_id
        or progress.get("base_url") != base_url
        or not isinstance(progress.get("samples"), list)
        or not isinstance(progress.get("attempts"), list)
    ):
        raise TapCalibrationError("校准进度与当前页面会话不一致，禁止混用证据。")
    return progress


def _save_progress(output_dir: Path, progress: dict[str, object]) -> None:
    progress["updated_at"] = datetime.now(timezone.utc).isoformat()
    _write_json(output_dir / "progress.json", progress)
    _write_json(output_dir / "samples.json", progress["samples"])


def _page_snapshot(base_url: str) -> tuple[str, list[dict[str, object]]]:
    snapshot = request_json(f"{base_url}/api/samples")
    session_id = str(snapshot.get("session_id") or "").strip()
    samples = snapshot.get("samples")
    if not session_id or not isinstance(samples, list) or any(
        not isinstance(sample, dict) for sample in samples
    ):
        raise TapCalibrationError("校准页样本快照缺少可信session_id或samples。")
    for index, sample in enumerate(samples):
        if int(sample.get("sequence", -1)) != index:
            raise TapCalibrationError("校准页样本序号不连续，禁止继续执行。")
    return session_id, samples


def _annotate_target(
    frame: Image.Image,
    detected: tuple[int, int, tuple[int, int, int, int]],
    command: tuple[int, int],
) -> Image.Image:
    target_x, target_y, box = detected
    command_x, command_y = command
    annotated = frame.copy()
    draw = ImageDraw.Draw(annotated)
    draw.rectangle(box, outline="#00ff66", width=3)
    draw.line(
        (command_x - 12, command_y, command_x + 12, command_y),
        fill="#ffff00",
        width=2,
    )
    draw.line(
        (command_x, command_y - 12, command_x, command_y + 12),
        fill="#ffff00",
        width=2,
    )
    draw.ellipse(
        (target_x - 4, target_y - 4, target_x + 4, target_y + 4),
        outline="#00ff66",
        width=2,
    )
    return annotated


def _frame_fingerprint(frame: Image.Image) -> str:
    """Record the exact fresh camera observation used for a calibration decision."""

    rgb = frame.convert("RGB")
    digest = hashlib.sha256()
    digest.update(f"{rgb.width}x{rgb.height}:RGB:".encode("ascii"))
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def _build_sample(
    attempt: dict[str, object], record: dict[str, object]
) -> dict[str, object]:
    sequence = int(attempt["sequence"])
    if int(record.get("sequence", -1)) != sequence:
        raise TapCalibrationError(
            f"校准页序号不一致：期望{sequence}，实际{record.get('sequence')}。"
        )
    return {
        "sequence": sequence,
        "desired_frame": list(attempt["desired_frame"]),
        "command_frame": list(attempt["command_frame"]),
        "target_dom": normalized_dom(record, "target"),
        "actual_dom": normalized_dom(record, "actual"),
        "viewport_size": [record["viewport_width"], record["viewport_height"]],
    }


def _attempt_path(output_dir: Path, attempt: dict[str, object]) -> Path:
    return output_dir / f"{attempt['attempt_id']}.json"


def _persist_attempt(
    output_dir: Path,
    progress: dict[str, object],
    attempt: dict[str, object],
) -> None:
    _write_json(_attempt_path(output_dir, attempt), attempt)
    _save_progress(output_dir, progress)


def _capture_after(
    robot: RobotController,
    output_dir: Path,
    attempt: dict[str, object],
) -> None:
    try:
        path = output_dir / f"{attempt['attempt_id']}_after.jpg"
        robot.vision_capture().convert("RGB").save(path, quality=94)
        attempt["after"] = str(path)
    except Exception as exc:
        attempt["after_capture_error"] = str(exc)


def _page_state_advanced(
    state: dict[str, object], before: dict[str, object]
) -> bool:
    current = _parse_page_timestamp(str(state.get("updated_at") or ""))
    previous = _parse_page_timestamp(str(before.get("updated_at") or ""))
    return current is not None and previous is not None and current > previous


def _recover_previous_attempt(
    *,
    base_url: str,
    robot: RobotController,
    output_dir: Path,
    progress: dict[str, object],
    page_state: dict[str, object],
    server_samples: list[dict[str, object]],
) -> dict[str, object] | None:
    attempts = progress["attempts"]
    if not attempts:
        return None
    attempt = attempts[-1]
    if attempt.get("status") not in {"action_started", "awaiting_recovery"}:
        return None
    before = attempt["page_state_before"]
    kind = str(attempt.get("kind") or "")
    if kind == "fullscreen_setup":
        advanced = _page_state_advanced(page_state, before)
        if advanced and calibration_page_ready(page_state):
            attempt["status"] = "verified_recovered"
            attempt["page_state_after"] = page_state
            _capture_after(robot, output_dir, attempt)
            _persist_attempt(output_dir, progress, attempt)
            return {
                "status": "recovered",
                "mode": progress["mode"],
                "recovered_kind": kind,
                "physical_actions": 0,
                "previous_physical_actions": 1,
                "page_state": page_state,
                "output_dir": str(output_dir),
            }
        if advanced and page_state.get("phase") == "blocked":
            attempt["status"] = "verified_failed"
            attempt["page_state_after"] = page_state
            _capture_after(robot, output_dir, attempt)
            _persist_attempt(output_dir, progress, attempt)
            raise TapCalibrationError(
                "上次中央准备动作已被页面明确阻断；禁止自动重试。"
            )
        raise TapCalibrationError(
            "上次中央准备动作结果仍不确定；本次仅允许恢复核验，禁止再次落笔。"
        )

    if kind != "calibration_point":
        raise TapCalibrationError("校准进度包含未知物理动作，禁止继续。")
    previous_count = int(attempt["sample_count_before"])
    if len(server_samples) == previous_count:
        raise TapCalibrationError(
            "上次边缘触点没有形成新样本，结果不确定；禁止自动重试。"
        )
    if len(server_samples) != previous_count + 1:
        raise TapCalibrationError("上次动作后的服务器样本数量异常，禁止继续。")
    expected_sequence = int(attempt["sequence"])
    expected_phase = "complete" if expected_sequence == 8 else "calibration"
    if (
        page_state.get("phase") != expected_phase
        or int(page_state.get("sequence", -1)) != expected_sequence + 1
        or not calibration_page_ready(page_state)
        or not _page_state_advanced(page_state, before)
    ):
        raise TapCalibrationError("上次动作后的页面阶段、序号或heartbeat不可信。")
    samples = progress["samples"]
    if len(samples) != previous_count:
        raise TapCalibrationError("本地样本数量与待恢复动作不一致。")
    sample = _build_sample(attempt, server_samples[-1])
    samples.append(sample)
    attempt["status"] = "verified_recovered"
    attempt["sample"] = sample
    attempt["page_state_after"] = page_state
    _capture_after(robot, output_dir, attempt)
    _persist_attempt(output_dir, progress, attempt)
    return {
        "status": "recovered",
        "mode": progress["mode"],
        "recovered_kind": kind,
        "sequence": expected_sequence,
        "sample_count": len(samples),
        "physical_actions": 0,
        "previous_physical_actions": 1,
        "page_state": page_state,
        "output_dir": str(output_dir),
    }


def _finalize_session(
    *,
    mode: str,
    progress: dict[str, object],
    output_dir: Path,
    calibration_path: Path,
) -> dict[str, object]:
    samples = progress["samples"]
    sequences = [int(sample.get("sequence", -1)) for sample in samples]
    if sequences != list(range(9)):
        raise TapCalibrationError("最终拟合或验证必须具备连续完整的九个触点。")
    frame_size = progress.get("frame_size")
    if not isinstance(frame_size, list) or len(frame_size) != 2:
        raise TapCalibrationError("九点会话缺少一致的相机画面尺寸。")
    normalized_size = (int(frame_size[0]), int(frame_size[1]))
    if mode == "collect":
        payload = build_calibration(samples, normalized_size)
        payload["collection_dir"] = str(output_dir)
        save_calibration(payload, calibration_path)
        progress["finalized"] = True
        progress["accepted_fit"] = bool(payload["accepted_fit"])
        _save_progress(output_dir, progress)
        if not payload["accepted_fit"]:
            raise TapCalibrationError("九点拟合质量或屏幕覆盖范围未通过，校准未启用。")
        return {
            "status": "fit_complete",
            "mode": mode,
            "physical_actions": 0,
            "sample_count": 9,
            "accepted_fit": True,
            "output_dir": str(output_dir),
            "calibration_path": str(calibration_path),
        }

    if not calibration_path.exists():
        raise TapCalibrationError("找不到待验证的tap_calibration.json。")
    payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    if not payload.get("accepted_fit"):
        raise TapCalibrationError("拟合质量未通过，禁止验证和启用。")
    if list(normalized_size) != payload.get("frame_size"):
        raise TapCalibrationError("验证时相机尺寸与采集时不一致。")
    validation = validate_samples(samples)
    validation_coverage = build_coverage(
        [
            (
                float(sample["desired_frame"][0]) / (normalized_size[0] - 1),
                float(sample["desired_frame"][1]) / (normalized_size[1] - 1),
            )
            for sample in samples
        ]
    )
    validation["coverage_passed"] = bool(validation_coverage["sufficient"])
    validation["passed"] = bool(
        validation["passed"] and validation["coverage_passed"]
    )
    validation["coverage"] = validation_coverage
    payload["validation"] = validation
    payload["validation_dir"] = str(output_dir)
    payload["validated_at"] = datetime.now(timezone.utc).isoformat()
    payload["validated"] = bool(validation["passed"])
    payload["enabled"] = bool(validation["passed"])
    save_calibration(payload, calibration_path)
    progress["finalized"] = True
    progress["validation_passed"] = bool(validation["passed"])
    _save_progress(output_dir, progress)
    if not validation["passed"]:
        raise TapCalibrationError("独立九点验证未通过，纠偏保持禁用。")
    return {
        "status": "validation_complete",
        "mode": mode,
        "physical_actions": 0,
        "sample_count": 9,
        "validation": validation,
        "output_dir": str(output_dir),
        "calibration_path": str(calibration_path),
    }


def _execute_calibration_action(
    *,
    mode: str,
    base_url: str,
    robot: RobotController,
    output_dir: Path,
    progress: dict[str, object],
    page_state: dict[str, object],
    page_session_id: str,
    server_samples: list[dict[str, object]],
    local_samples: list[dict[str, object]],
    frame: Image.Image,
    detected: tuple[int, int, tuple[int, int, int, int]],
    correction: Affine2D | None,
    kind: str,
    sequence: int | None,
    device_status_url: str,
    device_id: str,
    default_device_id: str,
    calibration_session_id: str,
) -> dict[str, object]:
    preview_x, preview_y, _preview_box = detected
    with _calibration_physical_lease(
        device_status_url=device_status_url,
        device_id=device_id,
        default_device_id=default_device_id,
        calibration_session_id=calibration_session_id,
        mode=mode,
        page_session_id=page_session_id,
    ):
        locked_state = wait_for_page_state(
            base_url,
            phases={str(page_state.get("phase") or "")},
            timeout=6.0,
            newer_than=str(page_state.get("updated_at") or ""),
        )
        if (
            locked_state.get("phase") != page_state.get("phase")
            or int(locked_state.get("sequence", -1))
            != int(page_state.get("sequence", -1))
            or locked_state.get("calibration_mode")
            != page_state.get("calibration_mode")
        ):
            raise TapCalibrationError(
                "物理锁内页面状态已变化，当前确认失效，保持0动作。"
            )
        if locked_state.get("phase") == "calibration" and not calibration_page_ready(
            locked_state
        ):
            raise TapCalibrationError("物理锁内校准模式证据失效，保持0动作。")
        locked_session_id, locked_samples = _page_snapshot(base_url)
        if (
            locked_session_id != page_session_id
            or len(locked_samples) != len(server_samples)
        ):
            raise TapCalibrationError(
                "物理锁内页面session或样本数量已变化，保持0动作。"
            )
        locked_frame, locked_detected = wait_for_stable_target(robot)
        desired_x, desired_y, box = locked_detected
        if (
            (locked_frame.width, locked_frame.height) != (frame.width, frame.height)
            or math.hypot(desired_x - preview_x, desired_y - preview_y) > 5.0
        ):
            raise TapCalibrationError("物理锁内靶点与预检观察不一致，保持0动作。")
        command_x, command_y = desired_x, desired_y
        if correction is not None:
            nx, ny = correction.apply(
                desired_x / (locked_frame.width - 1),
                desired_y / (locked_frame.height - 1),
            )
            command_x = int(round(nx * (locked_frame.width - 1)))
            command_y = int(round(ny * (locked_frame.height - 1)))
        attempt_number = len(progress["attempts"]) + 1
        attempt_id = f"action_{attempt_number:02d}_{kind}"
        before_path = output_dir / f"{attempt_id}_before.jpg"
        _annotate_target(
            locked_frame,
            (desired_x, desired_y, box),
            (command_x, command_y),
        ).save(before_path, quality=94)
        attempt: dict[str, object] = {
            "attempt_id": attempt_id,
            "kind": kind,
            "sequence": sequence,
            "status": "action_started",
            "physical_actions": 1,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "page_state_before": locked_state,
            "sample_count_before": len(server_samples),
            "desired_frame": [desired_x, desired_y],
            "command_frame": [command_x, command_y],
            "frame_size": [locked_frame.width, locked_frame.height],
            "observation_fingerprint": _frame_fingerprint(locked_frame),
            "before": str(before_path),
            "device_id": device_id,
            "calibration_session_id": calibration_session_id,
        }
        progress["attempts"].append(attempt)
        _persist_attempt(output_dir, progress, attempt)
        try:
            click_raw_pixel(robot, locked_frame, (command_x, command_y))
            if kind == "fullscreen_setup":
                after_state = wait_for_page_state(
                    base_url,
                    phases={"calibration", "blocked"},
                    timeout=8.0,
                    newer_than=str(locked_state.get("updated_at") or ""),
                )
                attempt["page_state_after"] = after_state
                if not calibration_page_ready(after_state):
                    attempt["status"] = "verified_failed"
                    reason = str(
                        after_state.get("error")
                        or "browser_fullscreen_api_unavailable"
                    )
                    raise TapCalibrationError(
                        "手机浏览器未提供可信全屏，且视口覆盖不足，"
                        f"禁止继续边缘标定：{reason}"
                    )
            else:
                assert sequence is not None
                record = wait_for_new_sample(base_url, len(server_samples))
                expected_phase = "complete" if sequence == 8 else "calibration"
                after_state = wait_for_page_state(
                    base_url,
                    phases={expected_phase},
                    timeout=8.0,
                    newer_than=str(locked_state.get("updated_at") or ""),
                )
                if (
                    int(after_state.get("sequence", -1)) != sequence + 1
                    or not calibration_page_ready(after_state)
                ):
                    raise TapCalibrationError(
                        "动作后页面sequence、phase或校准模式验证失败。"
                    )
                sample = _build_sample(attempt, record)
                local_samples.append(sample)
                attempt["sample"] = sample
                attempt["page_state_after"] = after_state
            attempt["status"] = "verified"
            attempt["verified_at"] = datetime.now(timezone.utc).isoformat()
        except TapCalibrationError:
            if attempt.get("status") != "verified_failed":
                attempt["status"] = "awaiting_recovery"
            raise
        except Exception as exc:
            attempt["status"] = "awaiting_recovery"
            attempt["error"] = str(exc)
            raise TapCalibrationError(
                "单步动作后验证异常；已记录为待恢复，禁止自动重试。"
            ) from exc
        finally:
            _capture_after(robot, output_dir, attempt)
            _persist_attempt(output_dir, progress, attempt)

    return {
        "status": "verified",
        "mode": mode,
        "executed_action": kind,
        "sequence": sequence,
        "physical_actions": 1,
        "sample_count": len(local_samples),
        "page_state": attempt["page_state_after"],
        "device_id": device_id,
        "output_dir": str(output_dir),
        "next_invocation_required": True,
    }


def _run_calibration_step_reserved(
    *,
    mode: str,
    base_url: str,
    robot: RobotController,
    execute: bool,
    device_status_url: str,
    reserved_device_id: str | None,
    default_device_id: str | None,
    calibration_session_id: str | None,
    output_root: Path = OUTPUT_ROOT,
    calibration_path: Path = CALIBRATION_PATH,
) -> dict[str, object]:
    """Run zero or one recoverable calibration action for one page session."""

    if mode not in {"collect", "validate"}:
        raise TapCalibrationError(f"不支持的校准阶段：{mode}")
    page_session_id, server_samples = _page_snapshot(base_url)
    output_dir = output_root / f"{mode}_{_safe_session_component(page_session_id)}"
    output_dir.mkdir(parents=True, exist_ok=True)
    lease = InterProcessLease(
        output_dir / "step.lease",
        owner_id=f"xy-calibration-{mode}-{page_session_id}",
        metadata={"mode": mode, "page_session_id": page_session_id},
    )
    if not lease.acquire():
        raise TapCalibrationError("当前校准页面会话已有另一个单步进程在运行。")
    try:
        progress = _load_progress(
            output_dir,
            mode=mode,
            page_session_id=page_session_id,
            base_url=base_url,
        )
        page_state = wait_for_page_state(
            base_url,
            phases={"fullscreen_setup", "calibration", "complete", "blocked"},
        )
        recovered = _recover_previous_attempt(
            base_url=base_url,
            robot=robot,
            output_dir=output_dir,
            progress=progress,
            page_state=page_state,
            server_samples=server_samples,
        )
        if recovered is not None:
            return recovered

        local_samples = progress["samples"]
        if len(local_samples) != len(server_samples):
            raise TapCalibrationError(
                "本地持久化样本与当前页面样本数量不一致，禁止混用或跳步。"
            )
        if len(local_samples) > 9:
            raise TapCalibrationError("校准样本超过九个，禁止继续。")
        if progress.get("finalized") is True:
            return {
                "status": "already_finalized",
                "mode": mode,
                "physical_actions": 0,
                "sample_count": len(local_samples),
                "output_dir": str(output_dir),
                "accepted_fit": progress.get("accepted_fit"),
                "validation_passed": progress.get("validation_passed"),
            }
        if page_state.get("phase") == "blocked":
            raise TapCalibrationError("校准页已安全阻断；刷新形成新会话前禁止重试。")
        if len(local_samples) == 9:
            if page_state.get("phase") != "complete" or int(
                page_state.get("sequence", -1)
            ) != 9:
                raise TapCalibrationError("九点齐全但页面未处于可信完成状态。")
            return _finalize_session(
                mode=mode,
                progress=progress,
                output_dir=output_dir,
                calibration_path=calibration_path,
            )

        phase = str(page_state.get("phase") or "")
        if phase == "fullscreen_setup":
            if server_samples:
                raise TapCalibrationError("全屏准备阶段不应已有边缘样本。")
            if any(
                attempt.get("kind") == "fullscreen_setup"
                for attempt in progress["attempts"]
            ):
                raise TapCalibrationError("本页面会话已尝试过中央准备，禁止再次落笔。")
            kind = "fullscreen_setup"
            sequence: int | None = None
            correction = None
        elif phase == "calibration":
            if not calibration_page_ready(page_state):
                raise TapCalibrationError("校准页缺少可信全屏或高覆盖视口证据。")
            sequence = len(local_samples)
            if int(page_state.get("sequence", -1)) != sequence:
                raise TapCalibrationError("校准页sequence与持久化样本数量不一致。")
            if any(
                attempt.get("kind") == "calibration_point"
                and attempt.get("sequence") == sequence
                for attempt in progress["attempts"]
            ):
                raise TapCalibrationError("当前sequence已有物理动作记录，禁止重复落笔。")
            kind = "calibration_point"
            correction = None
            if mode == "validate":
                if not calibration_path.exists():
                    raise TapCalibrationError("找不到待验证的tap_calibration.json。")
                payload = json.loads(calibration_path.read_text(encoding="utf-8"))
                if not payload.get("accepted_fit"):
                    raise TapCalibrationError("拟合质量未通过，禁止开始独立验证。")
                correction = Affine2D.from_json(payload["target_to_command"])
        else:
            raise TapCalibrationError(f"当前页面阶段不能执行新的单步动作：{phase}")

        frame, detected = wait_for_stable_target(robot)
        desired_x, desired_y, box = detected
        command_x, command_y = desired_x, desired_y
        if correction is not None:
            nx, ny = correction.apply(
                desired_x / (frame.width - 1), desired_y / (frame.height - 1)
            )
            command_x = int(round(nx * (frame.width - 1)))
            command_y = int(round(ny * (frame.height - 1)))
        if kind == "calibration_point":
            stored_size = progress.get("frame_size")
            current_size = [frame.width, frame.height]
            if stored_size is None:
                progress["frame_size"] = current_size
            elif stored_size != current_size:
                raise TapCalibrationError("跨调用采集时相机画面尺寸发生变化。")

        preview = {
            "status": "awaiting_explicit_execution",
            "mode": mode,
            "next_action": kind,
            "sequence": sequence,
            "physical_actions": 0,
            "target_frame": [desired_x, desired_y],
            "command_frame": [command_x, command_y],
            "frame_size": [frame.width, frame.height],
            "observation_fingerprint": _frame_fingerprint(frame),
            "page_state": page_state,
            "page_session_id": page_session_id,
            "sample_count": len(local_samples),
            "device_id": reserved_device_id,
            "output_dir": str(output_dir),
        }
        _save_progress(output_dir, progress)
        if not execute:
            return preview
        if not all(
            value
            for value in (
                reserved_device_id,
                default_device_id,
                calibration_session_id,
            )
        ):
            raise TapCalibrationError("真实单步执行缺少共享设备reservation。")
        return _execute_calibration_action(
            mode=mode,
            base_url=base_url,
            robot=robot,
            output_dir=output_dir,
            progress=progress,
            page_state=page_state,
            page_session_id=page_session_id,
            server_samples=server_samples,
            local_samples=local_samples,
            frame=frame,
            detected=(desired_x, desired_y, box),
            correction=correction,
            kind=kind,
            sequence=sequence,
            device_status_url=device_status_url,
            device_id=str(reserved_device_id),
            default_device_id=str(default_device_id),
            calibration_session_id=str(calibration_session_id),
        )
    finally:
        lease.release()


def run_calibration_step(
    *,
    mode: str,
    base_url: str,
    robot: RobotController,
    execute: bool,
    device_status_url: str = "http://127.0.0.1:8765/api/device",
    device_id: str | None = None,
    output_root: Path = OUTPUT_ROOT,
    calibration_path: Path = CALIBRATION_PATH,
) -> dict[str, object]:
    """Run zero or one recoverable action under the shared device gates."""

    if not execute:
        return _run_calibration_step_reserved(
            mode=mode,
            base_url=base_url,
            robot=robot,
            execute=False,
            device_status_url=device_status_url,
            reserved_device_id=None,
            default_device_id=None,
            calibration_session_id=None,
            output_root=output_root,
            calibration_path=calibration_path,
        )
    with _calibration_device_reservation(
        device_status_url=device_status_url,
        device_id=device_id,
    ) as (reserved_device_id, default_device_id, calibration_session_id):
        return _run_calibration_step_reserved(
            mode=mode,
            base_url=base_url,
            robot=robot,
            execute=True,
            device_status_url=device_status_url,
            reserved_device_id=reserved_device_id,
            default_device_id=default_device_id,
            calibration_session_id=calibration_session_id,
            output_root=output_root,
            calibration_path=calibration_path,
        )


def validate_samples(samples: list[dict[str, object]]) -> dict[str, float | bool]:
    distances = []
    for sample in samples:
        width, height = (float(value) for value in sample["viewport_size"])
        target = sample["target_dom"]
        actual = sample["actual_dom"]
        dx = (float(actual[0]) - float(target[0])) * width
        dy = (float(actual[1]) - float(target[1])) * height
        distances.append(math.hypot(dx, dy))
    rms = math.sqrt(sum(value * value for value in distances) / len(distances))
    maximum = max(distances)
    return {
        "rms_error_css_px": round(rms, 4),
        "max_error_css_px": round(maximum, 4),
        "passed": bool(rms <= 8.0 and maximum <= 16.0),
    }


def _probe_single_touch_reserved(
    *,
    base_url: str,
    robot: RobotController,
    execute: bool,
    output_dir: Path | None = None,
    device_status_url: str,
    reserved_device_id: str | None,
    default_device_id: str | None,
    calibration_session_id: str | None,
) -> dict[str, object]:
    """Probe one harmless browser target through the production tap path.

    Dry-run only proves that the calibration target is visible and stable.
    Execute mode issues at most one physical tap, never modifies the active
    calibration, and records whether the phone reported a touch at all before
    judging coordinate accuracy.
    """

    frame, detected = wait_for_stable_target(robot)
    target_x, target_y, box = detected
    requested_grid = (
        int(round(target_x * 1000 / max(1, frame.width - 1))),
        int(round(target_y * 1000 / max(1, frame.height - 1))),
    )
    result: dict[str, object] = {
        "mode": "single_touch_probe",
        "ready": True,
        "execute": bool(execute),
        "physical_actions": 0,
        "target_frame": [target_x, target_y],
        "requested_grid": list(requested_grid),
        "frame_size": [frame.width, frame.height],
        "contact_detected": False,
        "coordinate_passed": False,
        "passed": False,
    }
    if not execute:
        result["status"] = "awaiting_explicit_execution"
        return result

    if output_dir is None:
        raise TapCalibrationError("单点执行探测必须提供证据目录。")
    output_dir.mkdir(parents=True, exist_ok=False)
    annotated = frame.copy()
    draw = ImageDraw.Draw(annotated)
    draw.rectangle(box, outline="#00ff66", width=3)
    draw.line(
        (target_x - 12, target_y, target_x + 12, target_y),
        fill="#ffff00",
        width=2,
    )
    draw.line(
        (target_x, target_y - 12, target_x, target_y + 12),
        fill="#ffff00",
        width=2,
    )
    before_path = output_dir / "before.jpg"
    annotated.save(before_path, quality=94)
    result["evidence"] = [str(before_path)]

    if not all(
        value
        for value in (
            reserved_device_id,
            default_device_id,
            calibration_session_id,
        )
    ):
        raise TapCalibrationError("真实单点探测缺少共享设备reservation。")
    page_session_id, page_samples = _page_snapshot(base_url)
    previous_count = len(page_samples)
    with _calibration_physical_lease(
        device_status_url=device_status_url,
        device_id=str(reserved_device_id),
        default_device_id=str(default_device_id),
        calibration_session_id=str(calibration_session_id),
        mode="probe",
        page_session_id=page_session_id,
    ):
        locked_session_id, locked_samples = _page_snapshot(base_url)
        if (
            locked_session_id != page_session_id
            or len(locked_samples) != previous_count
        ):
            raise TapCalibrationError(
                "物理锁内探测页面session或样本数量已变化，保持0动作。"
            )
        locked_frame, locked_detected = wait_for_stable_target(robot)
        locked_x, locked_y, _locked_box = locked_detected
        if (
            (locked_frame.width, locked_frame.height) != (frame.width, frame.height)
            or math.hypot(locked_x - target_x, locked_y - target_y) > 5.0
        ):
            raise TapCalibrationError("物理锁内探测靶点与预检观察不一致，保持0动作。")
        requested_grid = (
            int(round(locked_x * 1000 / max(1, locked_frame.width - 1))),
            int(round(locked_y * 1000 / max(1, locked_frame.height - 1))),
        )
        result["target_frame"] = [locked_x, locked_y]
        result["requested_grid"] = list(requested_grid)
        result["frame_size"] = [locked_frame.width, locked_frame.height]
        try:
            # Count conservatively only after both shared gates and the fresh
            # lock-internal observation hold, exactly as the action method is
            # entered.
            result["physical_actions"] = 1
            command_frame = robot.vision_tap_relative(*requested_grid)
            result["command_frame"] = list(command_frame)
            record = wait_for_new_sample(base_url, previous_count)
            sample = {
                "sequence": int(record["sequence"]),
                "desired_frame": [locked_x, locked_y],
                "command_frame": list(command_frame),
                "target_dom": normalized_dom(record, "target"),
                "actual_dom": normalized_dom(record, "actual"),
                "viewport_size": [record["viewport_width"], record["viewport_height"]],
            }
            validation = validate_samples([sample])
            result.update(
                {
                    "status": "complete",
                    "contact_detected": True,
                    "coordinate_passed": bool(validation["passed"]),
                    "passed": bool(validation["passed"]),
                    "sample": sample,
                    "validation": validation,
                }
            )
        except Exception as exc:
            result.update(
                {
                    "status": "failed",
                    "error": str(exc),
                    "passed": False,
                }
            )
        finally:
            try:
                after_path = output_dir / "after.jpg"
                robot.vision_capture().convert("RGB").save(after_path, quality=94)
                result.setdefault("evidence", []).append(str(after_path))
            except Exception as capture_error:
                result["after_capture_error"] = str(capture_error)
            report_path = output_dir / "report.json"
            report_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            result.setdefault("evidence", []).append(str(report_path))
    return result


def probe_single_touch(
    *,
    base_url: str,
    robot: RobotController,
    execute: bool,
    output_dir: Path | None = None,
    device_status_url: str = "http://127.0.0.1:8765/api/device",
    device_id: str | None = None,
) -> dict[str, object]:
    if not execute:
        return _probe_single_touch_reserved(
            base_url=base_url,
            robot=robot,
            execute=False,
            output_dir=output_dir,
            device_status_url=device_status_url,
            reserved_device_id=None,
            default_device_id=None,
            calibration_session_id=None,
        )
    with _calibration_device_reservation(
        device_status_url=device_status_url,
        device_id=device_id,
    ) as (reserved_device_id, default_device_id, calibration_session_id):
        return _probe_single_touch_reserved(
            base_url=base_url,
            robot=robot,
            execute=True,
            output_dir=output_dir,
            device_status_url=device_status_url,
            reserved_device_id=reserved_device_id,
            default_device_id=default_device_id,
            calibration_session_id=calibration_session_id,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect and validate nine physical XY touch points")
    parser.add_argument("phase", choices=("probe", "collect", "validate"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8770")
    parser.add_argument(
        "--device-status-url",
        default="http://127.0.0.1:8765/api/device",
        help="Uvicorn设备状态接口；真实--execute必须在共享reservation后复查在线且空闲",
    )
    parser.add_argument(
        "--device-id",
        default=None,
        help="目标device_id；省略时使用/api/device报告的default_device_id",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "明确允许本次调用最多一次物理点击；省略时只做零动作预检。"
            "collect/validate 每次只处理中央准备或当前sequence的一个触点"
        ),
    )
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")
    request_json(f"{base_url}/api/health")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.phase == "probe":
        output = (
            OUTPUT_ROOT / f"probe_{timestamp}"
            if args.execute
            else None
        )
        result = probe_single_touch(
            base_url=base_url,
            robot=RobotController(),
            execute=bool(args.execute),
            output_dir=output,
            device_status_url=args.device_status_url,
            device_id=args.device_id,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if not args.execute:
            print("单点触控预检通过；尚未执行物理动作。")
            return 0
        return 0 if result["passed"] else 1

    result = run_calibration_step(
        mode=args.phase,
        base_url=base_url,
        robot=RobotController(),
        execute=bool(args.execute),
        device_status_url=args.device_status_url,
        device_id=args.device_id,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] == "awaiting_explicit_execution":
        print("单步预检通过；尚未执行物理动作。确认后仅对当前新鲜状态重跑 --execute。")
    elif result["status"] == "fit_complete":
        print("拟合完成但尚未启用。重置校准页形成新session后，逐次运行 validate。")
    elif result["status"] == "validation_complete":
        print("多位置XY纠偏已通过独立九点验证并启用。")
    else:
        print("本次单步已结束；必须重新调用命令获取新状态，禁止在本次继续下一动作。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
