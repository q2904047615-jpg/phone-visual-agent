from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image, ImageDraw

import robot_gui_poc as legacy
from robot_core import RobotController, load_workflow_config
from tap_calibration import (
    CALIBRATION_PATH,
    Affine2D,
    TapCalibrationError,
    build_calibration,
    build_coverage,
    save_calibration,
)


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "output" / "xy_calibration"


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
) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    latest: dict[str, object] = {}
    while time.monotonic() < deadline:
        latest = request_json(f"{base_url}/api/page-state")
        updated_at = str(latest.get("updated_at") or "")
        try:
            timestamp = datetime.fromisoformat(updated_at)
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            age = (datetime.now(timezone.utc) - timestamp).total_seconds()
        except ValueError:
            age = float("inf")
        if str(latest.get("phase") or "") in phases and 0 <= age <= 3.0:
            return latest
        time.sleep(0.2)
    raise TapCalibrationError(
        f"手机校准页没有进入预期阶段{sorted(phases)}；当前状态：{latest}"
    )


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


def ensure_fullscreen_calibration_page(
    *,
    base_url: str,
    robot: RobotController,
    output_dir: Path,
) -> int:
    """Prepare a bounded calibration viewport with one freshly observed tap.

    Real fullscreen is preferred. A viewport-only fallback is accepted only
    when both CSS screen ratios are tightly bounded; the later camera-space
    convex-hull gate remains authoritative before a calibration can activate.
    """

    state = wait_for_page_state(
        base_url,
        phases={"fullscreen_setup", "calibration", "complete"},
    )
    if state.get("phase") == "calibration" and calibration_page_ready(state):
        return 0
    if state.get("phase") == "complete" and calibration_page_ready(state):
        # A just-reset page can need one polling interval to show point 1 again.
        state = wait_for_page_state(base_url, phases={"calibration"}, timeout=3.0)
        if calibration_page_ready(state):
            return 0
    if state.get("phase") != "fullscreen_setup":
        raise TapCalibrationError(f"校准页状态不允许进入全屏：{state}")

    frame, detected = wait_for_stable_target(robot)
    target_x, target_y, box = detected
    annotated = frame.copy()
    draw = ImageDraw.Draw(annotated)
    draw.rectangle(box, outline="#00ff66", width=3)
    annotated.save(output_dir / "00_fullscreen_setup_before.jpg", quality=94)
    result: dict[str, object] = {
        "mode": "fullscreen_setup",
        "physical_actions": 1,
        "target_frame": [target_x, target_y],
        "before": str(output_dir / "00_fullscreen_setup_before.jpg"),
        "passed": False,
    }
    click_raw_pixel(robot, frame, (target_x, target_y))
    try:
        state = wait_for_page_state(
            base_url,
            phases={"calibration", "blocked"},
            timeout=8.0,
        )
        result["page_state"] = state
        result["passed"] = calibration_page_ready(state)
        result["fullscreen_entered"] = state.get("fullscreen") is True
        result["safe_viewport_fallback"] = (
            state.get("calibration_mode") == "viewport_coverage"
        )
    except Exception as exc:
        result["error"] = str(exc)
        result["page_state"] = request_json(f"{base_url}/api/page-state")
        raise
    finally:
        after_path = output_dir / "00_fullscreen_setup_after.jpg"
        robot.vision_capture().convert("RGB").save(after_path, quality=94)
        result["after"] = str(after_path)
        (output_dir / "00_fullscreen_setup.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if not calibration_page_ready(state):
        reason = str(state.get("error") or "browser_fullscreen_api_unavailable")
        raise TapCalibrationError(
            f"手机浏览器未提供可信全屏，且视口覆盖不足，禁止继续边缘标定：{reason}"
        )
    return 1


def collect_points(
    *,
    base_url: str,
    output_dir: Path,
    correction: Affine2D | None,
) -> tuple[list[dict[str, object]], tuple[int, int]]:
    robot = RobotController()
    output_dir.mkdir(parents=True, exist_ok=True)
    collected: list[dict[str, object]] = []
    frame_size: tuple[int, int] | None = None
    setup_actions = ensure_fullscreen_calibration_page(
        base_url=base_url,
        robot=robot,
        output_dir=output_dir,
    )

    for sequence in range(9):
        time.sleep(0.75 if sequence else 0.2)
        frame, detected = wait_for_stable_target(robot)
        if frame_size is None:
            frame_size = (frame.width, frame.height)
        elif frame_size != (frame.width, frame.height):
            raise TapCalibrationError("采集中相机画面尺寸发生变化。")
        desired_x, desired_y, box = detected
        command_x, command_y = desired_x, desired_y
        if correction is not None:
            nx, ny = correction.apply(
                desired_x / (frame.width - 1), desired_y / (frame.height - 1)
            )
            command_x = int(round(nx * (frame.width - 1)))
            command_y = int(round(ny * (frame.height - 1)))

        annotated = frame.copy()
        draw = ImageDraw.Draw(annotated)
        draw.rectangle(box, outline="#00ff66", width=3)
        draw.line((command_x - 12, command_y, command_x + 12, command_y), fill="#ffff00", width=2)
        draw.line((command_x, command_y - 12, command_x, command_y + 12), fill="#ffff00", width=2)
        annotated.save(output_dir / f"{sequence + 1:02d}_before.jpg", quality=94)

        previous = request_json(f"{base_url}/api/samples")
        previous_count = len(previous.get("samples", []))
        click_raw_pixel(robot, frame, (command_x, command_y))
        record = wait_for_new_sample(base_url, previous_count)
        if int(record["sequence"]) != sequence:
            raise TapCalibrationError(
                f"校准页序号不一致：期望{sequence}，实际{record['sequence']}。"
            )
        collected.append(
            {
                "sequence": sequence,
                "desired_frame": [desired_x, desired_y],
                "command_frame": [command_x, command_y],
                "target_dom": normalized_dom(record, "target"),
                "actual_dom": normalized_dom(record, "actual"),
                "viewport_size": [record["viewport_width"], record["viewport_height"]],
            }
        )
        (output_dir / "samples.json").write_text(
            json.dumps(collected, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    assert frame_size is not None
    (output_dir / "execution.json").write_text(
        json.dumps(
            {
                "fullscreen_setup_actions": setup_actions,
                "calibration_point_actions": len(collected),
                "physical_actions": setup_actions + len(collected),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return collected, frame_size


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


def probe_single_touch(
    *,
    base_url: str,
    robot: RobotController,
    execute: bool,
    output_dir: Path | None = None,
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

    previous = request_json(f"{base_url}/api/samples")
    previous_count = len(previous.get("samples", []))
    # Count conservatively as soon as the production action method is entered.
    # If the vendor layer raises after touching the phone, evidence must never
    # claim that zero physical actions were possible.
    result["physical_actions"] = 1
    try:
        command_frame = robot.vision_tap_relative(*requested_grid)
        result["command_frame"] = list(command_frame)
        record = wait_for_new_sample(base_url, previous_count)
        sample = {
            "sequence": int(record["sequence"]),
            "desired_frame": [target_x, target_y],
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect and validate nine physical XY touch points")
    parser.add_argument("phase", choices=("probe", "collect", "validate"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8770")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="probe 模式下明确允许最多一次物理点击；省略时只做零动作预检",
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
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if not args.execute:
            print("单点触控预检通过；尚未执行物理动作。")
            return 0
        return 0 if result["passed"] else 1

    if args.phase == "collect":
        output = OUTPUT_ROOT / f"collect_{timestamp}"
        samples, frame_size = collect_points(base_url=base_url, output_dir=output, correction=None)
        payload = build_calibration(samples, frame_size)
        payload["collection_dir"] = str(output)
        save_calibration(payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        if not payload["accepted_fit"]:
            raise TapCalibrationError("九点拟合质量或屏幕覆盖范围未通过，校准未启用。")
        print("拟合完成但尚未启用。请在手机刷新校准页后运行 validate。")
        return 0

    if not CALIBRATION_PATH.exists():
        raise TapCalibrationError("找不到待验证的tap_calibration.json。")
    payload = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    if not payload.get("accepted_fit"):
        raise TapCalibrationError("拟合质量未通过，禁止验证和启用。")
    correction = Affine2D.from_json(payload["target_to_command"])
    output = OUTPUT_ROOT / f"validate_{timestamp}"
    samples, frame_size = collect_points(base_url=base_url, output_dir=output, correction=correction)
    if list(frame_size) != payload.get("frame_size"):
        raise TapCalibrationError("验证时相机尺寸与采集时不一致。")
    validation = validate_samples(samples)
    validation_coverage = build_coverage(
        [
            (
                float(sample["desired_frame"][0]) / (frame_size[0] - 1),
                float(sample["desired_frame"][1]) / (frame_size[1] - 1),
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
    payload["validation_dir"] = str(output)
    payload["validated_at"] = datetime.now(timezone.utc).isoformat()
    payload["validated"] = bool(validation["passed"])
    payload["enabled"] = bool(validation["passed"])
    save_calibration(payload)
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    if not validation["passed"]:
        raise TapCalibrationError("独立九点验证未通过，纠偏保持禁用。")
    print("多位置XY纠偏已通过独立验证并启用。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
