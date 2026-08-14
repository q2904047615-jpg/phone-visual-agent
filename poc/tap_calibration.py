from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


CALIBRATION_PATH = Path(__file__).with_name("tap_calibration.json")
CALIBRATION_VERSION = 2
MIN_COVERAGE_SPAN_X = 0.68
MIN_COVERAGE_SPAN_Y = 0.82


class TapCalibrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Affine2D:
    """Two-row affine transform operating on normalized coordinates."""

    rows: tuple[tuple[float, float, float], tuple[float, float, float]]

    def apply(self, x: float, y: float) -> tuple[float, float]:
        first, second = self.rows
        return (
            first[0] * x + first[1] * y + first[2],
            second[0] * x + second[1] * y + second[2],
        )

    def inverse(self) -> "Affine2D":
        matrix = np.array(
            [
                [*self.rows[0]],
                [*self.rows[1]],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        if abs(float(np.linalg.det(matrix))) < 1e-7:
            raise TapCalibrationError("校准矩阵接近奇异，无法安全求逆。")
        inverse = np.linalg.inv(matrix)
        return Affine2D(
            (
                tuple(float(value) for value in inverse[0, :3]),
                tuple(float(value) for value in inverse[1, :3]),
            )
        )

    def to_json(self) -> list[list[float]]:
        return [list(self.rows[0]), list(self.rows[1])]

    @classmethod
    def from_json(cls, value: Sequence[Sequence[float]]) -> "Affine2D":
        if len(value) != 2 or any(len(row) != 3 for row in value):
            raise TapCalibrationError("二维仿射矩阵必须是2×3。")
        return cls(
            (
                tuple(float(item) for item in value[0]),
                tuple(float(item) for item in value[1]),
            )
        )


def fit_affine(
    source: Iterable[tuple[float, float]],
    destination: Iterable[tuple[float, float]],
) -> tuple[Affine2D, np.ndarray]:
    source_array = np.asarray(list(source), dtype=float)
    destination_array = np.asarray(list(destination), dtype=float)
    if source_array.shape != destination_array.shape:
        raise TapCalibrationError("源点和目标点数量不一致。")
    if source_array.ndim != 2 or source_array.shape[1] != 2:
        raise TapCalibrationError("校准点必须是二维坐标。")
    if source_array.shape[0] < 3:
        raise TapCalibrationError("至少需要3个不共线校准点。")
    design = np.column_stack((source_array, np.ones(source_array.shape[0])))
    solution, _residuals, rank, _singular = np.linalg.lstsq(
        design, destination_array, rcond=None
    )
    if rank < 3:
        raise TapCalibrationError("校准点共线，无法拟合二维纠偏。")
    transform = Affine2D(
        (
            tuple(float(value) for value in solution[:, 0]),
            tuple(float(value) for value in solution[:, 1]),
        )
    )
    predicted = design @ solution
    errors = destination_array - predicted
    return transform, errors


def _convex_hull(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return the counter-clockwise convex hull without repeating its first point."""

    ordered = sorted(set(points))
    if len(ordered) < 3:
        raise TapCalibrationError("校准覆盖点不足，无法形成安全区域。")

    def cross(
        origin: tuple[float, float],
        first: tuple[float, float],
        second: tuple[float, float],
    ) -> float:
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (
            first[1] - origin[1]
        ) * (second[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        raise TapCalibrationError("校准覆盖点共线，无法形成安全区域。")
    return hull


def build_coverage(
    normalized_points: Sequence[tuple[float, float]],
) -> dict[str, object]:
    hull = _convex_hull(normalized_points)
    xs = [point[0] for point in hull]
    ys = [point[1] for point in hull]
    bounds = [min(xs), min(ys), max(xs), max(ys)]
    span_x = bounds[2] - bounds[0]
    span_y = bounds[3] - bounds[1]
    sufficient = bool(
        span_x >= MIN_COVERAGE_SPAN_X and span_y >= MIN_COVERAGE_SPAN_Y
    )
    return {
        "kind": "convex_hull",
        "normalized_hull": [[float(x), float(y)] for x, y in hull],
        "normalized_bounds": [float(value) for value in bounds],
        "span": [round(float(span_x), 6), round(float(span_y), 6)],
        "sufficient": sufficient,
    }


def _point_in_convex_hull(
    point: tuple[float, float],
    hull: Sequence[Sequence[float]],
    *,
    tolerance: float = 0.003,
) -> bool:
    if len(hull) < 3:
        return False
    direction = 0
    x, y = point
    for index, first in enumerate(hull):
        second = hull[(index + 1) % len(hull)]
        cross = (float(second[0]) - float(first[0])) * (y - float(first[1])) - (
            float(second[1]) - float(first[1])
        ) * (x - float(first[0]))
        if abs(cross) <= tolerance:
            continue
        current = 1 if cross > 0 else -1
        if direction and current != direction:
            return False
        direction = current
    return True


def build_calibration(
    samples: Sequence[dict[str, object]], frame_size: tuple[int, int]
) -> dict[str, object]:
    """Build target->command correction from browser touch observations.

    Each sample contains the camera-space target/command point and the target
    and observed browser coordinates.  The browser plane is used only as an
    independent touch sensor; the saved correction remains in camera-normalized
    coordinates, which is what RobotController consumes.
    """

    if len(samples) < 6:
        raise TapCalibrationError("多位置校准至少需要6个有效触点。")
    width, height = frame_size
    if width < 2 or height < 2:
        raise TapCalibrationError("相机画面尺寸无效。")

    desired_frame: list[tuple[float, float]] = []
    command_frame = []
    target_dom = []
    actual_dom = []
    for sample in samples:
        desired = sample["desired_frame"]
        command = sample.get("command_frame", desired)
        target = sample["target_dom"]
        actual = sample["actual_dom"]
        desired_frame.append(
            (float(desired[0]) / (width - 1), float(desired[1]) / (height - 1))
        )
        command_frame.append(
            (float(command[0]) / (width - 1), float(command[1]) / (height - 1))
        )
        target_dom.append((float(target[0]), float(target[1])))
        actual_dom.append((float(actual[0]), float(actual[1])))

    frame_to_dom, projection_errors = fit_affine(desired_frame, target_dom)
    dom_to_frame = frame_to_dom.inverse()
    observed_frame = [dom_to_frame.apply(x, y) for x, y in actual_dom]
    command_to_observed, physical_errors = fit_affine(command_frame, observed_frame)
    target_to_command = command_to_observed.inverse()

    physical_pixel_errors = np.column_stack(
        (physical_errors[:, 0] * (width - 1), physical_errors[:, 1] * (height - 1))
    )
    distances = np.linalg.norm(physical_pixel_errors, axis=1)
    projection_scale = np.array([width - 1, height - 1], dtype=float)
    projection_distances = np.linalg.norm(projection_errors * projection_scale, axis=1)
    rms = float(math.sqrt(float(np.mean(distances**2))))
    maximum = float(np.max(distances))
    coverage = build_coverage(desired_frame)

    # A noisy/unstable page must never silently become an active correction.
    # A precise fit over only the middle of the screen is also unsafe because
    # small edge controls would otherwise rely on unmeasured extrapolation.
    accepted = bool(
        rms <= 12.0 and maximum <= 25.0 and coverage["sufficient"]
    )
    return {
        "version": CALIBRATION_VERSION,
        "enabled": False,
        "accepted_fit": accepted,
        "validated": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "frame_size": [width, height],
        "sample_count": len(samples),
        "target_to_command": target_to_command.to_json(),
        "frame_to_dom": frame_to_dom.to_json(),
        "fit": {
            "rms_error_px": round(rms, 4),
            "max_error_px": round(maximum, 4),
            "projection_rms_px": round(
                float(math.sqrt(float(np.mean(projection_distances**2)))), 4
            ),
        },
        "coverage": coverage,
        "samples": list(samples),
    }


def save_calibration(payload: dict[str, object], path: Path = CALIBRATION_PATH) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_active_calibration(
    frame_size: tuple[int, int], path: Path = CALIBRATION_PATH
) -> Affine2D | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not payload.get("enabled") or not payload.get("validated"):
            return None
        stored_size = payload.get("frame_size")
        if not isinstance(stored_size, list) or len(stored_size) != 2:
            return None
        stored_width = float(stored_size[0])
        stored_height = float(stored_size[1])
        current_width = float(frame_size[0])
        current_height = float(frame_size[1])
        if min(stored_width, stored_height, current_width, current_height) <= 0:
            return None
        # The affine transform is stored in normalized image coordinates, so
        # an isotropic DPI/display resize (for example 540x960 -> 810x1440)
        # must not disable a previously validated physical calibration.  A
        # non-uniform resize changes camera geometry and remains fail-closed.
        width_scale = current_width / stored_width
        height_scale = current_height / stored_height
        scale_delta = abs(width_scale - height_scale) / max(
            width_scale,
            height_scale,
        )
        if scale_delta > 0.02:
            return None
        return Affine2D.from_json(payload["target_to_command"])
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError, TapCalibrationError):
        return None


def corrected_grid_point(
    x: int,
    y: int,
    frame_size: tuple[int, int],
    path: Path = CALIBRATION_PATH,
) -> tuple[int, int]:
    calibration = load_active_calibration(frame_size, path)
    if calibration is None:
        return x, y
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload.get("version", 0)) < CALIBRATION_VERSION:
            raise TapCalibrationError("当前触控标定缺少屏幕覆盖边界，必须重新标定。")
        coverage = payload.get("coverage")
        if not isinstance(coverage, dict) or not coverage.get("sufficient"):
            raise TapCalibrationError("当前触控标定未覆盖足够屏幕区域，已拒绝点击。")
        hull = coverage.get("normalized_hull")
        if not isinstance(hull, list) or not _point_in_convex_hull(
            (x / 1000.0, y / 1000.0), hull
        ):
            bounds = coverage.get("normalized_bounds", [])
            raise TapCalibrationError(
                f"目标点({x}, {y})位于实测标定区域之外{bounds}，已拒绝外推点击。"
            )
    except TapCalibrationError:
        raise
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise TapCalibrationError("无法验证触控标定覆盖边界，已拒绝点击。") from exc
    corrected_x, corrected_y = calibration.apply(x / 1000.0, y / 1000.0)
    if not (-0.08 <= corrected_x <= 1.08 and -0.08 <= corrected_y <= 1.08):
        raise TapCalibrationError("纠偏结果超出安全边界，已拒绝点击。")
    return (
        min(1000, max(0, int(round(corrected_x * 1000)))),
        min(1000, max(0, int(round(corrected_y * 1000)))),
    )
