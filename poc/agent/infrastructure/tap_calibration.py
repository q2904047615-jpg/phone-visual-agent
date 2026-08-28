"""File-backed touch calibration and coordinate correction infrastructure."""

from __future__ import annotations

from agent.domain.validation import reject_if
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


POC_ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_PATH = POC_ROOT / "tap_calibration.json"
CALIBRATION_VERSION = 2
MIN_COVERAGE_SPAN_X = 0.68
MIN_COVERAGE_SPAN_Y = 0.82
SYSTEM_NAVIGATION_DOM_CENTER_X = 0.5
SYSTEM_NAVIGATION_DOM_INWARD_DISTANCE = 0.25
SYSTEM_NAVIGATION_MIN_BOTTOM_Y = 0.90
SYSTEM_NAVIGATION_MAX_FRAME_TO_DOM_ERROR = 0.02


class TapCalibrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Affine2D:
    """Two-row affine transform operating on normalized coordinates."""

    rows: tuple[tuple[float, float, float], tuple[float, float, float]]

    def apply(self, x: float, y: float) -> tuple[float, float]:
        first, second = self.rows
        return (first[0] * x + first[1] * y + first[2], second[0] * x + second[1] * y + second[2])

    def inverse(self) -> 'Affine2D':
        matrix = np.array([[*self.rows[0]], [*self.rows[1]], [0.0, 0.0, 1.0]], dtype=float)
        reject_if(abs(float(np.linalg.det(matrix))) < 1e-07, TapCalibrationError("校准矩阵接近奇异，无法安全求逆。"))
        inverse = np.linalg.inv(matrix)
        return Affine2D((tuple((float(value) for value in inverse[0, :3])), tuple((float(value) for value in inverse[1,
            :3]))))

    def to_json(self) -> list[list[float]]:
        return [list(self.rows[0]), list(self.rows[1])]

    @classmethod
    def from_json(cls, value: Sequence[Sequence[float]]) -> 'Affine2D':
        reject_if(len(value) != 2 or any((len(row) != 3 for row in value)), TapCalibrationError("二维仿射矩阵必须是2×3。"))
        return cls((tuple((float(item) for item in value[0])), tuple((float(item) for item in value[1]))))


def fit_affine(source: Iterable[tuple[float, float]], destination: Iterable[tuple[float, float]]) -> tuple[Affine2D,
    np.ndarray]:
    source_array = np.asarray(list(source), dtype=float)
    destination_array = np.asarray(list(destination), dtype=float)
    reject_if(source_array.shape != destination_array.shape, TapCalibrationError("源点和目标点数量不一致。"))
    reject_if(source_array.ndim != 2 or source_array.shape[1] != 2, TapCalibrationError("校准点必须是二维坐标。"))
    reject_if(source_array.shape[0] < 3, TapCalibrationError("至少需要3个不共线校准点。"))
    design = np.column_stack((source_array, np.ones(source_array.shape[0])))
    solution, _residuals, rank, _singular = np.linalg.lstsq(design, destination_array, rcond=None)
    reject_if(rank < 3, TapCalibrationError("校准点共线，无法拟合二维纠偏。"))
    transform = Affine2D((tuple((float(value) for value in solution[:, 0])),
        tuple((float(value) for value in solution[:, 1]))))
    predicted = design @ solution
    errors = destination_array - predicted
    return transform, errors


def _convex_hull(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return the counter-clockwise convex hull without repeating its first point."""

    ordered = sorted(set(points))
    reject_if(len(ordered) < 3, TapCalibrationError("校准覆盖点不足，无法形成安全区域。"))

    def cross(origin: tuple[float, float], first: tuple[float, float], second: tuple[float, float]) -> float:
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (first[1] - origin[1]) * (second[0] - origin[0])

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
    reject_if(len(hull) < 3, TapCalibrationError("校准覆盖点共线，无法形成安全区域。"))
    return hull


def build_coverage(normalized_points: Sequence[tuple[float, float]]) -> dict[str, object]:
    hull = _convex_hull(normalized_points)
    xs = [point[0] for point in hull]
    ys = [point[1] for point in hull]
    bounds = [min(xs), min(ys), max(xs), max(ys)]
    span_x = bounds[2] - bounds[0]
    span_y = bounds[3] - bounds[1]
    sufficient = bool(span_x >= MIN_COVERAGE_SPAN_X and span_y >= MIN_COVERAGE_SPAN_Y)
    return {'kind': 'convex_hull', 'normalized_hull': [[float(x), float(y)] for x, y in hull],
        'normalized_bounds': [float(value) for value in bounds], 'span': [round(float(span_x), 6), round(float(span_y),
        6)], 'sufficient': sufficient}


def _point_in_convex_hull(point: tuple[float, float], hull: Sequence[Sequence[float]], *,
    tolerance: float=0.003) -> bool:
    if len(hull) < 3:
        return False
    direction = 0
    x, y = point
    for (index, first) in enumerate(hull):
        second = hull[(index + 1) % len(hull)]
        cross = (float(second[0]) - float(first[0])) * (y - float(first[1])) - (float(second[1]) -
            float(first[1])) * (x - float(first[0]))
        if abs(cross) <= tolerance:
            continue
        current = 1 if cross > 0 else -1
        if direction and current != direction:
            return False
        direction = current
    return True


def build_calibration(samples: Sequence[dict[str, object]], frame_size: tuple[int, int]) -> dict[str, object]:
    """Build target->command correction from browser touch observations.

    Each sample contains the camera-space target/command point and the target
    and observed browser coordinates.  The browser plane is used only as an
    independent touch sensor; the saved correction remains in camera-normalized
    coordinates, which is what RobotController consumes.
    """

    reject_if(len(samples) < 6, TapCalibrationError("多位置校准至少需要6个有效触点。"))
    width, height = frame_size
    reject_if(width < 2 or height < 2, TapCalibrationError("相机画面尺寸无效。"))

    desired_frame: list[tuple[float, float]] = []
    command_frame = []
    target_dom = []
    actual_dom = []
    for sample in samples:
        desired = sample["desired_frame"]
        command = sample.get("command_frame", desired)
        target = sample["target_dom"]
        actual = sample["actual_dom"]
        desired_frame.append((float(desired[0]) / (width - 1), float(desired[1]) / (height - 1)))
        command_frame.append((float(command[0]) / (width - 1), float(command[1]) / (height - 1)))
        target_dom.append((float(target[0]), float(target[1])))
        actual_dom.append((float(actual[0]), float(actual[1])))

    frame_to_dom, projection_errors = fit_affine(desired_frame, target_dom)
    dom_to_frame = frame_to_dom.inverse()
    observed_frame = [dom_to_frame.apply(x, y) for x, y in actual_dom]
    command_to_observed, physical_errors = fit_affine(command_frame, observed_frame)
    target_to_command = command_to_observed.inverse()

    physical_pixel_errors = np.column_stack((physical_errors[:, 0] * (width - 1), physical_errors[:, 1] * (height - 1)))
    distances = np.linalg.norm(physical_pixel_errors, axis=1)
    projection_scale = np.array([width - 1, height - 1], dtype=float)
    projection_distances = np.linalg.norm(projection_errors * projection_scale, axis=1)
    rms = float(math.sqrt(float(np.mean(distances**2))))
    maximum = float(np.max(distances))
    coverage = build_coverage(desired_frame)

    # A noisy/unstable page must never silently become an active correction.
    # A precise fit over only the middle of the screen is also unsafe because
    # small edge controls would otherwise rely on unmeasured extrapolation.
    accepted = bool(rms <= 12.0 and maximum <= 25.0 and coverage['sufficient'])
    return {'version': CALIBRATION_VERSION, 'enabled': False, 'accepted_fit': accepted, 'validated': False,
        'created_at': datetime.now(timezone.utc).isoformat(), 'frame_size': [width, height],
        'sample_count': len(samples), 'target_to_command': target_to_command.to_json(),
        'frame_to_dom': frame_to_dom.to_json(), 'fit': {'rms_error_px': round(rms, 4), 'max_error_px': round(maximum,
        4), 'projection_rms_px': round(float(math.sqrt(float(np.mean(projection_distances ** 2)))), 4)},
        'coverage': coverage, 'samples': list(samples)}


def save_calibration(payload: dict[str, object], path: Path=CALIBRATION_PATH) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_active_calibration(frame_size: tuple[int, int], path: Path=CALIBRATION_PATH) -> Affine2D | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not payload.get('enabled') or not payload.get('validated'):
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
        scale_delta = abs(width_scale - height_scale) / max(width_scale, height_scale)
        if scale_delta > 0.02:
            return None
        return Affine2D.from_json(payload["target_to_command"])
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError, TapCalibrationError):
        return None


def corrected_grid_point(x: int, y: int, frame_size: tuple[int, int], path: Path=CALIBRATION_PATH) -> tuple[int, int]:
    calibration = load_active_calibration(frame_size, path)
    if calibration is None:
        return x, y
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        reject_if(int(payload.get('version', 0)) < CALIBRATION_VERSION, TapCalibrationError("当前触控标定缺少屏幕覆盖边界，必须重新标定。"))
        _validated_hull(payload.get("coverage"), label="当前触控标定")
    except TapCalibrationError:
        raise
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise TapCalibrationError("无法验证触控标定覆盖边界，已拒绝点击。") from exc
    corrected_x, corrected_y = calibration.apply(x / 1000.0, y / 1000.0)
    reject_if(not (-0.08 <= corrected_x <= 1.08 and -0.08 <= corrected_y <= 1.08), TapCalibrationError("纠偏结果超出安全边界，已拒绝点击。"))
    return (min(1000, max(0, int(round(corrected_x * 1000)))), min(1000, max(0, int(round(corrected_y * 1000)))))


def _validated_hull(value: object, *, label: str) -> list[tuple[float, float]]:
    reject_if(not isinstance(value, dict) or value.get('sufficient') is not True, TapCalibrationError(f"{label}缺少足够的验证覆盖范围。"))
    raw_hull = value.get("normalized_hull")
    reject_if(not isinstance(raw_hull, list) or len(raw_hull) < 3, TapCalibrationError(f"{label}缺少有效凸包。"))
    try:
        hull = [(float(point[0]), float(point[1])) for point in raw_hull if isinstance(point, (list,
            tuple)) and len(point) == 2]
    except (TypeError, ValueError) as exc:
        raise TapCalibrationError(f"{label}凸包包含非法坐标。") from exc
    reject_if(
        len(hull) != len(raw_hull) or any((not math.isfinite(value) or not 0.0 <= value <= 1.0 for point
        in hull for value in point)),
        TapCalibrationError(f"{label}凸包包含越界坐标。"),
    )
    _convex_hull(hull)
    return hull


def _vertical_polygon_slice(polygon: Sequence[tuple[float, float]], x: float) -> tuple[float, float]:
    intersections: list[float] = []
    tolerance = 1e-7
    for (index, first) in enumerate(polygon):
        second = polygon[(index + 1) % len(polygon)]
        x1, y1 = first
        x2, y2 = second
        if abs(x2 - x1) <= tolerance:
            if abs(x - x1) <= tolerance:
                intersections.extend((y1, y2))
            continue
        if min(x1, x2) - tolerance <= x <= max(x1, x2) + tolerance:
            ratio = (x - x1) / (x2 - x1)
            if -tolerance <= ratio <= 1.0 + tolerance:
                intersections.append(y1 + (y2 - y1) * ratio)
    reject_if(len(intersections) < 2, TapCalibrationError("验证凸包没有覆盖DOM底边中线。"))
    return min(intersections), max(intersections)


def reveal_system_navigation_path(frame_size: tuple[int, int], path: Path=CALIBRATION_PATH) -> dict[str, object]:
    """Derive one calibrated bottom-edge inward gesture without model coordinates."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise TapCalibrationError("无法读取系统导航唤出所需的触控标定。") from exc
    reject_if(not isinstance(payload, dict), TapCalibrationError("触控标定不是JSON对象。"))
    reject_if(
        int(payload.get('version', 0)) < CALIBRATION_VERSION or payload.get('enabled') is not True
        or payload.get('accepted_fit') is not True or (payload.get('validated') is not True),
        TapCalibrationError("系统导航唤出要求已启用且独立验证通过的v2标定。"),
    )
    validation = payload.get("validation")
    reject_if(
        not isinstance(validation, dict) or validation.get('passed') is not True or validation.get('coverage_passed')
        is not True,
        TapCalibrationError("系统导航唤出缺少独立九点验证证据。"),
    )

    stored_size = payload.get("frame_size")
    reject_if(not isinstance(stored_size, list) or len(stored_size) != 2, TapCalibrationError("触控标定缺少有效相机尺寸。"))
    try:
        stored_width, stored_height = (float(value) for value in stored_size)
        current_width, current_height = (float(value) for value in frame_size)
    except (TypeError, ValueError) as exc:
        raise TapCalibrationError("触控标定相机尺寸非法。") from exc
    reject_if(min(stored_width, stored_height, current_width, current_height) <= 1, TapCalibrationError("触控标定相机尺寸无效。"))
    width_scale = current_width / stored_width
    height_scale = current_height / stored_height
    scale_delta = abs(width_scale - height_scale) / max(width_scale, height_scale)
    reject_if(scale_delta > 0.02, TapCalibrationError("当前相机画面相对验证标定发生非等比漂移，保持0动作。"))

    try:
        frame_to_dom = Affine2D.from_json(payload["frame_to_dom"])
        dom_to_frame = frame_to_dom.inverse()
        target_to_command = Affine2D.from_json(payload["target_to_command"])
    except (KeyError, TypeError, ValueError, TapCalibrationError) as exc:
        raise TapCalibrationError("系统导航唤出缺少可信坐标变换。") from exc
    collection_hull = _validated_hull(payload.get("coverage"), label="采集标定")
    validation_hull = _validated_hull(validation.get("coverage"), label="独立验证")

    samples = payload.get("samples")
    reject_if(not isinstance(samples, list) or len(samples) < 6, TapCalibrationError("frame_to_dom缺少足够的原始标定样本。"))
    sample_points: list[tuple[float, float]] = []
    projection_errors: list[float] = []
    try:
        for sample in samples:
            if not isinstance(sample, dict):
                raise TypeError
            desired = sample["desired_frame"]
            target = sample["target_dom"]
            frame_point = (float(desired[0]) / (stored_width - 1), float(desired[1]) / (stored_height - 1))
            dom_point = (float(target[0]), float(target[1]))
            if any((not math.isfinite(value) for value in (*frame_point, *dom_point))):
                raise ValueError
            sample_points.append(frame_point)
            projection_errors.append(math.dist(frame_to_dom.apply(*frame_point), dom_point))
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise TapCalibrationError("frame_to_dom原始样本损坏。") from exc
    reject_if(max(projection_errors) > SYSTEM_NAVIGATION_MAX_FRAME_TO_DOM_ERROR, TapCalibrationError("frame_to_dom与原始标定样本漂移，保持0动作。"))
    rebuilt_coverage = build_coverage(sample_points)
    rebuilt_hull = _validated_hull(rebuilt_coverage, label="重建标定")
    reject_if(
        any((not _point_in_convex_hull(point, rebuilt_hull) for point in collection_hull))
        or any((not _point_in_convex_hull(point, collection_hull) for point in rebuilt_hull)),
        TapCalibrationError("保存的覆盖凸包与原始样本不一致，保持0动作。"),
    )

    collection_dom_hull = [frame_to_dom.apply(*point) for point in collection_hull]
    validation_dom_hull = [frame_to_dom.apply(*point) for point in validation_hull]
    collection_slice = _vertical_polygon_slice(collection_dom_hull, SYSTEM_NAVIGATION_DOM_CENTER_X)
    validation_slice = _vertical_polygon_slice(validation_dom_hull, SYSTEM_NAVIGATION_DOM_CENTER_X)
    top = max(collection_slice[0], validation_slice[0])
    bottom = min(collection_slice[1], validation_slice[1])
    reject_if(bottom < SYSTEM_NAVIGATION_MIN_BOTTOM_Y, TapCalibrationError("验证凸包没有覆盖DOM底边中点，保持0动作。"))

    requested_grid: list[tuple[int, int]] | None = None
    dom_path: list[tuple[float, float]] | None = None
    for inset_index in range(0, 21):
        start_y = min(0.995, bottom) - inset_index * 0.001
        end_y = start_y - SYSTEM_NAVIGATION_DOM_INWARD_DISTANCE
        if end_y <= top:
            break
        candidate_dom = [(SYSTEM_NAVIGATION_DOM_CENTER_X, start_y), (SYSTEM_NAVIGATION_DOM_CENTER_X, end_y)]
        candidate_grid = [tuple((int(round(value * 1000)) for value in dom_to_frame.apply(*point))) for point
            in candidate_dom]
        normalized = [(point[0] / 1000.0, point[1] / 1000.0) for point in candidate_grid]
        if (all((0 <= value <= 1000 for point in candidate_grid for value in point))
            and all((_point_in_convex_hull(point, collection_hull) and _point_in_convex_hull(point,
            validation_hull) for point in normalized))):
            requested_grid = candidate_grid
            dom_path = [frame_to_dom.apply(*point) for point in normalized]
            break
    reject_if(requested_grid is None or dom_path is None, TapCalibrationError("无法在采集与验证凸包内形成系统边缘轨迹，保持0动作。"))
    reject_if(
        dom_path[0][1] < SYSTEM_NAVIGATION_MIN_BOTTOM_Y or dom_path[0][1] - dom_path[1][1] < 0.2
        or any((abs(point[0] - SYSTEM_NAVIGATION_DOM_CENTER_X) > 0.02 for point in dom_path)),
        TapCalibrationError("本地推导的系统边缘轨迹语义不可信，保持0动作。"),
    )

    corrected_grid: list[tuple[int, int]] = []
    for point in requested_grid:
        corrected = target_to_command.apply(point[0] / 1000.0, point[1] / 1000.0)
        reject_if(any((not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in corrected)), TapCalibrationError("系统边缘轨迹纠偏结果越界，保持0动作。"))
        corrected_grid.append((int(round(corrected[0] * 1000)), int(round(corrected[1] * 1000))))

    return {'action': 'reveal_system_navigation', 'edge': 'bottom', 'frame_size': [int(current_width),
        int(current_height)], 'stored_frame_size': [int(stored_width), int(stored_height)], 'dom_path': [[float(x),
        float(y)] for x, y in dom_path], 'requested_grid': [list(point) for point in requested_grid],
        'corrected_grid': [list(point) for point in corrected_grid], 'collection_coverage': payload['coverage'],
        'validation_coverage': validation['coverage'], 'calibration_version': int(payload['version']),
        'calibration_created_at': payload.get('created_at'), 'calibration_validated_at': payload.get('validated_at')}
