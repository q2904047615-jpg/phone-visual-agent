"""Touch calibration: an affine correction from camera grid points to arm commands.

The calibration file is produced by ``python -m pva.tools.calibrate``. It stores a
2x3 affine ``target_to_command`` in normalized camera coordinates plus the convex
hull of the measured points. A correction is only applied when the file is
``enabled`` and ``validated``; otherwise grid points pass through unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from pva.util import read_json, reject_if, write_json

CALIBRATION_VERSION = 2
MIN_COVERAGE_SPAN_X = 0.68
MIN_COVERAGE_SPAN_Y = 0.82


class CalibrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Affine2D:
    rows: tuple[tuple[float, float, float], tuple[float, float, float]]

    def apply(self, x: float, y: float) -> tuple[float, float]:
        first, second = self.rows
        return first[0] * x + first[1] * y + first[2], second[0] * x + second[1] * y + second[2]

    def inverse(self) -> "Affine2D":
        matrix = np.array([[*self.rows[0]], [*self.rows[1]], [0.0, 0.0, 1.0]], dtype=float)
        reject_if(abs(float(np.linalg.det(matrix))) < 1e-7, CalibrationError("标定矩阵接近奇异，无法求逆。"))
        inverse = np.linalg.inv(matrix)
        return Affine2D((tuple(float(v) for v in inverse[0, :3]), tuple(float(v) for v in inverse[1, :3])))

    def to_json(self) -> list[list[float]]:
        return [list(self.rows[0]), list(self.rows[1])]

    @classmethod
    def from_json(cls, value: Sequence[Sequence[float]]) -> "Affine2D":
        reject_if(len(value) != 2 or any(len(row) != 3 for row in value), CalibrationError("仿射矩阵必须是 2x3。"))
        return cls((tuple(float(v) for v in value[0]), tuple(float(v) for v in value[1])))


def fit_affine(source: Iterable[tuple[float, float]],
               destination: Iterable[tuple[float, float]]) -> tuple[Affine2D, np.ndarray]:
    src = np.asarray(list(source), dtype=float)
    dst = np.asarray(list(destination), dtype=float)
    reject_if(src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2, CalibrationError("标定点格式错误。"))
    reject_if(src.shape[0] < 3, CalibrationError("至少需要 3 个不共线的标定点。"))
    design = np.column_stack((src, np.ones(src.shape[0])))
    solution, _residuals, rank, _singular = np.linalg.lstsq(design, dst, rcond=None)
    reject_if(rank < 3, CalibrationError("标定点共线，无法拟合。"))
    transform = Affine2D((tuple(float(v) for v in solution[:, 0]), tuple(float(v) for v in solution[:, 1])))
    return transform, dst - design @ solution


def convex_hull(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    ordered = sorted(set(points))
    reject_if(len(ordered) < 3, CalibrationError("标定覆盖点不足。"))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

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
    reject_if(len(hull) < 3, CalibrationError("标定覆盖点共线。"))
    return hull


def build_coverage(normalized_points: Sequence[tuple[float, float]]) -> dict[str, object]:
    hull = convex_hull(normalized_points)
    xs = [p[0] for p in hull]
    ys = [p[1] for p in hull]
    span_x, span_y = max(xs) - min(xs), max(ys) - min(ys)
    return {
        "kind": "convex_hull",
        "normalized_hull": [[float(x), float(y)] for x, y in hull],
        "normalized_bounds": [min(xs), min(ys), max(xs), max(ys)],
        "span": [round(span_x, 6), round(span_y, 6)],
        "sufficient": bool(span_x >= MIN_COVERAGE_SPAN_X and span_y >= MIN_COVERAGE_SPAN_Y),
    }


def build_calibration(samples: Sequence[dict[str, object]], frame_size: tuple[int, int]) -> dict[str, object]:
    """Fit the camera-grid -> arm-command correction from measured touches.

    Each sample has ``desired_frame`` (camera pixel the arm was told to hit),
    optional ``command_frame`` (pixel actually commanded), ``target_dom`` and
    ``actual_dom`` (normalized touch positions reported by the calibration page).
    """
    reject_if(len(samples) < 6, CalibrationError("至少需要 6 个有效触点。"))
    width, height = frame_size
    desired, command, target, actual = [], [], [], []
    for sample in samples:
        d = sample["desired_frame"]
        c = sample.get("command_frame", d)
        desired.append((float(d[0]) / (width - 1), float(d[1]) / (height - 1)))
        command.append((float(c[0]) / (width - 1), float(c[1]) / (height - 1)))
        target.append((float(sample["target_dom"][0]), float(sample["target_dom"][1])))
        actual.append((float(sample["actual_dom"][0]), float(sample["actual_dom"][1])))
    frame_to_dom, projection_errors = fit_affine(desired, target)
    dom_to_frame = frame_to_dom.inverse()
    observed = [dom_to_frame.apply(x, y) for x, y in actual]
    command_to_observed, physical_errors = fit_affine(command, observed)
    target_to_command = command_to_observed.inverse()
    pixel_errors = np.column_stack((physical_errors[:, 0] * (width - 1), physical_errors[:, 1] * (height - 1)))
    distances = np.linalg.norm(pixel_errors, axis=1)
    projection = np.linalg.norm(projection_errors * np.array([width - 1, height - 1], dtype=float), axis=1)
    rms = float(math.sqrt(float(np.mean(distances ** 2))))
    maximum = float(np.max(distances))
    coverage = build_coverage(desired)
    return {
        "version": CALIBRATION_VERSION,
        "enabled": False,
        "accepted_fit": bool(rms <= 12.0 and maximum <= 25.0 and coverage["sufficient"]),
        "validated": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "frame_size": [width, height],
        "sample_count": len(samples),
        "target_to_command": target_to_command.to_json(),
        "frame_to_dom": frame_to_dom.to_json(),
        "fit": {"rms_error_px": round(rms, 4), "max_error_px": round(maximum, 4),
                "projection_rms_px": round(float(math.sqrt(float(np.mean(projection ** 2)))), 4)},
        "coverage": coverage,
        "samples": list(samples),
    }


def save_calibration(payload: dict[str, object], path: Path) -> None:
    write_json(path, payload)


class Calibration:
    """Loaded correction for one device; identity when absent or not validated."""

    def __init__(self, path: Path | None):
        self.path = path
        self.payload = read_json(path, default=None) if path else None
        self.transform: Affine2D | None = None
        self.error: str | None = None
        if not isinstance(self.payload, dict):
            return
        if not (self.payload.get("enabled") and self.payload.get("validated")):
            return
        try:
            reject_if(int(self.payload.get("version", 0)) < CALIBRATION_VERSION,
                      CalibrationError("标定文件版本过旧，需要重新标定。"))
            coverage = self.payload.get("coverage")
            reject_if(not isinstance(coverage, dict) or coverage.get("sufficient") is not True,
                      CalibrationError("标定覆盖范围不足。"))
            self.transform = Affine2D.from_json(self.payload["target_to_command"])
        except (CalibrationError, KeyError, TypeError, ValueError) as exc:
            self.error = str(exc)

    @property
    def active(self) -> bool:
        return self.transform is not None

    def status(self) -> dict[str, object]:
        fit = (self.payload or {}).get("fit") if isinstance(self.payload, dict) else None
        return {"active": self.active, "path": str(self.path) if self.path else None,
                "error": self.error, "fit": fit}

    def correct(self, x: int, y: int, frame_size: tuple[int, int]) -> tuple[int, int]:
        """Map a 0..1000 grid point to the corrected 0..1000 command point."""
        if self.transform is None:
            return x, y
        stored = (self.payload or {}).get("frame_size") or frame_size
        sw, sh = float(stored[0]), float(stored[1])
        cw, ch = float(frame_size[0]), float(frame_size[1])
        # The affine works in normalized coordinates, so an isotropic resize is fine;
        # a non-uniform change means the camera geometry changed.
        if abs(cw / sw - ch / sh) / max(cw / sw, ch / sh) > 0.02:
            raise CalibrationError("相机画面比例与标定时不同，请重新标定。")
        cx, cy = self.transform.apply(x / 1000.0, y / 1000.0)
        reject_if(not (-0.08 <= cx <= 1.08 and -0.08 <= cy <= 1.08),
                  CalibrationError("纠偏后的坐标超出安全范围。"))
        return min(1000, max(0, round(cx * 1000))), min(1000, max(0, round(cy * 1000)))
