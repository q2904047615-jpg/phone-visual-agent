from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import run_xy_calibration
from device_exclusivity import InterProcessLease
from tap_calibration import (
    Affine2D,
    TapCalibrationError,
    build_calibration,
    corrected_grid_point,
    fit_affine,
    resolve_target_grid_point_within_calibration,
    reveal_system_navigation_path,
)
from run_xy_calibration import (
    calibration_page_ready,
    locate_magenta_target,
    probe_single_touch,
    run_calibration_step,
    wait_for_page_state,
)
from universal_agent_orchestrator import DeviceTaskRegistry


def fresh_page_state(
    phase: str,
    *,
    sequence: int,
    offset_seconds: float = 0,
) -> dict[str, object]:
    timestamp = datetime.now(timezone.utc) + timedelta(seconds=offset_seconds)
    return {
        "phase": phase,
        "sequence": sequence,
        "fullscreen": phase != "fullscreen_setup",
        "calibration_mode": (
            "setup" if phase == "fullscreen_setup" else "fullscreen"
        ),
        "fullscreen_attempted": phase != "fullscreen_setup",
        "viewport_coverage": {},
        "updated_at": timestamp.isoformat(),
    }


def ready_device_payload(*, busy: bool = False) -> dict[str, object]:
    return {
        "default_device_id": "device-local-01",
        "devices": [
            {
                "device_id": "device-local-01",
                "controller_online": True,
                "camera_online": True,
                "busy": busy,
            }
        ],
    }


def complete_nine_samples() -> list[dict[str, object]]:
    width, height = 501, 901
    points = [
        (0.05, 0.05),
        (0.50, 0.05),
        (0.95, 0.05),
        (0.05, 0.50),
        (0.50, 0.50),
        (0.95, 0.50),
        (0.05, 0.95),
        (0.50, 0.95),
        (0.95, 0.95),
    ]
    return [
        {
            "sequence": index,
            "desired_frame": [x * (width - 1), y * (height - 1)],
            "command_frame": [x * (width - 1), y * (height - 1)],
            "target_dom": [x, y],
            "actual_dom": [x, y],
            "viewport_size": [width, height],
        }
        for index, (x, y) in enumerate(points)
    ]


@contextmanager
def fake_device_reservation(**_kwargs):
    yield "device-local-01", "device-local-01", "calibration-test-session"


@contextmanager
def fake_physical_lease(**_kwargs):
    yield


class FakeProbeRobot:
    def __init__(self, frame: Image.Image) -> None:
        self.frame = frame
        self.tap_calls: list[tuple[int, int]] = []

    def vision_tap_relative(self, x: int, y: int) -> tuple[int, int]:
        self.tap_calls.append((x, y))
        return (111, 222)

    def vision_capture(self) -> Image.Image:
        return self.frame.copy()


class FakeFullscreenRobot:
    def __init__(self, frame: Image.Image) -> None:
        self.frame = frame

    def vision_capture(self) -> Image.Image:
        return self.frame.copy()


class TapCalibrationMathTests(unittest.TestCase):
    def test_calibration_page_ready_accepts_only_bounded_modes(self):
        self.assertTrue(
            calibration_page_ready(
                {"calibration_mode": "fullscreen", "fullscreen": True}
            )
        )
        fallback = {
            "calibration_mode": "viewport_coverage",
            "fullscreen": False,
            "fullscreen_attempted": True,
            "viewport_coverage": {
                "eligible": True,
                "width_ratio": 0.94,
                "height_ratio": 0.93,
            },
        }
        self.assertTrue(calibration_page_ready(fallback))
        self.assertFalse(
            calibration_page_ready(
                {
                    **fallback,
                    "viewport_coverage": {
                        "eligible": True,
                        "width_ratio": 0.94,
                        "height_ratio": 0.91,
                    },
                }
            )
        )
        self.assertFalse(
            calibration_page_ready(
                {"calibration_mode": "blocked", "fullscreen": False}
            )
        )
        self.assertFalse(
            calibration_page_ready(
                {
                    **fallback,
                    "viewport_coverage": {
                        "eligible": True,
                        "width_ratio": "invalid",
                        "height_ratio": 0.94,
                    },
                }
            )
        )

    def test_magenta_target_is_located(self):
        from PIL import Image, ImageDraw

        image = Image.new("RGB", (300, 500), "black")
        draw = ImageDraw.Draw(image)
        draw.ellipse((118, 208, 162, 252), outline="#ff00e6", width=7)
        draw.line((110, 230, 170, 230), fill="#ff00e6", width=4)
        draw.line((140, 200, 140, 260), fill="#ff00e6", width=4)
        x, y, _box = locate_magenta_target(image)
        self.assertLessEqual(abs(x - 140), 2)
        self.assertLessEqual(abs(y - 230), 2)

    def test_dim_magenta_target_clipped_by_camera_edge_is_located(self):
        from PIL import Image, ImageDraw

        image = Image.new("RGB", (300, 500), "black")
        draw = ImageDraw.Draw(image)
        dim_magenta = (125, 12, 122)
        draw.ellipse((118, -14, 162, 30), outline=dim_magenta, width=7)
        draw.line((110, 8, 170, 8), fill=dim_magenta, width=4)
        draw.line((140, 0, 140, 38), fill=dim_magenta, width=4)

        x, y, box = locate_magenta_target(image)

        self.assertLessEqual(abs(x - 140), 3)
        self.assertLess(y, 24)
        self.assertEqual(0, box[1])

    def test_fit_known_affine_and_inverse(self):
        source = [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9), (0.5, 0.5)]
        expected = Affine2D(((1.02, 0.03, -0.01), (-0.02, 0.98, 0.025)))
        destination = [expected.apply(*point) for point in source]
        fitted, errors = fit_affine(source, destination)
        self.assertLess(float(abs(errors).max()), 1e-10)
        for point in source:
            actual = fitted.inverse().apply(*expected.apply(*point))
            self.assertAlmostEqual(actual[0], point[0], places=7)
            self.assertAlmostEqual(actual[1], point[1], places=7)

    def test_fit_rejects_collinear_points(self):
        with self.assertRaises(TapCalibrationError):
            fit_affine([(0, 0), (0.5, 0.5), (1, 1)], [(0, 0), (0.5, 0.5), (1, 1)])

    def test_inactive_file_does_not_change_point(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tap.json"
            path.write_text(json.dumps({"enabled": False}), encoding="utf-8")
            self.assertEqual(corrected_grid_point(321, 654, (540, 960), path), (321, 654))

    def test_active_correction_is_applied(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tap.json"
            path.write_text(
                json.dumps(
                    {
                        "enabled": True,
                        "validated": True,
                        "version": 2,
                        "frame_size": [540, 960],
                        "target_to_command": [[1.0, 0.0, -0.01], [0.0, 1.0, 0.02]],
                        "coverage": {
                            "sufficient": True,
                            "normalized_hull": [[0.05, 0.05], [0.95, 0.05], [0.95, 0.95], [0.05, 0.95]],
                            "normalized_bounds": [0.05, 0.05, 0.95, 0.95],
                        },
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(corrected_grid_point(500, 500, (540, 960), path), (490, 520))

    def test_validated_normalized_correction_survives_isotropic_dpi_resize(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tap.json"
            path.write_text(
                json.dumps(
                    {
                        "enabled": True,
                        "validated": True,
                        "version": 2,
                        "frame_size": [540, 960],
                        "target_to_command": [
                            [1.0, 0.0, -0.01],
                            [0.0, 1.0, 0.02],
                        ],
                        "coverage": {
                            "sufficient": True,
                            "normalized_hull": [[0.05, 0.05], [0.95, 0.05], [0.95, 0.95], [0.05, 0.95]],
                            "normalized_bounds": [0.05, 0.05, 0.95, 0.95],
                        },
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                corrected_grid_point(500, 500, (810, 1440), path),
                (490, 520),
            )

    def test_validated_correction_rejects_non_uniform_resize(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tap.json"
            path.write_text(
                json.dumps(
                    {
                        "enabled": True,
                        "validated": True,
                        "version": 2,
                        "frame_size": [540, 960],
                        "target_to_command": [
                            [1.0, 0.0, -0.01],
                            [0.0, 1.0, 0.02],
                        ],
                        "coverage": {
                            "sufficient": True,
                            "normalized_hull": [[0.05, 0.05], [0.95, 0.05], [0.95, 0.95], [0.05, 0.95]],
                            "normalized_bounds": [0.05, 0.05, 0.95, 0.95],
                        },
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                corrected_grid_point(500, 500, (810, 1300), path),
                (500, 500),
            )

    def test_build_calibration_recovers_translation(self):
        width, height = 501, 901
        points = [(0.05, 0.05), (0.5, 0.05), (0.95, 0.05), (0.05, 0.5), (0.5, 0.5), (0.95, 0.5), (0.05, 0.95), (0.5, 0.95), (0.95, 0.95)]
        samples = []
        for index, (x, y) in enumerate(points):
            samples.append(
                {
                    "sequence": index,
                    "desired_frame": [x * (width - 1), y * (height - 1)],
                    "command_frame": [x * (width - 1), y * (height - 1)],
                    "target_dom": [x, y],
                    "actual_dom": [x + 0.02, y - 0.01],
                }
            )
        payload = build_calibration(samples, (width, height))
        correction = Affine2D.from_json(payload["target_to_command"])
        corrected = correction.apply(0.5, 0.5)
        self.assertAlmostEqual(corrected[0], 0.48, places=6)
        self.assertAlmostEqual(corrected[1], 0.51, places=6)
        self.assertTrue(payload["accepted_fit"])
        self.assertTrue(payload["coverage"]["sufficient"])
        self.assertFalse(payload["enabled"])

    def test_middle_only_fit_is_rejected_even_when_residual_is_zero(self):
        width, height = 501, 901
        points = [(0.25, 0.25), (0.5, 0.25), (0.75, 0.25), (0.25, 0.5), (0.5, 0.5), (0.75, 0.5), (0.25, 0.75), (0.5, 0.75), (0.75, 0.75)]
        samples = [
            {
                "sequence": index,
                "desired_frame": [x * (width - 1), y * (height - 1)],
                "command_frame": [x * (width - 1), y * (height - 1)],
                "target_dom": [x, y],
                "actual_dom": [x, y],
            }
            for index, (x, y) in enumerate(points)
        ]
        payload = build_calibration(samples, (width, height))
        self.assertFalse(payload["coverage"]["sufficient"])
        self.assertFalse(payload["accepted_fit"])

    def test_active_correction_rejects_target_outside_measured_hull(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tap.json"
            path.write_text(
                json.dumps(
                    {
                        "version": 2,
                        "enabled": True,
                        "validated": True,
                        "frame_size": [540, 960],
                        "target_to_command": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                        "coverage": {
                            "sufficient": True,
                            "normalized_hull": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.8], [0.1, 0.8]],
                            "normalized_bounds": [0.1, 0.1, 0.9, 0.8],
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(TapCalibrationError, "实测标定区域之外"):
                corrected_grid_point(735, 910, (540, 960), path)

    def test_dual_audited_edge_target_resolves_inside_measured_hull(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tap.json"
            path.write_text(
                json.dumps(
                    {
                        "version": 2,
                        "enabled": True,
                        "validated": True,
                        "frame_size": [810, 1440],
                        "target_to_command": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                        "coverage": {
                            "sufficient": True,
                            "normalized_hull": [
                                [0.11742892459826947, 0.48922863099374564],
                                [0.12855377008652658, 0.020152883947185545],
                                [0.5018541409147095, 0.017373175816539264],
                                [0.8825710754017305, 0.014593467685892982],
                                [0.8825710754017305, 0.9652536483669215],
                                [0.5006180469715699, 0.9645587213342599],
                                [0.1211372064276885, 0.9631688672689368],
                            ],
                            "normalized_bounds": [
                                0.11742892459826947,
                                0.014593467685892982,
                                0.8825710754017305,
                                0.9652536483669215,
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )

            point = resolve_target_grid_point_within_calibration(
                895,
                829,
                (0.802, 0.797861, 0.988, 0.859861),
                (810, 1440),
                path,
            )

            self.assertEqual((842, 829), point)
            self.assertEqual(point, corrected_grid_point(*point, (810, 1440), path))

    def test_calibrated_target_center_already_inside_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tap.json"
            path.write_text(
                json.dumps(
                    {
                        "version": 2,
                        "enabled": True,
                        "validated": True,
                        "frame_size": [540, 960],
                        "target_to_command": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                        "coverage": {
                            "sufficient": True,
                            "normalized_hull": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
                            "normalized_bounds": [0.1, 0.1, 0.9, 0.9],
                        },
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                (500, 500),
                resolve_target_grid_point_within_calibration(
                    500, 500, (0.45, 0.45, 0.55, 0.55), (540, 960), path
                ),
            )

    def test_dual_audited_wide_edge_target_uses_absolute_safe_intersection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tap.json"
            hull = [
                [0.11742892459826947, 0.48922863099374564],
                [0.12855377008652658, 0.020152883947185545],
                [0.5018541409147095, 0.017373175816539264],
                [0.8825710754017305, 0.014593467685892982],
                [0.8825710754017305, 0.9652536483669215],
                [0.5006180469715699, 0.9645587213342599],
                [0.1211372064276885, 0.9631688672689368],
            ]
            path.write_text(
                json.dumps(
                    {
                        "version": 2,
                        "enabled": True,
                        "validated": True,
                        "frame_size": [810, 1440],
                        "target_to_command": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                        "coverage": {
                            "sufficient": True,
                            "normalized_hull": hull,
                            "normalized_bounds": [
                                0.11742892459826947,
                                0.014593467685892982,
                                0.8825710754017305,
                                0.9652536483669215,
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )

            left_point = resolve_target_grid_point_within_calibration(
                99,
                941,
                (0.009, 0.9131388888888889, 0.189, 0.9691388888888889),
                (810, 1440),
                path,
            )
            right_point = resolve_target_grid_point_within_calibration(
                901,
                941,
                (0.811, 0.9131388888888889, 0.991, 0.9691388888888889),
                (810, 1440),
                path,
            )

            self.assertEqual((155, 938), left_point)
            self.assertEqual((847, 939), right_point)
            for point, bounds in (
                (left_point, (0.009, 0.9131388888888889, 0.189, 0.9691388888888889)),
                (right_point, (0.811, 0.9131388888888889, 0.991, 0.9691388888888889)),
            ):
                self.assertGreater(point[0] / 1000.0, bounds[0] + 0.01)
                self.assertLess(point[0] / 1000.0, bounds[2] - 0.01)
                self.assertGreater(point[1] / 1000.0, bounds[1] + 0.01)
                self.assertLess(point[1] / 1000.0, bounds[3] - 0.01)

    def test_dual_audited_narrow_literal_key_uses_safe_intersection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tap.json"
            hull = [
                [0.11742892459826947, 0.48922863099374564],
                [0.12855377008652658, 0.020152883947185545],
                [0.5018541409147095, 0.017373175816539264],
                [0.8825710754017305, 0.014593467685892982],
                [0.8825710754017305, 0.9652536483669215],
                [0.5006180469715699, 0.9645587213342599],
                [0.1211372064276885, 0.9631688672689368],
            ]
            path.write_text(
                json.dumps(
                    {
                        "version": 2,
                        "enabled": True,
                        "validated": True,
                        "frame_size": [810, 1440],
                        "target_to_command": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                        "coverage": {
                            "sufficient": True,
                            "normalized_hull": hull,
                            "normalized_bounds": [
                                0.11742892459826947,
                                0.014593467685892982,
                                0.8825710754017305,
                                0.9652536483669215,
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )
            left_bounds = (
                0.0828,
                0.6812916666666666,
                0.1668,
                0.742111111111111,
            )
            right_bounds = (
                0.8332,
                0.6812916666666666,
                0.9172,
                0.742111111111111,
            )

            left_point = resolve_target_grid_point_within_calibration(
                115,
                712,
                left_bounds,
                (810, 1440),
                path,
            )
            right_point = resolve_target_grid_point_within_calibration(
                885,
                712,
                right_bounds,
                (810, 1440),
                path,
            )

            for point, bounds in (
                (left_point, left_bounds),
                (right_point, right_bounds),
            ):
                self.assertGreater(point[0] / 1000.0, bounds[0] + 0.01)
                self.assertLess(point[0] / 1000.0, bounds[2] - 0.01)
                self.assertGreater(point[1] / 1000.0, bounds[1] + 0.01)
                self.assertLess(point[1] / 1000.0, bounds[3] - 0.01)

    def test_dual_audited_edge_target_rejects_large_but_too_narrow_fragment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tap.json"
            path.write_text(
                json.dumps(
                    {
                        "version": 2,
                        "enabled": True,
                        "validated": True,
                        "frame_size": [540, 960],
                        "target_to_command": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                        "coverage": {
                            "sufficient": True,
                            "normalized_hull": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
                            "normalized_bounds": [0.1, 0.1, 0.9, 0.9],
                        },
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(TapCalibrationError, "二维覆盖不足"):
                resolve_target_grid_point_within_calibration(
                    930,
                    500,
                    (0.87, 0.30, 0.98, 0.70),
                    (540, 960),
                    path,
                )

    def test_calibrated_target_rejects_insufficient_overlap_and_large_shift(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tap.json"
            path.write_text(
                json.dumps(
                    {
                        "version": 2,
                        "enabled": True,
                        "validated": True,
                        "frame_size": [540, 960],
                        "target_to_command": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                        "coverage": {
                            "sufficient": True,
                            "normalized_hull": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
                            "normalized_bounds": [0.1, 0.1, 0.9, 0.9],
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(TapCalibrationError, "二维覆盖不足"):
                resolve_target_grid_point_within_calibration(
                    950, 500, (0.88, 0.45, 0.99, 0.55), (540, 960), path
                )
            with self.assertRaisesRegex(TapCalibrationError, "偏离视觉目标中心过远"):
                resolve_target_grid_point_within_calibration(
                    990, 500, (0.70, 0.45, 1.00, 0.55), (540, 960), path
                )

    def test_legacy_active_calibration_without_coverage_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tap.json"
            path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "enabled": True,
                        "validated": True,
                        "frame_size": [540, 960],
                        "target_to_command": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(TapCalibrationError, "必须重新标定"):
                corrected_grid_point(500, 500, (540, 960), path)

    def test_reveal_system_navigation_path_is_local_calibrated_and_inward(self):
        path = Path(__file__).with_name("tap_calibration.json")
        result = reveal_system_navigation_path((810, 1440), path)

        start, end = result["dom_path"]
        self.assertEqual("bottom", result["edge"])
        self.assertAlmostEqual(0.5, start[0], delta=0.02)
        self.assertAlmostEqual(0.5, end[0], delta=0.02)
        self.assertGreaterEqual(start[1], 0.90)
        self.assertGreaterEqual(start[1] - end[1], 0.20)
        self.assertEqual(2, len(result["requested_grid"]))
        self.assertEqual(2, len(result["corrected_grid"]))

    def test_reveal_system_navigation_path_fails_closed_on_invalid_or_drifted_fit(self):
        source = json.loads(
            Path(__file__).with_name("tap_calibration.json").read_text(
                encoding="utf-8"
            )
        )
        out_of_bounds_hull = json.loads(json.dumps(source))
        out_of_bounds_hull["coverage"]["normalized_hull"][0][0] = -0.001
        cases = (
            ("disabled", {**source, "enabled": False}, (810, 1440)),
            (
                "unvalidated",
                {**source, "validation": {**source["validation"], "passed": False}},
                (810, 1440),
            ),
            (
                "frame_to_dom_drift",
                {**source, "frame_to_dom": [[1, 0, 0], [0, 1, 0]]},
                (810, 1440),
            ),
            ("non_uniform_frame", source, (810, 1515)),
            ("out_of_bounds_hull", out_of_bounds_hull, (810, 1440)),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tap.json"
            for label, payload, frame_size in cases:
                with self.subTest(label=label):
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaises(TapCalibrationError):
                        reveal_system_navigation_path(frame_size, path)

    def test_single_touch_probe_dry_run_never_calls_robot(self):
        frame = Image.new("RGB", (540, 960), "black")
        robot = FakeProbeRobot(frame)
        with patch(
            "run_xy_calibration.wait_for_stable_target",
            return_value=(frame, (120, 240, (100, 220, 140, 260))),
        ):
            result = probe_single_touch(
                base_url="http://127.0.0.1:8770",
                robot=robot,
                execute=False,
            )

        self.assertEqual("awaiting_explicit_execution", result["status"])
        self.assertEqual(0, result["physical_actions"])
        self.assertEqual([], robot.tap_calls)

    def test_fullscreen_setup_executes_only_one_action_and_stops(self):
        frame = Image.new("RGB", (540, 960), "black")
        robot = FakeFullscreenRobot(frame)
        before = fresh_page_state("fullscreen_setup", sequence=0, offset_seconds=-2)
        locked = fresh_page_state("fullscreen_setup", sequence=0, offset_seconds=-1)
        after = fresh_page_state("calibration", sequence=0)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = temporary.name
        with patch(
            "run_xy_calibration._calibration_device_reservation",
            new=fake_device_reservation,
        ), patch(
            "run_xy_calibration._calibration_physical_lease",
            new=fake_physical_lease,
        ), patch(
            "run_xy_calibration.request_json",
            return_value={"session_id": "page-setup", "samples": []},
        ), patch(
            "run_xy_calibration.wait_for_page_state",
            side_effect=[before, locked, after],
        ), patch(
            "run_xy_calibration.wait_for_stable_target",
            return_value=(frame, (270, 480, (250, 460, 290, 500))),
        ), patch(
            "run_xy_calibration.click_raw_pixel",
        ) as click:
            output_root = Path(directory)
            result = run_calibration_step(
                mode="collect",
                base_url="http://127.0.0.1:8770",
                robot=robot,
                execute=True,
                output_root=output_root,
            )
            progress = json.loads(
                (
                    output_root / "collect_page-setup" / "progress.json"
                ).read_text(encoding="utf-8")
            )

        click.assert_called_once()
        self.assertEqual(1, result["physical_actions"])
        self.assertEqual("fullscreen_setup", result["executed_action"])
        self.assertEqual(1, len(progress["attempts"]))
        self.assertEqual([], progress["samples"])

    def test_edge_point_persists_and_next_call_only_previews_next_sequence(self):
        frame = Image.new("RGB", (540, 960), "black")
        robot = FakeFullscreenRobot(frame)
        record = {
            "sequence": 0,
            "target_x": 20.0,
            "target_y": 30.0,
            "actual_x": 21.0,
            "actual_y": 31.0,
            "viewport_width": 400,
            "viewport_height": 800,
        }
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = temporary.name
        with patch(
            "run_xy_calibration._calibration_device_reservation",
            new=fake_device_reservation,
        ), patch(
            "run_xy_calibration._calibration_physical_lease",
            new=fake_physical_lease,
        ), patch(
            "run_xy_calibration.request_json",
            return_value={"session_id": "page-points", "samples": []},
        ), patch(
            "run_xy_calibration.wait_for_page_state",
            side_effect=[
                fresh_page_state("calibration", sequence=0, offset_seconds=-2),
                fresh_page_state("calibration", sequence=0, offset_seconds=-1),
                fresh_page_state("calibration", sequence=1),
            ],
        ), patch(
            "run_xy_calibration.wait_for_stable_target",
            return_value=(frame, (30, 50, (20, 40, 40, 60))),
        ), patch(
            "run_xy_calibration.click_raw_pixel",
        ) as first_click, patch(
            "run_xy_calibration.wait_for_new_sample",
            return_value=record,
        ):
            output_root = Path(directory)
            first = run_calibration_step(
                mode="collect",
                base_url="http://127.0.0.1:8770",
                robot=robot,
                execute=True,
                output_root=output_root,
            )
        with patch(
            "run_xy_calibration.request_json",
            return_value={"session_id": "page-points", "samples": [record]},
        ), patch(
            "run_xy_calibration.wait_for_page_state",
            return_value=fresh_page_state("calibration", sequence=1),
        ), patch(
            "run_xy_calibration.wait_for_stable_target",
            return_value=(frame, (270, 480, (250, 460, 290, 500))),
        ), patch(
            "run_xy_calibration.click_raw_pixel",
        ) as second_click:
            second = run_calibration_step(
                mode="collect",
                base_url="http://127.0.0.1:8770",
                robot=robot,
                execute=False,
                output_root=output_root,
            )

        first_click.assert_called_once()
        second_click.assert_not_called()
        self.assertEqual(1, first["sample_count"])
        self.assertEqual(0, second["physical_actions"])
        self.assertEqual(1, second["sequence"])

    def test_interrupted_point_is_recovered_without_second_action(self):
        frame = Image.new("RGB", (540, 960), "black")
        robot = FakeFullscreenRobot(frame)
        record = {
            "sequence": 0,
            "target_x": 20.0,
            "target_y": 30.0,
            "actual_x": 21.0,
            "actual_y": 31.0,
            "viewport_width": 400,
            "viewport_height": 800,
        }
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = temporary.name
        with patch(
            "run_xy_calibration._calibration_device_reservation",
            new=fake_device_reservation,
        ), patch(
            "run_xy_calibration._calibration_physical_lease",
            new=fake_physical_lease,
        ), patch(
            "run_xy_calibration.request_json",
            return_value={"session_id": "page-recover", "samples": []},
        ), patch(
            "run_xy_calibration.wait_for_page_state",
            side_effect=[
                fresh_page_state("calibration", sequence=0, offset_seconds=-2),
                fresh_page_state("calibration", sequence=0, offset_seconds=-1),
            ],
        ), patch(
            "run_xy_calibration.wait_for_stable_target",
            return_value=(frame, (30, 50, (20, 40, 40, 60))),
        ), patch(
            "run_xy_calibration.click_raw_pixel",
        ) as first_click, patch(
            "run_xy_calibration.wait_for_new_sample",
            side_effect=TapCalibrationError("sample timeout"),
        ):
            output_root = Path(directory)
            with self.assertRaisesRegex(TapCalibrationError, "sample timeout"):
                run_calibration_step(
                    mode="collect",
                    base_url="http://127.0.0.1:8770",
                    robot=robot,
                    execute=True,
                    output_root=output_root,
                )
        with patch(
            "run_xy_calibration.request_json",
            return_value={"session_id": "page-recover", "samples": [record]},
        ), patch(
            "run_xy_calibration.wait_for_page_state",
            return_value=fresh_page_state("calibration", sequence=1),
        ), patch(
            "run_xy_calibration.click_raw_pixel",
        ) as recovery_click:
            recovered = run_calibration_step(
                mode="collect",
                base_url="http://127.0.0.1:8770",
                robot=robot,
                execute=False,
                output_root=output_root,
            )

        first_click.assert_called_once()
        recovery_click.assert_not_called()
        self.assertEqual("recovered", recovered["status"])
        self.assertEqual(0, recovered["physical_actions"])
        self.assertEqual(1, recovered["sample_count"])

    def test_execute_holds_registry_and_calibration_physical_lease_during_click(self):
        frame = Image.new("RGB", (540, 960), "black")
        robot = FakeFullscreenRobot(frame)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lease_dir = root / "leases"

            def request(url, *_args, **_kwargs):
                if url == "http://device.test/api/device":
                    return ready_device_payload()
                if url.endswith("/api/samples"):
                    return {"session_id": "page-positive-lock", "samples": []}
                raise AssertionError(url)

            observed: dict[str, object] = {}

            def inspect_locks(*_args, **_kwargs):
                observed["physical"] = InterProcessLease.active_payload(
                    lease_dir / "physical_hardware_action.lease"
                )
                observed["device_session"] = DeviceTaskRegistry(
                    lease_directory=lease_dir
                ).active_session("device-local-01")

            with patch.object(
                run_xy_calibration,
                "SHARED_DEVICE_LEASE_DIR",
                lease_dir,
            ), patch(
                "run_xy_calibration.request_json",
                side_effect=request,
            ), patch(
                "run_xy_calibration.wait_for_page_state",
                side_effect=[
                    fresh_page_state(
                        "fullscreen_setup", sequence=0, offset_seconds=-2
                    ),
                    fresh_page_state(
                        "fullscreen_setup", sequence=0, offset_seconds=-1
                    ),
                    fresh_page_state("calibration", sequence=0),
                ],
            ), patch(
                "run_xy_calibration.wait_for_stable_target",
                return_value=(frame, (270, 480, (250, 460, 290, 500))),
            ), patch(
                "run_xy_calibration.click_raw_pixel",
                side_effect=inspect_locks,
            ) as click:
                result = run_calibration_step(
                    mode="collect",
                    base_url="http://127.0.0.1:8770",
                    robot=robot,
                    execute=True,
                    device_status_url="http://device.test/api/device",
                    output_root=root / "output",
                )

            active_after = DeviceTaskRegistry(
                lease_directory=lease_dir
            ).active_session("device-local-01")
            physical_after = InterProcessLease.active_payload(
                lease_dir / "physical_hardware_action.lease"
            )

        click.assert_called_once()
        self.assertEqual(1, result["physical_actions"])
        self.assertTrue(observed["physical"]["calibration_step"])
        self.assertEqual("calibration_step", observed["physical"]["purpose"])
        self.assertTrue(str(observed["device_session"]).startswith("calibration-step-"))
        self.assertIsNone(active_after)
        self.assertIsNone(physical_after)

    def test_busy_device_rejects_zero_actions_and_releases_registry(self):
        frame = Image.new("RGB", (540, 960), "black")
        robot = FakeFullscreenRobot(frame)
        with tempfile.TemporaryDirectory() as directory:
            lease_dir = Path(directory) / "leases"
            device_states = iter(
                [ready_device_payload(), ready_device_payload(busy=True)]
            )
            with patch.object(
                run_xy_calibration,
                "SHARED_DEVICE_LEASE_DIR",
                lease_dir,
            ), patch(
                "run_xy_calibration.request_json",
                side_effect=lambda *_args, **_kwargs: next(device_states),
            ), patch(
                "run_xy_calibration.wait_for_stable_target",
            ) as visual, patch(
                "run_xy_calibration.click_raw_pixel",
            ) as click, self.assertRaisesRegex(TapCalibrationError, "非空闲"):
                run_calibration_step(
                    mode="collect",
                    base_url="http://127.0.0.1:8770",
                    robot=robot,
                    execute=True,
                    device_status_url="http://device.test/api/device",
                    output_root=Path(directory) / "output",
                )
            registry = DeviceTaskRegistry(lease_directory=lease_dir)
            registry.reserve("device-local-01", "after-busy-rejection")
            registry.release("device-local-01", "after-busy-rejection")

        visual.assert_not_called()
        click.assert_not_called()

    def test_device_registry_occupancy_rejects_before_visual_or_action(self):
        frame = Image.new("RGB", (540, 960), "black")
        robot = FakeFullscreenRobot(frame)
        with tempfile.TemporaryDirectory() as directory:
            lease_dir = Path(directory) / "leases"
            competitor = DeviceTaskRegistry(lease_directory=lease_dir)
            competitor.reserve("device-local-01", "generic-awaiting-confirmation")
            try:
                with patch.object(
                    run_xy_calibration,
                    "SHARED_DEVICE_LEASE_DIR",
                    lease_dir,
                ), patch(
                    "run_xy_calibration.request_json",
                    return_value=ready_device_payload(),
                ), patch(
                    "run_xy_calibration.wait_for_stable_target",
                ) as visual, patch(
                    "run_xy_calibration.click_raw_pixel",
                ) as click, self.assertRaisesRegex(
                    TapCalibrationError, "已有活动任务"
                ):
                    run_calibration_step(
                        mode="collect",
                        base_url="http://127.0.0.1:8770",
                        robot=robot,
                        execute=True,
                        device_status_url="http://device.test/api/device",
                        output_root=Path(directory) / "output",
                    )
            finally:
                competitor.release(
                    "device-local-01", "generic-awaiting-confirmation"
                )

        visual.assert_not_called()
        click.assert_not_called()

    def test_physical_lease_occupancy_rejects_zero_actions_and_releases_registry(self):
        frame = Image.new("RGB", (540, 960), "black")
        robot = FakeFullscreenRobot(frame)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lease_dir = root / "leases"
            occupied = InterProcessLease(
                lease_dir / "physical_hardware_action.lease",
                owner_id="legacy-worker",
                metadata={"purpose": "legacy-action"},
            )
            self.assertTrue(occupied.acquire())

            def request(url, *_args, **_kwargs):
                if url == "http://device.test/api/device":
                    return ready_device_payload()
                if url.endswith("/api/samples"):
                    return {"session_id": "page-locked", "samples": []}
                raise AssertionError(url)

            try:
                with patch.object(
                    run_xy_calibration,
                    "SHARED_DEVICE_LEASE_DIR",
                    lease_dir,
                ), patch(
                    "run_xy_calibration.request_json",
                    side_effect=request,
                ), patch(
                    "run_xy_calibration.wait_for_page_state",
                    return_value=fresh_page_state(
                        "fullscreen_setup", sequence=0
                    ),
                ), patch(
                    "run_xy_calibration.wait_for_stable_target",
                    return_value=(frame, (270, 480, (250, 460, 290, 500))),
                ), patch(
                    "run_xy_calibration.click_raw_pixel",
                ) as click, self.assertRaisesRegex(
                    TapCalibrationError, "物理控制权"
                ):
                    run_calibration_step(
                        mode="collect",
                        base_url="http://127.0.0.1:8770",
                        robot=robot,
                        execute=True,
                        device_status_url="http://device.test/api/device",
                        output_root=root / "output",
                    )
            finally:
                occupied.release()

            progress = json.loads(
                (
                    root / "output" / "collect_page-locked" / "progress.json"
                ).read_text(encoding="utf-8")
            )
            registry = DeviceTaskRegistry(lease_directory=lease_dir)
            registry.reserve("device-local-01", "after-calibration-failure")
            registry.release("device-local-01", "after-calibration-failure")

        click.assert_not_called()
        self.assertEqual([], progress["attempts"])

    def test_fit_and_validation_finalize_only_after_nine_persisted_points(self):
        frame = Image.new("RGB", (501, 901), "black")
        robot = FakeFullscreenRobot(frame)
        samples = complete_nine_samples()
        server_samples = [{"sequence": index} for index in range(9)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration_path = root / "tap.json"
            for mode, session_id in (
                ("collect", "collect-complete"),
                ("validate", "validate-complete"),
            ):
                output_dir = root / f"{mode}_{session_id}"
                progress = {
                    "version": 1,
                    "mode": mode,
                    "page_session_id": session_id,
                    "base_url": "http://127.0.0.1:8770",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "frame_size": [501, 901],
                    "samples": samples,
                    "attempts": [],
                }
                run_xy_calibration._write_json(
                    output_dir / "progress.json", progress
                )
                with patch(
                    "run_xy_calibration.request_json",
                    return_value={
                        "session_id": session_id,
                        "samples": server_samples,
                    },
                ), patch(
                    "run_xy_calibration.wait_for_page_state",
                    return_value=fresh_page_state("complete", sequence=9),
                ), patch(
                    "run_xy_calibration.click_raw_pixel",
                ) as click:
                    result = run_calibration_step(
                        mode=mode,
                        base_url="http://127.0.0.1:8770",
                        robot=robot,
                        execute=False,
                        output_root=root,
                        calibration_path=calibration_path,
                    )
                click.assert_not_called()
                self.assertEqual(0, result["physical_actions"])

            payload = json.loads(calibration_path.read_text(encoding="utf-8"))

        self.assertEqual("validation_complete", result["status"])
        self.assertTrue(payload["enabled"])
        self.assertTrue(payload["validated"])

    def test_eight_points_cannot_finalize_fit(self):
        frame = Image.new("RGB", (501, 901), "black")
        robot = FakeFullscreenRobot(frame)
        samples = complete_nine_samples()[:8]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_dir = root / "collect_collect-eight"
            run_xy_calibration._write_json(
                output_dir / "progress.json",
                {
                    "version": 1,
                    "mode": "collect",
                    "page_session_id": "collect-eight",
                    "base_url": "http://127.0.0.1:8770",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "frame_size": [501, 901],
                    "samples": samples,
                    "attempts": [],
                },
            )
            with patch(
                "run_xy_calibration.request_json",
                return_value={
                    "session_id": "collect-eight",
                    "samples": [{"sequence": index} for index in range(8)],
                },
            ), patch(
                "run_xy_calibration.wait_for_page_state",
                return_value=fresh_page_state("calibration", sequence=8),
            ), patch(
                "run_xy_calibration.wait_for_stable_target",
                return_value=(frame, (475, 855, (465, 845, 485, 865))),
            ), patch(
                "run_xy_calibration.click_raw_pixel",
            ) as click:
                result = run_calibration_step(
                    mode="collect",
                    base_url="http://127.0.0.1:8770",
                    robot=robot,
                    execute=False,
                    output_root=root,
                    calibration_path=root / "tap.json",
                )
            calibration_created = (root / "tap.json").exists()

        click.assert_not_called()
        self.assertEqual("awaiting_explicit_execution", result["status"])
        self.assertEqual(8, result["sequence"])
        self.assertFalse(calibration_created)

    def test_page_state_wait_rejects_stale_calibration_heartbeat(self):
        stale = (datetime.now(timezone.utc) - timedelta(seconds=20)).isoformat()
        with patch(
            "run_xy_calibration.request_json",
            return_value={
                "phase": "calibration",
                "fullscreen": True,
                "updated_at": stale,
            },
        ), self.assertRaisesRegex(TapCalibrationError, "没有进入预期阶段"):
            wait_for_page_state(
                "http://127.0.0.1:8770",
                phases={"calibration"},
                timeout=0.01,
            )

    def test_page_state_wait_requires_heartbeat_newer_than_action_input(self):
        timestamp = datetime.now(timezone.utc).isoformat()
        with patch(
            "run_xy_calibration.request_json",
            return_value={
                "phase": "calibration",
                "fullscreen": True,
                "updated_at": timestamp,
            },
        ), self.assertRaisesRegex(TapCalibrationError, "没有进入预期阶段"):
            wait_for_page_state(
                "http://127.0.0.1:8770",
                phases={"calibration"},
                timeout=0.01,
                newer_than=timestamp,
            )

    def test_page_state_wait_skips_fresh_heartbeat_until_sequence_advances(self):
        before = datetime.now(timezone.utc) - timedelta(seconds=1)
        early = fresh_page_state("calibration", sequence=0)
        advanced = fresh_page_state("calibration", sequence=1)
        with patch(
            "run_xy_calibration.request_json",
            side_effect=[early, advanced],
        ), patch("run_xy_calibration.time.sleep"):
            result = wait_for_page_state(
                "http://127.0.0.1:8770",
                phases={"calibration"},
                timeout=1.0,
                newer_than=before.isoformat(),
                expected_sequence=1,
            )

        self.assertEqual(1, result["sequence"])

    def test_single_touch_probe_records_one_verified_contact(self):
        frame = Image.new("RGB", (540, 960), "black")
        robot = FakeProbeRobot(frame)
        record = {
            "sequence": 3,
            "target_x": 100.0,
            "target_y": 200.0,
            "actual_x": 103.0,
            "actual_y": 204.0,
            "viewport_width": 400,
            "viewport_height": 800,
        }
        with tempfile.TemporaryDirectory() as directory, patch(
            "run_xy_calibration._calibration_device_reservation",
            new=fake_device_reservation,
        ), patch(
            "run_xy_calibration._calibration_physical_lease",
            new=fake_physical_lease,
        ), patch(
            "run_xy_calibration.wait_for_stable_target",
            return_value=(frame, (120, 240, (100, 220, 140, 260))),
        ), patch(
            "run_xy_calibration.request_json",
            return_value={"session_id": "probe-success", "samples": []},
        ), patch(
            "run_xy_calibration.wait_for_new_sample",
            return_value=record,
        ):
            output = Path(directory) / "probe"
            result = probe_single_touch(
                base_url="http://127.0.0.1:8770",
                robot=robot,
                execute=True,
                output_dir=output,
            )

            saved = json.loads((output / "report.json").read_text(encoding="utf-8"))

        self.assertEqual(1, result["physical_actions"])
        self.assertTrue(result["contact_detected"])
        self.assertTrue(result["coordinate_passed"])
        self.assertEqual(1, len(robot.tap_calls))
        self.assertTrue(saved["passed"])

    def test_single_touch_probe_keeps_failure_evidence_without_retry(self):
        frame = Image.new("RGB", (540, 960), "black")
        robot = FakeProbeRobot(frame)
        with tempfile.TemporaryDirectory() as directory, patch(
            "run_xy_calibration._calibration_device_reservation",
            new=fake_device_reservation,
        ), patch(
            "run_xy_calibration._calibration_physical_lease",
            new=fake_physical_lease,
        ), patch(
            "run_xy_calibration.wait_for_stable_target",
            return_value=(frame, (120, 240, (100, 220, 140, 260))),
        ), patch(
            "run_xy_calibration.request_json",
            return_value={"session_id": "probe-failure", "samples": []},
        ), patch(
            "run_xy_calibration.wait_for_new_sample",
            side_effect=TapCalibrationError("手机没有回传触点"),
        ):
            output = Path(directory) / "probe"
            result = probe_single_touch(
                base_url="http://127.0.0.1:8770",
                robot=robot,
                execute=True,
                output_dir=output,
            )
            self.assertTrue((output / "report.json").exists())

        self.assertEqual("failed", result["status"])
        self.assertEqual(1, result["physical_actions"])
        self.assertFalse(result["contact_detected"])
        self.assertEqual(1, len(robot.tap_calls))

    def test_single_touch_probe_reobserves_under_physical_lease_before_action(self):
        frame = Image.new("RGB", (540, 960), "black")
        robot = FakeProbeRobot(frame)
        preview = (frame, (120, 240, (100, 220, 140, 260)))
        moved = (frame, (140, 240, (120, 220, 160, 260)))
        with tempfile.TemporaryDirectory() as directory, patch(
            "run_xy_calibration._calibration_device_reservation",
            new=fake_device_reservation,
        ), patch(
            "run_xy_calibration._calibration_physical_lease",
            new=fake_physical_lease,
        ), patch(
            "run_xy_calibration.wait_for_stable_target",
            side_effect=[preview, moved],
        ) as stable_target, patch(
            "run_xy_calibration.request_json",
            return_value={"session_id": "probe-moved", "samples": []},
        ):
            with self.assertRaisesRegex(
                TapCalibrationError,
                "物理锁内探测靶点与预检观察不一致",
            ):
                probe_single_touch(
                    base_url="http://127.0.0.1:8770",
                    robot=robot,
                    execute=True,
                    output_dir=Path(directory) / "probe",
                )

        self.assertEqual(2, stable_target.call_count)
        self.assertEqual([], robot.tap_calls)


if __name__ == "__main__":
    unittest.main()
