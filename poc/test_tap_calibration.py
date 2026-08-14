from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from tap_calibration import (
    Affine2D,
    TapCalibrationError,
    build_calibration,
    corrected_grid_point,
    fit_affine,
)
from run_xy_calibration import (
    calibration_page_ready,
    ensure_fullscreen_calibration_page,
    locate_magenta_target,
    probe_single_touch,
    wait_for_page_state,
)


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

    def test_fullscreen_setup_failure_keeps_one_action_evidence(self):
        frame = Image.new("RGB", (540, 960), "black")
        robot = FakeFullscreenRobot(frame)
        states = [
            {
                "phase": "fullscreen_setup",
                "fullscreen": False,
                "calibration_mode": "setup",
            },
            {
                "phase": "blocked",
                "fullscreen": False,
                "calibration_mode": "blocked",
                "fullscreen_attempted": True,
                "error": "fullscreen refused",
            },
        ]

        def page_state(*_args, **_kwargs):
            value = states.pop(0)
            if isinstance(value, Exception):
                raise value
            return value

        with tempfile.TemporaryDirectory() as directory, patch(
            "run_xy_calibration.wait_for_page_state",
            side_effect=page_state,
        ), patch(
            "run_xy_calibration.wait_for_stable_target",
            return_value=(frame, (270, 480, (250, 460, 290, 500))),
        ), patch(
            "run_xy_calibration.click_raw_pixel",
        ) as click, patch(
            "run_xy_calibration.request_json",
            return_value={"phase": "fullscreen_setup", "fullscreen": False},
        ):
            output = Path(directory)
            with self.assertRaisesRegex(TapCalibrationError, "fullscreen refused"):
                ensure_fullscreen_calibration_page(
                    base_url="http://127.0.0.1:8770",
                    robot=robot,
                    output_dir=output,
                )
            report = json.loads(
                (output / "00_fullscreen_setup.json").read_text(encoding="utf-8")
            )

        click.assert_called_once()
        self.assertEqual(1, report["physical_actions"])
        self.assertFalse(report["passed"])

    def test_viewport_fallback_keeps_one_setup_action_and_continues(self):
        frame = Image.new("RGB", (540, 960), "black")
        robot = FakeFullscreenRobot(frame)
        ready = {
            "phase": "calibration",
            "fullscreen": False,
            "calibration_mode": "viewport_coverage",
            "fullscreen_attempted": True,
            "viewport_coverage": {
                "eligible": True,
                "width_ratio": 0.95,
                "height_ratio": 0.94,
            },
        }
        with tempfile.TemporaryDirectory() as directory, patch(
            "run_xy_calibration.wait_for_page_state",
            side_effect=[
                {
                    "phase": "fullscreen_setup",
                    "fullscreen": False,
                    "calibration_mode": "setup",
                },
                ready,
            ],
        ), patch(
            "run_xy_calibration.wait_for_stable_target",
            return_value=(frame, (270, 480, (250, 460, 290, 500))),
        ), patch(
            "run_xy_calibration.click_raw_pixel",
        ) as click:
            output = Path(directory)
            actions = ensure_fullscreen_calibration_page(
                base_url="http://127.0.0.1:8770",
                robot=robot,
                output_dir=output,
            )
            report = json.loads(
                (output / "00_fullscreen_setup.json").read_text(encoding="utf-8")
            )

        click.assert_called_once()
        self.assertEqual(1, actions)
        self.assertTrue(report["passed"])
        self.assertTrue(report["safe_viewport_fallback"])
        self.assertFalse(report["fullscreen_entered"])

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
            "run_xy_calibration.wait_for_stable_target",
            return_value=(frame, (120, 240, (100, 220, 140, 260))),
        ), patch(
            "run_xy_calibration.request_json",
            return_value={"samples": []},
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
            "run_xy_calibration.wait_for_stable_target",
            return_value=(frame, (120, 240, (100, 220, 140, 260))),
        ), patch(
            "run_xy_calibration.request_json",
            return_value={"samples": []},
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


if __name__ == "__main__":
    unittest.main()
