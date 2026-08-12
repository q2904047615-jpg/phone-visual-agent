from __future__ import annotations

import json
import tempfile
import unittest
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
    locate_magenta_target,
    probe_single_touch,
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


class TapCalibrationMathTests(unittest.TestCase):
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
                        "frame_size": [540, 960],
                        "target_to_command": [[1.0, 0.0, -0.01], [0.0, 1.0, 0.02]],
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(corrected_grid_point(500, 500, (540, 960), path), (490, 520))

    def test_build_calibration_recovers_translation(self):
        width, height = 501, 901
        points = [(0.15, 0.15), (0.5, 0.15), (0.85, 0.15), (0.15, 0.5), (0.5, 0.5), (0.85, 0.5), (0.15, 0.85), (0.5, 0.85), (0.85, 0.85)]
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
        self.assertFalse(payload["enabled"])

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
