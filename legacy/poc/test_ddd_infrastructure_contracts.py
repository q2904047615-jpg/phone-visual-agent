"""Resource paths and pure infrastructure behavior; no hardware access."""
from pathlib import Path
import unittest


class InfrastructureContractTests(unittest.TestCase):
    def test_tap_calibration_has_one_infrastructure_entry(self) -> None:
        import agent.infrastructure.tap_calibration as tap_calibration

        root = Path(__file__).resolve().parent
        self.assertEqual(
            root / "tap_calibration.json",
            tap_calibration.CALIBRATION_PATH,
        )
        coverage = tap_calibration.build_coverage(
            [(0.05, 0.04), (0.95, 0.04), (0.95, 0.96), (0.05, 0.96)]
        )
        self.assertTrue(coverage["sufficient"])
        self.assertEqual([0.9, 0.92], coverage["span"])

    def test_seller_window_adapter_has_one_infrastructure_entry(self) -> None:
        import agent.infrastructure.seller_window_adapter as seller_window

        root = Path(__file__).resolve().parent
        self.assertEqual(root, seller_window.ROOT)
        self.assertEqual(root / "output", seller_window.OUTPUT_DIR)
        self.assertEqual(1.25, seller_window.seller_ui_scale(675))
        self.assertEqual(
            (2557, 2),
            seller_window.cursor_parking_screen_point(
                (0, 0, 830, 1600),
                (0, 0, 2560, 1600),
            ),
        )

    def test_robot_controller_has_one_infrastructure_entry(self) -> None:
        import agent.infrastructure.robot_controller as robot_controller

        root = Path(__file__).resolve().parent
        self.assertEqual(root, robot_controller.POC_ROOT)
        self.assertEqual(
            root / "controller_config.json",
            robot_controller.CONTROL_CONFIG_PATH,
        )
        self.assertEqual(root / "output" / "web", robot_controller.WEB_OUTPUT_DIR)
        mock = robot_controller.MockRobotController(device_id="architecture-test")
        self.assertEqual(root / "tap_calibration.json", mock.calibration_path)
        mock.request_stop()
        self.assertTrue(mock.stop_event.is_set())
        mock.begin_new_task()
        self.assertFalse(mock.stop_event.is_set())


if __name__ == "__main__":
    unittest.main()
