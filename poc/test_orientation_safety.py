from __future__ import annotations

import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw, ImageEnhance

from generic_action_adapter import GenericSingleActionAdapter
from orientation_safety import (
    OrientationCredential,
    OrientationSafetyError,
    PhysicalExecutionGate,
    _mint_audited_credential,
)
from robot_core import RobotController, WorkflowNotReady
from run_xy_calibration import click_raw_pixel


FRAME = Image.new("RGB", (540, 960), "gray")


def patterned_frame() -> Image.Image:
    frame = Image.new("RGB", (540, 960), (80, 90, 100))
    draw = ImageDraw.Draw(frame)
    draw.rectangle((40, 80, 500, 180), fill=(220, 225, 230))
    draw.rectangle((70, 300, 300, 520), fill=(25, 35, 45))
    draw.ellipse((340, 600, 480, 740), fill=(180, 70, 50))
    return frame


def audited_credential(*, device_id="device-a", scene="scene-a", frame=FRAME):
    return _mint_audited_credential(
        device_id=device_id,
        scene_fingerprint=scene,
        frame=frame,
        phone_content_rotation="upright",
        confidence=0.95,
        evidence=("手机状态文字正向",),
    )


class OrientationCredentialTests(unittest.TestCase):
    def test_gate_binds_device_scene_size_and_exact_action_frame(self):
        reference = patterned_frame()
        gate = PhysicalExecutionGate("device-a")
        credential = audited_credential(frame=reference)
        gate.arm(credential, action="tap_semantic", scene_fingerprint="scene-a")
        gate.consume(action="tap_semantic", frame=reference.copy())

        changed = reference.copy()
        changed.paste((255, 255, 255), (180, 360, 360, 600))
        credential = audited_credential(frame=reference)
        gate.arm(credential, action="tap_semantic", scene_fingerprint="scene-a")
        with self.assertRaisesRegex(OrientationSafetyError, "实际捕获帧.*视觉漂移"):
            gate.consume(action="tap_semantic", frame=changed)

    def test_small_camera_noise_and_exposure_pass_but_rotation_is_rejected(self):
        reference = patterned_frame()
        exposure = ImageEnhance.Brightness(reference).enhance(1.04)
        noisy = exposure.copy()
        draw = ImageDraw.Draw(noisy)
        for y in range(20, noisy.height, 53):
            for x in range(15, noisy.width, 47):
                draw.point((x, y), fill=(105, 110, 115))

        for actual in (exposure, noisy):
            gate = PhysicalExecutionGate("device-a")
            credential = audited_credential(frame=reference)
            gate.arm(
                credential,
                action="tap_semantic",
                scene_fingerprint="scene-a",
            )
            gate.consume(action="tap_semantic", frame=actual)

        rotated = reference.rotate(90, expand=False)
        gate = PhysicalExecutionGate("device-a")
        credential = audited_credential(frame=reference)
        gate.arm(credential, action="tap_semantic", scene_fingerprint="scene-a")
        with self.assertRaisesRegex(OrientationSafetyError, "视觉漂移"):
            gate.consume(action="tap_semantic", frame=rotated)

    def test_serialized_or_manually_copied_credential_cannot_arm(self):
        original = audited_credential()
        copied = OrientationCredential.from_dict(original.to_dict())
        gate = PhysicalExecutionGate("device-a")
        with self.assertRaisesRegex(OrientationSafetyError, "实际独立审计"):
            gate.arm(copied, action="tap_semantic", scene_fingerprint="scene-a")

        values = original.to_dict()
        manual = OrientationCredential(
            version=values["version"],
            credential_id=values["credential_id"],
            source=values["source"],
            device_id=values["device_id"],
            scene_fingerprint=values["scene_fingerprint"],
            frame_fingerprint=values["frame_fingerprint"],
            evidence_frame_fingerprint=values["evidence_frame_fingerprint"],
            frame_size=tuple(values["frame_size"]),
            camera_layout_orientation=values["camera_layout_orientation"],
            phone_content_rotation=values["phone_content_rotation"],
            confidence=values["confidence"],
            evidence=tuple(values["evidence"]),
        )
        with self.assertRaisesRegex(OrientationSafetyError, "实际独立审计"):
            gate.arm(manual, action="tap_semantic", scene_fingerprint="scene-a")

    def test_placeholder_device_ids_fail_before_model_or_robot(self):
        for value in ("", "unbound", "unknown"):
            with self.subTest(value=value), self.assertRaises((ValueError, OrientationSafetyError)):
                GenericSingleActionAdapter(
                    capture=lambda: FRAME.copy(),
                    observer=object(),
                    robot=object(),
                    device_id=value,
                )


class PublicPhysicalEntryGateTests(unittest.TestCase):
    def _controller(self) -> RobotController:
        return RobotController(
            title="test",
            device_id="device-a",
            verified_actions={
                "tap_semantic", "dismiss_overlay", "swipe", "back", "home",
                "input_verified_text", "long_press", "drag",
                "reveal_system_navigation",
            },
        )

    def test_every_public_physical_primitive_fails_before_controller_physical_call(self):
        calls = (
            ("tap", lambda c: c.vision_tap_relative(500, 500)),
            ("dismiss", lambda c: c.vision_dismiss_overlay_relative(500, 500)),
            ("long_press", lambda c: c.vision_long_press_relative(500, 500)),
            ("drag", lambda c: c.vision_drag_relative(200, 300, 700, 600)),
            ("reveal", lambda c: c.vision_reveal_system_navigation()),
            ("home", lambda c: c.vision_android_home()),
            ("back", lambda c: c.vision_android_back()),
            ("swipe_up", lambda c: c.vision_swipe_up()),
            ("swipe_down", lambda c: c.vision_swipe_down()),
            ("swipe_left", lambda c: c.vision_swipe_left()),
            ("swipe_right", lambda c: c.vision_swipe_right()),
            ("type_text", lambda c: c.vision_type_text("agent")),
            ("type_pinyin", lambda c: c.vision_type_pinyin("agent", "agent")),
            ("clear_text", lambda c: c.vision_clear_text(delete_count=2)),
        )
        physical_names = (
            "configure_single_click_count", "click_client_point",
            "drag_client_path", "configure_swipe", "trigger_selected_action",
        )
        for label, invoke in calls:
            with self.subTest(label=label):
                controller = self._controller()
                with (
                    patch("robot_core.legacy.find_window", return_value=(123, "test")),
                    patch.object(controller, "_capture_phone", return_value=FRAME.copy()),
                    patch("robot_core.legacy.configure_single_click_count") as configure,
                    patch("robot_core.legacy.click_client_point") as click,
                    patch("robot_core.legacy.drag_client_path") as drag,
                    patch("robot_core.legacy.configure_swipe") as configure_swipe,
                    patch("robot_core.legacy.trigger_selected_action") as trigger,
                ):
                    with self.assertRaisesRegex(OrientationSafetyError, "一次性方向授权"):
                        invoke(controller)
                    mocks = (configure, click, drag, configure_swipe, trigger)
                    self.assertEqual(
                        [0] * len(physical_names),
                        [item.call_count for item in mocks],
                    )

    def test_legacy_multi_step_and_calibration_bypasses_are_closed(self):
        controller = self._controller()
        with self.assertRaisesRegex(WorkflowNotReady, "旧多步 workflow"):
            controller.execute("wechat.send_text_to_file_transfer", {})

        with (
            patch("run_xy_calibration.legacy.find_window") as find_window,
            patch("run_xy_calibration.legacy.configure_single_click_count") as configure,
            patch("run_xy_calibration.legacy.click_client_point") as click,
        ):
            with self.assertRaisesRegex(OrientationSafetyError, "一次性方向授权"):
                click_raw_pixel(controller, FRAME.copy(), (100, 100))
            self.assertEqual(0, find_window.call_count)
            self.assertEqual(0, configure.call_count)
            self.assertEqual(0, click.call_count)


if __name__ == "__main__":
    unittest.main()
