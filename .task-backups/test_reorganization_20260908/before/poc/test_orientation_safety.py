from __future__ import annotations

import gc
import unittest
import weakref
from dataclasses import replace
from unittest.mock import patch

from PIL import Image, ImageDraw, ImageEnhance

from agent.infrastructure.generic_action_adapter import GenericSingleActionAdapter
from agent.infrastructure.orientation_safety import (
    OrientationCredential,
    OrientationSafetyError,
    PhysicalExecutionGate,
    _CLAIMED_AUDIT_CREDENTIALS,
    _mint_single_step_scene_credential,
)
from agent.infrastructure.robot_controller import RobotController
from run_xy_calibration import click_raw_pixel


FRAME = Image.new("RGB", (540, 960), "gray")


def patterned_frame() -> Image.Image:
    frame = Image.new("RGB", (540, 960), (80, 90, 100))
    draw = ImageDraw.Draw(frame)
    draw.rectangle((40, 80, 500, 180), fill=(220, 225, 230))
    draw.rectangle((70, 300, 300, 520), fill=(25, 35, 45))
    draw.ellipse((340, 600, 480, 740), fill=(180, 70, 50))
    return frame


def audited_credential(
    *,
    device_id="device-a",
    scene="scene-a",
    frame=FRAME,
    phone_content_rotation="upright",
):
    credential = _mint_single_step_scene_credential(
        device_id=device_id,
        scene_fingerprint=scene,
        frame=frame,
    )
    return replace(credential, phone_content_rotation=phone_content_rotation)


class OrientationCredentialTests(unittest.TestCase):
    def test_optional_model_alignment_metadata_cannot_veto_local_frame_binding(self):
        credential = _mint_single_step_scene_credential(
            device_id="device-a",
            scene_fingerprint="scene-optional-alignment",
            frame=FRAME,
        )

        self.assertEqual("portrait", credential.camera_layout_orientation)
        self.assertEqual("unknown", credential.phone_content_rotation)
        self.assertEqual(1.0, credential.confidence)
        self.assertEqual(
            ("本地当前帧尺寸与像素形成一次性方向绑定",),
            credential.evidence,
        )
        gate = PhysicalExecutionGate("device-a")
        gate.arm(
            credential,
            action="tap_semantic",
            scene_fingerprint="scene-optional-alignment",
        )
        gate.consume(action="tap_semantic", frame=FRAME.copy())

    def test_fixed_system_navigation_keeps_frame_binding_without_app_rotation(self):
        for action, rotation in (
            ("back", "rotated_90"),
            ("home", "unknown"),
            ("open_recent_apps", "rotated_270"),
        ):
            with self.subTest(action=action, rotation=rotation):
                gate = PhysicalExecutionGate("device-a")
                credential = audited_credential(
                    phone_content_rotation=rotation,
                )
                gate.arm(
                    credential,
                    action=action,
                    scene_fingerprint="scene-a",
                )
                consumed = gate.consume(action=action, frame=FRAME.copy())
                self.assertEqual(rotation, consumed.phone_content_rotation)



    def test_live_execution_source_claim_is_exact_once_and_weakly_held(self):
        gate = PhysicalExecutionGate("device-a")
        credential = audited_credential()
        gate.arm(credential, action="tap_semantic", scene_fingerprint="scene-a")

        credential.claim_live_execution_source()
        with self.assertRaisesRegex(OrientationSafetyError, "live 对象"):
            credential.claim_live_execution_source()

        pending = audited_credential()
        pending_seal = pending._audit_seal
        pending_ref = weakref.ref(pending)
        gate.clear()
        gate.arm(pending, action="tap_semantic", scene_fingerprint="scene-a")
        gate.clear()
        del pending
        gc.collect()

        self.assertIsNone(pending_ref())
        self.assertNotIn(pending_seal, _CLAIMED_AUDIT_CREDENTIALS)

    def test_gate_keeps_canvas_binding_but_does_not_veto_pixel_motion(self):
        reference = patterned_frame()
        gate = PhysicalExecutionGate("device-a")
        credential = audited_credential(frame=reference)
        gate.arm(credential, action="tap_semantic", scene_fingerprint="scene-a")
        gate.consume(action="tap_semantic", frame=reference.copy())

        changed = reference.copy()
        changed.paste((255, 255, 255), (180, 360, 360, 600))
        credential = audited_credential(frame=reference)
        gate.arm(credential, action="tap_semantic", scene_fingerprint="scene-a")
        self.assertIs(credential, gate.consume(action="tap_semantic", frame=changed))
        with self.assertRaisesRegex(OrientationSafetyError, "缺少一次性"):
            gate.consume(action="tap_semantic", frame=changed)

        credential = audited_credential(frame=reference)
        gate.arm(credential, action="tap_semantic", scene_fingerprint="scene-a")
        with self.assertRaisesRegex(OrientationSafetyError, "画布尺寸不匹配"):
            gate.consume(
                action="tap_semantic",
                frame=Image.new("RGB", (960, 540), "gray"),
            )

    def test_gate_freshness_is_the_one_shot_visual_binding_not_wall_clock(self):
        reference = patterned_frame()
        gate = PhysicalExecutionGate("device-a")
        credential = audited_credential(frame=reference)

        with patch(
            "time.monotonic",
            side_effect=AssertionError("wall clock must not authorize the frame"),
        ):
            gate.arm(
                credential,
                action="tap_semantic",
                scene_fingerprint="scene-a",
            )
            consumed = gate.consume(
                action="tap_semantic",
                frame=reference.copy(),
            )

        self.assertIs(consumed, credential)

    def test_pixel_noise_exposure_and_same_canvas_rotation_are_not_identity_vetoes(self):
        reference = patterned_frame()
        exposure = ImageEnhance.Brightness(reference).enhance(1.04)
        darker_exposure = ImageEnhance.Brightness(reference).enhance(0.90)
        noisy = exposure.copy()
        draw = ImageDraw.Draw(noisy)
        for y in range(20, noisy.height, 53):
            for x in range(15, noisy.width, 47):
                draw.point((x, y), fill=(105, 110, 115))

        for actual in (exposure, darker_exposure, noisy):
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
        # Accepted tradeoff: same-canvas UI movement is no longer locally
        # classified. Explicit rotated credentials and wrong sizes still fail.
        self.assertIs(credential, gate.consume(action="tap_semantic", frame=rotated))

    def test_live_seal_canvas_cannot_be_changed_by_copying_credential(self):
        original = audited_credential()
        forged = replace(original, frame_size=(600, 1000))
        gate = PhysicalExecutionGate("device-a")
        gate.arm(forged, action="tap_semantic", scene_fingerprint="scene-a")
        with self.assertRaisesRegex(OrientationSafetyError, "实际捕获帧尺寸发生变化"):
            gate.consume(action="tap_semantic", frame=Image.new("RGB", (600, 1000)))
        with self.assertRaisesRegex(OrientationSafetyError, "缺少一次性"):
            gate.consume(action="tap_semantic", frame=FRAME)

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

    def test_failed_rearm_clears_previous_authorization(self):
        gate = PhysicalExecutionGate("device-a")
        first = audited_credential()
        gate.arm(first, action="tap_semantic", scene_fingerprint="scene-a")

        invalid = OrientationCredential.from_dict(audited_credential().to_dict())
        with self.assertRaisesRegex(OrientationSafetyError, "实际独立审计"):
            gate.arm(
                invalid,
                action="tap_semantic",
                scene_fingerprint="scene-a",
            )

        with self.assertRaisesRegex(OrientationSafetyError, "缺少一次性方向授权"):
            gate.consume(action="tap_semantic", frame=FRAME.copy())

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
                "long_press", "drag",
                "reveal_system_navigation",
            },
        )

    def test_every_public_physical_primitive_fails_before_controller_physical_call(self):
        calls = (
            ("tap", lambda c: c.vision_tap_relative(500, 500)),
            ("dismiss", lambda c: c.vision_dismiss_overlay_relative(500, 500)),
            ("long_press", lambda c: c.vision_long_press_relative(500, 500)),
            ("drag", lambda c: c.vision_drag_relative(200, 300, 700, 600)),
            ("swipe_relative", lambda c: c.vision_swipe_relative(700, 500, 200, 500, "left")),
            ("reveal", lambda c: c.vision_reveal_system_navigation()),
            ("home", lambda c: c.vision_android_home()),
            ("back", lambda c: c.vision_android_back()),
            ("swipe_up", lambda c: c.vision_swipe_up()),
            ("swipe_down", lambda c: c.vision_swipe_down()),
            ("swipe_left", lambda c: c.vision_swipe_left()),
            ("swipe_right", lambda c: c.vision_swipe_right()),
        )
        physical_names = (
            "configure_single_click_count", "click_client_point",
            "long_press_client_point", "drag_client_path", "swipe_client_path", "configure_swipe",
            "trigger_selected_action",
        )
        for label, invoke in calls:
            with self.subTest(label=label):
                controller = self._controller()
                with (
                    patch("agent.infrastructure.robot_controller.seller_gui.find_window", return_value=(123, "test")),
                    patch.object(controller, "_capture_phone", return_value=FRAME.copy()),
                    patch("agent.infrastructure.robot_controller.seller_gui.configure_single_click_count") as configure,
                    patch("agent.infrastructure.robot_controller.seller_gui.click_client_point") as click,
                    patch("agent.infrastructure.robot_controller.seller_gui.long_press_client_point") as long_press,
                    patch("agent.infrastructure.robot_controller.seller_gui.drag_client_path") as drag,
                    patch("agent.infrastructure.robot_controller.seller_gui.swipe_client_path") as swipe_path,
                    patch("agent.infrastructure.robot_controller.seller_gui.configure_swipe") as configure_swipe,
                    patch("agent.infrastructure.robot_controller.seller_gui.trigger_selected_action") as trigger,
                ):
                    with self.assertRaisesRegex(OrientationSafetyError, "一次性方向授权"):
                        invoke(controller)
                    mocks = (
                        configure,
                        click,
                        long_press,
                        drag,
                        swipe_path,
                        configure_swipe,
                        trigger,
                    )
                    self.assertEqual(
                        [0] * len(physical_names),
                        [item.call_count for item in mocks],
                    )

    def test_explicit_rotated_credentials_cannot_reach_visual_robot_physical_entry(self):
        for rotation in (
            "rotated_90",
            "rotated_180",
            "rotated_270",
        ):
            with self.subTest(rotation=rotation):
                controller = self._controller()
                credential = audited_credential(
                    phone_content_rotation=rotation,
                )
                with self.assertRaisesRegex(
                    OrientationSafetyError,
                    "与执行坐标轴不一致",
                ):
                    controller.arm_physical_execution(
                        credential,
                        action="tap_semantic",
                        scene_fingerprint="scene-a",
                    )
                with (
                    patch(
                        "agent.infrastructure.robot_controller.seller_gui.find_window",
                        return_value=(123, "test"),
                    ),
                    patch.object(
                        controller,
                        "_capture_phone",
                        return_value=FRAME.copy(),
                    ),
                    patch("agent.infrastructure.robot_controller.seller_gui.configure_single_click_count") as configure,
                    patch("agent.infrastructure.robot_controller.seller_gui.click_client_point") as click,
                    patch("agent.infrastructure.robot_controller.seller_gui.drag_client_path") as drag,
                    patch("agent.infrastructure.robot_controller.seller_gui.configure_swipe") as configure_swipe,
                    patch("agent.infrastructure.robot_controller.seller_gui.trigger_selected_action") as trigger,
                ):
                    with self.assertRaisesRegex(
                        OrientationSafetyError,
                        "一次性方向授权",
                    ):
                        controller.vision_tap_relative(500, 500)
                    self.assertEqual(
                        [0, 0, 0, 0, 0],
                        [
                            configure.call_count,
                            click.call_count,
                            drag.call_count,
                            configure_swipe.call_count,
                            trigger.call_count,
                        ],
                    )

    def test_calibration_bypass_is_closed(self):
        controller = self._controller()
        with (
            patch("run_xy_calibration.seller_gui.find_window") as find_window,
            patch("run_xy_calibration.seller_gui.configure_single_click_count") as configure,
            patch("run_xy_calibration.seller_gui.click_client_point") as click,
        ):
            with self.assertRaisesRegex(OrientationSafetyError, "一次性方向授权"):
                click_raw_pixel(controller, FRAME.copy(), (100, 100))
            self.assertEqual(0, find_window.call_count)
            self.assertEqual(0, configure.call_count)
            self.assertEqual(0, click.call_count)

if __name__ == "__main__":
    unittest.main()
