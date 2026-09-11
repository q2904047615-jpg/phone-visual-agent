"""Dynamic pixels cannot veto an otherwise valid one-shot canonical action.

Only observation output and hardware I/O are fixtures. Adapter, controller,
device executor and physical execution gate remain the production classes.
"""
import inspect
import unittest

from PIL import Image, ImageDraw

from agent.application.action_adapter import GenericActionAdapterError
from agent.domain.semantic_action import SemanticAction
from agent.infrastructure.generic_action_adapter import GenericSingleActionAdapter
from agent.infrastructure.orientation_safety import (
    OrientationSafetyError, PhysicalExecutionGate, _mint_single_step_scene_credential,
)
from test_generic_action_adapter import FakeRobot, FakeSceneObserver, SequenceCapture, scene, goal


class GatedRecordingRobot(FakeRobot):
    def __init__(self, physical_frame):
        super().__init__()
        self.gate = PhysicalExecutionGate(self.device_id)
        self.physical_frame = physical_frame

    def arm_physical_execution(self, credential, *, action, scene_fingerprint):
        self.gate.arm(credential, action=action, scene_fingerprint=scene_fingerprint)

    def clear_physical_execution_authorization(self):
        self.gate.clear()

    def _consume(self, action):
        self.gate.consume(action=action, frame=self.physical_frame)


def dynamic_frame(color):
    frame = Image.new("RGB", (540, 960), "gray")
    ImageDraw.Draw(frame).rectangle((90, 200, 440, 750), fill=color)
    return frame


class DynamicFrameAuthorityTests(unittest.TestCase):
    def test_dynamic_and_static_pages_execute_once_without_reselection(self):
        for app, colors in (("media", ("red", "blue", "green")),
                            ("reader", ("white", "black", "yellow")),
                            ("settings", ("gray", "gray", "gray"))):
            with self.subTest(app=app):
                reference, confirmation, physical = map(dynamic_frame, colors)
                robot = GatedRecordingRobot(physical)
                observer = FakeSceneObserver([scene("after", app_id=app)])
                capture = SequenceCapture([confirmation] * 4 + [physical] * 4)
                adapter = GenericSingleActionAdapter(capture=capture, observer=observer, robot=robot,
                    device_id=robot.device_id, frame_interval=0, post_action_settle=0)
                requested = SemanticAction(node_id="current", action="tap_semantic", params={
                    "element_id": "e1", "target": "app_icon", "tap_point": (0.25, 0.45)})
                result = adapter.execute(requested_action=requested, planned_scene=scene("planned", app_id=app),
                    planned_frames=[reference] * 4, goal=goal(), confirmed=True)
                self.assertEqual([("tap", 250, 450)], robot.actions)
                self.assertIs(result.requested_action, requested)
                self.assertIs(result.rebound_action, requested)
                self.assertEqual(1, observer.calls)  # post-action observation, never a reselector
                self.assertEqual(8, capture.calls)  # still captures before and after
                self.assertEqual(4, len(result.after_frames))
                self.assertFalse(result.confirmation_frame_identity_verified)
                self.assertIsNone(result.confirmation_frame_delta)
                with self.assertRaisesRegex(OrientationSafetyError, "缺少一次性"):
                    robot.gate.consume(action="tap_semantic", frame=physical)

    def test_invalid_or_changed_canvas_stops_before_hardware(self):
        frame = dynamic_frame("red")
        for planned in ([], [Image.new("RGB", (541, 960))] * 4,
                        [frame] * 3 + [Image.new("RGB", (540, 961))]):
            with self.subTest(sizes=[item.size for item in planned]):
                robot = GatedRecordingRobot(frame)
                observer = FakeSceneObserver([])
                adapter = GenericSingleActionAdapter(capture=SequenceCapture([frame] * 4),
                    observer=observer, robot=robot, device_id=robot.device_id, frame_interval=0)
                with self.assertRaises(GenericActionAdapterError):
                    adapter.execute(requested_action=SemanticAction(node_id="current", action="tap_semantic",
                        params={"element_id": "e1", "target": "app_icon", "tap_point": (0.25, 0.45)}),
                        planned_scene=scene("planned"), planned_frames=planned, goal=goal(), confirmed=True)
                self.assertEqual([], robot.actions)
                self.assertEqual(0, observer.calls)

    def test_wrong_device_scene_action_and_replay_remain_rejected(self):
        frame = dynamic_frame("blue")
        for wrong_device, wrong_scene in (("other", "scene"), ("device", "other")):
            credential = _mint_single_step_scene_credential(device_id="device", scene_fingerprint="scene", frame=frame)
            with self.assertRaises(OrientationSafetyError):
                PhysicalExecutionGate(wrong_device).arm(credential, action="tap_semantic",
                    scene_fingerprint=wrong_scene)
        credential = _mint_single_step_scene_credential(device_id="device", scene_fingerprint="scene", frame=frame)
        gate = PhysicalExecutionGate("device")
        gate.arm(credential, action="tap_semantic", scene_fingerprint="scene")
        with self.assertRaisesRegex(OrientationSafetyError, "物理动作不匹配"):
            gate.consume(action="home", frame=frame)
        with self.assertRaisesRegex(OrientationSafetyError, "缺少一次性"):
            gate.consume(action="tap_semantic", frame=frame)
        with self.assertRaisesRegex(OrientationSafetyError, "已使用"):
            gate.arm(credential, action="tap_semantic", scene_fingerprint="scene")

    def test_old_pixel_vetoes_have_no_runtime_switch_or_entry(self):
        from agent.infrastructure import orientation_safety as orientation
        self.assertNotIn("confirmation_frame_delta_max", inspect.signature(GenericSingleActionAdapter).parameters)
        self.assertFalse(hasattr(GenericSingleActionAdapter, "_confirmation_frame_delta"))
        for name in ("MAX_ORIENTATION_MEAN_BRIGHTNESS_DELTA", "MAX_ORIENTATION_CENTERED_MAE",
                     "_assert_visually_bound", "_frame_visual_binding", "OrientationFrameMismatchError"):
            self.assertFalse(hasattr(orientation, name), name)


if __name__ == "__main__":
    unittest.main()
