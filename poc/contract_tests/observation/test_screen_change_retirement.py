"""Moving pixels are observations, not transport errors. No hardware or remote calls."""
from dataclasses import replace
import unittest
from unittest.mock import patch
from PIL import Image, ImageOps
from agent.application.qwen_visual_decision import QwenVisualDecisionObserver
from agent.domain.universal_action_controller import UniversalActionController
from agent.domain.vision_model import VisionAgentError
from agent.infrastructure.generic_scene_observer import SingleStepGenericSceneObserver
from agent.infrastructure.observation_images import measure_local_stability
from agent.infrastructure.trusted_observation_frames import build_trusted_observation, validate_trusted_observation_against_frames
from agent.infrastructure import seller_window_adapter as seller
from test_support.generic_action_adapter import (
    RawSceneProvider,
    textured_phone_frame,
)
from contract_tests.observation.test_point_scene_projection import wire, decision, task_context
from test_qwen_visual_decision import StatusOnlyProvider


class FakeWindows:
    def __init__(self):
        self.events = []
        self.positions = []

    def ClientToScreen(self, hwnd, point): return 1
    def GetCursorPos(self, point):
        point._obj.x, point._obj.y = 11, 12
        return 1
    def ShowWindow(self, *args): return 1
    def SetForegroundWindow(self, *args): return 1
    def SetCursorPos(self, x, y):
        self.positions.append((x, y))
        return 1
    def GetAsyncKeyState(self, *args): return 0
    def mouse_event(self, event, *args): self.events.append(event)


class ScreenChangeRetirementTests(unittest.TestCase):
    def test_real_observer_binder_controller_accept_dynamic_and_static_frames(self):
        frame = textured_phone_frame().resize((240, 480))
        inverted = ImageOps.invert(frame)
        for dynamic in (True, False):
            with self.subTest(dynamic=dynamic):
                frames = [frame, inverted, frame, inverted] if dynamic else [frame] * 4
                measured = measure_local_stability(frames, threshold=0, allow_leading_outlier=True)
                self.assertEqual(not dynamic, measured.stable)
                context = task_context('打开当前入口')
                provider = RawSceneProvider([wire(decision())])
                observer = SingleStepGenericSceneObserver(provider)
                allowed = frozenset({'tap_semantic'})
                # Even an arbitrarily strict diagnostic threshold cannot veto the real chain.
                with patch.dict('os.environ', {'ROBOT_LOCAL_FRAME_DELTA_MAX': '0'}):
                    scene, choice = observer.observe_with_decision(frames=frames,
                        goal_context={'objective': context.raw_goal}, device_id=context.device_id,
                        available_action_kinds=allowed)
                    trusted = build_trusted_observation(frames=frames, device_id=context.device_id, scene=scene)
                    result = QwenVisualDecisionObserver(StatusOnlyProvider(),
                        trusted_observation_frame_validator=validate_trusted_observation_against_frames).decide(
                        frames=frames, task_context=context, trusted_observation=trusted,
                        model_decision=choice, available_action_kinds=allowed)
                resolved = UniversalActionController().resolve_one(result.proposal.action, replace(scene, stable=False))
                self.assertEqual('tap_semantic', resolved.kind)
                self.assertEqual(1, provider.calls)
                self.assertEqual(not dynamic, trusted.local_stability.stable)
                with self.assertRaisesRegex(VisionAgentError, 'fingerprint'):
                    validate_trusted_observation_against_frames(replace(trusted, fingerprint='wrong'), frames,
                        allow_leading_outlier=True)

    def test_changed_dimensions_still_reject(self):
        with self.assertRaisesRegex(VisionAgentError, '尺寸'):
            measure_local_stability([Image.new('RGB', (540, 960)), Image.new('RGB', (541, 960))])

    def test_all_seller_mouse_paths_dispatch_without_screen_sampling(self):
        for kind in ('click', 'double_tap', 'long_press', 'swipe'):
            fake = FakeWindows()
            with self.subTest(kind=kind), patch.object(seller, 'user32', fake), \
                patch.object(seller, 'client_geometry', return_value=(0, 0, 540, 1038)), \
                patch.object(seller.ImageGrab, 'grab', side_effect=AssertionError('must not sample video')) as capture, \
                patch.object(seller.time, 'sleep'), patch.object(seller, 'sleep_interruptible'):
                if kind in ('click', 'double_tap'):
                    result = seller.click_client_point(123, 200, 400, 0, .35,
                        return_dispatch_receipt=True, click_count=2 if kind == 'double_tap' else 1)
                    expected = [seller.MOUSEEVENTF_LEFTDOWN, seller.MOUSEEVENTF_LEFTUP]
                    self.assertEqual([(200, 400), (11, 12)], fake.positions)
                elif kind == 'long_press':
                    result = seller.long_press_client_point(123, 200, 400, hold_seconds=.8)
                    expected = [seller.MOUSEEVENTF_RIGHTDOWN, seller.MOUSEEVENTF_RIGHTUP]
                    self.assertEqual([(200, 400), (11, 12)], fake.positions)
                else:
                    result = seller.swipe_client_path(123, (200, 700), (200, 300))
                    expected = [seller.MOUSEEVENTF_RIGHTDOWN, seller.MOUSEEVENTF_RIGHTUP]
                    self.assertEqual(8, len(fake.positions))  # start, 6 moves, restore; no barrier detour
                self.assertEqual(expected, fake.events)
                self.assertTrue(result['input_events_dispatched'])
                self.assertFalse(result['mechanical_contact_ack'])
                self.assertNotIn('seller_event_barrier_confirmed', result)
                self.assertNotIn('round_trip_position_confirmed', result)
                capture.assert_not_called()


if __name__ == '__main__':
    unittest.main()
