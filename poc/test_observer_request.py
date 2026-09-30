from __future__ import annotations
from PIL import Image
from agent.infrastructure.generic_scene_observer import (
    OBSERVATION_MAX_ATTEMPTS,
    OBSERVATION_TIMEOUT_SECONDS,
    SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
)
from agent.infrastructure.generic_scene_observer import SingleStepGenericSceneObserver
from agent.infrastructure.dashscope_vision_provider import _image_data_url
from agent.infrastructure.generic_scene_observer import _single_step_observation_prompt
import json
from pathlib import Path
import tempfile
import unittest
from test_support.generic_scene_observer import (
    SequenceProvider,
    _BaseStructuredDecisionContractTests,
    audited_text_input_scene,
    scene_payload,
    stable_frames,
    unique_goal_element,
)


class StructuredDecisionContractTests(_BaseStructuredDecisionContractTests):
    def test_prompt_exposes_whole_task_and_only_executed_history(self) -> None:
        context = {"objective": "打开消息应用然后回主屏幕", "entities": {"history": [{"action": "home", "transport_outcome": "executed"}]}}
        prompt = _single_step_observation_prompt(context, include_input_structure=False, image_count=3,
            request_image_size=(540, 960), available_action_kinds=("home", "tap_semantic"))
        self.assertIn("打开消息应用然后回主屏幕", prompt)
        self.assertIn('"transport_outcome":"executed"', prompt)
        self.assertNotIn("forbidden_future_effect_kinds", prompt)

    def test_prompt_uses_task_budget_not_gesture_retry_count(self) -> None:
        prompt = _single_step_observation_prompt(
            {}, include_input_structure=False, image_count=1,
            request_image_size=(540, 960),
            available_action_kinds=('scroll', 'swipe_element'),
        )
        self.assertIn('当前设备本轮可用动作', prompt)
        self.assertIn('整任务预算', prompt)
        self.assertNotIn('gesture_correction', prompt)
        self.assertNotIn('绝不能重复原轨迹', prompt)
        self.assertIn('HISTORY', prompt)
        self.assertIn('CURRENT', prompt)


class SingleStepGenericSceneObserverTests(unittest.TestCase):
    def test_visual_references_are_sent_as_separate_labeled_examples(self) -> None:
        envelope = {
            "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
            "coordinate_space": {"kind": "axis_grid", "width": 1000, "height": 1000},
            "scene": scene_payload(),
            "input_structure": None,
            "decision": {"status": "finish", "reason": "当前任务已完成"},
        }
        provider = SequenceProvider([envelope])
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "example.jpg"
            Image.new("RGB", (1179, 2556), (220, 20, 60)).save(reference)
            SingleStepGenericSceneObserver(provider).observe_with_decision(
                frames=stable_frames(), goal_context={"objective": "识别页面"},
                device_id="device-local-01", visual_reference_paths=(reference,),
            )
        content = provider.messages_seen[0][1]["content"]
        image_parts = [part for part in content if part.get("type") == "image_url"]
        self.assertEqual(5, len(image_parts))
        self.assertIn("REFERENCE IMAGE 1 - VISUAL EXAMPLE ONLY", content[1]["text"])
        self.assertIn("参考图坐标", content[0]["text"])
        self.assertTrue(image_parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,"))
        self.assertEqual(1, provider.calls)

    def test_explicit_system_home_observation_sends_unmasked_phone_frame(
        self,
    ) -> None:
        cases = (
            (
                "chinese",
                (173, 61, 211),
                {
                    "objective": "回到手机主屏幕",
                    "execution_class": "navigate",
                    "completion_conditions": ["手机主屏幕可见"],
                },
            ),
            (
                "english-variation",
                (27, 189, 116),
                {
                    "objective": "return to the phone home screen",
                    "execution_class": "navigate",
                    "completion_conditions": ["phone home screen is visible"],
                },
            ),
        )
        for name, color, goal_context in cases:
            with self.subTest(name=name):
                envelope = {
                    "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
                    "coordinate_space": {
                        "kind": "axis_grid",
                        "width": 1000,
                        "height": 1000,
                    },
                    "scene": scene_payload(),
                    "input_structure": None,
                }
                provider = SequenceProvider([envelope])
                frames = stable_frames(color)

                observed = SingleStepGenericSceneObserver(provider).observe(
                    frames=frames,
                    goal_context=goal_context,
                    device_id="device-local-01",
                    available_action_kinds={"back", "home"},
                )

                image_parts = [
                    part for part in provider.messages_seen[0][1]["content"] if part.get("type") == "image_url"
                ]
                self.assertEqual(4, len(image_parts))
                self.assertEqual(180.0, OBSERVATION_TIMEOUT_SECONDS)
                self.assertEqual(OBSERVATION_TIMEOUT_SECONDS, provider.call_options["timeout"])
                self.assertEqual(2, OBSERVATION_MAX_ATTEMPTS)
                self.assertEqual(OBSERVATION_MAX_ATTEMPTS, provider.call_options["max_attempts"])
                self.assertEqual(
                    _image_data_url(frames[-1].convert("RGB")),
                    image_parts[0]["image_url"]["url"],
                )
                self.assertEqual("calculator", observed.foreground_app_id)
                self.assertEqual("app_home", observed.screen_id)
                prompt = provider.messages_seen[0][1]["content"][0]["text"]
                self.assertNotIn("中央App内容未披露", prompt)
                self.assertNotIn("固定遮罩", prompt)
                for text in ("open_recent_apps", "Home", "HISTORY", "CURRENT",
                    '当前设备本轮可用动作（唯一运行时动作集合）：["back","home"]'):
                    self.assertIn(text, prompt)

    def test_runtime_action_set_is_sent_on_every_observation(self) -> None:
        envelope = {
            "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
            "coordinate_space": {
                "kind": "axis_grid",
                "width": 1000,
                "height": 1000,
            },
            "scene": scene_payload(),
            "input_structure": None,
        }
        provider = SequenceProvider(
            [
                json.loads(json.dumps(envelope)),
                json.loads(json.dumps(envelope)),
                json.loads(json.dumps(envelope)),
            ]
        )
        observer = SingleStepGenericSceneObserver(provider)
        frames = stable_frames()
        goal_context = {
            "objective": "打开目标应用",
            "execution_class": "navigate",
            "completion_conditions": ["目标应用主页面可见"],
        }

        observer.observe(
            frames=frames,
            goal_context=goal_context,
            device_id="device-local-01",
            available_action_kinds={"home"},
        )
        observer.observe(
            frames=frames,
            goal_context=goal_context,
            device_id="device-local-01",
            available_action_kinds={"back"},
        )
        observer.observe(
            frames=frames,
            goal_context=goal_context,
            device_id="device-local-01",
            available_action_kinds={"back"},
        )

        self.assertEqual(3, provider.calls)
        home_prompt = provider.messages_seen[0][1]["content"][0]["text"]
        back_prompt = provider.messages_seen[1][1]["content"][0]["text"]
        repeated_back_prompt = provider.messages_seen[2][1]["content"][0]["text"]
        self.assertIn(
            '当前设备本轮可用动作（唯一运行时动作集合）：["home"]',
            home_prompt,
        )
        self.assertIn(
            '当前设备本轮可用动作（唯一运行时动作集合）：["back"]',
            back_prompt,
        )
        self.assertIn(
            '当前设备本轮可用动作（唯一运行时动作集合）：["back"]',
            repeated_back_prompt,
        )

    def test_single_step_observer_normalizes_request_bound_y_axis_grid(
        self,
    ) -> None:
        context = {
            "entities": {
                "active_subgoal_visual_context": {
                    "subgoal_id": "input_message",
                    "objective": "在输入框中输入消息",
                    "constraints": [],
                    "completion_conditions": ["输入框显示指定消息"],
                    "execution_class": "navigate",
                    "goal_entities": {
                        "active_input_transaction_text": "aaazjie？你好",
                        "active_input_field_id": "message_field",
                        "active_input_multiline": False,
                    },
                }
            }
        }
        cases = (
            ((810, 1440), (720, 1280), [120, 906, 780, 953]),
            ((540, 960), (540, 960), [120, 906, 780, 953]),
        )
        canonical_bounds = []
        for frame_size, request_size, raw_bounds in cases:
            with self.subTest(request_size=request_size):
                scene, audit = audited_text_input_scene(raw_bounds)
                provider = SequenceProvider(
                    [
                        {
                            "protocol_version": SINGLE_STEP_OBSERVATION_PROTOCOL_VERSION,
                            "coordinate_space": {
                                "kind": "axis_grid",
                                "width": 1000,
                                "height": 1000,
                            },
                            "scene": scene,
                            "input_structure": audit,
                        }
                    ]
                )
                observer = SingleStepGenericSceneObserver(provider)

                observed = observer.observe(
                    frames=[Image.new("RGB", frame_size, (30, 40, 50)) for _ in range(4)],
                    goal_context=context,
                    device_id="device-local-01",
                )

                field = unique_goal_element(observed)
                canonical_bounds.append(tuple(round(value, 3) for value in field.bounds))
                normalization = observer.last_diagnostics["coordinate_normalization"]
                self.assertEqual("axis_grid", normalization["wire_kind"])
                self.assertEqual([1000, 1000], normalization["wire_extent"])
                self.assertEqual(list(frame_size), normalization["request_image_size"])
                self.assertTrue(normalization["applied"])
                self.assertEqual(1, provider.calls)
                prompt = provider.messages_seen[0][1]["content"][0]["text"]
                self.assertIn(
                    '"coordinate_space":{"kind":"axis_grid","width":1000,"height":1000}',
                    prompt,
                )

        self.assertEqual((0.12, 0.906, 0.78, 0.953), canonical_bounds[0])
        self.assertEqual(canonical_bounds[0], canonical_bounds[1])


if __name__ == "__main__":
    unittest.main()

