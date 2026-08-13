from __future__ import annotations

import json
import unittest

from PIL import Image, ImageFilter

from generic_scene_observer import GenericSceneObserver
from ui_scene import UISceneError
from vision_agent import VisionAgentError


class FakeProvider:
    configured = True

    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls = 0

    def status(self) -> dict:
        return {"configured": True, "model": "fake-qwen"}

    def _chat(
        self,
        messages: list[dict],
        max_tokens: int,
        *,
        timeout: float | None = None,
        max_attempts: int | None = None,
    ) -> str:
        self.calls += 1
        self.messages = messages
        self.max_tokens = max_tokens
        self.call_options = {
            "timeout": timeout,
            "max_attempts": max_attempts,
        }
        return json.dumps(self.payload, ensure_ascii=False)


class SequenceProvider(FakeProvider):
    def __init__(self, responses: list[str | dict | BaseException]) -> None:
        super().__init__({})
        self.responses = list(responses)
        self.max_tokens_seen: list[int] = []
        self.messages_seen: list[list[dict]] = []

    def _chat(
        self,
        messages: list[dict],
        max_tokens: int,
        *,
        timeout: float | None = None,
        max_attempts: int | None = None,
    ) -> str:
        self.calls += 1
        self.messages_seen.append(messages)
        self.max_tokens_seen.append(max_tokens)
        self.call_options = {"timeout": timeout, "max_attempts": max_attempts}
        value = self.responses.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def stable_frames(color: tuple[int, int, int] = (30, 40, 50)) -> list[Image.Image]:
    return [Image.new("RGB", (540, 960), color) for _ in range(4)]


def stable_frames_with_one_sharp_center() -> list[Image.Image]:
    base = Image.new("RGB", (540, 960), (30, 40, 50))
    checker = Image.new("RGB", (280, 420), (10, 10, 10))
    pixels = checker.load()
    for y in range(checker.height):
        for x in range(checker.width):
            value = 240 if ((x // 8) + (y // 8)) % 2 else 10
            pixels[x, y] = (value, value, value)
    sharp = base.copy()
    sharp.paste(checker, (130, 270))
    soft_center = checker.filter(ImageFilter.GaussianBlur(radius=6.0))
    soft = base.copy()
    soft.paste(soft_center, (130, 270))
    return [soft.copy(), sharp, soft.copy(), soft.copy()]


def scene_payload() -> dict:
    return {
        "protocol_version": "2026-08-10-ui-scene-v2",
        "foreground_app_id": "calculator",
        "screen_id": "app_home",
        "summary": "计算器首页",
        "elements": [
            {
                "element_id": "e1",
                "role": "button",
                "meaning": "digit_key",
                "label": "7",
                "bounds": [100, 600, 260, 760],
                "confidence": 0.98,
                "states": {"enabled": True},
                "evidence": ["7"],
            }
        ],
        "overlays": [],
        "stable": True,
        "confidence": 0.96,
        "fingerprint": "model-value-must-not-be-trusted",
    }


class GenericSceneObserverTests(unittest.TestCase):
    def test_observes_arbitrary_app_and_normalizes_bounds(self) -> None:
        provider = FakeProvider(scene_payload())
        observer = GenericSceneObserver(provider)
        scene = observer.observe(
            frames=stable_frames(),
            goal_context={"objective": "在计算器输入7"},
        )
        self.assertEqual(scene.app_id, "calculator")
        self.assertEqual(scene.foreground_app_id, "calculator")
        self.assertEqual(scene.elements[0].bounds, (0.1, 0.6, 0.26, 0.76))
        self.assertNotEqual(scene.fingerprint, "model-value-must-not-be-trusted")
        self.assertEqual(provider.calls, 1)
        self.assertEqual(provider.max_tokens, 800)
        self.assertEqual(provider.call_options["timeout"], 60.0)

    def test_low_confidence_scene_is_rejected_before_trusted_observation(self) -> None:
        payload = scene_payload()
        payload["confidence"] = 0.6
        provider = FakeProvider(payload)

        with self.assertRaisesRegex(VisionAgentError, "整体置信度不足"):
            GenericSceneObserver(provider).observe(
                frames=stable_frames(),
                goal_context={"objective": "目标内容可见"},
            )

        self.assertEqual(2, provider.calls)
        self.assertEqual(provider.call_options["max_attempts"], 2)

    def test_low_scene_confidence_accepts_one_strong_goal_element_only(self) -> None:
        payload = scene_payload()
        payload["confidence"] = 0.6
        payload["elements"][0]["states"]["goal_relevant"] = True
        provider = FakeProvider(payload)
        observer = GenericSceneObserver(provider)

        scene = observer.observe(
            frames=stable_frames(),
            goal_context={"objective": "点击唯一清晰目标"},
        )

        self.assertEqual("e1", scene.unique_trusted_goal_element().element_id)
        self.assertEqual("unique_goal_element", observer.last_diagnostics["confidence_basis"])

    def test_low_scene_confidence_rejects_multiple_strong_goal_elements(self) -> None:
        payload = scene_payload()
        payload["confidence"] = 0.6
        payload["elements"][0]["states"]["goal_relevant"] = True
        second = dict(payload["elements"][0])
        second.update({"element_id": "e2", "bounds": [300, 600, 460, 760]})
        payload["elements"].append(second)

        with self.assertRaisesRegex(VisionAgentError, "整体置信度不足"):
            GenericSceneObserver(FakeProvider(payload)).observe(
                frames=stable_frames(),
                goal_context={"objective": "点击目标"},
            )

    def test_low_scene_confidence_rejects_low_confidence_goal_element(self) -> None:
        payload = scene_payload()
        payload["confidence"] = 0.6
        payload["elements"][0]["confidence"] = 0.7
        payload["elements"][0]["states"]["goal_relevant"] = True

        with self.assertRaisesRegex(VisionAgentError, "整体置信度不足"):
            GenericSceneObserver(FakeProvider(payload)).observe(
                frames=stable_frames(),
                goal_context={"objective": "点击目标"},
            )

    def test_low_scene_confidence_accepts_read_only_completion_evidence(self) -> None:
        payload = scene_payload()
        payload["confidence"] = 0.6
        payload["elements"][0].update(
            {
                "role": "container",
                "meaning": "visible_result_count",
                "states": {"goal_relevant": True},
            }
        )
        observer = GenericSceneObserver(FakeProvider(payload))

        scene = observer.observe(
            frames=stable_frames(),
            goal_context={"objective": "确认结果已显示"},
        )

        self.assertEqual(("e1",), tuple(x.element_id for x in scene.trusted_completion_evidence()))
        self.assertEqual("completion_evidence_only", observer.last_diagnostics["confidence_basis"])

    def test_unstable_frames_do_not_call_model(self) -> None:
        provider = FakeProvider(scene_payload())
        frames = stable_frames()
        frames[-1] = Image.new("RGB", (540, 960), (255, 255, 255))
        with self.assertRaisesRegex(VisionAgentError, "稳定性检查未通过"):
            GenericSceneObserver(provider).observe(frames=frames)
        self.assertEqual(provider.calls, 0)

    def test_stable_group_uses_sharpest_frame_instead_of_last_frame(self) -> None:
        observer = GenericSceneObserver(FakeProvider(scene_payload()))
        observer.observe(frames=stable_frames_with_one_sharp_center())
        diagnostics = observer.last_diagnostics
        self.assertEqual(diagnostics["selected_frame_index"], 1)
        scores = diagnostics["frame_sharpness_scores"]
        self.assertEqual(scores[1], max(scores))

    def test_visual_action_field_is_rejected(self) -> None:
        payload = scene_payload()
        payload["action"] = "tap"
        with self.assertRaisesRegex(VisionAgentError, "动作字段"):
            GenericSceneObserver(FakeProvider(payload)).observe(frames=stable_frames())

    def test_goal_context_cannot_smuggle_coordinates(self) -> None:
        with self.assertRaisesRegex(VisionAgentError, "控制字段"):
            GenericSceneObserver(FakeProvider(scene_payload())).observe(
                frames=stable_frames(),
                goal_context={"objective": "打开设置", "x": 50},
            )

    def test_protocol_external_element_field_is_rejected(self) -> None:
        payload = scene_payload()
        payload["elements"][0]["next_action"] = "tap"
        with self.assertRaisesRegex(VisionAgentError, "动作字段"):
            GenericSceneObserver(FakeProvider(payload)).observe(frames=stable_frames())

    def test_home_screen_forces_launcher_even_when_goal_leaks_into_model(self) -> None:
        payload = scene_payload()
        payload["foreground_app_id"] = "douyin"
        payload["screen_id"] = "android_home"
        scene = GenericSceneObserver(FakeProvider(payload)).observe(
            frames=stable_frames(),
            goal_context={"app_id": "douyin", "objective": "打开抖音"},
        )
        self.assertEqual(scene.foreground_app_id, "launcher")

    def test_old_app_id_field_is_accepted_as_compatibility_input(self) -> None:
        payload = scene_payload()
        payload["app_id"] = payload.pop("foreground_app_id")
        scene = GenericSceneObserver(FakeProvider(payload)).observe(
            frames=stable_frames()
        )
        self.assertEqual(scene.foreground_app_id, "calculator")

    def test_invalid_json_gets_one_compact_retry(self) -> None:
        provider = SequenceProvider(["{", scene_payload()])
        observer = GenericSceneObserver(provider)
        scene = observer.observe(frames=stable_frames())
        self.assertEqual(scene.foreground_app_id, "calculator")
        self.assertEqual(provider.calls, 2)
        self.assertEqual(provider.max_tokens_seen, [800, 800])
        self.assertTrue(observer.last_diagnostics["compact_retry_used"])
        self.assertEqual(observer.last_diagnostics["model_calls"], 2)
        self.assertEqual(
            len(observer.last_diagnostics["model_call_elapsed_seconds"]),
            2,
        )
        self.assertGreaterEqual(observer.last_diagnostics["elapsed_seconds"], 0.0)

    def test_bounds_object_retry_prompt_requires_four_number_array(self) -> None:
        invalid = scene_payload()
        invalid["elements"][0]["bounds"] = {
            "x": 100,
            "y": 600,
            "width": 160,
            "height": 160,
        }
        provider = SequenceProvider([invalid, scene_payload()])
        scene = GenericSceneObserver(provider).observe(frames=stable_frames())
        self.assertEqual(scene.elements[0].bounds, (0.1, 0.6, 0.26, 0.76))
        retry_text = provider.messages_seen[1][1]["content"][0]["text"]
        self.assertIn(
            "bounds必须是恰好4个0..1000数值的数组[left,top,right,bottom]",
            retry_text,
        )

    def test_service_disconnect_is_not_misclassified_as_format_retry(self) -> None:
        provider = SequenceProvider(
            [VisionAgentError("千问视觉连接连续1次中断：Server disconnected")]
        )
        observer = GenericSceneObserver(provider)
        with self.assertRaisesRegex(VisionAgentError, "Server disconnected"):
            observer.observe(frames=stable_frames())
        self.assertEqual(provider.calls, 1)
        self.assertFalse(observer.last_diagnostics["format_retry_used"])
        self.assertEqual(observer.last_diagnostics["error_type"], "service_disconnect")
        self.assertEqual(observer.last_diagnostics["model_calls"], 1)
        self.assertIn("未建立可信候选", observer.last_diagnostics["safe_stop_reason"])

    def test_targeted_invalid_json_uses_the_only_format_retry(self) -> None:
        first = scene_payload()
        first["elements"] = []
        first["summary"] = "未知首页"
        refined = scene_payload()
        provider = SequenceProvider([first, "{", refined])
        observer = GenericSceneObserver(provider)
        scene = observer.observe(
            frames=stable_frames(),
            goal_context={"objective": "查找目标按钮"},
        )
        self.assertEqual(scene.elements[0].element_id, "e1")
        self.assertEqual(provider.calls, 3)
        self.assertEqual(provider.max_tokens_seen, [800, 1200, 1200])
        self.assertTrue(observer.last_diagnostics["format_retry_used"])
        self.assertTrue(observer.last_diagnostics["repair_retry_success"])

    def test_observation_never_uses_two_format_repairs(self) -> None:
        first_retry = scene_payload()
        first_retry["elements"] = []
        first_retry["summary"] = "未知首页"
        provider = SequenceProvider(["{", first_retry, "{"])
        observer = GenericSceneObserver(provider)
        with self.assertRaises(VisionAgentError):
            observer.observe(
                frames=stable_frames(),
                goal_context={"objective": "查找目标按钮"},
            )
        self.assertEqual(provider.calls, 3)
        self.assertEqual(provider.max_tokens_seen, [800, 800, 1200])
        self.assertTrue(observer.last_diagnostics["format_retry_used"])
        self.assertFalse(observer.last_diagnostics["repair_retry_success"])

    def test_single_evidence_string_is_normalized_before_strict_validation(self) -> None:
        payload = scene_payload()
        payload["elements"][0]["evidence"] = "7"
        scene = GenericSceneObserver(FakeProvider(payload)).observe(
            frames=stable_frames()
        )
        self.assertEqual(scene.elements[0].evidence, ("7",))

    def test_evidence_object_is_not_silently_repaired(self) -> None:
        payload = scene_payload()
        payload["elements"][0]["evidence"] = {"text": "7"}
        with self.assertRaisesRegex(VisionAgentError, "evidence 必须是数组"):
            GenericSceneObserver(FakeProvider(payload)).observe(frames=stable_frames())

    def test_unsupported_peripheral_structure_is_discarded(self) -> None:
        payload = scene_payload()
        payload["elements"].insert(
            0,
            {
                "element_id": "tabs",
                "role": "tab_group",
                "meaning": "top_channel_group",
                "label": "频道栏",
                "bounds": [100, 0, 900, 60],
                "confidence": 0.96,
                "states": {},
                "evidence": ["推荐"],
            },
        )
        scene = GenericSceneObserver(FakeProvider(payload)).observe(
            frames=stable_frames()
        )
        self.assertEqual([item.element_id for item in scene.elements], ["e1"])

    def test_goal_relevant_container_is_preserved(self) -> None:
        payload = scene_payload()
        payload["elements"][0]["role"] = "container"
        payload["elements"][0]["meaning"] = "video_content"
        payload["elements"][0]["states"] = {"goal_relevant": True}
        scene = GenericSceneObserver(FakeProvider(payload)).observe(
            frames=stable_frames()
        )
        self.assertEqual(scene.elements[0].role, "container")
        self.assertEqual(scene.elements[0].meaning, "video_content")

    def test_unsupported_goal_target_still_stops_controller(self) -> None:
        payload = scene_payload()
        payload["elements"][0]["role"] = "search_icon"
        payload["elements"][0]["states"] = {"goal_relevant": True}
        with self.assertRaisesRegex(VisionAgentError, "不支持的元素角色"):
            GenericSceneObserver(FakeProvider(payload)).observe(frames=stable_frames())

    def test_missing_goal_element_triggers_targeted_refinement(self) -> None:
        first = scene_payload()
        first["foreground_app_id"] = "launcher"
        first["screen_id"] = "android_home"
        first["summary"] = "安卓桌面"
        first["elements"] = []
        refined = dict(first)
        refined["elements"] = [
            {
                "element_id": "e1",
                "role": "icon",
                "meaning": "app_icon",
                "label": "微信",
                "bounds": [200, 700, 340, 850],
                "confidence": 0.96,
                "states": {"goal_relevant": True},
                "evidence": ["微信"],
            }
        ]
        provider = SequenceProvider([first, refined])
        observer = GenericSceneObserver(provider)
        scene = observer.observe(
            frames=stable_frames(),
            goal_context={
                "app_id": "wechat",
                "app_name": "微信",
                "objective": "打开微信",
            },
        )
        self.assertEqual(provider.calls, 2)
        self.assertEqual(provider.max_tokens_seen, [800, 1200])
        self.assertEqual(scene.elements[0].label, "微信")
        self.assertTrue(observer.last_diagnostics["targeted_refinement_used"])
        targeted_text = provider.messages_seen[1][1]["content"][0]["text"]
        self.assertIn("置信度只评价当前画面观察本身是否可靠", targeted_text)
        self.assertIn("系统级动作没有屏内按钮", targeted_text)
        self.assertIn("目标相关控件确实不存在时返回空elements", targeted_text)
        self.assertIn("不能因为目标尚未完成而降低", targeted_text)
        self.assertIn("模糊、遮挡或不唯一时仍必须降低", targeted_text)

    def test_target_app_already_open_does_not_refine_open_goal(self) -> None:
        payload = scene_payload()
        payload["foreground_app_id"] = "wechat"
        payload["screen_id"] = "chat_list"
        provider = SequenceProvider([payload])
        observer = GenericSceneObserver(provider)
        observer.observe(
            frames=stable_frames(),
            goal_context={
                "app_id": "wechat",
                "app_name": "微信",
                "objective": "打开微信",
            },
        )
        self.assertEqual(provider.calls, 1)
        self.assertFalse(observer.last_diagnostics["targeted_refinement_used"])

    def test_high_confidence_goal_element_does_not_refine_only_for_unknown_screen(self) -> None:
        payload = scene_payload()
        payload["foreground_app_id"] = "unknown"
        payload["screen_id"] = "unknown"
        payload["elements"][0]["states"] = {"goal_relevant": True}
        provider = SequenceProvider([payload])
        observer = GenericSceneObserver(provider)

        scene = observer.observe(
            frames=stable_frames(),
            goal_context={
                "app_id": "current_foreground",
                "objective": "让目标进入当前画面",
            },
        )

        self.assertEqual(provider.calls, 1)
        self.assertEqual("unknown", scene.screen_id)
        self.assertFalse(observer.last_diagnostics["targeted_refinement_used"])

    def test_status_exposes_observation_policy(self) -> None:
        status = GenericSceneObserver(FakeProvider(scene_payload())).status()
        self.assertEqual(status["compact_output_tokens"], 800)
        self.assertEqual(status["observation_timeout_seconds"], 60.0)
        self.assertEqual(status["max_compact_elements"], 12)
        self.assertEqual(status["current_stage"], "idle")


if __name__ == "__main__":
    unittest.main()
