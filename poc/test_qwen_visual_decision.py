from __future__ import annotations

import json
import unittest
from pathlib import Path

from PIL import Image

from qwen_visual_decision import QwenVisualDecisionObserver
from vision_agent import VisionAgentError


ROOT = Path(__file__).resolve().parent
EXISTING_SCREENSHOT = (
    ROOT / "evals" / "vision_replay" / "images" / "douyin_digit_local_input_com.jpg"
)


class FakeProvider:
    configured = True

    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls = 0
        self.messages: list[dict] = []

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
        self.options = {
            "max_tokens": max_tokens,
            "timeout": timeout,
            "max_attempts": max_attempts,
        }
        return json.dumps(self.payload, ensure_ascii=False)


class SequenceProvider(FakeProvider):
    def __init__(self, payloads: list[dict]) -> None:
        super().__init__({})
        self.payloads = list(payloads)

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
        self.options = {
            "max_tokens": max_tokens,
            "timeout": timeout,
            "max_attempts": max_attempts,
        }
        return json.dumps(self.payloads.pop(0), ensure_ascii=False)


def action_payload() -> dict:
    return {
        "protocol_version": "2026-08-11-qwen-visual-decision-v1",
        "device_id": "offline_device_01",
        "page_state": {
            "foreground_app_id": "douyin",
            "screen_id": "comment_editor",
            "summary": "评论输入框显示.com，符号键盘已展开",
            "elements": [
                {
                    "element_id": "e_send",
                    "role": "button",
                    "meaning": "send_comment",
                    "label": "发送",
                    "bounds": [780, 438, 925, 515],
                    "confidence": 0.96,
                    "states": {"goal_relevant": True, "enabled": True},
                    "evidence": ["发送"],
                }
            ],
            "overlays": ["comment_editor"],
            "stable": True,
            "confidence": 0.94,
            "fingerprint": "model-fingerprint-is-ignored",
        },
        "status": "action",
        "next_action": {
            "kind": "tap_semantic",
            "element_id": "e_send",
            "target": "send_comment",
            "role": "button",
            "label": "发送",
            "states": {"enabled": True},
        },
        "target_region": {
            "kind": "element",
            "element_id": "e_send",
            "bounds": [780, 438, 925, 515],
            "description": "右侧红色发送按钮",
        },
        "expected_result": {"scene_changed": True, "screen_id": "video"},
        "confidence": 0.93,
        "reason": "发送按钮清晰且唯一。",
        "completion_evidence": [],
    }


def existing_screenshot_frames() -> list[Image.Image]:
    with Image.open(EXISTING_SCREENSHOT) as image:
        frame = image.convert("RGB")
    return [frame.copy() for _ in range(4)]


class QwenVisualDecisionTests(unittest.TestCase):
    def test_existing_screenshot_produces_one_bound_action(self) -> None:
        provider = FakeProvider(action_payload())
        observer = QwenVisualDecisionObserver(provider)
        decision = observer.decide(
            frames=existing_screenshot_frames(),
            device_id="offline_device_01",
            current_subgoal={
                "objective": "评论输入框内容确认后，点击唯一发送按钮",
            },
            constraints=["只提出一个动作"],
        )

        self.assertEqual(provider.calls, 1)
        self.assertEqual(decision.proposal.action.action, "tap_semantic")
        self.assertEqual(decision.target_region.element_id, "e_send")
        self.assertEqual(
            decision.target_region.bounds,
            decision.page_state.get_element("e_send").bounds,
        )
        self.assertEqual(decision.expected_result["screen_id"], "video")
        self.assertAlmostEqual(decision.confidence, 0.93)
        self.assertTrue(
            provider.messages[0]["content"][1]["image_url"]["url"].startswith(
                "data:image/jpeg;base64,"
            )
        )
        self.assertFalse(observer.status()["hardware_actions_enabled"])

    def test_region_not_bound_to_element_is_rejected(self) -> None:
        payload = action_payload()
        payload["target_region"]["bounds"] = [790, 450, 900, 500]
        with self.assertRaisesRegex(VisionAgentError, "复用当前页面元素 bounds"):
            QwenVisualDecisionObserver(FakeProvider(payload)).decide(
                frames=existing_screenshot_frames(),
                device_id="offline_device_01",
                current_subgoal={"objective": "点击发送"},
                constraints=[],
            )

    def test_action_below_confidence_threshold_must_block(self) -> None:
        payload = action_payload()
        payload["confidence"] = 0.60
        with self.assertRaisesRegex(VisionAgentError, "必须返回 blocked"):
            QwenVisualDecisionObserver(FakeProvider(payload)).decide(
                frames=existing_screenshot_frames(),
                device_id="offline_device_01",
                current_subgoal={"objective": "点击发送"},
                constraints=[],
            )

    def test_action_must_copy_visible_element_fields(self) -> None:
        payload = action_payload()
        payload["next_action"]["label"] = "近似发送"
        with self.assertRaisesRegex(VisionAgentError, "逐字复制目标元素 label"):
            QwenVisualDecisionObserver(FakeProvider(payload)).decide(
                frames=existing_screenshot_frames(),
                device_id="offline_device_01",
                current_subgoal={"objective": "点击发送"},
                constraints=[],
            )

    def test_output_device_id_must_match_input(self) -> None:
        payload = action_payload()
        payload["device_id"] = "other_device"
        with self.assertRaisesRegex(VisionAgentError, "device_id不匹配"):
            QwenVisualDecisionObserver(FakeProvider(payload)).decide(
                frames=existing_screenshot_frames(),
                device_id="offline_device_01",
                current_subgoal={"objective": "点击发送"},
                constraints=[],
            )

    def test_blocked_response_discards_malformed_non_action_elements(self) -> None:
        payload = action_payload()
        payload.update(
            {
                "status": "blocked",
                "next_action": None,
                "target_region": None,
                "expected_result": {},
                "confidence": 0.4,
                "reason": "没有逐字匹配的可见候选词。",
            }
        )
        payload["page_state"]["elements"] = [
            {
                "element_id": "candidate",
                "role": "candidate_text",
                "meaning": "candidate",
                "label": "近似词",
                "bounds": [100, 100, 300, 160],
                "confidence": 0.9,
                "states": ["visible"],
                "evidence": ["近似词"],
            }
        ]
        decision = QwenVisualDecisionObserver(FakeProvider(payload)).decide(
            frames=existing_screenshot_frames(),
            device_id="offline_device_01",
            current_subgoal={"objective": "选择逐字相同的目标词"},
            constraints=["禁止近似替代"],
        )
        self.assertEqual(decision.proposal.status, "blocked")
        self.assertEqual(decision.page_state.elements, ())

    def test_finished_requires_visible_evidence_and_no_action(self) -> None:
        payload = action_payload()
        payload.update(
            {
                "status": "finished",
                "next_action": None,
                "target_region": None,
                "expected_result": {},
                "completion_evidence": ["输入框清晰显示.com"],
                "reason": "当前画面已满足子目标。",
            }
        )
        decision = QwenVisualDecisionObserver(FakeProvider(payload)).decide(
            frames=existing_screenshot_frames(),
            device_id="offline_device_01",
            current_subgoal={"objective": "确认输入框准确显示.com"},
            constraints=["不要发送"],
        )
        self.assertEqual(decision.proposal.status, "finished")
        self.assertIsNone(decision.proposal.action)
        self.assertEqual(
            decision.proposal.completion_evidence,
            ("输入框清晰显示.com",),
        )

    def test_one_protocol_retry_repairs_finished_target_region(self) -> None:
        invalid = action_payload()
        invalid.update(
            {
                "status": "finished",
                "next_action": None,
                "completion_evidence": ["输入框显示.com"],
            }
        )
        repaired = dict(invalid)
        repaired["target_region"] = None
        repaired["expected_result"] = {}
        provider = SequenceProvider([invalid, repaired])
        observer = QwenVisualDecisionObserver(provider)
        decision = observer.decide(
            frames=existing_screenshot_frames(),
            device_id="offline_device_01",
            current_subgoal={"objective": "确认输入框显示.com"},
            constraints=["不要发送"],
        )
        self.assertEqual(decision.proposal.status, "finished")
        self.assertEqual(provider.calls, 2)
        self.assertTrue(observer.last_diagnostics["protocol_retry_used"])

    def test_unstable_frames_stop_before_qwen_call(self) -> None:
        provider = FakeProvider(action_payload())
        frames = existing_screenshot_frames()
        frames[-1] = Image.new("RGB", frames[-1].size, "white")
        with self.assertRaisesRegex(VisionAgentError, "稳定性检查未通过"):
            QwenVisualDecisionObserver(provider).decide(
                frames=frames,
                device_id="offline_device_01",
                current_subgoal={"objective": "点击发送"},
                constraints=[],
            )
        self.assertEqual(provider.calls, 0)


if __name__ == "__main__":
    unittest.main()
