from __future__ import annotations

import unittest
from PIL import Image, ImageDraw

from agent.application.qwen_visual_decision import QwenVisualDecisionObserver
from agent.domain.qwen_task_context import QwenTaskContext
from agent.domain.ui_scene import UIScene
from agent.domain.vision_model import VisionAgentError
from agent.infrastructure.observation_images import local_frame_fingerprint
from agent.infrastructure.trusted_observation_frames import (
    build_trusted_observation,
    validate_trusted_observation_against_frames,
)


def patterned_frames() -> list[Image.Image]:
    """Stable offline frames shared by the visual-decision test modules."""
    image = Image.new("RGB", (240, 480), "black")
    draw = ImageDraw.Draw(image)
    for y in range(0, 480, 12):
        for x in range(0, 240, 12):
            if (x // 12 + y // 12) % 2:
                draw.rectangle((x, y, x + 5, y + 5), fill="white")
    return [image.copy() for _ in range(4)]


class StatusOnlyProvider:
    configured = True

    def status(self) -> dict:
        return {"configured": True, "model": "offline-qwen"}


class QwenSameResponseDecisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frames = patterned_frames()
        scene = UIScene(
            app_id="sample.app",
            screen_id="sample_home",
            summary="当前画面用于离线通用动作测试",
            elements=(),
            stable=True,
            confidence=0.98,
            fingerprint=local_frame_fingerprint(self.frames[-1]),
        )
        self.observation = build_trusted_observation(
            frames=self.frames,
            device_id="device-local-01",
            scene=scene,
            observation_id="obs_0123456789abcdef0123456789abcdef",
        )
        self.context = QwenTaskContext(
            task_id="task_same_response_decision",
            device_id="device-local-01",
            revision=1,
            raw_goal="完成当前页面目标",
        )
        self.context.validate()

    def decide(self, payload: dict, *, available_action_kinds: set[str] | None = None):
        observer = QwenVisualDecisionObserver(
            StatusOnlyProvider(),
            trusted_observation_frame_validator=validate_trusted_observation_against_frames,
        )
        decision = observer.decide(
            frames=self.frames,
            task_context=self.context,
            trusted_observation=self.observation,
            available_action_kinds=available_action_kinds,
            model_decision=payload,
        )
        return observer, decision

    def test_tap_uses_qwen_point_without_local_element_binding(self) -> None:
        observer, decision = self.decide(
            {
                "status": "action",
                "action": "tap",
                "point": [500, 500],
                "reason": "当前帧中的目标位置",
            },
            available_action_kinds={"tap_semantic"},
        )

        self.assertEqual("action", decision.proposal.status)
        self.assertEqual("tap_semantic", decision.proposal.action.action)
        self.assertEqual((0.5, 0.5), decision.proposal.action.params["tap_point"])
        self.assertIsNone(decision.proposal.action.params.get("element_id"))
        self.assertEqual(0, observer.last_diagnostics["model_calls"])

    def test_swipe_uses_qwen_start_and_end_without_element_ids(self) -> None:
        _observer, decision = self.decide(
            {
                "status": "action",
                "action": "swipe",
                "start": [500, 730],
                "end": [500, 250],
                "reason": "沿当前画面向上滑动",
            },
            available_action_kinds={"swipe_element"},
        )

        action = decision.proposal.action
        self.assertEqual("swipe_element", action.action)
        self.assertEqual((0.5, 0.73), action.params["start"])
        self.assertEqual((0.5, 0.25), action.params["end"])
        self.assertIsNone(action.params.get("element_id"))

    def test_finish_uses_current_observation_reason_only(self) -> None:
        _observer, decision = self.decide(
            {
                "status": "finish",
                "reason": "当前截图已经证明任务完成",
            }
        )

        self.assertEqual("finish", decision.proposal.status)
        self.assertIsNone(decision.proposal.action)
        self.assertEqual("当前截图已经证明任务完成", decision.proposal.reason)

    def test_old_element_protocol_is_rejected(self) -> None:
        with self.assertRaisesRegex(VisionAgentError, "协议外字段"):
            self.decide(
                {
                    "status": "action",
                    "action": "tap",
                    "point": [500, 500],
                    "target": {"element_id": "old-target"},
                },
                available_action_kinds={"tap_semantic"},
            )


if __name__ == "__main__":
    unittest.main()
