from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from PIL import Image

from semantic_executor import (
    PageObservation as SemanticPageObservation,
    SingleStepSemanticExecutor,
)
from state_controller import PageObservation as VisionPageObservation
from task_orchestrator import GoalSpec, TaskPlan


class ObservationOnlyProvider(Protocol):
    def observe(
        self,
        *,
        operation: str,
        params: dict[str, Any],
        frames: list[Image.Image],
        controller_context: dict[str, Any],
    ) -> VisionPageObservation: ...


@dataclass(frozen=True)
class DryRunPreview:
    goal: GoalSpec
    observation: SemanticPageObservation
    decision: dict[str, Any]
    captured_frames: int
    observation_pipeline: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": "real_observation_single_step_dry_run",
            "executed": False,
            "execution_enabled": False,
            "captured_frames": self.captured_frames,
            "observation_pipeline": dict(self.observation_pipeline),
            "goal": self.goal.to_dict(),
            "observation": self.observation.to_dict(),
            "decision": self.decision,
            "safety": {
                "robot_action_called": False,
                "task_created": False,
                "account_action_performed": False,
            },
        }


@dataclass(frozen=True)
class RichObservationCapture:
    observation: VisionPageObservation
    frames: tuple[Image.Image, ...]
    observation_pipeline: dict[str, Any] = field(default_factory=dict)


def to_semantic_observation(
    observed: VisionPageObservation,
) -> SemanticPageObservation:
    """Strip coordinates and keep only semantic evidence for the new controller."""
    active_input = (
        observed.input_text if observed.input_scope == "active_input" else None
    )
    properties: dict[str, Any] = {
        "base_state": observed.base_state,
        "overlays": list(observed.overlays),
        "blocking_overlays": list(observed.blocking_overlays),
        "visible_targets": sorted(observed.targets),
        "visible_texts": list(observed.visible_texts),
        "chat_title": observed.page_title,
        "active_input": active_input,
        "input_is_empty": observed.input_is_empty,
        "input_focused": observed.input_focused,
        "composition_text": observed.composition_text,
        "candidate_text": observed.candidate_text,
        "exact_match_count": observed.exact_match_count,
        "keyboard_visible": observed.keyboard_visible,
        "heart_state": observed.heart_state,
        "selection_count": observed.selection_count,
        "sent_message_visible": observed.sent_message_visible,
        "new_image_visible": observed.new_image_visible,
        "comment_sent": observed.comment_sent_visible,
        "search_query_text": observed.search_query_text,
        "search_query_verified": observed.search_query_verified,
        "search_results_relevant": observed.search_results_relevant,
        "search_result_evidence": list(observed.search_result_evidence),
        "observation_reason": observed.reason,
    }
    # Raw click coordinates and bounds deliberately do not cross this boundary.
    return SemanticPageObservation(
        page_state=observed.state,
        properties=properties,
        confidence=observed.confidence,
        stable=observed.stable,
        frame_id=observed.page_fingerprint,
    )


class ReadOnlySemanticDryRunner:
    """Capture real frames and expose one semantic decision without execution.

    The adapter receives only a frame-capture callable.  It has no reference to
    RobotController and therefore cannot tap, swipe, type, send or queue work.
    """

    def __init__(
        self,
        capture_frame: Callable[[], Image.Image],
        observer: ObservationOnlyProvider,
        *,
        frame_count: int = 4,
        observation_seconds: float = 1.5,
    ) -> None:
        if frame_count < 4:
            raise ValueError("真实页面判断至少需要4帧。")
        if observation_seconds < 0:
            raise ValueError("观察时间不能为负数。")
        self._capture_frame = capture_frame
        self._observer = observer
        self.frame_count = frame_count
        self.observation_seconds = float(observation_seconds)

    def preview(self, goal: GoalSpec, plan: TaskPlan) -> DryRunPreview:
        goal.validate()
        plan.validate()
        captured = self.observe_rich(
            goal,
            mode="real_observation_single_step_dry_run",
            executed_actions=[],
            explicitly_forbidden=[
                "tap",
                "swipe",
                "type",
                "send",
                "queue_task",
            ],
        )
        semantic_observation = to_semantic_observation(captured.observation)
        decision = SingleStepSemanticExecutor(plan).start(semantic_observation)
        return DryRunPreview(
            goal=goal,
            observation=semantic_observation,
            decision=decision.to_dict(),
            captured_frames=len(captured.frames),
            observation_pipeline=dict(captured.observation_pipeline),
        )

    def observe_rich(
        self,
        goal: GoalSpec,
        *,
        mode: str,
        executed_actions: list[dict[str, Any]],
        explicitly_forbidden: list[str],
    ) -> RichObservationCapture:
        """Return the real multi-frame observation without exposing execution."""
        goal.validate()
        frames = self._capture_frames()
        observation = self._observer.observe(
            operation=goal.source_operation,
            params=dict(goal.parameters),
            frames=frames,
            controller_context={
                "mode": mode,
                "executed_actions": list(executed_actions),
                "explicitly_forbidden": list(explicitly_forbidden),
            },
        )
        return RichObservationCapture(
            observation=observation,
            frames=tuple(frames),
            observation_pipeline=dict(
                getattr(self._observer, "last_observation_diagnostics", {}) or {}
            ),
        )

    def _capture_frames(self) -> list[Image.Image]:
        interval = (
            self.observation_seconds / (self.frame_count - 1)
            if self.frame_count > 1
            else 0.0
        )
        frames: list[Image.Image] = []
        for index in range(self.frame_count):
            frame = self._capture_frame()
            if not isinstance(frame, Image.Image):
                raise RuntimeError("摄像头取帧没有返回图像。")
            frame = frame.convert("RGB")
            if frame.width < 200 or frame.height < 300:
                raise RuntimeError(
                    f"摄像头画面尺寸异常：{frame.width}×{frame.height}。"
                )
            frames.append(frame.copy())
            if index + 1 < self.frame_count and interval:
                time.sleep(interval)
        return frames
