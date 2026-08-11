from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from live_semantic_dry_run import (
    ReadOnlySemanticDryRunner,
    RichObservationCapture,
    to_semantic_observation,
)
from semantic_executor import ActionResult, SemanticAction
from target_locator import LocalTargetResolver, TargetResolution
from task_orchestrator import GoalSpec
from vision_agent import VisionAgentError


MIN_OBSERVATION_CONFIDENCE = 0.72
ENABLED_REAL_ACTIONS = frozenset({"ensure_app", "tap_semantic", "swipe"})
ENABLED_READ_ONLY_ACTIONS = frozenset({"observe"})
APP_TARGETS = {
    "douyin": ("douyin_icon", "douyin_"),
    "wechat": ("wechat_icon", "wechat_"),
}


class SemanticActionAdapterError(RuntimeError):
    pass


@dataclass(frozen=True)
class EnsureAppStepResult:
    action_result: ActionResult
    before: dict[str, Any]
    after: dict[str, Any]
    robot_action_called: bool
    command_point: tuple[int, int] | None
    resolution: dict[str, Any] | None
    evidence: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_result": {
                "node_id": self.action_result.node_id,
                "success": self.action_result.success,
                "observation": self.action_result.observation.to_dict(),
                "details": dict(self.action_result.details),
            },
            "before": dict(self.before),
            "after": dict(self.after),
            "robot_action_called": self.robot_action_called,
            "command_point": (
                list(self.command_point) if self.command_point is not None else None
            ),
            "resolution": dict(self.resolution) if self.resolution else None,
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class ObserveStepResult:
    action_result: ActionResult
    observation: dict[str, Any]
    classification: str
    safe_for_next_action: bool
    robot_action_called: bool
    evidence: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_result": {
                "node_id": self.action_result.node_id,
                "success": self.action_result.success,
                "observation": self.action_result.observation.to_dict(),
                "details": dict(self.action_result.details),
            },
            "observation": dict(self.observation),
            "classification": self.classification,
            "safe_for_next_action": self.safe_for_next_action,
            "robot_action_called": self.robot_action_called,
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class TapHeartStepResult:
    action_result: ActionResult
    before: dict[str, Any]
    after: dict[str, Any]
    robot_action_called: bool
    command_point: tuple[int, int] | None
    resolution_before: dict[str, Any] | None
    resolution_after: dict[str, Any] | None
    evidence: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_result": {
                "node_id": self.action_result.node_id,
                "success": self.action_result.success,
                "observation": self.action_result.observation.to_dict(),
                "details": dict(self.action_result.details),
            },
            "before": dict(self.before),
            "after": dict(self.after),
            "robot_action_called": self.robot_action_called,
            "command_point": (
                list(self.command_point) if self.command_point is not None else None
            ),
            "resolution_before": (
                dict(self.resolution_before) if self.resolution_before else None
            ),
            "resolution_after": (
                dict(self.resolution_after) if self.resolution_after else None
            ),
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class SwipeStepResult:
    action_result: ActionResult
    before: dict[str, Any]
    after: dict[str, Any]
    robot_action_called: bool
    evidence: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_result": {
                "node_id": self.action_result.node_id,
                "success": self.action_result.success,
                "observation": self.action_result.observation.to_dict(),
                "details": dict(self.action_result.details),
            },
            "before": dict(self.before),
            "after": dict(self.after),
            "robot_action_called": self.robot_action_called,
            "evidence": list(self.evidence),
        }


class PhysicalActionVerificationError(SemanticActionAdapterError):
    """A physical action happened, but its result could not be safely verified."""

    def __init__(
        self,
        message: str,
        *,
        physical_actions: int,
        evidence: tuple[str, ...] = (),
        command_point: tuple[int, int] | None = None,
    ) -> None:
        super().__init__(message)
        self.physical_actions = int(physical_actions)
        self.evidence = tuple(evidence)
        self.command_point = command_point


class ObserveActionAdapter:
    """Read four camera frames and report state without a hardware capability."""

    def __init__(
        self,
        sensor: ReadOnlySemanticDryRunner,
        *,
        min_confidence: float = MIN_OBSERVATION_CONFIDENCE,
    ) -> None:
        self._sensor = sensor
        self.min_confidence = float(min_confidence)

    def execute(
        self,
        action: SemanticAction,
        goal: GoalSpec,
        *,
        evidence_dir: Path | None = None,
    ) -> ObserveStepResult:
        goal.validate()
        if action.action not in ENABLED_READ_ONLY_ACTIONS:
            raise SemanticActionAdapterError(
                f"只读适配器只允许 observe，拒绝 {action.action}。"
            )
        captured = self._sensor.observe_rich(
            goal,
            mode="semantic_observe",
            executed_actions=[],
            explicitly_forbidden=[
                "tap",
                "swipe",
                "type",
                "send",
                "queue_task",
                "close_overlay",
                "recover",
            ],
        )
        observation = captured.observation
        evidence = EnsureAppActionAdapter._save_frames(
            evidence_dir,
            "observe",
            captured,
        )
        safe = (
            observation.stable is True
            and float(observation.confidence) >= self.min_confidence
            and str(observation.state) not in {"", "unknown"}
        )
        classification = self._classification(observation)
        details = {
            "reason": observation.reason,
            "classification": classification,
            "safe_for_next_action": safe,
            "physical_actions": 0,
            "observation_pipeline": dict(captured.observation_pipeline),
        }
        return ObserveStepResult(
            action_result=ActionResult(
                node_id=action.node_id,
                success=True,
                observation=to_semantic_observation(observation),
                details=details,
            ),
            observation=observation.to_dict(),
            classification=classification,
            safe_for_next_action=safe,
            robot_action_called=False,
            evidence=tuple(evidence),
        )

    @staticmethod
    def _classification(observation: Any) -> str:
        state = str(observation.state)
        if state == "douyin_video":
            return "ordinary_video"
        if state.startswith("douyin_live"):
            return "live"
        if observation.blocking_overlays:
            return "blocking_overlay"
        if (
            state in {"", "unknown"}
            or observation.stable is not True
            or float(observation.confidence) < MIN_OBSERVATION_CONFIDENCE
        ):
            return "unknown"
        return "other"


class TapHeartActionAdapter:
    """Execute one guarded Douyin heart tap and verify it without retrying."""

    def __init__(
        self,
        sensor: ReadOnlySemanticDryRunner,
        tap_relative: Callable[[int, int], tuple[int, int]],
        *,
        target_resolver: LocalTargetResolver | None = None,
        min_confidence: float = MIN_OBSERVATION_CONFIDENCE,
        post_tap_wait: float = 2.0,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        self._sensor = sensor
        self._tap_relative = tap_relative
        self._target_resolver = target_resolver or LocalTargetResolver()
        self.min_confidence = float(min_confidence)
        self.post_tap_wait = max(0.0, float(post_tap_wait))
        self._sleep = sleeper or time.sleep

    def execute(
        self,
        action: SemanticAction,
        goal: GoalSpec,
        *,
        evidence_dir: Path | None = None,
        before_capture: RichObservationCapture | None = None,
    ) -> TapHeartStepResult:
        self._validate_action(action, goal)
        before_capture = before_capture or self._observe(
            goal,
            "tap_heart_before",
            [],
        )
        before = before_capture.observation
        self._require_safe_video(before, stage="点赞前")
        evidence = EnsureAppActionAdapter._save_frames(
            evidence_dir,
            "before",
            before_capture,
        )
        if before.blocking_overlays:
            raise SemanticActionAdapterError("点赞前存在阻塞弹层，禁止点击爱心。")
        if before.heart_state != "unliked":
            raise SemanticActionAdapterError(
                f"视觉观察到爱心状态为 {before.heart_state or 'unknown'}，只允许点击白心。"
            )
        proposed = before.targets.get("heart")
        if not EnsureAppActionAdapter._valid_point(proposed):
            raise SemanticActionAdapterError("稳定视频页没有经过视觉确认的爱心目标。")
        try:
            before_resolution = self._target_resolver.resolve(
                frames=list(before_capture.frames),
                observation=before,
                target="heart",
                proposed=(int(proposed[0]), int(proposed[1])),
                params={},
            )
        except VisionAgentError as exc:
            raise SemanticActionAdapterError(str(exc)) from exc
        if before_resolution.verified_state != "unliked":
            raise SemanticActionAdapterError(
                "本地多帧检测没有独立确认白心，禁止点击，避免取消已有点赞。"
            )

        command_point = self._tap_relative(*before_resolution.resolved_coordinate)
        if self.post_tap_wait:
            self._sleep(self.post_tap_wait)
        try:
            after_capture = self._observe(
                goal,
                "tap_heart_after",
                [
                    {
                        "action": "tap_semantic",
                        "target": "heart",
                        "resolved_coordinate": list(
                            before_resolution.resolved_coordinate
                        ),
                    }
                ],
            )
            after = after_capture.observation
            evidence.extend(
                EnsureAppActionAdapter._save_frames(
                    evidence_dir,
                    "after",
                    after_capture,
                )
            )
            self._require_safe_video(after, stage="点赞后")
            after_proposed = after.targets.get("heart")
            if not EnsureAppActionAdapter._valid_point(after_proposed):
                after_proposed = before_resolution.resolved_coordinate
            after_resolution = self._target_resolver.resolve(
                frames=list(after_capture.frames),
                observation=after,
                target="heart",
                proposed=(int(after_proposed[0]), int(after_proposed[1])),
                params={},
            )
        except (VisionAgentError, SemanticActionAdapterError, RuntimeError) as exc:
            raise PhysicalActionVerificationError(
                f"爱心已点击一次，但点赞后验证失败：{exc}；禁止补点。",
                physical_actions=1,
                evidence=tuple(evidence),
                command_point=command_point,
            ) from exc

        verified = (
            after.heart_state == "liked"
            and after_resolution.verified_state == "liked"
            and not after.blocking_overlays
        )
        details = {
            "reason": (
                "Qwen与本地多帧检测均确认爱心已变红。"
                if verified
                else "点击后未同时获得Qwen红心与本地红心证据；禁止补点。"
            ),
            "target": "heart",
            "before_heart_state": before.heart_state,
            "before_local_heart_state": before_resolution.verified_state,
            "after_heart_state": after.heart_state,
            "after_local_heart_state": after_resolution.verified_state,
            "physical_actions": 1,
            "retry_count": 0,
            "observation_pipeline_before": dict(
                before_capture.observation_pipeline
            ),
            "observation_pipeline_after": dict(
                after_capture.observation_pipeline
            ),
        }
        return TapHeartStepResult(
            action_result=ActionResult(
                node_id=action.node_id,
                success=verified,
                observation=to_semantic_observation(after),
                details=details,
            ),
            before=before.to_dict(),
            after=after.to_dict(),
            robot_action_called=True,
            command_point=command_point,
            resolution_before=before_resolution.to_dict(),
            resolution_after=after_resolution.to_dict(),
            evidence=tuple(evidence),
        )

    def _observe(
        self,
        goal: GoalSpec,
        mode: str,
        executed_actions: list[dict[str, Any]],
    ) -> RichObservationCapture:
        return self._sensor.observe_rich(
            goal,
            mode=mode,
            executed_actions=executed_actions,
            explicitly_forbidden=[
                "swipe",
                "type",
                "send",
                "queue_task",
                "second_tap",
                "retry_tap",
            ],
        )

    def _validate_action(self, action: SemanticAction, goal: GoalSpec) -> None:
        goal.validate()
        if action.action != "tap_semantic":
            raise SemanticActionAdapterError("点赞适配器只允许 tap_semantic。")
        if str(action.params.get("target") or "") != "heart":
            raise SemanticActionAdapterError("本阶段只允许点击抖音爱心。")
        if goal.app_id != "douyin":
            raise SemanticActionAdapterError("爱心点击动作只允许抖音任务。")
        if goal.parameters.get("like") is not True:
            raise SemanticActionAdapterError("任务目标没有明确包含点赞，禁止点击。")

    def _require_safe_video(self, observation: Any, *, stage: str) -> None:
        safe = (
            observation.stable is True
            and float(observation.confidence) >= self.min_confidence
            and str(observation.state) == "douyin_video"
        )
        if not safe:
            raise SemanticActionAdapterError(
                f"{stage}不是稳定的抖音普通视频页，或置信度低于"
                f"{self.min_confidence:.2f}。"
            )


class SwipeUpActionAdapter:
    """Execute one guarded upward navigation gesture and verify page identity.

    This adapter is intentionally narrower than a general swipe capability. It
    may only skip a confirmed Douyin live preview or advertisement. A full live
    room must be left through the separate close/back recovery path.
    """

    SKIPPABLE_STATES = frozenset({
        "douyin_live",  # Legacy name kept for old reviewed observations.
        "douyin_live_preview",
        "douyin_ad",
    })
    COMMON_VISIBLE_TEXTS = frozenset({
        "点击进入直播间",
        "直播中",
        "讲解中",
        "推荐",
        "首页",
        "朋友",
        "消息",
        "我",
    })

    def __init__(
        self,
        sensor: ReadOnlySemanticDryRunner,
        swipe_up: Callable[[], Any],
        *,
        min_confidence: float = MIN_OBSERVATION_CONFIDENCE,
        post_swipe_wait: float = 2.5,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        self._sensor = sensor
        self._swipe_up = swipe_up
        self.min_confidence = float(min_confidence)
        self.post_swipe_wait = max(0.0, float(post_swipe_wait))
        self._sleep = sleeper or time.sleep

    def execute(
        self,
        action: SemanticAction,
        goal: GoalSpec,
        *,
        evidence_dir: Path | None = None,
        before_capture: RichObservationCapture | None = None,
    ) -> SwipeStepResult:
        self._validate_action(action, goal)
        before_capture = before_capture or self._observe(
            goal,
            "swipe_up_before",
            [],
        )
        before = before_capture.observation
        self._require_skippable(before)
        evidence = EnsureAppActionAdapter._save_frames(
            evidence_dir,
            "before",
            before_capture,
        )

        self._swipe_up()
        if self.post_swipe_wait:
            self._sleep(self.post_swipe_wait)
        try:
            after_capture = self._observe(
                goal,
                "swipe_up_after",
                [{"action": "swipe", "direction": "up"}],
            )
            after = after_capture.observation
            evidence.extend(
                EnsureAppActionAdapter._save_frames(
                    evidence_dir,
                    "after",
                    after_capture,
                )
            )
            self._require_safe_douyin(after, stage="上划后")
            if not self._page_identity_changed(before, after):
                raise SemanticActionAdapterError(
                    "上划后没有获得新页面身份（作者、标题或页面指纹）证据。"
                )
        except (SemanticActionAdapterError, RuntimeError) as exc:
            raise PhysicalActionVerificationError(
                f"已经上划一次，但新页面验证失败：{exc}；禁止自动补划。",
                physical_actions=1,
                evidence=tuple(evidence),
            ) from exc

        details = {
            "reason": "上划前确认是可跳过页面，上划后确认进入了不同的抖音页面。",
            "before_state": before.state,
            "after_state": after.state,
            "before_page_identity": list(self._page_identity(before)),
            "after_page_identity": list(self._page_identity(after)),
            "physical_actions": 1,
            "retry_count": 0,
            "observation_pipeline_before": dict(
                before_capture.observation_pipeline
            ),
            "observation_pipeline_after": dict(
                after_capture.observation_pipeline
            ),
        }
        return SwipeStepResult(
            action_result=ActionResult(
                node_id=action.node_id,
                success=True,
                observation=to_semantic_observation(after),
                details=details,
            ),
            before=before.to_dict(),
            after=after.to_dict(),
            robot_action_called=True,
            evidence=tuple(evidence),
        )

    def _observe(
        self,
        goal: GoalSpec,
        mode: str,
        executed_actions: list[dict[str, Any]],
    ) -> RichObservationCapture:
        return self._sensor.observe_rich(
            goal,
            mode=mode,
            executed_actions=executed_actions,
            explicitly_forbidden=[
                "tap",
                "type",
                "send",
                "queue_task",
                "second_swipe",
                "retry_swipe",
                "close_overlay",
                "recover",
            ],
        )

    def _validate_action(self, action: SemanticAction, goal: GoalSpec) -> None:
        goal.validate()
        if action.action != "swipe":
            raise SemanticActionAdapterError("上划适配器只允许 swipe。")
        if str(action.params.get("direction") or "") != "up":
            raise SemanticActionAdapterError("本阶段只允许向上滑动。")
        if goal.app_id != "douyin":
            raise SemanticActionAdapterError("单次上划只允许抖音任务。")

    def _require_skippable(self, observation: Any) -> None:
        self._require_safe_douyin(observation, stage="上划前")
        if str(observation.state) not in self.SKIPPABLE_STATES:
            raise SemanticActionAdapterError(
                "上划前页面不是直播预览或广告；普通视频、完整直播间和未知页面均禁止上划。"
            )
        blocking = set(observation.blocking_overlays)
        # A lone model-reported loading layer is not allowed to deadlock the
        # non-account-effect skip gesture on a stable, explicitly identified
        # live preview/ad.  Qwen has confused Douyin's grey live-preview CTA
        # and translucent video content with a loading mask.  The swipe still
        # has strict post-action page-change verification and is never retried.
        # Every real dialog/permission/unknown overlay remains blocking.
        if blocking and blocking != {"loading"}:
            raise SemanticActionAdapterError("上划前存在阻塞弹层，禁止滑动。")

    def _require_safe_douyin(self, observation: Any, *, stage: str) -> None:
        safe = (
            observation.stable is True
            and float(observation.confidence) >= self.min_confidence
            and str(observation.state).startswith("douyin_")
            and str(observation.state) not in {"douyin_live_room"}
        )
        if not safe:
            raise SemanticActionAdapterError(
                f"{stage}不是稳定、可识别的抖音页面，或置信度低于"
                f"{self.min_confidence:.2f}。"
            )

    @classmethod
    def _page_identity(cls, observation: Any) -> tuple[str, ...]:
        identity: list[str] = []
        fingerprint = str(observation.page_fingerprint or "").strip()
        if fingerprint:
            identity.append(f"fingerprint:{fingerprint}")
        for text in observation.visible_texts:
            value = str(text).strip()
            if value and value not in cls.COMMON_VISIBLE_TEXTS:
                identity.append(f"text:{value}")
        return tuple(dict.fromkeys(identity))

    @classmethod
    def _page_identity_changed(cls, before: Any, after: Any) -> bool:
        before_identity = set(cls._page_identity(before))
        after_identity = set(cls._page_identity(after))
        if not before_identity or not after_identity:
            return False
        return before_identity != after_identity


class EnsureAppActionAdapter:
    """The sole real-action bridge for the generic controller's first phase.

    It owns exactly one hardware capability: a relative tap.  It cannot swipe,
    type, send, queue a task or execute any semantic action other than
    ``ensure_app``.
    """

    def __init__(
        self,
        sensor: ReadOnlySemanticDryRunner,
        tap_relative: Callable[[int, int], tuple[int, int]],
        *,
        target_resolver: LocalTargetResolver | None = None,
        min_confidence: float = MIN_OBSERVATION_CONFIDENCE,
        post_tap_wait: float = 2.0,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self._sensor = sensor
        self._tap_relative = tap_relative
        self._target_resolver = target_resolver or LocalTargetResolver()
        self.min_confidence = float(min_confidence)
        self.post_tap_wait = max(0.0, float(post_tap_wait))
        self._sleep = sleeper

    def execute(
        self,
        action: SemanticAction,
        goal: GoalSpec,
        *,
        evidence_dir: Path | None = None,
        before_capture: RichObservationCapture | None = None,
    ) -> EnsureAppStepResult:
        self._validate_action(action, goal)
        app_id = str(action.params["app_id"])
        target, state_prefix = APP_TARGETS[app_id]
        before_capture = before_capture or self._observe(
            goal,
            "ensure_app_before",
            [],
        )
        before = before_capture.observation
        self._require_safe_observation(before, stage="点击前")
        evidence = self._save_frames(evidence_dir, "before", before_capture)

        if before.state.startswith(state_prefix):
            semantic = to_semantic_observation(before)
            result = ActionResult(
                node_id=action.node_id,
                success=True,
                observation=semantic,
                details={
                    "reason": f"{app_id} 已经打开，无需点击。",
                    "no_op": True,
                    "verified_state": before.state,
                    "observation_pipeline_before": dict(
                        before_capture.observation_pipeline
                    ),
                },
            )
            return EnsureAppStepResult(
                action_result=result,
                before=before.to_dict(),
                after=before.to_dict(),
                robot_action_called=False,
                command_point=None,
                resolution=None,
                evidence=tuple(evidence),
            )

        if before.state != "android_home":
            raise SemanticActionAdapterError(
                f"当前页面是 {before.state}，第一阶段只允许从安卓桌面打开{app_id}。"
            )
        if before.blocking_overlays:
            raise SemanticActionAdapterError(
                "桌面存在阻塞弹层，第一阶段禁止点击 App 图标。"
            )
        proposed = before.targets.get(target)
        if not self._valid_point(proposed):
            raise SemanticActionAdapterError(
                f"稳定桌面中没有经过视觉确认的 {target}，禁止猜测坐标。"
            )

        try:
            resolution = self._target_resolver.resolve(
                frames=list(before_capture.frames),
                observation=before,
                target=target,
                proposed=(int(proposed[0]), int(proposed[1])),
                params={"app_id": app_id},
            )
        except VisionAgentError as exc:
            raise SemanticActionAdapterError(str(exc)) from exc

        command_point = self._tap_relative(*resolution.resolved_coordinate)
        try:
            if self.post_tap_wait:
                self._sleep(self.post_tap_wait)
            after_capture = self._observe(
                goal,
                "ensure_app_after",
                [
                    {
                        "action": "ensure_app",
                        "app_id": app_id,
                        "target": target,
                        "resolved_coordinate": list(
                            resolution.resolved_coordinate
                        ),
                    }
                ],
            )
            after = after_capture.observation
            evidence.extend(
                self._save_frames(evidence_dir, "after", after_capture)
            )
        except (VisionAgentError, SemanticActionAdapterError, RuntimeError) as exc:
            raise PhysicalActionVerificationError(
                f"已点击 {app_id} 图标一次，但打开后复核失败：{exc}；"
                "禁止自动补点。",
                physical_actions=1,
                evidence=tuple(evidence),
                command_point=command_point,
            ) from exc

        verified = self._is_safe(after) and after.state.startswith(state_prefix)
        details = {
            "reason": (
                f"已进入 {after.state}。"
                if verified
                else f"点击后观察到 {after.state}，没有可靠确认已进入{app_id}。"
            ),
            "no_op": False,
            "verified_state": after.state,
            "target": target,
            "resolution_method": resolution.method,
            "observation_pipeline_before": dict(
                before_capture.observation_pipeline
            ),
            "observation_pipeline_after": dict(
                after_capture.observation_pipeline
            ),
        }
        action_result = ActionResult(
            node_id=action.node_id,
            success=verified,
            observation=to_semantic_observation(after),
            details=details,
        )
        return EnsureAppStepResult(
            action_result=action_result,
            before=before.to_dict(),
            after=after.to_dict(),
            robot_action_called=True,
            command_point=command_point,
            resolution=resolution.to_dict(),
            evidence=tuple(evidence),
        )

    def _observe(
        self,
        goal: GoalSpec,
        mode: str,
        executed_actions: list[dict[str, Any]],
    ) -> RichObservationCapture:
        return self._sensor.observe_rich(
            goal,
            mode=mode,
            executed_actions=executed_actions,
            explicitly_forbidden=[
                "swipe",
                "type",
                "send",
                "queue_task",
                "second_tap",
            ],
        )

    def _validate_action(self, action: SemanticAction, goal: GoalSpec) -> None:
        goal.validate()
        if action.action != "ensure_app":
            raise SemanticActionAdapterError(
                f"语义动作 {action.action} 尚未接入实机；当前只允许 ensure_app。"
            )
        app_id = str(action.params.get("app_id") or "").strip()
        if app_id not in APP_TARGETS:
            raise SemanticActionAdapterError("ensure_app 只允许微信或抖音。")
        if app_id != goal.app_id:
            raise SemanticActionAdapterError(
                f"动作目标 {app_id} 与任务目标 {goal.app_id} 不一致。"
            )

    def _require_safe_observation(self, observation: Any, *, stage: str) -> None:
        if not self._is_safe(observation):
            raise SemanticActionAdapterError(
                f"{stage}页面不稳定或置信度低于{self.min_confidence:.2f}，禁止动作。"
            )

    def _is_safe(self, observation: Any) -> bool:
        return (
            observation.stable is True
            and float(observation.confidence) >= self.min_confidence
            and str(observation.state) not in {"", "unknown"}
        )

    @staticmethod
    def _valid_point(point: Any) -> bool:
        return (
            isinstance(point, tuple)
            and len(point) == 2
            and all(isinstance(value, int) and 0 <= value <= 1000 for value in point)
        )

    @staticmethod
    def _save_frames(
        evidence_dir: Path | None,
        prefix: str,
        captured: RichObservationCapture,
    ) -> list[str]:
        if evidence_dir is None:
            return []
        evidence_dir.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []
        for index, frame in enumerate(captured.frames, start=1):
            path = evidence_dir / f"{prefix}_{index}.jpg"
            frame.save(path, format="JPEG", quality=90)
            paths.append(str(path))
        return paths


SemanticStepResult = (
    EnsureAppStepResult
    | ObserveStepResult
    | TapHeartStepResult
    | SwipeStepResult
)


class SemanticActionRouter:
    """Single allow-list entry point for every currently approved adapter.

    The router deliberately contains no fallback and performs no coordinate
    conversion itself.  One semantic action is routed to exactly one adapter;
    unsupported actions stop before observing the camera or touching hardware.
    """

    def __init__(
        self,
        *,
        observe: ObserveActionAdapter,
        ensure_app: EnsureAppActionAdapter,
        tap_heart: TapHeartActionAdapter,
        swipe_up: SwipeUpActionAdapter,
    ) -> None:
        self._observe = observe
        self._ensure_app = ensure_app
        self._tap_heart = tap_heart
        self._swipe_up = swipe_up

    def execute(
        self,
        action: SemanticAction,
        goal: GoalSpec,
        *,
        evidence_dir: Path | None = None,
        before_capture: RichObservationCapture | None = None,
    ) -> SemanticStepResult:
        if action.action == "observe":
            if before_capture is not None:
                raise SemanticActionAdapterError(
                    "observe 必须重新采集页面，禁止复用旧观察。"
                )
            return self._observe.execute(
                action,
                goal,
                evidence_dir=evidence_dir,
            )

        if action.action == "ensure_app":
            return self._ensure_app.execute(
                action,
                goal,
                evidence_dir=evidence_dir,
                before_capture=before_capture,
            )

        if (
            action.action == "tap_semantic"
            and str(action.params.get("target") or "") == "heart"
        ):
            return self._tap_heart.execute(
                action,
                goal,
                evidence_dir=evidence_dir,
                before_capture=before_capture,
            )

        if action.action == "swipe":
            return self._swipe_up.execute(
                action,
                goal,
                evidence_dir=evidence_dir,
                before_capture=before_capture,
            )

        raise SemanticActionAdapterError(
            "语义动作尚未进入统一白名单；当前只允许 observe、ensure_app "
            "、tap_semantic:heart 和受监督 swipe:up。"
        )

    @staticmethod
    def supports(action: SemanticAction) -> bool:
        return bool(
            action.action in {"observe", "ensure_app"}
            or (
                action.action == "swipe"
                and str(action.params.get("direction") or "") == "up"
            )
            or (
                action.action == "tap_semantic"
                and str(action.params.get("target") or "") == "heart"
            )
        )
