from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from generic_intent import GenericIntentDraft
from generic_scene_observer import GenericSceneObserver
from observation_images import measure_local_stability
from semantic_executor import SemanticAction
from ui_scene import UIElement, UIScene, UISceneError
from universal_action_controller import (
    ResolvedSemanticAction,
    UniversalActionController,
    UniversalActionError,
)


class GenericActionAdapterError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        physical_actions: int = 0,
        evidence: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.physical_actions = int(physical_actions)
        self.evidence = tuple(evidence)


@dataclass(frozen=True)
class GenericActionExecutionResult:
    requested_action: SemanticAction
    rebound_action: SemanticAction
    resolved_action: ResolvedSemanticAction
    before_scene: UIScene
    after_scene: UIScene
    physical_actions: int
    robot_result: Any = None
    evidence: tuple[str, ...] = ()
    after_frames: tuple[Image.Image, ...] = field(
        default_factory=tuple,
        repr=False,
        compare=False,
    )
    after_frame_paths: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_action": self.requested_action.to_dict(),
            "rebound_action": self.rebound_action.to_dict(),
            "resolved_action": self.resolved_action.to_dict(),
            "before_scene": self.before_scene.to_dict(),
            "after_scene": self.after_scene.to_dict(),
            "physical_actions": self.physical_actions,
            "robot_result": self.robot_result,
            "evidence": list(self.evidence),
            "after_frame_count": len(self.after_frames),
            "after_frame_paths": list(self.after_frame_paths),
        }


class GenericSingleActionAdapter:
    """The only generic bridge from a verified scene to one robot action."""

    PHYSICAL_KINDS = frozenset({"tap_semantic", "dismiss_overlay", "swipe", "back"})

    def __init__(
        self,
        *,
        capture: Callable[[], Image.Image],
        observer: GenericSceneObserver,
        robot: Any,
        controller: UniversalActionController | None = None,
        frame_interval: float = 0.5,
        post_action_settle: float = 1.5,
        post_action_timeout: float = 10.0,
        post_action_max_observations: int = 2,
    ) -> None:
        self.capture = capture
        self.observer = observer
        self.robot = robot
        self.controller = controller or UniversalActionController()
        self.frame_interval = max(0.0, float(frame_interval))
        self.post_action_settle = max(0.0, float(post_action_settle))
        self.post_action_timeout = max(0.0, float(post_action_timeout))
        self.post_action_max_observations = max(
            1,
            int(post_action_max_observations),
        )

    def _capture_frame(self) -> Image.Image:
        frame = self.capture().convert("RGB")
        if frame.width < 400 or frame.height < 700:
            raise GenericActionAdapterError("摄像头返回残缺画面，停止单步动作。")
        return frame

    def capture_scene(
        self,
        goal: GenericIntentDraft,
        *,
        evidence_dir: Path | None = None,
        prefix: str,
    ) -> tuple[UIScene, list[Image.Image], tuple[str, ...]]:
        frames: list[Image.Image] = []
        for index in range(4):
            frames.append(self._capture_frame())
            if index < 3 and self.frame_interval:
                time.sleep(self.frame_interval)
        paths = self._save_frames(frames, evidence_dir, prefix)
        try:
            scene = self.observer.observe(
                frames=frames,
                goal_context=goal.to_dict(),
            )
        except RuntimeError as exc:
            raise GenericActionAdapterError(
                f"通用页面观察失败：{exc}",
                evidence=paths,
            ) from exc
        return scene, frames, paths

    def _capture_stable_post_action_frames(
        self,
        *,
        deadline: float,
        evidence_dir: Path | None,
        prefix: str,
    ) -> tuple[list[Image.Image], tuple[str, ...]]:
        """Wait for four consecutive locally stable frames within the deadline.

        This gate is deliberately local and cheap.  Qwen is called only after
        the camera's outer/static UI bands have settled, and no physical action
        is ever repeated while waiting.
        """

        frames: list[Image.Image] = []
        last_stability = None
        while True:
            frames.append(self._capture_frame())
            if len(frames) > 4:
                frames.pop(0)
            if len(frames) == 4:
                last_stability = measure_local_stability(frames)
                if last_stability.stable:
                    paths = self._save_frames(frames, evidence_dir, prefix)
                    return list(frames), paths
                if time.monotonic() >= deadline:
                    paths = self._save_frames(
                        frames,
                        evidence_dir,
                        f"{prefix}_timeout",
                    )
                    raise GenericActionAdapterError(
                        "动作后画面在限定时间内没有稳定："
                        f"{last_stability.reason}",
                        evidence=paths,
                    )
            if time.monotonic() >= deadline:
                paths = self._save_frames(
                    frames,
                    evidence_dir,
                    f"{prefix}_timeout",
                )
                reason = (
                    last_stability.reason
                    if last_stability is not None
                    else "未能采集到连续4帧"
                )
                raise GenericActionAdapterError(
                    f"动作后画面在限定时间内没有稳定：{reason}",
                    evidence=paths,
                )
            if self.frame_interval:
                time.sleep(
                    min(
                        self.frame_interval,
                        max(0.0, deadline - time.monotonic()),
                    )
                )

    def _observe_stable_post_action_scene(
        self,
        goal: GenericIntentDraft,
        *,
        before: UIScene,
        resolved: ResolvedSemanticAction,
        evidence_dir: Path | None,
    ) -> tuple[
        UIScene,
        tuple[Image.Image, ...],
        tuple[str, ...],
        tuple[str, ...],
    ]:
        deadline = time.monotonic() + self.post_action_timeout
        if self.post_action_settle:
            time.sleep(min(self.post_action_settle, self.post_action_timeout))

        all_paths: tuple[str, ...] = ()
        last_error: Exception | None = None
        for attempt in range(1, self.post_action_max_observations + 1):
            frames, paths = self._capture_stable_post_action_frames(
                deadline=deadline,
                evidence_dir=evidence_dir,
                prefix=f"after_confirmed_action_attempt_{attempt}",
            )
            all_paths += paths
            try:
                after = self.observer.observe(
                    frames=frames,
                    goal_context=goal.to_dict(),
                )
            except RuntimeError as exc:
                raise GenericActionAdapterError(
                    f"通用页面观察失败：{exc}",
                    evidence=all_paths,
                ) from exc

            try:
                self.controller.verify_after_action(resolved, before, after)
                return after, tuple(frames), paths, all_paths
            except UniversalActionError as exc:
                last_error = exc
                # A stable old page, a low-confidence transitional scene, or
                # an expected destination that is still loading can all be a
                # legitimate intermediate state.  Re-observe at most once,
                # within the same hard deadline, without repeating the action.
                if (
                    attempt >= self.post_action_max_observations
                    or time.monotonic() >= deadline
                ):
                    break
                if self.frame_interval:
                    time.sleep(
                        min(
                            self.frame_interval,
                            max(0.0, deadline - time.monotonic()),
                        )
                    )

        assert last_error is not None
        raise GenericActionAdapterError(
            f"动作后自适应观察仍未通过：{last_error}",
            evidence=all_paths,
        ) from last_error

    def execute(
        self,
        *,
        requested_action: SemanticAction,
        planned_scene: UIScene,
        goal: GenericIntentDraft,
        confirmed: bool,
        evidence_dir: Path | None = None,
    ) -> GenericActionExecutionResult:
        if confirmed is not True:
            raise GenericActionAdapterError("必须明确确认当前这一个语义动作。")
        before, _frames, before_paths = self.capture_scene(
            goal,
            evidence_dir=evidence_dir,
            prefix="before_confirmed_action",
        )
        rebound = self._rebind_action(requested_action, planned_scene, before)
        try:
            resolved = self.controller.resolve_one(
                rebound,
                before,
                confirmed=True,
            )
        except UniversalActionError as exc:
            raise GenericActionAdapterError(
                f"确认前控制器拒绝动作：{exc}",
                evidence=before_paths,
            ) from exc

        if resolved.kind not in self.PHYSICAL_KINDS and resolved.kind != "wait_for_change":
            raise GenericActionAdapterError(
                f"当前通用硬件适配器尚未开放：{resolved.kind}",
                evidence=before_paths,
            )

        physical_actions = 0
        robot_result: Any = None
        try:
            if resolved.kind in {"tap_semantic", "dismiss_overlay"}:
                if resolved.normalized_point is None:
                    raise GenericActionAdapterError("点击动作缺少已校验落点。")
                x = max(0, min(1000, round(resolved.normalized_point[0] * 1000)))
                y = max(0, min(1000, round(resolved.normalized_point[1] * 1000)))
                physical_actions = 1
                robot_result = self.robot.vision_tap_relative(x, y)
            elif resolved.kind == "swipe":
                method = getattr(self.robot, f"vision_swipe_{resolved.direction}", None)
                if not callable(method):
                    raise GenericActionAdapterError(
                        f"机械臂不支持滑动方向：{resolved.direction}"
                    )
                physical_actions = 1
                robot_result = method()
            elif resolved.kind == "back":
                physical_actions = 1
                robot_result = self.robot.vision_android_back()
            elif resolved.kind == "wait_for_change":
                time.sleep(max(0.5, self.post_action_settle))
        except GenericActionAdapterError:
            raise
        except Exception as exc:
            raise GenericActionAdapterError(
                f"机械臂单步动作调用失败：{exc}",
                physical_actions=physical_actions,
                evidence=before_paths,
            ) from exc

        try:
            (
                after,
                after_frames,
                after_frame_paths,
                all_after_paths,
            ) = self._observe_stable_post_action_scene(
                goal,
                before=before,
                resolved=resolved,
                evidence_dir=evidence_dir,
            )
        except (GenericActionAdapterError, UniversalActionError) as exc:
            evidence = before_paths + tuple(getattr(exc, "evidence", ()))
            raise GenericActionAdapterError(
                f"单步动作后验证失败：{exc}",
                physical_actions=physical_actions,
                evidence=evidence,
            ) from exc

        return GenericActionExecutionResult(
            requested_action=requested_action,
            rebound_action=rebound,
            resolved_action=resolved,
            before_scene=before,
            after_scene=after,
            physical_actions=physical_actions,
            robot_result=robot_result,
            evidence=before_paths + all_after_paths,
            after_frames=after_frames,
            after_frame_paths=after_frame_paths,
        )

    def _rebind_action(
        self,
        requested: SemanticAction,
        planned_scene: UIScene,
        fresh_scene: UIScene,
    ) -> SemanticAction:
        planned_app = planned_scene.foreground_app_id
        fresh_app = fresh_scene.foreground_app_id
        if (
            planned_app != "unknown"
            and fresh_app != "unknown"
            and planned_app != fresh_app
        ):
            raise GenericActionAdapterError(
                f"确认时前台 App 已变化：{planned_app} -> {fresh_app}"
            )
        if planned_scene.screen_id != fresh_scene.screen_id:
            raise GenericActionAdapterError(
                f"确认时页面已变化：{planned_scene.screen_id} -> {fresh_scene.screen_id}"
            )
        if requested.action not in {"tap_semantic", "dismiss_overlay"}:
            return requested

        original_id = str(requested.params.get("element_id") or "").strip()
        try:
            original = planned_scene.get_element(original_id)
        except UISceneError as exc:
            raise GenericActionAdapterError(f"原始场景目标无效：{exc}") from exc
        # Model-generated ``meaning`` is descriptive rather than a durable ID.
        # Across two observations the same labelled icon can legitimately be
        # described as, for example, "Douyin app icon" and "Douyin launch
        # entry".  Requiring both label and meaning to be byte-for-byte equal
        # made a correctly re-observed target disappear at confirmation time.
        #
        # Prefer visible label + role + requested states when a label exists;
        # this remains strict and must still resolve to exactly one element.
        # Unlabelled controls keep the previous exact-meaning requirement.
        match_filters = {
            "role": original.role,
            "states": dict(requested.params.get("states") or {}),
        }
        if original.label:
            matches = fresh_scene.find_elements(
                label=original.label,
                **match_filters,
            )
        else:
            matches = fresh_scene.find_elements(
                meaning=original.meaning,
                **match_filters,
            )
        if len(matches) != 1:
            raise GenericActionAdapterError(
                f"确认时目标控件不再唯一：{original.meaning}，匹配{len(matches)}个"
            )
        current = matches[0]
        return SemanticAction(
            node_id=requested.node_id,
            action=requested.action,
            params={
                **requested.params,
                "element_id": current.element_id,
                "target": current.meaning,
                "role": current.role,
                "label": current.label,
            },
        )

    @staticmethod
    def _save_frames(
        frames: list[Image.Image],
        evidence_dir: Path | None,
        prefix: str,
    ) -> tuple[str, ...]:
        if evidence_dir is None:
            return ()
        evidence_dir.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []
        for index, frame in enumerate(frames, start=1):
            path = evidence_dir / f"{prefix}_{index}.jpg"
            frame.save(path, format="JPEG", quality=92)
            paths.append(str(path))
        return tuple(paths)
