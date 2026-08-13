from __future__ import annotations

import time
import re
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageChops, ImageStat

from generic_intent import GenericIntentDraft
from generic_scene_observer import GenericSceneObserver
from observation_images import measure_local_stability
from qwen_runtime_errors import FORMAT_ERROR_TYPES, classify_qwen_error
from semantic_executor import SemanticAction
from ui_scene import UIElement, UIScene, UISceneError
from universal_action_controller import (
    ResolvedSemanticAction,
    UniversalActionController,
    UniversalActionError,
    navigation_semantic_class,
)


class GenericActionAdapterError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        physical_actions: int = 0,
        evidence: tuple[str, ...] = (),
        observation_errors: tuple[str, ...] = (),
        verification_errors: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.physical_actions = int(physical_actions)
        self.evidence = tuple(evidence)
        self.observation_errors = tuple(observation_errors)
        self.verification_errors = tuple(verification_errors)


@dataclass(frozen=True)
class GenericActionExecutionResult:
    requested_action: SemanticAction
    rebound_action: SemanticAction
    resolved_action: ResolvedSemanticAction
    before_scene: UIScene
    after_scene: UIScene
    physical_actions: int
    action_outcome: str = "matched"
    verification_errors: tuple[str, ...] = ()
    robot_result: Any = None
    evidence: tuple[str, ...] = ()
    after_frames: tuple[Image.Image, ...] = field(
        default_factory=tuple,
        repr=False,
        compare=False,
    )
    after_frame_paths: tuple[str, ...] = ()
    observation_errors: tuple[str, ...] = ()
    before_frames: tuple[Image.Image, ...] = field(
        default_factory=tuple,
        repr=False,
        compare=False,
    )
    before_frame_paths: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_action": self.requested_action.to_dict(),
            "rebound_action": self.rebound_action.to_dict(),
            "resolved_action": self.resolved_action.to_dict(),
            "before_scene": self.before_scene.to_dict(),
            "after_scene": self.after_scene.to_dict(),
            "physical_actions": self.physical_actions,
            "action_outcome": self.action_outcome,
            "verification_errors": list(self.verification_errors),
            "robot_result": self.robot_result,
            "evidence": list(self.evidence),
            "after_frame_count": len(self.after_frames),
            "after_frame_paths": list(self.after_frame_paths),
            "observation_errors": list(self.observation_errors),
            "before_frame_count": len(self.before_frames),
            "before_frame_paths": list(self.before_frame_paths),
        }


class GenericSingleActionAdapter:
    """The only generic bridge from a verified scene to one robot action."""

    PHYSICAL_KINDS = frozenset(
        {
            "tap_semantic",
            "dismiss_overlay",
            "swipe",
            "back",
            "input_verified_text",
            "long_press",
            "drag",
        }
    )

    def supported_action_kinds(self) -> frozenset[str]:
        """Return only actions backed by callable methods on this device."""

        capability_provider = getattr(self.robot, "hardware_capabilities", None)
        declared = capability_provider() if callable(capability_provider) else {}
        if not isinstance(declared, dict):
            declared = {}

        def available(action: str, method_name: str) -> bool:
            return callable(getattr(self.robot, method_name, None)) and bool(
                declared.get(action, True)
            )

        supported = (
            {"wait_for_change"}
            if bool(declared.get("wait_for_change", True))
            else set()
        )
        if available("tap_semantic", "vision_tap_relative"):
            supported.add("tap_semantic")
        if available("dismiss_overlay", "vision_dismiss_overlay_relative"):
            supported.add("dismiss_overlay")
        if bool(declared.get("swipe", True)) and any(
            callable(getattr(self.robot, f"vision_swipe_{direction}", None))
            for direction in ("up", "down", "left", "right")
        ):
            supported.add("swipe")
        if available("back", "vision_android_back"):
            supported.add("back")
        if available("input_verified_text", "vision_type_text"):
            supported.add("input_verified_text")
        if available("long_press", "vision_long_press_relative"):
            supported.add("long_press")
        if available("drag", "vision_drag_relative"):
            supported.add("drag")
        return frozenset(supported)

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
        confirmation_frame_delta_max: float = 6.0,
    ) -> None:
        self.capture = capture
        self.observer = observer
        self.robot = robot
        self.controller = controller or UniversalActionController()
        self.frame_interval = max(0.0, float(frame_interval))
        self.post_action_settle = max(0.0, float(post_action_settle))
        self.post_action_timeout = max(0.0, float(post_action_timeout))
        self.post_action_max_observations = min(
            2,
            max(1, int(post_action_max_observations)),
        )
        self.confirmation_frame_delta_max = max(
            0.0,
            float(confirmation_frame_delta_max),
        )

    @staticmethod
    def _confirmation_frame_delta(
        planned_frames: tuple[Image.Image, ...] | list[Image.Image],
        fresh_frames: list[Image.Image],
    ) -> float:
        if not planned_frames or not fresh_frames:
            raise GenericActionAdapterError("确认前缺少本地真实帧，不能验证画面身份。")
        planned_sizes = {frame.size for frame in planned_frames}
        fresh_sizes = {frame.size for frame in fresh_frames}
        if len(planned_sizes) != 1 or len(fresh_sizes) != 1 or planned_sizes != fresh_sizes:
            raise GenericActionAdapterError("确认前真实画面尺寸发生变化。")

        def compact(frame: Image.Image) -> Image.Image:
            return frame.convert("L").resize((96, 160), Image.Resampling.BILINEAR)

        planned = [compact(frame) for frame in planned_frames]
        fresh = [compact(frame) for frame in fresh_frames]
        return min(
            float(ImageStat.Stat(ImageChops.difference(first, second)).mean[0])
            for first in planned
            for second in fresh
        )

    def _capture_frame(self) -> Image.Image:
        frame = self.capture().convert("RGB")
        if frame.width < 400 or frame.height < 700:
            raise GenericActionAdapterError("摄像头返回残缺画面，停止单步动作。")
        return frame

    def _capture_confirmation_frames(
        self,
        *,
        evidence_dir: Path | None,
        prefix: str,
    ) -> tuple[list[Image.Image], tuple[str, ...]]:
        """Capture four fresh local frames without re-interpreting the scene."""

        frames: list[Image.Image] = []
        for index in range(4):
            frames.append(self._capture_frame())
            if index < 3 and self.frame_interval:
                time.sleep(self.frame_interval)
        stability = measure_local_stability(frames)
        paths = self._save_frames(frames, evidence_dir, prefix)
        if not stability.stable:
            raise GenericActionAdapterError(
                f"确认前本地多帧稳定性检查未通过：{stability.reason}",
                evidence=paths,
            )
        return frames, paths

    def _capture_scene_once(
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

    def capture_scene(
        self,
        goal: GenericIntentDraft,
        *,
        evidence_dir: Path | None,
        prefix: str,
    ) -> tuple[UIScene, list[Image.Image], tuple[str, ...]]:
        all_paths: tuple[str, ...] = ()
        errors: list[str] = []
        for attempt in range(1, 3):
            try:
                scene, frames, paths = self._capture_scene_once(
                    goal,
                    evidence_dir=evidence_dir,
                    prefix=f"{prefix}_attempt_{attempt}",
                )
                return scene, frames, all_paths + paths
            except GenericActionAdapterError as exc:
                all_paths += tuple(exc.evidence)
                errors.append(f"第{attempt}轮动作前观察失败：{exc}")
                if attempt >= 2 or not self._pre_action_observation_retryable(exc):
                    raise GenericActionAdapterError(
                        "动作前通用页面观察失败：" + "；".join(errors),
                        evidence=all_paths,
                        observation_errors=tuple(errors),
                    ) from exc
        raise AssertionError("unreachable")

    def _pre_action_observation_retryable(self, error: Exception) -> bool:
        text = str(error)
        return self._post_observation_retryable(error) or any(
            marker in text
            for marker in (
                "页面不稳定",
                "画面不稳定",
                "整体置信度不足",
                "不能建立可信候选",
            )
        )

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
        evidence_prefix: str,
    ) -> tuple[
        UIScene,
        tuple[Image.Image, ...],
        tuple[str, ...],
        tuple[str, ...],
        tuple[str, ...],
        tuple[str, ...],
    ]:
        if self.post_action_settle:
            time.sleep(min(self.post_action_settle, self.post_action_timeout))

        all_paths: tuple[str, ...] = ()
        observation_errors: list[str] = []
        verification_errors: list[str] = []
        last_error: Exception | None = None
        for attempt in range(1, self.post_action_max_observations + 1):
            attempt_deadline = time.monotonic() + self.post_action_timeout
            try:
                frames, paths = self._capture_stable_post_action_frames(
                    deadline=attempt_deadline,
                    evidence_dir=evidence_dir,
                    prefix=f"{evidence_prefix}_after_attempt_{attempt}",
                )
            except GenericActionAdapterError as exc:
                capture_evidence = all_paths + tuple(exc.evidence)
                prior_errors = (
                    "；此前" + "；".join(observation_errors + verification_errors)
                    if observation_errors or verification_errors
                    else ""
                )
                raise GenericActionAdapterError(
                    f"动作后画面采集失败：{exc}{prior_errors}",
                    evidence=capture_evidence,
                    observation_errors=tuple(observation_errors),
                    verification_errors=tuple(verification_errors),
                ) from exc
            all_paths += paths
            try:
                after = self.observer.observe(
                    frames=frames,
                    goal_context=goal.to_dict(),
                )
            except RuntimeError as exc:
                last_error = exc
                observation_errors.append(
                    f"第{attempt}轮动作后观察失败：{exc}"
                )
                if (
                    attempt >= self.post_action_max_observations
                    or not self._post_observation_retryable(exc)
                ):
                    raise GenericActionAdapterError(
                        "通用页面观察失败："
                        + "；".join(verification_errors + observation_errors),
                        evidence=all_paths,
                        observation_errors=tuple(observation_errors),
                        verification_errors=tuple(verification_errors),
                    ) from exc
                continue

            try:
                self.controller.verify_after_action(resolved, before, after)
                return (
                    after,
                    tuple(frames),
                    paths,
                    all_paths,
                    tuple(observation_errors),
                    tuple(verification_errors),
                )
            except UniversalActionError as exc:
                last_error = exc
                verification_errors.append(
                    f"第{attempt}轮动作结果不匹配：{exc}"
                )
                # A stable old page, a low-confidence transitional scene, or
                # an expected destination that is still loading can all be a
                # legitimate intermediate state.  Re-observe at most once,
                # with a fresh bounded capture, without repeating the action.
                if attempt >= self.post_action_max_observations:
                    break
                if self.frame_interval:
                    time.sleep(self.frame_interval)

        assert last_error is not None
        return (
            after,
            tuple(frames),
            paths,
            all_paths,
            tuple(observation_errors),
            tuple(verification_errors),
        )

    def _post_observation_retryable(self, error: Exception) -> bool:
        diagnostics = getattr(self.observer, "last_diagnostics", {})
        error_type = (
            diagnostics.get("error_type")
            if isinstance(diagnostics, dict)
            else None
        )
        return (
            error_type in FORMAT_ERROR_TYPES
            or classify_qwen_error(error) in FORMAT_ERROR_TYPES
        )

    def execute(
        self,
        *,
        requested_action: SemanticAction,
        planned_scene: UIScene,
        goal: GenericIntentDraft,
        confirmed: bool,
        evidence_dir: Path | None = None,
        planned_frames: tuple[Image.Image, ...] | list[Image.Image] = (),
    ) -> GenericActionExecutionResult:
        if confirmed is not True:
            raise GenericActionAdapterError("必须明确确认当前这一个语义动作。")
        safe_node = re.sub(r"[^a-zA-Z0-9_-]+", "_", requested_action.node_id)[:48]
        evidence_prefix = f"{safe_node or 'action'}_{uuid.uuid4().hex}"
        local_frame_identity_verified = False
        if planned_frames:
            before_frames, before_paths = self._capture_confirmation_frames(
                evidence_dir=evidence_dir,
                prefix=f"{evidence_prefix}_before",
            )
            frame_delta = self._confirmation_frame_delta(planned_frames, before_frames)
            if frame_delta > self.confirmation_frame_delta_max:
                raise GenericActionAdapterError(
                    "确认时本地真实画面已变化："
                    f"差异{frame_delta:.2f}超过阈值{self.confirmation_frame_delta_max:.2f}",
                    evidence=before_paths,
                )
            local_frame_identity_verified = True
            before = planned_scene
        else:
            before, before_frames, before_paths = self.capture_scene(
                goal,
                evidence_dir=evidence_dir,
                prefix=f"{evidence_prefix}_before",
            )
        try:
            rebound = self._rebind_action(
                requested_action,
                planned_scene,
                before,
                local_frame_identity_verified=local_frame_identity_verified,
            )
        except GenericActionAdapterError as exc:
            raise GenericActionAdapterError(
                str(exc),
                evidence=before_paths + tuple(getattr(exc, "evidence", ())),
            ) from exc
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
                if resolved.kind == "dismiss_overlay":
                    robot_result = self.robot.vision_dismiss_overlay_relative(x, y)
                else:
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
            elif resolved.kind == "input_verified_text":
                if not resolved.text:
                    raise GenericActionAdapterError("输入动作缺少已校验文字。")
                method = getattr(self.robot, "vision_type_text", None)
                if not callable(method):
                    raise GenericActionAdapterError("机械臂不支持经过验证的文字输入。")
                physical_actions = 1
                robot_result = method(resolved.text)
            elif resolved.kind == "long_press":
                if resolved.normalized_point is None or resolved.hold_seconds is None:
                    raise GenericActionAdapterError("长按动作缺少已校验落点或时长。")
                method = getattr(self.robot, "vision_long_press_relative", None)
                if not callable(method):
                    raise GenericActionAdapterError("机械臂不支持通用长按。")
                x = max(0, min(1000, round(resolved.normalized_point[0] * 1000)))
                y = max(0, min(1000, round(resolved.normalized_point[1] * 1000)))
                physical_actions = 1
                robot_result = method(x, y, resolved.hold_seconds)
            elif resolved.kind == "drag":
                if (
                    resolved.normalized_point is None
                    or resolved.normalized_end_point is None
                ):
                    raise GenericActionAdapterError("拖动动作缺少已校验起点或终点。")
                method = getattr(self.robot, "vision_drag_relative", None)
                if not callable(method):
                    raise GenericActionAdapterError(
                        "当前机械臂控制端没有经过验收的任意拖动能力。"
                    )
                start_x = max(
                    0, min(1000, round(resolved.normalized_point[0] * 1000))
                )
                start_y = max(
                    0, min(1000, round(resolved.normalized_point[1] * 1000))
                )
                end_x = max(
                    0, min(1000, round(resolved.normalized_end_point[0] * 1000))
                )
                end_y = max(
                    0, min(1000, round(resolved.normalized_end_point[1] * 1000))
                )
                physical_actions = 1
                robot_result = method(start_x, start_y, end_x, end_y)
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
                observation_errors,
                verification_errors,
            ) = self._observe_stable_post_action_scene(
                goal,
                before=before,
                resolved=resolved,
                evidence_dir=evidence_dir,
                evidence_prefix=evidence_prefix,
            )
        except (GenericActionAdapterError, UniversalActionError) as exc:
            evidence = before_paths + tuple(getattr(exc, "evidence", ()))
            raise GenericActionAdapterError(
                f"单步动作后验证失败：{exc}",
                physical_actions=physical_actions,
                evidence=evidence,
                observation_errors=tuple(
                    getattr(exc, "observation_errors", ())
                ),
                verification_errors=tuple(
                    getattr(exc, "verification_errors", ())
                ),
            ) from exc

        return GenericActionExecutionResult(
            requested_action=requested_action,
            rebound_action=rebound,
            resolved_action=resolved,
            before_scene=before,
            after_scene=after,
            physical_actions=physical_actions,
            action_outcome=(
                "mismatched" if verification_errors else "matched"
            ),
            verification_errors=verification_errors,
            robot_result=robot_result,
            evidence=before_paths + all_after_paths,
            after_frames=after_frames,
            after_frame_paths=after_frame_paths,
            observation_errors=observation_errors,
            before_frames=before_frames,
            before_frame_paths=before_paths,
        )

    def _rebind_action(
        self,
        requested: SemanticAction,
        planned_scene: UIScene,
        fresh_scene: UIScene,
        *,
        local_frame_identity_verified: bool = False,
    ) -> SemanticAction:
        planned_app = planned_scene.foreground_app_id
        fresh_app = fresh_scene.foreground_app_id
        if (
            not local_frame_identity_verified
            and
            planned_app != "unknown"
            and fresh_app != "unknown"
            and planned_app != fresh_app
        ):
            raise GenericActionAdapterError(
                f"确认时前台 App 已变化：{planned_app} -> {fresh_app}"
            )
        if (
            not local_frame_identity_verified
            and
            planned_scene.screen_id != "unknown"
            and planned_scene.screen_id != fresh_scene.screen_id
        ):
            raise GenericActionAdapterError(
                f"确认时页面已变化：{planned_scene.screen_id} -> {fresh_scene.screen_id}"
            )
        single_element_actions = {
            "tap_semantic",
            "dismiss_overlay",
            "input_verified_text",
            "long_press",
        }
        if requested.action not in single_element_actions | {"drag"}:
            return requested

        def rebind_element(prefix: str = "") -> UIElement:
            original_id = str(
                requested.params.get(f"{prefix}element_id") or ""
            ).strip()
            try:
                original = planned_scene.get_element(original_id)
            except UISceneError as exc:
                raise GenericActionAdapterError(f"原始场景目标无效：{exc}") from exc
            matches = fresh_scene.find_elements(
                label=original.label or None,
                role=original.role,
                states=dict(original.states),
            )
            if len(matches) != 1:
                raise GenericActionAdapterError(
                    "确认时目标语义不再严格唯一："
                    f"{original.meaning}，匹配{len(matches)}个"
                )
            current = matches[0]
            if current.meaning.casefold() != original.meaning.casefold():
                original_class = navigation_semantic_class(
                    original.meaning,
                    original.label,
                )
                current_class = navigation_semantic_class(
                    current.meaning,
                    current.label,
                )
                if (
                    not original_class
                    or original_class == "forbidden"
                    or current_class != original_class
                ):
                    raise GenericActionAdapterError(
                        "确认时目标语义已经变化，旧确认失效。"
                    )
            if current.label != original.label or current.states != original.states:
                raise GenericActionAdapterError(
                    "确认时目标标签或状态已经变化，旧确认失效。"
                )
            left = max(original.bounds[0], current.bounds[0])
            top = max(original.bounds[1], current.bounds[1])
            right = min(original.bounds[2], current.bounds[2])
            bottom = min(original.bounds[3], current.bounds[3])
            intersection = max(0.0, right - left) * max(0.0, bottom - top)
            original_area = max(
                0.0, original.bounds[2] - original.bounds[0]
            ) * max(0.0, original.bounds[3] - original.bounds[1])
            current_area = max(
                0.0, current.bounds[2] - current.bounds[0]
            ) * max(0.0, current.bounds[3] - current.bounds[1])
            union = original_area + current_area - intersection
            overlap = intersection / union if union > 0 else 0.0
            if overlap < 0.60:
                raise GenericActionAdapterError(
                    "确认时目标区域已明显移动，旧确认失效。"
                )
            return current

        prefixes = ("source_", "destination_") if requested.action == "drag" else ("",)
        params = dict(requested.params)
        for prefix in prefixes:
            current = rebind_element(prefix)
            params.update(
                {
                    f"{prefix}element_id": current.element_id,
                    f"{prefix}target": current.meaning,
                    f"{prefix}role": current.role,
                    f"{prefix}label": current.label,
                    f"{prefix}states": dict(current.states),
                }
            )
        return SemanticAction(
            node_id=requested.node_id,
            action=requested.action,
            params=params,
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
