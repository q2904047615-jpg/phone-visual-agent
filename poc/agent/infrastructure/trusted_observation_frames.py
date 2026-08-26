"""PIL-backed construction and freshness checks for trusted observations."""

from __future__ import annotations

import os
import uuid

from PIL import Image

from agent.domain.qwen_task_context import DEVICE_ID_PATTERN
from agent.domain.trusted_observation import (
    OBSERVATION_ID_PATTERN,
    TrustedObservation,
    canonicalize_trusted_scene,
    trusted_target_local_candidate,
)
from agent.domain.ui_scene import MIN_TARGET_CONFIDENCE, UIScene
from agent.domain.vision_model import VisionAgentError
from agent.infrastructure.observation_images import (
    local_frame_fingerprint,
    measure_frame_sharpness,
    measure_local_stability,
)


MIN_TRUSTED_FRAME_SHARPNESS = 4.0


def build_trusted_observation(
    *,
    frames: list[Image.Image],
    device_id: str,
    scene: UIScene,
    observation_id: str | None = None,
) -> TrustedObservation:
    if len(frames) < 4:
        raise VisionAgentError("可信观察至少需要4帧。")
    if not DEVICE_ID_PATTERN.fullmatch(str(device_id or "").strip()):
        raise VisionAgentError(f"可信观察 device_id 无效：{device_id!r}")
    stability = measure_local_stability(frames, allow_leading_outlier=True)
    if not stability.stable:
        raise VisionAgentError(
            f"本地多帧稳定性检查未通过：{stability.reason}；不能建立可信观察。"
        )
    sharpness = tuple(measure_frame_sharpness(frame) for frame in frames)
    # The observer permits one stale leading camera frame.  Only the converged
    # three-frame tail may provide the trusted fingerprint.
    stable_tail_start = max(0, len(frames) - min(3, len(frames)))
    selected = max(
        range(stable_tail_start, len(frames)),
        key=sharpness.__getitem__,
    )
    sharpness_floor = float(
        os.environ.get(
            "ROBOT_LOCAL_FRAME_SHARPNESS_MIN",
            str(MIN_TRUSTED_FRAME_SHARPNESS),
        )
    )
    if sharpness[selected] < sharpness_floor:
        raise VisionAgentError(
            "当前最清晰帧仍然模糊："
            f"sharpness={sharpness[selected]:.3f} < {sharpness_floor:.3f}。"
        )
    fingerprint = local_frame_fingerprint(frames[selected].convert("RGB"))
    scene.validate()
    canonical_scene, aliases, conflicts = canonicalize_trusted_scene(scene)
    target_local_candidate = trusted_target_local_candidate(
        canonical_scene,
        conflicts,
    )
    if not scene.stable or (
        float(scene.confidence) < MIN_TARGET_CONFIDENCE
        and target_local_candidate is None
        and not canonical_scene.trusted_completion_evidence()
    ):
        raise VisionAgentError("页面不稳定或整体置信度不足，不能建立可信候选。")
    if scene.fingerprint != fingerprint:
        raise VisionAgentError(
            "只读观察 fingerprint 与当前本地帧不一致，拒绝建立可信候选。"
        )
    resolved_id = observation_id or f"obs_{uuid.uuid4().hex}"
    if not OBSERVATION_ID_PATTERN.fullmatch(resolved_id):
        raise VisionAgentError(f"observation_id 格式无效：{resolved_id!r}")
    result = TrustedObservation(
        observation_id=resolved_id,
        device_id=str(device_id).strip(),
        fingerprint=fingerprint,
        scene=canonical_scene,
        local_stability=stability,
        selected_frame_index=selected,
        frame_sharpness_scores=sharpness,
        candidate_aliases=aliases,
        candidate_conflicts=conflicts,
    )
    validate_trusted_observation_against_frames(
        result,
        frames,
        allow_leading_outlier=True,
    )
    return result


def validate_trusted_observation_against_frames(
    observation: TrustedObservation,
    frames: list[Image.Image],
    *,
    allow_leading_outlier: bool = False,
) -> None:
    if len(frames) < 4:
        raise VisionAgentError("新鲜度校验至少需要4帧。")
    stability = measure_local_stability(
        frames,
        allow_leading_outlier=allow_leading_outlier,
    )
    if not stability.stable:
        raise VisionAgentError(
            f"当前画面已不稳定：{stability.reason}；旧观察失效。"
        )
    sharpness = [measure_frame_sharpness(frame) for frame in frames]
    eligible_start = (
        max(0, len(frames) - min(3, len(frames)))
        if allow_leading_outlier
        else 0
    )
    selected = max(
        range(eligible_start, len(frames)),
        key=sharpness.__getitem__,
    )
    sharpness_floor = float(
        os.environ.get(
            "ROBOT_LOCAL_FRAME_SHARPNESS_MIN",
            str(MIN_TRUSTED_FRAME_SHARPNESS),
        )
    )
    if sharpness[selected] < sharpness_floor:
        raise VisionAgentError("当前新鲜画面仍然模糊，旧动作失效。")
    current_fingerprint = local_frame_fingerprint(frames[selected].convert("RGB"))
    if current_fingerprint != observation.fingerprint:
        raise VisionAgentError("当前画面 fingerprint 已变化，旧动作失效。")
    if observation.scene.fingerprint != observation.fingerprint:
        raise VisionAgentError("可信观察内部 fingerprint 不一致。")
