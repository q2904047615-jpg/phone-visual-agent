from __future__ import annotations

import hashlib
import time
from datetime import datetime
from typing import Any, Callable, Mapping

from PIL import Image

from agent.domain.action_capabilities import build_device_capability_snapshot
from agent.infrastructure.observation_images import measure_local_stability
from agent.application.vision_usage import QWEN_PLUS_MODEL


RUNTIME_DOCTOR_VERSION = "2026-08-25-runtime-doctor-v1"
DOCTOR_FRAME_COUNT = 4
DOCTOR_FRAME_INTERVAL_SECONDS = 0.12


def _public_provider_status(provider: Any, *, role: str) -> tuple[dict[str, Any], str | None]:
    """Return a secret-free provider projection and an optional blocker."""

    try:
        raw = provider.status()
    except Exception as exc:
        return {
            "role": role,
            "configured": False,
            "status_error_type": type(exc).__name__,
        }, f"{role} provider 状态读取失败"
    if not isinstance(raw, Mapping):
        return {
            "role": role,
            "configured": False,
            "status_error_type": "invalid_status_shape",
        }, f"{role} provider 状态格式无效"
    allowed = (
        "configured",
        "provider",
        "model",
        "thinking",
        "thinking_enabled",
        "model_config_version",
        "coordinate_scale",
        "successful_call_count",
        "last_network_attempts",
        "last_finish_reason",
        "response_model",
    )
    value = {"role": role}
    value.update({key: raw[key] for key in allowed if key in raw})
    value["configured"] = bool(raw.get("configured"))
    blocker = None if value["configured"] else f"{role} provider 未配置"
    return value, blocker


def _public_controller_status(controller: Any) -> dict[str, Any]:
    try:
        raw = controller.device_status()
    except Exception as exc:
        return {
            "controller_online": False,
            "camera_online": False,
            "busy": False,
            "stop_requested": False,
            "window_title": "",
            "client_size": [0, 0],
            "status_error_type": type(exc).__name__,
        }
    if not isinstance(raw, Mapping):
        raw = {}
    client_size = raw.get("client_size")
    if not isinstance(client_size, (list, tuple)) or len(client_size) != 2:
        client_size = (0, 0)
    return {
        "controller_online": bool(raw.get("controller_online")),
        "camera_online": bool(raw.get("camera_online")),
        "busy": bool(raw.get("busy")),
        "stop_requested": bool(raw.get("stop_requested")),
        "window_title": str(raw.get("window_title") or "")[:200],
        "client_size": [int(client_size[0]), int(client_size[1])],
        "status_error_type": None,
    }


def _frame_fingerprint(frame: Image.Image) -> str:
    image = frame.convert("RGB")
    digest = hashlib.sha256()
    digest.update(f"{image.width}x{image.height}:RGB:".encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def _capture_stable_frames(
    controller: Any,
    *,
    sleep: Callable[[float], None],
) -> tuple[list[Image.Image], str | None]:
    frames: list[Image.Image] = []
    try:
        for index in range(DOCTOR_FRAME_COUNT):
            frame = controller.vision_capture()
            if not isinstance(frame, Image.Image):
                raise TypeError("vision_capture did not return PIL.Image")
            frames.append(frame.convert("RGB"))
            if index + 1 < DOCTOR_FRAME_COUNT:
                sleep(DOCTOR_FRAME_INTERVAL_SECONDS)
    except Exception as exc:
        return frames, type(exc).__name__
    return frames, None


def run_runtime_doctor(
    *,
    device_id: str,
    controller: Any,
    deepseek_provider: Any,
    qwen_provider: Any,
    active_session: str | None,
    protocols: Mapping[str, str],
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Inspect the current formal runtime without requesting a physical action."""

    resolved_device = str(device_id or "").strip()
    if not resolved_device:
        raise ValueError("device_id 不能为空。")

    blockers: list[str] = []
    controller_status = _public_controller_status(controller)
    if not controller_status["controller_online"]:
        blockers.append("机械臂控制端不可连接")
    if not controller_status["camera_online"]:
        blockers.append("手机摄像头画面不可用")
    if controller_status["busy"]:
        blockers.append("机械臂控制端正在忙碌")
    if controller_status["stop_requested"]:
        blockers.append("机械臂控制端处于停止状态")
    if active_session:
        blockers.append(f"设备已有活动会话：{active_session}")

    deepseek_status, deepseek_blocker = _public_provider_status(
        deepseek_provider,
        role="DeepSeek",
    )
    qwen_status, qwen_blocker = _public_provider_status(
        qwen_provider,
        role="Qwen",
    )
    if deepseek_blocker:
        blockers.append(deepseek_blocker)
    if qwen_blocker:
        blockers.append(qwen_blocker)
    if qwen_status.get("configured") and qwen_status.get("model") != QWEN_PLUS_MODEL:
        blockers.append(
            "正式视觉模型不是 qwen3.7-plus："
            + str(qwen_status.get("model") or "unknown")
        )

    capability_profile: Mapping[str, Any] = {}
    profile_provider = getattr(controller, "hardware_capability_profile", None)
    if callable(profile_provider):
        try:
            raw_profile = profile_provider()
            if isinstance(raw_profile, Mapping):
                capability_profile = raw_profile
        except Exception as exc:
            blockers.append("设备动作能力读取失败：" + type(exc).__name__)
    raw_actions = capability_profile.get("actions")
    if not isinstance(raw_actions, Mapping):
        raw_actions = {}
    supported_actions = tuple(
        str(name)
        for name, spec in raw_actions.items()
        if isinstance(name, str)
        and isinstance(spec, Mapping)
        and spec.get("enabled") is True
    )
    try:
        typed_capabilities = build_device_capability_snapshot(
            device_id=resolved_device,
            supported_actions=supported_actions,
            raw_profile=capability_profile,
        )
        capability_snapshot = typed_capabilities.to_dict()
        capability_snapshot["supported_actions"] = list(
            typed_capabilities.supported_actions
        )
    except Exception as exc:
        capability_snapshot = {
            "device_id": resolved_device,
            "error_type": type(exc).__name__,
        }
        blockers.append("设备动作能力合同无效：" + type(exc).__name__)

    frames: list[Image.Image] = []
    capture_error: str | None = None
    stability: dict[str, Any] | None = None
    may_capture = bool(
        controller_status["controller_online"]
        and controller_status["camera_online"]
        and not controller_status["busy"]
        and not controller_status["stop_requested"]
        and not active_session
    )
    if may_capture:
        frames, capture_error = _capture_stable_frames(controller, sleep=sleep)
        if capture_error:
            blockers.append("连续四帧采集失败：" + capture_error)
        if len(frames) != DOCTOR_FRAME_COUNT:
            blockers.append(
                f"只取得 {len(frames)} 帧，预期 {DOCTOR_FRAME_COUNT} 帧"
            )
        elif len({frame.size for frame in frames}) != 1:
            blockers.append("连续四帧尺寸不一致")
        else:
            measured = measure_local_stability(
                frames,
                allow_leading_outlier=True,
            )
            stability = measured.to_dict()
            if not measured.stable:
                blockers.append("连续四帧不稳定：" + measured.reason)

    unique_blockers = list(dict.fromkeys(blockers))
    return {
        "schema_version": RUNTIME_DOCTOR_VERSION,
        "checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "ready": not unique_blockers,
        "physical_actions": 0,
        "device": {
            "device_id": resolved_device,
            "exclusive_available": active_session is None,
            "active_session": active_session,
        },
        "controller": controller_status,
        "camera": {
            "captured": may_capture,
            "frame_count": len(frames),
            "frame_sizes": [list(frame.size) for frame in frames],
            "frame_fingerprints": [_frame_fingerprint(frame) for frame in frames],
            "stability": stability,
            "capture_error_type": capture_error,
        },
        "providers": {
            "deepseek": deepseek_status,
            "qwen": qwen_status,
        },
        "protocols": {str(key): str(value) for key, value in protocols.items()},
        "capabilities": capability_snapshot,
        "blockers": unique_blockers,
    }
