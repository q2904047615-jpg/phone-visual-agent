from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from observation_images import measure_local_stability
from universal_agent_orchestrator import (
    DeviceTaskRegistry,
    PhaseOneNavigationPolicy,
)


ROOT = Path(__file__).resolve().parent
PREFLIGHT_SCHEMA_VERSION = "2026-08-12-universal-agent-live-preflight-v1"
FRAME_COUNT = 4
FRAME_INTERVAL_SECONDS = 0.12


def _provider_is_configured(provider: Any) -> tuple[bool, bool]:
    """Return only readiness; never propagate provider credentials or URLs."""

    try:
        status = provider.status()
        return bool(status.get("configured")), False
    except Exception:
        configured = bool(getattr(provider, "configured", False))
        return configured, True


def _safe_controller_status(controller: Any) -> dict[str, Any]:
    try:
        raw = controller.device_status()
    except Exception as exc:
        raw = {
            "controller_online": False,
            "camera_online": False,
            "error": str(exc),
        }
    client_size = raw.get("client_size")
    if not isinstance(client_size, (list, tuple)) or len(client_size) != 2:
        client_size = [0, 0]
    return {
        "online": bool(raw.get("controller_online")),
        "camera_online": bool(raw.get("camera_online")),
        "window_title": str(raw.get("window_title") or ""),
        "client_size": [int(client_size[0]), int(client_size[1])],
        "busy": bool(raw.get("busy")),
        "stop_requested": bool(raw.get("stop_requested")),
        "error": str(raw.get("error") or raw.get("camera_error") or "") or None,
    }


def _capture_frames(
    controller: Any,
    *,
    sleep: Callable[[float], None],
) -> tuple[list[Image.Image], str | None]:
    frames: list[Image.Image] = []
    try:
        for index in range(FRAME_COUNT):
            frame = controller.vision_capture()
            if not isinstance(frame, Image.Image):
                raise TypeError("vision_capture() 未返回 PIL.Image")
            frames.append(frame.convert("RGB"))
            if index + 1 < FRAME_COUNT:
                sleep(FRAME_INTERVAL_SECONDS)
    except Exception as exc:
        return frames, str(exc)
    return frames, None


def _save_evidence(
    frames: list[Image.Image], evidence_dir: Path | None
) -> list[str]:
    if evidence_dir is None:
        return []
    evidence_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for index, frame in enumerate(frames, start=1):
        path = evidence_dir / f"frame_{index}.jpg"
        frame.save(path, format="JPEG", quality=90, optimize=True)
        paths.append(str(path.resolve()))
    return paths


def run_read_only_preflight(
    *,
    device_id: str,
    controller: Any,
    deepseek_provider: Any,
    qwen_provider: Any,
    registry: DeviceTaskRegistry,
    policy: PhaseOneNavigationPolicy,
    evidence_dir: Path | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Inspect readiness without invoking any physical-action entry point."""

    normalized_device_id = str(device_id or "").strip()
    if not normalized_device_id:
        raise ValueError("device_id 不能为空。")

    blockers: list[str] = []
    active_session = registry.active_session(normalized_device_id)
    exclusive_available = active_session is None
    if not exclusive_available:
        blockers.append(f"设备已有活动会话：{active_session}")

    controller_status = _safe_controller_status(controller)
    if not controller_status["online"]:
        blockers.append("机械臂控制端不可连接")
    if not controller_status["camera_online"]:
        blockers.append("手机摄像头画面不可用")
    if controller_status["busy"]:
        blockers.append("机械臂控制端正在忙碌")
    if controller_status["stop_requested"]:
        blockers.append("机械臂控制端处于停止状态")

    deepseek_configured, deepseek_status_failed = _provider_is_configured(
        deepseek_provider
    )
    qwen_configured, qwen_status_failed = _provider_is_configured(qwen_provider)
    if not deepseek_configured:
        blockers.append("DeepSeek provider 未配置")
    if not qwen_configured:
        blockers.append("Qwen provider 未配置")
    if deepseek_status_failed:
        blockers.append("DeepSeek provider 状态读取失败")
    if qwen_status_failed:
        blockers.append("Qwen provider 状态读取失败")

    policy_version = str(getattr(policy, "VERSION", ""))
    expected_policy_version = PhaseOneNavigationPolicy.VERSION
    policy_matches = policy_version == expected_policy_version
    if not policy_matches:
        blockers.append("当前编排器不是预期的第一阶段安全策略")

    frames: list[Image.Image] = []
    capture_error: str | None = None
    stability: dict[str, Any] | None = None
    if controller_status["online"] and controller_status["camera_online"]:
        frames, capture_error = _capture_frames(controller, sleep=sleep)
        if capture_error:
            blockers.append(f"连续四帧采集失败：{capture_error}")
        if len(frames) != FRAME_COUNT:
            blockers.append(f"只取得 {len(frames)} 帧，预期 {FRAME_COUNT} 帧")
        elif len({frame.size for frame in frames}) != 1:
            blockers.append("连续四帧尺寸不一致")
        else:
            measured = measure_local_stability(frames)
            stability = measured.to_dict()
            if not measured.stable:
                blockers.append(f"连续四帧不稳定：{measured.reason}")

    evidence_paths = _save_evidence(frames, evidence_dir)
    unique_blockers = list(dict.fromkeys(blockers))
    return {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "ready": not unique_blockers,
        "physical_actions": 0,
        "device": {
            "device_id": normalized_device_id,
            "exclusive_available": exclusive_available,
            "active_session": active_session,
        },
        "controller": controller_status,
        "camera": {
            "frame_count": len(frames),
            "frame_sizes": [list(frame.size) for frame in frames],
            "stability": stability,
            "capture_error": capture_error,
            "evidence_paths": evidence_paths,
        },
        "providers": {
            "deepseek_configured": deepseek_configured,
            "qwen_configured": qwen_configured,
        },
        "policy": {
            "version": policy_version,
            "expected_version": expected_policy_version,
            "phase_one_matches": policy_matches,
        },
        "blockers": unique_blockers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="通用手机视觉 Agent 只读实机预检（绝不发送物理动作）"
    )
    parser.add_argument("--device-id", default="default-device")
    parser.add_argument(
        "--no-save-frames",
        action="store_true",
        help="不保存只读摄像头证据帧",
    )
    args = parser.parse_args()

    # Importing the shared runtime constructs dependencies but does not start the
    # web worker.  This script calls only the read-only interfaces above.
    from web_app import runtime

    evidence_dir = None
    if not args.no_save_frames:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        evidence_dir = ROOT / "output" / f"universal_agent_preflight_{timestamp}"

    result = run_read_only_preflight(
        device_id=args.device_id,
        controller=runtime.controller,
        deepseek_provider=runtime.intent_provider,
        qwen_provider=runtime.vision_provider,
        registry=runtime.device_task_registry,
        policy=runtime.universal_agent_orchestrator.policy,
        evidence_dir=evidence_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
