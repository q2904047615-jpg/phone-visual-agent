"""Isolated two-observation lucky-bag runner.

The general visual task loop remains the authority for ordinary tasks.  This
feature has an explicitly approved, bounded flow: Qwen finds the entry once,
the two known form controls are pressed locally, and Qwen verifies the result.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import threading
import time
from typing import Any, Callable, Mapping
from uuid import uuid4

from PIL import Image

from agent.infrastructure.observation_images import local_frame_fingerprint
from agent.infrastructure.orientation_safety import _mint_single_step_scene_credential
from agent.infrastructure.task_screenshot_history import save_task_frames
from agent.domain.vision_model import VisionAgentError

from .profile import (
    COMMENT_SEND_DELAY_SECONDS,
    NO_LUCKY_BAG_REOBSERVE_SECONDS,
    POST_PARTICIPATION_REOBSERVE_SECONDS,
    LuckyBagProfile,
)


COMMENT_POINT = (498, 868)
SEND_POINT = (832, 476)
# These are the three raw camera-frame pixels supplied for the lucky-bag
# entry row.  Qwen selects one slot; it never supplies an arbitrary entry
# coordinate.  The runner converts the selected raw pixel to the controller's
# existing 0..1000 grid using the current frame dimensions.
LUCKY_BAG_SLOT_POINTS = {
    1: (70, 113),
    2: (118, 113),
    3: (163, 113),
}


@dataclass
class _FastSession:
    session_id: str
    profile: LuckyBagProfile
    goal: str
    run_dir: Path
    status: str = "running"
    phase: str = "find_bag"
    detail: str = "正在观察直播画面中的福袋入口。"
    physical_actions: int = 0
    observations: int = 0
    qwen_reply: str = ""
    qwen_decision: dict[str, Any] | None = None
    win_detected: bool = False
    draw_result: str = ""
    observed_at: float = 0.0
    next_observation_at: float = 0.0
    controller: Any = None
    visual_reference_paths: tuple[Path, ...] = ()


class FastLuckyBagSessionGateway:
    """Gateway implementing the approved lucky-bag flow without generic auto steps."""

    def __init__(self, *, controller_for_device: Callable[[str], Any],
        capture_frame: Callable[[str], Image.Image], observer: Any,
        ensure_device_ready: Callable[[str], None],
        exclusive_device_session: Callable[[str], Any] | None = None) -> None:
        self.controller_for_device = controller_for_device
        self.capture_frame = capture_frame
        self.observer = observer
        self.ensure_device_ready = ensure_device_ready
        self.exclusive_device_session = exclusive_device_session
        self._sessions: dict[str, _FastSession] = {}
        self._lock = threading.RLock()

    def start(self, *, goal: str, device_id: str, run_dir: Path,
        visual_reference_paths: tuple[Path, ...] = ()) -> Mapping[str, Any]:
        self.ensure_device_ready(device_id)
        controller = self.controller_for_device(device_id)
        begin = getattr(controller, "begin_new_task", None)
        if callable(begin):
            begin()
        session_id = f"lucky-fast-{uuid4().hex}"
        run_dir.mkdir(parents=True, exist_ok=True)
        state = _FastSession(session_id, LuckyBagProfile(
            device_id=device_id, recipient="local", duration_seconds=1,
        ), goal, run_dir, visual_reference_paths=tuple(visual_reference_paths))
        # The monitor, not this device-id copy, owns the deadline and profile.
        state.profile = getattr(self, "_profile_for_start", state.profile)
        state.controller = controller
        with self._lock:
            self._sessions[session_id] = state
        return {"session": self._snapshot(state)}

    def set_profile(self, session_id: str, profile: LuckyBagProfile) -> None:
        with self._lock:
            self._sessions[session_id].profile = profile

    def get(self, session_id: str) -> Mapping[str, Any]:
        with self._lock:
            return {"session": self._snapshot(self._sessions[session_id])}

    def pause(self, session_id: str) -> Mapping[str, Any]:
        with self._lock:
            state = self._sessions[session_id]
            state.status = "paused"
            state.detail = "用户已暂停福袋监控。"
        self._request_stop(state)
        return {"session": self._snapshot(state)}

    def cancel(self, session_id: str) -> Mapping[str, Any]:
        with self._lock:
            state = self._sessions[session_id]
            state.status = "cancelled"
            state.detail = "用户已停止福袋监控。"
        self._request_stop(state)
        return {"session": self._snapshot(state)}

    def auto(self, session_id: str, *, max_physical_actions: int,
        max_observations: int) -> Mapping[str, Any]:
        del max_physical_actions, max_observations
        with self._lock:
            state = self._sessions[session_id]
        if state.status != "running":
            return {"session": self._snapshot(state)}
        if state.next_observation_at > time.monotonic():
            # Return regularly so the supervisor can enforce its 24-hour
            # deadline even when the draw countdown is longer than time left.
            self._wait(state, min(1.0, state.next_observation_at - time.monotonic()))
            return {"session": self._snapshot(state)}
        controller = self.controller_for_device(state.profile.device_id)
        try:
            with self._execution_scope(state.profile.device_id):
                if state.status != "running":
                    return {"session": self._snapshot(state)}
                if state.phase == "find_bag":
                    decision = self._observe(state, stage="find_bag", references=True)
                    if state.status != "running":
                        return {"session": self._snapshot(state)}
                    action = str(decision.get("action") or "")
                    if action == "wait_for_change":
                        state.detail = "当前画面未确认福袋，等待后重新观察。"
                        state.next_observation_at = time.monotonic() + NO_LUCKY_BAG_REOBSERVE_SECONDS
                        return {"session": self._snapshot(state)}
                    if action != "tap":
                        raise VisionAgentError("Qwen 未选择福袋入口位置。")
                    slot = self._match_lucky_bag_slot(decision.get("point"), controller)
                    if slot is None:
                        raise VisionAgentError("Qwen 返回的福袋位置不是 1、2、3 号候选位置。")
                    self._local_raw_pixel_tap(controller, LUCKY_BAG_SLOT_POINTS[slot])
                    state.physical_actions += 1
                    state.phase = "fixed_comment"
                    state.detail = f"Qwen选择第{slot}号福袋位置，已打开福袋，按固定流程填写并发送。"
                    return {"session": self._snapshot(state)}
                if state.phase == "fixed_comment":
                    self._local_tap(controller, COMMENT_POINT)
                    state.physical_actions += 1
                    state.phase = "fixed_send"
                    state.next_observation_at = time.monotonic() + COMMENT_SEND_DELAY_SECONDS
                    state.detail = "已点击去发表评论，等待 3 秒后发送。"
                    return {"session": self._snapshot(state)}
                if state.phase == "fixed_send":
                    self._local_tap(controller, SEND_POINT)
                    state.physical_actions += 1
                    state.phase = "verify_participation"
                if state.phase == "verify_participation":
                    decision = self._observe(state, stage="verify", references=False)
                    if state.status != "running":
                        return {"session": self._snapshot(state)}
                    wait_seconds = self._participation_wait_seconds(decision)
                    if wait_seconds is None:
                        # Re-observe only: an uncertain send must never return
                        # to the fixed comment/send sequence.
                        state.detail = "参与结果或开奖倒计时尚未确认，等待后重新查看，不重复发送。"
                        state.next_observation_at = time.monotonic() + NO_LUCKY_BAG_REOBSERVE_SECONDS
                    else:
                        state.phase = "draw_result"
                        state.detail = f"已确认参与，等待开奖倒计时结束后复核结果（约 {wait_seconds:g} 秒）。"
                        state.next_observation_at = state.observed_at + wait_seconds
                    return {"session": self._snapshot(state)}
                if state.phase == "draw_result":
                    decision = self._observe(state, stage="draw_result", references=False)
                    if state.status != "running":
                        return {"session": self._snapshot(state)}
                    result = self._postcondition_status(decision)
                    action = str(decision.get("action") or "")
                    if result == "not_confirmed":
                        if action != "tap":
                            raise VisionAgentError("Qwen 判断未中奖，但没有返回‘知道了’按钮的点击动作。")
                        if not self._valid_point(decision.get("point")):
                            raise VisionAgentError("Qwen 判断未中奖，但没有返回有效的‘知道了’按钮坐标。")
                        self._local_tap(controller, tuple(int(round(v)) for v in decision["point"]))
                        state.physical_actions += 1
                        state.draw_result = "not_won"
                        state.phase = "find_bag"
                        state.detail = "本轮未中奖，已点击知道了，继续监控新的福袋。"
                        state.next_observation_at = 0.0
                    elif result == "confirmed":
                        state.win_detected = True
                        state.draw_result = "won"
                        state.phase = "find_bag"
                        state.detail = "本轮已中奖，继续监控新的福袋。"
                        state.next_observation_at = time.monotonic() + NO_LUCKY_BAG_REOBSERVE_SECONDS
                    else:
                        state.draw_result = "unknown"
                        state.detail = "开奖结果尚未明确，等待后重新查看，不点击也不重复参与。"
                        state.next_observation_at = time.monotonic() + POST_PARTICIPATION_REOBSERVE_SECONDS
        except Exception as exc:
            if state.status == "running":
                state.status = "failed"
                state.detail = str(exc) or type(exc).__name__
        return {"session": self._snapshot(state)}

    def _execution_scope(self, device_id: str):
        # The web adapter supplies the same inter-process/device lock used by
        # ordinary tasks.  Tests can provide a no-op context manager.
        return self._exclusive_device_session(device_id)

    def _exclusive_device_session(self, device_id: str):
        from contextlib import nullcontext
        callback = getattr(self, "exclusive_device_session", None)
        return callback(device_id) if callable(callback) else nullcontext()

    def _observe(self, state: _FastSession, *, stage: str, references: bool) -> dict[str, Any]:
        state.observed_at = time.monotonic()
        frames = [self.capture_frame(state.profile.device_id) for _ in range(4)]
        paths = save_task_frames(frames, state.run_dir, stage, state.profile.device_id)
        context = {
            "task_id": state.session_id,
            "app_name": state.profile.app_alias,
            "objective": (
                "识别当前直播画面中的福袋并完成一次参与"
                if stage == "find_bag" else
                "判断刚才发送评论后是否已明确参与福袋"
                if stage == "verify" else
                "开奖后判断是否中奖；如果未中奖，返回当前画面中‘知道了’按钮的点击坐标"
            ),
            "entities": {"task_id": state.session_id},
            "constraints": [
                "这是福袋专用流程。初始观察只找福袋入口；发送后只判断是否明确参与；开奖后只判断中奖结果。",
                "固定的去发表评论和发送评论步骤不需要再次调用模型。",
            ],
        }
        if stage == "verify":
            context["constraints"].append(
                "只判断刚才是否成功参与，不点击、不再次发送。decision.postcondition.status 用 confirmed 表示明确已参与，"
                "not_confirmed 表示明确未参与，unknown 表示无法确定；fact说明当前可见证据。"
                "已参与且倒计时可读时选择 wait_for_change，wait_seconds填写CURRENT采集时剩余的准确秒数（包括0），"
                "不得统一写300或截短倒计时；倒计时不可读时wait_seconds=null。"
            )
        elif stage == "draw_result":
            context["constraints"].append(
                "本轮只检查当前福袋是否中奖。decision.postcondition.status 用 confirmed 表示明确中奖，"
                "not_confirmed 表示明确未中奖，unknown 表示尚未开奖或无法判断；fact写当前画面证据。"
                "reply自然说明该结果。未中奖时同一响应选择tap且point填写CURRENT中‘知道了’的坐标；"
                "中奖时不点击，返回finish（仅结束本轮检查，监控继续）；未知时wait_for_change，不把未知当中奖或未中奖。"
            )
        else:
            context["constraints"].append("找不到福袋时选择wait_for_change，等待300秒后再次观察。")
        current_detail_images: tuple[Image.Image, ...] = ()
        if stage == "find_bag":
            current = frames[-1].convert("RGB")
            crop_right = min(current.width, max(1, round(current.width * 0.45)))
            crop_bottom = min(current.height, max(1, round(current.height * 0.30)))
            crop = current.crop((0, 0, crop_right, crop_bottom))
            current_detail_images = (
                crop.resize((crop.width * 3, crop.height * 3), Image.Resampling.LANCZOS),
            )
            context["constraints"].append(
                "本轮另附当前手机画面左上区域的放大图；它只用于确认福袋外观，点击位置仍只能从当前整图给出的1、2、3号候选中选择。"
            )
            width, height = frames[-1].size
            candidates = {
                str(slot): [round(x * 1000 / max(1, width - 1), 2),
                            round(y * 1000 / max(1, height - 1), 2)]
                for slot, (x, y) in LUCKY_BAG_SLOT_POINTS.items()
            }
            context["constraints"].append(
                "福袋入口只允许从 1、2、3 号候选中选择；不要返回其它入口坐标。"
                f"当前帧候选的 decision.point（1000×1000 坐标）为 {json.dumps(candidates, ensure_ascii=False)}；"
                "看到福袋时只能返回其中一个候选点并执行tap，找不到时返回wait_for_change。"
            )
        actions = ("tap_semantic", "wait_for_change") if stage in {"find_bag", "draw_result"} else ("wait_for_change",)
        _scene, decision = self.observer.observe_with_decision(
            frames=frames, goal_context=context, device_id=state.profile.device_id,
            available_action_kinds=actions, response_evidence_dir=state.run_dir,
            response_evidence_prefix=stage, current_frame_paths=paths,
            visual_reference_paths=(state.visual_reference_paths or state.profile.visual_reference_paths)
                if references else (),
            current_detail_images=current_detail_images,
        )
        state.observations += 1
        state.qwen_decision = dict(decision)
        raw = getattr(self.observer, "last_raw_response", "")
        try:
            payload = json.loads(raw)
            reply = payload.get("reply") if isinstance(payload, dict) else ""
            state.qwen_reply = reply.strip() if isinstance(reply, str) else ""
        except (TypeError, ValueError):
            state.qwen_reply = ""
        return decision

    @staticmethod
    def _participation_wait_seconds(decision: Mapping[str, Any]) -> float | None:
        """Return the countdown only after Qwen confirms participation."""

        if FastLuckyBagSessionGateway._postcondition_status(decision) != "confirmed":
            return None
        action = str(decision.get("action") or "")
        if action == "wait_for_change":
            value = decision.get("wait_seconds")
            if (isinstance(value, (int, float)) and not isinstance(value, bool)
                    and value >= 0 and math.isfinite(value)):
                return float(value)
        return None

    @staticmethod
    def _postcondition_status(decision: Mapping[str, Any]) -> str:
        postcondition = decision.get("postcondition")
        return postcondition.get("status", "unknown") if isinstance(postcondition, Mapping) else "unknown"

    @staticmethod
    def _valid_point(value: Any) -> bool:
        return (isinstance(value, (list, tuple)) and len(value) == 2
            and all(isinstance(part, (int, float)) and not isinstance(part, bool)
                and 0 <= float(part) <= 1000 for part in value))

    @classmethod
    def _match_lucky_bag_slot(cls, value: Any, controller: Any) -> int | None:
        if not cls._valid_point(value):
            return None
        frame = controller.vision_capture()
        width, height = frame.size
        candidates = {
            slot: (x * 1000 / max(1, width - 1), y * 1000 / max(1, height - 1))
            for slot, (x, y) in LUCKY_BAG_SLOT_POINTS.items()
        }
        point = (float(value[0]), float(value[1]))
        ranked = sorted(
            ((slot, math.hypot(point[0] - candidate[0], point[1] - candidate[1]))
             for slot, candidate in candidates.items()),
            key=lambda item: item[1],
        )
        # The three points are close on a tall frame.  Refuse an ambiguous
        # answer instead of silently clicking a neighboring slot.
        best_slot, best_distance = ranked[0]
        second_distance = ranked[1][1]
        return best_slot if best_distance <= 28 and second_distance - best_distance >= 8 else None

    @classmethod
    def _local_raw_pixel_tap(cls, controller: Any, point: tuple[int, int]) -> None:
        frame = controller.vision_capture()
        x, y = point
        if not (0 <= x < frame.width and 0 <= y < frame.height):
            raise VisionAgentError(
                f"福袋候选位置 {point} 超出当前画面 {frame.width}×{frame.height}。"
            )
        grid_point = (
            int(round(x * 1000 / max(1, frame.width - 1))),
            int(round(y * 1000 / max(1, frame.height - 1))),
        )
        cls._local_tap(controller, grid_point)

    @staticmethod
    def _local_tap(controller: Any, point: tuple[int, int]) -> None:
        frame = controller.vision_capture()
        fingerprint = local_frame_fingerprint(frame)
        credential = _mint_single_step_scene_credential(
            device_id=controller.device_id, scene_fingerprint=fingerprint, frame=frame,
        )
        controller.arm_physical_execution(credential, action="tap_semantic", scene_fingerprint=fingerprint)
        try:
            controller.vision_tap_relative(*point)
            receipt = controller.consume_last_click_receipt()
            if not isinstance(receipt, dict) or receipt.get("input_events_dispatched") is not True:
                raise VisionAgentError("固定点击没有返回有效的事件派发凭据。")
        finally:
            controller.clear_physical_execution_authorization()

    @staticmethod
    def _wait(state: _FastSession, seconds: float) -> None:
        event = getattr(getattr(state, "controller", None), "stop_event", None)
        if event is not None:
            event.wait(max(0.0, seconds))
        else:
            time.sleep(max(0.0, seconds))

    @staticmethod
    def _request_stop(state: _FastSession) -> None:
        controller = getattr(state, "controller", None)
        request_stop = getattr(controller, "request_stop", None)
        if callable(request_stop):
            request_stop()

    @staticmethod
    def _snapshot(state: _FastSession) -> dict[str, Any]:
        return {
            "session_id": state.session_id, "status": state.status,
            "physical_actions": state.physical_actions,
            "execution_budget": {"observation_attempts": state.observations},
            "qwen_reply": state.qwen_reply, "qwen_decision": state.qwen_decision,
            "win_detected": state.win_detected,
            "draw_result": state.draw_result,
            "detail": state.detail, "current_phase": state.phase,
            "device_id": state.profile.device_id,
        }


__all__ = ["COMMENT_POINT", "SEND_POINT", "LUCKY_BAG_SLOT_POINTS", "FastLuckyBagSessionGateway"]
