"""One task on one phone: observe -> ask Qwen -> execute one action -> observe again.

Status values
-------------
running            the loop thread is working
awaiting_approval  step mode: an action is proposed and waits for the user
needs_confirmation the model marked the action as login or payment
paused             paused by the user, or the action/observation budget ran out
succeeded          the model reported the whole task finished
failed             a device/model fault, or the model said the task cannot be done
cancelled          stopped by the user
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from pva.actions import Action, ActionError, describe, parse_action
from pva.devices import Device, DeviceError, DeviceManager, StopRequested
from pva.model.prompt import Shot, build_messages, redact_images
from pva.model.qwen import ModelError, VisionModel, extract_json
from pva.util import now_iso, write_json

ACTIVE = {"running", "awaiting_approval", "needs_confirmation", "paused"}
FINAL = {"succeeded", "failed", "cancelled"}


@dataclass
class TaskOptions:
    device_id: str
    task: str
    mode: str = "auto"                      # auto | step
    max_actions: int = 100
    max_observations: int = 200
    enable_thinking: bool = False
    settle_seconds: float = 1.2
    thumbnail: tuple[int, int] = (360, 640)
    current_max: tuple[int, int] = (1080, 1920)
    max_consecutive_model_errors: int = 3


@dataclass
class Step:
    index: int
    screenshot: str
    started_at: str = field(default_factory=now_iso)
    observation: str = ""
    last_result: str = ""
    reply: str = ""
    reason: str = ""
    risk: str = "none"
    action: dict[str, Any] | None = None
    action_text: str = ""
    status: str = "thinking"   # thinking | proposed | executed | error | finished | rejected
    error: str = ""
    receipt: dict[str, Any] | None = None
    result_text: str = ""      # what the model sees in the history for this action
    model: dict[str, Any] = field(default_factory=dict)
    extra_screenshots: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class Session:
    def __init__(self, options: TaskOptions, *, devices: DeviceManager, model: VisionModel,
                 output_root: Path, on_change: Callable[["Session"], None] | None = None):
        self.id = uuid.uuid4().hex[:12]
        self.options = options
        self.devices = devices
        self.device: Device = devices.get(options.device_id)
        self.model = model
        self.created_at = now_iso()
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(output_root) / f"{stamp}_{self.id}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.status = "running"
        self.detail = ""
        self.result_message = ""
        self.steps: list[Step] = []
        self.notes: list[dict[str, str]] = []
        self.actions_used = 0
        self.observations_used = 0
        self.model_calls = 0
        self.tokens = 0
        self.pending: Action | None = None
        self._pending_frame_size: tuple[int, int] | None = None
        self._shots: list[tuple[str, str]] = []      # (file name, label) for every saved screenshot
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._on_change = on_change
        self._model_errors = 0

    # -- public control -----------------------------------------------------------
    def start(self) -> None:
        self.devices.reserve(self.device.id, self.id)
        try:
            with self.devices.hardware(self.device.id):
                self.device.prepare()
        except Exception:
            self.devices.release(self.device.id, self.id)
            raise
        self._spawn()

    def pause(self) -> None:
        with self._lock:
            if self.status == "running":
                self._stop.set()
                self.device.request_stop()
                self._set("paused", "用户已暂停；继续时会重新截图判断。")

    def resume(self, *, max_actions: int | None = None, max_observations: int | None = None) -> None:
        with self._lock:
            if self.status != "paused":
                raise RuntimeError(f"当前状态不能继续：{self.status}")
            if max_actions:
                self.options.max_actions = max(max_actions, self.actions_used + 1)
            if max_observations:
                self.options.max_observations = max(max_observations, self.observations_used + 1)
            if self._thread and self._thread.is_alive():
                raise RuntimeError("上一步还在收尾，请稍后再试。")
            self.devices.reserve(self.device.id, self.id)
            self.device.clear_stop()
            self._stop.clear()
            self._set("running", "")
        self._spawn()

    def cancel(self) -> None:
        with self._lock:
            if self.status in FINAL:
                return
            self._stop.set()
            self.device.request_stop()
            self.pending = None
            self._finish("cancelled", "用户已停止任务。")
            self._wake.set()

    def add_note(self, text: str) -> None:
        text = text.strip()
        if not text:
            raise ValueError("说明不能为空。")
        with self._lock:
            self.notes.append({"at": now_iso(), "text": text})
            self._save()

    def approve(self) -> None:
        """Execute the pending action (step mode or a login/payment confirmation)."""
        with self._lock:
            if self.status not in {"awaiting_approval", "needs_confirmation"} or self.pending is None:
                raise RuntimeError("没有等待确认的动作。")
            self._set("running", "")
            self._approved = True
        self._spawn()

    def reject(self, note: str = "") -> None:
        """Discard the pending action; the next observation decides again."""
        with self._lock:
            if self.status not in {"awaiting_approval", "needs_confirmation"} or self.pending is None:
                raise RuntimeError("没有等待确认的动作。")
            step = self.steps[-1]
            step.status = "rejected"
            self._history_result(step, "用户拒绝执行")
            self.pending = None
            if note.strip():
                self.notes.append({"at": now_iso(), "text": note.strip()})
            if self.options.mode == "step":
                self._set("running", "")
                self._spawn()
            else:
                self._set("paused", "已拒绝该动作；可补充说明后继续，或停止任务。")

    def observe_once(self) -> None:
        """Step mode: ask for a fresh proposal (discarding any pending one)."""
        with self._lock:
            if self.status not in {"awaiting_approval", "paused"}:
                raise RuntimeError(f"当前状态不能重新观察：{self.status}")
            if self.pending is not None:
                self.steps[-1].status = "rejected"
                self._history_result(self.steps[-1], "未执行（用户要求重新观察）")
                self.pending = None
            self.devices.reserve(self.device.id, self.id)
            self.device.clear_stop()
            self._stop.clear()
            self._set("running", "")
        self._spawn()

    # -- snapshot -----------------------------------------------------------------------
    def snapshot(self, *, full: bool = True) -> dict[str, Any]:
        with self._lock:
            data = {
                "id": self.id, "device_id": self.device.id, "device_name": self.device.name,
                "task": self.options.task, "mode": self.options.mode, "status": self.status,
                "detail": self.detail, "result": self.result_message, "created_at": self.created_at,
                "budget": {"actions": self.actions_used, "max_actions": self.options.max_actions,
                           "observations": self.observations_used,
                           "max_observations": self.options.max_observations},
                "model_calls": self.model_calls, "tokens": self.tokens,
                "step_count": len(self.steps),
                "last_reply": next((s.reply for s in reversed(self.steps) if s.reply), ""),
                "pending": self.pending.to_dict() if self.pending else None,
                "pending_text": describe(self.pending) if self.pending else "",
                "pending_risk": self.steps[-1].risk if self.pending and self.steps else "none",
                "run_dir": str(self.run_dir),
            }
            if full:
                data["steps"] = [s.to_dict() for s in self.steps]
                data["notes"] = list(self.notes)
            return data

    # -- loop ---------------------------------------------------------------------------
    _approved = False

    def _spawn(self) -> None:
        self._thread = threading.Thread(target=self._run, name=f"task-{self.id}", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            if self._approved and self.pending is not None:
                self._approved = False
                if not self._execute_pending():
                    return
            while not self._stop.is_set():
                if not self._iteration():
                    return
        except StopRequested as exc:
            with self._lock:
                if self.status == "running":
                    self._set("paused", f"{exc}；继续时会重新截图判断。")
        except Exception as exc:  # pragma: no cover - last-resort guard
            (self.run_dir / "crash.txt").write_text(traceback.format_exc(), encoding="utf-8")
            with self._lock:
                if self.status not in FINAL:
                    self._finish("failed", f"内部错误：{exc}")
        finally:
            with self._lock:
                # A pending approval keeps the device; paused or finished tasks free it.
                if self.status not in {"running", "awaiting_approval", "needs_confirmation"}:
                    self.devices.release(self.device.id, self.id)
                self._save()

    def _iteration(self) -> bool:
        """One observe/decide/act cycle. Returns False when the loop should stop."""
        with self._lock:
            if self.observations_used >= self.options.max_observations:
                self._set("paused", f"观察次数已达上限 {self.options.max_observations}；可调高上限后继续。")
                return False
            if self.actions_used >= self.options.max_actions:
                self._set("paused", f"动作次数已达上限 {self.options.max_actions}；可调高上限后继续。")
                return False
        frame = self._capture()
        if frame is None:
            return False
        step = self._new_step(frame)
        decision = self._decide(step, frame)
        if decision is None:
            return self.status == "running"
        action, risk = decision
        with self._lock:
            if self._stop.is_set():
                return False
            if action.type == "finish":
                step.status = "finished"
                ok = action.success is not False
                self._finish("succeeded" if ok else "failed", action.message or step.reply or describe(action))
                return False
            self.pending = action
            self._pending_frame_size = frame.size
            if risk in {"authentication", "payment"}:
                step.status = "proposed"
                label = "登录/身份认证" if risk == "authentication" else "付款/资金交易"
                self._set("needs_confirmation", f"该动作涉及{label}，需要你确认后执行：{describe(action)}")
                return False
            if self.options.mode == "step":
                step.status = "proposed"
                self._set("awaiting_approval", f"建议动作：{describe(action)}")
                return False
        return self._execute_pending()

    def _capture(self) -> Image.Image | None:
        try:
            with self.devices.hardware(self.device.id):
                frame = self.device.capture()
        except StopRequested:
            raise
        except DeviceError as exc:
            with self._lock:
                self._finish("failed", f"截图失败：{exc}")
            return None
        if frame.width < 100 or frame.height < 100:
            with self._lock:
                self._finish("failed", f"截图尺寸异常：{frame.size}")
            return None
        return frame

    def _new_step(self, frame: Image.Image) -> Step:
        with self._lock:
            self.observations_used += 1
            index = len(self.steps) + 1
            name = f"step_{index:03d}.jpg"
            frame.save(self.run_dir / name, quality=90)
            step = Step(index=index, screenshot=name)
            self.steps.append(step)
            self._save()
            return step

    def _history(self) -> list[dict[str, Any]]:
        return [{"step": s.index, "action": s.action_text, "reason": s.reason, "result": s.result_text}
                for s in self.steps if s.action_text and s.result_text]

    def _history_result(self, step: Step, text: str) -> None:
        step.result_text = text

    def _past_shots(self, current_name: str) -> list[Shot]:
        shots = []
        for name, label in self._shots:
            if name == current_name:
                continue
            with Image.open(self.run_dir / name) as image:
                shots.append(Shot(image=image.copy(), label=label))
        return shots

    def _decide(self, step: Step, frame: Image.Image) -> tuple[Action, str] | None:
        apps = self.device.apps()
        supported = self.device.supported_actions()
        prior = self.steps[-2] if len(self.steps) > 1 else None
        messages = build_messages(
            task=self.options.task, history=self._history(), notes=[n["text"] for n in self.notes],
            apps=apps, actions=sorted(supported), past_shots=self._past_shots(step.screenshot), current=frame,
            thumbnail=self.options.thumbnail, current_max=self.options.current_max)
        self._shots.append((step.screenshot, f"第{step.index}步决策前" + (f"（第{prior.index}步动作后）" if prior else "（任务开始）")))
        write_json(self.run_dir / f"step_{step.index:03d}_request.json", redact_images(messages))
        try:
            reply = self.model.complete(messages, enable_thinking=self.options.enable_thinking)
        except ModelError as exc:
            return self._model_failure(step, str(exc))
        with self._lock:
            self.model_calls += 1
            self.tokens += reply.usage.get("total_tokens", 0)
            step.model = {"elapsed_seconds": reply.elapsed_seconds, "usage": reply.usage,
                          "request_id": reply.request_id, "model": reply.model}
        write_json(self.run_dir / f"step_{step.index:03d}_response.json",
                   {"content": reply.content, "reasoning": reply.reasoning, **step.model})
        try:
            data = extract_json(reply.content)
            step.observation = str(data.get("observation") or "")
            step.last_result = str(data.get("last_result") or "")
            step.reply = str(data.get("reply") or "")
            step.reason = str(data.get("reason") or "")
            risk = str(data.get("risk") or "none").strip().lower()
            step.risk = risk if risk in {"authentication", "payment"} else "none"
            action = parse_action(data.get("action"), apps=apps, supported=supported | {"scroll"})
        except (ModelError, ActionError) as exc:
            return self._model_failure(step, f"模型输出无法执行：{exc}")
        self._model_errors = 0
        with self._lock:
            step.action = action.to_dict()
            step.action_text = describe(action)
            self._save()
        return action, step.risk

    def _model_failure(self, step: Step, message: str) -> None:
        with self._lock:
            self._model_errors += 1
            step.status = "error"
            step.error = message
            if self._model_errors >= self.options.max_consecutive_model_errors:
                self._finish("failed", f"连续 {self._model_errors} 次模型错误，已停止：{message}")
            else:
                self.detail = f"{message}（将重新截图再试）"
                self._save()
        return None

    def _execute_pending(self) -> bool:
        with self._lock:
            action, frame_size = self.pending, self._pending_frame_size
            step = self.steps[-1]
            if action is None or frame_size is None:
                return True
            self.pending = None
            self.actions_used += 1
            self.detail = f"执行：{describe(action)}"
            self._save()
        try:
            with self.devices.hardware(self.device.id):
                if action.type == "recents" and self.device.driver != "adb":
                    receipt = self._recents_via_home(step, frame_size)
                else:
                    receipt = self.device.execute(action, frame_size)
        except StopRequested:
            with self._lock:
                step.status = "error"
                step.error = "执行中被停止，动作可能已部分发出。"
                self._history_result(step, "执行中被停止，效果未知")
            raise
        except DeviceError as exc:
            with self._lock:
                step.status = "error"
                step.error = str(exc)
                self._history_result(step, f"执行失败：{exc}")
                self._finish("failed", f"设备执行失败：{exc}")
            return False
        with self._lock:
            step.status = "executed"
            step.receipt = receipt
            self._history_result(step, "已执行")
            self.detail = ""
            self._save()
        settle = 0 if action.type == "wait" else self.options.settle_seconds
        if settle and not self._stop.wait(settle):
            return True
        return not self._stop.is_set()

    def _recents_via_home(self, step: Step, frame_size: tuple[int, int]) -> dict[str, Any]:
        """Approved navigation rule: go Home first, take a screenshot, then open Recents."""
        home = self.device.execute(Action("home"), frame_size)
        self.device.sleep(self.options.settle_seconds)
        frame = self.device.capture()
        name = f"step_{step.index:03d}_home.jpg"
        frame.save(self.run_dir / name, quality=90)
        with self._lock:
            step.extra_screenshots.append(name)
            self._shots.append((name, f"第{step.index}步：打开最近任务前先回到桌面"))
        recents = self.device.execute(Action("recents"), frame.size)
        return {"home": home, "recents": recents}

    # -- state helpers ------------------------------------------------------------------
    def _set(self, status: str, detail: str) -> None:
        self.status = status
        self.detail = detail
        if status == "paused":
            self.devices.release(self.device.id, self.id)
        self._save()

    def _finish(self, status: str, message: str) -> None:
        self.status = status
        self.result_message = message
        self.detail = ""
        self._stop.set()
        self.devices.release(self.device.id, self.id)
        self._save()

    def _save(self) -> None:
        data = self.snapshot()
        data["history_shots"] = [{"file": name, "label": label} for name, label in self._shots]
        write_json(self.run_dir / "session.json", data)
        if self._on_change:
            self._on_change(self)
