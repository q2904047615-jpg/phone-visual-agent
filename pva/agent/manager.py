"""Creates tasks, tracks live ones, and lists finished ones from disk."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from pva.agent.session import ACTIVE, Session, TaskOptions
from pva.config import AppConfig
from pva.devices import DeviceManager
from pva.model.qwen import VisionModel
from pva.util import read_json


class TaskManager:
    def __init__(self, config: AppConfig, devices: DeviceManager, model: VisionModel, output_root: Path):
        self.config = config
        self.devices = devices
        self.model = model
        self.output_root = Path(output_root) / "tasks"
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def start(self, *, device_id: str | None, task: str, mode: str = "auto", max_actions: int | None = None,
              max_observations: int | None = None, enable_thinking: bool | None = None) -> Session:
        task = (task or "").strip()
        if not task:
            raise ValueError("请输入任务。")
        if mode not in {"auto", "step"}:
            raise ValueError("mode 只能是 auto 或 step。")
        agent = self.config.agent
        options = TaskOptions(
            device_id=self.devices.get(device_id).id, task=task, mode=mode,
            max_actions=int(max_actions or agent["max_actions"]),
            max_observations=int(max_observations or agent["max_observations"]),
            enable_thinking=bool(agent["enable_thinking"] if enable_thinking is None else enable_thinking),
            settle_seconds=float(agent["settle_seconds"]),
            thumbnail=tuple(agent["history_thumbnail"]), current_max=tuple(agent["current_image_max"]),
            max_consecutive_model_errors=int(agent["max_consecutive_model_errors"]),
        )
        if options.max_actions < 1 or options.max_observations < 1:
            raise ValueError("动作和观察上限必须大于 0。")
        session = Session(options, devices=self.devices, model=self.model, output_root=self.output_root)
        session.start()
        with self._lock:
            self.sessions[session.id] = session
        return session

    def get(self, session_id: str) -> Session:
        session = self.sessions.get(session_id)
        if session is None:
            raise KeyError("任务不存在或服务已重启（历史任务只读）。")
        return session

    def active(self) -> list[Session]:
        return [s for s in self.sessions.values() if s.status in ACTIVE]

    def active_for_device(self, device_id: str) -> Session | None:
        return next((s for s in self.active() if s.device.id == device_id), None)

    def detail(self, session_id: str) -> dict[str, Any]:
        if session_id in self.sessions:
            return self.sessions[session_id].snapshot()
        run_dir = self._run_dir(session_id)
        data = read_json(run_dir / "session.json", default=None) if run_dir else None
        if not data:
            raise KeyError("任务不存在。")
        if data.get("status") in ACTIVE:
            data["status"], data["detail"] = "interrupted", "服务重启时任务未结束。"
        data["readonly"] = True
        return data

    def history(self, limit: int = 30) -> list[dict[str, Any]]:
        items = []
        for run_dir in sorted(self.output_root.iterdir(), reverse=True)[: limit * 2]:
            session_id = run_dir.name.rsplit("_", 1)[-1]
            if session_id in self.sessions:
                items.append(self.sessions[session_id].snapshot(full=False))
                continue
            data = read_json(run_dir / "session.json", default=None)
            if not data:
                continue
            data.pop("steps", None)
            data.pop("notes", None)
            data.pop("history_shots", None)
            if data.get("status") in ACTIVE:
                data["status"] = "interrupted"
            items.append(data)
            if len(items) >= limit:
                break
        return items[:limit]

    def screenshot_path(self, session_id: str, name: str) -> Path:
        if "/" in name or "\\" in name or not name.endswith(".jpg"):
            raise KeyError("非法文件名。")
        run_dir = self.sessions[session_id].run_dir if session_id in self.sessions else self._run_dir(session_id)
        if run_dir is None or not (run_dir / name).is_file():
            raise KeyError("截图不存在。")
        return run_dir / name

    def _run_dir(self, session_id: str) -> Path | None:
        if not session_id.isalnum():
            return None
        return next(iter(sorted(self.output_root.glob(f"*_{session_id}"))), None)

    def shutdown(self) -> None:
        for session in self.active():
            session.cancel()
