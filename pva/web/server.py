"""Local web console and JSON API.

Run with ``python -m pva`` (see ``pva/__main__.py``). The server only accepts
requests from this computer unless ``PVA_ALLOW_REMOTE=1``.
"""

from __future__ import annotations

import ipaddress
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from pva.actions import ActionError, describe, parse_action
from pva.agent.manager import TaskManager
from pva.config import AppConfig, load_config, output_dir
from pva.devices import DeviceBusy, DeviceError, DeviceManager
from pva.model.qwen import QwenClient, VisionModel
from pva.plugins import PluginContext, load_plugins

STATIC = Path(__file__).resolve().parent / "static"
STATUS_TTL = 3.0


class StartTask(BaseModel):
    device_id: str | None = None
    task: str = Field(min_length=1, max_length=20000)
    mode: str = Field(default="auto", pattern=r"^(auto|step)$")
    max_actions: int | None = Field(default=None, ge=1, le=10000)
    max_observations: int | None = Field(default=None, ge=1, le=20000)
    enable_thinking: bool | None = None


class Resume(BaseModel):
    max_actions: int | None = Field(default=None, ge=1, le=10000)
    max_observations: int | None = Field(default=None, ge=1, le=20000)


class Note(BaseModel):
    text: str = Field(default="", max_length=5000)


class ManualAction(BaseModel):
    action: dict[str, Any]


class Position(BaseModel):
    position: int = Field(ge=1, le=10)


def create_app(config: AppConfig | None = None, model: VisionModel | None = None) -> FastAPI:
    config = config or load_config()
    devices = DeviceManager(config)
    model = model or QwenClient(timeout=float(config.agent["model_timeout_seconds"]))
    out = output_dir()
    tasks = TaskManager(config, devices, model, out)
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        tasks.shutdown()
        for item in plugins.values():
            state = item.get("state")
            if state is not None and hasattr(state, "shutdown"):
                try:
                    state.shutdown()
                except Exception:
                    pass

    app = FastAPI(title="手机视觉 Agent", docs_url="/api/docs", openapi_url="/api/openapi.json", lifespan=lifespan)
    app.state.devices, app.state.tasks, app.state.model = devices, tasks, model
    status_cache: dict[str, tuple[float, dict[str, Any]]] = {}

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        if os.environ.get("PVA_ALLOW_REMOTE") != "1":
            host = request.client.host if request.client else ""
            try:
                allowed = host in {"testclient", "localhost"} or ipaddress.ip_address(host).is_loopback
            except ValueError:
                allowed = False
            if not allowed:
                return JSONResponse({"detail": "只允许本机访问。"}, status_code=403)
        return await call_next(request)

    def device_status(device_id: str, force: bool = False) -> dict[str, Any]:
        cached = status_cache.get(device_id)
        if cached and not force and time.time() - cached[0] < STATUS_TTL:
            data = dict(cached[1])
        else:
            try:
                data = devices.get(device_id).status()
            except Exception as exc:  # status must never break the page
                data = {**devices.get(device_id).base_status(), "online": False, "camera": False, "error": str(exc)}
            status_cache[device_id] = (time.time(), data)
            data = dict(data)
        active = tasks.active_for_device(device_id)
        data["task_id"] = active.id if active else None
        data["owner"] = devices.owner(device_id)
        return data

    def session_or_404(session_id: str):
        try:
            return tasks.get(session_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    def guarded(call):
        try:
            return call()
        except (DeviceBusy, RuntimeError, ValueError) as exc:
            raise HTTPException(409, str(exc)) from exc

    # -- pages ---------------------------------------------------------------------------
    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    # -- status ---------------------------------------------------------------------------
    @app.get("/api/status")
    def status() -> dict[str, Any]:
        return {
            "model": model.status(),
            "mock": config.mock,
            "config_path": str(config.path),
            "output_dir": str(out),
            "default_device": devices.default_device,
            "defaults": {k: config.agent[k] for k in ("max_actions", "max_observations", "enable_thinking")},
            "devices": [device_status(device_id) for device_id in devices.devices],
            "active_tasks": [s.snapshot(full=False) for s in tasks.active()],
            "plugins": {key: {k: v for k, v in item.items() if k != "state"} for key, item in plugins.items()},
        }

    @app.get("/api/devices/{device_id}/status")
    def one_device(device_id: str) -> dict[str, Any]:
        guarded(lambda: devices.get(device_id))
        return device_status(device_id, force=True)

    @app.get("/api/devices/{device_id}/preview.jpg")
    def preview(device_id: str) -> Response:
        try:
            with devices.hardware(device_id, timeout=0.3) as device:
                data = device.preview()
        except DeviceBusy as exc:
            raise HTTPException(409, str(exc)) from exc
        except (DeviceError, OSError, RuntimeError) as exc:
            raise HTTPException(503, str(exc)) from exc
        return Response(data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.post("/api/devices/{device_id}/action")
    def manual_action(device_id: str, body: ManualAction) -> dict[str, Any]:
        """Operator control from the console (only while no task owns the controller)."""
        device = guarded(lambda: devices.get(device_id))
        try:
            action = parse_action(body.action, apps=device.apps(), supported=device.supported_actions() | {"scroll"})
        except ActionError as exc:
            raise HTTPException(422, str(exc)) from exc
        if action.type == "finish":
            raise HTTPException(422, "手动控制不支持 finish。")
        owner = f"manual-{time.time_ns()}"
        guarded(lambda: devices.reserve(device_id, owner))
        try:
            with devices.hardware(device_id):
                device.clear_stop()
                frame_size = device.capture().size if action.type not in {"type", "clear", "enter", "launch"} else (1000, 1000)
                receipt = device.execute(action, frame_size)
        except DeviceError as exc:
            raise HTTPException(409, str(exc)) from exc
        finally:
            devices.release(device_id, owner)
        return {"action": action.to_dict(), "text": describe(action), "receipt": receipt}

    @app.post("/api/devices/{device_id}/machine-position")
    def machine_position(device_id: str, body: Position) -> dict[str, Any]:
        device = guarded(lambda: devices.get(device_id))
        if device.driver != "robot":
            raise HTTPException(422, "只有机械臂设备有机位。")
        from pva.devices import robot, seller_window
        owner = f"manual-{time.time_ns()}"
        guarded(lambda: devices.reserve(device_id, owner))
        try:
            with devices.hardware(device_id):
                hwnd = device._window()
                seller_window.select_machine_position(hwnd, body.position)
                robot._selected_positions[device.title] = body.position
        except (DeviceError, RuntimeError, ValueError, OSError) as exc:
            raise HTTPException(409, str(exc)) from exc
        finally:
            devices.release(device_id, owner)
        return {"device_id": device_id, "position": body.position}

    # -- tasks ------------------------------------------------------------------------------
    @app.post("/api/tasks")
    def start_task(body: StartTask) -> dict[str, Any]:
        session = guarded(lambda: tasks.start(device_id=body.device_id, task=body.task, mode=body.mode,
                                              max_actions=body.max_actions, max_observations=body.max_observations,
                                              enable_thinking=body.enable_thinking))
        return session.snapshot()

    @app.get("/api/tasks")
    def list_tasks(limit: int = 30) -> dict[str, Any]:
        return {"tasks": tasks.history(min(max(limit, 1), 200))}

    @app.get("/api/tasks/{session_id}")
    def get_task(session_id: str) -> dict[str, Any]:
        try:
            return tasks.detail(session_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/api/tasks/{session_id}/files/{name}")
    def task_file(session_id: str, name: str) -> FileResponse:
        try:
            return FileResponse(tasks.screenshot_path(session_id, name))
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.post("/api/tasks/{session_id}/pause")
    def pause(session_id: str) -> dict[str, Any]:
        session = session_or_404(session_id)
        guarded(session.pause)
        return session.snapshot()

    @app.post("/api/tasks/{session_id}/resume")
    def resume(session_id: str, body: Resume | None = None) -> dict[str, Any]:
        session = session_or_404(session_id)
        body = body or Resume()
        guarded(lambda: session.resume(max_actions=body.max_actions, max_observations=body.max_observations))
        return session.snapshot()

    @app.post("/api/tasks/{session_id}/cancel")
    def cancel(session_id: str) -> dict[str, Any]:
        session = session_or_404(session_id)
        session.cancel()
        return session.snapshot()

    @app.post("/api/tasks/{session_id}/approve")
    def approve(session_id: str) -> dict[str, Any]:
        session = session_or_404(session_id)
        guarded(session.approve)
        return session.snapshot()

    @app.post("/api/tasks/{session_id}/reject")
    def reject(session_id: str, body: Note | None = None) -> dict[str, Any]:
        session = session_or_404(session_id)
        guarded(lambda: session.reject((body or Note()).text))
        return session.snapshot()

    @app.post("/api/tasks/{session_id}/observe")
    def observe(session_id: str) -> dict[str, Any]:
        session = session_or_404(session_id)
        guarded(session.observe_once)
        return session.snapshot()

    @app.post("/api/tasks/{session_id}/note")
    def note(session_id: str, body: Note) -> dict[str, Any]:
        session = session_or_404(session_id)
        guarded(lambda: session.add_note(body.text))
        return session.snapshot()

    @app.post("/api/stop")
    def stop_all() -> dict[str, Any]:
        stopped = [s.id for s in tasks.active()]
        for session in tasks.active():
            session.cancel()
        for device in devices.devices.values():
            device.request_stop()
        return {"stopped": stopped}

    plugins = load_plugins(app, PluginContext(devices=devices, tasks=tasks, output_root=out))
    app.state.plugins = plugins
    return app
