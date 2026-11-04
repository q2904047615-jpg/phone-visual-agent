"""Douyin lucky-bag monitor (optional plugin).

Watches a live room the user opened, taps the lucky bag, sends the prefilled
comment, waits for the draw and e-mails on a suspected win. Detection uses local
OCR plus either the pinned upstream detector ("upstream" mode, ADB) or templates
("local" mode, robot arm); see ``settings.json``. It does not use Qwen.

Extra dependencies: ``python -m pva.plugins.lucky_bag.setup``.
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from pva.plugins.notifications import JsonlNotificationOutbox

from .dedicated_monitor import DedicatedLuckyBagMonitor
from .gmail import DurableNotificationRouter, configured_gmail_sink, gmail_configuration_status
from .profile import LuckyBagProfile, build_lucky_bag_goal

TITLE = "抖音福袋监控"


class StartRequest(BaseModel):
    device_id: str = Field(min_length=1)
    recipient: str = Field(min_length=3)
    duration_seconds: int = Field(default=24 * 3600, ge=60, le=7 * 24 * 3600)


def default_recipient() -> str:
    return os.environ.get("LUCKY_BAG_NOTIFICATION_RECIPIENT", "q2904047615@gmail.com")


def create(context) -> tuple[APIRouter, Any]:
    output = context.output_root / "lucky_bag"
    notifications = DurableNotificationRouter(
        outbox=JsonlNotificationOutbox(output / "lucky_bag_notifications.jsonl"),
        remote=configured_gmail_sink())
    monitor = DedicatedLuckyBagMonitor(devices=context.devices, output_root=output,
                                       notification_sink=notifications)
    router = APIRouter(prefix="/api/plugins/lucky-bag", tags=["lucky-bag"])

    def require(monitor_id: str):
        record = monitor.get(monitor_id)
        if record is None:
            raise HTTPException(404, "福袋监控不存在。")
        return record

    @router.get("")
    def info() -> dict[str, Any]:
        return {"title": TITLE, "mode": monitor.config.get("mode"), "notification": gmail_configuration_status(),
                "default_recipient": default_recipient(), "restore_error": monitor.restore_error,
                "monitors": [r.snapshot() for r in monitor.list()[:10]]}

    @router.post("/start")
    def start(body: StartRequest) -> dict[str, Any]:
        if context.tasks.active_for_device(body.device_id):
            raise HTTPException(409, "该设备正在执行通用任务。")
        profile = LuckyBagProfile(device_id=body.device_id, recipient=body.recipient,
                                  duration_seconds=body.duration_seconds)
        try:
            record = monitor.start(profile=profile, goal=build_lucky_bag_goal(profile))
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(409, str(exc)) from exc
        return record.snapshot()

    @router.post("/{monitor_id}/{command}")
    def control(monitor_id: str, command: str) -> dict[str, Any]:
        require(monitor_id)
        handler = {"pause": monitor.pause, "resume": monitor.resume, "cancel": monitor.cancel}.get(command)
        if handler is None:
            raise HTTPException(404, "未知操作。")
        try:
            return handler(monitor_id).snapshot()
        except (RuntimeError, ValueError, KeyError) as exc:
            raise HTTPException(409, str(exc)) from exc

    return router, monitor
