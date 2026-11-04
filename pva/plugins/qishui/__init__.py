"""Soda Music (汽水音乐) single-ad page test (optional plugin).

The user opens one ad by hand. The plugin runs a step-mode task: Qwen reports
what the page shows and proposes an action, but nothing executes on its own.
The only action the user may approve is one ordinary ``back``; reward/claim
buttons are never pressed.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .profile import QishuiAdTestProfile, build_qishui_ad_test_goal

TITLE = "汽水音乐广告页测试"


class StartRequest(BaseModel):
    device_id: str = Field(min_length=1)
    duration_seconds: int = Field(default=180, ge=10, le=3600)


class ConfirmRequest(BaseModel):
    mode: str = Field(pattern=r"^(back|end)$")


class QishuiTests:
    def __init__(self, tasks):
        self.tasks = tasks
        self.records: dict[str, dict[str, Any]] = {}

    def _view(self, record: dict[str, Any]) -> dict[str, Any]:
        session = self.tasks.sessions.get(record["session_id"])
        snap = session.snapshot(full=False) if session else {}
        pending = (snap.get("pending") or {}).get("type")
        if record["status"] == "observing":
            if session is None or snap.get("status") in {"failed", "cancelled"}:
                record.update(status="failed", detail=snap.get("result") or "会话已结束。")
            elif snap.get("status") == "succeeded":
                record.update(status="awaiting_confirmation", confirm="end",
                              detail="模型报告已到达奖励/结束页；请确认结束（不会点击领取）。")
            elif time.time() > record["deadline"]:
                session.cancel()
                record.update(status="expired", detail="测试达到时限。")
        confirm = record.get("confirm") or ("back" if pending == "back" else None)
        return {**record, "confirm": confirm, "model_reply": snap.get("last_reply", ""),
                "proposal": snap.get("pending_text", ""), "session_status": snap.get("status"),
                "run_dir": snap.get("run_dir")}

    def start(self, profile: QishuiAdTestProfile) -> dict[str, Any]:
        session = self.tasks.start(device_id=profile.device_id, task=build_qishui_ad_test_goal(profile),
                                   mode="step", max_actions=1, max_observations=40)
        test_id = uuid.uuid4().hex[:12]
        self.records[test_id] = {"test_id": test_id, "session_id": session.id, "device_id": profile.device_id,
                                 "status": "observing", "detail": "请手动打开一次广告，然后点“观察”。",
                                 "deadline": time.time() + profile.duration_seconds, "confirm": None}
        return self._view(self.records[test_id])

    def get(self, test_id: str) -> dict[str, Any]:
        if test_id not in self.records:
            raise KeyError("测试不存在。")
        return self._view(self.records[test_id])

    def observe(self, test_id: str) -> dict[str, Any]:
        record = self.records[test_id]
        if record["status"] != "observing":
            raise RuntimeError("测试已结束。")
        self.tasks.get(record["session_id"]).observe_once()
        return self._view(record)

    def confirm(self, test_id: str, mode: str) -> dict[str, Any]:
        record = self.records[test_id]
        session = self.tasks.get(record["session_id"])
        if mode == "end":
            if record.get("confirm") != "end":
                raise RuntimeError("当前没有待确认的结束页。")
            session.cancel()
            record.update(status="completed", detail="已记录奖励/结束页；未点击领取。", confirm=None)
        else:
            if session.pending is None or session.pending.type != "back":
                raise RuntimeError("当前没有待确认的普通返回。")
            session.approve()
            record.update(status="completed", detail="已按确认执行一次返回；未点击领取。", confirm=None)
        return self._view(record)

    def cancel(self, test_id: str) -> dict[str, Any]:
        record = self.records[test_id]
        session = self.tasks.sessions.get(record["session_id"])
        if session:
            session.cancel()
        record.update(status="cancelled", detail="用户已停止测试。", confirm=None)
        return self._view(record)


def create(context) -> tuple[APIRouter, QishuiTests]:
    tests = QishuiTests(context.tasks)
    router = APIRouter(prefix="/api/plugins/qishui", tags=["qishui"])

    def guard(call):
        try:
            return call()
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @router.get("")
    def info() -> dict[str, Any]:
        return {"title": TITLE, "tests": [tests._view(r) for r in list(tests.records.values())[-10:]][::-1]}

    @router.post("/start")
    def start(body: StartRequest) -> dict[str, Any]:
        return guard(lambda: tests.start(QishuiAdTestProfile(device_id=body.device_id,
                                                             duration_seconds=body.duration_seconds)))

    @router.get("/{test_id}")
    def get(test_id: str) -> dict[str, Any]:
        return guard(lambda: tests.get(test_id))

    @router.post("/{test_id}/observe")
    def observe(test_id: str) -> dict[str, Any]:
        return guard(lambda: tests.observe(test_id))

    @router.post("/{test_id}/confirm")
    def confirm(test_id: str, body: ConfirmRequest) -> dict[str, Any]:
        return guard(lambda: tests.confirm(test_id, body.mode))

    @router.post("/{test_id}/cancel")
    def cancel(test_id: str) -> dict[str, Any]:
        return guard(lambda: tests.cancel(test_id))

    return router, tests
