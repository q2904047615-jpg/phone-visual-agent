"""Long-running supervisor for the additive lucky-bag feature."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from threading import RLock
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Mapping, Protocol
from uuid import uuid4

from features.notifications import JsonlNotificationOutbox, NotificationEvent, NotificationSink, utc_timestamp
from .profile import LuckyBagProfile


class LuckyBagSessionGateway(Protocol):
    def start(self, *, goal: str, device_id: str, run_dir: Path) -> Mapping[str, Any]: ...
    def get(self, session_id: str) -> Mapping[str, Any]: ...
    def auto(self, session_id: str, *, max_physical_actions: int, max_observations: int) -> Mapping[str, Any]: ...
    def pause(self, session_id: str) -> Mapping[str, Any]: ...
    def cancel(self, session_id: str) -> Mapping[str, Any]: ...


@dataclass
class LuckyBagMonitorRecord:
    monitor_id: str
    profile: LuckyBagProfile
    goal: str
    run_dir: Path
    status: str = "starting"
    session_id: str = ""
    started_at: str = field(default_factory=utc_timestamp)
    updated_at: str = field(default_factory=utc_timestamp)
    deadline_monotonic: float = 0.0
    deadline_epoch: float = 0.0
    detail: str = ""
    notified: bool = False

    def snapshot(self) -> dict[str, Any]:
        return {
            "monitor_id": self.monitor_id,
            "feature_id": "lucky_bag",
            "status": self.status,
            "session_id": self.session_id,
            "device_id": self.profile.device_id,
            "recipient": self.profile.recipient,
            "duration_seconds": self.profile.duration_seconds,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "deadline_epoch": self.deadline_epoch,
            "detail": self.detail,
            "notified": self.notified,
            "run_dir": str(self.run_dir),
        }


class LuckyBagMonitor:
    """Resume the existing universal session in bounded cumulative chunks."""

    def __init__(self, *, gateway: LuckyBagSessionGateway, output_root: Path,
        chunk_actions: int = 20, chunk_observations: int = 40,
        outbox_path: Path | None = None, notification_sink: NotificationSink | None = None,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        state_path: Path | None = None) -> None:
        if chunk_actions < 1 or chunk_observations < 1:
            raise ValueError("监督器分段预算必须为正整数。")
        self.gateway = gateway
        self.output_root = Path(output_root)
        self.chunk_actions = int(chunk_actions)
        self.chunk_observations = int(chunk_observations)
        self.outbox = JsonlNotificationOutbox(outbox_path or self.output_root / "lucky_bag_notifications.jsonl")
        self.notification_sink = notification_sink or self.outbox
        self.clock = clock
        self.wall_clock = wall_clock
        self.state_path = Path(state_path or self.output_root / "lucky_bag_monitors.json")
        self._records: dict[str, LuckyBagMonitorRecord] = {}
        self._lock = RLock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lucky-bag")
        self._restore()

    def start(self, *, profile: LuckyBagProfile, goal: str) -> LuckyBagMonitorRecord:
        monitor_id = uuid4().hex
        run_dir = self.output_root / f"lucky_bag_{monitor_id[:12]}"
        run_dir.mkdir(parents=True, exist_ok=True)
        record = LuckyBagMonitorRecord(
            monitor_id=monitor_id, profile=profile, goal=goal, run_dir=run_dir,
            deadline_monotonic=self.clock() + profile.duration_seconds,
            deadline_epoch=self.wall_clock() + profile.duration_seconds,
        )
        with self._lock:
            self._records[monitor_id] = record
            self._persist_locked()
        self._executor.submit(self._run, monitor_id)
        return record

    def get(self, monitor_id: str) -> LuckyBagMonitorRecord | None:
        with self._lock:
            return self._records.get(monitor_id)

    def list(self) -> list[LuckyBagMonitorRecord]:
        with self._lock:
            return sorted(
                self._records.values(),
                key=lambda item: item.updated_at,
                reverse=True,
            )

    def pause(self, monitor_id: str) -> LuckyBagMonitorRecord:
        record = self._require(monitor_id)
        if record.session_id:
            self.gateway.pause(record.session_id)
        self._set(record, "paused", "用户已暂停福袋监控。")
        return record

    def resume(self, monitor_id: str) -> LuckyBagMonitorRecord:
        record = self._require(monitor_id)
        if record.status in {"succeeded", "expired", "failed", "cancelled"}:
            raise RuntimeError(f"当前监控状态不能恢复：{record.status}")
        if record.status == "recovery_required":
            # The previous process-owned session cannot be deserialized safely.
            # Start a fresh observation only after explicit user action.
            record.session_id = ""
        self._set(record, "running", "正在恢复并重新观察。")
        self._executor.submit(
            self._run if not record.session_id else self._advance,
            record.monitor_id,
        )
        return record

    def cancel(self, monitor_id: str) -> LuckyBagMonitorRecord:
        record = self._require(monitor_id)
        if record.session_id:
            self.gateway.cancel(record.session_id)
        self._set(record, "cancelled", "用户已停止福袋监控。")
        return record

    def shutdown(self) -> None:
        with self._lock:
            self._persist_locked()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _require(self, monitor_id: str) -> LuckyBagMonitorRecord:
        record = self.get(monitor_id)
        if record is None:
            raise KeyError("福袋监控不存在。")
        return record

    def _set(self, record: LuckyBagMonitorRecord, status: str, detail: str) -> None:
        with self._lock:
            record.status = status
            record.detail = detail
            record.updated_at = utc_timestamp()
            self._persist_locked()

    @staticmethod
    def _profile_from_snapshot(payload: Mapping[str, Any]) -> LuckyBagProfile:
        return LuckyBagProfile(
            device_id=str(payload.get("device_id") or ""),
            recipient=str(payload.get("recipient") or ""),
            duration_seconds=int(payload.get("duration_seconds") or 0),
            app_alias=str(payload.get("app_alias") or "抖音"),
            subject=str(payload.get("subject") or "疑似中奖"),
            body=str(payload.get("body") or "疑似中奖"),
        )

    def _persist_locked(self) -> None:
        rows = []
        for record in self._records.values():
            item = record.snapshot()
            item.update({
                "goal": record.goal,
                "run_dir": str(record.run_dir),
                "profile": {
                    "device_id": record.profile.device_id,
                    "recipient": record.profile.recipient,
                    "duration_seconds": record.profile.duration_seconds,
                    "app_alias": record.profile.app_alias,
                    "subject": record.profile.subject,
                    "body": record.profile.body,
                },
            })
            rows.append(item)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, self.state_path)

    def _restore(self) -> None:
        try:
            rows = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError):
            return
        if not isinstance(rows, list):
            return
        now_wall = self.wall_clock()
        now_mono = self.clock()
        with self._lock:
            for item in rows:
                if not isinstance(item, Mapping):
                    continue
                try:
                    profile_payload = item.get("profile")
                    if not isinstance(profile_payload, Mapping):
                        continue
                    profile = self._profile_from_snapshot(profile_payload)
                    deadline_epoch = float(item.get("deadline_epoch") or 0.0)
                    remaining = max(0.0, deadline_epoch - now_wall)
                    status = str(item.get("status") or "failed")
                    if status in {"starting", "running", "waiting_confirmation", "paused", "recovery_required"}:
                        if remaining <= 0:
                            status = "expired"
                            detail = "恢复时发现监控已达到时限。"
                        else:
                            status = "recovery_required"
                            detail = "API 已重启；原 Agent 会话不能安全反序列化，等待用户恢复后重新观察。"
                    else:
                        detail = str(item.get("detail") or "")
                    record = LuckyBagMonitorRecord(
                        monitor_id=str(item.get("monitor_id") or ""),
                        profile=profile,
                        goal=str(item.get("goal") or ""),
                        run_dir=Path(str(item.get("run_dir") or self.output_root)),
                        status=status,
                        session_id="",
                        started_at=str(item.get("started_at") or utc_timestamp()),
                        updated_at=utc_timestamp(),
                        deadline_monotonic=now_mono + remaining,
                        deadline_epoch=deadline_epoch,
                        detail=detail,
                        notified=bool(item.get("notified")),
                    )
                    if record.monitor_id:
                        self._records[record.monitor_id] = record
                except (TypeError, ValueError):
                    continue
            self._persist_locked()

    @staticmethod
    def _session_snapshot(result: Mapping[str, Any]) -> Mapping[str, Any]:
        value = result.get("session") if isinstance(result.get("session"), Mapping) else result
        if not isinstance(value, Mapping):
            raise RuntimeError("会话网关没有返回有效状态。")
        return value

    def _run(self, monitor_id: str) -> None:
        record = self._require(monitor_id)
        try:
            started = self.gateway.start(goal=record.goal, device_id=record.profile.device_id, run_dir=record.run_dir)
            session = self._session_snapshot(started)
            record.session_id = str(session.get("session_id") or "")
            if not record.session_id:
                raise RuntimeError("会话启动没有返回session_id。")
            with self._lock:
                self._persist_locked()
            if record.status in {"paused", "cancelled"}:
                if record.status == "paused":
                    self.gateway.pause(record.session_id)
                else:
                    self.gateway.cancel(record.session_id)
                return
            self._set(record, "running", "福袋监控已启动。")
            self._advance(monitor_id)
        except Exception as exc:
            self._set(record, "failed", str(exc) or type(exc).__name__)

    def _advance(self, monitor_id: str) -> None:
        record = self._require(monitor_id)
        try:
            while True:
                if record.status in {"paused", "cancelled", "failed", "succeeded", "expired"}:
                    return
                if self.clock() >= record.deadline_monotonic:
                    if record.session_id:
                        self.gateway.pause(record.session_id)
                    self._set(record, "expired", "已达到福袋监控时限。")
                    return
                current = self._session_snapshot(self.gateway.get(record.session_id))
                status = str(current.get("status") or "")
                if status == "succeeded":
                    self._notify(record, current)
                    self._set(record, "succeeded", "Qwen报告任务终态，已记录通知事件。")
                    return
                if status in {"failed", "blocked", "cancelled"}:
                    self._set(record, status, str(current.get("failed_reason") or current.get("blocked_reason") or "会话已结束。"))
                    return
                if status == "paused" and record.status == "paused":
                    return
                if status == "awaiting_effect_confirmation":
                    self._set(record, "waiting_confirmation", "当前会话涉及登录或付款，需要用户确认后才能继续。")
                    return
                budget = current.get("execution_budget") if isinstance(current.get("execution_budget"), Mapping) else {}
                actions = int(current.get("physical_actions") or 0)
                observations = int(budget.get("observation_attempts") or 0)
                result = self.gateway.auto(
                    record.session_id,
                    max_physical_actions=actions + self.chunk_actions,
                    max_observations=observations + self.chunk_observations,
                )
                updated = self._session_snapshot(result)
                updated_status = str(updated.get("status") or "")
                if updated_status == "succeeded":
                    self._notify(record, updated)
                    self._set(record, "succeeded", "Qwen报告任务终态，已记录通知事件。")
                    return
                if updated_status in {"failed", "blocked", "cancelled"}:
                    self._set(record, updated_status, str(updated.get("failed_reason") or updated.get("blocked_reason") or "会话已结束。"))
                    return
                if updated_status == "paused":
                    self._set(record, "paused", "用户已暂停福袋监控。")
                    return
                if updated_status == "awaiting_effect_confirmation":
                    self._set(record, "waiting_confirmation", "当前会话涉及登录或付款，需要用户确认后才能继续。")
                    return
                self._set(record, "running", "已续接下一段累计预算。")
        except Exception as exc:
            self._set(record, "failed", str(exc) or type(exc).__name__)

    def _notify(self, record: LuckyBagMonitorRecord, session: Mapping[str, Any]) -> None:
        if record.notified:
            return
        decision = session.get("qwen_decision") if isinstance(session.get("qwen_decision"), Mapping) else {}
        reason = str(decision.get("reason") or session.get("failed_reason") or "")
        evidence = sorted(record.run_dir.glob("*.jpg"))
        screenshot_path = str(evidence[-1]) if evidence else str(record.run_dir)
        self.notification_sink.publish(NotificationEvent(
            event_id=record.monitor_id, recipient=record.profile.recipient,
            subject=record.profile.subject, body=record.profile.body,
            observed_at=utc_timestamp(), screenshot_path=screenshot_path, reason=reason,
        ))
        record.notified = True


__all__ = ["LuckyBagMonitor", "LuckyBagMonitorRecord", "LuckyBagSessionGateway"]
