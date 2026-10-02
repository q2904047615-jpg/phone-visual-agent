"""Single-run, manual-confirmation supervisor for Soda Music ad-page tests.

This feature deliberately does not run the generic autonomous loop.  It only
starts a normal visual session, lets the operator request fresh observations,
and gates the one optional ``back`` action behind an explicit confirmation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from threading import RLock
import time
from typing import Any, Callable, Mapping, Protocol
from uuid import uuid4

from features.notifications import utc_timestamp

from .profile import QishuiAdTestProfile


class QishuiAdTestSessionGateway(Protocol):
    def start(self, *, goal: str, device_id: str, run_dir: Path) -> Mapping[str, Any]: ...
    def get(self, session_id: str) -> Mapping[str, Any]: ...
    def observe(self, session_id: str, *, device_id: str) -> Mapping[str, Any]: ...
    def confirm(self, session_id: str, *, device_id: str, confirmation: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def cancel(self, session_id: str, *, device_id: str) -> Mapping[str, Any]: ...


@dataclass
class QishuiAdTestRecord:
    test_id: str
    profile: QishuiAdTestProfile
    goal: str
    run_dir: Path
    status: str = "starting"
    phase: str = "manual_start"
    session_id: str = ""
    started_at: str = field(default_factory=utc_timestamp)
    updated_at: str = field(default_factory=utc_timestamp)
    deadline_monotonic: float = 0.0
    deadline_epoch: float = 0.0
    observation_count: int = 0
    physical_actions: int = 0
    confirmation_mode: str = ""
    detail: str = ""
    failure_reason: str = ""
    model_report: str = ""
    evidence_paths: list[str] = field(default_factory=list)

    def snapshot(self) -> dict[str, Any]:
        return {
            "test_id": self.test_id,
            "feature_id": "qishui_ad_test",
            "status": self.status,
            "phase": self.phase,
            "session_id": self.session_id,
            "device_id": self.profile.device_id,
            "app_alias": self.profile.app_alias,
            "duration_seconds": self.profile.duration_seconds,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "deadline_epoch": self.deadline_epoch,
            "observation_count": self.observation_count,
            "physical_actions": self.physical_actions,
            "confirmation_mode": self.confirmation_mode,
            "detail": self.detail,
            "failure_reason": self.failure_reason,
            "model_report": self.model_report,
            "evidence_paths": list(self.evidence_paths),
            "run_dir": str(self.run_dir),
            "manual_only": True,
            "claim_action_allowed": False,
        }


class QishuiAdTestMonitor:
    """Keep one bounded ad-page test and never advance it autonomously."""

    _TERMINAL = {"completed", "failed", "cancelled", "expired"}

    def __init__(self, *, gateway: QishuiAdTestSessionGateway, output_root: Path,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        state_path: Path | None = None) -> None:
        self.gateway = gateway
        self.output_root = Path(output_root)
        self.clock = clock
        self.wall_clock = wall_clock
        self.state_path = Path(state_path or self.output_root / "qishui_ad_tests.json")
        self._records: dict[str, QishuiAdTestRecord] = {}
        self._lock = RLock()

    def start(self, *, profile: QishuiAdTestProfile, goal: str) -> QishuiAdTestRecord:
        test_id = uuid4().hex
        run_dir = self.output_root / f"qishui_ad_test_{test_id[:12]}"
        run_dir.mkdir(parents=True, exist_ok=True)
        record = QishuiAdTestRecord(
            test_id=test_id,
            profile=profile,
            goal=goal,
            run_dir=run_dir,
            deadline_monotonic=self.clock() + profile.duration_seconds,
            deadline_epoch=self.wall_clock() + profile.duration_seconds,
        )
        try:
            started = self.gateway.start(goal=goal, device_id=profile.device_id, run_dir=run_dir)
            session = self._session_snapshot(started)
            record.session_id = str(session.get("session_id") or "")
            if not record.session_id:
                raise RuntimeError("会话启动没有返回session_id。")
            record.status = "observing"
            record.phase = "manual_observe"
            record.detail = "测试已启动；请用户手动打开一次广告，再请求观察。"
            self._record_session(record, session)
        except Exception as exc:
            record.status = "failed"
            record.phase = "stopped"
            record.failure_reason = str(exc) or type(exc).__name__
            record.detail = "测试会话启动失败，未执行设备动作。"
        with self._lock:
            self._records[test_id] = record
            self._persist_locked()
        return record

    def get(self, test_id: str) -> QishuiAdTestRecord | None:
        with self._lock:
            return self._records.get(test_id)

    def list(self) -> list[QishuiAdTestRecord]:
        with self._lock:
            return sorted(self._records.values(), key=lambda item: item.updated_at, reverse=True)

    def observe(self, test_id: str, *, device_id: str) -> QishuiAdTestRecord:
        record = self._require(test_id)
        self._require_device(record, device_id)
        if record.status in self._TERMINAL or record.status == "awaiting_confirmation":
            return record
        if self.clock() >= record.deadline_monotonic:
            return self._fail(record, "测试达到时限，未能确认广告页面状态。")
        try:
            current = self._session_snapshot(self.gateway.observe(record.session_id, device_id=device_id))
            record.observation_count += 1
            self._record_session(record, current)
            self._map_session(record, current)
        except Exception as exc:
            self._fail(record, str(exc) or type(exc).__name__)
        self._save(record)
        return record

    def confirm(self, test_id: str, *, device_id: str, confirmed: bool,
        mode: str, confirmation: Mapping[str, Any] | None) -> QishuiAdTestRecord:
        record = self._require(test_id)
        self._require_device(record, device_id)
        if confirmed is not True:
            raise ValueError("必须明确确认一次人工操作。")
        if record.status != "awaiting_confirmation":
            raise ValueError("当前测试没有待确认的人工操作。")
        if mode != record.confirmation_mode:
            raise ValueError("确认模式与当前待确认操作不一致。")
        if mode == "end":
            record.status = "completed"
            record.phase = "completed"
            record.detail = "已记录奖励/结束页观察结果；未点击领取金币。"
            self._save(record)
            return record
        if mode != "back" or not confirmation:
            raise ValueError("普通返回必须提交当前截图绑定的confirmation scope。")
        session = self._session_snapshot(self.gateway.get(record.session_id))
        if self._proposal_action(session) != "back":
            raise ValueError("当前页面没有唯一的普通back待确认动作。")
        try:
            updated = self._session_snapshot(self.gateway.confirm(
                record.session_id, device_id=device_id, confirmation=confirmation,
            ))
            record.physical_actions += 1
            if record.physical_actions > 1:
                raise RuntimeError("测试会话超过一个物理动作预算。")
            self._record_session(record, updated)
            record.status = "completed"
            record.phase = "completed"
            record.detail = "已按人工确认执行一次普通返回；未点击领取金币。"
            self._save(record)
            return record
        except Exception as exc:
            self._fail(record, str(exc) or type(exc).__name__)
            raise

    def cancel(self, test_id: str, *, device_id: str) -> QishuiAdTestRecord:
        record = self._require(test_id)
        self._require_device(record, device_id)
        if record.session_id and record.status not in self._TERMINAL:
            self.gateway.cancel(record.session_id, device_id=device_id)
        record.status = "cancelled"
        record.phase = "stopped"
        record.detail = "用户已停止广告页面测试；未点击领取金币。"
        self._save(record)
        return record

    def shutdown(self) -> None:
        with self._lock:
            self._persist_locked()

    def _require(self, test_id: str) -> QishuiAdTestRecord:
        record = self.get(test_id)
        if record is None:
            raise KeyError("汽水音乐广告测试不存在。")
        return record

    @staticmethod
    def _require_device(record: QishuiAdTestRecord, device_id: str) -> None:
        if device_id != record.profile.device_id:
            raise ValueError("请求device_id与广告测试设备不一致。")

    @staticmethod
    def _session_snapshot(result: Mapping[str, Any]) -> Mapping[str, Any]:
        value = result.get("session") if isinstance(result.get("session"), Mapping) else result
        if not isinstance(value, Mapping):
            raise RuntimeError("会话网关没有返回有效状态。")
        return value

    @staticmethod
    def _proposal_action(session: Mapping[str, Any]) -> str:
        proposal = session.get("proposal")
        if isinstance(proposal, Mapping):
            action = proposal.get("action")
            if isinstance(action, Mapping):
                return str(action.get("action") or "").strip()
            return str(action or "").strip()
        decision = session.get("qwen_decision")
        if isinstance(decision, Mapping):
            action = decision.get("action")
            if isinstance(action, Mapping):
                return str(action.get("action") or "").strip()
        return ""

    @staticmethod
    def _model_report(session: Mapping[str, Any]) -> str:
        decision = session.get("qwen_decision")
        if isinstance(decision, Mapping):
            return str(decision.get("reason") or decision.get("postcondition") or "").strip()[:500]
        return ""

    def _map_session(self, record: QishuiAdTestRecord, session: Mapping[str, Any]) -> None:
        status = str(session.get("status") or "")
        if status in {"failed", "blocked", "cancelled"}:
            return self._fail(record, str(session.get("failed_reason") or session.get("blocked_reason") or "会话报告失败或状态不确定。"))
        if status == "awaiting_effect_confirmation":
            return self._fail(record, "出现效果确认请求；广告测试不会批准登录、付款或其他外部效果。")
        if status == "succeeded":
            record.status = "awaiting_confirmation"
            record.phase = "reward_or_end_page"
            record.confirmation_mode = "end"
            record.detail = "已到达模型报告的奖励/结束页；等待用户确认结束，不会点击领取金币。"
            return
        if status == "awaiting_confirmation":
            action = self._proposal_action(session)
            if action == "back":
                record.status = "awaiting_confirmation"
                record.phase = "reward_or_end_page"
                record.confirmation_mode = "back"
                record.detail = "页面需要一次普通返回；等待用户明确确认，不会自动执行。"
                return
            if action == "wait_for_change":
                record.status = "observing"
                record.phase = "manual_observe"
                record.detail = "广告仍需等待或重新观察；本功能不自动点击、不自动循环。"
                return
            record.status = "observing"
            record.phase = "manual_observe"
            record.detail = "模型提出了非返回动作；本测试不执行该动作，请人工检查页面后继续或停止。"
            return
        if status in {"paused", "needs_reobservation", "budget_paused"}:
            record.status = "observing"
            record.phase = "manual_observe"
            record.detail = "会话已暂停，等待人工再次观察；未执行领取或其他广告动作。"
            return
        record.status = "observing"
        record.phase = "manual_observe"
        record.detail = "已记录当前页面状态，等待下一次人工观察。"

    def _record_session(self, record: QishuiAdTestRecord, session: Mapping[str, Any]) -> None:
        record.physical_actions = max(record.physical_actions, int(session.get("physical_actions") or 0))
        evidence = session.get("evidence")
        if isinstance(evidence, list):
            record.evidence_paths = [str(item) for item in evidence if str(item).strip()]
        record.model_report = self._model_report(session)

    def _fail(self, record: QishuiAdTestRecord, reason: str) -> QishuiAdTestRecord:
        record.status = "failed"
        record.phase = "stopped"
        record.failure_reason = reason
        record.detail = "状态不确定或测试无法继续，已停止；未点击领取金币。"
        self._save(record)
        return record

    def _save(self, record: QishuiAdTestRecord) -> None:
        with self._lock:
            record.updated_at = utc_timestamp()
            self._persist_locked()

    def _persist_locked(self) -> None:
        rows = []
        for record in self._records.values():
            item = record.snapshot()
            item.update({"goal": record.goal, "profile": {
                "device_id": record.profile.device_id,
                "duration_seconds": record.profile.duration_seconds,
                "app_alias": record.profile.app_alias,
            }})
            rows.append(item)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, self.state_path)


__all__ = ["QishuiAdTestMonitor", "QishuiAdTestRecord", "QishuiAdTestSessionGateway"]
