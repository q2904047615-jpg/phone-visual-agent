"""Dedicated trial monitor. Generic Qwen sessions keep their original owner."""
from dataclasses import asdict, dataclass, field
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
from threading import Event, RLock
import time
from uuid import uuid4
from features.notifications import NotificationEvent, utc_timestamp
from .flow import Flow
from .vision import LocalOcr, TemplateDetector, interpret
from .transport import AdbTransport, ArmTransport
from .upstream import UpstreamDetector


PHASES = {"search":"寻找福袋", "open_bag":"打开福袋", "open_comment":"打开预填评论",
    "verify_join":"确认参与结果", "joined":"已参与，等待开奖", "read_countdown":"读取倒计时",
    "dismiss_loss":"关闭没抽中结果", "result":"核对开奖结果", "waiting_room":"等待手动换房",
    "empty_comment":"评论没有预填", "inspect_conditions":"识别参与条件", "suspected_win":"疑似中奖，已停手"}
TERMINAL = {"succeeded", "failed", "cancelled", "expired"}


@dataclass
class Record:
    monitor_id: str
    profile: object
    goal: str
    run_dir: Path
    status: str = "starting"
    flow: Flow = field(default_factory=Flow)
    started_at: str = field(default_factory=utc_timestamp)
    updated_at: str = field(default_factory=utc_timestamp)
    deadline_epoch: float = 0
    detail: str = ""
    current_phase: str = "准备启动"
    last_action: str = ""
    last_decision_reason: str = ""
    last_qwen_reply: str = ""
    notified: bool = False
    notification_status: str = "pending"
    physical_actions: int = 0
    observations: int = 0
    next_observation_epoch: float = 0
    screenshot_path: str = ""
    event: Event = field(default_factory=Event)

    @property
    def session_id(self):
        # This dedicated monitor is not a Qwen session.
        return ""

    def snapshot(self):
        return {"monitor_id":self.monitor_id, "feature_id":"lucky_bag", "session_id":"",
            "device_id":self.profile.device_id, "recipient":self.profile.recipient,
            "duration_seconds":self.profile.duration_seconds, "started_at":self.started_at,
            "updated_at":self.updated_at, "deadline_epoch":self.deadline_epoch,
            **{name:getattr(self,name) for name in ("status","detail","current_phase","last_action",
                "last_decision_reason","last_qwen_reply","notified","notification_status",
                "physical_actions","observations","next_observation_epoch","screenshot_path")},
            "run_dir":str(self.run_dir), "draw_at":self.flow.draw_at}


class DedicatedLuckyBagMonitor:
    def __init__(self, *, runtime, hardware_lock, output_root, notification_sink):
        self.runtime, self.hardware_lock = runtime, hardware_lock
        self.output_root = Path(output_root)
        self.notification_sink = notification_sink
        self.state_path = self.output_root / "dedicated_monitors.json"
        self.lock = RLock()
        self.records, self.workers = {}, set()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dedicated-fudai")
        self.config = json.loads((Path(__file__).resolve().parents[3] / "trial.json").read_text(encoding="utf-8"))
        self.restore_error = ""
        try:
            self._restore()
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.records.clear()
            self.restore_error = "福袋状态文件无法恢复，原文件已保留："+str(exc)

    def _save(self):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [{"snapshot":r.snapshot(),"goal":r.goal,"profile":asdict(r.profile),"flow":asdict(r.flow)} for r in self.records.values()]
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding="utf-8")
        os.replace(temporary,self.state_path)

    def _restore(self):
        from .profile import LuckyBagProfile
        if not self.state_path.exists():
            return
        for item in json.loads(self.state_path.read_text(encoding="utf-8")):
            s = item["snapshot"]
            r = Record(s["monitor_id"], LuckyBagProfile(**item["profile"]), item["goal"], Path(s["run_dir"]))
            for name in ("status","detail","current_phase","deadline_epoch","started_at","notified",
                "notification_status","physical_actions","observations","screenshot_path"):
                setattr(r,name,s[name])
            r.flow = Flow(**item["flow"])
            if r.flow.halted and r.status not in TERMINAL:
                r.status = "failed"
                r.detail = "服务在通知期间中断；手机保持停手，请检查通知是否送达。"
            elif r.status not in TERMINAL:
                r.status = "recovery_required"
                r.detail = "服务已重启，点击继续后读取新画面；已经尝试的发送不会重放。"
            self.records[r.monitor_id] = r

    def get(self, monitor_id):
        with self.lock:
            return self.records.get(monitor_id)

    def list(self):
        with self.lock:
            return sorted(self.records.values(),key=lambda r:r.updated_at,reverse=True)

    def start(self, *, profile, goal):
        with self.lock:
            if self.restore_error:
                raise RuntimeError(self.restore_error)
            monitor_id = uuid4().hex
            # Same registry/OS lease as generic tasks and both trial processes.
            self.runtime.device_task_registry.reserve(profile.device_id, "lucky-"+monitor_id)
            try:
                run_dir = self.output_root / ("lucky_bag_"+monitor_id)
                run_dir.mkdir(parents=True)
                r = Record(monitor_id,profile,goal,run_dir,deadline_epoch=time.time()+profile.duration_seconds)
                self.records[monitor_id] = r
                self._save()
                self._submit(r)
                return r
            except Exception:
                self.records.pop(monitor_id,None)
                self.runtime.device_task_registry.release(profile.device_id,"lucky-"+monitor_id)
                raise

    def _require(self, monitor_id):
        r = self.get(monitor_id)
        if r is None:
            raise KeyError("福袋监控不存在。")
        return r

    def pause(self, monitor_id):
        with self.lock:
            r = self._require(monitor_id)
            if r.status not in TERMINAL:
                r.status, r.detail = "paused", "已暂停；继续时重新观察，不重复发送。"
                r.event.set()
                self._save()
            return r

    def resume(self, monitor_id):
        with self.lock:
            r = self._require(monitor_id)
            if r.status in TERMINAL or r.flow.halted:
                raise RuntimeError("该监控已结束，不能继续手机操作。")
            if r.status in {"running","starting"}:
                return r
            if r.monitor_id in self.workers:
                raise RuntimeError("上一轮正在暂停，请稍后点击继续。")
            self.runtime.device_task_registry.reserve(r.profile.device_id,"lucky-"+r.monitor_id)
            r.status, r.detail = "starting", "正在重新观察当前画面。"
            r.event.clear()
            self._save()
            self._submit(r)
            return r

    def cancel(self, monitor_id):
        with self.lock:
            r = self._require(monitor_id)
            if r.status not in TERMINAL:
                r.status, r.detail = "cancelled", "用户已停止福袋监控。"
                r.event.set()
                self._save()
            if monitor_id not in self.workers:
                self.runtime.device_task_registry.release(r.profile.device_id,"lucky-"+monitor_id)
            return r

    def shutdown(self):
        for r in self.list():
            self.pause(r.monitor_id)
        self.executor.shutdown(wait=False,cancel_futures=True)

    def _submit(self, r):
        self.workers.add(r.monitor_id)
        self.executor.submit(self._run,r)

    def _run(self, r):
        try:
            mode = self.config["mode"]
            transport = (AdbTransport if mode == "upstream" else ArmTransport)(self.runtime,r.profile.device_id,self.hardware_lock)
            transport.preflight()
            root = Path(__file__).resolve().parents[3]
            detector = UpstreamDetector(root / "external/douyin_guaji", self.config.get("upstream_offset",0)) if mode == "upstream" else TemplateDetector(Path(__file__).parent / "templates")
            ocr = LocalOcr()
            if mode == "local":
                transport.controller.begin_new_task()
            while True:
                with self.lock:
                    if r.status in TERMINAL or r.status == "paused":
                        break
                    r.status = "running"
                    if time.time() >= r.deadline_epoch:
                        r.status, r.detail = "expired", "已达到监控时限，停止操作。"
                        self._save()
                        break
                frame = transport.capture()
                captured_at = time.time()
                tokens = ocr.read(frame)
                bag = detector.locate(frame)
                page = interpret(tokens,frame.size,bag)
                if page.countdown is not None:
                    # OCR processing time must not postpone the actual draw.
                    page.countdown = max(0,page.countdown-(time.time()-captured_at))
                with self.lock:
                    if r.status != "running":
                        break
                    r.observations += 1
                    r.screenshot_path = str(r.run_dir / f"{r.observations:06d}.jpg")
                    frame.save(r.screenshot_path,quality=92)
                    step = r.flow.decide(page,time.time())
                    r.flow.phase = step.phase
                    r.current_phase = PHASES.get(step.phase,step.phase)
                    r.detail = r.last_decision_reason = step.message
                    r.last_action = "tap" if step.point else "wait" if not step.notify and not step.pause else "stop"
                    r.updated_at = utc_timestamp()
                    r.next_observation_epoch = time.time()+step.wait
                    # Commit send_attempted/halted before any external effect.
                    self._save()
                    (r.run_dir / f"{r.observations:06d}.json").write_text(json.dumps({"captured_at":captured_at,"page":asdict(page),"step":asdict(step),"flow":asdict(r.flow)},ensure_ascii=False,indent=2),encoding="utf-8")
                    if step.notify:
                        self._notify(r)
                        break
                    if step.pause:
                        r.status = "paused"
                        self._save()
                        break
                    if step.point:
                        # Count dispatched attempts even if transport returns an
                        # uncertain error; sending is never automatically retried.
                        r.physical_actions += 1
                        self._save()
                        transport.tap(step.point,frame)
                r.event.wait(min(step.wait,max(0,r.deadline_epoch-time.time())))
                if r.event.is_set():
                    break
        except Exception as exc:
            with self.lock:
                if r.status not in {"paused","cancelled"}:
                    r.status, r.detail = "failed", str(exc) or type(exc).__name__
                self._save()
        finally:
            with self.lock:
                self.workers.discard(r.monitor_id)
                if r.status in TERMINAL:
                    self.runtime.device_task_registry.release(r.profile.device_id,"lucky-"+r.monitor_id)

    def _notify(self, r):
        r.notification_status = "sending"
        self._save()
        event = NotificationEvent(event_id=r.monitor_id,recipient=r.profile.recipient,
            subject=r.profile.subject,body=r.profile.body,observed_at=utc_timestamp(),
            screenshot_path=r.screenshot_path,reason=r.detail)
        try:
            self.notification_sink.publish(event)
            remote = getattr(self.notification_sink,"remote",None)
            r.notified = remote is not None
            r.notification_status = "sent" if r.notified else "local_queue_only"
            r.status = "succeeded" if r.notified else "failed"
            r.detail += "邮件已发送。" if r.notified else "Gmail未配置，通知已保存到本地队列。"
        except Exception as exc:
            r.status, r.notification_status = "failed", "failed"
            r.detail += "邮件发送失败，手机保持停手："+str(exc)
        self._save()


__all__ = ["DedicatedLuckyBagMonitor", "Record"]

