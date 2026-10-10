"""Bounded, configurable seller-window test motions.

This module deliberately uses window messages instead of SetCursorPos,
mouse_event, keyboard input, or window activation.  A successful receipt only
means the messages were queued for the seller window; it does not claim that a
phone or the seller application consumed them.
"""

from __future__ import annotations

import ctypes
import json
import math
import threading
import time
import uuid
from ctypes import wintypes
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable

from agent.infrastructure import seller_window_adapter as seller_gui


BASELINE_WIDTH = seller_gui.BASELINE_CLIENT_WIDTH
BASELINE_HEIGHT = seller_gui.DEFAULT_CAMERA_HEIGHT
CLICK_FIRST_DELAY_SECONDS = 30.0


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} 必须是大于0的有限数字。")
    return value


def _point(value: tuple[int, int] | list[int], name: str) -> tuple[int, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} 必须是 [x, y]。")
    x, y = value
    if any(isinstance(item, bool) or not isinstance(item, int) for item in (x, y)):
        raise ValueError(f"{name} 必须使用整数 PX 坐标。")
    if not (0 <= x < BASELINE_WIDTH and 0 <= y < BASELINE_HEIGHT):
        raise ValueError(f"{name} 超出卖家基准画面 0..{BASELINE_WIDTH - 1} x 0..{BASELINE_HEIGHT - 1}。")
    return int(x), int(y)


@dataclass(frozen=True)
class MotionProfile:
    device_id: str
    kind: str
    point_a: tuple[int, int]
    point_b: tuple[int, int]
    interval_seconds: float
    duration_seconds: float

    def __post_init__(self) -> None:
        if self.kind not in {"click_pair", "swipe"}:
            raise ValueError("kind 必须是 click_pair 或 swipe。")
        if not str(self.device_id).strip():
            raise ValueError("device_id 不能为空。")
        _point(self.point_a, "point_a")
        _point(self.point_b, "point_b")
        if self.kind == "swipe" and self.point_a == self.point_b:
            raise ValueError("上滑起点和终点不能相同。")
        _finite(self.interval_seconds, "interval_seconds")
        _finite(self.duration_seconds, "duration_seconds")

    def snapshot(self) -> dict[str, Any]:
        result = asdict(self)
        result["point_a"] = list(self.point_a)
        result["point_b"] = list(self.point_b)
        result["baseline"] = {"width": BASELINE_WIDTH, "height": BASELINE_HEIGHT}
        return result


def map_baseline_point(point: tuple[int, int], client_width: int, camera_height: int) -> tuple[int, int]:
    """Map seller's documented 540x960 PX coordinates into its client area."""

    if int(client_width) <= 0 or int(camera_height) <= 0:
        raise ValueError("卖家窗口客户区尺寸无效。")
    x, y = _point(point, "point")
    mapped_x = min(int(client_width) - 1, max(0, round(x * int(client_width) / BASELINE_WIDTH)))
    mapped_y = min(int(camera_height) - 1, max(0, round(y * int(camera_height) / BASELINE_HEIGHT)))
    return mapped_x, mapped_y


def _lparam(x: int, y: int) -> int:
    return (int(y) << 16) | (int(x) & 0xFFFF)


class BackgroundWindowTransport:
    """Post input messages to the seller window without foreground activation."""

    WM_MOUSEMOVE = 0x0200
    WM_LBUTTONDOWN = 0x0201
    WM_LBUTTONUP = 0x0202
    WM_RBUTTONDOWN = 0x0204
    WM_RBUTTONUP = 0x0205
    MK_LBUTTON = 0x0001
    MK_RBUTTON = 0x0002

    def __init__(self, controller: Any, *, user32: Any | None = None,
                 find_window: Callable[[str], tuple[int, str]] | None = None,
                 client_geometry: Callable[[int], tuple[int, int, int, int]] | None = None,
                 camera_height: Callable[[int, int], int] | None = None,
                 mock: bool = False) -> None:
        self.controller = controller
        self.user32 = user32 if user32 is not None else seller_gui.user32
        self.find_window = find_window or seller_gui.find_window
        self.client_geometry = client_geometry or seller_gui.client_geometry
        self.camera_height = camera_height or seller_gui.seller_camera_height
        self.mock = bool(mock)

    def _geometry(self) -> tuple[int, int, int, int, int]:
        hwnd, title = self.find_window(getattr(self.controller, "title", seller_gui.DEFAULT_WINDOW_TITLE))
        left, top, width, height = self.client_geometry(hwnd)
        camera_height = self.camera_height(width, height)
        if width <= 0 or camera_height <= 0:
            raise RuntimeError("卖家窗口客户区不可用于测试动作。")
        return int(hwnd), int(width), int(height), int(camera_height), title

    def _post(self, hwnd: int, message: int, flags: int, x: int, y: int) -> None:
        if self.mock:
            return
        post = getattr(self.user32, "PostMessageW", None)
        if post is None:
            raise RuntimeError("当前平台没有可用的 Win32 PostMessageW。")
        if hasattr(post, "argtypes"):
            post.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
            post.restype = wintypes.BOOL
        if not post(hwnd, message, flags, _lparam(x, y)):
            raise OSError("PostMessageW 发送窗口消息失败。")

    def click_pair(self, profile: MotionProfile, stop_event: threading.Event) -> dict[str, Any]:
        hwnd, width, _height, camera_height, _title = self._geometry()
        points = [map_baseline_point(point, width, camera_height) for point in (profile.point_a, profile.point_b)]
        for x, y in points:
            if stop_event.is_set():
                raise InterruptedError("动作已停止。")
            self._post(hwnd, self.WM_LBUTTONDOWN, self.MK_LBUTTON, x, y)
            self._post(hwnd, self.WM_LBUTTONUP, 0, x, y)
        return {
            "kind": profile.kind,
            "mapped_points": [list(point) for point in points],
            "input_events_dispatched": 4,
            "cursor_moved": False,
            "window_activated": False,
            "mechanical_contact_ack": False,
            "device_touched": "unknown",
        }

    def swipe(self, profile: MotionProfile, stop_event: threading.Event) -> dict[str, Any]:
        hwnd, width, _height, camera_height, _title = self._geometry()
        start = map_baseline_point(profile.point_a, width, camera_height)
        end = map_baseline_point(profile.point_b, width, camera_height)
        self._post(hwnd, self.WM_RBUTTONDOWN, self.MK_RBUTTON, *start)
        sent = 1
        try:
            # A finite path gives the seller control the same movement events as
            # a drag, while every wait remains interruptible.
            for index in range(1, 7):
                if stop_event.wait(0.05):
                    raise InterruptedError("动作已停止。")
                x = round(start[0] + (end[0] - start[0]) * index / 6)
                y = round(start[1] + (end[1] - start[1]) * index / 6)
                self._post(hwnd, self.WM_MOUSEMOVE, self.MK_RBUTTON, x, y)
                sent += 1
        finally:
            self._post(hwnd, self.WM_RBUTTONUP, 0, *end)
            sent += 1
        return {
            "kind": profile.kind,
            "mapped_start": list(start),
            "mapped_end": list(end),
            "input_events_dispatched": sent,
            "cursor_moved": False,
            "window_activated": False,
            "mechanical_contact_ack": False,
            "device_touched": "unknown",
        }


class ManualMotionMonitor:
    """One bounded worker per device, with durable summary evidence."""

    ACTIVE = {"starting", "running", "stopping"}

    def __init__(self, *, registry: Any, transport_factory: Callable[[str], BackgroundWindowTransport],
                 output_dir: Path, clock: Callable[[], float] = time.monotonic,
                 wall_clock: Callable[[], float] = time.time) -> None:
        self.registry = registry
        self.transport_factory = transport_factory
        self.output_dir = Path(output_dir)
        self.clock = clock
        self.wall_clock = wall_clock
        self._lock = threading.RLock()
        self._records: dict[str, dict[str, Any]] = {}
        self._events: dict[str, threading.Event] = {}
        self._threads: dict[str, threading.Thread] = {}

    def _write_evidence(self, record: dict[str, Any], event: dict[str, Any] | None = None) -> None:
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            path = self.output_dir / f"{record['motion_id']}.jsonl"
            payload = {"motion": record, "event": event} if event else {"motion": record}
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        except OSError:
            pass

    def list(self, device_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            values = [dict(item) for item in self._records.values()]
        if device_id:
            values = [item for item in values if item["profile"]["device_id"] == device_id]
        return sorted(values, key=lambda item: item.get("started_at") or "", reverse=True)

    def get(self, motion_id: str) -> dict[str, Any] | None:
        with self._lock:
            item = self._records.get(motion_id)
            return dict(item) if item else None

    def start(self, profile: MotionProfile) -> dict[str, Any]:
        motion_id = uuid.uuid4().hex
        self.registry.reserve(profile.device_id, motion_id)
        event = threading.Event()
        record = {
            "motion_id": motion_id,
            "status": "starting",
            "profile": profile.snapshot(),
            "started_at": self.wall_clock(),
            "finished_at": None,
            "cycles_dispatched": 0,
            "last_receipt": None,
            "next_cycle_in_seconds": None,
            "error": None,
        }
        with self._lock:
            if any(item["profile"]["device_id"] == profile.device_id and item["status"] in self.ACTIVE
                   for item in self._records.values()):
                self.registry.release(profile.device_id, motion_id)
                raise RuntimeError("该设备已有固定测试在运行。")
            self._records[motion_id] = record
            self._events[motion_id] = event
        try:
            thread = threading.Thread(target=self._run, args=(motion_id, profile, event),
                                      name=f"manual-motion-{motion_id[:8]}", daemon=True)
            with self._lock:
                self._threads[motion_id] = thread
            thread.start()
        except BaseException:
            self.registry.release(profile.device_id, motion_id)
            with self._lock:
                self._records.pop(motion_id, None)
                self._events.pop(motion_id, None)
                self._threads.pop(motion_id, None)
            raise
        return dict(record)

    def _update(self, motion_id: str, **changes: Any) -> dict[str, Any]:
        with self._lock:
            record = self._records[motion_id]
            record.update(changes)
            snapshot = dict(record)
        self._write_evidence(snapshot)
        return snapshot

    def _run(self, motion_id: str, profile: MotionProfile, event: threading.Event) -> None:
        start = self.clock()
        deadline = start + profile.duration_seconds
        first_delay = CLICK_FIRST_DELAY_SECONDS if profile.kind == "click_pair" else profile.interval_seconds
        released = False

        def finish(**changes: Any) -> None:
            """Publish a terminal record only after releasing the device lease."""
            nonlocal released
            self.registry.release(profile.device_id, motion_id)
            released = True
            self._update(motion_id, **changes)

        try:
            self._update(motion_id, status="running", next_cycle_in_seconds=first_delay)
            while True:
                remaining = deadline - self.clock()
                if remaining <= 0:
                    finish(status="completed", finished_at=self.wall_clock(), next_cycle_in_seconds=0)
                    return
                wait_for = min(first_delay if self.get(motion_id)["cycles_dispatched"] == 0 else profile.interval_seconds, remaining)
                self._update(motion_id, next_cycle_in_seconds=max(0.0, wait_for))
                if event.wait(wait_for) or self.clock() >= deadline:
                    status = "stopped" if event.is_set() else "completed"
                    finish(status=status, finished_at=self.wall_clock(), next_cycle_in_seconds=0)
                    return
                if event.is_set():
                    finish(status="stopped", finished_at=self.wall_clock(), next_cycle_in_seconds=0)
                    return
                transport = self.transport_factory(profile.device_id)
                receipt = (transport.click_pair(profile, event) if profile.kind == "click_pair"
                           else transport.swipe(profile, event))
                with self._lock:
                    count = int(self._records[motion_id]["cycles_dispatched"]) + 1
                self._update(motion_id, cycles_dispatched=count, last_receipt=receipt)
                first_delay = profile.interval_seconds
        except InterruptedError:
            finish(status="stopped", finished_at=self.wall_clock(), next_cycle_in_seconds=0)
        except BaseException as exc:
            finish(status="failed", finished_at=self.wall_clock(), error=str(exc), next_cycle_in_seconds=0)
        finally:
            if not released:
                self.registry.release(profile.device_id, motion_id)

    def stop(self, motion_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._records.get(motion_id)
            event = self._events.get(motion_id)
            if record is None:
                raise KeyError(motion_id)
            if record["status"] in {"completed", "stopped", "failed"}:
                return dict(record)
            record["status"] = "stopping"
        if event:
            event.set()
        return self.get(motion_id) or record

    def stop_all(self, device_id: str | None = None) -> list[dict[str, Any]]:
        result = []
        for item in self.list(device_id):
            if item["status"] in self.ACTIVE:
                result.append(self.stop(item["motion_id"]))
        return result

    def shutdown(self) -> None:
        self.stop_all()
        with self._lock:
            threads = list(self._threads.values())
        for thread in threads:
            thread.join(timeout=2.0)


__all__ = ["BASELINE_WIDTH", "BASELINE_HEIGHT", "CLICK_FIRST_DELAY_SECONDS", "MotionProfile",
           "map_baseline_point", "BackgroundWindowTransport", "ManualMotionMonitor"]
