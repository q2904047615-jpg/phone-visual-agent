"""Device interface: capture a screenshot, execute one action."""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from typing import Any

from PIL import Image

from pva.actions import Action, NAV_ACTIONS, PATH_ACTIONS, POINT_ACTIONS
from pva.config import DeviceConfig
from pva.devices.adb import AdbBridge, AdbError
from pva.util import jpeg_bytes


class DeviceError(RuntimeError):
    """A technical device fault; the task must stop and report it."""


class StopRequested(DeviceError):
    pass


class Device(ABC):
    """One phone. Subclasses implement capture and the physical primitives."""

    driver = "base"

    def __init__(self, config: DeviceConfig, adb_executable: str = "adb"):
        self.config = config
        self.id = config.id
        self.name = config.name
        self.adb = AdbBridge(adb_executable, config.adb_serial) if config.adb_serial else None
        self._stop = threading.Event()

    # -- identity / capability ------------------------------------------------
    @property
    def hardware_key(self) -> str:
        """Devices sharing this key share one physical controller and one lock."""
        return f"{self.driver}:{self.id}"

    def apps(self) -> list[str]:
        return sorted(self.config.apps) if self.adb else []

    def text_input_available(self) -> bool:
        return self.adb is not None

    def supported_actions(self) -> set[str]:
        actions = POINT_ACTIONS | PATH_ACTIONS | NAV_ACTIONS | {"scroll", "wait", "finish"}
        if self.text_input_available():
            actions |= {"type", "clear", "enter"}
        if self.apps():
            actions.add("launch")
        return actions

    @abstractmethod
    def status(self) -> dict[str, Any]:
        ...

    def base_status(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "driver": self.driver,
                "machine_position": self.config.machine_position, "apps": self.apps(),
                "text_input": self.text_input_available(),
                "actions": sorted(self.supported_actions())}

    # -- screen ---------------------------------------------------------------
    @abstractmethod
    def capture(self) -> Image.Image:
        """Screenshot for the model (may bring the controller window forward)."""

    def preview(self) -> bytes:
        """Lightweight JPEG for the console; must not disturb the desktop."""
        return jpeg_bytes(self.capture(), quality=70, max_size=(540, 960))

    # -- control --------------------------------------------------------------
    def request_stop(self) -> None:
        self._stop.set()

    def clear_stop(self) -> None:
        self._stop.clear()

    def checkpoint(self) -> None:
        if self._stop.is_set():
            raise StopRequested("用户已停止。")

    def prepare(self) -> None:
        """Called once before a task starts using the device."""
        self.clear_stop()

    def execute(self, action: Action, frame_size: tuple[int, int]) -> dict[str, Any]:
        """Run exactly one action. Returns a small receipt for the evidence log."""
        self.checkpoint()
        t = action.type
        try:
            if t in TEXT_HANDLERS:
                return TEXT_HANDLERS[t](self, action)
            if t == "launch":
                package = self.config.apps[action.app]
                self._require_adb().launch(package)
                return {"channel": "adb_launch", "package": package}
            if t == "wait":
                self.sleep(action.seconds or 2)
                return {"channel": "wait"}
            if t == "tap":
                return self.tap(action.point, frame_size)
            if t == "double_tap":
                return self.double_tap(action.point, frame_size)
            if t == "long_press":
                return self.long_press(action.point, frame_size)
            if t == "swipe":
                return self.swipe(action.start, action.end, frame_size)
            if t == "drag":
                return self.drag(action.start, action.end, frame_size)
            if t in NAV_ACTIONS:
                return self.navigate(t, frame_size)
        except AdbError as exc:
            raise DeviceError(str(exc)) from exc
        raise DeviceError(f"设备无法执行动作 {t}。")

    def sleep(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.checkpoint()
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))

    def _require_adb(self) -> AdbBridge:
        if self.adb is None:
            raise DeviceError("该设备未配置 adb_serial，无法使用 ADB Keyboard 或包名直启。")
        return self.adb

    @abstractmethod
    def tap(self, point: tuple[int, int], frame_size: tuple[int, int]) -> dict[str, Any]: ...

    @abstractmethod
    def double_tap(self, point: tuple[int, int], frame_size: tuple[int, int]) -> dict[str, Any]: ...

    @abstractmethod
    def long_press(self, point: tuple[int, int], frame_size: tuple[int, int]) -> dict[str, Any]: ...

    @abstractmethod
    def swipe(self, start: tuple[int, int], end: tuple[int, int], frame_size: tuple[int, int]) -> dict[str, Any]: ...

    @abstractmethod
    def drag(self, start: tuple[int, int], end: tuple[int, int], frame_size: tuple[int, int]) -> dict[str, Any]: ...

    @abstractmethod
    def navigate(self, key: str, frame_size: tuple[int, int]) -> dict[str, Any]: ...


def _type(device: Device, action: Action) -> dict[str, Any]:
    device._require_adb().input_text(action.text or "")
    return {"channel": "adb_keyboard", "chars": len(action.text or "")}


def _clear(device: Device, action: Action) -> dict[str, Any]:
    device._require_adb().clear_text()
    return {"channel": "adb_keyboard"}


def _enter(device: Device, action: Action) -> dict[str, Any]:
    device._require_adb().input_text("\n")
    return {"channel": "adb_keyboard"}


TEXT_HANDLERS = {"type": _type, "clear": _clear, "enter": _enter}


def grid_to_pixel(point: tuple[int, int], size: tuple[int, int]) -> tuple[int, int]:
    return tuple(min(s - 1, max(0, round(v * (s - 1) / 1000))) for v, s in zip(point, size))  # type: ignore[return-value]
