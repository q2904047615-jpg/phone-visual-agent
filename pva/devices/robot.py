"""Robot-arm device: the vendor control window (main.exe) shows the camera image;
a mouse event on that image makes the arm touch the same spot on the phone.

* left click            -> tap (hold ``tap_hold_seconds``)
* left click, count 2   -> double tap (vendor "连点次数" field set to 2, then back to 1)
* right-button hold     -> long press
* right-button path     -> swipe / drag
* Home / Back / Recents -> taps on the phone's navigation bar (ratios in config)

Several machine positions can share one vendor window and camera. They share one
hardware lock, and the right position is selected before each task.
"""

from __future__ import annotations

import threading
from typing import Any

from PIL import Image

from pva.config import DeviceConfig
from pva.devices import seller_window as seller
from pva.devices.base import Device, DeviceError, StopRequested, grid_to_pixel
from pva.devices.calibration import Calibration, CalibrationError
from pva.util import jpeg_bytes

# Which machine position each vendor window currently shows.
_selected_positions: dict[str, int] = {}
_capture_locks: dict[str, threading.RLock] = {}


class RobotDevice(Device):
    driver = "robot"

    def __init__(self, config: DeviceConfig, controller: dict[str, Any], adb_executable: str = "adb"):
        super().__init__(config, adb_executable)
        self.controller = controller
        self.title = str(controller.get("window_title") or seller.DEFAULT_WINDOW_TITLE)
        self.calibration = Calibration(config.calibration)
        self._capture_lock = _capture_locks.setdefault(self.title, threading.RLock())

    @property
    def hardware_key(self) -> str:
        return f"robot:{self.title}"

    # -- status / capture -------------------------------------------------------
    def _window(self) -> int:
        try:
            hwnd, _title = seller.find_window(self.title)
        except RuntimeError as exc:
            raise DeviceError(str(exc)) from exc
        return hwnd

    def status(self) -> dict[str, Any]:
        info = {**self.base_status(), "online": False, "camera": False, "error": None,
                "calibration": self.calibration.status()}
        try:
            hwnd = self._window()
            _l, _t, width, height = seller.client_geometry(hwnd)
            info["online"] = width > 0 and height > 0
            info["camera"] = seller.seller_layout_has_full_camera(width, height) if height > width else (
                width >= 800 and height >= 450)
            if not info["camera"]:
                info["error"] = "控制端没有显示完整的相机画面和底部操作栏。"
        except (DeviceError, OSError, RuntimeError, ValueError) as exc:
            info["error"] = str(exc)
        if self.adb is not None:
            info["keyboard"] = self.adb.keyboard_status()
        return info

    def capture(self) -> Image.Image:
        hwnd = self._window()
        try:
            with self._capture_lock, seller.temporarily_park_cursor_outside_camera(hwnd):
                return seller.camera_crop(seller.capture_client(hwnd), seller.DEFAULT_CAMERA_HEIGHT)
        except (OSError, RuntimeError, ValueError) as exc:
            raise DeviceError(f"相机取图失败：{exc}") from exc

    def preview(self) -> bytes:
        hwnd = self._window()
        try:
            with self._capture_lock:
                image = seller.camera_crop(seller.capture_client_passive(hwnd), seller.DEFAULT_CAMERA_HEIGHT)
        except (OSError, RuntimeError, ValueError) as exc:
            raise DeviceError(str(exc)) from exc
        return jpeg_bytes(image, quality=70, max_size=(540, 960))

    # -- task lifecycle ------------------------------------------------------------
    def prepare(self) -> None:
        super().prepare()
        hwnd = self._window()
        try:
            position = self.config.machine_position
            if position is not None and _selected_positions.get(self.title) != position:
                seller.select_machine_position(hwnd, position)
                _selected_positions[self.title] = position
            seller.configure_single_click_count(hwnd)
        except (OSError, RuntimeError, ValueError) as exc:
            raise DeviceError(f"控制端准备失败：{exc}") from exc

    def checkpoint(self) -> None:
        super().checkpoint()
        try:
            seller._check_escape()
        except RuntimeError as exc:
            raise StopRequested(str(exc)) from exc

    # -- primitives ----------------------------------------------------------------
    def _client_point(self, point: tuple[int, int], frame_size: tuple[int, int]) -> tuple[int, int]:
        try:
            corrected = self.calibration.correct(point[0], point[1], frame_size)
        except CalibrationError as exc:
            raise DeviceError(str(exc)) from exc
        # The camera crop starts at the client origin, so frame pixels are client pixels.
        return grid_to_pixel(corrected, frame_size)

    def _run(self, label: str, operation) -> dict[str, Any]:
        hwnd = self._window()
        self.checkpoint()
        try:
            receipt = operation(hwnd) or {}
        except RuntimeError as exc:
            if "Esc" in str(exc):
                raise StopRequested(str(exc)) from exc
            raise DeviceError(f"{label}失败：{exc}") from exc
        except (OSError, ValueError) as exc:
            raise DeviceError(f"{label}失败：{exc}") from exc
        finally:
            try:
                seller.clear_seller_camera_overlay(hwnd)
            except (OSError, RuntimeError, ValueError):
                pass
        return {"channel": "robot", **receipt}

    def _click(self, point: tuple[int, int], frame_size: tuple[int, int], count: int) -> dict[str, Any]:
        x, y = self._client_point(point, frame_size)
        hold = float(self.controller["tap_hold_seconds"])

        def operation(hwnd: int) -> dict[str, Any]:
            if count == 1:
                receipt = seller.click_client_point(hwnd, x, y, 0, hold, return_dispatch_receipt=True)
            else:
                seller.configure_click_count(hwnd, count)
                try:
                    receipt = seller.click_client_point(hwnd, x, y, 0, hold, return_dispatch_receipt=True,
                                                        click_count=count)
                finally:
                    seller.configure_single_click_count(hwnd)
            return {"client": [x, y], **(receipt or {})}

        return self._run("点击", operation)

    def tap(self, point, frame_size):
        return self._click(point, frame_size, 1)

    def double_tap(self, point, frame_size):
        return self._click(point, frame_size, 2)

    def long_press(self, point, frame_size):
        x, y = self._client_point(point, frame_size)
        hold = float(self.controller["long_press_seconds"])
        return self._run("长按", lambda hwnd: {"client": [x, y],
                                              **seller.long_press_client_point(hwnd, x, y, hold_seconds=hold)})

    def swipe(self, start, end, frame_size):
        a, b = self._client_point(start, frame_size), self._client_point(end, frame_size)
        if a == b:
            raise DeviceError("标定后滑动起点和终点重合。")
        cfg = self.controller["swipe"]
        return self._run("滑动", lambda hwnd: seller.swipe_client_path(
            hwnd, a, b, touch_down_seconds=float(cfg["touch_down_seconds"]),
            movement_seconds=float(cfg["movement_seconds"]), steps=int(cfg["steps"])))

    def drag(self, start, end, frame_size):
        a, b = self._client_point(start, frame_size), self._client_point(end, frame_size)
        if a == b:
            raise DeviceError("标定后拖动起点和终点重合。")

        def operation(hwnd: int) -> dict[str, Any]:
            seller.drag_client_path(hwnd, a, b, duration_seconds=float(self.controller["drag_seconds"]))
            return {"client_start": list(a), "client_end": list(b)}

        return self._run("拖动", operation)

    def navigate(self, key, frame_size):
        x_ratio, y_ratio = self.controller["nav"][key]
        if frame_size[0] > frame_size[1]:
            # Landscape camera image: portrait (x, y) maps to (y, 1 - x).
            x_ratio, y_ratio = y_ratio, 1.0 - x_ratio
        width, height = frame_size
        x = min(width - 1, max(0, round(width * x_ratio)))
        y = min(height - 1, max(0, round(height * y_ratio)))
        hold = float(self.controller["tap_hold_seconds"])
        return self._run("导航键", lambda hwnd: {"client": [x, y], "key": key, **(
            seller.click_client_point(hwnd, x, y, 0, hold, return_dispatch_receipt=True) or {})})
