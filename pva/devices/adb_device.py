"""Optional pure-ADB device (screencap + input), for phones without the robot arm.

Enabled only when a device entry sets ``"driver": "adb"``. Navigation uses the
standard Android key events; everything else matches the robot device.
"""

from __future__ import annotations

from typing import Any

from PIL import Image

from pva.devices.adb import AdbError
from pva.devices.base import Device, DeviceError, grid_to_pixel

KEYCODES = {"home": 3, "back": 4, "recents": 187}


class AdbDevice(Device):
    driver = "adb"

    def status(self) -> dict[str, Any]:
        info = {**self.base_status(), "online": False, "camera": False, "error": None}
        if self.adb is None:
            info["error"] = "未配置 adb_serial。"
            return info
        try:
            info["online"] = info["camera"] = self.adb.run("get-state") == "device"
        except AdbError as exc:
            info["error"] = str(exc)
        info["keyboard"] = self.adb.keyboard_status()
        return info

    def capture(self) -> Image.Image:
        try:
            return self._require_adb().screencap()
        except AdbError as exc:
            raise DeviceError(f"ADB 截图失败：{exc}") from exc

    def _pixel(self, point, frame_size):
        return grid_to_pixel(point, frame_size)

    def tap(self, point, frame_size):
        x, y = self._pixel(point, frame_size)
        self._require_adb().tap(x, y)
        return {"channel": "adb", "pixel": [x, y]}

    def double_tap(self, point, frame_size):
        x, y = self._pixel(point, frame_size)
        self._require_adb().tap(x, y)
        self._require_adb().tap(x, y)
        return {"channel": "adb", "pixel": [x, y]}

    def long_press(self, point, frame_size):
        x, y = self._pixel(point, frame_size)
        self._require_adb().swipe((x, y), (x, y), 900)
        return {"channel": "adb", "pixel": [x, y]}

    def swipe(self, start, end, frame_size):
        a, b = self._pixel(start, frame_size), self._pixel(end, frame_size)
        self._require_adb().swipe(a, b, 350)
        return {"channel": "adb", "start": list(a), "end": list(b)}

    def drag(self, start, end, frame_size):
        a, b = self._pixel(start, frame_size), self._pixel(end, frame_size)
        self._require_adb().swipe(a, b, 1200)
        return {"channel": "adb", "start": list(a), "end": list(b)}

    def navigate(self, key, frame_size):
        self._require_adb().keyevent(KEYCODES[key])
        return {"channel": "adb", "key": key}
