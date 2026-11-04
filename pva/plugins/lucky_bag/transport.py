"""Screen/tap transports for the lucky-bag monitor.

* ``AdbTransport`` ("upstream" mode): screencap and ``input tap`` over the device's adb_serial.
* ``ArmTransport`` ("local" mode): camera capture and an arm tap through the normal device layer.
"""

from __future__ import annotations

from pva.actions import Action
from pva.devices import DeviceManager
from pva.devices.adb import AdbError


class AdbTransport:
    def __init__(self, devices: DeviceManager, device_id: str):
        self.devices = devices
        self.device_id = device_id
        self.adb = devices.get(device_id).adb
        if self.adb is None:
            raise RuntimeError("该设备未配置 adb_serial，无法使用 ADB 模式。")

    def preflight(self):
        try:
            if self.adb.run("get-state") != "device":
                raise RuntimeError("手机 ADB 未连接。")
        except AdbError as exc:
            raise RuntimeError(str(exc)) from exc

    def prepare(self):
        pass

    def capture(self):
        with self.devices.hardware(self.device_id):
            return self.adb.screencap()

    def tap(self, point, frame):
        x, y = point
        if not 0 <= x < frame.width or not 0 <= y < frame.height:
            raise RuntimeError("福袋坐标超出当前截图范围。")
        with self.devices.hardware(self.device_id):
            self.adb.tap(int(x), int(y))


class ArmTransport:
    def __init__(self, devices: DeviceManager, device_id: str):
        self.devices = devices
        self.device_id = device_id
        self.device = devices.get(device_id)

    def preflight(self):
        status = self.device.status()
        if not status.get("online") or not status.get("camera"):
            raise RuntimeError(status.get("error") or "相机或机械臂控制端离线。")

    def prepare(self):
        with self.devices.hardware(self.device_id):
            self.device.prepare()

    def capture(self):
        with self.devices.hardware(self.device_id):
            return self.device.capture().convert("RGB")

    def tap(self, point, frame):
        x, y = point
        if not 0 <= x < frame.width or not 0 <= y < frame.height:
            raise RuntimeError("福袋坐标超出当前相机画面。")
        grid = (round(x * 1000 / (frame.width - 1)), round(y * 1000 / (frame.height - 1)))
        with self.devices.hardware(self.device_id):
            return self.device.execute(Action("tap", point=grid), frame.size)
