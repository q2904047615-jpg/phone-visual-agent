"""Explicit trial transports: native ADB in A, existing camera/arm in B."""
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
import subprocess
from PIL import Image
from agent.infrastructure.device_exclusivity import InterProcessLease, SHARED_DEVICE_LEASE_DIR
from agent.infrastructure.orientation_safety import _mint_single_step_scene_credential
from agent.infrastructure.orientation_safety import frame_fingerprint


class AdbTransport:
    def __init__(self, runtime, device_id, hardware_lock=None):
        self.device_id = device_id
        transport = runtime.text_transport_for_device(device_id)
        if transport is None:
            raise RuntimeError("A版该设备未配置可信 ADB 地址，请先在原项目的配对入口完成连接。")
        self.executable = str(transport.adb_executable)
        self.serial = transport.profile.adb_serial

    def run(self, *args, binary=False):
        result = subprocess.run([self.executable,"-s",self.serial,*args], capture_output=True, timeout=20)
        if result.returncode:
            raise RuntimeError("ADB 操作失败："+result.stderr.decode("utf-8",errors="replace").strip())
        return result.stdout if binary else result.stdout.decode("utf-8",errors="replace").strip()

    def preflight(self):
        if self.run("get-state") != "device":
            raise RuntimeError("A版手机 ADB 未连接。")

    @contextmanager
    def lock(self):
        lease = InterProcessLease(SHARED_DEVICE_LEASE_DIR / "physical_hardware_action.lease",
            owner_id="lucky-adb-"+self.device_id, metadata={"device_id":self.device_id})
        with lease:
            yield

    def capture(self):
        with self.lock():
            return Image.open(BytesIO(self.run("exec-out","screencap","-p",binary=True))).convert("RGB")

    def tap(self, point, frame):
        x,y = point
        if not 0 <= x < frame.width or not 0 <= y < frame.height:
            raise RuntimeError("福袋坐标超出当前截图范围。")
        with self.lock():
            self.run("shell","input","tap",str(int(x)),str(int(y)))


class ArmTransport:
    def __init__(self, runtime, device_id, hardware_lock):
        self.device_id, self.hardware_lock = device_id, hardware_lock
        self.controller = runtime.controller_for_device(device_id)

    def preflight(self):
        status = self.controller.device_status()
        if not status.get("controller_online") or not status.get("camera_online"):
            raise RuntimeError("B版相机或机械臂控制端离线。")
        if status.get("busy"):
            raise RuntimeError("机械臂正在执行其他任务。")

    def capture(self):
        with self.hardware_lock(self.device_id):
            return self.controller.vision_capture().convert("RGB")

    def tap(self, point, frame):
        x,y = point
        if not 0 <= x < frame.width or not 0 <= y < frame.height:
            raise RuntimeError("福袋坐标超出当前相机画面。")
        with self.hardware_lock(self.device_id):
            credential = _mint_single_step_scene_credential(device_id=self.device_id,
                scene_fingerprint=frame_fingerprint(frame), frame=frame)
            self.controller.arm_physical_execution(credential, action="tap_semantic", scene_fingerprint=frame_fingerprint(frame))
            try:
                return self.controller.vision_tap_relative(round(x*1000/(frame.width-1)), round(y*1000/(frame.height-1)))
            finally:
                self.controller.clear_physical_execution_authorization()
