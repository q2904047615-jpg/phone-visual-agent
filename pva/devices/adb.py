"""Fixed ADB command family: ADB Keyboard text, package launch, and (adb driver only) touch.

The model never supplies ADB arguments. Text is sent through the ADB Keyboard IME
(``com.android.adbkeyboard/.AdbIME``) as a base64 broadcast; app launches use a
package name looked up in the local device config.
"""

from __future__ import annotations

import base64
import io
import shutil
import subprocess
from pathlib import Path

from PIL import Image

ADB_KEYBOARD_IME = "com.android.adbkeyboard/.AdbIME"


class AdbError(RuntimeError):
    pass


class AdbBridge:
    def __init__(self, executable: str, serial: str, *, timeout: float = 15.0):
        self.executable = executable
        self.serial = serial
        self.timeout = timeout

    def _resolve_executable(self) -> str:
        path = Path(self.executable)
        if path.is_file():
            return str(path)
        found = shutil.which(self.executable)
        if not found:
            raise AdbError(f"找不到 adb 可执行文件：{self.executable}")
        return found

    def run(self, *args: str, binary: bool = False) -> str | bytes:
        if not self.serial:
            raise AdbError("该设备没有配置 adb_serial。")
        try:
            result = subprocess.run([self._resolve_executable(), "-s", self.serial, *args],
                                    capture_output=True, timeout=self.timeout, check=False)
        except subprocess.TimeoutExpired as exc:
            raise AdbError(f"ADB 命令超时：{args[0]}") from exc
        except OSError as exc:
            raise AdbError(f"ADB 不可用：{exc}") from exc
        if result.returncode != 0:
            raise AdbError("ADB 命令失败：" + result.stderr.decode("utf-8", errors="replace").strip())
        return result.stdout if binary else result.stdout.decode("utf-8", errors="replace").strip()

    # -- readiness ---------------------------------------------------------
    def keyboard_status(self) -> dict[str, object]:
        """Return whether text input can work, without changing the phone."""
        if not self.serial:
            return {"ready": False, "reason": "未配置 adb_serial"}
        try:
            if self.run("get-state") != "device":
                return {"ready": False, "reason": "ADB 设备离线"}
            enabled = str(self.run("shell", "ime", "list", "-s")).splitlines()
            if ADB_KEYBOARD_IME not in enabled:
                return {"ready": False, "reason": "手机未安装/启用 ADB Keyboard"}
            selected = self.run("shell", "settings", "get", "secure", "default_input_method")
            if selected != ADB_KEYBOARD_IME:
                return {"ready": False, "reason": "ADB Keyboard 不是当前输入法"}
        except AdbError as exc:
            return {"ready": False, "reason": str(exc)}
        return {"ready": True, "reason": None}

    # -- ADB Keyboard -------------------------------------------------------
    def _broadcast(self, *args: str) -> None:
        output = str(self.run("shell", "am", "broadcast", *args))
        if "Broadcast completed" not in output:
            raise AdbError("ADB Keyboard 广播未被接收：" + output[-200:])

    def input_text(self, text: str) -> None:
        encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
        self._broadcast("-a", "ADB_INPUT_B64", "--es", "msg", encoded)

    def clear_text(self) -> None:
        self._broadcast("-a", "ADB_CLEAR_TEXT")

    def launch(self, package: str) -> None:
        self.run("shell", "monkey", "-p", package, "-c", "android.intent.category.LAUNCHER", "1")

    # -- adb driver only ----------------------------------------------------
    def screencap(self) -> Image.Image:
        data = self.run("exec-out", "screencap", "-p", binary=True)
        return Image.open(io.BytesIO(data)).convert("RGB")

    def tap(self, x: int, y: int) -> None:
        self.run("shell", "input", "tap", str(x), str(y))

    def swipe(self, start: tuple[int, int], end: tuple[int, int], duration_ms: int) -> None:
        self.run("shell", "input", "swipe", *map(str, (*start, *end, duration_ms)))

    def keyevent(self, code: int) -> None:
        self.run("shell", "input", "keyevent", str(code))
