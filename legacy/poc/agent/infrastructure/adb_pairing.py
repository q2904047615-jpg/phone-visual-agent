"""Local Android wireless-debugging pairing used by the web setup panel."""

from __future__ import annotations

import ipaddress
from pathlib import Path
import re
import subprocess
from typing import Any, Callable


_CODE = re.compile(r"^[0-9]{6}$")


class AdbPairingError(RuntimeError):
    """A pairing or connection command failed before the device was ready."""

    def __init__(self, message: str, *, code: str = "pairing_failed") -> None:
        super().__init__(message)
        self.code = code


Runner = Callable[..., subprocess.CompletedProcess[str]]


def _ipv4(value: str) -> str:
    try:
        address = ipaddress.ip_address(value)
    except ValueError as exc:
        raise AdbPairingError("配对地址必须是手机的 IPv4 地址。", code="invalid_pairing_host") from exc
    if address.version != 4:
        raise AdbPairingError("暂只支持局域网 IPv4 配对地址。", code="invalid_pairing_host")
    return str(address)


class AdbPairingService:
    """Run only the fixed adb pair/connect/get-state setup sequence."""

    def __init__(self, adb_executable: Path, adb_serial: str, *, runner: Runner = subprocess.run,
                 timeout_seconds: float = 20.0) -> None:
        executable = Path(adb_executable)
        if executable.name.casefold() not in {"adb", "adb.exe"}:
            raise AdbPairingError("ADB 配置不是 adb 可执行文件。", code="adb_config_invalid")
        if not isinstance(adb_serial, str) or not re.fullmatch(r"[^:]+:[0-9]{1,5}", adb_serial):
            raise AdbPairingError("当前设备的 ADB 连接地址配置无效。", code="adb_config_invalid")
        self._adb = executable
        self._serial = adb_serial
        self._runner = runner
        self._timeout = float(timeout_seconds)

    @property
    def serial(self) -> str:
        return self._serial

    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        try:
            return self._runner(
                [str(self._adb), *args],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self._timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise AdbPairingError("ADB 配对超时，请确认手机仍显示配对码。", code="pairing_timeout") from exc
        except OSError as exc:
            raise AdbPairingError(f"ADB 不可用：{exc}", code="adb_unavailable") from exc

    @staticmethod
    def _failure(result: subprocess.CompletedProcess[str], action: str) -> AdbPairingError:
        # Keep ADB output out of the page response; it may echo connection details.
        return AdbPairingError(
            f"{action}失败：returncode={result.returncode}",
            code=f"{action}_failed",
        )

    def pair_and_connect(self, *, pairing_host: str, pairing_port: int, pairing_code: str) -> dict[str, Any]:
        host = _ipv4(str(pairing_host).strip())
        if not isinstance(pairing_port, int) or isinstance(pairing_port, bool) or not 1 <= pairing_port <= 65535:
            raise AdbPairingError("配对端口无效。", code="invalid_pairing_port")
        if not isinstance(pairing_code, str) or not _CODE.fullmatch(pairing_code):
            raise AdbPairingError("配对码必须是 6 位数字。", code="invalid_pairing_code")
        pairing_endpoint = f"{host}:{pairing_port}"
        paired = self._run("pair", pairing_endpoint, pairing_code)
        if paired.returncode != 0 or "successfully paired" not in (paired.stdout or "").casefold():
            raise self._failure(paired, "pair")
        connected = self._run("connect", self._serial)
        connect_text = f"{connected.stdout or ''} {connected.stderr or ''}".casefold()
        if connected.returncode != 0 or not any(marker in connect_text for marker in ("connected to", "already connected")):
            raise self._failure(connected, "connect")
        state = self._run("-s", self._serial, "get-state")
        if state.returncode != 0 or (state.stdout or "").strip() != "device":
            raise self._failure(state, "verify")
        return {
            "paired": True,
            "connected": True,
            "adb_serial": self._serial,
            "state": "device",
        }


__all__ = ["AdbPairingError", "AdbPairingService"]
