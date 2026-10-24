"""Mock-mode ADB Keyboard bound to the controller's ``SimPhone``.

It is the production ``AdbKeyboardTextTransport`` (unchanged scope, nonce and
digest checks) whose adb process is replaced by an interpreter of only the two
fixed broadcasts, ``ADB_INPUT_B64`` and ``ADB_CLEAR_TEXT``. No adb runs.
"""

from __future__ import annotations

import base64
import subprocess
from pathlib import Path

from agent.domain.text_transport import TEXT_TRANSPORT_PROTOCOL, TextTransportProfile
from agent.infrastructure.adb_keyboard_transport import AdbKeyboardTextTransport
from agent.infrastructure.simulated_phone import SimPhone


class SimulatedTextTransport(AdbKeyboardTextTransport):
    def __init__(self, phone: SimPhone, device_id: str) -> None:
        self.phone = phone
        self.broadcasts: list[str] = []
        profile = TextTransportProfile(protocol_version=TEXT_TRANSPORT_PROTOCOL,
            profile_id=f"simulated-{device_id}", device_id=device_id, adb_serial="simulated-phone",
            enabled=True, capabilities=("append_text", "clear_text"), command_timeout_seconds=5.0)
        super().__init__(profile, Path("simulated/adb.exe"), runner=self._broadcast_runner)

    def _preflight_reason(self) -> str | None:
        # No executable or IME to probe; the simulated keyboard is always selected.
        return None if self._profile.enabled else "profile_disabled"

    def _broadcast_runner(self, argv: list[str], **_kwargs) -> subprocess.CompletedProcess[str]:
        args = argv[3:]  # drop "<adb> -s <serial>"
        if args[:5] == ["shell", "am", "broadcast", "-a", "ADB_INPUT_B64"] and args[5:7] == ["--es", "msg"]:
            self.phone.type_text(base64.b64decode(args[7]).decode("utf-8"))
        elif args == ["shell", "am", "broadcast", "-a", "ADB_CLEAR_TEXT"]:
            self.phone.clear_text()
        else:
            return subprocess.CompletedProcess(argv, 1, "", "simulated adb: command not allowed")
        self.broadcasts.append(args[4])
        return subprocess.CompletedProcess(argv, 0, "Broadcast completed: result=0\n", "")


__all__ = ["SimulatedTextTransport"]
