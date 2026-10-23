"""Mock-mode text and app-launch tools bound to the controller's ``SimPhone``.

They keep the production contracts: text goes through the unchanged
``AdbKeyboardTextTransport`` scope/nonce/digest checks and only its fixed
``ADB_INPUT_B64`` / ``ADB_CLEAR_TEXT`` broadcasts are interpreted; app launch
only accepts registered ``launch_ref`` values. No real ADB process is started.
"""

from __future__ import annotations

import base64
import subprocess
from pathlib import Path
from typing import Any

from agent.application.action_adapter import AppLaunchTarget
from agent.domain.text_transport import TEXT_TRANSPORT_PROTOCOL, TextTransportProfile
from agent.domain.validation import reject_if
from agent.infrastructure.adb_keyboard_transport import ADB_KEYBOARD_IME_ID, AdbKeyboardTextTransport
from agent.infrastructure.adb_package_launcher import AdbPackageLauncherError
from agent.infrastructure.simulated_phone import SIMULATED_APP_PACKAGES, SimPhone

SIMULATED_SERIAL = "simulated-phone"


class _SimulatedAdb:
    """Interprets only the fixed ADB Keyboard command family against a SimPhone."""

    def __init__(self, phone: SimPhone) -> None:
        self.phone = phone
        self.commands: list[tuple[str, ...]] = []

    def __call__(self, argv: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        args = tuple(argv[3:])  # drop "<adb> -s <serial>"
        self.commands.append(args)
        stdout = ""
        if args == ("get-state",):
            stdout = "device\n"
        elif args[:4] == ("shell", "ime", "list", "-s"):
            stdout = ADB_KEYBOARD_IME_ID + "\n"
        elif args[:5] == ("shell", "settings", "get", "secure", "default_input_method"):
            stdout = ADB_KEYBOARD_IME_ID + "\n"
        elif args[:5] == ("shell", "am", "broadcast", "-a", "ADB_INPUT_B64") and args[5:7] == ("--es", "msg"):
            self.phone.type_text(base64.b64decode(args[7]).decode("utf-8"))
            stdout = "Broadcasting: Intent { act=ADB_INPUT_B64 }\nBroadcast completed: result=0\n"
        elif args == ("shell", "am", "broadcast", "-a", "ADB_CLEAR_TEXT"):
            self.phone.clear_text()
            stdout = "Broadcasting: Intent { act=ADB_CLEAR_TEXT }\nBroadcast completed: result=0\n"
        else:
            return subprocess.CompletedProcess(argv, 1, "", "simulated adb: command not allowed")
        return subprocess.CompletedProcess(argv, 0, stdout, "")


class SimulatedTextTransport(AdbKeyboardTextTransport):
    """Production ADB Keyboard transport whose adb process is the simulator."""

    def __init__(self, phone: SimPhone, device_id: str) -> None:
        profile = TextTransportProfile(
            protocol_version=TEXT_TRANSPORT_PROTOCOL, profile_id=f"simulated-{device_id}", device_id=device_id,
            adb_serial=SIMULATED_SERIAL, enabled=True, capabilities=("append_text", "clear_text"),
            command_timeout_seconds=5.0)
        self.simulated_adb = _SimulatedAdb(phone)
        super().__init__(profile, Path("simulated/adb.exe"), runner=self.simulated_adb)

    def _preflight_reason(self) -> str | None:
        # The simulator has no executable on disk; run the same IME checks without it.
        if not self._profile.enabled:
            return "profile_disabled"
        state = self._run("get-state")
        selected = self._run("shell", "settings", "get", "secure", "default_input_method")
        if state.stdout.strip() != "device" or selected.stdout.strip() != ADB_KEYBOARD_IME_ID:
            return "ime_not_selected"
        return None


class SimulatedAppLauncher:
    """Registry-bound launcher for the simulated phone's own apps."""

    enabled = True

    def __init__(self, phone: SimPhone) -> None:
        self.phone = phone
        self._by_alias = {name.casefold(): AppLaunchTarget(f"sim.{package.rsplit('.', 1)[-1]}", package)
                          for name, package in SIMULATED_APP_PACKAGES.items()}
        self._names = {target.launch_ref: name for name, target in
                       zip(SIMULATED_APP_PACKAGES, self._by_alias.values())}
        self.launches: list[str] = []

    def aliases(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_alias))

    def catalog(self) -> list[dict[str, str]]:
        return [{"alias": alias, "launch_ref": target.launch_ref, "expected_app_id": target.expected_app_id}
                for alias, target in sorted(self._by_alias.items())]

    def resolve(self, app_id: str, app_name: str) -> AppLaunchTarget | None:
        matches = {self._by_alias[key] for key in (str(v).strip().casefold() for v in (app_id, app_name) if v)
                   if key in self._by_alias}
        reject_if(len(matches) > 1, AdbPackageLauncherError("app_id 与 app_name 映射到不同启动目标。"))
        return next(iter(matches), None)

    def launch(self, launch_ref: str) -> subprocess.CompletedProcess[str]:
        name = self._names.get(str(launch_ref or ""))
        reject_if(name is None, AdbPackageLauncherError("launch_ref 未在当前设备登记。"))
        self.launches.append(launch_ref)
        with self.phone.lock:
            self.phone.open_app(name)
        return subprocess.CompletedProcess(["simulated", "launch", launch_ref], 0, "Events injected: 1\n", "")


__all__ = ["SimulatedAppLauncher", "SimulatedTextTransport"]
