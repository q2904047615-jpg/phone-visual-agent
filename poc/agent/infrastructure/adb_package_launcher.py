"""Registry-bound transport for the one approved ADB package launch."""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Callable, Mapping

from agent.application.action_adapter import AppLaunchTarget
from agent.domain.validation import reject_if


_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_PACKAGE = re.compile(r"[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)+\Z")


class AdbPackageLauncherError(RuntimeError):
    def __init__(self, message: str, *, attempted: bool=False) -> None:
        super().__init__(message)
        self.attempted = bool(attempted)


def _keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    reject_if(set(value) != expected, AdbPackageLauncherError(f"{label} 字段不符合合同。"))


def _match(value: Any, pattern: re.Pattern[str], label: str) -> str:
    reject_if(not isinstance(value, str) or not pattern.fullmatch(value),
        AdbPackageLauncherError(f"{label} 格式无效。"))
    return value


def _alias(value: Any) -> str:
    reject_if(not isinstance(value, str) or not value or value != value.strip() or len(value) > 128
        or not value.isprintable(), AdbPackageLauncherError("App alias 格式无效。"))
    return value.casefold()


class AdbPackageLauncher:
    """Resolve local aliases and execute only a fixed monkey package launch."""

    def __init__(self, registry_path: str | Path, device_id: str, *, runner: Callable[..., Any]=subprocess.run,
        timeout_seconds: float=10.0) -> None:
        resolved_device = _match(device_id, _ID, 'device_id')
        reject_if(not callable(runner) or isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float)) or not 0 < float(timeout_seconds) <= 120,
            AdbPackageLauncherError("runner 或 timeout_seconds 无效。"))
        self._runner, self._timeout = runner, float(timeout_seconds)
        try:
            payload = json.loads(Path(registry_path).read_text(encoding='utf-8'))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise AdbPackageLauncherError(f"无法读取 App 启动注册表：{exc}") from exc
        reject_if(not isinstance(payload, Mapping), AdbPackageLauncherError("App 启动注册表顶层必须是对象。"))
        _keys(payload, {'version', 'devices'}, '注册表顶层')
        devices = payload['devices']
        reject_if(payload['version'] != 1 or isinstance(payload['version'], bool) or not isinstance(devices, list),
            AdbPackageLauncherError("App 启动注册表 version/devices 无效。"))
        matches = [raw for raw in devices if isinstance(raw, Mapping) and raw.get('device_id') == resolved_device]
        reject_if(len(matches) > 1,
            AdbPackageLauncherError(f"App 包名启动注册表设备登记不唯一：{resolved_device}。"))
        self.enabled = False
        self._argv_by_ref: dict[str, tuple[str, ...]] = {}
        self._by_alias: dict[str, AppLaunchTarget] = {}
        if not matches:
            return
        profile = matches[0]
        _keys(profile, {'device_id', 'enabled', 'adb_executable', 'adb_serial', 'apps'}, 'device profile')
        raw_enabled, executable, serial, apps = (profile[key] for key in ('enabled', 'adb_executable', 'adb_serial',
            'apps'))
        reject_if(not isinstance(raw_enabled, bool) or not isinstance(executable, str) or not executable.strip()
            or executable != executable.strip() or '\x00' in executable or not isinstance(serial, str)
            or (serial and not _ID.fullmatch(serial)) or not isinstance(apps, list),
            AdbPackageLauncherError("device profile 的启用状态、ADB、serial 或 apps 无效。"))
        by_ref: dict[str, AppLaunchTarget] = {}
        by_alias: dict[str, AppLaunchTarget] = {}
        packages: set[str] = set()
        for raw in apps:
            reject_if(not isinstance(raw, Mapping), AdbPackageLauncherError("App 映射必须是对象。"))
            _keys(raw, {'launch_ref', 'aliases', 'package'}, 'App 映射')
            ref, package, raw_aliases = (_match(raw['launch_ref'], _ID, 'launch_ref'), _match(raw['package'],
                _PACKAGE, 'package'), raw['aliases'])
            normalized = tuple(_alias(item) for item in raw_aliases) if isinstance(raw_aliases, list) else ()
            reject_if(not normalized or len(normalized) != len(set(normalized)) or ref in by_ref
                or package.casefold() in packages or any(alias in by_alias for alias in normalized),
                AdbPackageLauncherError("App 映射的 ref、alias 或 package 缺失/重复。"))
            target = AppLaunchTarget(ref, package)
            by_ref[ref] = target
            packages.add(package.casefold())
            by_alias.update((alias, target) for alias in normalized)
        executable_path = Path(executable)
        reject_if(executable_path.name.casefold() not in {'adb', 'adb.exe'},
            AdbPackageLauncherError("adb_executable 必须指向 adb 或 adb.exe。"))
        resolved = str(executable_path) if executable_path.is_file() else shutil.which(executable)
        executable_command = str(resolved or executable)
        self.enabled = bool(raw_enabled and resolved and serial and by_ref)
        self._argv_by_ref = {ref: (executable_command, '-s', serial, 'shell', 'monkey', '-p',
            target.expected_app_id, '-c', 'android.intent.category.LAUNCHER', '1') for ref, target in by_ref.items()}
        self._by_alias = by_alias

    def resolve(self, app_id: str, app_name: str) -> AppLaunchTarget | None:
        if not self.enabled:
            return None
        matches = {self._by_alias[key] for key in (_alias(value) for value in (app_id, app_name) if value)
            if key in self._by_alias}
        reject_if(len(matches) > 1, AdbPackageLauncherError("app_id 与 app_name 映射到不同启动目标。"))
        return next(iter(matches), None)

    def launch(self, launch_ref: str) -> Any:
        reject_if(not self.enabled, AdbPackageLauncherError("当前设备的 ADB 包名启动能力未启用。"))
        argv = self._argv_by_ref.get(_match(launch_ref, _ID, 'launch_ref'))
        reject_if(argv is None, AdbPackageLauncherError("launch_ref 未在当前设备登记。"))
        try:
            result = self._runner(list(argv), check=False, capture_output=True, text=True, timeout=self._timeout)
        except subprocess.TimeoutExpired as exc:
            raise AdbPackageLauncherError("ADB 包名启动超时。", attempted=True) from exc
        except OSError as exc:
            raise AdbPackageLauncherError(f"ADB 启动器不可用：{exc}", attempted=True) from exc
        except Exception as exc:
            raise AdbPackageLauncherError("ADB 包名启动 transport 异常。", attempted=True) from exc
        reject_if(not isinstance(getattr(result, 'returncode', None), int) or result.returncode != 0,
            AdbPackageLauncherError(f"ADB 包名启动失败：returncode={getattr(result, 'returncode', 'missing')}",
                attempted=True))
        return result
