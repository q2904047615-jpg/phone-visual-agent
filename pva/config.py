"""Runtime configuration: one device file plus a few environment variables.

Environment variables
---------------------
DASHSCOPE_API_KEY      Qwen (Aliyun Model Studio) key. Required for real tasks.
VISION_MODEL           Model id, default ``qwen3-vl-plus``.
VISION_MODEL_BASE_URL  OpenAI-compatible endpoint, default DashScope.
PVA_MOCK=1             Replace every device with the built-in simulated phone.
PVA_CONFIG             Path to the device file (default ``config/devices.json``).
PVA_OUTPUT             Evidence directory (default ``output/``).
PVA_PORT               Web console port (default 8765).
"""

from __future__ import annotations

import copy
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pva.util import read_json

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "devices.json"
EXAMPLE_CONFIG_PATH = CONFIG_DIR / "devices.example.json"

DEFAULT_MODEL = "qwen3-vl-plus"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

DEFAULT_CONTROLLER: dict[str, Any] = {
    "window_title": "智联新途",
    "tap_hold_seconds": 0.35,
    "long_press_seconds": 0.8,
    "nav": {"home": [0.5, 0.976], "back": [0.685, 0.976], "recents": [0.33, 0.976]},
    "swipe": {"touch_down_seconds": 0.35, "movement_seconds": 0.3, "steps": 6},
    "drag_seconds": 0.8,
}

DEFAULT_AGENT: dict[str, Any] = {
    "max_actions": 100,
    "max_observations": 200,
    "settle_seconds": 1.2,
    "history_thumbnail": [360, 640],
    "current_image_max": [1080, 1920],
    "model_timeout_seconds": 180,
    "max_consecutive_model_errors": 3,
    "enable_thinking": False,
}


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class DeviceConfig:
    id: str
    name: str
    driver: str  # robot | adb | mock
    enabled: bool = True
    machine_position: int | None = None
    calibration: Path | None = None
    adb_serial: str = ""
    apps: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class AppConfig:
    path: Path
    controller: dict[str, Any]
    agent: dict[str, Any]
    adb_executable: str
    default_device: str
    devices: tuple[DeviceConfig, ...]
    mock: bool

    def device(self, device_id: str) -> DeviceConfig:
        for device in self.devices:
            if device.id == device_id:
                return device
        raise ConfigError(f"未知设备：{device_id}")


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def _parse_device(raw: dict[str, Any], base: Path, mock: bool) -> DeviceConfig:
    device_id = str(raw.get("id") or "").strip()
    if not device_id:
        raise ConfigError("设备缺少 id。")
    driver = "mock" if mock else str(raw.get("driver") or "robot")
    if driver not in {"robot", "adb", "mock"}:
        raise ConfigError(f"设备 {device_id} 的 driver 只能是 robot / adb / mock。")
    position = raw.get("machine_position")
    if position is not None and (not isinstance(position, int) or not 1 <= position <= 10):
        raise ConfigError(f"设备 {device_id} 的 machine_position 必须是 1-10。")
    calibration = raw.get("calibration")
    apps = raw.get("apps") or {}
    if not isinstance(apps, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in apps.items()):
        raise ConfigError(f"设备 {device_id} 的 apps 必须是 {{名称: 包名}}。")
    return DeviceConfig(
        id=device_id,
        name=str(raw.get("name") or device_id),
        driver=driver,
        enabled=bool(raw.get("enabled", True)),
        machine_position=position,
        calibration=(base / calibration) if calibration else None,
        adb_serial=str(raw.get("adb_serial") or "").strip(),
        apps=dict(apps),
    )


def load_config(path: Path | None = None) -> AppConfig:
    mock = os.environ.get("PVA_MOCK", "").strip() in {"1", "true", "yes"}
    resolved = Path(path or os.environ.get("PVA_CONFIG") or DEFAULT_CONFIG_PATH)
    if not resolved.exists() and resolved == DEFAULT_CONFIG_PATH and EXAMPLE_CONFIG_PATH.exists():
        shutil.copyfile(EXAMPLE_CONFIG_PATH, resolved)
    raw = read_json(resolved, default=None)
    if raw is None:
        raise ConfigError(f"找不到设备配置：{resolved}")
    if not isinstance(raw, dict):
        raise ConfigError("设备配置必须是 JSON 对象。")
    devices = tuple(_parse_device(item, resolved.parent, mock) for item in raw.get("devices") or [])
    if not devices:
        raise ConfigError("设备配置中至少需要一台设备。")
    ids = [device.id for device in devices]
    if len(ids) != len(set(ids)):
        raise ConfigError("设备 id 重复。")
    default_device = str(raw.get("default_device") or ids[0])
    if default_device not in ids:
        raise ConfigError(f"default_device {default_device} 不在设备列表中。")
    return AppConfig(
        path=resolved,
        controller=_merge(DEFAULT_CONTROLLER, raw.get("controller") or {}),
        agent=_merge(DEFAULT_AGENT, raw.get("agent") or {}),
        adb_executable=str(raw.get("adb_executable") or "adb"),
        default_device=default_device,
        devices=devices,
        mock=mock,
    )


def output_dir() -> Path:
    return Path(os.environ.get("PVA_OUTPUT") or ROOT / "output")


def model_settings() -> dict[str, str]:
    return {
        "api_key": os.environ.get("DASHSCOPE_API_KEY", "").strip(),
        "model": os.environ.get("VISION_MODEL", "").strip() or DEFAULT_MODEL,
        "base_url": (os.environ.get("VISION_MODEL_BASE_URL", "").strip() or DEFAULT_BASE_URL).rstrip("/"),
    }
