"""Build config/devices.json from the old poc/ configuration files.

    python -m pva.tools.migrate_config [--adb-registry PATH] [--app-registry PATH]

Reads legacy/poc/device_registry.json and controller_config.json, plus your local
(un-committed) ADB Keyboard and app-package registries if you point at them
(defaults: the ROBOT_ADB_KEYBOARD_REGISTRY / ROBOT_APP_PACKAGE_REGISTRY variables).
Calibration files are copied to config/calibration/<device>.json.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from pva.config import CONFIG_DIR, EXAMPLE_CONFIG_PATH, ROOT
from pva.util import read_json, write_json

LEGACY = ROOT / "legacy" / "poc"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adb-registry", default=os.environ.get("ROBOT_ADB_KEYBOARD_REGISTRY"))
    parser.add_argument("--app-registry", default=os.environ.get("ROBOT_APP_PACKAGE_REGISTRY"))
    parser.add_argument("--output", default=str(CONFIG_DIR / "devices.json"))
    args = parser.parse_args()

    config = read_json(EXAMPLE_CONFIG_PATH)
    registry = read_json(LEGACY / "device_registry.json", default={"devices": []})
    controller = read_json(LEGACY / "controller_config.json", default={})
    adb = read_json(Path(args.adb_registry), default={"devices": []}) if args.adb_registry else {"devices": []}
    apps = read_json(Path(args.app_registry), default={"devices": []}) if args.app_registry else {"devices": []}

    serials = {item["profile"]["device_id"]: item["profile"].get("adb_serial", "") for item in adb.get("devices", [])}
    for item in adb.get("devices", []):
        executable = item.get("adb_executable")
        if executable and executable != "adb.exe":
            config["adb_executable"] = executable
    packages = {}
    for item in apps.get("devices", []):
        packages[item["device_id"]] = {alias: app["package"] for app in item.get("apps", []) for alias in app["aliases"]}
        serials.setdefault(item["device_id"], item.get("adb_serial", ""))

    nav = config["controller"]["nav"]
    for key, prefix in (("home", "android_home"), ("back", "android_back"), ("recents", "android_recents")):
        if f"{prefix}_x_ratio" in controller:
            nav[key] = [controller[f"{prefix}_x_ratio"], controller[f"{prefix}_y_ratio"]]
    if "tap_hold" in controller:
        config["controller"]["tap_hold_seconds"] = controller["tap_hold"]

    devices = []
    for old in registry.get("devices", []):
        device_id = old["device_id"]
        calibration = f"calibration/{device_id}.json"
        source = LEGACY / old.get("calibration_path", "")
        target = CONFIG_DIR / calibration
        if source.is_file() and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        devices.append({"id": device_id, "name": f"{old.get('machine_position', '')}号机位", "driver": "robot",
                        "enabled": bool(old.get("enabled")), "machine_position": old.get("machine_position"),
                        "calibration": calibration, "adb_serial": serials.get(device_id, "") or "",
                        "apps": packages.get(device_id, {})})
    devices += [d for d in config["devices"] if d["driver"] == "mock"]
    config["devices"] = devices or config["devices"]
    write_json(Path(args.output), config)
    print(f"已写入 {args.output}（{len(devices)} 台设备）。请检查 adb_serial 与 apps。")


if __name__ == "__main__":
    main()
