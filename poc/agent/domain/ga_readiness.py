"""Read-only readiness contracts for the multi-device/GA transition.

This module deliberately does not participate in task admission or policy.  It
turns the current device registry into an auditable report and keeps the
physical-evidence gate separate from the offline virtual-device contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping


GA_READINESS_PROTOCOL = "2026-10-14-ga-readiness-v1"
READINESS_STATUSES = frozenset({"pass", "blocked", "not_started", "partial"})


@dataclass(frozen=True)
class ReadinessCheck:
    check_id: str
    status: str
    detail: str
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in READINESS_STATUSES:
            raise ValueError(f"未知 readiness status：{self.status}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "status": self.status,
            "detail": self.detail,
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class GAReadinessReport:
    protocol_version: str
    registry_path: str
    enabled_devices: tuple[str, ...]
    missing_calibrations: tuple[str, ...]
    offline_virtual_status: str
    physical_multi_device_status: str
    checks: tuple[ReadinessCheck, ...]
    ga_scope: tuple[ReadinessCheck, ...]

    def __post_init__(self) -> None:
        if self.protocol_version != GA_READINESS_PROTOCOL:
            raise ValueError("GA readiness protocol_version 无效。")

    @property
    def overall_offline_status(self) -> str:
        return "pass" if self.offline_virtual_status == "pass" else "blocked"

    def as_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "registry_path": self.registry_path,
            "enabled_devices": list(self.enabled_devices),
            "missing_calibrations": list(self.missing_calibrations),
            "offline_virtual_status": self.offline_virtual_status,
            "physical_multi_device_status": self.physical_multi_device_status,
            "overall_offline_status": self.overall_offline_status,
            "checks": [item.as_dict() for item in self.checks],
            "ga_scope": [item.as_dict() for item in self.ga_scope],
        }


def _load_registry(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"设备注册表无法读取：{exc}") from exc
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError("设备注册表 version 必须为 1。")
    if not isinstance(payload.get("devices"), list):
        raise ValueError("设备注册表 devices 必须为数组。")
    return payload


def evaluate_device_registry(path: Path, *, real_evidence: bool = False) -> GAReadinessReport:
    """Evaluate only static registry/readiness facts; never changes the file."""

    registry_path = Path(path)
    payload = _load_registry(registry_path)
    devices = payload["devices"]
    all_ids: set[str] = set()
    enabled: list[dict[str, Any]] = []
    duplicate_ids: list[str] = []
    for raw in devices:
        if not isinstance(raw, dict):
            raise ValueError("设备注册表中的设备项必须为对象。")
        device_id = str(raw.get("device_id") or "").strip()
        if not device_id:
            raise ValueError("设备注册表存在空 device_id。")
        if device_id in all_ids:
            duplicate_ids.append(device_id)
        all_ids.add(device_id)
        if raw.get("enabled") is True:
            enabled.append(raw)

    enabled_ids = tuple(str(item["device_id"]).strip() for item in enabled)
    default_id = str(payload.get("default_device_id") or "").strip()
    default_ok = default_id in enabled_ids
    windows: dict[str, set[int]] = {}
    position_conflicts: list[str] = []
    missing_calibrations: list[str] = []
    for item in enabled:
        device_id = str(item["device_id"]).strip()
        raw_position = item.get("machine_position")
        window = str(item.get("window_title") or "").strip() or "__default_window__"
        if isinstance(raw_position, bool) or not isinstance(raw_position, int) or not 1 <= raw_position <= 10:
            position_conflicts.append(device_id)
        else:
            seen = windows.setdefault(window, set())
            if raw_position in seen:
                position_conflicts.append(device_id)
            seen.add(raw_position)
        calibration_value = str(item.get("calibration_path") or "tap_calibration.json").strip()
        calibration = Path(calibration_value)
        if not calibration.is_absolute():
            calibration = registry_path.parent / calibration
        if not calibration.is_file():
            missing_calibrations.append(device_id)

    schema_status = "pass" if not duplicate_ids else "blocked"
    default_status = "pass" if default_ok else "blocked"
    calibration_status = "pass" if not missing_calibrations else "blocked"
    position_status = "pass" if not position_conflicts else "blocked"
    virtual_status = "pass" if len(enabled) >= 2 and calibration_status == "pass" and position_status == "pass" else "blocked"
    if virtual_status != "pass":
        physical_status = "blocked_registry_not_ready"
    elif not real_evidence:
        physical_status = "blocked_no_real_evidence"
    else:
        physical_status = "pass"

    checks = (
        ReadinessCheck("device_registry_schema", schema_status,
                       "设备 ID 唯一。" if schema_status == "pass" else f"重复 device_id：{duplicate_ids}。",
                       (str(registry_path),)),
        ReadinessCheck("default_device", default_status,
                       f"default_device_id={default_id} 已启用。" if default_status == "pass" else
                       f"default_device_id={default_id or 'missing'} 未绑定到已启用设备。", (str(registry_path),)),
        ReadinessCheck("enabled_device_calibration", calibration_status,
                       "所有已启用设备都有独立校准文件。" if calibration_status == "pass" else
                       f"已启用设备缺少校准文件：{missing_calibrations}。", (str(registry_path),)),
        ReadinessCheck("machine_position_isolation", position_status,
                       "已启用设备的控制窗口/机位不冲突。" if position_status == "pass" else
                       f"控制窗口或 machine_position 冲突：{position_conflicts}。", (str(registry_path),)),
        ReadinessCheck("offline_virtual_multi_device", virtual_status,
                       "虚拟双设备配置满足静态隔离前提。" if virtual_status == "pass" else
                       "当前配置不足以证明虚拟双设备隔离；不启用未校准设备。", (str(registry_path),)),
        ReadinessCheck("physical_multi_device_evidence", "pass" if real_evidence else "blocked",
                       "已提供真实第二机位证据。" if real_evidence else
                       "尚无第二台真机、独立校准和动作后证据；此项不由 Mock 结果替代。", (str(registry_path),)),
    )
    ga_scope = (
        ReadinessCheck("multi_device_runtime_contract", "pass",
                       "设备租约、fencing epoch、资源隔离和共享机械窗口锁已有离线合同。",
                       ("poc/test_device_exclusivity.py", "poc/test_device_runtime_resources.py")),
        ReadinessCheck("multi_device_physical_acceptance", "pass" if physical_status == "pass" else "blocked",
                       "等待真实第二机位及独立校准证据。" if physical_status != "pass" else "真实多设备证据已登记。"),
        ReadinessCheck("users_teams_permissions", "partial", "用户/团队/角色权限、主体会话、会话撤销和写入权限已接入本机 Control Plane；外部 provider 和运行时授权尚未接入。",
                       ("poc/test_control_plane_contract.py", "poc/test_control_plane_api.py")),
        ReadinessCheck("quotas_billing", "partial", "配额计算、写入和幂等持久化已验证，计费与运行时扣减尚未接入；本批不新增隐式阻断。",
                       ("poc/test_control_plane_contract.py", "poc/test_control_plane_api.py")),
        ReadinessCheck("api_sdk", "partial", "API/SDK 版本、幂等、错误封套、主体会话、撤销和本机读写路由已验证，具体外部身份 provider 尚未配置。",
                       ("poc/test_control_plane_contract.py", "poc/test_control_plane_api.py", "docs/control_plane_offline_contract.md")),
        ReadinessCheck("version_rollback", "partial", "工具/能力晋级已有回滚合同，平台版本回滚尚未形成 GA 发布流程。"),
    )
    return GAReadinessReport(
        protocol_version=GA_READINESS_PROTOCOL,
        registry_path=str(registry_path),
        enabled_devices=enabled_ids,
        missing_calibrations=tuple(missing_calibrations),
        offline_virtual_status=virtual_status,
        physical_multi_device_status=physical_status,
        checks=checks,
        ga_scope=ga_scope,
    )


__all__ = ["GA_READINESS_PROTOCOL", "GAReadinessReport", "ReadinessCheck", "evaluate_device_registry"]
