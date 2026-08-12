from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping
import uuid

from PIL import Image, UnidentifiedImageError

from device_exclusivity import InterProcessLease


PROMOTABLE_ACTIONS = frozenset(
    {
        "tap_semantic",
        "dismiss_overlay",
        "swipe",
        "back",
        "input_verified_text",
        "long_press",
        "drag",
    }
)


class CapabilityAcceptanceError(RuntimeError):
    pass


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    try:
        return _sha256_bytes(Path(path).read_bytes())
    except OSError as exc:
        raise CapabilityAcceptanceError(f"文件无法读取：{path}：{exc}") from exc


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise CapabilityAcceptanceError(f"{label}无法读取：{exc}") from exc
    if not isinstance(value, dict):
        raise CapabilityAcceptanceError(f"{label}必须是 JSON 对象。")
    return value


def _required_text(value: Any, *, field: str, max_length: int = 256) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CapabilityAcceptanceError(f"验收报告字段 {field} 不能为空。")
    clean = value.strip()
    if len(clean) > max_length:
        raise CapabilityAcceptanceError(f"验收报告字段 {field} 过长。")
    return clean


def _observation(value: Any, *, field: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise CapabilityAcceptanceError(f"验收报告字段 {field} 必须是对象。")
    return {
        "observation_id": _required_text(
            value.get("observation_id"), field=f"{field}.observation_id"
        ),
        "fingerprint": _required_text(
            value.get("fingerprint"), field=f"{field}.fingerprint"
        ),
    }


def _validate_frame_paths(
    value: Any,
    *,
    field: str,
    trial_root: Path,
    expected_sha256: Any,
) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) != 4:
        raise CapabilityAcceptanceError(f"{field} 必须恰好包含四张 JPEG 证据。")
    if (
        not isinstance(expected_sha256, list)
        or len(expected_sha256) != 4
        or any(
            not isinstance(item, str)
            or len(item) != 64
            or any(character not in "0123456789abcdef" for character in item)
            for item in expected_sha256
        )
    ):
        raise CapabilityAcceptanceError(f"{field} 的证据摘要格式无效。")
    resolved_root = trial_root.resolve(strict=True)
    result: list[str] = []
    seen: set[Path] = set()
    for index, item in enumerate(value, start=1):
        if not isinstance(item, str) or not item.strip():
            raise CapabilityAcceptanceError(f"{field}[{index}] 证据路径无效。")
        candidate = Path(item.strip())
        if not candidate.is_absolute():
            candidate = trial_root / candidate
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise CapabilityAcceptanceError(
                f"{field}[{index}] 证据文件不存在。"
            ) from exc
        try:
            resolved.relative_to(resolved_root)
        except ValueError as exc:
            raise CapabilityAcceptanceError(
                f"{field}[{index}] 证据必须位于本次 trial 目录。"
            ) from exc
        if resolved in seen:
            raise CapabilityAcceptanceError(f"{field} 不能重复引用同一张证据。")
        if resolved.suffix.lower() not in {".jpg", ".jpeg"}:
            raise CapabilityAcceptanceError(f"{field}[{index}] 证据必须是 JPEG。")
        try:
            with Image.open(resolved) as image:
                image.verify()
                if image.format != "JPEG":
                    raise CapabilityAcceptanceError(
                        f"{field}[{index}] 证据内容不是 JPEG。"
                    )
        except (OSError, UnidentifiedImageError) as exc:
            raise CapabilityAcceptanceError(
                f"{field}[{index}] 证据无法读取。"
            ) from exc
        if sha256_file(resolved) != expected_sha256[index - 1]:
            raise CapabilityAcceptanceError(
                f"{field}[{index}] 证据摘要与文件不一致。"
            )
        seen.add(resolved)
        result.append(str(resolved))
    return tuple(result)


def validate_acceptance_report(report_path: Path) -> dict[str, Any]:
    """Validate evidence strong enough to make one capability promotable."""

    resolved_report = Path(report_path).resolve(strict=True)
    report = _load_json_object(resolved_report, label="验收报告")
    if report.get("version") != 1:
        raise CapabilityAcceptanceError("验收报告版本无效。")

    trial_id = _required_text(report.get("trial_id"), field="trial_id", max_length=128)
    session_id = _required_text(
        report.get("session_id"), field="session_id", max_length=128
    )
    task_id = _required_text(report.get("task_id"), field="task_id", max_length=128)
    device_id = _required_text(
        report.get("device_id"), field="device_id", max_length=128
    )
    action = _required_text(
        report.get("candidate_action"), field="candidate_action", max_length=64
    )
    if action not in PROMOTABLE_ACTIONS:
        raise CapabilityAcceptanceError(f"动作类型不能进入真机验收：{action}。")
    _required_text(report.get("code_revision"), field="code_revision", max_length=128)
    if report.get("status") != "passed":
        raise CapabilityAcceptanceError("验收报告结果不是 passed，不能晋级。")
    physical_actions = report.get("physical_actions")
    if isinstance(physical_actions, bool) or physical_actions != 1:
        raise CapabilityAcceptanceError("验收报告物理动作数必须严格等于 1。")
    if report.get("action_outcome") != "matched":
        raise CapabilityAcceptanceError("验收结果不是 matched，不能晋级。")

    before = _observation(report.get("before_observation"), field="before_observation")
    after = _observation(report.get("after_observation"), field="after_observation")
    if after["observation_id"] == before["observation_id"]:
        raise CapabilityAcceptanceError("动作后 observation_id 未变化。")
    if after["fingerprint"] == before["fingerprint"]:
        raise CapabilityAcceptanceError("动作后 fingerprint 未变化。")

    execution = report.get("execution")
    if not isinstance(execution, dict):
        raise CapabilityAcceptanceError("验收报告 execution 必须是对象。")
    resolved_action = execution.get("resolved_action")
    if not isinstance(resolved_action, dict) or resolved_action.get("kind") != action:
        raise CapabilityAcceptanceError("执行动作类型与候选动作类型不一致。")
    observation_errors = execution.get("observation_errors")
    if not isinstance(observation_errors, list) or observation_errors:
        raise CapabilityAcceptanceError("验收报告包含观察错误，不能晋级。")
    verification_errors = execution.get("verification_errors")
    if not isinstance(verification_errors, list) or verification_errors:
        raise CapabilityAcceptanceError("验收报告包含验证错误，不能晋级。")

    trial_root = resolved_report.parent
    before_paths = _validate_frame_paths(
        report.get("before_frame_paths"),
        field="before_frame_paths",
        trial_root=trial_root,
        expected_sha256=report.get("before_frame_sha256"),
    )
    after_paths = _validate_frame_paths(
        report.get("after_frame_paths"),
        field="after_frame_paths",
        trial_root=trial_root,
        expected_sha256=report.get("after_frame_sha256"),
    )
    if set(before_paths) & set(after_paths):
        raise CapabilityAcceptanceError("动作前后证据不能引用同一文件。")

    normalized = dict(report)
    normalized.update(
        {
            "trial_id": trial_id,
            "session_id": session_id,
            "task_id": task_id,
            "device_id": device_id,
            "candidate_action": action,
            "before_observation": before,
            "after_observation": after,
            "before_frame_paths": list(before_paths),
            "after_frame_paths": list(after_paths),
        }
    )
    return normalized


@dataclass(frozen=True)
class PromotionScope:
    trial_id: str
    device_id: str
    action: str
    report_sha256: str
    registry_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "device_id": self.device_id,
            "action": self.action,
            "report_sha256": self.report_sha256,
            "registry_sha256": self.registry_sha256,
        }


class PromotionAuthority:
    """One-shot local authority bound to one report and registry revision."""

    def __init__(self, scope: PromotionScope) -> None:
        self.scope = scope
        self.consumed = False

    def validate_and_consume(self, value: Mapping[str, Any]) -> None:
        if self.consumed:
            raise CapabilityAcceptanceError("能力晋级确认已使用，禁止重放。")
        self.consumed = True
        if not isinstance(value, Mapping):
            raise CapabilityAcceptanceError("能力晋级确认范围必须是对象。")
        expected = self.scope.to_dict()
        requested = dict(value)
        if set(requested) != set(expected) or any(
            not isinstance(requested.get(key), str) or requested.get(key) != expected[key]
            for key in expected
        ):
            raise CapabilityAcceptanceError("能力晋级确认范围不匹配。")


class CapabilityRegistryPromoter:
    """Promote one report-bound action through an atomic registry replacement."""

    def __init__(
        self,
        registry_path: Path,
        *,
        lease_path: Path | None = None,
        replace_file: Callable[[Path, Path], None] | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.registry_path = Path(registry_path)
        self.lease_path = Path(
            lease_path
            if lease_path is not None
            else self.registry_path.with_suffix(".promotion.lease")
        )
        self._replace_file = replace_file or (
            lambda source, target: os.replace(source, target)
        )
        self._now = now or (lambda: datetime.now(timezone.utc))

    @staticmethod
    def _registry_device(
        payload: dict[str, Any], device_id: str
    ) -> dict[str, Any]:
        if payload.get("version") != 1 or not isinstance(payload.get("devices"), list):
            raise CapabilityAcceptanceError("设备注册表版本或 devices 格式无效。")
        matches = [
            item
            for item in payload["devices"]
            if isinstance(item, dict)
            and item.get("enabled") is True
            and item.get("device_id") == device_id
        ]
        if len(matches) != 1:
            raise CapabilityAcceptanceError(
                f"设备注册表没有唯一的已启用设备：{device_id}。"
            )
        device = matches[0]
        actions = device.get("verified_actions")
        if not isinstance(actions, list) or any(
            not isinstance(item, str) or not item.strip() for item in actions
        ):
            raise CapabilityAcceptanceError("设备 verified_actions 格式无效。")
        normalized = [item.strip() for item in actions]
        if len(normalized) != len(set(normalized)):
            raise CapabilityAcceptanceError("设备 verified_actions 存在重复动作。")
        device["verified_actions"] = normalized
        return device

    def _load_registry(self) -> tuple[dict[str, Any], bytes]:
        try:
            raw = self.registry_path.read_bytes()
            payload = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise CapabilityAcceptanceError(f"设备注册表无法读取：{exc}") from exc
        if not isinstance(payload, dict):
            raise CapabilityAcceptanceError("设备注册表必须是 JSON 对象。")
        return payload, raw

    def preview(self, report_path: Path) -> PromotionScope:
        report = validate_acceptance_report(report_path)
        registry, raw = self._load_registry()
        device = self._registry_device(registry, report["device_id"])
        action = report["candidate_action"]
        if action in device["verified_actions"]:
            raise CapabilityAcceptanceError(f"设备能力 {action} 已经启用。")
        return PromotionScope(
            trial_id=report["trial_id"],
            device_id=report["device_id"],
            action=action,
            report_sha256=sha256_file(Path(report_path)),
            registry_sha256=_sha256_bytes(raw),
        )

    @staticmethod
    def _write_new_file(path: Path, payload: bytes) -> None:
        try:
            with path.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError as exc:
            raise CapabilityAcceptanceError(f"晋级证据文件已经存在：{path.name}。") from exc
        except OSError as exc:
            raise CapabilityAcceptanceError(f"晋级证据无法写入：{path.name}：{exc}") from exc

    def _replace_registry(self, payload: bytes) -> None:
        temporary = self.registry_path.parent / (
            f".{self.registry_path.name}.{uuid.uuid4().hex}.tmp"
        )
        try:
            with temporary.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            self._replace_file(temporary, self.registry_path)
        except OSError as exc:
            raise CapabilityAcceptanceError(f"设备注册表原子替换失败：{exc}") from exc
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    def promote(
        self,
        report_path: Path,
        *,
        confirmation: Mapping[str, Any],
        authority: PromotionAuthority,
    ) -> dict[str, Any]:
        authority.validate_and_consume(confirmation)
        scope = authority.scope
        lease = InterProcessLease(
            self.lease_path,
            owner_id=scope.trial_id,
            metadata={
                "kind": "capability_registry_promotion",
                "trial_id": scope.trial_id,
                "device_id": scope.device_id,
                "action": scope.action,
            },
        )
        if not lease.acquire():
            raise CapabilityAcceptanceError("设备注册表晋级锁已被其他进程占用。")
        try:
            if sha256_file(Path(report_path)) != scope.report_sha256:
                raise CapabilityAcceptanceError("验收报告在确认后发生变化。")
            report = validate_acceptance_report(report_path)
            if (
                report["trial_id"] != scope.trial_id
                or report["device_id"] != scope.device_id
                or report["candidate_action"] != scope.action
            ):
                raise CapabilityAcceptanceError("验收报告范围在确认后发生变化。")

            registry, registry_raw = self._load_registry()
            if _sha256_bytes(registry_raw) != scope.registry_sha256:
                raise CapabilityAcceptanceError("设备注册表在确认后发生变化。")
            device = self._registry_device(registry, scope.device_id)
            if scope.action in device["verified_actions"]:
                raise CapabilityAcceptanceError(f"设备能力 {scope.action} 已经启用。")

            trial_dir = Path(report_path).resolve(strict=True).parent
            backup_path = trial_dir / "registry_before.json"
            promotion_path = trial_dir / "promotion.json"
            if backup_path.exists() or promotion_path.exists():
                raise CapabilityAcceptanceError("本次 trial 已存在晋级证据，禁止重复晋级。")

            device["verified_actions"] = sorted(
                {*device["verified_actions"], scope.action}
            )
            encoded_registry = (
                json.dumps(registry, ensure_ascii=False, indent=2) + "\n"
            ).encode("utf-8")
            after_sha256 = _sha256_bytes(encoded_registry)
            promoted_at = self._now().astimezone(timezone.utc).isoformat()
            result = {
                "version": 1,
                "trial_id": scope.trial_id,
                "device_id": scope.device_id,
                "action": scope.action,
                "report_sha256": scope.report_sha256,
                "registry_before_sha256": scope.registry_sha256,
                "registry_after_sha256": after_sha256,
                "promoted_at": promoted_at,
                "requires_restart": True,
            }
            encoded_promotion = (
                json.dumps(result, ensure_ascii=False, indent=2) + "\n"
            ).encode("utf-8")

            self._write_new_file(backup_path, registry_raw)
            try:
                self._replace_registry(encoded_registry)
            except CapabilityAcceptanceError:
                try:
                    backup_path.unlink(missing_ok=True)
                except OSError:
                    pass
                raise
            try:
                self._write_new_file(promotion_path, encoded_promotion)
            except CapabilityAcceptanceError:
                # The registry update is already durable.  Restore the exact
                # previous bytes so a missing audit record never leaves an
                # enabled capability behind.
                self._replace_registry(registry_raw)
                raise
            return result
        finally:
            lease.release()
