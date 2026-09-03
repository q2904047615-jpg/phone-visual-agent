"""File-backed capability acceptance validation and promotion infrastructure."""

from __future__ import annotations

from agent.domain.validation import DataclassWire, reject_if
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import threading
from typing import Any, Callable, Mapping

from PIL import Image, UnidentifiedImageError
from agent.infrastructure.orientation_safety import OrientationCredential, OrientationSafetyError, frame_fingerprint
from agent.infrastructure.atomic_files import atomic_replace_bytes, write_new_bytes

from agent.infrastructure.device_exclusivity import InterProcessLease
from agent.infrastructure.tap_calibration import (
    Affine2D,
    CALIBRATION_VERSION,
    MIN_COVERAGE_SPAN_X,
    MIN_COVERAGE_SPAN_Y,
    TapCalibrationError,
)
from agent.domain.action_capabilities import (
    CALIBRATION_BOUND_ACTIONS,
    PROMOTABLE_ACTIONS,
    physical_capability_for_action,
)


ACCEPTANCE_REPORT_VERSION = 3


def _normalized_coverage_bounds(value: Any, *, label: str) -> list[float]:
    reject_if(
        not isinstance(value, list) or len(value) != 4 or any((isinstance(item, bool) or not isinstance(item, (int,
        float)) or (not math.isfinite(float(item))) for item in value)),
        CapabilityAcceptanceError(f"{label}边界格式无效。"),
    )
    bounds = [float(item) for item in value]
    reject_if(
        any((not 0.0 <= item <= 1.0 for item in bounds)) or bounds[2] - bounds[0] < MIN_COVERAGE_SPAN_X
        or bounds[3] - bounds[1] < MIN_COVERAGE_SPAN_Y,
        CapabilityAcceptanceError(f"{label}没有覆盖足够的归一化屏幕范围。"),
    )
    return bounds


def _validated_coverage(value: Any, *, label: str) -> list[float]:
    reject_if(not isinstance(value, dict) or value.get('sufficient') is not True, CapabilityAcceptanceError(f"{label}没有足够的实测屏幕覆盖。"))
    bounds = _normalized_coverage_bounds(value.get('normalized_bounds'), label=label)
    hull = value.get("normalized_hull")
    reject_if(
        not isinstance(hull, list) or len(hull) < 3 or any((not isinstance(point,
        list) or len(point) != 2 or any((isinstance(item, bool) or not isinstance(item, (int,
        float)) or (not math.isfinite(float(item))) or (not 0.0 <= float(item) <= 1.0) for item in point)) for point
        in hull)),
        CapabilityAcceptanceError(f"{label}凸包格式无效。"),
    )
    derived = [min((float(point[0]) for point in hull)), min((float(point[1]) for point in hull)),
        max((float(point[0]) for point in hull)), max((float(point[1]) for point in hull))]
    reject_if(any((abs(stored - computed) > 1e-06 for stored, computed in zip(bounds, derived))), CapabilityAcceptanceError(f"{label}边界与凸包不一致。"))
    return bounds


def validated_calibration_evidence(path: Path) -> dict[str, Any]:
    """Load the exact active calibration state used by one gesture trial."""

    try:
        resolved = Path(path).resolve(strict=True)
        raw = resolved.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise CapabilityAcceptanceError(f"触控标定无法读取：{exc}") from exc
    reject_if(not isinstance(payload, dict), CapabilityAcceptanceError("触控标定必须是 JSON 对象。"))
    version = payload.get("version")
    reject_if(isinstance(version, bool) or not isinstance(version, int) or version < CALIBRATION_VERSION, CapabilityAcceptanceError("触控标定版本过旧，不能用于正式手势验收。"))
    reject_if(payload.get('enabled') is not True or payload.get('validated') is not True, CapabilityAcceptanceError("触控标定尚未启用并完成独立验证。"))
    reject_if(payload.get('accepted_fit') is not True, CapabilityAcceptanceError("触控标定拟合尚未达到验收标准。"))
    try:
        Affine2D.from_json(payload["target_to_command"])
    except (KeyError, TypeError, ValueError, TapCalibrationError) as exc:
        raise CapabilityAcceptanceError("触控标定变换矩阵无效。") from exc
    frame_size = payload.get("frame_size")
    reject_if(
        not isinstance(frame_size, list) or len(frame_size) != 2 or any((isinstance(item, bool) or not isinstance(item,
        (int, float)) or item <= 0 for item in frame_size)),
        CapabilityAcceptanceError("触控标定 frame_size 无效。"),
    )
    bounds = _validated_coverage(payload.get("coverage"), label="触控标定采集覆盖")
    validation = payload.get("validation")
    reject_if(
        not isinstance(validation, dict) or validation.get('passed') is not True or validation.get('coverage_passed')
        is not True,
        CapabilityAcceptanceError("触控标定缺少通过的独立验证记录。"),
    )
    validation_bounds = _validated_coverage(validation.get('coverage'), label='触控标定独立验证覆盖')
    return {'version': version, 'sha256': _sha256_bytes(raw), 'frame_size': [float(frame_size[0]),
        float(frame_size[1])], 'coverage_bounds': bounds, 'validation_coverage_bounds': validation_bounds}


def _calibration_evidence(value: Any, *, action: str) -> dict[str, Any] | None:
    if action not in CALIBRATION_BOUND_ACTIONS:
        reject_if(value is not None, CapabilityAcceptanceError("非点位手势验收不能携带触控标定证据。"))
        return None
    reject_if(
        not isinstance(value, dict) or set(value) != {'version', 'sha256', 'frame_size', 'coverage_bounds',
        'validation_coverage_bounds'},
        CapabilityAcceptanceError("手势验收缺少完整触控标定证据。"),
    )
    version = value.get("version")
    digest = value.get("sha256")
    frame_size = value.get("frame_size")
    bounds = _normalized_coverage_bounds(value.get('coverage_bounds'), label='手势验收采集覆盖')
    validation_bounds = _normalized_coverage_bounds(value.get('validation_coverage_bounds'), label='手势验收独立验证覆盖')
    reject_if(isinstance(version, bool) or not isinstance(version, int) or version < CALIBRATION_VERSION, CapabilityAcceptanceError("手势验收触控标定版本无效。"))
    reject_if(
        not isinstance(digest, str) or len(digest) != 64 or any((character not in '0123456789abcdef' for character
        in digest)),
        CapabilityAcceptanceError("手势验收触控标定摘要无效。"),
    )
    reject_if(
        not isinstance(frame_size, list) or len(frame_size) != 2 or any((isinstance(item, bool) or not isinstance(item,
        (int, float)) or item <= 0 for item in frame_size)),
        CapabilityAcceptanceError("手势验收触控标定范围无效。"),
    )
    return {'version': version, 'sha256': digest, 'frame_size': [float(item) for item in frame_size],
        'coverage_bounds': bounds, 'validation_coverage_bounds': validation_bounds}


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
    reject_if(not isinstance(value, dict), CapabilityAcceptanceError(f"{label}必须是 JSON 对象。"))
    return value


def _required_text(value: Any, *, field: str, max_length: int=256) -> str:
    reject_if(not isinstance(value, str) or not value.strip(), CapabilityAcceptanceError(f"验收报告字段 {field} 不能为空。"))
    clean = value.strip()
    reject_if(len(clean) > max_length, CapabilityAcceptanceError(f"验收报告字段 {field} 过长。"))
    return clean


def _observation(value: Any, *, field: str) -> dict[str, str]:
    reject_if(not isinstance(value, dict), CapabilityAcceptanceError(f"验收报告字段 {field} 必须是对象。"))
    return {'observation_id': _required_text(value.get('observation_id'), field=f'{field}.observation_id'),
        'fingerprint': _required_text(value.get('fingerprint'), field=f'{field}.fingerprint')}


def _confirmation_scope(value: Any, *, session_id: str, task_id: str, device_id: str) -> dict[str, Any]:
    reject_if(not isinstance(value, dict), CapabilityAcceptanceError("验收报告 confirmation_scope 必须是对象。"))
    required = {'session_id', 'task_id', 'device_id', 'revision', 'subgoal_id', 'effect_ids', 'observation_id',
        'fingerprint'}
    reject_if(set(value) != required, CapabilityAcceptanceError("验收报告 confirmation_scope 字段不完整。"))
    normalized = dict(value)
    for (field, expected) in (('session_id', session_id), ('task_id', task_id), ('device_id', device_id)):
        reject_if(normalized.get(field) != expected, CapabilityAcceptanceError(f'验收报告 confirmation_scope.{field} 与报告范围不一致。'))
    revision = normalized.get("revision")
    reject_if(isinstance(revision, bool) or not isinstance(revision, int) or revision < 1, CapabilityAcceptanceError("验收报告 confirmation_scope.revision 无效。"))
    _required_text(normalized.get("subgoal_id"), field="confirmation_scope.subgoal_id")
    _required_text(normalized.get('observation_id'), field='confirmation_scope.observation_id')
    _required_text(normalized.get('fingerprint'), field='confirmation_scope.fingerprint')
    effect_ids = normalized.get("effect_ids")
    reject_if(
        not isinstance(effect_ids, list) or any((not isinstance(item, str) or not item.strip() for item in effect_ids)),
        CapabilityAcceptanceError("验收报告 confirmation_scope.effect_ids 无效。"),
    )
    reject_if(effect_ids != sorted(set(effect_ids)), CapabilityAcceptanceError('验收报告 confirmation_scope.effect_ids 必须去重并排序。'))
    return normalized


def _validate_frame_paths(value: Any, *, field: str, trial_root: Path, expected_sha256: Any) -> tuple[str, ...]:
    reject_if(not isinstance(value, list) or len(value) != 4, CapabilityAcceptanceError(f"{field} 必须恰好包含四张 JPEG 证据。"))
    reject_if(
        not isinstance(expected_sha256, list) or len(expected_sha256) != 4 or any((not isinstance(item,
        str) or len(item) != 64 or any((character not in '0123456789abcdef' for character in item)) for item
        in expected_sha256)),
        CapabilityAcceptanceError(f"{field} 的证据摘要格式无效。"),
    )
    resolved_root = trial_root.resolve(strict=True)
    result: list[str] = []
    seen: set[Path] = set()
    for (index, item) in enumerate(value, start=1):
        reject_if(not isinstance(item, str) or not item.strip(), CapabilityAcceptanceError(f"{field}[{index}] 证据路径无效。"))
        candidate = Path(item.strip())
        if not candidate.is_absolute():
            candidate = trial_root / candidate
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise CapabilityAcceptanceError(f'{field}[{index}] 证据文件不存在。') from exc
        try:
            resolved.relative_to(resolved_root)
        except ValueError as exc:
            raise CapabilityAcceptanceError(f'{field}[{index}] 证据必须位于本次 trial 目录。') from exc
        reject_if(resolved in seen, CapabilityAcceptanceError(f"{field} 不能重复引用同一张证据。"))
        reject_if(resolved.suffix.lower() not in {'.jpg', '.jpeg'}, CapabilityAcceptanceError(f"{field}[{index}] 证据必须是 JPEG。"))
        try:
            with Image.open(resolved) as image:
                image.verify()
                reject_if(image.format != 'JPEG', CapabilityAcceptanceError(f'{field}[{index}] 证据内容不是 JPEG。'))
        except (OSError, UnidentifiedImageError) as exc:
            raise CapabilityAcceptanceError(f'{field}[{index}] 证据无法读取。') from exc
        reject_if(sha256_file(resolved) != expected_sha256[index - 1], CapabilityAcceptanceError(f'{field}[{index}] 证据摘要与文件不一致。'))
        seen.add(resolved)
        result.append(str(resolved))
    return tuple(result)


def _consistent_frame_size(paths: tuple[str, ...], *, field: str) -> tuple[int, int]:
    sizes: set[tuple[int, int]] = set()
    for value in paths:
        try:
            with Image.open(value) as image:
                sizes.add(tuple(image.size))
        except (OSError, UnidentifiedImageError) as exc:
            raise CapabilityAcceptanceError(f"{field} 证据尺寸无法读取。") from exc
    reject_if(len(sizes) != 1, CapabilityAcceptanceError(f"{field} 四张证据尺寸不一致。"))
    width, height = sizes.pop()
    reject_if(width <= 0 or height <= 0, CapabilityAcceptanceError(f"{field} 证据尺寸无效。"))
    return width, height


def validate_acceptance_report(report_path: Path) -> dict[str, Any]:
    """Validate evidence strong enough to make one capability promotable."""

    resolved_report = Path(report_path).resolve(strict=True)
    report = _load_json_object(resolved_report, label="验收报告")
    reject_if(report.get('version') != ACCEPTANCE_REPORT_VERSION, CapabilityAcceptanceError("验收报告版本无效。"))

    trial_id = _required_text(report.get("trial_id"), field="trial_id", max_length=128)
    session_id = _required_text(report.get('session_id'), field='session_id', max_length=128)
    task_id = _required_text(report.get("task_id"), field="task_id", max_length=128)
    device_id = _required_text(report.get('device_id'), field='device_id', max_length=128)
    action = _required_text(report.get('candidate_action'), field='candidate_action', max_length=64)
    reject_if(action not in PROMOTABLE_ACTIONS, CapabilityAcceptanceError(f"动作类型不能进入真机验收：{action}。"))
    calibration_evidence = _calibration_evidence(report.get('calibration_evidence'), action=action)
    code_revision = _required_text(report.get('code_revision'), field='code_revision', max_length=128)
    reject_if(code_revision.endswith('+dirty'), CapabilityAcceptanceError("验收报告来自未提交代码，不能晋级。"))
    reject_if(report.get('status') != 'passed', CapabilityAcceptanceError("验收报告结果不是 passed，不能晋级。"))
    physical_actions = report.get("physical_actions")
    reject_if(isinstance(physical_actions, bool) or physical_actions != 1, CapabilityAcceptanceError("验收报告物理动作数必须严格等于 1。"))
    reject_if(report.get('action_outcome') != 'matched', CapabilityAcceptanceError("验收结果不是 matched，不能晋级。"))

    before = _observation(report.get("before_observation"), field="before_observation")
    after = _observation(report.get("after_observation"), field="after_observation")
    confirmation_scope = _confirmation_scope(report.get('confirmation_scope'), session_id=session_id, task_id=task_id,
        device_id=device_id)
    reject_if(
        confirmation_scope['observation_id'] != before['observation_id']
        or confirmation_scope['fingerprint'] != before['fingerprint'],
        CapabilityAcceptanceError('动作前 observation/fingerprint 与确认作用域不一致。'),
    )
    reject_if(after['observation_id'] == before['observation_id'], CapabilityAcceptanceError("动作后 observation_id 未变化。"))
    reject_if(after['fingerprint'] == before['fingerprint'], CapabilityAcceptanceError("动作后 fingerprint 未变化。"))

    execution = report.get("execution")
    reject_if(not isinstance(execution, dict), CapabilityAcceptanceError("验收报告 execution 必须是对象。"))
    resolved_action = execution.get("resolved_action")
    reject_if(not isinstance(resolved_action, dict) or resolved_action.get('kind') != action, CapabilityAcceptanceError("执行动作类型与候选动作类型不一致。"))
    observation_errors = execution.get("observation_errors")
    reject_if(not isinstance(observation_errors, list) or observation_errors, CapabilityAcceptanceError("验收报告包含观察错误，不能晋级。"))
    verification_errors = execution.get("verification_errors")
    reject_if(not isinstance(verification_errors, list) or verification_errors, CapabilityAcceptanceError("验收报告包含验证错误，不能晋级。"))
    transition_evidence = execution.get("controller_transition_evidence")
    reject_if(
        not isinstance(transition_evidence, list) or not transition_evidence
        or any(not isinstance(item, str) or not item.strip() for item in transition_evidence),
        CapabilityAcceptanceError("验收报告缺少 Controller 已验证的 typed transition evidence。"),
    )
    before_scene = execution.get("before_scene")
    after_scene = execution.get("after_scene")
    reject_if(not isinstance(before_scene, dict) or not isinstance(after_scene, dict), CapabilityAcceptanceError("验收报告缺少动作前后场景。"))
    reject_if(str(after_scene.get('fingerprint') or '') != after['fingerprint'], CapabilityAcceptanceError("动作后场景 fingerprint 与验收观察不一致。"))
    execution_before_fingerprint = str(before_scene.get("fingerprint") or "")
    reject_if(not execution_before_fingerprint, CapabilityAcceptanceError("执行前场景缺少 fingerprint。"))
    reject_if(execution_before_fingerprint == after['fingerprint'], CapabilityAcceptanceError("动作后 fingerprint 与执行前场景相同。"))

    trial_root = resolved_report.parent
    before_paths = _validate_frame_paths(report.get('before_frame_paths'), field='before_frame_paths',
        trial_root=trial_root, expected_sha256=report.get('before_frame_sha256'))
    after_paths = _validate_frame_paths(report.get('after_frame_paths'), field='after_frame_paths',
        trial_root=trial_root, expected_sha256=report.get('after_frame_sha256'))
    reject_if(set(before_paths) & set(after_paths), CapabilityAcceptanceError("动作前后证据不能引用同一文件。"))
    before_frame_size = _consistent_frame_size(before_paths, field='before_frame_paths')
    after_frame_size = _consistent_frame_size(after_paths, field='after_frame_paths')
    reject_if(before_frame_size != after_frame_size, CapabilityAcceptanceError("动作前后证据画面尺寸不一致。"))
    try:
        orientation_credential = OrientationCredential.from_dict(execution.get('orientation_credential'))
        orientation_credential.assert_authorizes(device_id=device_id, scene_fingerprint=execution_before_fingerprint,
            frame_size=before_frame_size, action=physical_capability_for_action(action))
    except OrientationSafetyError as exc:
        raise CapabilityAcceptanceError(f'独立方向凭据不能支持能力晋级：{exc}') from exc
    before_fingerprints: set[str] = set()
    for path in before_paths:
        with Image.open(path) as image:
            before_fingerprints.add(frame_fingerprint(image.convert("RGB")))
    reject_if(
        not orientation_credential.evidence_frame_fingerprint or orientation_credential.evidence_frame_fingerprint not
        in before_fingerprints,
        CapabilityAcceptanceError('独立方向凭据未绑定动作前保存的稳定帧。'),
    )
    raw_alignment = before_scene.get("camera_alignment")
    if isinstance(raw_alignment, dict):
        compact_layout = raw_alignment.get("camera_layout_orientation")
        compact_rotation = raw_alignment.get("phone_content_rotation")
        reject_if(
            compact_layout not in {'unknown', orientation_credential.camera_layout_orientation} or compact_rotation not
            in {'unknown', orientation_credential.phone_content_rotation},
            CapabilityAcceptanceError('主场景方向事实与独立方向凭据冲突，不能晋级。'),
        )
    normalized = dict(report)
    normalized.update({'trial_id': trial_id, 'session_id': session_id, 'task_id': task_id, 'device_id': device_id,
        'candidate_action': action, 'calibration_evidence': calibration_evidence, 'before_observation': before,
        'after_observation': after, 'confirmation_scope': confirmation_scope, 'before_frame_paths': list(before_paths),
        'after_frame_paths': list(after_paths)})
    return normalized


@dataclass(frozen=True)
class PromotionScope(DataclassWire):
    trial_id: str
    device_id: str
    action: str
    report_sha256: str
    registry_sha256: str

_PROMOTION_AUTHORITY_FACTORY_TOKEN = object()


def _resolved_execution_kind(execution_result: Any) -> str:
    resolved = getattr(execution_result, "resolved_action", None)
    if isinstance(resolved, Mapping):
        return str(resolved.get("kind") or "").strip()
    return str(getattr(resolved, "kind", "") or "").strip()


def _validate_live_promotion_source(*, report: Mapping[str, Any], orientation_credential: OrientationCredential,
    execution_result: Any) -> None:
    reject_if(not isinstance(orientation_credential, OrientationCredential), CapabilityAcceptanceError('能力晋级必须接收本进程真实方向凭据对象。'))
    reject_if(getattr(execution_result, 'orientation_credential', None) is not orientation_credential, CapabilityAcceptanceError("能力晋级方向凭据不是本次动作结果持有的同一对象。"))
    physical_actions = getattr(execution_result, "physical_actions", None)
    reject_if(isinstance(physical_actions, bool) or physical_actions != 1, CapabilityAcceptanceError("能力晋级来源必须是恰好一次物理动作结果。"))
    reject_if(str(getattr(execution_result, 'action_outcome', '') or '') != 'matched', CapabilityAcceptanceError("能力晋级来源动作结果未通过闭环验证。"))
    action = str(report.get("candidate_action") or "")
    reject_if(_resolved_execution_kind(execution_result) != action, CapabilityAcceptanceError("能力晋级来源动作类型与报告不一致。"))
    reject_if(orientation_credential.device_id != report.get('device_id'), CapabilityAcceptanceError("能力晋级 live 方向凭据与报告设备不一致。"))
    before_scene = getattr(execution_result, "before_scene", None)
    before_fingerprint = str(getattr(before_scene, "fingerprint", "") or "")
    reject_if(before_fingerprint != orientation_credential.scene_fingerprint, CapabilityAcceptanceError("能力晋级 live 方向凭据与动作前场景不一致。"))
    before_frames = getattr(execution_result, "before_frames", None)
    reject_if(not isinstance(before_frames, tuple) or not before_frames, CapabilityAcceptanceError("能力晋级来源缺少本进程动作前原始帧对象。"))
    matching_frames = [frame for frame in before_frames if isinstance(frame,
        Image.Image) and tuple(frame.size) == orientation_credential.frame_size
        and (frame_fingerprint(frame) == orientation_credential.frame_fingerprint)]
    reject_if(not matching_frames, CapabilityAcceptanceError("能力晋级 live 方向凭据未绑定动作前帧对象。"))
    execution = report.get("execution")
    reject_if(not isinstance(execution, Mapping), CapabilityAcceptanceError("能力晋级报告缺少执行对象。"))
    live_execution = getattr(execution_result, "to_dict", lambda: None)()
    try:
        same_execution = json.dumps(dict(execution), ensure_ascii=False, sort_keys=True) == json.dumps(dict(live_execution), ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        same_execution = False
    reject_if(not same_execution, CapabilityAcceptanceError("能力晋级报告 execution 不是本进程已验证动作结果的完整序列化视图。"))
    reject_if(execution.get('orientation_credential') != orientation_credential.to_dict(), CapabilityAcceptanceError("能力晋级报告方向凭据不是 live 对象的序列化视图。"))
    reject_if(report.get('physical_actions') != physical_actions, CapabilityAcceptanceError("能力晋级报告与 live 动作计数不一致。"))
    reject_if(report.get('action_outcome') != getattr(execution_result, 'action_outcome', None), CapabilityAcceptanceError("能力晋级报告与 live 动作结果不一致。"))
    before_paths = tuple(str(value) for value in report.get("before_frame_paths", ()))
    result_paths = tuple((str(value) for value in getattr(execution_result, 'before_frame_paths', ())))
    reject_if(before_paths != result_paths, CapabilityAcceptanceError("能力晋级报告与 live 动作前证据路径不一致。"))


class PromotionAuthority:
    """One-shot local authority bound to one report and registry revision."""

    def __init__(self, scope: PromotionScope, *, orientation_credential: OrientationCredential, execution_result: Any,
        report: Mapping[str, Any], _factory_token: object | None=None) -> None:
        reject_if(_factory_token is not _PROMOTION_AUTHORITY_FACTORY_TOKEN, CapabilityAcceptanceError("PromotionAuthority 只能由 live preview 签发。"))
        self.scope = scope
        self.consumed = False
        self._orientation_credential = orientation_credential
        self._execution_result = execution_result
        self._report = json.loads(json.dumps(dict(report), ensure_ascii=False))
        self._source_nonce = object()
        self._lifecycle_lock = threading.Lock()

    def begin_promotion(self) -> None:
        reject_if(not self._lifecycle_lock.acquire(blocking=False), CapabilityAcceptanceError("能力晋级 authority 正在使用。"))

    def end_promotion(self) -> None:
        self._lifecycle_lock.release()

    def _live_source(self) -> tuple[OrientationCredential, Any, Mapping[str, Any]]:
        reject_if(
            self._source_nonce is None or self._orientation_credential is None or self._execution_result is None
            or (self._report is None),
            CapabilityAcceptanceError("能力晋级 live 来源已失效。"),
        )
        return self._orientation_credential, self._execution_result, self._report

    def validate_and_consume(self, value: Mapping[str, Any]) -> None:
        reject_if(self.consumed, CapabilityAcceptanceError("能力晋级确认已使用，禁止重放。"))
        self.consumed = True
        orientation_credential, execution_result, report = self._live_source()
        _validate_live_promotion_source(report=report, orientation_credential=orientation_credential,
            execution_result=execution_result)
        reject_if(not isinstance(value, Mapping), CapabilityAcceptanceError("能力晋级确认范围必须是对象。"))
        expected = self.scope.to_dict()
        requested = dict(value)
        reject_if(
            set(requested) != set(expected) or any((not isinstance(requested.get(key),
            str) or requested.get(key) != expected[key] for key in expected)),
            CapabilityAcceptanceError("能力晋级确认范围不匹配。"),
        )

    def assert_current_report(self, report: Mapping[str, Any]) -> None:
        orientation_credential, execution_result, _stored_report = self._live_source()
        _validate_live_promotion_source(report=report, orientation_credential=orientation_credential,
            execution_result=execution_result)

    def release_source(self) -> None:
        self._source_nonce = None
        self._orientation_credential = None
        self._execution_result = None
        self._report = None

    def invalidate(self) -> None:
        with self._lifecycle_lock:
            self.consumed = True
            self.release_source()


class CapabilityRegistryPromoter:
    """Promote one report-bound action through an atomic registry replacement."""

    def __init__(self, registry_path: Path, *, lease_path: Path | None=None, replace_file: Callable[[Path, Path],
        None] | None=None, now: Callable[[], datetime] | None=None) -> None:
        self.registry_path = Path(registry_path)
        self.lease_path = Path(lease_path if lease_path is not None else self.registry_path.with_suffix(
            '.promotion.lease'))
        self._replace_file = replace_file or (lambda source, target: os.replace(source, target))
        self._now = now or (lambda: datetime.now(timezone.utc))

    @staticmethod
    def _registry_device(payload: dict[str, Any], device_id: str) -> dict[str, Any]:
        reject_if(payload.get('version') != 1 or not isinstance(payload.get('devices'), list), CapabilityAcceptanceError("设备注册表版本或 devices 格式无效。"))
        matches = [item for item in payload['devices'] if isinstance(item,
            dict) and item.get('enabled') is True and (item.get('device_id') == device_id)]
        reject_if(len(matches) != 1, CapabilityAcceptanceError(f'设备注册表没有唯一的已启用设备：{device_id}。'))
        device = matches[0]
        actions = device.get("verified_actions")
        reject_if(not isinstance(actions, list) or any((not isinstance(item, str) or not item.strip() for item in actions)), CapabilityAcceptanceError("设备 verified_actions 格式无效。"))
        normalized = [item.strip() for item in actions]
        reject_if(len(normalized) != len(set(normalized)), CapabilityAcceptanceError("设备 verified_actions 存在重复动作。"))
        device["verified_actions"] = normalized
        return device

    def _load_registry(self) -> tuple[dict[str, Any], bytes]:
        try:
            raw = self.registry_path.read_bytes()
            payload = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise CapabilityAcceptanceError(f"设备注册表无法读取：{exc}") from exc
        reject_if(not isinstance(payload, dict), CapabilityAcceptanceError("设备注册表必须是 JSON 对象。"))
        return payload, raw

    def _require_matching_calibration(self, report: Mapping[str, Any], device: Mapping[str, Any]) -> None:
        action = str(report.get("candidate_action") or "")
        if action not in CALIBRATION_BOUND_ACTIONS:
            return
        calibration_value = device.get("calibration_path")
        reject_if(not isinstance(calibration_value, str) or not calibration_value.strip(), CapabilityAcceptanceError("设备注册表缺少触控标定路径。"))
        calibration_path = Path(calibration_value.strip())
        if not calibration_path.is_absolute():
            calibration_path = self.registry_path.parent / calibration_path
        current = validated_calibration_evidence(calibration_path)
        reject_if(current != report.get('calibration_evidence'), CapabilityAcceptanceError('触控标定与真机验收报告不一致；必须在当前标定上重新验收。'))

    def preview(self, report_path: Path, *, orientation_credential: OrientationCredential | None=None,
        execution_result: Any | None=None) -> PromotionAuthority:
        reject_if(orientation_credential is None or execution_result is None, CapabilityAcceptanceError('能力晋级 preview 必须由 live manager 提供方向凭据和一次动作结果。'))
        report = validate_acceptance_report(report_path)
        _validate_live_promotion_source(report=report, orientation_credential=orientation_credential,
            execution_result=execution_result)
        registry, raw = self._load_registry()
        device = self._registry_device(registry, report["device_id"])
        action = report["candidate_action"]
        physical_action = physical_capability_for_action(action)
        self._require_matching_calibration(report, device)
        reject_if(physical_action in device['verified_actions'],
            CapabilityAcceptanceError(f"设备能力 {action} 对应的物理能力已经启用。"))
        scope = PromotionScope(trial_id=report['trial_id'], device_id=report['device_id'], action=action,
            report_sha256=sha256_file(Path(report_path)), registry_sha256=_sha256_bytes(raw))
        try:
            orientation_credential.claim_live_execution_source()
        except OrientationSafetyError as exc:
            raise CapabilityAcceptanceError(f'能力晋级缺少 live-trial 方向来源：{exc}') from exc
        return PromotionAuthority(scope, orientation_credential=orientation_credential,
            execution_result=execution_result, report=report, _factory_token=_PROMOTION_AUTHORITY_FACTORY_TOKEN)

    @staticmethod
    def _write_new_file(path: Path, payload: bytes) -> None:
        try:
            write_new_bytes(path, payload)
        except FileExistsError as exc:
            raise CapabilityAcceptanceError(f"晋级证据文件已经存在：{path.name}。") from exc
        except OSError as exc:
            raise CapabilityAcceptanceError(f"晋级证据无法写入：{path.name}：{exc}") from exc

    def _replace_registry(self, payload: bytes) -> None:
        try:
            atomic_replace_bytes(self.registry_path, payload, replace_file=self._replace_file)
        except OSError as exc:
            raise CapabilityAcceptanceError(f"设备注册表原子替换失败：{exc}") from exc

    def promote(self, report_path: Path, *, confirmation: Mapping[str, Any], authority: PromotionAuthority) -> dict[str,
        Any]:
        authority.begin_promotion()
        try:
            return self._promote_bound(report_path, confirmation=confirmation, authority=authority)
        finally:
            authority.release_source()
            authority.end_promotion()

    def _promote_bound(self, report_path: Path, *, confirmation: Mapping[str, Any],
        authority: PromotionAuthority) -> dict[str, Any]:
        authority.validate_and_consume(confirmation)
        scope = authority.scope
        lease = InterProcessLease(self.lease_path, owner_id=scope.trial_id,
            metadata={'kind': 'capability_registry_promotion', 'trial_id': scope.trial_id, 'device_id': scope.device_id,
            'action': scope.action})
        reject_if(not lease.acquire(), CapabilityAcceptanceError("设备注册表晋级锁已被其他进程占用。"))
        try:
            reject_if(sha256_file(Path(report_path)) != scope.report_sha256, CapabilityAcceptanceError("验收报告在确认后发生变化。"))
            report = validate_acceptance_report(report_path)
            authority.assert_current_report(report)
            reject_if(
                report['trial_id'] != scope.trial_id or report['device_id'] != scope.device_id
                or report['candidate_action'] != scope.action,
                CapabilityAcceptanceError("验收报告范围在确认后发生变化。"),
            )

            registry, registry_raw = self._load_registry()
            reject_if(_sha256_bytes(registry_raw) != scope.registry_sha256, CapabilityAcceptanceError("设备注册表在确认后发生变化。"))
            device = self._registry_device(registry, scope.device_id)
            self._require_matching_calibration(report, device)
            physical_action = physical_capability_for_action(scope.action)
            reject_if(physical_action in device['verified_actions'],
                CapabilityAcceptanceError(f"设备能力 {scope.action} 对应的物理能力已经启用。"))

            trial_dir = Path(report_path).resolve(strict=True).parent
            backup_path = trial_dir / "registry_before.json"
            promotion_path = trial_dir / "promotion.json"
            reject_if(backup_path.exists() or promotion_path.exists(), CapabilityAcceptanceError("本次 trial 已存在晋级证据，禁止重复晋级。"))

            device['verified_actions'] = sorted({*device['verified_actions'], physical_action})
            encoded_registry = (json.dumps(registry, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
            after_sha256 = _sha256_bytes(encoded_registry)
            promoted_at = self._now().astimezone(timezone.utc).isoformat()
            result = {'version': 1, 'trial_id': scope.trial_id, 'device_id': scope.device_id,
                'action': scope.action, 'physical_action': physical_action,
                'report_sha256': scope.report_sha256, 'registry_before_sha256': scope.registry_sha256,
                'registry_after_sha256': after_sha256, 'promoted_at': promoted_at, 'requires_restart': True}
            encoded_promotion = (json.dumps(result, ensure_ascii=False, indent=2) + '\n').encode('utf-8')

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
