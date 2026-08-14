from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import threading
from typing import Any, Callable, Mapping
import uuid

from capability_acceptance import (
    ACCEPTANCE_REPORT_VERSION,
    CALIBRATION_BOUND_ACTIONS,
    CapabilityAcceptanceError,
    CapabilityRegistryPromoter,
    PROMOTABLE_ACTIONS,
    PromotionAuthority,
    PromotionScope,
    action_execution_evidence_error,
    exact_input_evidence_error,
    validated_calibration_evidence,
    validate_acceptance_report,
)


def _payload(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    method = getattr(value, "to_dict", None)
    if callable(method):
        result = method()
        if isinstance(result, Mapping):
            return dict(result)
    raise CapabilityAcceptanceError("验收对象不能转换为 JSON 对象。")


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    encoded = (
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except OSError as exc:
        raise CapabilityAcceptanceError(f"验收状态无法原子写入：{exc}") from exc
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
    return target


def _sha256_paths(paths: list[str]) -> list[str]:
    digests: list[str] = []
    for value in paths:
        try:
            digests.append(hashlib.sha256(Path(value).read_bytes()).hexdigest())
        except OSError as exc:
            raise CapabilityAcceptanceError(f"验收证据无法读取：{value}：{exc}") from exc
    return digests


def _proposal_action(snapshot: Mapping[str, Any]) -> str:
    proposal = snapshot.get("proposal")
    if not isinstance(proposal, Mapping) or proposal.get("status") != "action":
        return ""
    action = proposal.get("action")
    if not isinstance(action, Mapping):
        return ""
    return str(action.get("action") or "").strip()


@dataclass
class CapabilityTrial:
    trial_id: str
    candidate_action: str
    text: str
    device_id: str
    run_dir: Path
    controller: Any = field(repr=False)
    orchestrator: Any = field(repr=False)
    session: Any = field(repr=False)
    code_revision: str
    report_path: Path
    calibration_evidence: dict[str, Any] | None = None
    promotion_authority: PromotionAuthority | None = field(default=None, repr=False)
    promotion_result: dict[str, Any] | None = None
    confirmation_attempted: bool = False
    operation_lock: threading.Lock = field(
        default_factory=threading.Lock,
        repr=False,
        compare=False,
    )

    def snapshot(self) -> dict[str, Any]:
        report: dict[str, Any] | None = None
        if self.report_path.is_file():
            try:
                loaded = json.loads(self.report_path.read_text(encoding="utf-8"))
                report = loaded if isinstance(loaded, dict) else None
            except (OSError, UnicodeError, ValueError, TypeError):
                report = None
        return {
            "trial_id": self.trial_id,
            "candidate_action": self.candidate_action,
            "text": self.text,
            "device_id": self.device_id,
            "code_revision": self.code_revision,
            "calibration_evidence": self.calibration_evidence,
            "session": self.session.snapshot(),
            "report": report,
            "promotion_scope": (
                self.promotion_authority.scope.to_dict()
                if self.promotion_authority is not None
                and not self.promotion_authority.consumed
                else None
            ),
            "promotion": self.promotion_result,
            "requires_restart": bool(
                self.promotion_result
                and self.promotion_result.get("requires_restart")
            ),
            "confirmation_attempted": self.confirmation_attempted,
        }


@dataclass
class RecoveredCapabilityTrial:
    """Read-only trial metadata; one-shot authorities never survive restart."""

    trial_id: str
    candidate_action: str
    text: str
    device_id: str
    run_dir: Path
    code_revision: str
    report_path: Path
    stored_snapshot: dict[str, Any] = field(repr=False)

    def __post_init__(self) -> None:
        raw_session = self.stored_snapshot.get("session")
        session_snapshot = dict(raw_session) if isinstance(raw_session, Mapping) else {}

        class RecoveredSession:
            def __init__(self, payload: dict[str, Any]) -> None:
                self._payload = payload
                self.physical_actions = int(payload.get("physical_actions", 0) or 0)
                self.session_id = str(payload.get("session_id") or "")

            def snapshot(self) -> dict[str, Any]:
                return json.loads(json.dumps(self._payload, ensure_ascii=False))

        self.session = RecoveredSession(session_snapshot)
        self.controller = None
        self.orchestrator = None
        self.promotion_authority = None
        self.promotion_result = (
            dict(self.stored_snapshot.get("promotion"))
            if isinstance(self.stored_snapshot.get("promotion"), Mapping)
            else None
        )
        if self.promotion_result is None:
            try:
                loaded_promotion = json.loads(
                    (self.run_dir / "promotion.json").read_text(encoding="utf-8")
                )
                if isinstance(loaded_promotion, dict):
                    self.promotion_result = loaded_promotion
            except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError):
                pass

    def snapshot(self) -> dict[str, Any]:
        payload = json.loads(json.dumps(self.stored_snapshot, ensure_ascii=False))
        report = None
        if self.report_path.is_file():
            try:
                loaded = json.loads(self.report_path.read_text(encoding="utf-8"))
                report = loaded if isinstance(loaded, dict) else None
            except (OSError, UnicodeError, ValueError, TypeError):
                report = None
        payload.update(
            {
                "trial_id": self.trial_id,
                "candidate_action": self.candidate_action,
                "text": self.text,
                "device_id": self.device_id,
                "code_revision": self.code_revision,
                "session": self.session.snapshot(),
                "report": report,
                "promotion_scope": None,
                "promotion": self.promotion_result,
                "requires_restart": bool(
                    self.promotion_result
                    and self.promotion_result.get("requires_restart")
                ),
                "read_only_recovered": True,
            }
        )
        return payload


class CapabilityAcceptanceManager:
    """Run one provisional generic action without mutating product controllers."""

    def __init__(
        self,
        *,
        provisional_controller_factory: Callable[[str, str], Any],
        orchestrator_factory: Callable[[Any], Any],
        device_registry: Any,
        output_dir: Path,
        registry_path: Path,
        code_revision_provider: Callable[[], str],
        id_factory: Callable[[], str] | None = None,
        promoter_factory: Callable[[Path], CapabilityRegistryPromoter] | None = None,
    ) -> None:
        self.provisional_controller_factory = provisional_controller_factory
        self.orchestrator_factory = orchestrator_factory
        self.device_registry = device_registry
        self.output_dir = Path(output_dir)
        self.registry_path = Path(registry_path)
        self.code_revision_provider = code_revision_provider
        self.id_factory = id_factory or (lambda: uuid.uuid4().hex)
        self.promoter_factory = promoter_factory or CapabilityRegistryPromoter
        self._trials: dict[str, CapabilityTrial | RecoveredCapabilityTrial] = {}
        self._guard = threading.RLock()
        self._recover_read_only_trials()

    def _recover_read_only_trials(self) -> None:
        if not self.output_dir.is_dir():
            return
        output_root = self.output_dir.resolve()
        for run_dir in sorted(self.output_dir.glob("capability_acceptance_*")):
            try:
                resolved_dir = run_dir.resolve(strict=True)
                if output_root not in resolved_dir.parents:
                    continue
                stored = json.loads(
                    (resolved_dir / "trial.json").read_text(encoding="utf-8")
                )
                if not isinstance(stored, dict):
                    continue
                trial_id = str(stored.get("trial_id") or "").strip()
                device_id = str(stored.get("device_id") or "").strip()
                action = str(stored.get("candidate_action") or "").strip()
                text_value = str(stored.get("text") or "").strip()
                revision = str(stored.get("code_revision") or "").strip()
                if (
                    not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", trial_id)
                    or resolved_dir.name != f"capability_acceptance_{trial_id}"
                    or not device_id
                    or action not in PROMOTABLE_ACTIONS
                    or not text_value
                    or not revision
                ):
                    continue
                self._trials[trial_id] = RecoveredCapabilityTrial(
                    trial_id=trial_id,
                    candidate_action=action,
                    text=text_value,
                    device_id=device_id,
                    run_dir=resolved_dir,
                    code_revision=revision,
                    report_path=resolved_dir / "acceptance_report.json",
                    stored_snapshot=stored,
                )
            except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError):
                continue

    @staticmethod
    def _require_live_trial(trial: Any) -> CapabilityTrial:
        if isinstance(trial, RecoveredCapabilityTrial):
            raise CapabilityAcceptanceError(
                "该验收会话来自服务重启前，仅可查看；确认权限不会跨进程恢复。"
            )
        return trial

    @staticmethod
    def _validate_start_values(device_id: str, action: str, text: str) -> tuple[str, str, str]:
        resolved_device = str(device_id or "").strip()
        candidate = str(action or "").strip()
        goal = " ".join(str(text or "").split())
        if not resolved_device or len(resolved_device) > 128:
            raise CapabilityAcceptanceError("验收 device_id 格式无效。")
        if candidate not in PROMOTABLE_ACTIONS:
            raise CapabilityAcceptanceError(
                f"动作 {candidate or 'missing'} 不能进入真机能力验收。"
            )
        if not goal or len(goal) > 500:
            raise CapabilityAcceptanceError("验收目标长度必须在 1～500 个字符之间。")
        return resolved_device, candidate, goal

    @staticmethod
    def _ensure_candidate(trial: CapabilityTrial) -> None:
        snapshot = trial.session.snapshot()
        if snapshot.get("status") != "awaiting_confirmation":
            return
        proposed = _proposal_action(snapshot)
        if proposed != trial.candidate_action:
            try:
                trial.orchestrator.cancel(trial.session)
            finally:
                raise CapabilityAcceptanceError(
                    "Qwen 当前唯一动作不是本次候选动作："
                    f"期望 {trial.candidate_action}，实际 {proposed or 'missing'}。"
                )

    @staticmethod
    def _controller_calibration_evidence(
        controller: Any,
        action: str,
    ) -> dict[str, Any] | None:
        if action not in CALIBRATION_BOUND_ACTIONS:
            return None
        calibration_path = getattr(controller, "calibration_path", None)
        if calibration_path is None:
            raise CapabilityAcceptanceError(
                "正式长按/拖动/系统边缘唤栏验收要求设备控制器提供触控标定路径。"
            )
        return validated_calibration_evidence(Path(calibration_path))

    def start(
        self,
        *,
        device_id: str,
        candidate_action: str,
        text: str,
    ) -> CapabilityTrial:
        resolved_device, action, goal = self._validate_start_values(
            device_id,
            candidate_action,
            text,
        )
        active = self.device_registry.active_session(resolved_device)
        if active is not None:
            raise CapabilityAcceptanceError(
                f"设备 {resolved_device} 已有活动任务：{active}。"
            )
        trial_id = str(self.id_factory() or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", trial_id):
            raise CapabilityAcceptanceError("验收 trial_id 格式无效。")
        with self._guard:
            if trial_id in self._trials:
                raise CapabilityAcceptanceError(f"验收 trial_id 已存在：{trial_id}。")

        revision = str(self.code_revision_provider() or "").strip()
        if not revision or len(revision) > 128:
            raise CapabilityAcceptanceError("无法记录当前代码提交，验收已取消。")
        if revision.endswith("+dirty"):
            raise CapabilityAcceptanceError("当前代码存在未提交修改，不能开始真机验收。")

        controller = self.provisional_controller_factory(resolved_device, action)
        calibration_evidence = self._controller_calibration_evidence(
            controller,
            action,
        )
        orchestrator = self.orchestrator_factory(controller)
        session_id = f"capability-trial-{trial_id}"
        run_dir = self.output_dir / f"capability_acceptance_{trial_id}"
        run_dir.mkdir(parents=True, exist_ok=False)
        session = orchestrator.start(
            session_id=session_id,
            raw_goal=goal,
            device_id=resolved_device,
            run_dir=run_dir,
        )
        if int(getattr(session, "physical_actions", 0)) != 0:
            try:
                orchestrator.cancel(session)
            finally:
                raise CapabilityAcceptanceError("验收启动阶段错误地产生了物理动作。")
        trial = CapabilityTrial(
            trial_id=trial_id,
            candidate_action=action,
            text=goal,
            device_id=resolved_device,
            run_dir=run_dir,
            controller=controller,
            orchestrator=orchestrator,
            session=session,
            code_revision=revision,
            report_path=run_dir / "acceptance_report.json",
            calibration_evidence=calibration_evidence,
        )
        self._ensure_candidate(trial)
        with self._guard:
            self._trials[trial_id] = trial
        _atomic_write_json(run_dir / "trial.json", trial.snapshot())
        return trial

    def get(self, trial_id: str) -> CapabilityTrial:
        with self._guard:
            trial = self._trials.get(str(trial_id or "").strip())
        if trial is None:
            raise CapabilityAcceptanceError("真机能力验收会话不存在。")
        return trial

    def snapshots(self) -> list[dict[str, Any]]:
        with self._guard:
            trials = list(self._trials.values())
        return [trial.snapshot() for trial in trials]

    def request_stop_all(self) -> list[str]:
        with self._guard:
            trials = list(self._trials.values())
        requested: list[str] = []
        for trial in trials:
            if isinstance(trial, RecoveredCapabilityTrial) or trial.report_path.exists():
                continue
            request_stop = getattr(trial.controller, "request_stop", None)
            if callable(request_stop):
                request_stop()
                requested.append(trial.trial_id)
        return requested

    def approve_risks(
        self,
        trial_id: str,
        confirmation: Mapping[str, Any],
    ) -> Any:
        trial = self._require_live_trial(self.get(trial_id))
        result = trial.orchestrator.approve_risks(trial.session, confirmation)
        if int(getattr(trial.session, "physical_actions", 0)) != 0:
            raise CapabilityAcceptanceError("验收风险确认错误地产生了物理动作。")
        self._ensure_candidate(trial)
        _atomic_write_json(trial.run_dir / "trial.json", trial.snapshot())
        return result

    @staticmethod
    def _task_id(snapshot: Mapping[str, Any]) -> str:
        scope = snapshot.get("confirmation_scope")
        if isinstance(scope, Mapping) and isinstance(scope.get("task_id"), str):
            return str(scope["task_id"])
        graph = snapshot.get("task_graph")
        if isinstance(graph, Mapping) and isinstance(graph.get("task_id"), str):
            return str(graph["task_id"])
        return ""

    def _write_pass_or_fail_report(
        self,
        trial: CapabilityTrial,
        *,
        before_snapshot: Mapping[str, Any],
        result: Any,
    ) -> dict[str, Any]:
        execution = _payload(result)
        before_paths = [str(path) for path in getattr(result, "before_frame_paths", ())]
        after_paths = [str(path) for path in getattr(result, "after_frame_paths", ())]
        resolved = getattr(result, "resolved_action", None)
        resolved_kind = str(getattr(resolved, "kind", "") or "").strip()
        physical_actions = getattr(result, "physical_actions", 0)
        action_outcome = str(getattr(result, "action_outcome", "") or "").strip()
        observation_errors = list(getattr(result, "observation_errors", ()) or ())
        verification_errors = list(getattr(result, "verification_errors", ()) or ())
        if trial.candidate_action == "input_verified_text":
            exact_error = exact_input_evidence_error(execution)
            if exact_error:
                verification_errors.append(exact_error)
                action_outcome = "mismatched"
        before_scene = getattr(result, "before_scene", None)
        after_scene = getattr(result, "after_scene", None)
        before_fingerprint = str(getattr(before_scene, "fingerprint", "") or "")
        after_fingerprint = str(getattr(after_scene, "fingerprint", "") or "")
        after_observation = getattr(trial.session, "trusted_observation", None)
        after_observation_id = str(
            getattr(after_observation, "observation_id", "") or ""
        )
        confirmation_scope = before_snapshot.get("confirmation_scope")
        confirmation_scope = (
            dict(confirmation_scope) if isinstance(confirmation_scope, Mapping) else {}
        )
        before_observation_id = str(
            confirmation_scope.get("observation_id") or ""
        )
        scoped_before_fingerprint = str(
            confirmation_scope.get("fingerprint") or ""
        )
        action_evidence_error = action_execution_evidence_error(
            trial.candidate_action,
            execution,
        )
        if action_evidence_error:
            verification_errors.append(action_evidence_error)
            action_outcome = "mismatched"
        passed = bool(
            not isinstance(physical_actions, bool)
            and physical_actions == 1
            and resolved_kind == trial.candidate_action
            and action_outcome == "matched"
            and not observation_errors
            and not verification_errors
            and len(before_paths) == 4
            and len(after_paths) == 4
            and before_fingerprint
            and scoped_before_fingerprint
            and after_fingerprint
            and before_fingerprint != after_fingerprint
            and scoped_before_fingerprint != after_fingerprint
            and before_observation_id
            and after_observation_id
            and before_observation_id != after_observation_id
        )
        execution["resolved_action"] = {
            **(
                execution.get("resolved_action")
                if isinstance(execution.get("resolved_action"), dict)
                else {}
            ),
            "kind": resolved_kind,
        }
        execution["observation_errors"] = observation_errors
        execution["verification_errors"] = verification_errors
        report = {
            "version": ACCEPTANCE_REPORT_VERSION,
            "trial_id": trial.trial_id,
            "session_id": str(getattr(trial.session, "session_id", "")),
            "task_id": self._task_id(before_snapshot),
            "device_id": trial.device_id,
            "candidate_action": trial.candidate_action,
            "calibration_evidence": trial.calibration_evidence,
            "status": "passed" if passed else "failed",
            "code_revision": trial.code_revision,
            "physical_actions": physical_actions,
            "action_outcome": action_outcome,
            "confirmation_scope": confirmation_scope,
            "before_observation": {
                "observation_id": before_observation_id,
                "fingerprint": scoped_before_fingerprint,
            },
            "after_observation": {
                "observation_id": after_observation_id,
                "fingerprint": after_fingerprint,
            },
            "execution": execution,
            "before_frame_paths": before_paths,
            "after_frame_paths": after_paths,
            "before_frame_sha256": _sha256_paths(before_paths),
            "after_frame_sha256": _sha256_paths(after_paths),
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        if passed:
            candidate_path = trial.run_dir / "acceptance_report.candidate.json"
            _atomic_write_json(candidate_path, report)
            try:
                validate_acceptance_report(candidate_path)
            finally:
                candidate_path.unlink(missing_ok=True)
        _atomic_write_json(trial.report_path, report)
        return report

    def _write_exception_report(
        self,
        trial: CapabilityTrial,
        *,
        before_snapshot: Mapping[str, Any],
        before_actions: int,
        exc: Exception,
        result: Any | None = None,
    ) -> dict[str, Any]:
        current_actions = int(getattr(trial.session, "physical_actions", 0))
        request_actions = max(
            0,
            current_actions - before_actions,
            int(getattr(exc, "physical_actions", 0) or 0),
            int(getattr(result, "physical_actions", 0) or 0) if result is not None else 0,
        )
        evidence = list(getattr(exc, "evidence", ()) or ())
        if result is not None:
            evidence.extend(getattr(result, "before_frame_paths", ()) or ())
            evidence.extend(getattr(result, "after_frame_paths", ()) or ())
        evidence = list(dict.fromkeys(str(path) for path in evidence))
        observation_errors = [
            str(value) for value in getattr(exc, "observation_errors", ()) or ()
        ]
        verification_errors = [
            str(value) for value in getattr(exc, "verification_errors", ()) or ()
        ]
        if result is not None:
            observation_errors.extend(
                str(value)
                for value in getattr(result, "observation_errors", ()) or ()
            )
            verification_errors.extend(
                str(value)
                for value in getattr(result, "verification_errors", ()) or ()
            )
        failure = {
            "version": ACCEPTANCE_REPORT_VERSION,
            "trial_id": trial.trial_id,
            "session_id": str(getattr(trial.session, "session_id", "")),
            "task_id": self._task_id(before_snapshot),
            "device_id": trial.device_id,
            "candidate_action": trial.candidate_action,
            "calibration_evidence": trial.calibration_evidence,
            "status": "failed",
            "code_revision": trial.code_revision,
            "physical_actions": request_actions,
            "action_outcome": str(
                getattr(result, "action_outcome", "") or "observation_failed"
            ),
            "confirmation_scope": (
                dict(before_snapshot.get("confirmation_scope"))
                if isinstance(before_snapshot.get("confirmation_scope"), Mapping)
                else {}
            ),
            "error": str(exc),
            "evidence": evidence,
            "observation_errors": list(dict.fromkeys(observation_errors)),
            "verification_errors": list(dict.fromkeys(verification_errors)),
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        _atomic_write_json(trial.report_path, failure)
        return failure

    def _pause_if_owned(self, trial: CapabilityTrial) -> None:
        active_session = self.device_registry.active_session(trial.device_id)
        if active_session == getattr(trial.session, "session_id", None):
            trial.orchestrator.pause(trial.session)

    def confirm(
        self,
        trial_id: str,
        confirmation: Mapping[str, Any],
    ) -> Any:
        trial = self._require_live_trial(self.get(trial_id))
        if not trial.operation_lock.acquire(blocking=False):
            raise CapabilityAcceptanceError("验收确认或晋级正在处理中。")
        try:
            if trial.report_path.exists():
                raise CapabilityAcceptanceError("验收报告已经生成，禁止重复执行或覆盖。")
            if trial.confirmation_attempted:
                raise CapabilityAcceptanceError("验收动作确认已经尝试，禁止重复执行。")
            self._ensure_candidate(trial)
            before_snapshot = trial.session.snapshot()
            before_actions = int(getattr(trial.session, "physical_actions", 0))
            trial.confirmation_attempted = True
            result = None
            try:
                current_calibration = self._controller_calibration_evidence(
                    trial.controller,
                    trial.candidate_action,
                )
                if current_calibration != trial.calibration_evidence:
                    raise CapabilityAcceptanceError(
                        "验收开始后触控标定发生变化；本次会话已失效，必须重新创建。"
                    )
                result = trial.orchestrator.confirm_one(trial.session, confirmation)
                request_actions = (
                    int(getattr(trial.session, "physical_actions", 0)) - before_actions
                )
                if request_actions != 1 or int(getattr(result, "physical_actions", 0)) != 1:
                    raise CapabilityAcceptanceError(
                        "验收确认必须恰好产生一个物理动作，"
                        f"实际为 {request_actions}。"
                    )
                report = self._write_pass_or_fail_report(
                    trial,
                    before_snapshot=before_snapshot,
                    result=result,
                )
                if report["status"] != "passed":
                    raise CapabilityAcceptanceError("真机动作未满足验收通过标准。")
            except Exception as exc:
                if not trial.report_path.exists():
                    self._write_exception_report(
                        trial,
                        before_snapshot=before_snapshot,
                        before_actions=before_actions,
                        exc=exc,
                        result=result,
                    )
                raise
            finally:
                self._pause_if_owned(trial)
                _atomic_write_json(trial.run_dir / "trial.json", trial.snapshot())

            promoter = self.promoter_factory(self.registry_path)
            trial.promotion_authority = PromotionAuthority(
                promoter.preview(trial.report_path)
            )
            _atomic_write_json(trial.run_dir / "trial.json", trial.snapshot())
            return result
        finally:
            trial.operation_lock.release()

    def promotion_scope(self, trial_id: str) -> PromotionScope:
        trial = self._require_live_trial(self.get(trial_id))
        authority = trial.promotion_authority
        if authority is None or authority.consumed:
            raise CapabilityAcceptanceError("当前验收没有可用的能力晋级确认。")
        return authority.scope

    def promote(
        self,
        trial_id: str,
        confirmation: Mapping[str, Any],
    ) -> dict[str, Any]:
        trial = self._require_live_trial(self.get(trial_id))
        if not trial.operation_lock.acquire(blocking=False):
            raise CapabilityAcceptanceError("验收确认或晋级正在处理中。")
        try:
            authority = trial.promotion_authority
            if authority is None:
                raise CapabilityAcceptanceError("当前验收没有可用的能力晋级确认。")
            active_session = self.device_registry.active_session(trial.device_id)
            if active_session is not None:
                raise CapabilityAcceptanceError(
                    f"设备 {trial.device_id} 仍有活动任务：{active_session}，不能晋级。"
                )
            try:
                current_revision = str(self.code_revision_provider() or "").strip()
            except Exception:
                authority.consumed = True
                raise
            if current_revision != trial.code_revision:
                authority.consumed = True
                raise CapabilityAcceptanceError(
                    "验收后代码状态发生变化，晋级确认已作废；请重启后重新验收。"
                )
            result = self.promoter_factory(self.registry_path).promote(
                trial.report_path,
                confirmation=confirmation,
                authority=authority,
            )
            trial.promotion_result = dict(result)
            _atomic_write_json(trial.run_dir / "trial.json", trial.snapshot())
            return result
        finally:
            trial.operation_lock.release()

    def cancel(self, trial_id: str) -> None:
        trial = self._require_live_trial(self.get(trial_id))
        request_stop = getattr(trial.controller, "request_stop", None)
        if callable(request_stop):
            request_stop()
        with trial.operation_lock:
            trial.orchestrator.cancel(trial.session)
            if trial.promotion_authority is not None:
                trial.promotion_authority.consumed = True
            _atomic_write_json(trial.run_dir / "trial.json", trial.snapshot())
