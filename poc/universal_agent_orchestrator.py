from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
import json
import os
from pathlib import Path
import re
import threading
from typing import Any, Callable, Mapping
import uuid

from deepseek_task_graph import DynamicTaskGraph, ObservedState
from device_exclusivity import InterProcessLease
from generic_action_adapter import GenericActionAdapterError
from generic_intent import GenericIntentDraft
from qwen_visual_decision import TrustedObservation
from ui_scene import MIN_TARGET_CONFIDENCE, UISceneError
from universal_action_controller import action_has_account_effect


class UniversalAgentOrchestratorError(RuntimeError):
    pass


class EvidenceStoreError(UniversalAgentOrchestratorError):
    pass


class AgentEvidenceStore:
    """Persist authoritative session evidence with same-directory replaces."""

    def __init__(
        self,
        run_dir: Path,
        *,
        replace_file: Callable[[Path, Path], None] | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self._replace_file = replace_file or (
            lambda source, target: os.replace(source, target)
        )

    @staticmethod
    def _payload(value: Any) -> dict[str, Any]:
        if isinstance(value, Mapping):
            return dict(value)
        for method_name in ("snapshot", "to_dict"):
            method = getattr(value, method_name, None)
            if callable(method):
                payload = method()
                if isinstance(payload, Mapping):
                    return dict(payload)
        raise EvidenceStoreError("证据对象不能转换为 JSON 对象。")

    def write_json(self, name: str, payload: Any) -> Path:
        clean_name = str(name or "").strip()
        if (
            not clean_name
            or Path(clean_name).name != clean_name
            or not clean_name.endswith(".json")
        ):
            raise EvidenceStoreError(f"证据文件名无效：{clean_name!r}")
        try:
            encoded = json.dumps(
                self._payload(payload),
                ensure_ascii=False,
                indent=2,
            )
        except (TypeError, ValueError, EvidenceStoreError) as exc:
            raise EvidenceStoreError(f"证据不能序列化：{exc}") from exc

        self.run_dir.mkdir(parents=True, exist_ok=True)
        target = self.run_dir / clean_name
        temporary = self.run_dir / f".{clean_name}.{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(encoded)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._replace_file(temporary, target)
        except OSError as exc:
            try:
                if temporary.exists():
                    temporary.unlink()
            except OSError:
                pass
            raise EvidenceStoreError(
                f"证据原子写入失败：{clean_name}：{exc}"
            ) from exc
        return target

    def write_session(self, session: Any) -> Path:
        return self.write_json("session.json", session)

    def write_task_graph(self, graph: DynamicTaskGraph) -> Path:
        return self.write_json(
            f"task_graph_revision_{graph.revision}.json",
            graph,
        )

    def write_risk_audit(self, graph: DynamicTaskGraph) -> Path:
        graph_payload = graph.to_dict()
        return self.write_json(
            f"risk_audit_revision_{graph.revision}.json",
            {
                "task_id": graph.task_id,
                "device_id": graph.device_id,
                "revision": graph.revision,
                "current_subgoal": graph_payload.get("current_subgoal"),
                "risk_actions": graph_payload["risk_actions"],
            },
        )

    def write_trusted_observation(self, step_number: int, observation: Any) -> Path:
        return self.write_json(
            f"trusted_observation_step_{int(step_number)}.json",
            observation,
        )

    def write_qwen_decision(self, step_number: int, decision: Any) -> Path:
        return self.write_json(
            f"qwen_decision_step_{int(step_number)}.json",
            decision,
        )

    def write_controller_decision(self, step_number: int, decision: Any) -> Path:
        return self.write_json(
            f"controller_decision_step_{int(step_number)}.json",
            decision,
        )

    def write_verification(self, step_number: int, verification: Any) -> Path:
        return self.write_json(
            f"verification_step_{int(step_number)}.json",
            verification,
        )

    def write_report(self, report: Any) -> Path:
        return self.write_json("report.json", report)


class ObservationBridge:
    """Translate protocol objects without inventing actions or business flow."""

    def goal_draft(self, graph: DynamicTaskGraph) -> GenericIntentDraft:
        graph.validate()
        if not graph.goal.target_apps:
            raise UniversalAgentOrchestratorError(
                "任务图没有目标 App，不能建立通用观察上下文。"
            )
        active = graph.active_subgoal()
        constraints = list(graph.constraints)
        if active is not None:
            constraints.extend(active.constraints)
        entities = dict(graph.goal.entities)
        entities["target_apps"] = [
            {"app_id": item.app_id, "app_name": item.app_name}
            for item in graph.goal.target_apps
        ]
        success_criteria = {
            item.condition_id: {
                "description": item.description,
                "evidence_required": list(item.evidence_required),
                "satisfied": item.satisfied,
            }
            for item in graph.completion_conditions
        }
        account_effects = tuple(
            dict.fromkeys(item.risk_type for item in graph.risk_actions)
        )
        primary_app = graph.goal.target_apps[0]
        draft = GenericIntentDraft(
            understood=True,
            app_id=primary_app.app_id,
            app_name=primary_app.app_name,
            objective=graph.goal.objective,
            entities=entities,
            constraints=tuple(dict.fromkeys(constraints)),
            success_criteria=success_criteria,
            account_effects=account_effects,
            needs_confirmation=True,
        )
        draft.validate()
        return draft

    @staticmethod
    def _text_items(value: Any) -> tuple[str, ...]:
        if not isinstance(value, (list, tuple)):
            return ()
        return tuple(
            text
            for item in value
            if (text := str(item or "").strip())
        )

    def observed_state(
        self,
        *,
        graph: DynamicTaskGraph,
        trusted_observation: Any,
        action_outcome: str,
        verification: dict[str, Any],
    ) -> ObservedState:
        graph.validate()
        observation_device = str(
            getattr(trusted_observation, "device_id", "")
        ).strip()
        if observation_device != graph.device_id:
            raise UniversalAgentOrchestratorError(
                "任务图与可信观察 device_id 不一致。"
            )
        scene = getattr(trusted_observation, "scene", None)
        if scene is None:
            raise UniversalAgentOrchestratorError("可信观察缺少 UIScene。")
        scene.validate()
        fingerprint = str(
            getattr(trusted_observation, "fingerprint", "")
        ).strip()
        if not fingerprint or scene.fingerprint != fingerprint:
            raise UniversalAgentOrchestratorError(
                "可信观察与 UIScene fingerprint 不一致。"
            )
        if not isinstance(verification, dict):
            raise UniversalAgentOrchestratorError("控制器验证结果必须是对象。")

        evidence: list[str] = []

        def add(items: Any) -> None:
            for item in self._text_items(items):
                if item not in evidence:
                    evidence.append(item)

        if scene.summary.strip():
            add((scene.summary.strip(),))
        add(scene.overlays)
        for element in scene.elements:
            add(element.evidence)
            visible = " / ".join(
                item for item in (element.label.strip(), element.meaning.strip()) if item
            )
            if visible:
                add((visible,))
        add(verification.get("visible_evidence"))
        if action_outcome == "matched":
            add(verification.get("completion_evidence"))
        if not evidence:
            raise UniversalAgentOrchestratorError(
                "当前观察没有可交给 DeepSeek 的可见证据。"
            )

        scene_id = str(
            getattr(trusted_observation, "observation_id", "")
        ).strip() or f"{scene.screen_id}:{fingerprint[:16]}"
        observed = ObservedState(
            scene_id=scene_id,
            summary=scene.summary.strip() or "当前可信页面观察",
            visible_evidence=tuple(evidence),
            last_action_outcome=str(action_outcome or "not_applicable"),
            blocked_reasons=self._text_items(verification.get("blocked_reasons")),
        )
        observed.validate()
        return observed


@dataclass
class ConfirmationAuthority:
    session_id: str
    task_id: str
    device_id: str
    revision: int
    subgoal_id: str
    risk_ids: tuple[str, ...]
    observation_id: str
    fingerprint: str
    consumed: bool = False
    invalid_reason: str = ""

    def scope(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "task_id": self.task_id,
            "device_id": self.device_id,
            "revision": self.revision,
            "subgoal_id": self.subgoal_id,
            "risk_ids": sorted(self.risk_ids),
            "observation_id": self.observation_id,
            "fingerprint": self.fingerprint,
        }


@dataclass
class UniversalAgentSessionState:
    session_id: str
    raw_goal: str
    device_id: str
    run_dir: Path
    adapter: Any = field(repr=False)
    evidence_store: AgentEvidenceStore = field(repr=False)
    task_graph: DynamicTaskGraph | None = None
    goal_draft: GenericIntentDraft | None = None
    trusted_observation: Any = None
    qwen_decision: Any = None
    controller_decision: NavigationPolicyDecision | None = None
    confirmation_authority: Any = field(default=None, repr=False)
    status: str = "created"
    step_number: int = 1
    physical_actions: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)
    evidence_paths: list[str] = field(default_factory=list)
    failed_reason: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now().astimezone().isoformat(
            timespec="seconds"
        )
    )

    @staticmethod
    def _serialize(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, Mapping):
            return dict(value)
        method = getattr(value, "to_dict", None)
        if callable(method):
            return method()
        return value

    def snapshot(self) -> dict[str, Any]:
        decision = self._serialize(self.qwen_decision)
        graph = self._serialize(self.task_graph)
        observation = self._serialize(self.trusted_observation)
        proposal = (
            self._serialize(getattr(self.qwen_decision, "proposal", None))
            if self.qwen_decision is not None
            else None
        )
        controller = (
            {
                "allowed": self.controller_decision.allowed,
                "reason": self.controller_decision.reason,
                "canonical_class": self.controller_decision.canonical_class,
                "policy_version": PhaseOneNavigationPolicy.VERSION,
            }
            if self.controller_decision is not None
            else None
        )
        scene = (
            self._serialize(getattr(self.trusted_observation, "scene", None))
            if self.trusted_observation is not None
            else None
        )
        return {
            "session_id": self.session_id,
            "raw_goal": self.raw_goal,
            "device_id": self.device_id,
            "created_at": self.created_at,
            "status": self.status,
            "step_number": self.step_number,
            "physical_actions": self.physical_actions,
            "failed_reason": self.failed_reason,
            "task_graph": graph,
            "goal": self._serialize(self.goal_draft),
            "trusted_observation": observation,
            "current_scene": scene,
            "qwen_decision": decision,
            "proposal": proposal,
            "controller_decision": controller,
            "history": list(self.history),
            "evidence": list(dict.fromkeys(self.evidence_paths)),
            "automatic_loop_enabled": False,
            "confirmation_scope": (
                self.confirmation_authority.scope()
                if self.confirmation_authority is not None
                and not self.confirmation_authority.consumed
                else None
            ),
            "confirmation_ready": bool(
                self.status == "awaiting_confirmation"
                and self.controller_decision is not None
                and self.controller_decision.allowed
                and self.confirmation_authority is not None
                and not self.confirmation_authority.consumed
            ),
        }


class DeviceTaskRegistry:
    """Own active-session identity and one re-entrant lock per device."""

    TERMINAL_STATUSES = frozenset(
        {"succeeded", "blocked", "failed", "paused", "cancelled"}
    )

    def __init__(self, *, lease_directory: Path | None = None) -> None:
        self._guard = threading.RLock()
        self._locks: dict[str, threading.RLock] = {}
        self._active: dict[str, str] = {}
        self._owners: dict[str, tuple[int, int]] = {}
        self._lease_directory = (
            Path(lease_directory) if lease_directory is not None else None
        )
        self._leases: dict[str, InterProcessLease] = {}

    def _lease_path(self, device_id: str) -> Path | None:
        if self._lease_directory is None:
            return None
        import hashlib

        digest = hashlib.sha256(device_id.encode("utf-8")).hexdigest()[:24]
        return self._lease_directory / f"device_{digest}.lease"

    @staticmethod
    def _id(value: str, field_name: str) -> str:
        result = str(value or "").strip()
        if not result:
            raise UniversalAgentOrchestratorError(f"{field_name} 不能为空。")
        return result

    def reserve(self, device_id: str, session_id: str) -> None:
        device = self._id(device_id, "device_id")
        session = self._id(session_id, "session_id")
        with self._guard:
            active = self._active.get(device)
            if active is not None and active != session:
                raise UniversalAgentOrchestratorError(
                    f"设备 {device} 已有活动任务：{active}。"
                )
            lease_path = self._lease_path(device)
            if lease_path is not None and device not in self._leases:
                lease = InterProcessLease(
                    lease_path,
                    owner_id=session,
                    metadata={"device_id": device, "session_id": session},
                )
                if not lease.acquire():
                    payload = InterProcessLease.active_payload(lease_path) or {}
                    owner = str(payload.get("session_id") or "另一个进程")
                    raise UniversalAgentOrchestratorError(
                        f"设备 {device} 已有活动任务：{owner}。"
                    )
                self._leases[device] = lease
            self._active[device] = session
            self._locks.setdefault(device, threading.RLock())

    def release(self, device_id: str, session_id: str) -> None:
        device = self._id(device_id, "device_id")
        session = self._id(session_id, "session_id")
        with self._guard:
            if self._active.get(device) == session:
                self._active.pop(device, None)
                lease = self._leases.pop(device, None)
                if lease is not None:
                    lease.release()

    def active_session(self, device_id: str) -> str | None:
        device = self._id(device_id, "device_id")
        with self._guard:
            local = self._active.get(device)
            if local is not None:
                return local
            lease_path = self._lease_path(device)
            if lease_path is None:
                return None
            payload = InterProcessLease.active_payload(lease_path) or {}
            return str(payload.get("session_id") or "").strip() or None

    @contextmanager
    def device_lock(self, device_id: str):
        device = self._id(device_id, "device_id")
        with self._guard:
            lock = self._locks.setdefault(device, threading.RLock())
        lock.acquire()
        thread_id = threading.get_ident()
        with self._guard:
            owner, depth = self._owners.get(device, (thread_id, 0))
            if depth and owner != thread_id:
                lock.release()
                raise UniversalAgentOrchestratorError(
                    f"设备锁所有者异常：{device}。"
                )
            self._owners[device] = (thread_id, depth + 1)
        try:
            yield
        finally:
            with self._guard:
                owner, depth = self._owners.get(device, (thread_id, 1))
                if owner == thread_id and depth <= 1:
                    self._owners.pop(device, None)
                elif owner == thread_id:
                    self._owners[device] = (owner, depth - 1)
            lock.release()

    def is_locked_by_current_thread(self, device_id: str) -> bool:
        device = self._id(device_id, "device_id")
        with self._guard:
            owner = self._owners.get(device)
            return bool(owner and owner[0] == threading.get_ident() and owner[1] > 0)


class UniversalAgentOrchestrator:
    """Coordinate the generic one-action visual loop without App workflows."""

    def __init__(
        self,
        *,
        deepseek_planner: Any,
        qwen_observer: Any,
        adapter_factory: Callable[[str], Any],
        trusted_observation_factory: Callable[..., Any] | None = None,
        evidence_store_factory: Callable[[Path], AgentEvidenceStore] | None = None,
        policy: PhaseOneNavigationPolicy | None = None,
        bridge: ObservationBridge | None = None,
        device_registry: DeviceTaskRegistry | None = None,
    ) -> None:
        self.deepseek_planner = deepseek_planner
        self.qwen_observer = qwen_observer
        self.adapter_factory = adapter_factory
        self.trusted_observation_factory = (
            trusted_observation_factory or TrustedObservation.from_scene
        )
        self.evidence_store_factory = evidence_store_factory or AgentEvidenceStore
        self.policy = policy or PhaseOneNavigationPolicy()
        self.bridge = bridge or ObservationBridge()
        self.device_registry = device_registry or DeviceTaskRegistry()

    def _release_if_terminal(self, session: UniversalAgentSessionState) -> None:
        if session.status in DeviceTaskRegistry.TERMINAL_STATUSES:
            self.device_registry.release(session.device_id, session.session_id)

    @staticmethod
    def _policy_payload(decision: NavigationPolicyDecision) -> dict[str, Any]:
        return {
            "allowed": decision.allowed,
            "reason": decision.reason,
            "canonical_class": decision.canonical_class,
            "policy_version": PhaseOneNavigationPolicy.VERSION,
        }

    @staticmethod
    def _validate_graph_identity(
        graph: DynamicTaskGraph,
        *,
        device_id: str,
        previous: DynamicTaskGraph | None = None,
    ) -> None:
        graph.validate()
        if graph.device_id != device_id:
            raise UniversalAgentOrchestratorError(
                "DeepSeek 任务图 device_id 与会话设备不一致。"
            )
        if previous is not None:
            if graph.task_id != previous.task_id or graph.device_id != previous.device_id:
                raise UniversalAgentOrchestratorError(
                    "DeepSeek 重规划改变了 task_id 或 device_id。"
                )
            if graph.revision <= previous.revision:
                raise UniversalAgentOrchestratorError(
                    "DeepSeek 重规划 revision 没有增加。"
                )

    @staticmethod
    def _validate_decision_binding(
        graph: DynamicTaskGraph,
        observation: Any,
        decision: Any,
    ) -> None:
        expected = {
            "task_id": graph.task_id,
            "device_id": graph.device_id,
            "revision": graph.revision,
            "observation_id": str(
                getattr(observation, "observation_id", "")
            ),
            "fingerprint": str(getattr(observation, "fingerprint", "")),
        }
        for field_name, expected_value in expected.items():
            actual = getattr(decision, field_name, None)
            if actual != expected_value:
                raise UniversalAgentOrchestratorError(
                    f"Qwen 决策 {field_name} 与当前权威状态不一致。"
                )
        bound = getattr(decision, "trusted_observation", None)
        if bound is None or (
            str(getattr(bound, "observation_id", ""))
            != expected["observation_id"]
            or str(getattr(bound, "fingerprint", ""))
            != expected["fingerprint"]
        ):
            raise UniversalAgentOrchestratorError(
                "Qwen 决策没有绑定当前可信观察。"
            )
        proposal = getattr(decision, "proposal", None)
        if proposal is None:
            raise UniversalAgentOrchestratorError("Qwen 决策缺少 proposal。")
        proposal.validate(observation.scene)

    @staticmethod
    def _remember(session: UniversalAgentSessionState, *paths: Any) -> None:
        for path in paths:
            if path is None:
                continue
            values = path if isinstance(path, (list, tuple)) else (path,)
            for item in values:
                text = str(item or "").strip()
                if text and text not in session.evidence_paths:
                    session.evidence_paths.append(text)

    def _write_terminal_snapshot(self, session: UniversalAgentSessionState) -> None:
        session_path = session.evidence_store.write_session(session)
        self._remember(session, session_path)
        report_path = session.evidence_store.write_report(
            {
                "mode": "universal_agent_safe_live_loop",
                "policy_version": PhaseOneNavigationPolicy.VERSION,
                "session": session.snapshot(),
            }
        )
        self._remember(session, report_path)

    def _current_confirmation_scope(
        self,
        session: UniversalAgentSessionState,
    ) -> dict[str, Any]:
        graph = session.task_graph
        observation = session.trusted_observation
        decision = session.qwen_decision
        if graph is None or observation is None or decision is None:
            raise UniversalAgentOrchestratorError(
                "当前会话没有完整的确认权威状态。"
            )
        self._validate_graph_identity(graph, device_id=session.device_id)
        self._validate_decision_binding(graph, observation, decision)
        if decision.proposal.status != "action":
            raise UniversalAgentOrchestratorError(
                "blocked/finished 决策没有可确认动作。"
            )
        current = graph.active_subgoal()
        if current is None:
            raise UniversalAgentOrchestratorError("当前任务没有活动子目标。")
        return {
            "session_id": session.session_id,
            "task_id": graph.task_id,
            "device_id": graph.device_id,
            "revision": graph.revision,
            "subgoal_id": current.subgoal_id,
            "risk_ids": sorted(current.risk_action_ids),
            "observation_id": str(observation.observation_id),
            "fingerprint": str(observation.fingerprint),
        }

    def _bind_confirmation(self, session: UniversalAgentSessionState) -> None:
        scope = self._current_confirmation_scope(session)
        session.confirmation_authority = ConfirmationAuthority(
            session_id=scope["session_id"],
            task_id=scope["task_id"],
            device_id=scope["device_id"],
            revision=scope["revision"],
            subgoal_id=scope["subgoal_id"],
            risk_ids=tuple(scope["risk_ids"]),
            observation_id=scope["observation_id"],
            fingerprint=scope["fingerprint"],
        )

    @staticmethod
    def _normalize_confirmation(value: Mapping[str, Any]) -> dict[str, Any]:
        required = {
            "session_id",
            "task_id",
            "device_id",
            "revision",
            "subgoal_id",
            "risk_ids",
            "observation_id",
            "fingerprint",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise UniversalAgentOrchestratorError(
                "确认作用域字段缺失或包含额外字段。"
            )
        risk_ids = value.get("risk_ids")
        if not isinstance(risk_ids, list):
            raise UniversalAgentOrchestratorError("确认作用域 risk_ids 必须是数组。")
        revision = value.get("revision")
        if isinstance(revision, bool) or not isinstance(revision, int):
            raise UniversalAgentOrchestratorError("确认作用域 revision 格式无效。")
        return {
            "session_id": str(value.get("session_id") or ""),
            "task_id": str(value.get("task_id") or ""),
            "device_id": str(value.get("device_id") or ""),
            "revision": revision,
            "subgoal_id": str(value.get("subgoal_id") or ""),
            "risk_ids": sorted(str(item) for item in risk_ids),
            "observation_id": str(value.get("observation_id") or ""),
            "fingerprint": str(value.get("fingerprint") or ""),
        }

    def _validate_and_consume_confirmation(
        self,
        session: UniversalAgentSessionState,
        confirmation: Mapping[str, Any],
    ) -> None:
        authority = session.confirmation_authority
        if session.status != "awaiting_confirmation":
            raise UniversalAgentOrchestratorError(
                f"会话已推进，当前状态不能确认：{session.status}。"
            )
        if authority is None:
            raise UniversalAgentOrchestratorError("当前会话没有可用确认作用域。")
        if authority.consumed:
            raise UniversalAgentOrchestratorError("当前确认已使用，禁止重放。")
        current_scope = self._current_confirmation_scope(session)
        if current_scope != authority.scope():
            authority.consumed = True
            authority.invalid_reason = "authoritative_state_changed"
            raise UniversalAgentOrchestratorError(
                "任务、画面或视觉决策已经变化，当前确认已失效。"
            )
        try:
            requested = self._normalize_confirmation(confirmation)
        except UniversalAgentOrchestratorError:
            authority.consumed = True
            authority.invalid_reason = "invalid_confirmation_shape"
            raise
        if requested != current_scope:
            authority.consumed = True
            authority.invalid_reason = "confirmation_scope_mismatch"
            raise UniversalAgentOrchestratorError(
                "确认作用域与当前 task/device/revision/subgoal/risk/observation 不一致。"
            )

    def _advance_after_observation(
        self,
        session: UniversalAgentSessionState,
        *,
        result: Any,
        new_observation: Any,
    ) -> None:
        previous_graph = session.task_graph
        assert previous_graph is not None
        verification = {
            "matched": True,
            "action_outcome": "matched",
            "physical_actions": result.physical_actions,
            "before_fingerprint": result.before_scene.fingerprint,
            "after_fingerprint": result.after_scene.fingerprint,
            "visible_evidence": [result.after_scene.summary],
            "after_frame_paths": list(result.after_frame_paths),
        }
        self._remember(
            session,
            session.evidence_store.write_verification(
                max(1, session.step_number - 1),
                verification,
            ),
        )
        observed = self.bridge.observed_state(
            graph=previous_graph,
            trusted_observation=new_observation,
            action_outcome="matched",
            verification=verification,
        )
        try:
            revised = self.deepseek_planner.replan(
                previous_graph,
                observed,
                trigger="observation_changed",
                reason="一个动作已经执行并由新的可信画面验证。",
            )
            self._validate_graph_identity(
                revised,
                device_id=session.device_id,
                previous=previous_graph,
            )
        except Exception as exc:
            session.status = "blocked"
            session.failed_reason = f"DeepSeek 重规划失败：{exc}"
            session.confirmation_authority = None
            return

        session.task_graph = revised
        session.goal_draft = self.bridge.goal_draft(revised)
        self._remember(
            session,
            session.evidence_store.write_task_graph(revised),
            session.evidence_store.write_risk_audit(revised),
        )
        if revised.status == "completed":
            session.status = "succeeded"
            session.confirmation_authority = None
            return
        current = revised.active_subgoal()
        impact = current.external_impact if current is not None else "unknown"
        if current is None or impact in {"external_state", "unknown"}:
            session.status = "blocked"
            session.failed_reason = (
                "重规划后的当前子目标属于第一阶段禁止范围："
                f"{impact}。"
            )
            session.confirmation_authority = None
            return

        frames = list(result.after_frames)
        context = revised.to_qwen_context()
        decision = self.qwen_observer.decide(
            frames=frames,
            task_context=context,
            trusted_observation=new_observation,
            decision_number=session.step_number,
        )
        self._validate_decision_binding(revised, new_observation, decision)
        session.qwen_decision = decision
        self._remember(
            session,
            session.evidence_store.write_qwen_decision(
                session.step_number,
                decision,
            ),
        )
        if decision.proposal.status != "action":
            session.status = "blocked"
            session.failed_reason = (
                decision.proposal.reason
                if decision.proposal.status == "blocked"
                else "Qwen 完成候选未被当前 DeepSeek revision 确认为完成。"
            )
            session.controller_decision = NavigationPolicyDecision(
                allowed=False,
                reason=session.failed_reason,
            )
            session.confirmation_authority = None
            return
        policy_decision = self.policy.evaluate(
            task_context=context,
            trusted_observation=new_observation,
            decision=decision,
        )
        session.controller_decision = policy_decision
        self._remember(
            session,
            session.evidence_store.write_controller_decision(
                session.step_number,
                self._policy_payload(policy_decision),
            ),
        )
        if not policy_decision.allowed:
            session.status = "blocked"
            session.failed_reason = policy_decision.reason
            session.confirmation_authority = None
            return
        session.status = "awaiting_confirmation"
        self._bind_confirmation(session)

    def refresh_decision(self, session: UniversalAgentSessionState) -> Any:
        """Capture a fresh trusted scene and replace the pending decision.

        This is the read-only implementation behind the web ``/next`` route.
        It deliberately never calls ``adapter.execute`` and always invalidates
        the previous confirmation before touching the camera.
        """

        if self.device_registry.active_session(session.device_id) != session.session_id:
            raise UniversalAgentOrchestratorError(
                "当前会话已不再拥有该设备，禁止重新观察。"
            )
        try:
            with self.device_registry.device_lock(session.device_id):
                return self._refresh_decision_locked(session)
        finally:
            self._release_if_terminal(session)

    def _refresh_decision_locked(self, session: UniversalAgentSessionState) -> Any:
        graph = session.task_graph
        goal = session.goal_draft
        if graph is None or goal is None:
            raise UniversalAgentOrchestratorError("当前会话缺少任务图或目标投影。")
        self._validate_graph_identity(graph, device_id=session.device_id)
        current = graph.active_subgoal()
        impact = current.external_impact if current is not None else "unknown"
        if current is None or impact in {"external_state", "unknown"}:
            raise UniversalAgentOrchestratorError(
                f"第一阶段禁止重新观察后推进 {impact} 子目标。"
            )

        authority = session.confirmation_authority
        if authority is not None:
            authority.consumed = True
            authority.invalid_reason = "fresh_observation_requested"
        session.confirmation_authority = None
        before_actions = session.physical_actions
        try:
            session.status = "observing"
            observation_id = f"obs_{uuid.uuid4().hex}"
            scene, frames, frame_paths = session.adapter.capture_scene(
                goal,
                evidence_dir=session.run_dir,
                prefix=(
                    f"refresh_step_{session.step_number}_"
                    f"{observation_id[-8:]}_frame"
                ),
            )
            self._remember(session, frame_paths)
            observation = self.trusted_observation_factory(
                frames=frames,
                device_id=session.device_id,
                scene=scene,
                observation_id=observation_id,
            )
            session.trusted_observation = observation
            self._remember(
                session,
                session.evidence_store.write_trusted_observation(
                    session.step_number,
                    observation,
                ),
            )

            context = graph.to_qwen_context()
            decision = self.qwen_observer.decide(
                frames=frames,
                task_context=context,
                trusted_observation=observation,
                decision_number=session.step_number,
            )
            self._validate_decision_binding(graph, observation, decision)
            session.qwen_decision = decision
            self._remember(
                session,
                session.evidence_store.write_qwen_decision(
                    session.step_number,
                    decision,
                ),
            )

            if decision.proposal.status == "action":
                policy_decision = self.policy.evaluate(
                    task_context=context,
                    trusted_observation=observation,
                    decision=decision,
                )
                session.controller_decision = policy_decision
                self._remember(
                    session,
                    session.evidence_store.write_controller_decision(
                        session.step_number,
                        self._policy_payload(policy_decision),
                    ),
                )
                if policy_decision.allowed:
                    session.status = "awaiting_confirmation"
                    session.failed_reason = ""
                    self._bind_confirmation(session)
                else:
                    session.status = "blocked"
                    session.failed_reason = policy_decision.reason
            else:
                session.status = "blocked"
                session.failed_reason = (
                    decision.proposal.reason
                    if decision.proposal.status == "blocked"
                    else "重新观察后的完成候选必须由 DeepSeek 新 revision 复核。"
                )
                session.controller_decision = NavigationPolicyDecision(
                    allowed=False,
                    reason=session.failed_reason,
                )
            if session.physical_actions != before_actions:
                raise UniversalAgentOrchestratorError(
                    "重新观察路径错误地改变了物理动作计数。"
                )
            self._write_terminal_snapshot(session)
            return decision
        except Exception as exc:
            session.status = "failed"
            session.failed_reason = str(exc)
            try:
                self._write_terminal_snapshot(session)
            except Exception:
                pass
            raise

    def confirm_one(
        self,
        session: UniversalAgentSessionState,
        confirmation: Mapping[str, Any],
    ) -> Any:
        if self.device_registry.active_session(session.device_id) != session.session_id:
            raise UniversalAgentOrchestratorError(
                "当前会话已不再拥有该设备，禁止执行。"
            )
        try:
            with self.device_registry.device_lock(session.device_id):
                return self._confirm_one_locked(session, confirmation)
        finally:
            self._release_if_terminal(session)

    def _confirm_one_locked(
        self,
        session: UniversalAgentSessionState,
        confirmation: Mapping[str, Any],
    ) -> Any:
        self._validate_and_consume_confirmation(session, confirmation)
        authority = session.confirmation_authority
        assert authority is not None
        graph = session.task_graph
        observation = session.trusted_observation
        decision = session.qwen_decision
        assert graph is not None and observation is not None and decision is not None

        context = graph.to_qwen_context()
        policy_decision = self.policy.evaluate(
            task_context=context,
            trusted_observation=observation,
            decision=decision,
        )
        session.controller_decision = policy_decision
        authority.consumed = True
        authority.invalid_reason = "consumed_before_execution"
        if not policy_decision.allowed:
            session.status = "blocked"
            session.failed_reason = policy_decision.reason
            self._remember(
                session,
                session.evidence_store.write_controller_decision(
                    session.step_number,
                    self._policy_payload(policy_decision),
                ),
            )
            self._write_terminal_snapshot(session)
            raise UniversalAgentOrchestratorError(policy_decision.reason)

        try:
            self._remember(
                session,
                session.evidence_store.write_controller_decision(
                    session.step_number,
                    {
                        **self._policy_payload(policy_decision),
                        "phase": "pre_execute_recheck",
                    },
                ),
            )
        except Exception:
            session.status = "failed"
            session.failed_reason = "执行前控制器证据写入失败。"
            raise

        session.status = "executing_one_action"
        try:
            result = session.adapter.execute(
                requested_action=decision.proposal.action,
                planned_scene=observation.scene,
                goal=session.goal_draft,
                confirmed=True,
                evidence_dir=session.run_dir,
            )
        except GenericActionAdapterError as exc:
            session.physical_actions += max(0, int(exc.physical_actions))
            self._remember(session, exc.evidence)
            session.status = "failed"
            session.failed_reason = str(exc)
            try:
                self._write_terminal_snapshot(session)
            except Exception:
                pass
            raise

        physical_actions = int(result.physical_actions)
        if physical_actions < 0 or physical_actions > 1:
            session.physical_actions += max(0, physical_actions)
            session.status = "failed"
            session.failed_reason = (
                f"单次确认返回了非法物理动作数：{physical_actions}。"
            )
            raise UniversalAgentOrchestratorError(session.failed_reason)
        session.physical_actions += physical_actions
        self._remember(
            session,
            result.evidence,
            result.after_frame_paths,
        )
        session.status = "verifying"
        if (
            result.resolved_action.kind != "wait_for_change"
            and result.after_scene.fingerprint == observation.fingerprint
        ):
            session.status = "failed"
            session.failed_reason = "动作后 fingerprint 没有变化，禁止继续。"
            try:
                self._write_terminal_snapshot(session)
            except Exception:
                pass
            raise UniversalAgentOrchestratorError(session.failed_reason)

        new_observation = self.trusted_observation_factory(
            frames=list(result.after_frames),
            device_id=session.device_id,
            scene=result.after_scene,
            observation_id=f"obs_{uuid.uuid4().hex}",
        )
        if (
            new_observation.observation_id == observation.observation_id
            or (
                result.resolved_action.kind != "wait_for_change"
                and new_observation.fingerprint == observation.fingerprint
            )
        ):
            session.status = "failed"
            session.failed_reason = "动作后可信观察 observation/fingerprint 未更新。"
            raise UniversalAgentOrchestratorError(session.failed_reason)
        session.trusted_observation = new_observation
        session.step_number += 1
        self._remember(
            session,
            session.evidence_store.write_trusted_observation(
                session.step_number,
                new_observation,
            ),
        )
        session.history.append(
            {
                "step_number": session.step_number - 1,
                "task_revision": graph.revision,
                "qwen_decision": UniversalAgentSessionState._serialize(decision),
                "execution": result.to_dict(),
                "after_observation_id": new_observation.observation_id,
                "after_fingerprint": new_observation.fingerprint,
            }
        )
        try:
            session.status = "replanning"
            self._advance_after_observation(
                session,
                result=result,
                new_observation=new_observation,
            )
            self._write_terminal_snapshot(session)
            return result
        except EvidenceStoreError as exc:
            session.status = "failed"
            session.failed_reason = str(exc)
            raise
        except Exception as exc:
            session.status = "failed"
            session.failed_reason = str(exc)
            try:
                self._write_terminal_snapshot(session)
            except Exception:
                pass
            raise

    def start(
        self,
        *,
        session_id: str,
        raw_goal: str,
        device_id: str,
        run_dir: Path,
    ) -> UniversalAgentSessionState:
        resolved_session = str(session_id or "").strip()
        resolved_device = str(device_id or "").strip()
        self.device_registry.reserve(resolved_device, resolved_session)
        try:
            with self.device_registry.device_lock(resolved_device):
                session = self._start_reserved(
                    session_id=resolved_session,
                    raw_goal=raw_goal,
                    device_id=resolved_device,
                    run_dir=run_dir,
                )
        except Exception:
            self.device_registry.release(resolved_device, resolved_session)
            raise
        self._release_if_terminal(session)
        return session

    def _start_reserved(
        self,
        *,
        session_id: str,
        raw_goal: str,
        device_id: str,
        run_dir: Path,
    ) -> UniversalAgentSessionState:
        adapter = self.adapter_factory(device_id)
        store = self.evidence_store_factory(Path(run_dir))
        session = UniversalAgentSessionState(
            session_id=str(session_id or "").strip(),
            raw_goal=" ".join(str(raw_goal or "").split()),
            device_id=str(device_id or "").strip(),
            run_dir=Path(run_dir),
            adapter=adapter,
            evidence_store=store,
        )
        if not session.session_id or not session.raw_goal or not session.device_id:
            raise UniversalAgentOrchestratorError(
                "启动通用 Agent 需要 session_id、目标和 device_id。"
            )
        try:
            session.status = "planning"
            graph = self.deepseek_planner.plan(
                session.raw_goal,
                device_id=session.device_id,
            )
            self._validate_graph_identity(graph, device_id=session.device_id)
            session.task_graph = graph
            session.goal_draft = self.bridge.goal_draft(graph)
            self._remember(
                session,
                store.write_task_graph(graph),
                store.write_risk_audit(graph),
            )

            current = graph.active_subgoal()
            impact = current.external_impact if current is not None else "unknown"
            if impact in {"external_state", "unknown"}:
                session.status = "blocked"
                session.failed_reason = (
                    f"第一阶段禁止 {impact} 子目标进入 Qwen 或机械臂执行。"
                )
                session.controller_decision = NavigationPolicyDecision(
                    allowed=False,
                    reason=session.failed_reason,
                )
                self._remember(
                    session,
                    store.write_controller_decision(
                        session.step_number,
                        self._policy_payload(session.controller_decision),
                    ),
                )
                self._write_terminal_snapshot(session)
                return session
            if current is None:
                session.status = "blocked"
                session.failed_reason = "任务图没有活动子目标。"
                self._write_terminal_snapshot(session)
                return session

            session.status = "observing"
            scene, frames, frame_paths = adapter.capture_scene(
                session.goal_draft,
                evidence_dir=session.run_dir,
                prefix=f"before_step_{session.step_number}_frame",
            )
            self._remember(session, frame_paths)
            observation = self.trusted_observation_factory(
                frames=frames,
                device_id=session.device_id,
                scene=scene,
            )
            session.trusted_observation = observation
            self._remember(
                session,
                store.write_trusted_observation(session.step_number, observation),
            )

            task_context = graph.to_qwen_context()
            decision = self.qwen_observer.decide(
                frames=frames,
                task_context=task_context,
                trusted_observation=observation,
                decision_number=session.step_number,
            )
            self._validate_decision_binding(graph, observation, decision)
            session.qwen_decision = decision
            self._remember(
                session,
                store.write_qwen_decision(session.step_number, decision),
            )

            proposal = decision.proposal
            if proposal.status == "action":
                policy_decision = self.policy.evaluate(
                    task_context=task_context,
                    trusted_observation=observation,
                    decision=decision,
                )
                session.controller_decision = policy_decision
                self._remember(
                    session,
                    store.write_controller_decision(
                        session.step_number,
                        self._policy_payload(policy_decision),
                    ),
                )
                if policy_decision.allowed:
                    session.status = "awaiting_confirmation"
                    self._bind_confirmation(session)
                else:
                    session.status = "blocked"
                    session.failed_reason = policy_decision.reason
            elif proposal.status == "blocked":
                session.status = "blocked"
                session.failed_reason = proposal.reason
                session.controller_decision = NavigationPolicyDecision(
                    allowed=False,
                    reason=proposal.reason,
                )
            elif proposal.status == "finished":
                observed = self.bridge.observed_state(
                    graph=graph,
                    trusted_observation=observation,
                    action_outcome="not_applicable",
                    verification={
                        "completion_evidence": list(proposal.completion_evidence),
                        "visible_evidence": list(proposal.completion_evidence),
                    },
                )
                revised = self.deepseek_planner.replan(
                    graph,
                    observed,
                    trigger="subgoal_completed",
                    reason="Qwen 在当前可信画面中提出完成候选，要求 DeepSeek 复核。",
                )
                self._validate_graph_identity(
                    revised,
                    device_id=session.device_id,
                    previous=graph,
                )
                session.task_graph = revised
                self._remember(
                    session,
                    store.write_task_graph(revised),
                    store.write_risk_audit(revised),
                )
                if revised.status == "completed":
                    session.status = "succeeded"
                else:
                    session.status = "blocked"
                    session.failed_reason = (
                        "Qwen 的完成候选没有被 DeepSeek 新 revision 确认为完成。"
                    )
            else:
                raise UniversalAgentOrchestratorError(
                    f"不支持的 Qwen 状态：{proposal.status}"
                )

            if session.physical_actions != 0:
                raise UniversalAgentOrchestratorError(
                    "start 路径错误地触发了物理动作。"
                )
            self._write_terminal_snapshot(session)
            return session
        except Exception as exc:
            session.status = "failed"
            session.failed_reason = str(exc)
            try:
                self._write_terminal_snapshot(session)
            except Exception:
                pass
            raise

    def pause(self, session: UniversalAgentSessionState) -> None:
        with self.device_registry.device_lock(session.device_id):
            authority = session.confirmation_authority
            if authority is not None:
                authority.consumed = True
                authority.invalid_reason = "paused"
            session.status = "paused"
            session.failed_reason = "用户已暂停；旧确认和旧观察不可复用。"
            self._write_terminal_snapshot(session)
        self.device_registry.release(session.device_id, session.session_id)

    def invalidate_confirmation(
        self,
        session: UniversalAgentSessionState,
        *,
        reason: str,
    ) -> None:
        """Invalidate a pending scope while retaining the device for re-observation."""

        if self.device_registry.active_session(session.device_id) != session.session_id:
            return
        with self.device_registry.device_lock(session.device_id):
            authority = session.confirmation_authority
            if authority is not None:
                authority.consumed = True
                authority.invalid_reason = str(reason or "invalidated")
            session.confirmation_authority = None
            session.status = "needs_reobservation"
            session.failed_reason = (
                "设备或摄像头状态变化；旧确认已失效，必须重新观察后再确认。"
            )
            self._write_terminal_snapshot(session)

    def cancel(self, session: UniversalAgentSessionState) -> None:
        with self.device_registry.device_lock(session.device_id):
            authority = session.confirmation_authority
            if authority is not None:
                authority.consumed = True
                authority.invalid_reason = "cancelled"
            session.status = "cancelled"
            session.failed_reason = "用户已取消任务。"
            self._write_terminal_snapshot(session)
        self.device_registry.release(session.device_id, session.session_id)


@dataclass(frozen=True)
class NavigationPolicyDecision:
    allowed: bool
    reason: str
    canonical_class: str = ""


class PhaseOneNavigationPolicy:
    """Fail-closed gate for the first real-device navigation milestone.

    This class classifies one already proposed visual action.  It never plans
    a task, chooses an App, invents an element, or changes coordinates.
    """

    VERSION = "2026-08-12-phase-one-navigation-v1"
    ALLOWED_ACTIONS = frozenset(
        {"swipe", "back", "wait_for_change", "tap_semantic", "dismiss_overlay"}
    )
    FORBIDDEN_ROLES = frozenset({"toggle", "input", "keyboard_key"})
    NAVIGATION_ROLES = frozenset(
        {"button", "icon", "text", "tab", "image", "list_item"}
    )
    FORBIDDEN_ENGLISH = frozenset(
        {
            "send",
            "publish",
            "post",
            "comment",
            "follow",
            "unfollow",
            "like",
            "favorite",
            "subscribe",
            "pay",
            "purchase",
            "buy",
            "order",
            "delete",
            "remove",
            "submit",
            "save",
            "invite",
            "join",
            "input",
            "type",
            "drag",
            "longpress",
            "confirm",
            "approve",
            "accept",
            "agree",
            "authorize",
        }
    )
    FORBIDDEN_CHINESE = (
        "发送",
        "发布",
        "评论",
        "关注",
        "取关",
        "点赞",
        "收藏",
        "订阅",
        "支付",
        "购买",
        "下单",
        "删除",
        "移除",
        "提交",
        "保存",
        "邀请",
        "加入",
        "输入",
        "长按",
        "拖动",
        "确认",
        "确定",
        "同意",
        "批准",
        "授权",
    )
    NAVIGATION_CLASSES = (
        ("back", frozenset({"back", "return", "previous"}), ("返回", "后退", "上一页")),
        ("close", frozenset({"close", "cancel", "dismiss"}), ("关闭", "取消", "收起")),
        ("tab", frozenset({"tab", "switch"}), ("标签", "切换")),
        ("menu", frozenset({"menu", "more"}), ("菜单", "更多")),
        ("list", frozenset({"list", "item"}), ("列表", "条目")),
        ("search", frozenset({"search"}), ("搜索",)),
        (
            "open",
            frozenset({"open", "enter", "navigate", "entry", "launcher"}),
            ("打开", "进入", "入口"),
        ),
        ("view", frozenset({"view", "details", "detail"}), ("查看", "详情")),
    )

    def __init__(self, *, min_confidence: float = MIN_TARGET_CONFIDENCE) -> None:
        self.min_confidence = float(min_confidence)

    @staticmethod
    def _value(source: Any, name: str, default: Any = "") -> Any:
        if isinstance(source, dict):
            return source.get(name, default)
        return getattr(source, name, default)

    @staticmethod
    def _deny(reason: str) -> NavigationPolicyDecision:
        return NavigationPolicyDecision(allowed=False, reason=reason)

    @staticmethod
    def _tokens(value: str) -> set[str]:
        return {
            token
            for token in re.split(r"[^a-z0-9]+", value.casefold())
            if token
        }

    def _semantic_class(self, *values: str) -> str:
        combined = " ".join(str(value or "").strip() for value in values)
        tokens = self._tokens(combined)
        if tokens.intersection(self.FORBIDDEN_ENGLISH) or any(
            marker in combined for marker in self.FORBIDDEN_CHINESE
        ):
            return "forbidden"
        for canonical, english, chinese in self.NAVIGATION_CLASSES:
            if tokens.intersection(english) or any(marker in combined for marker in chinese):
                return canonical
        return ""

    def evaluate(
        self,
        *,
        task_context: Any,
        trusted_observation: Any,
        decision: Any,
    ) -> NavigationPolicyDecision:
        impact = str(
            self._value(task_context, "current_external_impact", "unknown")
        ).strip()
        if impact in {"external_state", "unknown"}:
            return self._deny(f"第一阶段禁止 {impact} 子目标进入视觉或机械臂执行。")

        proposal = self._value(decision, "proposal", None)
        if proposal is None or str(self._value(proposal, "status", "")) != "action":
            return self._deny("当前 Qwen 决策没有唯一可执行动作。")
        action = self._value(proposal, "action", None)
        if action is None:
            return self._deny("当前 Qwen 决策缺少动作。")
        action_kind = str(self._value(action, "action", "")).strip()
        if action_kind not in self.ALLOWED_ACTIONS:
            return self._deny(f"第一阶段不允许动作：{action_kind or 'missing'}。")
        if action_kind == "wait_for_change":
            if impact not in {"read_only", "navigation_only"}:
                return self._deny(f"等待动作不能用于 {impact} 子目标。")
        elif impact != "navigation_only":
            return self._deny(f"物理导航动作要求 navigation_only，当前为 {impact}。")

        for field in ("task_id", "device_id", "revision"):
            expected = self._value(task_context, field, None)
            actual = self._value(decision, field, None)
            if expected is not None and actual != expected:
                return self._deny(f"Qwen 决策 {field} 与任务上下文不一致。")

        observation_device = str(
            self._value(trusted_observation, "device_id", "")
        ).strip()
        context_device = str(self._value(task_context, "device_id", "")).strip()
        decision_device = str(self._value(decision, "device_id", "")).strip()
        if not observation_device or observation_device not in {
            context_device,
            decision_device,
        } or context_device != decision_device:
            return self._deny("可信观察、任务和 Qwen 决策的 device_id 不一致。")

        observation_fingerprint = str(
            self._value(trusted_observation, "fingerprint", "")
        ).strip()
        decision_fingerprint = str(
            self._value(decision, "fingerprint", "")
        ).strip()
        if not observation_fingerprint or decision_fingerprint != observation_fingerprint:
            return self._deny("Qwen 决策 fingerprint 与当前可信观察不一致。")

        decision_observation = self._value(decision, "trusted_observation", None)
        if decision_observation is not None:
            bound_fingerprint = str(
                self._value(decision_observation, "fingerprint", "")
            ).strip()
            if bound_fingerprint != observation_fingerprint:
                return self._deny("Qwen 决策绑定了不同的可信观察。")

        scene = self._value(trusted_observation, "scene", None)
        if scene is None:
            return self._deny("可信观察缺少页面场景。")
        try:
            scene.validate()
        except (AttributeError, UISceneError) as exc:
            return self._deny(f"可信页面场景无效：{exc}")
        if not scene.stable or float(scene.confidence) < self.min_confidence:
            return self._deny("页面不稳定或整体置信度不足。")
        if scene.fingerprint != observation_fingerprint:
            return self._deny("页面 fingerprint 与可信观察不一致。")
        if float(self._value(decision, "confidence", 0.0)) < self.min_confidence:
            return self._deny("Qwen 决策置信度不足。")

        if action_has_account_effect(action):
            return self._deny("动作语义可能改变账号或外部状态。")

        if action_kind == "swipe":
            direction = str(action.params.get("direction") or "").strip()
            if direction not in {"up", "down", "left", "right"}:
                return self._deny("滑动方向无效。")
            return NavigationPolicyDecision(True, "允许一个四向导航滑动。", "swipe")
        if action_kind == "back":
            return NavigationPolicyDecision(True, "允许一个系统返回动作。", "back")
        if action_kind == "wait_for_change":
            return NavigationPolicyDecision(True, "允许等待页面变化，不产生物理动作。", "wait")

        element_id = str(action.params.get("element_id") or "").strip()
        try:
            element = scene.get_element(element_id, min_confidence=self.min_confidence)
        except UISceneError as exc:
            return self._deny(f"当前可信观察不能唯一解析候选：{exc}")
        if element.role in self.FORBIDDEN_ROLES or element.role not in self.NAVIGATION_ROLES:
            return self._deny(f"候选角色 {element.role} 不允许作为第一阶段导航点击。")
        expected_fields = {
            "target": element.meaning,
            "role": element.role,
            "label": element.label,
        }
        for field, expected in expected_fields.items():
            if str(action.params.get(field) or "") != expected:
                return self._deny(f"动作 {field} 没有逐字复用可信候选。")

        region = self._value(decision, "target_region", None)
        if region is None:
            return self._deny("点击动作缺少可信目标区域。")
        if (
            str(self._value(region, "kind", "")) != "element"
            or str(self._value(region, "element_id", "")) != element.element_id
            or tuple(self._value(region, "bounds", ())) != tuple(element.bounds)
        ):
            return self._deny("目标区域没有逐项复用可信候选 bounds。")

        conflicts = self._value(trusted_observation, "candidate_conflicts", ()) or ()
        for conflict in conflicts:
            if not isinstance(conflict, dict):
                if element.element_id in str(conflict):
                    return self._deny("当前候选存在语义冲突或不唯一。")
                continue
            conflict_ids = conflict.get("element_ids") or []
            resolved_duplicate = (
                conflict.get("kind") == "duplicate_visual_object_collapsed"
                and conflict.get("canonical_element_id") == element.element_id
                and element.element_id in conflict_ids
            )
            if resolved_duplicate:
                continue
            if element.element_id in conflict_ids or element.element_id in str(conflict):
                return self._deny("当前候选存在语义冲突或不唯一。")

        canonical = self._semantic_class(
            element.meaning,
            element.label,
            str(action.params.get("target") or ""),
        )
        if canonical == "forbidden":
            return self._deny("候选包含外部状态、输入或破坏性语义。")
        if not canonical:
            return self._deny("本地策略无法证明候选属于通用导航语义。")
        if action_kind == "dismiss_overlay" and canonical not in {"close", "back"}:
            return self._deny("关闭弹层动作只能指向关闭、取消或返回语义。")

        return NavigationPolicyDecision(
            allowed=True,
            reason="当前唯一候选通过第一阶段低风险导航策略。",
            canonical_class=canonical,
        )
