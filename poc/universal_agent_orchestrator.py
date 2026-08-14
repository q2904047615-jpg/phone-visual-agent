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
from qwen_visual_decision import QwenTaskContext, TrustedObservation
from ui_scene import MIN_TARGET_CONFIDENCE, UISceneError
from universal_action_controller import (
    action_has_account_effect,
    navigation_semantic_class,
)


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
        # DeepSeek intentionally abstracts low-level wording out of the task
        # graph. Preserve the user's original visual descriptors only inside
        # the read-only scene-observation draft so labels, colors, shapes and
        # coarse positions are not lost. The Qwen action context is still
        # produced directly from ``DynamicTaskGraph.to_qwen_context()``, so
        # this value cannot authorize or specify an action.
        if graph.raw_user_goal.strip():
            entities["original_goal_visual_context"] = graph.raw_user_goal.strip()
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
class RiskConfirmationAuthority:
    session_id: str
    task_id: str
    device_id: str
    revision: int
    subgoal_id: str
    risk_ids: tuple[str, ...]
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
    trusted_frames: tuple[Any, ...] = field(default_factory=tuple, repr=False)
    qwen_decision: Any = None
    controller_decision: NavigationPolicyDecision | None = None
    confirmation_authority: Any = field(default=None, repr=False)
    risk_confirmation_authority: Any = field(default=None, repr=False)
    confirmed_risk_ids: tuple[str, ...] = ()
    status: str = "created"
    step_number: int = 1
    physical_actions: int = 0
    automatic_loop_enabled: bool = False
    auto_pause_reason: str = ""
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
            "automatic_loop_enabled": self.automatic_loop_enabled,
            "auto_pause_reason": self.auto_pause_reason,
            "available_action_kinds": sorted(
                self.adapter.supported_action_kinds()
                if callable(getattr(self.adapter, "supported_action_kinds", None))
                else PhaseOneNavigationPolicy.ALLOWED_ACTIONS
            ),
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
            "risk_confirmation_scope": (
                self.risk_confirmation_authority.scope()
                if self.risk_confirmation_authority is not None
                and not self.risk_confirmation_authority.consumed
                else None
            ),
            "risk_confirmation_ready": bool(
                self.status == "awaiting_risk_confirmation"
                and self.risk_confirmation_authority is not None
                and not self.risk_confirmation_authority.consumed
            ),
            "confirmed_risk_ids": list(self.confirmed_risk_ids),
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
    def _available_action_kinds(
        session: UniversalAgentSessionState,
    ) -> frozenset[str]:
        provider = getattr(session.adapter, "supported_action_kinds", None)
        if not callable(provider):
            return PhaseOneNavigationPolicy.ALLOWED_ACTIONS
        actions = frozenset(str(item or "").strip() for item in provider())
        if not actions or "" in actions:
            raise UniversalAgentOrchestratorError(
                "设备动作能力为空或包含无效动作。"
            )
        unexpected = actions - PhaseOneNavigationPolicy.ALLOWED_ACTIONS
        if unexpected:
            raise UniversalAgentOrchestratorError(
                "设备报告了协议外动作：" + ", ".join(sorted(unexpected))
            )
        return actions

    def _decide_next_action(
        self,
        session: UniversalAgentSessionState,
        *,
        frames: list[Any],
        task_context: Mapping[str, Any],
        trusted_observation: Any,
    ) -> Any:
        kwargs = {
            "frames": frames,
            "task_context": task_context,
            "trusted_observation": trusted_observation,
            "decision_number": session.step_number,
            "available_action_kinds": self._available_action_kinds(session),
        }
        try:
            return self.qwen_observer.decide(**kwargs)
        except TypeError as exc:
            text = str(exc)
            if "available_action_kinds" not in text or "unexpected keyword" not in text:
                raise
            kwargs.pop("available_action_kinds")
            return self.qwen_observer.decide(**kwargs)

    @staticmethod
    def _is_presence_only_read_only_subgoal(subgoal: Any) -> bool:
        """Allow zero-action completion only for locating one visible object.

        A visible element can prove that an object exists, but it cannot prove
        its exact value, text, state, or a result produced elsewhere.  Keep the
        lexical gate deliberately narrow so ambiguous read-only work fails
        closed and can be retried with a fresh observation instead.
        """

        text = " ".join(
            str(item or "").strip()
            for item in (
                getattr(subgoal, "objective", ""),
                *tuple(getattr(subgoal, "completion_conditions", ()) or ()),
            )
            if str(item or "").strip()
        ).casefold()
        if not text:
            return False
        presence_markers = (
            "定位",
            "找到",
            "寻找",
            "识别",
            "可见",
            "存在",
            "locate",
            "find",
            "identify",
            "visible",
            "present",
            "exists",
        )
        value_verification_markers = (
            "内容",
            "文字",
            "文本",
            "数值",
            "字段值",
            "包含",
            "等于",
            "是否为",
            "状态为",
            "验证",
            "核对",
            "读取",
            "content",
            "text equals",
            "contains",
            "value",
            "verify",
            "read the",
        )
        return any(marker in text for marker in presence_markers) and not any(
            marker in text for marker in value_verification_markers
        )

    @staticmethod
    def _candidate_has_unresolved_conflict(
        trusted_observation: Any,
        element_id: str,
    ) -> bool:
        for conflict in getattr(trusted_observation, "candidate_conflicts", ()) or ():
            if not isinstance(conflict, Mapping):
                if element_id in str(conflict):
                    return True
                continue
            conflict_ids = conflict.get("element_ids") or []
            resolved_duplicate = (
                conflict.get("kind") == "duplicate_visual_object_collapsed"
                and conflict.get("canonical_element_id") == element_id
                and element_id in conflict_ids
            )
            if not resolved_duplicate and (
                element_id in conflict_ids or element_id in str(conflict)
            ):
                return True
        return False

    def _try_advance_read_only_presence_subgoal(
        self,
        session: UniversalAgentSessionState,
        *,
        graph: DynamicTaskGraph,
        trusted_observation: Any,
    ) -> DynamicTaskGraph | None:
        """Use one exact visible candidate to advance one read-only checkpoint."""

        current = graph.active_subgoal()
        if (
            current is None
            or current.external_impact != "read_only"
            or not self._is_presence_only_read_only_subgoal(current)
        ):
            return None
        scene = getattr(trusted_observation, "scene", None)
        if scene is None:
            return None
        candidate = scene.unique_trusted_goal_element(
            min_confidence=MIN_TARGET_CONFIDENCE,
        )
        if (
            candidate is None
            or candidate.states.get("fully_visible") is not True
            or self._candidate_has_unresolved_conflict(
                trusted_observation,
                candidate.element_id,
            )
        ):
            return None

        candidate_fact = (
            "当前可信画面仅有一个完整可见的目标元素："
            f"element_id={candidate.element_id}, role={candidate.role}, "
            f"label={candidate.label or '[empty]'}, meaning={candidate.meaning}, "
            f"confidence={float(candidate.confidence):.3f}, fully_visible=true。"
        )
        observed = self.bridge.observed_state(
            graph=graph,
            trusted_observation=trusted_observation,
            action_outcome="not_applicable",
            verification={"visible_evidence": [candidate_fact, *candidate.evidence]},
        )
        revised = self.deepseek_planner.replan(
            graph,
            observed,
            trigger="subgoal_completed",
            reason=(
                "当前可信画面已经以唯一、高置信、完整可见的目标元素证明"
                "定位类 read_only 子目标；只允许推进这一个子目标，不得推断"
                "元素值、外部状态或执行动作。"
            ),
        )
        self._validate_graph_identity(
            revised,
            device_id=session.device_id,
            previous=graph,
        )
        if revised.revision != graph.revision + 1:
            raise UniversalAgentOrchestratorError(
                "只读证据推进必须且只能产生一个新 revision。"
            )
        old_ids = tuple(item.subgoal_id for item in graph.subgoals)
        new_ids = tuple(item.subgoal_id for item in revised.subgoals)
        if old_ids != new_ids:
            raise UniversalAgentOrchestratorError(
                "只读证据推进不得增加、删除或重排子目标。"
            )
        old_by_id = {item.subgoal_id: item for item in graph.subgoals}
        new_by_id = {item.subgoal_id: item for item in revised.subgoals}
        if (
            revised.goal != graph.goal
            or revised.constraints != graph.constraints
            or revised.completion_conditions != graph.completion_conditions
            or revised.risk_actions != graph.risk_actions
        ):
            raise UniversalAgentOrchestratorError(
                "只读证据推进不得修改目标、约束、全局完成条件或风险定义。"
            )
        for subgoal_id in old_ids:
            old_item = old_by_id[subgoal_id]
            new_item = new_by_id[subgoal_id]
            if (
                new_item.objective != old_item.objective
                or new_item.depends_on != old_item.depends_on
                or new_item.constraints != old_item.constraints
                or new_item.completion_conditions != old_item.completion_conditions
                or new_item.risk_action_ids != old_item.risk_action_ids
                or new_item.external_impact != old_item.external_impact
            ):
                raise UniversalAgentOrchestratorError(
                    "只读证据推进只能改变子目标状态和完成证据。"
                )
        newly_completed = tuple(
            subgoal_id
            for subgoal_id in old_ids
            if old_by_id[subgoal_id].status != "completed"
            and new_by_id[subgoal_id].status == "completed"
        )
        completed_current = new_by_id[current.subgoal_id]
        if (
            newly_completed != (current.subgoal_id,)
            or not completed_current.completion_evidence
        ):
            raise UniversalAgentOrchestratorError(
                "只读证据只能完成当前定位子目标，且必须记录可见证据。"
            )
        for subgoal_id in old_ids:
            old_status = old_by_id[subgoal_id].status
            new_status = new_by_id[subgoal_id].status
            if subgoal_id == current.subgoal_id:
                continue
            if old_status == "completed" and new_status != "completed":
                raise UniversalAgentOrchestratorError(
                    "只读证据推进不得回退已完成子目标。"
                )
            if old_status == "pending" and new_status not in {"pending", "active"}:
                raise UniversalAgentOrchestratorError(
                    "只读证据推进不得越过后续子目标。"
                )
        newly_active = tuple(
            subgoal_id
            for subgoal_id in old_ids
            if old_by_id[subgoal_id].status == "pending"
            and new_by_id[subgoal_id].status == "active"
        )
        if len(newly_active) > 1:
            raise UniversalAgentOrchestratorError(
                "只读证据推进最多只能激活一个后续子目标。"
            )
        if revised.status != "completed" and (
            len(newly_active) != 1
            or revised.active_subgoal_id != newly_active[0]
        ):
            raise UniversalAgentOrchestratorError(
                "只读证据推进后必须精确激活一个后续子目标。"
            )
        return revised

    def _store_revised_graph(
        self,
        session: UniversalAgentSessionState,
        revised: DynamicTaskGraph,
    ) -> None:
        session.task_graph = revised
        session.goal_draft = self.bridge.goal_draft(revised)
        session.confirmation_authority = None
        session.risk_confirmation_authority = None
        session.confirmed_risk_ids = ()
        self._remember(
            session,
            session.evidence_store.write_task_graph(revised),
            session.evidence_store.write_risk_audit(revised),
        )

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

    def _bind_risk_confirmation(self, session: UniversalAgentSessionState) -> None:
        graph = session.task_graph
        if graph is None:
            raise UniversalAgentOrchestratorError("风险确认缺少任务图。")
        current = graph.active_subgoal()
        if current is None or not current.risk_action_ids:
            raise UniversalAgentOrchestratorError("当前子目标没有可确认风险。")
        session.risk_confirmation_authority = RiskConfirmationAuthority(
            session_id=session.session_id,
            task_id=graph.task_id,
            device_id=graph.device_id,
            revision=graph.revision,
            subgoal_id=current.subgoal_id,
            risk_ids=tuple(current.risk_action_ids),
        )

    @staticmethod
    def _normalize_risk_confirmation(value: Mapping[str, Any]) -> dict[str, Any]:
        required = {
            "session_id",
            "task_id",
            "device_id",
            "revision",
            "subgoal_id",
            "risk_ids",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise UniversalAgentOrchestratorError(
                "风险确认作用域字段缺失或包含额外字段。"
            )
        risk_ids = value.get("risk_ids")
        revision = value.get("revision")
        if not isinstance(risk_ids, list):
            raise UniversalAgentOrchestratorError("风险确认 risk_ids 必须是数组。")
        if isinstance(revision, bool) or not isinstance(revision, int):
            raise UniversalAgentOrchestratorError("风险确认 revision 格式无效。")
        return {
            "session_id": str(value.get("session_id") or ""),
            "task_id": str(value.get("task_id") or ""),
            "device_id": str(value.get("device_id") or ""),
            "revision": revision,
            "subgoal_id": str(value.get("subgoal_id") or ""),
            "risk_ids": sorted(str(item) for item in risk_ids),
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
        action_outcome = str(getattr(result, "action_outcome", "matched"))
        matched = action_outcome == "matched"
        verification_errors = tuple(
            str(item)
            for item in getattr(result, "verification_errors", ())
            if str(item).strip()
        )
        verification = {
            "matched": matched,
            "action_outcome": action_outcome,
            "physical_actions": result.physical_actions,
            "before_fingerprint": result.before_scene.fingerprint,
            "after_fingerprint": result.after_scene.fingerprint,
            "visible_evidence": [result.after_scene.summary],
            "blocked_reasons": list(verification_errors),
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
            action_outcome=action_outcome,
            verification=verification,
        )
        try:
            revised = self.deepseek_planner.replan(
                previous_graph,
                observed,
                trigger=(
                    "observation_changed" if matched else "action_result_mismatch"
                ),
                reason=(
                    "一个动作已经执行并由新的可信画面验证。"
                    if matched
                    else "动作已执行，但新画面没有证明预期语义变化，必须重规划。"
                ),
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
        session.confirmation_authority = None
        session.risk_confirmation_authority = None
        session.confirmed_risk_ids = ()
        self._remember(
            session,
            session.evidence_store.write_task_graph(revised),
            session.evidence_store.write_risk_audit(revised),
        )
        if revised.status == "completed":
            session.status = "succeeded"
            return
        current = revised.active_subgoal()
        impact = current.external_impact if current is not None else "unknown"
        if current is None:
            session.status = "blocked"
            session.failed_reason = "重规划后的任务图没有活动子目标。"
            return
        if impact in {"external_state", "unknown"}:
            session.status = "awaiting_risk_confirmation"
            session.failed_reason = ""
            self._bind_risk_confirmation(session)
            return
        if impact == "read_only":
            try:
                reviewed = self.deepseek_planner.replan(
                    revised,
                    observed,
                    trigger="subgoal_completed",
                    reason=(
                        "当前 read_only 子目标只能用已经采集的当前可信画面"
                        "完成或阻塞；不得请求任何新的物理动作。"
                    ),
                )
                self._validate_graph_identity(
                    reviewed,
                    device_id=session.device_id,
                    previous=revised,
                )
                session.task_graph = reviewed
                session.goal_draft = self.bridge.goal_draft(reviewed)
                self._remember(
                    session,
                    session.evidence_store.write_task_graph(reviewed),
                    session.evidence_store.write_risk_audit(reviewed),
                )
            except Exception as exc:
                session.status = "blocked"
                session.failed_reason = f"只读完成复核失败：{exc}"
                return
            if reviewed.status == "completed":
                session.status = "succeeded"
                session.failed_reason = ""
                return
            reviewed_current = reviewed.active_subgoal()
            reviewed_impact = (
                reviewed_current.external_impact
                if reviewed_current is not None
                else "unknown"
            )
            if reviewed_current is None:
                session.status = "blocked"
                session.failed_reason = "只读复核后的任务图没有活动子目标。"
                return
            if reviewed_impact in {"external_state", "unknown"}:
                session.status = "awaiting_risk_confirmation"
                session.failed_reason = ""
                self._bind_risk_confirmation(session)
                return
            if reviewed_impact == "read_only":
                session.status = "blocked"
                session.failed_reason = (
                    "当前可信画面没有让 DeepSeek 完成 read_only 子目标；"
                    "禁止为只读验证请求物理动作。"
                )
                return
            # A read-only checkpoint may be completed while the overall task still
            # has a later navigation-only subgoal. Reuse the same trusted frames;
            # do not capture again and do not execute anything without a new scope.
            revised = reviewed

        frames = list(result.after_frames)
        context = revised.to_qwen_context()
        decision = self._decide_next_action(
            session,
            frames=frames,
            task_context=context,
            trusted_observation=new_observation,
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
        if decision.proposal.status == "finished":
            completion_observed = self.bridge.observed_state(
                graph=revised,
                trusted_observation=new_observation,
                action_outcome="not_applicable",
                verification={
                    "completion_evidence": list(
                        decision.proposal.completion_evidence
                    ),
                    "visible_evidence": list(
                        decision.proposal.completion_evidence
                    ),
                },
            )
            try:
                completed = self.deepseek_planner.replan(
                    revised,
                    completion_observed,
                    trigger="subgoal_completed",
                    reason=(
                        "Qwen 在动作后的当前可信画面中提出完成候选，"
                        "要求 DeepSeek 复核整个任务。"
                    ),
                )
                self._validate_graph_identity(
                    completed,
                    device_id=session.device_id,
                    previous=revised,
                )
                session.task_graph = completed
                session.goal_draft = self.bridge.goal_draft(completed)
                self._remember(
                    session,
                    session.evidence_store.write_task_graph(completed),
                    session.evidence_store.write_risk_audit(completed),
                )
            except Exception as exc:
                session.status = "blocked"
                session.failed_reason = f"完成候选复核失败：{exc}"
                session.controller_decision = NavigationPolicyDecision(
                    allowed=False,
                    reason=session.failed_reason,
                )
                return
            if completed.status == "completed":
                session.status = "succeeded"
                session.failed_reason = ""
            else:
                session.status = "blocked"
                session.failed_reason = (
                    "Qwen 的完成候选没有被 DeepSeek 新 revision 确认为完成。"
                )
                session.controller_decision = NavigationPolicyDecision(
                    allowed=False,
                    reason=session.failed_reason,
                )
            return
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
            task_context=QwenTaskContext.from_dict(context),
            trusted_observation=new_observation,
            decision=decision,
            available_action_kinds=self._available_action_kinds(session),
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
        if session.status == "awaiting_risk_confirmation":
            raise UniversalAgentOrchestratorError(
                "当前子目标必须先确认风险范围，禁止提前调用 Qwen。"
            )
        if current is None or (
            impact in {"external_state", "unknown"}
            and not session.confirmed_risk_ids
        ):
            raise UniversalAgentOrchestratorError(
                f"当前 {impact} 子目标缺少有效风险确认。"
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
            session.trusted_frames = tuple(frames)
            self._remember(
                session,
                session.evidence_store.write_trusted_observation(
                    session.step_number,
                    observation,
                ),
            )

            if session.confirmed_risk_ids:
                context = graph.to_qwen_context(
                    confirmed_risk_ids=session.confirmed_risk_ids,
                    confirmed_task_id=graph.task_id,
                    confirmed_device_id=graph.device_id,
                    confirmed_subgoal_id=graph.active_subgoal_id,
                    confirmed_revision=graph.revision,
                )
            else:
                context = graph.to_qwen_context()
            decision = self._decide_next_action(
                session,
                frames=frames,
                task_context=context,
                trusted_observation=observation,
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
                    task_context=QwenTaskContext.from_dict(context),
                    trusted_observation=observation,
                    decision=decision,
                    available_action_kinds=self._available_action_kinds(session),
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

        if session.confirmed_risk_ids:
            context = graph.to_qwen_context(
                confirmed_risk_ids=session.confirmed_risk_ids,
                confirmed_task_id=graph.task_id,
                confirmed_device_id=graph.device_id,
                confirmed_subgoal_id=graph.active_subgoal_id,
                confirmed_revision=graph.revision,
            )
        else:
            context = graph.to_qwen_context()
        policy_decision = self.policy.evaluate(
            task_context=QwenTaskContext.from_dict(context),
            trusted_observation=observation,
            decision=decision,
            available_action_kinds=self._available_action_kinds(session),
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
                planned_frames=session.trusted_frames,
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
            getattr(result, "action_outcome", "matched") == "matched"
            and
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
                getattr(result, "action_outcome", "matched") == "matched"
                and
                result.resolved_action.kind != "wait_for_change"
                and new_observation.fingerprint == observation.fingerprint
            )
        ):
            session.status = "failed"
            session.failed_reason = "动作后可信观察 observation/fingerprint 未更新。"
            raise UniversalAgentOrchestratorError(session.failed_reason)
        session.trusted_observation = new_observation
        session.trusted_frames = tuple(result.after_frames)
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

    def approve_risks(
        self,
        session: UniversalAgentSessionState,
        confirmation: Mapping[str, Any],
    ) -> Any:
        """Consume one graph-bound risk approval, then observe without acting."""

        if self.device_registry.active_session(session.device_id) != session.session_id:
            raise UniversalAgentOrchestratorError(
                "当前会话已不再拥有该设备，禁止确认风险。"
            )
        try:
            with self.device_registry.device_lock(session.device_id):
                if session.status != "awaiting_risk_confirmation":
                    raise UniversalAgentOrchestratorError(
                        f"当前状态不能确认风险：{session.status}。"
                    )
                authority = session.risk_confirmation_authority
                if authority is None or authority.consumed:
                    raise UniversalAgentOrchestratorError("当前风险确认已失效或已使用。")
                requested = self._normalize_risk_confirmation(confirmation)
                if requested != authority.scope():
                    authority.consumed = True
                    authority.invalid_reason = "risk_scope_mismatch"
                    raise UniversalAgentOrchestratorError(
                        "风险确认与当前 task/device/revision/subgoal/risk 不一致。"
                    )
                authority.consumed = True
                authority.invalid_reason = "consumed_before_observation"
                session.confirmed_risk_ids = tuple(authority.risk_ids)
                graph = session.task_graph
                assert graph is not None
                context = graph.to_qwen_context(
                    confirmed_risk_ids=session.confirmed_risk_ids,
                    confirmed_task_id=graph.task_id,
                    confirmed_device_id=graph.device_id,
                    confirmed_subgoal_id=graph.active_subgoal_id,
                    confirmed_revision=graph.revision,
                )
                result = self._observe_after_risk_confirmation(session, context)
                self._write_terminal_snapshot(session)
                return result
        finally:
            self._release_if_terminal(session)

    def run_safe_loop(
        self,
        session: UniversalAgentSessionState,
        confirmation: Mapping[str, Any],
        *,
        max_physical_actions: int = 1,
        max_iterations: int = 1,
    ) -> dict[str, Any]:
        """Compatibility entry point for exactly one scoped confirmation."""

        if self.device_registry.active_session(session.device_id) != session.session_id:
            raise UniversalAgentOrchestratorError(
                "当前会话已不再拥有该设备，禁止自动推进。"
            )
        if isinstance(max_physical_actions, bool) or not isinstance(
            max_physical_actions, int
        ):
            raise UniversalAgentOrchestratorError("自动推进动作上限格式无效。")
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
            raise UniversalAgentOrchestratorError("自动推进迭代上限格式无效。")
        if max_physical_actions != 1:
            raise UniversalAgentOrchestratorError(
                "每次确认最多执行一个物理动作。"
            )
        if max_iterations != 1:
            raise UniversalAgentOrchestratorError(
                "每次确认最多处理一个动作轮次。"
            )

        try:
            with self.device_registry.device_lock(session.device_id):
                return self._run_safe_loop_locked(
                    session,
                    confirmation,
                    max_physical_actions=max_physical_actions,
                    max_iterations=max_iterations,
                )
        finally:
            self._release_if_terminal(session)

    def _run_safe_loop_locked(
        self,
        session: UniversalAgentSessionState,
        confirmation: Mapping[str, Any],
        *,
        max_physical_actions: int,
        max_iterations: int,
    ) -> dict[str, Any]:
        start_actions = session.physical_actions
        if session.status != "awaiting_confirmation":
            raise UniversalAgentOrchestratorError(
                f"当前状态不能启动安全自动推进：{session.status}。"
            )
        session.automatic_loop_enabled = False
        session.auto_pause_reason = ""
        try:
            result = self._confirm_one_locked(session, dict(confirmation))
            if getattr(result, "action_outcome", "matched") != "matched":
                session.auto_pause_reason = (
                    "动作后没有出现预期语义变化，已停止并完成重规划。"
                )
            else:
                session.auto_pause_reason = {
                    "awaiting_confirmation": "已执行一个已确认动作；下一动作需要重新确认。",
                    "awaiting_risk_confirmation": "下一子目标需要单独确认风险范围。",
                    "succeeded": "目标已由新画面和 DeepSeek revision 证明完成。",
                    "blocked": "当前视觉决策或本地策略已阻止继续。",
                    "failed": "当前动作或验证失败，禁止自动重试。",
                }.get(session.status, f"会话状态 {session.status} 不允许继续。")
        finally:
            session.automatic_loop_enabled = False
            self._write_terminal_snapshot(session)

        return {
            "physical_actions": session.physical_actions - start_actions,
            "iterations": 1,
            "status": session.status,
            "pause_reason": session.auto_pause_reason,
        }

    def _observe_after_risk_confirmation(
        self,
        session: UniversalAgentSessionState,
        task_context: Mapping[str, Any],
    ) -> Any:
        graph = session.task_graph
        assert graph is not None and session.goal_draft is not None
        before_actions = session.physical_actions
        session.status = "observing"
        scene, frames, frame_paths = session.adapter.capture_scene(
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
        session.trusted_frames = tuple(frames)
        self._remember(
            session,
            session.evidence_store.write_trusted_observation(
                session.step_number,
                observation,
            ),
        )
        decision = self._decide_next_action(
            session,
            frames=frames,
            task_context=task_context,
            trusted_observation=observation,
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
                task_context=QwenTaskContext.from_dict(dict(task_context)),
                trusted_observation=observation,
                decision=decision,
                available_action_kinds=self._available_action_kinds(session),
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
                else "外部状态目标的完成候选必须由 DeepSeek 新 revision 复核。"
            )
        if session.physical_actions != before_actions:
            raise UniversalAgentOrchestratorError("风险确认路径错误地触发了物理动作。")
        return decision

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
            if current is None:
                session.status = "blocked"
                session.failed_reason = "任务图没有活动子目标。"
                self._write_terminal_snapshot(session)
                return session
            if impact in {"external_state", "unknown"}:
                session.status = "awaiting_risk_confirmation"
                session.failed_reason = ""
                session.confirmed_risk_ids = ()
                session.confirmation_authority = None
                self._bind_risk_confirmation(session)
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
            session.trusted_frames = tuple(frames)
            self._remember(
                session,
                store.write_trusted_observation(session.step_number, observation),
            )

            if impact == "read_only":
                revised = self._try_advance_read_only_presence_subgoal(
                    session,
                    graph=graph,
                    trusted_observation=observation,
                )
                if revised is None:
                    session.status = "blocked"
                    session.failed_reason = (
                        "当前 read_only 子目标不是可由唯一完整可见元素证明的"
                        "定位目标，或当前画面证据不唯一；未请求物理动作。"
                    )
                    self._write_terminal_snapshot(session)
                    return session
                self._store_revised_graph(session, revised)
                graph = revised
                if revised.status == "completed":
                    session.status = "succeeded"
                    session.failed_reason = ""
                    self._write_terminal_snapshot(session)
                    return session
                current = revised.active_subgoal()
                impact = current.external_impact if current is not None else "unknown"
                if current is None:
                    session.status = "blocked"
                    session.failed_reason = "只读证据推进后没有活动子目标。"
                    self._write_terminal_snapshot(session)
                    return session
                if impact in {"external_state", "unknown"}:
                    session.status = "awaiting_risk_confirmation"
                    session.failed_reason = ""
                    self._bind_risk_confirmation(session)
                    self._write_terminal_snapshot(session)
                    return session
                if impact == "read_only":
                    session.status = "blocked"
                    session.failed_reason = (
                        "同一可信画面最多推进一个 read_only 子目标；"
                        "必须重新观察后再继续。"
                    )
                    self._write_terminal_snapshot(session)
                    return session

            task_context = graph.to_qwen_context()
            decision = self._decide_next_action(
                session,
                frames=frames,
                task_context=task_context,
                trusted_observation=observation,
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
                    task_context=QwenTaskContext.from_dict(task_context),
                    trusted_observation=observation,
                    decision=decision,
                    available_action_kinds=self._available_action_kinds(session),
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
        try:
            with self.device_registry.device_lock(session.device_id):
                for authority in (
                    session.confirmation_authority,
                    session.risk_confirmation_authority,
                ):
                    if authority is not None:
                        authority.consumed = True
                        authority.invalid_reason = "paused"
                session.confirmed_risk_ids = ()
                session.status = "paused"
                session.failed_reason = "用户已暂停；旧确认和旧观察不可复用。"
                self._write_terminal_snapshot(session)
        finally:
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
            for authority in (
                session.confirmation_authority,
                session.risk_confirmation_authority,
            ):
                if authority is not None:
                    authority.consumed = True
                    authority.invalid_reason = str(reason or "invalidated")
            session.confirmation_authority = None
            session.risk_confirmation_authority = None
            session.confirmed_risk_ids = ()
            session.status = "needs_reobservation"
            session.failed_reason = (
                "设备或摄像头状态变化；旧确认已失效，必须重新观察后再确认。"
            )
            self._write_terminal_snapshot(session)

    def cancel(self, session: UniversalAgentSessionState) -> None:
        try:
            with self.device_registry.device_lock(session.device_id):
                for authority in (
                    session.confirmation_authority,
                    session.risk_confirmation_authority,
                ):
                    if authority is not None:
                        authority.consumed = True
                        authority.invalid_reason = "cancelled"
                session.confirmed_risk_ids = ()
                session.status = "cancelled"
                session.failed_reason = "用户已取消任务。"
                self._write_terminal_snapshot(session)
        finally:
            self.device_registry.release(session.device_id, session.session_id)


@dataclass(frozen=True)
class NavigationPolicyDecision:
    allowed: bool
    reason: str
    canonical_class: str = ""


class PhaseOneNavigationPolicy:
    """Fail-closed gate for one generic visual action.

    This class classifies one already proposed visual action.  It never plans
    a task, chooses an App, invents an element, or changes coordinates.
    """

    VERSION = "2026-08-13-universal-action-policy-v3"
    ALLOWED_ACTIONS = frozenset(
        {
            "swipe",
            "back",
            "home",
            "wait_for_change",
            "tap_semantic",
            "dismiss_overlay",
            "input_verified_text",
            "long_press",
            "drag",
        }
    )
    FORBIDDEN_ROLES = frozenset({"keyboard_key"})
    NAVIGATION_ROLES = frozenset(
        {"button", "icon", "text", "tab", "image", "list_item"}
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
        return navigation_semantic_class(*values)

    def _matches_target_app(self, task_context: Any, element: Any) -> bool:
        """Bind a visible App entry to the formal task target without App rules."""

        goal = self._value(task_context, "goal", {})
        target_apps = self._value(goal, "target_apps", ()) or ()
        candidate_tokens = self._tokens(
            " ".join(
                (
                    str(self._value(element, "meaning", "")),
                    str(self._value(element, "label", "")),
                )
            )
        )
        candidate_label = str(self._value(element, "label", "")).strip().casefold()
        for target_app in target_apps:
            app_id_tokens = self._tokens(
                str(self._value(target_app, "app_id", ""))
            )
            if app_id_tokens and app_id_tokens.issubset(candidate_tokens):
                return True
            app_name = str(self._value(target_app, "app_name", "")).strip().casefold()
            if app_name and candidate_label == app_name:
                return True
        return False

    def evaluate(
        self,
        *,
        task_context: Any,
        trusted_observation: Any,
        decision: Any,
        available_action_kinds: frozenset[str] | None = None,
    ) -> NavigationPolicyDecision:
        impact = str(
            self._value(task_context, "current_external_impact", "unknown")
        ).strip()
        if impact == "unknown":
            return self._deny("unknown 子目标禁止进入视觉或机械臂执行。")
        external_allowed = bool(
            self._value(task_context, "external_action_allowed", False)
        )
        if impact == "external_state" and not external_allowed:
            return self._deny("external_state 子目标缺少当前作用域确认。")

        proposal = self._value(decision, "proposal", None)
        if proposal is None or str(self._value(proposal, "status", "")) != "action":
            return self._deny("当前 Qwen 决策没有唯一可执行动作。")
        action = self._value(proposal, "action", None)
        if action is None:
            return self._deny("当前 Qwen 决策缺少动作。")
        action_kind = str(self._value(action, "action", "")).strip()
        if action_kind not in self.ALLOWED_ACTIONS:
            return self._deny(f"通用策略不允许动作：{action_kind or 'missing'}。")
        if (
            available_action_kinds is not None
            and action_kind not in available_action_kinds
        ):
            return self._deny(f"当前设备没有本地验证动作能力：{action_kind}。")
        if action_kind == "wait_for_change":
            if impact not in {"read_only", "navigation_only"}:
                return self._deny(f"等待动作不能用于 {impact} 子目标。")
        elif impact not in {"navigation_only", "external_state"}:
            return self._deny(f"物理动作不能用于 {impact} 子目标。")

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
        if not scene.stable:
            return self._deny("页面不稳定或整体置信度不足。")
        if float(scene.confidence) < self.min_confidence:
            local_candidate = trusted_observation.target_local_candidate()
            action_element_id = str(action.params.get("element_id") or "").strip()
            if (
                action_kind not in {
                    "tap_semantic",
                    "dismiss_overlay",
                    "input_verified_text",
                    "long_press",
                }
                or local_candidate is None
                or local_candidate.element_id != action_element_id
            ):
                return self._deny("页面整体置信度不足，且没有唯一可信的目标局部证据。")
        if scene.fingerprint != observation_fingerprint:
            return self._deny("页面 fingerprint 与可信观察不一致。")
        if float(self._value(decision, "confidence", 0.0)) < self.min_confidence:
            return self._deny("Qwen 决策置信度不足。")

        if action_has_account_effect(action) and not external_allowed:
            return self._deny("动作语义可能改变账号或外部状态。")

        if action_kind == "swipe":
            direction = str(action.params.get("direction") or "").strip()
            if direction not in {"up", "down", "left", "right"}:
                return self._deny("滑动方向无效。")
            return NavigationPolicyDecision(True, "允许一个四向导航滑动。", "swipe")
        if action_kind == "back":
            return NavigationPolicyDecision(True, "允许一个系统返回动作。", "back")
        if action_kind == "home":
            return NavigationPolicyDecision(
                True,
                "允许一个Android系统Home动作，返回系统Launcher。",
                "home",
            )
        if action_kind == "wait_for_change":
            return NavigationPolicyDecision(True, "允许等待页面变化，不产生物理动作。", "wait")

        if action_kind == "drag":
            source_id = str(action.params.get("source_element_id") or "").strip()
            destination_id = str(
                action.params.get("destination_element_id") or ""
            ).strip()
            try:
                source = scene.get_element(source_id, min_confidence=self.min_confidence)
                destination = scene.get_element(
                    destination_id,
                    min_confidence=self.min_confidence,
                )
            except UISceneError as exc:
                return self._deny(f"拖动端点不能由可信观察唯一解析：{exc}")
            if source.element_id == destination.element_id:
                return self._deny("拖动起点和终点不能相同。")
            for prefix, element in (
                ("source_", source),
                ("destination_", destination),
            ):
                for field, expected in {
                    "target": element.meaning,
                    "role": element.role,
                    "label": element.label,
                }.items():
                    if str(action.params.get(f"{prefix}{field}") or "") != expected:
                        return self._deny(f"拖动动作没有逐字复用 {prefix}{field}。")
            region = self._value(decision, "target_region", None)
            if (
                region is None
                or str(self._value(region, "kind", "")) != "element_path"
                or str(self._value(region, "element_id", "")) != source.element_id
                or tuple(self._value(region, "bounds", ())) != tuple(source.bounds)
                or str(self._value(region, "destination_element_id", ""))
                != destination.element_id
                or tuple(self._value(region, "destination_bounds", ()))
                != tuple(destination.bounds)
            ):
                return self._deny("拖动路径没有逐项复用两端可信候选 bounds。")
            if any(
                self._semantic_class(element.meaning, element.label) == "forbidden"
                for element in (source, destination)
            ) and not external_allowed:
                return self._deny("拖动端点包含外部状态、输入或破坏性语义。")
            return NavigationPolicyDecision(True, "允许一个双候选语义拖动。", "drag")

        element_id = str(action.params.get("element_id") or "").strip()
        try:
            element = scene.get_element(element_id, min_confidence=self.min_confidence)
        except UISceneError as exc:
            return self._deny(f"当前可信观察不能唯一解析候选：{exc}")
        if element.role in self.FORBIDDEN_ROLES:
            return self._deny(f"候选角色 {element.role} 不允许进入通用动作。")
        if action_kind == "input_verified_text":
            if element.role != "input" or element.states.get("focused") is not True:
                return self._deny("输入动作要求最新画面证明 input 候选已聚焦。")
            text = action.params.get("text")
            if (
                not isinstance(text, str)
                or not text
                or len(text) > 100
                or "\n" in text
                or "\r" in text
            ):
                return self._deny("输入文字格式无效。")
        elif (
            action_kind == "tap_semantic"
            and element.role == "input"
            and impact == "navigation_only"
        ):
            # Focusing one exact visible input changes only the local UI state.
            # Text entry remains a separate, newly observed and confirmed
            # input_verified_text action.
            pass
        elif element.role not in self.NAVIGATION_ROLES and not (
            impact == "external_state" and external_allowed and element.role == "toggle"
        ):
            return self._deny(f"候选角色 {element.role} 不允许用于当前动作。")
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

        if action_kind == "tap_semantic" and element.role == "input":
            return NavigationPolicyDecision(
                True,
                "允许对一个精确可信输入候选执行本地聚焦。",
                "focus_input",
            )
        canonical = self._semantic_class(
            element.meaning,
            element.label,
            str(action.params.get("target") or ""),
        )
        if canonical == "forbidden":
            if impact == "external_state" and external_allowed:
                canonical = "external"
            elif action_kind == "input_verified_text":
                canonical = "input"
            else:
                return self._deny("候选包含外部状态、输入或破坏性语义。")
        if not canonical:
            if action_kind == "input_verified_text":
                canonical = "input"
            elif action_kind == "long_press":
                canonical = "long_press"
            elif (
                action_kind == "tap_semantic"
                and impact == "navigation_only"
                and element.role in {"button", "icon", "image", "list_item"}
                and self._matches_target_app(task_context, element)
            ):
                canonical = "open"
            elif impact == "external_state" and external_allowed:
                canonical = "external"
            else:
                return self._deny("本地策略无法证明候选属于通用导航语义或动作语义。")
        if action_kind == "dismiss_overlay" and canonical not in {"close", "back"}:
            return self._deny("关闭弹层动作只能指向关闭、取消或返回语义。")

        return NavigationPolicyDecision(
            allowed=True,
            reason="当前唯一候选通过第一阶段低风险导航策略。",
            canonical_class=canonical,
        )
