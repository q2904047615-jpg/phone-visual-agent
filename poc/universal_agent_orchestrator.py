from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import threading
from types import SimpleNamespace
from typing import Any, Callable
import uuid

from deepseek_task_graph import (
    ControllerTransitionEvidenceRef,
    TaskGraphError,
    DynamicTaskGraph,
    ObservedState,
    VerifiedActionTransition,
    named_visual_identity_is_grounded,
)
from deepseek_failure_diagnostics import persist_deepseek_failure_diagnostic
from device_exclusivity import InterProcessLease
from generic_action_adapter import GenericActionAdapterError
from generic_intent import GenericIntentDraft
from generic_step_planner import GenericStepProposal
from message_intent import CanonicalMessageIntent, MessageIntentError
from qwen_visual_decision import QwenTaskContext, TrustedObservation
from ui_scene import (
    MIN_TARGET_CONFIDENCE,
    UISceneError,
    compact_drag_source_container_error,
)
from universal_action_controller import (
    action_has_account_effect,
    navigation_semantic_class,
)
from verified_text_transaction import (
    VerifiedTextTransactionError,
    plan_next_verified_input,
)
from constraint_target_filter import constraint_excludes_candidate


POST_ACTION_TRANSITION_PROTOCOL_VERSION = (
    "2026-08-16-universal-post-action-transition-v1"
)
POST_ACTION_OUTCOMES = frozenset({"matched", "mismatched"})
MAX_VISIBLE_PRESENCE_ADVANCES_PER_OBSERVATION = 4
_TRANSIENT_ACTION_KEYS = frozenset(
    {
        "node_id",
        "element_id",
        "source_element_id",
        "destination_element_id",
        "bounds",
        "source_bounds",
        "destination_bounds",
        "point",
        "normalized_point",
        "before_fingerprint",
        "observation_id",
        "fingerprint",
    }
)


def _stable_action_payload(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _stable_action_payload(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in _TRANSIENT_ACTION_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_stable_action_payload(item) for item in value]
    return value


def _action_digest(action: Any) -> str:
    if action is None:
        raise UniversalAgentOrchestratorError("动作摘要缺少语义动作。")
    payload = action.to_dict() if callable(getattr(action, "to_dict", None)) else action
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _action_equivalence_digest(action: Any) -> str:
    if action is None:
        raise UniversalAgentOrchestratorError("动作等价摘要缺少语义动作。")
    payload = action.to_dict() if callable(getattr(action, "to_dict", None)) else action
    canonical = json.dumps(
        _stable_action_payload(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _subgoal_progress_signature(subgoal: Any) -> str:
    if subgoal is None:
        return ""
    payload = {
        "subgoal_id": str(getattr(subgoal, "subgoal_id", "")),
        "objective": str(getattr(subgoal, "objective", "")),
        "completion_conditions": list(
            getattr(subgoal, "completion_conditions", ())
        ),
        "external_impact": str(getattr(subgoal, "external_impact", "")),
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _risk_intent_material(
    graph: DynamicTaskGraph,
    current: Any,
) -> tuple[str, dict[str, Any]]:
    risk_ids = tuple(sorted(str(item) for item in current.risk_action_ids))
    risks = [
        risk
        for risk in graph.risk_actions
        if risk.risk_id in set(risk_ids)
    ]
    if {risk.risk_id for risk in risks} != set(risk_ids):
        raise UniversalAgentOrchestratorError(
            "风险确认引用了任务图中不存在的风险。"
        )
    message_risk = any(
        risk.risk_type == "message_or_communication" for risk in risks
    )
    preview: dict[str, Any] = {
        "kind": "external_state",
        "risk_ids": list(risk_ids),
    }
    if message_risk:
        try:
            intent = CanonicalMessageIntent.from_goal(
                target_apps=graph.goal.target_apps,
                entities=graph.goal.entities,
            )
        except MessageIntentError as exc:
            raise UniversalAgentOrchestratorError(
                f"消息发送风险缺少可逐字确认的收件人或消息原文：{exc}"
            ) from exc
        preview = {"kind": "message_or_communication", **intent.preview()}
    payload = {
        "protocol_version": "2026-08-18-risk-intent-v1",
        "task_id": graph.task_id,
        "device_id": graph.device_id,
        "revision": graph.revision,
        "subgoal_id": current.subgoal_id,
        "risk_ids": list(risk_ids),
        "risks": [
            {
                "risk_id": risk.risk_id,
                "description": risk.description,
                "external_effect": risk.external_effect,
                "risk_type": risk.risk_type,
                "risk_level": risk.risk_level,
                "subgoal_ids": list(risk.subgoal_ids),
            }
            for risk in risks
        ],
        "goal": {
            "target_apps": [
                {"app_id": app.app_id, "app_name": app.app_name}
                for app in graph.goal.target_apps
            ],
            "entities": graph.goal.entities,
        },
        "preview": preview,
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest(), preview


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

    def write_post_action_transition(
        self,
        step_number: int,
        transition: Any,
    ) -> Path:
        return self.write_json(
            f"post_action_transition_step_{int(step_number)}.json",
            transition,
        )

    def write_confirmation_failure(
        self,
        step_number: int,
        transition: Any,
    ) -> Path:
        return self.write_json(
            f"confirmation_failure_step_{int(step_number)}.json",
            transition,
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
        if active is not None:
            entities["active_subgoal_visual_context"] = {
                "subgoal_id": active.subgoal_id,
                "objective": active.objective,
                "constraints": list(active.constraints),
                "completion_conditions": list(active.completion_conditions),
                "external_impact": active.external_impact,
                "goal_entities": dict(graph.goal.entities),
            }
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
        verified_action_transition: VerifiedActionTransition | None = None,
        controller_transition_evidence_refs: tuple[
            ControllerTransitionEvidenceRef, ...
        ] = (),
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
        grounded_visual_facts: list[str] = []

        def add(items: Any) -> None:
            for item in self._text_items(items):
                if item not in evidence:
                    evidence.append(item)

        def add_grounded(item: str) -> None:
            value = str(item or "").strip()
            if value and value not in grounded_visual_facts:
                grounded_visual_facts.append(value)

        if scene.summary.strip():
            add((scene.summary.strip(),))
        add(scene.overlays)
        add_grounded(
            json.dumps(
                {
                    "app_id": scene.app_id,
                    "screen_id": scene.screen_id,
                    "overlays": list(scene.overlays),
                    "stable": scene.stable,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        for element in scene.elements:
            add(element.evidence)
            visible = " / ".join(
                item for item in (element.label.strip(), element.meaning.strip()) if item
            )
            if visible:
                add((visible,))
            add_grounded(
                json.dumps(
                    {
                        "element_id": element.element_id,
                        "role": element.role,
                        "label": element.label,
                        "meaning": element.meaning,
                        "states": dict(element.states),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
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
            grounded_visual_facts=tuple(grounded_visual_facts),
            last_action_outcome=str(action_outcome or "not_applicable"),
            blocked_reasons=self._text_items(verification.get("blocked_reasons")),
            verified_action_transition=verified_action_transition,
            controller_transition_evidence_refs=(
                controller_transition_evidence_refs
            ),
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
    decision_node_id: str
    action_digest: str
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
            "decision_node_id": self.decision_node_id,
            "action_digest": self.action_digest,
        }


@dataclass
class RiskConfirmationAuthority:
    session_id: str
    task_id: str
    device_id: str
    revision: int
    subgoal_id: str
    risk_ids: tuple[str, ...]
    intent_digest: str
    intent_preview: dict[str, Any]
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
            "intent_digest": self.intent_digest,
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
    last_post_action_transition: dict[str, Any] | None = None
    last_confirmation_failure: dict[str, Any] | None = None
    confirm_stage: str = ""
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
            "confirm_stage": self.confirm_stage,
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
            "post_action_transition_protocol": (
                POST_ACTION_TRANSITION_PROTOCOL_VERSION
            ),
            "last_post_action_transition": (
                dict(self.last_post_action_transition)
                if self.last_post_action_transition is not None
                else None
            ),
            "last_confirmation_failure": (
                dict(self.last_confirmation_failure)
                if self.last_confirmation_failure is not None
                else None
            ),
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
            "risk_confirmation_preview": (
                dict(self.risk_confirmation_authority.intent_preview)
                if self.risk_confirmation_authority is not None
                and not self.risk_confirmation_authority.consumed
                else None
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
        transition_markers = (
            "刷新",
            "重新加载",
            "重新载入",
            "重新获取",
            "重新读取",
            "重新连接",
            "加载完成",
            "更新完成",
            "同步完成",
            "导航",
            "跳转",
            "进入",
            "返回",
            "切换",
            "打开",
            "启动",
            "收起",
            "隐藏",
            "关闭",
            "消失",
            "移除",
            "不可见",
            "不存在",
            "缺失",
            "refresh",
            "reload",
            "reloaded",
            "updated",
            "synchronized",
            "navigated",
            "redirected",
            "entered",
            "returned",
            "switched",
            "opened",
            "launched",
            "dismiss",
            "hide",
            "hidden",
            "close",
            "closed",
            "disappear",
            "remove",
            "not visible",
            "absent",
            "missing",
            "retrieved",
            "refetched",
            "reconnected",
        )
        return any(marker in text for marker in presence_markers) and not any(
            marker in text for marker in value_verification_markers
        ) and not any(marker in text for marker in transition_markers)

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

    @staticmethod
    def _presence_binding_terms(*values: Any) -> frozenset[str]:
        """Return bounded literal terms for a zero-action presence check."""

        text = " ".join(
            str(value or "").casefold().replace("_", " ") for value in values
        )
        generic = {
            "action", "button", "control", "current", "element", "image",
            "item", "page", "screen", "target", "view", "visible",
            "当前", "页面", "画面", "目标", "元素", "控件", "可见", "出现",
        }
        terms = {
            token
            for token in re.findall(r"[a-z0-9]{3,}", text)
            if token not in generic
        }
        for run in re.findall(r"[\u4e00-\u9fff]{2,}", text):
            for size in range(2, min(6, len(run)) + 1):
                terms.update(
                    run[index:index + size]
                    for index in range(0, len(run) - size + 1)
                )
        return frozenset(term for term in terms if term not in generic)

    @staticmethod
    def _presence_surface_classes(*values: Any) -> frozenset[str]:
        """Keep destination/container nouns from collapsing into ordinal overlap."""

        text = " ".join(
            str(value or "").casefold().replace("_", " ").replace("-", " ")
            for value in values
        )
        classes = {
            name
            for name, markers in {
                "page": (
                    "页面",
                    "网页",
                    "界面",
                    "首页",
                    "主界面",
                    " page",
                    "screen",
                    "view",
                    "interface",
                    "app home",
                ),
                "title": ("标题", "题头", "title", "heading"),
                "list": ("列表", "清单", " list"),
                "input": ("输入框", "文本框", "input field", "textbox"),
                "menu": ("菜单", " menu"),
                "dialog": ("对话框", "弹窗", "dialog", "modal"),
                "destination": (
                    "对应页面",
                    "目标页面",
                    "下一页",
                    "详情页",
                    "详情",
                    "destination page",
                    "target page",
                    "next page",
                    "detail page",
                    "details",
                ),
                "foreground_app": (
                    "应用在前台",
                    "前台应用",
                    "前台可见",
                    "foreground app",
                    "in the foreground",
                    "is foreground",
                ),
            }.items()
            if any(marker in text for marker in markers)
        }
        return frozenset(classes)

    @classmethod
    def _target_app_identity_terms(cls, *values: Any) -> frozenset[str]:
        return cls._presence_binding_terms(*values).difference(
            {"app", "application", "android", "com", "应用", "程序"}
        )

    @classmethod
    def _referenced_target_app_pages(
        cls,
        *,
        graph: DynamicTaskGraph,
        presence_text: str,
    ) -> tuple[Any, ...]:
        """Return target Apps whose named page is the claimed visible state.

        A launcher affordance labelled with an App name proves that the App can
        be opened; it does not prove that the named App page is already in the
        foreground.  Keep the binding structural and graph-derived so the same
        rule applies to every App and every natural-language goal.
        """

        required_surfaces = cls._presence_surface_classes(presence_text)
        if not required_surfaces.intersection({"page", "foreground_app"}):
            return ()
        presence_terms = cls._presence_binding_terms(presence_text)
        referenced = []
        for target_app in graph.goal.target_apps:
            if str(target_app.app_id or "").strip().casefold() == "current_foreground":
                continue
            app_terms = cls._target_app_identity_terms(
                target_app.app_id,
                target_app.app_name,
            )
            if app_terms and app_terms.intersection(presence_terms):
                referenced.append(target_app)
        return tuple(referenced)

    @classmethod
    def _scene_foreground_matches_target_app_page(
        cls,
        *,
        scene: Any,
        target_apps: tuple[Any, ...],
    ) -> bool:
        foreground = str(getattr(scene, "foreground_app_id", "") or "").strip()
        if not foreground or foreground.casefold() == "unknown":
            return False
        foreground_terms = cls._target_app_identity_terms(foreground)
        for target_app in target_apps:
            app_id = str(target_app.app_id or "").strip()
            if app_id and foreground.casefold() == app_id.casefold():
                return True
            app_terms = cls._target_app_identity_terms(
                target_app.app_id,
                target_app.app_name,
            )
            if foreground_terms and foreground_terms.intersection(app_terms):
                return True
        return False

    @staticmethod
    def _scene_page_identity_facts(scene: Any) -> tuple[str, ...]:
        facts = [
            json.dumps(
                {
                    "app_id": str(getattr(scene, "app_id", "") or ""),
                    "foreground_app_id": str(
                        getattr(scene, "foreground_app_id", "") or ""
                    ),
                    "screen_id": str(getattr(scene, "screen_id", "") or ""),
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        ]
        for element in tuple(getattr(scene, "elements", ()) or ()):
            role = str(getattr(element, "role", "") or "").casefold()
            meaning = str(getattr(element, "meaning", "") or "").casefold()
            if role not in {"text", "container"}:
                continue
            if role != "container" and not any(
                marker in meaning
                for marker in ("page", "screen", "view", "home", "title", "heading")
            ):
                continue
            facts.append(
                json.dumps(
                    {
                        "role": role,
                        "meaning": meaning,
                        "label": str(getattr(element, "label", "") or ""),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        return tuple(facts)

    @classmethod
    def _scene_named_presence_is_grounded(
        cls,
        *,
        scene: Any,
        texts: tuple[str, ...],
    ) -> bool:
        return named_visual_identity_is_grounded(
            texts,
            cls._scene_page_identity_facts(scene),
        )

    @classmethod
    def _intrinsic_presence_surface_classes(cls, item: Any) -> frozenset[str]:
        """Return surface types owned by an element, not words near it."""

        role = str(getattr(item, "role", "") or "").casefold()
        classes = set(
            cls._presence_surface_classes(
                role,
                getattr(item, "meaning", ""),
            )
        )
        role_classes = {
            "input": "input",
            "textbox": "input",
            "text_input": "input",
            "list": "list",
            "list_item": "list",
            "menu": "menu",
            "menu_item": "menu",
            "dialog": "dialog",
            "modal": "dialog",
            "title": "title",
            "heading": "title",
        }
        if role in role_classes:
            classes.add(role_classes[role])
        return frozenset(classes)

    def _multi_presence_candidates(
        self,
        *,
        subgoal: Any,
        scene: Any,
        trusted_observation: Any,
    ) -> tuple[Any, ...] | None:
        """Bind two to four explicitly conjoined visible objects, fail closed."""

        text = " ".join(
            str(item or "").strip()
            for item in (
                getattr(subgoal, "objective", ""),
                *tuple(getattr(subgoal, "completion_conditions", ()) or ()),
            )
            if str(item or "").strip()
        ).casefold()
        if not self._is_explicit_multi_presence_text(text):
            return None
        text_terms = self._presence_binding_terms(text)
        if not text_terms:
            return None
        matched: list[tuple[Any, frozenset[str]]] = []
        for candidate in scene.elements:
            terms = self._presence_binding_terms(
                candidate.label,
                candidate.meaning,
                *candidate.evidence,
            ).intersection(text_terms)
            if not terms:
                continue
            if (
                float(candidate.confidence) < MIN_TARGET_CONFIDENCE
                or candidate.states.get("visible") is False
                or candidate.states.get("fully_visible") is False
                or self._candidate_has_unresolved_conflict(
                    trusted_observation,
                    candidate.element_id,
                )
            ):
                return None
            left, top, right, bottom = candidate.bounds
            if not (
                0.02 <= left < right <= 0.98
                and 0.02 <= top < bottom <= 0.98
            ):
                return None
            matched.append((candidate, frozenset(terms)))
        # An instruction card may repeat every endpoint name. It is aggregate
        # evidence, not either endpoint. Remove it only when two or more
        # smaller candidates together cover all of its bound terms.
        reduced: list[tuple[Any, frozenset[str]]] = []
        for index, item in enumerate(matched):
            _candidate, terms = item
            others = [
                other_terms
                for other_index, (_other, other_terms) in enumerate(matched)
                if other_index != index and other_terms.intersection(terms)
            ]
            aggregate = False
            if len(others) >= 2:
                for first in range(len(others)):
                    for second in range(first + 1, len(others)):
                        combined = others[first].union(others[second])
                        if combined and combined.issubset(terms):
                            aggregate = True
                            break
                    if aggregate:
                        break
            if not aggregate:
                reduced.append(item)
        if not 2 <= len(reduced) <= 4:
            return None
        goal_candidates = tuple(item[0] for item in reduced)
        candidate_terms = [item[1] for item in reduced]
        for index, terms in enumerate(candidate_terms):
            other_terms = frozenset().union(
                *(item for other_index, item in enumerate(candidate_terms)
                  if other_index != index)
            )
            if not terms.difference(other_terms):
                return None
        return goal_candidates

    @staticmethod
    def _is_explicit_multi_presence_text(text: str) -> bool:
        return bool(
            re.search(
                r"(?:和|与|及|同时|均|都|两者|两个|多个|分别|"
                r"\bboth\b|\band\b|\ball\b|\btwo\b|\bmultiple\b)",
                str(text or "").casefold(),
            )
        )

    def _try_advance_visible_presence_subgoal(
        self,
        session: UniversalAgentSessionState,
        *,
        graph: DynamicTaskGraph,
        trusted_observation: Any,
    ) -> DynamicTaskGraph | None:
        """Use bounded visible evidence to advance one safe presence checkpoint."""

        current = graph.active_subgoal()
        if (
            current is None
            or current.external_impact not in {"read_only", "navigation_only"}
            or not self._is_presence_only_read_only_subgoal(current)
        ):
            return None
        scene = getattr(trusted_observation, "scene", None)
        if scene is None:
            return None
        candidates: tuple[Any, ...]
        presence_text = " ".join(
            (
                current.objective,
                *tuple(current.completion_conditions or ()),
            )
        )
        if not self._scene_named_presence_is_grounded(
            scene=scene,
            texts=(current.objective, *tuple(current.completion_conditions or ())),
        ):
            return None
        referenced_app_pages = self._referenced_target_app_pages(
            graph=graph,
            presence_text=presence_text,
        )
        if referenced_app_pages and not self._scene_foreground_matches_target_app_page(
            scene=scene,
            target_apps=referenced_app_pages,
        ):
            return None
        if self._is_explicit_multi_presence_text(presence_text):
            candidates = self._multi_presence_candidates(
                subgoal=current,
                scene=scene,
                trusted_observation=trusted_observation,
            ) or ()
            if not candidates:
                return None
        else:
            completion_terms = self._presence_binding_terms(
                *tuple(current.completion_conditions or ())
            )
            required_surfaces = self._presence_surface_classes(
                *tuple(current.completion_conditions or ())
            )
            candidate = scene.unique_trusted_goal_element(
                min_confidence=MIN_TARGET_CONFIDENCE,
            )
            if candidate is not None:
                candidate_terms = self._presence_binding_terms(
                    candidate.label,
                    candidate.meaning,
                    *candidate.evidence,
                )
                scene_terms = self._presence_binding_terms(
                    scene.screen_id,
                    scene.summary,
                )
                candidate_surfaces = self._intrinsic_presence_surface_classes(
                    candidate
                )
                scene_container_surfaces = self._presence_surface_classes(
                    scene.screen_id,
                    scene.summary,
                ).intersection({"page"})
                if referenced_app_pages:
                    scene_container_surfaces = scene_container_surfaces.union(
                        required_surfaces.intersection({"foreground_app"})
                    )
                required_element_surfaces = required_surfaces.difference(
                    scene_container_surfaces
                )
                if (
                    not completion_terms
                    or not (
                        candidate_terms.intersection(completion_terms)
                        or scene_terms.intersection(completion_terms)
                    )
                    or not required_element_surfaces.issubset(candidate_surfaces)
                    or candidate.states.get("fully_visible") is not True
                    or self._candidate_has_unresolved_conflict(
                        trusted_observation,
                        candidate.element_id,
                    )
                ):
                    return None
                candidates = (candidate,)
            elif current.external_impact == "navigation_only":
                presence_terms = self._presence_binding_terms(presence_text)
                summary_terms = self._presence_binding_terms(scene.summary)
                if not presence_terms or not presence_terms.intersection(summary_terms):
                    return None
                scene_container_surfaces = self._presence_surface_classes(
                    scene.screen_id,
                    scene.summary,
                ).intersection({"page"})
                if referenced_app_pages:
                    scene_container_surfaces = scene_container_surfaces.union(
                        required_surfaces.intersection({"foreground_app"})
                    )
                required_element_surfaces = required_surfaces.difference(
                    scene_container_surfaces
                )
                matched = []
                for item in scene.elements:
                    item_terms = self._presence_binding_terms(
                        item.label,
                        item.meaning,
                        *item.evidence,
                    )
                    # Container identity (for example, "the current page") is
                    # a scene-level fact.  The required control type must be
                    # intrinsic to the candidate itself; a nearby instruction
                    # merely mentioning an input must not become that input.
                    intrinsic_item_surfaces = (
                        self._intrinsic_presence_surface_classes(item)
                    )
                    if (
                        not presence_terms.intersection(item_terms)
                        or not required_element_surfaces.issubset(
                            intrinsic_item_surfaces
                        )
                    ):
                        continue
                    if (
                        float(item.confidence) < MIN_TARGET_CONFIDENCE
                        or item.states.get("fully_visible") is not True
                        or self._candidate_has_unresolved_conflict(
                            trusted_observation,
                            item.element_id,
                        )
                    ):
                        return None
                    left, top, right, bottom = item.bounds
                    if not (
                        0.02 <= left < right <= 0.98
                        and 0.02 <= top < bottom <= 0.98
                    ):
                        return None
                    matched.append(item)
                if not 1 <= len(matched) <= 4:
                    return None
                candidates = tuple(matched)
            else:
                return None

        candidate_facts = tuple(
            "当前可信画面的目标元素："
            f"element_id={item.element_id}, role={item.role}, "
            f"label={item.label or '[empty]'}, meaning={item.meaning}, "
            f"confidence={float(item.confidence):.3f}, "
            f"fully_visible={item.states.get('fully_visible', 'unknown')}, "
            "bounds_inside_safe_frame=true。"
            for item in candidates
        )
        observed = self.bridge.observed_state(
            graph=graph,
            trusted_observation=trusted_observation,
            action_outcome="not_applicable",
            verification={
                "visible_evidence": [
                    scene.summary,
                    *candidate_facts,
                    *(fact for item in candidates for fact in item.evidence),
                ]
            },
        )
        revised = self.deepseek_planner.replan(
            graph,
            observed,
            trigger="subgoal_completed",
            reason=(
                f"当前可信画面已经以{len(candidates)}个逐项语义绑定、"
                f"高置信且无冲突的目标元素证明定位类 {current.external_impact} 子目标；"
                f"本轮只能把当前 subgoal_id={current.subgoal_id} 标为 completed，"
                "其 completion_evidence 必须逐字选择 visible_evidence 中至少一项；"
                "最多激活一个直接后继，其他节点不得越级完成。不得推断元素值、"
                "外部状态或执行动作；无法满足这些约束时必须 blocked。"
            ),
        )
        self._validate_graph_identity(
            revised,
            device_id=session.device_id,
            previous=graph,
            trusted_observation=trusted_observation,
        )
        if revised.revision != graph.revision + 1:
            raise UniversalAgentOrchestratorError(
                "可见状态证据推进必须且只能产生一个新 revision。"
            )
        old_ids = tuple(item.subgoal_id for item in graph.subgoals)
        new_ids = tuple(item.subgoal_id for item in revised.subgoals)
        if old_ids != new_ids:
            raise UniversalAgentOrchestratorError(
                "可见状态证据推进不得增加、删除或重排子目标。"
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
                "可见状态证据推进不得修改目标、约束、全局完成条件或风险定义。"
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
                    "可见状态证据推进只能改变子目标状态和完成证据。"
                )
        newly_completed = tuple(
            subgoal_id
            for subgoal_id in old_ids
            if old_by_id[subgoal_id].status != "completed"
            and new_by_id[subgoal_id].status == "completed"
        )
        completed_current = new_by_id[current.subgoal_id]
        completed_before = {
            item.subgoal_id for item in graph.subgoals if item.status == "completed"
        }
        accepted_prefix: list[str] = []
        prefix_valid = bool(
            newly_completed and newly_completed[0] == current.subgoal_id
        )
        for subgoal_id in newly_completed:
            old_item = old_by_id[subgoal_id]
            new_item = new_by_id[subgoal_id]
            dependencies_ready = all(
                dependency in completed_before or dependency in accepted_prefix
                for dependency in old_item.depends_on
            )
            if (
                old_item.external_impact not in {"read_only", "navigation_only"}
                or not self._is_presence_only_read_only_subgoal(old_item)
                or not dependencies_ready
                or not new_item.completion_evidence
            ):
                prefix_valid = False
                break
            accepted_prefix.append(subgoal_id)
        if not prefix_valid or not completed_current.completion_evidence:
            raise UniversalAgentOrchestratorError(
                "可见状态证据只能完成从当前节点开始、依赖连续满足的安全定位前缀，"
                "且每个节点必须记录可见证据："
                f"current={current.subgoal_id}, newly_completed={newly_completed}, "
                f"current_evidence_count={len(completed_current.completion_evidence)}。"
            )
        for subgoal_id in old_ids:
            old_status = old_by_id[subgoal_id].status
            new_status = new_by_id[subgoal_id].status
            if subgoal_id in newly_completed:
                continue
            if old_status == "completed" and new_status != "completed":
                raise UniversalAgentOrchestratorError(
                    "可见状态证据推进不得回退已完成子目标。"
                )
            if old_status == "pending" and new_status not in {"pending", "active"}:
                raise UniversalAgentOrchestratorError(
                    "可见状态证据推进不得越过后续子目标。"
                )
        newly_active = tuple(
            subgoal_id
            for subgoal_id in old_ids
            if old_by_id[subgoal_id].status == "pending"
            and new_by_id[subgoal_id].status == "active"
        )
        if len(newly_active) > 1:
            raise UniversalAgentOrchestratorError(
                "可见状态证据推进最多只能激活一个后续子目标。"
            )
        if revised.status != "completed" and (
            len(newly_active) != 1
            or revised.active_subgoal_id != newly_active[0]
        ):
            raise UniversalAgentOrchestratorError(
                "可见状态证据推进后必须精确激活一个后续子目标。"
            )
        return revised

    def _advance_visible_presence_prefix(
        self,
        session: UniversalAgentSessionState,
        *,
        graph: DynamicTaskGraph,
        trusted_observation: Any,
    ) -> tuple[DynamicTaskGraph, int]:
        """Consume a bounded, independently validated visible-state prefix."""

        current_graph = graph
        advances = 0
        while advances < MAX_VISIBLE_PRESENCE_ADVANCES_PER_OBSERVATION:
            revised = self._try_advance_visible_presence_subgoal(
                session,
                graph=current_graph,
                trusted_observation=trusted_observation,
            )
            if revised is None:
                break
            self._store_revised_graph(session, revised)
            current_graph = revised
            advances += 1
            if revised.status == "completed":
                break
        return current_graph, advances

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

    def _review_completion_candidate(
        self,
        session: UniversalAgentSessionState,
        *,
        graph: DynamicTaskGraph,
        trusted_observation: Any,
        decision: Any,
        reason: str,
    ) -> DynamicTaskGraph:
        """Require DeepSeek to approve a Qwen-only visible completion claim."""

        proposal = decision.proposal
        if proposal.status != "finished":
            raise UniversalAgentOrchestratorError(
                "只有 Qwen finished 决策可以进入完成复核。"
            )
        observed = self.bridge.observed_state(
            graph=graph,
            trusted_observation=trusted_observation,
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
            reason=reason,
        )
        self._validate_graph_identity(
            revised,
            device_id=session.device_id,
            previous=graph,
            trusted_observation=trusted_observation,
        )
        self._store_revised_graph(session, revised)
        if revised.status == "completed":
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
        return revised

    @staticmethod
    def _policy_payload(decision: NavigationPolicyDecision) -> dict[str, Any]:
        return {
            "allowed": decision.allowed,
            "reason": decision.reason,
            "canonical_class": decision.canonical_class,
            "policy_version": PhaseOneNavigationPolicy.VERSION,
        }

    @classmethod
    def _validate_newly_completed_named_app_surfaces(
        cls,
        *,
        previous: DynamicTaskGraph,
        revised: DynamicTaskGraph,
        trusted_observation: Any,
    ) -> None:
        scene = getattr(trusted_observation, "scene", None)
        if scene is None:
            raise UniversalAgentOrchestratorError(
                "DeepSeek revision 缺少可复核的可信场景。"
            )
        old_by_id = {item.subgoal_id: item for item in previous.subgoals}
        for item in revised.subgoals:
            old = old_by_id.get(item.subgoal_id)
            if item.status != "completed" or (
                old is not None and old.status == "completed"
            ):
                continue
            presence_text = " ".join(
                (item.objective, *tuple(item.completion_conditions or ()))
            )
            referenced_app_pages = cls._referenced_target_app_pages(
                graph=previous,
                presence_text=presence_text,
            )
            if referenced_app_pages and not cls._scene_foreground_matches_target_app_page(
                scene=scene,
                target_apps=referenced_app_pages,
            ):
                raise UniversalAgentOrchestratorError(
                    "Launcher 或其他页面中的 App 入口不能证明目标 App 页面已在前台："
                    f"subgoal_id={item.subgoal_id}。"
                )

    @classmethod
    def _validate_graph_identity(
        cls,
        graph: DynamicTaskGraph,
        *,
        device_id: str,
        previous: DynamicTaskGraph | None = None,
        trusted_observation: Any | None = None,
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
            if graph.revision != previous.revision + 1:
                raise UniversalAgentOrchestratorError(
                    "DeepSeek 重规划 revision 必须严格等于上一 revision + 1。"
                )
            if trusted_observation is not None:
                cls._validate_newly_completed_named_app_surfaces(
                    previous=previous,
                    revised=graph,
                    trusted_observation=trusted_observation,
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

    def _record_deepseek_failure(
        self,
        session: UniversalAgentSessionState,
        error: Exception,
        *,
        stage: str,
    ) -> None:
        if not isinstance(error, TaskGraphError):
            return
        try:
            paths = persist_deepseek_failure_diagnostic(
                self.deepseek_planner,
                evidence_dir=session.run_dir,
                prefix=f"deepseek_{stage}_step_{session.step_number}",
                failed_stage=stage,
                error=error,
            )
        except Exception:
            return
        self._remember(session, paths)

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

    def _ensure_terminal_snapshot(self, session: UniversalAgentSessionState) -> None:
        """Write a terminal report only when the persisted one is missing/stale."""

        report_path = session.run_dir / "report.json"
        try:
            persisted = json.loads(report_path.read_text(encoding="utf-8"))["session"]
            current = session.snapshot()
            compared_fields = (
                "status",
                "failed_reason",
                "confirm_stage",
                "physical_actions",
                "last_post_action_transition",
                "last_confirmation_failure",
            )
            if all(persisted.get(key) == current.get(key) for key in compared_fields):
                return
        except (OSError, ValueError, KeyError, TypeError):
            pass
        self._write_terminal_snapshot(session)

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
            "decision_node_id": str(decision.proposal.action.node_id),
            "action_digest": _action_digest(decision.proposal.action),
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
            decision_node_id=scope["decision_node_id"],
            action_digest=scope["action_digest"],
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
            "decision_node_id",
            "action_digest",
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
        decision_node_id = str(value.get("decision_node_id") or "").strip()
        action_digest = str(value.get("action_digest") or "").strip()
        if not decision_node_id:
            raise UniversalAgentOrchestratorError(
                "确认作用域 decision_node_id 不能为空。"
            )
        if not re.fullmatch(r"[0-9a-f]{64}", action_digest):
            raise UniversalAgentOrchestratorError(
                "确认作用域 action_digest 必须是 64 位小写 SHA-256。"
            )
        return {
            "session_id": str(value.get("session_id") or ""),
            "task_id": str(value.get("task_id") or ""),
            "device_id": str(value.get("device_id") or ""),
            "revision": revision,
            "subgoal_id": str(value.get("subgoal_id") or ""),
            "risk_ids": sorted(str(item) for item in risk_ids),
            "observation_id": str(value.get("observation_id") or ""),
            "fingerprint": str(value.get("fingerprint") or ""),
            "decision_node_id": decision_node_id,
            "action_digest": action_digest,
        }

    def _bind_risk_confirmation(self, session: UniversalAgentSessionState) -> None:
        graph = session.task_graph
        if graph is None:
            raise UniversalAgentOrchestratorError("风险确认缺少任务图。")
        current = graph.active_subgoal()
        if current is None or not current.risk_action_ids:
            raise UniversalAgentOrchestratorError("当前子目标没有可确认风险。")
        intent_digest, intent_preview = _risk_intent_material(graph, current)
        session.risk_confirmation_authority = RiskConfirmationAuthority(
            session_id=session.session_id,
            task_id=graph.task_id,
            device_id=graph.device_id,
            revision=graph.revision,
            subgoal_id=current.subgoal_id,
            risk_ids=tuple(current.risk_action_ids),
            intent_digest=intent_digest,
            intent_preview=intent_preview,
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
            "intent_digest",
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
        intent_digest = str(value.get("intent_digest") or "").strip()
        if not re.fullmatch(r"[0-9a-f]{64}", intent_digest):
            raise UniversalAgentOrchestratorError(
                "风险确认 intent_digest 必须是 64 位小写 SHA-256。"
            )
        return {
            "session_id": str(value.get("session_id") or ""),
            "task_id": str(value.get("task_id") or ""),
            "device_id": str(value.get("device_id") or ""),
            "revision": revision,
            "subgoal_id": str(value.get("subgoal_id") or ""),
            "risk_ids": sorted(str(item) for item in risk_ids),
            "intent_digest": intent_digest,
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
        before_observation: Any,
        new_observation: Any,
    ) -> None:
        previous_graph = session.task_graph
        assert previous_graph is not None
        action_outcome = str(getattr(result, "action_outcome", "matched"))
        if action_outcome not in POST_ACTION_OUTCOMES:
            raise UniversalAgentOrchestratorError(
                f"动作后验证返回了不支持的 outcome：{action_outcome}。"
            )
        matched = action_outcome == "matched"
        verification_errors = tuple(
            str(item)
            for item in getattr(result, "verification_errors", ())
            if str(item).strip()
        )
        if matched and verification_errors:
            raise UniversalAgentOrchestratorError(
                "动作后验证同时返回 matched 与 verification_errors。"
            )
        if not matched and not verification_errors:
            raise UniversalAgentOrchestratorError(
                "动作后验证返回 mismatched 但没有结构化错误。"
            )
        previous_current = previous_graph.active_subgoal()
        previous_decision = session.qwen_decision
        authority = session.confirmation_authority
        if (
            previous_current is None
            or previous_decision is None
            or previous_decision.proposal.action is None
            or authority is None
        ):
            raise UniversalAgentOrchestratorError(
                "动作后重规划缺少上一子目标、决策或确认权威。"
            )
        wait_transition = (
            result.resolved_action.kind == "wait_for_change"
            and int(result.physical_actions) == 0
        )
        receipt: VerifiedActionTransition | None = None
        controller_refs: tuple[ControllerTransitionEvidenceRef, ...] = ()
        if not wait_transition:
            receipt = VerifiedActionTransition(
                receipt_id=f"receipt_{uuid.uuid4().hex}",
                session_id=session.session_id,
                task_id=previous_graph.task_id,
                device_id=previous_graph.device_id,
                prior_revision=previous_graph.revision,
                subgoal_id=previous_current.subgoal_id,
                decision_node_id=authority.decision_node_id,
                action_digest=authority.action_digest,
                rebound_action_digest=_action_digest(result.rebound_action),
                resolved_action_digest=_action_digest(result.resolved_action),
                action_kind=str(previous_decision.proposal.action.action),
                before_observation_id=str(before_observation.observation_id),
                before_fingerprint=str(before_observation.fingerprint),
                after_observation_id=str(new_observation.observation_id),
                after_fingerprint=str(new_observation.fingerprint),
                physical_actions=int(result.physical_actions),
                outcome=action_outcome,
                errors=verification_errors,
                controller_completion_evidence=tuple(
                    str(item)
                    for item in getattr(
                        result, "controller_completion_evidence", ()
                    )
                    if str(item).strip()
                ),
            )
            receipt.validate()
            if previous_current.external_impact == "navigation_only":
                controller_refs = tuple(
                    ControllerTransitionEvidenceRef(
                        ref_id=(
                            f"controller_transition:{receipt.receipt_id}:{index}"
                        ),
                        receipt_id=receipt.receipt_id,
                        subgoal_id=receipt.subgoal_id,
                        text=text,
                    )
                    for index, text in enumerate(
                        receipt.controller_completion_evidence,
                        start=1,
                    )
                )
        verification = {
            "matched": matched,
            "action_outcome": action_outcome,
            "physical_actions": result.physical_actions,
            "before_fingerprint": before_observation.fingerprint,
            "execution_before_fingerprint": result.before_scene.fingerprint,
            "after_fingerprint": result.after_scene.fingerprint,
            "visible_evidence": [result.after_scene.summary],
            "blocked_reasons": list(verification_errors),
            "after_frame_paths": list(result.after_frame_paths),
            "controller_completion_evidence": list(
                receipt.controller_completion_evidence if receipt is not None else ()
            ),
            "verified_action_transition": (
                receipt.to_dict() if receipt is not None else None
            ),
            "transition_kind": (
                "wait_observation" if wait_transition else "physical_action"
            ),
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
            action_outcome=("not_applicable" if wait_transition else action_outcome),
            verification=verification,
            verified_action_transition=receipt,
            controller_transition_evidence_refs=controller_refs,
        )
        previous_signature = _subgoal_progress_signature(previous_current)
        transition_record = {
            "protocol_version": POST_ACTION_TRANSITION_PROTOCOL_VERSION,
            "receipt": receipt.to_dict() if receipt is not None else None,
            "transition_kind": (
                "wait_observation" if wait_transition else "physical_action"
            ),
            "prior_subgoal_signature": previous_signature,
            "prior_action_equivalence_digest": _action_equivalence_digest(
                previous_decision.proposal.action
            ),
            "disposition": "replanning",
        }

        def persist_transition() -> None:
            session.last_post_action_transition = dict(transition_record)
            self._remember(
                session,
                session.evidence_store.write_post_action_transition(
                    max(1, session.step_number - 1),
                    transition_record,
                ),
            )

        persist_transition()
        # The consumed decision and every authority derived from the old frame
        # become unusable before any model replan attempt.
        session.qwen_decision = None
        session.controller_decision = None
        session.confirmation_authority = None
        session.risk_confirmation_authority = None
        session.confirmed_risk_ids = ()
        try:
            revised = self.deepseek_planner.replan(
                previous_graph,
                observed,
                trigger=(
                    "observation_changed"
                    if wait_transition
                    else "action_result_matched"
                    if matched
                    else "action_result_mismatch"
                ),
                reason=(
                    "wait_for_change 未产生物理动作；仅依据新的可信画面重规划。"
                    if wait_transition
                    else
                    "一个动作已经执行并由新的可信画面验证。"
                    if matched
                    else "动作已执行，但新画面没有证明预期语义变化，必须重规划。"
                ),
            )
            self._validate_graph_identity(
                revised,
                device_id=session.device_id,
                previous=previous_graph,
                trusted_observation=new_observation,
            )
        except Exception as exc:
            session.status = "blocked"
            session.failed_reason = f"DeepSeek 重规划失败：{exc}"
            transition_record["disposition"] = "blocked_replan_failure"
            transition_record["diagnostic"] = session.failed_reason
            persist_transition()
            return

        session.task_graph = revised
        session.goal_draft = self.bridge.goal_draft(revised)
        revised_current = revised.active_subgoal()
        transition_record["revised_revision"] = revised.revision
        transition_record["revised_subgoal_id"] = (
            revised_current.subgoal_id if revised_current is not None else None
        )
        transition_record["revised_subgoal_signature"] = (
            _subgoal_progress_signature(revised_current)
        )
        if receipt is not None:
            transition_record["receipt_consumed_revision"] = revised.revision
        self._remember(
            session,
            session.evidence_store.write_task_graph(revised),
            session.evidence_store.write_risk_audit(revised),
        )
        if revised.status == "completed":
            session.status = "succeeded"
            transition_record["disposition"] = "task_completed"
            persist_transition()
            return
        if controller_refs:
            prior_in_revised = next(
                (
                    item
                    for item in revised.subgoals
                    if item.subgoal_id == previous_current.subgoal_id
                ),
                None,
            )
            if prior_in_revised is None or prior_in_revised.status != "completed":
                session.status = "blocked"
                session.failed_reason = (
                    "本地一次性 controller_transition 完成证据已满足，"
                    "但重规划未完成其绑定的 navigation_only 子目标；"
                    "禁止继续产生动作或第二确认。"
                )
                transition_record["disposition"] = (
                    "blocked_unconsumed_controller_completion"
                )
                transition_record["diagnostic"] = session.failed_reason
                persist_transition()
                return
        current = revised.active_subgoal()
        impact = current.external_impact if current is not None else "unknown"
        if current is None:
            session.status = "blocked"
            session.failed_reason = "重规划后的任务图没有活动子目标。"
            transition_record["disposition"] = "blocked_missing_active_subgoal"
            persist_transition()
            return
        if impact in {"external_state", "unknown"}:
            session.status = "awaiting_risk_confirmation"
            session.failed_reason = ""
            transition_record["disposition"] = "advanced_to_risk_confirmation"
            self._bind_risk_confirmation(session)
            persist_transition()
            return
        if impact == "read_only":
            try:
                reviewed = self.deepseek_planner.replan(
                    revised,
                    replace(
                        observed,
                        last_action_outcome="not_applicable",
                        verified_action_transition=None,
                        controller_transition_evidence_refs=(),
                    ),
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
                    trusted_observation=new_observation,
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
                transition_record["disposition"] = "blocked_read_only_review"
                transition_record["diagnostic"] = session.failed_reason
                persist_transition()
                return
            if reviewed.status == "completed":
                session.status = "succeeded"
                session.failed_reason = ""
                transition_record["disposition"] = "task_completed_after_read_only_review"
                persist_transition()
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
                transition_record["disposition"] = "blocked_read_only_no_active"
                transition_record["diagnostic"] = session.failed_reason
                persist_transition()
                return
            if reviewed_impact in {"external_state", "unknown"}:
                session.status = "awaiting_risk_confirmation"
                session.failed_reason = ""
                self._bind_risk_confirmation(session)
                transition_record["disposition"] = (
                    "advanced_to_risk_confirmation_after_read_only"
                )
                persist_transition()
                return
            if reviewed_impact == "read_only":
                session.status = "blocked"
                session.failed_reason = (
                    "当前可信画面没有让 DeepSeek 完成 read_only 子目标；"
                    "禁止为只读验证请求物理动作。"
                )
                transition_record["disposition"] = "blocked_read_only_incomplete"
                transition_record["diagnostic"] = session.failed_reason
                persist_transition()
                return
            # A read-only checkpoint may be completed while the overall task still
            # has a later navigation-only subgoal. Reuse the same trusted frames;
            # do not capture again and do not execute anything without a new scope.
            revised = reviewed
            transition_record["revised_revision"] = revised.revision
            transition_record["revised_subgoal_id"] = revised.active_subgoal_id
            transition_record["revised_subgoal_signature"] = (
                _subgoal_progress_signature(revised.active_subgoal())
            )

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
            try:
                self._review_completion_candidate(
                    session,
                    graph=revised,
                    trusted_observation=new_observation,
                    decision=decision,
                    reason=(
                        "Qwen 在动作后的当前可信画面中提出完成候选，"
                        "要求 DeepSeek 复核整个任务。"
                    ),
                )
                transition_record["disposition"] = (
                    "task_completed_after_qwen_review"
                    if session.status == "succeeded"
                    else "blocked_completion_review"
                )
                if session.failed_reason:
                    transition_record["diagnostic"] = session.failed_reason
                persist_transition()
            except Exception as exc:
                session.status = "blocked"
                session.failed_reason = f"完成候选复核失败：{exc}"
                session.controller_decision = NavigationPolicyDecision(
                    allowed=False,
                    reason=session.failed_reason,
                )
                transition_record["disposition"] = "blocked_completion_review"
                transition_record["diagnostic"] = session.failed_reason
                persist_transition()
                return
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
            transition_record["disposition"] = "blocked_qwen_no_action"
            transition_record["diagnostic"] = session.failed_reason
            persist_transition()
            return
        new_equivalence_digest = _action_equivalence_digest(
            decision.proposal.action
        )
        transition_record["next_action_equivalence_digest"] = (
            new_equivalence_digest
        )
        if (
            bool(controller_refs)
            and matched
            and transition_record.get("revised_subgoal_signature")
            == previous_signature
            and new_equivalence_digest
            == transition_record["prior_action_equivalence_digest"]
        ):
            session.status = "blocked"
            session.failed_reason = (
                "一次性 controller_transition 完成证据已满足，"
                "但同一活动子目标仍提出等价动作；禁止生成第二确认。"
            )
            session.controller_decision = NavigationPolicyDecision(
                allowed=False,
                reason=session.failed_reason,
            )
            session.confirmation_authority = None
            transition_record["disposition"] = "blocked_equivalent_repeat"
            transition_record["diagnostic"] = session.failed_reason
            persist_transition()
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
            transition_record["disposition"] = "blocked_policy"
            transition_record["diagnostic"] = session.failed_reason
            persist_transition()
            return
        session.status = "awaiting_confirmation"
        self._bind_confirmation(session)
        transition_record["disposition"] = "advanced_to_new_confirmation"
        transition_record["next_confirmation_scope"] = (
            session.confirmation_authority.scope()
        )
        persist_transition()

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

        prior_observation = session.trusted_observation
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
            try:
                observation = self.trusted_observation_factory(
                    frames=frames,
                    device_id=session.device_id,
                    scene=scene,
                    observation_id=observation_id,
                )
            except Exception as exc:
                session.status = "blocked"
                session.failed_reason = f"重新观察证据不足：{exc}"
                session.qwen_decision = None
                session.controller_decision = NavigationPolicyDecision(
                    allowed=False,
                    reason=session.failed_reason,
                )
                if session.physical_actions != before_actions:
                    raise UniversalAgentOrchestratorError(
                        "重新观察证据失败路径错误地改变了物理动作计数。"
                    )
                blocked_decision = SimpleNamespace(
                    proposal=GenericStepProposal(
                        status="blocked",
                        reason=session.failed_reason,
                    )
                )
                self._write_terminal_snapshot(session)
                return blocked_decision
            session.trusted_observation = observation
            session.trusted_frames = tuple(frames)
            self._remember(
                session,
                session.evidence_store.write_trusted_observation(
                    session.step_number,
                    observation,
                ),
            )

            if (
                prior_observation is not None
                and str(getattr(prior_observation, "fingerprint", ""))
                != str(observation.fingerprint)
            ):
                observed = self.bridge.observed_state(
                    graph=graph,
                    trusted_observation=observation,
                    action_outcome="not_applicable",
                    verification={
                        "visible_evidence": [scene.summary],
                        "blocked_reasons": [],
                    },
                )
                try:
                    revised = self.deepseek_planner.replan(
                        graph,
                        observed,
                        trigger="observation_changed",
                        reason=(
                            "只读重新观察发现页面指纹变化；必须先修订高层状态，"
                            "再允许 Qwen 规划下一动作。"
                        ),
                    )
                    self._validate_graph_identity(
                        revised,
                        device_id=session.device_id,
                        previous=graph,
                        trusted_observation=observation,
                    )
                except Exception as exc:
                    session.status = "blocked"
                    session.failed_reason = f"页面变化重规划失败：{exc}"
                    session.qwen_decision = None
                    session.controller_decision = NavigationPolicyDecision(
                        allowed=False,
                        reason=session.failed_reason,
                    )
                    session.confirmed_risk_ids = ()
                    session.risk_confirmation_authority = None
                    if session.physical_actions != before_actions:
                        raise UniversalAgentOrchestratorError(
                            "页面变化重规划失败路径错误地改变了物理动作计数。"
                        )
                    blocked_decision = SimpleNamespace(
                        proposal=GenericStepProposal(
                            status="blocked",
                            reason=session.failed_reason,
                        )
                    )
                    self._write_terminal_snapshot(session)
                    return blocked_decision
                session.task_graph = revised
                session.goal_draft = self.bridge.goal_draft(revised)
                session.qwen_decision = None
                session.controller_decision = None
                session.confirmed_risk_ids = ()
                session.risk_confirmation_authority = None
                graph = revised
                goal = session.goal_draft
                self._remember(
                    session,
                    session.evidence_store.write_task_graph(revised),
                    session.evidence_store.write_risk_audit(revised),
                )
                if revised.status == "completed":
                    session.status = "succeeded"
                    session.failed_reason = ""
                    terminal_decision = SimpleNamespace(
                        proposal=GenericStepProposal(
                            status="finished",
                            reason="DeepSeek 已依据新的可信画面确认任务完成。",
                            completion_evidence=observed.visible_evidence[:3],
                        )
                    )
                    if session.physical_actions != before_actions:
                        raise UniversalAgentOrchestratorError(
                            "重新观察路径错误地改变了物理动作计数。"
                        )
                    self._write_terminal_snapshot(session)
                    return terminal_decision
                current = revised.active_subgoal()
                impact = (
                    current.external_impact if current is not None else "unknown"
                )
                if current is None:
                    session.status = "blocked"
                    session.failed_reason = "页面变化重规划后没有活动子目标。"
                    blocked_decision = SimpleNamespace(
                        proposal=GenericStepProposal(
                            status="blocked",
                            reason=session.failed_reason,
                        )
                    )
                    self._write_terminal_snapshot(session)
                    return blocked_decision
                if impact in {"external_state", "unknown"}:
                    session.status = "awaiting_risk_confirmation"
                    session.failed_reason = ""
                    self._bind_risk_confirmation(session)
                    risk_decision = SimpleNamespace(
                        proposal=GenericStepProposal(
                            status="blocked",
                            reason="页面变化后必须重新确认当前风险范围。",
                        )
                    )
                    self._write_terminal_snapshot(session)
                    return risk_decision

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
            elif decision.proposal.status == "finished":
                self._review_completion_candidate(
                    session,
                    graph=graph,
                    trusted_observation=observation,
                    decision=decision,
                    reason=(
                        "Qwen 在重新观察后的当前可信画面中提出完成候选，"
                        "要求 DeepSeek 复核整个任务。"
                    ),
                )
            else:
                session.status = "blocked"
                session.failed_reason = (
                    decision.proposal.reason
                    if decision.proposal.status == "blocked"
                    else f"不支持的 Qwen 状态：{decision.proposal.status}"
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
            self._record_deepseek_failure(
                session,
                exc,
                stage="refresh_decision",
            )
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
        before_actions = session.physical_actions
        authority_before = session.confirmation_authority
        post_transition_before = session.last_post_action_transition
        try:
            with self.device_registry.device_lock(session.device_id):
                try:
                    return self._confirm_one_locked(session, confirmation)
                except Exception as exc:
                    self._finalize_confirm_failure(
                        session,
                        exc,
                        before_actions=before_actions,
                        authority_before=authority_before,
                        post_transition_before=post_transition_before,
                    )
                    raise
        finally:
            self._release_if_terminal(session)

    def _finalize_confirm_failure(
        self,
        session: UniversalAgentSessionState,
        error: Exception,
        *,
        before_actions: int,
        authority_before: Any,
        post_transition_before: Any,
    ) -> None:
        """Best-effort memory/artifact convergence after a consumed confirm.

        The original exception remains authoritative and is always re-raised by
        the caller.  This method never invokes the adapter or any observation
        source and never invents unavailable receipt/frame fields.
        """

        authority = session.confirmation_authority or authority_before
        authority_consumed = bool(
            authority is not None and getattr(authority, "consumed", False)
        )
        request_actions = max(0, session.physical_actions - int(before_actions))
        if not (authority_consumed or request_actions):
            return

        # A transition produced by the normal after-observation path is more
        # specific than this exception boundary.  Never replace it; only make a
        # best-effort attempt to converge the terminal report with memory.
        if session.last_post_action_transition is not post_transition_before:
            try:
                self._ensure_terminal_snapshot(session)
            except Exception:
                pass
            return

        failed_stage = session.confirm_stage or "confirmation"
        reason = str(error).strip() or error.__class__.__name__
        if request_actions == 0 and session.status == "blocked":
            # Policy recheck already produced the authoritative blocked terminal
            # state and snapshot.  Consuming the confirmation token alone does
            # not turn that pre-action rejection into an execution failure.
            try:
                self._ensure_terminal_snapshot(session)
            except Exception:
                pass
            return
        recoverable_reobservation = bool(
            request_actions == 0 and session.status == "needs_reobservation"
        )
        if not recoverable_reobservation:
            session.status = "failed"
        session.failed_reason = reason
        if request_actions == 0:
            requested_kind = str(
                getattr(
                    getattr(
                        getattr(session.qwen_decision, "proposal", None),
                        "action",
                        None,
                    ),
                    "action",
                    "",
                )
                or ""
            )
            if failed_stage == "validating_confirmation":
                transition_kind = "confirmation_failure"
            elif requested_kind == "wait_for_change":
                transition_kind = "wait_observation_failure"
            else:
                transition_kind = "pre_action_failure"
            transition = {
                "protocol_version": POST_ACTION_TRANSITION_PROTOCOL_VERSION,
                "transition_kind": transition_kind,
                "disposition": (
                    "needs_reobservation"
                    if recoverable_reobservation
                    else "failed"
                ),
                "failed_stage": failed_stage,
                "error_type": error.__class__.__name__,
                "error": reason,
                "authority_consumed": authority_consumed,
                "physical_actions_before": int(before_actions),
                "physical_actions": int(session.physical_actions),
                "request_physical_actions": 0,
                "evidence": list(dict.fromkeys(session.evidence_paths)),
            }
            if authority is not None and callable(getattr(authority, "scope", None)):
                try:
                    transition["authority_scope"] = authority.scope()
                except Exception:
                    pass
            session.last_confirmation_failure = transition
            failure_step = max(1, session.step_number)
            try:
                self._remember(
                    session,
                    session.evidence_store.write_confirmation_failure(
                        failure_step,
                        transition,
                    ),
                )
            except Exception:
                pass
            try:
                self._write_terminal_snapshot(session)
            except Exception:
                pass
            return

        transition: dict[str, Any] = {
            "protocol_version": POST_ACTION_TRANSITION_PROTOCOL_VERSION,
            "transition_kind": "post_action_failure",
            "disposition": "failed",
            "failed_stage": failed_stage,
            "error_type": error.__class__.__name__,
            "error": reason,
            "authority_consumed": authority_consumed,
            "physical_actions_before": int(before_actions),
            "physical_actions": int(session.physical_actions),
            "request_physical_actions": request_actions,
            "evidence": list(dict.fromkeys(session.evidence_paths)),
        }
        if authority is not None and callable(getattr(authority, "scope", None)):
            try:
                transition["authority_scope"] = authority.scope()
            except Exception:
                pass
        session.last_post_action_transition = transition
        failure_step = max(1, session.step_number)
        if session.history and authority is not None:
            latest = session.history[-1]
            if latest.get("task_revision") == getattr(authority, "revision", None):
                failure_step = max(1, int(latest.get("step_number") or failure_step))
        try:
            self._remember(
                session,
                session.evidence_store.write_post_action_transition(
                    failure_step,
                    transition,
                ),
            )
        except Exception:
            pass
        try:
            self._write_terminal_snapshot(session)
        except Exception:
            pass

    def _confirm_one_locked(
        self,
        session: UniversalAgentSessionState,
        confirmation: Mapping[str, Any],
    ) -> Any:
        session.confirm_stage = "validating_confirmation"
        self._validate_and_consume_confirmation(session, confirmation)
        authority = session.confirmation_authority
        assert authority is not None
        graph = session.task_graph
        observation = session.trusted_observation
        decision = session.qwen_decision
        assert graph is not None and observation is not None and decision is not None

        session.confirm_stage = "policy_recheck"
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

        session.confirm_stage = "pre_execute_evidence"
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
        session.confirm_stage = "executing"
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
            session.status = (
                "needs_reobservation"
                if int(exc.physical_actions) == 0
                else "failed"
            )
            session.failed_reason = str(exc)
            try:
                self._write_terminal_snapshot(session)
            except Exception:
                pass
            raise

        self._remember(
            session,
            getattr(result, "evidence", ()),
            getattr(result, "after_frame_paths", ()),
        )
        session.confirm_stage = "validating_execution_result"
        physical_actions = int(result.physical_actions)
        wait_transition = result.resolved_action.kind == "wait_for_change"
        if physical_actions != 1 and not (wait_transition and physical_actions == 0):
            session.physical_actions += max(0, physical_actions)
            session.status = "failed"
            session.failed_reason = (
                "已确认动作必须产生一次物理动作，或仅 wait_for_change 产生零动作；"
                f"实际返回：{physical_actions}。"
            )
            raise UniversalAgentOrchestratorError(session.failed_reason)
        session.physical_actions += physical_actions

        requested_action = decision.proposal.action
        rebound_params = dict(getattr(result.rebound_action, "params", {}) or {})
        resolved_kind = str(getattr(result.resolved_action, "kind", ""))
        resolved_target_binding_ok = True
        if resolved_kind in {
            "tap_semantic",
            "dismiss_overlay",
            "input_verified_text",
            "clear_verified_text",
            "long_press",
        }:
            resolved_target_binding_ok = (
                str(getattr(result.resolved_action, "target_element_id", "") or "")
                == str(rebound_params.get("element_id") or "")
            )
        elif resolved_kind == "drag":
            resolved_target_binding_ok = (
                str(getattr(result.resolved_action, "target_element_id", "") or "")
                == str(rebound_params.get("source_element_id") or "")
                and str(
                    getattr(result.resolved_action, "destination_element_id", "")
                    or ""
                )
                == str(rebound_params.get("destination_element_id") or "")
            )
        if (
            requested_action is None
            or _action_digest(requested_action) != authority.action_digest
            or _action_digest(result.requested_action) != authority.action_digest
            or str(getattr(result.requested_action, "node_id", ""))
            != authority.decision_node_id
            or str(getattr(result.rebound_action, "node_id", ""))
            != authority.decision_node_id
            or str(getattr(result.resolved_action, "node_id", ""))
            != authority.decision_node_id
            or str(getattr(result.resolved_action, "kind", ""))
            != str(getattr(result.rebound_action, "action", ""))
            or not resolved_target_binding_ok
            or dict(getattr(result.resolved_action, "expected_effect", {}) or {})
            != dict(
                getattr(result.rebound_action, "params", {}).get(
                    "expected_effect", {}
                )
                or {}
            )
        ):
            session.status = "failed"
            session.failed_reason = (
                "执行结果没有严格绑定 confirmed/requested/rebound/resolved 动作链。"
            )
            raise UniversalAgentOrchestratorError(session.failed_reason)
        if (
            getattr(result, "controller_completion_evidence", ())
            and result.resolved_action.expected_effect.get(
                "goal_complete_on_success"
            )
            is not True
        ):
            session.status = "failed"
            session.failed_reason = (
                "控制器完成证据缺少 resolved expected_effect 的一次性完成声明。"
            )
            raise UniversalAgentOrchestratorError(session.failed_reason)
        session.status = "verifying"
        session.confirm_stage = "validating_post_action_evidence"
        action_outcome = str(getattr(result, "action_outcome", ""))
        verification_errors = tuple(
            str(item)
            for item in getattr(result, "verification_errors", ())
            if str(item).strip()
        )
        if action_outcome not in POST_ACTION_OUTCOMES:
            session.status = "failed"
            session.failed_reason = f"动作结果 outcome 无效：{action_outcome}。"
            raise UniversalAgentOrchestratorError(session.failed_reason)
        if (action_outcome == "matched") == bool(verification_errors):
            session.status = "failed"
            session.failed_reason = (
                "动作结果 outcome 与 verification_errors 不一致。"
            )
            raise UniversalAgentOrchestratorError(session.failed_reason)
        if (
            result.planned_scene_fingerprint != observation.fingerprint
            or result.confirmation_frame_identity_verified is not True
        ):
            session.status = "failed"
            session.failed_reason = "动作结果没有绑定确认 scope 的规划画面。"
            raise UniversalAgentOrchestratorError(session.failed_reason)
        if result.resolved_action.before_fingerprint != result.before_scene.fingerprint:
            session.status = "failed"
            session.failed_reason = "动作结果没有绑定复核后的执行前画面。"
            raise UniversalAgentOrchestratorError(session.failed_reason)
        after_frames = tuple(getattr(result, "after_frames", ()))
        after_paths = tuple(
            str(item).strip() for item in getattr(result, "after_frame_paths", ())
        )
        if (
            len(after_frames) < 4
            or len(after_paths) != len(after_frames)
            or any(not item for item in after_paths)
            or len(set(after_paths)) != len(after_paths)
        ):
            session.status = "failed"
            session.failed_reason = "动作后可信观察缺少完整且唯一的原始帧证据。"
            raise UniversalAgentOrchestratorError(session.failed_reason)
        if (
            action_outcome == "matched"
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

        session.confirm_stage = "building_trusted_observation"
        new_observation = self.trusted_observation_factory(
            frames=list(after_frames),
            device_id=session.device_id,
            scene=result.after_scene,
            observation_id=f"obs_{uuid.uuid4().hex}",
        )
        if (
            str(getattr(new_observation, "device_id", "")) != session.device_id
            or str(getattr(new_observation, "fingerprint", ""))
            != result.after_scene.fingerprint
            or str(getattr(getattr(new_observation, "scene", None), "fingerprint", ""))
            != result.after_scene.fingerprint
        ):
            session.status = "failed"
            session.failed_reason = "动作后可信观察未严格绑定 device/after scene fingerprint。"
            raise UniversalAgentOrchestratorError(session.failed_reason)
        if (
            new_observation.observation_id == observation.observation_id
            or (
                action_outcome == "matched"
                and
                result.resolved_action.kind != "wait_for_change"
                and new_observation.fingerprint == observation.fingerprint
            )
        ):
            session.status = "failed"
            session.failed_reason = "动作后可信观察 observation/fingerprint 未更新。"
            raise UniversalAgentOrchestratorError(session.failed_reason)
        session.confirm_stage = "persisting_post_observation"
        session.trusted_observation = new_observation
        session.trusted_frames = after_frames
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
                "before_observation_id": observation.observation_id,
                "before_fingerprint": observation.fingerprint,
                "after_observation_id": new_observation.observation_id,
                "after_fingerprint": new_observation.fingerprint,
            }
        )
        try:
            session.status = "replanning"
            session.confirm_stage = "replanning"
            self._advance_after_observation(
                session,
                result=result,
                before_observation=observation,
                new_observation=new_observation,
            )
            session.confirm_stage = "completed"
            self._write_terminal_snapshot(session)
            return result
        except EvidenceStoreError as exc:
            session.status = "failed"
            session.failed_reason = str(exc)
            raise
        except Exception as exc:
            session.status = "failed"
            session.failed_reason = str(exc)
            self._record_deepseek_failure(
                session,
                exc,
                stage="post_action_replan",
            )
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

            if impact in {"read_only", "navigation_only"}:
                initial_safe = graph.active_subgoal()
                revised, visible_advances = self._advance_visible_presence_prefix(
                    session,
                    graph=graph,
                    trusted_observation=observation,
                )
                if visible_advances:
                    graph = revised
                if (
                    impact == "read_only"
                    and not visible_advances
                    and initial_safe is not None
                    and not self._is_presence_only_read_only_subgoal(
                        initial_safe
                    )
                ):
                    observed = self.bridge.observed_state(
                        graph=graph,
                        trusted_observation=observation,
                        action_outcome="not_applicable",
                        verification={
                            "visible_evidence": [scene.summary],
                            "blocked_reasons": [],
                        },
                    )
                    revised = self.deepseek_planner.replan(
                        graph,
                        observed,
                        trigger="observation_changed",
                        reason=(
                            "初始可信画面不能直接证明当前 read_only 结果。"
                            "如果目标页面或区域尚未出现，必须先修订为一个"
                            " navigation_only 中间状态；不得请求低层动作、"
                            "不得直接宣称结果完成。"
                        ),
                    )
                    self._validate_graph_identity(
                        revised,
                        device_id=session.device_id,
                        previous=graph,
                        trusted_observation=observation,
                    )
                    self._store_revised_graph(session, revised)
                    graph = revised
                    visible_advances = 1
                if not visible_advances and impact == "read_only":
                    session.status = "blocked"
                    session.failed_reason = (
                        "当前 read_only 子目标不是可由唯一完整可见元素证明的"
                        "定位目标，或当前画面证据不唯一；未请求物理动作。"
                    )
                    self._write_terminal_snapshot(session)
                    return session
                if visible_advances:
                    if revised.status == "completed":
                        session.status = "succeeded"
                        session.failed_reason = ""
                        self._write_terminal_snapshot(session)
                        return session
                    current = revised.active_subgoal()
                    impact = (
                        current.external_impact if current is not None else "unknown"
                    )
                    if current is None:
                        session.status = "blocked"
                        session.failed_reason = "可见状态证据推进后没有活动子目标。"
                        self._write_terminal_snapshot(session)
                        return session
                    if impact in {"external_state", "unknown"}:
                        session.status = "awaiting_risk_confirmation"
                        session.failed_reason = ""
                        self._bind_risk_confirmation(session)
                        self._write_terminal_snapshot(session)
                        return session
                    if impact == "read_only":
                        session.status = "needs_reobservation"
                        session.failed_reason = (
                            "同一可信画面最多推进一个可见状态子目标；"
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
                self._review_completion_candidate(
                    session,
                    graph=graph,
                    trusted_observation=observation,
                    decision=decision,
                    reason="Qwen 在当前可信画面中提出完成候选，要求 DeepSeek 复核。",
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
            self._record_deepseek_failure(
                session,
                exc,
                stage=(
                    "initial_task_graph"
                    if session.task_graph is None
                    else "start_replan"
                ),
            )
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

    VERSION = "2026-08-16-universal-action-policy-v15"
    ALLOWED_ACTIONS = frozenset(
        {
            "swipe",
            "reveal_system_navigation",
            "back",
            "home",
            "wait_for_change",
            "tap_semantic",
            "dismiss_overlay",
            "input_verified_text",
            "clear_verified_text",
            "long_press",
            "drag",
        }
    )
    FORBIDDEN_ROLES = frozenset({"keyboard_key"})
    NAVIGATION_ROLES = frozenset(
        {"button", "icon", "text", "tab", "image", "list_item"}
    )
    GOAL_BOUND_TAP_ROLES = frozenset(
        {"button", "icon", "tab", "image", "list_item"}
    )
    GENERIC_BINDING_TERMS = frozenset(
        {
            "action",
            "button",
            "control",
            "current",
            "element",
            "icon",
            "image",
            "item",
            "page",
            "screen",
            "setup",
            "target",
            "view",
            "元素",
            "图标",
            "当前",
            "按钮",
            "控件",
            "入口",
            "操作",
            "目标",
            "画面",
            "视图",
            "页面",
        }
    )
    LOCAL_ACTION_LABEL_MARKER_GROUPS = (
        ("长按", "long_press", "longpress"),
        ("拖动", "drag"),
        ("输入", "input", "type"),
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

    @staticmethod
    def _parse_small_ordinal(value: str) -> int | None:
        text = str(value or "").strip().casefold()
        if text.isdigit():
            number = int(text)
            return number if 1 <= number <= 99 else None
        english = {
            "first": 1,
            "second": 2,
            "third": 3,
            "fourth": 4,
            "fifth": 5,
            "sixth": 6,
            "seventh": 7,
            "eighth": 8,
            "ninth": 9,
            "tenth": 10,
        }
        if text in english:
            return english[text]
        chinese_digits = {
            "零": 0,
            "〇": 0,
            "一": 1,
            "二": 2,
            "两": 2,
            "三": 3,
            "四": 4,
            "五": 5,
            "六": 6,
            "七": 7,
            "八": 8,
            "九": 9,
        }
        if text == "十":
            return 10
        if "十" in text:
            left, right = text.split("十", 1)
            tens = chinese_digits.get(left, 1 if left == "" else -1)
            ones = chinese_digits.get(right, 0 if right == "" else -1)
            number = tens * 10 + ones
            return number if 1 <= number <= 99 else None
        if text and all(character in chinese_digits for character in text):
            number = 0
            for character in text:
                number = number * 10 + chinese_digits[character]
            return number if 1 <= number <= 99 else None
        match = re.fullmatch(r"(\d+)(?:st|nd|rd|th)", text)
        if match:
            number = int(match.group(1))
            return number if 1 <= number <= 99 else None
        return None

    @classmethod
    def _vertical_list_ordinal(cls, task_context: Any) -> int | None:
        current = cls._value(task_context, "current_subgoal", None)
        goal = cls._value(task_context, "goal", None)
        values: list[str] = []
        if isinstance(current, Mapping):
            values.extend(cls._structured_strings(current.get("objective")))
            values.extend(
                cls._structured_strings(current.get("completion_conditions"))
            )
        elif isinstance(goal, Mapping):
            values.extend(cls._structured_strings(goal.get("objective")))
        if isinstance(goal, Mapping):
            entities = goal.get("entities")
            if isinstance(entities, Mapping):
                for key in ("target_ordinal", "ordinal", "target_index"):
                    values.extend(cls._structured_strings(entities.get(key)))
        text = " ".join(values).casefold()
        if not re.search(r"(?:列表|清单|\blist\b)", text):
            return None
        chinese = re.search(
            r"第([零〇一二两三四五六七八九十\d]+)(?:项|个|条|行|入口|选项)",
            text,
        )
        if chinese:
            return cls._parse_small_ordinal(chinese.group(1))
        english = re.search(
            r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|\d+(?:st|nd|rd|th)?)\s+"
            r"(?:item|entry|option|row)\b",
            text,
        )
        return cls._parse_small_ordinal(english.group(1)) if english else None

    def _ordinal_binding_error(
        self,
        *,
        task_context: Any,
        scene: Any,
        element: Any,
        action_kind: str,
    ) -> str:
        ordinal = self._vertical_list_ordinal(task_context)
        if ordinal is None or action_kind != "tap_semantic":
            return ""
        if element.role not in {"list_item", "button"}:
            return "序数列表目标必须绑定 list_item 或同列 button。"
        if element.states.get("fully_visible") is not True:
            return "序数列表目标必须完整可见。"
        el, _et, er, _eb = element.bounds
        element_width = max(1e-9, er - el)
        peers = []
        for candidate in scene.elements:
            if (
                candidate.role != element.role
                or float(candidate.confidence) < self.min_confidence
                or candidate.states.get("visible") is False
                or candidate.states.get("fully_visible") is not True
            ):
                continue
            cl, ct, cr, cb = candidate.bounds
            candidate_width = max(1e-9, cr - cl)
            horizontal_overlap = max(0.0, min(er, cr) - max(el, cl))
            width_ratio = candidate_width / element_width
            if (
                horizontal_overlap / min(element_width, candidate_width) < 0.75
                or not 0.65 <= width_ratio <= 1.54
            ):
                continue
            peers.append((float(ct + cb) / 2.0, candidate))
        peers.sort(key=lambda item: (item[0], item[1].element_id))
        if len(peers) < ordinal:
            return (
                "序数列表目标缺少完整可见的前序同列兄弟项："
                f"需要第{ordinal}项，仅证明{len(peers)}项。"
            )
        if peers[ordinal - 1][1].element_id != element.element_id:
            return "候选按可信几何从上到下排序后不在任务指定序位。"
        return ""

    def _is_exact_literal_local_action_label(
        self,
        *,
        task_context: Any,
        action: Any,
        element: Any,
    ) -> bool:
        """Recognize a quoted gesture word as a UI label, never as authority.

        DeepSeek keeps these literals only in ``target_ui_label``.  This narrow
        exception lets a normal tap enter that labelled local mode while all
        other destructive/account semantics remain fail-closed.
        """

        if str(self._value(action, "action", "")) not in {
            "tap_semantic",
            "long_press",
        }:
            return False
        goal = self._value(task_context, "goal", None)
        if not isinstance(goal, Mapping):
            return False
        entities = goal.get("entities")
        if not isinstance(entities, Mapping):
            return False
        literal = str(entities.get("target_ui_label") or "").strip()
        action_kind = str(self._value(action, "action", ""))
        if (
            not literal
            and str(entities.get("capability_trial_kind") or "") == action_kind
            and action_kind == "long_press"
        ):
            # A deterministic capability trial intentionally carries only the
            # primitive kind in its task graph.  The visual label must still be
            # copied byte-for-byte by Qwen and bound to the observed element;
            # destructive words left after removing the gesture marker remain
            # forbidden below.
            literal = str(element.label or "").strip()
        if not literal or str(element.label or "").strip() != literal:
            return False
        if str(action.params.get("label") or "").strip() != literal:
            return False
        if str(action.params.get("target") or "").strip() != str(
            element.meaning or ""
        ).strip():
            return False
        normalized = literal.casefold()
        matched_groups = tuple(
            group
            for group in self.LOCAL_ACTION_LABEL_MARKER_GROUPS
            if any(marker in normalized for marker in group)
        )
        return len(matched_groups) == 1

    def _is_goal_bound_local_capability_entry(
        self,
        *,
        task_context: Any,
        action: Any,
        element: Any,
    ) -> bool:
        """Identify a tap target whose label names a local capability.

        This classification grants no primitive-action authority.  It only
        lets the later goal-bound navigation checks distinguish tapping a
        visible entry named ``input``/``drag``/``long press`` from executing
        the named action.  The later checks still require unique semantic
        binding, an empty risk graph, and a verifiable navigation result.
        """

        if (
            str(self._value(action, "action", "")) != "tap_semantic"
            or str(
                self._value(task_context, "current_external_impact", "")
            )
            != "navigation_only"
            or bool(
                self._value(task_context, "external_action_allowed", False)
            )
            or element.role not in self.GOAL_BOUND_TAP_ROLES
            or element.states.get("goal_relevant") is not True
            or element.states.get("fully_visible") is not True
        ):
            return False
        if str(action.params.get("label") or "").strip() != str(
            element.label or ""
        ).strip():
            return False
        if str(action.params.get("target") or "").strip() != str(
            element.meaning or ""
        ).strip():
            return False
        requested_states = action.params.get("states")
        if not isinstance(requested_states, dict) or requested_states != element.states:
            return False
        normalized = str(element.label or "").strip().casefold()
        matched_groups = tuple(
            group
            for group in self.LOCAL_ACTION_LABEL_MARKER_GROUPS
            if any(marker in normalized for marker in group)
        )
        return len(matched_groups) == 1

    @staticmethod
    def _without_literal_local_action_markers(values: Any) -> tuple[str, ...]:
        sanitized: list[str] = []
        for value in PhaseOneNavigationPolicy._structured_strings(values):
            current = value.casefold()
            for marker in (
                "long_press",
                "longpress",
                "drag",
                "input",
                "type",
                "长按",
                "拖动",
                "输入",
            ):
                current = current.replace(marker, " ")
            sanitized.append(current)
        return tuple(sanitized)

    @staticmethod
    def _has_unresolved_candidate_conflict(
        conflicts: Any,
        element_id: str,
    ) -> bool:
        for conflict in conflicts or ():
            if not isinstance(conflict, dict):
                if element_id in str(conflict):
                    return True
                continue
            conflict_ids = conflict.get("element_ids") or []
            resolved_duplicate = (
                conflict.get("kind") == "duplicate_visual_object_collapsed"
                and conflict.get("canonical_element_id") == element_id
                and element_id in conflict_ids
            )
            if resolved_duplicate:
                continue
            if element_id in conflict_ids or element_id in str(conflict):
                return True
        return False

    @classmethod
    def _structured_strings(cls, value: Any) -> tuple[str, ...]:
        if isinstance(value, str):
            text = value.strip()
            return (text,) if text else ()
        if isinstance(value, Mapping):
            result: list[str] = []
            for item in value.values():
                result.extend(cls._structured_strings(item))
            return tuple(result)
        if isinstance(value, (list, tuple)):
            result = []
            for item in value:
                result.extend(cls._structured_strings(item))
            return tuple(result)
        return ()

    @classmethod
    def _binding_terms(cls, values: Any) -> set[str]:
        terms: set[str] = set()
        chinese_generic_terms = tuple(
            term
            for term in cls.GENERIC_BINDING_TERMS
            if re.fullmatch(r"[\u3400-\u9fff]+", term)
        )
        for value in cls._structured_strings(values):
            normalized = value.casefold()
            terms.update(
                token
                for token in re.findall(r"[a-z0-9]+", normalized)
                if len(token) >= 2 and token not in cls.GENERIC_BINDING_TERMS
            )
            for sequence in re.findall(r"[\u3400-\u9fff]+", normalized):
                for generic in chinese_generic_terms:
                    sequence = sequence.replace(generic, "")
                terms.update(
                    sequence[index : index + 2]
                    for index in range(len(sequence) - 1)
                    if sequence[index : index + 2]
                    not in cls.GENERIC_BINDING_TERMS
                )
        return terms

    @classmethod
    def _explicit_target_exclusion_error(
        cls,
        *,
        task_context: Any,
        element: Any,
    ) -> str:
        """Reject a visible target named inside an explicit negative constraint.

        This is a generic local safety check.  It does not plan an alternative
        action or recognize any App/page; it only prevents Qwen from selecting
        an element whose visible semantics overlap a user-provided prohibition.
        """

        current_subgoal = cls._value(task_context, "current_subgoal", None)
        constraint_values: list[str] = list(
            cls._structured_strings(cls._value(task_context, "constraints", ()))
        )
        if isinstance(current_subgoal, Mapping):
            constraint_values.extend(
                cls._structured_strings(current_subgoal.get("constraints"))
            )
        if constraint_excludes_candidate(
            constraint_values,
            (
                cls._value(element, "meaning", ""),
                cls._value(element, "label", ""),
                cls._value(element, "evidence", ()),
            ),
            candidate_role=str(cls._value(element, "role", "")),
        ):
            return "当前候选与任务明确排除的可见目标语义重叠。"
        return ""

    @staticmethod
    def _has_structured_postcondition(expected: Any, scene: Any) -> bool:
        if not isinstance(expected, Mapping) or expected.get("allow_unchanged") is True:
            return False
        if any(
            expected.get(key) is True
            for key in ("scene_changed", "content_changed", "current_video_changed")
        ):
            return True
        expected_app = str(expected.get("app_id") or "").strip()
        if expected_app and expected_app != scene.foreground_app_id:
            return True
        expected_screen = str(expected.get("screen_id") or "").strip()
        if expected_screen and expected_screen != scene.screen_id:
            return True
        element_state = expected.get("element_state")
        return bool(
            isinstance(element_state, Mapping)
            and str(element_state.get("meaning") or "").strip()
            and isinstance(element_state.get("states"), Mapping)
            and element_state["states"]
        )

    def _goal_bound_navigation_fallback_error(
        self,
        *,
        task_context: Any,
        trusted_observation: Any,
        action: Any,
        element: Any,
        scene: Any,
    ) -> str:
        if str(self._value(task_context, "current_external_impact", "")) != "navigation_only":
            return "通用目标绑定回退只允许 navigation_only 子目标。"
        if element.role not in self.GOAL_BOUND_TAP_ROLES:
            return "通用目标绑定回退要求候选具有明确可点击角色。"
        relevance = element.states.get("goal_relevant")
        if relevance is False:
            return "通用目标绑定回退拒绝明确 goal_relevant=false 的候选。"
        requested_states = action.params.get("states")
        if not isinstance(requested_states, dict) or requested_states != element.states:
            return "通用目标绑定回退要求动作逐项复用候选 states。"

        conflicts = self._value(trusted_observation, "candidate_conflicts", None)
        if not isinstance(conflicts, (list, tuple)):
            return "通用目标绑定回退缺少候选冲突证据。"
        if self._has_unresolved_candidate_conflict(conflicts, element.element_id):
            return "通用目标绑定回退候选存在语义冲突或不唯一。"
        requested_label = str(action.params.get("label") or "").strip().casefold()
        requested_role = str(action.params.get("role") or "").strip().casefold()
        requested_target = str(action.params.get("target") or "").strip().casefold()
        eligible = tuple(
            candidate
            for candidate in scene.elements
            if float(candidate.confidence) >= self.min_confidence
            and candidate.states.get("visible") is not False
            and (
                candidate.states.get("goal_relevant") is True
                if relevance is True
                else candidate.states.get("goal_relevant") is not False
            )
            and (
                not requested_label
                or candidate.label.strip().casefold() == requested_label
            )
            and (
                not requested_role
                or candidate.role.strip().casefold() == requested_role
            )
            and (
                not requested_target
                or requested_target
                in {
                    candidate.meaning.strip().casefold(),
                    candidate.label.strip().casefold(),
                }
            )
            and candidate.states == requested_states
        )
        if len(eligible) != 1 or eligible[0].element_id != element.element_id:
            return "通用目标绑定回退要求动作完整语义绑定下只有一个高置信候选。"

        current_subgoal = self._value(task_context, "current_subgoal", None)
        goal = self._value(task_context, "goal", None)
        if not isinstance(current_subgoal, Mapping) or not isinstance(goal, Mapping):
            return "通用目标绑定回退缺少结构化目标或当前子目标。"
        if (
            str(current_subgoal.get("external_impact") or "") != "navigation_only"
            or str(current_subgoal.get("status") or "") != "active"
        ):
            return "通用目标绑定回退的当前子目标状态或影响类型无效。"
        risk_actions = self._value(task_context, "risk_actions", None)
        risk_ids = current_subgoal.get("risk_action_ids")
        if (
            bool(self._value(task_context, "external_action_allowed", False))
            or not isinstance(risk_actions, (list, tuple))
            or risk_actions
            or not isinstance(risk_ids, (list, tuple))
            or risk_ids
        ):
            return "通用目标绑定回退要求任务与当前子目标均无风险动作。"

        entities = goal.get("entities")
        objective = str(current_subgoal.get("objective") or "").strip()
        completion_conditions = current_subgoal.get("completion_conditions")
        if (
            not isinstance(entities, Mapping)
            or not entities
            or not objective
            or not isinstance(completion_conditions, (list, tuple))
            or not completion_conditions
        ):
            return "通用目标绑定回退缺少结构化目标实体、子目标或完成条件。"
        candidate_values = (
            element.meaning,
            element.label,
            *element.evidence,
        )
        literal_local_action_label = self._is_exact_literal_local_action_label(
            task_context=task_context,
            action=action,
            element=element,
        )
        local_capability_entry = self._is_goal_bound_local_capability_entry(
            task_context=task_context,
            action=action,
            element=element,
        )
        candidate_terms = self._binding_terms(candidate_values)
        entity_terms = self._binding_terms(entities)
        subgoal_values = (
            objective,
            completion_conditions,
        )
        subgoal_terms = self._binding_terms(subgoal_values)
        scene_terms = self._binding_terms((scene.summary,))
        subgoal_bound = bool(candidate_terms.intersection(subgoal_terms))
        entity_bound = bool(candidate_terms.intersection(entity_terms)) or bool(
            subgoal_bound and scene_terms.intersection(entity_terms)
        )
        if (
            not candidate_terms
            or not entity_bound
            or (
                not literal_local_action_label
                and not subgoal_bound
            )
        ):
            return "通用目标绑定回退无法证明候选同时绑定目标实体与当前子目标。"
        safety_values = (
            candidate_values,
            entities,
            subgoal_values,
        )
        if literal_local_action_label:
            # The raw goal and negative constraints (for example "不得提交")
            # have already passed the independent risk audit.  Re-scanning
            # those strings as a positive action would create a false denial.
            # Keep this exception candidate-local and retain the exact label.
            safety_values = (
                candidate_values,
                {"target_ui_label": entities.get("target_ui_label")},
            )
        elif local_capability_entry:
            # Goal/subgoal binding and the empty risk graph were proved above.
            # Inspect only the observed entry here so negative task wording
            # such as "unsubmitted" cannot masquerade as the tap's effect.
            safety_values = (candidate_values,)
        safety_strings = self._structured_strings(safety_values)
        if local_capability_entry:
            task_literals = self._structured_strings((entities, subgoal_values))
            if task_literals:
                safety_strings = tuple(
                    self._strip_exact_literals(value, task_literals)
                    for value in safety_strings
                )
        sibling_literal_labels = tuple(
            candidate.label.strip()
            for candidate in scene.elements
            if candidate.element_id != element.element_id
            and candidate.label.strip()
        )
        if sibling_literal_labels:
            safety_strings = tuple(
                self._strip_exact_literals(value, sibling_literal_labels)
                for value in safety_strings
            )
        if literal_local_action_label or local_capability_entry:
            safety_strings = self._without_literal_local_action_markers(
                safety_strings
            )
        if any(self._contains_control_instruction(value) for value in safety_strings):
            return "通用目标绑定回退检测到标签外控制指令。"
        if self._semantic_class(*safety_strings) == "forbidden":
            return "通用目标绑定回退检测到外部状态、破坏、账号或交易语义。"
        if action_has_account_effect(action):
            return "通用目标绑定回退检测到账号或外部状态动作。"
        if not self._has_structured_postcondition(
            action.params.get("expected_effect"),
            scene,
        ):
            return "通用目标绑定回退缺少可由新画面验证的结构化动作后预期。"
        return ""

    @staticmethod
    def _strip_exact_literals(value: str, literals: tuple[str, ...]) -> str:
        result = str(value or "")
        for literal in sorted(set(literals), key=len, reverse=True):
            result = result.replace(literal, "")
        return result

    @staticmethod
    def _contains_control_instruction(value: str) -> bool:
        text = str(value or "").casefold()
        return bool(
            re.search(
                r"(?:请|建议|应当|应该|需要|然后|随后|直接|先)"
                r".{0,12}(?:点击|滑动|拖动|拖拽|长按|提交|发送|保存|登录)"
                r"|(?:点击|滑动|拖动|拖拽|长按)(?:这个|该|目标|按钮|后|到)"
                r"|\b(?:please|should|must|then|next)\b.{0,24}"
                r"\b(?:tap|click|swipe|drag|press|submit|send|save|login)\b",
                text,
            )
        )

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

        if action_kind == "reveal_system_navigation":
            if impact != "navigation_only":
                return self._deny(
                    "系统导航栏唤出动作只允许 navigation_only 子目标。"
                )
            current_subgoal = self._value(task_context, "current_subgoal", {})
            risk_ids = self._value(current_subgoal, "risk_action_ids", ()) or ()
            if not isinstance(risk_ids, (list, tuple)) or risk_ids:
                return self._deny("系统导航栏唤出动作要求当前子目标没有风险动作。")
            if set(action.params) - {"expected_effect"}:
                return self._deny(
                    "系统导航栏唤出动作不能携带坐标、方向、距离或其他参数。"
                )
            if action.params.get("expected_effect") != {
                "system_ui": {"navigation_bar_visible": True}
            }:
                return self._deny(
                    "系统导航栏唤出动作缺少精确的结构化导航栏可见后置条件。"
                )
            system_ui = getattr(scene, "system_ui", None)
            if system_ui is None:
                return self._deny(
                    "系统导航栏唤出动作缺少结构化 scene.system_ui。"
                )
            immersive = getattr(system_ui, "immersive_or_fullscreen", None)
            navigation_visible = getattr(
                system_ui,
                "navigation_bar_visible",
                None,
            )
            if isinstance(system_ui, Mapping):
                if immersive is None:
                    immersive = system_ui.get("immersive_or_fullscreen")
                if navigation_visible is None:
                    navigation_visible = system_ui.get("navigation_bar_visible")
            if (
                immersive is not True
                or navigation_visible is not False
            ):
                return self._deny(
                    "系统导航栏唤出动作要求当前画面明确处于沉浸态且导航栏隐藏。"
                )
            region = self._value(decision, "target_region", None)
            if (
                region is None
                or str(self._value(region, "kind", "")) != "system_navigation"
                or str(self._value(region, "element_id", ""))
                or tuple(self._value(region, "bounds", ()))
                != (0.0, 0.0, 1.0, 1.0)
            ):
                return self._deny(
                    "系统导航栏唤出动作必须绑定整屏 system_navigation 区域。"
                )
            return NavigationPolicyDecision(
                True,
                "允许一次无坐标的Android系统导航栏唤出动作。",
                "reveal_system_navigation",
            )
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
            source_container_error = compact_drag_source_container_error(
                scene,
                source,
            )
            if source_container_error:
                return self._deny(source_container_error)
            if source.role in self.FORBIDDEN_ROLES or destination.role in self.FORBIDDEN_ROLES:
                return self._deny("拖动端点不能使用禁止进入通用动作的角色。")
            conflicts = self._value(trusted_observation, "candidate_conflicts", ()) or ()
            if any(
                self._has_unresolved_candidate_conflict(conflicts, element.element_id)
                for element in (source, destination)
            ):
                return self._deny("拖动起点或终点存在语义冲突或不唯一。")
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
            endpoint_safety_strings = self._without_literal_local_action_markers(
                tuple(
                    value
                    for element in (source, destination)
                    for value in (element.meaning, element.label)
                )
            )
            if (
                self._semantic_class(*endpoint_safety_strings) == "forbidden"
                and not external_allowed
            ):
                return self._deny("拖动端点包含外部状态、输入或破坏性语义。")
            return NavigationPolicyDecision(True, "允许一个双候选语义拖动。", "drag")

        element_id = str(action.params.get("element_id") or "").strip()
        try:
            element = scene.get_element(element_id, min_confidence=self.min_confidence)
        except UISceneError as exc:
            return self._deny(f"当前可信观察不能唯一解析候选：{exc}")
        if element.role in self.FORBIDDEN_ROLES:
            return self._deny(f"候选角色 {element.role} 不允许进入通用动作。")
        if action_kind in {"input_verified_text", "clear_verified_text"}:
            if element.role != "input" or element.states.get("focused") is not True:
                return self._deny("输入或清空动作要求最新画面证明 input 候选已聚焦。")
            if element.states.get("goal_relevant") is not True:
                return self._deny("输入动作要求最新画面证明 input 候选与当前目标相关。")
            if action_kind == "clear_verified_text" and (
                not isinstance(element.states.get("value"), str)
                or not element.states.get("value")
            ):
                return self._deny("精确文字清空要求最新画面确认非空输入值。")
            if element.states.get("keyboard_layout") != "qwerty":
                return self._deny("精确文字输入要求最新画面确认 QWERTY 键盘。")
            input_step = None
            if action_kind == "input_verified_text":
                try:
                    input_step = plan_next_verified_input(
                        action.params.get("text"),
                        element.states.get("value"),
                    )
                except (ValueError, VerifiedTextTransactionError) as exc:
                    return self._deny(f"无法建立精确文字输入事务：{exc}")
                if input_step is None:
                    return self._deny("输入框已经逐字等于目标文字，不得重复输入。")
                if input_step.kind == "literal_key":
                    return self._deny("下一字符需要独立可见键位审计，禁止猜测输入。")
                if (
                    element.states.get("keyboard_input_mode")
                    != input_step.required_mode
                ):
                    return self._deny("当前键盘模式与下一确定性文字分段不一致。")
                if element.states.get("ime_preedit_text"):
                    return self._deny("当前仍有未完成输入法组合，禁止继续键入。")
                if (
                    input_step.required_case_mode
                    and element.states.get("keyboard_case_mode")
                    != input_step.required_case_mode
                ):
                    return self._deny("当前键盘大小写状态与下一英文分段不一致。")
            elif element.states.get("keyboard_input_mode") != "direct_latin":
                return self._deny("精确文字清空要求 direct_latin 键盘证据。")
            eligible_inputs = tuple(
                candidate
                for candidate in scene.elements
                if candidate.role == "input"
                and float(candidate.confidence) >= self.min_confidence
                and candidate.states.get("visible") is not False
                and candidate.states.get("goal_relevant") is True
                and candidate.states.get("focused") is True
                and (
                    isinstance(candidate.states.get("value"), str)
                    and (
                        action_kind == "input_verified_text"
                        or bool(candidate.states.get("value"))
                    )
                )
                and candidate.states.get("keyboard_layout") == "qwerty"
                and candidate.states.get("keyboard_input_mode")
                == (
                    input_step.required_mode
                    if input_step is not None
                    else "direct_latin"
                )
                and (
                    input_step is None
                    or not input_step.required_case_mode
                    or candidate.states.get("keyboard_case_mode")
                    == input_step.required_case_mode
                )
                and (
                    action_kind != "input_verified_text"
                    or not candidate.states.get("ime_preedit_text")
                )
            )
            if len(eligible_inputs) != 1 or eligible_inputs[0].element_id != element.element_id:
                return self._deny("输入动作要求唯一符合安全条件的目标输入框。")
            if action_kind == "input_verified_text":
                text = action.params.get("text")
                if (
                    not isinstance(text, str)
                    or not text
                    or len(text) > 100
                    or "\n" in text
                    or "\r" in text
                ):
                    return self._deny("输入文字格式无效。")
            else:
                if impact != "navigation_only":
                    return self._deny("精确文字清空只允许 navigation_only 子目标。")
                expected = action.params.get("expected_effect")
                if not isinstance(expected, dict) or expected.get("element_state") != {
                    "meaning": element.meaning,
                    "states": {"value": ""},
                }:
                    return self._deny("精确文字清空缺少绑定原输入框的空值后置条件。")
                return NavigationPolicyDecision(
                    True,
                    "允许按当前精确非空值清空唯一已聚焦目标输入框。",
                    "clear_verified_text",
                )
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

        exclusion_error = self._explicit_target_exclusion_error(
            task_context=task_context,
            element=element,
        )
        if exclusion_error:
            return self._deny(exclusion_error)

        ordinal_error = self._ordinal_binding_error(
            task_context=task_context,
            scene=scene,
            element=element,
            action_kind=action_kind,
        )
        if ordinal_error:
            return self._deny(ordinal_error)

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
        if self._has_unresolved_candidate_conflict(conflicts, element.element_id):
            return self._deny("当前候选存在语义冲突或不唯一。")

        if (
            action_kind == "tap_semantic"
            and element.meaning == "ime_exact_candidate"
        ):
            states = element.states
            if (
                impact != "navigation_only"
                or element.role != "button"
                or states.get("ime_candidate") is not True
                or states.get("fully_visible") is not True
                or states.get("goal_relevant") is not True
                or float(element.confidence) < 0.9
            ):
                return self._deny("输入法候选缺少本轮唯一、完整、高置信审计证据。")
            target_text = self._value(task_context, "requested_input_text", None)
            if target_text is None:
                goal_value = self._value(task_context, "goal", {})
                goal_entities = (
                    goal_value.get("entities")
                    if isinstance(goal_value, Mapping)
                    else None
                )
                if isinstance(goal_entities, Mapping):
                    target_text = goal_entities.get("input_text")
            prior_value = states.get("prior_input_value")
            try:
                input_step = plan_next_verified_input(target_text, prior_value)
            except (ValueError, VerifiedTextTransactionError) as exc:
                return self._deny(f"输入法候选无法绑定精确文字事务：{exc}")
            if (
                input_step is None
                or input_step.kind != "chinese_pinyin"
                or element.label != input_step.segment
                or states.get("expected_input_value") != input_step.expected_value
                or states.get("pinyin") != input_step.pinyin
            ):
                return self._deny("输入法候选与本地下一中文分段不一致。")
            expected = action.params.get("expected_effect")
            if not isinstance(expected, dict) or expected.get("element_state") != {
                "meaning": "application_text_input",
                "states": {"value": input_step.expected_value},
            }:
                return self._deny("输入法候选缺少绑定应用输入框精确前缀的后置条件。")
            return NavigationPolicyDecision(
                True,
                "允许选择本轮拼音组合中唯一逐字一致的中文候选。",
                "ime_exact_candidate",
            )

        if (
            action_kind == "tap_semantic"
            and element.meaning == "input_exact_literal_key"
        ):
            states = element.states
            target_text = self._value(task_context, "requested_input_text", None)
            if target_text is None:
                goal_value = self._value(task_context, "goal", {})
                goal_entities = goal_value.get("entities") if isinstance(goal_value, Mapping) else None
                if isinstance(goal_entities, Mapping):
                    target_text = goal_entities.get("input_text")
            prior_value = states.get("prior_input_value")
            try:
                input_step = plan_next_verified_input(target_text, prior_value)
            except (ValueError, VerifiedTextTransactionError) as exc:
                return self._deny(f"可见逐键候选无法绑定精确文字事务：{exc}")
            if (
                impact != "navigation_only"
                or element.role != "button"
                or float(element.confidence) < 0.9
                or states.get("goal_relevant") is not True
                or states.get("fully_visible") is not True
                or states.get("input_literal_key") is not True
                or input_step is None
                or input_step.kind != "literal_key"
                or states.get("key_value") != input_step.segment
                or states.get("expected_input_value") != input_step.expected_value
            ):
                return self._deny("可见逐键候选与本地下一字符事务不一致。")
            expected = action.params.get("expected_effect")
            if not isinstance(expected, dict) or expected.get("element_state") != {
                "meaning": "application_text_input",
                "states": {"value": input_step.expected_value},
            }:
                return self._deny("可见逐键候选缺少精确输入值后置条件。")
            return NavigationPolicyDecision(
                True,
                "允许点击本轮唯一完整可见且逐字绑定的下一字符键。",
                "input_exact_literal_key",
            )

        if (
            action_kind == "tap_semantic"
            and element.meaning == "switch_keyboard_layout"
        ):
            states = element.states
            target_text = self._value(task_context, "requested_input_text", None)
            if target_text is None:
                goal_value = self._value(task_context, "goal", {})
                goal_entities = goal_value.get("entities") if isinstance(goal_value, Mapping) else None
                if isinstance(goal_entities, Mapping):
                    target_text = goal_entities.get("input_text")
            prior_value = states.get("prior_input_value")
            try:
                input_step = plan_next_verified_input(target_text, prior_value)
            except (ValueError, VerifiedTextTransactionError) as exc:
                return self._deny(f"布局切换候选无法绑定精确文字事务：{exc}")
            desired_layout = (
                "numeric" if input_step and input_step.segment.isdecimal()
                else "qwerty" if input_step and (
                    input_step.kind in {"direct_latin", "chinese_pinyin"}
                    or input_step.segment == " "
                    or input_step.segment.isalpha()
                )
                else "symbol"
            )
            if (
                impact != "navigation_only"
                or element.role != "button"
                or float(element.confidence) < 0.9
                or states.get("goal_relevant") is not True
                or states.get("fully_visible") is not True
                or states.get("keyboard_layout_switch") is not True
                or input_step is None
                or states.get("next_input_value") != input_step.segment
                or states.get("target_layout") != desired_layout
                or states.get("current_layout") == desired_layout
            ):
                return self._deny("键盘布局切换没有绑定下一字符所需的唯一方向。")
            expected = action.params.get("expected_effect")
            if not isinstance(expected, dict) or expected.get("element_state") != {
                "meaning": "application_text_input",
                "states": {
                    "value": input_step.current_text,
                    "keyboard_layout": desired_layout,
                },
            }:
                return self._deny("键盘布局切换缺少保持输入值并到达目标布局的后置条件。")
            return NavigationPolicyDecision(
                True,
                "允许切换到下一逐键字符所需的已审计键盘布局。",
                "switch_keyboard_layout",
            )

        if (
            action_kind == "tap_semantic"
            and element.meaning == "switch_keyboard_case"
        ):
            states = element.states
            target_text = self._value(task_context, "requested_input_text", None)
            if target_text is None:
                goal_value = self._value(task_context, "goal", {})
                goal_entities = goal_value.get("entities") if isinstance(goal_value, Mapping) else None
                if isinstance(goal_entities, Mapping):
                    target_text = goal_entities.get("input_text")
            prior_value = states.get("prior_input_value")
            try:
                input_step = plan_next_verified_input(target_text, prior_value)
            except (ValueError, VerifiedTextTransactionError) as exc:
                return self._deny(f"大小写切换候选无法绑定精确文字事务：{exc}")
            if (
                impact != "navigation_only"
                or element.role != "button"
                or float(element.confidence) < 0.9
                or states.get("goal_relevant") is not True
                or states.get("fully_visible") is not True
                or states.get("keyboard_case_switch") is not True
                or input_step is None
                or input_step.kind != "direct_latin"
                or not input_step.required_case_mode
                or states.get("target_mode") != input_step.required_case_mode
                or states.get("current_mode") == input_step.required_case_mode
            ):
                return self._deny("大小写切换没有绑定下一大写英文分段。")
            expected = action.params.get("expected_effect")
            if not isinstance(expected, dict) or expected.get("element_state") != {
                "meaning": "application_text_input",
                "states": {
                    "value": input_step.current_text,
                    "keyboard_case_mode": input_step.required_case_mode,
                },
            }:
                return self._deny("大小写切换缺少保持输入值并到达大写状态的后置条件。")
            return NavigationPolicyDecision(
                True,
                "允许为下一英文分段切换唯一已审计的大小写状态。",
                "switch_keyboard_case",
            )

        if action_kind == "tap_semantic" and element.role == "input":
            if element.states.get("focused") is True:
                return self._deny(
                    "输入框已由当前画面证明聚焦，禁止再次点击制造冗余物理动作。"
                )
            return NavigationPolicyDecision(
                True,
                "允许对一个精确可信输入候选执行本地聚焦。",
                "focus_input",
            )
        if (
            action_kind == "tap_semantic"
            and element.meaning == "switch_keyboard_input_mode"
        ):
            states = element.states
            if (
                impact != "navigation_only"
                or element.role not in {"button", "icon"}
                or states.get("keyboard_input_mode_switch") is not True
                or states.get("current_mode") != "chinese_pinyin"
                or states.get("target_mode") != "direct_latin"
            ):
                return self._deny("键盘模式切换候选缺少从中文拼音到英文直输的可信状态。")
            focused_inputs = tuple(
                candidate
                for candidate in scene.elements
                if candidate.role == "input"
                and float(candidate.confidence) >= self.min_confidence
                and candidate.states.get("focused") is True
                and candidate.states.get("goal_relevant") is True
                and candidate.states.get("value") == ""
                and candidate.states.get("keyboard_layout") == "qwerty"
                and candidate.states.get("keyboard_input_mode") == "chinese_pinyin"
            )
            if len(focused_inputs) != 1:
                return self._deny("键盘模式切换要求唯一空白、已聚焦的中文拼音 QWERTY 输入框。")
            left, top, right, bottom = element.bounds
            if (
                top < 0.72
                or right - left > 0.2
                or bottom - top > 0.12
                or right <= left
                or bottom <= top
            ):
                return self._deny("键盘模式切换候选不在可信的底部紧凑按键区域。")
            visible = " ".join(
                [element.label, *element.evidence]
            ).strip().casefold()
            if not visible or not (
                "中" in visible or "chinese" in visible or "中文" in visible
            ):
                return self._deny("键盘模式切换候选缺少可见中文模式证据。")
            return NavigationPolicyDecision(
                True,
                "允许把唯一空白目标输入框从中文拼音切换到英文直输；动作后必须重新观察。",
                "switch_keyboard_input_mode",
            )
        clear_claimed = (
            element.meaning == "clear_local_text"
            or str(action.params.get("target") or "") == "clear_local_text"
        )
        if clear_claimed and element.states.get("local_text_clear") is not True:
            return self._deny("本地文字清空候选缺少可信 local_text_clear 状态。")
        if action_kind == "tap_semantic" and element.states.get("local_text_clear") is True:
            if impact != "navigation_only" or element.role not in {"button", "icon"}:
                return self._deny("本地文字清空控件只允许用于 navigation_only 的独立按钮或图标。")
            if element.label.strip().casefold() not in {"×", "✕", "✖", "x"}:
                return self._deny("本地文字清空候选缺少逐字可见的 × 图形。")
            visible_semantics = " ".join(
                [element.meaning, element.label, *element.evidence]
            ).casefold()
            if "取消" in visible_semantics or re.search(
                r"\bcancel(?:led|ing)?\b", visible_semantics
            ):
                return self._deny("文字取消控件不能冒充本地文字清空图标。")
            focused_inputs = tuple(
                candidate
                for candidate in scene.elements
                if candidate.role == "input"
                and float(candidate.confidence) >= self.min_confidence
                and candidate.states.get("focused") is True
                and candidate.states.get("goal_relevant") is True
                and isinstance(candidate.states.get("value"), str)
                and bool(candidate.states.get("value"))
                and candidate.states.get("keyboard_layout")
                in {"qwerty", "numeric", "symbol", "unknown"}
                and candidate.states.get("visible") is not False
            )
            if len(focused_inputs) != 1:
                return self._deny("本地文字清空要求唯一非空、已聚焦的目标输入框。")
            input_element = focused_inputs[0]
            il, it, ir, ib = input_element.bounds
            el, et, er, eb = element.bounds
            vertical_overlap = max(0.0, min(ib, eb) - max(it, et))
            input_height = max(1e-9, ib - it)
            element_width = er - el
            element_height = max(1e-9, eb - et)
            gap = max(0.0, el - ir)
            geometrically_bound = (
                vertical_overlap / element_height >= 0.6
                and el >= il + 0.4 * (ir - il)
                and er <= min(1.0, ir + 0.2)
                and gap <= max(0.04, input_height)
                and element_width <= 2.0 * input_height
                and element_height <= 1.5 * input_height
            )
            if not geometrically_bound:
                return self._deny("本地文字清空控件没有与唯一目标输入框形成可信几何绑定。")
            return NavigationPolicyDecision(
                True,
                "允许清空唯一已聚焦输入框中的本地临时文字。",
                "clear_local_text",
            )
        canonical = self._semantic_class(
            element.meaning,
            element.label,
            str(action.params.get("target") or ""),
        )
        local_capability_entry = self._is_goal_bound_local_capability_entry(
            task_context=task_context,
            action=action,
            element=element,
        )
        if canonical == "forbidden":
            if impact == "external_state" and external_allowed:
                canonical = "external"
            elif action_kind in {"input_verified_text", "clear_verified_text"}:
                canonical = "input"
            elif self._is_exact_literal_local_action_label(
                task_context=task_context,
                action=action,
                element=element,
            ):
                sanitized = self._without_literal_local_action_markers(
                    (
                        element.meaning,
                        element.label,
                        str(action.params.get("target") or ""),
                    )
                )
                sanitized_class = self._semantic_class(*sanitized)
                if sanitized_class == "forbidden":
                    return self._deny("候选包含外部状态、输入或破坏性语义。")
                canonical = (
                    "long_press"
                    if action_kind == "long_press"
                    else sanitized_class
                )
            elif local_capability_entry:
                sanitized = self._without_literal_local_action_markers(
                    (
                        element.meaning,
                        element.label,
                        str(action.params.get("target") or ""),
                    )
                )
                if self._semantic_class(*sanitized) == "forbidden":
                    return self._deny("候选包含外部状态、输入或破坏性语义。")
                # Do not authorize from the sanitized label.  An empty class
                # deliberately falls through to the complete goal-bound tap
                # proof below.
                canonical = ""
            else:
                return self._deny("候选包含外部状态、输入或破坏性语义。")
        if canonical == "refresh":
            if (
                impact != "navigation_only"
                or action_kind != "tap_semantic"
                or element.role not in {"button", "icon"}
                or element.meaning != "reload"
                or float(element.confidence) < 0.90
                or element.states.get("goal_relevant") is not True
                or element.states.get("fully_visible") is not True
                or element.states.get("reload_visual_audit") is not True
            ):
                return self._deny(
                    "刷新候选缺少当前目标关联、本地图标簇审计、完整可见或高置信证据。"
                )
            requested_states = action.params.get("states")
            if not isinstance(requested_states, dict) or requested_states != element.states:
                return self._deny("刷新动作没有逐项复用本地审计 states。")
        if not canonical:
            if action_kind in {"input_verified_text", "clear_verified_text"}:
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
            elif action_kind == "tap_semantic":
                fallback_error = self._goal_bound_navigation_fallback_error(
                    task_context=task_context,
                    trusted_observation=trusted_observation,
                    action=action,
                    element=element,
                    scene=scene,
                )
                if fallback_error:
                    return self._deny(
                        "本地策略无法证明候选属于通用导航语义或动作语义："
                        + fallback_error
                    )
                canonical = "goal_bound_tap"
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
