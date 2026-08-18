from __future__ import annotations

import json
import os
import re
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Iterable

from PIL import Image

from generic_scene_observer import _local_frame_fingerprint, _safe_goal_context
from constraint_target_filter import constraint_excludes_candidate
from generic_step_planner import (
    ALLOWED_STEP_ACTIONS,
    GenericStepPlanningError,
    GenericStepProposal,
    _reject_raw_control_data,
)
from observation_images import (
    LocalFrameStability,
    measure_frame_sharpness,
    measure_local_stability,
)
from message_intent import (
    subgoal_binds_recipient,
    subgoal_targets_recipient_control,
)
from qwen_runtime_errors import classify_qwen_error, failure_diagnostics
from semantic_executor import SemanticAction
from ui_scene import MIN_TARGET_CONFIDENCE, UIElement, UIScene, UISceneError
from universal_action_controller import UniversalActionController, UniversalActionError
from verified_text_transaction import (
    VerifiedTextTransactionError,
    plan_next_verified_input,
)
from vision_agent import VisionAgentError, _image_data_url
from vision_model_config import public_model_identity


QWEN_VISUAL_DECISION_PROTOCOL_VERSION = "2026-08-14-qwen-visual-decision-v5"
QWEN_VISUAL_SELECTION_PROTOCOL_VERSION = "2026-08-16-qwen-visual-selection-v2"
SUPPORTED_TASK_CONTEXT_PROTOCOL = "2026-08-11-deepseek-task-graph-v3"
MIGRATION_TASK_CONTEXT_PROTOCOL = "2026-08-11-deepseek-task-graph-v2"
SUPPORTED_TASK_CONTEXT_PROTOCOLS = frozenset(
    {SUPPORTED_TASK_CONTEXT_PROTOCOL, MIGRATION_TASK_CONTEXT_PROTOCOL}
)
QWEN_VISUAL_DECISION_MODEL_ROLE = "trusted_observation_single_step_selector"
DECISION_TIMEOUT_SECONDS = 60.0
DECISION_OUTPUT_TOKENS = 1800
DECISION_RETRY_TOKENS = 0
MIN_DECISION_CONFIDENCE = 0.72
MIN_TRUSTED_FRAME_SHARPNESS = 4.0
SINGLE_ELEMENT_ACTIONS = frozenset(
    {
        "tap_semantic", "dismiss_overlay", "input_verified_text",
        "clear_verified_text", "long_press",
    }
)
QWEN_PROTOCOL_ACTIONS = frozenset(ALLOWED_STEP_ACTIONS)

TASK_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
DEVICE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
OBSERVATION_ID_PATTERN = re.compile(r"^obs_[A-Za-z0-9]{16,64}$")
ALLOWED_TASK_STATUSES = {
    "ready",
    "running",
    "awaiting_confirmation",
    "completed",
    "blocked",
}
ALLOWED_EXTERNAL_IMPACTS = {
    "read_only",
    "navigation_only",
    "external_state",
    "unknown",
}
ACTIONABLE_EXACT_TEXT_ROLES = frozenset(
    {"button", "icon", "input", "tab", "toggle", "list_item", "keyboard_key"}
)
ROLE_PRIORITY = {
    "input": 100,
    "button": 95,
    "icon": 90,
    "keyboard_key": 85,
    "list_item": 80,
    "tab": 75,
    "toggle": 75,
    "dialog": 60,
    "text": 40,
    "image": 35,
    "container": 10,
    "unknown": 0,
}


def _structured_system_ui(scene: UIScene) -> dict[str, Any] | None:
    facts = getattr(scene, "system_ui", None)
    if facts is None:
        return None
    immersive = getattr(facts, "immersive_or_fullscreen", None)
    navigation_visible = getattr(facts, "navigation_bar_visible", None)
    if isinstance(facts, Mapping):
        if immersive is None:
            immersive = facts.get("immersive_or_fullscreen")
        if navigation_visible is None:
            navigation_visible = facts.get("navigation_bar_visible")
    return {
        "immersive_or_fullscreen": immersive,
        "navigation_bar_visible": navigation_visible,
    }

TRANSITION_COMPLETION_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"(?:已经|已|完成|成功|重新)(?:刷新|重新加载|加载|更新|同步|导航|跳转|进入|返回|切换|打开|启动|重新获取|重新读取|重新连接)",
        r"(?:刷新|重新加载|更新|同步|导航|跳转|进入|返回|切换|打开|启动|重新获取|重新读取|重新连接)(?:已经|已|完成|成功)",
        r"\b(?:has|have|was|were|is)\s+(?:been\s+)?(?:refreshed|reloaded|updated|synchronized|navigated|redirected|entered|returned|switched|opened|launched|retrieved|refetched|reacquired)\b",
        r"\b(?:refresh|reload|update|sync|navigation|redirect|retrieval|refetch|reacquisition)\s+(?:completed|complete|succeeded|occurred)\b",
        r"\b(?:re)?fetch(?:ed|es|ing)?\s+(?:the\s+)?latest\b",
    )
)
EXPLICIT_TRANSITION_EVIDENCE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"动作(?:回执|结果|已执行|执行成功)",
        r"(?:刷新|重新加载|加载|更新|同步|导航|跳转|进入|返回|切换|打开|启动|重新获取|重新读取|重新连接)(?:已经|已)?(?:成功|完成)",
        r"(?:页面|内容|画面|fingerprint|observation)(?:已经|已)?(?:发生变化|变化|改变|更新)",
        r"前后(?:画面|观察|fingerprint|observation)",
        r"刚刚更新",
        r"\b(?:action receipt|action result|fingerprint changed|observation changed|scene changed|content changed)\b",
        r"\b(?:refresh|reload|update|sync|navigation|redirect|retrieval|refetch)\s+(?:completed|complete|succeeded|successful)\b",
        r"\bupdated just now\b",
        r"\bbefore\b.{0,80}\bafter\b",
    )
)


def _current_subgoal_requires_transition_evidence(context: "QwenTaskContext") -> bool:
    completion_conditions = _text_tuple(
        context.current_subgoal.get("completion_conditions") or [],
        "current_subgoal.completion_conditions",
    )
    texts = completion_conditions or (
        str(context.current_subgoal.get("objective") or ""),
    )
    return any(
        pattern.search(text)
        for text in texts
        for pattern in TRANSITION_COMPLETION_PATTERNS
    )


def _contains_explicit_transition_evidence(texts: Iterable[str]) -> bool:
    return any(
        pattern.search(str(text))
        for text in texts
        for pattern in EXPLICIT_TRANSITION_EVIDENCE_PATTERNS
    )


def _finished_has_transition_evidence(
    context: "QwenTaskContext",
    observation: "TrustedObservation",
    evidence_ids: tuple[str, ...],
) -> bool:
    prior_evidence = _text_tuple(
        context.current_subgoal.get("completion_evidence") or [],
        "current_subgoal.completion_evidence",
    )
    if _contains_explicit_transition_evidence(prior_evidence):
        return True
    visible_evidence: list[str] = []
    for evidence_id in evidence_ids:
        if evidence_id == "scene":
            visible_evidence.extend(
                [observation.scene.summary, *observation.scene.overlays]
            )
            continue
        element = observation.get_candidate(evidence_id)
        visible_evidence.extend(
            [element.meaning, element.label, *element.evidence]
        )
    return _contains_explicit_transition_evidence(visible_evidence)


@dataclass(frozen=True)
class QwenTaskContext:
    protocol_version: str
    task_id: str
    device_id: str
    revision: int
    task_status: str
    goal: dict[str, Any]
    global_constraints: tuple[str, ...]
    goal_completion_conditions: tuple[dict[str, Any], ...]
    current_subgoal: dict[str, Any]
    current_external_impact: str
    risk_actions: tuple[dict[str, Any], ...]
    confirmation_gate: dict[str, Any]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "QwenTaskContext":
        if not isinstance(value, dict):
            raise VisionAgentError("Qwen任务上下文必须是JSON对象。")
        required = {
            "protocol_version",
            "task_id",
            "device_id",
            "revision",
            "task_status",
            "goal",
            "global_constraints",
            "goal_completion_conditions",
            "current_subgoal",
            "current_external_impact",
            "risk_actions",
            "confirmation_gate",
        }
        missing = required - set(value)
        unexpected = set(value) - required
        if missing:
            raise VisionAgentError(
                "Qwen任务上下文缺少字段：" + ", ".join(sorted(missing))
            )
        if unexpected:
            raise VisionAgentError(
                "Qwen任务上下文包含协议外字段："
                + ", ".join(sorted(unexpected))
            )

        context = cls(
            protocol_version=str(value["protocol_version"] or "").strip(),
            task_id=str(value["task_id"] or "").strip(),
            device_id=str(value["device_id"] or "").strip(),
            revision=value["revision"],
            task_status=str(value["task_status"] or "").strip(),
            goal=_require_dict(value["goal"], "goal"),
            global_constraints=_text_tuple(
                value["global_constraints"], "global_constraints"
            ),
            goal_completion_conditions=_dict_tuple(
                value["goal_completion_conditions"],
                "goal_completion_conditions",
            ),
            current_subgoal=_require_dict(
                value["current_subgoal"], "current_subgoal"
            ),
            current_external_impact=str(
                value["current_external_impact"] or ""
            ).strip(),
            risk_actions=_dict_tuple(value["risk_actions"], "risk_actions"),
            confirmation_gate=_require_dict(
                value["confirmation_gate"], "confirmation_gate"
            ),
        )
        context.validate()
        return context

    def validate(self) -> None:
        if self.protocol_version not in SUPPORTED_TASK_CONTEXT_PROTOCOLS:
            raise VisionAgentError(
                f"不支持的DeepSeek任务上下文协议：{self.protocol_version}"
            )
        if not TASK_ID_PATTERN.fullmatch(self.task_id):
            raise VisionAgentError(f"task_id 格式无效：{self.task_id!r}")
        if not DEVICE_ID_PATTERN.fullmatch(self.device_id):
            raise VisionAgentError(f"device_id 格式无效：{self.device_id!r}")
        if (
            isinstance(self.revision, bool)
            or not isinstance(self.revision, int)
            or self.revision < 1
        ):
            raise VisionAgentError("revision 必须是正整数。")
        if self.task_status not in ALLOWED_TASK_STATUSES:
            raise VisionAgentError(f"task_status 无效：{self.task_status}")
        if self.current_external_impact not in ALLOWED_EXTERNAL_IMPACTS:
            raise VisionAgentError(
                f"current_external_impact 无效：{self.current_external_impact}"
            )

        subgoal_allowed = {
            "subgoal_id",
            "objective",
            "status",
            "depends_on",
            "constraints",
            "completion_conditions",
            "completion_evidence",
            "risk_action_ids",
            "external_impact",
        }
        unexpected_subgoal = set(self.current_subgoal) - subgoal_allowed
        if unexpected_subgoal:
            raise VisionAgentError(
                "current_subgoal 包含协议外字段："
                + ", ".join(sorted(unexpected_subgoal))
            )
        if not str(self.current_subgoal.get("subgoal_id") or "").strip():
            raise VisionAgentError("current_subgoal 缺少 subgoal_id。")
        if not str(self.current_subgoal.get("objective") or "").strip():
            raise VisionAgentError("current_subgoal 缺少 objective。")
        if str(self.current_subgoal.get("status") or "") != "active":
            raise VisionAgentError("Qwen入口只接受 status=active 的 current_subgoal。")
        if (
            str(self.current_subgoal.get("external_impact") or "")
            != self.current_external_impact
        ):
            raise VisionAgentError(
                "current_subgoal.external_impact 与顶层上下文不一致。"
            )
        _safe_goal_context(self.to_dict())

        risk_ids = [str(item.get("risk_id") or "").strip() for item in self.risk_actions]
        if any(not item for item in risk_ids) or len(risk_ids) != len(set(risk_ids)):
            raise VisionAgentError("risk_actions 含空ID或重复ID。")
        subgoal_risk_ids = _text_tuple(
            self.current_subgoal.get("risk_action_ids") or [],
            "current_subgoal.risk_action_ids",
        )
        if len(subgoal_risk_ids) != len(set(subgoal_risk_ids)):
            raise VisionAgentError("current_subgoal.risk_action_ids 含重复风险ID。")
        gate_allowed = {
            "required",
            "state",
            "risk_ids",
            "external_state_action_allowed",
        }
        if self.protocol_version == SUPPORTED_TASK_CONTEXT_PROTOCOL:
            gate_allowed.add("scope")
        if set(self.confirmation_gate) != gate_allowed:
            raise VisionAgentError("confirmation_gate 字段不完整或包含协议外字段。")
        required = self.confirmation_gate.get("required")
        allowed = self.confirmation_gate.get("external_state_action_allowed")
        if not isinstance(required, bool) or not isinstance(allowed, bool):
            raise VisionAgentError("confirmation_gate 布尔字段格式无效。")
        state = str(self.confirmation_gate.get("state") or "").strip()
        if state not in {"not_required", "awaiting_confirmation", "confirmed"}:
            raise VisionAgentError(f"confirmation_gate.state 无效：{state}")
        gate_risk_ids = _text_tuple(
            self.confirmation_gate.get("risk_ids") or [],
            "confirmation_gate.risk_ids",
        )
        if len(gate_risk_ids) != len(set(gate_risk_ids)):
            raise VisionAgentError("confirmation_gate.risk_ids 含重复风险ID。")
        confirmation_risk_ids = {
            str(item.get("risk_id") or "").strip()
            for item in self.risk_actions
            if item.get("confirmation_required") is True
        }
        if (
            set(gate_risk_ids) != confirmation_risk_ids
            or set(risk_ids) != set(subgoal_risk_ids)
        ):
            raise VisionAgentError(
                "risk_actions、current_subgoal 与 confirmation_gate 风险ID不一致。"
            )

        if self.protocol_version == SUPPORTED_TASK_CONTEXT_PROTOCOL:
            scope = _require_dict(
                self.confirmation_gate.get("scope"),
                "confirmation_gate.scope",
            )
            scope_allowed = {"task_id", "device_id", "revision", "subgoal_id"}
            if set(scope) != scope_allowed:
                raise VisionAgentError(
                    "confirmation_gate.scope 字段缺失或包含协议外字段。"
                )
            expected_scope = {
                "task_id": self.task_id,
                "device_id": self.device_id,
                "revision": self.revision,
                "subgoal_id": str(self.current_subgoal["subgoal_id"]),
            }
            for field, expected in expected_scope.items():
                if type(scope[field]) is not type(expected) or scope[field] != expected:
                    raise VisionAgentError(
                        f"confirmation_gate.scope.{field} 与当前上下文不一致。"
                    )

        external = self.current_external_impact in {"external_state", "unknown"}
        if self.current_external_impact == "unknown":
            raise VisionAgentError("unknown 子目标禁止进入视觉动作协议。")
        if external and confirmation_risk_ids:
            if not required or not gate_risk_ids:
                raise VisionAgentError("需确认的外部状态子目标必须关闭风险确认门。")
            if state not in {"awaiting_confirmation", "confirmed"}:
                raise VisionAgentError("外部状态子目标的确认门状态无效。")
            if state == "confirmed" and not allowed:
                raise VisionAgentError("确认门状态与 external_state_action_allowed 冲突。")
            if state != "confirmed" and allowed:
                raise VisionAgentError("未确认风险不能允许外部状态动作。")
        elif external:
            if required or gate_risk_ids or state != "not_required" or not allowed:
                raise VisionAgentError("自动外部效果的本地策略授权状态无效。")
        elif required or gate_risk_ids or allowed or state != "not_required":
            raise VisionAgentError("只读/导航子目标不得伪造风险确认状态。")

    @property
    def external_action_allowed(self) -> bool:
        return bool(
            self.protocol_version == SUPPORTED_TASK_CONTEXT_PROTOCOL
            and self.confirmation_gate["external_state_action_allowed"]
        )

    @property
    def pre_observation_block_reason(self) -> str | None:
        """Return the local gate that must run before either Qwen call."""

        if self.current_external_impact == "unknown":
            return "unknown 子目标禁止调用观察或决策模型。"
        if self.current_external_impact == "external_state":
            if self.protocol_version == MIGRATION_TASK_CONTEXT_PROTOCOL:
                return "v2迁移上下文缺少确认作用域，禁止调用观察或决策模型。"
            if not self.external_action_allowed:
                return "风险确认门未满足，本轮禁止调用观察或决策模型。"
        return None

    @property
    def exact_text_requirements(self) -> tuple[str, ...]:
        """Structured exact-text requirements supplied by the task graph.

        The visual selector must not infer an exact string from prose. When
        DeepSeek supplies one of these generic entity fields, local candidate
        matching becomes authoritative and is performed before Qwen is called.
        """

        entities = self.goal.get("entities") or {}
        if not isinstance(entities, dict):
            raise VisionAgentError("goal.entities 必须是JSON对象。")
        values: list[str] = []
        target_label = entities.get("target_ui_label")
        if target_label is not None:
            if not isinstance(target_label, str) or not target_label.strip():
                raise VisionAgentError("goal.entities.target_ui_label 格式无效。")
            values.append(target_label.strip())
        recipient = entities.get("recipient")
        if recipient is not None:
            if (
                not isinstance(recipient, str)
                or not recipient
                or len(recipient) > 100
                or recipient != recipient.strip()
                or "\n" in recipient
                or "\r" in recipient
            ):
                raise VisionAgentError("goal.entities.recipient 格式无效。")
            if subgoal_targets_recipient_control(recipient, self.current_subgoal):
                if recipient not in values:
                    values.append(recipient)
        return tuple(values)

    @property
    def identity_text_requirements(self) -> tuple[str, ...]:
        entities = self.goal.get("entities") or {}
        if not isinstance(entities, dict):
            raise VisionAgentError("goal.entities 必须是JSON对象。")
        recipient = entities.get("recipient")
        if recipient is None:
            return ()
        if (
            not isinstance(recipient, str)
            or not recipient
            or len(recipient) > 100
            or recipient != recipient.strip()
        ):
            raise VisionAgentError("goal.entities.recipient 格式无效。")
        if (
            subgoal_binds_recipient(recipient, self.current_subgoal)
            and not subgoal_targets_recipient_control(
                recipient,
                self.current_subgoal,
            )
        ):
            return (recipient,)
        return ()

    @property
    def exact_text_target_roles(self) -> tuple[str, ...]:
        # Role/meaning hints are observation facts, not task-authority fields.
        return ()

    @property
    def requested_input_text(self) -> str | None:
        """Return the exact text authorized by DeepSeek, never model-invented text."""

        entities = self.goal.get("entities") or {}
        raw = entities.get("input_text")
        if raw is None:
            return None
        if not isinstance(raw, str) or not raw or len(raw) > 100:
            raise VisionAgentError("goal.entities.input_text 必须为1～100个字符。")
        if "\n" in raw or "\r" in raw:
            raise VisionAgentError("goal.entities.input_text 不得包含换行。")
        return raw

    @property
    def exact_text_target_meanings(self) -> tuple[str, ...]:
        return ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "task_id": self.task_id,
            "device_id": self.device_id,
            "revision": self.revision,
            "task_status": self.task_status,
            "goal": dict(self.goal),
            "global_constraints": list(self.global_constraints),
            "goal_completion_conditions": [
                dict(item) for item in self.goal_completion_conditions
            ],
            "current_subgoal": dict(self.current_subgoal),
            "current_external_impact": self.current_external_impact,
            "risk_actions": [dict(item) for item in self.risk_actions],
            "confirmation_gate": dict(self.confirmation_gate),
        }

    def to_observation_context(self) -> dict[str, Any]:
        """Small read-only goal context for candidate discovery.

        The decision selector still receives ``to_dict()`` in full. This view
        removes task-graph bookkeeping that the observation model cannot use,
        reducing malformed or truncated scene JSON without hiding the active
        objective, entities, constraints, completion conditions, or device.
        """

        return {
            "device_id": self.device_id,
            "objective": str(self.current_subgoal.get("objective") or ""),
            "entities": dict(self.goal.get("entities") or {}),
            "constraints": [
                *self.global_constraints,
                *_text_tuple(
                    self.current_subgoal.get("constraints") or [],
                    "current_subgoal.constraints",
                ),
            ],
            "completion_conditions": list(
                self.current_subgoal.get("completion_conditions") or []
            ),
            "external_impact": self.current_external_impact,
        }


@dataclass(frozen=True)
class TrustedObservation:
    observation_id: str
    device_id: str
    fingerprint: str
    scene: UIScene
    local_stability: LocalFrameStability
    selected_frame_index: int
    frame_sharpness_scores: tuple[float, ...]
    candidate_aliases: tuple[tuple[str, str], ...] = ()
    candidate_conflicts: tuple[dict[str, Any], ...] = ()

    @classmethod
    def from_scene(
        cls,
        *,
        frames: list[Image.Image],
        device_id: str,
        scene: UIScene,
        observation_id: str | None = None,
    ) -> "TrustedObservation":
        if len(frames) < 4:
            raise VisionAgentError("可信观察至少需要4帧。")
        if not DEVICE_ID_PATTERN.fullmatch(str(device_id or "").strip()):
            raise VisionAgentError(f"可信观察 device_id 无效：{device_id!r}")
        stability = measure_local_stability(frames, allow_leading_outlier=True)
        if not stability.stable:
            raise VisionAgentError(
                f"本地多帧稳定性检查未通过：{stability.reason}；不能建立可信观察。"
            )
        sharpness = tuple(measure_frame_sharpness(frame) for frame in frames)
        # GenericSceneObserver permits one stale leading camera frame and only
        # exposes a fingerprint from the converged three-frame tail.  Reusing
        # the leading sample here could make the same read-only capture reject
        # itself merely because a transient overlay looked sharper.
        stable_tail_start = max(0, len(frames) - min(3, len(frames)))
        selected = max(
            range(stable_tail_start, len(frames)),
            key=sharpness.__getitem__,
        )
        sharpness_floor = float(
            os.environ.get(
                "ROBOT_LOCAL_FRAME_SHARPNESS_MIN",
                str(MIN_TRUSTED_FRAME_SHARPNESS),
            )
        )
        if sharpness[selected] < sharpness_floor:
            raise VisionAgentError(
                "当前最清晰帧仍然模糊："
                f"sharpness={sharpness[selected]:.3f} < {sharpness_floor:.3f}。"
            )
        fingerprint = _local_frame_fingerprint(frames[selected].convert("RGB"))
        scene.validate()
        canonical_scene, aliases, conflicts = _canonicalize_trusted_scene(scene)
        target_local_candidate = _trusted_target_local_candidate(
            canonical_scene,
            conflicts,
        )
        if not scene.stable or (
            float(scene.confidence) < MIN_TARGET_CONFIDENCE
            and target_local_candidate is None
            and not canonical_scene.trusted_completion_evidence()
        ):
            raise VisionAgentError("页面不稳定或整体置信度不足，不能建立可信候选。")
        if scene.fingerprint != fingerprint:
            raise VisionAgentError(
                "只读观察 fingerprint 与当前本地帧不一致，拒绝建立可信候选。"
            )
        resolved_id = observation_id or f"obs_{uuid.uuid4().hex}"
        if not OBSERVATION_ID_PATTERN.fullmatch(resolved_id):
            raise VisionAgentError(f"observation_id 格式无效：{resolved_id!r}")
        result = cls(
            observation_id=resolved_id,
            device_id=str(device_id).strip(),
            fingerprint=fingerprint,
            scene=canonical_scene,
            local_stability=stability,
            selected_frame_index=selected,
            frame_sharpness_scores=sharpness,
            candidate_aliases=aliases,
            candidate_conflicts=conflicts,
        )
        result.validate_against_frames(frames, allow_leading_outlier=True)
        return result

    def validate_against_frames(
        self,
        frames: list[Image.Image],
        *,
        allow_leading_outlier: bool = False,
    ) -> None:
        if len(frames) < 4:
            raise VisionAgentError("新鲜度校验至少需要4帧。")
        stability = measure_local_stability(
            frames,
            allow_leading_outlier=allow_leading_outlier,
        )
        if not stability.stable:
            raise VisionAgentError(
                f"当前画面已不稳定：{stability.reason}；旧观察失效。"
            )
        sharpness = [measure_frame_sharpness(frame) for frame in frames]
        eligible_start = (
            max(0, len(frames) - min(3, len(frames)))
            if allow_leading_outlier
            else 0
        )
        selected = max(
            range(eligible_start, len(frames)),
            key=sharpness.__getitem__,
        )
        sharpness_floor = float(
            os.environ.get(
                "ROBOT_LOCAL_FRAME_SHARPNESS_MIN",
                str(MIN_TRUSTED_FRAME_SHARPNESS),
            )
        )
        if sharpness[selected] < sharpness_floor:
            raise VisionAgentError("当前新鲜画面仍然模糊，旧动作失效。")
        current_fingerprint = _local_frame_fingerprint(
            frames[selected].convert("RGB")
        )
        if current_fingerprint != self.fingerprint:
            raise VisionAgentError("当前画面 fingerprint 已变化，旧动作失效。")
        if self.scene.fingerprint != self.fingerprint:
            raise VisionAgentError("可信观察内部 fingerprint 不一致。")

    def get_candidate(self, element_id: str) -> UIElement:
        return self.scene.get_element(element_id)

    def target_local_candidate(self) -> UIElement | None:
        """Return the sole conflict-free goal element usable on a dynamic page."""

        return _trusted_target_local_candidate(
            self.scene,
            self.candidate_conflicts,
        )

    def prompt_dict(self) -> dict[str, Any]:
        candidates: list[dict[str, Any]] = []
        for item in self.scene.elements:
            value = item.to_dict()
            value["bounds"] = [round(part * 1000) for part in item.bounds]
            candidates.append(value)
        value = {
            "observation_id": self.observation_id,
            "device_id": self.device_id,
            "fingerprint": self.fingerprint,
            "foreground_app_id": self.scene.foreground_app_id,
            "screen_id": self.scene.screen_id,
            "summary": self.scene.summary,
            "overlays": list(self.scene.overlays),
            "scene_confidence": float(self.scene.confidence),
            "candidate_bounds_scale": 1000,
            "candidates": candidates,
            "candidate_aliases": dict(self.candidate_aliases),
            "candidate_conflicts": [dict(item) for item in self.candidate_conflicts],
        }
        system_ui = _structured_system_ui(self.scene)
        if system_ui is not None:
            value["system_ui"] = system_ui
        return value

    def to_dict(self) -> dict[str, Any]:
        scene = self.scene.to_dict()
        system_ui = _structured_system_ui(self.scene)
        if system_ui is not None:
            scene["system_ui"] = system_ui
        return {
            "observation_id": self.observation_id,
            "device_id": self.device_id,
            "fingerprint": self.fingerprint,
            "scene": scene,
            "local_stability": self.local_stability.to_dict(),
            "selected_frame_index": self.selected_frame_index,
            "frame_sharpness_scores": [
                round(value, 3) for value in self.frame_sharpness_scores
            ],
            "candidate_aliases": dict(self.candidate_aliases),
            "candidate_conflicts": [dict(item) for item in self.candidate_conflicts],
        }


@dataclass(frozen=True)
class ModelPageState:
    foreground_app_id: str
    screen_id: str
    summary: str
    overlays: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: Any) -> "ModelPageState":
        if not isinstance(value, dict):
            raise GenericStepPlanningError("page_state 必须是JSON对象。")
        allowed = {"foreground_app_id", "screen_id", "summary", "overlays"}
        unexpected = set(value) - allowed
        if unexpected:
            raise GenericStepPlanningError(
                "Qwen page_state 只能描述页面，禁止携带候选元素："
                + ", ".join(sorted(unexpected))
            )
        overlays = value.get("overlays") or []
        if not isinstance(overlays, list):
            raise GenericStepPlanningError("page_state.overlays 必须是数组。")
        return cls(
            foreground_app_id=str(value.get("foreground_app_id") or "unknown")[:80],
            screen_id=str(value.get("screen_id") or "unknown")[:120],
            summary=str(value.get("summary") or "").strip()[:500],
            overlays=tuple(str(item).strip()[:120] for item in overlays if str(item).strip()),
        )

    @classmethod
    def from_trusted_scene(cls, scene: UIScene) -> "ModelPageState":
        return cls(
            foreground_app_id=scene.foreground_app_id,
            screen_id=scene.screen_id,
            summary=scene.summary,
            overlays=scene.overlays,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "foreground_app_id": self.foreground_app_id,
            "screen_id": self.screen_id,
            "summary": self.summary,
            "overlays": list(self.overlays),
        }


@dataclass(frozen=True)
class VisualTargetRegion:
    kind: str
    bounds: tuple[float, float, float, float]
    description: str
    element_id: str = ""
    destination_element_id: str = ""
    destination_bounds: tuple[float, float, float, float] | None = None

    def validate(self, observation: TrustedObservation, action: SemanticAction) -> None:
        if self.kind not in {"element", "element_path", "screen", "system_navigation"}:
            raise GenericStepPlanningError(f"不支持的目标区域类型：{self.kind}")
        if len(self.bounds) != 4:
            raise GenericStepPlanningError("目标区域 bounds 必须包含4个数值。")
        left, top, right, bottom = self.bounds
        if not (0.0 <= left < right <= 1.0 and 0.0 <= top < bottom <= 1.0):
            raise GenericStepPlanningError(f"目标区域超出归一化画面：{self.bounds}")
        if not self.description.strip():
            raise GenericStepPlanningError("目标区域缺少可读描述。")
        if action.action in SINGLE_ELEMENT_ACTIONS:
            if self.kind != "element":
                raise GenericStepPlanningError("元素动作必须绑定可信候选元素。")
            action_id = str(action.params.get("element_id") or "").strip()
            if not self.element_id or self.element_id != action_id:
                raise GenericStepPlanningError("目标区域 element_id 与动作不一致。")
            element = observation.get_candidate(self.element_id)
            if any(abs(a - b) > 0.0001 for a, b in zip(self.bounds, element.bounds)):
                raise GenericStepPlanningError(
                    "目标区域必须逐项复用可信候选的原始 bounds。"
                )
            if self.destination_element_id or self.destination_bounds is not None:
                raise GenericStepPlanningError("单元素动作不能携带拖动终点。")
        elif action.action == "drag":
            if self.kind != "element_path":
                raise GenericStepPlanningError("拖动动作必须绑定可信元素路径。")
            source_id = str(action.params.get("source_element_id") or "").strip()
            destination_id = str(
                action.params.get("destination_element_id") or ""
            ).strip()
            if (
                not source_id
                or not destination_id
                or self.element_id != source_id
                or self.destination_element_id != destination_id
            ):
                raise GenericStepPlanningError("拖动目标区域与动作元素不一致。")
            source = observation.get_candidate(source_id)
            destination = observation.get_candidate(destination_id)
            if any(abs(a - b) > 0.0001 for a, b in zip(self.bounds, source.bounds)):
                raise GenericStepPlanningError("拖动起点必须复用可信候选 bounds。")
            if self.destination_bounds is None or any(
                abs(a - b) > 0.0001
                for a, b in zip(self.destination_bounds, destination.bounds)
            ):
                raise GenericStepPlanningError("拖动终点必须复用可信候选 bounds。")
        else:
            if self.element_id or self.destination_element_id:
                raise GenericStepPlanningError("屏幕/系统动作不能伪造 element_id。")
            if self.destination_bounds is not None:
                raise GenericStepPlanningError("屏幕/系统动作不能携带拖动终点。")
            if self.bounds != (0.0, 0.0, 1.0, 1.0):
                raise GenericStepPlanningError("屏幕/系统动作只能描述整屏区域。")
            expected_kind = (
                "system_navigation"
                if action.action in {"back", "home", "reveal_system_navigation"}
                else "screen"
            )
            if self.kind != expected_kind:
                raise GenericStepPlanningError("动作与目标区域类型不一致。")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "element_id": self.element_id or None,
            "bounds": list(self.bounds),
            "destination_element_id": self.destination_element_id or None,
            "destination_bounds": (
                list(self.destination_bounds)
                if self.destination_bounds is not None
                else None
            ),
            "description": self.description,
        }


@dataclass(frozen=True)
class QwenVisualDecision:
    task_id: str
    device_id: str
    revision: int
    observation_id: str
    fingerprint: str
    page_state: ModelPageState
    trusted_observation: TrustedObservation
    proposal: GenericStepProposal
    target_region: VisualTargetRegion | None
    expected_result: dict[str, Any]
    confidence: float
    reason: str
    completion_evidence_element_ids: tuple[str, ...] = ()
    protocol_version: str = QWEN_VISUAL_DECISION_PROTOCOL_VERSION

    def validate(self, context: QwenTaskContext) -> None:
        self.validate_fresh(context, self.trusted_observation)
        self.proposal.validate(self.trusted_observation.scene)
        exact_text_block = _exact_text_candidate_block(
            context,
            self.trusted_observation,
        )
        if exact_text_block is not None:
            if self.proposal.status != "blocked":
                raise GenericStepPlanningError(
                    "逐字一致文字约束缺少本地唯一可信候选，必须 blocked。"
                )
            exact_candidate_ids: set[str] = set()
        else:
            exact_candidate_ids = _required_exact_candidate_ids(
                context,
                self.trusted_observation,
            )
        identity_candidate_ids: set[str] = set()
        identity_block = _identity_text_candidate_block(
            context,
            self.trusted_observation,
        )
        if identity_block is not None:
            if self.proposal.status != "blocked":
                raise GenericStepPlanningError(
                    "当前收件人身份缺少本地唯一逐字视觉证据，必须 blocked。"
                )
        else:
            identity_candidate_ids = _required_identity_candidate_ids(
                context,
                self.trusted_observation,
            )
        if self.protocol_version != QWEN_VISUAL_DECISION_PROTOCOL_VERSION:
            raise GenericStepPlanningError("Qwen视觉决策协议版本无效。")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise GenericStepPlanningError("Qwen视觉决策置信度必须在0到1之间。")
        if not isinstance(self.expected_result, dict):
            raise GenericStepPlanningError("expected_result 必须是JSON对象。")

        action = self.proposal.action
        if self.proposal.status == "action":
            if action is None or self.target_region is None:
                raise GenericStepPlanningError("唯一下一动作缺少可信目标区域。")
            if not self.expected_result:
                raise GenericStepPlanningError("唯一下一动作缺少可验证预期结果。")
            if (
                context.current_external_impact in {"external_state", "unknown"}
                and not context.external_action_allowed
            ):
                raise GenericStepPlanningError("风险确认门未满足，禁止产生外部状态动作。")
            self.target_region.validate(self.trusted_observation, action)
            if action.action in SINGLE_ELEMENT_ACTIONS:
                element = self.trusted_observation.get_candidate(
                    str(action.params.get("element_id") or "")
                )
                if exact_candidate_ids and element.element_id not in exact_candidate_ids:
                    raise GenericStepPlanningError(
                        "动作目标不是本地确认的逐字一致唯一候选。"
                    )
                expected_fields = {
                    "target": element.meaning,
                    "role": element.role,
                    "label": element.label,
                }
                for key, expected in expected_fields.items():
                    if str(action.params.get(key) or "") != expected:
                        raise GenericStepPlanningError(
                            f"动作未逐字复制可信候选 {key}。"
                        )
                requested_states = action.params.get("states") or {}
                if any(element.states.get(k) != v for k, v in requested_states.items()):
                    raise GenericStepPlanningError("动作 states 与可信候选不一致。")
                if action.action == "input_verified_text":
                    if element.role != "input":
                        raise GenericStepPlanningError("输入动作必须绑定 input 候选。")
                    authorized = context.requested_input_text
                    if authorized is None or action.params.get("text") != authorized:
                        raise GenericStepPlanningError(
                            "输入文字没有逐字复用DeepSeek结构化 input_text。"
                        )
                if action.action == "clear_verified_text":
                    if element.role != "input":
                        raise GenericStepPlanningError("清空动作必须绑定 input 候选。")
                    if action.params.get("text") is not None:
                        raise GenericStepPlanningError("清空动作不能携带模型生成的文字。")
                if action.action == "long_press":
                    duration_ms = action.params.get("duration_ms", 800)
                    if (
                        isinstance(duration_ms, bool)
                        or not isinstance(duration_ms, (int, float))
                        or not 500 <= float(duration_ms) <= 2000
                    ):
                        raise GenericStepPlanningError(
                            "长按 duration_ms 必须在500～2000之间。"
                        )
            elif action.action == "drag":
                for prefix in ("source_", "destination_"):
                    element = self.trusted_observation.get_candidate(
                        str(action.params.get(f"{prefix}element_id") or "")
                    )
                    for field, expected in {
                        "target": element.meaning,
                        "role": element.role,
                        "label": element.label,
                    }.items():
                        if str(action.params.get(f"{prefix}{field}") or "") != expected:
                            raise GenericStepPlanningError(
                                f"拖动动作未逐字复制可信候选 {prefix}{field}。"
                            )
                    requested_states = action.params.get(f"{prefix}states") or {}
                    if any(
                        element.states.get(k) != v
                        for k, v in requested_states.items()
                    ):
                        raise GenericStepPlanningError(
                            f"拖动动作 {prefix}states 与可信候选不一致。"
                        )
            if dict(action.params.get("expected_effect") or {}) != self.expected_result:
                raise GenericStepPlanningError("动作 expected_effect 与顶层预期不一致。")
            if float(self.confidence) < MIN_DECISION_CONFIDENCE:
                raise GenericStepPlanningError("动作置信度不足，必须 blocked。")
            try:
                UniversalActionController().resolve_one(
                    action,
                    self.trusted_observation.scene,
                    confirmed=True,
                )
            except UniversalActionError as exc:
                raise GenericStepPlanningError(f"本地控制器拒绝动作：{exc}") from exc
            if exact_candidate_ids and action.action not in {
                "tap_semantic",
                "dismiss_overlay",
            }:
                raise GenericStepPlanningError(
                    "存在逐字一致文字约束时，动作必须绑定该唯一候选。"
                )
        elif self.target_region is not None:
            raise GenericStepPlanningError("finished/blocked 不能携带动作目标区域。")

        if self.proposal.status == "finished":
            if not self.completion_evidence_element_ids:
                raise GenericStepPlanningError("finished 缺少可信完成证据ID。")
            if (
                _current_subgoal_requires_transition_evidence(context)
                and not _finished_has_transition_evidence(
                    context,
                    self.trusted_observation,
                    self.completion_evidence_element_ids,
                )
            ):
                raise GenericStepPlanningError(
                    "发生型完成条件不能由单帧静态视觉内容单独满足；"
                    "必须提供前后变化、动作回执或明确动态证据。"
                )
            if exact_candidate_ids and not exact_candidate_ids.issubset(
                set(self.completion_evidence_element_ids)
            ):
                raise GenericStepPlanningError(
                    "finished 未引用本地确认的逐字一致候选。"
                )
            if identity_candidate_ids and not identity_candidate_ids.issubset(
                set(self.completion_evidence_element_ids)
            ):
                raise GenericStepPlanningError(
                    "finished 未引用当前收件人的唯一逐字身份候选。"
                )
            _resolve_completion_evidence(
                self.completion_evidence_element_ids,
                self.trusted_observation,
            )
            if float(self.trusted_observation.scene.confidence) < MIN_TARGET_CONFIDENCE:
                allowed_evidence = {
                    item.element_id
                    for item in self.trusted_observation.scene.trusted_completion_evidence()
                }
                if (
                    "scene" in self.completion_evidence_element_ids
                    or not set(self.completion_evidence_element_ids).issubset(
                        allowed_evidence
                    )
                ):
                    raise GenericStepPlanningError(
                        "低整页置信度的finished只能引用高置信只读目标证据，不能引用scene。"
                    )

    def validate_fresh(
        self,
        current_context: QwenTaskContext | dict[str, Any],
        current_observation: TrustedObservation,
    ) -> None:
        context = (
            current_context
            if isinstance(current_context, QwenTaskContext)
            else QwenTaskContext.from_dict(current_context)
        )
        expected = (
            context.task_id,
            context.device_id,
            context.revision,
            current_observation.observation_id,
            current_observation.fingerprint,
        )
        actual = (
            self.task_id,
            self.device_id,
            self.revision,
            self.observation_id,
            self.fingerprint,
        )
        if actual != expected:
            raise GenericStepPlanningError(
                "task/device/revision/observation/fingerprint 已过期或不匹配。"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "task_id": self.task_id,
            "device_id": self.device_id,
            "revision": self.revision,
            "observation_id": self.observation_id,
            "fingerprint": self.fingerprint,
            "page_state": self.page_state.to_dict(),
            "trusted_observation": self.trusted_observation.to_dict(),
            "status": self.proposal.status,
            "next_action": self.proposal.action.to_dict() if self.proposal.action else None,
            "target_region": self.target_region.to_dict() if self.target_region else None,
            "expected_result": dict(self.expected_result),
            "confidence": float(self.confidence),
            "reason": self.reason,
            "completion_evidence_element_ids": list(
                self.completion_evidence_element_ids
            ),
            "completion_evidence": list(self.proposal.completion_evidence),
        }


class QwenVisualDecisionObserver:
    """Select one action from a separately established trusted observation."""

    def __init__(self, provider: Any) -> None:
        self.provider = provider
        self.last_raw_response = ""
        self.last_diagnostics: dict[str, Any] = {}
        self._metrics = {
            "decision_count": 0,
            "model_attempted_count": 0,
            "first_pass_success_count": 0,
            "retry_success_count": 0,
            "final_blocked_count": 0,
        }

    def status(self) -> dict[str, Any]:
        value = dict(self.provider.status())
        attempted = self._metrics["model_attempted_count"]
        decisions = self._metrics["decision_count"]
        value.update(
            {
                "visual_decision_protocol": QWEN_VISUAL_DECISION_PROTOCOL_VERSION,
                "visual_selection_protocol": QWEN_VISUAL_SELECTION_PROTOCOL_VERSION,
                "task_context_protocol": SUPPORTED_TASK_CONTEXT_PROTOCOL,
                "model_role": QWEN_VISUAL_DECISION_MODEL_ROLE,
                "hardware_actions_enabled": False,
                "decision_timeout_seconds": DECISION_TIMEOUT_SECONDS,
                "decision_output_tokens": DECISION_OUTPUT_TOKENS,
                "decision_retry_tokens": DECISION_RETRY_TOKENS,
                **self._metrics,
                "first_pass_rate": _ratio(
                    self._metrics["first_pass_success_count"], attempted
                ),
                "repair_retry_rate": _ratio(
                    self._metrics["retry_success_count"], attempted
                ),
                "final_blocked_rate": _ratio(
                    self._metrics["final_blocked_count"], decisions
                ),
            }
        )
        return value

    def decide(
        self,
        *,
        frames: list[Image.Image],
        task_context: QwenTaskContext | dict[str, Any],
        trusted_observation: TrustedObservation,
        decision_number: int = 1,
        available_action_kinds: Iterable[str] | None = None,
    ) -> QwenVisualDecision:
        started = time.perf_counter()
        self.last_raw_response = ""
        self.last_diagnostics = {}
        model_call_elapsed_seconds: list[float] = []
        context = (
            task_context
            if isinstance(task_context, QwenTaskContext)
            else QwenTaskContext.from_dict(task_context)
        )
        context.validate()
        available_actions = _normalize_available_action_kinds(
            available_action_kinds
        )
        available_actions = _precondition_eligible_action_kinds(
            context,
            trusted_observation,
            available_actions,
        )
        # These are the same read-only frames that established the trusted
        # observation, so apply the observer's one-leading-frame tolerance.
        # Confirmation-time recapture and post-action verification use their
        # own stricter full-window stability checks.
        trusted_observation.validate_against_frames(
            frames,
            allow_leading_outlier=True,
        )
        if context.device_id != trusted_observation.device_id:
            raise VisionAgentError("任务 device_id 与可信观察不一致。")
        self._metrics["decision_count"] += 1

        model_identity = public_model_identity(self.provider.status())
        base_diagnostics = {
            "visual_decision_protocol": QWEN_VISUAL_DECISION_PROTOCOL_VERSION,
            "vision_model": model_identity,
            "task_id": context.task_id,
            "device_id": context.device_id,
            "revision": context.revision,
            "observation_id": trusted_observation.observation_id,
            "fingerprint": trusted_observation.fingerprint,
            "model_calls": 0,
            "protocol_retry_used": False,
            "first_output_rejected": False,
            "candidate_action_from_first_output": False,
            "first_pass_success": False,
            "repair_retry_success": False,
            "hardware_actions_enabled": False,
            "available_action_kinds": sorted(available_actions),
            "model_call_elapsed_seconds": model_call_elapsed_seconds,
        }
        self.last_diagnostics = dict(base_diagnostics)

        def model_chat(messages: list[dict[str, Any]], *, max_tokens: int) -> str:
            base_diagnostics["model_calls"] = int(base_diagnostics["model_calls"]) + 1
            self.last_diagnostics = dict(base_diagnostics)
            call_started = time.perf_counter()
            try:
                return self._provider_chat(messages, max_tokens=max_tokens)
            finally:
                model_identity.clear()
                model_identity.update(public_model_identity(self.provider.status()))
                model_call_elapsed_seconds.append(
                    round(time.perf_counter() - call_started, 3)
                )
                base_diagnostics["model_call_elapsed_seconds"] = list(
                    model_call_elapsed_seconds
                )
                self.last_diagnostics = dict(base_diagnostics)

        if (
            context.current_external_impact in {"external_state", "unknown"}
            and not context.external_action_allowed
        ):
            decision = _local_blocked_decision(
                context,
                trusted_observation,
                reason="风险确认门未满足，本轮禁止提出外部状态动作。",
            )
            self._metrics["final_blocked_count"] += 1
            self.last_diagnostics.update(
                {
                    "local_safety_block": "confirmation_gate",
                    "decision_status": "blocked",
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                }
            )
            return decision

        exact_text_block = _exact_text_candidate_block(
            context,
            trusted_observation,
        )
        if exact_text_block is not None:
            reason, block_code = exact_text_block
            decision = _local_blocked_decision(
                context,
                trusted_observation,
                reason=reason,
            )
            self._metrics["final_blocked_count"] += 1
            self.last_diagnostics.update(
                {
                    "local_safety_block": block_code,
                    "decision_status": "blocked",
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                }
            )
            return decision

        identity_block = _identity_text_candidate_block(
            context,
            trusted_observation,
        )
        if identity_block is not None:
            reason, block_code = identity_block
            decision = _local_blocked_decision(
                context,
                trusted_observation,
                reason=reason,
            )
            self._metrics["final_blocked_count"] += 1
            self.last_diagnostics.update(
                {
                    "local_safety_block": block_code,
                    "decision_status": "blocked",
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                }
            )
            return decision

        prompt = _selection_decision_prompt(
            context,
            trusted_observation,
            decision_number=max(1, int(decision_number)),
            available_action_kinds=available_actions,
        )
        image = frames[trusted_observation.selected_frame_index].convert("RGB")
        messages = _decision_messages(prompt, image)
        self._metrics["model_attempted_count"] += 1
        try:
            raw = model_chat(messages, max_tokens=DECISION_OUTPUT_TOKENS)
        except VisionAgentError as service_error:
            reason = f"Qwen决策服务不可用，本轮安全阻塞：{service_error}"
            decision = _local_blocked_decision(
                context,
                trusted_observation,
                reason=reason,
            )
            self._metrics["final_blocked_count"] += 1
            self.last_diagnostics.update(
                failure_diagnostics(
                    service_error,
                    stage="requesting_first_decision",
                    model_calls=int(base_diagnostics["model_calls"]),
                    elapsed_seconds=time.perf_counter() - started,
                    safe_stop_reason="决策服务失败，未形成候选动作，控制器与机械臂均未执行。",
                )
            )
            self.last_diagnostics["decision_status"] = "blocked"
            return decision
        self.last_raw_response = raw
        self.last_diagnostics = dict(base_diagnostics)
        try:
            decision = _parse_model_decision(
                raw,
                context=context,
                observation=trusted_observation,
                available_action_kinds=available_actions,
            )
            self._metrics["first_pass_success_count"] += 1
            first_pass = True
        except VisionAgentError as first_error:
            reason = (
                "Qwen单次结构化输出不符合可信选择合同；本轮安全阻塞："
                f"{first_error}"
            )
            decision = _local_blocked_decision(
                context,
                trusted_observation,
                reason=reason,
            )
            self._metrics["final_blocked_count"] += 1
            self.last_diagnostics.update(
                {
                    "failed_stage": "parsing_first_decision",
                    "error": str(first_error),
                    "error_type": classify_qwen_error(
                        first_error,
                        raw_response=self.last_raw_response,
                    ),
                    "first_output_rejected": True,
                    "candidate_action_from_first_output": False,
                    "protocol_retry_used": False,
                    "retry_failure_blocked": False,
                    "decision_status": "blocked",
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                    "safe_stop_reason": (
                        "单次模型输出非法，输出已丢弃；未发起远程格式重生成，"
                        "控制器与机械臂均未执行。"
                    ),
                    "raw_response_length": len(self.last_raw_response),
                    "raw_response_excerpt": self.last_raw_response[:1000],
                }
            )
            return decision

        if decision.proposal.status == "blocked":
            self._metrics["final_blocked_count"] += 1
        self.last_diagnostics.update(
            {
                "model_calls": int(base_diagnostics["model_calls"]),
                "protocol_retry_used": False,
                "first_pass_success": first_pass,
                "repair_retry_success": False,
                "decision_status": decision.proposal.status,
                "decision_confidence": decision.confidence,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        )
        return decision

    def _provider_chat(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
    ) -> str:
        try:
            return self.provider._chat(
                messages,
                max_tokens=max_tokens,
                timeout=DECISION_TIMEOUT_SECONDS,
                max_attempts=2,
                response_format={"type": "json_object"},
            )
        except TypeError as exc:
            text = str(exc)
            if "unexpected keyword" not in text and "keyword argument" not in text:
                raise
            return self.provider._chat(messages, max_tokens=max_tokens)


def _decision_observation_prompt_dict(
    context: QwenTaskContext,
    observation: TrustedObservation,
) -> dict[str, Any]:
    """Hide explicitly excluded elements from the model's candidate surface.

    The authoritative observation remains unchanged for fingerprinting, local
    policy and evidence.  This projection only prevents an excluded element
    from competing for Qwen's single next-action selection.
    """

    value = observation.prompt_dict()
    constraints = (
        context.global_constraints,
        context.current_subgoal.get("constraints") or (),
    )
    excluded_ids: set[str] = set()
    filtered_candidates: list[dict[str, Any]] = []
    for candidate in value.get("candidates", []):
        candidate = dict(candidate)
        states = candidate.get("states")
        if isinstance(states, Mapping) and "keyboard_geometry" in states:
            # Current-frame keyboard anchors are a local execution credential,
            # not part of Qwen's semantic choice surface. Qwen selects only the
            # trusted element_id; local hydration restores authoritative states.
            candidate["states"] = {
                key: item
                for key, item in states.items()
                if key != "keyboard_geometry"
            }
        if constraint_excludes_candidate(
            constraints,
            (
                candidate.get("meaning"),
                candidate.get("label"),
                candidate.get("evidence"),
            ),
            candidate_role=str(candidate.get("role") or ""),
        ):
            element_id = str(candidate.get("element_id") or "").strip()
            if element_id:
                excluded_ids.add(element_id)
            continue
        filtered_candidates.append(candidate)
    value["candidates"] = filtered_candidates
    if excluded_ids:
        aliases = value.get("candidate_aliases")
        if isinstance(aliases, dict):
            value["candidate_aliases"] = {
                key: target
                for key, target in aliases.items()
                if str(key) not in excluded_ids and str(target) not in excluded_ids
            }
        conflicts = value.get("candidate_conflicts")
        if isinstance(conflicts, list):
            value["candidate_conflicts"] = [
                conflict
                for conflict in conflicts
                if not (
                    isinstance(conflict, Mapping)
                    and excluded_ids.intersection(
                        str(item) for item in conflict.get("element_ids", [])
                    )
                )
            ]
    return value


def _selection_choices(
    context: QwenTaskContext,
    observation: TrustedObservation,
    available_action_kinds: frozenset[str],
) -> tuple[dict[str, Any], ...]:
    """Build generic action choices from the trusted scene, never app steps."""

    prompt_observation = _decision_observation_prompt_dict(context, observation)
    candidates = tuple(
        item
        for item in prompt_observation.get("candidates", ())
        if isinstance(item, Mapping)
        and str(item.get("element_id") or "").strip()
    )
    choices: list[dict[str, Any]] = []

    def append_choice(
        action: str,
        *,
        expected_result: Mapping[str, Any],
        **parts: Any,
    ) -> None:
        choices.append(
            {
                "choice_id": f"choice_{len(choices) + 1}",
                "action": action,
                "expected_result": dict(expected_result),
                **parts,
            }
        )

    for action in sorted(available_action_kinds):
        if action in {"back", "home", "reveal_system_navigation", "wait_for_change"}:
            expected_result = (
                {"system_ui": {"navigation_bar_visible": True}}
                if action == "reveal_system_navigation"
                else {"scene_changed": True}
            )
            append_choice(action, expected_result=expected_result)
            continue
        if action == "swipe":
            for direction in ("up", "down", "left", "right"):
                append_choice(
                    action,
                    direction=direction,
                    expected_result={"content_changed": True},
                )
            continue
        eligible = tuple(
            item
            for item in candidates
            if str(item.get("role") or "") not in {"keyboard_key", "dialog"}
            and isinstance(item.get("states"), Mapping)
            and (
                item["states"].get("goal_relevant") is True
                or item["states"].get("ime_candidate") is True
                or item["states"].get("input_literal_key") is True
                or item["states"].get("keyboard_layout_switch") is True
                or item["states"].get("keyboard_case_switch") is True
            )
        )
        if action in {"input_verified_text", "clear_verified_text"}:
            eligible = tuple(
                item for item in eligible if str(item.get("role") or "") == "input"
            )
            if action == "input_verified_text":
                eligible = tuple(
                    item for item in eligible
                    if isinstance(item.get("states"), Mapping)
                    and item["states"].get("focused") is True
                )
            else:
                eligible = tuple(
                    item for item in eligible
                    if isinstance(item.get("states"), Mapping)
                    and item["states"].get("focused") is True
                    and isinstance(item["states"].get("value"), str)
                    and bool(item["states"].get("value"))
                    and item["states"].get("goal_relevant") is True
                )
        if action in SINGLE_ELEMENT_ACTIONS:
            for item in eligible:
                if action == "input_verified_text":
                    try:
                        input_step = plan_next_verified_input(
                            context.requested_input_text,
                            item["states"].get("value"),
                        )
                    except (ValueError, VerifiedTextTransactionError):
                        continue
                    if (
                        input_step is None
                        or input_step.kind == "literal_key"
                        or item["states"].get("keyboard_layout") != "qwerty"
                        or item["states"].get("keyboard_input_mode")
                        != input_step.required_mode
                        or (
                            bool(input_step.required_case_mode)
                            and item["states"].get("keyboard_case_mode")
                            != input_step.required_case_mode
                        )
                        or item["states"].get("ime_preedit_text")
                    ):
                        continue
                    expected_states = (
                        {
                            "value": input_step.current_text,
                            "ime_preedit_text": input_step.pinyin,
                            "ime_exact_candidate_text": input_step.segment,
                        }
                        if input_step.kind == "chinese_pinyin"
                        else {"value": input_step.expected_value}
                    )
                    expected_result = {
                        "element_state": {
                            "meaning": str(item.get("meaning") or "").strip(),
                            "states": expected_states,
                        }
                    }
                elif action == "clear_verified_text":
                    expected_result = {
                        "element_state": {
                            "meaning": str(item.get("meaning") or "").strip(),
                            "states": {"value": ""},
                        }
                    }
                elif (
                    action == "tap_semantic"
                    and str(item.get("meaning") or "") == "ime_exact_candidate"
                    and isinstance(item.get("states"), Mapping)
                    and item["states"].get("ime_candidate") is True
                ):
                    expected_result = {
                        "element_state": {
                            "meaning": "application_text_input",
                            "states": {
                                "value": item["states"].get("expected_input_value"),
                            },
                        }
                    }
                elif (
                    action == "tap_semantic"
                    and str(item.get("meaning") or "") == "input_exact_literal_key"
                    and isinstance(item.get("states"), Mapping)
                    and item["states"].get("input_literal_key") is True
                ):
                    expected_result = {
                        "element_state": {
                            "meaning": "application_text_input",
                            "states": {
                                "value": item["states"].get("expected_input_value"),
                            },
                        }
                    }
                elif (
                    action == "tap_semantic"
                    and str(item.get("meaning") or "") == "switch_keyboard_layout"
                    and isinstance(item.get("states"), Mapping)
                    and item["states"].get("keyboard_layout_switch") is True
                ):
                    expected_result = {
                        "element_state": {
                            "meaning": "application_text_input",
                            "states": {
                                "value": item["states"].get("prior_input_value"),
                                "keyboard_layout": item["states"].get("target_layout"),
                            },
                        }
                    }
                elif (
                    action == "tap_semantic"
                    and str(item.get("meaning") or "") == "switch_keyboard_case"
                    and isinstance(item.get("states"), Mapping)
                    and item["states"].get("keyboard_case_switch") is True
                ):
                    expected_result = {
                        "element_state": {
                            "meaning": "application_text_input",
                            "states": {
                                "value": item["states"].get("prior_input_value"),
                                "keyboard_case_mode": item["states"].get("target_mode"),
                            },
                        }
                    }
                elif (
                    action == "tap_semantic"
                    and str(item.get("role") or "") == "input"
                    and not bool(
                        isinstance(item.get("states"), Mapping)
                        and item["states"].get("focused") is True
                    )
                ):
                    expected_result = {
                        "element_state": {
                            "meaning": str(item.get("meaning") or "").strip(),
                            "states": {"focused": True},
                        }
                    }
                else:
                    expected_result = {"scene_changed": True}
                append_choice(
                    action,
                    element_id=str(item["element_id"]),
                    expected_result=expected_result,
                )
            continue
        if action == "drag":
            for source in eligible:
                for destination in eligible:
                    if source["element_id"] == destination["element_id"]:
                        continue
                    append_choice(
                        action,
                        source_element_id=str(source["element_id"]),
                        destination_element_id=str(destination["element_id"]),
                        expected_result={"scene_changed": True},
                    )
    return tuple(choices)


def _selection_decision_prompt(
    context: QwenTaskContext,
    observation: TrustedObservation,
    *,
    decision_number: int,
    available_action_kinds: frozenset[str],
) -> str:
    """Ask Qwen only for semantic selection; local code binds all authority."""

    observation_prompt = _decision_observation_prompt_dict(context, observation)
    choices = _selection_choices(context, observation, available_action_kinds)
    return f"""
你是通用手机视觉操作 Agent 的 Qwen 单步视觉选择层。必须先做完成判定，再考虑动作。
你只能根据当前 DeepSeek 子目标、本轮可信画面和本地提供的 choices 选择一个下一动作，
或判断 finished/blocked。禁止规划后续步骤、编造候选、输出坐标、执行机械臂或批准风险。

任务上下文：
{json.dumps(context.to_dict(), ensure_ascii=False, separators=(',', ':'))}

本轮可信观察：
{json.dumps(observation_prompt, ensure_ascii=False, separators=(',', ':'))}

本地合法动作候选：
{json.dumps(choices, ensure_ascii=False, separators=(',', ':'))}

只返回一个短JSON对象，顶层只允许以下字段：
{{"status":"action|finished|blocked","choice_id":"action时逐字复制一个choice_id，否则null",
"completes_current_subgoal_on_success":false,"confidence":0.0,"reason":"当前画面依据",
"completion_evidence_element_ids":[]}}

严格规则：
1. status=action时choice_id必须逐字来自choices，completion_evidence_element_ids必须为空。
   即使只有一个choice，也必须由你明确选择；本地不会替你选择。
   若且仅若该choice的expected_result经动作后验证即可直接满足current_subgoal的全部完成条件，
   completes_current_subgoal_on_success=true；仍需后续本地控制器验证，不能凭此字段判定完成。
2. status=finished时choice_id必须为null；完成证据只能引用可信候选ID或"scene"。
   当前状态已经满足完成条件时禁止再点击或选择入口；
   completes_current_subgoal_on_success必须为false。
3. status=blocked时choice_id必须为null、完成证据必须为空，
   completes_current_subgoal_on_success必须为false。
4. global_constraints和current_subgoal.constraints是选择前硬过滤；无法安全满足时blocked。
5. current_external_impact=read_only时只能finished/blocked，除非目标明确要求等待异步变化且choices含wait_for_change。
6. choices中的action、element_id、direction和expected_result都由本地控制器绑定；禁止复制、改写或另行输出。
7. input_verified_text的文字由DeepSeek结构化目标和本地控制器逐字绑定，你只选择对应choice_id；
   不得在输出中重复、改写或补全文字。
8. choices没有合适动作时blocked；不得返回choices之外的动作名称或element_id。
9. 这是第{decision_number}轮。不要Markdown，不要identity、page_state、next_action、target_region、
   expected_result、bounds或额外字段。
"""


def _decision_prompt(
    context: QwenTaskContext,
    observation: TrustedObservation,
    *,
    decision_number: int,
    available_action_kinds: frozenset[str],
) -> str:
    available_actions = "|".join(sorted(available_action_kinds))
    observation_prompt = _decision_observation_prompt_dict(context, observation)
    return f"""
你是通用手机视觉操作 Agent 的 Qwen 单步视觉选择层。DeepSeek 已给出当前动态任务上下文，
本地只读观察阶段已从本轮稳定画面生成可信候选。单元素动作只能在可信候选中选择一个已有 element_id；
back/home/reveal_system_navigation是无元素、无坐标的系统动作，不得伪装成候选元素点击；
你不能创建候选、修改候选文字或 bounds，也不能执行机械臂、批准风险或规划后续步骤。

完整任务上下文：
{json.dumps(context.to_dict(), ensure_ascii=False, separators=(',', ':'))}

本轮可信观察（唯一可执行证据源）：
{json.dumps(observation_prompt, ensure_ascii=False, separators=(',', ':'))}

当前设备已经本地验证可用的动作：{available_actions}

只返回一个JSON对象：
{{
  "protocol_version":"{QWEN_VISUAL_DECISION_PROTOCOL_VERSION}",
  "task_id":"逐字复制输入",
  "device_id":"逐字复制输入",
  "revision":{context.revision},
  "observation_id":"逐字复制输入",
  "fingerprint":"逐字复制输入",
  "page_state":{{"foreground_app_id":"语义描述","screen_id":"语义描述","summary":"短描述","overlays":[]}},
  "status":"action|finished|blocked",
   "next_action":{{"kind":"{available_actions}","element_id":"单元素动作的可信候选ID","target":"复制meaning","role":"复制role","label":"复制label","states":{{}},"text":"输入时逐字复制goal.entities.input_text","duration_ms":"长按500到2000；默认800","source_element_id":"拖动起点候选","destination_element_id":"拖动终点候选","direction":"仅swipe使用；不要返回distance"}},
  "target_region":{{"kind":"element|element_path|screen|system_navigation","element_id":"单元素或拖动起点候选ID","bounds":[0,0,1000,1000],"destination_element_id":"仅拖动终点","destination_bounds":[0,0,1000,1000],"description":"语义区域"}},
  "expected_result":{{"scene_changed":true}},
  "confidence":0.0,
  "reason":"当前画面与当前子目标支持此结论的依据",
  "completion_evidence_element_ids":[]
}}

严格规则：
0. 必须先做完成判定，再考虑动作：逐项比较current_subgoal.completion_conditions与可信scene摘要、
   overlays和候选证据。若当前scene已语义证明目标状态/界面已经存在，必须finished；此时即使还有
   meaning含open/enter/start/launch或label像入口的候选，也禁止再点击它。只有当前证据尚未完成目标
   才能考虑action；不要把当前界面的标题、数量指示或已选中tab误判为“打开当前界面”的按钮。
   role=dialog的候选不能作为单元素动作目标或drag起点。广阔、无标签、未完整可见或包含其他控件的
   页面container也不能作为drag起点；只有紧凑、逐字有标签、fully_visible=true且代表单个源物体的
   container才可逐字复制为source_element_id，并仍须通过本地独立几何审计。drag终点本身是可信
   目标区域时，container可以逐字复制为destination_element_id。其他情况下container/dialog只能被
   completion_evidence_element_ids引用。一个数量指示加一个可见卡片/列表容器足以证明列表已打开时，
   应finished并引用这些候选ID。
1. 每轮最多一个next_action，禁止actions、steps、plan、后续动作或裸坐标。
   global_constraints与current_subgoal.constraints是候选选择前的硬过滤条件。若某条否定约束明确排除
   某个可见元素、区域、角色或语义，即使它看起来是最短路径，也绝不能选择该element_id。不得以目标
   objective是肯定表达为由覆盖否定约束。若navigation_only子目标要求沿访问层级离开当前页面，所有
   页面内导航候选又被明确排除，且可信scene证明system_ui.navigation_bar_visible=true、设备能力包含
   back，则应使用无element_id、无坐标的back；绝不能把back伪装成页面元素tap_semantic。
2. page_state只是语义描述，禁止elements、bounds或任何可执行候选字段。
3. tap_semantic/dismiss_overlay/input_verified_text/long_press只能引用可信观察中现有且置信度>=0.72的唯一element_id；
   target/role/label/states必须逐字复制，target_region.bounds必须逐项复制候选原始bounds。
   role=keyboard_key绝不能作为动作目标。若目标要求本地临时输入值为空，且观察同时提供非空、已聚焦
   input和states.local_text_clear=true的独立button/icon，只能选择该独立清空控件，不能选择输入框本体
   或键盘退格键。
4. input_verified_text只能绑定role=input的候选，text必须逐字复制goal.entities.input_text；不能改写、补全或推断。
   使用该动作时expected_result必须精确写成
   {{"element_state":{{"meaning":"逐字复制输入候选meaning","states":{{"value":"逐字复制goal.entities.input_text"}}}}}}；
   element_state和states都必须是JSON对象，绝不能返回字符串、数组或自然语言。
   role=input且states.focused=true时禁止再用tap_semantic重复聚焦；这不会推进子目标。
   当当前高层子目标需要输入，画面只有一个与目标相关的role=input候选，但它没有
   states.focused=true时，input_verified_text不会出现在本轮可用动作集合中。若tap_semantic
   可用，本轮应只绑定该唯一input候选并提出tap_semantic，expected_result为
   {{"element_state":{{"meaning":"逐字复制输入候选meaning","states":{{"focused":true}}}}}}；动作后
   必须重新观察，不得在同一轮输入文字。
   对小写英文字母精确输入，候选还必须同时提供states.value=""、keyboard_layout="qwerty"和
   keyboard_input_mode="direct_latin"。QWERTY但keyboard_input_mode="chinese_pinyin"时禁止直接输入；
   若可信观察另有meaning=switch_keyboard_input_mode、keyboard_input_mode_switch=true且明确从
   chinese_pinyin切到direct_latin的独立button，可先选择一次tap_semantic，随后必须重新观察，禁止在
   同一轮继续输入。
5. drag必须绑定两个不同可信候选。next_action只能使用扁平字段
   source_element_id/source_target/source_role/source_label/source_states和
   destination_element_id/destination_target/destination_role/destination_label/destination_states；
   绝不能返回source或destination嵌套对象。target_region.kind必须是element_path，并逐字复制
   element_id/bounds以及destination_element_id/destination_bounds。long_press时长限制500到2000毫秒。
6. swipe/wait使用整屏[0,0,1000,1000]和kind=screen；back/home/reveal_system_navigation
   使用整屏和kind=system_navigation。
   home只表示按下Android系统Home键、回到系统Launcher；绝不能用它表示浏览器或任何App里的“首页”。
   swipe只返回direction=up|down|left|right，绝对不要返回distance；距离由本地已校准控制器决定。
   reveal_system_navigation 只在可信观察 system_ui 明确 immersive_or_fullscreen=true 且
   navigation_bar_visible=false时使用；动作本身不得返回坐标、方向或距离，expected_result必须精确为
   {{"system_ui":{{"navigation_bar_visible":true}}}}。
7. 找不到可靠候选、文字不完全一致、候选不唯一、画面模糊或置信度不足时必须blocked。
8. finished只能用completion_evidence_element_ids引用可信候选ID，或用scene引用可信scene摘要；
   禁止自由编写完成证据。若scene_confidence<0.72，只能引用goal_relevant=true且role为
   container/dialog的高置信候选ID，禁止引用scene；这些只读证据不能用于任何动作。
   若current_subgoal的完成条件声称“已刷新/已重新加载/已导航/已重新获取/已同步”等变化已经发生，
   单张当前画面的静态元素或目标结果外观不能单独证明该事件。只有上下文已有前后变化或动作回执，
   或被引用候选逐字显示“刷新成功/加载完成/刚刚更新”等明确动态证据时才可finished；否则选择
   一个当前可信动作，找不到就blocked。禁止用刷新图标、页面标题或目标内容的静态存在冒充事件证据。
9. confirmation_gate没有允许外部状态动作时必须blocked；你不能自行改写或批准确认门。
10. task_id/device_id/revision/observation_id/fingerprint必须逐字复制；任何旧值都会被拒绝。
11. expected_result只描述一个动作后可由新画面验证的变化，且只能按需使用：
    scene_changed、content_changed、current_video_changed、app_id、screen_id、system_ui、
    element_state={{"meaning":"逐字语义","states":{{"状态":true}}}}；不得编写自然语言条件或其他键。
12. 这是第{decision_number}轮，只根据本轮上下文与本轮观察作答。不要Markdown。
13. next_action.kind只能来自当前设备可用动作集合；缺少所需动作能力时必须blocked。
14. status是互斥判别字段：只要返回非null next_action，就必须是status=action并同时给出target_region和
    非空expected_result；status=blocked或finished时next_action和target_region必须为null、
    expected_result必须为空对象。不得把动作字段与终止状态混合。
15. current_external_impact=read_only 时禁止点击、滑动、返回、输入、长按和拖动；当前可信画面已
    证明子目标时必须 finished，并可用 completion_evidence_element_ids=["scene"] 引用可信场景摘要；
    尚未证明时必须 blocked。只有目标明确要求等待异步变化时才可使用 wait_for_change。
16. current_external_impact=navigation_only 且目标字面标签或目标区域尚未出现在可信候选中时，
    只有原图明确显示当前就是与目标相关、仍可继续浏览的列表/信息流/结构化分步流程，并且边缘存在
    被裁切的后续内容，或属于页面内容的连续引导轨/连接线明确接触该边缘、同时没有遮挡层时，才允许
    返回一次整屏 swipe 去显示更多内容。根据原图内容延伸方向选择
    up/down/left/right，expected_result只写{{"content_changed":true}}；不得点击无关候选，不得猜测
    目标已经存在，也不得在无法证明可继续浏览时滑动。动作后必须重新观察，不能连续执行。
"""


def _decision_retry_prompt(
    context: QwenTaskContext,
    observation: TrustedObservation,
    *,
    error: Exception,
    decision_number: int,
    available_action_kinds: frozenset[str],
) -> str:
    available_actions = "|".join(sorted(available_action_kinds))
    observation_prompt = _decision_observation_prompt_dict(context, observation)
    return f"""
上一次输出未通过本地协议，任何候选动作均已丢弃，系统没有执行动作。
错误：{str(error)[:500]}
任务上下文：{json.dumps(context.to_dict(), ensure_ascii=False, separators=(',', ':'))}
可信观察：{json.dumps(observation_prompt, ensure_ascii=False, separators=(',', ':'))}

最多只允许这一次格式修复。重新独立观察并返回完整JSON：
- 协议版本必须是{QWEN_VISUAL_DECISION_PROTOCOL_VERSION}。
- 必须逐字复制task_id={context.task_id}、device_id={context.device_id}、revision={context.revision}、
  observation_id={observation.observation_id}、fingerprint={observation.fingerprint}。
- page_state只能包含foreground_app_id、screen_id、summary、overlays，禁止elements。
- 先比较current_subgoal.completion_conditions与可信scene；当前摘要或候选证据已经语义证明目标界面/
  状态存在时必须finished，禁止再选open/enter/start/launch类动作，也禁止把标题、计数或已选tab当入口。
- role=dialog不能作为单元素动作目标或drag起点。页面级、无标签、未完整可见或包含其他控件的container
  不能作为drag起点；只有紧凑、逐字有标签、fully_visible=true且代表单个源物体的container可以作为
  source_element_id。可信container也可作为drag的destination_element_id；其他情况只能作为finished证据。
- action只能选择可信候选已有element_id并复制原始字段与bounds；不能新建元素。
- global_constraints和current_subgoal.constraints中的否定约束必须先过滤候选；被明确排除的元素即使是
  最短路径也不得选择。navigation_only要求沿访问层级离开当前页面、页面内候选均被排除、导航栏可见
  且back能力可用时，使用无element_id、无坐标的back，不得返回页面元素tap_semantic。
- 找不到逐字匹配且唯一的可信候选就blocked；finished只引用可信证据ID或scene。
- “已刷新/已重新加载/已导航/已重新获取/已同步”等发生型完成条件必须有前后变化、动作回执或被引用
  候选中的明确动态成功文字；单帧静态页面内容、标题或图标不能证明事件已经发生。
- confirmation_gate未允许外部动作时blocked；每轮只允许一个动作，不要计划后续步骤。
- 顶层只允许下方JSON中的字段；绝对不要action、actions、reasoning、analysis、plan或额外字段。
- expected_result只能按需使用scene_changed、content_changed、current_video_changed、app_id、screen_id、
  element_state、system_ui；reveal_system_navigation 的 system_ui 必须精确证明导航栏可见。
- input_verified_text的expected_result必须精确为
  {{"element_state":{{"meaning":"逐字复制输入候选meaning","states":{{"value":"逐字复制goal.entities.input_text"}}}}}}；
  element_state和states都必须是JSON对象，绝不能返回字符串、数组或自然语言。
- 当高层目标需要输入，但唯一相关role=input候选没有states.focused=true时，
  input_verified_text会被本地从本轮可用动作集合中移除。若tap_semantic可用，只提出绑定
  该唯一input候选的聚焦动作，expected_result写其meaning的states.focused=true；动作后
  重新观察，本轮不得同时输入文字。
- 这是第{decision_number}轮。不要Markdown，不要解释，不要把JSON转义成字符串。
- 当前设备只允许动作：{available_actions}；不得返回集合外动作，无法继续就blocked。
- current_external_impact=read_only 时禁止点击、滑动、返回、输入、长按和拖动；画面已证明结果就
  finished，并可用 completion_evidence_element_ids=["scene"]，否则blocked；只有明确等待异步变化
  才可 wait_for_change。
- navigation_only 的目标候选尚未出现时，只有原图明确显示相关列表/信息流/结构化分步流程可继续浏览，
  且边缘有被裁切内容或属于页面内容的连续引导轨/连接线明确接触该边缘、没有遮挡层，才可返回一次
  整屏swipe；expected_result必须是{{"content_changed":true}}，
  不得点击无关候选或连续滑动。
- swipe只允许direction=up|down|left|right，绝对不要返回distance；距离由本地控制器决定。
- status是互斥判别字段，必须先选择且只选择下面一种完整形状：
  A. 执行动作：status="action"，next_action为一个对象，target_region为一个对象，
     expected_result为非空对象，completion_evidence_element_ids=[]。
  B. 安全阻塞：status="blocked"，next_action=null，target_region=null，expected_result={{}}，
     completion_evidence_element_ids=[]。
  C. 已经完成：status="finished"，next_action=null，target_region=null，expected_result={{}}，
     completion_evidence_element_ids只引用当前可信候选。
- 如果你能从当前可信观察选择一个动作，必须使用A并明确写status="action"；绝不能保留B/C的status。
- 如果使用B或C，绝不能携带任何next_action或target_region。不要混合三种形状。

下面是三种互斥形状的完整骨架。身份字段必须逐字保留；只能选择其中一个，不能混合：

A1. 当前可信观察明确支持一个单元素动作时：
{{"protocol_version":"{QWEN_VISUAL_DECISION_PROTOCOL_VERSION}",
"task_id":"{context.task_id}","device_id":"{context.device_id}","revision":{context.revision},
"observation_id":"{observation.observation_id}","fingerprint":"{observation.fingerprint}",
"page_state":{{"foreground_app_id":"unknown","screen_id":"unknown","summary":"短描述","overlays":[]}},
"status":"action","next_action":{{"kind":"从允许动作中选择","element_id":"逐字复制可信候选ID"}},
"target_region":{{"kind":"element","element_id":"逐字复制可信候选ID","bounds":[0,0,0,0],"description":"逐字复制可信候选meaning"}},
"expected_result":{{"scene_changed":true}},"confidence":0.0,"reason":"当前画面依据",
"completion_evidence_element_ids":[]}}
其中两个element_id必须相同且来自本轮可信观察，bounds必须逐项复制该候选；
next_action还必须按动作类型补齐主提示要求的text/duration_ms/direction字段。

A2. 当前可信观察明确支持drag时，只能使用下面的扁平字段，绝不能返回source或destination对象：
{{"protocol_version":"{QWEN_VISUAL_DECISION_PROTOCOL_VERSION}",
"task_id":"{context.task_id}","device_id":"{context.device_id}","revision":{context.revision},
"observation_id":"{observation.observation_id}","fingerprint":"{observation.fingerprint}",
"page_state":{{"foreground_app_id":"unknown","screen_id":"unknown","summary":"短描述","overlays":[]}},
"status":"action","next_action":{{"kind":"drag",
"source_element_id":"逐字复制起点候选ID","source_target":"逐字复制起点meaning",
"source_role":"逐字复制起点role","source_label":"逐字复制起点label","source_states":{{}},
"destination_element_id":"逐字复制终点候选ID","destination_target":"逐字复制终点meaning",
"destination_role":"逐字复制终点role","destination_label":"逐字复制终点label","destination_states":{{}}}},
"target_region":{{"kind":"element_path","element_id":"逐字复制同一起点候选ID",
"bounds":[0,0,0,0],"destination_element_id":"逐字复制同一终点候选ID",
"destination_bounds":[0,0,0,0],"description":"起点到终点"}},
"expected_result":{{"scene_changed":true}},"confidence":0.0,"reason":"当前画面依据",
"completion_evidence_element_ids":[]}}
四个bounds数组必须逐项复制对应可信候选，禁止使用零值占位；起点不能是container/dialog，
终点可以是可信container目标区域。

B. 无可靠动作时：
{{"protocol_version":"{QWEN_VISUAL_DECISION_PROTOCOL_VERSION}",
"task_id":"{context.task_id}","device_id":"{context.device_id}","revision":{context.revision},
"observation_id":"{observation.observation_id}","fingerprint":"{observation.fingerprint}",
"page_state":{{"foreground_app_id":"unknown","screen_id":"unknown","summary":"短描述","overlays":[]}},
"status":"blocked","next_action":null,"target_region":null,"expected_result":{{}},
"confidence":0.0,"reason":"安全停止原因","completion_evidence_element_ids":[]}}

C. 当前可信画面已经证明目标完成时：
{{"protocol_version":"{QWEN_VISUAL_DECISION_PROTOCOL_VERSION}",
"task_id":"{context.task_id}","device_id":"{context.device_id}","revision":{context.revision},
"observation_id":"{observation.observation_id}","fingerprint":"{observation.fingerprint}",
"page_state":{{"foreground_app_id":"unknown","screen_id":"unknown","summary":"短描述","overlays":[]}},
"status":"finished","next_action":null,"target_region":null,"expected_result":{{}},
"confidence":0.0,"reason":"当前可信证据","completion_evidence_element_ids":["scene"]}}

这些只是结构骨架。不得复制不存在的候选、不得使用骨架中的占位文字或零bounds，仍不得增加任何键。
"""


def _parse_model_decision(
    raw: str,
    *,
    context: QwenTaskContext,
    observation: TrustedObservation,
    available_action_kinds: frozenset[str] | None = None,
) -> QwenVisualDecision:
    """Hydrate the model's minimal selection into the existing formal object.

    Full legacy-shaped payloads remain accepted as migration/test input, but
    production prompts only request the minimal selection envelope.  All
    authority-bearing identity, candidate semantics and geometry are local.
    """

    payload = _extract_qwen_json_object(raw)
    if "protocol_version" in payload or "next_action" in payload:
        return _parse_decision(
            raw,
            context=context,
            observation=observation,
            available_action_kinds=available_action_kinds,
        )

    allowed = {
        "status",
        "choice_id",
        "completes_current_subgoal_on_success",
        "confidence",
        "reason",
        "completion_evidence_element_ids",
    }
    unexpected = set(payload) - allowed
    if unexpected:
        raise VisionAgentError(
            "Qwen最小选择包含协议外字段：" + ", ".join(sorted(unexpected))
        )
    required = {
        "status",
        "choice_id",
        "completes_current_subgoal_on_success",
        "confidence",
        "reason",
        "completion_evidence_element_ids",
    }
    missing = required - set(payload)
    if missing:
        raise VisionAgentError(
            "Qwen最小选择缺少字段：" + ", ".join(sorted(missing))
        )

    status = str(payload.get("status") or "").strip().lower()
    if status not in {"action", "finished", "blocked"}:
        raise VisionAgentError("Qwen最小选择 status 必须是action、finished或blocked。")
    choices = _selection_choices(
        context,
        observation,
        available_action_kinds or QWEN_PROTOCOL_ACTIONS,
    )
    choices_by_id = {str(item["choice_id"]): item for item in choices}
    choice_id = str(payload.get("choice_id") or "").strip()
    completes_current_subgoal = payload.get(
        "completes_current_subgoal_on_success"
    )
    if not isinstance(completes_current_subgoal, bool):
        raise VisionAgentError(
            "Qwen最小选择 completes_current_subgoal_on_success 必须是布尔值。"
        )
    completion_ids = payload.get("completion_evidence_element_ids")
    if not isinstance(completion_ids, list) or any(
        not isinstance(item, str) for item in completion_ids
    ):
        raise VisionAgentError(
            "Qwen最小选择 completion_evidence_element_ids 必须是字符串数组。"
        )
    next_action: dict[str, Any] | None = None
    expected_result: dict[str, Any] = {}
    if status == "action":
        if choice_id not in choices_by_id:
            raise VisionAgentError("Qwen最小选择引用了不存在或不允许的 choice_id。")
        if completion_ids:
            raise VisionAgentError("action 不能携带完成证据。")
        choice = choices_by_id[choice_id]
        local_expected_result = choice.get("expected_result")
        if not isinstance(local_expected_result, Mapping) or not local_expected_result:
            raise VisionAgentError("本地动作选择缺少可验证 expected_result。")
        expected_result = dict(local_expected_result)
        if completes_current_subgoal:
            expected_result["goal_complete_on_success"] = True
        next_action = {
            key: value
            for key, value in choice.items()
            if key not in {"choice_id", "expected_result"}
        }
        next_action["kind"] = next_action.pop("action")
        if next_action["kind"] == "input_verified_text":
            next_action["text"] = context.requested_input_text
        elif next_action["kind"] == "long_press":
            next_action["duration_ms"] = 800
    else:
        if choice_id:
            raise VisionAgentError("finished/blocked 不能携带 choice_id。")
        if completes_current_subgoal:
            raise VisionAgentError(
                "finished/blocked 不能声明动作后完成当前子目标。"
            )
        if status == "blocked" and completion_ids:
            raise VisionAgentError("blocked 不能携带完成证据。")

    scene = observation.scene
    hydrated = {
        "protocol_version": QWEN_VISUAL_DECISION_PROTOCOL_VERSION,
        "task_id": context.task_id,
        "device_id": context.device_id,
        "revision": context.revision,
        "observation_id": observation.observation_id,
        "fingerprint": observation.fingerprint,
        "page_state": {
            "foreground_app_id": scene.foreground_app_id,
            "screen_id": scene.screen_id,
            "summary": scene.summary,
            "overlays": list(scene.overlays),
        },
        "status": status,
        "next_action": next_action,
        "target_region": None,
        "expected_result": expected_result,
        "confidence": payload.get("confidence"),
        "reason": payload.get("reason"),
        "completion_evidence_element_ids": completion_ids,
    }
    return _parse_decision(
        json.dumps(hydrated, ensure_ascii=False, separators=(",", ":")),
        context=context,
        observation=observation,
        available_action_kinds=available_action_kinds,
    )


def _parse_decision(
    raw: str,
    *,
    context: QwenTaskContext,
    observation: TrustedObservation,
    available_action_kinds: frozenset[str] | None = None,
) -> QwenVisualDecision:
    try:
        payload = _extract_qwen_json_object(raw)
        allowed = {
            "protocol_version",
            "task_id",
            "device_id",
            "revision",
            "observation_id",
            "fingerprint",
            "page_state",
            "status",
            "next_action",
            "target_region",
            "expected_result",
            "confidence",
            "reason",
            "completion_evidence_element_ids",
        }
        unexpected = set(payload) - allowed
        if unexpected:
            raise GenericStepPlanningError(
                "Qwen视觉决策包含协议外字段：" + ", ".join(sorted(unexpected))
            )
        expected_identity = {
            "protocol_version": QWEN_VISUAL_DECISION_PROTOCOL_VERSION,
            "task_id": context.task_id,
            "device_id": context.device_id,
            "revision": context.revision,
            "observation_id": observation.observation_id,
            "fingerprint": observation.fingerprint,
        }
        for key, expected in expected_identity.items():
            if payload.get(key) != expected:
                raise GenericStepPlanningError(f"Qwen返回的{key}不匹配或已过期。")

        page_state = ModelPageState.from_dict(payload.get("page_state"))
        status = str(payload.get("status") or "").strip().lower()
        raw_action = payload.get("next_action")
        raw_action_kind = ""
        if isinstance(raw_action, dict):
            for kind_field in ("kind", "action", "action_type", "type"):
                if raw_action.get(kind_field):
                    raw_action_kind = str(raw_action[kind_field]).strip().lower()
                    break
        expected_result = _normalize_expected_result(
            payload.get("expected_result") or {},
            action_kind=raw_action_kind,
        )
        nested_target_region = None
        nested_expected_result = None
        if isinstance(raw_action, dict):
            raw_action = dict(raw_action)
            nested_target_region = raw_action.pop("target_region", None)
            nested_expected_result = raw_action.pop("expected_result", None)
        top_level_target_region = payload.get("target_region")
        if nested_target_region not in (None, {}):
            if (
                top_level_target_region not in (None, {})
                and top_level_target_region != nested_target_region
            ):
                raise GenericStepPlanningError(
                    "next_action.target_region 与顶层 target_region 冲突。"
                )
            top_level_target_region = nested_target_region
        if nested_expected_result not in (None, {}):
            if not isinstance(nested_expected_result, dict):
                raise GenericStepPlanningError(
                    "next_action.expected_result 必须是JSON对象。"
                )
            normalized_nested = _normalize_expected_result(
                nested_expected_result,
                action_kind=raw_action_kind,
            )
            if expected_result and expected_result != normalized_nested:
                raise GenericStepPlanningError(
                    "next_action.expected_result 与顶层 expected_result 冲突。"
                )
            expected_result = normalized_nested
        action = _parse_action(
            raw_action,
            status=status,
            expected_result=expected_result,
            observation=observation,
            revision=context.revision,
        )
        if (
            context.current_external_impact == "read_only"
            and action is not None
            and action.action != "wait_for_change"
        ):
            raise GenericStepPlanningError(
                "read_only 子目标禁止点击、滑动、系统导航、返回、输入、长按或拖动；"
                "当前画面已证明结果时必须 finished，否则 blocked。"
            )
        if (
            action is not None
            and available_action_kinds is not None
            and action.action not in available_action_kinds
        ):
            raise GenericStepPlanningError(
                f"当前设备没有本地验证动作能力：{action.action}"
            )
        evidence_ids = (
            _text_tuple(
                payload.get("completion_evidence_element_ids") or [],
                "completion_evidence_element_ids",
            )
            if status == "finished"
            else ()
        )
        completion_evidence = (
            _resolve_completion_evidence(evidence_ids, observation)
            if status == "finished"
            else ()
        )
        proposal = GenericStepProposal(
            status=status,
            action=action,
            reason=str(payload.get("reason") or "").strip()[:500],
            completion_evidence=completion_evidence,
        )
        target_region = _parse_target_region(
            top_level_target_region,
            action=action,
            observation=observation,
        )
        raw_confidence = payload.get("confidence", 0.0)
        if isinstance(raw_confidence, bool):
            raise GenericStepPlanningError("confidence 不能是布尔值。")
        confidence = min(float(raw_confidence), float(observation.scene.confidence))
        if action and action.action in SINGLE_ELEMENT_ACTIONS:
            element = observation.get_candidate(
                str(action.params.get("element_id") or "")
            )
            local_candidate = observation.target_local_candidate()
            confidence_ceiling = (
                float(element.confidence)
                if local_candidate is not None
                and local_candidate.element_id == element.element_id
                else min(
                    float(observation.scene.confidence),
                    float(element.confidence),
                )
            )
            confidence = min(
                float(raw_confidence),
                confidence_ceiling,
            )
        elif action and action.action == "drag":
            confidence = min(
                confidence,
                *(
                    float(
                        observation.get_candidate(
                            str(action.params.get(f"{prefix}element_id") or "")
                        ).confidence
                    )
                    for prefix in ("source_", "destination_")
                ),
            )
        decision = QwenVisualDecision(
            task_id=context.task_id,
            device_id=context.device_id,
            revision=context.revision,
            observation_id=observation.observation_id,
            fingerprint=observation.fingerprint,
            page_state=page_state,
            trusted_observation=observation,
            proposal=proposal,
            target_region=target_region,
            expected_result=dict(expected_result),
            confidence=confidence,
            reason=str(payload.get("reason") or "").strip()[:500],
            completion_evidence_element_ids=evidence_ids,
        )
        decision.validate(context)
        return decision
    except (UISceneError, GenericStepPlanningError, ValueError, TypeError) as exc:
        raise VisionAgentError(f"Qwen视觉单步决策不符合协议：{exc}") from exc


def _extract_qwen_json_object(raw: str) -> dict[str, Any]:
    """Accept one JSON object, or exact duplicate copies of that object only."""

    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text).strip()
    decoder = json.JSONDecoder()
    values: list[Any] = []
    index = 0
    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            break
        try:
            value, end = decoder.raw_decode(text, index)
        except json.JSONDecodeError as exc:
            raise VisionAgentError(f"模型返回的 JSON 无法解析：{exc}") from exc
        values.append(value)
        index = end
    if not values:
        raise VisionAgentError("模型没有返回 JSON 对象。")
    if any(not isinstance(value, dict) for value in values):
        raise VisionAgentError("模型返回值必须是 JSON 对象。")
    first = values[0]
    if any(value != first for value in values[1:]):
        raise VisionAgentError("模型返回了多个互相冲突的 JSON 对象。")
    return dict(first)


def _normalize_expected_result(
    value: Any,
    *,
    action_kind: str = "",
) -> dict[str, Any]:
    """Normalize model wording into the controller's verifiable effect schema."""

    if not isinstance(value, dict):
        raise GenericStepPlanningError("expected_result 必须是JSON对象。")
    _reject_raw_control_data(value)
    aliases = {
        "foreground_app_id": "app_id",
        "new_foreground_app_id": "app_id",
        "new_app_id": "app_id",
        "new_screen_id": "screen_id",
        "screen_change": "scene_changed",
        "page_changed": "scene_changed",
        "list_content_changed": "content_changed",
        "scroll_occurred": "content_changed",
        "new_items_visible": "content_changed",
    }
    allowed = {
        "scene_changed",
        "content_changed",
        "current_video_changed",
        "app_id",
        "screen_id",
        "element_state",
        "system_ui",
        "allow_unchanged",
        "goal_complete_on_success",
    }
    normalized: dict[str, Any] = {}
    for raw_key, item in value.items():
        key = aliases.get(str(raw_key), str(raw_key))
        if (
            key == "system_ui"
            and item in (
                {"overlays": []},
                {"soft_keyboard_visible": False},
            )
            and action_kind in {"back", "home", "dismiss_overlay"}
        ):
            # Model-authored overlay/keyboard absence is not itself a
            # controller-owned system_ui fact.  For dismissal actions retain
            # only the weaker, independently verifiable scene transition; the
            # object must match one exact absence shape, and every other nested
            # field/value still fails closed below.
            key = "scene_changed"
            item = True
        if key not in allowed:
            raise GenericStepPlanningError(
                f"expected_result 包含协议外字段：{raw_key}"
            )
        if key in normalized and normalized[key] != item:
            raise GenericStepPlanningError(
                f"expected_result.{raw_key} 与 {key} 冲突。"
            )
        normalized[key] = item

    for key in (
        "scene_changed",
        "content_changed",
        "current_video_changed",
        "allow_unchanged",
        "goal_complete_on_success",
    ):
        if key in normalized and not isinstance(normalized[key], bool):
            raise GenericStepPlanningError(f"expected_result.{key} 必须是布尔值。")
    for key in ("app_id", "screen_id"):
        if key in normalized:
            if not isinstance(normalized[key], str) or not normalized[key].strip():
                raise GenericStepPlanningError(
                    f"expected_result.{key} 必须是非空字符串。"
                )
            normalized[key] = normalized[key].strip()
    if "element_state" in normalized:
        element_state = normalized["element_state"]
        if not isinstance(element_state, dict):
            raise GenericStepPlanningError(
                "expected_result.element_state 必须是JSON对象。"
            )
        unexpected = set(element_state) - {"meaning", "states"}
        if unexpected:
            raise GenericStepPlanningError(
                "expected_result.element_state 包含协议外字段："
                + ", ".join(sorted(unexpected))
            )
        meaning = element_state.get("meaning")
        states = element_state.get("states")
        if not isinstance(meaning, str) or not meaning.strip():
            raise GenericStepPlanningError(
                "expected_result.element_state.meaning 必须是非空字符串。"
            )
        if not isinstance(states, dict) or not states:
            raise GenericStepPlanningError(
                "expected_result.element_state.states 必须是非空对象。"
            )
        normalized["element_state"] = {
            "meaning": meaning.strip(),
            "states": dict(states),
        }
    if "system_ui" in normalized:
        system_ui = normalized["system_ui"]
        if not isinstance(system_ui, dict):
            raise GenericStepPlanningError(
                "expected_result.system_ui 必须是JSON对象。"
            )
        unexpected = set(system_ui) - {"navigation_bar_visible"}
        if unexpected:
            raise GenericStepPlanningError(
                "expected_result.system_ui 包含协议外字段："
                + ", ".join(sorted(unexpected))
            )
        if system_ui.get("navigation_bar_visible") is not True:
            raise GenericStepPlanningError(
                "expected_result.system_ui.navigation_bar_visible 必须为true。"
            )
        normalized["system_ui"] = {"navigation_bar_visible": True}
    return normalized


def _normalize_available_action_kinds(
    value: Iterable[str] | None,
) -> frozenset[str]:
    if value is None:
        return QWEN_PROTOCOL_ACTIONS
    try:
        normalized = frozenset(str(item or "").strip() for item in value)
    except TypeError as exc:
        raise VisionAgentError("设备动作能力必须是可迭代字符串集合。") from exc
    if "" in normalized:
        raise VisionAgentError("设备动作能力不能包含空值。")
    unexpected = normalized - QWEN_PROTOCOL_ACTIONS
    if unexpected:
        raise VisionAgentError(
            "设备动作能力包含协议外动作：" + ", ".join(sorted(unexpected))
        )
    if not normalized:
        raise VisionAgentError("设备没有任何可供 Qwen 选择的通用动作。")
    return normalized


def _precondition_eligible_action_kinds(
    context: QwenTaskContext,
    observation: TrustedObservation,
    available_action_kinds: frozenset[str],
) -> frozenset[str]:
    """Hide actions whose controller-owned visual preconditions are absent.

    This does not add an action or infer focus. It only prevents Qwen from
    proposing verified text input before the trusted scene proves that an
    input is focused; a separate tap and fresh observation must establish that
    state first.
    """

    eligible = set(available_action_kinds)
    if "input_verified_text" in eligible:
        eligible_inputs = []
        if context.requested_input_text is not None:
            for element in observation.scene.elements:
                if element.role != "input" or element.states.get("focused") is not True:
                    continue
                try:
                    step = plan_next_verified_input(
                        context.requested_input_text,
                        element.states.get("value"),
                    )
                except (ValueError, VerifiedTextTransactionError):
                    continue
                if (
                    step is not None
                    and step.kind != "literal_key"
                    and element.states.get("keyboard_layout") == "qwerty"
                    and element.states.get("keyboard_input_mode") == step.required_mode
                    and (
                        not step.required_case_mode
                        or element.states.get("keyboard_case_mode")
                        == step.required_case_mode
                    )
                    and not element.states.get("ime_preedit_text")
                ):
                    eligible_inputs.append(element)
        if len(eligible_inputs) != 1:
            eligible.remove("input_verified_text")
    if "clear_verified_text" in eligible:
        clearable_inputs = tuple(
            element
            for element in observation.scene.elements
            if element.role == "input"
            and element.states.get("focused") is True
            and isinstance(element.states.get("value"), str)
            and bool(element.states.get("value"))
            and element.states.get("keyboard_layout") == "qwerty"
            and element.states.get("goal_relevant") is True
        )
        if len(clearable_inputs) != 1:
            eligible.remove("clear_verified_text")
        elif _current_subgoal_requests_verified_clear(context):
            # Clearing a currently visible local draft is a certified atomic
            # action.  Do not offer indirect long-press/selection flows or
            # keyboard keys when the trusted scene already proves the exact
            # preconditions for the one-shot clear contract.
            eligible.intersection_update({"clear_verified_text"})
    if _current_subgoal_requests_keyboard_dismissal(context) and (
        _trusted_scene_proves_visible_keyboard(observation.scene)
    ):
        # Android back is the certified device primitive for dismissing a
        # currently visible soft keyboard.  Keep Qwen as the single-step
        # selector, but do not offer element taps (especially keyboard keys),
        # Home, or gestures for this exact structural state transition.
        eligible.intersection_update({"back"})
    return frozenset(eligible)


_KEYBOARD_REFERENCE_PATTERN = re.compile(
    r"(?:软键盘|键盘|输入法|\b(?:soft\s+)?keyboard\b|\bime\b)",
    re.IGNORECASE,
)
_KEYBOARD_DISMISSAL_PATTERN = re.compile(
    r"(?:收起|隐藏|关闭|不再显示|不可见|"
    r"\b(?:hide|hidden|dismiss|close|closed|not\s+visible|no\s+longer\s+visible)\b)",
    re.IGNORECASE,
)

_VERIFIED_CLEAR_PATTERN = re.compile(
    r"(?:清空|清除|删(?:除|掉)|恢复(?:为|成)?(?:空白|空)|"
    r"(?:内容|输入框|草稿|文本|文字|值)(?:恢复)?(?:为|成|是)?(?:空白|空)|"
    r"\b(?:clear|empty|blank|remove|delete)\b)",
    re.IGNORECASE,
)


def _current_subgoal_requests_verified_clear(
    context: QwenTaskContext,
) -> bool:
    """Recognize only the active subgoal's explicit empty-value request."""

    if context.requested_input_text is not None:
        return False
    visible = " ".join(
        [
            str(context.current_subgoal.get("objective") or ""),
            *(
                str(item)
                for item in context.current_subgoal.get(
                    "completion_conditions",
                    [],
                )
            ),
        ]
    )
    return bool(_VERIFIED_CLEAR_PATTERN.search(visible))


def _current_subgoal_requests_keyboard_dismissal(
    context: QwenTaskContext,
) -> bool:
    """Match only the active subgoal, never a keyboard mention in the goal."""

    visible = " ".join(
        [
            str(context.current_subgoal.get("objective") or ""),
            *(
                str(item)
                for item in context.current_subgoal.get(
                    "completion_conditions",
                    [],
                )
            ),
        ]
    )
    return bool(
        _KEYBOARD_REFERENCE_PATTERN.search(visible)
        and _KEYBOARD_DISMISSAL_PATTERN.search(visible)
    )


def _trusted_scene_proves_visible_keyboard(scene: UIScene) -> bool:
    """Require the independent input audit's complete visible-keyboard facts."""

    candidates = tuple(
        element
        for element in scene.elements
        if element.role == "input"
        and float(element.confidence) >= MIN_TARGET_CONFIDENCE
        and element.states.get("goal_relevant") is True
        and element.states.get("focused") is True
        and element.states.get("keyboard_layout")
        in {"qwerty", "numeric", "symbol", "unknown"}
        and element.states.get("keyboard_input_mode")
        in {"direct_latin", "chinese_pinyin", "unknown"}
    )
    return len(candidates) == 1


def _parse_action(
    value: Any,
    *,
    status: str,
    expected_result: dict[str, Any],
    observation: TrustedObservation,
    revision: int,
) -> SemanticAction | None:
    if status != "action":
        if value not in (None, {}):
            raise GenericStepPlanningError("finished/blocked 不能携带 next_action。")
        return None
    if not isinstance(value, dict):
        raise GenericStepPlanningError("action 状态缺少唯一 next_action 对象。")
    value = dict(value)
    nested_params = value.pop("params", None)
    if nested_params is not None:
        if not isinstance(nested_params, dict):
            raise GenericStepPlanningError("next_action.params 必须是JSON对象。")
        _reject_raw_control_data(nested_params)
        for field, nested_value in nested_params.items():
            if field in value and value[field] != nested_value:
                raise GenericStepPlanningError(
                    f"next_action.params.{field} 与顶层字段冲突。"
                )
            value[field] = nested_value
    # Qwen occasionally uses two conventional JSON aliases even after a
    # format-only retry.  Normalize names only; candidate identity and every
    # semantic field are still checked against the trusted observation below.
    for alias, canonical in {
        "action": "kind",
        "action_type": "kind",
        "type": "kind",
        "target_element_id": "element_id",
    }.items():
        if alias not in value:
            continue
        if canonical in value and value[canonical] != value[alias]:
            raise GenericStepPlanningError(
                f"next_action.{alias} 与 {canonical} 冲突。"
            )
        value[canonical] = value.pop(alias)
    kind = str(value.get("kind") or "").strip().lower()
    redundant_bounds = value.pop("bounds", None)
    if redundant_bounds is not None and kind not in SINGLE_ELEMENT_ACTIONS:
        raise GenericStepPlanningError(
            "next_action.bounds 只允许逐项复用单元素可信候选区域。"
        )
    if "distance" in value:
        value.pop("distance")
        if kind != "swipe":
            raise GenericStepPlanningError(
                "next_action.distance 只允许作为swipe的非权威提示。"
            )
        # The device exposes only a calibrated fixed swipe.  Model-authored
        # distance never reaches the controller or hardware.  Discard legacy
        # numeric or descriptive hints instead of treating display-only data
        # as an executable protocol failure.
    allowed = {
        "kind", "element_id", "target", "role", "label", "states", "direction",
        "text", "duration_ms",
        "source_element_id", "source_target", "source_role", "source_label",
        "source_states", "destination_element_id", "destination_target",
        "destination_role", "destination_label", "destination_states",
    }
    unexpected = set(value) - allowed
    if unexpected:
        raise GenericStepPlanningError(
            "next_action 包含协议外字段：" + ", ".join(sorted(unexpected))
        )
    if kind not in ALLOWED_STEP_ACTIONS:
        raise GenericStepPlanningError(f"唯一下一动作不在通用白名单：{kind}")
    parameter_fields_by_kind = {
        "tap_semantic": {"element_id", "target", "role", "label", "states"},
        "dismiss_overlay": {"element_id", "target", "role", "label", "states"},
        "input_verified_text": {
            "element_id", "target", "role", "label", "states", "text",
        },
        "clear_verified_text": {
            "element_id", "target", "role", "label", "states",
        },
        "long_press": {
            "element_id", "target", "role", "label", "states", "duration_ms",
        },
        "drag": {
            "source_element_id", "source_target", "source_role", "source_label",
            "source_states", "destination_element_id", "destination_target",
            "destination_role", "destination_label", "destination_states",
        },
        "swipe": {"direction"},
        "reveal_system_navigation": set(),
        "back": set(),
        "home": set(),
        "wait_for_change": set(),
    }
    effective_fields = parameter_fields_by_kind[kind]
    params = {
        key: value[key]
        for key in effective_fields
        if key in value and value[key] not in (None, "", {}, [])
    }
    params["expected_effect"] = dict(expected_result)
    if kind == "reveal_system_navigation":
        if expected_result != {"system_ui": {"navigation_bar_visible": True}}:
            raise GenericStepPlanningError(
                "reveal_system_navigation 必须精确声明结构化导航栏可见后置条件。"
            )
    elif "system_ui" in expected_result:
        raise GenericStepPlanningError(
            "结构化 system_ui 后置条件只允许用于 reveal_system_navigation。"
        )
    for field in ("states", "source_states", "destination_states"):
        if not isinstance(params.get(field, {}), dict):
            raise GenericStepPlanningError(f"next_action.{field} 必须是对象。")
    _reject_raw_control_data(params)
    if kind in SINGLE_ELEMENT_ACTIONS:
        element_id = str(params.get("element_id") or "").strip()
        if not element_id:
            raise GenericStepPlanningError("元素动作缺少可信候选 element_id。")
        element = observation.get_candidate(element_id)
        if element.role == "keyboard_key":
            raise GenericStepPlanningError(
                "keyboard_key 不能作为通用元素动作目标。"
            )
        if redundant_bounds is not None:
            if (
                not isinstance(redundant_bounds, (list, tuple))
                or len(redundant_bounds) != 4
                or any(
                    isinstance(item, bool) or not isinstance(item, (int, float))
                    for item in redundant_bounds
                )
            ):
                raise GenericStepPlanningError(
                    "next_action.bounds 必须包含4个0到1000数值。"
                )
            normalized_bounds = tuple(
                float(item) / 1000.0 for item in redundant_bounds
            )
            if any(
                abs(actual - trusted) > 0.0001
                for actual, trusted in zip(normalized_bounds, element.bounds)
            ):
                raise GenericStepPlanningError(
                    "next_action.bounds 没有逐项复用可信候选区域。"
                )
        # element_id is Qwen's only semantic selection.  All descriptive
        # fields are authoritative local data and must never depend on the
        # model repeating strings exactly (or on model-authored synonyms).
        params.update(
            {
                "target": element.meaning,
                "role": element.role,
                "label": element.label,
                "states": dict(element.states),
            }
        )
    elif kind == "drag":
        source_id = str(params.get("source_element_id") or "").strip()
        destination_id = str(params.get("destination_element_id") or "").strip()
        if not source_id or not destination_id or source_id == destination_id:
            raise GenericStepPlanningError("拖动必须绑定两个不同的可信候选。")
        source = observation.get_candidate(source_id)
        destination = observation.get_candidate(destination_id)
        for prefix, element in (
            ("source_", source),
            ("destination_", destination),
        ):
            params.update(
                {
                    f"{prefix}target": element.meaning,
                    f"{prefix}role": element.role,
                    f"{prefix}label": element.label,
                    f"{prefix}states": dict(element.states),
                }
            )
    return SemanticAction(
        node_id=f"qwen_visual_revision_{revision}",
        action=kind,
        params=params,
    )


def _parse_target_region(
    value: Any,
    *,
    action: SemanticAction | None = None,
    observation: TrustedObservation | None = None,
) -> VisualTargetRegion | None:
    if value in (None, {}) and action is None:
        return None
    if value in (None, {}):
        value = {}
    elif not isinstance(value, dict):
        raise GenericStepPlanningError("target_region 必须是对象或null。")
    value = dict(value)
    for alias, canonical in {
        "type": "kind",
        "region_type": "kind",
        "target_element_id": "element_id",
        "target_bounds": "bounds",
    }.items():
        if alias not in value:
            continue
        if canonical in value and value[canonical] != value[alias]:
            raise GenericStepPlanningError(
                f"target_region.{alias} 与 {canonical} 冲突。"
            )
        value[canonical] = value.pop(alias)
    if action is not None:
        if observation is None:
            raise GenericStepPlanningError("本地构造目标区域缺少可信观察。")
        if action.action in SINGLE_ELEMENT_ACTIONS:
            element_id = str(action.params.get("element_id") or "").strip()
            element = observation.get_candidate(element_id)
            defaults = {
                "kind": "element",
                "element_id": element.element_id,
                "bounds": [item * 1000.0 for item in element.bounds],
                "description": element.label or element.meaning,
            }
        elif action.action == "drag":
            source = observation.get_candidate(
                str(action.params.get("source_element_id") or "").strip()
            )
            destination = observation.get_candidate(
                str(action.params.get("destination_element_id") or "").strip()
            )
            defaults = {
                "kind": "element_path",
                "element_id": source.element_id,
                "bounds": [item * 1000.0 for item in source.bounds],
                "destination_element_id": destination.element_id,
                "destination_bounds": [
                    item * 1000.0 for item in destination.bounds
                ],
                "description": (
                    f"{source.label or source.meaning} 到 "
                    f"{destination.label or destination.meaning}"
                ),
            }
        elif action.action in {"back", "home", "reveal_system_navigation"}:
            defaults = {
                "kind": "system_navigation",
                "bounds": [0.0, 0.0, 1000.0, 1000.0],
                "description": (
                    "Android系统Home键"
                    if action.action == "home"
                    else "Android系统导航栏"
                    if action.action == "reveal_system_navigation"
                    else "系统返回区域"
                ),
            }
        else:
            defaults = {
                "kind": "screen",
                "bounds": [0.0, 0.0, 1000.0, 1000.0],
                "description": "当前屏幕",
            }
        if action.action in SINGLE_ELEMENT_ACTIONS or action.action == "drag":
            for key, default in defaults.items():
                if value.get(key) in (None, "", []):
                    value[key] = default
        else:
            # Screen/system actions have no model-authoritative region.  The
            # local controller always records the actual full-screen region;
            # model-authored container bounds or element IDs are ignored.
            value = {
                **defaults,
                "description": str(value.get("description") or "").strip()
                or defaults["description"],
            }
    allowed = {
        "kind",
        "element_id",
        "bounds",
        "destination_element_id",
        "destination_bounds",
        "description",
    }
    unexpected = set(value) - allowed
    if unexpected:
        raise GenericStepPlanningError(
            "target_region 包含协议外字段：" + ", ".join(sorted(unexpected))
        )
    raw_bounds = value.get("bounds")
    if not isinstance(raw_bounds, (list, tuple)) or len(raw_bounds) != 4:
        raise GenericStepPlanningError("target_region.bounds 必须包含4个数值。")
    try:
        bounds = tuple(float(item) / 1000.0 for item in raw_bounds)
    except (TypeError, ValueError) as exc:
        raise GenericStepPlanningError("target_region.bounds 含有非数值。") from exc
    destination_bounds = None
    raw_destination = value.get("destination_bounds")
    if raw_destination is not None:
        if not isinstance(raw_destination, (list, tuple)) or len(raw_destination) != 4:
            raise GenericStepPlanningError(
                "target_region.destination_bounds 必须包含4个数值。"
            )
        try:
            destination_bounds = tuple(
                float(item) / 1000.0 for item in raw_destination
            )
        except (TypeError, ValueError) as exc:
            raise GenericStepPlanningError(
                "target_region.destination_bounds 含有非数值。"
            ) from exc
    return VisualTargetRegion(
        kind=str(value.get("kind") or "").strip().lower(),
        element_id=str(value.get("element_id") or "").strip(),
        bounds=bounds,  # type: ignore[arg-type]
        destination_element_id=str(
            value.get("destination_element_id") or ""
        ).strip(),
        destination_bounds=destination_bounds,  # type: ignore[arg-type]
        description=str(value.get("description") or "").strip()[:200],
    )


def _resolve_completion_evidence(
    evidence_ids: tuple[str, ...],
    observation: TrustedObservation,
) -> tuple[str, ...]:
    evidence: list[str] = []
    for evidence_id in evidence_ids:
        if evidence_id == "scene":
            if not observation.scene.summary.strip():
                raise GenericStepPlanningError("可信scene没有可引用的完成摘要。")
            evidence.append(f"scene:{observation.scene.summary}")
            continue
        element = observation.get_candidate(evidence_id)
        visible = element.label or (element.evidence[0] if element.evidence else element.meaning)
        evidence.append(f"{element.element_id}:{visible}")
    return tuple(evidence)


def _canonicalize_trusted_scene(
    scene: UIScene,
) -> tuple[UIScene, tuple[tuple[str, str], ...], tuple[dict[str, Any], ...]]:
    """Collapse duplicate descriptions of one visual object, preserving bounds.

    The canonical element is always one of the original observed elements. No
    coordinate is averaged or invented. Strongly overlapping but semantically
    conflicting elements stay separate and are reported as conflicts.
    """

    elements = list(scene.elements)
    if len(elements) < 2:
        return scene, (), ()
    parents = list(range(len(elements)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    conflicts: list[dict[str, Any]] = []
    for left in range(len(elements)):
        for right in range(left + 1, len(elements)):
            overlap = _bounds_overlap(elements[left].bounds, elements[right].bounds)
            compatible = _elements_semantically_compatible(
                elements[left],
                elements[right],
            )
            if overlap["intersection_over_smaller"] >= 0.85 and compatible:
                union(left, right)
            elif overlap["iou"] >= 0.5:
                conflicts.append(
                    {
                        "kind": "overlapping_semantic_conflict",
                        "element_ids": [
                            elements[left].element_id,
                            elements[right].element_id,
                        ],
                        "iou": round(overlap["iou"], 4),
                    }
                )

    groups: dict[int, list[UIElement]] = {}
    for index, element in enumerate(elements):
        groups.setdefault(find(index), []).append(element)
    canonical: list[UIElement] = []
    aliases: list[tuple[str, str]] = []
    for group in groups.values():
        selected = max(group, key=_canonical_element_rank)
        canonical.append(selected)
        if len(group) > 1:
            duplicate_ids = sorted(item.element_id for item in group)
            conflicts.append(
                {
                    "kind": "duplicate_visual_object_collapsed",
                    "canonical_element_id": selected.element_id,
                    "element_ids": duplicate_ids,
                }
            )
            aliases.extend(
                (item.element_id, selected.element_id)
                for item in group
                if item.element_id != selected.element_id
            )
    canonical.sort(key=lambda item: elements.index(item))
    if len(canonical) == len(elements):
        return scene, tuple(sorted(aliases)), tuple(conflicts)
    canonical_scene = UIScene(
            app_id=scene.app_id,
            screen_id=scene.screen_id,
            summary=scene.summary,
            elements=tuple(canonical),
            overlays=scene.overlays,
            stable=scene.stable,
            confidence=scene.confidence,
            fingerprint=scene.fingerprint,
            protocol_version=scene.protocol_version,
            system_ui=scene.system_ui,
            camera_alignment=scene.camera_alignment,
        )
    return (
        canonical_scene,
        tuple(sorted(aliases)),
        tuple(conflicts),
    )


def _trusted_target_local_candidate(
    scene: UIScene,
    conflicts: tuple[dict[str, Any], ...],
) -> UIElement | None:
    """Resolve one strong goal element and fail closed on unresolved overlap."""

    candidate = scene.unique_trusted_goal_element()
    if candidate is None:
        return None
    for conflict in conflicts:
        conflict_ids = tuple(str(item) for item in conflict.get("element_ids") or ())
        if candidate.element_id not in conflict_ids:
            continue
        if (
            conflict.get("kind") == "duplicate_visual_object_collapsed"
            and conflict.get("canonical_element_id") == candidate.element_id
        ):
            continue
        return None
    return candidate


def _canonical_element_rank(element: UIElement) -> tuple[int, float, float]:
    left, top, right, bottom = element.bounds
    area = (right - left) * (bottom - top)
    return (
        ROLE_PRIORITY.get(element.role, 0),
        float(element.confidence),
        -area,
    )


def _elements_semantically_compatible(left: UIElement, right: UIElement) -> bool:
    left_texts = {
        text.strip().casefold()
        for text in (left.label, *left.evidence)
        if text.strip()
    }
    right_texts = {
        text.strip().casefold()
        for text in (right.label, *right.evidence)
        if text.strip()
    }
    if left_texts and right_texts and left_texts.intersection(right_texts):
        return True
    return left.meaning.strip().casefold() == right.meaning.strip().casefold()


def _bounds_overlap(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> dict[str, float]:
    intersection_width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    intersection_height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    smaller = min(left_area, right_area)
    return {
        "iou": intersection / union if union > 0 else 0.0,
        "intersection_over_smaller": intersection / smaller if smaller > 0 else 0.0,
    }


def _exact_text_candidate_block(
    context: QwenTaskContext,
    observation: TrustedObservation,
) -> tuple[str, str] | None:
    """Reject missing or ambiguous structured exact-text targets locally."""

    for required_text in context.exact_text_requirements:
        matches = _matching_exact_text_candidates(
            context,
            observation,
            required_text,
        )
        if not matches:
            return (
                f"当前可信候选中不存在逐字一致文字：{required_text}",
                "exact_text_missing",
            )
        if len(matches) != 1:
            return (
                f"逐字一致文字目标不唯一：{required_text}，共{len(matches)}个",
                "exact_text_ambiguous",
            )
    return None


def _required_exact_candidate_ids(
    context: QwenTaskContext,
    observation: TrustedObservation,
) -> set[str]:
    result: set[str] = set()
    for required_text in context.exact_text_requirements:
        matches = _matching_exact_text_candidates(
            context,
            observation,
            required_text,
        )
        if len(matches) != 1:
            raise GenericStepPlanningError(
                "逐字一致文字约束缺少本地唯一可信候选。"
            )
        result.add(matches[0])
    return result


def _identity_text_candidate_block(
    context: QwenTaskContext,
    observation: TrustedObservation,
) -> tuple[str, str] | None:
    for required_text in context.identity_text_requirements:
        matches = _matching_identity_text_candidates(observation, required_text)
        if not matches:
            return (f"当前画面不存在收件人逐字身份：{required_text}", "identity_missing")
        if len(matches) != 1:
            return (f"当前画面收件人身份不唯一：{required_text}", "identity_ambiguous")
    return None


def _required_identity_candidate_ids(
    context: QwenTaskContext,
    observation: TrustedObservation,
) -> set[str]:
    result: set[str] = set()
    for required_text in context.identity_text_requirements:
        matches = _matching_identity_text_candidates(observation, required_text)
        if len(matches) != 1:
            raise GenericStepPlanningError("收件人身份缺少本地唯一逐字候选。")
        result.add(matches[0])
    return result


def _matching_identity_text_candidates(
    observation: TrustedObservation,
    required_text: str,
) -> list[str]:
    return [
        element.element_id
        for element in observation.scene.elements
        if float(element.confidence) >= MIN_TARGET_CONFIDENCE
        and element.states.get("visible") is not False
        and element.role != "input"
        and (
            element.states.get("identity_anchor") is True
            or element.states.get("goal_relevant") is True
            or element.meaning.strip().casefold()
            in {
                "recipient_identity",
                "conversation_identity",
                "conversation_title",
                "page_title",
            }
        )
        and required_text in (element.label, *element.evidence)
    ]


def _matching_exact_text_candidates(
    context: QwenTaskContext,
    observation: TrustedObservation,
    required_text: str,
) -> list[str]:
    roles = set(context.exact_text_target_roles)
    meanings = set(context.exact_text_target_meanings)
    matches: list[str] = []
    for element in observation.scene.elements:
        if float(element.confidence) < MIN_TARGET_CONFIDENCE:
            continue
        if required_text not in (element.label, *element.evidence):
            continue
        if roles and element.role not in roles:
            continue
        if meanings and element.meaning.strip().casefold() not in meanings:
            continue
        if context.current_external_impact != "read_only" and not roles:
            if element.role not in ACTIONABLE_EXACT_TEXT_ROLES:
                continue
        matches.append(element.element_id)
    return matches


def _local_blocked_decision(
    context: QwenTaskContext,
    observation: TrustedObservation,
    *,
    reason: str,
) -> QwenVisualDecision:
    decision = QwenVisualDecision(
        task_id=context.task_id,
        device_id=context.device_id,
        revision=context.revision,
        observation_id=observation.observation_id,
        fingerprint=observation.fingerprint,
        page_state=ModelPageState.from_trusted_scene(observation.scene),
        trusted_observation=observation,
        proposal=GenericStepProposal(status="blocked", reason=reason),
        target_region=None,
        expected_result={},
        confidence=min(float(observation.scene.confidence), 1.0),
        reason=reason,
    )
    decision.validate(context)
    return decision


def _vision_message(prompt: str, image: Image.Image) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": _image_data_url(image)}},
        ],
    }


def _decision_messages(prompt: str, image: Image.Image) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": (
                "你是受本地协议约束的单步视觉选择器。只输出一个语法完整的JSON对象；"
                "禁止Markdown、解释、思考过程、reasoning、analysis、额外字段、代码围栏"
                "或JSON对象前后的任何文字。"
            ),
        },
        _vision_message(prompt, image),
    ]


def _require_dict(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise VisionAgentError(f"{name} 必须是JSON对象。")
    return dict(value)


def _text_tuple(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise VisionAgentError(f"{name} 必须是字符串数组。")
    result = tuple(str(item).strip()[:500] for item in value)
    if any(not item for item in result):
        raise VisionAgentError(f"{name} 不能包含空字符串。")
    return result


def _dict_tuple(value: Any, name: str) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, (list, tuple)):
        raise VisionAgentError(f"{name} 必须是对象数组。")
    if any(not isinstance(item, dict) for item in value):
        raise VisionAgentError(f"{name} 只能包含JSON对象。")
    return tuple(dict(item) for item in value)


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0
