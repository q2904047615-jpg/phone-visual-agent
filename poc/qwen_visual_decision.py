from __future__ import annotations

import json
import os
import re
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Iterable

from PIL import Image

from generic_scene_observer import _local_frame_fingerprint, _safe_goal_context
from canonical_action_protocol import (
    SUPPORTED_ACTIONS,
    CanonicalActionProtocolError as GenericStepPlanningError,
    GenericStepProposal,
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
from semantic_action import SemanticAction
from task_semantic_ir import TaskSemanticIR
from ui_scene import MIN_TARGET_CONFIDENCE, UIElement, UIScene
from vision_agent import VisionAgentError
from vision_model_config import public_model_identity


QWEN_VISUAL_DECISION_PROTOCOL_VERSION = "2026-08-14-qwen-visual-decision-v5"
SUPPORTED_TASK_CONTEXT_PROTOCOL = "2026-08-20-deepseek-typed-task-graph-v4"
SUPPORTED_TASK_CONTEXT_PROTOCOLS = frozenset({SUPPORTED_TASK_CONTEXT_PROTOCOL})
QWEN_VISUAL_DECISION_MODEL_ROLE = "trusted_observation_single_step_selector"
MIN_DECISION_CONFIDENCE = 0.72
MIN_TRUSTED_FRAME_SHARPNESS = 4.0
SINGLE_ELEMENT_ACTIONS = frozenset(
    {
        "tap_semantic", "dismiss_overlay", "input_verified_text", "press_enter",
        "clear_verified_text", "double_tap", "long_press",
    }
)


def _targets_single_element(
    action_or_kind: SemanticAction | str,
    params: Mapping[str, Any] | None = None,
) -> bool:
    """Return whether this exact canonical action binds one observed element.

    Ordinary viewport swipes remain screen actions.  A swipe becomes an
    element action only when the canonical catalog binds its immutable
    ``element_id``; Qwen never invents that identity or any coordinates.
    """

    if isinstance(action_or_kind, SemanticAction):
        kind = action_or_kind.action
        values = action_or_kind.params
    else:
        kind = str(action_or_kind or "").strip()
        values = params or {}
    return kind in SINGLE_ELEMENT_ACTIONS or (
        kind == "swipe"
        and bool(str(values.get("element_id") or "").strip())
    )


QWEN_PROTOCOL_ACTIONS = frozenset(SUPPORTED_ACTIONS)

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


ALLOWED_EXECUTION_CLASSES = {
    "observe",
    "navigate",
    "effect",
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

@dataclass(frozen=True)
class QwenTaskContext(Mapping[str, Any]):
    protocol_version: str
    task_id: str
    device_id: str
    revision: int
    task_status: str
    goal: dict[str, Any]
    global_constraints: tuple[str, ...]
    goal_completion_conditions: tuple[dict[str, Any], ...]
    current_subgoal: dict[str, Any]
    current_execution_class: str
    effect_intents: tuple[dict[str, Any], ...]
    effect_gate: dict[str, Any]
    semantic_ir: TaskSemanticIR | None = field(
        default=None,
        repr=False,
        compare=False,
    )

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
            "current_execution_class",
            "effect_intents",
            "effect_gate",
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
            current_execution_class=str(
                value["current_execution_class"] or ""
            ).strip(),
            effect_intents=_dict_tuple(value["effect_intents"], "effect_intents"),
            effect_gate=_require_dict(
                value["effect_gate"], "effect_gate"
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
        if self.current_execution_class not in ALLOWED_EXECUTION_CLASSES:
            raise VisionAgentError(
                f"current_execution_class 无效：{self.current_execution_class}"
            )
        if self.semantic_ir is not None:
            self.semantic_ir.validate()
            expected_scope = (self.task_id, self.device_id, self.revision)
            actual_scope = (
                self.semantic_ir.task_id,
                self.semantic_ir.device_id,
                self.semantic_ir.revision,
            )
            if actual_scope != expected_scope:
                raise VisionAgentError("TaskSemanticIR 与 Qwen task scope 不一致。")

        subgoal_allowed = {
            "subgoal_id",
            "objective",
            "status",
            "depends_on",
            "constraints",
            "completion_conditions",
            "completion_evidence",
            "effect_ids",
            "execution_class",
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
            str(self.current_subgoal.get("execution_class") or "")
            != self.current_execution_class
        ):
            raise VisionAgentError(
                "current_subgoal.execution_class 与顶层上下文不一致。"
            )
        _safe_goal_context(self.to_dict())

        effect_allowed = {
            "effect_id",
            "kind",
            "target_entity_roles",
            "payload_entity_roles",
            "source_subgoal_ids",
            "expected_results",
            "local_policy",
        }
        policy_allowed = {"effect_id", "confirmation_required", "policy_level"}
        for index, item in enumerate(self.effect_intents):
            if set(item) != effect_allowed:
                raise VisionAgentError(
                    f"effect_intents[{index}] 字段不完整或包含协议外字段。"
                )
            for field in (
                "target_entity_roles",
                "payload_entity_roles",
                "source_subgoal_ids",
                "expected_results",
            ):
                _text_tuple(item[field], f"effect_intents[{index}].{field}")
            if not str(item.get("kind") or "").strip():
                raise VisionAgentError(f"effect_intents[{index}].kind 不能为空。")
            policy = _require_dict(
                item.get("local_policy"),
                f"effect_intents[{index}].local_policy",
            )
            if set(policy) != policy_allowed:
                raise VisionAgentError(
                    f"effect_intents[{index}].local_policy 字段不完整或包含协议外字段。"
                )
            if policy.get("effect_id") != item.get("effect_id"):
                raise VisionAgentError(
                    f"effect_intents[{index}].local_policy.effect_id 不一致。"
                )
            if not isinstance(policy.get("confirmation_required"), bool):
                raise VisionAgentError(
                    f"effect_intents[{index}].local_policy.confirmation_required 必须是布尔值。"
                )
            if not str(policy.get("policy_level") or "").strip():
                raise VisionAgentError(
                    f"effect_intents[{index}].local_policy.policy_level 不能为空。"
                )

        effect_ids = [str(item.get("effect_id") or "").strip() for item in self.effect_intents]
        if any(not item for item in effect_ids) or len(effect_ids) != len(set(effect_ids)):
            raise VisionAgentError("effect_intents 含空ID或重复ID。")
        subgoal_effect_ids = _text_tuple(
            self.current_subgoal.get("effect_ids") or [],
            "current_subgoal.effect_ids",
        )
        if len(subgoal_effect_ids) != len(set(subgoal_effect_ids)):
            raise VisionAgentError("current_subgoal.effect_ids 含重复效果ID。")
        gate_allowed = {
            "required",
            "state",
            "effect_ids",
            "effect_action_allowed",
        }
        if self.protocol_version == SUPPORTED_TASK_CONTEXT_PROTOCOL:
            gate_allowed.add("scope")
        if set(self.effect_gate) != gate_allowed:
            raise VisionAgentError("effect_gate 字段不完整或包含协议外字段。")
        required = self.effect_gate.get("required")
        allowed = self.effect_gate.get("effect_action_allowed")
        if not isinstance(required, bool) or not isinstance(allowed, bool):
            raise VisionAgentError("effect_gate 布尔字段格式无效。")
        state = str(self.effect_gate.get("state") or "").strip()
        if state not in {"not_required", "awaiting_confirmation", "confirmed"}:
            raise VisionAgentError(f"effect_gate.state 无效：{state}")
        gate_effect_ids = _text_tuple(
            self.effect_gate.get("effect_ids") or [],
            "effect_gate.effect_ids",
        )
        if len(gate_effect_ids) != len(set(gate_effect_ids)):
            raise VisionAgentError("effect_gate.effect_ids 含重复效果ID。")
        confirmation_effect_ids = {
            str(item.get("effect_id") or "").strip()
            for item in self.effect_intents
            if isinstance(item.get("local_policy"), dict)
            and item["local_policy"].get("confirmation_required") is True
        }
        if (
            set(gate_effect_ids) != confirmation_effect_ids
            or set(effect_ids) != set(subgoal_effect_ids)
        ):
            raise VisionAgentError(
                "effect_intents、current_subgoal 与 effect_gate 效果ID不一致。"
            )

        if self.protocol_version == SUPPORTED_TASK_CONTEXT_PROTOCOL:
            scope = _require_dict(
                self.effect_gate.get("scope"),
                "effect_gate.scope",
            )
            scope_allowed = {"task_id", "device_id", "revision", "subgoal_id"}
            if set(scope) != scope_allowed:
                raise VisionAgentError(
                    "effect_gate.scope 字段缺失或包含协议外字段。"
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
                        f"effect_gate.scope.{field} 与当前上下文不一致。"
                    )

        external = self.current_execution_class in {"effect", "unknown"}
        if self.current_execution_class == "unknown":
            raise VisionAgentError("unknown 子目标禁止进入视觉动作协议。")
        if external and confirmation_effect_ids:
            if not required or not gate_effect_ids:
                raise VisionAgentError("需确认的效果子目标必须关闭效果确认门。")
            if state not in {"awaiting_confirmation", "confirmed"}:
                raise VisionAgentError("外部状态子目标的确认门状态无效。")
            if state == "confirmed" and not allowed:
                raise VisionAgentError("确认门状态与 effect_action_allowed 冲突。")
            if state != "confirmed" and allowed:
                raise VisionAgentError("未确认效果不能允许受限效果动作。")
        elif external:
            if required or gate_effect_ids or state != "not_required" or not allowed:
                raise VisionAgentError("自动外部效果的本地策略授权状态无效。")
        elif required or gate_effect_ids or allowed or state != "not_required":
            raise VisionAgentError("只读/导航子目标不得伪造效果确认状态。")

    @property
    def effect_action_allowed(self) -> bool:
        return bool(
            self.protocol_version == SUPPORTED_TASK_CONTEXT_PROTOCOL
            and self.effect_gate["effect_action_allowed"]
        )

    @property
    def pre_observation_block_reason(self) -> str | None:
        """Return the local gate that must run before either Qwen call."""

        if self.current_execution_class == "unknown":
            return "unknown 子目标禁止调用观察或决策模型。"
        if self.current_execution_class == "effect":
            if not self.effect_action_allowed:
                return "本地效果确认门未满足，本轮禁止调用观察或决策模型。"
        return None

    @property
    def exact_text_requirements(self) -> tuple[str, ...]:
        """Return only literals whose active operation is literal selection.

        ``target_ui_label`` is visual context, not a second action authority.
        A phrase such as ``刷新图标`` or ``发送键`` may describe an icon whose
        visible label is empty.  Pre-blocking Qwen because that phrase is not
        present verbatim duplicates the canonical catalog and turns ordinary
        semantic navigation into a product boundary.  Recipient selection is
        different: the requested party is user data and must remain exact.
        Typed input payloads and field identities are enforced by the typed
        input transaction rather than by this visual-label gate.
        """

        values: list[str] = []
        for recipient in self.recipient_values:
            if subgoal_targets_recipient_control(recipient, self.current_subgoal):
                if recipient not in values:
                    values.append(recipient)
        return tuple(values)

    @property
    def recipient_values(self) -> tuple[str, ...]:
        entities = self.goal.get("entities") or {}
        if not isinstance(entities, dict):
            raise VisionAgentError("goal.entities 必须是JSON对象。")
        values: list[str] = []
        recipient = entities.get("recipient")
        if recipient is not None:
            values.append(recipient)
        recipients = entities.get("recipients")
        if recipients is not None:
            if not isinstance(recipients, list) or not 1 <= len(recipients) <= 32:
                raise VisionAgentError("goal.entities.recipients 格式无效。")
            values.extend(recipients)
        if any(
            not isinstance(item, str)
            or not item
            or len(item) > 100
            or item != item.strip()
            or "\n" in item
            or "\r" in item
            for item in values
        ) or len(values) != len(set(values)):
            raise VisionAgentError("goal.entities recipient/recipients 格式无效。")
        return tuple(values)

    @property
    def identity_text_requirements(self) -> tuple[str, ...]:
        entities = self.goal.get("entities") or {}
        if not isinstance(entities, dict):
            raise VisionAgentError("goal.entities 必须是JSON对象。")
        return tuple(
            recipient
            for recipient in self.recipient_values
            if subgoal_binds_recipient(recipient, self.current_subgoal)
            and not subgoal_targets_recipient_control(
                recipient,
                self.current_subgoal,
            )
        )

    @property
    def exact_text_target_roles(self) -> tuple[str, ...]:
        # Role/meaning hints are observation facts, not task-authority fields.
        return ()

    @property
    def requested_input_text(self) -> str | None:
        """Return the exact text authorized by DeepSeek, never model-invented text."""

        if self.semantic_ir is not None and self.semantic_ir.input_fields:
            active_id = str(self.current_subgoal.get("subgoal_id") or "")
            entities = {item.entity_id: item for item in self.semantic_ir.entities}
            relevant = [
                item
                for item in self.semantic_ir.input_fields
                if active_id in item.source_subgoal_ids
            ]
            if not relevant and len(self.semantic_ir.input_fields) == 1:
                relevant = [self.semantic_ir.input_fields[0]]
            if not relevant and len(self.semantic_ir.input_fields) > 1:
                raise VisionAgentError(
                    "当前子目标没有绑定唯一 typed input field，禁止猜测多个字段。"
                )
            if len(relevant) > 1:
                raise VisionAgentError("当前子目标同时绑定多个输入字段，缺少唯一字段选择。")
            if relevant:
                raw = entities[relevant[0].payload_ref].value
                if not isinstance(raw, str):
                    raise VisionAgentError("typed input payload 不是文字。")
                return raw

        entities = self.goal.get("entities") or {}
        raw = entities.get("input_text")
        if raw is None:
            return None
        if not isinstance(raw, str) or not raw or len(raw) > 4000 or "\r" in raw:
            raise VisionAgentError("goal.entities.input_text 必须为1～4000个字符。")
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
            "current_execution_class": self.current_execution_class,
            "effect_intents": [dict(item) for item in self.effect_intents],
            "effect_gate": dict(self.effect_gate),
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self):
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())

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
            "execution_class": self.current_execution_class,
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
        # The single-step observer permits one stale leading camera frame and only
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
        if _targets_single_element(action):
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
                if action.action
                in {
                    "back",
                    "home",
                    "open_recent_apps",
                    "reveal_system_navigation",
                }
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
        identity_block = _identity_text_candidate_block(
            context,
            self.trusted_observation,
        )
        if identity_block is not None:
            if self.proposal.status != "blocked":
                raise GenericStepPlanningError(
                    "当前收件人身份缺少本地唯一逐字视觉证据，必须 blocked。"
                )
        if self.protocol_version != QWEN_VISUAL_DECISION_PROTOCOL_VERSION:
            raise GenericStepPlanningError("Qwen视觉决策协议版本无效。")
        if self.proposal.status not in {"action", "blocked"}:
            raise GenericStepPlanningError(
                "Qwen本地选择器只能返回 action 或 blocked；完成状态由任务图裁决。"
            )
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
                context.current_execution_class in {"effect", "unknown"}
                and not context.effect_action_allowed
            ):
                raise GenericStepPlanningError("风险确认门未满足，禁止产生外部状态动作。")
            self.target_region.validate(self.trusted_observation, action)
            if _targets_single_element(action):
                element = self.trusted_observation.get_candidate(
                    str(action.params.get("element_id") or "")
                )
                exact_bound_element_ids = {
                    element.element_id,
                    str(element.states.get("input_element_id") or "").strip(),
                }
                exact_bound_element_ids.discard("")
                if exact_candidate_ids and exact_candidate_ids.isdisjoint(
                    exact_bound_element_ids
                ):
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
                if action.action == "press_enter":
                    if (
                        element.meaning != "input_exact_enter_key"
                        or element.states.get("input_enter_key") is not True
                        or element.states.get("key_action") != "newline"
                    ):
                        raise GenericStepPlanningError(
                            "换行动作必须绑定本地审计的可见 newline 键。"
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
            local_semantic_target = self.trusted_observation.target_local_candidate()
            if (
                local_semantic_target is not None
                and _targets_single_element(action)
                and not _formal_action_applies_effect(action)
                and str(action.params.get("element_id") or "")
                != local_semantic_target.element_id
            ):
                raise GenericStepPlanningError(
                    "动作没有绑定当前画面唯一的语义目标候选。"
                )
            if exact_candidate_ids and not _targets_single_element(action):
                raise GenericStepPlanningError(
                    "存在逐字一致文字约束时，动作必须绑定该唯一候选。"
                )
        elif self.target_region is not None:
            raise GenericStepPlanningError("blocked 不能携带动作目标区域。")

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
        }


class QwenVisualDecisionObserver:
    """Select one canonical action locally from one trusted observation."""

    def __init__(self, provider: Any) -> None:
        self.provider = provider
        self.last_raw_response = ""
        self.last_diagnostics: dict[str, Any] = {}
        self._metrics = {
            "decision_count": 0,
            "deterministic_action_count": 0,
            "final_blocked_count": 0,
        }

    def status(self) -> dict[str, Any]:
        value = dict(self.provider.status())
        decisions = self._metrics["decision_count"]
        value.update(
            {
                "visual_decision_protocol": QWEN_VISUAL_DECISION_PROTOCOL_VERSION,
                "task_context_protocol": SUPPORTED_TASK_CONTEXT_PROTOCOL,
                "model_role": QWEN_VISUAL_DECISION_MODEL_ROLE,
                "hardware_actions_enabled": False,
                "selection_authority": "canonical_action_catalog_local",
                **self._metrics,
                "deterministic_action_rate": _ratio(
                    self._metrics["deterministic_action_count"], decisions
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
        navigation_history: Iterable[Mapping[str, Any]] = (),
    ) -> QwenVisualDecision:
        started = time.perf_counter()
        self.last_raw_response = ""
        self.last_diagnostics = {}
        context = (
            task_context
            if isinstance(task_context, QwenTaskContext)
            else QwenTaskContext.from_dict(task_context)
        )
        context.validate()
        available_actions = _normalize_available_action_kinds(
            available_action_kinds
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
        canonical_choices = _selection_choices(
            context,
            trusted_observation,
            available_actions,
        )
        canonical_action_kinds = sorted(
            {str(item["action"]) for item in canonical_choices}
        )

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
            "hardware_actions_enabled": False,
            "available_action_kinds": canonical_action_kinds,
            "canonical_choice_count": len(canonical_choices),
            "canonical_choices": [
                {
                    "action": str(item.get("action") or ""),
                    "direction": str(item.get("direction") or ""),
                    "element_id": str(item.get("element_id") or ""),
                }
                for item in canonical_choices
            ],
            "device_action_kinds": sorted(available_actions),
        }
        self.last_diagnostics = dict(base_diagnostics)

        if (
            context.current_execution_class in {"effect", "unknown"}
            and not context.effect_action_allowed
        ):
            decision = _local_blocked_decision(
                context,
                trusted_observation,
                reason="风险确认门未满足，本轮禁止提出外部状态动作。",
            )
            self._metrics["final_blocked_count"] += 1
            self.last_diagnostics.update(
                {
                    "local_safety_block": "effect_gate",
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

        deterministic_selection = _deterministic_exact_selection_payload(
            context,
            canonical_choices,
            observation=trusted_observation,
        )
        if deterministic_selection is not None:
            raw = json.dumps(
                deterministic_selection,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            decision = _hydrate_canonical_selection(
                deterministic_selection,
                context=context,
                observation=trusted_observation,
                choices=canonical_choices,
            )
            self.last_raw_response = raw
            self._metrics["deterministic_action_count"] += 1
            self.last_diagnostics.update(
                {
                    "local_deterministic_selection": True,
                    "decision_status": decision.proposal.status,
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                }
            )
            return decision

        reason = (
            "当前canonical目录未能依据本步当前截图中的唯一视觉目标确定单一动作；"
            "本地选择器停止，不沿用历史页面动作，也不发起重复模型请求。"
        )
        decision = _local_blocked_decision(
            context,
            trusted_observation,
            reason=reason,
        )
        self._metrics["final_blocked_count"] += 1
        self.last_diagnostics.update(
            {
                "local_safety_block": "single_step_candidate_not_unique",
                "local_deterministic_selection": False,
                "model_calls": 0,
                "decision_status": "blocked",
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        )
        return decision


def _selection_choices(
    context: QwenTaskContext,
    observation: TrustedObservation,
    available_action_kinds: frozenset[str],
) -> tuple[dict[str, Any], ...]:
    """Build generic action choices from the trusted scene, never app steps."""

    choices: list[dict[str, Any]] = []
    if context.semantic_ir is None:
        raise VisionAgentError("typed v4 视觉选择缺少 canonical TaskSemanticIR。")
    try:
        from canonical_action_protocol import (
            canonical_candidate_expected_result,
            compile_canonical_action_catalog,
        )

        formal_report = compile_canonical_action_catalog(
            observation.scene,
            context.semantic_ir,
            available_action_kinds,
        )
    except Exception as exc:
        raise VisionAgentError(
            f"canonical action catalog 构建失败：{exc}"
        ) from exc

    # Qwen receives a presentation of the canonical catalog, not a separately
    # rebuilt action list.  Every action parameter and postcondition below is a
    # deterministic projection of the same immutable candidate that Policy
    # later selects by digest and ID.
    for candidate in formal_report.candidates:
        choices.append(
            {
                "choice_id": f"choice_{len(choices) + 1}",
                "action": candidate.action_kind,
                **dict(candidate.parameters),
                "expected_result": canonical_candidate_expected_result(
                    candidate,
                    observation.scene,
                ),
                "formal_candidate_id": candidate.candidate_id,
                "formal_report_digest": formal_report.report_digest,
                "formal_transition": candidate.transition.to_dict(),
            }
        )
    return tuple(choices)


def _deterministic_exact_selection_payload(
    context: QwenTaskContext,
    choices: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    *,
    observation: TrustedObservation | None = None,
) -> dict[str, Any] | None:
    """Select one canonical candidate from the sole step observation.

    Exact wrappers retain their stronger typed filters.  General goals may
    select only when the scene's unique ``goal_relevant`` element and the
    canonical catalog identify exactly one same candidate, or when the catalog
    itself contains one coordinate-free primitive.  No wording heuristic or
    second action owner is introduced here.
    """

    def action_payload(choice: Mapping[str, Any], reason: str) -> dict[str, Any] | None:
        choice_id = str(choice.get("choice_id") or "").strip()
        if not choice_id:
            return None
        return {
            "status": "action",
            "choice_id": choice_id,
            "confidence": 1.0,
            "reason": reason,
        }

    active_id = str(context.current_subgoal.get("subgoal_id") or "").strip()
    if active_id == "input_exact_text":
        local_target = (
            observation.target_local_candidate()
            if observation is not None
            else None
        )
        if local_target is None:
            return None
        current_value = getattr(local_target, "states", {}).get("value")
        authorized_text = getattr(context, "requested_input_text", None)
        permitted_actions = {
            "clear_verified_text"
        } if (
            isinstance(current_value, str)
            and isinstance(authorized_text, str)
            and not authorized_text.startswith(current_value)
        ) else {
            "tap_semantic",
            "input_verified_text",
            "press_enter",
            "clear_verified_text",
        }
        matching_choices = tuple(
            choice
            for choice in choices
            if str(choice.get("element_id") or "").strip()
            == local_target.element_id
            and str(choice.get("action") or "").strip()
            in permitted_actions
        )
    else:
        expected_action = {
            "exact_back": "back",
            "exact_home": "home",
            "exact_open_recent_apps": "open_recent_apps",
            "exact_tap_semantic": "tap_semantic",
        }.get(active_id)
        matching_choices = (
            tuple(
                choice
                for choice in choices
                if str(choice.get("action") or "").strip() == expected_action
            )
            if expected_action is not None
            else ()
        )
        if active_id == "exact_tap_semantic" and observation is not None:
            local_target = observation.target_local_candidate()
            if local_target is not None:
                matching_choices = tuple(
                    choice
                    for choice in matching_choices
                    if str(choice.get("element_id") or "").strip()
                    == local_target.element_id
                )
    if len(matching_choices) == 1:
        return action_payload(
            matching_choices[0],
            "结构化直推目录只有一个合法 canonical candidate。",
        )

    if (
        getattr(context, "current_execution_class", "") == "effect"
        and bool(getattr(context, "effect_action_allowed", False))
    ):
        effect_choices = tuple(
            choice for choice in choices if _choice_applies_effect(choice)
        )
        if len(effect_choices) == 1:
            return action_payload(
                effect_choices[0],
                "canonical目录只有一个绑定当前EffectIntent的动作。",
            )
        if effect_choices:
            return None

    if observation is None:
        return None
    local_target = observation.target_local_candidate()
    if local_target is not None:
        same_target = tuple(
            choice
            for choice in choices
            if str(choice.get("element_id") or "").strip()
            == local_target.element_id
        )
        if len(same_target) == 1:
            return action_payload(
                same_target[0],
                "单次Qwen画面的唯一目标与canonical目录唯一候选一致。",
            )

    if len(choices) == 1 and str(choices[0].get("action") or "") in {
        "back",
        "home",
        "open_recent_apps",
        "reveal_system_navigation",
        "swipe",
        "wait_for_change",
    }:
        return action_payload(
            choices[0],
            "canonical目录只有一个坐标无关或容器级合法动作。",
        )
    return None


def _choice_applies_effect(choice: Mapping[str, Any]) -> bool:
    transition = choice.get("formal_transition")
    if not isinstance(transition, Mapping):
        return False
    expectations = transition.get("expectations")
    if not isinstance(expectations, list):
        return False
    return any(
        isinstance(item, Mapping)
        and item.get("predicate") == "effect.applied"
        and item.get("operator") == "equals"
        and item.get("value") is True
        for item in expectations
    )


def _formal_action_applies_effect(action: SemanticAction) -> bool:
    """Recognize only a locally hydrated canonical effect transition.

    Qwen selects a ``choice_id`` and cannot author ``formal_transition``.
    Therefore an exact ``effect.applied = true`` expectation identifies the
    canonical effect control without letting the observer's generic
    ``goal_relevant`` flag establish a second action owner.  Non-effect and
    legacy actions keep the existing unique semantic-target check.
    """

    transition = action.params.get("formal_transition")
    if not isinstance(transition, Mapping):
        return False
    expectations = transition.get("expectations")
    if not isinstance(expectations, list):
        return False
    return any(
        isinstance(item, Mapping)
        and item.get("predicate") == "effect.applied"
        and item.get("operator") == "equals"
        and item.get("value") is True
        for item in expectations
    )


def _hydrate_canonical_selection(
    payload: Mapping[str, Any],
    *,
    context: QwenTaskContext,
    observation: TrustedObservation,
    choices: tuple[dict[str, Any], ...],
) -> QwenVisualDecision:
    """Hydrate the already-selected immutable canonical candidate."""

    allowed = {"status", "choice_id", "confidence", "reason"}
    if set(payload) != allowed:
        missing = sorted(allowed - set(payload))
        extra = sorted(set(payload) - allowed)
        details = []
        if missing:
            details.append("缺少字段：" + ", ".join(missing))
        if extra:
            details.append("包含协议外字段：" + ", ".join(extra))
        raise VisionAgentError("本地 canonical 选择结构无效；" + "；".join(details))
    if str(payload.get("status") or "").strip().lower() != "action":
        raise VisionAgentError("本地 canonical 选择结果必须是 action。")
    choice_id = str(payload.get("choice_id") or "").strip()
    matches = tuple(
        item for item in choices if str(item.get("choice_id") or "") == choice_id
    )
    if len(matches) != 1:
        raise VisionAgentError("本地选择引用了不存在或不唯一的 choice_id。")
    choice = matches[0]
    kind = str(choice.get("action") or "").strip()
    if kind not in SUPPORTED_ACTIONS:
        raise VisionAgentError("canonical candidate 包含未知动作。")
    raw_expected = choice.get("expected_result")
    if not isinstance(raw_expected, Mapping) or not raw_expected:
        raise VisionAgentError("canonical candidate 缺少可验证 expected_result。")
    expected_result = dict(raw_expected)
    params = {
        key: value
        for key, value in choice.items()
        if key
        not in {
            "choice_id",
            "action",
            "expected_result",
            "selection_context",
        }
    }
    if _targets_single_element(kind, params):
        element_id = str(params.get("element_id") or "").strip()
        element = observation.get_candidate(element_id)
        params.update(
            {
                "element_id": element.element_id,
                "target": element.meaning,
                "role": element.role,
                "label": element.label,
                "states": dict(element.states),
            }
        )
        if kind == "input_verified_text":
            params["text"] = context.requested_input_text
        elif kind == "long_press":
            params["duration_ms"] = 800
    elif kind == "drag":
        for prefix in ("source_", "destination_"):
            element = observation.get_candidate(
                str(params.get(f"{prefix}element_id") or "").strip()
            )
            params.update(
                {
                    f"{prefix}element_id": element.element_id,
                    f"{prefix}target": element.meaning,
                    f"{prefix}role": element.role,
                    f"{prefix}label": element.label,
                    f"{prefix}states": dict(element.states),
                }
            )
    params["expected_effect"] = expected_result
    action = SemanticAction(
        node_id=f"qwen_visual_revision_{context.revision}",
        action=kind,
        params=params,
    )
    raw_confidence = payload.get("confidence")
    if isinstance(raw_confidence, bool) or not isinstance(
        raw_confidence, (int, float)
    ):
        raise VisionAgentError("本地 canonical 选择 confidence 无效。")
    confidence = min(
        float(raw_confidence),
        float(observation.scene.confidence),
    )
    if _targets_single_element(kind, params):
        confidence = min(
            confidence,
            float(
                observation.get_candidate(
                    str(params.get("element_id") or "")
                ).confidence
            ),
        )
    elif kind == "drag":
        confidence = min(
            confidence,
            *(
                float(
                    observation.get_candidate(
                        str(params.get(f"{prefix}element_id") or "")
                    ).confidence
                )
                for prefix in ("source_", "destination_")
            ),
        )
    reason = str(payload.get("reason") or "").strip()[:500]
    decision = QwenVisualDecision(
        task_id=context.task_id,
        device_id=context.device_id,
        revision=context.revision,
        observation_id=observation.observation_id,
        fingerprint=observation.fingerprint,
        page_state=ModelPageState.from_trusted_scene(observation.scene),
        trusted_observation=observation,
        proposal=GenericStepProposal(
            status="action",
            action=action,
            reason=reason,
        ),
        target_region=_canonical_target_region(action, observation),
        expected_result=expected_result,
        confidence=confidence,
        reason=reason,
    )
    decision.validate(context)
    return decision


def _canonical_target_region(
    action: SemanticAction,
    observation: TrustedObservation,
) -> VisualTargetRegion:
    if _targets_single_element(action):
        element = observation.get_candidate(
            str(action.params.get("element_id") or "").strip()
        )
        return VisualTargetRegion(
            kind="element",
            element_id=element.element_id,
            bounds=element.bounds,
            description=element.label or element.meaning,
        )
    if action.action == "drag":
        source = observation.get_candidate(
            str(action.params.get("source_element_id") or "").strip()
        )
        destination = observation.get_candidate(
            str(action.params.get("destination_element_id") or "").strip()
        )
        return VisualTargetRegion(
            kind="element_path",
            element_id=source.element_id,
            bounds=source.bounds,
            destination_element_id=destination.element_id,
            destination_bounds=destination.bounds,
            description=(
                f"{source.label or source.meaning} 到 "
                f"{destination.label or destination.meaning}"
            ),
        )
    is_system = action.action in {
        "back",
        "home",
        "open_recent_apps",
        "reveal_system_navigation",
    }
    description = {
        "back": "系统返回区域",
        "home": "Android系统Home键",
        "open_recent_apps": "Android系统最近任务键",
        "reveal_system_navigation": "Android系统导航栏",
    }.get(action.action, "当前屏幕")
    return VisualTargetRegion(
        kind="system_navigation" if is_system else "screen",
        bounds=(0.0, 0.0, 1.0, 1.0),
        description=description,
    )


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
        raise VisionAgentError("设备没有任何可供本地选择的 canonical 动作。")
    return normalized


def _typed_required_action_kinds(context: QwenTaskContext) -> frozenset[str]:
    """Read typed action requirements for identity/evidence checks only."""

    semantic_ir = context.semantic_ir
    if semantic_ir is None:
        return frozenset()
    active_id = str(context.current_subgoal.get("subgoal_id") or "")
    subgoal = next(
        (item for item in semantic_ir.subgoals if item.subgoal_id == active_id),
        None,
    )
    if subgoal is None:
        return frozenset()
    constraints = {
        item.constraint_id: item for item in semantic_ir.constraints
    }
    return frozenset(
        str(constraints[constraint_ref].value)
        for constraint_ref in subgoal.constraint_refs
        if constraint_ref in constraints
        and constraints[constraint_ref].kind == "required_action"
    )


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
            exact_same_role = bool(
                elements[left].label.strip()
                and elements[left].label.strip().casefold()
                == elements[right].label.strip().casefold()
                and elements[left].role == elements[right].role
            )
            if compatible and (
                overlap["intersection_over_smaller"] >= 0.85
                or (exact_same_role and overlap["iou"] >= 0.5)
            ):
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


def _canonical_element_rank(element: UIElement) -> tuple[int, int, float, float]:
    left, top, right, bottom = element.bounds
    area = (right - left) * (bottom - top)
    locally_audited_input_control = int(
        element.element_id.startswith("local_audited_")
        and (
            (
                element.meaning == "ime_exact_candidate"
                and element.states.get("ime_candidate") is True
            )
            or (
                element.meaning == "input_exact_literal_key"
                and element.states.get("input_literal_key") is True
            )
            or element.states.get("keyboard_layout_switch") is True
            or element.states.get("keyboard_case_switch") is True
            or element.states.get("keyboard_input_mode_switch") is True
        )
    )
    return (
        locally_audited_input_control,
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


def _launcher_app_entry_candidate_ids(
    context: QwenTaskContext,
    observation: TrustedObservation,
) -> tuple[str, ...]:
    """Return one typed App entry before applying inner-page text gates."""

    semantic_ir = context.semantic_ir
    scene = observation.scene
    if semantic_ir is None:
        return ()
    current_identity = f"{scene.foreground_app_id} {scene.screen_id}".casefold()
    if not any(token in current_identity for token in ("launcher", "home_screen", "desktop")):
        return ()
    active_id = str(context.current_subgoal.get("subgoal_id") or "")
    typed_subgoal = next(
        (item for item in semantic_ir.subgoals if item.subgoal_id == active_id),
        None,
    )
    surfaces = {item.surface_id: item for item in semantic_ir.surfaces}
    target_surface = (
        surfaces.get(typed_subgoal.surface_ref)
        if typed_subgoal is not None
        else None
    )
    if target_surface is None or target_surface.kind != "app":
        return ()
    app_name = str(target_surface.app_name or "").strip()
    if not app_name:
        return ()
    matches = tuple(
        element.element_id
        for element in scene.elements
        if element.role in {"button", "icon", "list_item"}
        and element.label == app_name
        and float(element.confidence) >= MIN_TARGET_CONFIDENCE
        and element.states.get("goal_relevant") is True
        and element.states.get("fully_visible") is True
    )
    return matches if len(matches) == 1 else ()


def _exact_text_candidate_block(
    context: QwenTaskContext,
    observation: TrustedObservation,
) -> tuple[str, str] | None:
    """Reject missing or ambiguous structured exact-text targets locally."""

    if _launcher_app_entry_candidate_ids(context, observation):
        return None
    for required_text in context.exact_text_requirements:
        identity_matches = _identity_scoped_exact_text_matches(
            context,
            observation,
            required_text,
        )
        if identity_matches is not None:
            if not identity_matches:
                return (
                    f"当前可信页面身份中不存在逐字一致文字：{required_text}",
                    "exact_text_missing",
                )
            if len(identity_matches) != 1:
                return (
                    f"逐字一致页面身份不唯一：{required_text}，"
                    f"共{len(identity_matches)}个",
                    "exact_text_ambiguous",
                )
            continue
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
    if _launcher_app_entry_candidate_ids(context, observation):
        return set()
    result: set[str] = set()
    for required_text in context.exact_text_requirements:
        identity_matches = _identity_scoped_exact_text_matches(
            context,
            observation,
            required_text,
        )
        if identity_matches is not None:
            if len(identity_matches) != 1:
                raise GenericStepPlanningError(
                    "逐字一致页面身份缺少本地唯一可信候选。"
                )
            # Swipe/back/home/wait/reveal-navigation do not act on the title
            # or identity anchor itself.  The exact text proves the current
            # surface only; it must not become an element-bound action target.
            continue
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
    if _launcher_app_entry_candidate_ids(context, observation):
        return None
    for required_text in context.identity_text_requirements:
        matches = _matching_identity_text_candidates(observation, required_text)
        if not matches:
            return (f"当前画面不存在收件人逐字身份：{required_text}", "identity_missing")
        if len(matches) != 1:
            return (f"当前画面收件人身份不唯一：{required_text}", "identity_ambiguous")
    return None


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


_IDENTITY_SCOPED_EXACT_TEXT_ACTIONS = frozenset(
    {
        "swipe",
        "back",
        "home",
        "open_recent_apps",
        "reveal_system_navigation",
        "wait_for_change",
    }
)

_SURFACE_IDENTITY_TYPE_SUFFIXES = frozenset(
    {
        "页",
        "页面",
        "界面",
        "屏幕",
        "窗口",
        "主页",
        "首页",
        "聊天页",
        "聊天页面",
        "对话页",
        "对话页面",
        "详情页",
        "详情页面",
        "列表页",
        "列表页面",
        "设置页",
        "设置页面",
        " page",
        " screen",
        " window",
        " chat page",
        " conversation page",
        " detail page",
        " list page",
        " settings page",
    }
)


def _surface_identity_text_matches(label: str, required_text: str) -> bool:
    """Match one literal title plus a bounded generic surface-type suffix."""

    literal = label.strip()
    required = required_text.strip()
    if not literal or not required:
        return False
    if literal == required:
        return True
    if required.startswith(literal):
        return required[len(literal) :].casefold() in _SURFACE_IDENTITY_TYPE_SUFFIXES
    if literal.startswith(required):
        return literal[len(required) :].casefold() in _SURFACE_IDENTITY_TYPE_SUFFIXES
    return False


def _matching_surface_identity_candidates(
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
                "conversation_title",
                "page_title",
            }
        )
        and _surface_identity_text_matches(element.label, required_text)
    ]


def _surface_descriptor_identity_candidate_ids(
    scene: UIScene,
    required_text: str,
) -> tuple[str, ...]:
    """Bind a generic page descriptor to its visible literal title.

    A typed ``target_ui_label`` can name the current page while the physical
    target is a separate input or button.  Only a strict generic type suffix
    plus a shorter visible title establishes this relation.  Exact labels stay
    element targets; zero or multiple titles never grant action authority.
    """

    required = str(required_text or "").strip()
    if not required:
        return ()
    matches = tuple(
        element.element_id
        for element in scene.elements
        if float(element.confidence) >= MIN_TARGET_CONFIDENCE
        and element.states.get("visible") is not False
        and element.role != "input"
        and (
            element.states.get("identity_anchor") is True
            or element.states.get("goal_relevant") is True
            or element.meaning.strip().casefold()
            in {"conversation_title", "page_title"}
        )
        and _surface_identity_text_matches(element.label, required)
    )
    has_descriptor_title = any(
        element.element_id in matches and element.label.strip() != required
        for element in scene.elements
    )
    return matches if has_descriptor_title else ()


def _identity_scoped_exact_text_matches(
    context: QwenTaskContext,
    observation: TrustedObservation,
    required_text: str,
) -> list[str] | None:
    """Resolve exact text as surface identity for non-element actions.

    A literal carried by the typed goal may name the current page or
    container (for example a conversation title) while the typed active
    action is a viewport gesture or a coordinate-free system action.  In that
    case the literal must still be uniquely visible, but binding the physical
    action to that title would invert the entity relation.  Element-bound
    actions deliberately keep the existing strict target requirement.
    """

    descriptor_matches = _surface_descriptor_identity_candidate_ids(
        observation.scene,
        required_text,
    )
    if descriptor_matches:
        return list(descriptor_matches)
    required_actions = _typed_required_action_kinds(context)
    if (
        len(required_actions) != 1
        or not required_actions.issubset(_IDENTITY_SCOPED_EXACT_TEXT_ACTIONS)
    ):
        return None
    return _matching_surface_identity_candidates(observation, required_text)


def _matching_exact_text_candidates(
    context: QwenTaskContext,
    observation: TrustedObservation,
    required_text: str,
) -> list[str]:
    active_input_matches = _active_input_transaction_exact_candidate_ids(
        context,
        observation,
        required_text,
    )
    if active_input_matches is not None:
        return active_input_matches
    roles = set(context.exact_text_target_roles)
    meanings = set(context.exact_text_target_meanings)
    matches: list[str] = []
    for element in observation.scene.elements:
        if float(element.confidence) < MIN_TARGET_CONFIDENCE:
            continue
        literal_match = required_text in (element.label, *element.evidence)
        if not literal_match:
            continue
        if roles and element.role not in roles:
            continue
        if meanings and element.meaning.strip().casefold() not in meanings:
            continue
        if context.current_execution_class != "observe" and not roles:
            if element.role not in ACTIONABLE_EXACT_TEXT_ROLES:
                continue
        matches.append(element.element_id)
    return matches


def _active_input_transaction_exact_candidate_ids(
    context: QwenTaskContext,
    observation: TrustedObservation,
    required_text: str,
) -> list[str] | None:
    """Keep one typed input field bound after its placeholder disappears.

    The field label is task authority, while the current value, geometry and
    uniqueness remain fresh local observation facts.  This bridge applies only
    to the active typed input field and only when its current value is an exact
    prefix of the authorized payload.  It therefore cannot turn another input
    or an arbitrary similarly labelled element into the action target.
    """

    semantic_ir = context.semantic_ir
    if semantic_ir is None:
        return None
    active_id = str(context.current_subgoal.get("subgoal_id") or "")
    fields = tuple(
        field
        for field in semantic_ir.input_fields
        if active_id in field.source_subgoal_ids
        and field.field_label in {"", required_text}
    )
    if len(fields) != 1:
        return None
    field = fields[0]
    entities = {item.entity_id: item for item in semantic_ir.entities}
    payload = entities.get(field.payload_ref)
    if (
        payload is None
        or payload.role != "input_text"
        or not isinstance(payload.value, str)
        or not payload.value
    ):
        return None

    matches: list[str] = []
    active_field_nonempty_seen = False
    for element in observation.scene.elements:
        current_value = element.states.get("value")
        if (
            element.states.get("input_field_id") == field.field_id
            and isinstance(current_value, str)
            and current_value
        ):
            active_field_nonempty_seen = True
        if (
            element.role != "input"
            or element.meaning != "application_text_input"
            or float(element.confidence) < MIN_TARGET_CONFIDENCE
            or element.states.get("goal_relevant") is not True
            or element.states.get("fully_visible") is not True
            or element.states.get("input_field_id") != field.field_id
            or (
                bool(field.field_label)
                and element.states.get("input_field_label") != field.field_label
            )
            or not isinstance(current_value, str)
            or not current_value
            or not payload.value.startswith(current_value)
            or element.label != current_value
        ):
            continue
        matches.append(element.element_id)
    # An empty field still visibly carries its placeholder/label, so the
    # ordinary literal matcher remains authoritative until text replaces it.
    return matches if active_field_nonempty_seen else None


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
