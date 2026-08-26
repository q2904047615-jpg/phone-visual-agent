from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

from task_semantic_ir import (
    ConstraintIntent,
    EffectIntent,
    SemanticEntity,
    TaskSemanticIR,
)
from semantic_action import SemanticAction
from ui_scene import UIElement, UIScene, scene_surface_kind
from verified_text_transaction import (
    VerifiedTextTransactionError,
    plan_from_input_states,
)


CANONICAL_ACTION_PROTOCOL = "2026-08-20-canonical-action-v1"

_IDEMPOTENT_SYSTEM_ACTION_SURFACE_KINDS = {
    "home": "launcher",
    "open_recent_apps": "recent_tasks",
}


def expected_idempotent_system_surface_kind(action_kind: str) -> str | None:
    """Return the canonical surface that makes a system action a no-op."""

    return _IDEMPOTENT_SYSTEM_ACTION_SURFACE_KINDS.get(
        str(action_kind or "").strip()
    )

_EFFECT_CONTROL_MEANINGS = {
    "send_message": frozenset({"send_message"}),
    "publish_content": frozenset(
        {"publish_content", "publish", "post_content", "comment", "reply"}
    ),
    "relationship_change": frozenset(
        {"follow", "unfollow", "subscribe", "unsubscribe", "favorite", "unfavorite"}
    ),
    "membership_change": frozenset(
        {"join", "leave", "invite", "remove_member"}
    ),
}
_ALL_EFFECT_CONTROL_MEANINGS = frozenset(
    meaning
    for meanings in _EFFECT_CONTROL_MEANINGS.values()
    for meaning in meanings
)


def _element_realizes_effect(element: UIElement, effect_kind: str) -> bool:
    """Bind a typed effect only to its canonical visible action control."""

    meanings = _EFFECT_CONTROL_MEANINGS.get(effect_kind, frozenset())
    return bool(
        meanings
        and element.role in {"button", "icon", "toggle"}
        and element.meaning in meanings
        and element.states.get("visible") is not False
        and element.states.get("enabled") is not False
        and element.states.get("fully_visible") is True
    )

MIN_ELEMENT_CONFIDENCE = 0.72
MIN_READY_CANDIDATES = 1
MAX_READY_CANDIDATES = 24

ELEMENT_ACTION_ROLES = frozenset(
    {"button", "icon", "input", "tab", "toggle", "list_item"}
)
SUPPORTED_ACTIONS = frozenset(
    {
        "tap_semantic",
        "dismiss_overlay",
        "swipe",
        "back",
        "home",
        "open_recent_apps",
        "reveal_system_navigation",
        "input_verified_text",
        "press_enter",
        "clear_verified_text",
        "double_tap",
        "long_press",
        "drag",
        "wait_for_change",
    }
)
RELATION_KINDS = frozenset(
    {
        "on_surface",
        "contains",
        "overlaps",
        "above",
        "below",
        "left_of",
        "right_of",
        "exact_literal_match",
        "binds_effect_target",
        "binds_effect_payload",
        "binds_surface",
        "binds_next_input_field",
    }
)
EXPECTATION_OPERATORS = frozenset(
    {"equals", "not_equals", "present", "absent", "changed"}
)
CLAIM_PREDICATES = frozenset(
    {
        "surface.kind",
        "surface.foreground_app_id",
        "surface.screen_id",
        "surface.stable",
        "surface.overlay_present",
        "element.exists",
        "element.role",
        "element.meaning",
        "element.label",
    }
)
EXPECTATION_PREDICATES = frozenset(
    {
        "surface.kind",
        "surface.active_ref",
        "surface.focused_entity_ref",
        "surface.overlay_present",
        "surface.navigation_depth",
        "surface.viewport",
        "system_ui.navigation_bar_visible",
        "observation.changed",
        "scene.changed",
        "element.exists",
        "element.state.focused",
        "element.state.value",
        "element.state.ime_preedit_text",
        "element.state.ime_exact_candidate_text",
        "element.state.keyboard_layout",
        "element.state.keyboard_input_mode",
        "element.state.keyboard_case_mode",
        "element.state.interaction_result",
        "element.state.location_relation",
        "input_field.focused",
        "effect.applied",
    }
)
_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_SAFE_STATE_KEYS = frozenset(
    {
        "enabled",
        "visible",
        "fully_visible",
        "focused",
        "selected",
        "checked",
        "value",
        "keyboard_layout",
        "keyboard_input_mode",
        "keyboard_case_mode",
        "scrollable",
        "scroll_axis",
        "page_index",
        "page_count",
        "input_next_field_key",
        "key_action",
        "source_input_field_id",
        "target_input_field_id",
        "target_input_field_label",
        "ime_preedit_text",
        "navigation_bar_visible",
    }
)


class CanonicalActionProtocolError(ValueError):
    pass


_FORMAL_AUTHORITY_PARAMS = frozenset(
    {"formal_candidate_id", "formal_report_digest", "formal_transition"}
)


@dataclass(frozen=True)
class GenericStepProposal:
    """Selected canonical action or a local blocked state.

    This is a transport value only.  The canonical catalog remains the sole
    action-candidate authority. Task completion is not representable here.
    """

    status: str
    action: SemanticAction | None = None
    reason: str = ""

    def validate(self, scene: UIScene) -> None:
        scene.validate()
        if self.status not in {"action", "blocked"}:
            raise CanonicalActionProtocolError(
                f"不支持的单步状态：{self.status}"
            )
        if self.status == "action":
            if self.action is None:
                raise CanonicalActionProtocolError("action 状态缺少唯一动作。")
            if self.action.action not in SUPPORTED_ACTIONS:
                raise CanonicalActionProtocolError(
                    f"单步动作不在 canonical 动作集合：{self.action.action}"
                )
            if self.action.action in {
                "tap_semantic",
                "dismiss_overlay",
                "input_verified_text",
                "press_enter",
                "clear_verified_text",
                "double_tap",
                "long_press",
            }:
                element_id = str(
                    self.action.params.get("element_id") or ""
                ).strip()
                if not element_id:
                    raise CanonicalActionProtocolError(
                        "元素动作必须引用当前场景 element_id。"
                    )
                scene.get_element(element_id)
            if self.action.action == "input_verified_text":
                text = self.action.params.get("text")
                if (
                    not isinstance(text, str)
                    or not text
                    or len(text) > 4000
                    or "\r" in text
                ):
                    raise CanonicalActionProtocolError(
                        "输入动作 text 必须为1～4000字符；换行由可见 Enter 键分段执行。"
                    )
                if (
                    scene.get_element(str(self.action.params["element_id"])).role
                    != "input"
                ):
                    raise CanonicalActionProtocolError(
                        "输入动作必须绑定 input 元素。"
                    )
            if self.action.action == "clear_verified_text":
                unexpected = set(self.action.params) - {
                    "element_id",
                    "target",
                    "role",
                    "label",
                    "states",
                    "expected_effect",
                } - _FORMAL_AUTHORITY_PARAMS
                if unexpected:
                    raise CanonicalActionProtocolError(
                        "清空动作包含协议外参数：" + ", ".join(sorted(unexpected))
                    )
                if (
                    scene.get_element(str(self.action.params["element_id"])).role
                    != "input"
                ):
                    raise CanonicalActionProtocolError(
                        "清空动作必须绑定 input 元素。"
                    )
            if self.action.action == "long_press":
                duration_ms = self.action.params.get("duration_ms", 800)
                if (
                    isinstance(duration_ms, bool)
                    or not isinstance(duration_ms, (int, float))
                    or not 500 <= float(duration_ms) <= 2000
                ):
                    raise CanonicalActionProtocolError(
                        "长按 duration_ms 必须在500～2000之间。"
                    )
            if self.action.action == "drag":
                source_id = str(
                    self.action.params.get("source_element_id") or ""
                ).strip()
                destination_id = str(
                    self.action.params.get("destination_element_id") or ""
                ).strip()
                if not source_id or not destination_id or source_id == destination_id:
                    raise CanonicalActionProtocolError(
                        "拖动必须绑定两个不同的可信元素。"
                    )
                scene.get_element(source_id)
                scene.get_element(destination_id)
            if self.action.action == "swipe":
                direction = str(
                    self.action.params.get("direction") or ""
                ).strip()
                if direction not in {"up", "down", "left", "right"}:
                    raise CanonicalActionProtocolError("滑动动作方向无效。")
            if self.action.action == "reveal_system_navigation":
                unexpected = (
                    set(self.action.params)
                    - {"expected_effect"}
                    - _FORMAL_AUTHORITY_PARAMS
                )
                if unexpected:
                    raise CanonicalActionProtocolError(
                        "系统导航栏唤出动作不能携带坐标、方向、距离或其他参数。"
                    )
                system_ui = getattr(scene, "system_ui", None)
                if system_ui is None:
                    raise CanonicalActionProtocolError(
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
                        navigation_visible = system_ui.get(
                            "navigation_bar_visible"
                        )
                if immersive is not True or navigation_visible is not False:
                    raise CanonicalActionProtocolError(
                        "系统导航栏唤出动作要求当前画面明确处于沉浸态且导航栏隐藏。"
                    )
                if self.action.params.get("expected_effect") != {
                    "system_ui": {"navigation_bar_visible": True}
                }:
                    raise CanonicalActionProtocolError(
                        "系统导航栏唤出动作必须精确声明结构化导航栏可见后置条件。"
                    )
        elif self.action is not None:
            raise CanonicalActionProtocolError(
                "blocked 状态不能携带动作。"
            )
        if self.status == "blocked" and not self.reason.strip():
            raise CanonicalActionProtocolError("阻塞报告必须说明原因。")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["action"] = self.action.to_dict() if self.action else None
        return value


def reject_raw_control_data(value: Any) -> None:
    """Reject model-authored coordinates or multi-action payloads."""

    forbidden = {
        "actions",
        "steps",
        "plan",
        "coordinate",
        "coordinates",
        "tap_point",
        "x",
        "y",
        "shell",
        "command",
    }
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).strip().lower() in forbidden:
                raise CanonicalActionProtocolError(
                    f"单步动作包含禁止字段：{key}"
                )
            reject_raw_control_data(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            reject_raw_control_data(item)
    elif not isinstance(value, (str, int, float, bool, type(None))):
        raise CanonicalActionProtocolError("单步动作参数类型无效。")


def _json_value(value: Any, field_name: str) -> Any:
    try:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise CanonicalActionProtocolError(f"{field_name} 必须是可序列化 JSON 值。") from exc


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stable_id(prefix: str, value: Any) -> str:
    return f"{prefix}_{_digest(value)[:20]}"


def _element_ref(element_id: str) -> str:
    return _stable_id("element", {"source_element_id": str(element_id)})


def _validate_id(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not _ID_PATTERN.fullmatch(value):
        raise CanonicalActionProtocolError(f"{field_name} 无效：{value!r}")


def _required_text(value: Any, field_name: str, *, max_length: int = 300) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CanonicalActionProtocolError(f"{field_name} 必须是非空字符串。")
    text = value.strip()
    if len(text) > max_length:
        raise CanonicalActionProtocolError(f"{field_name} 超过长度限制。")
    return text


def _typed_element_source(element: UIElement) -> dict[str, Any]:
    states = {
        key: _json_value(value, f"element.{element.element_id}.states.{key}")
        for key, value in sorted(element.states.items())
        if key in _SAFE_STATE_KEYS
    }
    return {
        "element_id": element.element_id,
        "role": element.role,
        "meaning": element.meaning,
        "label": element.label,
        "bounds": [float(value) for value in element.bounds],
        "confidence": float(element.confidence),
        "states": states,
    }


def _scene_source(scene: UIScene) -> dict[str, Any]:
    return {
        "foreground_app_id": scene.foreground_app_id,
        "screen_id": scene.screen_id,
        "stable": scene.stable,
        "confidence": float(scene.confidence),
        "overlays": list(scene.overlays),
        "system_ui": scene.system_ui.to_dict(),
        "elements": [
            _typed_element_source(element)
            for element in sorted(scene.elements, key=lambda item: item.element_id)
        ],
    }


def _surface_kind(scene: UIScene) -> str:
    return scene_surface_kind(scene)


@dataclass(frozen=True)
class VisualClaim:
    claim_id: str
    subject_ref: str
    predicate: str
    value: Any
    confidence: float
    source_digest: str
    def validate(self) -> None:
        _validate_id(self.claim_id, "claim.claim_id")
        _validate_id(self.subject_ref, "claim.subject_ref")
        _required_text(self.predicate, "claim.predicate", max_length=100)
        if self.predicate not in CLAIM_PREDICATES and not (
            self.predicate.startswith("element.state.")
            and self.predicate.removeprefix("element.state.") in _SAFE_STATE_KEYS
        ):
            raise CanonicalActionProtocolError(
                f"claim.predicate 无效：{self.predicate}"
            )
        _json_value(self.value, "claim.value")
        if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)):
            raise CanonicalActionProtocolError("claim.confidence 格式无效。")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise CanonicalActionProtocolError("claim.confidence 超出范围。")
        if not re.fullmatch(r"[0-9a-f]{64}", self.source_digest):
            raise CanonicalActionProtocolError("claim.source_digest 必须是 SHA-256。")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "claim_id": self.claim_id,
            "subject_ref": self.subject_ref,
            "predicate": self.predicate,
            "value": _json_value(self.value, "claim.value"),
            "confidence": float(self.confidence),
            "source_digest": self.source_digest,
        }


@dataclass(frozen=True)
class VisualRelation:
    relation_id: str
    subject_ref: str
    relation: str
    object_ref: str
    support_claim_ids: tuple[str, ...]

    def validate(self) -> None:
        _validate_id(self.relation_id, "relation.relation_id")
        _validate_id(self.subject_ref, "relation.subject_ref")
        _validate_id(self.object_ref, "relation.object_ref")
        if self.relation not in RELATION_KINDS:
            raise CanonicalActionProtocolError(f"relation.relation 无效：{self.relation}")
        if not self.support_claim_ids:
            raise CanonicalActionProtocolError("relation.support_claim_ids 不能为空。")
        if len(set(self.support_claim_ids)) != len(self.support_claim_ids):
            raise CanonicalActionProtocolError("relation.support_claim_ids 重复。")
        for value in self.support_claim_ids:
            _validate_id(value, "relation.support_claim_ids")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "relation_id": self.relation_id,
            "subject_ref": self.subject_ref,
            "relation": self.relation,
            "object_ref": self.object_ref,
            "support_claim_ids": list(self.support_claim_ids),
        }


@dataclass(frozen=True)
class Affordance:
    affordance_id: str
    subject_ref: str
    action_kind: str
    support_claim_ids: tuple[str, ...]

    def validate(self) -> None:
        _validate_id(self.affordance_id, "affordance.affordance_id")
        _validate_id(self.subject_ref, "affordance.subject_ref")
        if self.action_kind not in SUPPORTED_ACTIONS:
            raise CanonicalActionProtocolError(
                f"affordance.action_kind 无效：{self.action_kind}"
            )
        if not self.support_claim_ids:
            raise CanonicalActionProtocolError("affordance.support_claim_ids 不能为空。")
        for value in self.support_claim_ids:
            _validate_id(value, "affordance.support_claim_ids")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "affordance_id": self.affordance_id,
            "subject_ref": self.subject_ref,
            "action_kind": self.action_kind,
            "support_claim_ids": list(self.support_claim_ids),
        }


@dataclass(frozen=True)
class StateExpectation:
    subject_ref: str
    predicate: str
    operator: str
    value: Any = None

    def validate(self) -> None:
        _validate_id(self.subject_ref, "expectation.subject_ref")
        _required_text(self.predicate, "expectation.predicate", max_length=100)
        if self.predicate not in EXPECTATION_PREDICATES:
            raise CanonicalActionProtocolError(
                f"expectation.predicate 无效：{self.predicate}"
            )
        if self.operator not in EXPECTATION_OPERATORS:
            raise CanonicalActionProtocolError(
                f"expectation.operator 无效：{self.operator}"
            )
        if self.operator in {"equals", "not_equals"}:
            _json_value(self.value, "expectation.value")
        elif self.value is not None:
            raise CanonicalActionProtocolError(
                f"expectation.{self.operator} 不得携带 value。"
            )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        result = {
            "subject_ref": self.subject_ref,
            "predicate": self.predicate,
            "operator": self.operator,
        }
        if self.operator in {"equals", "not_equals"}:
            result["value"] = _json_value(self.value, "expectation.value")
        return result


@dataclass(frozen=True)
class TypedStateTransition:
    transition_id: str
    precondition_claim_ids: tuple[str, ...]
    expectations: tuple[StateExpectation, ...]
    exploratory: bool = False

    def validate(self) -> None:
        _validate_id(self.transition_id, "transition.transition_id")
        if not self.precondition_claim_ids:
            raise CanonicalActionProtocolError("transition.precondition_claim_ids 不能为空。")
        for value in self.precondition_claim_ids:
            _validate_id(value, "transition.precondition_claim_ids")
        if not self.expectations:
            raise CanonicalActionProtocolError("transition.expectations 不能为空。")
        for expectation in self.expectations:
            expectation.validate()
            if (
                expectation.predicate in {"scene.changed", "observation.changed"}
                and not self.exploratory
            ):
                raise CanonicalActionProtocolError(
                    "scene/observation changed 只能用于 exploratory transition。"
                )
        if not isinstance(self.exploratory, bool):
            raise CanonicalActionProtocolError("transition.exploratory 必须是布尔值。")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "transition_id": self.transition_id,
            "precondition_claim_ids": list(self.precondition_claim_ids),
            "expectations": [item.to_dict() for item in self.expectations],
            "exploratory": self.exploratory,
        }


@dataclass(frozen=True)
class CanonicalActionCandidate:
    candidate_id: str
    action_kind: str
    subject_refs: tuple[str, ...]
    affordance_ids: tuple[str, ...]
    relation_ids: tuple[str, ...]
    transition: TypedStateTransition
    parameters: dict[str, Any] = field(default_factory=dict)
    effect_ref: str = ""

    def validate(self) -> None:
        _validate_id(self.candidate_id, "candidate.candidate_id")
        if self.action_kind not in SUPPORTED_ACTIONS:
            raise CanonicalActionProtocolError(f"candidate.action_kind 无效：{self.action_kind}")
        if not self.subject_refs or not self.affordance_ids:
            raise CanonicalActionProtocolError("candidate 缺少 subject/affordance 绑定。")
        for field_name, values in (
            ("subject_refs", self.subject_refs),
            ("affordance_ids", self.affordance_ids),
            ("relation_ids", self.relation_ids),
        ):
            if len(set(values)) != len(values):
                raise CanonicalActionProtocolError(f"candidate.{field_name} 重复。")
            for value in values:
                _validate_id(value, f"candidate.{field_name}")
        _json_value(self.parameters, "candidate.parameters")
        if any(key in self.parameters for key in {"bounds", "point", "x", "y"}):
            raise CanonicalActionProtocolError("canonical candidate 不得携带坐标。")
        if self.effect_ref:
            _validate_id(self.effect_ref, "candidate.effect_ref")
        self.transition.validate()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "candidate_id": self.candidate_id,
            "action_kind": self.action_kind,
            "subject_refs": list(self.subject_refs),
            "affordance_ids": list(self.affordance_ids),
            "relation_ids": list(self.relation_ids),
            "parameters": _json_value(self.parameters, "candidate.parameters"),
            "effect_ref": self.effect_ref,
            "transition": self.transition.to_dict(),
        }


@dataclass(frozen=True)
class CanonicalActionCatalog:
    task_id: str
    device_id: str
    revision: int
    scene_digest: str
    semantic_digest: str
    claims: tuple[VisualClaim, ...]
    relations: tuple[VisualRelation, ...]
    affordances: tuple[Affordance, ...]
    candidates: tuple[CanonicalActionCandidate, ...]
    status: str
    warnings: tuple[str, ...] = ()
    protocol_version: str = CANONICAL_ACTION_PROTOCOL

    def validate(self) -> None:
        if self.protocol_version != CANONICAL_ACTION_PROTOCOL:
            raise CanonicalActionProtocolError("canonical action protocol_version 无效。")
        _required_text(self.task_id, "report.task_id", max_length=128)
        _required_text(self.device_id, "report.device_id", max_length=128)
        if isinstance(self.revision, bool) or not isinstance(self.revision, int) or self.revision < 1:
            raise CanonicalActionProtocolError("report.revision 必须是正整数。")
        for field_name, value in (
            ("scene_digest", self.scene_digest),
            ("semantic_digest", self.semantic_digest),
        ):
            if not re.fullmatch(r"[0-9a-f]{64}", value):
                raise CanonicalActionProtocolError(f"report.{field_name} 必须是 SHA-256。")
        if self.status not in {"ready", "blocked"}:
            raise CanonicalActionProtocolError("report.status 无效。")
        if self.status == "ready" and not MIN_READY_CANDIDATES <= len(self.candidates) <= MAX_READY_CANDIDATES:
            raise CanonicalActionProtocolError(
                f"ready report 必须包含{MIN_READY_CANDIDATES}至"
                f"{MAX_READY_CANDIDATES}个候选。"
            )
        if self.status == "blocked" and len(self.candidates) >= MIN_READY_CANDIDATES:
            raise CanonicalActionProtocolError("候选已足够时不得标记 blocked。")

        collections = (
            ("claim", self.claims),
            ("relation", self.relations),
            ("affordance", self.affordances),
            ("candidate", self.candidates),
        )
        for field_name, items in collections:
            seen: set[str] = set()
            key = f"{field_name}_id"
            for item in items:
                item.validate()
                item_id = str(getattr(item, key))
                if item_id in seen:
                    raise CanonicalActionProtocolError(f"report.{field_name} ID 重复。")
                seen.add(item_id)

        claim_ids = {item.claim_id for item in self.claims}
        claimed_subjects = {item.subject_ref for item in self.claims}
        relation_ids = {item.relation_id for item in self.relations}
        relation_by_id = {item.relation_id: item for item in self.relations}
        affordance_ids = {item.affordance_id for item in self.affordances}
        affordance_by_id = {item.affordance_id: item for item in self.affordances}
        for relation in self.relations:
            if relation.subject_ref not in claimed_subjects:
                raise CanonicalActionProtocolError("relation.subject_ref 没有事实主体。")
            if not set(relation.support_claim_ids).issubset(claim_ids):
                raise CanonicalActionProtocolError("relation 引用未知 claim。")
        for affordance in self.affordances:
            if affordance.subject_ref not in claimed_subjects:
                raise CanonicalActionProtocolError("affordance.subject_ref 没有事实主体。")
            if not set(affordance.support_claim_ids).issubset(claim_ids):
                raise CanonicalActionProtocolError("affordance 引用未知 claim。")
        for candidate in self.candidates:
            if not set(candidate.subject_refs).issubset(claimed_subjects):
                raise CanonicalActionProtocolError("candidate.subject_refs 没有事实主体。")
            if not set(candidate.relation_ids).issubset(relation_ids):
                raise CanonicalActionProtocolError("candidate 引用未知 relation。")
            if not set(candidate.affordance_ids).issubset(affordance_ids):
                raise CanonicalActionProtocolError("candidate 引用未知 affordance。")
            bound_affordances = [
                affordance_by_id[value] for value in candidate.affordance_ids
            ]
            if any(
                item.action_kind != candidate.action_kind
                or item.subject_ref not in candidate.subject_refs
                for item in bound_affordances
            ):
                raise CanonicalActionProtocolError("candidate 与 affordance 绑定不一致。")
            if not set(candidate.transition.precondition_claim_ids).issubset(claim_ids):
                raise CanonicalActionProtocolError("transition 引用未知 claim。")
            candidate_next_field_subjects = {
                relation_by_id[value].object_ref
                for value in candidate.relation_ids
                if relation_by_id[value].relation == "binds_next_input_field"
            }
            if any(
                item.subject_ref not in claimed_subjects
                and item.subject_ref != candidate.effect_ref
                and item.subject_ref not in candidate_next_field_subjects
                for item in candidate.transition.expectations
            ):
                raise CanonicalActionProtocolError("transition expectation 没有事实主体。")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "protocol_version": self.protocol_version,
            "task_id": self.task_id,
            "device_id": self.device_id,
            "revision": self.revision,
            "scene_digest": self.scene_digest,
            "semantic_digest": self.semantic_digest,
            "status": self.status,
            "warnings": list(self.warnings),
            "claims": [item.to_dict() for item in self.claims],
            "relations": [item.to_dict() for item in self.relations],
            "affordances": [item.to_dict() for item in self.affordances],
            "candidates": [item.to_dict() for item in self.candidates],
        }

    @property
    def report_digest(self) -> str:
        return _digest(self.to_dict())


def _claim(subject_ref: str, predicate: str, value: Any, confidence: float, source: Any) -> VisualClaim:
    payload = {
        "subject_ref": subject_ref,
        "predicate": predicate,
        "value": _json_value(value, "claim.value"),
    }
    return VisualClaim(
        claim_id=_stable_id("claim", payload),
        subject_ref=subject_ref,
        predicate=predicate,
        value=payload["value"],
        confidence=float(confidence),
        source_digest=_digest(source),
    )


def _relation(
    subject_ref: str,
    relation: str,
    object_ref: str,
    support_claim_ids: Iterable[str],
) -> VisualRelation:
    support = tuple(sorted(set(support_claim_ids)))
    payload = {
        "subject_ref": subject_ref,
        "relation": relation,
        "object_ref": object_ref,
        "support_claim_ids": support,
    }
    return VisualRelation(
        relation_id=_stable_id("relation", payload),
        subject_ref=subject_ref,
        relation=relation,
        object_ref=object_ref,
        support_claim_ids=support,
    )


def _affordance(subject_ref: str, action_kind: str, support_claim_ids: Iterable[str]) -> Affordance:
    support = tuple(sorted(set(support_claim_ids)))
    payload = {
        "subject_ref": subject_ref,
        "action_kind": action_kind,
        "support_claim_ids": support,
    }
    return Affordance(
        affordance_id=_stable_id("affordance", payload),
        subject_ref=subject_ref,
        action_kind=action_kind,
        support_claim_ids=support,
    )


def _relation_between(first: UIElement, second: UIElement) -> str:
    al, at, ar, ab = first.bounds
    bl, bt, br, bb = second.bounds
    if al <= bl and at <= bt and ar >= br and ab >= bb:
        return "contains"
    overlap_x = max(0.0, min(ar, br) - max(al, bl))
    overlap_y = max(0.0, min(ab, bb) - max(at, bt))
    if overlap_x > 0 and overlap_y > 0:
        return "overlaps"
    acx, acy = (al + ar) / 2.0, (at + ab) / 2.0
    bcx, bcy = (bl + br) / 2.0, (bt + bb) / 2.0
    if abs(acx - bcx) >= abs(acy - bcy):
        return "left_of" if acx < bcx else "right_of"
    return "above" if acy < bcy else "below"


def _element_eligible(element: UIElement) -> bool:
    return (
        element.role in ELEMENT_ACTION_ROLES
        and float(element.confidence) >= MIN_ELEMENT_CONFIDENCE
        and element.states.get("enabled") is not False
        and element.states.get("visible") is not False
        and element.states.get("fully_visible") is True
    )


def _verified_text_affordance_ready(
    element: UIElement,
    target_text: str,
) -> bool:
    """Expose batch text input only when the next typed segment is executable."""

    if element.role != "input" or element.states.get("focused") is not True:
        return False
    try:
        step = plan_from_input_states(target_text, element.states)
    except (ValueError, VerifiedTextTransactionError):
        return False
    if step is None or step.kind == "literal_key":
        return False
    return bool(
        element.states.get("keyboard_layout") == "qwerty"
        and element.states.get("keyboard_input_mode") == step.required_mode
        and (
            not step.required_case_mode
            or element.states.get("keyboard_case_mode") == step.required_case_mode
        )
        and not element.states.get("ime_preedit_text")
    )


def _element_proves_scrollable_viewport(element: UIElement) -> bool:
    """Grant swipe affordance only from a typed, evidenced viewport fact."""

    base_evidence = bool(
        element.role == "container"
        and float(element.confidence) >= MIN_ELEMENT_CONFIDENCE
        and element.states.get("visible") is not False
        and element.states.get("scrollable") is True
        and element.states.get("scroll_axis") in {"vertical", "horizontal"}
        and any(str(item).strip() for item in element.evidence)
    )
    if not base_evidence:
        return False
    if element.states.get("fully_visible") is True:
        # Preserve the already verified cross-App swipe contract.
        return True
    # For a scrollable viewport, fully_visible=false is the typed content-crop
    # fact; non-empty element evidence above must independently support it.
    return bool(
        element.states.get("fully_visible") is False
        and element.states.get("goal_relevant") is True
    )


def _swipe_directions_for_viewport(element: UIElement) -> tuple[str, ...]:
    """Return only directions supported by the observed viewport axis/edge."""

    axis = element.states.get("scroll_axis")
    if axis == "horizontal":
        backward, forward = "right", "left"
    elif axis == "vertical":
        backward, forward = "down", "up"
    else:
        return ()
    page_index = element.states.get("page_index")
    page_count = element.states.get("page_count")
    if (
        isinstance(page_index, int)
        and not isinstance(page_index, bool)
        and isinstance(page_count, int)
        and not isinstance(page_count, bool)
        and page_count >= 2
        and 0 <= page_index < page_count
    ):
        directions: list[str] = []
        if page_index < page_count - 1:
            directions.append(forward)
        if page_index > 0:
            directions.append(backward)
        return tuple(directions)
    return (forward, backward)


_EXPLICIT_SWIPE_DIRECTION_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "up": (
        re.compile(r"(?:向|往|朝)\s*上(?!方)"),
        re.compile(r"上\s*(?:滑|划|拖|推)"),
        re.compile(r"\b(?:swipe|flick|drag)\s+up\b", re.IGNORECASE),
        re.compile(
            r"\b(?:swipe|flick|drag)\b[^.;!?\n]{0,80}"
            r"\b(?:upward|upwards)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:swipe|flick|drag)\b[^.;!?\n]{0,80}\bup\b"
            r"\s*(?:away|off|out)?\s*(?:[.;!?]|$)",
            re.IGNORECASE,
        ),
        re.compile(r"\b(?:upward|upwards)\s+(?:swipe|flick|drag)\b", re.IGNORECASE),
    ),
    "down": (
        re.compile(r"(?:向|往|朝)\s*下(?!方)"),
        re.compile(r"下\s*(?:滑|划|拖|推)"),
        re.compile(r"\b(?:swipe|flick|drag)\s+down\b", re.IGNORECASE),
        re.compile(
            r"\b(?:swipe|flick|drag)\b[^.;!?\n]{0,80}"
            r"\b(?:downward|downwards)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:swipe|flick|drag)\b[^.;!?\n]{0,80}\bdown\b"
            r"\s*(?:away|off|out)?\s*(?:[.;!?]|$)",
            re.IGNORECASE,
        ),
        re.compile(r"\b(?:downward|downwards)\s+(?:swipe|flick|drag)\b", re.IGNORECASE),
    ),
    "left": (
        re.compile(r"(?:向|往|朝)\s*左(?!方)"),
        re.compile(r"左\s*(?:滑|划|拖|推)"),
        re.compile(r"\b(?:swipe|flick|drag)\s+left\b", re.IGNORECASE),
        re.compile(
            r"\b(?:swipe|flick|drag)\b[^.;!?\n]{0,80}\bleftward\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:swipe|flick|drag)\b[^.;!?\n]{0,80}\bleft\b"
            r"\s*(?:away|off|out)?\s*(?:[.;!?]|$)",
            re.IGNORECASE,
        ),
        re.compile(r"\bleftward\s+(?:swipe|flick|drag)\b", re.IGNORECASE),
    ),
    "right": (
        re.compile(r"(?:向|往|朝)\s*右(?!方)"),
        re.compile(r"右\s*(?:滑|划|拖|推)"),
        re.compile(r"\b(?:swipe|flick|drag)\s+right\b", re.IGNORECASE),
        re.compile(
            r"\b(?:swipe|flick|drag)\b[^.;!?\n]{0,80}\brightward\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:swipe|flick|drag)\b[^.;!?\n]{0,80}\bright\b"
            r"\s*(?:away|off|out)?\s*(?:[.;!?]|$)",
            re.IGNORECASE,
        ),
        re.compile(r"\brightward\s+(?:swipe|flick|drag)\b", re.IGNORECASE),
    ),
}


def _explicit_required_swipe_direction(
    constraints: Iterable[ConstraintIntent],
) -> str | None:
    """Extract one explicit direction from authoritative swipe wording.

    The free-form text may narrow an already typed ``required_action=swipe``;
    it can never mint a swipe by itself. Ambiguous or missing directions fail
    closed instead of becoming a guessed physical action.
    """

    source_text = " ".join(
        item.source_text
        for item in constraints
        if item.kind == "required_action"
        and item.value == "swipe"
        and item.authoritative
        and item.source_text.strip()
    )
    matches = {
        direction
        for direction, patterns in _EXPLICIT_SWIPE_DIRECTION_PATTERNS.items()
        if any(pattern.search(source_text) for pattern in patterns)
    }
    if len(matches) != 1:
        return None
    return next(iter(matches))


def _unique_directional_swipe_presence(scene: UIScene) -> UIElement | None:
    """Prove one object can anchor a typed directional swipe-away gesture.

    A local-action role may provide the identity directly.  A container or
    dialog may contribute its observed bounds only to this non-click gesture;
    the canonical candidate still carries no coordinates, and its direction
    comes exclusively from the authoritative typed task constraint.
    """

    target = scene.unique_trusted_goal_element()
    if target is None:
        completion_evidence = scene.trusted_completion_evidence()
        if len(completion_evidence) != 1:
            return None
        target = completion_evidence[0]
        if any(
            other.element_id != target.element_id
            and other.states.get("goal_relevant") is True
            and float(other.confidence) >= MIN_ELEMENT_CONFIDENCE
            for other in scene.elements
        ):
            return None
    if target.states.get("fully_visible") is not True:
        return None
    if not any(str(item).strip() for item in target.evidence):
        return None
    return target


def scene_matches_target_app_surface(scene: UIScene, target_surface: Any) -> bool:
    """Bind a typed App surface to one structured foreground identity.

    Android package IDs, product IDs, and user-facing App names are not always
    lexical aliases (for example ``com.vendor.runtime`` versus a product
    name).  ``screen_id`` is already a typed fact about the current foreground
    surface, so its bounded identity terms may bridge that gap.  Child labels,
    summaries, and ordinary body text remain ineligible and therefore cannot
    turn a launcher icon into foreground-App proof.
    """

    def identity_terms(*values: Any) -> frozenset[str]:
        generic = {
            "android", "app", "application", "com", "current", "foreground",
            "home", "interface", "list", "main", "page", "screen", "the",
            "view", "应用", "程序", "当前", "主页", "主界面", "列表", "页面",
            "界面", "画面", "屏幕",
        }
        text = " ".join(
            str(value or "").casefold().replace("_", " ").replace("-", " ")
            for value in values
        )
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

    foreground = str(scene.foreground_app_id or "").strip().casefold()
    target_app_id = str(getattr(target_surface, "app_id", "") or "").strip().casefold()
    if not foreground or foreground == "unknown" or not target_app_id:
        return False
    if foreground == target_app_id:
        return True

    foreground_parts = tuple(part for part in foreground.split(".") if part)
    target_parts = tuple(part for part in target_app_id.split(".") if part)
    # A package suffix such as ``app`` is not an App identity.  Leaf matching is
    # only valid when one side is an intentionally unqualified identifier
    # (for example ``settings`` versus ``com.android.settings``).
    if foreground_parts and target_parts and (
        (len(foreground_parts) == 1 and foreground_parts[0] == target_parts[-1])
        or (len(target_parts) == 1 and target_parts[0] == foreground_parts[-1])
    ):
        return True

    app_name = str(getattr(target_surface, "app_name", "") or "").strip().casefold()
    target_terms = identity_terms(target_app_id, app_name)
    screen_terms = identity_terms(scene.screen_id)
    if target_terms and screen_terms.intersection(target_terms):
        return True
    if not app_name:
        return False
    if foreground == app_name:
        return True
    title_matches = tuple(
        element
        for element in scene.elements
        if element.role in {"text", "container"}
        and any(
            marker in str(element.meaning or "").strip().casefold()
            for marker in ("page_title", "title", "heading", "app_header")
        )
        and str(element.label or "").strip().casefold() == app_name
        and float(element.confidence) >= MIN_ELEMENT_CONFIDENCE
        and element.states.get("fully_visible") is True
    )
    return len(title_matches) == 1


def _exact_tap_text_target_eligible(element: UIElement) -> bool:
    """Admit only an exact-authorized visible text target for a semantic tap."""

    return (
        element.role == "text"
        and float(element.confidence) >= MIN_ELEMENT_CONFIDENCE
        and element.states.get("enabled") is not False
        and element.states.get("visible") is not False
        and element.states.get("fully_visible") is True
        and element.states.get("goal_relevant") is True
    )


def _unique_exact_matches(
    elements: tuple[UIElement, ...],
    entity: SemanticEntity,
    *,
    allow_exact_tap_text: bool = False,
) -> tuple[UIElement, ...]:
    if not isinstance(entity.value, str) or not entity.value:
        return ()
    literal = entity.value.casefold()
    return tuple(
        element
        for element in elements
        if (
            _element_eligible(element)
            or (
                allow_exact_tap_text
                and _exact_tap_text_target_eligible(element)
            )
        )
        and literal
        in {
            element.label.strip().casefold(),
            str(element.states.get("value", "")).strip().casefold(),
        }
    )


def _candidate(
    *,
    action_kind: str,
    subject_refs: tuple[str, ...],
    affordance_ids: tuple[str, ...],
    relation_ids: tuple[str, ...],
    precondition_claim_ids: tuple[str, ...],
    expectations: tuple[StateExpectation, ...],
    exploratory: bool = False,
    parameters: Mapping[str, Any] | None = None,
    effect_ref: str = "",
) -> CanonicalActionCandidate:
    transition_payload = {
        "action_kind": action_kind,
        "subjects": subject_refs,
        "preconditions": precondition_claim_ids,
        "expectations": [item.to_dict() for item in expectations],
        "exploratory": exploratory,
    }
    transition = TypedStateTransition(
        transition_id=_stable_id("transition", transition_payload),
        precondition_claim_ids=tuple(sorted(set(precondition_claim_ids))),
        expectations=expectations,
        exploratory=exploratory,
    )
    payload = {
        "action_kind": action_kind,
        "subjects": subject_refs,
        "affordances": affordance_ids,
        "relations": relation_ids,
        "parameters": dict(parameters or {}),
        "effect_ref": effect_ref,
        "transition": transition.to_dict(),
    }
    return CanonicalActionCandidate(
        candidate_id=_stable_id("candidate", payload),
        action_kind=action_kind,
        subject_refs=subject_refs,
        affordance_ids=affordance_ids,
        relation_ids=relation_ids,
        transition=transition,
        parameters=dict(parameters or {}),
        effect_ref=effect_ref,
    )


def compile_canonical_action_catalog(
    scene: UIScene,
    semantic_ir: TaskSemanticIR,
    available_action_kinds: Iterable[str],
) -> CanonicalActionCatalog:
    """Compile the sole deterministic action catalog for the active subgoal."""

    scene.validate()
    semantic_ir.validate()
    active_subgoals = tuple(
        item for item in semantic_ir.subgoals if item.status == "active"
    )
    if len(active_subgoals) != 1:
        raise CanonicalActionProtocolError(
            "canonical action catalog 要求且只允许一个 active subgoal。"
        )
    active_subgoal = active_subgoals[0]
    constraints_by_id = {
        item.constraint_id: item for item in semantic_ir.constraints
    }
    active_required_action_constraints = tuple(
        constraints_by_id[ref]
        for ref in active_subgoal.constraint_refs
        if ref in constraints_by_id
        and constraints_by_id[ref].kind == "required_action"
    )
    active_required_actions = frozenset(
        str(item.value) for item in active_required_action_constraints
    )
    active_input_fields = tuple(
        item
        for item in semantic_ir.input_fields
        if active_subgoal.subgoal_id in item.source_subgoal_ids
    )
    active_input_payload_refs = frozenset(
        item.payload_ref for item in active_input_fields
    )
    predecessor_input_field_ids = {
        item.field_id
        for item in semantic_ir.input_fields
        if set(item.source_subgoal_ids).intersection(active_subgoal.depends_on)
    }
    direct_successor_ids = {
        item.subgoal_id
        for item in semantic_ir.subgoals
        if active_subgoal.subgoal_id in item.depends_on
    }
    successor_input_field_ids = {
        item.field_id
        for item in semantic_ir.input_fields
        if set(item.source_subgoal_ids).intersection(direct_successor_ids)
    }
    if predecessor_input_field_ids and successor_input_field_ids:
        boundary_input_field_ids = (
            predecessor_input_field_ids & successor_input_field_ids
        )
    else:
        boundary_input_field_ids = (
            predecessor_input_field_ids | successor_input_field_ids
        )
    enter_input_field_ids = frozenset(
        {item.field_id for item in active_input_fields}
        | boundary_input_field_ids
    )
    active_effect_refs = frozenset(active_subgoal.effect_refs)
    effects_by_id = {item.effect_id: item for item in semantic_ir.effects}
    active_entity_refs = set(active_subgoal.entity_refs)
    for effect_ref in active_effect_refs:
        effect = effects_by_id.get(effect_ref)
        if effect is not None:
            active_entity_refs.update(effect.target_refs)
            active_entity_refs.update(effect.payload_refs)
    active_entity_refs.update(active_input_payload_refs)
    desired_by_id = {
        item.state_id: item for item in semantic_ir.desired_states
    }
    active_desired_states = tuple(
        desired_by_id[ref]
        for ref in active_subgoal.desired_state_refs
        if ref in desired_by_id
    )
    active_entity_refs.update(
        item.subject_ref
        for item in active_desired_states
        if item.subject_ref in {entity.entity_id for entity in semantic_ir.entities}
    )
    active_text = " ".join(
        str(item.value or "") for item in active_desired_states
    ).casefold()
    active_targets_input = bool(
        active_input_fields
        or "clear_verified_text" in active_required_actions
        or any(
            token in active_text
            for token in ("input", "text field", "输入框", "文本框", "编辑框")
        )
    )
    available = frozenset(str(value) for value in available_action_kinds)
    unknown = available - SUPPORTED_ACTIONS
    if unknown:
        raise CanonicalActionProtocolError(
            "available_action_kinds 含未知动作：" + ", ".join(sorted(unknown))
        )

    surface_ref = "surface_current"
    source = _scene_source(scene)
    scene_digest = _digest(source)
    claims: list[VisualClaim] = [
        _claim(surface_ref, "surface.kind", _surface_kind(scene), scene.confidence, source),
        _claim(
            surface_ref,
            "surface.foreground_app_id",
            scene.foreground_app_id,
            scene.confidence,
            source,
        ),
        _claim(surface_ref, "surface.screen_id", scene.screen_id, scene.confidence, source),
        _claim(surface_ref, "surface.stable", scene.stable, scene.confidence, source),
        _claim(
            surface_ref,
            "surface.overlay_present",
            bool(scene.overlays),
            scene.confidence,
            source,
        ),
    ]
    element_claim_ids: dict[str, list[str]] = {}
    element_claim_by_predicate: dict[tuple[str, str], str] = {}
    sorted_elements = tuple(sorted(scene.elements, key=lambda item: item.element_id))
    for element in sorted_elements:
        element_ref = _element_ref(element.element_id)
        typed_source = _typed_element_source(element)
        element_claims = [
            _claim(element_ref, "element.exists", True, element.confidence, typed_source),
            _claim(element_ref, "element.role", element.role, element.confidence, typed_source),
            _claim(element_ref, "element.meaning", element.meaning, element.confidence, typed_source),
        ]
        if element.label:
            element_claims.append(
                _claim(element_ref, "element.label", element.label, element.confidence, typed_source)
            )
        for key, value in sorted(element.states.items()):
            if key in _SAFE_STATE_KEYS:
                element_claims.append(
                    _claim(
                        element_ref,
                        f"element.state.{key}",
                        value,
                        element.confidence,
                        typed_source,
                    )
                )
        claims.extend(element_claims)
        element_claim_ids[element.element_id] = [item.claim_id for item in element_claims]
        for item in element_claims:
            element_claim_by_predicate[(element.element_id, item.predicate)] = item.claim_id

    claims = sorted(claims, key=lambda item: item.claim_id)
    surface_claim_ids = tuple(
        item.claim_id for item in claims if item.subject_ref == surface_ref
    )
    relations: list[VisualRelation] = []
    for element in sorted_elements:
        relations.append(
            _relation(
                _element_ref(element.element_id),
                "on_surface",
                surface_ref,
                element_claim_ids[element.element_id],
            )
        )
    for index, first in enumerate(sorted_elements):
        for second in sorted_elements[index + 1 :]:
            relations.append(
                _relation(
                    _element_ref(first.element_id),
                    _relation_between(first, second),
                    _element_ref(second.element_id),
                    (
                        element_claim_by_predicate[(first.element_id, "element.exists")],
                        element_claim_by_predicate[(second.element_id, "element.exists")],
                    ),
                )
            )

    entity_by_id = {item.entity_id: item for item in semantic_ir.entities}
    active_input_payload_entities = tuple(
        entity_by_id[ref]
        for ref in sorted(active_input_payload_refs)
        if ref in entity_by_id
        and entity_by_id[ref].role == "input_text"
        and isinstance(entity_by_id[ref].value, str)
    )
    effect_by_entity: dict[str, list[tuple[EffectIntent, str]]] = {}
    for effect in semantic_ir.effects:
        for entity_ref in effect.target_refs:
            effect_by_entity.setdefault(entity_ref, []).append((effect, "binds_effect_target"))
        for entity_ref in effect.payload_refs:
            effect_by_entity.setdefault(entity_ref, []).append((effect, "binds_effect_payload"))

    exact_elements_by_entity: dict[str, tuple[UIElement, ...]] = {}
    relation_ids_by_element: dict[str, list[str]] = {}
    element_id_by_ref = {
        _element_ref(element.element_id): element.element_id
        for element in sorted_elements
    }
    for relation in relations:
        if relation.relation == "on_surface":
            element_id = element_id_by_ref.get(relation.subject_ref)
            if element_id:
                relation_ids_by_element.setdefault(element_id, []).append(
                    relation.relation_id
                )
    relation_effects_by_element: dict[str, list[tuple[str, str, str]]] = {}
    focused_inputs = tuple(
        element
        for element in sorted_elements
        if _element_eligible(element)
        and element.role == "input"
        and element.states.get("focused") is True
    )
    input_payload_entities = tuple(
        entity
        for entity in semantic_ir.entities
        if entity.role == "input_text" and isinstance(entity.value, str)
    )
    direct_payload_bindings: set[tuple[str, str, str]] = set()
    if len(focused_inputs) == 1 and len(input_payload_entities) == 1:
        element = focused_inputs[0]
        entity = input_payload_entities[0]
        support = tuple(
            claim_id
            for predicate in ("element.role", "element.state.focused")
            if (claim_id := element_claim_by_predicate.get((element.element_id, predicate)))
        )
        for effect, relation_kind in effect_by_entity.get(entity.entity_id, ()):
            if relation_kind != "binds_effect_payload":
                continue
            binding = _relation(
                _element_ref(element.element_id),
                relation_kind,
                effect.effect_id,
                support,
            )
            relations.append(binding)
            relation_ids_by_element.setdefault(element.element_id, []).append(
                binding.relation_id
            )
            relation_effects_by_element.setdefault(element.element_id, []).append(
                (effect.effect_id, entity.entity_id, relation_kind)
            )
            direct_payload_bindings.add(
                (element.element_id, effect.effect_id, entity.entity_id)
            )
    if len(active_input_fields) == 1 and len(predecessor_input_field_ids) == 1:
        active_field = active_input_fields[0]
        predecessor_field_id = next(iter(predecessor_input_field_ids))
        predecessor_field = next(
            item for item in semantic_ir.input_fields
            if item.field_id == predecessor_field_id
        )
        predecessor_payload = entity_by_id.get(predecessor_field.payload_ref)
        focused_predecessors = [
            item for item in focused_inputs
            if item.states.get("input_field_id") == predecessor_field_id
            and item.states.get("input_field_label") == predecessor_field.field_label
            and predecessor_payload is not None
            and predecessor_payload.role == "input_text"
            and item.states.get("value") == predecessor_payload.value
        ]
        for element in sorted_elements:
            if not (
                len(focused_predecessors) == 1
                and _element_eligible(element)
                and element.meaning == "input_next_field_key"
                and element.role == "button"
                and element.states.get("input_next_field_key") is True
                and element.states.get("key_action") == "next"
                and element.states.get("source_input_field_id") == predecessor_field_id
                and element.states.get("target_input_field_id") == active_field.field_id
                and element.states.get("target_input_field_label") == active_field.field_label
            ):
                continue
            support = tuple(element_claim_ids[element.element_id])
            binding = _relation(
                _element_ref(element.element_id),
                "binds_next_input_field",
                active_field.field_id,
                support,
            )
            relations.append(binding)
            relation_ids_by_element.setdefault(element.element_id, []).append(
                binding.relation_id
            )
    exact_tap_authority = bool(
        active_subgoal.subgoal_id == "exact_tap_semantic"
        and active_required_actions == {"tap_semantic"}
    )
    exact_tap_text_element_ids: set[str] = set()
    for entity in semantic_ir.entities:
        allow_exact_tap_text = bool(
            exact_tap_authority
            and entity.role == "target_ui_label"
        )
        matches = _unique_exact_matches(
            sorted_elements,
            entity,
            allow_exact_tap_text=allow_exact_tap_text,
        )
        exact_elements_by_entity[entity.entity_id] = matches
        if (
            allow_exact_tap_text
            and len(matches) == 1
            and matches[0].role == "text"
        ):
            exact_tap_text_element_ids.add(matches[0].element_id)
        for element in matches:
            literal_claims = tuple(
                claim_id
                for predicate in ("element.label", "element.state.value")
                if (claim_id := element_claim_by_predicate.get((element.element_id, predicate)))
            )
            exact_relation = _relation(
                _element_ref(element.element_id),
                "exact_literal_match",
                entity.entity_id,
                literal_claims,
            )
            relations.append(exact_relation)
            relation_ids_by_element.setdefault(element.element_id, []).append(
                exact_relation.relation_id
            )
            for effect, relation_kind in effect_by_entity.get(entity.entity_id, ()):
                binding = _relation(
                    _element_ref(element.element_id),
                    relation_kind,
                    effect.effect_id,
                    literal_claims,
                )
                relations.append(binding)
                relation_ids_by_element.setdefault(element.element_id, []).append(
                    binding.relation_id
                )
                relation_effects_by_element.setdefault(element.element_id, []).append(
                    (effect.effect_id, entity.entity_id, relation_kind)
                )

    for surface in semantic_ir.surfaces:
        if surface.kind != "app" or not surface.app_name:
            continue
        matches = tuple(
            element
            for element in sorted_elements
            if _element_eligible(element)
            and element.label.strip().casefold() == surface.app_name.casefold()
        )
        for element in matches:
            label_claim = element_claim_by_predicate.get((element.element_id, "element.label"))
            if not label_claim:
                continue
            binding = _relation(
                _element_ref(element.element_id),
                "binds_surface",
                surface.surface_id,
                (label_claim,),
            )
            relations.append(binding)
            relation_ids_by_element.setdefault(element.element_id, []).append(
                binding.relation_id
            )

    relation_by_id = {item.relation_id: item for item in relations}
    relations = sorted(relation_by_id.values(), key=lambda item: item.relation_id)

    scrollable_viewports = tuple(
        element
        for element in sorted_elements
        if _element_proves_scrollable_viewport(element)
    )
    target_swipe_presence: UIElement | None = None
    explicit_target_swipe_direction = _explicit_required_swipe_direction(
        active_required_action_constraints
    )
    if (
        not scrollable_viewports
        and active_required_actions == {"swipe"}
        and explicit_target_swipe_direction is not None
    ):
        target_swipe_presence = _unique_directional_swipe_presence(scene)
    if len(scrollable_viewports) == 1:
        swipe_directions = _swipe_directions_for_viewport(scrollable_viewports[0])
    elif target_swipe_presence is not None:
        swipe_directions = (explicit_target_swipe_direction,)
    else:
        swipe_directions = ()
    target_swipe_claim_ids = (
        tuple(element_claim_ids[target_swipe_presence.element_id])
        if target_swipe_presence is not None
        else ()
    )
    affordances: list[Affordance] = []
    for action_kind in sorted(available):
        if action_kind in {
            "back",
            "home",
            "open_recent_apps",
            "reveal_system_navigation",
            "swipe",
            "wait_for_change",
        }:
            if action_kind == "swipe" and not swipe_directions:
                continue
            if action_kind == "reveal_system_navigation" and not (
                scene.system_ui.immersive_or_fullscreen is True
                and scene.system_ui.navigation_bar_visible is False
            ):
                continue
            support_claim_ids = (
                tuple(surface_claim_ids) + target_swipe_claim_ids
                if action_kind == "swipe" and target_swipe_presence is not None
                else surface_claim_ids
            )
            affordances.append(
                _affordance(surface_ref, action_kind, support_claim_ids)
            )
    for element in sorted_elements:
        normally_actionable = _element_eligible(element)
        exact_tap_text_target = element.element_id in exact_tap_text_element_ids
        if not normally_actionable and not exact_tap_text_target:
            continue
        focus_only_input_surface = bool(
            element.role == "input"
            and element.states.get("focus_only_input_surface") is True
        )
        supported: set[str] = set()
        if "tap_semantic" in available:
            supported.add("tap_semantic")
        if (
            normally_actionable
            and not focus_only_input_surface
            and "double_tap" in available
        ):
            supported.add("double_tap")
        if (
            normally_actionable
            and not focus_only_input_surface
            and "long_press" in available
        ):
            supported.add("long_press")
        if normally_actionable and not focus_only_input_surface and "drag" in available:
            supported.add("drag")
        if element.role == "input" and element.states.get("focused") is True:
            active_field = active_input_fields[0] if len(active_input_fields) == 1 else None
            field_identity_matches = bool(
                active_field is not None
                and (
                    (
                        element.states.get("input_field_id")
                        == active_field.field_id
                        and (
                            not active_field.field_label
                            or element.states.get("input_field_label")
                            == active_field.field_label
                        )
                    )
                    or (
                        len(semantic_ir.input_fields) == 1
                        and not active_field.field_label
                    )
                )
            )
            if (
                "input_verified_text" in available
                and len(active_input_payload_entities) == 1
                and field_identity_matches
                and _verified_text_affordance_ready(
                    element,
                    active_input_payload_entities[0].value,
                )
            ):
                supported.add("input_verified_text")
            if (
                "clear_verified_text" in available
                and (
                    bool(element.states.get("value"))
                    or bool(element.states.get("ime_preedit_text"))
                )
            ):
                supported.add("clear_verified_text")
        if (
            "press_enter" in available
            and element.meaning == "input_exact_enter_key"
            and element.states.get("input_enter_key") is True
            and element.states.get("key_action") == "newline"
        ):
            supported.add("press_enter")
        if "dismiss_overlay" in available and scene.overlays and element.role in {"button", "icon"}:
            supported.add("dismiss_overlay")
        for action_kind in sorted(supported):
            affordances.append(
                _affordance(
                    _element_ref(element.element_id),
                    action_kind,
                    element_claim_ids[element.element_id],
                )
            )
    affordance_by_pair = {
        (item.subject_ref, item.action_kind): item for item in affordances
    }
    affordances = sorted(affordances, key=lambda item: item.affordance_id)

    candidates: list[CanonicalActionCandidate] = []
    active_external_effect_refs = tuple(
        active_subgoal.effect_refs
        if active_subgoal.external_impact == "external_state"
        else ()
    )
    unique_effect_control_by_ref: dict[str, str] = {}
    for effect_ref in active_external_effect_refs:
        effect = effects_by_id.get(effect_ref)
        if effect is None:
            continue
        matches = [
            element.element_id
            for element in sorted_elements
            if _element_eligible(element)
            and _element_realizes_effect(element, effect.kind)
        ]
        if len(matches) == 1:
            unique_effect_control_by_ref[effect_ref] = matches[0]
    # Element candidates require a unique exact entity/surface binding. A model
    # boolean such as goal_relevant never grants eligibility here.
    for element in sorted_elements:
        element_ref = _element_ref(element.element_id)
        relation_ids = tuple(sorted(set(relation_ids_by_element.get(element.element_id, ()))))
        if not relation_ids:
            continue
        unique_relation_ids: list[str] = []
        bound_entity_ids: list[str] = []
        for relation_id in relation_ids:
            relation = relation_by_id[relation_id]
            if relation.relation == "exact_literal_match":
                matches = exact_elements_by_entity.get(relation.object_ref, ())
                if len(matches) != 1:
                    continue
                bound_entity_ids.append(relation.object_ref)
                unique_relation_ids.append(relation_id)
            elif relation.relation == "binds_surface":
                same_surface_matches = [
                    item
                    for item in relations
                    if item.relation == "binds_surface"
                    and item.object_ref == relation.object_ref
                ]
                if len(same_surface_matches) != 1:
                    continue
                unique_relation_ids.append(relation_id)
            elif relation.relation in {"binds_effect_target", "binds_effect_payload"}:
                # These are retained only when their exact entity binding is unique.
                related_entities = [
                    entity_id
                    for effect_id, entity_id, kind in relation_effects_by_element.get(
                        element.element_id, ()
                    )
                    if effect_id == relation.object_ref and kind == relation.relation
                ]
                if any(
                    len(exact_elements_by_entity.get(entity_id, ())) == 1
                    or (
                        relation.relation == "binds_effect_payload"
                        and (element.element_id, relation.object_ref, entity_id)
                        in direct_payload_bindings
                    )
                    for entity_id in related_entities
                ):
                    unique_relation_ids.append(relation_id)
            elif relation.relation == "binds_next_input_field":
                same_bindings = [
                    item for item in relations
                    if item.relation == "binds_next_input_field"
                    and item.object_ref == relation.object_ref
                ]
                if len(same_bindings) == 1:
                    unique_relation_ids.append(relation_id)
            elif relation.relation == "on_surface":
                unique_relation_ids.append(relation_id)
        if not unique_relation_ids:
            continue

        tap_affordance = affordance_by_pair.get((element_ref, "tap_semantic"))
        if element.role == "input" and element.states.get("focused") is True:
            # The typed focus postcondition is already satisfied.  Keeping a
            # tap candidate here lets Qwen spend a physical action on a no-op
            # instead of choosing the bound input transaction.
            tap_affordance = None
        if tap_affordance is not None:
            extra_expectations: tuple[StateExpectation, ...] = ()
            surface_binding = next(
                (
                    relation_by_id[value]
                    for value in unique_relation_ids
                    if relation_by_id[value].relation == "binds_surface"
                ),
                None,
            )
            if surface_binding is not None:
                expectation = StateExpectation(
                    surface_ref,
                    "surface.active_ref",
                    "equals",
                    surface_binding.object_ref,
                )
            elif element.meaning == "input_next_field_key":
                target_field_id = str(
                    element.states.get("target_input_field_id") or ""
                ).strip()
                if not any(
                    relation_by_id[value].relation == "binds_next_input_field"
                    and relation_by_id[value].object_ref == target_field_id
                    for value in unique_relation_ids
                ):
                    continue
                expectation = StateExpectation(
                    target_field_id,
                    "input_field.focused",
                    "equals",
                    True,
                )
            elif element.meaning in {
                "ime_exact_candidate",
                "input_exact_literal_key",
                "input_exact_enter_key",
                "switch_keyboard_layout",
                "switch_keyboard_case",
                "switch_keyboard_input_mode",
            }:
                expected_input_value = (
                    element.states.get("expected_input_value")
                    if element.meaning
                    in {"ime_exact_candidate", "input_exact_literal_key"}
                    or element.meaning == "input_exact_enter_key"
                    else element.states.get("prior_input_value")
                )
                if not isinstance(expected_input_value, str):
                    continue
                expected_element_id = str(
                    element.states.get("input_element_id") or element.element_id
                ).strip()
                expected_element = next(
                    (
                        item
                        for item in sorted_elements
                        if item.element_id == expected_element_id
                    ),
                    None,
                )
                expected_subject_ref = (
                    _element_ref(expected_element.element_id)
                    if expected_element is not None
                    else element_ref
                )
                expectation = StateExpectation(
                    expected_subject_ref,
                    "element.state.value",
                    "equals",
                    expected_input_value,
                )
                switch_predicate = {
                    "switch_keyboard_layout": "element.state.keyboard_layout",
                    "switch_keyboard_case": "element.state.keyboard_case_mode",
                    "switch_keyboard_input_mode": "element.state.keyboard_input_mode",
                }.get(element.meaning)
                switch_value = (
                    element.states.get("target_layout")
                    if element.meaning == "switch_keyboard_layout"
                    else element.states.get("target_mode")
                )
                if switch_predicate and isinstance(switch_value, str) and switch_value:
                    extra_expectations = (
                        StateExpectation(
                            expected_subject_ref,
                            switch_predicate,
                            "equals",
                            switch_value,
                        ),
                    )
            elif element.role == "input":
                expectation = StateExpectation(
                    element_ref,
                    "element.state.focused",
                    "equals",
                    True,
                )
            else:
                entity_ref = sorted(set(bound_entity_ids))[0] if bound_entity_ids else ""
                expectation = (
                    StateExpectation(
                        surface_ref,
                        "surface.focused_entity_ref",
                        "equals",
                        entity_ref,
                    )
                    if entity_ref
                    else StateExpectation(
                        surface_ref,
                        "surface.navigation_depth",
                        "changed",
                    )
                )
            effect_ref = next(
                (
                    active_effect_ref
                    for active_effect_ref, control_element_id in
                    unique_effect_control_by_ref.items()
                    if control_element_id == element.element_id
                ),
                "",
            )
            if effect_ref:
                expectation = StateExpectation(
                    effect_ref,
                    "effect.applied",
                    "equals",
                    True,
                )
            candidates.append(
                _candidate(
                    action_kind="tap_semantic",
                    subject_refs=(element_ref,),
                    affordance_ids=(tap_affordance.affordance_id,),
                    relation_ids=tuple(sorted(set(unique_relation_ids))),
                    precondition_claim_ids=tuple(element_claim_ids[element.element_id]),
                    expectations=(expectation, *extra_expectations),
                    effect_ref=effect_ref,
                    parameters={"element_id": element.element_id},
                    exploratory=(
                        not effect_ref
                        and surface_binding is None
                        and element.role != "input"
                        and element.meaning
                        not in {
                            "ime_exact_candidate",
                            "input_exact_literal_key",
                            "input_exact_enter_key",
                            "switch_keyboard_layout",
                            "switch_keyboard_case",
                            "switch_keyboard_input_mode",
                            "input_next_field_key",
                        }
                    ),
                )
            )

        enter_affordance = affordance_by_pair.get((element_ref, "press_enter"))
        if enter_affordance is not None:
            expected_value = element.states.get("expected_input_value")
            input_element_id = str(
                element.states.get("input_element_id") or ""
            ).strip()
            input_element = next(
                (
                    item
                    for item in sorted_elements
                    if item.element_id == input_element_id and item.role == "input"
                ),
                None,
            )
            if isinstance(expected_value, str) and input_element is not None:
                candidates.append(
                    _candidate(
                        action_kind="press_enter",
                        subject_refs=(element_ref,),
                        affordance_ids=(enter_affordance.affordance_id,),
                        relation_ids=tuple(sorted(set(unique_relation_ids))),
                        precondition_claim_ids=tuple(
                            element_claim_ids[element.element_id]
                        ),
                        expectations=(
                            StateExpectation(
                                _element_ref(input_element.element_id),
                                "element.state.value",
                                "equals",
                                expected_value,
                            ),
                        ),
                        parameters={"element_id": element.element_id},
                    )
                )

        input_affordance = affordance_by_pair.get((element_ref, "input_verified_text"))
        if input_affordance is not None:
            payload_entities = [
                entity_by_id[entity_id]
                for effect_id, entity_id, relation_kind in relation_effects_by_element.get(
                    element.element_id, ()
                )
                if relation_kind == "binds_effect_payload"
                and entity_id in active_input_payload_refs
                and entity_id in entity_by_id
                and entity_by_id[entity_id].role == "input_text"
            ]
            # A focused empty input does not literally contain the future payload.
            # Bind the unique typed input_text payload directly to the input affordance.
            if not payload_entities:
                payload_entities = list(active_input_payload_entities)
            if len(payload_entities) == 1:
                payload = payload_entities[0]
                try:
                    deterministic_input_step = plan_from_input_states(
                        payload.value,
                        element.states,
                    )
                except (ValueError, VerifiedTextTransactionError):
                    deterministic_input_step = None
                if deterministic_input_step is None:
                    continue
                expected_input_states = (
                    {
                        "value": deterministic_input_step.current_text,
                        "ime_preedit_text": deterministic_input_step.pinyin,
                        "ime_exact_candidate_text": deterministic_input_step.segment,
                    }
                    if deterministic_input_step.kind == "chinese_pinyin"
                    else {"value": deterministic_input_step.expected_value}
                )
                candidates.append(
                    _candidate(
                        action_kind="input_verified_text",
                        subject_refs=(element_ref,),
                        affordance_ids=(input_affordance.affordance_id,),
                        relation_ids=tuple(sorted(set(unique_relation_ids))),
                        precondition_claim_ids=tuple(element_claim_ids[element.element_id]),
                        expectations=tuple(
                            StateExpectation(
                                element_ref,
                                f"element.state.{state_name}",
                                "equals",
                                state_value,
                            )
                            for state_name, state_value in expected_input_states.items()
                        ),
                        parameters={"element_id": element.element_id},
                    )
                )

        clear_affordance = affordance_by_pair.get((element_ref, "clear_verified_text"))
        if clear_affordance is not None:
            clear_expectations = [
                StateExpectation(
                    element_ref,
                    "element.state.value",
                    "equals",
                    "",
                )
            ]
            if element.states.get("ime_preedit_text"):
                clear_expectations.append(
                    StateExpectation(
                        element_ref,
                        "element.state.ime_preedit_text",
                        "absent",
                    )
                )
            candidates.append(
                _candidate(
                    action_kind="clear_verified_text",
                    subject_refs=(element_ref,),
                    affordance_ids=(clear_affordance.affordance_id,),
                    relation_ids=tuple(sorted(set(unique_relation_ids))),
                    precondition_claim_ids=tuple(element_claim_ids[element.element_id]),
                    expectations=tuple(clear_expectations),
                    parameters={"element_id": element.element_id},
                )
            )

        dismiss_affordance = affordance_by_pair.get((element_ref, "dismiss_overlay"))
        if dismiss_affordance is not None:
            candidates.append(
                _candidate(
                    action_kind="dismiss_overlay",
                    subject_refs=(element_ref,),
                    affordance_ids=(dismiss_affordance.affordance_id,),
                    relation_ids=tuple(sorted(set(unique_relation_ids))),
                    precondition_claim_ids=tuple(element_claim_ids[element.element_id]),
                    expectations=(
                        StateExpectation(
                            surface_ref,
                            "surface.overlay_present",
                            "equals",
                            False,
                        ),
                    ),
                    parameters={"element_id": element.element_id},
                )
            )

        long_press_affordance = affordance_by_pair.get((element_ref, "long_press"))
        if long_press_affordance is not None:
            candidates.append(
                _candidate(
                    action_kind="long_press",
                    subject_refs=(element_ref,),
                    affordance_ids=(long_press_affordance.affordance_id,),
                    relation_ids=tuple(sorted(set(unique_relation_ids))),
                    precondition_claim_ids=tuple(element_claim_ids[element.element_id]),
                    expectations=(
                        StateExpectation(
                            element_ref,
                            "element.state.interaction_result",
                            "changed",
                        ),
                    ),
                    exploratory=True,
                    parameters={"element_id": element.element_id},
                )
            )

        double_tap_affordance = affordance_by_pair.get((element_ref, "double_tap"))
        if double_tap_affordance is not None:
            candidates.append(
                _candidate(
                    action_kind="double_tap",
                    subject_refs=(element_ref,),
                    affordance_ids=(double_tap_affordance.affordance_id,),
                    relation_ids=tuple(sorted(set(unique_relation_ids))),
                    precondition_claim_ids=tuple(element_claim_ids[element.element_id]),
                    expectations=(
                        StateExpectation(
                            element_ref,
                            "element.state.interaction_result",
                            "changed",
                        ),
                    ),
                    exploratory=True,
                    parameters={"element_id": element.element_id},
                )
            )

    source_roles = {"drag_source", "source", "item"}
    destination_roles = {"drag_destination", "destination", "target"}
    unique_entity_elements = {
        entity_id: matches[0]
        for entity_id, matches in exact_elements_by_entity.items()
        if len(matches) == 1
    }
    source_bindings = [
        (entity, unique_entity_elements[entity.entity_id])
        for entity in semantic_ir.entities
        if entity.role in source_roles and entity.entity_id in unique_entity_elements
    ]
    destination_bindings = [
        (entity, unique_entity_elements[entity.entity_id])
        for entity in semantic_ir.entities
        if entity.role in destination_roles and entity.entity_id in unique_entity_elements
    ]
    if len(source_bindings) == 1 and len(destination_bindings) == 1:
        source_entity, source_element = source_bindings[0]
        destination_entity, destination_element = destination_bindings[0]
        if source_element.element_id != destination_element.element_id:
            source_ref = _element_ref(source_element.element_id)
            destination_ref = _element_ref(destination_element.element_id)
            source_affordance = affordance_by_pair.get((source_ref, "drag"))
            destination_affordance = affordance_by_pair.get((destination_ref, "drag"))
            if source_affordance is not None and destination_affordance is not None:
                relation_ids = tuple(
                    sorted(
                        set(relation_ids_by_element.get(source_element.element_id, ()))
                        | set(
                            relation_ids_by_element.get(
                                destination_element.element_id, ()
                            )
                        )
                    )
                )
                candidates.append(
                    _candidate(
                        action_kind="drag",
                        subject_refs=(source_ref, destination_ref),
                        affordance_ids=(
                            source_affordance.affordance_id,
                            destination_affordance.affordance_id,
                        ),
                        relation_ids=relation_ids,
                        precondition_claim_ids=tuple(
                            element_claim_ids[source_element.element_id]
                            + element_claim_ids[destination_element.element_id]
                        ),
                        expectations=(
                            StateExpectation(
                                source_ref,
                                "element.state.location_relation",
                                "equals",
                                destination_ref,
                            ),
                        ),
                        parameters={
                            "source_element_id": source_element.element_id,
                            "destination_element_id": destination_element.element_id,
                        },
                    )
                )

    surface_affordance = {
        item.action_kind: item
        for item in affordances
        if item.subject_ref == surface_ref
    }
    target_swipe_ref = (
        _element_ref(target_swipe_presence.element_id)
        if target_swipe_presence is not None
        else ""
    )
    swipe_specs = tuple(
        (
            "swipe",
            (
                StateExpectation(
                    target_swipe_ref,
                    "element.exists",
                    "absent",
                ),
            )
            if target_swipe_presence is not None
            else (
                StateExpectation(
                    surface_ref,
                    "surface.viewport",
                    "changed",
                ),
            ),
            target_swipe_presence is None,
            {
                "direction": direction,
                **(
                    {"element_id": target_swipe_presence.element_id}
                    if target_swipe_presence is not None
                    else {}
                ),
            },
        )
        for direction in swipe_directions
    )
    system_specs: tuple[tuple[str, tuple[StateExpectation, ...], bool, dict[str, Any]], ...] = (
        (
            "open_recent_apps",
            (
                StateExpectation(
                    surface_ref,
                    "surface.kind",
                    "equals",
                    expected_idempotent_system_surface_kind(
                        "open_recent_apps"
                    ),
                ),
            ),
            False,
            {},
        ),
        (
            "home",
            (
                StateExpectation(
                    surface_ref,
                    "surface.kind",
                    "equals",
                    expected_idempotent_system_surface_kind("home"),
                ),
            ),
            False,
            {},
        ),
        (
            "reveal_system_navigation",
            (
                StateExpectation(
                    surface_ref,
                    "system_ui.navigation_bar_visible",
                    "equals",
                    True,
                ),
            ),
            False,
            {},
        ),
        (
            "back",
            (StateExpectation(surface_ref, "surface.navigation_depth", "changed"),),
            True,
            {},
        ),
        *swipe_specs,
        (
            "wait_for_change",
            (StateExpectation(surface_ref, "observation.changed", "changed"),),
            True,
            {},
        ),
    )
    for action_kind, expectations, exploratory, parameters in system_specs:
        affordance = surface_affordance.get(action_kind)
        if affordance is None:
            continue
        idempotent_surface_kind = expected_idempotent_system_surface_kind(
            action_kind
        )
        if (
            idempotent_surface_kind is not None
            and _surface_kind(scene) == idempotent_surface_kind
        ):
            continue
        if (
            action_kind == "reveal_system_navigation"
            and scene.system_ui.navigation_bar_visible is True
        ):
            continue
        candidates.append(
            _candidate(
                action_kind=action_kind,
                subject_refs=(
                    (surface_ref, target_swipe_ref)
                    if action_kind == "swipe"
                    and target_swipe_presence is not None
                    else (surface_ref,)
                ),
                affordance_ids=(affordance.affordance_id,),
                relation_ids=(),
                precondition_claim_ids=(
                    tuple(surface_claim_ids) + target_swipe_claim_ids
                    if action_kind == "swipe"
                    and target_swipe_presence is not None
                    else surface_claim_ids
                ),
                expectations=expectations,
                exploratory=exploratory,
                parameters=parameters,
            )
        )

    action_priority = {
        "tap_semantic": 0,
        "input_verified_text": 1,
        "press_enter": 2,
        "clear_verified_text": 3,
        "dismiss_overlay": 4,
        "double_tap": 5,
        "long_press": 6,
        "drag": 7,
        "open_recent_apps": 8,
        "home": 9,
        "reveal_system_navigation": 10,
        "back": 11,
        "swipe": 12,
        "wait_for_change": 13,
    }
    unique_candidates = {item.candidate_id: item for item in candidates}
    element_by_id = {item.element_id: item for item in sorted_elements}

    def clears_nonprefix_active_input(candidate: CanonicalActionCandidate) -> bool:
        if candidate.action_kind != "clear_verified_text":
            return False
        element = element_by_id.get(
            str(candidate.parameters.get("element_id") or "")
        )
        if element is None or len(active_input_payload_entities) != 1:
            return False
        current_value = element.states.get("value")
        current_preedit = element.states.get("ime_preedit_text")
        exact_candidate = element.states.get("ime_exact_candidate_text")
        authorized_value = active_input_payload_entities[0].value
        if not isinstance(current_value, str) or not isinstance(authorized_value, str):
            return False
        if isinstance(current_preedit, str) and current_preedit:
            useful_exact_candidate = bool(
                isinstance(exact_candidate, str)
                and exact_candidate
                and authorized_value.startswith(current_value + exact_candidate)
            )
            # A raw IME preedit is not committed application text.  Its glyphs
            # merely forming a prefix of the authorized payload cannot make it
            # reusable, because the current IME mode may assign different
            # semantics to that composition.  Preserve it only when the same
            # scene binds an exact, verifiable candidate that can commit the
            # authorized next value; otherwise clearing is the sole recovery.
            return not useful_exact_candidate
        return bool(current_value and not authorized_value.startswith(current_value))

    def clears_useful_active_preedit(candidate: CanonicalActionCandidate) -> bool:
        if candidate.action_kind != "clear_verified_text":
            return False
        element = element_by_id.get(
            str(candidate.parameters.get("element_id") or "")
        )
        if element is None or len(active_input_payload_entities) != 1:
            return False
        current_value = element.states.get("value")
        current_preedit = element.states.get("ime_preedit_text")
        exact_candidate = element.states.get("ime_exact_candidate_text")
        authorized_value = active_input_payload_entities[0].value
        return bool(
            isinstance(current_value, str)
            and isinstance(current_preedit, str)
            and current_preedit
            and isinstance(authorized_value, str)
            and isinstance(exact_candidate, str)
            and exact_candidate
            and authorized_value.startswith(current_value + exact_candidate)
        )

    def belongs_to_active_subgoal(candidate: CanonicalActionCandidate) -> bool:
        if candidate.effect_ref:
            return candidate.effect_ref in active_effect_refs
        action_kind = candidate.action_kind
        if (
            action_kind == "clear_verified_text"
            and clears_useful_active_preedit(candidate)
            and not (
                "clear_verified_text" in active_required_actions
                and "input_verified_text" not in active_required_actions
            )
        ):
            return False
        if active_required_actions:
            if "input_verified_text" in active_required_actions:
                if action_kind == "input_verified_text":
                    pass
                elif (
                    action_kind == "press_enter"
                    and "press_enter" in active_required_actions
                ):
                    pass
                elif action_kind == "tap_semantic":
                    required_element = element_by_id.get(
                        str(candidate.parameters.get("element_id") or "")
                    )
                    if required_element is None or not (
                        required_element.role == "input"
                        or required_element.meaning
                        in {
                            "ime_exact_candidate",
                            "input_exact_literal_key",
                            "input_exact_enter_key",
                            "switch_keyboard_layout",
                            "switch_keyboard_case",
                            "switch_keyboard_input_mode",
                            "input_next_field_key",
                        }
                    ):
                        return False
                elif (
                    action_kind == "clear_verified_text"
                    and (
                        "clear_verified_text" in active_required_actions
                        or clears_nonprefix_active_input(candidate)
                    )
                ):
                    pass
                else:
                    return False
            elif action_kind not in active_required_actions:
                return False
        if action_kind in {
            "input_verified_text",
            "press_enter",
            "clear_verified_text",
        }:
            if action_kind == "clear_verified_text":
                return bool(active_input_payload_refs) or bool(
                    "clear_verified_text" in active_required_actions
                    and active_targets_input
                )
            if action_kind == "press_enter":
                enter_element = element_by_id.get(
                    str(candidate.parameters.get("element_id") or "")
                )
                enter_field_id = str(
                    enter_element.states.get("input_field_id")
                    if enter_element is not None
                    else ""
                ).strip()
                return bool(
                    active_input_payload_refs
                    or (
                        "press_enter" in active_required_actions
                        and len(enter_input_field_ids) == 1
                        and enter_field_id in enter_input_field_ids
                    )
                )
            return bool(active_input_payload_refs)
        if action_kind == "tap_semantic":
            element_id = str(candidate.parameters.get("element_id") or "")
            element = element_by_id.get(element_id)
            if element is None:
                return False
            surfaces = {item.surface_id: item for item in semantic_ir.surfaces}
            target_surface = surfaces.get(active_subgoal.surface_ref)
            current_kind = _surface_kind(scene)
            if (
                target_surface is not None
                and target_surface.kind == "app"
                and current_kind != "launcher"
                and not scene_matches_target_app_surface(scene, target_surface)
            ):
                return False
            if element.meaning == "input_next_field_key":
                return any(
                    relation_by_id[relation_id].relation
                    == "binds_next_input_field"
                    and relation_by_id[relation_id].object_ref
                    in {item.field_id for item in active_input_fields}
                    for relation_id in candidate.relation_ids
                )
            if element.meaning in {
                "ime_exact_candidate",
                "input_exact_literal_key",
                "input_exact_enter_key",
                "switch_keyboard_layout",
                "switch_keyboard_case",
                "switch_keyboard_input_mode",
            }:
                if element.meaning == "input_exact_enter_key":
                    return False
                return bool(active_input_payload_refs)
            if element.role == "input":
                if element.states.get("focused") is True:
                    # A focused field already satisfies the navigation
                    # affordance.  Advertising another tap beside the typed
                    # input transaction creates two candidates for the same
                    # target and may move the caret away from the verified end
                    # position.  Only the deterministic input action remains.
                    return False
                if len(active_input_fields) == 1:
                    active_field = active_input_fields[0]
                    return bool(
                        (
                            element.states.get("input_field_id")
                            == active_field.field_id
                            and (
                                not active_field.field_label
                                or element.states.get("input_field_label")
                                == active_field.field_label
                            )
                        )
                        or (
                            len(semantic_ir.input_fields) == 1
                            and not active_field.field_label
                        )
                    )
                return active_targets_input
            target_surface = surfaces.get(active_subgoal.surface_ref)
            if target_surface is not None and target_surface.kind == "app":
                if current_kind == "launcher":
                    return any(
                        relation_by_id[relation_id].relation == "binds_surface"
                        and relation_by_id[relation_id].object_ref
                        == active_subgoal.surface_ref
                        for relation_id in candidate.relation_ids
                    )
                if not scene_matches_target_app_surface(scene, target_surface):
                    return False
            for relation_id in candidate.relation_ids:
                relation = relation_by_id[relation_id]
                if (
                    relation.relation == "binds_surface"
                    and relation.object_ref == active_subgoal.surface_ref
                ):
                    return True
                if (
                    relation.relation == "exact_literal_match"
                    and relation.object_ref in active_entity_refs
                ):
                    return True
            return bool(
                active_subgoal.external_impact == "navigation_only"
                and element.meaning not in _ALL_EFFECT_CONTROL_MEANINGS
            )
        if action_kind == "dismiss_overlay":
            return active_subgoal.external_impact == "navigation_only"
        if action_kind in {"double_tap", "long_press", "drag"}:
            return action_kind in active_required_actions
        if action_kind == "home":
            surfaces = {
                item.surface_id: item for item in semantic_ir.surfaces
            }
            target = surfaces.get(active_subgoal.surface_ref)
            current_kind = _surface_kind(scene)
            if target is None:
                return False
            if target.kind == "launcher":
                return current_kind != "launcher"
            if target.kind != "app" or current_kind == "launcher":
                return False
            return not scene_matches_target_app_surface(scene, target)
        if action_kind == "wait_for_change":
            if active_input_fields:
                return action_kind in active_required_actions
            surfaces = {
                item.surface_id: item for item in semantic_ir.surfaces
            }
            target = surfaces.get(active_subgoal.surface_ref)
            if (
                target is not None
                and target.kind == "app"
                and _surface_kind(scene) != "launcher"
                and not scene_matches_target_app_surface(scene, target)
            ):
                return False
            return active_subgoal.external_impact in {
                "read_only",
                "navigation_only",
            }
        if action_kind in {
            "back",
            "open_recent_apps",
            "swipe",
            "reveal_system_navigation",
        }:
            if active_input_fields:
                return action_kind in active_required_actions
            surfaces = {
                item.surface_id: item for item in semantic_ir.surfaces
            }
            target = surfaces.get(active_subgoal.surface_ref)
            if (
                target is not None
                and target.kind == "app"
                and _surface_kind(scene) != "launcher"
                and not scene_matches_target_app_surface(scene, target)
            ):
                # A stable non-target App must use the canonical Home reset.
                # Back, scroll or passive waiting cannot establish the unique
                # launcher checkpoint required before choosing the target App.
                return False
            return active_subgoal.external_impact == "navigation_only"
        return False

    unique_candidates = {
        candidate_id: candidate
        for candidate_id, candidate in unique_candidates.items()
        if belongs_to_active_subgoal(candidate)
    }
    candidates = sorted(
        unique_candidates.values(),
        key=lambda item: (
            action_priority[item.action_kind],
            item.subject_refs,
            item.candidate_id,
        ),
    )[:MAX_READY_CANDIDATES]
    status = "ready" if len(candidates) >= MIN_READY_CANDIDATES else "blocked"
    warnings: list[str] = []
    if status == "blocked":
        warnings.append("insufficient_local_candidates")
    if any(len(matches) > 1 for matches in exact_elements_by_entity.values()):
        warnings.append("duplicate_exact_literal_binding")

    report = CanonicalActionCatalog(
        task_id=semantic_ir.task_id,
        device_id=semantic_ir.device_id,
        revision=semantic_ir.revision,
        scene_digest=scene_digest,
        semantic_digest=semantic_ir.semantic_digest,
        claims=tuple(claims),
        relations=tuple(relations),
        affordances=tuple(affordances),
        candidates=tuple(candidates),
        status=status,
        warnings=tuple(sorted(set(warnings))),
    )
    report.validate()
    return report


def canonical_candidate_expected_result(
    candidate: CanonicalActionCandidate,
    scene: UIScene,
) -> dict[str, Any]:
    """Project one canonical transition into the controller's visual result shape."""

    candidate.validate()
    if candidate.action_kind == "reveal_system_navigation":
        return {"system_ui": {"navigation_bar_visible": True}}
    if candidate.action_kind == "swipe":
        element_id = str(candidate.parameters.get("element_id") or "").strip()
        if element_id:
            matches = tuple(
                element
                for element in scene.elements
                if element.element_id == element_id
            )
            if len(matches) != 1:
                raise CanonicalActionProtocolError(
                    "元素绑定 swipe 引用了不存在或不唯一的当前元素。"
                )
            element = matches[0]
            return {
                "content_changed": True,
                "element_absent": {
                    "element_id": element.element_id,
                    "meaning": element.meaning,
                    "role": element.role,
                    "label": element.label,
                },
            }
        return {"content_changed": True}

    element_by_ref = {
        _element_ref(element.element_id): element for element in scene.elements
    }
    state_expectations = [
        item
        for item in candidate.transition.expectations
        if item.predicate.startswith("element.state.")
        and item.operator == "equals"
        and item.subject_ref in element_by_ref
    ]
    if state_expectations:
        subjects = {item.subject_ref for item in state_expectations}
        if len(subjects) != 1:
            raise CanonicalActionProtocolError(
                "canonical candidate 包含多个元素的状态结果，无法形成唯一验证目标。"
            )
        subject_ref = next(iter(subjects))
        element = element_by_ref[subject_ref]
        states = {
            item.predicate.removeprefix("element.state."): item.value
            for item in state_expectations
        }
        return {
            "element_state": {
                "meaning": element.meaning,
                "states": states,
            }
        }
    return {"scene_changed": True}


def select_canonical_action_candidate(
    report: CanonicalActionCatalog,
    *,
    report_digest: str,
    candidate_id: str,
) -> CanonicalActionCandidate:
    """Return exactly one immutable candidate from the current catalog."""

    report.validate()
    if report_digest != report.report_digest:
        raise CanonicalActionProtocolError(
            "canonical action catalog digest 已过期或不匹配。"
        )
    matches = [item for item in report.candidates if item.candidate_id == candidate_id]
    if len(matches) != 1:
        raise CanonicalActionProtocolError(
            "canonical candidate_id 不存在或不唯一。"
        )
    return matches[0]
