from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping

from task_semantic_ir import EffectIntent, SemanticEntity, TaskSemanticIR
from ui_scene import UIElement, UIScene


VISUAL_ACTION_SHADOW_PROTOCOL = "2026-08-18-visual-action-shadow-v1"
VISUAL_ACTION_AUTHORITY_PROTOCOL = "2026-08-19-visual-action-authority-v1"
VISUAL_CLAIM_PROTOCOL = "2026-08-18-visual-claim-v1-shadow"
SHADOW_SELECTION_PROTOCOL = "2026-08-18-shadow-candidate-selection-v1"

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
        "reveal_system_navigation",
        "input_verified_text",
        "clear_verified_text",
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
        "element.state.focused",
        "element.state.value",
        "element.state.interaction_result",
        "element.state.location_relation",
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
        "ime_preedit_text",
        "navigation_bar_visible",
    }
)


class VisualActionShadowError(ValueError):
    pass


def _json_value(value: Any, field_name: str) -> Any:
    try:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise VisualActionShadowError(f"{field_name} 必须是可序列化 JSON 值。") from exc


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
        raise VisualActionShadowError(f"{field_name} 无效：{value!r}")


def _required_text(value: Any, field_name: str, *, max_length: int = 300) -> str:
    if not isinstance(value, str) or not value.strip():
        raise VisualActionShadowError(f"{field_name} 必须是非空字符串。")
    text = value.strip()
    if len(text) > max_length:
        raise VisualActionShadowError(f"{field_name} 超过长度限制。")
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
    identity = f"{scene.foreground_app_id} {scene.screen_id}".casefold()
    if any(token in identity for token in ("launcher", "home_screen", "desktop")):
        return "launcher"
    if scene.overlays:
        return "system_dialog" if "system" in identity else "app"
    return "app"


@dataclass(frozen=True)
class VisualClaim:
    claim_id: str
    subject_ref: str
    predicate: str
    value: Any
    confidence: float
    source_digest: str
    protocol_version: str = VISUAL_CLAIM_PROTOCOL

    def validate(self) -> None:
        if self.protocol_version != VISUAL_CLAIM_PROTOCOL:
            raise VisualActionShadowError("VisualClaim protocol_version 无效。")
        _validate_id(self.claim_id, "claim.claim_id")
        _validate_id(self.subject_ref, "claim.subject_ref")
        _required_text(self.predicate, "claim.predicate", max_length=100)
        if self.predicate not in CLAIM_PREDICATES and not (
            self.predicate.startswith("element.state.")
            and self.predicate.removeprefix("element.state.") in _SAFE_STATE_KEYS
        ):
            raise VisualActionShadowError(
                f"claim.predicate 无效：{self.predicate}"
            )
        _json_value(self.value, "claim.value")
        if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)):
            raise VisualActionShadowError("claim.confidence 格式无效。")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise VisualActionShadowError("claim.confidence 超出范围。")
        if not re.fullmatch(r"[0-9a-f]{64}", self.source_digest):
            raise VisualActionShadowError("claim.source_digest 必须是 SHA-256。")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "protocol_version": self.protocol_version,
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
            raise VisualActionShadowError(f"relation.relation 无效：{self.relation}")
        if not self.support_claim_ids:
            raise VisualActionShadowError("relation.support_claim_ids 不能为空。")
        if len(set(self.support_claim_ids)) != len(self.support_claim_ids):
            raise VisualActionShadowError("relation.support_claim_ids 重复。")
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
            raise VisualActionShadowError(
                f"affordance.action_kind 无效：{self.action_kind}"
            )
        if not self.support_claim_ids:
            raise VisualActionShadowError("affordance.support_claim_ids 不能为空。")
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
            raise VisualActionShadowError(
                f"expectation.predicate 无效：{self.predicate}"
            )
        if self.operator not in EXPECTATION_OPERATORS:
            raise VisualActionShadowError(
                f"expectation.operator 无效：{self.operator}"
            )
        if self.operator in {"equals", "not_equals"}:
            _json_value(self.value, "expectation.value")
        elif self.value is not None:
            raise VisualActionShadowError(
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
            raise VisualActionShadowError("transition.precondition_claim_ids 不能为空。")
        for value in self.precondition_claim_ids:
            _validate_id(value, "transition.precondition_claim_ids")
        if not self.expectations:
            raise VisualActionShadowError("transition.expectations 不能为空。")
        for expectation in self.expectations:
            expectation.validate()
            if (
                expectation.predicate in {"scene.changed", "observation.changed"}
                and not self.exploratory
            ):
                raise VisualActionShadowError(
                    "scene/observation changed 只能用于 exploratory transition。"
                )
        if not isinstance(self.exploratory, bool):
            raise VisualActionShadowError("transition.exploratory 必须是布尔值。")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "transition_id": self.transition_id,
            "precondition_claim_ids": list(self.precondition_claim_ids),
            "expectations": [item.to_dict() for item in self.expectations],
            "exploratory": self.exploratory,
        }


@dataclass(frozen=True)
class ShadowActionCandidate:
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
            raise VisualActionShadowError(f"candidate.action_kind 无效：{self.action_kind}")
        if not self.subject_refs or not self.affordance_ids:
            raise VisualActionShadowError("candidate 缺少 subject/affordance 绑定。")
        for field_name, values in (
            ("subject_refs", self.subject_refs),
            ("affordance_ids", self.affordance_ids),
            ("relation_ids", self.relation_ids),
        ):
            if len(set(values)) != len(values):
                raise VisualActionShadowError(f"candidate.{field_name} 重复。")
            for value in values:
                _validate_id(value, f"candidate.{field_name}")
        _json_value(self.parameters, "candidate.parameters")
        if any(key in self.parameters for key in {"bounds", "point", "x", "y"}):
            raise VisualActionShadowError("shadow candidate 不得携带坐标。")
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
class VisualActionShadowReport:
    task_id: str
    device_id: str
    revision: int
    scene_digest: str
    semantic_digest: str
    claims: tuple[VisualClaim, ...]
    relations: tuple[VisualRelation, ...]
    affordances: tuple[Affordance, ...]
    candidates: tuple[ShadowActionCandidate, ...]
    status: str
    warnings: tuple[str, ...] = ()
    authoritative: bool = False
    execution_allowed: bool = False
    protocol_version: str = VISUAL_ACTION_SHADOW_PROTOCOL

    def validate(self) -> None:
        expected_protocol = (
            VISUAL_ACTION_AUTHORITY_PROTOCOL
            if self.authoritative
            else VISUAL_ACTION_SHADOW_PROTOCOL
        )
        if self.protocol_version != expected_protocol:
            raise VisualActionShadowError("visual report protocol_version 无效。")
        if self.execution_allowed:
            raise VisualActionShadowError("视觉候选报告本身不得授予物理执行权限。")
        _required_text(self.task_id, "report.task_id", max_length=128)
        _required_text(self.device_id, "report.device_id", max_length=128)
        if isinstance(self.revision, bool) or not isinstance(self.revision, int) or self.revision < 1:
            raise VisualActionShadowError("report.revision 必须是正整数。")
        for field_name, value in (
            ("scene_digest", self.scene_digest),
            ("semantic_digest", self.semantic_digest),
        ):
            if not re.fullmatch(r"[0-9a-f]{64}", value):
                raise VisualActionShadowError(f"report.{field_name} 必须是 SHA-256。")
        if self.status not in {"ready", "blocked"}:
            raise VisualActionShadowError("report.status 无效。")
        if self.status == "ready" and not MIN_READY_CANDIDATES <= len(self.candidates) <= MAX_READY_CANDIDATES:
            raise VisualActionShadowError(
                f"ready report 必须包含{MIN_READY_CANDIDATES}至"
                f"{MAX_READY_CANDIDATES}个候选。"
            )
        if self.status == "blocked" and len(self.candidates) >= MIN_READY_CANDIDATES:
            raise VisualActionShadowError("候选已足够时不得标记 blocked。")

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
                    raise VisualActionShadowError(f"report.{field_name} ID 重复。")
                seen.add(item_id)

        claim_ids = {item.claim_id for item in self.claims}
        claimed_subjects = {item.subject_ref for item in self.claims}
        relation_ids = {item.relation_id for item in self.relations}
        affordance_ids = {item.affordance_id for item in self.affordances}
        affordance_by_id = {item.affordance_id: item for item in self.affordances}
        for relation in self.relations:
            if relation.subject_ref not in claimed_subjects:
                raise VisualActionShadowError("relation.subject_ref 没有事实主体。")
            if not set(relation.support_claim_ids).issubset(claim_ids):
                raise VisualActionShadowError("relation 引用未知 claim。")
        for affordance in self.affordances:
            if affordance.subject_ref not in claimed_subjects:
                raise VisualActionShadowError("affordance.subject_ref 没有事实主体。")
            if not set(affordance.support_claim_ids).issubset(claim_ids):
                raise VisualActionShadowError("affordance 引用未知 claim。")
        for candidate in self.candidates:
            if not set(candidate.subject_refs).issubset(claimed_subjects):
                raise VisualActionShadowError("candidate.subject_refs 没有事实主体。")
            if not set(candidate.relation_ids).issubset(relation_ids):
                raise VisualActionShadowError("candidate 引用未知 relation。")
            if not set(candidate.affordance_ids).issubset(affordance_ids):
                raise VisualActionShadowError("candidate 引用未知 affordance。")
            bound_affordances = [
                affordance_by_id[value] for value in candidate.affordance_ids
            ]
            if any(
                item.action_kind != candidate.action_kind
                or item.subject_ref not in candidate.subject_refs
                for item in bound_affordances
            ):
                raise VisualActionShadowError("candidate 与 affordance 绑定不一致。")
            if not set(candidate.transition.precondition_claim_ids).issubset(claim_ids):
                raise VisualActionShadowError("transition 引用未知 claim。")
            if any(
                item.subject_ref not in claimed_subjects
                and item.subject_ref != candidate.effect_ref
                for item in candidate.transition.expectations
            ):
                raise VisualActionShadowError("transition expectation 没有事实主体。")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "protocol_version": self.protocol_version,
            "authoritative": self.authoritative,
            "execution_allowed": self.execution_allowed,
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


@dataclass(frozen=True)
class ShadowCandidateSelection:
    report_digest: str
    candidate: ShadowActionCandidate
    authoritative: bool = False
    execution_allowed: bool = False
    protocol_version: str = SHADOW_SELECTION_PROTOCOL

    def validate(self) -> None:
        if self.protocol_version != SHADOW_SELECTION_PROTOCOL:
            raise VisualActionShadowError("shadow selection protocol_version 无效。")
        if self.authoritative or self.execution_allowed:
            raise VisualActionShadowError("shadow selection 不得取得执行权限。")
        if not re.fullmatch(r"[0-9a-f]{64}", self.report_digest):
            raise VisualActionShadowError("selection.report_digest 必须是 SHA-256。")
        self.candidate.validate()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "protocol_version": self.protocol_version,
            "authoritative": self.authoritative,
            "execution_allowed": self.execution_allowed,
            "report_digest": self.report_digest,
            "candidate": self.candidate.to_dict(),
        }


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


def _element_proves_scrollable_viewport(element: UIElement) -> bool:
    """Grant swipe affordance only from a typed, evidenced viewport fact."""

    return (
        element.role == "container"
        and float(element.confidence) >= MIN_ELEMENT_CONFIDENCE
        and element.states.get("visible") is not False
        and element.states.get("fully_visible") is True
        and element.states.get("scrollable") is True
        and element.states.get("scroll_axis") in {"vertical", "horizontal"}
        and any(str(item).strip() for item in element.evidence)
    )


def _unique_exact_matches(
    elements: tuple[UIElement, ...],
    entity: SemanticEntity,
) -> tuple[UIElement, ...]:
    if not isinstance(entity.value, str) or not entity.value:
        return ()
    literal = entity.value.casefold()
    return tuple(
        element
        for element in elements
        if _element_eligible(element)
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
) -> ShadowActionCandidate:
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
    return ShadowActionCandidate(
        candidate_id=_stable_id("candidate", payload),
        action_kind=action_kind,
        subject_refs=subject_refs,
        affordance_ids=affordance_ids,
        relation_ids=relation_ids,
        transition=transition,
        parameters=dict(parameters or {}),
        effect_ref=effect_ref,
    )


def compile_visual_action_shadow(
    scene: UIScene,
    semantic_ir: TaskSemanticIR,
    available_action_kinds: Iterable[str],
) -> VisualActionShadowReport:
    """Compile a deterministic, non-authoritative visual/action shadow report."""

    scene.validate()
    semantic_ir.validate()
    available = frozenset(str(value) for value in available_action_kinds)
    unknown = available - SUPPORTED_ACTIONS
    if unknown:
        raise VisualActionShadowError(
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
    for entity in semantic_ir.entities:
        matches = _unique_exact_matches(sorted_elements, entity)
        exact_elements_by_entity[entity.entity_id] = matches
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

    affordances: list[Affordance] = []
    for action_kind in sorted(available):
        if action_kind in {"back", "home", "reveal_system_navigation", "swipe", "wait_for_change"}:
            if action_kind == "swipe" and not any(
                _element_proves_scrollable_viewport(element)
                for element in sorted_elements
            ):
                continue
            if action_kind == "reveal_system_navigation" and not (
                scene.system_ui.immersive_or_fullscreen is True
                and scene.system_ui.navigation_bar_visible is False
            ):
                continue
            affordances.append(_affordance(surface_ref, action_kind, surface_claim_ids))
    for element in sorted_elements:
        if not _element_eligible(element):
            continue
        supported: set[str] = set()
        if "tap_semantic" in available:
            supported.add("tap_semantic")
        if "long_press" in available:
            supported.add("long_press")
        if "drag" in available:
            supported.add("drag")
        if element.role == "input" and element.states.get("focused") is True:
            if "input_verified_text" in available:
                supported.add("input_verified_text")
            if "clear_verified_text" in available and bool(element.states.get("value")):
                supported.add("clear_verified_text")
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

    candidates: list[ShadowActionCandidate] = []
    active_external_effect_refs = tuple(
        dict.fromkeys(
            effect_ref
            for subgoal in semantic_ir.subgoals
            if subgoal.status == "active"
            and subgoal.external_impact == "external_state"
            for effect_ref in subgoal.effect_refs
        )
    )
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
            elif relation.relation == "on_surface":
                unique_relation_ids.append(relation_id)
        if not unique_relation_ids:
            continue

        tap_affordance = affordance_by_pair.get((element_ref, "tap_semantic"))
        if tap_affordance is not None:
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
            elif element.meaning in {
                "ime_exact_candidate",
                "input_exact_literal_key",
                "switch_keyboard_layout",
                "switch_keyboard_case",
                "switch_keyboard_input_mode",
            }:
                expected_input_value = (
                    element.states.get("expected_input_value")
                    if element.meaning
                    in {"ime_exact_candidate", "input_exact_literal_key"}
                    else element.states.get("prior_input_value")
                )
                if not isinstance(expected_input_value, str):
                    continue
                expectation = StateExpectation(
                    element_ref,
                    "element.state.value",
                    "equals",
                    expected_input_value,
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
            effect_ref = ""
            effects = sorted(
                {
                    effect_id
                    for effect_id, entity_id, relation_kind in relation_effects_by_element.get(
                        element.element_id, ()
                    )
                    if len(exact_elements_by_entity.get(entity_id, ())) == 1
                    and relation_kind == "binds_effect_target"
                }
            )
            if len(effects) == 1:
                effect_ref = effects[0]
                expectation = StateExpectation(
                    effect_ref,
                    "effect.applied",
                    "equals",
                    True,
                )
            elif not effects and len(active_external_effect_refs) == 1:
                # Qwen selects which visible control realizes the already
                # authorized typed effect.  The model cannot change the effect
                # identity, target, payload or local policy.
                effect_ref = active_external_effect_refs[0]
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
                    expectations=(expectation,),
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
                            "switch_keyboard_layout",
                            "switch_keyboard_case",
                            "switch_keyboard_input_mode",
                        }
                    ),
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
                and entity_id in entity_by_id
                and entity_by_id[entity_id].role == "input_text"
            ]
            # A focused empty input does not literally contain the future payload.
            # Bind the unique typed input_text payload directly to the input affordance.
            if not payload_entities:
                payload_entities = [
                    entity
                    for entity in semantic_ir.entities
                    if entity.role == "input_text" and isinstance(entity.value, str)
                ]
            if len(payload_entities) == 1:
                payload = payload_entities[0]
                payload_effects = sorted(
                    effect.effect_id
                    for effect in semantic_ir.effects
                    if payload.entity_id in effect.payload_refs
                )
                candidates.append(
                    _candidate(
                        action_kind="input_verified_text",
                        subject_refs=(element_ref,),
                        affordance_ids=(input_affordance.affordance_id,),
                        relation_ids=tuple(sorted(set(unique_relation_ids))),
                        precondition_claim_ids=tuple(element_claim_ids[element.element_id]),
                        expectations=(
                            StateExpectation(
                                element_ref,
                                "element.state.value",
                                "equals",
                                payload.value,
                            ),
                        ),
                        effect_ref=payload_effects[0] if len(payload_effects) == 1 else "",
                        parameters={"element_id": element.element_id},
                    )
                )

        clear_affordance = affordance_by_pair.get((element_ref, "clear_verified_text"))
        if clear_affordance is not None:
            candidates.append(
                _candidate(
                    action_kind="clear_verified_text",
                    subject_refs=(element_ref,),
                    affordance_ids=(clear_affordance.affordance_id,),
                    relation_ids=tuple(sorted(set(unique_relation_ids))),
                    precondition_claim_ids=tuple(element_claim_ids[element.element_id]),
                    expectations=(
                        StateExpectation(
                            element_ref,
                            "element.state.value",
                            "equals",
                            "",
                        ),
                    ),
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
    system_specs: tuple[tuple[str, tuple[StateExpectation, ...], bool, dict[str, Any]], ...] = (
        (
            "home",
            (StateExpectation(surface_ref, "surface.kind", "equals", "launcher"),),
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
        *tuple(
            (
                "swipe",
                (StateExpectation(surface_ref, "surface.viewport", "changed"),),
                True,
                {"direction": direction},
            )
            for direction in ("up", "down", "left", "right")
        ),
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
        if action_kind == "home" and _surface_kind(scene) == "launcher":
            continue
        if (
            action_kind == "reveal_system_navigation"
            and scene.system_ui.navigation_bar_visible is True
        ):
            continue
        candidates.append(
            _candidate(
                action_kind=action_kind,
                subject_refs=(surface_ref,),
                affordance_ids=(affordance.affordance_id,),
                relation_ids=(),
                precondition_claim_ids=surface_claim_ids,
                expectations=expectations,
                exploratory=exploratory,
                parameters=parameters,
            )
        )

    action_priority = {
        "tap_semantic": 0,
        "input_verified_text": 1,
        "clear_verified_text": 2,
        "dismiss_overlay": 3,
        "long_press": 4,
        "drag": 5,
        "home": 6,
        "reveal_system_navigation": 7,
        "back": 8,
        "swipe": 9,
        "wait_for_change": 10,
    }
    unique_candidates = {item.candidate_id: item for item in candidates}
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

    report = VisualActionShadowReport(
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


def select_shadow_candidate(
    report: VisualActionShadowReport,
    *,
    report_digest: str,
    candidate_id: str,
) -> ShadowCandidateSelection:
    """Select exactly one existing candidate without allowing any mutation."""

    report.validate()
    if report_digest != report.report_digest:
        raise VisualActionShadowError("shadow report digest 已过期或不匹配。")
    matches = [item for item in report.candidates if item.candidate_id == candidate_id]
    if len(matches) != 1:
        raise VisualActionShadowError("shadow candidate_id 不存在或不唯一。")
    selection = ShadowCandidateSelection(
        report_digest=report_digest,
        candidate=matches[0],
    )
    selection.validate()
    return selection


def compile_visual_action_authority(
    scene: UIScene,
    semantic_ir: TaskSemanticIR,
    available_action_kinds: Iterable[str],
) -> VisualActionShadowReport:
    """Promote the deterministic local candidate graph, never model prose.

    Authority here means only candidate identity and typed transition authority.
    It cannot execute hardware; the fresh-observation, policy, geometry and
    one-shot confirmation gates remain mandatory downstream.
    """

    shadow = compile_visual_action_shadow(
        scene,
        semantic_ir,
        available_action_kinds,
    )
    report = replace(
        shadow,
        protocol_version=VISUAL_ACTION_AUTHORITY_PROTOCOL,
        authoritative=True,
        execution_allowed=False,
    )
    report.validate()
    return report


def compile_visual_action_shadow_safe(
    scene: UIScene,
    semantic_ir: TaskSemanticIR,
    available_action_kinds: Iterable[str],
) -> tuple[VisualActionShadowReport | None, dict[str, Any] | None]:
    """Fail-open for production behavior, fail-closed for shadow authority."""

    try:
        return compile_visual_action_shadow(scene, semantic_ir, available_action_kinds), None
    except Exception as exc:  # diagnostic boundary: never changes the formal path
        return None, {
            "protocol_version": VISUAL_ACTION_SHADOW_PROTOCOL,
            "authoritative": False,
            "execution_allowed": False,
            "status": "shadow_error",
            "error_type": type(exc).__name__,
            "error_message": str(exc)[:500],
        }
