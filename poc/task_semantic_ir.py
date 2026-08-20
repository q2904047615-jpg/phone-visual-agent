from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping


TASK_SEMANTIC_IR_PROTOCOL = "2026-08-20-task-semantic-ir-v3"
RISK_POLICY_PROTOCOL = "2026-08-18-local-risk-policy-v1"
COMPILATION_REPORT_PROTOCOL = "2026-08-20-semantic-compilation-v1"
AUTHORITY_REPORT_PROTOCOL = "2026-08-20-typed-effect-authority-v1"
POLICY_TRACE_PROTOCOL = "2026-08-20-semantic-policy-trace-v1"
EFFECT_PREVIEW_PROTOCOL = "2026-08-19-effect-preview-v1"

AUTOMATIC = "automatic"
CONFIRMATION_REQUIRED = "confirmation_required"
RISK_POLICIES = frozenset({AUTOMATIC, CONFIRMATION_REQUIRED})

DEFAULT_CONFIRMATION_EFFECT_KINDS = frozenset(
    {
        "authentication",
        "financial_transaction",
    }
)

SURFACE_KINDS = frozenset(
    {
        "launcher",
        "app",
        "system_dialog",
        "keyboard",
        "file_picker",
        "current_surface",
        "device",
        "system",
    }
)

ENTITY_AUTHORITIES = frozenset({"user_literal", "planner_context"})
CONSTRAINT_KINDS = frozenset(
    {
        "exact_entity",
        "forbidden_effect",
        "required_state",
        "required_action",
        "planner_context",
    }
)
REQUIRED_ACTION_KINDS = frozenset(
    {
        "tap_semantic",
        "double_tap",
        "swipe",
        "long_press",
        "drag",
        "input_verified_text",
        "clear_verified_text",
        "press_enter",
        "pinch",
        "home",
        "back",
        "hardware_key",
        "dismiss_overlay",
        "reveal_system_navigation",
    }
)
STATE_PREDICATES = frozenset(
    {
        "input.value_equals",
        "surface.state_visible",
        "effect.applied",
        "effect.result_visible",
        "observation.matches_description",
    }
)
EVIDENCE_SOURCE_KINDS = frozenset(
    {"visual_claim", "controller_transition", "effect_receipt"}
)
SUBGOAL_IMPACTS = frozenset(
    {"read_only", "navigation_only", "external_state", "unknown"}
)
_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
_EXTERNAL_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")

_RUNTIME_RISK_EFFECT_KIND = {
    "message_or_communication": "send_message",
    "content_publication": "publish_content",
    "account_relationship_change": "relationship_change",
    "membership_change": "membership_change",
    "permission_role_change": "sensitive_permission_change",
    "data_mutation": "data_mutation",
    "data_deletion": "irreversible_data_deletion",
    "transaction_or_payment": "financial_transaction",
    "account_or_permission_change": "sensitive_permission_change",
    "unknown_external_effect": "generic_effect",
}

_ENTITY_TYPE_BY_ROLE = {
    "recipient": "party",
    "input_text": "text",
    "target_ui_label": "ui_literal",
    "target_surface": "surface",
    "spatial_hint": "spatial_hint",
    "amount": "money",
    "currency": "currency",
    "merchant": "party",
    "payee": "party",
    "account": "account",
    "file": "file",
    "product": "product",
    "date": "date",
    "time": "time",
}

_TARGET_ROLES_BY_EFFECT = {
    "send_message": ("recipient",),
    "relationship_change": ("account", "recipient", "target"),
    "membership_change": ("account", "recipient", "target"),
    "financial_transaction": ("merchant", "payee", "recipient", "account"),
    "authentication": ("account",),
    "publish_content": ("account", "target"),
    "data_mutation": ("file", "target", "product"),
    "data_deletion": ("file", "target", "product", "account"),
    "sensitive_permission_change": ("account", "target"),
    "generic_effect": ("target", "target_ui_label", "account", "file", "product"),
}

_PAYLOAD_ROLES_BY_EFFECT = {
    "send_message": ("input_text",),
    "publish_content": ("input_text", "file"),
    "financial_transaction": ("amount", "currency", "product"),
    "data_mutation": ("input_text", "value"),
    "generic_effect": ("input_text", "value"),
}


class TaskSemanticIRError(ValueError):
    pass


def _required_text(value: Any, field_name: str, *, max_length: int = 500) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TaskSemanticIRError(f"{field_name} 必须是非空字符串。")
    text = value.strip()
    if len(text) > max_length:
        raise TaskSemanticIRError(f"{field_name} 超过长度限制。")
    return text


def _validate_id(value: str, field_name: str) -> None:
    if not _ID_PATTERN.fullmatch(str(value or "")):
        raise TaskSemanticIRError(f"{field_name} 无效：{value!r}")


def _validate_external_id(value: str, field_name: str) -> None:
    if not _EXTERNAL_ID_PATTERN.fullmatch(str(value or "")):
        raise TaskSemanticIRError(f"{field_name} 无效：{value!r}")


def _slug(value: Any, *, fallback: str) -> str:
    normalized = re.sub(r"[^a-z0-9_.-]+", "_", str(value or "").casefold())
    normalized = normalized.strip("_.-")[:72]
    if not normalized or not normalized[0].isalpha():
        normalized = f"{fallback}_{normalized}".rstrip("_")
    return normalized or fallback


def _json_value(value: Any, field_name: str) -> Any:
    try:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise TaskSemanticIRError(f"{field_name} 必须是可序列化 JSON 值。") from exc


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class SourceSpan:
    start: int
    end: int

    def validate(self, raw_goal: str) -> None:
        if (
            isinstance(self.start, bool)
            or isinstance(self.end, bool)
            or not isinstance(self.start, int)
            or not isinstance(self.end, int)
            or not 0 <= self.start < self.end <= len(raw_goal)
        ):
            raise TaskSemanticIRError("entity.source_span 超出用户原始目标。")

    def to_dict(self) -> dict[str, int]:
        return {"start": self.start, "end": self.end}


@dataclass(frozen=True)
class SemanticEntity:
    entity_id: str
    entity_type: str
    role: str
    value: Any
    source_span: SourceSpan | None = None
    authority: str = "planner_context"

    def validate(self, raw_goal: str) -> None:
        _validate_id(self.entity_id, "entity.entity_id")
        _validate_id(self.entity_type, "entity.entity_type")
        _validate_id(self.role, "entity.role")
        _json_value(self.value, f"entity.{self.entity_id}.value")
        if self.authority not in ENTITY_AUTHORITIES:
            raise TaskSemanticIRError(
                f"entity.{self.entity_id}.authority 无效：{self.authority}"
            )
        if self.source_span is not None:
            self.source_span.validate(raw_goal)
            literal = raw_goal[self.source_span.start : self.source_span.end]
            if not isinstance(self.value, str) or literal != self.value:
                raise TaskSemanticIRError(
                    f"entity.{self.entity_id}.source_span 未逐字绑定 value。"
                )
            if self.authority != "user_literal":
                raise TaskSemanticIRError(
                    f"entity.{self.entity_id} 有 source_span 时必须是 user_literal。"
                )
        elif self.authority == "user_literal":
            raise TaskSemanticIRError(
                f"entity.{self.entity_id} 的 user_literal 缺少 source_span。"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "role": self.role,
            "value": _json_value(self.value, f"entity.{self.entity_id}.value"),
            "source_span": self.source_span.to_dict() if self.source_span else None,
            "authority": self.authority,
        }


@dataclass(frozen=True)
class SurfaceRef:
    surface_id: str
    kind: str
    app_id: str = ""
    app_name: str = ""

    def validate(self) -> None:
        _validate_id(self.surface_id, "surface.surface_id")
        if self.kind not in SURFACE_KINDS:
            raise TaskSemanticIRError(f"surface.kind 无效：{self.kind}")
        if self.kind == "app":
            _validate_id(self.app_id, "surface.app_id")
            _required_text(self.app_name, "surface.app_name", max_length=120)
        elif self.app_id or self.app_name:
            raise TaskSemanticIRError("非 App surface 不得携带 app_id/app_name。")

    def to_dict(self) -> dict[str, str]:
        return {
            "surface_id": self.surface_id,
            "kind": self.kind,
            "app_id": self.app_id,
            "app_name": self.app_name,
        }


@dataclass(frozen=True)
class EffectIntent:
    effect_id: str
    kind: str
    target_refs: tuple[str, ...] = ()
    payload_refs: tuple[str, ...] = ()
    source_subgoal_ids: tuple[str, ...] = ()
    expected_result_texts: tuple[str, ...] = ()
    attributes: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _validate_id(self.effect_id, "effect.effect_id")
        _validate_id(self.kind, "effect.kind")
        for field_name, values in (
            ("target_refs", self.target_refs),
            ("payload_refs", self.payload_refs),
            ("source_subgoal_ids", self.source_subgoal_ids),
        ):
            if len(set(values)) != len(values):
                raise TaskSemanticIRError(f"effect.{self.effect_id}.{field_name} 重复。")
            for value in values:
                _validate_id(value, f"effect.{self.effect_id}.{field_name}")
        for value in self.expected_result_texts:
            _required_text(
                value,
                f"effect.{self.effect_id}.expected_result_texts",
                max_length=500,
            )
        _json_value(self.attributes, f"effect.{self.effect_id}.attributes")

    def to_dict(self) -> dict[str, Any]:
        return {
            "effect_id": self.effect_id,
            "kind": self.kind,
            "target_refs": list(self.target_refs),
            "payload_refs": list(self.payload_refs),
            "source_subgoal_ids": list(self.source_subgoal_ids),
            "expected_result_texts": list(self.expected_result_texts),
            "attributes": _json_value(
                self.attributes,
                f"effect.{self.effect_id}.attributes",
            ),
        }


@dataclass(frozen=True)
class CriticalBinding:
    binding_id: str
    effect_id: str
    binding_kind: str
    entity_ref: str

    def validate(self) -> None:
        _validate_id(self.binding_id, "critical_binding.binding_id")
        _validate_id(self.effect_id, "critical_binding.effect_id")
        if self.binding_kind not in {"effect_target_equals", "effect_payload_equals"}:
            raise TaskSemanticIRError(
                f"critical_binding.binding_kind 无效：{self.binding_kind}"
            )
        _validate_id(self.entity_ref, "critical_binding.entity_ref")

    def to_dict(self) -> dict[str, str]:
        return {
            "binding_id": self.binding_id,
            "effect_id": self.effect_id,
            "binding_kind": self.binding_kind,
            "entity_ref": self.entity_ref,
        }


@dataclass(frozen=True)
class ConstraintIntent:
    constraint_id: str
    kind: str
    subject_refs: tuple[str, ...] = ()
    object_refs: tuple[str, ...] = ()
    value: Any = None
    source_text: str = ""
    authoritative: bool = False

    def validate(self) -> None:
        _validate_id(self.constraint_id, "constraint.constraint_id")
        if self.kind not in CONSTRAINT_KINDS:
            raise TaskSemanticIRError(f"constraint.kind 无效：{self.kind}")
        for field_name, values in (
            ("subject_refs", self.subject_refs),
            ("object_refs", self.object_refs),
        ):
            if len(values) != len(set(values)):
                raise TaskSemanticIRError(
                    f"constraint.{self.constraint_id}.{field_name} 重复。"
                )
            for value in values:
                _validate_id(value, f"constraint.{self.constraint_id}.{field_name}")
        _json_value(self.value, f"constraint.{self.constraint_id}.value")
        if self.source_text:
            _required_text(
                self.source_text,
                f"constraint.{self.constraint_id}.source_text",
                max_length=500,
            )
        if not isinstance(self.authoritative, bool):
            raise TaskSemanticIRError("constraint.authoritative 必须是布尔值。")
        if self.kind == "planner_context" and self.authoritative:
            raise TaskSemanticIRError("planner_context 约束不得取得 authority。")
        if self.kind != "planner_context" and not self.authoritative:
            raise TaskSemanticIRError("typed constraint 必须明确取得 authority。")
        if self.kind == "required_action" and self.value not in REQUIRED_ACTION_KINDS:
            raise TaskSemanticIRError("required_action 约束的动作类型无效。")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "constraint_id": self.constraint_id,
            "kind": self.kind,
            "subject_refs": list(self.subject_refs),
            "object_refs": list(self.object_refs),
            "value": _json_value(self.value, f"constraint.{self.constraint_id}.value"),
            "source_text": self.source_text,
            "authoritative": self.authoritative,
        }


@dataclass(frozen=True)
class DesiredState:
    state_id: str
    subject_ref: str
    predicate: str
    value: Any
    source_subgoal_id: str = ""

    def validate(self) -> None:
        _validate_id(self.state_id, "desired_state.state_id")
        _validate_id(self.subject_ref, "desired_state.subject_ref")
        if self.predicate not in STATE_PREDICATES:
            raise TaskSemanticIRError(
                f"desired_state.predicate 无效：{self.predicate}"
            )
        _json_value(self.value, f"desired_state.{self.state_id}.value")
        if self.source_subgoal_id:
            _validate_id(
                self.source_subgoal_id,
                f"desired_state.{self.state_id}.source_subgoal_id",
            )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "state_id": self.state_id,
            "subject_ref": self.subject_ref,
            "predicate": self.predicate,
            "value": _json_value(self.value, f"desired_state.{self.state_id}.value"),
            "source_subgoal_id": self.source_subgoal_id,
        }


@dataclass(frozen=True)
class EvidenceRequirement:
    requirement_id: str
    desired_state_ref: str
    allowed_sources: tuple[str, ...]

    def validate(self) -> None:
        _validate_id(self.requirement_id, "evidence_requirement.requirement_id")
        _validate_id(
            self.desired_state_ref,
            "evidence_requirement.desired_state_ref",
        )
        if not self.allowed_sources or len(self.allowed_sources) != len(
            set(self.allowed_sources)
        ):
            raise TaskSemanticIRError(
                "evidence_requirement.allowed_sources 必须非空且不重复。"
            )
        unknown = set(self.allowed_sources) - EVIDENCE_SOURCE_KINDS
        if unknown:
            raise TaskSemanticIRError(
                "evidence_requirement.allowed_sources 无效："
                + ", ".join(sorted(unknown))
            )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "requirement_id": self.requirement_id,
            "desired_state_ref": self.desired_state_ref,
            "allowed_sources": list(self.allowed_sources),
        }


@dataclass(frozen=True)
class SemanticSubgoal:
    subgoal_id: str
    surface_ref: str
    status: str
    external_impact: str
    depends_on: tuple[str, ...] = ()
    constraint_refs: tuple[str, ...] = ()
    entity_refs: tuple[str, ...] = ()
    desired_state_refs: tuple[str, ...] = ()
    effect_refs: tuple[str, ...] = ()

    def validate(self) -> None:
        _validate_id(self.subgoal_id, "semantic_subgoal.subgoal_id")
        _validate_id(self.surface_ref, "semantic_subgoal.surface_ref")
        if not self.status:
            raise TaskSemanticIRError("semantic_subgoal.status 不能为空。")
        if self.external_impact not in SUBGOAL_IMPACTS:
            raise TaskSemanticIRError(
                f"semantic_subgoal.external_impact 无效：{self.external_impact}"
            )
        for field_name, values in (
            ("depends_on", self.depends_on),
            ("constraint_refs", self.constraint_refs),
            ("entity_refs", self.entity_refs),
            ("desired_state_refs", self.desired_state_refs),
            ("effect_refs", self.effect_refs),
        ):
            if len(values) != len(set(values)):
                raise TaskSemanticIRError(
                    f"semantic_subgoal.{self.subgoal_id}.{field_name} 重复。"
                )
            for value in values:
                _validate_id(value, f"semantic_subgoal.{self.subgoal_id}.{field_name}")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "subgoal_id": self.subgoal_id,
            "surface_ref": self.surface_ref,
            "status": self.status,
            "external_impact": self.external_impact,
            "depends_on": list(self.depends_on),
            "constraint_refs": list(self.constraint_refs),
            "entity_refs": list(self.entity_refs),
            "desired_state_refs": list(self.desired_state_refs),
            "effect_refs": list(self.effect_refs),
        }


@dataclass(frozen=True)
class InputFieldIntent:
    field_id: str
    payload_ref: str
    field_label: str = ""
    recipient_refs: tuple[str, ...] = ()
    source_subgoal_ids: tuple[str, ...] = ()
    multiline: bool = False

    def validate(self) -> None:
        _validate_id(self.field_id, "input_field.field_id")
        _validate_id(self.payload_ref, "input_field.payload_ref")
        if (
            not isinstance(self.field_label, str)
            or len(self.field_label) > 120
            or self.field_label != self.field_label.strip()
            or "\n" in self.field_label
            or "\r" in self.field_label
        ):
            raise TaskSemanticIRError("input_field.field_label 无效。")
        for field_name, values in (
            ("recipient_refs", self.recipient_refs),
            ("source_subgoal_ids", self.source_subgoal_ids),
        ):
            if len(values) != len(set(values)):
                raise TaskSemanticIRError(f"input_field.{field_name} 重复。")
            for value in values:
                _validate_id(value, f"input_field.{field_name}")
        if not isinstance(self.multiline, bool):
            raise TaskSemanticIRError("input_field.multiline 必须是布尔值。")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "field_id": self.field_id,
            "payload_ref": self.payload_ref,
            "field_label": self.field_label,
            "recipient_refs": list(self.recipient_refs),
            "source_subgoal_ids": list(self.source_subgoal_ids),
            "multiline": self.multiline,
        }


@dataclass(frozen=True)
class TaskSemanticIR:
    task_id: str
    device_id: str
    revision: int
    raw_goal: str
    surfaces: tuple[SurfaceRef, ...]
    entities: tuple[SemanticEntity, ...]
    effects: tuple[EffectIntent, ...]
    critical_bindings: tuple[CriticalBinding, ...] = ()
    constraints: tuple[ConstraintIntent, ...] = ()
    desired_states: tuple[DesiredState, ...] = ()
    evidence_requirements: tuple[EvidenceRequirement, ...] = ()
    subgoals: tuple[SemanticSubgoal, ...] = ()
    input_fields: tuple[InputFieldIntent, ...] = ()
    protocol_version: str = TASK_SEMANTIC_IR_PROTOCOL

    def validate(self) -> None:
        if self.protocol_version != TASK_SEMANTIC_IR_PROTOCOL:
            raise TaskSemanticIRError("TaskSemanticIR protocol_version 无效。")
        _validate_external_id(self.task_id, "semantic_ir.task_id")
        _validate_external_id(self.device_id, "semantic_ir.device_id")
        if isinstance(self.revision, bool) or not isinstance(self.revision, int) or self.revision < 1:
            raise TaskSemanticIRError("semantic_ir.revision 必须是正整数。")
        _required_text(self.raw_goal, "semantic_ir.raw_goal", max_length=10000)

        def unique(items: tuple[Any, ...], field_name: str, key: str) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for item in items:
                value = str(getattr(item, key))
                if value in result:
                    raise TaskSemanticIRError(f"{field_name} ID 重复：{value}")
                result[value] = item
            return result

        surfaces = unique(self.surfaces, "surface", "surface_id")
        entities = unique(self.entities, "entity", "entity_id")
        effects = unique(self.effects, "effect", "effect_id")
        bindings = unique(self.critical_bindings, "critical_binding", "binding_id")
        constraints = unique(self.constraints, "constraint", "constraint_id")
        states = unique(self.desired_states, "desired_state", "state_id")
        requirements = unique(
            self.evidence_requirements,
            "evidence_requirement",
            "requirement_id",
        )
        subgoals = unique(self.subgoals, "semantic_subgoal", "subgoal_id")
        input_fields = unique(self.input_fields, "input_field", "field_id")
        for surface in surfaces.values():
            surface.validate()
        for entity in entities.values():
            entity.validate(self.raw_goal)
        for effect in effects.values():
            effect.validate()
            unknown_refs = set(effect.target_refs).union(effect.payload_refs) - set(entities)
            if unknown_refs:
                raise TaskSemanticIRError(
                    f"effect.{effect.effect_id} 引用未知实体："
                    + ", ".join(sorted(unknown_refs))
                )
        for binding in bindings.values():
            binding.validate()
            if binding.effect_id not in effects or binding.entity_ref not in entities:
                raise TaskSemanticIRError(
                    f"critical_binding.{binding.binding_id} 引用未知 effect/entity。"
                )
            effect = effects[binding.effect_id]
            expected_refs = (
                effect.target_refs
                if binding.binding_kind == "effect_target_equals"
                else effect.payload_refs
            )
            if binding.entity_ref not in expected_refs:
                raise TaskSemanticIRError(
                    f"critical_binding.{binding.binding_id} 未绑定 effect 对应引用。"
                )
        known_subjects = set(entities).union(effects).union(surfaces)
        for constraint in constraints.values():
            constraint.validate()
            unknown_refs = set(constraint.subject_refs).union(
                constraint.object_refs
            ) - known_subjects
            if unknown_refs:
                raise TaskSemanticIRError(
                    f"constraint.{constraint.constraint_id} 引用未知对象："
                    + ", ".join(sorted(unknown_refs))
                )
        for state in states.values():
            state.validate()
            if state.subject_ref not in known_subjects:
                raise TaskSemanticIRError(
                    f"desired_state.{state.state_id} 引用未知 subject。"
                )
            if state.source_subgoal_id and state.source_subgoal_id not in subgoals:
                raise TaskSemanticIRError(
                    f"desired_state.{state.state_id} 引用未知 subgoal。"
                )
        for requirement in requirements.values():
            requirement.validate()
            if requirement.desired_state_ref not in states:
                raise TaskSemanticIRError(
                    f"evidence_requirement.{requirement.requirement_id} 引用未知 state。"
                )
        for subgoal in subgoals.values():
            subgoal.validate()
            if subgoal.surface_ref not in surfaces:
                raise TaskSemanticIRError(
                    f"semantic_subgoal.{subgoal.subgoal_id} 引用未知 surface。"
                )
            for field_name, refs, known in (
                ("depends_on", subgoal.depends_on, subgoals),
                ("constraint_refs", subgoal.constraint_refs, constraints),
                ("entity_refs", subgoal.entity_refs, entities),
                ("desired_state_refs", subgoal.desired_state_refs, states),
                ("effect_refs", subgoal.effect_refs, effects),
            ):
                unknown = set(refs) - set(known)
                if unknown:
                    raise TaskSemanticIRError(
                        f"semantic_subgoal.{subgoal.subgoal_id}.{field_name} 引用未知 ID："
                        + ", ".join(sorted(unknown))
                    )
        for input_field in input_fields.values():
            input_field.validate()
            if input_field.payload_ref not in entities:
                raise TaskSemanticIRError(
                    f"input_field.{input_field.field_id} 引用未知 payload。"
                )
            payload = entities[input_field.payload_ref]
            if payload.role != "input_text" or payload.entity_type != "text":
                raise TaskSemanticIRError(
                    f"input_field.{input_field.field_id} payload 必须是 typed input_text。"
                )
            if set(input_field.recipient_refs) - set(entities):
                raise TaskSemanticIRError(
                    f"input_field.{input_field.field_id} 引用未知 recipient。"
                )
            if set(input_field.source_subgoal_ids) - set(subgoals):
                raise TaskSemanticIRError(
                    f"input_field.{input_field.field_id} 引用未知 subgoal。"
                )
        for subgoal in subgoals.values():
            requires_typed_input = any(
                constraints[constraint_ref].kind == "required_action"
                and constraints[constraint_ref].value == "input_verified_text"
                for constraint_ref in subgoal.constraint_refs
                if constraint_ref in constraints
            )
            if requires_typed_input and not any(
                subgoal.subgoal_id in input_field.source_subgoal_ids
                for input_field in input_fields.values()
            ):
                raise TaskSemanticIRError(
                    "typed input action 未绑定 InputFieldIntent："
                    f"{subgoal.subgoal_id}"
                )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "protocol_version": self.protocol_version,
            "task_id": self.task_id,
            "device_id": self.device_id,
            "revision": self.revision,
            "raw_goal_sha256": hashlib.sha256(self.raw_goal.encode("utf-8")).hexdigest(),
            "surfaces": [item.to_dict() for item in self.surfaces],
            "entities": [item.to_dict() for item in self.entities],
            "effects": [item.to_dict() for item in self.effects],
            "critical_bindings": [item.to_dict() for item in self.critical_bindings],
            "constraints": [item.to_dict() for item in self.constraints],
            "desired_states": [item.to_dict() for item in self.desired_states],
            "evidence_requirements": [
                item.to_dict() for item in self.evidence_requirements
            ],
            "subgoals": [item.to_dict() for item in self.subgoals],
            "input_fields": [item.to_dict() for item in self.input_fields],
        }

    @property
    def semantic_digest(self) -> str:
        value = self.to_dict()
        authoritative_constraint_ids = {
            item.constraint_id for item in self.constraints if item.authoritative
        }
        value["constraints"] = [
            item
            for item in value["constraints"]
            if item["constraint_id"] in authoritative_constraint_ids
        ]
        value["subgoals"] = [
            {
                **item,
                "constraint_refs": [
                    ref
                    for ref in item["constraint_refs"]
                    if ref in authoritative_constraint_ids
                ],
            }
            for item in value["subgoals"]
        ]
        return _canonical_digest(value)


@dataclass(frozen=True)
class RiskDecision:
    effect_id: str
    policy: str
    matched_rule: str
    policy_id: str
    policy_version: int

    def validate(self) -> None:
        _validate_id(self.effect_id, "risk_decision.effect_id")
        if self.policy not in RISK_POLICIES:
            raise TaskSemanticIRError(f"risk_decision.policy 无效：{self.policy}")
        _required_text(self.matched_rule, "risk_decision.matched_rule", max_length=200)
        _validate_id(self.policy_id, "risk_decision.policy_id")
        if (
            isinstance(self.policy_version, bool)
            or not isinstance(self.policy_version, int)
            or self.policy_version < 1
        ):
            raise TaskSemanticIRError("risk_decision.policy_version 必须是正整数。")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "effect_id": self.effect_id,
            "policy": self.policy,
            "matched_rule": self.matched_rule,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
        }


@dataclass(frozen=True)
class LocalRiskPolicyConfig:
    policy_id: str = "default_low_friction"
    version: int = 2
    confirmation_effect_kinds: frozenset[str] = DEFAULT_CONFIRMATION_EFFECT_KINDS
    overrides: tuple[tuple[str, str], ...] = ()
    protocol_version: str = RISK_POLICY_PROTOCOL

    def validate(self) -> None:
        if self.protocol_version != RISK_POLICY_PROTOCOL:
            raise TaskSemanticIRError("风险策略协议版本无效。")
        _validate_id(self.policy_id, "risk_policy.policy_id")
        if isinstance(self.version, bool) or not isinstance(self.version, int) or self.version < 1:
            raise TaskSemanticIRError("risk_policy.version 必须是正整数。")
        for kind in self.confirmation_effect_kinds:
            _validate_id(kind, "risk_policy.confirmation_effect_kinds")
        seen: set[str] = set()
        for kind, policy in self.overrides:
            _validate_id(kind, "risk_policy.overrides.kind")
            if kind in seen:
                raise TaskSemanticIRError(f"风险策略 override 重复：{kind}")
            seen.add(kind)
            if policy not in RISK_POLICIES:
                raise TaskSemanticIRError(f"风险策略 override 无效：{policy}")

    def decide(self, effect: EffectIntent) -> RiskDecision:
        self.validate()
        effect.validate()
        override_map = dict(self.overrides)
        if effect.kind in override_map:
            policy = override_map[effect.kind]
            matched_rule = f"override:{effect.kind}"
        elif effect.kind in self.confirmation_effect_kinds:
            policy = CONFIRMATION_REQUIRED
            matched_rule = f"confirmation_kind:{effect.kind}"
        else:
            policy = AUTOMATIC
            matched_rule = "default_automatic"
        return RiskDecision(
            effect_id=effect.effect_id,
            policy=policy,
            matched_rule=matched_rule,
            policy_id=self.policy_id,
            policy_version=self.version,
        )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "protocol_version": self.protocol_version,
            "policy_id": self.policy_id,
            "version": self.version,
            "confirmation_effect_kinds": sorted(self.confirmation_effect_kinds),
            "overrides": [
                {"effect_kind": kind, "policy": policy}
                for kind, policy in self.overrides
            ],
        }


@dataclass(frozen=True)
class EffectEntityPreview:
    entity_ref: str
    role: str
    entity_type: str
    value: Any

    def validate(self) -> None:
        _validate_id(self.entity_ref, "effect_preview.entity_ref")
        _validate_id(self.role, "effect_preview.role")
        _validate_id(self.entity_type, "effect_preview.entity_type")
        _json_value(self.value, "effect_preview.value")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "entity_ref": self.entity_ref,
            "role": self.role,
            "entity_type": self.entity_type,
            "value": _json_value(self.value, "effect_preview.value"),
        }


@dataclass(frozen=True)
class EffectPreview:
    task_id: str
    device_id: str
    revision: int
    effect_id: str
    effect_kind: str
    targets: tuple[EffectEntityPreview, ...]
    payloads: tuple[EffectEntityPreview, ...]
    policy: str
    policy_id: str
    policy_version: int
    expected_result_texts: tuple[str, ...] = ()
    protocol_version: str = EFFECT_PREVIEW_PROTOCOL

    def validate(self) -> None:
        if self.protocol_version != EFFECT_PREVIEW_PROTOCOL:
            raise TaskSemanticIRError("EffectPreview protocol_version 无效。")
        _required_text(self.task_id, "effect_preview.task_id", max_length=128)
        _required_text(self.device_id, "effect_preview.device_id", max_length=128)
        if isinstance(self.revision, bool) or not isinstance(self.revision, int) or self.revision < 1:
            raise TaskSemanticIRError("effect_preview.revision 必须是正整数。")
        _validate_id(self.effect_id, "effect_preview.effect_id")
        _validate_id(self.effect_kind, "effect_preview.effect_kind")
        if self.policy not in RISK_POLICIES:
            raise TaskSemanticIRError("effect_preview.policy 无效。")
        _validate_id(self.policy_id, "effect_preview.policy_id")
        if isinstance(self.policy_version, bool) or not isinstance(self.policy_version, int) or self.policy_version < 1:
            raise TaskSemanticIRError("effect_preview.policy_version 必须是正整数。")
        refs: list[str] = []
        for item in (*self.targets, *self.payloads):
            item.validate()
            refs.append(item.entity_ref)
        if len(refs) != len(set(refs)):
            raise TaskSemanticIRError("EffectPreview target/payload 引用重复。")
        for item in self.expected_result_texts:
            _required_text(item, "effect_preview.expected_result", max_length=1000)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "protocol_version": self.protocol_version,
            "task_id": self.task_id,
            "device_id": self.device_id,
            "revision": self.revision,
            "effect_id": self.effect_id,
            "effect_kind": self.effect_kind,
            "targets": [item.to_dict() for item in self.targets],
            "payloads": [item.to_dict() for item in self.payloads],
            "policy": self.policy,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "expected_result_texts": list(self.expected_result_texts),
        }

    @property
    def preview_digest(self) -> str:
        return _canonical_digest(self.to_dict())


def local_risk_policy_from_dict(payload: Mapping[str, Any]) -> LocalRiskPolicyConfig:
    if not isinstance(payload, Mapping):
        raise TaskSemanticIRError("风险策略配置必须是 JSON 对象。")
    expected_keys = {
        "protocol_version",
        "policy_id",
        "version",
        "confirmation_effect_kinds",
        "overrides",
    }
    actual_keys = set(payload)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise TaskSemanticIRError(
            f"风险策略配置字段不匹配：missing={missing}, extra={extra}"
        )
    raw_confirmation = payload["confirmation_effect_kinds"]
    raw_overrides = payload["overrides"]
    if not isinstance(raw_confirmation, list) or not isinstance(raw_overrides, list):
        raise TaskSemanticIRError("风险策略种类和 overrides 必须是数组。")
    overrides: list[tuple[str, str]] = []
    for index, item in enumerate(raw_overrides):
        if not isinstance(item, Mapping) or set(item) != {"effect_kind", "policy"}:
            raise TaskSemanticIRError(f"风险策略 overrides[{index}] 字段无效。")
        overrides.append((str(item["effect_kind"]), str(item["policy"])))
    config = LocalRiskPolicyConfig(
        policy_id=str(payload["policy_id"]),
        version=payload["version"],
        confirmation_effect_kinds=frozenset(
            str(item) for item in raw_confirmation
        ),
        overrides=tuple(overrides),
        protocol_version=str(payload["protocol_version"]),
    )
    config.validate()
    return config


def load_local_risk_policy(path: str | Path) -> LocalRiskPolicyConfig:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskSemanticIRError(f"无法读取风险策略配置：{exc}") from exc
    return local_risk_policy_from_dict(payload)


@dataclass(frozen=True)
class SemanticCompilationReport:
    semantic_ir: TaskSemanticIR
    risk_policy: LocalRiskPolicyConfig
    risk_decisions: tuple[RiskDecision, ...]
    warnings: tuple[str, ...] = ()
    authoritative: bool = False
    execution_allowed: bool = False
    protocol_version: str = COMPILATION_REPORT_PROTOCOL

    def validate(self) -> None:
        if self.protocol_version != COMPILATION_REPORT_PROTOCOL:
            raise TaskSemanticIRError("语义编译报告协议版本无效。")
        if self.authoritative is not False or self.execution_allowed is not False:
            raise TaskSemanticIRError("影子报告不得携带执行权限。")
        self.semantic_ir.validate()
        self.risk_policy.validate()
        decisions: dict[str, RiskDecision] = {}
        for decision in self.risk_decisions:
            decision.validate()
            if decision.effect_id in decisions:
                raise TaskSemanticIRError(
                    f"影子风险决定 effect_id 重复：{decision.effect_id}"
                )
            decisions[decision.effect_id] = decision
        expected = {item.effect_id for item in self.semantic_ir.effects}
        if set(decisions) != expected:
            raise TaskSemanticIRError("影子风险决定必须逐项覆盖全部 EffectIntent。")
        for warning in self.warnings:
            _required_text(warning, "semantic_compilation.warning", max_length=500)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "protocol_version": self.protocol_version,
            "authoritative": False,
            "execution_allowed": False,
            "semantic_ir": self.semantic_ir.to_dict(),
            "semantic_digest": self.semantic_ir.semantic_digest,
            "risk_policy": self.risk_policy.to_dict(),
            "risk_decisions": [item.to_dict() for item in self.risk_decisions],
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class RiskPolicyTrace:
    effect_id: str
    effect_kind: str
    formal_policy: str
    allowed: bool
    reason: str

    def validate(self) -> None:
        _validate_id(self.effect_id, "risk_policy_trace.effect_id")
        _validate_id(self.effect_kind, "risk_policy_trace.effect_kind")
        if self.formal_policy not in RISK_POLICIES:
            raise TaskSemanticIRError("risk_policy_trace.formal_policy 无效。")
        if not isinstance(self.allowed, bool):
            raise TaskSemanticIRError("risk_policy_trace.allowed 必须是布尔值。")
        _required_text(self.reason, "risk_policy_trace.reason", max_length=300)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "effect_id": self.effect_id,
            "effect_kind": self.effect_kind,
            "formal_policy": self.formal_policy,
            "allowed": self.allowed,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class SemanticRiskAuthorityReport:
    """Formal authority for field roles and confirmation policy only.

    It cannot grant a visual action or physical execution.  Its only authority
    is to bind typed effects to the local, versioned confirmation policy.
    """

    semantic_ir: TaskSemanticIR
    risk_policy: LocalRiskPolicyConfig
    risk_decisions: tuple[RiskDecision, ...]
    policy_traces: tuple[RiskPolicyTrace, ...]
    effect_previews: tuple[EffectPreview, ...]
    source_graph_digest: str
    authoritative_scope: str = "semantic_task_and_risk"
    physical_execution_allowed: bool = False
    protocol_version: str = AUTHORITY_REPORT_PROTOCOL

    def validate(self) -> None:
        if self.protocol_version != AUTHORITY_REPORT_PROTOCOL:
            raise TaskSemanticIRError("正式语义风险报告协议版本无效。")
        if self.authoritative_scope != "semantic_task_and_risk":
            raise TaskSemanticIRError("正式语义风险报告权威范围无效。")
        if self.physical_execution_allowed is not False:
            raise TaskSemanticIRError("语义风险权威不得授予物理执行权限。")
        if not re.fullmatch(r"[0-9a-f]{64}", self.source_graph_digest):
            raise TaskSemanticIRError("source_graph_digest 必须是 SHA-256。")
        self.semantic_ir.validate()
        self.risk_policy.validate()
        decisions = {item.effect_id: item for item in self.risk_decisions}
        if len(decisions) != len(self.risk_decisions):
            raise TaskSemanticIRError("正式风险决定 effect_id 重复。")
        expected = {item.effect_id for item in self.semantic_ir.effects}
        if set(decisions) != expected:
            raise TaskSemanticIRError("正式风险决定必须覆盖全部 EffectIntent。")
        for item in self.risk_decisions:
            item.validate()
        previews = {item.effect_id: item for item in self.effect_previews}
        if len(previews) != len(self.effect_previews) or set(previews) != expected:
            raise TaskSemanticIRError("EffectPreview 必须逐项覆盖全部 EffectIntent。")
        for effect in self.semantic_ir.effects:
            preview = previews[effect.effect_id]
            preview.validate()
            decision = decisions[effect.effect_id]
            if (
                preview.task_id != self.semantic_ir.task_id
                or preview.device_id != self.semantic_ir.device_id
                or preview.revision != self.semantic_ir.revision
                or preview.effect_kind != effect.kind
                or preview.policy != decision.policy
                or preview.policy_id != decision.policy_id
                or preview.policy_version != decision.policy_version
            ):
                raise TaskSemanticIRError("EffectPreview 与 effect/risk authority 不一致。")
        unsupported = [
            item.effect_id
            for item in self.semantic_ir.effects
            if item.kind == "generic_effect"
        ]
        if unsupported:
            raise TaskSemanticIRError(
                "未知外部效果没有可执行语义类型：" + ", ".join(unsupported)
            )
        traces = {item.effect_id: item for item in self.policy_traces}
        if len(traces) != len(self.policy_traces) or set(traces) != expected:
            raise TaskSemanticIRError("正式策略轨迹必须逐项覆盖全部 EffectIntent。")
        for item in self.policy_traces:
            item.validate()
            if not item.allowed:
                raise TaskSemanticIRError(
                    f"正式策略存在未允许效果：{item.effect_id}"
                )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "protocol_version": self.protocol_version,
            "authoritative_scope": self.authoritative_scope,
            "physical_execution_allowed": False,
            "source_graph_digest": self.source_graph_digest,
            "semantic_ir": self.semantic_ir.to_dict(),
            "semantic_digest": self.semantic_ir.semantic_digest,
            "risk_policy": self.risk_policy.to_dict(),
            "risk_decisions": [item.to_dict() for item in self.risk_decisions],
            "effect_previews": [
                {**item.to_dict(), "preview_digest": item.preview_digest}
                for item in self.effect_previews
            ],
            "policy_trace_protocol": POLICY_TRACE_PROTOCOL,
            "policy_traces": [item.to_dict() for item in self.policy_traces],
        }


def _source_span(raw_goal: str, value: Any) -> SourceSpan | None:
    if not isinstance(value, str) or not value:
        return None
    start = raw_goal.find(value)
    if start < 0:
        return None
    return SourceSpan(start=start, end=start + len(value))


def _entity_refs_for_roles(
    entities: tuple[SemanticEntity, ...],
    roles: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(
        item.entity_id
        for role in roles
        for item in entities
        if item.role == role
    )


def _runtime_graph_digest(graph: Any) -> str:
    goal = getattr(graph, "goal", None)
    payload = {
        "task_id": str(getattr(graph, "task_id", "")),
        "device_id": str(getattr(graph, "device_id", "")),
        "revision": getattr(graph, "revision", None),
        "raw_goal": str(getattr(graph, "raw_user_goal", "")),
        "goal": {
            "objective": str(getattr(goal, "objective", "")),
            "target_apps": [
                {
                    "app_id": str(getattr(item, "app_id", "")),
                    "app_name": str(getattr(item, "app_name", "")),
                }
                for item in tuple(getattr(goal, "target_apps", ()) or ())
            ],
            "entities": dict(getattr(goal, "entities", {}) or {}),
        },
        "effect_intents": [
            {
                "effect_id": str(getattr(item, "risk_id", "")),
                "kind": str(getattr(item, "effect_kind", "")),
                "target_entity_roles": list(
                    getattr(item, "target_roles", ()) or ()
                ),
                "payload_entity_roles": list(
                    getattr(item, "payload_roles", ()) or ()
                ),
                "source_subgoal_ids": list(
                    getattr(item, "subgoal_ids", ()) or ()
                ),
                "expected_results": list(
                    getattr(item, "expected_result_texts", ()) or ()
                ),
            }
            for item in tuple(getattr(graph, "risk_actions", ()) or ())
        ],
        "subgoals": [
            {
                "subgoal_id": str(getattr(item, "subgoal_id", "")),
                "execution_class": str(getattr(item, "external_impact", "")),
                "effect_ids": list(getattr(item, "risk_action_ids", ()) or ()),
            }
            for item in tuple(getattr(graph, "subgoals", ()) or ())
        ],
    }
    return _canonical_digest(payload)


def compile_runtime_graph_semantics(
    graph: Any,
    *,
    risk_policy: LocalRiskPolicyConfig | None = None,
) -> SemanticCompilationReport:
    """Project the locally generated runtime graph into semantic IR."""

    task_id = str(getattr(graph, "task_id", "")).strip()
    device_id = str(getattr(graph, "device_id", "")).strip()
    revision = getattr(graph, "revision", 0)
    raw_goal = str(getattr(graph, "raw_user_goal", "") or "").strip()
    goal = getattr(graph, "goal", None)
    if not raw_goal:
        raw_goal = str(getattr(goal, "objective", "") or "").strip()

    target_apps = tuple(getattr(goal, "target_apps", ()) or ())
    surfaces: list[SurfaceRef] = []
    raw_goal_folded = raw_goal.casefold()
    launcher_terms = ("主桌面", "桌面", "主页", "home screen", "launcher")
    current_surface_terms = (
        "当前",
        "当前页面",
        "当前界面",
        "当前应用",
        "当前前台",
        "眼前",
        "current page",
        "current screen",
        "current app",
        "current",
        "foreground",
    )
    raw_uses_current_surface = any(
        term in raw_goal_folded for term in current_surface_terms
    )
    needs_launcher = len(target_apps) > 1 or any(
        term in raw_goal_folded for term in launcher_terms
    )
    if needs_launcher:
        surfaces.append(SurfaceRef(surface_id="surface_launcher", kind="launcher"))
    for app in target_apps:
        app_id = _slug(getattr(app, "app_id", ""), fallback="app")
        surfaces.append(
            SurfaceRef(
                surface_id=f"surface_{app_id}",
                kind="app",
                app_id=app_id,
                app_name=str(getattr(app, "app_name", "") or app_id),
            )
        )
    if len(target_apps) > 1 or raw_uses_current_surface:
        surfaces.append(
            SurfaceRef(
                surface_id="surface_current",
                kind="current_surface",
            )
        )

    raw_entities = getattr(goal, "entities", {}) or {}
    if not isinstance(raw_entities, Mapping):
        raise TaskSemanticIRError("goal.entities 必须是映射。")
    entities: list[SemanticEntity] = []
    input_field_id_by_entity: dict[str, str] = {}
    input_field_label_by_entity: dict[str, str] = {}
    raw_recipients = raw_entities.get("recipients")
    if isinstance(raw_recipients, list):
        for index, value in enumerate(raw_recipients, 1):
            entity_id = f"entity_recipient_{index}"
            entities.append(
                SemanticEntity(
                    entity_id=entity_id,
                    entity_type="party",
                    role="recipient",
                    value=_json_value(value, f"entities.recipients[{index - 1}]"),
                    source_span=_source_span(raw_goal, value),
                    authority=(
                        "user_literal"
                        if _source_span(raw_goal, value) is not None
                        else "planner_context"
                    ),
                )
            )
    raw_input_fields = raw_entities.get("input_fields")
    if isinstance(raw_input_fields, list):
        for index, spec in enumerate(raw_input_fields, 1):
            if not isinstance(spec, Mapping):
                continue
            value = spec.get("text")
            field_name = _slug(spec.get("field_id"), fallback=f"field_{index}")
            entity_id = f"entity_input_text_{field_name}"
            entities.append(
                SemanticEntity(
                    entity_id=entity_id,
                    entity_type="text",
                    role="input_text",
                    value=_json_value(
                        value,
                        f"entities.input_fields[{index - 1}].text",
                    ),
                    source_span=_source_span(raw_goal, value),
                    authority=(
                        "user_literal"
                        if _source_span(raw_goal, value) is not None
                        else "planner_context"
                    ),
                )
            )
            input_field_id_by_entity[entity_id] = field_name
            field_label = spec.get("field_label")
            input_field_label_by_entity[entity_id] = (
                field_label if isinstance(field_label, str) else ""
            )
    for index, key in enumerate(sorted(raw_entities, key=lambda item: str(item))):
        if key in {"recipients", "input_fields"}:
            continue
        role = _slug(key, fallback=f"role_{index + 1}")
        value = raw_entities[key]
        span = _source_span(raw_goal, value)
        entities.append(
            SemanticEntity(
                entity_id=f"entity_{role}_{index + 1}",
                entity_type=_ENTITY_TYPE_BY_ROLE.get(role, "opaque"),
                role=role,
                value=_json_value(value, f"entities.{role}"),
                source_span=span,
                authority="user_literal" if span is not None else "planner_context",
            )
        )

    if not surfaces:
        surface_entity = next(
            (item for item in entities if item.role == "target_surface"),
            None,
        )
        surface_kind = str(surface_entity.value) if surface_entity is not None else "current_surface"
        if surface_kind not in SURFACE_KINDS:
            surface_kind = "current_surface"
        surfaces.append(
            SurfaceRef(
                surface_id=f"surface_{_slug(surface_kind, fallback='current')}",
                kind=surface_kind,
            )
        )

    subgoals = {
        str(getattr(item, "subgoal_id", "")): item
        for item in tuple(getattr(graph, "subgoals", ()) or ())
        if str(getattr(item, "subgoal_id", ""))
    }
    effects: list[EffectIntent] = []
    represented_subgoals: set[str] = set()
    warnings: list[str] = []
    for index, risk in enumerate(tuple(getattr(graph, "risk_actions", ()) or ())):
        runtime_type = str(getattr(risk, "risk_type", "") or "")
        declared_kind = str(getattr(risk, "effect_kind", "") or "")
        effect_kind = declared_kind or _RUNTIME_RISK_EFFECT_KIND.get(
            runtime_type,
            "generic_effect",
        )
        risk_id = _slug(getattr(risk, "risk_id", ""), fallback=f"effect_{index + 1}")
        source_subgoal_ids = tuple(
            _slug(value, fallback="subgoal")
            for value in tuple(getattr(risk, "subgoal_ids", ()) or ())
        )
        represented_subgoals.update(source_subgoal_ids)
        expected_results = [
            str(value).strip()
            for value in tuple(
                getattr(risk, "expected_result_texts", ()) or ()
            )
            if str(value).strip()
        ]
        if not expected_results:
            for subgoal_id in source_subgoal_ids:
                subgoal = subgoals.get(subgoal_id)
                if subgoal is None:
                    continue
                expected_results.extend(
                    str(value).strip()
                    for value in tuple(
                        getattr(subgoal, "completion_conditions", ()) or ()
                    )
                    if str(value).strip()
                )
        declared_target_roles = tuple(
            str(value)
            for value in tuple(getattr(risk, "target_roles", ()) or ())
        )
        declared_payload_roles = tuple(
            str(value)
            for value in tuple(getattr(risk, "payload_roles", ()) or ())
        )
        target_refs = _entity_refs_for_roles(
            tuple(entities),
            declared_target_roles
            or _TARGET_ROLES_BY_EFFECT.get(effect_kind, ("target",)),
        )
        payload_refs = _entity_refs_for_roles(
            tuple(entities),
            declared_payload_roles
            or _PAYLOAD_ROLES_BY_EFFECT.get(effect_kind, ("input_text", "value")),
        )
        if not target_refs:
            warnings.append(f"effect_{risk_id}:missing_typed_target")
        if effect_kind == "generic_effect" and not expected_results:
            warnings.append(f"effect_{risk_id}:missing_expected_result")
        effects.append(
            EffectIntent(
                effect_id=f"effect_{risk_id}",
                kind=effect_kind,
                target_refs=target_refs,
                payload_refs=payload_refs,
                source_subgoal_ids=source_subgoal_ids,
                expected_result_texts=tuple(dict.fromkeys(expected_results)),
                attributes={
                    "runtime_effect_id": str(getattr(risk, "risk_id", "") or ""),
                    "planner_declared_typed_effect": bool(declared_kind),
                },
            )
        )

    bindings: list[CriticalBinding] = []
    for effect in effects:
        for index, entity_ref in enumerate(effect.target_refs):
            bindings.append(
                CriticalBinding(
                    binding_id=_slug(
                        f"binding_{effect.effect_id}_target_{index + 1}",
                        fallback="binding_target",
                    ),
                    effect_id=effect.effect_id,
                    binding_kind="effect_target_equals",
                    entity_ref=entity_ref,
                )
            )
        for index, entity_ref in enumerate(effect.payload_refs):
            bindings.append(
                CriticalBinding(
                    binding_id=_slug(
                        f"binding_{effect.effect_id}_payload_{index + 1}",
                        fallback="binding_payload",
                    ),
                    effect_id=effect.effect_id,
                    binding_kind="effect_payload_equals",
                    entity_ref=entity_ref,
                )
            )

    entity_by_role: dict[str, list[SemanticEntity]] = {}
    for entity in entities:
        entity_by_role.setdefault(entity.role, []).append(entity)

    typed_constraints: list[ConstraintIntent] = []
    entity_constraint_ids: list[str] = []
    for index, entity in enumerate(entities, 1):
        if entity.authority != "user_literal":
            continue
        constraint_id = f"constraint_exact_{index}"
        entity_constraint_ids.append(constraint_id)
        typed_constraints.append(
            ConstraintIntent(
                constraint_id=constraint_id,
                kind="exact_entity",
                subject_refs=(entity.entity_id,),
                value=entity.value,
                source_text=str(entity.value) if isinstance(entity.value, str) else "",
                authoritative=True,
            )
        )

    planner_constraint_ids_by_subgoal: dict[str, list[str]] = {}
    action_constraint_ids_by_subgoal: dict[str, list[str]] = {}
    all_planner_constraints: list[tuple[str, str]] = [
        ("", str(item).strip())
        for item in tuple(getattr(graph, "constraints", ()) or ())
        if str(item).strip()
    ]
    for subgoal_id, subgoal in subgoals.items():
        all_planner_constraints.extend(
            (subgoal_id, str(item).strip())
            for item in tuple(getattr(subgoal, "constraints", ()) or ())
            if str(item).strip()
        )
    for index, (subgoal_id, text) in enumerate(all_planner_constraints, 1):
        constraint_id = f"constraint_context_{index}"
        typed_constraints.append(
            ConstraintIntent(
                constraint_id=constraint_id,
                kind="planner_context",
                value=text,
                source_text=text,
                authoritative=False,
            )
        )
        planner_constraint_ids_by_subgoal.setdefault(subgoal_id, []).append(
            constraint_id
        )

    action_patterns = (
        ("double_tap", re.compile(r"双击|double[ _-]?(?:tap|click)", re.I)),
        ("pinch", re.compile(r"捏合|双指|pinch|zoom", re.I)),
        ("press_enter", re.compile(r"回车|enter(?:\s+key)?", re.I)),
        (
            "clear_verified_text",
            re.compile(
                r"(?:清空|清除|置空).{0,8}(?:输入框|文本|文字|内容|草稿)|"
                r"(?:clear|empty).{0,8}(?:input|text|draft)",
                re.I,
            ),
        ),
        (
            "dismiss_overlay",
            re.compile(r"关闭.{0,6}(?:弹窗|弹层|对话框)|dismiss[ _-]?overlay", re.I),
        ),
        (
            "reveal_system_navigation",
            re.compile(r"(?:唤出|显示).{0,6}(?:系统)?导航栏|reveal[ _-]?navigation", re.I),
        ),
        ("long_press", re.compile(r"长按|long[ _-]?press", re.I)),
        ("drag", re.compile(r"拖动|拖拽|drag", re.I)),
        ("swipe", re.compile(r"滑动|上划|下划|左划|右划|swipe", re.I)),
        (
            "input_verified_text",
            re.compile(
                r"输入(?!框|法|区域|状态|控件|字段|页面|界面|模式|键盘)|填写|键入|"
                r"\btype\b|\binput\b(?!\s*(?:field|box|area|control|state|"
                r"mode|method|page|screen|keyboard)\b)",
                re.I,
            ),
        ),
        ("home", re.compile(r"home\s*键|回到主页|回到主桌面", re.I)),
        (
            "back",
            re.compile(
                r"返回键|后退键|back\s*key|"
                r"(?:收起|隐藏|关闭).{0,6}(?:软?键盘|输入法)",
                re.I,
            ),
        ),
        ("hardware_key", re.compile(r"音量键|电源键|hardware\s*key", re.I)),
        ("tap_semantic", re.compile(r"点击|轻触|点按|tap|click", re.I)),
    )
    for subgoal_id, subgoal in subgoals.items():
        # A read-only node can describe an already completed action (for
        # example, "after input, verify no send").  It never owns a new
        # physical action, so legacy wording must not mint required_action.
        if str(getattr(subgoal, "external_impact", "") or "") == "read_only":
            continue
        objective = str(getattr(subgoal, "objective", "") or "")
        for action_kind, pattern in action_patterns:
            if not pattern.search(objective):
                continue
            constraint_id = f"constraint_action_{len(typed_constraints) + 1}"
            typed_constraints.append(
                ConstraintIntent(
                    constraint_id=constraint_id,
                    kind="required_action",
                    value=action_kind,
                    source_text=objective,
                    authoritative=True,
                )
            )
            action_constraint_ids_by_subgoal.setdefault(subgoal_id, []).append(
                constraint_id
            )

    surface_by_app_term: dict[str, str] = {}
    for surface in surfaces:
        if surface.kind == "app":
            surface_by_app_term[surface.app_id.casefold()] = surface.surface_id
            surface_by_app_term[surface.app_name.casefold()] = surface.surface_id

    def surface_for_subgoal(subgoal: Any) -> str:
        values = (
            str(getattr(subgoal, "objective", "") or ""),
            *tuple(str(item) for item in tuple(getattr(subgoal, "constraints", ()) or ())),
            *tuple(
                str(item)
                for item in tuple(
                    getattr(subgoal, "completion_conditions", ()) or ()
                )
            ),
        )
        normalized = " ".join(values).casefold()
        if any(term in normalized for term in launcher_terms):
            launcher = next(
                (item.surface_id for item in surfaces if item.kind == "launcher"),
                "",
            )
            if launcher:
                return launcher
        if any(term in normalized for term in current_surface_terms):
            current = next(
                (item.surface_id for item in surfaces if item.kind == "current_surface"),
                "",
            )
            if current:
                return current
        matches = {
            surface_id
            for term, surface_id in surface_by_app_term.items()
            if term and term in normalized
        }
        if len(matches) == 1:
            return next(iter(matches))
        if raw_uses_current_surface:
            current = next(
                (item.surface_id for item in surfaces if item.kind == "current_surface"),
                "",
            )
            if current:
                return current
        app_surfaces = [item.surface_id for item in surfaces if item.kind == "app"]
        if len(app_surfaces) == 1:
            return app_surfaces[0]
        current = next(
            (item.surface_id for item in surfaces if item.kind == "current_surface"),
            "",
        )
        if current:
            return current
        return surfaces[0].surface_id

    effect_refs_by_subgoal: dict[str, list[str]] = {}
    for effect in effects:
        for subgoal_id in effect.source_subgoal_ids:
            effect_refs_by_subgoal.setdefault(subgoal_id, []).append(effect.effect_id)

    desired_states: list[DesiredState] = []
    evidence_requirements: list[EvidenceRequirement] = []
    desired_by_subgoal: dict[str, list[str]] = {}

    def append_desired_state(
        *,
        description: str,
        source_subgoal_id: str,
        surface_ref: str,
    ) -> None:
        state_number = len(desired_states) + 1
        state_id = f"state_{state_number}"
        input_entities = tuple(entity_by_role.get("input_text", ()))
        effect_refs = effect_refs_by_subgoal.get(source_subgoal_id, [])
        compact_description = description.casefold()
        if len(effect_refs) == 1:
            subject_ref = effect_refs[0]
            receipt_only = any(
                marker in compact_description
                for marker in (
                    "动作已执行",
                    "操作已执行",
                    "效果已触发",
                    "已触发操作",
                    "action executed",
                    "effect applied",
                )
            )
            predicate = (
                "effect.applied" if receipt_only else "effect.result_visible"
            )
            value = True
            sources = (
                ("effect_receipt",)
                if receipt_only
                else ("visual_claim", "effect_receipt")
            )
        elif (
            len(input_entities) == 1
            and isinstance(input_entities[0].value, str)
            and input_entities[0].value
            and input_entities[0].value.casefold() in compact_description
        ):
            subject_ref = input_entities[0].entity_id
            predicate = "input.value_equals"
            value: Any = input_entities[0].value
            sources = ("visual_claim",)
        elif any(
            term and term in compact_description
            for term in surface_by_app_term
        ):
            subject_ref = surface_ref
            predicate = "surface.state_visible"
            value = description
            sources = ("visual_claim",)
        else:
            subject_ref = surface_ref
            predicate = "observation.matches_description"
            value = description
            sources = ("visual_claim", "controller_transition")
        desired_states.append(
            DesiredState(
                state_id=state_id,
                subject_ref=subject_ref,
                predicate=predicate,
                value=value,
                source_subgoal_id=source_subgoal_id,
            )
        )
        desired_by_subgoal.setdefault(source_subgoal_id, []).append(state_id)
        evidence_requirements.append(
            EvidenceRequirement(
                requirement_id=f"evidence_{state_number}",
                desired_state_ref=state_id,
                allowed_sources=tuple(sources),
            )
        )

    semantic_subgoals: list[SemanticSubgoal] = []
    for subgoal_id, subgoal in subgoals.items():
        surface_ref = surface_for_subgoal(subgoal)
        subgoal_text = " ".join(
            [
                str(getattr(subgoal, "objective", "") or ""),
                *(
                    str(item)
                    for item in tuple(
                        getattr(subgoal, "completion_conditions", ()) or ()
                    )
                ),
            ]
        ).casefold()
        entity_refs = tuple(
            entity.entity_id
            for entity in entities
            if isinstance(entity.value, str)
            and entity.value.strip()
            and entity.value.strip().casefold() in subgoal_text
        )
        for description in tuple(
            getattr(subgoal, "completion_conditions", ()) or ()
        ):
            if str(description).strip():
                append_desired_state(
                    description=str(description).strip(),
                    source_subgoal_id=subgoal_id,
                    surface_ref=surface_ref,
                )
        semantic_subgoals.append(
            SemanticSubgoal(
                subgoal_id=subgoal_id,
                surface_ref=surface_ref,
                status=str(getattr(subgoal, "status", "") or "pending"),
                external_impact=str(
                    getattr(subgoal, "external_impact", "") or "unknown"
                ),
                depends_on=tuple(
                    str(item)
                    for item in tuple(getattr(subgoal, "depends_on", ()) or ())
                ),
                constraint_refs=tuple(
                    dict.fromkeys(
                        [
                            *entity_constraint_ids,
                            *planner_constraint_ids_by_subgoal.get("", ()),
                            *planner_constraint_ids_by_subgoal.get(subgoal_id, ()),
                            *action_constraint_ids_by_subgoal.get(subgoal_id, ()),
                        ]
                    )
                ),
                entity_refs=entity_refs,
                desired_state_refs=tuple(desired_by_subgoal.get(subgoal_id, ())),
                effect_refs=tuple(effect_refs_by_subgoal.get(subgoal_id, ())),
            )
        )

    global_conditions = tuple(
        getattr(graph, "completion_conditions", ()) or ()
    )
    default_surface = surfaces[0].surface_id
    for condition in global_conditions:
        description = str(getattr(condition, "description", "") or "").strip()
        if description:
            append_desired_state(
                description=description,
                source_subgoal_id="",
                surface_ref=default_surface,
            )

    constraints_by_id = {
        item.constraint_id: item for item in typed_constraints
    }
    semantic_subgoal_by_id = {
        item.subgoal_id: item for item in semantic_subgoals
    }
    desired_by_id = {
        item.state_id: item for item in desired_states
    }
    recipient_refs = tuple(
        item.entity_id for item in entity_by_role.get("recipient", ())
    )
    input_action_subgoal_ids = tuple(
        subgoal.subgoal_id
        for subgoal in semantic_subgoals
        if any(
            constraints_by_id[constraint_ref].kind == "required_action"
            and constraints_by_id[constraint_ref].value == "input_verified_text"
            for constraint_ref in subgoal.constraint_refs
            if constraint_ref in constraints_by_id
        )
    )
    clear_action_subgoal_ids = tuple(
        subgoal.subgoal_id
        for subgoal in semantic_subgoals
        if any(
            constraints_by_id[constraint_ref].kind == "required_action"
            and constraints_by_id[constraint_ref].value == "clear_verified_text"
            for constraint_ref in subgoal.constraint_refs
            if constraint_ref in constraints_by_id
        )
    )
    input_entities = tuple(entity_by_role.get("input_text", ()))

    def clear_subgoals_for_input(entity_id: str) -> tuple[str, ...]:
        if len(input_entities) == 1:
            return clear_action_subgoal_ids
        field_labels = {
            item.entity_id: input_field_label_by_entity.get(item.entity_id, "").strip()
            for item in input_entities
        }
        selected: list[str] = []
        for subgoal_id in clear_action_subgoal_ids:
            source = subgoals.get(subgoal_id)
            if source is None:
                continue
            context = " ".join(
                [
                    str(getattr(source, "objective", "") or ""),
                    *tuple(
                        str(item)
                        for item in tuple(getattr(source, "constraints", ()) or ())
                    ),
                    *tuple(
                        str(item)
                        for item in tuple(
                            getattr(source, "completion_conditions", ()) or ()
                        )
                    ),
                ]
            ).casefold()
            matches = tuple(
                item_id
                for item_id, label in field_labels.items()
                if label and label.casefold() in context
            )
            if matches == (entity_id,):
                selected.append(subgoal_id)
        return tuple(selected)

    input_fields: list[InputFieldIntent] = []
    for index, payload in enumerate(input_entities, 1):
        typed_source_subgoal_ids = tuple(
            semantic_subgoal.subgoal_id
            for semantic_subgoal in semantic_subgoals
            if (
                payload.entity_id in semantic_subgoal.entity_refs
                or semantic_subgoal.subgoal_id in input_action_subgoal_ids
                or any(
                    desired_by_id[desired_ref].subject_ref == payload.entity_id
                    for desired_ref in semantic_subgoal.desired_state_refs
                    if desired_ref in desired_by_id
                )
            )
        )
        # With one canonical input field, the typed action itself is sufficient
        # to bind ownership.  Legacy prose may omit, quote differently, or even
        # contradict the literal; it is context only and cannot replace the
        # canonical payload.  Multiple fields still require an unambiguous
        # literal-to-subgoal projection until the transport exposes field refs.
        source_subgoal_ids = tuple(
            dict.fromkeys(
                [
                    *typed_source_subgoal_ids,
                    *(input_action_subgoal_ids if len(input_entities) == 1 else ()),
                    *clear_subgoals_for_input(payload.entity_id),
                ]
            )
        )
        input_fields.append(
            InputFieldIntent(
                field_id=input_field_id_by_entity.get(
                    payload.entity_id,
                    f"input_field_{index}",
                ),
                payload_ref=payload.entity_id,
                field_label=input_field_label_by_entity.get(payload.entity_id, ""),
                recipient_refs=recipient_refs,
                source_subgoal_ids=source_subgoal_ids,
                multiline=isinstance(payload.value, str)
                and ("\n" in payload.value or "\r" in payload.value),
            )
        )
    for input_field in input_fields:
        if not input_field.multiline:
            continue
        candidate_subgoal_ids = list(input_field.source_subgoal_ids)
        if not candidate_subgoal_ids:
            candidate_subgoal_ids = [
                item.subgoal_id
                for item in semantic_subgoals
                if any(
                    constraints_by_id[constraint_ref].kind == "required_action"
                    and constraints_by_id[constraint_ref].value
                    == "input_verified_text"
                    for constraint_ref in item.constraint_refs
                    if constraint_ref in constraints_by_id
                )
            ]
        if len(candidate_subgoal_ids) != 1:
            continue
        subgoal_id = candidate_subgoal_ids[0]
        current_subgoal = semantic_subgoal_by_id.get(subgoal_id)
        if current_subgoal is None or any(
            constraints_by_id[constraint_ref].kind == "required_action"
            and constraints_by_id[constraint_ref].value == "press_enter"
            for constraint_ref in current_subgoal.constraint_refs
            if constraint_ref in constraints_by_id
        ):
            continue
        constraint_id = f"constraint_action_{len(typed_constraints) + 1}"
        press_enter_constraint = ConstraintIntent(
            constraint_id=constraint_id,
            kind="required_action",
            value="press_enter",
            source_text="multiline typed input field",
            authoritative=True,
        )
        typed_constraints.append(press_enter_constraint)
        constraints_by_id[constraint_id] = press_enter_constraint
        semantic_subgoal_by_id[subgoal_id] = replace(
            current_subgoal,
            constraint_refs=(*current_subgoal.constraint_refs, constraint_id),
        )
    semantic_subgoals = [
        semantic_subgoal_by_id[item.subgoal_id] for item in semantic_subgoals
    ]
    semantic_ir = TaskSemanticIR(
        task_id=task_id,
        device_id=device_id,
        revision=revision,
        raw_goal=raw_goal,
        surfaces=tuple(surfaces),
        entities=tuple(entities),
        effects=tuple(effects),
        critical_bindings=tuple(bindings),
        constraints=tuple(typed_constraints),
        desired_states=tuple(desired_states),
        evidence_requirements=tuple(evidence_requirements),
        subgoals=tuple(semantic_subgoals),
        input_fields=tuple(input_fields),
    )
    semantic_ir.validate()
    policy = risk_policy or LocalRiskPolicyConfig()
    decisions = tuple(policy.decide(effect) for effect in semantic_ir.effects)
    warnings.insert(0, f"runtime_graph_digest:{_runtime_graph_digest(graph)}")
    report = SemanticCompilationReport(
        semantic_ir=semantic_ir,
        risk_policy=policy,
        risk_decisions=decisions,
        warnings=tuple(warnings),
    )
    report.validate()
    return report


def compile_formal_semantic_authority(
    graph: Any,
    *,
    risk_policy: LocalRiskPolicyConfig | None = None,
) -> SemanticRiskAuthorityReport:
    """Compile the sole formal field-role and confirmation authority.

    The runtime graph is already a deterministic projection of the strict
    typed planner transport.  Only ``EffectIntent.kind`` and the local policy
    decide confirmation; model-supplied legacy risk fields are not accepted.
    """

    if any(
        not str(getattr(item, "effect_kind", "") or "")
        for item in tuple(getattr(graph, "risk_actions", ()) or ())
    ):
        raise TaskSemanticIRError(
            "正式语义权威拒绝旧风险投影；必须由 typed effect_intents 创建新任务图。"
        )
    compilation = compile_runtime_graph_semantics(graph, risk_policy=risk_policy)
    decisions = {item.effect_id: item for item in compilation.risk_decisions}
    entity_by_id = {
        item.entity_id: item for item in compilation.semantic_ir.entities
    }
    traces: list[RiskPolicyTrace] = []
    for effect in compilation.semantic_ir.effects:
        decision = decisions[effect.effect_id]
        formal_required = decision.policy == CONFIRMATION_REQUIRED
        allowed = effect.kind != "generic_effect"
        reason = "typed_effect_local_policy"
        traces.append(
            RiskPolicyTrace(
                effect_id=effect.effect_id,
                effect_kind=effect.kind,
                formal_policy=decision.policy,
                allowed=allowed,
                reason=reason,
            )
        )
    previews = tuple(
        EffectPreview(
            task_id=compilation.semantic_ir.task_id,
            device_id=compilation.semantic_ir.device_id,
            revision=compilation.semantic_ir.revision,
            effect_id=effect.effect_id,
            effect_kind=effect.kind,
            targets=tuple(
                EffectEntityPreview(
                    entity_ref=entity_ref,
                    role=entity_by_id[entity_ref].role,
                    entity_type=entity_by_id[entity_ref].entity_type,
                    value=entity_by_id[entity_ref].value,
                )
                for entity_ref in effect.target_refs
            ),
            payloads=tuple(
                EffectEntityPreview(
                    entity_ref=entity_ref,
                    role=entity_by_id[entity_ref].role,
                    entity_type=entity_by_id[entity_ref].entity_type,
                    value=entity_by_id[entity_ref].value,
                )
                for entity_ref in effect.payload_refs
            ),
            policy=decisions[effect.effect_id].policy,
            policy_id=decisions[effect.effect_id].policy_id,
            policy_version=decisions[effect.effect_id].policy_version,
            expected_result_texts=effect.expected_result_texts,
        )
        for effect in compilation.semantic_ir.effects
    )
    report = SemanticRiskAuthorityReport(
        semantic_ir=compilation.semantic_ir,
        risk_policy=compilation.risk_policy,
        risk_decisions=compilation.risk_decisions,
        policy_traces=tuple(traces),
        effect_previews=previews,
        source_graph_digest=_runtime_graph_digest(graph),
    )
    report.validate()
    return report


def apply_formal_semantic_risk_policy(
    graph: Any,
    authority: SemanticRiskAuthorityReport,
) -> Any:
    """Project formal confirmation decisions back onto the transport graph."""

    authority.validate()
    if authority.source_graph_digest != _runtime_graph_digest(graph):
        raise TaskSemanticIRError("正式语义风险权威未绑定当前任务图。")
    decisions = {item.effect_id: item for item in authority.risk_decisions}
    decision_by_runtime_effect_id: dict[str, RiskDecision] = {}
    for effect in authority.semantic_ir.effects:
        risk_id = str(effect.attributes.get("runtime_effect_id") or "").strip()
        if not risk_id:
            raise TaskSemanticIRError(
                f"EffectIntent 缺少 runtime effect 绑定：{effect.effect_id}"
            )
        if risk_id in decision_by_runtime_effect_id:
            raise TaskSemanticIRError(f"runtime effect 重复映射：{risk_id}")
        decision_by_runtime_effect_id[risk_id] = decisions[effect.effect_id]

    projected_risks = []
    for risk in tuple(getattr(graph, "risk_actions", ()) or ()):
        risk_id = str(getattr(risk, "risk_id", "") or "")
        decision = decision_by_runtime_effect_id.get(risk_id)
        if decision is None:
            raise TaskSemanticIRError(f"正式风险权威遗漏 runtime effect：{risk_id}")
        projected_risks.append(
            replace(
                risk,
                confirmation_required=(
                    decision.policy == CONFIRMATION_REQUIRED
                ),
                risk_level=(
                    "high"
                    if decision.policy == CONFIRMATION_REQUIRED
                    else "low"
                ),
            )
        )

    status = str(getattr(graph, "status", "") or "")
    active_id = str(getattr(graph, "active_subgoal_id", "") or "")
    active = next(
        (
            item
            for item in tuple(getattr(graph, "subgoals", ()) or ())
            if str(getattr(item, "subgoal_id", "") or "") == active_id
        ),
        None,
    )
    required_ids = {
        str(getattr(item, "risk_id", "") or "")
        for item in projected_risks
        if bool(getattr(item, "confirmation_required", False))
    }
    active_requires_confirmation = bool(
        active is not None
        and set(tuple(getattr(active, "risk_action_ids", ()) or ()))
        .intersection(required_ids)
    )
    if status == "awaiting_confirmation" and not active_requires_confirmation:
        status = "ready"
    elif (
        status in {"ready", "running"}
        and active_requires_confirmation
    ):
        status = "awaiting_confirmation"
    return replace(graph, risk_actions=tuple(projected_risks), status=status)
