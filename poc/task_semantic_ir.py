from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


TASK_SEMANTIC_IR_PROTOCOL = "2026-08-18-task-semantic-ir-v1-shadow"
RISK_POLICY_PROTOCOL = "2026-08-18-local-risk-policy-v1"
SHADOW_REPORT_PROTOCOL = "2026-08-18-semantic-shadow-report-v1"

AUTOMATIC = "automatic"
CONFIRMATION_REQUIRED = "confirmation_required"
RISK_POLICIES = frozenset({AUTOMATIC, CONFIRMATION_REQUIRED})

DEFAULT_CONFIRMATION_EFFECT_KINDS = frozenset(
    {
        "authentication",
        "financial_transaction",
        "sensitive_permission_change",
        "irreversible_account_deletion",
        "irreversible_data_deletion",
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
_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
_EXTERNAL_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")

_LEGACY_RISK_EFFECT_KIND = {
    "message_or_communication": "send_message",
    "content_publication": "publish_content",
    "account_relationship_change": "relationship_change",
    "membership_change": "membership_change",
    "permission_role_change": "sensitive_permission_change",
    "data_mutation": "data_mutation",
    "data_deletion": "data_deletion",
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
class TaskSemanticIR:
    task_id: str
    device_id: str
    revision: int
    raw_goal: str
    surfaces: tuple[SurfaceRef, ...]
    entities: tuple[SemanticEntity, ...]
    effects: tuple[EffectIntent, ...]
    critical_bindings: tuple[CriticalBinding, ...] = ()
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
        }

    @property
    def semantic_digest(self) -> str:
        return _canonical_digest(self.to_dict())


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
    version: int = 1
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
class SemanticShadowReport:
    semantic_ir: TaskSemanticIR
    risk_policy: LocalRiskPolicyConfig
    risk_decisions: tuple[RiskDecision, ...]
    warnings: tuple[str, ...] = ()
    authoritative: bool = False
    execution_allowed: bool = False
    protocol_version: str = SHADOW_REPORT_PROTOCOL

    def validate(self) -> None:
        if self.protocol_version != SHADOW_REPORT_PROTOCOL:
            raise TaskSemanticIRError("影子报告协议版本无效。")
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
            _required_text(warning, "semantic_shadow.warning", max_length=500)

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


def _legacy_graph_digest(graph: Any) -> str:
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
        "risk_actions": [
            {
                "risk_id": str(getattr(item, "risk_id", "")),
                "risk_type": str(getattr(item, "risk_type", "")),
                "subgoal_ids": list(getattr(item, "subgoal_ids", ()) or ()),
            }
            for item in tuple(getattr(graph, "risk_actions", ()) or ())
        ],
        "subgoals": [
            {
                "subgoal_id": str(getattr(item, "subgoal_id", "")),
                "external_impact": str(getattr(item, "external_impact", "")),
                "risk_action_ids": list(getattr(item, "risk_action_ids", ()) or ()),
            }
            for item in tuple(getattr(graph, "subgoals", ()) or ())
        ],
    }
    return _canonical_digest(payload)


def compile_legacy_graph_shadow(
    graph: Any,
    *,
    risk_policy: LocalRiskPolicyConfig | None = None,
) -> SemanticShadowReport:
    """Project one legacy DeepSeek graph into a non-authoritative semantic IR.

    The projector deliberately ignores every free-text constraint for risk
    classification. It does not validate or mutate the legacy graph and cannot
    grant execution authority.
    """

    task_id = str(getattr(graph, "task_id", "")).strip()
    device_id = str(getattr(graph, "device_id", "")).strip()
    revision = getattr(graph, "revision", 0)
    raw_goal = str(getattr(graph, "raw_user_goal", "") or "").strip()
    goal = getattr(graph, "goal", None)
    if not raw_goal:
        raw_goal = str(getattr(goal, "objective", "") or "").strip()

    target_apps = tuple(getattr(goal, "target_apps", ()) or ())
    surfaces: list[SurfaceRef] = []
    if len(target_apps) > 1:
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

    raw_entities = getattr(goal, "entities", {}) or {}
    if not isinstance(raw_entities, Mapping):
        raise TaskSemanticIRError("legacy goal.entities 必须是映射。")
    entities: list[SemanticEntity] = []
    for index, key in enumerate(sorted(raw_entities, key=lambda item: str(item))):
        role = _slug(key, fallback=f"role_{index + 1}")
        value = raw_entities[key]
        span = _source_span(raw_goal, value)
        entities.append(
            SemanticEntity(
                entity_id=f"entity_{role}_{index + 1}",
                entity_type=_ENTITY_TYPE_BY_ROLE.get(role, "opaque"),
                role=role,
                value=_json_value(value, f"legacy.entities.{role}"),
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
        legacy_type = str(getattr(risk, "risk_type", "") or "")
        effect_kind = _LEGACY_RISK_EFFECT_KIND.get(legacy_type, "generic_effect")
        risk_id = _slug(getattr(risk, "risk_id", ""), fallback=f"effect_{index + 1}")
        source_subgoal_ids = tuple(
            _slug(value, fallback="subgoal")
            for value in tuple(getattr(risk, "subgoal_ids", ()) or ())
        )
        represented_subgoals.update(source_subgoal_ids)
        expected_results: list[str] = []
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
        target_refs = _entity_refs_for_roles(
            tuple(entities),
            _TARGET_ROLES_BY_EFFECT.get(effect_kind, ("target",)),
        )
        payload_refs = _entity_refs_for_roles(
            tuple(entities),
            _PAYLOAD_ROLES_BY_EFFECT.get(effect_kind, ("input_text", "value")),
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
                    "legacy_risk_type": legacy_type,
                    "legacy_risk_level": str(getattr(risk, "risk_level", "") or ""),
                    "legacy_confirmation_required": bool(
                        getattr(risk, "confirmation_required", False)
                    ),
                },
            )
        )

    for index, (subgoal_id, subgoal) in enumerate(subgoals.items()):
        if subgoal_id in represented_subgoals:
            continue
        impact = str(getattr(subgoal, "external_impact", "") or "")
        if impact not in {"external_state", "unknown"}:
            continue
        expected_results = tuple(
            str(value).strip()
            for value in tuple(getattr(subgoal, "completion_conditions", ()) or ())
            if str(value).strip()
        )
        effect_id = f"effect_generic_{index + 1}"
        target_refs = _entity_refs_for_roles(
            tuple(entities),
            _TARGET_ROLES_BY_EFFECT["generic_effect"],
        )
        if not target_refs:
            warnings.append(f"{effect_id}:missing_typed_target")
        if not expected_results:
            warnings.append(f"{effect_id}:missing_expected_result")
        effects.append(
            EffectIntent(
                effect_id=effect_id,
                kind="generic_effect",
                target_refs=target_refs,
                payload_refs=_entity_refs_for_roles(
                    tuple(entities),
                    _PAYLOAD_ROLES_BY_EFFECT["generic_effect"],
                ),
                source_subgoal_ids=(subgoal_id,),
                expected_result_texts=expected_results,
                attributes={"legacy_external_impact": impact},
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

    semantic_ir = TaskSemanticIR(
        task_id=task_id,
        device_id=device_id,
        revision=revision,
        raw_goal=raw_goal,
        surfaces=tuple(surfaces),
        entities=tuple(entities),
        effects=tuple(effects),
        critical_bindings=tuple(bindings),
    )
    semantic_ir.validate()
    policy = risk_policy or LocalRiskPolicyConfig()
    decisions = tuple(policy.decide(effect) for effect in semantic_ir.effects)
    warnings.insert(0, f"legacy_graph_digest:{_legacy_graph_digest(graph)}")
    report = SemanticShadowReport(
        semantic_ir=semantic_ir,
        risk_policy=policy,
        risk_decisions=decisions,
        warnings=tuple(warnings),
    )
    report.validate()
    return report
