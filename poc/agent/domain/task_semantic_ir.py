from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, fields, replace
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
DEFAULT_CONFIRMATION_EFFECT_KINDS = frozenset({"authentication", "financial_transaction"})
SURFACE_KINDS = frozenset(
    {"launcher", "app", "system_dialog", "keyboard", "file_picker", "current_surface", "device", "system", "recent_tasks"}
)
ENTITY_AUTHORITIES = frozenset({"user_literal", "planner_context"})
CONSTRAINT_KINDS = frozenset({"exact_entity", "forbidden_effect", "required_state", "required_action", "planner_context"})
REQUIRED_ACTION_KINDS = frozenset(
    {
        "tap_semantic", "double_tap", "swipe", "long_press", "drag", "input_verified_text",
        "clear_verified_text", "press_enter", "pinch", "home", "back", "open_recent_apps",
        "hardware_key", "dismiss_overlay", "reveal_system_navigation",
    }
)
STATE_PREDICATES = frozenset(
    {"input.value_equals", "surface.state_visible", "effect.applied", "effect.result_visible", "observation.matches_description"}
)
EVIDENCE_SOURCE_KINDS = frozenset({"visual_claim", "controller_transition", "effect_receipt"})
SUBGOAL_IMPACTS = frozenset({"read_only", "navigation_only", "external_state", "unknown"})
_ID = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
_EXTERNAL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_EDITABLE = re.compile(
    r"输入框|文本框|编辑框|输入栏|编辑栏|输入区域|编辑区域|字段|表单项|搜索框|地址栏|正文框|"
    r"\binput\s+(?:field|box|area|control)\b|\btext\s*(?:field|box|area)\b|\btextarea\b|"
    r"\b(?:editor|composer|form\s+field|search\s+box|address\s+bar)\b",
    re.I,
)
_RECENTS = re.compile(r"最近任务|最近应用|任务概览|后台(?:任务|应用)|系统多任务|\brecents?\b|\brecent\s+(?:tasks?|apps?)\b", re.I)
_RECENT_CARD = re.compile(r"(?:应用)?(?:预览|任务)?卡片|\b(?:app\s+)?(?:preview\s+|task\s+)?card\b", re.I)
_DISMISS = re.compile(r"划掉|滑走|移出|移除|关闭|清除|\b(?:dismiss|remove|close|clear)\b|\bswipe\b.{0,80}\b(?:away|off)\b", re.I)
_DIRECTION = re.compile(r"向(?:上|下|左|右)|(?:上|下|左|右)(?:划|滑)|\b(?:up|down|left|right)(?:ward)?\b", re.I)
_ENTITY_TYPES = {
    "recipient": "party", "input_text": "text", "target_ui_label": "ui_literal", "target_surface": "surface",
    "spatial_hint": "spatial_hint", "amount": "money", "currency": "currency", "merchant": "party",
    "payee": "party", "account": "account", "file": "file", "product": "product", "date": "date", "time": "time",
}
_TARGET_ROLES = {
    "send_message": ("recipient",), "relationship_change": ("account", "recipient", "target"),
    "membership_change": ("account", "recipient", "target"),
    "financial_transaction": ("merchant", "payee", "recipient", "account"), "authentication": ("account",),
    "publish_content": ("account", "target"), "data_mutation": ("file", "target", "product"),
    "irreversible_data_deletion": ("file", "target", "product", "account"),
    "sensitive_permission_change": ("account", "target"),
}
_PAYLOAD_ROLES = {
    "send_message": ("input_text",), "publish_content": ("input_text", "file"),
    "financial_transaction": ("amount", "currency", "product"), "data_mutation": ("input_text", "value"),
}
_ACTION_PATTERNS = (
    ("double_tap", re.compile(r"双击|double[ _-]?(?:tap|click)", re.I)),
    ("pinch", re.compile(r"捏合|双指|pinch|zoom", re.I)),
    ("press_enter", re.compile(r"回车|换行(?:键)?|enter(?:\s+key)?|new[ _-]?line(?:\s+key)?", re.I)),
    ("clear_verified_text", re.compile(
        r"(?:清空|清除|置空|删除)[^，。；;,.!?！？\r\n]*?(?:输入框|文本框|编辑框|输入栏|编辑栏|输入区域|编辑区域|草稿|(?:输入法|IME\s*)?预编辑(?:文本)?)|"
        r"(?:clear|empty|erase)[^，。；;,.!?！？\r\n]*?(?:input\s+(?:field|box|area)|text\s*(?:field|box|area)|editor|composer|draft|preedit|composition)", re.I)),
    ("dismiss_overlay", re.compile(r"关闭.{0,6}(?:弹窗|弹层|对话框)|dismiss[ _-]?overlay", re.I)),
    ("reveal_system_navigation", re.compile(r"(?:唤出|显示).{0,6}(?:系统)?导航栏|reveal[ _-]?navigation", re.I)),
    ("long_press", re.compile(r"长按|long[ _-]?press", re.I)),
    ("drag", re.compile(r"拖动|拖拽|drag", re.I)),
    ("swipe", re.compile(r"滑动|上划|下划|左划|右划|swipe", re.I)),
    ("input_verified_text", re.compile(
        r"输入(?!框|法|值|区域|表面|载体|状态|控件|字段|页面|界面|模式|键盘)|填写|键入|\btype\b|"
        r"\binput\b(?!\s*(?:field|box|area|control|state|mode|method|page|screen|keyboard)\b)", re.I)),
    ("home", re.compile(r"home\s*键|回到主页|回到主桌面|回到(?:手机|系统)主屏幕", re.I)),
    ("open_recent_apps", re.compile(r"(?:打开|进入|调出|显示)(?:系统)?(?:最近任务|最近应用|系统多任务)(?:页面|界面|列表)?|(?:open|show)\s+(?:the\s+)?(?:recent\s+(?:tasks|apps)|recents)", re.I)),
    ("back", re.compile(r"返回键|后退键|back\s*key|(?:收起|隐藏|关闭).{0,6}(?:软?键盘|输入法)", re.I)),
    ("hardware_key", re.compile(r"音量键|电源键|hardware\s*key", re.I)),
    ("tap_semantic", re.compile(r"点击|轻触|点按|\btap\b|\bclick\b", re.I)),
)


class TaskSemanticIRError(ValueError):
    pass


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TaskSemanticIRError(f"{name} 必须是非空字符串。")
    return value.strip()


def _valid_id(value: str, name: str, *, external: bool = False) -> None:
    if not (_EXTERNAL_ID if external else _ID).fullmatch(value):
        raise TaskSemanticIRError(f"{name} 无效：{value!r}")


def _slug(value: Any, fallback: str) -> str:
    text = re.sub(r"[^a-z0-9_.-]+", "_", str(value or "").strip().casefold()).strip("_.-")
    if not text or not text[0].isalpha():
        text = fallback
    return text[:96]


def _json_value(value: Any, name: str) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))
    except (TypeError, ValueError) as exc:
        raise TaskSemanticIRError(f"{name} 不是 JSON 值。") from exc


def _jsonify(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_jsonify(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    return value


def _wire(record: Any) -> dict[str, Any]:
    return {item.name: _jsonify(getattr(record, item.name)) for item in fields(record)}


def _digest(value: Any) -> str:
    payload = json.dumps(_jsonify(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _unique(items: tuple[Any, ...], attr: str, name: str) -> dict[str, Any]:
    values = {getattr(item, attr): item for item in items}
    if len(values) != len(items):
        raise TaskSemanticIRError(f"{name} ID 重复。")
    return values


class _Record:
    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return _wire(self)


@dataclass(frozen=True)
class SourceSpan(_Record):
    start: int
    end: int

    def validate(self) -> None:
        if isinstance(self.start, bool) or not isinstance(self.start, int) or not isinstance(self.end, int) or self.start < 0 or self.end <= self.start:
            raise TaskSemanticIRError("source_span 必须是非空正向区间。")


@dataclass(frozen=True)
class SemanticEntity(_Record):
    entity_id: str
    entity_type: str
    role: str
    value: Any
    source_span: SourceSpan | None = None
    authority: str = "planner_context"

    def validate(self, raw_goal: str | None = None) -> None:
        _valid_id(self.entity_id, "entity_id")
        _required_text(self.entity_type, "entity_type")
        _valid_id(self.role, "entity role")
        _json_value(self.value, f"entity {self.entity_id}.value")
        if self.authority not in ENTITY_AUTHORITIES:
            raise TaskSemanticIRError(f"实体 authority 无效：{self.authority}")
        if self.source_span is not None:
            self.source_span.validate()
            if raw_goal is not None and raw_goal[self.source_span.start:self.source_span.end] != self.value:
                raise TaskSemanticIRError("user_literal source_span 未逐字绑定实体值。")


@dataclass(frozen=True)
class SurfaceRef(_Record):
    surface_id: str
    kind: str
    app_id: str = ""
    app_name: str = ""

    def validate(self) -> None:
        _valid_id(self.surface_id, "surface_id")
        if self.kind not in SURFACE_KINDS:
            raise TaskSemanticIRError(f"surface kind 无效：{self.kind}")
        if self.kind == "app":
            _valid_id(self.app_id, "surface.app_id")
            _required_text(self.app_name, "surface.app_name")


@dataclass(frozen=True)
class EffectIntent(_Record):
    effect_id: str
    kind: str
    target_refs: tuple[str, ...] = ()
    payload_refs: tuple[str, ...] = ()
    source_subgoal_ids: tuple[str, ...] = ()
    expected_result_texts: tuple[str, ...] = ()
    attributes: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _valid_id(self.effect_id, "effect_id")
        _valid_id(self.kind, "effect kind")
        for value in (*self.target_refs, *self.payload_refs, *self.source_subgoal_ids):
            _valid_id(value, "effect reference")
        if len(set(self.target_refs)) != len(self.target_refs) or len(set(self.payload_refs)) != len(self.payload_refs):
            raise TaskSemanticIRError("effect entity reference 重复。")
        for text in self.expected_result_texts:
            _required_text(text, "expected_result_texts")
        _json_value(self.attributes, "effect.attributes")


@dataclass(frozen=True)
class CriticalBinding(_Record):
    binding_id: str
    effect_id: str
    binding_kind: str
    entity_ref: str

    def validate(self) -> None:
        for value, name in ((self.binding_id, "binding_id"), (self.effect_id, "effect_id"), (self.entity_ref, "entity_ref")):
            _valid_id(value, name)
        if self.binding_kind not in {"effect_target_equals", "effect_payload_equals"}:
            raise TaskSemanticIRError(f"binding_kind 无效：{self.binding_kind}")


@dataclass(frozen=True)
class ConstraintIntent(_Record):
    constraint_id: str
    kind: str
    subject_refs: tuple[str, ...] = ()
    object_refs: tuple[str, ...] = ()
    value: Any = None
    source_text: str = ""
    authoritative: bool = False

    def validate(self) -> None:
        _valid_id(self.constraint_id, "constraint_id")
        if self.kind not in CONSTRAINT_KINDS:
            raise TaskSemanticIRError(f"constraint kind 无效：{self.kind}")
        for value in (*self.subject_refs, *self.object_refs):
            _valid_id(value, "constraint reference")
        _json_value(self.value, "constraint.value")
        if self.kind == "required_action" and self.value not in REQUIRED_ACTION_KINDS:
            raise TaskSemanticIRError(f"required_action 无效：{self.value}")


@dataclass(frozen=True)
class DesiredState(_Record):
    state_id: str
    subject_ref: str
    predicate: str
    value: Any
    source_subgoal_id: str = ""

    def validate(self) -> None:
        _valid_id(self.state_id, "state_id")
        _valid_id(self.subject_ref, "state.subject_ref")
        if self.predicate not in STATE_PREDICATES:
            raise TaskSemanticIRError(f"state predicate 无效：{self.predicate}")
        if self.source_subgoal_id:
            _valid_id(self.source_subgoal_id, "state.source_subgoal_id")
        _json_value(self.value, "state.value")


@dataclass(frozen=True)
class EvidenceRequirement(_Record):
    requirement_id: str
    desired_state_ref: str
    allowed_sources: tuple[str, ...]

    def validate(self) -> None:
        _valid_id(self.requirement_id, "requirement_id")
        _valid_id(self.desired_state_ref, "desired_state_ref")
        if not self.allowed_sources or any(item not in EVIDENCE_SOURCE_KINDS for item in self.allowed_sources):
            raise TaskSemanticIRError("evidence allowed_sources 无效。")


@dataclass(frozen=True)
class SemanticSubgoal(_Record):
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
        for value in (self.subgoal_id, self.surface_ref, *self.depends_on, *self.constraint_refs, *self.entity_refs, *self.desired_state_refs, *self.effect_refs):
            _valid_id(value, "subgoal reference")
        if self.external_impact not in SUBGOAL_IMPACTS:
            raise TaskSemanticIRError(f"subgoal impact 无效：{self.external_impact}")
        if self.status not in {"pending", "active", "completed", "blocked", "skipped"}:
            raise TaskSemanticIRError(f"subgoal status 无效：{self.status}")


@dataclass(frozen=True)
class InputFieldIntent(_Record):
    field_id: str
    payload_ref: str
    field_label: str = ""
    recipient_refs: tuple[str, ...] = ()
    source_subgoal_ids: tuple[str, ...] = ()
    multiline: bool = False

    def validate(self) -> None:
        _valid_id(self.field_id, "field_id")
        for value in (self.payload_ref, *self.recipient_refs, *self.source_subgoal_ids):
            _valid_id(value, "input field reference")
        if not isinstance(self.field_label, str) or not isinstance(self.multiline, bool):
            raise TaskSemanticIRError("input field metadata 无效。")


@dataclass(frozen=True)
class TaskSemanticIR(_Record):
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
            raise TaskSemanticIRError(f"TaskSemanticIR 协议无效：{self.protocol_version}")
        _valid_id(self.task_id, "task_id", external=True)
        _valid_id(self.device_id, "device_id", external=True)
        if isinstance(self.revision, bool) or not isinstance(self.revision, int) or self.revision < 1:
            raise TaskSemanticIRError("revision 必须是正整数。")
        _required_text(self.raw_goal, "raw_goal")
        groups = (
            (self.surfaces, "surface_id", "surface"), (self.entities, "entity_id", "entity"),
            (self.effects, "effect_id", "effect"), (self.critical_bindings, "binding_id", "binding"),
            (self.constraints, "constraint_id", "constraint"), (self.desired_states, "state_id", "state"),
            (self.evidence_requirements, "requirement_id", "evidence"), (self.subgoals, "subgoal_id", "subgoal"),
            (self.input_fields, "field_id", "input field"),
        )
        maps = {name: _unique(items, attr, name) for items, attr, name in groups}
        for items, _, _ in groups:
            for item in items:
                item.validate()
        for item in self.entities:
            item.validate(self.raw_goal)
        entity_ids = set(maps["entity"])
        effect_ids = set(maps["effect"])
        subgoal_ids = set(maps["subgoal"])
        surface_ids = set(maps["surface"])
        constraint_ids = set(maps["constraint"])
        state_ids = set(maps["state"])
        for effect in self.effects:
            if set((*effect.target_refs, *effect.payload_refs)) - entity_ids:
                raise TaskSemanticIRError(f"EffectIntent 引用未知实体：{effect.effect_id}")
            if set(effect.source_subgoal_ids) - subgoal_ids:
                raise TaskSemanticIRError(f"EffectIntent 引用未知子目标：{effect.effect_id}")
        for binding in self.critical_bindings:
            if binding.effect_id not in effect_ids or binding.entity_ref not in entity_ids:
                raise TaskSemanticIRError(f"CriticalBinding 引用不存在：{binding.binding_id}")
            effect = maps["effect"][binding.effect_id]
            expected = effect.target_refs if binding.binding_kind == "effect_target_equals" else effect.payload_refs
            if binding.entity_ref not in expected:
                raise TaskSemanticIRError(f"CriticalBinding 未绑定 effect 对应角色：{binding.binding_id}")
        for subgoal in self.subgoals:
            if subgoal.surface_ref not in surface_ids or set(subgoal.constraint_refs) - constraint_ids or set(subgoal.entity_refs) - entity_ids or set(subgoal.desired_state_refs) - state_ids or set(subgoal.effect_refs) - effect_ids:
                raise TaskSemanticIRError(f"SemanticSubgoal 引用不存在：{subgoal.subgoal_id}")
        for state in self.desired_states:
            if state.subject_ref not in entity_ids | effect_ids | surface_ids:
                raise TaskSemanticIRError(f"DesiredState subject_ref 不存在：{state.subject_ref}")
        for item in self.input_fields:
            if item.payload_ref not in entity_ids or set(item.recipient_refs) - entity_ids or set(item.source_subgoal_ids) - subgoal_ids:
                raise TaskSemanticIRError(f"InputFieldIntent 引用不存在：{item.field_id}")

    @property
    def semantic_digest(self) -> str:
        self.validate()
        value = self.to_dict()
        authoritative = {item.constraint_id for item in self.constraints if item.authoritative}
        value["constraints"] = [item for item in value["constraints"] if item["constraint_id"] in authoritative]
        value["subgoals"] = [
            {**item, "constraint_refs": [ref for ref in item["constraint_refs"] if ref in authoritative]}
            for item in value["subgoals"]
        ]
        return _digest(value)


@dataclass(frozen=True)
class RiskDecision(_Record):
    effect_id: str
    policy: str
    matched_rule: str
    policy_id: str
    policy_version: int

    def validate(self) -> None:
        _valid_id(self.effect_id, "risk.effect_id")
        if self.policy not in RISK_POLICIES or self.policy_version < 1:
            raise TaskSemanticIRError("risk decision 无效。")
        _required_text(self.matched_rule, "matched_rule")
        _valid_id(self.policy_id, "policy_id")


@dataclass(frozen=True)
class LocalRiskPolicyConfig(_Record):
    policy_id: str = "default_low_friction"
    version: int = 2
    confirmation_effect_kinds: frozenset[str] = DEFAULT_CONFIRMATION_EFFECT_KINDS
    overrides: tuple[tuple[str, str], ...] = ()
    protocol_version: str = RISK_POLICY_PROTOCOL

    def validate(self) -> None:
        if self.protocol_version != RISK_POLICY_PROTOCOL or self.version < 1:
            raise TaskSemanticIRError("风险策略版本无效。")
        _valid_id(self.policy_id, "policy_id")
        if any(not _ID.fullmatch(kind) for kind in self.confirmation_effect_kinds):
            raise TaskSemanticIRError("confirmation_effect_kinds 无效。")
        ids = [effect_kind for effect_kind, policy in self.overrides]
        if len(ids) != len(set(ids)):
            raise TaskSemanticIRError("风险策略 override 重复。")
        if any(not _ID.fullmatch(effect_kind) or policy not in RISK_POLICIES for effect_kind, policy in self.overrides):
            raise TaskSemanticIRError("风险策略 override 无效。")

    def decide(self, effect: EffectIntent) -> RiskDecision:
        self.validate()
        effect.validate()
        override = dict(self.overrides).get(effect.kind)
        policy = override or (CONFIRMATION_REQUIRED if effect.kind in self.confirmation_effect_kinds else AUTOMATIC)
        return RiskDecision(effect.effect_id, policy, f"effect_kind:{effect.kind}" if override is None else "effect_id_override", self.policy_id, self.version)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "protocol_version": self.protocol_version, "policy_id": self.policy_id, "version": self.version,
            "confirmation_effect_kinds": sorted(self.confirmation_effect_kinds),
            "overrides": [{"effect_id": effect_id, "policy": policy} for effect_id, policy in self.overrides],
        }


@dataclass(frozen=True)
class EffectEntityPreview(_Record):
    entity_ref: str
    role: str
    entity_type: str
    value: Any

    def validate(self) -> None:
        _valid_id(self.entity_ref, "preview.entity_ref")
        _valid_id(self.role, "preview.role")
        _required_text(self.entity_type, "preview.entity_type")
        _json_value(self.value, "preview.value")


@dataclass(frozen=True)
class EffectPreview(_Record):
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
        if self.protocol_version != EFFECT_PREVIEW_PROTOCOL or self.policy not in RISK_POLICIES or self.revision < 1 or self.policy_version < 1:
            raise TaskSemanticIRError("effect preview 元数据无效。")
        for value, name in ((self.task_id, "task_id"), (self.device_id, "device_id")):
            _valid_id(value, name, external=True)
        for value in (self.effect_id, self.effect_kind, self.policy_id):
            _valid_id(value, "effect preview id")
        for item in (*self.targets, *self.payloads):
            item.validate()

    @property
    def preview_digest(self) -> str:
        return _digest(self.to_dict())


@dataclass(frozen=True)
class SemanticCompilationReport(_Record):
    semantic_ir: TaskSemanticIR
    risk_policy: LocalRiskPolicyConfig
    risk_decisions: tuple[RiskDecision, ...]
    warnings: tuple[str, ...] = ()
    authoritative: bool = False
    execution_allowed: bool = False
    protocol_version: str = COMPILATION_REPORT_PROTOCOL

    def validate(self) -> None:
        if self.protocol_version != COMPILATION_REPORT_PROTOCOL or self.authoritative or self.execution_allowed:
            raise TaskSemanticIRError("semantic compilation 不得携带执行权限。")
        self.semantic_ir.validate()
        self.risk_policy.validate()
        if {item.effect_id for item in self.risk_decisions} != {item.effect_id for item in self.semantic_ir.effects}:
            raise TaskSemanticIRError("risk decisions 未覆盖全部 effect。")
        for item in self.risk_decisions:
            item.validate()


@dataclass(frozen=True)
class RiskPolicyTrace(_Record):
    effect_id: str
    effect_kind: str
    formal_policy: str
    allowed: bool
    reason: str

    def validate(self) -> None:
        for value in (self.effect_id, self.effect_kind):
            _valid_id(value, "policy trace id")
        if self.formal_policy not in RISK_POLICIES or not isinstance(self.allowed, bool):
            raise TaskSemanticIRError("policy trace 无效。")
        _required_text(self.reason, "policy trace reason")


@dataclass(frozen=True)
class SemanticRiskAuthorityReport(_Record):
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
        if self.protocol_version != AUTHORITY_REPORT_PROTOCOL or self.authoritative_scope != "semantic_task_and_risk" or self.physical_execution_allowed:
            raise TaskSemanticIRError("semantic authority 元数据无效。")
        if not re.fullmatch(r"[0-9a-f]{64}", self.source_graph_digest):
            raise TaskSemanticIRError("source_graph_digest 无效。")
        self.semantic_ir.validate()
        self.risk_policy.validate()
        effect_ids = {item.effect_id for item in self.semantic_ir.effects}
        if {item.effect_id for item in self.risk_decisions} != effect_ids or {item.effect_id for item in self.policy_traces} != effect_ids or {item.effect_id for item in self.effect_previews} != effect_ids:
            raise TaskSemanticIRError("semantic authority 未逐 effect 覆盖。")
        for item in (*self.risk_decisions, *self.policy_traces, *self.effect_previews):
            item.validate()


def local_risk_policy_from_dict(payload: Mapping[str, Any]) -> LocalRiskPolicyConfig:
    if not isinstance(payload, Mapping) or set(payload) != {"protocol_version", "policy_id", "version", "confirmation_effect_kinds", "overrides"}:
        raise TaskSemanticIRError("风险策略字段不匹配。")
    kinds = payload.get("confirmation_effect_kinds")
    overrides = payload.get("overrides")
    if not isinstance(kinds, list) or not isinstance(overrides, list):
        raise TaskSemanticIRError("风险策略集合必须是数组。")
    parsed: list[tuple[str, str]] = []
    for item in overrides:
        if not isinstance(item, Mapping) or set(item) != {"effect_id", "policy"}:
            raise TaskSemanticIRError("风险策略 override 字段无效。")
        parsed.append((str(item["effect_id"]), str(item["policy"])))
    config = LocalRiskPolicyConfig(
        protocol_version=str(payload["protocol_version"]), policy_id=str(payload["policy_id"]),
        version=payload["version"], confirmation_effect_kinds=frozenset(str(item) for item in kinds), overrides=tuple(parsed),
    )
    config.validate()
    return config


def _source_span(raw_goal: str, value: Any) -> SourceSpan | None:
    if not isinstance(value, str) or not value:
        return None
    start = raw_goal.find(value)
    return SourceSpan(start, start + len(value)) if start >= 0 else None


def _entity_refs(entities: tuple[SemanticEntity, ...], roles: tuple[str, ...]) -> tuple[str, ...]:
    role_set = set(roles)
    return tuple(item.entity_id for item in entities if item.role in role_set)


def _runtime_graph_digest(graph: Any) -> str:
    if hasattr(graph, "to_dict"):
        payload = graph.to_dict()
        payload.pop("current_subgoal", None)
        payload.pop("replan_history", None)
        return _digest(payload)
    return _digest({"task_id": getattr(graph, "task_id", ""), "revision": getattr(graph, "revision", 0)})


def _positive_match(text: str, pattern: re.Pattern[str], action: str) -> bool:
    for match in pattern.finditer(text):
        prefix = re.split(r"[，。；;,.!?！？\n\r]", text[max(0, match.start() - 40):match.start()])[-1]
        if re.search(r"(?:不(?:要|得|可|能|用|是)?|别|勿|禁止|避免|无需|无须|do\s+not|don['’]?t|must\s+not|never|without|avoid|no)(?:执行|进行|使用)?\s*$", prefix, re.I):
            continue
        if action == "input_verified_text" and re.search(r"(?:可|可以|能|能够|支持)(?:直接|正常|精确)?\s*$", prefix, re.I):
            continue
        return True
    return False


def _surface_for_subgoal(subgoal: Any, surfaces: tuple[SurfaceRef, ...]) -> str:
    text = " ".join((str(getattr(subgoal, "objective", "")), *map(str, getattr(subgoal, "constraints", ()) or ()), *map(str, getattr(subgoal, "completion_conditions", ()) or ()))).casefold()
    if _RECENTS.search(text):
        match = next((item.surface_id for item in surfaces if item.kind == "recent_tasks"), "")
        if match:
            return match
    if any(term in text for term in ("主桌面", "桌面", "主页", "手机主屏幕", "系统主屏幕", "主屏幕", "home screen", "launcher")):
        match = next((item.surface_id for item in surfaces if item.kind == "launcher"), "")
        if match:
            return match
    app_matches = [item.surface_id for item in surfaces if item.kind == "app" and (item.app_id.casefold() in text or item.app_name.casefold() in text)]
    if len(app_matches) == 1:
        return app_matches[0]
    current = next((item.surface_id for item in surfaces if item.kind == "current_surface"), "")
    if current:
        return current
    apps = [item.surface_id for item in surfaces if item.kind == "app"]
    return apps[0] if len(apps) == 1 else surfaces[0].surface_id


def compile_runtime_graph_semantics(graph: Any, *, risk_policy: LocalRiskPolicyConfig | None = None) -> SemanticCompilationReport:
    task_id = str(getattr(graph, "task_id", "")).strip()
    device_id = str(getattr(graph, "device_id", "")).strip()
    revision = getattr(graph, "revision", 0)
    goal = getattr(graph, "goal", None)
    raw_goal = str(getattr(graph, "raw_user_goal", "") or getattr(goal, "objective", "") or "").strip()
    raw_entities = getattr(goal, "entities", {}) or {}
    if not isinstance(raw_entities, Mapping):
        raise TaskSemanticIRError("goal.entities 必须是映射。")
    target_apps = tuple(getattr(goal, "target_apps", ()) or ())
    subgoals = tuple(getattr(graph, "subgoals", ()) or ())
    structured = " ".join(str(getattr(item, "objective", "")) for item in subgoals)
    surfaces: list[SurfaceRef] = []
    explicit_system = not target_apps and str(raw_entities.get("target_surface") or "") in {"device", "system"}
    launcher_requested = bool(re.search(r"主桌面|桌面|主页|手机主屏幕|系统主屏幕|home screen|launcher", raw_goal, re.I))
    if launcher_requested or (explicit_system and re.search(r"主屏幕|home screen|launcher", structured, re.I)):
        surfaces.append(SurfaceRef("surface_launcher", "launcher"))
    if _RECENTS.search(structured):
        surfaces.append(SurfaceRef("surface_recent_tasks", "recent_tasks"))
    for app in target_apps:
        app_id = _slug(getattr(app, "app_id", ""), "app")
        surfaces.append(SurfaceRef(f"surface_{app_id}", "app", app_id, str(getattr(app, "app_name", "") or app_id)))
    if len(target_apps) > 1 or re.search(r"当前|眼前|current|foreground", raw_goal, re.I):
        surfaces.append(SurfaceRef("surface_current", "current_surface"))
    if not surfaces:
        kind = str(raw_entities.get("target_surface") or "current_surface")
        kind = kind if kind in SURFACE_KINDS else "current_surface"
        surfaces.append(SurfaceRef(f"surface_{_slug(kind, 'current')}", kind))

    entities: list[SemanticEntity] = []
    field_meta: dict[str, tuple[str, str]] = {}
    recipients = raw_entities.get("recipients")
    if isinstance(recipients, list):
        for index, value in enumerate(recipients, 1):
            span = _source_span(raw_goal, value)
            entities.append(SemanticEntity(f"entity_recipient_{index}", "party", "recipient", _json_value(value, "recipients"), span, "user_literal" if span else "planner_context"))
    input_fields = raw_entities.get("input_fields")
    if isinstance(input_fields, list):
        for index, spec in enumerate(input_fields, 1):
            if not isinstance(spec, Mapping):
                continue
            field_id = _slug(spec.get("field_id"), f"field_{index}")
            entity_id = f"entity_input_text_{field_id}"
            value = spec.get("text")
            span = _source_span(raw_goal, value)
            entities.append(SemanticEntity(entity_id, "text", "input_text", _json_value(value, "input_fields.text"), span, "user_literal" if span else "planner_context"))
            field_meta[entity_id] = (field_id, str(spec.get("field_label") or ""))
    for index, key in enumerate(sorted(raw_entities, key=str), 1):
        if key in {"recipients", "input_fields"}:
            continue
        role = _slug(key, f"role_{index}")
        value = raw_entities[key]
        span = _source_span(raw_goal, value)
        entities.append(SemanticEntity(f"entity_{role}_{index}", _ENTITY_TYPES.get(role, "opaque"), role, _json_value(value, f"entities.{role}"), span, "user_literal" if span else "planner_context"))
    entity_tuple = tuple(entities)
    by_role: dict[str, list[SemanticEntity]] = {}
    for entity in entity_tuple:
        by_role.setdefault(entity.role, []).append(entity)

    runtime_subgoals = {str(getattr(item, "subgoal_id", "")): item for item in subgoals}
    effects: list[EffectIntent] = []
    warnings: list[str] = []
    for index, risk in enumerate(tuple(getattr(graph, "risk_actions", ()) or ()), 1):
        runtime_id = str(getattr(risk, "risk_id", "") or "")
        kind = str(getattr(risk, "effect_kind", "") or "generic_effect")
        effect_id = f"effect_{_slug(runtime_id, f'effect_{index}')}"
        source_ids = tuple(_slug(item, "subgoal") for item in getattr(risk, "subgoal_ids", ()) or ())
        expected = tuple(str(item).strip() for item in getattr(risk, "expected_result_texts", ()) or () if str(item).strip())
        if not expected:
            expected = tuple(dict.fromkeys(text for subgoal_id in source_ids for text in map(str, getattr(runtime_subgoals.get(subgoal_id), "completion_conditions", ()) or ()) if text.strip()))
        targets = _entity_refs(entity_tuple, tuple(getattr(risk, "target_roles", ()) or ()) or _TARGET_ROLES.get(kind, ("target",)))
        payloads = _entity_refs(entity_tuple, tuple(getattr(risk, "payload_roles", ()) or ()) or _PAYLOAD_ROLES.get(kind, ("input_text", "value")))
        if not targets:
            warnings.append(f"effect_{runtime_id}:missing_typed_target")
        effects.append(EffectIntent(effect_id, kind, targets, payloads, source_ids, expected, {"runtime_effect_id": runtime_id, "planner_declared_typed_effect": bool(getattr(risk, "effect_kind", ""))}))
    effect_tuple = tuple(effects)
    bindings = tuple(
        CriticalBinding(f"binding_{effect.effect_id}_{kind}_{index}", effect.effect_id, f"effect_{kind}_equals", ref)
        for effect in effect_tuple for kind, refs in (("target", effect.target_refs), ("payload", effect.payload_refs)) for index, ref in enumerate(refs, 1)
    )

    constraints: list[ConstraintIntent] = []
    exact_ids: list[str] = []
    for entity in entity_tuple:
        if entity.authority == "user_literal":
            constraint_id = f"constraint_exact_{len(exact_ids) + 1}"
            exact_ids.append(constraint_id)
            constraints.append(ConstraintIntent(constraint_id, "exact_entity", (entity.entity_id,), value=entity.value, source_text=str(entity.value) if isinstance(entity.value, str) else "", authoritative=True))
    context_ids: dict[str, list[str]] = {"": []}
    context_specs = [("", str(item).strip()) for item in getattr(graph, "constraints", ()) or () if str(item).strip()]
    context_specs.extend((str(getattr(subgoal, "subgoal_id", "")), str(item).strip()) for subgoal in subgoals for item in getattr(subgoal, "constraints", ()) or () if str(item).strip())
    for owner, text in context_specs:
        constraint_id = f"constraint_context_{sum(map(len, context_ids.values())) + 1}"
        constraints.append(ConstraintIntent(constraint_id, "planner_context", value=text, source_text=text))
        context_ids.setdefault(owner, []).append(constraint_id)
    action_ids: dict[str, list[str]] = {}
    has_input_payload = bool(by_role.get("input_text"))
    for subgoal in subgoals:
        subgoal_id = str(getattr(subgoal, "subgoal_id", ""))
        if str(getattr(subgoal, "external_impact", "")) == "read_only":
            continue
        objective = str(getattr(subgoal, "objective", ""))
        state_text = " ".join((objective, *map(str, getattr(subgoal, "completion_conditions", ()) or ())))
        card_dismiss = bool(_RECENTS.search(state_text) and _RECENT_CARD.search(state_text) and _DISMISS.search(state_text))
        actions: list[tuple[str, str]] = []
        if explicit_system and re.search(r"回到主屏幕|return\s+to\s+(?:the\s+)?home\s+screen", objective, re.I):
            actions.append(("home", objective))
        if card_dismiss:
            actions.append(("swipe", objective if _DIRECTION.search(objective) else f"{objective}；系统最近任务卡片向左滑动清除"))
        for action, pattern in _ACTION_PATTERNS:
            if action == "swipe" and card_dismiss:
                continue
            if _positive_match(objective, pattern, action) and (action != "input_verified_text" or has_input_payload):
                actions.append((action, objective))
        for action, source_text in dict(actions).items():
            constraint_id = f"constraint_action_{1 + sum(map(len, action_ids.values()))}"
            constraints.append(ConstraintIntent(constraint_id, "required_action", value=action, source_text=source_text, authoritative=True))
            action_ids.setdefault(subgoal_id, []).append(constraint_id)

    effect_refs: dict[str, list[str]] = {}
    for effect in effect_tuple:
        for subgoal_id in effect.source_subgoal_ids:
            effect_refs.setdefault(subgoal_id, []).append(effect.effect_id)
    surfaces_tuple = tuple(surfaces)
    desired: list[DesiredState] = []
    evidence: list[EvidenceRequirement] = []
    desired_refs: dict[str, list[str]] = {}

    def add_state(description: str, owner: str, surface: str) -> None:
        description = description.strip()
        active_effects = effect_refs.get(owner, [])
        input_matches = [item for item in by_role.get("input_text", ()) if isinstance(item.value, str) and item.value and item.value.casefold() in description.casefold() and _EDITABLE.search(description)]
        if len(active_effects) == 1:
            receipt_only = bool(re.search(r"动作已执行|操作已执行|效果已触发|action executed|effect applied", description, re.I))
            specs = [(active_effects[0], "effect.applied" if receipt_only else "effect.result_visible", True, ("effect_receipt",) if receipt_only else ("visual_claim", "effect_receipt"))]
        elif input_matches:
            specs = [(item.entity_id, "input.value_equals", item.value, ("visual_claim",)) for item in input_matches]
        else:
            specs = [(surface, "observation.matches_description", description, ("visual_claim", "controller_transition"))]
        for subject, predicate, value, sources in specs:
            state_id = f"state_{len(desired) + 1}"
            desired.append(DesiredState(state_id, subject, predicate, value, owner))
            desired_refs.setdefault(owner, []).append(state_id)
            evidence.append(EvidenceRequirement(f"evidence_{len(evidence) + 1}", state_id, sources))

    semantic_subgoals: list[SemanticSubgoal] = []
    for subgoal in subgoals:
        subgoal_id = str(getattr(subgoal, "subgoal_id", ""))
        surface = _surface_for_subgoal(subgoal, surfaces_tuple)
        text = " ".join((str(getattr(subgoal, "objective", "")), *map(str, getattr(subgoal, "completion_conditions", ()) or ()))).casefold()
        entity_refs = tuple(item.entity_id for item in entity_tuple if isinstance(item.value, str) and item.value.strip() and item.value.strip().casefold() in text)
        for description in getattr(subgoal, "completion_conditions", ()) or ():
            if str(description).strip():
                add_state(str(description), subgoal_id, surface)
        semantic_subgoals.append(SemanticSubgoal(
            subgoal_id, surface, str(getattr(subgoal, "status", "pending")), str(getattr(subgoal, "external_impact", "unknown")),
            tuple(map(str, getattr(subgoal, "depends_on", ()) or ())),
            tuple(dict.fromkeys((*exact_ids, *context_ids.get("", ()), *context_ids.get(subgoal_id, ()), *action_ids.get(subgoal_id, ())))),
            entity_refs, tuple(desired_refs.get(subgoal_id, ())), tuple(effect_refs.get(subgoal_id, ())),
        ))
    default_surface = surfaces_tuple[0].surface_id
    for condition in getattr(graph, "completion_conditions", ()) or ():
        description = str(getattr(condition, "description", "") or "").strip()
        if description:
            add_state(description, "", default_surface)

    constraint_map = {item.constraint_id: item for item in constraints}
    desired_map = {item.state_id: item for item in desired}
    recipient_refs = tuple(item.entity_id for item in by_role.get("recipient", ()))
    input_entities = tuple(by_role.get("input_text", ()))
    typed_fields: list[InputFieldIntent] = []
    for index, entity in enumerate(input_entities, 1):
        sources = []
        field_id, label = field_meta.get(entity.entity_id, (f"input_field_{index}", ""))
        for subgoal, semantic in zip(subgoals, semantic_subgoals):
            actions = {constraint_map[ref].value for ref in semantic.constraint_refs if ref in constraint_map and constraint_map[ref].kind == "required_action"}
            context = " ".join((str(getattr(subgoal, "objective", "")), *map(str, getattr(subgoal, "constraints", ()) or ()), *map(str, getattr(subgoal, "completion_conditions", ()) or ()))).casefold()
            owns_typed_action = bool(actions & {"input_verified_text", "clear_verified_text"})
            literal_matches = [
                item for item in input_entities
                if isinstance(item.value, str) and item.value and item.value.casefold() in context
            ]
            owns_exact_literal = len(literal_matches) == 1 and literal_matches[0].entity_id == entity.entity_id
            owns = owns_typed_action and (
                len(input_entities) == 1 or bool(label and label.casefold() in context)
            ) or owns_exact_literal
            verifies_exact_value = any(
                desired_map[ref].subject_ref == entity.entity_id and desired_map[ref].predicate == "input.value_equals"
                for ref in semantic.desired_state_refs if ref in desired_map
            )
            if verifies_exact_value or (owns and semantic.external_impact != "read_only"):
                sources.append(semantic.subgoal_id)
        typed_fields.append(InputFieldIntent(field_id, entity.entity_id, label, recipient_refs, tuple(dict.fromkeys(sources)), isinstance(entity.value, str) and "\n" in entity.value))
    for semantic in semantic_subgoals:
        actions = {constraint_map[ref].value for ref in semantic.constraint_refs if ref in constraint_map and constraint_map[ref].kind == "required_action"}
        if "input_verified_text" not in actions:
            continue
        owners = [item for item in typed_fields if semantic.subgoal_id in item.source_subgoal_ids]
        if len(owners) != 1:
            raise TaskSemanticIRError("input_verified_text 子目标必须且只能绑定一个 InputFieldIntent。")
    for typed in typed_fields:
        if not typed.multiline or len(typed.source_subgoal_ids) != 1:
            continue
        owner = typed.source_subgoal_ids[0]
        semantic = next(item for item in semantic_subgoals if item.subgoal_id == owner)
        if any(constraint_map[ref].kind == "required_action" and constraint_map[ref].value == "press_enter" for ref in semantic.constraint_refs):
            continue
        constraint_id = f"constraint_action_{1 + sum(map(len, action_ids.values()))}"
        new_constraint = ConstraintIntent(constraint_id, "required_action", value="press_enter", source_text="multiline typed input field", authoritative=True)
        constraints.append(new_constraint)
        constraint_map[constraint_id] = new_constraint
        semantic_subgoals = [replace(item, constraint_refs=(*item.constraint_refs, constraint_id)) if item.subgoal_id == owner else item for item in semantic_subgoals]

    semantic_ir = TaskSemanticIR(
        task_id, device_id, revision, raw_goal, surfaces_tuple, entity_tuple, effect_tuple, bindings,
        tuple(constraints), tuple(desired), tuple(evidence), tuple(semantic_subgoals), tuple(typed_fields),
    )
    semantic_ir.validate()
    policy = risk_policy or LocalRiskPolicyConfig()
    decisions = tuple(policy.decide(effect) for effect in effect_tuple)
    report = SemanticCompilationReport(semantic_ir, policy, decisions, (f"runtime_graph_digest:{_runtime_graph_digest(graph)}", *warnings))
    report.validate()
    return report


def compile_formal_semantic_authority(graph: Any, *, risk_policy: LocalRiskPolicyConfig | None = None) -> SemanticRiskAuthorityReport:
    if any(not str(getattr(item, "effect_kind", "") or "") for item in getattr(graph, "risk_actions", ()) or ()):
        raise TaskSemanticIRError("正式语义权威只接受由 typed effect_intents 创建的风险条目。")
    compilation = compile_runtime_graph_semantics(graph, risk_policy=risk_policy)
    decisions = {item.effect_id: item for item in compilation.risk_decisions}
    entities = {item.entity_id: item for item in compilation.semantic_ir.entities}
    traces = tuple(RiskPolicyTrace(effect.effect_id, effect.kind, decisions[effect.effect_id].policy, effect.kind != "generic_effect", "typed_effect_local_policy") for effect in compilation.semantic_ir.effects)

    def preview(ref: str) -> EffectEntityPreview:
        entity = entities[ref]
        return EffectEntityPreview(ref, entity.role, entity.entity_type, entity.value)

    previews = tuple(EffectPreview(
        compilation.semantic_ir.task_id, compilation.semantic_ir.device_id, compilation.semantic_ir.revision,
        effect.effect_id, effect.kind, tuple(map(preview, effect.target_refs)), tuple(map(preview, effect.payload_refs)),
        decisions[effect.effect_id].policy, decisions[effect.effect_id].policy_id, decisions[effect.effect_id].policy_version,
        effect.expected_result_texts,
    ) for effect in compilation.semantic_ir.effects)
    report = SemanticRiskAuthorityReport(compilation.semantic_ir, compilation.risk_policy, compilation.risk_decisions, traces, previews, _runtime_graph_digest(graph))
    report.validate()
    return report


def apply_formal_semantic_risk_policy(graph: Any, authority: SemanticRiskAuthorityReport) -> Any:
    authority.validate()
    if authority.source_graph_digest != _runtime_graph_digest(graph):
        raise TaskSemanticIRError("正式语义风险权威未绑定当前任务图。")
    decisions = {effect.attributes.get("runtime_effect_id"): decision for effect, decision in zip(authority.semantic_ir.effects, authority.risk_decisions)}
    projected = []
    for risk in getattr(graph, "risk_actions", ()) or ():
        risk_id = str(getattr(risk, "risk_id", ""))
        decision = decisions.get(risk_id)
        if decision is None:
            raise TaskSemanticIRError(f"正式风险权威遗漏 runtime effect：{risk_id}")
        projected.append(replace(risk, confirmation_required=decision.policy == CONFIRMATION_REQUIRED, risk_level="high" if decision.policy == CONFIRMATION_REQUIRED else "low"))
    active_id = str(getattr(graph, "active_subgoal_id", "") or "")
    active = next((item for item in getattr(graph, "subgoals", ()) or () if getattr(item, "subgoal_id", "") == active_id), None)
    required = {item.risk_id for item in projected if item.confirmation_required}
    needs_confirmation = bool(active and set(getattr(active, "risk_action_ids", ()) or ()) & required)
    status = str(getattr(graph, "status", ""))
    if status == "awaiting_confirmation" and not needs_confirmation:
        status = "ready"
    elif status in {"ready", "running"} and needs_confirmation:
        status = "awaiting_confirmation"
    return replace(graph, risk_actions=tuple(projected), status=status)
