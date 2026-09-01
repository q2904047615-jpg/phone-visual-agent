from __future__ import annotations

from .validation import ValidatedDataclassWire, canonical_digest, reject_if, wire_value
import json
import re
from dataclasses import dataclass, field, replace
from typing import Any, Mapping


TASK_SEMANTIC_IR_PROTOCOL = "2026-08-20-task-semantic-ir-v3"
RISK_POLICY_PROTOCOL = "2026-08-18-local-risk-policy-v1"
AUTHORITY_REPORT_PROTOCOL = "2026-08-20-typed-effect-authority-v1"
EFFECT_PREVIEW_PROTOCOL = "2026-08-19-effect-preview-v1"
AUTOMATIC = "automatic"
CONFIRMATION_REQUIRED = "confirmation_required"
RISK_POLICIES = frozenset({AUTOMATIC, CONFIRMATION_REQUIRED})
DEFAULT_CONFIRMATION_EFFECT_KINDS = frozenset({"authentication", "financial_transaction"})
SURFACE_KINDS = frozenset({'launcher', 'app', 'system_dialog', 'keyboard', 'file_picker', 'current_surface', 'device',
    'system', 'recent_tasks'})
ENTITY_AUTHORITIES = frozenset({"user_literal", "planner_context"})
CONSTRAINT_KINDS = frozenset({'exact_entity', 'forbidden_effect', 'required_state', 'required_action',
    'planner_context'})
REQUIRED_ACTION_KINDS = frozenset({'tap_semantic', 'double_tap', 'swipe', 'long_press', 'drag', 'input_verified_text',
    'clear_verified_text', 'press_enter', 'pinch', 'home', 'back', 'open_recent_apps', 'hardware_key',
    'dismiss_overlay', 'reveal_system_navigation'})
STATE_PREDICATES = frozenset({'input.value_equals', 'surface.state_visible', 'effect.applied', 'effect.result_visible',
    'observation.matches_description'})
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
_ENTITY_TYPES = {'recipient': 'party', 'input_text': 'text', 'target_ui_label': 'ui_literal',
    'target_surface': 'surface', 'spatial_hint': 'spatial_hint', 'amount': 'money', 'currency': 'currency',
    'merchant': 'party', 'payee': 'party', 'account': 'account', 'file': 'file', 'product': 'product', 'date': 'date',
    'time': 'time'}
_TARGET_ROLES = {'send_message': ('recipient',), 'relationship_change': ('account', 'recipient', 'target'),
    'membership_change': ('account', 'recipient', 'target'), 'financial_transaction': ('merchant', 'payee', 'recipient',
    'account'), 'authentication': ('account',), 'publish_content': ('account', 'target'), 'data_mutation': ('file',
    'target', 'product'), 'irreversible_data_deletion': ('file', 'target', 'product', 'account'),
    'sensitive_permission_change': ('account', 'target')}
_PAYLOAD_ROLES = {'send_message': ('input_text',), 'publish_content': ('input_text', 'file'),
    'financial_transaction': ('amount', 'currency', 'product'), 'data_mutation': ('input_text', 'value')}
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
    reject_if(not isinstance(value, str) or not value.strip(), TaskSemanticIRError(f"{name} 必须是非空字符串。"))
    return value.strip()


def _valid_id(value: str, name: str, *, external: bool=False) -> None:
    reject_if(not (_EXTERNAL_ID if external else _ID).fullmatch(value), TaskSemanticIRError(f"{name} 无效：{value!r}"))


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


def _digest(value: Any) -> str:
    return canonical_digest(wire_value(value))


def _unique(items: tuple[Any, ...], attr: str, name: str) -> dict[str, Any]:
    values = {getattr(item, attr): item for item in items}
    reject_if(len(values) != len(items), TaskSemanticIRError(f"{name} ID 重复。"))
    return values


@dataclass(frozen=True)
class SourceSpan(ValidatedDataclassWire):
    start: int
    end: int

    def validate(self) -> None:
        reject_if(
            isinstance(self.start, bool) or not isinstance(self.start, int) or (not isinstance(self.end,
            int)) or (self.start < 0) or (self.end <= self.start),
            TaskSemanticIRError("source_span 必须是非空正向区间。"),
        )


@dataclass(frozen=True)
class SemanticEntity(ValidatedDataclassWire):
    entity_id: str
    entity_type: str
    role: str
    value: Any
    source_span: SourceSpan | None = None
    authority: str = "planner_context"

    def validate(self, raw_goal: str | None=None) -> None:
        _valid_id(self.entity_id, "entity_id")
        _required_text(self.entity_type, "entity_type")
        _valid_id(self.role, "entity role")
        _json_value(self.value, f"entity {self.entity_id}.value")
        reject_if(self.authority not in ENTITY_AUTHORITIES, TaskSemanticIRError(f"实体 authority 无效：{self.authority}"))
        if self.source_span is not None:
            self.source_span.validate()
            reject_if(raw_goal is not None and raw_goal[self.source_span.start:self.source_span.end] != self.value, TaskSemanticIRError("user_literal source_span 未逐字绑定实体值。"))


@dataclass(frozen=True)
class SurfaceRef(ValidatedDataclassWire):
    surface_id: str
    kind: str
    app_id: str = ""
    app_name: str = ""

    def validate(self) -> None:
        _valid_id(self.surface_id, "surface_id")
        reject_if(self.kind not in SURFACE_KINDS, TaskSemanticIRError(f"surface kind 无效：{self.kind}"))
        if self.kind == 'app':
            _valid_id(self.app_id, "surface.app_id")
            _required_text(self.app_name, "surface.app_name")


@dataclass(frozen=True)
class EffectIntent(ValidatedDataclassWire):
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
        reject_if(len(set(self.target_refs)) != len(self.target_refs) or len(set(self.payload_refs)) != len(self.payload_refs), TaskSemanticIRError("effect entity reference 重复。"))
        for text in self.expected_result_texts:
            _required_text(text, "expected_result_texts")
        _json_value(self.attributes, "effect.attributes")


@dataclass(frozen=True)
class ConstraintIntent(ValidatedDataclassWire):
    constraint_id: str
    kind: str
    subject_refs: tuple[str, ...] = ()
    object_refs: tuple[str, ...] = ()
    value: Any = None
    source_text: str = ""
    authoritative: bool = False

    def validate(self) -> None:
        _valid_id(self.constraint_id, "constraint_id")
        reject_if(self.kind not in CONSTRAINT_KINDS, TaskSemanticIRError(f"constraint kind 无效：{self.kind}"))
        for value in (*self.subject_refs, *self.object_refs):
            _valid_id(value, "constraint reference")
        _json_value(self.value, "constraint.value")
        reject_if(self.kind == 'required_action' and self.value not in REQUIRED_ACTION_KINDS, TaskSemanticIRError(f"required_action 无效：{self.value}"))


@dataclass(frozen=True)
class DesiredState(ValidatedDataclassWire):
    state_id: str
    subject_ref: str
    predicate: str
    value: Any
    source_subgoal_id: str = ""

    def validate(self) -> None:
        _valid_id(self.state_id, "state_id")
        _valid_id(self.subject_ref, "state.subject_ref")
        reject_if(self.predicate not in STATE_PREDICATES, TaskSemanticIRError(f"state predicate 无效：{self.predicate}"))
        if self.source_subgoal_id:
            _valid_id(self.source_subgoal_id, "state.source_subgoal_id")
        _json_value(self.value, "state.value")


@dataclass(frozen=True)
class SemanticSubgoal(ValidatedDataclassWire):
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
        for value in (self.subgoal_id, self.surface_ref, *self.depends_on, *self.constraint_refs, *self.entity_refs,
            *self.desired_state_refs, *self.effect_refs):
            _valid_id(value, "subgoal reference")
        reject_if(self.external_impact not in SUBGOAL_IMPACTS, TaskSemanticIRError(f"subgoal impact 无效：{self.external_impact}"))
        reject_if(self.status not in {'pending', 'active', 'completed', 'blocked', 'skipped'}, TaskSemanticIRError(f"subgoal status 无效：{self.status}"))


@dataclass(frozen=True)
class InputFieldIntent(ValidatedDataclassWire):
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
        reject_if(not isinstance(self.field_label, str) or not isinstance(self.multiline, bool), TaskSemanticIRError("input field metadata 无效。"))


@dataclass(frozen=True)
class TaskSemanticIR(ValidatedDataclassWire):
    task_id: str
    device_id: str
    revision: int
    raw_goal: str
    surfaces: tuple[SurfaceRef, ...]
    entities: tuple[SemanticEntity, ...]
    effects: tuple[EffectIntent, ...]
    constraints: tuple[ConstraintIntent, ...] = ()
    desired_states: tuple[DesiredState, ...] = ()
    subgoals: tuple[SemanticSubgoal, ...] = ()
    input_fields: tuple[InputFieldIntent, ...] = ()
    protocol_version: str = TASK_SEMANTIC_IR_PROTOCOL

    def validate(self) -> None:
        reject_if(self.protocol_version != TASK_SEMANTIC_IR_PROTOCOL, TaskSemanticIRError(f"TaskSemanticIR 协议无效：{self.protocol_version}"))
        _valid_id(self.task_id, "task_id", external=True)
        _valid_id(self.device_id, "device_id", external=True)
        reject_if(isinstance(self.revision, bool) or not isinstance(self.revision, int) or self.revision < 1, TaskSemanticIRError("revision 必须是正整数。"))
        _required_text(self.raw_goal, "raw_goal")
        groups = ((self.surfaces, 'surface_id', 'surface'), (self.entities, 'entity_id', 'entity'), (self.effects,
            'effect_id', 'effect'), (self.constraints, 'constraint_id', 'constraint'), (self.desired_states,
            'state_id', 'state'), (self.subgoals, 'subgoal_id', 'subgoal'), (self.input_fields, 'field_id',
            'input field'))
        maps = {name: _unique(items, attr, name) for items, attr, name in groups}
        for (items, _, _) in groups:
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
            reject_if(set((*effect.target_refs, *effect.payload_refs)) - entity_ids, TaskSemanticIRError(f"EffectIntent 引用未知实体：{effect.effect_id}"))
            reject_if(set(effect.source_subgoal_ids) - subgoal_ids, TaskSemanticIRError(f"EffectIntent 引用未知子目标：{effect.effect_id}"))
        for subgoal in self.subgoals:
            reject_if(
                subgoal.surface_ref not in surface_ids or set(subgoal.constraint_refs) - constraint_ids
                or set(subgoal.entity_refs) - entity_ids or set(subgoal.desired_state_refs) - state_ids
                or set(subgoal.effect_refs) - effect_ids,
                TaskSemanticIRError(f"SemanticSubgoal 引用不存在：{subgoal.subgoal_id}"),
            )
        for state in self.desired_states:
            reject_if(state.subject_ref not in entity_ids | effect_ids | surface_ids, TaskSemanticIRError(f"DesiredState subject_ref 不存在：{state.subject_ref}"))
        for item in self.input_fields:
            reject_if(
                item.payload_ref not in entity_ids or set(item.recipient_refs) - entity_ids
                or set(item.source_subgoal_ids) - subgoal_ids,
                TaskSemanticIRError(f"InputFieldIntent 引用不存在：{item.field_id}"),
            )

@dataclass(frozen=True)
class RiskDecision(ValidatedDataclassWire):
    effect_id: str
    policy: str
    matched_rule: str
    policy_id: str
    policy_version: int

    def validate(self) -> None:
        _valid_id(self.effect_id, "risk.effect_id")
        reject_if(self.policy not in RISK_POLICIES or self.policy_version < 1, TaskSemanticIRError("risk decision 无效。"))
        _required_text(self.matched_rule, "matched_rule")
        _valid_id(self.policy_id, "policy_id")


def _risk_policy_wire() -> dict[str, Any]:
    return {'protocol_version': RISK_POLICY_PROTOCOL, 'policy_id': 'default_low_friction', 'version': 2,
        'confirmation_effect_kinds': sorted(DEFAULT_CONFIRMATION_EFFECT_KINDS), 'overrides': []}


def decide_effect_risk(effect: EffectIntent) -> RiskDecision:
    effect.validate()
    policy = CONFIRMATION_REQUIRED if effect.kind in DEFAULT_CONFIRMATION_EFFECT_KINDS else AUTOMATIC
    decision = RiskDecision(effect.effect_id, policy, f'effect_kind:{effect.kind}', 'default_low_friction', 2)
    decision.validate()
    return decision


def effect_preview_digest(preview: Mapping[str, Any]) -> str:
    return _digest(dict(preview))


@dataclass(frozen=True)
class SemanticRiskAuthorityReport(ValidatedDataclassWire):
    semantic_ir: TaskSemanticIR
    risk_policy: dict[str, Any]
    risk_decisions: tuple[RiskDecision, ...]
    effect_previews: tuple[dict[str, Any], ...]
    source_graph_digest: str
    authoritative_scope: str = "semantic_task_and_risk"
    physical_execution_allowed: bool = False
    protocol_version: str = AUTHORITY_REPORT_PROTOCOL

    def validate(self) -> None:
        reject_if(
            self.protocol_version != AUTHORITY_REPORT_PROTOCOL or self.authoritative_scope != 'semantic_task_and_risk'
            or self.physical_execution_allowed,
            TaskSemanticIRError("semantic authority 元数据无效。"),
        )
        reject_if(not re.fullmatch('[0-9a-f]{64}', self.source_graph_digest), TaskSemanticIRError("source_graph_digest 无效。"))
        self.semantic_ir.validate()
        reject_if(self.risk_policy != _risk_policy_wire(), TaskSemanticIRError("风险策略不是当前唯一批准边界。"))
        effect_ids = {item.effect_id for item in self.semantic_ir.effects}
        reject_if({item.effect_id for item in self.risk_decisions} != effect_ids or {str(item.get('effect_id') or '')
            for item in self.effect_previews} != effect_ids, TaskSemanticIRError("semantic authority 未逐 effect 覆盖。"))
        for item in self.risk_decisions:
            item.validate()
        reject_if(any((set(item) != {'task_id', 'device_id', 'revision', 'effect_id', 'effect_kind', 'targets',
            'payloads', 'policy', 'policy_id', 'policy_version', 'expected_result_texts', 'protocol_version'}
            or item.get('protocol_version') != EFFECT_PREVIEW_PROTOCOL or item.get('policy') not in RISK_POLICIES
            or item.get('policy_id') != 'default_low_friction' or item.get('policy_version') != 2
            for item in self.effect_previews)), TaskSemanticIRError("effect preview 元数据无效。"))


def _source_span(raw_goal: str, value: Any) -> SourceSpan | None:
    if not isinstance(value, str) or not value:
        return None
    start = raw_goal.find(value)
    return SourceSpan(start, start + len(value)) if start >= 0 else None


def _entity_refs(entities: tuple[SemanticEntity, ...], roles: tuple[str, ...]) -> tuple[str, ...]:
    role_set = set(roles)
    return tuple(item.entity_id for item in entities if item.role in role_set)


def _runtime_graph_digest(graph: Any) -> str:
    if hasattr(graph, 'to_dict'):
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
        if action == 'input_verified_text' and re.search('(?:可|可以|能|能够|支持)(?:直接|正常|精确)?\\s*$', prefix, re.I):
            continue
        return True
    return False


def _surface_for_subgoal(subgoal: Any, surfaces: tuple[SurfaceRef, ...]) -> str:
    text = ' '.join((str(getattr(subgoal, 'objective', '')), *map(str, getattr(subgoal, 'constraints', ()) or ()),
        *map(str, getattr(subgoal, 'completion_conditions', ()) or ()))).casefold()
    if _RECENTS.search(text):
        match = next((item.surface_id for item in surfaces if item.kind == "recent_tasks"), "")
        if match:
            return match
    if any((term in text for term in ('主桌面', '桌面', '主页', '手机主屏幕', '系统主屏幕', '主屏幕', 'home screen', 'launcher'))):
        match = next((item.surface_id for item in surfaces if item.kind == "launcher"), "")
        if match:
            return match
    app_matches = [item.surface_id for item in surfaces if item.kind == 'app' and (item.app_id.casefold() in text
        or item.app_name.casefold() in text)]
    if len(app_matches) == 1:
        return app_matches[0]
    current = next((item.surface_id for item in surfaces if item.kind == "current_surface"), "")
    if current:
        return current
    apps = [item.surface_id for item in surfaces if item.kind == "app"]
    return apps[0] if len(apps) == 1 else surfaces[0].surface_id


def _compile_runtime_graph_semantics(graph: Any) -> tuple[TaskSemanticIR, tuple[RiskDecision, ...]]:
    task_id = str(getattr(graph, "task_id", "")).strip()
    device_id = str(getattr(graph, "device_id", "")).strip()
    revision = getattr(graph, "revision", 0)
    goal = getattr(graph, "goal", None)
    raw_goal = str(getattr(graph, "raw_user_goal", "") or getattr(goal, "objective", "") or "").strip()
    raw_entities = getattr(goal, "entities", {}) or {}
    reject_if(not isinstance(raw_entities, Mapping), TaskSemanticIRError("goal.entities 必须是映射。"))
    target_apps = tuple(getattr(goal, "target_apps", ()) or ())
    subgoals = tuple(getattr(graph, "subgoals", ()) or ())
    structured = " ".join(str(getattr(item, "objective", "")) for item in subgoals)
    surfaces: list[SurfaceRef] = []
    explicit_system = not target_apps and str(raw_entities.get("target_surface") or "") in {"device", "system"}
    launcher_requested = bool(re.search(r"主桌面|桌面|主页|手机主屏幕|系统主屏幕|home screen|launcher", raw_goal, re.I))
    if launcher_requested or (explicit_system and re.search('主屏幕|home screen|launcher', structured, re.I)):
        surfaces.append(SurfaceRef("surface_launcher", "launcher"))
    if _RECENTS.search(structured):
        surfaces.append(SurfaceRef("surface_recent_tasks", "recent_tasks"))
    for app in target_apps:
        app_id = _slug(getattr(app, "app_id", ""), "app")
        surfaces.append(SurfaceRef(f"surface_{app_id}", "app", app_id, str(getattr(app, "app_name", "") or app_id)))
    if len(target_apps) > 1 or re.search('当前|眼前|current|foreground', raw_goal, re.I):
        surfaces.append(SurfaceRef("surface_current", "current_surface"))
    if not surfaces:
        kind = str(raw_entities.get("target_surface") or "current_surface")
        kind = kind if kind in SURFACE_KINDS else "current_surface"
        surfaces.append(SurfaceRef(f"surface_{_slug(kind, 'current')}", kind))

    entities: list[SemanticEntity] = []
    field_meta: dict[str, tuple[str, str]] = {}
    recipients = raw_entities.get("recipients")
    if isinstance(recipients, list):
        for (index, value) in enumerate(recipients, 1):
            span = _source_span(raw_goal, value)
            entities.append(SemanticEntity(f'entity_recipient_{index}', 'party', 'recipient', _json_value(value,
                'recipients'), span, 'user_literal' if span else 'planner_context'))
    input_fields = raw_entities.get("input_fields")
    if isinstance(input_fields, list):
        for (index, spec) in enumerate(input_fields, 1):
            if not isinstance(spec, Mapping):
                continue
            field_id = _slug(spec.get("field_id"), f"field_{index}")
            entity_id = f"entity_input_text_{field_id}"
            value = spec.get("text")
            span = _source_span(raw_goal, value)
            entities.append(SemanticEntity(entity_id, 'text', 'input_text', _json_value(value, 'input_fields.text'),
                span, 'user_literal' if span else 'planner_context'))
            field_meta[entity_id] = (field_id, str(spec.get("field_label") or ""))
    for (index, key) in enumerate(sorted(raw_entities, key=str), 1):
        if key in {'recipients', 'input_fields'}:
            continue
        role = _slug(key, f"role_{index}")
        value = raw_entities[key]
        span = _source_span(raw_goal, value)
        entities.append(SemanticEntity(f'entity_{role}_{index}', _ENTITY_TYPES.get(role, 'opaque'), role,
            _json_value(value, f'entities.{role}'), span, 'user_literal' if span else 'planner_context'))
    entity_tuple = tuple(entities)
    by_role: dict[str, list[SemanticEntity]] = {}
    for entity in entity_tuple:
        by_role.setdefault(entity.role, []).append(entity)

    runtime_subgoals = {str(getattr(item, "subgoal_id", "")): item for item in subgoals}
    effects: list[EffectIntent] = []
    for (index, risk) in enumerate(tuple(getattr(graph, 'risk_actions', ()) or ()), 1):
        runtime_id = str(getattr(risk, "risk_id", "") or "")
        kind = str(getattr(risk, "effect_kind", "") or "generic_effect")
        effect_id = f"effect_{_slug(runtime_id, f'effect_{index}')}"
        source_ids = tuple(_slug(item, "subgoal") for item in getattr(risk, "subgoal_ids", ()) or ())
        expected = tuple((str(item).strip() for item in getattr(risk, 'expected_result_texts',
            ()) or () if str(item).strip()))
        if not expected:
            expected = tuple(dict.fromkeys((text for subgoal_id in source_ids for text in map(str,
                getattr(runtime_subgoals.get(subgoal_id), 'completion_conditions', ()) or ()) if text.strip())))
        targets = _entity_refs(entity_tuple, tuple(getattr(risk, 'target_roles', ()) or ()) or _TARGET_ROLES.get(kind,
            ('target',)))
        payloads = _entity_refs(entity_tuple, tuple(getattr(risk, 'payload_roles',
            ()) or ()) or _PAYLOAD_ROLES.get(kind, ('input_text', 'value')))
        effects.append(EffectIntent(effect_id, kind, targets, payloads, source_ids, expected,
            {'runtime_effect_id': runtime_id, 'planner_declared_typed_effect': bool(getattr(risk, 'effect_kind', ''))}))
    effect_tuple = tuple(effects)
    constraints: list[ConstraintIntent] = []
    exact_ids: list[str] = []
    for entity in entity_tuple:
        if entity.authority == 'user_literal':
            constraint_id = f"constraint_exact_{len(exact_ids) + 1}"
            exact_ids.append(constraint_id)
            constraints.append(ConstraintIntent(constraint_id, 'exact_entity', (entity.entity_id,), value=entity.value,
                source_text=str(entity.value) if isinstance(entity.value, str) else '', authoritative=True))
    context_ids: dict[str, list[str]] = {"": []}
    context_specs = [("", str(item).strip()) for item in getattr(graph, "constraints", ()) or () if str(item).strip()]
    context_specs.extend(((str(getattr(subgoal, 'subgoal_id', '')),
        str(item).strip()) for subgoal in subgoals for item in getattr(subgoal, 'constraints',
        ()) or () if str(item).strip()))
    for (owner, text) in context_specs:
        constraint_id = f"constraint_context_{sum(map(len, context_ids.values())) + 1}"
        constraints.append(ConstraintIntent(constraint_id, "planner_context", value=text, source_text=text))
        context_ids.setdefault(owner, []).append(constraint_id)
    action_ids: dict[str, list[str]] = {}
    action_values: dict[str, set[str]] = {}
    has_input_payload = bool(by_role.get("input_text"))
    for subgoal in subgoals:
        subgoal_id = str(getattr(subgoal, "subgoal_id", ""))
        if str(getattr(subgoal, 'external_impact', '')) == 'read_only':
            continue
        objective = str(getattr(subgoal, "objective", ""))
        state_text = " ".join((objective, *map(str, getattr(subgoal, "completion_conditions", ()) or ())))
        card_dismiss = bool(_RECENTS.search(state_text) and _RECENT_CARD.search(state_text)
            and _DISMISS.search(state_text))
        actions: list[tuple[str, str]] = []
        if explicit_system and re.search('回到主屏幕|return\\s+to\\s+(?:the\\s+)?home\\s+screen', objective, re.I):
            actions.append(("home", objective))
        if card_dismiss:
            actions.append(("swipe", objective if _DIRECTION.search(objective) else f"{objective}；系统最近任务卡片向左滑动清除"))
        for (action, pattern) in _ACTION_PATTERNS:
            if action == 'swipe' and card_dismiss:
                continue
            if _positive_match(objective, pattern, action) and (action != 'input_verified_text' or has_input_payload):
                actions.append((action, objective))
        for (action, source_text) in dict(actions).items():
            constraint_id = f"constraint_action_{1 + sum(map(len, action_ids.values()))}"
            constraints.append(ConstraintIntent(constraint_id, 'required_action', value=action, source_text=source_text,
                authoritative=True))
            action_ids.setdefault(subgoal_id, []).append(constraint_id)
            action_values.setdefault(subgoal_id, set()).add(action)

    effect_refs: dict[str, list[str]] = {}
    for effect in effect_tuple:
        for subgoal_id in effect.source_subgoal_ids:
            effect_refs.setdefault(subgoal_id, []).append(effect.effect_id)
    surfaces_tuple = tuple(surfaces)
    desired: list[DesiredState] = []
    desired_refs: dict[str, list[str]] = {}
    verified_input_subjects: dict[str, set[str]] = {}

    def add_state(description: str, owner: str, surface: str) -> None:
        description = description.strip()
        active_effects = effect_refs.get(owner, [])
        input_matches = [item for item in by_role.get('input_text', ()) if isinstance(item.value,
            str) and item.value and (item.value.casefold() in description.casefold()) and _EDITABLE.search(description)]
        if len(active_effects) == 1:
            receipt_only = bool(re.search(r"动作已执行|操作已执行|效果已触发|action executed|effect applied", description, re.I))
            specs = [(active_effects[0], 'effect.applied' if receipt_only else 'effect.result_visible', True)]
        elif input_matches:
            specs = [(item.entity_id, "input.value_equals", item.value) for item in input_matches]
        else:
            specs = [(surface, 'observation.matches_description', description)]
        for (subject, predicate, value) in specs:
            state_id = f"state_{len(desired) + 1}"
            desired.append(DesiredState(state_id, subject, predicate, value, owner))
            desired_refs.setdefault(owner, []).append(state_id)
            if predicate == 'input.value_equals':
                verified_input_subjects.setdefault(owner, set()).add(subject)

    semantic_subgoals: list[SemanticSubgoal] = []
    for subgoal in subgoals:
        subgoal_id = str(getattr(subgoal, "subgoal_id", ""))
        surface = _surface_for_subgoal(subgoal, surfaces_tuple)
        text = ' '.join((str(getattr(subgoal, 'objective', '')), *map(str, getattr(subgoal, 'completion_conditions',
            ()) or ()))).casefold()
        entity_refs = tuple((item.entity_id for item in entity_tuple if isinstance(item.value,
            str) and item.value.strip() and (item.value.strip().casefold() in text)))
        for description in getattr(subgoal, 'completion_conditions', ()) or ():
            if str(description).strip():
                add_state(str(description), subgoal_id, surface)
        semantic_subgoals.append(SemanticSubgoal(subgoal_id, surface, str(getattr(subgoal, 'status', 'pending')),
            str(getattr(subgoal, 'external_impact', 'unknown')), tuple(map(str, getattr(subgoal, 'depends_on',
            ()) or ())), tuple(dict.fromkeys((*exact_ids, *context_ids.get('', ()), *context_ids.get(subgoal_id, ()),
            *action_ids.get(subgoal_id, ())))), entity_refs, tuple(desired_refs.get(subgoal_id, ())),
            tuple(effect_refs.get(subgoal_id, ()))))
    default_surface = surfaces_tuple[0].surface_id
    for condition in getattr(graph, 'completion_conditions', ()) or ():
        description = str(getattr(condition, "description", "") or "").strip()
        if description:
            add_state(description, "", default_surface)

    recipient_refs = tuple(item.entity_id for item in by_role.get("recipient", ()))
    input_entities = tuple(by_role.get("input_text", ()))
    input_contexts = tuple(' '.join((str(getattr(subgoal, 'objective', '')),
        *map(str, getattr(subgoal, 'constraints', ()) or ()), *map(str, getattr(subgoal, 'completion_conditions',
        ()) or ()))).casefold() for subgoal in subgoals)
    typed_fields: list[InputFieldIntent] = []
    for (index, entity) in enumerate(input_entities, 1):
        sources = []
        field_id, label = field_meta.get(entity.entity_id, (f"input_field_{index}", ""))
        for semantic, context in zip(semantic_subgoals, input_contexts):
            actions = action_values.get(semantic.subgoal_id, set())
            owns_typed_action = bool(actions & {"input_verified_text", "clear_verified_text"})
            owns = owns_typed_action and (len(input_entities) == 1 or bool(label and label.casefold()
                in context))
            verifies_exact_value = entity.entity_id in verified_input_subjects.get(semantic.subgoal_id, set())
            if verifies_exact_value or owns:
                sources.append(semantic.subgoal_id)
        typed_fields.append(InputFieldIntent(field_id, entity.entity_id, label, recipient_refs,
            tuple(dict.fromkeys(sources)), isinstance(entity.value, str) and '\n' in entity.value))
    for semantic in semantic_subgoals:
        actions = action_values.get(semantic.subgoal_id, set())
        if 'input_verified_text' not in actions:
            continue
        owners = [item for item in typed_fields if semantic.subgoal_id in item.source_subgoal_ids]
        reject_if(len(owners) != 1, TaskSemanticIRError("input_verified_text 子目标必须且只能绑定一个 InputFieldIntent。"))
    for typed in typed_fields:
        if not typed.multiline or len(typed.source_subgoal_ids) != 1:
            continue
        owner = typed.source_subgoal_ids[0]
        if 'press_enter' in action_values.get(owner, set()):
            continue
        constraint_id = f"constraint_action_{1 + sum(map(len, action_ids.values()))}"
        new_constraint = ConstraintIntent(constraint_id, 'required_action', value='press_enter',
            source_text='multiline typed input field', authoritative=True)
        constraints.append(new_constraint)
        semantic_subgoals = [replace(item, constraint_refs=(*item.constraint_refs,
            constraint_id)) if item.subgoal_id == owner else item for item in semantic_subgoals]

    semantic_ir = TaskSemanticIR(task_id, device_id, revision, raw_goal, surfaces_tuple, entity_tuple, effect_tuple,
        tuple(constraints), tuple(desired), tuple(semantic_subgoals), tuple(typed_fields))
    semantic_ir.validate()
    return semantic_ir, tuple(map(decide_effect_risk, effect_tuple))


def compile_formal_semantic_authority(graph: Any) -> SemanticRiskAuthorityReport:
    reject_if(
        any((not str(getattr(item, 'effect_kind', '') or '') for item in getattr(graph, 'risk_actions', ()) or ())),
        TaskSemanticIRError("正式语义权威只接受由 typed effect_intents 创建的风险条目。"),
    )
    semantic_ir, risk_decisions = _compile_runtime_graph_semantics(graph)
    decisions = {item.effect_id: item for item in risk_decisions}
    entities = {item.entity_id: item for item in semantic_ir.entities}
    def preview(ref: str) -> dict[str, Any]:
        entity = entities[ref]
        return {'entity_ref': ref, 'role': entity.role, 'entity_type': entity.entity_type, 'value': entity.value}

    previews = tuple(({'task_id': semantic_ir.task_id, 'device_id': semantic_ir.device_id,
        'revision': semantic_ir.revision, 'effect_id': effect.effect_id, 'effect_kind': effect.kind,
        'targets': list(map(preview, effect.target_refs)), 'payloads': list(map(preview, effect.payload_refs)),
        'policy': decisions[effect.effect_id].policy, 'policy_id': decisions[effect.effect_id].policy_id,
        'policy_version': decisions[effect.effect_id].policy_version,
        'expected_result_texts': list(effect.expected_result_texts), 'protocol_version': EFFECT_PREVIEW_PROTOCOL}
        for effect in semantic_ir.effects))
    report = SemanticRiskAuthorityReport(semantic_ir, _risk_policy_wire(), risk_decisions, previews,
        _runtime_graph_digest(graph))
    report.validate()
    return report


def apply_formal_semantic_risk_policy(graph: Any, authority: SemanticRiskAuthorityReport) -> Any:
    authority.validate()
    reject_if(authority.source_graph_digest != _runtime_graph_digest(graph), TaskSemanticIRError("正式语义风险权威未绑定当前任务图。"))
    decisions = {effect.attributes.get('runtime_effect_id'): decision for effect,
        decision in zip(authority.semantic_ir.effects, authority.risk_decisions)}
    projected = []
    for risk in getattr(graph, 'risk_actions', ()) or ():
        risk_id = str(getattr(risk, "risk_id", ""))
        decision = decisions.get(risk_id)
        reject_if(decision is None, TaskSemanticIRError(f"正式风险权威遗漏 runtime effect：{risk_id}"))
        projected.append(replace(risk, confirmation_required=decision.policy == CONFIRMATION_REQUIRED))
    active_id = str(getattr(graph, "active_subgoal_id", "") or "")
    active = next((item for item in getattr(graph, 'subgoals', ()) or () if getattr(item, 'subgoal_id',
        '') == active_id), None)
    required = {item.risk_id for item in projected if item.confirmation_required}
    needs_confirmation = bool(active and set(getattr(active, "risk_action_ids", ()) or ()) & required)
    status = str(getattr(graph, "status", ""))
    if status == 'awaiting_confirmation' and (not needs_confirmation):
        status = "ready"
    elif status in {'ready', 'running'} and needs_confirmation:
        status = "awaiting_confirmation"
    return replace(graph, risk_actions=tuple(projected), status=status)
