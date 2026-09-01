from __future__ import annotations

from .validation import ValidatedDataclassWire, dataclass_wire, reject_if
import re
import uuid
from dataclasses import asdict, dataclass, field, replace
from typing import Any

DEEPSEEK_TASK_GRAPH_PROTOCOL_VERSION = "2026-08-20-deepseek-typed-task-graph-v4"
SUBGOAL_EXTERNAL_IMPACTS = frozenset({'read_only', 'navigation_only', 'external_state', 'unknown'})
PLANNER_EFFECT_KINDS = frozenset({'send_message', 'publish_content', 'relationship_change', 'membership_change',
    'data_mutation', 'authentication', 'financial_transaction', 'sensitive_permission_change',
    'irreversible_account_deletion', 'irreversible_data_deletion'})
CONFIRMATION_EFFECT_KINDS = frozenset({'authentication', 'financial_transaction'})
INPUT_OPERATIONS = frozenset({'focus', 'input_verified_text', 'clear_verified_text', 'press_enter'})
PRIMARY_INPUT_FIELD_ID = 'primary_input'
_EXECUTION_CLASS_BY_RUNTIME_IMPACT = {'read_only': 'observe', 'navigation_only': 'navigate',
    'external_state': 'effect', 'unknown': 'unknown'}
GRAPH_STATUSES = frozenset({'ready', 'running', 'awaiting_confirmation', 'completed', 'blocked'})
SUBGOAL_STATUSES = frozenset({'pending', 'active', 'completed', 'blocked', 'skipped'})
ID_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
DEVICE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
TASK_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
TARGET_SURFACES = frozenset({"device", "system", "current_surface"})
MAX_CANONICAL_INPUT_CHARS = 4000
MAX_INPUT_FIELDS = 32
MAX_RECIPIENTS = 32

class TaskGraphError(ValueError):
    pass


@dataclass(frozen=True)
class TargetApp:
    app_id: str
    app_name: str

    def validate(self) -> None:
        reject_if(not ID_PATTERN.fullmatch(self.app_id), TaskGraphError(f"目标 App ID 无效：{self.app_id!r}"))
        _require_text(self.app_name, "target_apps.app_name")


@dataclass(frozen=True)
class GraphGoal:
    objective: str
    target_apps: tuple[TargetApp, ...]
    entities: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _require_text(self.objective, "goal.objective")
        apps = _unique_by_id(self.target_apps, lambda app: app.app_id, "目标 App")
        for app in apps.values():
            app.validate()
        _reject_control_fields(self.entities, "goal.entities")
        target_surface = self.entities.get("target_surface")
        reject_if(target_surface is not None and target_surface not in TARGET_SURFACES, TaskGraphError("goal.entities.target_surface 必须是 device、system 或 current_surface。"))
        input_text = self.entities.get("input_text")
        reject_if(input_text is not None and (not _is_exact_input_text(input_text)), TaskGraphError('goal.entities.input_text 必须为1～4000个逐字输入字符；允许换行但不允许回车控制符。'))
        recipient = self.entities.get("recipient")
        reject_if(recipient is not None and (not _is_single_line_literal(recipient, 100)), TaskGraphError("goal.entities.recipient 必须为1～100个首尾无空白的逐字收件人字符。"))
        recipients = self.entities.get("recipients")
        reject_if(recipient is not None and recipients is not None, TaskGraphError("recipient 与 recipients 只能使用一种表达。"))
        reject_if(
            recipients is not None and (not isinstance(recipients,
            list) or not 1 <= len(recipients) <= MAX_RECIPIENTS or any((not _is_single_line_literal(item,
            100) for item in recipients)) or (len(recipients) != len(set(recipients)))),
            TaskGraphError("goal.entities.recipients 必须为1～32个互不重复的逐字收件人。"),
        )
        input_fields = self.entities.get("input_fields")
        reject_if(input_text is not None and input_fields is not None, TaskGraphError("input_text 与 input_fields 只能使用一种表达。"))
        if input_fields is not None:
            _validate_input_fields(input_fields)


@dataclass(frozen=True)
class CompletionCondition:
    condition_id: str
    description: str
    evidence_required: tuple[str, ...]
    satisfied: bool = False
    evidence: tuple[str, ...] = ()

    def validate(self) -> None:
        _validate_id(self.condition_id, "完成条件 ID")
        _require_text(self.description, "completion_conditions.description")
        _validate_text_list(self.evidence_required, "evidence_required", required=True)
        _validate_text_list(self.evidence, "evidence", required=False)
        reject_if(self.satisfied and (not self.evidence), TaskGraphError(f"已满足的完成条件缺少可见证据：{self.condition_id}"))
        reject_if(not self.satisfied and self.evidence, TaskGraphError(f"未满足的完成条件不能携带完成证据：{self.condition_id}"))


@dataclass(frozen=True)
class RiskAction:
    risk_id: str
    subgoal_ids: tuple[str, ...]
    confirmation_required: bool = True
    effect_kind: str = ""
    target_roles: tuple[str, ...] = ()
    payload_roles: tuple[str, ...] = ()
    expected_result_texts: tuple[str, ...] = ()

    def validate(self) -> None:
        _validate_id(self.risk_id, "风险 ID")
        reject_if(not isinstance(self.confirmation_required, bool), TaskGraphError(f'effect_intents.local_policy.confirmation_required 必须是布尔值：{self.risk_id}'))
        _validate_id_list(self.subgoal_ids, "effect_intents.source_subgoal_ids", required=True)
        if self.effect_kind:
            reject_if(self.effect_kind not in PLANNER_EFFECT_KINDS, TaskGraphError(f'正式效果类型无效：{self.effect_kind}'))
            _validate_id_list(self.target_roles, 'effect_intents.target_entity_roles', required=False)
            _validate_id_list(self.payload_roles, 'effect_intents.payload_entity_roles', required=False)
            _validate_text_list(self.expected_result_texts, 'effect_intents.expected_results', required=True)


@dataclass(frozen=True)
class Subgoal:
    subgoal_id: str
    objective: str
    status: str
    depends_on: tuple[str, ...]
    constraints: tuple[str, ...]
    completion_conditions: tuple[str, ...]
    completion_evidence: tuple[str, ...]
    risk_action_ids: tuple[str, ...]
    external_impact: str
    input_field_id: str = ''
    input_operation: str = ''

    def validate(self) -> None:
        _validate_id(self.subgoal_id, "子目标 ID")
        _require_text(self.objective, "subgoals.objective")
        reject_if(self.status not in SUBGOAL_STATUSES, TaskGraphError(f"子目标状态无效：{self.status}"))
        _validate_id_list(self.depends_on, "subgoals.depends_on", required=False)
        _validate_text_list(self.constraints, "subgoals.constraints", required=False)
        _validate_text_list(self.completion_conditions, 'subgoals.completion_conditions', required=True)
        _validate_text_list(self.completion_evidence, 'subgoals.completion_evidence', required=False)
        _validate_id_list(self.risk_action_ids, 'subgoals.effect_ids', required=False)
        reject_if(self.external_impact not in SUBGOAL_EXTERNAL_IMPACTS, TaskGraphError(f'子目标执行类别无效：{self.external_impact}'))
        reject_if(bool(self.input_field_id) != bool(self.input_operation), TaskGraphError(
            f'子目标输入绑定必须同时提供 input_field_id 与 input_operation：{self.subgoal_id}'))
        if self.input_field_id:
            _validate_id(self.input_field_id, 'subgoals.input_field_id')
            reject_if(self.input_operation not in INPUT_OPERATIONS, TaskGraphError(
                f'子目标 input_operation 无效：{self.input_operation}'))
        reject_if(self.status == 'completed' and (not self.completion_evidence), TaskGraphError(f"已完成子目标缺少完成证据：{self.subgoal_id}"))
        reject_if(self.status != 'completed' and self.completion_evidence, TaskGraphError(f"未完成子目标不能携带完成证据：{self.subgoal_id}"))


def _goal_wire(goal: GraphGoal) -> dict[str, Any]:
    return {'objective': goal.objective, 'target_apps': [asdict(app) for app in goal.target_apps],
        'entities': dict(goal.entities)}


def _condition_wire(condition: CompletionCondition) -> dict[str, Any]:
    return dataclass_wire(condition)


def _effect_wire(effect: RiskAction, *, local_policy: bool) -> dict[str, Any]:
    value = {'effect_id': effect.risk_id, 'kind': effect.effect_kind, 'target_entity_roles': list(effect.target_roles),
        'payload_entity_roles': list(effect.payload_roles), 'source_subgoal_ids': list(effect.subgoal_ids),
        'expected_results': list(effect.expected_result_texts)}
    if local_policy:
        value['local_policy'] = {'effect_id': effect.risk_id, 'confirmation_required': effect.confirmation_required,
            'policy_level': 'high' if effect.confirmation_required else 'low'}
    return value


def _subgoal_wire(subgoal: Subgoal) -> dict[str, Any]:
    value = {'subgoal_id': subgoal.subgoal_id, 'objective': subgoal.objective, 'status': subgoal.status,
        'depends_on': list(subgoal.depends_on), 'constraints': list(subgoal.constraints),
        'completion_conditions': list(subgoal.completion_conditions),
        'completion_evidence': list(subgoal.completion_evidence), 'effect_ids': list(subgoal.risk_action_ids),
        'execution_class': _EXECUTION_CLASS_BY_RUNTIME_IMPACT[subgoal.external_impact]}
    if subgoal.input_field_id:
        value['input_field_id'] = subgoal.input_field_id
        value['input_operation'] = subgoal.input_operation
    return value


@dataclass(frozen=True)
class DynamicTaskGraph:
    task_id: str
    device_id: str
    revision: int
    status: str
    goal: GraphGoal
    constraints: tuple[str, ...]
    completion_conditions: tuple[CompletionCondition, ...]
    risk_actions: tuple[RiskAction, ...]
    subgoals: tuple[Subgoal, ...]
    active_subgoal_id: str | None
    clarification_questions: tuple[str, ...] = ()
    protocol_version: str = DEEPSEEK_TASK_GRAPH_PROTOCOL_VERSION
    raw_user_goal: str = field(default="", repr=False, compare=False)

    def validate(self) -> None:
        reject_if(self.protocol_version != DEEPSEEK_TASK_GRAPH_PROTOCOL_VERSION, TaskGraphError(f"任务图协议版本无效：{self.protocol_version}"))
        _validate_task_id(self.task_id)
        _validate_device_id(self.device_id)
        reject_if(isinstance(self.revision, bool) or not isinstance(self.revision, int) or self.revision < 1, TaskGraphError("任务图 revision 必须是正整数。"))
        reject_if(self.status not in GRAPH_STATUSES, TaskGraphError(f"任务图状态无效：{self.status}"))
        self.goal.validate()
        reject_if(
            self.status != 'blocked' and (not self.goal.target_apps) and (self.goal.entities.get('target_surface') not
            in TARGET_SURFACES),
            TaskGraphError("可推进任务图必须声明目标 App，或声明 device/system/current_surface 目标表面。"),
        )
        _validate_text_list(self.constraints, "constraints", required=False)
        _validate_text_list(self.clarification_questions, "clarification_questions", required=False)

        conditions = _unique_by_id(self.completion_conditions, lambda item: item.condition_id, "完成条件")
        reject_if(not conditions, TaskGraphError("任务图至少需要一个全局完成条件。"))
        risks = _unique_by_id(self.risk_actions, lambda item: item.risk_id, "风险")
        subgoals = _unique_by_id(self.subgoals, lambda item: item.subgoal_id, "子目标")
        for item in (*conditions.values(), *risks.values()):
            item.validate()
        for subgoal in subgoals.values():
            subgoal.validate()
            reject_if(subgoal.subgoal_id in subgoal.depends_on, TaskGraphError(f"子目标不能依赖自身：{subgoal.subgoal_id}"))
            _reject_missing_refs(subgoal.depends_on, subgoals, f'子目标 {subgoal.subgoal_id} 依赖不存在节点：')
            _reject_missing_refs(subgoal.risk_action_ids, risks, f'子目标 {subgoal.subgoal_id} 引用不存在风险：')
        _validate_subgoal_input_bindings(self.goal.entities, tuple(subgoals.values()))
        for risk in risks.values():
            _reject_missing_refs(risk.subgoal_ids, subgoals, f'风险 {risk.risk_id} 引用不存在子目标：')
            for subgoal_id in risk.subgoal_ids:
                reject_if(risk.risk_id not in subgoals[subgoal_id].risk_action_ids, TaskGraphError(f"风险与子目标引用不对称：{risk.risk_id} / {subgoal_id}"))
        _reject_dependency_cycles(subgoals)

        active = [item.subgoal_id for item in self.subgoals if item.status == "active"]
        if self.status in {'ready', 'running', 'awaiting_confirmation'}:
            reject_if(len(active) != 1 or self.active_subgoal_id != active[0], TaskGraphError("可推进任务图必须且只能有一个活动子目标。"))
            active_node = subgoals[active[0]]
            unfinished_dependencies = [dependency for dependency
                in active_node.depends_on if subgoals[dependency].status != 'completed']
            reject_if(unfinished_dependencies, TaskGraphError("活动子目标存在未完成依赖：" + ", ".join(unfinished_dependencies)))
            confirmation_risk_ids = {risk_id for risk_id in active_node.risk_action_ids if risks[
                risk_id].confirmation_required}
            reject_if(self.status == 'awaiting_confirmation' and (not confirmation_risk_ids), TaskGraphError("等待确认状态必须关联当前子目标的效果意图。"))
            reject_if(confirmation_risk_ids and self.status != 'awaiting_confirmation', TaskGraphError("本地风险策略要求确认的子目标必须等待用户确认。"))
            reject_if(
                self.status == 'awaiting_confirmation' and active_node.external_impact not in {'external_state',
                'unknown'},
                TaskGraphError("等待确认状态只能用于外部状态或未知影响子目标。"),
            )
        elif active or self.active_subgoal_id is not None:
            raise TaskGraphError("完成或阻塞任务图不能保留活动子目标。")

        if self.status == 'completed':
            reject_if(not all((item.satisfied for item in self.completion_conditions)), TaskGraphError("任务完成必须满足全部全局完成条件。"))
            reject_if(any((item.status in {'pending', 'active', 'blocked'} for item in self.subgoals)), TaskGraphError("任务完成时不能保留未决子目标。"))
        reject_if(
            self.status == 'blocked' and (not self.clarification_questions)
            and (not any((item.status == 'blocked' for item in self.subgoals))),
            TaskGraphError("阻塞任务图必须说明澄清问题或阻塞子目标。"),
        )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        subgoals = [_subgoal_wire(item) for item in self.subgoals]
        value = {'protocol_version': self.protocol_version, 'task_id': self.task_id, 'device_id': self.device_id,
            'revision': self.revision, 'status': self.status, 'goal': _goal_wire(self.goal),
            'constraints': list(self.constraints), 'completion_conditions': [_condition_wire(item) for item
            in self.completion_conditions], 'effect_intents': [_effect_wire(item,
            local_policy=True) for item in self.risk_actions], 'subgoals': subgoals,
            'active_subgoal_id': self.active_subgoal_id, 'clarification_questions': list(self.clarification_questions)}
        value['current_subgoal'] = next((item for item in subgoals if item['subgoal_id'] == self.active_subgoal_id),
            None)
        return value

    def active_subgoal(self) -> Subgoal | None:
        self.validate()
        if self.active_subgoal_id is None:
            return None
        return next(item for item in self.subgoals if item.subgoal_id == self.active_subgoal_id)

    def to_qwen_context(self, *, confirmed_effect_ids: tuple[str, ...]=(), confirmed_task_id: str | None=None,
        confirmed_device_id: str | None=None, confirmed_subgoal_id: str | None=None,
        confirmed_revision: int | None=None) -> dict[str, Any]:
        """Expose only the current high-level target and safety context to Qwen."""

        value = self.to_dict()
        current = value["current_subgoal"]
        current_effect_ids = set(current["effect_ids"] if current else [])
        current_context = dict(current) if current else None
        if current_context is not None:
            current_context.pop("effect_ids", None)
        effect_by_id = {item["effect_id"]: item for item in value["effect_intents"]}
        confirmation_effect_ids = {effect_id for effect_id
            in current_effect_ids if effect_by_id[effect_id]['local_policy']['confirmation_required']}
        confirmed = set(confirmed_effect_ids)
        scope = (confirmed_task_id, confirmed_device_id, confirmed_subgoal_id, confirmed_revision)
        expected_scope = (self.task_id, self.device_id, self.active_subgoal_id, self.revision)
        if confirmed and scope != expected_scope:
            labels = ("task_id", "device_id", "current_subgoal", "revision")
            mismatch = next((label for label, actual, expected in zip(labels, scope,
                expected_scope) if actual != expected))
            raise TaskGraphError(f"确认记录 {mismatch} 不匹配，禁止跨作用域复用。")
        reject_if(not confirmed and any((item is not None for item in scope)), TaskGraphError("确认作用域不能脱离 confirmed_effect_ids 单独提供。"))
        unknown_confirmations = confirmed - confirmation_effect_ids
        reject_if(unknown_confirmations, TaskGraphError("确认记录不属于 current_subgoal：" + ", ".join(sorted(unknown_confirmations))))
        confirmation_required = bool(confirmation_effect_ids)
        confirmation_granted = confirmation_required and confirmation_effect_ids.issubset(confirmed)
        automatic_external_allowed = bool(current and current['execution_class'] == 'effect'
            and (not confirmation_required))
        return {
            "protocol_version": self.protocol_version,
            "task_id": self.task_id,
            "device_id": self.device_id,
            "revision": self.revision,
            "task_status": self.status,
            "goal": value["goal"],
            "global_constraints": value["constraints"],
            "goal_completion_conditions": value["completion_conditions"],
            "current_subgoal": current_context,
            "current_execution_class": current["execution_class"] if current else None,
            "effect_intents": [
                item for item in value["effect_intents"] if item["effect_id"] in current_effect_ids
            ],
            "effect_gate": {
                "state": (
                    "confirmed" if confirmation_granted else
                    "awaiting_confirmation" if confirmation_required else "not_required"
                ),
                "scope": {
                    "task_id": self.task_id,
                    "device_id": self.device_id,
                    "revision": self.revision,
                    "subgoal_id": self.active_subgoal_id,
                },
                "effect_action_allowed": confirmation_granted or automatic_external_allowed,
            },
        }


def build_exact_input_task_graph(raw_goal: str, *, exact_input_text: str, device_id: str,
    task_id: str | None=None) -> DynamicTaskGraph:
    """Build a typed graph for one explicitly authorized input value."""
    goal_text = str(raw_goal or "").strip()
    canonical = str(exact_input_text or "")
    reject_if(not goal_text, TaskGraphError("用户目标不能为空。"))
    reject_if(not _is_exact_input_text(canonical), TaskGraphError('exact_input_text 必须为1～4000个逐字输入字符；允许换行但不允许回车控制符。'))
    return _build_exact_graph(goal_text=goal_text, device_id=device_id, task_id=task_id, objective='使当前唯一输入框内容精确等于授权文字',
        entities={'target_surface': 'current_surface', 'target_ui_label': '当前唯一输入框', 'input_text': canonical},
        constraints=('不要发送或提交',), condition_id='exact_input_value', completion='当前唯一输入框内容与授权文字逐字一致',
        evidence='输入框中可见的完整文字', subgoal_id='input_exact_text', subgoal_objective='在当前唯一输入框中逐字输入授权文字')


def build_exact_action_task_graph(raw_goal: str, *, action_kind: str, target_label: str='', device_id: str,
    task_id: str | None=None) -> DynamicTaskGraph:
    """Build one locally authorized navigation action without model planning."""

    goal_text = str(raw_goal or "").strip()
    resolved_action = str(action_kind or "").strip()
    label = str(target_label or "").strip()
    reject_if(not goal_text, TaskGraphError("用户目标不能为空。"))
    reject_if(resolved_action not in {'back', 'home', 'open_recent_apps', 'tap_semantic'}, TaskGraphError('exact_action_kind 只允许 back、home、open_recent_apps 或 tap_semantic。'))
    reject_if(resolved_action == 'tap_semantic' and (not label), TaskGraphError("tap_semantic 直推必须提供 exact_target_label。"))
    reject_if(resolved_action != 'tap_semantic' and label, TaskGraphError('back/home/open_recent_apps 直推不得携带 exact_target_label。'))
    reject_if(len(label) > 120 or '\n' in label or '\r' in label, TaskGraphError("exact_target_label 必须为不超过120字符的单行文字。"))
    objective_by_action = {'back': '按一次返回键', 'home': '回到系统主屏幕', 'open_recent_apps': '打开系统最近任务页面',
        'tap_semantic': '点击当前画面中的目标控件'}
    entities: dict[str, Any] = {"target_surface": "current_surface"}
    if label:
        entities["target_ui_label"] = label
    completion = "动作后出现新的稳定画面"
    graph = _build_exact_graph(goal_text=goal_text, device_id=device_id, task_id=task_id,
        objective=objective_by_action[resolved_action], entities=entities, constraints=(),
        condition_id='action_completed', completion=completion, evidence='动作后的稳定画面',
        subgoal_id=f'exact_{resolved_action}', subgoal_objective=objective_by_action[resolved_action])
    return graph


def _build_exact_graph(*, goal_text: str, device_id: str, task_id: str | None, objective: str, entities: dict[str, Any],
    constraints: tuple[str, ...], condition_id: str, completion: str, evidence: str, subgoal_id: str,
    subgoal_objective: str) -> DynamicTaskGraph:
    _validate_device_id(device_id)
    resolved_task_id = task_id or uuid.uuid4().hex
    _validate_task_id(resolved_task_id)
    graph = DynamicTaskGraph(task_id=resolved_task_id, device_id=device_id, revision=1, status='ready',
        goal=GraphGoal(objective, (), entities), constraints=constraints,
        completion_conditions=(CompletionCondition(condition_id, completion, (evidence,)),), risk_actions=(),
        subgoals=(Subgoal(subgoal_id=subgoal_id, objective=subgoal_objective, status='active', depends_on=(),
        constraints=constraints, completion_conditions=(completion,), completion_evidence=(), risk_action_ids=(),
        external_impact='navigation_only',
        input_field_id=PRIMARY_INPUT_FIELD_ID if 'input_text' in entities else '',
        input_operation='input_verified_text' if 'input_text' in entities else ''),),
        active_subgoal_id=subgoal_id, raw_user_goal=goal_text)
    graph.validate()
    return graph


def _graph_from_payload(payload: dict[str, Any], *, task_id: str, device_id: str, revision: int,
    raw_user_goal: str) -> DynamicTaskGraph:
    """Parse model-owned plan definitions and derive the only runtime state locally."""

    raw_payload = _expect_dict(payload, '任务计划')
    plan_keys = {'goal', 'constraints', 'completion_conditions', 'effect_intents', 'subgoals',
        'clarification_questions'}
    # Retired runtime copies are deliberately not read, repaired or compared.  Projecting
    # the definition fields keeps an old extra field from regaining veto power.
    definition = {key: raw_payload[key] for key in plan_keys if key in raw_payload}
    _expect_keys(definition, plan_keys, '任务计划')
    raw_goal = _definition_object(definition.get("goal"), {"objective", "target_apps", "entities"}, "goal")
    target_apps = tuple((_target_app_from_payload(item) for item in _expect_list(raw_goal.get('target_apps'),
        'goal.target_apps')))
    entities = dict(_expect_dict(raw_goal.get("entities"), "goal.entities"))
    entities = _normalize_unique_input_newline_escapes(entities, raw_user_goal=raw_user_goal)
    optional_input_text = entities.get("input_text")
    if optional_input_text is None or optional_input_text == '':
        entities.pop("input_text", None)
    goal = GraphGoal(objective=_require_text(raw_goal.get('objective'), 'goal.objective'), target_apps=target_apps,
        entities=entities)
    goal.validate()
    raw_subgoals = tuple((_subgoal_from_payload(item) for item
        in _expect_list(definition.get('subgoals'), 'subgoals')))
    raw_subgoals = _bind_single_step_input_structurally(raw_subgoals, entities)
    subgoals_by_id = {item.subgoal_id: item for item in raw_subgoals}
    reject_if(len(subgoals_by_id) != len(raw_subgoals), TaskGraphError("子目标 ID 重复。"))
    for subgoal in raw_subgoals:
        subgoal.validate()
    completion_conditions = tuple((_condition_from_payload(item) for item
        in _expect_list(definition.get('completion_conditions'), 'completion_conditions')))
    raw_effects = tuple(_expect_list(definition.get('effect_intents'), 'effect_intents'))
    effects = tuple((_effect_from_payload(item, subgoals=subgoals_by_id, entities=entities)
        for item in raw_effects))
    effects_by_id = {item.risk_id: item for item in effects}
    reject_if(len(effects_by_id) != len(effects), TaskGraphError("effect_intents.effect_id 重复。"))
    linked_subgoals = _derive_subgoal_effect_links(raw_subgoals, effects)
    constraints = _text_tuple(definition.get('constraints'), 'constraints')
    graph = DynamicTaskGraph(task_id=task_id, device_id=device_id, revision=revision, status='ready', goal=goal,
        constraints=constraints,
        completion_conditions=completion_conditions, risk_actions=effects, subgoals=linked_subgoals,
        active_subgoal_id=None,
        clarification_questions=_text_tuple(definition.get('clarification_questions'), 'clarification_questions'),
        raw_user_goal=raw_user_goal)
    return _activate_runtime_frontier(graph, initial=True)


def _normalize_unique_input_newline_escapes(entities: dict[str, Any], *, raw_user_goal: str) -> dict[str, Any]:
    """Repair ``\\n`` only when its LF form occurs verbatim in the untouched user goal."""

    def normalized(value: Any) -> Any:
        if not isinstance(value, str) or '\\n' not in value or value in raw_user_goal:
            return value
        candidate = value.replace("\\n", "\n")
        return candidate if candidate in raw_user_goal else value

    result = dict(entities)
    if 'input_text' in result:
        result["input_text"] = normalized(result["input_text"])
    raw_fields = result.get("input_fields")
    if isinstance(raw_fields, list):
        normalized_fields: list[Any] = []
        for item in raw_fields:
            if isinstance(item, dict) and 'text' in item:
                normalized_fields.append({**item, 'text': normalized(item.get('text'))})
            else:
                normalized_fields.append(item)
        result["input_fields"] = normalized_fields
    return result


def _target_app_from_payload(value: Any) -> TargetApp:
    item = _expect_dict(value, "target_apps[]")
    _expect_keys(item, {"app_id", "app_name"}, "target_apps[]")
    return TargetApp(app_id=str(item.get('app_id') or '').strip().lower(), app_name=_require_text(item.get('app_name'),
        'target_apps.app_name'))


def _condition_from_payload(value: Any) -> CompletionCondition:
    item = _definition_object(value, {'condition_id', 'description', 'evidence_required'},
        'completion_conditions[]')
    return CompletionCondition(condition_id=str(item.get('condition_id') or '').strip().lower(),
        description=_require_text(item.get('description'), 'completion_conditions.description'),
        evidence_required=_text_tuple(item.get('evidence_required'), 'evidence_required'))


def _effect_from_payload(value: Any, *, subgoals: dict[str, Subgoal], entities: dict[str, Any]) -> RiskAction:
    item = _definition_object(value, {'effect_id', 'kind', 'target_entity_roles', 'payload_entity_roles',
        'source_subgoal_ids'}, 'effect_intents[]')
    effect_id = str(item.get("effect_id") or "").strip().lower()
    _validate_id(effect_id, "effect_intents.effect_id")
    kind = str(item.get("kind") or "").strip().lower()
    reject_if(kind not in PLANNER_EFFECT_KINDS, TaskGraphError(f"正式效果类型无效：{kind}"))
    source_subgoal_ids = _id_tuple(item.get('source_subgoal_ids'), 'effect_intents.source_subgoal_ids')
    missing_subgoals = set(source_subgoal_ids) - set(subgoals)
    reject_if(missing_subgoals, TaskGraphError('effect_intents 引用不存在子目标：' + ', '.join(sorted(missing_subgoals))))
    target_roles = _id_tuple(item.get('target_entity_roles'), 'effect_intents.target_entity_roles')
    payload_roles = _id_tuple(item.get('payload_entity_roles'), 'effect_intents.payload_entity_roles')
    available_roles = set(entities)
    if isinstance(entities.get('recipients'), list):
        available_roles.add("recipient")
    if isinstance(entities.get('input_fields'), list):
        available_roles.add("input_text")
    missing_roles = (set(target_roles) | set(payload_roles)) - available_roles
    reject_if(missing_roles, TaskGraphError('effect_intents 引用不存在的 goal.entities 角色：' + ', '.join(sorted(missing_roles))))
    expected_results = tuple(condition for subgoal_id in source_subgoal_ids
        for condition in subgoals[subgoal_id].completion_conditions)
    reject_if(not expected_results, TaskGraphError('effect 子目标必须包含至少一个正向完成条件。'))
    return RiskAction(risk_id=effect_id, subgoal_ids=source_subgoal_ids,
        confirmation_required=kind in CONFIRMATION_EFFECT_KINDS,
        effect_kind=kind, target_roles=target_roles, payload_roles=payload_roles,
        expected_result_texts=expected_results)


def _subgoal_from_payload(value: Any) -> Subgoal:
    raw = _expect_dict(value, 'subgoals[]')
    required = {'subgoal_id', 'objective', 'depends_on', 'constraints', 'completion_conditions'}
    allowed = required | {'input_field_id', 'input_operation'}
    retired_runtime_copies = {'status', 'completion_evidence', 'effect_ids', 'execution_class'}
    unexpected = set(raw) - allowed - retired_runtime_copies
    reject_if(unexpected, TaskGraphError('subgoals[] 包含协议外字段：' + ', '.join(sorted(unexpected))))
    reject_if(required - set(raw), TaskGraphError('subgoals[] 缺少字段：' + ', '.join(sorted(required - set(raw)))))
    item = {key: raw[key] for key in allowed if key in raw}
    field_id = str(item.get('input_field_id') or '').strip().lower()
    operation = str(item.get('input_operation') or '').strip().lower()
    return Subgoal(subgoal_id=str(item.get('subgoal_id') or '').strip().lower(),
        objective=_require_text(item.get('objective'), 'subgoals.objective'),
        status='pending', depends_on=_id_tuple(item.get('depends_on'),
        'subgoals.depends_on'), constraints=_text_tuple(item.get('constraints'), 'subgoals.constraints'),
        completion_conditions=_text_tuple(item.get('completion_conditions'), 'subgoals.completion_conditions'),
        completion_evidence=(), risk_action_ids=(),
        external_impact='navigation_only', input_field_id=field_id, input_operation=operation)


def _definition_object(value: Any, keys: set[str], path: str) -> dict[str, Any]:
    raw = _expect_dict(value, path)
    projected = {key: raw[key] for key in keys if key in raw}
    _expect_keys(projected, keys, path)
    return projected


def _derive_subgoal_effect_links(subgoals: tuple[Subgoal, ...], effects: tuple[RiskAction, ...]) -> tuple[Subgoal, ...]:
    """Derive the sole runtime effect relationship from typed effect intents."""

    links: dict[str, list[str]] = {item.subgoal_id: [] for item in subgoals}
    for effect in effects:
        for subgoal_id in effect.subgoal_ids:
            links[subgoal_id].append(effect.risk_id)
    linked: list[Subgoal] = []
    for subgoal in subgoals:
        effect_ids = tuple(links[subgoal.subgoal_id])
        linked.append(replace(subgoal, risk_action_ids=effect_ids,
            external_impact='external_state' if effect_ids else 'navigation_only'))
    return tuple(linked)


def _bind_single_step_input_structurally(subgoals: tuple[Subgoal, ...], entities: dict[str, Any]) -> tuple[Subgoal, ...]:
    """Supply the one unambiguous field binding without reading subgoal prose."""

    if len(subgoals) != 1 or subgoals[0].input_field_id:
        return subgoals
    if isinstance(entities.get('input_text'), str):
        return (replace(subgoals[0], input_field_id=PRIMARY_INPUT_FIELD_ID,
            input_operation='input_verified_text'),)
    raw_fields = entities.get('input_fields')
    fields = [item for item in raw_fields if isinstance(item, dict)] if isinstance(raw_fields, list) else []
    if len(fields) != 1:
        return subgoals
    field = fields[0]
    operation = 'clear_verified_text' if field.get('target_only') is True else 'input_verified_text'
    return (replace(subgoals[0], input_field_id=str(field.get('field_id') or ''), input_operation=operation),)


def _activate_runtime_frontier(graph: DynamicTaskGraph, *, initial: bool) -> DynamicTaskGraph:
    subgoals = {item.subgoal_id: item for item in graph.subgoals}
    if not subgoals:
        # An omitted model subgoal is not proof that the user's goal is ambiguous.
        # The validated goal, device scope and positive completion conditions already
        # contain enough high-level context for Qwen to choose the next action from a
        # fresh screenshot.  Keep model clarification text as diagnostics instead of
        # granting it a second runtime stop authority.
        derived = Subgoal(subgoal_id='goal', objective=graph.goal.objective, status='pending', depends_on=(),
            constraints=graph.constraints,
            completion_conditions=tuple(item.description for item in graph.completion_conditions),
            completion_evidence=(), risk_action_ids=(), external_impact='navigation_only')
        graph = replace(graph,
            subgoals=_bind_single_step_input_structurally((derived,), graph.goal.entities))
        subgoals = {item.subgoal_id: item for item in graph.subgoals}
    for item in subgoals.values():
        _reject_missing_refs(item.depends_on, subgoals, f'子目标 {item.subgoal_id} 依赖不存在节点：')
    _reject_dependency_cycles(subgoals)
    completed_ids = {item.subgoal_id for item in graph.subgoals if item.status == 'completed'}
    pending = [item for item in graph.subgoals if item.status not in {'completed', 'skipped'}]
    if not pending:
        return _finalize_runtime_graph(replace(graph, status='running', active_subgoal_id=None))
    current = next((item for item in pending if set(item.depends_on).issubset(completed_ids)), None)
    reject_if(current is None, TaskGraphError('任务计划没有依赖已满足的可执行子目标。'))
    projected = tuple(replace(item, status='active') if item.subgoal_id == current.subgoal_id
        else replace(item, status='pending', completion_evidence=()) if item.status != 'completed' else item
        for item in graph.subgoals)
    confirmation_ids = {risk.risk_id for risk in graph.risk_actions if risk.confirmation_required}
    status = ('awaiting_confirmation' if confirmation_ids.intersection(current.risk_action_ids)
        else 'ready' if initial else 'running')
    return replace(graph, status=status, subgoals=projected,
        active_subgoal_id=current.subgoal_id)


def _finalize_runtime_graph(graph: DynamicTaskGraph) -> DynamicTaskGraph:
    evidence = tuple(dict.fromkeys(ref for item in reversed(graph.subgoals)
        if item.status == 'completed' for ref in item.completion_evidence))[:3]
    reject_if(not evidence, TaskGraphError('任务完成缺少当前运行事实。'))
    conditions = tuple(condition if condition.satisfied else replace(condition, satisfied=True, evidence=evidence)
        for condition in graph.completion_conditions)
    return replace(graph, status='completed', completion_conditions=conditions, active_subgoal_id=None)


def complete_active_subgoal(graph: DynamicTaskGraph, *, evidence: tuple[str, ...]) -> DynamicTaskGraph:
    """Advance the sole local plan frontier from Qwen's current-scene finish."""

    graph.validate()
    current = graph.active_subgoal()
    normalized_evidence = tuple(dict.fromkeys(str(item).strip() for item in evidence if str(item).strip()))[:3]
    reject_if(current is None or not normalized_evidence,
        TaskGraphError("当前scene finish缺少活动子目标或可见证据。"))
    subgoals = tuple(replace(item, status='completed', completion_evidence=normalized_evidence)
        if item.subgoal_id == current.subgoal_id else item for item in graph.subgoals)
    advanced = _activate_runtime_frontier(replace(graph, revision=graph.revision + 1,
        status='running', subgoals=subgoals, active_subgoal_id=None), initial=False)
    if advanced.status != 'completed':
        active = next(item for item in advanced.subgoals if item.subgoal_id == advanced.active_subgoal_id)
        confirmation_ids = {risk.risk_id for risk in advanced.risk_actions if risk.confirmation_required}
        if confirmation_ids.intersection(active.risk_action_ids):
            advanced = replace(advanced, status='awaiting_confirmation')
    advanced.validate()
    return advanced


def _reject_dependency_cycles(subgoals: dict[str, Subgoal]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(subgoal_id: str) -> None:
        reject_if(subgoal_id in visiting, TaskGraphError(f"子目标依赖形成环：{subgoal_id}"))
        if subgoal_id in visited:
            return
        visiting.add(subgoal_id)
        for dependency in subgoals[subgoal_id].depends_on:
            visit(dependency)
        visiting.remove(subgoal_id)
        visited.add(subgoal_id)

    for subgoal_id in subgoals:
        visit(subgoal_id)


def _unique_by_id(values: tuple[Any, ...], key: Any, label: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value in values:
        item_id = key(value)
        reject_if(item_id in result, TaskGraphError(f"{label} ID 重复：{item_id}"))
        result[item_id] = value
    return result


def _reject_missing_refs(values: tuple[str, ...], available: dict[str, Any], message: str) -> None:
    missing = set(values) - set(available)
    reject_if(missing, TaskGraphError(message + ", ".join(sorted(missing))))


def _expect_keys(value: dict[str, Any], allowed: set[str], path: str) -> None:
    unexpected = set(value) - allowed
    reject_if(unexpected, TaskGraphError(f'{path} 包含协议外字段：' + ', '.join(sorted((str(item) for item in unexpected)))))
    missing = allowed - set(value)
    reject_if(missing, TaskGraphError(f'{path} 缺少字段：' + ', '.join(sorted(missing))))


def _expect_dict(value: Any, path: str) -> dict[str, Any]:
    reject_if(not isinstance(value, dict), TaskGraphError(f"{path} 必须是对象。"))
    return value


def _expect_list(value: Any, path: str) -> list[Any]:
    reject_if(not isinstance(value, list), TaskGraphError(f"{path} 必须是数组。"))
    return value


def _is_single_line_literal(value: Any, max_length: int, *, allow_empty: bool=False) -> bool:
    return bool(isinstance(value, str) and (allow_empty or value) and (len(value) <= max_length)
        and (value == value.strip()) and ('\n' not in value) and ('\r' not in value))


def _is_exact_input_text(value: Any) -> bool:
    return bool(isinstance(value, str) and value and (len(value) <= MAX_CANONICAL_INPUT_CHARS) and ('\r' not in value))


def _validate_input_fields(value: Any) -> None:
    reject_if(not isinstance(value, list) or not 1 <= len(value) <= MAX_INPUT_FIELDS, TaskGraphError("goal.entities.input_fields 必须为1～32个输入字段。"))
    field_ids: set[str] = set()
    labels: set[str] = set()
    for (index, item) in enumerate(value):
        allowed_shapes = ({"field_id", "text"}, {"field_id", "field_label", "text"},
            {"field_id", "text", "target_only"}, {"field_id", "field_label", "text", "target_only"})
        reject_if(not isinstance(item, dict) or set(item) not in allowed_shapes, TaskGraphError(
            f'goal.entities.input_fields[{index}] 只允许 field_id/可选field_label/text/可选target_only。'))
        field_id = item.get("field_id")
        label = item.get("field_label", "")
        reject_if(not isinstance(field_id, str) or not ID_PATTERN.fullmatch(field_id), TaskGraphError(f"goal.entities.input_fields[{index}].field_id 无效。"))
        reject_if(field_id in field_ids, TaskGraphError("goal.entities.input_fields.field_id 重复。"))
        field_ids.add(field_id)
        reject_if(not _is_single_line_literal(label, 120, allow_empty=True), TaskGraphError(f"goal.entities.input_fields[{index}].field_label 无效。"))
        folded = label.casefold()
        reject_if(label and folded in labels, TaskGraphError("goal.entities.input_fields.field_label 重复。"))
        labels.add(folded)
        target_only = item.get('target_only') is True
        reject_if('target_only' in item and not target_only, TaskGraphError(
            f'goal.entities.input_fields[{index}].target_only 只能为 true。'))
        text = item.get('text')
        reject_if(not (_is_exact_input_text(text) or (target_only and text == '')), TaskGraphError(
            f"goal.entities.input_fields[{index}].text 必须为1～4000字符，或在 target_only=true 时为空。"))


def _validate_subgoal_input_bindings(entities: dict[str, Any], subgoals: tuple[Subgoal, ...]) -> None:
    """Bind typed input work to declared fields without reading objective prose."""

    fields: set[str] = set()
    if isinstance(entities.get('input_text'), str):
        fields.add(PRIMARY_INPUT_FIELD_ID)
    raw_fields = entities.get('input_fields')
    if isinstance(raw_fields, list):
        fields.update(str(item.get('field_id') or '') for item in raw_fields if isinstance(item, dict))
    for subgoal in subgoals:
        if not subgoal.input_field_id:
            continue
        reject_if(subgoal.input_field_id not in fields, TaskGraphError(
            f'子目标引用未声明的 typed input field：{subgoal.subgoal_id}/{subgoal.input_field_id}'))


def _require_text(value: Any, path: str, *, max_length: int=1000) -> str:
    reject_if(not isinstance(value, str) or not value.strip(), TaskGraphError(f"{path} 必须是非空字符串。"))
    text = value.strip()
    reject_if(len(text) > max_length, TaskGraphError(f"{path} 超过最大长度 {max_length}。"))
    return text


def _text_tuple(value: Any, path: str) -> tuple[str, ...]:
    items = _expect_list(value, path)
    return tuple(_require_text(item, f"{path}[]") for item in items)


def _id_tuple(value: Any, path: str) -> tuple[str, ...]:
    items = _expect_list(value, path)
    return tuple(str(item or "").strip().lower() for item in items)


def _validate_text_list(values: tuple[str, ...], path: str, *, required: bool) -> None:
    reject_if(required and (not values), TaskGraphError(f"{path} 不能为空。"))
    reject_if(len(values) != len(set(values)), TaskGraphError(f"{path} 不能包含重复项。"))
    for value in values:
        _require_text(value, path)


def _validate_id_list(values: tuple[str, ...], path: str, *, required: bool) -> None:
    reject_if(required and (not values), TaskGraphError(f"{path} 不能为空。"))
    reject_if(len(values) != len(set(values)), TaskGraphError(f"{path} 不能包含重复 ID。"))
    for value in values:
        _validate_id(value, path)


def _validate_id(value: str, label: str) -> None:
    reject_if(not ID_PATTERN.fullmatch(value), TaskGraphError(f"{label} 无效：{value!r}"))


def _validate_device_id(device_id: str) -> None:
    reject_if(not DEVICE_ID_PATTERN.fullmatch(str(device_id or '')), TaskGraphError(f"device_id 无效：{device_id!r}"))


def _validate_task_id(task_id: str) -> None:
    reject_if(not TASK_ID_PATTERN.fullmatch(str(task_id or '')), TaskGraphError(f"task_id 无效：{task_id!r}"))


def _reject_control_fields(value: Any, path: str) -> None:
    forbidden = {'action', 'actions', 'step', 'steps', 'tap', 'click', 'swipe', 'coordinate', 'coordinates', 'x', 'y',
        'shell', 'command', 'powershell', 'python', 'main_exe', 'execution_plan'}
    if isinstance(value, dict):
        for (key, item) in value.items():
            reject_if(str(key).strip().lower() in forbidden, TaskGraphError(f"任务图包含低层控制字段：{path}.{key}"))
            _reject_control_fields(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for (index, item) in enumerate(value):
            _reject_control_fields(item, f"{path}[{index}]")
    elif not isinstance(value, (str, int, float, bool, type(None))):
        raise TaskGraphError(f"任务图字段类型不受支持：{path}")
