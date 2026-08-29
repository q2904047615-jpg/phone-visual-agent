from __future__ import annotations

from .validation import ValidatedDataclassWire, dataclass_wire, reject_if
import json
import re
import uuid
from dataclasses import asdict, dataclass, field, replace
from difflib import SequenceMatcher
from typing import Any

from agent.domain.task_semantic_ir import compile_formal_semantic_authority


DEEPSEEK_TASK_GRAPH_PROTOCOL_VERSION = "2026-08-20-deepseek-typed-task-graph-v4"
SUBGOAL_EXTERNAL_IMPACTS = frozenset({'read_only', 'navigation_only', 'external_state', 'unknown'})
PLANNER_EXECUTION_CLASSES = frozenset({'observe', 'navigate', 'effect', 'unknown'})
PLANNER_EFFECT_KINDS = frozenset({'send_message', 'publish_content', 'relationship_change', 'membership_change',
    'data_mutation', 'authentication', 'financial_transaction', 'sensitive_permission_change',
    'irreversible_account_deletion', 'irreversible_data_deletion'})
_RUNTIME_IMPACT_BY_EXECUTION_CLASS = {'observe': 'read_only', 'navigate': 'navigation_only', 'effect': 'external_state',
    'unknown': 'unknown'}
NON_EFFECT_RESULT_PATTERN = re.compile(
    r"(?:保持|维持|仍然|仍旧).{0,24}(?:不变|原样|未发生|未触发|未执行)|"
    r"(?:未|没有|尚未|不得|不要|禁止|不能).{0,20}"
    r"(?:发送|提交|发布|关注|评论|付款|支付|转账|登录|授权|删除|修改|保存|同步)|"
    r"\b(?:remain|keep|stay)\b.{0,24}\b(?:unchanged|not\s+sent)\b|"
    r"\b(?:not|never|without)\b.{0,20}"
    r"\b(?:send|submit|publish|follow|comment|pay|login|authorize|delete|modify|save|sync)\b",
    re.IGNORECASE,
)
_EXECUTION_CLASS_BY_RUNTIME_IMPACT = {value: key for key, value in _RUNTIME_IMPACT_BY_EXECUTION_CLASS.items()}
GRAPH_STATUSES = frozenset({'ready', 'running', 'awaiting_confirmation', 'completed', 'blocked'})
SUBGOAL_STATUSES = frozenset({'pending', 'active', 'completed', 'blocked', 'skipped'})
REPLAN_TRIGGERS = frozenset({'observation_changed', 'action_result_matched', 'action_mismatch',
    'action_result_mismatch', 'subgoal_completed', 'risk_detected', 'constraint_discovered', 'recovery_needed'})
VERIFIED_ACTION_TRANSITION_PROTOCOL_VERSION = '2026-08-16-verified-action-transition-v1'
ID_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
DEVICE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
TASK_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
FORBIDDEN_EXECUTION_INSTRUCTION_PATTERN = re.compile(
    r"(?:"
    r"裸坐标|像素坐标|归一化坐标|坐标点|系统命令|"
    r"shell|powershell|cmd\.exe|adb|main\.exe|"
    r"\(\s*\d{1,4}\s*[,，]\s*\d{1,4}\s*\)|"
    r"\b[xy]\s*[:=]\s*\d+|"
    r"\b(?:coordinate|keycode|system[ _-]?command|shell[ _-]?command)\b"
    r")",
    re.IGNORECASE,
)
TARGET_SURFACES = frozenset({"device", "system", "current_surface"})
MAX_CANONICAL_INPUT_CHARS = 4000
MAX_INPUT_FIELDS = 32
MAX_RECIPIENTS = 32

QWEN_SUBGOAL_SCOPED_ENTITY_KEYS = frozenset({'recipient', 'recipients', 'input_text', 'input_fields', 'target_ui_label',
    'spatial_hint', 'amount', 'currency', 'merchant', 'payee', 'account', 'file', 'product', 'date', 'time', 'target',
    'value'})
GENERIC_UI_ROLE_ONLY_LABEL_PATTERN = re.compile(
    r"(?:输入框|文本框|搜索框|文本区域|输入区域|编辑区域|"
    r"按钮|入口|选项|控件|元素|列表项|标签页|页签)$",
    re.IGNORECASE,
)
NEGATED_LOW_LEVEL_INSTRUCTION_PREFIX_PATTERN = re.compile(
    r"(?:不|未|没有|未曾|勿|不要|不得|禁止|不能|避免|无需|无须|"
    r"do\s+not|don't|never|without)\s*"
    r"(?:(?:进行|执行)\s*)?"
    r"(?:(?:任何|任意|一切|all|any)\s*)?"
    r"(?!(?:忘记|漏掉|只|仅|forget\b|fail\b))",
    re.IGNORECASE,
)
LOW_LEVEL_NEGATION_SCOPE_RESET_PATTERN = re.compile(
    r"[。；;！？!?\r\n]+|"
    r"\b(?:but|however|then|afterwards|next|may|can|need(?:s|ed)?\s+to)\b|"
    r"(?:但是|但|然而|不过|然后|随后|接着|可以|仍可|需要|应当)",
    re.IGNORECASE,
)
INPUT_CONTENT_STATE_CONSTRAINT_PATTERN = re.compile(
    r"^\s*(?:当前)?输入内容"
    r"(?:(?:必须|应当|应|需要)(?:为|是|等于)|保持为)\s*"
    r"(?:“[^”\r\n]{1,100}”|\"[^\"\r\n]{1,100}\")\s*$",
    re.IGNORECASE,
)
READ_ONLY_RISK_CONTROL_STATE_PATTERN = re.compile(
    r"(?:(?:停在|保持在).{0,20}(?:按钮|控件|入口).{0,8}(?:之前|前)|"
    r"(?:(?:发送|提交|删除|清除|转发|发布|保存|分享|回复|关注|支付|"
    r"send|submit|delete|erase|forward|publish|save|share|reply|follow|pay)\s*)?"
    r"(?:按钮|控件|入口|button|control).{0,12}(?:可见|显示|仍能看见|可核对|"
    r"未被触发|没有触发|未触发|未被点击|没有被点击|未点击|"
    r"未被激活|未激活|没有激活|"
    r"未被启用|未启用|没有启用|"
    r"visible|shown|not\s+triggered|not\s+activated|not\s+enabled)|"
    r"\b(?:stop|stay|remain)\b.{0,28}\bbefore\b.{0,16}\b(?:button|control)\b|"
    r"\b(?:button|control)\b.{0,16}\b(?:visible|shown)\b)",
    re.IGNORECASE,
)
DIRECT_PROHIBITION_CLAUSE_PATTERN = re.compile(
    r"^\s*(?:(?:且|并且|并|and)\s*)?"
    r"(?:不要|不得|禁止|不能|避免|勿|"
    r"不(?!要|得|能|应|可|只|仅|忘记)|do\s+not|don't|never)\s*"
    r"(?!(?:忘记|漏掉|只|仅|forget\b|fail\b))",
    re.IGNORECASE,
)


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
        _reject_low_level_instruction(self.objective, "goal.objective")
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
        _reject_low_level_instruction(self.description, 'completion_conditions.description')
        _validate_text_list(self.evidence_required, "evidence_required", required=True)
        for item in self.evidence_required:
            _reject_low_level_completion_evidence(item, 'completion_conditions.evidence_required')
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

    def validate(self) -> None:
        _validate_id(self.subgoal_id, "子目标 ID")
        _require_text(self.objective, "subgoals.objective")
        reject_if(self.status not in SUBGOAL_STATUSES, TaskGraphError(f"子目标状态无效：{self.status}"))
        _validate_id_list(self.depends_on, "subgoals.depends_on", required=False)
        _validate_text_list(self.constraints, "subgoals.constraints", required=False)
        for item in self.constraints:
            _reject_low_level_instruction(item, 'subgoals.constraints', allow_negated=True)
        _validate_text_list(self.completion_conditions, 'subgoals.completion_conditions', required=True)
        _reject_low_level_instruction(self.objective, "subgoals.objective")
        for item in self.completion_conditions:
            _reject_low_level_completion_evidence(item, 'subgoals.completion_conditions')
        _validate_text_list(self.completion_evidence, 'subgoals.completion_evidence', required=False)
        _validate_id_list(self.risk_action_ids, 'subgoals.effect_ids', required=False)
        reject_if(self.external_impact not in SUBGOAL_EXTERNAL_IMPACTS, TaskGraphError(f'子目标执行类别无效：{self.external_impact}'))
        # Only typed InputFieldIntent owns input values; free-form preparation prose never does.
        reject_if(self.status == 'completed' and (not self.completion_evidence), TaskGraphError(f"已完成子目标缺少完成证据：{self.subgoal_id}"))
        reject_if(self.status != 'completed' and self.completion_evidence, TaskGraphError(f"未完成子目标不能携带完成证据：{self.subgoal_id}"))


@dataclass(frozen=True)
class ReplanRecord:
    revision: int
    trigger: str
    reason: str
    scene_id: str
    evidence: tuple[str, ...]
    retained_completed_subgoal_ids: tuple[str, ...]
    added_subgoal_ids: tuple[str, ...]
    skipped_subgoal_ids: tuple[str, ...]
    consumed_action_transition_receipt_id: str = ""


@dataclass(frozen=True)
class VerifiedActionTransition(ValidatedDataclassWire):
    """Controller receipt for one scoped action followed by one fresh observation."""

    receipt_id: str
    session_id: str
    task_id: str
    device_id: str
    prior_revision: int
    subgoal_id: str
    decision_node_id: str
    action_digest: str
    rebound_action_digest: str
    resolved_action_digest: str
    action_kind: str
    before_observation_id: str
    before_fingerprint: str
    after_observation_id: str
    after_fingerprint: str
    physical_actions: int
    outcome: str
    errors: tuple[str, ...] = ()
    controller_transition_evidence: tuple[str, ...] = ()
    protocol_version: str = VERIFIED_ACTION_TRANSITION_PROTOCOL_VERSION

    def validate(self) -> None:
        reject_if(self.protocol_version != VERIFIED_ACTION_TRANSITION_PROTOCOL_VERSION, TaskGraphError(f'动作转换回执协议版本无效：{self.protocol_version}'))
        _require_text(self.receipt_id, "action_transition.receipt_id")
        _require_text(self.session_id, "action_transition.session_id")
        reject_if(not ID_PATTERN.fullmatch(self.receipt_id), TaskGraphError(f'动作转换回执 receipt_id 无效：{self.receipt_id!r}'))
        reject_if(not TASK_ID_PATTERN.fullmatch(self.task_id), TaskGraphError(f"动作转换回执 task_id 无效：{self.task_id!r}"))
        reject_if(not DEVICE_ID_PATTERN.fullmatch(self.device_id), TaskGraphError(f"动作转换回执 device_id 无效：{self.device_id!r}"))
        reject_if(isinstance(self.prior_revision, bool) or not isinstance(self.prior_revision, int) or self.prior_revision < 1, TaskGraphError("动作转换回执 prior_revision 必须是正整数。"))
        for field_name in ('subgoal_id', 'decision_node_id', 'action_digest', 'rebound_action_digest',
            'resolved_action_digest', 'action_kind', 'before_observation_id', 'before_fingerprint',
            'after_observation_id', 'after_fingerprint'):
            _require_text(getattr(self, field_name), f"action_transition.{field_name}")
        for field_name in ('action_digest', 'rebound_action_digest', 'resolved_action_digest'):
            reject_if(not re.fullmatch('[0-9a-f]{64}', getattr(self, field_name)), TaskGraphError(f'动作转换回执 {field_name} 必须是 64 位小写 SHA-256。'))
        reject_if(self.before_observation_id == self.after_observation_id, TaskGraphError("动作转换回执必须绑定新的动作后 observation_id。"))
        reject_if(
            isinstance(self.physical_actions, bool) or not isinstance(self.physical_actions,
            int) or self.physical_actions != 1,
            TaskGraphError("动作转换回执必须且只能证明 1 次物理动作。"),
        )
        reject_if(self.outcome not in {'matched', 'mismatched'}, TaskGraphError(f"动作转换回执 outcome 无效：{self.outcome}"))
        _validate_text_list(self.errors, "action_transition.errors", required=False)
        _validate_text_list(self.controller_transition_evidence, 'action_transition.controller_transition_evidence',
            required=False)
        reject_if(self.outcome == 'matched' and self.errors, TaskGraphError("matched 动作转换回执不能同时包含验证错误。"))
        reject_if(self.outcome == 'mismatched' and (not self.errors), TaskGraphError("mismatched 动作转换回执必须包含验证错误。"))
        reject_if(self.outcome == 'matched' and self.before_fingerprint == self.after_fingerprint, TaskGraphError("matched 动作转换回执必须绑定变化后的 fingerprint。"))

@dataclass(frozen=True)
class ControllerTransitionEvidenceRef(ValidatedDataclassWire):
    ref_id: str
    receipt_id: str
    subgoal_id: str
    text: str
    source: str = "controller_transition"

    def validate(self) -> None:
        reject_if(self.source != 'controller_transition', TaskGraphError("控制器转换证据来源无效。"))
        for field_name in ('ref_id', 'receipt_id', 'subgoal_id', 'text'):
            _require_text(getattr(self, field_name), f'controller_transition_evidence.{field_name}')
        reject_if(not self.ref_id.startswith(f'controller_transition:{self.receipt_id}:'), TaskGraphError("控制器转换证据 ref_id 未绑定 receipt_id。"))

@dataclass(frozen=True)
class VisualClaimEvidenceRef(ValidatedDataclassWire):
    ref_id: str
    claim_id: str
    scene_id: str
    subject_ref: str
    predicate: str
    fact: str
    source: str = "visual_claim"

    def validate(self) -> None:
        reject_if(self.source != 'visual_claim', TaskGraphError("视觉 claim 证据来源无效。"))
        for field_name in ('ref_id', 'claim_id', 'scene_id', 'subject_ref', 'predicate', 'fact'):
            _require_text(getattr(self, field_name), f'visual_claim_evidence.{field_name}')
        reject_if(not re.fullmatch('[0-9a-f]{64}', self.claim_id), TaskGraphError("视觉 claim_id 必须是 SHA-256。"))
        reject_if(self.ref_id != f'visual_claim:{self.scene_id}:{self.claim_id}', TaskGraphError("视觉 claim ref_id 未绑定 scene_id/claim_id。"))

@dataclass(frozen=True)
class ObservedState(ValidatedDataclassWire):
    scene_id: str
    summary: str
    visible_evidence: tuple[str, ...]
    grounded_visual_facts: tuple[str, ...] = ()
    last_action_outcome: str = "not_applicable"
    blocked_reasons: tuple[str, ...] = ()
    verified_action_transition: VerifiedActionTransition | None = None
    controller_transition_evidence_refs: tuple[ControllerTransitionEvidenceRef, ...] = ()
    visual_claim_evidence_refs: tuple[VisualClaimEvidenceRef, ...] = ()

    def validate(self) -> None:
        _require_text(self.scene_id, "observation.scene_id")
        _require_text(self.summary, "observation.summary")
        _validate_text_list(self.visible_evidence, 'observation.visible_evidence', required=True)
        _validate_text_list(self.grounded_visual_facts, 'observation.grounded_visual_facts', required=False)
        reject_if(self.last_action_outcome not in {'not_applicable', 'matched', 'mismatched', 'uncertain'}, TaskGraphError(f'观察中的动作结果无效：{self.last_action_outcome}'))
        _validate_text_list(self.blocked_reasons, 'observation.blocked_reasons', required=False)
        if self.verified_action_transition is not None:
            self.verified_action_transition.validate()
            reject_if(self.last_action_outcome != self.verified_action_transition.outcome, TaskGraphError('观察动作结果与 verified_action_transition outcome 不一致。'))
        for item in self.controller_transition_evidence_refs:
            item.validate()
            reject_if(
                self.verified_action_transition is None or item.receipt_id !=
                self.verified_action_transition.receipt_id or item.subgoal_id !=
                self.verified_action_transition.subgoal_id or (item.text not
                in self.verified_action_transition.controller_transition_evidence),
                TaskGraphError('控制器转换证据未绑定当前 verified action transition。'),
            )
        reject_if(self.verified_action_transition is None and self.controller_transition_evidence_refs, TaskGraphError("无动作回执时不得携带控制器转换证据。"))
        visual_ref_ids: set[str] = set()
        for item in self.visual_claim_evidence_refs:
            item.validate()
            reject_if(item.scene_id != self.scene_id, TaskGraphError("视觉 claim 未绑定当前 scene_id。"))
            reject_if(item.ref_id in visual_ref_ids, TaskGraphError("视觉 claim ref_id 重复。"))
            visual_ref_ids.add(item.ref_id)

def _goal_wire(goal: GraphGoal) -> dict[str, Any]:
    return {'objective': goal.objective, 'target_apps': [asdict(app) for app in goal.target_apps],
        'entities': dict(goal.entities)}


def _literal_strings(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    nested = value.values() if isinstance(value, dict) else value if isinstance(value, (list, tuple)) else ()
    return tuple(literal for item in nested for literal in _literal_strings(item))


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
    return {'subgoal_id': subgoal.subgoal_id, 'objective': subgoal.objective, 'status': subgoal.status,
        'depends_on': list(subgoal.depends_on), 'constraints': list(subgoal.constraints),
        'completion_conditions': list(subgoal.completion_conditions),
        'completion_evidence': list(subgoal.completion_evidence), 'effect_ids': list(subgoal.risk_action_ids),
        'execution_class': _EXECUTION_CLASS_BY_RUNTIME_IMPACT[subgoal.external_impact]}


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
    replan_history: tuple[ReplanRecord, ...] = ()
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
        for item in self.constraints:
            _reject_low_level_instruction(item, "constraints", allow_negated=True)
        _validate_text_list(self.clarification_questions, "clarification_questions", required=False)
        for item in self.clarification_questions:
            _reject_low_level_instruction(item, "clarification_questions")

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
            'active_subgoal_id': self.active_subgoal_id, 'clarification_questions': list(self.clarification_questions),
            'replan_history': [dataclass_wire(item) for item in self.replan_history]}
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
        goal_context = dict(value["goal"])
        goal_entities = dict(goal_context.get("entities") or {})
        if current is not None:
            current_text = '\n'.join((str(item) for item in (current.get('objective') or '',
                *(current.get('constraints') or ()), *(current.get('completion_conditions') or ())))).casefold()
            for key in tuple(goal_entities):
                if key not in QWEN_SUBGOAL_SCOPED_ENTITY_KEYS:
                    continue
                literals = _literal_strings(goal_entities[key])
                if not literals or not any((literal.casefold() in current_text for literal in literals)):
                    goal_entities.pop(key, None)
            exact_label = str(goal_entities.get("target_ui_label") or "").strip()
            if exact_label and GENERIC_UI_ROLE_ONLY_LABEL_PATTERN.search(exact_label):
                escaped_label = re.escape(exact_label)
                raw_goal = self.raw_user_goal or self.goal.objective
                explicitly_literal = re.search(
                    rf"[“\"]{escaped_label}[”\"]|"
                    rf"(?:名为|名称为|标有|标签为|文字为|显示文字为)\s*[“\"]?{escaped_label}[”\"]?",
                    raw_goal,
                    flags=re.IGNORECASE,
                )
                if not explicitly_literal:
                    goal_entities.pop("target_ui_label", None)
        goal_context["entities"] = goal_entities
        return {
            "protocol_version": self.protocol_version,
            "task_id": self.task_id,
            "device_id": self.device_id,
            "revision": self.revision,
            "task_status": self.status,
            "goal": goal_context,
            "global_constraints": value["constraints"],
            "goal_completion_conditions": value["completion_conditions"],
            "current_subgoal": current,
            "current_execution_class": current["execution_class"] if current else None,
            "effect_intents": [
                item for item in value["effect_intents"] if item["effect_id"] in current_effect_ids
            ],
            "effect_gate": {
                "required": confirmation_required,
                "state": (
                    "confirmed" if confirmation_granted else
                    "awaiting_confirmation" if confirmation_required else "not_required"
                ),
                "effect_ids": sorted(confirmation_effect_ids),
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
    semantic_ir = compile_formal_semantic_authority(graph).semantic_ir
    required_actions = {str(constraint.value) for constraint
        in semantic_ir.constraints if constraint.kind == 'required_action'}
    reject_if(resolved_action not in required_actions, TaskGraphError("本地直推动作没有编译为请求的 canonical action。"))
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
        external_impact='navigation_only'),), active_subgoal_id=subgoal_id, raw_user_goal=goal_text)
    graph.validate()
    compile_formal_semantic_authority(graph)
    return graph


def _planner_transport_snapshot(graph: DynamicTaskGraph) -> dict[str, Any]:
    """Serialize only the current typed planner transport."""

    graph.validate()
    reject_if(any((not effect.effect_kind for effect in graph.risk_actions)), TaskGraphError("风险条目缺少 typed effect_kind，不能进入正式重规划。"))
    return {'status': 'running' if graph.status == 'awaiting_confirmation' else graph.status,
        'goal': _goal_wire(graph.goal), 'constraints': list(graph.constraints),
        'completion_conditions': [_condition_wire(item) for item in graph.completion_conditions],
        'effect_intents': [_effect_wire(item, local_policy=False) for item in graph.risk_actions],
        'subgoals': [_subgoal_wire(item) for item in graph.subgoals], 'active_subgoal_id': graph.active_subgoal_id,
        'clarification_questions': list(graph.clarification_questions)}


def _graph_from_payload(payload: dict[str, Any], *, task_id: str, device_id: str, revision: int,
    raw_user_goal: str) -> DynamicTaskGraph:
    _expect_keys(payload, {'status', 'goal', 'constraints', 'completion_conditions', 'effect_intents', 'subgoals',
        'active_subgoal_id', 'clarification_questions'}, '任务图')
    raw_goal = _expect_dict(payload.get("goal"), "goal")
    _expect_keys(raw_goal, {"objective", "target_apps", "entities"}, "goal")
    target_apps = tuple((_target_app_from_payload(item) for item in _expect_list(raw_goal.get('target_apps'),
        'goal.target_apps')))
    entities = dict(_expect_dict(raw_goal.get("entities"), "goal.entities"))
    entities = _normalize_unique_input_newline_escapes(entities, raw_user_goal=raw_user_goal)
    # Optional examples may be null/empty; every non-empty value remains strictly validated.
    optional_input_text = entities.get("input_text")
    if optional_input_text is None or optional_input_text == '':
        entities.pop("input_text", None)
    goal = GraphGoal(objective=_require_text(raw_goal.get('objective'), 'goal.objective'), target_apps=target_apps,
        entities=entities)
    # Validate typed entities before normalizing redundant effect-result references.
    goal.validate()
    active_value = payload.get("active_subgoal_id")
    active_subgoal_id = None if active_value is None else str(active_value).strip()
    raw_subgoals = tuple((_subgoal_from_payload(item) for item in _expect_list(payload.get('subgoals'), 'subgoals')))
    subgoals_by_id = {item.subgoal_id: item for item in raw_subgoals}
    reject_if(len(subgoals_by_id) != len(raw_subgoals), TaskGraphError("子目标 ID 重复。"))
    for subgoal in raw_subgoals:
        subgoal.validate()
    completion_conditions = tuple((_condition_from_payload(item) for item
        in _expect_list(payload.get('completion_conditions'), 'completion_conditions')))
    raw_effects = tuple(_expect_list(payload.get('effect_intents'), 'effect_intents'))
    effects = tuple((_effect_from_payload(item, subgoals=subgoals_by_id, entities=entities,
        goal_completion_results=tuple((condition.description for condition in completion_conditions)),
        effect_count=len(raw_effects)) for item in raw_effects))
    effects_by_id = {item.risk_id: item for item in effects}
    reject_if(len(effects_by_id) != len(effects), TaskGraphError("effect_intents.effect_id 重复。"))
    for subgoal in raw_subgoals:
        missing = set(subgoal.risk_action_ids) - set(effects_by_id)
        reject_if(missing, TaskGraphError(f'子目标 {subgoal.subgoal_id} 引用不存在 effect：' + ', '.join(sorted(missing))))
        reject_if(subgoal.external_impact == 'external_state' and (not subgoal.risk_action_ids), TaskGraphError(f'effect 子目标必须引用至少一个 effect_intent：{subgoal.subgoal_id}'))
        reject_if(subgoal.external_impact != 'external_state' and subgoal.risk_action_ids, TaskGraphError(f'非 effect 子目标不能引用 effect_intent：{subgoal.subgoal_id}'))
    reject_if(str(payload.get('status') or '').strip().lower() == 'awaiting_confirmation', TaskGraphError("模型不得决定 awaiting_confirmation；确认状态只由本地策略生成。"))
    return DynamicTaskGraph(task_id=task_id, device_id=device_id, revision=revision,
        status=str(payload.get('status') or '').strip().lower(), goal=goal,
        constraints=_text_tuple(payload.get('constraints'), 'constraints'), completion_conditions=completion_conditions,
        risk_actions=effects, subgoals=raw_subgoals, active_subgoal_id=active_subgoal_id,
        clarification_questions=_text_tuple(payload.get('clarification_questions'), 'clarification_questions'),
        raw_user_goal=raw_user_goal)


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
    item = _expect_dict(value, "completion_conditions[]")
    _expect_keys(item, {'condition_id', 'description', 'evidence_required', 'satisfied', 'evidence'},
        'completion_conditions[]')
    reject_if(not isinstance(item.get('satisfied'), bool), TaskGraphError("completion_conditions.satisfied 必须是布尔值。"))
    return CompletionCondition(condition_id=str(item.get('condition_id') or '').strip().lower(),
        description=_require_text(item.get('description'), 'completion_conditions.description'),
        evidence_required=_text_tuple(item.get('evidence_required'), 'evidence_required'), satisfied=item['satisfied'],
        evidence=_text_tuple(item.get('evidence'), 'evidence'))


def _effect_from_payload(value: Any, *, subgoals: dict[str, Subgoal], entities: dict[str, Any],
    goal_completion_results: tuple[str, ...], effect_count: int) -> RiskAction:
    item = _expect_dict(value, "effect_intents[]")
    _expect_keys(item, {'effect_id', 'kind', 'target_entity_roles', 'payload_entity_roles', 'source_subgoal_ids',
        'expected_results'}, 'effect_intents[]')
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
    expected_results = _text_tuple(item.get('expected_results'), 'effect_intents.expected_results')
    source_results = tuple((condition for subgoal_id in source_subgoal_ids for condition
        in subgoals[subgoal_id].completion_conditions))
    allowed_results = set(source_results)
    reject_if(not expected_results, TaskGraphError('effect_intents.expected_results 必须逐字来自绑定子目标的正向完成条件。'))
    reject_if(
        any((DIRECT_PROHIBITION_CLAUSE_PATTERN.search(result) or NON_EFFECT_RESULT_PATTERN.search(result) for result
        in expected_results)),
        TaskGraphError("禁止或未发生状态不能声明为 effect_intent。"),
    )
    if any(result not in allowed_results for result in expected_results):
        # Repair only a uniquely owned redundant expected_result; ambiguous graphs remain fail-closed.
        exact_unique_goal_result = effect_count == 1 and len(source_subgoal_ids) == 1 and (len(expected_results) ==
            1) and (sum((result == expected_results[0] for result in goal_completion_results)) == 1)
        exact_unique_source_result = len(source_results) == 1 and (not DIRECT_PROHIBITION_CLAUSE_PATTERN.search(
            source_results[0])) and (not NON_EFFECT_RESULT_PATTERN.search(source_results[0]))
        if not exact_unique_goal_result and exact_unique_source_result:
            expected_results = source_results
        elif not exact_unique_goal_result:
            raise TaskGraphError('effect_intents.expected_results 必须逐字来自绑定子目标的正向完成条件。')
    return RiskAction(risk_id=effect_id, subgoal_ids=source_subgoal_ids, confirmation_required=False,
        effect_kind=kind, target_roles=target_roles, payload_roles=payload_roles,
        expected_result_texts=expected_results)


def _subgoal_from_payload(value: Any) -> Subgoal:
    item = _expect_dict(value, "subgoals[]")
    _expect_keys(item, {'subgoal_id', 'objective', 'status', 'depends_on', 'constraints', 'completion_conditions',
        'completion_evidence', 'effect_ids', 'execution_class'}, 'subgoals[]')
    execution_class = str(item.get("execution_class") or "").strip().lower()
    reject_if(execution_class not in PLANNER_EXECUTION_CLASSES, TaskGraphError(f'subgoals.execution_class 无效：{execution_class}'))
    return Subgoal(subgoal_id=str(item.get('subgoal_id') or '').strip().lower(),
        objective=_require_text(item.get('objective'), 'subgoals.objective'),
        status=str(item.get('status') or '').strip().lower(), depends_on=_id_tuple(item.get('depends_on'),
        'subgoals.depends_on'), constraints=_text_tuple(item.get('constraints'), 'subgoals.constraints'),
        completion_conditions=_text_tuple(item.get('completion_conditions'), 'subgoals.completion_conditions'),
        completion_evidence=_text_tuple(item.get('completion_evidence'), 'subgoals.completion_evidence'),
        risk_action_ids=_id_tuple(item.get('effect_ids'), 'subgoals.effect_ids'),
        external_impact=_RUNTIME_IMPACT_BY_EXECUTION_CLASS[execution_class])


_VISUAL_IDENTITY_CONTAINER_PATTERN = re.compile(
    r"主页面|主页|首页|页面|界面|屏幕|视图|面板|卡片|"
    r"(?:^|\b)(?:page|screen|view|panel|card)(?:\b|$)",
    re.IGNORECASE,
)
_GENERIC_VISUAL_LOCATION_PATTERN = re.compile(
    r"^(?:(?:找到|定位|查找|确认|观察|查看|识别|进入|打开)\s*)?"
    r"(?:当前|同一|该|目标|原来|原有)(?:本地)?(?:页面|界面|屏幕|视图)"
    r"(?:中|内|上)?\s*|"
    r"^(?:(?:find|locate|identify|observe|verify|inspect|open|enter|view)\s+)?"
    r"(?:the\s+)?(?:current|same|this|target|original)\s+"
    r"(?:local\s+)?(?:page|screen|view)\b\s*|"
    r"(?:在|位于)\s*(?:当前|同一|该)?(?:页面|界面|屏幕|视图)"
    r"(?:中|内|上)?\s*(?:可见|出现|显示|存在)?|"
    r"(?:visible|present|shown)\s+(?:in|on)\s+(?:the\s+)?"
    r"(?:current\s+)?(?:page|screen|view)|"
    r"(?:in|on)\s+(?:the\s+)?(?:current|same|this|target|original)\s+"
    r"(?:local\s+)?(?:page|screen|view)\b",
    re.IGNORECASE,
)
_LEADING_UNNAMED_VISUAL_CONTAINER_PATTERN = re.compile(
    r"^(?:主页面|主页|首页|页面|界面|屏幕|视图|面板|卡片)(?:中|内|上)?|"
    r"^(?:the\s+)?(?:page|screen|view|panel|card)\b",
    re.IGNORECASE,
)
_TEMPORAL_REFERENTIAL_VISUAL_CONTAINER_PATTERN = re.compile(
    r"(?:打开|进入|操作|动作|加载|刷新|跳转|切换|返回|退出|完成)"
    r"[^，。；;\r\n]*?后(?:的)?(?:页面|界面|屏幕|视图)(?:中|内|上)?",
    re.IGNORECASE,
)
_FUNCTIONAL_VISUAL_CONTAINER_MODIFIER_PATTERN = re.compile(
    r"(?:可|能|能够|可以|用于|供|允许|支持|包含|带有|显示|展示|存在|具有|提供)"
    r"[^，。；;\r\n]{0,80}的\s*$|"
    r"(?:editable|searchable|input|entry|selection|results?|"
    r"used\s+to|intended\s+for|allows?|supports?|contains?|shows?)"
    r"(?:[\s_-]+[a-z0-9]+){0,8}\s*$",
    re.IGNORECASE,
)
_VISUAL_IDENTITY_GENERIC_TOKENS = tuple(
    "原来的|原有的|当前的|指定的|目标的|已经|清晰|完整|当前|原来|原有|指定|目标|主页面|主页|首页|"
    "页面|界面|屏幕|视图|面板|卡片|控件|元素|入口|主标题|标题|文字|逐字|显示|出现|可见|打开|进入|"
    "返回|回到|通过|流程|结果|page|screen|view|panel|card|visible|shown|displayed|open|opened|result|"
    "process|flow|main title|title|heading|text|verbatim|current|target|original|the|is".split("|")
)

# This small semantic bridge maps generic screen categories without replacing verbatim title checks.
_VISUAL_IDENTITY_SEMANTIC_ALIASES = {'conversation': (re.compile('聊天|会话'),
    re.compile('(?<![a-z0-9])(?:chat|conversation)(?![a-z0-9])', re.I)), 'browser': (re.compile('浏览器'),
    re.compile('(?<![a-z0-9])browser(?![a-z0-9])', re.I)),
    'launcher': (re.compile('(?:手机|系统|android)?(?:主)?桌面(?!版)|(?:手机|系统)?主屏(?:幕)?', re.I),
    re.compile('(?<![a-z0-9])(?:launcher|home[ _-]?screen)(?![a-z0-9])', re.I))}

_VISUAL_IDENTITY_SEMANTIC_COMPACT_MARKERS = {'conversation': frozenset({'聊天', '会话', 'chat', 'conversation'}),
    'browser': frozenset({'浏览器', 'browser'}), 'launcher': frozenset({'launcher', 'homescreen', '桌面', '主屏', '主屏幕'})}

_SYSTEM_HOME_SURFACE_PATTERN = re.compile(
    r"(?:手机|系统|android)?(?:主)?桌面(?!版)|(?:手机|系统)?主屏(?:幕)?|"
    r"(?<![a-z0-9])(?:launcher|home[ _-]?screen)(?![a-z0-9])",
    re.IGNORECASE,
)


def _compact_identity_text(value: str) -> str:
    return "".join(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", value.casefold()))


def _visual_identity_semantic_keys(value: str) -> frozenset[str]:
    text = str(value or "").casefold()
    return frozenset((key for key, patterns in _VISUAL_IDENTITY_SEMANTIC_ALIASES.items() if any((pattern.search(
        text) for pattern in patterns))))


def _is_semantic_only_visual_identity(value: str, semantic_keys: frozenset[str] | None=None) -> bool:
    """Return whether an identity contains only a generic container category."""

    compact = _compact_identity_text(value)
    keys = semantic_keys or _visual_identity_semantic_keys(value)
    return bool(compact and any((compact in _VISUAL_IDENTITY_SEMANTIC_COMPACT_MARKERS.get(key, ()) for key in keys)))


def _named_visual_identity_anchor(texts: tuple[str, ...]) -> str:
    anchors: list[str] = []
    for item in texts:
        value = str(item or "").strip()
        if _SYSTEM_HOME_SURFACE_PATTERN.search(value):
            if 'launcher' not in anchors:
                anchors.append("launcher")
            continue
        if _TEMPORAL_REFERENTIAL_VISUAL_CONTAINER_PATTERN.search(value):
            continue
        identity_value = _GENERIC_VISUAL_LOCATION_PATTERN.sub(" ", value)
        if _LEADING_UNNAMED_VISUAL_CONTAINER_PATTERN.search(identity_value.strip()):
            continue
        container = _VISUAL_IDENTITY_CONTAINER_PATTERN.search(identity_value)
        if not identity_value.strip() or container is None:
            continue
        identity_name = identity_value[:container.start()]
        if _FUNCTIONAL_VISUAL_CONTAINER_MODIFIER_PATTERN.search(identity_name.strip()):
            continue
        cleaned = identity_name.casefold()
        for token in _VISUAL_IDENTITY_GENERIC_TOKENS:
            cleaned = cleaned.replace(token, " ")
        anchor = _compact_identity_text(cleaned)
        has_stable_length = len(anchor) >= 4 or len(re.findall(r"[\u4e00-\u9fff]", anchor)) >= 2
        if has_stable_length and anchor not in anchors:
            anchors.append(anchor)
    return min(anchors, key=len) if anchors else ""


def _identity_anchor_is_grounded(anchor: str, facts: tuple[str, ...]) -> bool:
    anchor_semantics = _visual_identity_semantic_keys(anchor)
    semantic_only_anchor = _is_semantic_only_visual_identity(anchor, anchor_semantics)
    for fact in facts:
        compact = _compact_identity_text(fact)
        if anchor in compact:
            return True
        if semantic_only_anchor and anchor_semantics.intersection(_visual_identity_semantic_keys(fact)):
            return True
        if not compact:
            continue
        longest = SequenceMatcher(None, anchor, compact, autojunk=False).find_longest_match()
        matched_anchor_part = anchor[longest.a : longest.a + longest.size]
        if longest.size / len(anchor) >= 0.5 and (not _is_semantic_only_visual_identity(matched_anchor_part)):
            return True
    return False


def named_visual_identity_is_grounded(texts: tuple[str, ...], facts: tuple[str, ...]) -> bool:
    """Require structured facts for every named visual container identity."""

    anchor = _named_visual_identity_anchor(texts)
    if not anchor:
        return True
    return bool(facts and _identity_anchor_is_grounded(anchor, facts))


def _restore_completed_history_evidence(previous: DynamicTaskGraph, candidate: DynamicTaskGraph) -> DynamicTaskGraph:
    """Restore only controller-owned evidence after the model preserves completed-node semantics."""

    completed = {item.subgoal_id: item for item in previous.subgoals if item.status == 'completed'}
    restored = tuple((replace(item, completion_evidence=completed[
        item.subgoal_id].completion_evidence) if item.subgoal_id in completed
        and item.status == 'completed' else item for item in candidate.subgoals))
    return replace(candidate, subgoals=restored)


def _apply_verified_navigation_completion(previous: DynamicTaskGraph, candidate: DynamicTaskGraph,
    observation: ObservedState, *, trigger: str) -> DynamicTaskGraph:
    """Complete only the navigation node bound to one matched local receipt."""
    transition = observation.verified_action_transition
    current = previous.active_subgoal()
    if (trigger != 'action_result_matched' or transition is None or transition.outcome != 'matched' or (current
        is None) or (current.external_impact != 'navigation_only') or (not transition.session_id.strip())
        or (transition.task_id != previous.task_id) or (transition.device_id != previous.device_id)
        or (transition.prior_revision != previous.revision) or (transition.subgoal_id != current.subgoal_id)
        or (transition.after_observation_id != observation.scene_id) or (transition.receipt_id
        in {item.consumed_action_transition_receipt_id for item
        in previous.replan_history if item.consumed_action_transition_receipt_id})):
        return candidate
    ref_ids = tuple((item.ref_id for item in observation.controller_transition_evidence_refs if item.receipt_id ==
        transition.receipt_id and item.subgoal_id == current.subgoal_id))
    if not ref_ids:
        return candidate
    candidate_current = _by_attr(candidate.subgoals, "subgoal_id").get(current.subgoal_id)
    reject_if(candidate_current is None, TaskGraphError("matched controller_transition 的候选图删除了其绑定子目标。"))
    reject_if(not _same_subgoal_semantics(current, candidate_current), TaskGraphError("matched controller_transition 的候选图改写了其绑定子目标语义。"))
    subgoals = tuple((replace(item, status='completed', completion_evidence=ref_ids) if item.subgoal_id ==
        current.subgoal_id else item for item in candidate.subgoals))
    active_ids = tuple(item.subgoal_id for item in subgoals if item.status == "active")
    return _finalize_terminal_graph_from_exact_subgoal_conditions(replace(candidate, subgoals=subgoals,
        active_subgoal_id=active_ids[0] if len(active_ids) == 1 else None))


def _condition_key(value: str) -> str:
    return re.sub(r"[\W_]+", "", str(value or ""), flags=re.UNICODE).casefold()


def _by_attr(values: Any, attribute: str) -> dict[str, Any]:
    return {str(getattr(item, attribute)): item for item in values}


def _same_fields(old: Any, new: Any, names: tuple[str, ...]) -> bool:
    return all(getattr(old, name) == getattr(new, name) for name in names)


def _same_subgoal_semantics(old: Subgoal, new: Subgoal) -> bool:
    return _same_fields(old, new, ('objective', 'depends_on', 'constraints', 'completion_conditions', 'risk_action_ids',
        'external_impact'))


def _terminal_navigation_target(previous: DynamicTaskGraph, observation: ObservedState, trigger: str) -> Subgoal | None:
    transition = observation.verified_action_transition
    current = previous.active_subgoal()
    if (trigger == 'action_result_matched' and transition is not None and (transition.outcome == 'matched')
        and (current is not None) and (len(previous.subgoals) == 1) and (current.external_impact == 'navigation_only')
        and (not previous.risk_actions) and (not previous.clarification_questions)
        and (transition.task_id == previous.task_id) and (transition.device_id == previous.device_id)
        and (transition.prior_revision == previous.revision) and (transition.subgoal_id == current.subgoal_id)
        and (transition.after_observation_id == observation.scene_id)):
        return current
    return None


def _valid_observation_refs(observation: ObservedState) -> set[str]:
    return {item.ref_id for item in (*observation.controller_transition_evidence_refs,
        *observation.visual_claim_evidence_refs)}


def _complete_exact_conditions(conditions: tuple[CompletionCondition, ...], subgoal_conditions: tuple[str, ...],
    evidence: tuple[str, ...]) -> tuple[CompletionCondition, ...] | None:
    source_keys = {_condition_key(value) for value in subgoal_conditions}
    source_keys.discard("")
    completed: list[CompletionCondition] = []
    for condition in conditions:
        target_keys = {_condition_key(condition.description),
            *(_condition_key(value) for value in condition.evidence_required)}
        target_keys.discard("")
        if not source_keys.intersection(target_keys):
            return None
        completed.append(replace(condition, satisfied=True, evidence=evidence))
    return tuple(completed)


def _project_terminal_single_navigation_candidate(previous: DynamicTaskGraph, candidate: DynamicTaskGraph,
    observation: ObservedState, *, trigger: str) -> DynamicTaskGraph:
    current = _terminal_navigation_target(previous, observation, trigger)
    candidate_current = next((item for item in candidate.subgoals if current and item.subgoal_id == current.subgoal_id),
        None)
    valid_refs = _valid_observation_refs(observation)
    if (current is None or candidate_current is None or candidate_current.status != 'completed'
        or (not _same_subgoal_semantics(current, candidate_current)) or (not candidate_current.completion_evidence)
        or (not valid_refs) or (not set(candidate_current.completion_evidence).issubset(valid_refs))):
        return candidate
    completed = _complete_exact_conditions(previous.completion_conditions, current.completion_conditions,
        candidate_current.completion_evidence)
    if completed is None:
        return candidate
    return replace(candidate, status='completed', goal=previous.goal, constraints=previous.constraints,
        completion_conditions=completed, risk_actions=previous.risk_actions, subgoals=(replace(current,
        status='completed', completion_evidence=candidate_current.completion_evidence),), active_subgoal_id=None,
        clarification_questions=())


def _finalize_terminal_graph_from_exact_subgoal_conditions(graph: DynamicTaskGraph) -> DynamicTaskGraph:
    """Complete terminal global conditions from exact typed subgoal conditions."""

    if (graph.active_subgoal_id is not None or not graph.subgoals or graph.clarification_questions
        or any((item.status not in {'completed', 'skipped'} for item in graph.subgoals))):
        return graph
    evidence_by_key = {_condition_key(text): subgoal.completion_evidence for subgoal
        in graph.subgoals if subgoal.status == 'completed' and subgoal.completion_evidence for text
        in subgoal.completion_conditions if _condition_key(text)}
    completed: list[CompletionCondition] = []
    for condition in graph.completion_conditions:
        if condition.satisfied and condition.evidence:
            completed.append(condition)
            continue
        keys = (_condition_key(condition.description), *(_condition_key(value) for value
            in condition.evidence_required))
        evidence = next((evidence_by_key[key] for key in keys if key in evidence_by_key), ())
        if not evidence:
            return graph
        completed.append(replace(condition, satisfied=True, evidence=evidence))
    return replace(graph, status="completed", completion_conditions=tuple(completed), active_subgoal_id=None)


def _validate_revision(previous: DynamicTaskGraph, candidate: DynamicTaskGraph, observation: ObservedState) -> None:
    reject_if(candidate.goal != previous.goal, TaskGraphError("重规划不能改写用户目标、目标 App 或目标实体。"))
    reject_if(not set(previous.constraints).issubset(candidate.constraints), TaskGraphError("重规划不能删除已有全局约束。"))
    old_conditions = _by_attr(previous.completion_conditions, "condition_id")
    new_conditions = _by_attr(candidate.completion_conditions, "condition_id")
    _reject_missing_refs(tuple(old_conditions), new_conditions, "重规划不能删除全局完成条件：")
    visual_claim_refs = {item.ref_id: item for item in observation.visual_claim_evidence_refs}
    evidence = set(visual_claim_refs) if visual_claim_refs else {item for item in (*observation.visible_evidence,
        *observation.grounded_visual_facts) if not item.startswith('controller_transition:')}
    controller_refs = _by_attr(observation.controller_transition_evidence_refs, "ref_id")
    transition = observation.verified_action_transition
    previous_current = previous.active_subgoal()
    old_subgoals = _by_attr(previous.subgoals, "subgoal_id")
    new_subgoals = _by_attr(candidate.subgoals, "subgoal_id")
    candidate_current = new_subgoals.get(previous_current.subgoal_id) if previous_current else None
    consumed_receipts = {item.consumed_action_transition_receipt_id for item
        in previous.replan_history if item.consumed_action_transition_receipt_id}
    if (transition is not None and transition.outcome == 'matched' and (transition.receipt_id not in consumed_receipts)
        and (previous_current is not None) and (previous_current.external_impact == 'navigation_only')
        and (candidate_current is not None) and (candidate_current.status == 'completed')
        and transition.session_id.strip() and (transition.task_id == previous.task_id)
        and (transition.device_id == previous.device_id) and (transition.prior_revision == previous.revision)
        and (transition.subgoal_id == previous_current.subgoal_id)
        and (transition.after_observation_id == observation.scene_id)):
        evidence.update({ref_id for ref_id, ref in controller_refs.items() if ref.receipt_id == transition.receipt_id
            and ref.subgoal_id == previous_current.subgoal_id and (ref_id in candidate_current.completion_evidence)})
    for (condition_id, old) in old_conditions.items():
        new = new_conditions[condition_id]
        reject_if(not _same_fields(old, new, ('description', 'evidence_required')), TaskGraphError(f"重规划不能改写全局完成条件：{condition_id}"))
        reject_if(old.satisfied and (not new.satisfied or not set(old.evidence).issubset(new.evidence)), TaskGraphError(f"重规划不能撤销已满足完成条件：{condition_id}"))
        reject_if(not old.satisfied and new.satisfied and (not set(new.evidence).issubset(evidence)), TaskGraphError(f"完成条件使用了当前观察之外的证据：{condition_id}"))
    for condition_id in set(new_conditions) - set(old_conditions):
        condition = new_conditions[condition_id]
        reject_if(condition.satisfied and (not set(condition.evidence).issubset(evidence)), TaskGraphError(f"新增完成条件使用了当前观察之外的证据：{condition_id}"))

    old_risks = _by_attr(previous.risk_actions, "risk_id")
    new_risks = _by_attr(candidate.risk_actions, "risk_id")
    _reject_missing_refs(tuple(old_risks), new_risks, "重规划不能删除既有风险：")
    for (risk_id, old) in old_risks.items():
        new = new_risks[risk_id]
        reject_if(
            old.confirmation_required and not new.confirmation_required
            or (not set(old.subgoal_ids).issubset(new.subgoal_ids)),
            TaskGraphError(f"重规划不能改写或降低既有风险：{risk_id}"),
        )

    completed = {key: item for key, item in old_subgoals.items() if item.status == "completed"}
    _reject_missing_refs(tuple(completed), new_subgoals, "重规划不能删除已完成子目标：")
    for (subgoal_id, old) in completed.items():
        new = new_subgoals[subgoal_id]
        reject_if(
            new.status != 'completed' or not _same_subgoal_semantics(old,
            new) or (not set(old.completion_evidence).issubset(new.completion_evidence)),
            TaskGraphError(f"重规划不能复活或改写已完成子目标：{subgoal_id}"),
        )
    for (subgoal_id, new) in new_subgoals.items():
        old = old_subgoals.get(subgoal_id)
        if new.status != 'completed' or (old is not None and old.status == 'completed'):
            continue
        claimed = set(new.completion_evidence)
        controller_claims = claimed.intersection(controller_refs)
        reject_if(claimed - evidence - set(controller_refs), TaskGraphError(f"子目标使用了当前观察之外的完成证据：{subgoal_id}"))
        reject_if(
            controller_claims and (old is None or old.external_impact != 'navigation_only'
            or previous.active_subgoal_id != subgoal_id or (transition is None) or (transition.outcome != 'matched')),
            TaskGraphError("controller_transition 证据只能完成其严格绑定的上一 navigation_only 活动子目标。"),
        )
        reject_if(any((controller_refs[ref_id].subgoal_id != subgoal_id for ref_id in controller_claims)), TaskGraphError("controller_transition 证据跨子目标使用。"))
    if (transition is not None and transition.outcome == 'matched' and controller_refs and (previous.active_subgoal_id
        is not None)):
        old_active = old_subgoals.get(previous.active_subgoal_id)
        revised_old_active = new_subgoals.get(previous.active_subgoal_id)
        reject_if(
            old_active is not None and old_active.external_impact == 'navigation_only'
            and (transition.subgoal_id == old_active.subgoal_id)
            and any((ref.subgoal_id == old_active.subgoal_id for ref in controller_refs.values()))
            and (revised_old_active is None or revised_old_active.status != 'completed'),
            TaskGraphError(f'matched controller_transition 未完成其绑定的 navigation_only 子目标：{old_active.subgoal_id}'),
        )


def _validate_execution_class_revision(previous: DynamicTaskGraph, candidate: DynamicTaskGraph) -> None:
    old_subgoals = {item.subgoal_id: item for item in previous.subgoals}
    new_subgoals = {item.subgoal_id: item for item in candidate.subgoals}
    for (subgoal_id, old) in old_subgoals.items():
        new = new_subgoals.get(subgoal_id)
        if new is None:
            continue
        reject_if(old.external_impact == 'external_state' and new.external_impact != 'external_state', TaskGraphError(f'重规划不能把既有 effect 子目标降级：{subgoal_id}'))
        reject_if(old.external_impact == 'unknown' and new.external_impact in {'read_only', 'navigation_only'}, TaskGraphError(f'重规划不能未经证据把 unknown 降级为安全分类：{subgoal_id}'))


def _validate_preserved_effect_intents(previous: DynamicTaskGraph, candidate: DynamicTaskGraph) -> None:
    previous_effects = {item.risk_id: item for item in previous.risk_actions}
    candidate_effects = {item.risk_id: item for item in candidate.risk_actions}
    missing = set(previous_effects) - set(candidate_effects)
    reject_if(missing, TaskGraphError('重规划不能删除既有 EffectIntent：' + ', '.join(sorted(missing))))
    for (effect_id, previous_effect) in previous_effects.items():
        candidate_effect = candidate_effects[effect_id]
        reject_if(
            candidate_effect.effect_kind != previous_effect.effect_kind
            or candidate_effect.target_roles != previous_effect.target_roles
            or candidate_effect.payload_roles != previous_effect.payload_roles
            or (candidate_effect.subgoal_ids != previous_effect.subgoal_ids)
            or (candidate_effect.expected_result_texts != previous_effect.expected_result_texts),
            TaskGraphError(f'重规划不能改写既有 EffectIntent：{effect_id}'),
        )


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
        allowed_shapes = ({"field_id", "text"}, {"field_id", "field_label", "text"})
        reject_if(not isinstance(item, dict) or set(item) not in allowed_shapes, TaskGraphError(f'goal.entities.input_fields[{index}] 只允许 field_id/可选field_label/text。'))
        field_id = item.get("field_id")
        label = item.get("field_label", "")
        reject_if(not isinstance(field_id, str) or not ID_PATTERN.fullmatch(field_id), TaskGraphError(f"goal.entities.input_fields[{index}].field_id 无效。"))
        reject_if(field_id in field_ids, TaskGraphError("goal.entities.input_fields.field_id 重复。"))
        field_ids.add(field_id)
        reject_if(not _is_single_line_literal(label, 120, allow_empty=True), TaskGraphError(f"goal.entities.input_fields[{index}].field_label 无效。"))
        folded = label.casefold()
        reject_if(label and folded in labels, TaskGraphError("goal.entities.input_fields.field_label 重复。"))
        labels.add(folded)
        reject_if(not _is_exact_input_text(item.get('text')), TaskGraphError(f"goal.entities.input_fields[{index}].text 必须为1～4000字符。"))


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


def _reject_low_level_instruction(value: str, path: str, *, allow_negated: bool=False) -> None:
    if path == 'subgoals.constraints' and INPUT_CONTENT_STATE_CONSTRAINT_PATTERN.fullmatch(value):
        return
    for match in FORBIDDEN_EXECUTION_INSTRUCTION_PATTERN.finditer(value):
        prefix = value[: match.start()].lower()
        resets = tuple(LOW_LEVEL_NEGATION_SCOPE_RESET_PATTERN.finditer(prefix))
        if resets:
            prefix = prefix[resets[-1].end() :]
        if allow_negated and NEGATED_LOW_LEVEL_INSTRUCTION_PREFIX_PATTERN.search(prefix):
            continue
        raise TaskGraphError(f"DeepSeek 任务图包含越权执行细节：{path}")


def _reject_low_level_completion_evidence(value: str, path: str) -> None:
    """Reject executable control details while preserving natural action facts."""

    try:
        _reject_low_level_instruction(value, path)
    except TaskGraphError:
        if not READ_ONLY_RISK_CONTROL_STATE_PATTERN.search(value):
            raise
        _reject_low_level_instruction(value, path, allow_negated=True)


def _normalize_unique_planner_transport_aliases(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize only bijective planner/runtime aliases; leave conflicts for strict validation."""

    if not isinstance(payload, dict):
        return payload
    value = json.loads(json.dumps(payload, ensure_ascii=False))
    changed = False
    goal = value.get("goal")
    if isinstance(goal, dict) and 'target_surface' in goal:
        target_surface = goal.get("target_surface")
        entities = goal.get("entities")
        if target_surface in TARGET_SURFACES and isinstance(entities, dict) and ('target_surface' not in entities):
            entities["target_surface"] = target_surface
            del goal["target_surface"]
            changed = True

    planner_class_by_runtime_impact = {runtime_impact: planner_class for planner_class,
        runtime_impact in _RUNTIME_IMPACT_BY_EXECUTION_CLASS.items()}
    subgoals = value.get("subgoals")
    if isinstance(subgoals, list):
        for item in subgoals:
            if not isinstance(item, dict):
                continue
            execution_class = str(item.get("execution_class") or "").strip().lower()
            normalized = planner_class_by_runtime_impact.get(execution_class)
            if normalized is not None and execution_class not in PLANNER_EXECUTION_CLASSES:
                item["execution_class"] = normalized
                changed = True
    return value if changed else payload


def _normalize_single_effect_result_string(payload: dict[str, Any]) -> dict[str, Any]:
    """Wrap one non-empty scalar result before ordinary exact source binding."""

    effects = payload.get("effect_intents")
    if not isinstance(effects, list):
        return payload
    normalized_effects: list[Any] = []
    changed = False
    for effect in effects:
        if not isinstance(effect, dict):
            normalized_effects.append(effect)
            continue
        result = effect.get("expected_results")
        if isinstance(result, str) and result.strip():
            normalized_effects.append({**effect, "expected_results": [result]})
            changed = True
        else:
            normalized_effects.append(effect)
    if not changed:
        return payload
    return {**payload, "effect_intents": normalized_effects}


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
