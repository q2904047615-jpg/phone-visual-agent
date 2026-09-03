"""Typed task context accepted by the Qwen visual selector."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from .validation import dataclass_wire, reject_if
import re
from dataclasses import dataclass
from typing import Any

import agent.domain.generic_goal as generic_goal_domain
from agent.domain.canonical_action_kinds import CANONICAL_ACTION_KINDS
from agent.domain.vision_model import VisionAgentError


SUPPORTED_TASK_CONTEXT_PROTOCOL = "2026-09-03-deepseek-required-action-v6"
TASK_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
DEVICE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
ALLOWED_TASK_STATUSES = {'ready', 'running', 'awaiting_confirmation', 'completed', 'blocked'}
ALLOWED_EXECUTION_CLASSES = {'observe', 'navigate', 'effect', 'unknown'}
ALLOWED_INPUT_OPERATIONS = {'focus', 'input_verified_text', 'clear_verified_text', 'press_enter'}


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

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> 'QwenTaskContext':
        reject_if(not isinstance(value, dict), VisionAgentError("Qwen任务上下文必须是JSON对象。"))
        required = {'protocol_version', 'task_id', 'device_id', 'revision', 'task_status', 'goal', 'global_constraints',
            'goal_completion_conditions', 'current_subgoal', 'current_execution_class', 'effect_intents', 'effect_gate'}
        missing = required - set(value)
        unexpected = set(value) - required
        reject_if(missing, VisionAgentError('Qwen任务上下文缺少字段：' + ', '.join(sorted(missing))))
        reject_if(unexpected, VisionAgentError('Qwen任务上下文包含协议外字段：' + ', '.join(sorted(unexpected))))

        context = cls(protocol_version=str(value['protocol_version'] or '').strip(),
            task_id=str(value['task_id'] or '').strip(), device_id=str(value['device_id'] or '').strip(),
            revision=value['revision'], task_status=str(value['task_status'] or '').strip(),
            goal=_require_dict(value['goal'], 'goal'), global_constraints=_text_tuple(value['global_constraints'],
            'global_constraints'), goal_completion_conditions=_dict_tuple(value['goal_completion_conditions'],
            'goal_completion_conditions'), current_subgoal=_require_dict(value['current_subgoal'], 'current_subgoal'),
            current_execution_class=str(value['current_execution_class'] or '').strip(),
            effect_intents=_dict_tuple(value['effect_intents'], 'effect_intents'),
            effect_gate=_require_dict(value['effect_gate'], 'effect_gate'))
        context.validate()
        return context

    def validate(self) -> None:
        reject_if(self.protocol_version != SUPPORTED_TASK_CONTEXT_PROTOCOL, VisionAgentError(f'不支持的DeepSeek任务上下文协议：{self.protocol_version}'))
        reject_if(not TASK_ID_PATTERN.fullmatch(self.task_id), VisionAgentError(f"task_id 格式无效：{self.task_id!r}"))
        reject_if(not DEVICE_ID_PATTERN.fullmatch(self.device_id), VisionAgentError(f"device_id 格式无效：{self.device_id!r}"))
        reject_if(isinstance(self.revision, bool) or not isinstance(self.revision, int) or self.revision < 1, VisionAgentError("revision 必须是正整数。"))
        reject_if(self.task_status not in ALLOWED_TASK_STATUSES, VisionAgentError(f"task_status 无效：{self.task_status}"))
        reject_if(self.current_execution_class not in ALLOWED_EXECUTION_CLASSES, VisionAgentError(f'current_execution_class 无效：{self.current_execution_class}'))
        subgoal_allowed = {'subgoal_id', 'objective', 'status', 'depends_on', 'constraints', 'completion_conditions',
            'completion_evidence', 'execution_class', 'input_field_id', 'input_operation', 'required_action_kind'}
        unexpected_subgoal = set(self.current_subgoal) - subgoal_allowed
        reject_if(unexpected_subgoal, VisionAgentError('current_subgoal 包含协议外字段：' + ', '.join(sorted(unexpected_subgoal))))
        reject_if(not str(self.current_subgoal.get('subgoal_id') or '').strip(), VisionAgentError("current_subgoal 缺少 subgoal_id。"))
        reject_if(not str(self.current_subgoal.get('objective') or '').strip(), VisionAgentError("current_subgoal 缺少 objective。"))
        reject_if(str(self.current_subgoal.get('status') or '') != 'active', VisionAgentError("Qwen入口只接受 status=active 的 current_subgoal。"))
        reject_if(str(self.current_subgoal.get('execution_class') or '') != self.current_execution_class, VisionAgentError('current_subgoal.execution_class 与顶层上下文不一致。'))
        field_id = str(self.current_subgoal.get('input_field_id') or '').strip()
        operation = str(self.current_subgoal.get('input_operation') or '').strip()
        reject_if(bool(field_id) != bool(operation), VisionAgentError(
            'current_subgoal typed input 绑定必须同时包含 input_field_id 与 input_operation。'))
        reject_if(operation and operation not in ALLOWED_INPUT_OPERATIONS, VisionAgentError(
            f'current_subgoal.input_operation 无效：{operation}'))
        required_action_kind = str(self.current_subgoal.get('required_action_kind') or '').strip()
        reject_if(required_action_kind and required_action_kind not in CANONICAL_ACTION_KINDS,
            VisionAgentError(f'current_subgoal.required_action_kind 无效：{required_action_kind}'))
        reject_if(bool(required_action_kind) and bool(operation), VisionAgentError(
            'current_subgoal 不能同时声明 required_action_kind 与 input_operation。'))
        generic_goal_domain.safe_goal_context(self.to_dict())

        effect_allowed = {'effect_id', 'kind', 'target_entity_roles', 'payload_entity_roles', 'source_subgoal_ids',
            'expected_results', 'local_policy'}
        policy_allowed = {"effect_id", "confirmation_required", "policy_level"}
        for (index, item) in enumerate(self.effect_intents):
            reject_if(set(item) != effect_allowed, VisionAgentError(f'effect_intents[{index}] 字段不完整或包含协议外字段。'))
            for field in ('target_entity_roles', 'payload_entity_roles', 'source_subgoal_ids', 'expected_results'):
                _text_tuple(item[field], f"effect_intents[{index}].{field}")
            reject_if(not str(item.get('kind') or '').strip(), VisionAgentError(f"effect_intents[{index}].kind 不能为空。"))
            policy = _require_dict(item.get('local_policy'), f'effect_intents[{index}].local_policy')
            reject_if(set(policy) != policy_allowed, VisionAgentError(f'effect_intents[{index}].local_policy 字段不完整或包含协议外字段。'))
            reject_if(policy.get('effect_id') != item.get('effect_id'), VisionAgentError(f'effect_intents[{index}].local_policy.effect_id 不一致。'))
            reject_if(not isinstance(policy.get('confirmation_required'), bool), VisionAgentError(f'effect_intents[{index}].local_policy.confirmation_required 必须是布尔值。'))
            reject_if(not str(policy.get('policy_level') or '').strip(), VisionAgentError(f'effect_intents[{index}].local_policy.policy_level 不能为空。'))

        effect_ids = [str(item.get("effect_id") or "").strip() for item in self.effect_intents]
        reject_if(any((not item for item in effect_ids)) or len(effect_ids) != len(set(effect_ids)), VisionAgentError("effect_intents 含空ID或重复ID。"))
        gate_allowed = {'state', 'effect_action_allowed', 'scope'}
        reject_if(set(self.effect_gate) != gate_allowed, VisionAgentError("effect_gate 字段不完整或包含协议外字段。"))
        allowed = self.effect_gate.get("effect_action_allowed")
        reject_if(not isinstance(allowed, bool), VisionAgentError("effect_gate.effect_action_allowed 格式无效。"))
        state = str(self.effect_gate.get("state") or "").strip()
        reject_if(state not in {'not_required', 'awaiting_confirmation', 'confirmed'}, VisionAgentError(f"effect_gate.state 无效：{state}"))
        confirmation_effect_ids = {str(item.get('effect_id') or '').strip() for item
            in self.effect_intents if isinstance(item.get('local_policy'),
            dict) and item['local_policy'].get('confirmation_required') is True}

        scope = _require_dict(self.effect_gate.get('scope'), 'effect_gate.scope')
        scope_allowed = {"task_id", "device_id", "revision", "subgoal_id"}
        reject_if(set(scope) != scope_allowed, VisionAgentError('effect_gate.scope 字段缺失或包含协议外字段。'))
        expected_scope = {'task_id': self.task_id, 'device_id': self.device_id, 'revision': self.revision,
            'subgoal_id': str(self.current_subgoal['subgoal_id'])}
        for (field, expected) in expected_scope.items():
            reject_if(type(scope[field]) is not type(expected) or scope[field] != expected, VisionAgentError(f'effect_gate.scope.{field} 与当前上下文不一致。'))

        external = self.current_execution_class in {"effect", "unknown"}
        reject_if(self.current_execution_class == 'unknown', VisionAgentError("unknown 子目标禁止进入视觉动作协议。"))
        if external and confirmation_effect_ids:
            reject_if(state not in {'awaiting_confirmation', 'confirmed'}, VisionAgentError("外部状态子目标的确认门状态无效。"))
            reject_if(state == 'confirmed' and (not allowed), VisionAgentError("确认门状态与 effect_action_allowed 冲突。"))
            reject_if(state != 'confirmed' and allowed, VisionAgentError("未确认效果不能允许受限效果动作。"))
        elif external:
            reject_if(state != 'not_required' or (not allowed), VisionAgentError("自动外部效果的本地策略授权状态无效。"))
        elif allowed or (state != 'not_required'):
            raise VisionAgentError("只读/导航子目标不得伪造效果确认状态。")

    @property
    def effect_action_allowed(self) -> bool:
        return bool(self.effect_gate['effect_action_allowed'])

    @property
    def requested_input_text(self) -> str | None:
        """Return the exact text authorized by DeepSeek, never model-invented text."""

        entities = self.goal.get("entities") or {}
        field_id = str(self.current_subgoal.get('input_field_id') or '').strip()
        operation = str(self.current_subgoal.get('input_operation') or '').strip()
        if not field_id or operation != 'input_verified_text':
            return None
        if field_id == 'primary_input':
            raw = entities.get("input_text")
        else:
            fields = entities.get('input_fields')
            reject_if(not isinstance(fields, list), VisionAgentError(
                '当前子目标引用多字段输入，但 goal.entities.input_fields 不存在。'))
            matches = [item for item in fields if isinstance(item, dict)
                and str(item.get('field_id') or '') == field_id]
            reject_if(len(matches) != 1, VisionAgentError(
                f'当前子目标 typed input field 不是唯一声明字段：{field_id}'))
            raw = matches[0].get('text')
        if raw is None:
            return None
        reject_if(not isinstance(raw, str) or not raw or len(raw) > 4000 or ('\r' in raw), VisionAgentError("goal.entities.input_text 必须为1～4000个字符。"))
        return raw

    def to_dict(self) -> dict[str, Any]:
        return dataclass_wire(self)

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())

def _require_dict(value: Any, name: str) -> dict[str, Any]:
    reject_if(not isinstance(value, dict), VisionAgentError(f"{name} 必须是JSON对象。"))
    return dict(value)


def _text_tuple(value: Any, name: str) -> tuple[str, ...]:
    reject_if(not isinstance(value, (list, tuple)), VisionAgentError(f"{name} 必须是字符串数组。"))
    result = tuple(str(item).strip()[:500] for item in value)
    reject_if(any((not item for item in result)), VisionAgentError(f"{name} 不能包含空字符串。"))
    return result


def _dict_tuple(value: Any, name: str) -> tuple[dict[str, Any], ...]:
    reject_if(not isinstance(value, (list, tuple)), VisionAgentError(f"{name} 必须是对象数组。"))
    reject_if(any((not isinstance(item, dict) for item in value)), VisionAgentError(f"{name} 只能包含JSON对象。"))
    return tuple(dict(item) for item in value)
