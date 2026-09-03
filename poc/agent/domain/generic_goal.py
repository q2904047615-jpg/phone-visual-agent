from __future__ import annotations

from .validation import ValidatedDataclassWire, reject_if
import json
import re
from dataclasses import dataclass, field
from typing import Any

from .vision_model import VisionAgentError


GOAL_PROJECTION_PROTOCOL = "2026-08-20-typed-goal-projection-v1"
APP_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")


class GenericIntentError(ValueError):
    pass


@dataclass(frozen=True)
class GenericIntentDraft(ValidatedDataclassWire):
    """Read-only task-graph projection with no risk or action authority."""

    understood: bool
    app_id: str = ""
    app_name: str = ""
    objective: str = ""
    entities: dict[str, Any] = field(default_factory=dict)
    constraints: tuple[str, ...] = ()
    success_criteria: dict[str, Any] = field(default_factory=dict)
    account_effects: tuple[str, ...] = ()
    message: str = ""
    needs_confirmation: bool = False
    protocol_version: str = GOAL_PROJECTION_PROTOCOL

    def validate(self) -> None:
        reject_if(not isinstance(self.understood, bool), GenericIntentError("understood 格式无效。"))
        if not self.understood:
            reject_if(not self.message.strip(), GenericIntentError("未理解任务时必须说明缺少的信息。"))
            return
        reject_if(not APP_ID_PATTERN.fullmatch(self.app_id), GenericIntentError(f"App ID 无效：{self.app_id!r}"))
        reject_if(not self.app_name.strip() or not self.objective.strip(), GenericIntentError("通用任务缺少 App 名称或目标。"))
        reject_if(not isinstance(self.needs_confirmation, bool), GenericIntentError("needs_confirmation 格式无效。"))
        _validate_json_value(self.entities, "entities")
        _validate_json_value(self.success_criteria, "success_criteria")
        for value in (*self.constraints, *self.account_effects):
            reject_if(not isinstance(value, str) or not value.strip(), GenericIntentError("约束和账号影响必须是非空字符串。"))

_ACTIVE_VISUAL_REQUIRED_FIELDS = {'subgoal_id', 'objective', 'constraints', 'completion_conditions',
    'execution_class', 'goal_entities'}
_ACTIVE_VISUAL_FIELDS = _ACTIVE_VISUAL_REQUIRED_FIELDS | {'transition_receipt', 'current_effect_kinds',
    'forbidden_future_effect_kinds', 'gesture_correction'}


@dataclass(frozen=True)
class ActiveVisualGoal:
    """Read-only view of the current typed graph node for visual observation."""

    root: dict[str, Any]
    focus: dict[str, Any]
    root_entities: dict[str, Any]
    goal_entities: dict[str, Any]
    has_active_focus: bool

    @classmethod
    def from_context(cls, context: dict[str, Any]) -> 'ActiveVisualGoal':
        root_entities = context.get("entities")
        root_entities = root_entities if isinstance(root_entities, dict) else {}
        candidate = root_entities.get("active_subgoal_visual_context")
        valid = bool(isinstance(candidate, dict)
            and _ACTIVE_VISUAL_REQUIRED_FIELDS.issubset(candidate)
            and set(candidate).issubset(_ACTIVE_VISUAL_FIELDS)
            and str(candidate.get('subgoal_id') or '').strip() and str(candidate.get('objective') or '').strip()
            and isinstance(candidate.get('constraints'), list) and isinstance(candidate.get('completion_conditions'),
            list) and isinstance(candidate.get('goal_entities'), dict))
        focus = candidate if valid else context
        entities = focus.get("goal_entities") if valid else root_entities
        return cls(context, focus, root_entities, entities if isinstance(entities, dict) else {}, valid)

    @property
    def observation_context(self) -> dict[str, Any]:
        if not self.has_active_focus:
            return self.root
        return {key: list(value) if key in {'constraints',
            'completion_conditions'} else dict(value) if key == 'goal_entities' else value for key,
            value in self.focus.items()}

    @property
    def explicit_text(self) -> str:
        value = self.goal_entities.get("input_text")
        return value.strip() if isinstance(value, str) else ""

    @property
    def transaction_text(self) -> str:
        value = self.goal_entities.get("active_input_transaction_text") if self.has_active_focus else None
        return value if isinstance(value, str) and value else ""

    @property
    def field(self) -> tuple[str, str, bool]:
        if not self.has_active_focus:
            return "", "", False
        field_id = self.goal_entities.get("active_input_field_id")
        label = self.goal_entities.get("active_input_field_label", "")
        multiline = self.goal_entities.get("active_input_multiline", False)
        if (not isinstance(field_id, str) or not field_id or (not isinstance(label, str)) or (not isinstance(multiline,
            bool))):
            return "", "", False
        return field_id, label, multiline

    @property
    def target_only(self) -> bool:
        return bool(self.has_active_focus and self.goal_entities.get('active_input_target_only') is True
            and self.field[0])

    @property
    def unique_typed_field(self) -> bool:
        fields = self.root_entities.get("input_fields")
        field_id, label, _ = self.field
        text = self.transaction_text
        if not isinstance(fields, list) or not field_id or (not text):
            return False
        exact = sum((isinstance(item, dict) and item.get('field_id') == field_id and (item.get('field_label') == label)
            and (item.get('text') == text) for item in fields))
        identities = sum(isinstance(item, dict) and item.get("field_id") == field_id for item in fields)
        return exact == identities == 1

    @property
    def predecessor(self) -> tuple[str, str, str]:
        if not self.has_active_focus:
            return "", "", ""
        values = tuple((self.goal_entities.get(key) for key in ('active_input_predecessor_field_id',
            'active_input_predecessor_field_label', 'active_input_predecessor_text')))
        fields = self.root_entities.get("input_fields")
        if not all((isinstance(value, str) and value for value in values)) or not isinstance(fields, list):
            return "", "", ""
        field_id, label, text = values
        active_id, active_label, _ = self.field
        active_text = self.transaction_text
        matches = lambda fid, flabel, value: sum((isinstance(item,
            dict) and item.get('field_id') == fid and (item.get('field_label',
            '') == flabel) and (item.get('text') == value) for item in fields))
        return (field_id, label, text) if field_id != active_id and matches(field_id, label, text) == matches(active_id,
            active_label, active_text) == 1 else ('', '', '')

    @property
    def mode_switch_requested(self) -> bool:
        return False

    @property
    def input_requested(self) -> bool:
        return bool(self.field[0])

    @property
    def clear_requested(self) -> bool:
        return bool(self.input_requested
            and self.goal_entities.get('active_input_operation') == 'clear_verified_text')

    @property
    def has_explicit_text(self) -> bool:
        return bool(self.explicit_text or self.transaction_text)


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()
    if text.startswith('```'):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise GenericIntentError(f"文本模型没有返回有效 JSON：{exc}") from exc
    reject_if(not isinstance(value, dict), GenericIntentError("文本模型返回内容不是 JSON 对象。"))
    return value


def _validate_json_value(value: Any, path: str) -> None:
    if isinstance(value, dict):
        for (key, item) in value.items():
            reject_if(not isinstance(key, str) or not key.strip(), GenericIntentError(f"目标参数字段无效：{path}"))
            _validate_json_value(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for (index, item) in enumerate(value):
            _validate_json_value(item, f"{path}[{index}]")
    elif not isinstance(value, (str, int, float, bool, type(None))):
        raise GenericIntentError(f"目标参数类型不受支持：{path}")


def safe_goal_context(value: dict[str, Any]) -> dict[str, Any]:
    """Keep goal data useful to observation while refusing control fields."""

    forbidden = {'action', 'actions', 'step', 'steps', 'tap', 'swipe', 'coordinate', 'coordinates', 'x', 'y', 'command',
        'shell', 'execution_plan'}

    def clean(item: Any, depth: int=0) -> Any:
        reject_if(depth > 5, VisionAgentError("目标上下文嵌套过深。"))
        if isinstance(item, dict):
            result: dict[str, Any] = {}
            for (raw_key, raw_value) in item.items():
                key = str(raw_key).strip()
                reject_if(key.lower() in forbidden, VisionAgentError(f"目标上下文包含控制字段：{key}"))
                result[key[:80]] = clean(raw_value, depth + 1)
            return result
        if isinstance(item, (list, tuple)):
            return [clean(part, depth + 1) for part in list(item)[:50]]
        if isinstance(item, str):
            return item[:1000]
        if isinstance(item, (int, float, bool)) or item is None:
            return item
        raise VisionAgentError("目标上下文包含不支持的数据类型。")

    cleaned = clean(value)
    reject_if(not isinstance(cleaned, dict), VisionAgentError("目标上下文必须是对象。"))
    return cleaned
