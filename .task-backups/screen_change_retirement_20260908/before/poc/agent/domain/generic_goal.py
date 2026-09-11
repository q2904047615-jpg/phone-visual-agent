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

@dataclass(frozen=True)
class ActiveVisualGoal:
    """Whole-task observation context without planned nodes or a progress pointer."""
    root: dict[str, Any]

    @classmethod
    def from_context(cls, context: dict[str, Any]) -> "ActiveVisualGoal":
        return cls(context)

    @property
    def observation_context(self) -> dict[str, Any]:
        return self.root

    @property
    def field(self) -> tuple[str, str, bool]:
        return "current_input", "", False

    @property
    def input_requested(self) -> bool:
        return True


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
    """Validate local JSON context; factual action history is allowed."""
    reject_if(not isinstance(value, dict), VisionAgentError("目标上下文必须是对象。"))
    _validate_json_value(value, "context")
    return json.loads(json.dumps(value, ensure_ascii=False))
