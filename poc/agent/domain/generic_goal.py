from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .vision_model import VisionAgentError


GOAL_PROJECTION_PROTOCOL = "2026-08-20-typed-goal-projection-v1"
APP_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")


class GenericIntentError(ValueError):
    pass


@dataclass(frozen=True)
class GenericIntentDraft:
    """Read-only projection of the authoritative typed task graph.

    This carrier has no risk or action veto.  Risk is decided by the typed
    policy and action availability by the canonical action catalog.
    """

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
        if not isinstance(self.understood, bool):
            raise GenericIntentError("understood 格式无效。")
        if not self.understood:
            if not self.message.strip():
                raise GenericIntentError("未理解任务时必须说明缺少的信息。")
            return
        if not APP_ID_PATTERN.fullmatch(self.app_id):
            raise GenericIntentError(f"App ID 无效：{self.app_id!r}")
        if not self.app_name.strip() or not self.objective.strip():
            raise GenericIntentError("通用任务缺少 App 名称或目标。")
        if not isinstance(self.needs_confirmation, bool):
            raise GenericIntentError("needs_confirmation 格式无效。")
        _validate_json_value(self.entities, "entities")
        _validate_json_value(self.success_criteria, "success_criteria")
        for value in (*self.constraints, *self.account_effects):
            if not isinstance(value, str) or not value.strip():
                raise GenericIntentError("约束和账号影响必须是非空字符串。")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        value = asdict(self)
        value["constraints"] = list(self.constraints)
        value["account_effects"] = list(self.account_effects)
        return value


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise GenericIntentError(f"文本模型没有返回有效 JSON：{exc}") from exc
    if not isinstance(value, dict):
        raise GenericIntentError("文本模型返回内容不是 JSON 对象。")
    return value


def _validate_json_value(value: Any, path: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                raise GenericIntentError(f"目标参数字段无效：{path}")
            _validate_json_value(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{path}[{index}]")
    elif not isinstance(value, (str, int, float, bool, type(None))):
        raise GenericIntentError(f"目标参数类型不受支持：{path}")


def safe_goal_context(value: dict[str, Any]) -> dict[str, Any]:
    """Keep goal data useful to observation while refusing control fields."""

    forbidden = {
        "action",
        "actions",
        "step",
        "steps",
        "tap",
        "swipe",
        "coordinate",
        "coordinates",
        "x",
        "y",
        "command",
        "shell",
        "execution_plan",
    }

    def clean(item: Any, depth: int = 0) -> Any:
        if depth > 5:
            raise VisionAgentError("目标上下文嵌套过深。")
        if isinstance(item, dict):
            result: dict[str, Any] = {}
            for raw_key, raw_value in item.items():
                key = str(raw_key).strip()
                if key.lower() in forbidden:
                    raise VisionAgentError(f"目标上下文包含控制字段：{key}")
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
    if not isinstance(cleaned, dict):
        raise VisionAgentError("目标上下文必须是对象。")
    return cleaned
