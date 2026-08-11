from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from intent_provider import IntentProviderError
from task_orchestrator import GoalSpec, TaskPlanError


GENERIC_INTENT_PROTOCOL_VERSION = "2026-08-10-generic-intent-v1"
APP_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")


class JsonIntentProvider(Protocol):
    configured: bool

    def chat_json(self, messages: list[dict[str, Any]], max_tokens: int = 500) -> str: ...


class GenericIntentError(ValueError):
    pass


@dataclass(frozen=True)
class GenericIntentDraft:
    understood: bool
    app_id: str = ""
    app_name: str = ""
    objective: str = ""
    entities: dict[str, Any] = field(default_factory=dict)
    constraints: tuple[str, ...] = ()
    success_criteria: dict[str, Any] = field(default_factory=dict)
    account_effects: tuple[str, ...] = ()
    message: str = ""
    needs_confirmation: bool = True
    protocol_version: str = GENERIC_INTENT_PROTOCOL_VERSION

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
        if self.needs_confirmation is not True:
            raise GenericIntentError("通用任务必须经过人工确认。")
        _reject_control_fields(self.entities, "entities")
        _reject_control_fields(self.success_criteria, "success_criteria")
        for value in (*self.constraints, *self.account_effects):
            if not isinstance(value, str) or not value.strip():
                raise GenericIntentError("约束和账号影响必须是非空字符串。")

    def to_goal_spec(self) -> GoalSpec:
        self.validate()
        if not self.understood:
            raise GenericIntentError("信息不足的草稿不能转换为可执行目标。")
        try:
            return GoalSpec.from_dynamic(
                app_id=self.app_id,
                objective=self.objective,
                parameters={
                    "app_name": self.app_name,
                    "entities": dict(self.entities),
                    "constraints": list(self.constraints),
                    "account_effects": list(self.account_effects),
                },
                success_criteria=dict(self.success_criteria),
                limits={"single_action_loop": True, "max_steps": 100},
            )
        except TaskPlanError as exc:
            raise GenericIntentError(str(exc)) from exc

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        value = asdict(self)
        value["constraints"] = list(self.constraints)
        value["account_effects"] = list(self.account_effects)
        return value


class GenericIntentParser:
    """DeepSeek extracts goals only; it cannot author device actions."""

    def __init__(self, provider: JsonIntentProvider) -> None:
        self.provider = provider

    def parse(self, raw_text: str) -> GenericIntentDraft:
        text = " ".join(raw_text.strip().split())
        if not text:
            return GenericIntentDraft(
                understood=False,
                message="请输入要完成的手机操作目标。",
            )
        if not self.provider.configured:
            raise IntentProviderError("DeepSeek 文本理解尚未配置。")
        prompt = f"""
你是通用手机视觉操作 Agent 的目标理解层。你只提取用户想完成的结果，
不能规划点击步骤，不能输出坐标，不能调用工具。

返回一个 JSON 对象，且只允许这些字段：
{{
  "understood": true或false,
  "app_id": "稳定的小写英文ID，例如 wechat、douyin、calculator",
  "app_name": "用户所说的App名称",
  "objective": "不增加用户未要求内容的明确目标",
  "entities": {{"目标对象、文本、数量、搜索词等": "值"}},
  "constraints": ["用户明确限制"],
  "success_criteria": {{"可从手机页面验证的完成条件": "值"}},
  "account_effects": ["send_message、like、comment、follow等账号变化，没有则空数组"],
  "message": "信息不足时说明缺少什么"
}}

规则：
1. 适用于任意 App，不得把任务限制为微信或抖音。
2. 不得输出 action、steps、tap、swipe、coordinate、x、y、shell、command。
3. 信息不足或 App 不明确时 understood=false，不能猜。
4. success_criteria 必须描述结果证据，不能描述操作步骤。
5. 常见指令也必须完整理解，不使用本地关键词替代。

用户原文：{json.dumps(text, ensure_ascii=False)}
"""
        raw = self.provider.chat_json(
            [{"role": "user", "content": prompt}],
            max_tokens=800,
        )
        payload = _parse_json_object(raw)
        allowed_keys = {
            "understood",
            "app_id",
            "app_name",
            "objective",
            "entities",
            "constraints",
            "success_criteria",
            "account_effects",
            "message",
        }
        unexpected = set(payload) - allowed_keys
        if unexpected:
            raise GenericIntentError(
                "文本模型返回了不允许的字段：" + ", ".join(sorted(unexpected))
            )
        understood = payload.get("understood") is True
        draft = GenericIntentDraft(
            understood=understood,
            app_id=str(payload.get("app_id") or "").strip().lower(),
            app_name=str(payload.get("app_name") or "").strip(),
            objective=str(payload.get("objective") or "").strip(),
            entities=dict(payload.get("entities") or {}),
            constraints=tuple(str(item).strip() for item in payload.get("constraints") or []),
            success_criteria=dict(payload.get("success_criteria") or {}),
            account_effects=tuple(
                str(item).strip() for item in payload.get("account_effects") or []
            ),
            message=str(payload.get("message") or "").strip(),
            needs_confirmation=True,
        )
        draft.validate()
        return draft


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


def _reject_control_fields(value: Any, path: str) -> None:
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
        "shell",
        "command",
        "execution_plan",
    }
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).strip().lower() in forbidden:
                raise GenericIntentError(f"目标草稿包含控制字段：{path}.{key}")
            _reject_control_fields(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_control_fields(item, f"{path}[{index}]")
    elif not isinstance(value, (str, int, float, bool, type(None))):
        raise GenericIntentError(f"目标参数类型不受支持：{path}")
