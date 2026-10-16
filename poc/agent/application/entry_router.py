"""Platform entry routing for natural-language requests.

The entry router only decides whether a message is conversational or should
create a Runtime task.  It never selects a canonical device action, approves
an effect, or completes a task.  The selected Model Gateway is the provider
boundary; provider-specific names must not appear in this contract.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import json
from typing import Any

AGENT_ENTRY_ROUTE_PROTOCOL_VERSION = "2026-10-16-agent-entry-route-v1"

AGENT_ENTRY_ROUTE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "agent_entry_route",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "route": {"type": "string", "enum": ["chat_only", "phone_task"]},
                "reply": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["route", "reply", "reason"],
        },
    },
}

AGENT_ENTRY_ROUTER_SYSTEM_PROMPT = """你是本项目 Agent 平台的入口路由器。
你只能根据用户消息和已有对话判断本轮是否需要创建手机任务：
- chat_only：只回答问题，不需要读取手机画面，也不创建手机任务；
- phone_task：用户明确要求手机完成目标，或必须读取手机当前画面才能继续。
你不是 Planner、Executor 或 Verifier，不得选择 canonical 动作、放行风险或声称任务已经完成。
只返回 JSON，不要 Markdown：
{"route":"chat_only或phone_task","reply":"给用户的自然语言回答","reason":"简短判断"}
不要声称已经查看手机或已经执行动作。phone_task 只表示准备把目标交给 Runtime。
"""


@dataclass(frozen=True)
class AgentEntryRoute:
    route: str
    reply: str
    reason: str

    def validate(self) -> None:
        if self.route not in {"chat_only", "phone_task"}:
            raise AgentEntryRouteError("Agent 入口路由缺少有效 route。")
        if not isinstance(self.reply, str) or not isinstance(self.reason, str):
            raise AgentEntryRouteError("Agent 入口路由的 reply/reason 必须是字符串。")

    def to_dict(self) -> dict[str, str]:
        self.validate()
        return {
            "protocol_version": AGENT_ENTRY_ROUTE_PROTOCOL_VERSION,
            "route": self.route,
            "reply": self.reply.strip(),
            "reason": self.reason.strip(),
        }


class AgentEntryRouter:
    """Turn one user message into a routing suggestion from the Model Gateway."""

    def __init__(self, model_gateway: Any) -> None:
        self.model_gateway = model_gateway

    def route(
        self,
        *,
        text: str,
        conversation: Iterable[Mapping[str, str]] = (),
    ) -> AgentEntryRoute:
        message = str(text or "").strip()
        if not message:
            raise AgentEntryRouteError("消息不能为空。")
        history = [
            {"role": str(item.get("role") or ""), "content": str(item.get("content") or "")}
            for item in conversation
        ]
        messages = [
            {"role": "system", "content": AGENT_ENTRY_ROUTER_SYSTEM_PROMPT},
            *history,
            {"role": "user", "content": message},
        ]
        chat = getattr(self.model_gateway, "chat", None)
        if not callable(chat):
            raise AgentEntryRouteError("默认 Model Gateway 缺少公开 chat 合同。")
        call_scope = getattr(self.model_gateway, "call_scope", None)
        context = call_scope(stage="agent_entry_route") if callable(call_scope) else _null_context()
        with context:
            raw = chat(
                messages,
                max_tokens=512,
                response_format=AGENT_ENTRY_ROUTE_RESPONSE_FORMAT,
            )
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise AgentEntryRouteError(f"Agent 入口路由不是有效 JSON：{exc}") from exc
        if not isinstance(payload, dict):
            raise AgentEntryRouteError("Agent 入口路由必须是 JSON 对象。")
        result = AgentEntryRoute(
            route=str(payload.get("route") or ""),
            reply=str(payload.get("reply") or ""),
            reason=str(payload.get("reason") or ""),
        )
        result.validate()
        return result


def _null_context():
    from contextlib import nullcontext

    return nullcontext()


class AgentEntryRouteError(RuntimeError):
    """The Model Gateway could not produce a valid entry routing suggestion."""


__all__ = [
    "AGENT_ENTRY_ROUTE_PROTOCOL_VERSION",
    "AGENT_ENTRY_ROUTE_RESPONSE_FORMAT",
    "AGENT_ENTRY_ROUTER_SYSTEM_PROMPT",
    "AgentEntryRoute",
    "AgentEntryRouteError",
    "AgentEntryRouter",
]
