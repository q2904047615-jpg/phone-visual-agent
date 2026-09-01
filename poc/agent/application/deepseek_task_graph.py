from __future__ import annotations

from agent.domain.validation import reject_if
import json
from functools import lru_cache
from importlib.resources import files
import uuid
from typing import Any, Protocol

from agent.domain.generic_goal import GenericIntentError, _parse_json_object
from agent.domain.task_graph import (
    DynamicTaskGraph,
    TaskGraphError,
    _graph_from_payload,
    _validate_device_id,
    _validate_task_id,
)

__all__ = ("DeepSeekTaskGraphPlanner", "JsonTaskGraphProvider")

class JsonTaskGraphProvider(Protocol):
    configured: bool

    def chat_json(self, messages: list[dict[str, Any]], max_tokens: int = 2000) -> str: ...

class DeepSeekTaskGraphPlanner:
    """Create one high-level task graph, then leave the screenshot loop."""

    def __init__(self, provider: JsonTaskGraphProvider) -> None:
        self.provider = provider
        self.last_raw_response = ""

    def plan(
        self,
        raw_goal: str,
        *,
        device_id: str,
        task_id: str | None = None,
    ) -> DynamicTaskGraph:
        # Internal whitespace can be literal user payload.  In particular, a
        # line feed is an authorized input character that must survive into
        # the typed graph unchanged.
        text = str(raw_goal or "").strip()
        reject_if(not text, TaskGraphError("用户目标不能为空。"))
        _validate_device_id(device_id)
        resolved_task_id = task_id or uuid.uuid4().hex
        _validate_task_id(resolved_task_id)
        self._require_provider()
        prompt = _initial_prompt(text)
        graph = self._request_graph(
            prompt,
            task_id=resolved_task_id,
            device_id=device_id,
            revision=1,
            raw_user_goal=text,
        )
        graph.validate()
        return graph

    def _request_graph(self, prompt: str, *, task_id: str, device_id: str, revision: int,
        raw_user_goal: str) -> DynamicTaskGraph:
        raw = self.provider.chat_json([{'role': 'user', 'content': prompt}], max_tokens=2400)
        self.last_raw_response = raw
        try:
            payload = _parse_json_object(raw)
        except GenericIntentError as exc:
            raise TaskGraphError(str(exc)) from exc
        graph = _graph_from_payload(
            payload,
            task_id=task_id,
            device_id=device_id,
            revision=revision,
            raw_user_goal=raw_user_goal,
        )
        graph.validate()
        return graph

    def _require_provider(self) -> None:
        reject_if(not self.provider.configured, TaskGraphError("DeepSeek 动态任务图尚未配置。"))

@lru_cache(maxsize=3)
def _prompt_template(name: str) -> str:
    with files(__package__).joinpath("prompts", name).open("r", encoding="utf-8") as stream:
        return stream.read()


def _render_prompt(name: str, **values: str) -> str:
    template = _prompt_template(name)
    for key, value in values.items():
        template = template.replace("{{" + key + "}}", value)
    return template


def _initial_prompt(raw_goal: str) -> str:
    return _render_prompt("deepseek_initial.txt", RAW_GOAL=json.dumps(raw_goal, ensure_ascii=False),
        SCHEMA=_schema_prompt())


def _schema_prompt() -> str:
    return _prompt_template("deepseek_schema.txt").rstrip("\n")
