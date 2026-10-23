"""Minimal OpenAI-compatible client for Qwen-VL on DashScope."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from pva.config import model_settings


class ModelError(RuntimeError):
    pass


@dataclass
class ModelReply:
    content: str
    usage: dict[str, int] = field(default_factory=dict)
    request_id: str = ""
    model: str = ""
    elapsed_seconds: float = 0.0
    reasoning: str = ""


class VisionModel(Protocol):
    def complete(self, messages: list[dict[str, Any]], *, enable_thinking: bool = False) -> ModelReply: ...

    def status(self) -> dict[str, Any]: ...


class QwenClient:
    TRANSIENT = {408, 429, 500, 502, 503, 504}

    def __init__(self, *, api_key: str | None = None, model: str | None = None, base_url: str | None = None,
                 timeout: float = 180.0, attempts: int = 3):
        settings = model_settings()
        self.api_key = api_key if api_key is not None else settings["api_key"]
        self.model = model or settings["model"]
        self.base_url = (base_url or settings["base_url"]).rstrip("/")
        self.timeout = timeout
        self.attempts = attempts
        self.calls = 0
        self.tokens = 0

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def status(self) -> dict[str, Any]:
        return {"provider": "dashscope", "model": self.model, "base_url": self.base_url,
                "configured": self.configured, "calls": self.calls, "total_tokens": self.tokens,
                "error": None if self.configured else "未设置 DASHSCOPE_API_KEY 环境变量"}

    def complete(self, messages: list[dict[str, Any]], *, enable_thinking: bool = False) -> ModelReply:
        if not self.configured:
            raise ModelError("千问未配置：请设置 DASHSCOPE_API_KEY 后重启控制台。")
        body = {"model": self.model, "messages": messages, "temperature": 0.0,
                "enable_thinking": bool(enable_thinking), "vl_high_resolution_images": True}
        started = time.perf_counter()
        last_error = ""
        for attempt in range(1, self.attempts + 1):
            try:
                response = httpx.post(f"{self.base_url}/chat/completions", json=body, timeout=self.timeout,
                                      headers={"Authorization": f"Bearer {self.api_key}"})
            except httpx.TimeoutException:
                last_error = f"请求超时（{self.timeout:g} 秒）"
            except httpx.TransportError as exc:
                last_error = f"网络错误：{exc}"
            else:
                if response.status_code == 200:
                    return self._reply(response, started)
                last_error = f"HTTP {response.status_code}：{response.text[:300]}"
                if response.status_code not in self.TRANSIENT:
                    break
            if attempt < self.attempts:
                time.sleep(1.5 * attempt)
        raise ModelError(f"千问请求失败：{last_error}")

    def _reply(self, response: httpx.Response, started: float) -> ModelReply:
        try:
            payload = response.json()
            message = payload["choices"][0]["message"]
            content = message.get("content") or ""
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ModelError("千问响应格式异常。") from exc
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        usage = {k: v for k, v in (payload.get("usage") or {}).items() if isinstance(v, int)}
        self.calls += 1
        self.tokens += usage.get("total_tokens", 0)
        return ModelReply(content=str(content), usage=usage, request_id=str(payload.get("id") or ""),
                          model=str(payload.get("model") or self.model),
                          elapsed_seconds=round(time.perf_counter() - started, 2),
                          reasoning=str(message.get("reasoning_content") or ""))


def extract_json(text: str) -> dict[str, Any]:
    """Return the first JSON object in a model reply (tolerates code fences and prose)."""
    cleaned = re.sub(r"```(?:json)?", "", text or "").strip()
    start = cleaned.find("{")
    while start != -1:
        depth, in_string, escaped = 0, False, False
        for index in range(start, len(cleaned)):
            char = cleaned[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        value = json.loads(cleaned[start:index + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(value, dict):
                        return value
                    break
        start = cleaned.find("{", start + 1)
    raise ModelError("模型回复中没有可解析的 JSON 对象。")
