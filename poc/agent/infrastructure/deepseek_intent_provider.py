"""DeepSeek HTTP provider for the task-graph application service."""

from __future__ import annotations

from agent.domain.validation import reject_if
import os
import time
from typing import Any

import httpx


DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"


def _read_windows_user_environment(name: str) -> str:
    """Read a user-scoped variable when the parent process has stale env data."""
    if os.name != 'nt':
        return ""
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Environment') as key:
            value, _ = winreg.QueryValueEx(key, name)
    except (FileNotFoundError, OSError):
        return ""
    return str(value or "").strip()


def _default_deepseek_api_key() -> str:
    return os.getenv('DEEPSEEK_API_KEY', '').strip() or _read_windows_user_environment('DEEPSEEK_API_KEY')


class IntentProviderError(RuntimeError):
    """Raised when the text-intent provider cannot return a safe result."""


class DeepSeekIntentProvider:
    """OpenAI-compatible DeepSeek adapter used only for intent extraction."""

    TRANSIENT_HTTP_STATUS_CODES = {408, 429, 500, 502, 503, 504}

    def __init__(self, *, api_key: str | None=None, model: str | None=None, base_url: str | None=None,
        timeout: float=30.0, max_attempts: int=3, retry_base_delay: float=0.8) -> None:
        self.api_key = api_key if api_key is not None else _default_deepseek_api_key()
        self.model = model or os.getenv('DEEPSEEK_INTENT_MODEL', DEFAULT_DEEPSEEK_MODEL)
        self.base_url = (base_url or os.getenv('DEEPSEEK_BASE_URL', DEFAULT_DEEPSEEK_BASE_URL)).rstrip('/')
        self.timeout = max(1.0, float(timeout))
        self.max_attempts = max(1, int(max_attempts))
        self.retry_base_delay = max(0.0, float(retry_base_delay))
        self.last_usage: dict[str, Any] = {}
        self.last_request_id = ""
        self.last_network_attempts = 0

    @property
    def configured(self) -> bool:
        return bool(self.api_key.strip())

    def status(self) -> dict[str, Any]:
        return {'provider': 'deepseek', 'role': 'structured_text_intent_only', 'model': self.model,
            'configured': self.configured, 'base_url': self.base_url, 'thinking': 'disabled',
            'last_usage': self.last_usage, 'last_request_id': self.last_request_id,
            'last_network_attempts': self.last_network_attempts,
            'error': None if self.configured else '未配置 DEEPSEEK_API_KEY'}

    def chat_json(self, messages: list[dict[str, Any]], max_tokens: int=500) -> str:
        reject_if(not self.configured, IntentProviderError('DeepSeek 文本理解尚未配置：请先设置 DEEPSEEK_API_KEY。'))

        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            self.last_network_attempts = attempt
            try:
                response = httpx.post(f'{self.base_url}/chat/completions', headers={'Authorization': f'Bearer {
                    self.api_key}', 'Content-Type': 'application/json'}, json={'model': self.model,
                    'messages': messages, 'thinking': {'type': 'disabled'}, 'response_format': {'type': 'json_object'},
                    'temperature': 0.0, 'max_tokens': max_tokens}, timeout=self.timeout)
                response.raise_for_status()
                payload = response.json()
                self.last_usage = payload.get("usage") or {}
                self.last_request_id = str(payload.get("id") or "")
                content = payload["choices"][0]["message"]["content"]
                reject_if(not isinstance(content, str) or not content.strip(), IntentProviderError("DeepSeek 文本理解返回了空内容。"))
                return content
            except httpx.HTTPStatusError as exc:
                last_error = exc
                status_code = exc.response.status_code
                if status_code not in self.TRANSIENT_HTTP_STATUS_CODES or attempt >= self.max_attempts:
                    detail = exc.response.text[:500]
                    raise IntentProviderError(f'DeepSeek 文本理解请求失败（HTTP {status_code}）：{detail}') from exc
            except httpx.TimeoutException as exc:
                last_error = exc
                if attempt >= self.max_attempts:
                    raise IntentProviderError(f'DeepSeek 文本理解连续{attempt}次超时。') from exc
            except httpx.TransportError as exc:
                last_error = exc
                if attempt >= self.max_attempts:
                    raise IntentProviderError(f'DeepSeek 文本理解连接连续{attempt}次中断：{exc}') from exc
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                raise IntentProviderError(f'DeepSeek 文本理解响应格式无效：{exc}') from exc

            if self.retry_base_delay > 0:
                time.sleep(self.retry_base_delay * (2 ** (attempt - 1)))

        raise IntentProviderError(f"DeepSeek 文本理解连接失败：{last_error}")
