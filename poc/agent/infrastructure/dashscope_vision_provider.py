"""DashScope visual-model transport and bounded JPEG encoding."""

from __future__ import annotations

from agent.domain.validation import reject_if
import base64
import binascii
import contextvars
import json
import os
import re
import threading
import time
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from typing import Any, Iterator

import httpx
from PIL import Image

from agent.domain.vision_model import VisionAgentError, VisionModelConfig
from agent.infrastructure.environment_vision_model_config import load_vision_model_config
from agent.application.vision_usage import VisionSessionUsageLedger
from agent.infrastructure.atomic_files import atomic_replace_bytes, json_bytes


class _DuplicateJSONKeyError(ValueError):
    pass


def _validated_response_format(value: Any) -> dict[str, Any] | None:
    """Accept only the two documented, bounded structured-output modes."""

    if value is None:
        return None
    if value == {'type': 'json_object'}:
        return {'type': 'json_object'}
    reject_if(not isinstance(value, dict) or set(value) != {'type', 'json_schema'}
        or value.get('type') != 'json_schema', VisionAgentError("千问视觉 response_format 不符合受支持的结构化输出协议。"))
    wrapper = value.get('json_schema')
    reject_if(not isinstance(wrapper, dict) or set(wrapper) != {'name', 'strict', 'schema'},
        VisionAgentError("千问视觉 json_schema 封装结构无效。"))
    name = wrapper.get('name')
    schema = wrapper.get('schema')
    reject_if(not isinstance(name, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_-]{0,63}', name),
        VisionAgentError("千问视觉 json_schema.name 无效。"))
    reject_if(wrapper.get('strict') is not True or not isinstance(schema, dict)
        or schema.get('type') != 'object' or schema.get('additionalProperties') is not False,
        VisionAgentError("千问视觉严格 json_schema 必须是禁止额外字段的对象。"))
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise VisionAgentError("千问视觉 json_schema 不能序列化为有效 JSON。") from exc
    return value


def _reject_duplicate_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for (key, item) in pairs:
        reject_if(key in value, _DuplicateJSONKeyError(key))
        value[key] = item
    return value


def _extract_json_object(raw: str, *, reject_duplicate_keys: bool=False,
    unwrap_singleton_object_array: bool=False) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith('```'):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    load_options = {'object_pairs_hook': _reject_duplicate_json_pairs} if reject_duplicate_keys else {}
    try:
        value = json.loads(text, **load_options)
    except _DuplicateJSONKeyError as exc:
        raise VisionAgentError(f"模型返回的 JSON 包含重复字段：{exc}") from exc
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        reject_if(start < 0 or end <= start, VisionAgentError("模型没有返回 JSON 对象。"))
        try:
            value = json.loads(text[start : end + 1], **load_options)
        except _DuplicateJSONKeyError as exc:
            raise VisionAgentError(f'模型返回的 JSON 包含重复字段：{exc}') from exc
        except json.JSONDecodeError as exc:
            raise VisionAgentError(f"模型返回的 JSON 无法解析：{exc}") from exc
    if unwrap_singleton_object_array and isinstance(value, list):
        reject_if(len(value) != 1 or not isinstance(value[0], dict),
            VisionAgentError("模型返回的单步观察数组必须恰好包含一个 JSON 对象。"))
        value = value[0]
    reject_if(not isinstance(value, dict), VisionAgentError("模型返回值必须是 JSON 对象。"))
    return value


def _image_request_size(image: Image.Image) -> tuple[int, int]:
    """Return the exact JPEG dimensions sent to the visual model."""
    return image.width, image.height


def _image_data_url(image: Image.Image) -> str:
    """Encode a readable, bounded JPEG for visual-model requests."""

    result = image.convert("RGB")
    request_size = _image_request_size(result)
    if result.size != request_size:
        result = result.resize(request_size, Image.Resampling.LANCZOS)
    buffer = BytesIO()
    result.save(buffer, format="JPEG", quality=82, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _has_only_valid_inline_jpeg_images(messages: list[dict[str, Any]]) -> bool:
    """Return true only when every visual input is a valid inline JPEG."""

    found = False
    prefix = "data:image/jpeg;base64,"
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get('type') != 'image_url':
                continue
            found = True
            image_url = part.get("image_url")
            url = image_url.get("url") if isinstance(image_url, dict) else None
            if not isinstance(url, str) or not url.startswith(prefix):
                return False
            try:
                payload = base64.b64decode(url[len(prefix) :], validate=True)
            except (ValueError, binascii.Error):
                return False
            if len(payload) < 4 or not payload.startswith(b'\xff\xd8'):
                return False
            if not payload.endswith(b'\xff\xd9'):
                return False
    return found


def _is_retryable_dashscope_inline_url_rejection(response: httpx.Response, messages: list[dict[str, Any]]) -> bool:
    if response.status_code != 400 or not _has_only_valid_inline_jpeg_images(messages):
        return False
    detail = response.text.lower()
    return 'internalerror.algo.invalidparameter' in detail and 'provided url does not appear to be valid' in detail


class DashScopeVisionProvider:
    """Minimal OpenAI-compatible transport for the canonical visual observers."""

    TRANSIENT_HTTP_STATUS_CODES = {408, 429, 500, 502, 503, 504}

    def __init__(self, *, api_key: str | None=None, model: str | None=None, base_url: str | None=None,
        model_config: VisionModelConfig | None=None, enable_thinking: bool=False, timeout: float=45.0,
        max_attempts: int=3, retry_base_delay: float=0.8) -> None:
        reject_if(model_config is not None and (model is not None or base_url is not None), ValueError("model_config 不能与 model/base_url 同时传入。"))
        self.api_key = api_key if api_key is not None else os.getenv("DASHSCOPE_API_KEY", "")
        self.model_config = model_config or load_vision_model_config(model=model, base_url=base_url,
            enable_thinking=enable_thinking)
        self.model = self.model_config.model
        self.base_url = self.model_config.base_url
        self.timeout = timeout
        self.max_attempts = max(1, int(max_attempts))
        self.retry_base_delay = max(0.0, float(retry_base_delay))
        self.last_usage: dict[str, Any] = {}
        self.usage_totals = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
        self.successful_call_count = 0
        self.last_request_id = ""
        self.last_network_attempts = 0
        self.last_finish_reason = ""
        self.last_response_model = ""
        self._request_lock = threading.RLock()
        self._active_usage_ledger: contextvars.ContextVar[VisionSessionUsageLedger |
            None] = contextvars.ContextVar(f'qwen_usage_ledger_{id(self)}', default=None)
        self._active_call_metadata: contextvars.ContextVar[tuple[str,
            str]] = contextvars.ContextVar(f'qwen_call_metadata_{id(self)}', default=('unscoped', ''))
        self._active_request_evidence: contextvars.ContextVar[Path | None] = contextvars.ContextVar(
            f'qwen_request_evidence_{id(self)}', default=None)

    @property
    def configured(self) -> bool:
        return bool(self.api_key.strip())

    def status(self) -> dict[str, Any]:
        return {'model_config_version': self.model_config.config_version, 'provider': self.model_config.provider,
            'model': self.model, 'thinking_enabled': self.model_config.enable_thinking,
            'coordinate_scale': self.model_config.coordinate_scale, 'configured': self.configured,
            'base_url': self.base_url, 'last_usage': self.last_usage, 'usage_totals': dict(self.usage_totals),
            'successful_call_count': self.successful_call_count, 'last_request_id': self.last_request_id,
            'last_network_attempts': self.last_network_attempts, 'last_finish_reason': self.last_finish_reason,
            'response_model': self.last_response_model, 'error': None if self.configured else '未配置 DASHSCOPE_API_KEY'}

    @contextmanager
    def session_usage_scope(self, ledger: VisionSessionUsageLedger | None) -> Iterator[None]:
        token = self._active_usage_ledger.set(ledger)
        try:
            yield
        finally:
            self._active_usage_ledger.reset(token)

    @contextmanager
    def call_scope(self, *, stage: str, fingerprint: str='',
        request_evidence_path: Path | None=None) -> Iterator[None]:
        token = self._active_call_metadata.set((str(stage or 'unscoped'), str(fingerprint or '')))
        evidence_token = self._active_request_evidence.set(request_evidence_path)
        try:
            yield
        finally:
            self._active_request_evidence.reset(evidence_token)
            self._active_call_metadata.reset(token)

    def _chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int | None,
        *,
        timeout: float | None = None,
        max_attempts: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        # This provider instance is shared by all device sessions.  Its
        # response metadata is intentionally serialized with the request so a
        # second device cannot overwrite request_id/usage before the first
        # session ledger records them.
        with self._request_lock:
            return self._chat_locked(messages, max_tokens, timeout=timeout, max_attempts=max_attempts,
                response_format=response_format)

    def _chat_locked(self, messages: list[dict[str, Any]], max_tokens: int | None, *, timeout: float | None=None,
        max_attempts: int | None=None, response_format: dict[str, Any] | None=None) -> str:
        reject_if(not self.configured, VisionAgentError("千问视觉尚未配置：请先设置 DASHSCOPE_API_KEY。"))
        ledger = self._active_usage_ledger.get()
        stage, fingerprint = self._active_call_metadata.get()
        local_request_id = ""
        if ledger is not None:
            local_request_id = ledger.reserve_request(model=self.model, stage=stage, fingerprint=fingerprint,
                max_completion_tokens=max_tokens)
        request_started = time.perf_counter()
        try:
            content = self._chat_untracked(messages, max_tokens, timeout=timeout, max_attempts=max_attempts,
                response_format=response_format)
        except Exception as exc:
            if ledger is not None and local_request_id:
                if self.last_usage:
                    ledger.record_success(local_request_id, provider_request_id=self.last_request_id,
                        response_model=self.last_response_model, network_attempts=self.last_network_attempts,
                        usage=self.last_usage, finish_reason=self.last_finish_reason,
                        elapsed_seconds=time.perf_counter() - request_started)
                else:
                    ledger.record_failure(local_request_id, network_attempts=self.last_network_attempts, error=exc,
                        elapsed_seconds=time.perf_counter() - request_started)
            raise
        if ledger is not None and local_request_id:
            ledger.record_success(local_request_id, provider_request_id=self.last_request_id,
                response_model=self.last_response_model, network_attempts=self.last_network_attempts,
                usage=self.last_usage, finish_reason=self.last_finish_reason,
                elapsed_seconds=time.perf_counter() - request_started)
        return content

    def _chat_untracked(self, messages: list[dict[str, Any]], max_tokens: int | None, *, timeout: float | None=None,
        max_attempts: int | None=None, response_format: dict[str, Any] | None=None) -> str:
        reject_if(not self.configured, VisionAgentError("千问视觉尚未配置：请先设置 DASHSCOPE_API_KEY。"))
        self.last_usage = {}
        self.last_request_id = ""
        self.last_network_attempts = 0
        self.last_finish_reason = ""
        self.last_response_model = ""
        effective_timeout = self.timeout if timeout is None else max(1.0, float(timeout))
        effective_attempts = self.max_attempts if max_attempts is None else max(1, int(max_attempts))
        reject_if(max_tokens is not None and (isinstance(max_tokens, bool) or not isinstance(max_tokens, int)
            or max_tokens <= 0), VisionAgentError("千问视觉 max_tokens 必须为正整数或省略。"))
        response_format = _validated_response_format(response_format)
        request_body = {'model': self.model, 'messages': messages, 'temperature': 0.0,
            **({'max_tokens': max_tokens} if max_tokens is not None else {}),
            **({'response_format': response_format} if response_format is not None else {}),
            **self.model_config.request_options()}
        evidence_path = self._active_request_evidence.get()
        if evidence_path is not None:
            # Preserve the actual JSON body and encoded images, never auth headers.
            # Saving before I/O also retains requests whose response/parse fails.
            # This is request content evidence, not a claim of server receipt.
            atomic_replace_bytes(evidence_path, json_bytes(request_body))
        last_error: Exception | None = None
        for attempt in range(1, effective_attempts + 1):
            self.last_network_attempts = attempt
            try:
                response = httpx.post(f'{self.base_url}/chat/completions', headers={'Authorization': f'Bearer {
                    self.api_key}', 'Content-Type': 'application/json'}, json=request_body, timeout=effective_timeout)
                response.raise_for_status()
                payload = response.json()
                break
            except httpx.HTTPStatusError as exc:
                last_error = exc
                status_code = exc.response.status_code
                retryable_inline_rejection = (
                    attempt < effective_attempts
                    and _is_retryable_dashscope_inline_url_rejection(exc.response, messages)
                )
                if (status_code not in self.TRANSIENT_HTTP_STATUS_CODES and (not retryable_inline_rejection)
                    or attempt >= effective_attempts):
                    detail = exc.response.text[:500]
                    raise VisionAgentError(f'千问视觉请求失败（HTTP {status_code}）：{detail}') from exc
            except httpx.TimeoutException as exc:
                last_error = exc
                if attempt >= effective_attempts:
                    raise VisionAgentError(f'千问视觉请求连续{attempt}次超时，未执行本轮动作。') from exc
            except httpx.TransportError as exc:
                last_error = exc
                if attempt >= effective_attempts:
                    raise VisionAgentError(f'千问视觉连接连续{attempt}次中断：{exc}') from exc
            except ValueError as exc:
                raise VisionAgentError(f"千问视觉响应不是有效 JSON：{exc}") from exc

            if self.retry_base_delay > 0:
                time.sleep(self.retry_base_delay * (2 ** (attempt - 1)))
        else:
            raise VisionAgentError(f"千问视觉连接失败：{last_error}")

        raw_usage = payload.get("usage") or {}
        self.last_usage = raw_usage if isinstance(raw_usage, dict) else {}
        normalized_usage: dict[str, int] = {}
        for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
            value = self.last_usage.get(key)
            if isinstance(value, int) and (not isinstance(value, bool)) and (value >= 0):
                normalized_usage[key] = value
        if 'total_tokens' not in normalized_usage:
            component_keys = ("prompt_tokens", "completion_tokens")
            if all((key in normalized_usage for key in component_keys)):
                normalized_usage['total_tokens'] = sum((normalized_usage[key] for key in component_keys))
        for (key, value) in normalized_usage.items():
            self.usage_totals[key] += value
        self.successful_call_count += 1
        self.last_request_id = str(payload.get("id") or "")
        self.last_response_model = str(payload.get("model") or "")
        try:
            choice = payload["choices"][0]
            self.last_finish_reason = str(choice.get("finish_reason") or "")
            content = choice["message"]["content"]
        except (AttributeError, KeyError, IndexError, TypeError) as exc:
            raise VisionAgentError("千问视觉响应缺少 message.content。") from exc
        reject_if(not isinstance(content, str) or not content.strip(), VisionAgentError("千问视觉返回了空内容。"))
        return content
