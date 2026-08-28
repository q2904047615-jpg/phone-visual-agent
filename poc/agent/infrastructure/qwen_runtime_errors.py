"""Diagnostics for failures emitted by the Qwen infrastructure path."""

from __future__ import annotations

import json
from typing import Any


FORMAT_ERROR_TYPES = frozenset({'invalid_json', 'truncated_json', 'protocol_invalid'})


def classify_qwen_error(error: BaseException | str, *, raw_response: str='') -> str:
    """Classify Qwen failures without weakening any protocol validation."""

    text = str(error).strip()
    lowered = text.casefold()
    if 'vision_model_identity_mismatch' in lowered:
        return "vision_model_identity_mismatch"
    if 'vision_step_contract_violation' in lowered:
        return "vision_step_contract_violation"
    if 'suite_timeout' in lowered or ('整套' in text and '超时' in text):
        return "suite_timeout"
    if 'case_timeout' in lowered or ('单用例' in text and '超时' in text):
        return "case_timeout"
    if (any((marker in lowered for marker in ('server disconnected', 'connection reset', 'connection aborted',
        'connection refused', 'remote protocol error', 'transporterror'))) or ('连接' in text and '中断' in text)):
        return "service_disconnect"
    if 'timeout' in lowered or '超时' in text:
        return "service_timeout"
    if 'http ' in lowered or '请求失败（http' in lowered:
        return "http_error"
    if (any((marker in lowered for marker in ('json无法解析', 'json 无法解析', 'not valid json', '不是有效 json', '不是有效json',
        '没有返回 json')))):
        return "truncated_json" if looks_like_truncated_json(raw_response) else "invalid_json"
    if any((marker in text for marker in ('不符合协议', '协议外字段', '缺少字段', '必须是JSON对象', '必须包含4个'))):
        return "protocol_invalid"
    if any((marker in text for marker in ('不稳定', '模糊', 'fingerprint', '确认门'))):
        return "local_safety_block"
    return "unknown_error"


def looks_like_truncated_json(raw_response: str) -> bool:
    text = str(raw_response or "").strip()
    if not text:
        return False
    fenced = text
    if fenced.startswith('```'):
        fenced = fenced.strip("`").strip()
        if fenced.casefold().startswith('json'):
            fenced = fenced[4:].lstrip()
    start = fenced.find("{")
    if start < 0:
        return False
    candidate = fenced[start:]
    try:
        json.loads(candidate)
        return False
    except json.JSONDecodeError as exc:
        if exc.pos >= max(0, len(candidate) - 3):
            return True
    braces = candidate.count("{") - candidate.count("}")
    brackets = candidate.count("[") - candidate.count("]")
    unescaped_quotes = 0
    escaped = False
    for char in candidate:
        if escaped:
            escaped = False
            continue
        if char == '\\':
            escaped = True
        elif char == '"':
            unescaped_quotes += 1
    return braces > 0 or brackets > 0 or unescaped_quotes % 2 == 1


def failure_diagnostics(error: BaseException | str, *, raw_response: str='', stage: str, model_calls: int,
    elapsed_seconds: float, safe_stop_reason: str) -> dict[str, Any]:
    return {'error_type': classify_qwen_error(error, raw_response=raw_response), 'error': str(error),
        'failed_stage': stage, 'model_calls': int(model_calls), 'elapsed_seconds': round(float(elapsed_seconds), 3),
        'safe_stop_reason': safe_stop_reason, 'hardware_actions_enabled': False}
