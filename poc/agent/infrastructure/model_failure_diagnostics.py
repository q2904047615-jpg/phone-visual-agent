"""Shared redaction and atomic persistence for model failure evidence."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Any, Mapping

from agent.infrastructure.atomic_files import atomic_replace_bytes, json_bytes

MAX_REDACTED_MODEL_RESPONSE_CHARS = 16000
_IMAGE_DATA_URL_RE = re.compile('data:image/[^;\\s\\"\']+;base64,[A-Za-z0-9+/=_-]+', re.IGNORECASE)
_SECRET_FIELD_RE = re.compile('(?P<prefix>[\\"\']?(?:authorization|api[_-]?key|access[_-]?token|refresh[_-]?token|token|secret|password)[\\"\']?\\s*[:=]\\s*)(?P<quote>[\\"\'])(?P<value>.*?)(?P=quote)', re.IGNORECASE)
_UNQUOTED_SECRET_FIELD_RE = re.compile('(?P<prefix>[\\"\']?(?:authorization|api[_-]?key|access[_-]?token|refresh[_-]?token|token|secret|password)[\\"\']?\\s*[:=]\\s*)(?![\\"\'])(?P<value>[^,}\\]\\s]+)', re.IGNORECASE)
_BEARER_RE = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]+", re.IGNORECASE)
_URL_SECRET_RE = re.compile(
    r"(?P<prefix>[?&](?:api[_-]?key|access[_-]?token|token|secret|password)=)"
    r"[^&#\s\"']+", re.IGNORECASE)
_URL_USERINFO_RE = re.compile(r"(https?://)[^/@\s:]+:[^/@\s]+@", re.IGNORECASE)
_OPENAI_STYLE_SECRET_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}")

def redact_model_failure_response(raw: str) -> str:
    redacted = _IMAGE_DATA_URL_RE.sub("[REDACTED_IMAGE_DATA_URL]", str(raw or ""))
    redacted = _SECRET_FIELD_RE.sub(lambda match: f'{match.group('prefix')}{match.group('quote')}[REDACTED_SECRET]{match.group('quote')}', redacted)
    redacted = _UNQUOTED_SECRET_FIELD_RE.sub(lambda match: f"{match.group('prefix')}[REDACTED_SECRET]", redacted)
    redacted = _BEARER_RE.sub("Bearer [REDACTED_SECRET]", redacted)
    redacted = _URL_SECRET_RE.sub(lambda match: f"{match.group('prefix')}[REDACTED_SECRET]", redacted)
    redacted = _URL_USERINFO_RE.sub(r"\1[REDACTED_CREDENTIALS]@", redacted)
    return _OPENAI_STYLE_SECRET_RE.sub("[REDACTED_SECRET]", redacted)

def model_failure_payload(*, artifact_version: str, raw: str, failed_stage: str, error_type: str, error_message: str, extra: Mapping[str, Any] | None=None) -> dict[str, Any]:
    redacted = redact_model_failure_response(raw)
    bounded = redacted[:MAX_REDACTED_MODEL_RESPONSE_CHARS]
    return {'artifact_version': artifact_version, **dict(extra or {}), 'failed_stage': str(failed_stage or 'unknown')[:120],
        'error_type': str(error_type or 'unknown')[:120], 'error_message': redact_model_failure_response(
        error_message)[:1000], 'raw_response_sha256': hashlib.sha256(raw.encode('utf-8')).hexdigest(),
        'raw_response_length': len(raw), 'redacted_response_truncated': len(redacted) > len(bounded),
        'redacted_raw_response': bounded}

def persist_model_failure_payload(payload: Mapping[str, Any], *, evidence_dir: Path, prefix: str, default_prefix: str, suffix: str) -> Path:
    output_dir = Path(evidence_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_prefix = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(prefix or ""))[:96]
    target = output_dir / f"{safe_prefix or default_prefix}_{suffix}.json"
    return atomic_replace_bytes(target, json_bytes(payload, trailing_newline=False))
