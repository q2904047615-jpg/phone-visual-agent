from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import uuid
from typing import Any


DEEPSEEK_FAILURE_DIAGNOSTIC_VERSION = (
    "2026-08-17-deepseek-failure-diagnostic-v1"
)
MAX_REDACTED_DEEPSEEK_RESPONSE_CHARS = 16000
_IMAGE_DATA_URL_RE = re.compile(
    r"data:image/[^;\s\"']+;base64,[A-Za-z0-9+/=_-]+",
    re.IGNORECASE,
)
_SECRET_FIELD_RE = re.compile(
    r"(?P<prefix>[\"']?(?:authorization|api[_-]?key|access[_-]?token|"
    r"refresh[_-]?token|token|secret|password)[\"']?\s*[:=]\s*)"
    r"(?P<quote>[\"'])(?P<value>.*?)(?P=quote)",
    re.IGNORECASE,
)
_UNQUOTED_SECRET_FIELD_RE = re.compile(
    r"(?P<prefix>[\"']?(?:authorization|api[_-]?key|access[_-]?token|"
    r"refresh[_-]?token|token|secret|password)[\"']?\s*[:=]\s*)"
    r"(?![\"'])(?P<value>[^,}\]\s]+)",
    re.IGNORECASE,
)
_BEARER_RE = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]+", re.IGNORECASE)
_URL_SECRET_RE = re.compile(
    r"(?P<prefix>[?&](?:api[_-]?key|access[_-]?token|token|secret|password)=)"
    r"[^&#\s\"']+",
    re.IGNORECASE,
)
_URL_USERINFO_RE = re.compile(r"(https?://)[^/@\s:]+:[^/@\s]+@", re.IGNORECASE)
_OPENAI_STYLE_SECRET_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}")


def _redact_deepseek_failure_response(raw: str) -> str:
    redacted = _IMAGE_DATA_URL_RE.sub("[REDACTED_IMAGE_DATA_URL]", str(raw or ""))
    redacted = _SECRET_FIELD_RE.sub(
        lambda match: (
            f"{match.group('prefix')}{match.group('quote')}"
            f"[REDACTED_SECRET]{match.group('quote')}"
        ),
        redacted,
    )
    redacted = _UNQUOTED_SECRET_FIELD_RE.sub(
        lambda match: f"{match.group('prefix')}[REDACTED_SECRET]",
        redacted,
    )
    redacted = _BEARER_RE.sub("Bearer [REDACTED_SECRET]", redacted)
    redacted = _URL_SECRET_RE.sub(
        lambda match: f"{match.group('prefix')}[REDACTED_SECRET]",
        redacted,
    )
    redacted = _URL_USERINFO_RE.sub(r"\1[REDACTED_CREDENTIALS]@", redacted)
    return _OPENAI_STYLE_SECRET_RE.sub("[REDACTED_SECRET]", redacted)


def _classify_deepseek_error(error: Exception) -> str:
    message = str(error)
    if "低层动作表达" in message:
        return "low_level_instruction"
    if "外部状态变化但未声明" in message:
        return "external_impact_mismatch"
    if "active 子目标" in message or "活动子目标" in message:
        return "active_frontier"
    if "JSON" in message or "json" in message:
        return "invalid_json"
    return "task_graph_validation"


def persist_deepseek_failure_diagnostic(
    planner: Any,
    *,
    evidence_dir: Path | None,
    prefix: str,
    failed_stage: str,
    error: Exception,
) -> tuple[str, ...]:
    """Persist bounded redacted planner output without changing fail-closed policy."""

    raw = str(getattr(planner, "last_raw_response", "") or "")
    if evidence_dir is None or not raw:
        return ()

    output_dir = Path(evidence_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_prefix = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(prefix or ""))[:96]
    target = output_dir / f"{safe_prefix or 'task_graph'}_deepseek_failure.json"
    redacted = _redact_deepseek_failure_response(raw)
    bounded = redacted[:MAX_REDACTED_DEEPSEEK_RESPONSE_CHARS]
    payload = {
        "artifact_version": DEEPSEEK_FAILURE_DIAGNOSTIC_VERSION,
        "model_role": "high_level_task_planner",
        "provider": "deepseek",
        "failed_stage": str(failed_stage or "unknown")[:120],
        "error_type": _classify_deepseek_error(error),
        "error_message": _redact_deepseek_failure_response(str(error))[:1000],
        "raw_response_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "raw_response_length": len(raw),
        "redacted_response_truncated": len(redacted) > len(bounded),
        "redacted_raw_response": bounded,
    }
    encoded = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    temporary = output_dir / f".{target.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_bytes(encoded)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return (str(target),)
