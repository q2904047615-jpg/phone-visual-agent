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
        return "execution_class_mismatch"
    if "active 子目标" in message or "活动子目标" in message:
        return "active_frontier"
    if "JSON" in message or "json" in message:
        return "invalid_json"
    return "task_graph_validation"


def _structured_candidate_diff(raw: str, previous_graph: Any) -> dict[str, Any] | None:
    """Compare only whitelisted high-level graph fields from valid JSON."""

    if previous_graph is None or not hasattr(previous_graph, "to_dict"):
        return None
    try:
        candidate = json.loads(str(raw or "").strip())
        previous = previous_graph.to_dict()
    except (AttributeError, TypeError, ValueError):
        return None
    if not isinstance(candidate, dict) or not isinstance(previous, dict):
        return None

    def project(payload: dict[str, Any]) -> dict[str, Any] | None:
        goal = payload.get("goal")
        if not isinstance(goal, dict):
            return None
        target_apps = goal.get("target_apps")
        safe_apps = None
        if isinstance(target_apps, list):
            safe_apps = [
                {
                    key: item.get(key)
                    for key in ("app_id", "app_name")
                    if isinstance(item, dict) and key in item
                }
                for item in target_apps
                if isinstance(item, dict)
            ]
        entities = goal.get("entities")
        input_fields = (
            entities.get("input_fields") if isinstance(entities, dict) else None
        )
        safe_fields = None
        if isinstance(input_fields, list):
            safe_fields = [
                {
                    key: item.get(key)
                    for key in ("field_id", "field_label", "text")
                    if isinstance(item, dict) and key in item
                }
                for item in input_fields
                if isinstance(item, dict)
            ]
        safe_subgoals = None
        if isinstance(payload.get("subgoals"), list):
            allowed = (
                "subgoal_id",
                "objective",
                "status",
                "depends_on",
                "constraints",
                "completion_conditions",
                "effect_ids",
                "execution_class",
            )
            safe_subgoals = [
                {key: item.get(key) for key in allowed if key in item}
                for item in payload["subgoals"]
                if isinstance(item, dict)
            ]
        return {
            "goal_objective": goal.get("objective"),
            "target_apps": safe_apps,
            "input_fields": safe_fields,
            "subgoals": safe_subgoals,
        }

    before = project(previous)
    after = project(candidate)
    if before is None or after is None:
        return None
    result = {
        "changed_fields": [key for key in before if before[key] != after[key]],
        "previous": before,
        "candidate": after,
    }
    encoded = json.dumps(result, ensure_ascii=False, sort_keys=True)
    return json.loads(_redact_deepseek_failure_response(encoded))


def persist_deepseek_failure_diagnostic(
    planner: Any,
    *,
    evidence_dir: Path | None,
    prefix: str,
    failed_stage: str,
    error: Exception,
    previous_graph: Any = None,
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
    structured_diff = _structured_candidate_diff(raw, previous_graph)
    if structured_diff is not None:
        payload["structured_candidate_diff"] = structured_diff
    authority = getattr(planner, "last_semantic_authority", None)
    if authority is not None:
        try:
            authority_json = json.dumps(
                authority.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
            )
            payload["typed_effect_authority"] = json.loads(
                _redact_deepseek_failure_response(authority_json)
            )
        except (AttributeError, TypeError, ValueError):
            payload["typed_effect_authority_error"] = (
                "正式类型化效果报告无法安全序列化。"
            )
    authority_error = str(
        getattr(planner, "last_semantic_authority_error", "") or ""
    ).strip()
    if authority_error:
        payload["typed_effect_authority_error"] = (
            _redact_deepseek_failure_response(authority_error)[:1000]
        )
    encoded = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    temporary = output_dir / f".{target.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_bytes(encoded)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return (str(target),)
