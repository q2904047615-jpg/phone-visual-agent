"""File-system persistence for redacted DeepSeek failure evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent.infrastructure.model_failure_diagnostics import (
    model_failure_payload,
    persist_model_failure_payload,
    redact_model_failure_response,
)

DEEPSEEK_FAILURE_DIAGNOSTIC_VERSION = '2026-08-17-deepseek-failure-diagnostic-v1'


def _redact_deepseek_failure_response(raw: str) -> str:
    return redact_model_failure_response(raw)


def _classify_deepseek_error(error: Exception) -> str:
    message = str(error)
    if 'JSON' in message or 'json' in message:
        return "invalid_json"
    return "task_graph_validation"


def _structured_candidate_diff(raw: str, previous_graph: Any) -> dict[str, Any] | None:
    """Compare only whitelisted high-level graph fields from valid JSON."""

    if previous_graph is None or not hasattr(previous_graph, 'to_dict'):
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
            safe_apps = [{key: item.get(key) for key in ('app_id', 'app_name') if isinstance(item,
                dict) and key in item} for item in target_apps if isinstance(item, dict)]
        entities = goal.get("entities")
        input_fields = entities.get('input_fields') if isinstance(entities, dict) else None
        safe_fields = None
        if isinstance(input_fields, list):
            safe_fields = [{key: item.get(key) for key in ('field_id', 'field_label', 'text') if isinstance(item,
                dict) and key in item} for item in input_fields if isinstance(item, dict)]
        safe_subgoals = None
        if isinstance(payload.get('subgoals'), list):
            allowed = ('subgoal_id', 'objective', 'status', 'depends_on', 'constraints', 'completion_conditions',
                'effect_ids', 'execution_class', 'input_field_id', 'input_operation')
            safe_subgoals = [{key: item.get(key) for key in allowed if key in item} for item
                in payload['subgoals'] if isinstance(item, dict)]
        return {'goal_objective': goal.get('objective'), 'target_apps': safe_apps, 'input_fields': safe_fields,
            'subgoals': safe_subgoals}

    before = project(previous)
    after = project(candidate)
    if before is None or after is None:
        return None
    result = {'changed_fields': [key for key in before if before[key] != after[key]], 'previous': before,
        'candidate': after}
    encoded = json.dumps(result, ensure_ascii=False, sort_keys=True)
    return json.loads(_redact_deepseek_failure_response(encoded))


def persist_deepseek_failure_diagnostic(planner: Any, *, evidence_dir: Path | None, prefix: str, failed_stage: str,
    error: Exception, previous_graph: Any=None) -> tuple[str, ...]:
    """Persist bounded redacted planner output without changing fail-closed policy."""

    raw = str(getattr(planner, "last_raw_response", "") or "")
    if evidence_dir is None or not raw:
        return ()

    payload = model_failure_payload(
        artifact_version=DEEPSEEK_FAILURE_DIAGNOSTIC_VERSION,
        raw=raw,
        failed_stage=failed_stage,
        error_type=_classify_deepseek_error(error),
        error_message=str(error),
        extra={'model_role': 'high_level_task_planner', 'provider': 'deepseek'},
    )
    structured_diff = _structured_candidate_diff(raw, previous_graph)
    if structured_diff is not None:
        payload["structured_candidate_diff"] = structured_diff
    target = persist_model_failure_payload(
        payload,
        evidence_dir=evidence_dir,
        prefix=prefix,
        default_prefix='task_graph',
        suffix='deepseek_failure',
    )
    return (str(target),)
