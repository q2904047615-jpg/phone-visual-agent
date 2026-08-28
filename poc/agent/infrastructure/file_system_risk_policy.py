"""File-system loader for the formal local semantic risk policy."""

from __future__ import annotations

import json
from pathlib import Path

from agent.domain.task_semantic_ir import LocalRiskPolicyConfig, TaskSemanticIRError, local_risk_policy_from_dict


def load_local_risk_policy(path: str | Path) -> LocalRiskPolicyConfig:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskSemanticIRError(f"无法读取风险策略配置：{exc}") from exc
    return local_risk_policy_from_dict(payload)
