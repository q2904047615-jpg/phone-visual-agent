from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
import re
from typing import Any


ACTION_SCOPE_FIELDS = frozenset({'session_id', 'task_id', 'device_id', 'revision', 'step_id',
    'effect_ids', 'observation_id', 'fingerprint', 'decision_node_id', 'action_digest'})


def normalize_action_scope(value: Mapping[str, Any]) -> dict[str, Any]:
    """Sole wire parser for the current one-action authority."""
    if not isinstance(value, Mapping) or set(value) != ACTION_SCOPE_FIELDS:
        raise ValueError('动作scope字段缺失或包含额外字段。')
    effects, revision = value.get('effect_ids'), value.get('revision')
    digest = str(value.get('action_digest') or '')
    if not isinstance(effects, list) or isinstance(revision, bool) or not isinstance(revision, int) or not re.fullmatch('[0-9a-f]{64}', digest):
        raise ValueError('动作scope格式无效。')
    result = {key: str(value.get(key) or '') for key in ACTION_SCOPE_FIELDS - {'revision', 'effect_ids'}}
    result.update(revision=revision, effect_ids=sorted(str(item) for item in effects))
    if not all(result[key] for key in ('observation_id', 'fingerprint', 'decision_node_id')):
        raise ValueError('动作scope缺少观察或决策标识。')
    return result


@dataclass
class ConfirmationAuthority:
    session_id: str
    task_id: str
    device_id: str
    revision: int
    step_id: str
    effect_ids: tuple[str, ...]
    observation_id: str
    fingerprint: str
    decision_node_id: str
    action_digest: str
    consumed: bool = False
    invalid_reason: str = ""

    def scope(self) -> dict[str, Any]:
        return {'session_id': self.session_id, 'task_id': self.task_id, 'device_id': self.device_id,
            'revision': self.revision, 'step_id': self.step_id, 'effect_ids': sorted(self.effect_ids),
            'observation_id': self.observation_id, 'fingerprint': self.fingerprint,
            'decision_node_id': self.decision_node_id, 'action_digest': self.action_digest}


@dataclass
class EffectConfirmationAuthority:
    session_id: str
    task_id: str
    device_id: str
    revision: int
    step_id: str
    effect_ids: tuple[str, ...]
    intent_digest: str
    intent_preview: dict[str, Any]
    consumed: bool = False
    invalid_reason: str = ""

    def scope(self) -> dict[str, Any]:
        return {'session_id': self.session_id, 'task_id': self.task_id, 'device_id': self.device_id,
            'revision': self.revision, 'step_id': self.step_id, 'effect_ids': sorted(self.effect_ids),
            'intent_digest': self.intent_digest}
