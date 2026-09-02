from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from agent.domain import (
    AgentEvidenceStorePort,
    CanonicalSelectionReceipt,
    ConfirmationAuthority,
    EffectConfirmationAuthority,
)
from agent.domain.canonical_action_kinds import CANONICAL_ACTION_KINDS
from agent.domain.task_graph import DynamicTaskGraph
from agent.domain.generic_goal import GenericIntentDraft
from agent.application.vision_usage import VisionSessionUsageLedger


POST_ACTION_TRANSITION_PROTOCOL_VERSION = '2026-08-16-universal-post-action-transition-v1'


@dataclass
class UniversalAgentSessionState:
    session_id: str
    raw_goal: str
    device_id: str
    run_dir: Path
    adapter: Any = field(repr=False)
    evidence_store: AgentEvidenceStorePort = field(repr=False)
    vision_usage: VisionSessionUsageLedger | None = field(default=None, repr=False)
    task_graph: DynamicTaskGraph | None = None
    goal_draft: GenericIntentDraft | None = None
    trusted_observation: Any = None
    trusted_frames: tuple[Any, ...] = field(default_factory=tuple, repr=False)
    qwen_decision: Any = None
    controller_decision: CanonicalSelectionReceipt | None = None
    confirmation_authority: ConfirmationAuthority | None = field(default=None, repr=False)
    effect_confirmation_authority: EffectConfirmationAuthority | None = field(default=None, repr=False)
    confirmed_effect_ids: tuple[str, ...] = ()
    status: str = "created"
    step_number: int = 1
    physical_actions: int = 0
    local_exact_input_authority: bool = field(default=False, repr=False)
    automatic_loop_enabled: bool = False
    auto_pause_reason: str = ""
    history: list[dict[str, Any]] = field(default_factory=list)
    evidence_paths: list[str] = field(default_factory=list)
    last_post_action_transition: dict[str, Any] | None = None
    input_focus_retry_key: str = field(default='', repr=False)
    input_focus_retry_state_digest: str = field(default='', repr=False)
    effect_previews: tuple[dict[str, Any], ...] = ()
    failed_reason: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().astimezone().isoformat(timespec='seconds'))

    @staticmethod
    def _serialize(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, Mapping):
            return dict(value)
        method = getattr(value, "to_dict", None)
        if callable(method):
            return method()
        return value

    @staticmethod
    def _active_scope(authority: Any) -> dict[str, Any] | None:
        return authority.scope() if authority is not None and not authority.consumed else None

    def snapshot(self) -> dict[str, Any]:
        decision = self._serialize(self.qwen_decision)
        graph = self._serialize(self.task_graph)
        observation = self._serialize(self.trusted_observation)
        proposal = self._serialize(getattr(self.qwen_decision, 'proposal',
            None)) if self.qwen_decision is not None else None
        controller = self._serialize(self.controller_decision)
        scene = self._serialize(getattr(self.trusted_observation, 'scene',
            None)) if self.trusted_observation is not None else None
        return {'session_id': self.session_id, 'raw_goal': self.raw_goal, 'device_id': self.device_id,
            'created_at': self.created_at, 'status': self.status, 'step_number': self.step_number,
            'physical_actions': self.physical_actions, 'qwen_usage': self._serialize(self.vision_usage),
            'local_exact_input_authority': self.local_exact_input_authority,
            'failed_reason': self.failed_reason, 'confirm_stage': '', 'task_graph': graph,
            'goal': self._serialize(self.goal_draft), 'trusted_observation': observation, 'current_scene': scene,
            'qwen_decision': decision, 'proposal': proposal, 'controller_decision': controller,
            'history': list(self.history), 'evidence': list(dict.fromkeys(self.evidence_paths)),
            'automatic_loop_enabled': self.automatic_loop_enabled, 'auto_pause_reason': self.auto_pause_reason,
            'corrective_retry_protocol': None, 'corrective_retry_history': [],
            'post_action_transition_protocol': POST_ACTION_TRANSITION_PROTOCOL_VERSION,
            'last_post_action_transition': self._serialize(self.last_post_action_transition),
            'last_confirmation_failure': None,
            'available_action_kinds': sorted(self.adapter.supported_action_kinds() if callable(getattr(self.adapter,
            'supported_action_kinds', None)) else CANONICAL_ACTION_KINDS),
            'confirmation_scope': self._active_scope(self.confirmation_authority),
            'confirmation_ready': bool(self.status == 'awaiting_confirmation' and self.controller_decision is not None
            and self.controller_decision.allowed and (self.confirmation_authority is not None)
            and (not self.confirmation_authority.consumed)),
            'effect_confirmation_scope': self._active_scope(self.effect_confirmation_authority),
            'effect_confirmation_ready': bool(self.status == 'awaiting_effect_confirmation'
            and self.effect_confirmation_authority is not None and (not self.effect_confirmation_authority.consumed)),
            'capability_gap': None,
            # Retain the public response key while the retired lineage authority no longer exists.
            'verified_app_surface_lineage': None,
            'effect_previews': [dict(item) for item in self.effect_previews],
            'effect_verification': None,
            'device_capability': self.adapter.capability_snapshot().to_dict() if callable(getattr(self.adapter,
            'capability_snapshot', None)) else None, 'effect_confirmation_preview': dict(
            self.effect_confirmation_authority.intent_preview) if self.effect_confirmation_authority is not None
            and (not self.effect_confirmation_authority.consumed) else None,
            'confirmed_effect_ids': list(self.confirmed_effect_ids)}
