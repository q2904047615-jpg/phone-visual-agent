"""Application port for the configured trusted text transport."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from agent.domain.text_transport import (
    TextTransportActionScope,
    TextTransportProfile,
    TextTransportResult,
)


@runtime_checkable
class TrustedTextTransportPort(Protocol):
    """Execute one canonical append/clear without deciding UI or visual success."""

    @property
    def profile(self) -> TextTransportProfile: ...

    def mint_action_scope(self, *, session_id: str, task_id: str, revision: int, action_id: str,
        input_field_id: str, observation_fingerprint: str, prior_text_digest: str,
        fragment_text_digest: str, expected_text_digest: str) -> TextTransportActionScope: ...

    def append_text(self, scope: TextTransportActionScope, text: str) -> TextTransportResult: ...

    def clear_text(self, scope: TextTransportActionScope) -> TextTransportResult: ...


__all__ = ["TrustedTextTransportPort"]
