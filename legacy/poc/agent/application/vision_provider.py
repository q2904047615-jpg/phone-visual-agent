"""Application-facing contract for visual-model providers."""

from __future__ import annotations

from typing import Any, Protocol


class VisionProviderPort(Protocol):
    """Public provider boundary consumed by visual observers."""

    def chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int | None,
        *,
        timeout: float | None = None,
        max_attempts: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str: ...

    def status(self) -> dict[str, Any]: ...
