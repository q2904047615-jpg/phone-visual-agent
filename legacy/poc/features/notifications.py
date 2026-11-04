"""Generic notification event records for optional product features."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class NotificationEvent:
    """A durable, provider-neutral notification request."""

    event_id: str
    recipient: str
    subject: str
    body: str
    observed_at: str
    screenshot_path: str = ""
    reason: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


class NotificationSink(Protocol):
    def publish(self, event: NotificationEvent) -> None: ...


class JsonlNotificationOutbox:
    """Append notification requests without sending external mail."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def publish(self, event: NotificationEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")


def utc_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")

