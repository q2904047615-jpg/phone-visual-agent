from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


class MessageIntentError(ValueError):
    pass


def _canonical_text(value: Any, field: str, *, max_length: int) -> str:
    if not isinstance(value, str) or not value or len(value) > max_length:
        raise MessageIntentError(f"{field} 必须为1～{max_length}个字符。")
    if value != value.strip():
        raise MessageIntentError(f"{field} 首尾不能包含空白。")
    if "\n" in value or "\r" in value:
        raise MessageIntentError(f"{field} 不得包含换行。")
    return value


@dataclass(frozen=True)
class CanonicalMessageIntent:
    target_apps: tuple[tuple[str, str], ...]
    recipient: str
    message_text: str

    @classmethod
    def from_goal(
        cls,
        *,
        target_apps: Iterable[Any],
        entities: Mapping[str, Any],
    ) -> "CanonicalMessageIntent":
        recipient = _canonical_text(
            entities.get("recipient"),
            "goal.entities.recipient",
            max_length=100,
        )
        message_text = _canonical_text(
            entities.get("input_text"),
            "goal.entities.input_text",
            max_length=100,
        )
        apps: list[tuple[str, str]] = []
        for raw in target_apps:
            if isinstance(raw, Mapping):
                app_id = raw.get("app_id")
                app_name = raw.get("app_name")
            else:
                app_id = getattr(raw, "app_id", None)
                app_name = getattr(raw, "app_name", None)
            if not isinstance(app_id, str) or not app_id.strip():
                raise MessageIntentError("消息目标 App 缺少 app_id。")
            if not isinstance(app_name, str) or not app_name.strip():
                raise MessageIntentError("消息目标 App 缺少 app_name。")
            pair = (app_id.strip(), app_name.strip())
            if pair not in apps:
                apps.append(pair)
        if not apps:
            raise MessageIntentError("消息目标至少需要一个目标 App。")
        return cls(tuple(apps), recipient, message_text)

    def payload(
        self,
        *,
        task_id: str,
        device_id: str,
        revision: int,
        subgoal_id: str,
        risk_ids: Iterable[str],
    ) -> dict[str, Any]:
        return {
            "protocol_version": "2026-08-18-message-intent-v1",
            "task_id": str(task_id),
            "device_id": str(device_id),
            "revision": int(revision),
            "subgoal_id": str(subgoal_id),
            "risk_ids": sorted(str(item) for item in risk_ids),
            "target_apps": [
                {"app_id": app_id, "app_name": app_name}
                for app_id, app_name in self.target_apps
            ],
            "recipient": self.recipient,
            "message_text": self.message_text,
        }

    def digest(self, **scope: Any) -> str:
        encoded = json.dumps(
            self.payload(**scope),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def preview(self) -> dict[str, Any]:
        return {
            "target_apps": [
                {"app_id": app_id, "app_name": app_name}
                for app_id, app_name in self.target_apps
            ],
            "recipient": self.recipient,
            "message_text": self.message_text,
        }


def subgoal_binds_recipient(
    recipient: str,
    *values: Any,
) -> bool:
    """Return true only when the current subgoal literally carries the recipient.

    App-opening and other prerequisite navigation must not be forced to match a
    contact that cannot yet be visible.  Once DeepSeek puts the canonical
    recipient into the active state description, local exact-text matching can
    become authoritative without asking Qwen to infer identity from prose.
    """

    expected = str(recipient or "")
    if not expected:
        return False

    def strings(value: Any) -> Iterable[str]:
        if isinstance(value, str):
            yield value
        elif isinstance(value, Mapping):
            for item in value.values():
                yield from strings(item)
        elif isinstance(value, (list, tuple, set, frozenset)):
            for item in value:
                yield from strings(item)

    return any(expected in item for value in values for item in strings(value))


def subgoal_targets_recipient_control(
    recipient: str,
    *values: Any,
) -> bool:
    """Distinguish selecting a recipient from using it as page identity."""

    if not subgoal_binds_recipient(recipient, *values):
        return False

    def flatten(value: Any) -> Iterable[str]:
        if isinstance(value, str):
            yield value
        elif isinstance(value, Mapping):
            for item in value.values():
                yield from flatten(item)
        elif isinstance(value, (list, tuple, set, frozenset)):
            for item in value:
                yield from flatten(item)

    visible = " ".join(item for value in values for item in flatten(value)).casefold()
    selection_markers = (
        "打开", "进入", "选择", "查找", "搜索", "定位", "匹配",
        "open", "enter", "select", "choose", "find", "search", "locate", "match",
    )
    non_selector_markers = (
        "输入", "草稿", "编辑", "发送", "消息正文",
        "input", "draft", "edit", "type", "send", "message body",
    )
    return any(marker in visible for marker in selection_markers) and not any(
        marker in visible for marker in non_selector_markers
    )
