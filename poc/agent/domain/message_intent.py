"""Pure recipient-binding semantics for the active task goal."""

from __future__ import annotations

from typing import Any, Iterable, Mapping


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
