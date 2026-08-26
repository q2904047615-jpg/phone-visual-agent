"""Single canonical vocabulary for executable universal-agent actions."""

from __future__ import annotations


CANONICAL_ACTION_KINDS = frozenset(
    {
        "tap_semantic",
        "dismiss_overlay",
        "swipe",
        "back",
        "home",
        "open_recent_apps",
        "reveal_system_navigation",
        "input_verified_text",
        "press_enter",
        "clear_verified_text",
        "double_tap",
        "long_press",
        "drag",
        "wait_for_change",
    }
)
