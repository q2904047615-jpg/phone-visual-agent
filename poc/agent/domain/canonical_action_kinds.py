"""Single canonical vocabulary for executable universal-agent actions."""

from __future__ import annotations


CANONICAL_ACTION_KINDS = frozenset({'tap_semantic', 'dismiss_overlay', 'scroll', 'swipe_element', 'back', 'home', 'open_recent_apps',
    'reveal_system_navigation', 'input_verified_text', 'press_enter', 'clear_verified_text', 'double_tap', 'long_press',
    'drag', 'launch_app', 'wait_for_change'})

_IDEMPOTENT_SYSTEM_SURFACES = {"home": "launcher", "open_recent_apps": "recent_tasks"}


def expected_idempotent_system_surface_kind(action_kind: str) -> str | None:
    return _IDEMPOTENT_SYSTEM_SURFACES.get(str(action_kind or "").strip())
