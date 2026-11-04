"""Single domain catalogue for canonical device actions."""

from __future__ import annotations

_RAW_ACTION_CATALOG = {
    "tap_semantic": {"physical": True, "promotable": True},
    "dismiss_overlay": {"physical": True, "promotable": True},
    "scroll": {"physical": True, "promotable": True},
    "swipe_element": {"physical": True, "promotable": True},
    "back": {"physical": True, "promotable": True},
    "home": {"physical": True, "promotable": True},
    "open_recent_apps": {"physical": True, "promotable": False},
    "reveal_system_navigation": {"physical": True, "promotable": True},
    "input_verified_text": {"physical": True, "promotable": False},
    "press_enter": {"physical": True, "promotable": False},
    "clear_verified_text": {"physical": True, "promotable": False},
    "double_tap": {"physical": True, "promotable": True},
    "long_press": {"physical": True, "promotable": True},
    "drag": {"physical": True, "promotable": True},
    # Package launch is a device transport, but it is not part of the
    # mechanical physical-action set used by the adapter's orientation gate.
    "launch_app": {"physical": False, "promotable": False},
    "wait_for_change": {"physical": False, "promotable": False},
}

CANONICAL_ACTION_KINDS = frozenset(_RAW_ACTION_CATALOG)
PHYSICAL_ACTION_KINDS = frozenset(
    key for key, value in _RAW_ACTION_CATALOG.items() if value["physical"]
)
PROMOTABLE_ACTION_KINDS = frozenset(
    key for key, value in _RAW_ACTION_CATALOG.items() if value["promotable"]
)
