"""User-approved Home -> fresh launcher -> recents navigation exception.

This is not task-intent detection or a task-completion checklist. Only a
selected canonical open_recent_apps starts it; every returned action still
passes the existing current-observation binder and one-shot execution scope.
"""
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

RECENT_NAVIGATION_PROTOCOL = '2026-09-07-recent-navigation-v1'
LOCAL_NAVIGATION_SOURCE = 'local_recent_navigation'


@dataclass
class RecentAppsNavigation:
    pending: bool = False

    def select(self, model_decision: Mapping[str, Any], *, foreground_app_id: str,
        available_action_kinds: frozenset[str]) -> dict[str, Any] | None:
        requested = (model_decision.get('status') == 'action'
            and model_decision.get('action') == 'open_recent_apps')
        if not self.pending and not requested:
            return None
        if 'open_recent_apps' not in available_action_kinds:
            raise ValueError('当前设备/观察没有打开后台能力。')
        kind = 'open_recent_apps' if foreground_app_id == 'launcher' else 'home'
        if kind not in available_action_kinds:
            raise ValueError('打开后台需要先回主屏幕，但当前设备/观察没有 Home 能力。')
        self.pending = True
        return {'status': 'action', 'action': kind,
            'previous_action_outcome': model_decision.get('previous_action_outcome'),
            'reason': ('本地后台导航：当前新图确认 Launcher，打开后台。' if kind == 'open_recent_apps'
                else '本地后台导航：当前新图未确认 Launcher，先回主屏幕再重新观察。')}

    def record_execution(self, kind: str) -> None:
        # A staged action, transport exception, pause or stale scope is not execution.
        if kind == 'open_recent_apps':
            self.pending = False

    def to_dict(self) -> dict[str, Any]:
        return {'protocol_version': RECENT_NAVIGATION_PROTOCOL, 'pending': self.pending}
