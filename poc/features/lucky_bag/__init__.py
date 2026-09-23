"""Lucky-bag monitoring feature configuration.

This package only describes the user-facing task and its notification policy.
The universal Agent remains responsible for visual decisions and device actions.
"""

from .monitor import LuckyBagMonitor, LuckyBagMonitorRecord, LuckyBagSessionGateway
from .profile import LuckyBagProfile, build_lucky_bag_goal

__all__ = ["LuckyBagMonitor", "LuckyBagMonitorRecord", "LuckyBagSessionGateway", "LuckyBagProfile", "build_lucky_bag_goal"]

