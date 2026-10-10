"""Lucky-bag monitoring feature variants and the bounded execution flow.

The ordinary LuckyBagMonitor keeps the universal Qwen-session integration from
phone-visual-agent.  DedicatedLuckyBagMonitor is selected only by the trial
variant and runs that variant's local detector and transport.
"""
from .monitor import LuckyBagMonitor, LuckyBagMonitorRecord, LuckyBagSessionGateway
from .fast_runner import COMMENT_POINT, SEND_POINT, LUCKY_BAG_SLOT_POINTS, FastLuckyBagSessionGateway
from .dedicated_monitor import DedicatedLuckyBagMonitor
from .profile import LuckyBagProfile, build_lucky_bag_goal

__all__ = [
    "LuckyBagMonitor",
    "LuckyBagMonitorRecord",
    "LuckyBagSessionGateway",
    "FastLuckyBagSessionGateway",
    "COMMENT_POINT",
    "SEND_POINT",
    "LUCKY_BAG_SLOT_POINTS",
    "DedicatedLuckyBagMonitor",
    "LuckyBagProfile",
    "build_lucky_bag_goal",
]
