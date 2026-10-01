"""Additive lucky-bag feature variants.

The ordinary LuckyBagMonitor keeps the universal Qwen-session integration from
phone-visual-agent.  DedicatedLuckyBagMonitor is selected only by the trial
variant and runs that variant's local detector and transport.
"""
from .monitor import LuckyBagMonitor, LuckyBagMonitorRecord, LuckyBagSessionGateway
from .dedicated_monitor import DedicatedLuckyBagMonitor
from .profile import LuckyBagProfile, build_lucky_bag_goal

__all__ = [
    "LuckyBagMonitor",
    "LuckyBagMonitorRecord",
    "LuckyBagSessionGateway",
    "DedicatedLuckyBagMonitor",
    "LuckyBagProfile",
    "build_lucky_bag_goal",
]
