"""User-configurable lucky-bag task profile.

The profile is a goal template and policy data. It does not select coordinates,
sequence device actions, or inspect screenshots locally.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LuckyBagProfile:
    """Configuration for one long-running lucky-bag monitoring task."""

    device_id: str
    recipient: str
    duration_seconds: int = 24 * 60 * 60
    app_alias: str = "抖音"
    subject: str = "疑似中奖"
    body: str = "疑似中奖"

    def __post_init__(self) -> None:
        if not self.device_id.strip():
            raise ValueError("device_id 不能为空。")
        if not self.recipient.strip():
            raise ValueError("recipient 不能为空。")
        if self.duration_seconds < 1:
            raise ValueError("duration_seconds 必须为正整数。")
        if not self.app_alias.strip():
            raise ValueError("app_alias 不能为空。")
        if not self.subject or not self.body:
            raise ValueError("通知标题和正文不能为空。")


def build_lucky_bag_goal(profile: LuckyBagProfile) -> str:
    """Build the natural-language goal sent to the existing universal Agent."""

    return "\n".join(
        (
            f"持续观察用户手动打开的{profile.app_alias}当前直播间，最长运行 {profile.duration_seconds} 秒。",
            "发现直播间左上角福袋后打开，依据当前 Android 实时画面查看参与条件和倒计时。",
            "如果参与条件要求发表评论，只有画面已预填评论时点击发送；不要输入、改写或补充评论文字。",
            "如果要求评论但画面没有预填内容，停止手机操作并报告该情况。",
            "发送后先观察参与结果；效果不明确时核对同一个福袋的当前状态，不重复发送评论。",
            "已参与后留在当前直播间，按实时倒计时等待开奖，不假定固定时长。",
            "明确显示没抽中时点击知道了，再继续观察当前直播间是否出现新福袋。",
            f"开奖后只要没有明确显示没抽中，立即停止手机操作，并触发通知：标题“{profile.subject}”，正文“{profile.body}”，收件人 {profile.recipient}。",
            "直播间下播或用户换房期间不要自行切换房间；等待用户手动打开下一个直播间。",
            "所有点击位置和动作必须依据当前 Android 实时画面，不使用参考截图坐标。",
        )
    )

