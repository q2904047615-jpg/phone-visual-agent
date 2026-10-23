"""Configuration for the manual Soda Music advertisement page test."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QishuiAdTestProfile:
    """One user-started test run; it never contains account or payout data."""

    device_id: str
    duration_seconds: int = 180
    app_alias: str = "汽水音乐"

    def __post_init__(self) -> None:
        if not self.device_id.strip():
            raise ValueError("device_id 不能为空。")
        if self.duration_seconds < 1:
            raise ValueError("duration_seconds 必须为正整数。")
        if not self.app_alias.strip():
            raise ValueError("app_alias 不能为空。")


def build_qishui_ad_test_goal(profile: QishuiAdTestProfile) -> str:
    """Describe a single manual test without adding a protocol or action kind."""

    return "\n".join(
        (
            f"这是一次汽水音乐广告页面测试，最长观察 {profile.duration_seconds} 秒，不是收益任务。",
            "用户负责手动打开一次广告；本会话最多测试一个广告，不得循环、重复或 farming。",
            "只根据当前手机画面检查汽水音乐是否在前台、广告是否加载、倒计时是否出现并完成、关闭或返回控件是否出现，以及返回后的奖励页面是否出现。",
            "不得点击广告内的安装、下载、跳转、领取、提现、金币或任何奖励控件；不得输入账号、Cookie、支付或个人资料。",
            "如果页面仍在加载或倒计时尚未完成，选择 wait_for_change；如果画面不唯一、倒计时异常、广告未加载或状态不确定，立即以当前事实结束并说明原因。",
            "到达奖励/领取页面或确认测试无法继续时停止推进；不要自行领取奖励。若选择普通 back，只把它作为待人工确认的一次返回动作。",
            "所有结论只依据当前实时画面和本会话实际观察历史，不使用固定坐标、旧截图或预设页面步骤。",
        )
    )
