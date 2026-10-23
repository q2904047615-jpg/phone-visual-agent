"""User-configurable lucky-bag task profile.

The profile is a goal template and policy data. It does not select coordinates,
sequence device actions, or inspect screenshots locally.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


NO_LUCKY_BAG_REOBSERVE_SECONDS = 300
POST_PARTICIPATION_REOBSERVE_SECONDS = 300
COMMENT_SEND_DELAY_SECONDS = 3
REFERENCE_IMAGE_DIR = Path(__file__).resolve().parent / "reference_images"
REFERENCE_IMAGE_NAMES = (
    "00_lucky_bag_closeup.jpg",
    "01_live_room_lucky_bag.png",
    "04_participation_success.jpg",
    "05_already_participated.jpg",
    "06_not_selected.jpg",
)

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

    @property
    def visual_reference_paths(self) -> tuple[Path, ...]:
        """Stable project-owned visual examples for the Qwen observation prompt."""

        return tuple(REFERENCE_IMAGE_DIR / name for name in REFERENCE_IMAGE_NAMES)


def build_lucky_bag_goal(profile: LuckyBagProfile) -> str:
    """Build the natural-language goal sent to the existing universal Agent."""

    return "\n".join(
        (
            "【ACTIVE_EXECUTION_POLICY:lucky_bag】",
            f"这是福袋监控任务，不是关于福袋的知识问答。持续观察用户手动打开的{profile.app_alias}当前直播间，最长运行 {profile.duration_seconds} 秒。",
            "以下顺序是本次任务的活动执行约束：每轮先读取CURRENT，再依据当前画面和真实动作历史选择一个通用动作或finish；不得把后面的等待条件提前替代当前已经满足的动作。",
            "五张用户参考图只用于理解界面语义，不复制其中的苹果手机坐标：图1是用户提供的福袋入口近照；图2是直播间中的福袋入口；图3是发送后短暂出现的“成功参与福袋”提示；图4是重新打开同一个福袋后显示“已参与”；图5是开奖后的“没抽中福袋”和“知道了”。",
            "实际 Android 画面可能改变布局、比例、颜色和文字样式；只能把参考图中的语义状态与当前 Android 实时画面相互核对，不能照搬参考图坐标。",
            "福袋入口请严格按这个外观判断：一个较小的红色或粉红色布袋/礼包图标，袋口或袋身有醒目的黄色、金色绳结（像蝴蝶结或扎口），图标下方或旁边常叠着黑色圆角倒计时标签，例如“07:07”；它通常出现在直播间画面的上方区域。Android 上位置、大小、比例、亮度和文字清晰度可能变化，但红粉色袋身加金黄色扎口加倒计时标签的组合是关键特征。",
            "只要当前画面出现与上述外观一致的福袋入口，即使图标较小、部分被裁切、倒计时数字模糊或背景很暗，也必须先点击该入口打开；此时禁止选择wait_for_change，不能在可参与福袋可见时等待。普通礼物、红包、游戏礼包和推荐卡没有这种红粉色袋身加金黄色绳结的组合时才排除。",
            f"只有当前画面明确没有上述福袋视觉证据、倒计时或开奖结果时，才选择 wait_for_change，并将 wait_seconds 填为 {NO_LUCKY_BAG_REOBSERVE_SECONDS}；等待结束后用新的 Android 画面重新判断。",
            "如果参与条件要求发表评论，只有画面已预填评论时继续福袋专用固定流程；不要输入、改写或补充评论文字。去发表评论使用固定归一化坐标 [0.498, 0.868]，发送评论使用固定归一化坐标 [0.832, 0.476]，这两步不再调用Qwen或做场景绑定；点击去发表评论后固定等待 3 秒，再点击发送评论。",
            "如果要求评论但画面没有预填内容，停止手机操作并报告该情况。",
            "发送后先调用一次Qwen确认是否已经参与；效果不明确时核对同一个福袋的当前状态，不重复发送评论。",
            "确认已参与后，wait_seconds填写当前画面实际开奖倒计时的剩余秒数，不截短、不固定为300秒；倒计时不可读则保持空值，等待后再读，不重复参与。倒计时结束后用新画面判断开奖结果。",
            "开奖后只调用一次Qwen判断中奖或未中奖：reply必须明确说明结果；未中奖时decision必须同时返回当前画面中“知道了”按钮的点击坐标，点击后继续观察当前直播间是否出现新福袋；中奖时不点击，记录结果并继续监控。",
            f"明确中奖时沿用通知：标题“{profile.subject}”，正文“{profile.body}”，收件人 {profile.recipient}；通知不结束监控。未知结果不能当作中奖或未中奖。",
            "直播间下播或用户换房期间不要自行切换房间；等待用户手动打开下一个直播间。",
            "福袋入口坐标必须来自当前 Android 实时画面；仅去发表评论和发送评论使用本任务批准的两个固定归一化坐标。",
        )
    )

