"""User-configurable lucky-bag task profile.

The profile is a goal template and policy data. It does not select coordinates,
sequence device actions, or inspect screenshots locally.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


NO_LUCKY_BAG_REOBSERVE_SECONDS = 60
POST_PARTICIPATION_REOBSERVE_SECONDS = 300
REFERENCE_IMAGE_DIR = Path(__file__).resolve().parent / "reference_images"
REFERENCE_IMAGE_NAMES = (
    "01_live_room_lucky_bag.png",
    "02_lucky_bag_detail.jpg",
    "03_prefilled_comment.jpg",
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
            "六张用户参考图只用于理解界面语义，不复制其中的苹果手机坐标：图1是直播间左上角的红色或粉红色礼包袋入口，带金色绳结并常有倒计时；图2是打开后的“福袋”详情页，显示参与人数、倒计时、参与条件和“去发表评论”；图3是评论框中已经自动填好的评论和“发送”按钮；图4是发送后短暂出现的“成功参与福袋”提示；图5是重新打开同一个福袋后显示“已参与”；图6是开奖后的“没抽中福袋”和“知道了”。",
            "实际 Android 画面可能改变布局、比例、颜色和文字样式；只能把参考图中的语义状态与当前 Android 实时画面相互核对，不能照搬参考图坐标。",
            "福袋入口通常是直播画面左上区域与参考图相似的红色或粉红色小礼包袋轮廓，带金色绳结、袋口或礼物结，旁边可能有数字倒计时；Android 上位置、大小、颜色和文字清晰度都可能变化。不要因为图标较小、部分被裁切或数字难读就否定袋状入口，也不能使用固定坐标或只凭单一颜色判断。",
            "只要当前画面出现可信的袋状礼包入口，或袋状轮廓与金色绳结/袋口等特征的明显变体，就必须先点击该入口打开；此时禁止选择wait_for_change，不能在可参与福袋可见时等待。再用详情页的“福袋”标题、参与人数、参与条件或倒计时确认；普通礼物、红包、游戏礼包和推荐卡没有这种袋状入口时才排除。",
            f"只有当前画面明确没有上述福袋视觉证据、倒计时或开奖结果时，才选择 wait_for_change，并将 wait_seconds 填为 {NO_LUCKY_BAG_REOBSERVE_SECONDS}；等待结束后用新的 Android 画面重新判断。",
            "如果参与条件要求发表评论，只有画面已预填评论时点击发送；不要输入、改写或补充评论文字。",
            "如果要求评论但画面没有预填内容，停止手机操作并报告该情况。",
            "发送后先观察参与结果；效果不明确时核对同一个福袋的当前状态，不重复发送评论。",
            f"只有在当前画面明确确认已参与后，才留在当前直播间选择 wait_for_change，并将 wait_seconds 填为 {POST_PARTICIPATION_REOBSERVE_SECONDS}；若当前画面明确显示的开奖倒计时少于 {POST_PARTICIPATION_REOBSERVE_SECONDS} 秒，填写剩余秒数；等待结束后用新的 Android 画面立即判断并处理结果。",
            "明确显示没抽中时点击知道了，再继续观察当前直播间是否出现新福袋。",
            f"开奖后只要没有明确显示没抽中，立即停止手机操作，并触发通知：标题“{profile.subject}”，正文“{profile.body}”，收件人 {profile.recipient}。",
            "直播间下播或用户换房期间不要自行切换房间；等待用户手动打开下一个直播间。",
            "所有点击位置和动作必须依据当前 Android 实时画面，不使用参考截图坐标。",
        )
    )

