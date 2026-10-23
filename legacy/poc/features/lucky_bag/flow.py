"""Dedicated lucky-bag policy. Only this feature uses this state machine."""
from dataclasses import dataclass


@dataclass
class Page:
    bag: tuple[int, int] | None = None
    comment_button: tuple[int, int] | None = None
    send_button: tuple[int, int] | None = None
    dismiss_button: tuple[int, int] | None = None
    detail: bool = False
    joined: bool = False
    lost: bool = False
    closed: bool = False
    live: bool = False
    prefilled: bool = False
    countdown: int | None = None
    summary: str = ""


@dataclass
class Step:
    phase: str
    message: str
    point: tuple[int, int] | None = None
    wait: float = 2
    notify: bool = False
    pause: bool = False


@dataclass
class Flow:
    phase: str = "search"
    draw_at: float | None = None
    joined: bool = False
    send_attempted: bool = False
    clicked_comment: bool = False
    dismiss_pending: bool = False
    halted: bool = False

    def decide(self, page: Page, now: float) -> Step:
        if self.halted:
            return Step("halted", "已停止手机操作。", pause=True)
        if self.dismiss_pending and not page.lost:
            self.draw_at = None
            self.joined = self.send_attempted = self.clicked_comment = self.dismiss_pending = False
        # Draw outcome takes priority over new icons, room changes and other UI.
        if self.joined and self.draw_at is not None and now >= self.draw_at:
            if not page.lost:
                self.halted = True
                return Step("suspected_win", "开奖时未看到明确没抽中，已停手。", notify=True)
            if page.dismiss_button:
                self.dismiss_pending = True
                return Step("dismiss_loss", "明确没抽中，关闭结果。", page.dismiss_button)
            return Step("result", "看到没抽中，但尚未定位到知道了按钮。")
        if page.lost:
            if page.dismiss_button:
                self.dismiss_pending = True
                return Step("dismiss_loss", "关闭明确的没抽中结果。", page.dismiss_button)
            return Step("result", "结果已识别，正在寻找知道了。")
        if page.closed:
            return Step("waiting_room", "直播结束，等待你手动打开下一间直播间。", wait=60)
        if page.joined:
            self.joined = True
            if page.countdown is not None:
                self.draw_at = now + page.countdown
            if self.draw_at is None:
                return Step("joined", "已参与，正在读取开奖倒计时。")
            return Step("joined", "已确认参与，等待开奖。", wait=max(0, min(300, self.draw_at - now)))
        if self.joined:
            if self.draw_at is None:
                if page.countdown is not None:
                    self.draw_at = now + page.countdown
                elif page.bag:
                    return Step("read_countdown", "重新打开同一福袋读取倒计时。", page.bag)
                else:
                    return Step("joined", "参与已确认，暂未读到倒计时。")
            return Step("joined", "保持当前房间等待开奖。", wait=max(0, min(300, self.draw_at - now)))
        if page.detail and page.countdown is not None:
            self.draw_at = now + page.countdown
        if self.send_attempted:
            if page.bag and not page.detail and not page.send_button:
                return Step("verify_join", "核对同一个福袋是否已参与，不重复发送。", page.bag)
            return Step("verify_join", "发送已尝试，参与结果尚未确认；等待新画面核对。")
        if self.clicked_comment and page.send_button:
            if not page.prefilled:
                return Step("empty_comment", "评论没有可确认的预填内容，已暂停。", pause=True)
            # This state must be durably written before dispatching send.
            self.send_attempted = True
            return Step("verify_join", "发送预填评论一次，随后确认参与结果。", page.send_button)
        if page.comment_button:
            self.clicked_comment = True
            return Step("open_comment", "打开自动预填评论。", page.comment_button)
        if page.bag and not page.detail and not page.send_button:
            return Step("open_bag", "发现福袋，打开详情。", page.bag)
        if page.detail or page.send_button or self.clicked_comment:
            return Step("inspect_conditions", "当前参与界面尚未完整识别，保留画面继续核对。")
        return Step("search", "当前没有识别到福袋，每分钟检查一次。", wait=60)
