from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any


STRUCTURED_OPERATIONS = {
    "wechat.send_text",
    "wechat.send_album_image",
    "douyin.search",
    "douyin.batch_interact",
}

MAX_TEXT_LENGTH = 4000
MAX_CHAT_NAME_LENGTH = 40
MAX_DOUYIN_TARGETS = 10
MAX_ALBUM_INDEX = 20
MAX_FULL_RETYPES = 2

_SUPPORTED_TEXT_RE = re.compile(
    r"^[\u3400-\u4dbf\u4e00-\u9fffA-Za-z0-9 \n"
    r"，。！？、；：,.!?;:'\"（）()《》【】\[\]<>“”‘’…—\-+_@#%&/\\=]+$"
)
_CHINESE_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def normalize_user_text(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name}必须是文字。")
    text = unicodedata.normalize("NFC", value)
    if not text:
        raise ValueError(f"{field_name}不能为空。")
    if len(text) > MAX_TEXT_LENGTH:
        raise ValueError(f"{field_name}不能超过{MAX_TEXT_LENGTH}个字符。")
    if "\r" in text:
        raise ValueError("输入文字不允许回车控制符；请使用换行字符。")
    if not _SUPPORTED_TEXT_RE.fullmatch(text):
        raise ValueError(
            f"{field_name}含暂不支持的表情、生僻符号或控制字符。"
        )
    return text


def normalize_chat_name(value: Any) -> str:
    text = normalize_user_text(value, field_name="聊天名称")
    if len(text) > MAX_CHAT_NAME_LENGTH:
        raise ValueError(f"聊天名称不能超过{MAX_CHAT_NAME_LENGTH}个字符。")
    return text


def editable_character_count(text: str) -> int:
    """Count visible editable units used for deterministic backspace taps."""
    normalized = unicodedata.normalize("NFC", text)
    # Pinyin IMEs may draw spaces/apostrophes between syllables although only
    # the letters are editable key presses.  Keep this identical to the live
    # decision validator so the controller never disagrees about backspaces.
    if re.fullmatch(r"[A-Za-z\s'’]+", normalized):
        return len(re.sub(r"[\s'’]", "", normalized))
    return sum(
        1
        for char in normalized
        if not unicodedata.combining(char)
        and char not in {"\ufe0e", "\ufe0f", "\u200d"}
    )


def split_input_segments(text: str) -> list[str]:
    """Split text into keyboard-safe chunks while preserving exact order.

    Chinese is capped at four characters per candidate selection. ASCII runs
    are capped at twenty keys. Punctuation is isolated so the controller can
    require a freshly observed symbol keyboard before each symbol action.
    """

    text = normalize_user_text(text, field_name="输入文字")
    segments: list[str] = []
    current = ""
    current_kind: str | None = None

    def flush() -> None:
        nonlocal current, current_kind
        if current:
            segments.append(current)
        current = ""
        current_kind = None

    for char in text:
        if _CHINESE_RE.fullmatch(char):
            kind, limit = "chinese", 4
        elif char.isascii() and (char.isalnum() or char == " "):
            kind, limit = "ascii", 20
        else:
            flush()
            segments.append(char)
            continue
        if current_kind != kind or len(current) >= limit:
            flush()
        current_kind = kind
        current += char
    flush()
    return segments


@dataclass
class InputAttemptState:
    """Controller-owned full-input recovery state used by unit/live runners."""

    target_text: str
    segments: list[str]
    max_retypes: int = MAX_FULL_RETYPES
    completed_segments: list[str] = field(default_factory=list)
    retypes_used: int = 0
    awaiting_empty_confirmation: bool = False

    @property
    def expected_prefix(self) -> str:
        return "".join(self.completed_segments)

    def accept_segment(self, segment: str, observed_text: str) -> bool:
        expected = self.expected_prefix + segment
        if observed_text != expected:
            return False
        self.completed_segments.append(segment)
        return True

    def begin_full_retype(self, observed_text: str) -> int:
        if self.retypes_used >= self.max_retypes:
            raise ValueError("完整重输已达到2次上限。")
        count = editable_character_count(observed_text)
        if count < 1:
            raise ValueError("无法确认输入框内实际字符数，拒绝猜测退格次数。")
        self.retypes_used += 1
        self.awaiting_empty_confirmation = True
        return count

    def confirm_empty(self, is_empty: bool) -> None:
        if not self.awaiting_empty_confirmation:
            raise ValueError("当前不在清空复核阶段。")
        if not is_empty:
            raise ValueError("输入框尚未确认完全为空。")
        self.completed_segments.clear()
        self.awaiting_empty_confirmation = False


class InputRecoveryCoordinator:
    """Map frozen plan steps to controller-owned full-retype sessions."""

    def __init__(self, sessions: list[dict[str, Any]], plan: list[dict[str, Any]]) -> None:
        self.states: dict[str, InputAttemptState] = {}
        self.step_to_session: dict[str, str] = {}
        self.start_cursor: dict[str, int] = {}
        plan_indexes = {str(step.get("id")): index for index, step in enumerate(plan)}
        for index, raw in enumerate(sessions):
            if not isinstance(raw, dict):
                raise ValueError("input_sessions 每一项必须是对象。")
            session_id = f"input_{index + 1}"
            target_text = normalize_user_text(
                raw.get("target_text"), field_name="输入会话目标"
            )
            segments = raw.get("segments")
            step_ids = raw.get("step_ids")
            if (
                not isinstance(segments, list)
                or not segments
                or not all(isinstance(value, str) and value for value in segments)
                or "".join(segments) != target_text
                or not isinstance(step_ids, list)
                or len(step_ids) != len(segments)
                or not all(step_id in plan_indexes for step_id in step_ids)
            ):
                raise ValueError("input_sessions 与冻结计划的文字分段不一致。")
            max_retypes = raw.get("max_full_retypes", MAX_FULL_RETYPES)
            if max_retypes != MAX_FULL_RETYPES:
                raise ValueError("每个输入会话必须固定最多完整重输2次。")
            state = InputAttemptState(
                target_text=target_text,
                segments=list(segments),
                max_retypes=max_retypes,
            )
            self.states[session_id] = state
            self.start_cursor[session_id] = plan_indexes[step_ids[0]]
            for step_id in step_ids:
                if step_id in self.step_to_session:
                    raise ValueError("同一冻结步骤不能属于多个输入会话。")
                self.step_to_session[step_id] = session_id

    def state_for_step(self, step_id: str) -> InputAttemptState | None:
        session_id = self.step_to_session.get(step_id)
        return self.states.get(session_id) if session_id else None

    def begin_recovery(self, step_id: str, observed_text: str) -> tuple[int, int]:
        session_id = self.step_to_session.get(step_id)
        if not session_id:
            raise ValueError("当前步骤不属于可恢复的输入会话。")
        count = self.states[session_id].begin_full_retype(observed_text)
        return count, self.start_cursor[session_id]

    def confirm_empty(self, step_id: str) -> int:
        session_id = self.step_to_session.get(step_id)
        if not session_id:
            raise ValueError("当前步骤不属于可恢复的输入会话。")
        self.states[session_id].confirm_empty(True)
        return self.start_cursor[session_id]

    def metrics(self) -> dict[str, Any]:
        """Return non-secret recovery progress for reports and the local UI."""

        sessions = []
        for session_id, state in self.states.items():
            sessions.append(
                {
                    "session_id": session_id,
                    "target_length": editable_character_count(state.target_text),
                    "segment_count": len(state.segments),
                    "completed_segments": len(state.completed_segments),
                    "retypes_used": state.retypes_used,
                    "max_retypes": state.max_retypes,
                    "awaiting_empty_confirmation": state.awaiting_empty_confirmation,
                }
            )
        return {
            "total_retypes": sum(item["retypes_used"] for item in sessions),
            "sessions": sessions,
        }


def _unique_allowed_texts(values: list[str]) -> tuple[list[str], dict[str, int]]:
    result: list[str] = []
    indexes: dict[str, int] = {}
    for value in values:
        if value not in indexes:
            indexes[value] = len(result)
            result.append(value)
    return result, indexes


def _step(
    index: int,
    intent: str,
    label: str,
    app_id: str,
    checkpoint: str,
    *,
    target: str | None = None,
    text_ref: int | None = None,
    count: int = 1,
    expected_input: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "id": f"step_{index}",
        "intent": intent,
        "label": label,
        "app_id": app_id,
        "target": target,
        "text_ref": text_ref,
        "count": count,
        "checkpoint": checkpoint,
    }
    if expected_input is not None:
        item["expected_input"] = expected_input
    item.update(extra)
    return item


def _text_steps(
    *,
    start_index: int,
    app_id: str,
    field_target: str,
    text: str,
    text_indexes: dict[str, int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    segments = split_input_segments(text)
    steps: list[dict[str, Any]] = []
    prefix = ""
    for offset, segment in enumerate(segments):
        prefix += segment
        step_id = f"step_{start_index + offset}"
        step = _step(
            start_index + offset,
            "enter_text",
            f"输入第{offset + 1}段：{segment}",
            app_id,
            f"只检查{field_target}，逐字显示：{prefix}",
            target=field_target,
            text_ref=text_indexes[segment],
            expected_input=prefix,
        )
        steps.append(step)
    session = {
        "field_target": field_target,
        "target_text": text,
        "segments": segments,
        "step_ids": [step["id"] for step in steps],
        "start_step_id": steps[0]["id"],
        "max_full_retypes": MAX_FULL_RETYPES,
        "require_initial_empty": True,
    }
    return steps, session


def build_operation_agent_params(
    operation: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Convert a public structured operation into a frozen vision plan."""

    if operation not in STRUCTURED_OPERATIONS:
        raise ValueError(f"不支持的结构化操作：{operation}")

    if operation == "wechat.send_text":
        chat_name = normalize_chat_name(params.get("chat_name"))
        text = normalize_user_text(params.get("text"), field_name="消息正文")
        chat_segments = split_input_segments(chat_name)
        message_segments = split_input_segments(text)
        allowed, refs = _unique_allowed_texts(chat_segments + message_segments)
        plan = [
            _step(1, "open_app", "打开微信", "wechat", "微信主界面清晰可见", target="微信"),
            _step(2, "open_target", "打开微信搜索", "wechat", "微信搜索框清晰可见", target="微信搜索"),
            _step(3, "focus_input", "聚焦聊天搜索框", "wechat", "搜索框为空且键盘完整清晰", target="微信搜索框"),
        ]
        search_steps, search_session = _text_steps(
            start_index=4,
            app_id="wechat",
            field_target="微信搜索框",
            text=chat_name,
            text_indexes=refs,
        )
        plan.extend(search_steps)
        plan.append(_step(len(plan) + 1, "submit", "提交聊天搜索", "wechat", "出现聊天搜索结果", target="搜索"))
        plan.append(_step(len(plan) + 1, "open_target", f"打开唯一匹配聊天：{chat_name}", "wechat", f"顶部标题逐字显示{chat_name}", target=chat_name))
        plan.append(_step(len(plan) + 1, "focus_input", "聚焦聊天输入框", "wechat", "底部输入框为空且键盘完整清晰", target="聊天输入框"))
        message_steps, message_session = _text_steps(
            start_index=len(plan) + 1,
            app_id="wechat",
            field_target="聊天页底部输入框",
            text=text,
            text_indexes=refs,
        )
        plan.extend(message_steps)
        plan.append(_step(len(plan) + 1, "submit", "发送消息", "wechat", "输入框清空并出现新的本人消息", target="发送按钮"))
        plan.append(_step(len(plan) + 1, "verify_result", "复核发送结果", "wechat", f"标题仍为{chat_name}且新消息逐字等于{text}", target=chat_name))
        goal = (
            f"打开微信，搜索聊天名称{chat_name}，只接受唯一完全匹配结果；"
            f"进入后发送文字{text}。任何输入不一致都必须清空本次输入并从头重打。"
        )
        sessions = [search_session, message_session]
    elif operation == "wechat.send_album_image":
        chat_name = normalize_chat_name(params.get("chat_name"))
        image_index = params.get("image_index")
        if isinstance(image_index, bool) or not isinstance(image_index, int) or not 1 <= image_index <= MAX_ALBUM_INDEX:
            raise ValueError("图片序号必须是1～20之间的整数。")
        chat_segments = split_input_segments(chat_name)
        allowed, refs = _unique_allowed_texts(chat_segments)
        plan = [
            _step(1, "open_app", "打开微信", "wechat", "微信主界面清晰可见", target="微信"),
            _step(2, "open_target", "打开微信搜索", "wechat", "微信搜索框清晰可见", target="微信搜索"),
            _step(3, "focus_input", "聚焦聊天搜索框", "wechat", "搜索框为空且键盘完整清晰", target="微信搜索框"),
        ]
        search_steps, search_session = _text_steps(
            start_index=4,
            app_id="wechat",
            field_target="微信搜索框",
            text=chat_name,
            text_indexes=refs,
        )
        plan.extend(search_steps)
        plan.append(_step(len(plan) + 1, "submit", "提交聊天搜索", "wechat", "出现聊天搜索结果", target="搜索"))
        plan.append(_step(len(plan) + 1, "open_target", f"打开唯一匹配聊天：{chat_name}", "wechat", f"顶部标题逐字显示{chat_name}", target=chat_name))
        plan.append(_step(len(plan) + 1, "open_target", "打开加号菜单", "wechat", "加号菜单完整显示", target="＋菜单"))
        plan.append(_step(len(plan) + 1, "open_target", "打开相册最近", "wechat", "相册标题为最近且缩略图网格稳定", target="相册最近"))
        plan.append(_step(len(plan) + 1, "select_media", f"选择最近第{image_index}张图片", "wechat", "只选中一张且选择计数为1", target=f"最近相册第{image_index}张", image_index=image_index))
        plan.append(_step(len(plan) + 1, "submit", "发送所选图片", "wechat", "返回原聊天并出现新的图片消息", target="发送按钮"))
        plan.append(_step(len(plan) + 1, "verify_result", "复核图片消息", "wechat", f"标题仍为{chat_name}且出现新的图片消息", target=chat_name))
        goal = (
            f"打开微信并进入唯一完全匹配的聊天{chat_name}，打开相册最近，"
            f"从左上到右下选择第{image_index}张图片，只选一张并发送。"
        )
        sessions = [search_session]
    elif operation == "douyin.search":
        keyword = normalize_user_text(params.get("keyword"), field_name="搜索关键词")
        segments = split_input_segments(keyword)
        allowed, refs = _unique_allowed_texts(segments)
        plan = [
            _step(1, "open_app", "打开抖音", "douyin", "抖音首页清晰可见", target="抖音"),
            _step(2, "open_target", "打开抖音搜索", "douyin", "搜索框清晰可见", target="搜索"),
            _step(3, "focus_input", "聚焦搜索框", "douyin", "搜索框为空且键盘完整清晰", target="抖音搜索框"),
        ]
        text_steps, search_session = _text_steps(
            start_index=4,
            app_id="douyin",
            field_target="抖音搜索框",
            text=keyword,
            text_indexes=refs,
        )
        plan.extend(text_steps)
        plan.append(_step(len(plan) + 1, "submit", "提交搜索", "douyin", f"搜索框逐字显示{keyword}并进入结果页", target="搜索按钮"))
        plan.append(_step(len(plan) + 1, "open_target", "进入视频结果", "douyin", "视频结果页稳定显示", target="视频"))
        plan.append(_step(len(plan) + 1, "verify_result", "复核搜索结果", "douyin", f"结果页关键词逐字显示{keyword}", target=keyword))
        goal = f"打开抖音，搜索关键词{keyword}，进入视频结果页后停止。输入错误必须全部清空后从头重打。"
        sessions = [search_session]
    else:
        keyword_value = params.get("keyword")
        keyword = normalize_user_text(keyword_value, field_name="搜索关键词") if isinstance(keyword_value, str) and keyword_value.strip() else None
        target_count = params.get("target_count")
        if isinstance(target_count, bool) or not isinstance(target_count, int) or not 1 <= target_count <= MAX_DOUYIN_TARGETS:
            raise ValueError("目标数量必须是1～10之间的整数。")
        like = params.get("like") is True
        comment = params.get("comment") is True
        if not like and not comment:
            raise ValueError("批量任务至少选择点赞或评论之一。")
        comment_text = None
        text_values: list[str] = []
        if keyword:
            text_values.extend(split_input_segments(keyword))
        comment_segments: list[str] = []
        if comment:
            comment_text = normalize_user_text(params.get("comment_text"), field_name="评论内容")
            comment_segments = split_input_segments(comment_text)
            text_values.extend(comment_segments)
            text_values.append(comment_text)
        allowed, refs = _unique_allowed_texts(text_values)
        plan = [_step(1, "open_app", "打开抖音", "douyin", "抖音首页清晰可见", target="抖音")]
        sessions = []
        if keyword:
            plan.extend([
                _step(2, "open_target", "打开抖音搜索", "douyin", "搜索框清晰可见", target="搜索"),
                _step(3, "focus_input", "聚焦搜索框", "douyin", "搜索框为空且键盘完整清晰", target="抖音搜索框"),
            ])
            text_steps, search_session = _text_steps(
                start_index=4,
                app_id="douyin",
                field_target="抖音搜索框",
                text=keyword,
                text_indexes=refs,
            )
            plan.extend(text_steps)
            plan.append(_step(len(plan) + 1, "submit", "提交搜索", "douyin", "进入搜索结果页", target="搜索按钮"))
            plan.append(_step(len(plan) + 1, "open_target", "进入视频结果", "douyin", "视频结果页稳定显示", target="视频"))
            sessions.append(search_session)
        plan.append(
            _step(
                len(plan) + 1,
                "interact_batch",
                f"逐页完成{target_count}个目标视频",
                "douyin",
                f"成功完成{target_count}个普通视频，评论区均已关闭",
                target="普通视频",
                text_ref=refs[comment_text] if comment_text else None,
                comment_text_refs=(
                    [refs[segment] for segment in comment_segments]
                    if comment_text
                    else []
                ),
                count=target_count,
                like=like,
                comment=comment,
                max_pages=target_count + 5,
            )
        )
        actions = "点赞并评论" if like and comment else ("点赞" if like else "评论")
        scope = f"搜索关键词{keyword}后" if keyword else "在当前推荐流"
        comment_part = f"，评论原文为{comment_text}" if comment_text else ""
        goal = (
            f"打开抖音，{scope}{actions}{target_count}个普通视频{comment_part}。"
            f"最多检查{target_count + 5}个页面；直播、广告直接跳过，页面未知则安全停止。"
            "评论输入错误必须全部清空后从头重打；评论发送后先关闭评论区再上划。"
        )

    return {
        "goal": goal,
        "allowed_texts": allowed,
        "execution_plan": plan,
        "input_sessions": sessions,
        "input_policy": {
            "verify_roi_only": True,
            "require_initial_empty": True,
            "clear_entire_attempt": True,
            "max_full_retypes": MAX_FULL_RETYPES,
        },
        "task_mode": "operate",
        "expected_result": None,
        "source_operation": operation,
        "source_params": dict(params),
    }


def build_state_workflow_params(
    operation: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Validate a public operation without producing a linear action plan.

    The state-graph runner receives only the immutable task target.  It derives
    the next action from the freshly observed page instead of advancing a plan
    cursor created before execution.
    """

    if operation not in STRUCTURED_OPERATIONS:
        raise ValueError(f"不支持的结构化操作：{operation}")

    if operation == "wechat.send_text":
        source_params = {
            "chat_name": normalize_chat_name(params.get("chat_name")),
            "text": normalize_user_text(params.get("text"), field_name="消息正文"),
        }
        summary = (
            f"微信：向唯一完全匹配的聊天“{source_params['chat_name']}”"
            f"发送文字“{source_params['text']}”"
        )
    elif operation == "wechat.send_album_image":
        image_index = params.get("image_index")
        if (
            isinstance(image_index, bool)
            or not isinstance(image_index, int)
            or not 1 <= image_index <= MAX_ALBUM_INDEX
        ):
            raise ValueError("图片序号必须是1～20之间的整数。")
        source_params = {
            "chat_name": normalize_chat_name(params.get("chat_name")),
            "image_index": image_index,
        }
        summary = (
            f"微信：向唯一完全匹配的聊天“{source_params['chat_name']}”"
            f"发送最近相册第{image_index}张图片"
        )
    elif operation == "douyin.search":
        source_params = {
            "keyword": normalize_user_text(params.get("keyword"), field_name="搜索关键词")
        }
        summary = f"抖音：搜索“{source_params['keyword']}”并进入视频结果"
    else:
        keyword_value = params.get("keyword")
        keyword = (
            normalize_user_text(keyword_value, field_name="搜索关键词")
            if isinstance(keyword_value, str) and keyword_value.strip()
            else None
        )
        target_count = params.get("target_count")
        if (
            isinstance(target_count, bool)
            or not isinstance(target_count, int)
            or not 1 <= target_count <= MAX_DOUYIN_TARGETS
        ):
            raise ValueError("目标数量必须是1～10之间的整数。")
        like = params.get("like") is True
        comment = params.get("comment") is True
        if not like and not comment:
            raise ValueError("批量任务至少选择点赞或评论之一。")
        comment_text = (
            normalize_user_text(params.get("comment_text"), field_name="评论内容")
            if comment
            else None
        )
        source_params = {
            "keyword": keyword,
            "target_count": target_count,
            "like": like,
            "comment": comment,
            "comment_text": comment_text,
        }
        action_name = "点赞并评论" if like and comment else ("点赞" if like else "评论")
        scope = f"搜索“{keyword}”后" if keyword else "当前推荐流"
        summary = f"抖音：{scope}{action_name}{target_count}个普通视频"

    return {
        "workflow_version": "page_state_graph_v1",
        "controller": "single_state_controller",
        "model_role": "observation_only",
        "summary": summary,
        "source_operation": operation,
        "source_params": source_params,
        "safety": {
            "min_confidence": 0.72,
            "one_action_per_observation": True,
            "unknown_state_stops": True,
            "verify_after_every_action": True,
            "max_full_retypes": MAX_FULL_RETYPES,
        },
    }
