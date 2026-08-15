from __future__ import annotations

import base64
import binascii
import json
import os
import re
import time
from dataclasses import asdict, dataclass, replace
from io import BytesIO
from pathlib import Path
from typing import Any, Protocol

import httpx
from PIL import Image, ImageChops, ImageStat

import robot_gui_poc as legacy
from operation_specs import (
    InputAttemptState,
    InputRecoveryCoordinator,
    editable_character_count,
)
from vision_model_config import (
    DEFAULT_VISION_BASE_URL,
    DEFAULT_VISION_MODEL,
    VisionModelConfig,
    load_vision_model_config,
)


DEFAULT_MODEL = DEFAULT_VISION_MODEL
DEFAULT_BASE_URL = DEFAULT_VISION_BASE_URL
ALLOWED_ACTIONS = {
    "tap",
    "type_symbol",
    "type_text",
    "type_pinyin",
    "clear_text",
    "swipe_up",
    "swipe_down",
    "swipe_left",
    "swipe_right",
    "android_home",
    "android_back",
    "wait",
    "finish",
    "stop",
}
ALLOWED_SCREEN_TYPES = {
    "android_home",
    "app_page",
    "chat_list",
    "chat",
    "video",
    "live",
    "dialog",
    "keyboard",
    "loading",
    "unknown",
}
FORBIDDEN_GOAL_RE = re.compile(
    r"(支付|付款|下单|购买|转账|红包|提现|银行卡|密码|验证码|"
    r"删除(?:文件|聊天|账号|数据)|注销账号|修改密码|绕过验证|"
    r"批量关注|连续.{0,8}关注|\d+\s*个.{0,8}关注)"
)
SOCIAL_INTERACTION_RE = re.compile(r"(点赞|评论)")
ARABIC_INTERACTION_COUNT_RE = re.compile(
    r"(?:接下来|连续|前|后|下)?(?:的)?\s*(\d{1,3})\s*(?:条|个|次|段)?"
    r".{0,12}(?:视频)?(?:点赞|评论)|"
    r"(?:点赞|评论).{0,12}?(\d{1,3})\s*(?:条|个|次|段)?"
)
CHINESE_DIGITS = {
    "一": 1,
    "两": 2,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
CHINESE_INTERACTION_COUNT_RE = re.compile(
    r"(?:接下来|连续|前|后|下)?(?:的)?\s*"
    r"([一二两三四五六七八九十]{1,3})\s*(?:条|个|次|段)?"
    r".{0,12}(?:视频)?(?:点赞|评论)"
)
MAX_SOCIAL_INTERACTIONS_PER_TASK = 10
MAX_EXECUTION_PLAN_STEPS = 110
MAX_ALLOWED_TEXTS = 100
ALLOWED_PLAN_INTENTS = {
    "open_app",
    "open_target",
    "focus_input",
    "clear_input",
    "enter_text",
    "submit",
    "like_current",
    "comment_current",
    "interact_batch",
    "select_media",
    "next_item",
    "verify_result",
    "navigate_back",
    "navigate_home",
}
PLAN_APP_ALIASES = {
    "微信": "wechat",
    "wechat": "wechat",
    "抖音": "douyin",
    "douyin": "douyin",
    "设置": "settings",
    "系统设置": "settings",
    "settings": "settings",
    "安卓系统": "android",
    "android": "android",
}
PLAN_APP_DISPLAY_NAMES = {
    "wechat": "微信",
    "douyin": "抖音",
    "settings": "设置",
    "android": "安卓系统",
}
PLAN_INTENT_ACTIONS = {
    "open_app": {"tap", "android_home", "android_back", "wait", "stop"},
    "open_target": {"tap", "android_back", "wait", "stop"},
    "focus_input": {"tap", "wait", "stop"},
    "clear_input": {"clear_text", "wait", "stop"},
    "enter_text": {
        "tap",
        "type_symbol",
        "type_text",
        "type_pinyin",
        "clear_text",
        "wait",
        "finish",
        "stop",
    },
    "submit": {"tap", "wait", "finish", "stop"},
    "like_current": {"tap", "swipe_up", "android_back", "wait", "finish", "stop"},
    "comment_current": {
        "tap",
        "type_symbol",
        "type_text",
        "type_pinyin",
        "clear_text",
        "android_back",
        "wait",
        "finish",
        "stop",
    },
    "interact_batch": {
        "tap",
        "type_symbol",
        "type_text",
        "type_pinyin",
        "clear_text",
        "swipe_up",
        "android_back",
        "wait",
        "finish",
        "stop",
    },
    "select_media": {"tap", "swipe_up", "swipe_down", "wait", "stop"},
    "next_item": {"swipe_up", "android_back", "wait", "finish", "stop"},
    "verify_result": {"wait", "finish", "stop"},
    "navigate_back": {"android_back", "wait", "stop"},
    "navigate_home": {"android_home", "wait", "stop"},
}
UNCHANGED_FRAME_MEAN_DELTA = 3.0
PINYIN_CANDIDATE_MISSED_TAP_ROI_DELTA = 3.0
SEND_INPUT_TRANSITION_MIN_DELTA = 3.0
SEND_MESSAGE_TRANSITION_MIN_DELTA = 1.5
SEND_POST_STABLE_MAX_DELTA = 2.0
# The seller preview places WeChat's real editor row around 51.5%-59.5% of
# the captured phone image.  The IME candidate / association strip can retain
# the last committed word immediately above that row.  Starting the crop at
# 48% included that strip and caused a cleared editor to be misread as still
# containing the old text.  Keep this ROI deliberately narrow: it is for the
# editor only, not candidate selection.
WECHAT_INPUT_ROI_TOP = 0.515
WECHAT_INPUT_ROI_BOTTOM = 0.595
WECHAT_CANDIDATE_ROI_TOP = 0.50
WECHAT_CANDIDATE_ROI_BOTTOM = 0.71
SAFE_NAVIGATION_RETRY_SCREENS = {"android_home", "chat_list"}
SAFE_NAVIGATION_TARGET_RE = re.compile(r"(应用图标|App图标|图标|文件传输助手|聊天|会话)")
UNSAFE_NAVIGATION_TARGET_RE = re.compile(
    r"(删除|清空|取消|发送|确认|支付|购买|点赞|评论|关注|转账)"
)


class VisionAgentError(RuntimeError):
    """The remote model or its response cannot be used safely."""


def goal_is_forbidden(text: str) -> bool:
    return bool(FORBIDDEN_GOAL_RE.search(text))


def requested_social_interaction_count(text: str) -> int:
    """Return the explicit like/comment count, or one for a single action."""
    if not SOCIAL_INTERACTION_RE.search(text):
        return 0
    match = ARABIC_INTERACTION_COUNT_RE.search(text)
    if match:
        return int(match.group(1) or match.group(2))
    chinese_match = CHINESE_INTERACTION_COUNT_RE.search(text)
    if chinese_match:
        token = chinese_match.group(1)
        if "十" in token:
            tens_text, ones_text = token.split("十", 1)
            tens = CHINESE_DIGITS.get(tens_text, 1) if tens_text else 1
            ones = CHINESE_DIGITS.get(ones_text, 0) if ones_text else 0
            return tens * 10 + ones
        return CHINESE_DIGITS.get(token, 1)
    return 1


def validate_execution_plan(
    raw_plan: Any,
    *,
    goal: str,
    allowed_texts: list[str],
) -> list[dict[str, Any]]:
    """Validate a frozen, high-level plan without accepting low-level commands.

    Coordinates and shell-like operations are deliberately absent.  Text steps
    may only reference an already validated ``allowed_texts`` entry.
    """

    if raw_plan in (None, []):
        return []
    if not isinstance(raw_plan, list):
        raise VisionAgentError(
            f"execution_plan 必须是1～{MAX_EXECUTION_PLAN_STEPS}步的数组。"
        )
    candidate_steps = list(raw_plan)
    # Finishing the last frozen step already stops the controller.  Some
    # models mirror a user's "然后停止" as a terminal stop/finish step; remove
    # only that harmless trailing marker.  The same intent anywhere else is
    # still rejected by the whitelist below.
    while candidate_steps and isinstance(candidate_steps[-1], dict):
        terminal_intent = str(candidate_steps[-1].get("intent", "")).strip()
        if terminal_intent not in {"stop", "finish"}:
            break
        candidate_steps.pop()
    if not 1 <= len(candidate_steps) <= MAX_EXECUTION_PLAN_STEPS:
        raise VisionAgentError(
            f"execution_plan 必须是1～{MAX_EXECUTION_PLAN_STEPS}步的数组。"
        )

    plan: list[dict[str, Any]] = []
    plan_apps: set[str] = set()
    social_count = requested_social_interaction_count(goal)
    send_forbidden = any(
        token in goal for token in ("不要发送", "不发送", "别发送", "禁止发送")
    )
    for index, raw_step in enumerate(candidate_steps, start=1):
        if not isinstance(raw_step, dict):
            raise VisionAgentError(f"execution_plan 第{index}步必须是对象。")
        intent = str(raw_step.get("intent", "")).strip()
        if intent not in ALLOWED_PLAN_INTENTS:
            raise VisionAgentError(
                f"execution_plan 第{index}步包含未授权意图：{intent or '(empty)'}。"
            )
        step_id = str(raw_step.get("id") or f"step_{index}").strip()
        if step_id != f"step_{index}":
            raise VisionAgentError("execution_plan 的 id 必须按 step_1、step_2 顺序排列。")
        label = str(raw_step.get("label", "")).strip()
        checkpoint = str(raw_step.get("checkpoint", "")).strip()
        if not label or len(label) > 80 or not checkpoint or len(checkpoint) > 120:
            raise VisionAgentError(
                f"execution_plan 第{index}步缺少简短 label/checkpoint。"
            )
        raw_app_id = str(raw_step.get("app_id", "")).strip()
        app_id = PLAN_APP_ALIASES.get(raw_app_id.lower(), raw_app_id.lower())
        if not app_id or not re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", app_id):
            raise VisionAgentError(f"execution_plan 第{index}步 app_id 无效。")
        plan_apps.add(app_id)

        target_value = raw_step.get("target")
        target = str(target_value).strip() if target_value is not None else None
        if intent == "open_app" and not target:
            target = PLAN_APP_DISPLAY_NAMES.get(app_id, app_id)
        elif intent == "open_target" and not target:
            inferred_target = re.sub(
                r"^(?:打开|进入|选择|点击|前往|定位到)\s*",
                "",
                label,
            ).strip()
            target = inferred_target or None
        if target is not None and (not target or len(target) > 80):
            raise VisionAgentError(f"execution_plan 第{index}步 target 无效。")
        if intent in {"open_app", "open_target"} and not target:
            raise VisionAgentError(
                f"execution_plan 第{index}步 {intent} 必须提供明确 target。"
            )

        text_ref_value = raw_step.get("text_ref")
        text_ref: int | None = None
        if text_ref_value is not None:
            if (
                isinstance(text_ref_value, bool)
                or not isinstance(text_ref_value, int)
                or not 0 <= text_ref_value < len(allowed_texts)
            ):
                raise VisionAgentError(
                    f"execution_plan 第{index}步 text_ref 未指向文字白名单。"
                )
            text_ref = text_ref_value
        if intent in {"enter_text", "comment_current"} and text_ref is None:
            raise VisionAgentError(
                f"execution_plan 第{index}步 {intent} 必须提供 text_ref。"
            )
        if intent not in {"enter_text", "comment_current", "interact_batch"} and text_ref is not None:
            raise VisionAgentError(
                f"execution_plan 第{index}步 {intent} 不允许携带 text_ref。"
            )

        count_value = raw_step.get("count", 1)
        if isinstance(count_value, bool) or not isinstance(count_value, int):
            raise VisionAgentError(f"execution_plan 第{index}步 count 必须是整数。")
        if not 1 <= count_value <= MAX_SOCIAL_INTERACTIONS_PER_TASK:
            raise VisionAgentError(f"execution_plan 第{index}步 count 超出安全范围。")
        if intent in {"like_current", "comment_current", "interact_batch"} and social_count:
            if count_value != social_count:
                raise VisionAgentError(
                    f"execution_plan 第{index}步 count 与用户原文数量不一致。"
                )
        elif count_value != 1:
            raise VisionAgentError(
                f"execution_plan 第{index}步只有点赞、评论或批量互动允许 count>1。"
            )

        if intent == "submit" and send_forbidden:
            raise VisionAgentError("用户明确禁止发送，execution_plan 不得包含 submit。")
        if intent in {"like_current", "comment_current", "interact_batch", "next_item"} and app_id != "douyin":
            raise VisionAgentError(f"execution_plan 第{index}步 {intent} 只允许用于抖音。")

        expected_input_value = raw_step.get("expected_input")
        expected_input = None
        if expected_input_value is not None:
            if (
                intent != "enter_text"
                or not isinstance(expected_input_value, str)
                or not expected_input_value
                or len(expected_input_value) > 100
            ):
                raise VisionAgentError(
                    f"execution_plan 第{index}步 expected_input 无效。"
                )
            expected_input = expected_input_value

        extra: dict[str, Any] = {}
        if intent == "interact_batch":
            like_value = raw_step.get("like") is True
            comment_value = raw_step.get("comment") is True
            max_pages = raw_step.get("max_pages")
            if not like_value and not comment_value:
                raise VisionAgentError("批量互动至少启用点赞或评论之一。")
            if comment_value and text_ref is None:
                raise VisionAgentError("批量评论必须提供 text_ref。")
            comment_text_refs = raw_step.get("comment_text_refs", [])
            if not isinstance(comment_text_refs, list):
                raise VisionAgentError("批量评论 comment_text_refs 必须是列表。")
            if comment_value:
                if (
                    not comment_text_refs
                    or not all(
                        isinstance(ref, int)
                        and not isinstance(ref, bool)
                        and 0 <= ref < len(allowed_texts)
                        for ref in comment_text_refs
                    )
                    or "".join(allowed_texts[ref] for ref in comment_text_refs)
                    != allowed_texts[text_ref]
                ):
                    raise VisionAgentError(
                        "批量评论分段必须完整拼回用户确认的评论原文。"
                    )
            elif comment_text_refs:
                raise VisionAgentError("未启用评论时不允许 comment_text_refs。")
            if (
                isinstance(max_pages, bool)
                or not isinstance(max_pages, int)
                or max_pages != min(15, count_value + 5)
            ):
                raise VisionAgentError("批量互动 max_pages 必须等于目标数量+5。")
            extra = {
                "like": like_value,
                "comment": comment_value,
                "max_pages": max_pages,
                "comment_text_refs": list(comment_text_refs),
            }
        elif intent == "select_media":
            image_index = raw_step.get("image_index")
            if (
                isinstance(image_index, bool)
                or not isinstance(image_index, int)
                or not 1 <= image_index <= 20
            ):
                raise VisionAgentError("相册图片序号必须是1～20。")
            extra = {"image_index": image_index}

        plan.append(
            {
                "id": step_id,
                "intent": intent,
                "label": label,
                "app_id": app_id,
                "target": target,
                "text_ref": text_ref,
                "count": count_value,
                "checkpoint": checkpoint,
                **({"expected_input": expected_input} if expected_input else {}),
                **extra,
            }
        )

    if len(plan_apps) != 1:
        raise VisionAgentError("单个 execution_plan 只能操作一个 App。")
    if any(step["intent"] == "submit" for step in plan) and not any(
        step["intent"] in {"enter_text", "comment_current"} for step in plan
    ):
        raise VisionAgentError("submit 前必须有输入或评论步骤。")
    return plan


def plan_checkpoint_is_proven(
    step: dict[str, Any],
    decision: "VisionDecision",
    allowed_texts: list[str],
) -> bool:
    """Apply conservative local evidence rules before advancing a plan step."""

    intent = str(step.get("intent") or "")
    target = str(step.get("target") or "").strip()
    evidence = " ".join(
        (decision.page_title or "", decision.target, decision.reason)
    ).strip()
    if intent == "open_app":
        return (
            decision.screen_type
            in {"app_page", "chat_list", "chat", "video", "keyboard"}
            and target in evidence
        )
    if intent == "open_target":
        # A resumed app may open directly on a conversation with the keyboard
        # already visible.  That is still valid target-page evidence; requiring
        # the intermediate chat-list layout would incorrectly force the plan
        # backwards.
        return (
            decision.screen_type in {"chat", "app_page", "keyboard"}
            and target in evidence
        )
    if intent == "focus_input":
        focused = decision.screen_type in {"chat", "keyboard"} and any(
            token in evidence for token in ("输入框", "编辑框", "键盘已显示", "键盘可见")
        )
        if "为空" in str(step.get("checkpoint") or ""):
            return focused and decision.input_is_empty is True
        return focused
    if intent == "clear_input":
        return decision.input_is_empty is True
    if intent == "enter_text":
        text_ref = step.get("text_ref")
        return (
            isinstance(text_ref, int)
            and 0 <= text_ref < len(allowed_texts)
            and decision.observed_input_text
            == str(step.get("expected_input") or allowed_texts[text_ref])
        )
    if intent == "select_media":
        return (
            decision.screen_type == "app_page"
            and decision.success
            and any(
                token in evidence
                for token in ("选择计数为1", "已选1张", "只选中一张")
            )
        )
    if intent == "navigate_home":
        return decision.screen_type == "android_home"
    if intent == "navigate_back":
        return decision.screen_type not in {"loading", "unknown"} and "返回" in evidence
    if intent == "verify_result":
        return (
            decision.success
            and decision.screen_type not in {"loading", "unknown"}
            and (not target or target in evidence)
        )
    # submit/like/comment/next_item have dedicated controller-owned completion
    # checks in the execution loop and are intentionally not advanced from a
    # free-form model assertion here.
    return False


def infer_safe_plan_resume_cursor(
    execution_plan: list[dict[str, Any]],
    plan_cursor: int,
    decision: "VisionDecision",
    allowed_texts: list[str],
) -> int:
    """Resume a frozen plan when an app restores a clearly deeper UI state.

    Only navigation/focus prerequisites may be skipped, and only when the same
    observation proves the named target and focused keyboard.  Input, submit,
    social interaction and result-verification checkpoints are never skipped.
    """

    if not execution_plan or not 0 <= plan_cursor < len(execution_plan):
        return plan_cursor
    evidence = " ".join(
        (decision.page_title or "", decision.target, decision.reason)
    ).strip()
    resumable_actions = {
        "clear_text": "clear_input",
        "type_text": "enter_text",
        "type_pinyin": "enter_text",
    }
    expected_intent = resumable_actions.get(decision.action)
    if expected_intent is None:
        return plan_cursor

    for candidate in range(plan_cursor + 1, len(execution_plan)):
        candidate_step = execution_plan[candidate]
        if candidate_step.get("intent") != expected_intent:
            continue
        if decision.action in {"type_text", "type_pinyin"}:
            text_ref = candidate_step.get("text_ref")
            if (
                not isinstance(text_ref, int)
                or not 0 <= text_ref < len(allowed_texts)
                or decision.text != allowed_texts[text_ref]
            ):
                continue

        # A restored app can bypass an entire search/navigation prefix and
        # reopen directly on a later named chat.  In that case the earlier
        # search typing/submission steps are no longer prerequisites.  Resume
        # only from the latest visually proven named target, and only across
        # focus-input steps.  If the focused/empty checkpoint itself is not yet
        # proven, return that checkpoint first; the execution loop will turn
        # the premature typing proposal into a wait and obtain fresh evidence.
        matched_target_index: int | None = None
        for index in range(candidate - 1, plan_cursor - 1, -1):
            step = execution_plan[index]
            if (
                step.get("intent") == "open_target"
                and step.get("app_id") == candidate_step.get("app_id")
                and plan_checkpoint_is_proven(step, decision, allowed_texts)
            ):
                matched_target_index = index
                break
        if matched_target_index is not None:
            target_suffix = execution_plan[matched_target_index + 1 : candidate]
            if target_suffix and all(
                step.get("intent") == "focus_input" for step in target_suffix
            ):
                for index in range(matched_target_index + 1, candidate):
                    if not plan_checkpoint_is_proven(
                        execution_plan[index], decision, allowed_texts
                    ):
                        return index
                return candidate

        prerequisites = execution_plan[plan_cursor:candidate]
        if any(
            step.get("intent") not in {"open_app", "open_target", "focus_input"}
            for step in prerequisites
        ):
            continue

        direct_proof: dict[int, bool] = {
            index: plan_checkpoint_is_proven(step, decision, allowed_texts)
            for index, step in enumerate(
                execution_plan[plan_cursor:candidate], start=plan_cursor
            )
        }
        named_target_proven = any(
            direct_proof[index]
            and execution_plan[index].get("intent") == "open_target"
            and execution_plan[index].get("app_id") == candidate_step.get("app_id")
            for index in range(plan_cursor, candidate)
        )
        prerequisites_proven = True
        for index in range(plan_cursor, candidate):
            step = execution_plan[index]
            intent = step.get("intent")
            if direct_proof[index]:
                continue
            # Reaching a named target inside the same app proves that the app
            # itself is open, even when the launcher step was bypassed by the
            # app restoring its previous activity.
            if intent == "open_app" and named_target_proven:
                continue
            prerequisites_proven = False
            break
        if not prerequisites_proven:
            continue
        if decision.screen_type not in {"chat", "keyboard"}:
            continue
        if decision.action in {"type_pinyin", "clear_text"} and (
            decision.keyboard_layout is None
            or decision.keyboard_layout.get("type") != "qwerty"
        ):
            continue
        if not any(token in evidence for token in ("输入框", "编辑框", "键盘")):
            continue
        return candidate
    return plan_cursor


def frame_mean_delta(first: Image.Image, second: Image.Image) -> float:
    """Return a low-resolution grayscale difference score for loop detection."""
    if first.size != second.size:
        return float("inf")
    first_small = first.convert("L").resize((64, 64))
    second_small = second.convert("L").resize((64, 64))
    return float(ImageStat.Stat(ImageChops.difference(first_small, second_small)).mean[0])


def frame_region_delta(
    first: Image.Image,
    second: Image.Image,
    *,
    left_ratio: float,
    top_ratio: float,
    right_ratio: float,
    bottom_ratio: float,
) -> float:
    """Compare the same normalized rectangular region in two camera frames."""

    if first.size != second.size:
        return float("inf")
    if not (
        0.0 <= left_ratio < right_ratio <= 1.0
        and 0.0 <= top_ratio < bottom_ratio <= 1.0
    ):
        raise ValueError("区域比例必须满足 0 <= left < right <= 1 且 0 <= top < bottom <= 1。")

    def crop(image: Image.Image) -> Image.Image:
        left = int(round(image.width * left_ratio))
        top = int(round(image.height * top_ratio))
        right = max(left + 1, int(round(image.width * right_ratio)))
        bottom = max(top + 1, int(round(image.height * bottom_ratio)))
        return image.crop((left, top, right, bottom))

    return frame_mean_delta(crop(first), crop(second))


def sent_message_transition_metrics(
    before_send: Image.Image,
    after_send: Image.Image,
) -> dict[str, float | bool]:
    """Return controller-owned evidence that a WeChat send tap took effect.

    The input area must change from the pre-send state and the right-hand chat
    area must independently change.  This prevents an old green bubble that is
    present in both frames from being treated as proof of a new send.
    """

    input_delta = frame_region_delta(
        before_send,
        after_send,
        left_ratio=0.0,
        top_ratio=WECHAT_INPUT_ROI_TOP,
        right_ratio=1.0,
        bottom_ratio=WECHAT_INPUT_ROI_BOTTOM,
    )
    message_delta = frame_region_delta(
        before_send,
        after_send,
        left_ratio=0.35,
        top_ratio=0.05,
        right_ratio=0.95,
        bottom_ratio=0.55,
    )
    return {
        "input_delta": input_delta,
        "message_delta": message_delta,
        "transition_visible": (
            input_delta >= SEND_INPUT_TRANSITION_MIN_DELTA
            and message_delta >= SEND_MESSAGE_TRANSITION_MIN_DELTA
        ),
    }


def enlarged_vertical_roi(
    image: Image.Image,
    *,
    top_ratio: float,
    bottom_ratio: float,
    scale: int = 2,
) -> Image.Image:
    """Return an enlarged vertical crop without changing the source frame.

    The full camera frames remain available for page-level context.  This crop
    is appended as the last image so text decisions can be based on the small
    input area instead of visually similar chat bubbles above it.
    """

    if not 0.0 <= top_ratio < bottom_ratio <= 1.0:
        raise ValueError("ROI 比例必须满足 0 <= top < bottom <= 1。")
    top = max(0, int(round(image.height * top_ratio)))
    bottom = min(
        image.height,
        max(top + 1, int(round(image.height * bottom_ratio))),
    )
    roi = image.crop((0, top, image.width, bottom))
    return roi.resize(
        (roi.width * max(1, scale), roi.height * max(1, scale)),
        Image.Resampling.LANCZOS,
    )


def tap_coordinates_are_close(
    first: Any,
    second: Any,
    *,
    tolerance: int = 20,
) -> bool:
    first_point = _validate_coordinate(first)
    second_point = _validate_coordinate(second)
    if first_point is None or second_point is None:
        return first_point == second_point
    return (
        abs(first_point[0] - second_point[0]) <= tolerance
        and abs(first_point[1] - second_point[1]) <= tolerance
    )


def repeated_action_matches(previous: dict[str, Any], current: dict[str, Any]) -> bool:
    if previous.get("action") != current.get("action"):
        return False
    if previous.get("target") != current.get("target"):
        return False
    if previous.get("text") != current.get("text"):
        return False
    if previous.get("pinyin") != current.get("pinyin"):
        return False
    if previous.get("observed_input_text") != current.get("observed_input_text"):
        return False
    if previous.get("delete_count") != current.get("delete_count"):
        return False
    if current.get("action") == "tap":
        return tap_coordinates_are_close(
            previous.get("coordinate"),
            current.get("coordinate"),
        )
    return previous.get("coordinate") == current.get("coordinate")


def safe_navigation_retry_key(item: dict[str, Any]) -> tuple[str, str] | None:
    if item.get("action") != "tap":
        return None
    screen_type = str(item.get("screen_type") or "")
    target = str(item.get("target") or "")
    if screen_type not in SAFE_NAVIGATION_RETRY_SCREENS:
        return None
    if not SAFE_NAVIGATION_TARGET_RE.search(target):
        return None
    if UNSAFE_NAVIGATION_TARGET_RE.search(target):
        return None
    return screen_type, target


def canonical_visible_pinyin(value: str) -> str:
    """Normalize only separators that a Pinyin IME may insert automatically."""
    return re.sub(r"[\s'’]", "", value).lower()


def exact_backspace_count(value: str) -> int:
    """Return the number of editable units represented by visible input.

    A Pinyin IME may render apostrophes or spaces between syllables even though
    the user never typed those separators.  Backspace removes the underlying
    letters, so those automatic separators must not be counted.  For Chinese,
    digits, symbols, and mixed text every visible character remains one unit.
    """

    if re.fullmatch(r"[A-Za-z\s'’]+", value):
        return len(canonical_visible_pinyin(value))
    return len(value)


DOUYIN_HEART_CENTER = (860, 514)


def camera_point_to_normalized(
    image: Image.Image,
    center: tuple[int, int],
) -> tuple[int, int]:
    width, height = image.size
    return (
        int(round(center[0] * 1000 / max(1, width - 1))),
        int(round(center[1] * 1000 / max(1, height - 1))),
    )


def classify_douyin_heart_roi(
    image: Image.Image,
    center: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Measure heart colour in a small camera-pixel ROI.

    ``center`` is supplied by the dynamic right-rail detector during real
    execution.  The legacy calibrated coordinate remains only as a compatibility
    fallback for isolated unit tests.
    """
    rgb = image.convert("RGB")
    width, height = rgb.size
    if center is None:
        center_x = int(round((width - 1) * DOUYIN_HEART_CENTER[0] / 1000))
        center_y = int(round((height - 1) * DOUYIN_HEART_CENTER[1] / 1000))
    else:
        center_x, center_y = center
    radius = max(22, min(34, int(round(min(width, height) * 0.052))))
    left = max(0, center_x - radius)
    top = max(0, center_y - radius)
    right = min(width, center_x + radius + 1)
    bottom = min(height, center_y + radius + 1)
    roi_image = rgb.crop((left, top, right, bottom))
    get_pixels = getattr(roi_image, "get_flattened_data", roi_image.getdata)
    pixels = list(get_pixels())
    pixel_count = max(1, len(pixels))
    red_count = 0
    white_count = 0
    for red, green, blue in pixels:
        if (
            red >= 145
            and red >= green * 1.35
            and red >= blue * 1.15
        ):
            red_count += 1
        if (
            min(red, green, blue) >= 165
            and max(red, green, blue) - min(red, green, blue) <= 70
        ):
            white_count += 1
    red_ratio = red_count / (white_count + 1)
    red_fraction = red_count / pixel_count
    white_fraction = white_count / pixel_count
    if (
        red_fraction >= 0.22
        or (red_fraction >= 0.17 and red_ratio >= 2.5)
    ):
        state = "liked"
    elif white_fraction >= 0.14 and red_fraction < 0.18:
        state = "unliked"
    else:
        state = "unknown"
    return {
        "state": state,
        "center_px": [center_x, center_y],
        "center": list(camera_point_to_normalized(image, (center_x, center_y))),
        "red_count": red_count,
        "white_count": white_count,
        "red_ratio": round(red_ratio, 3),
        "red_fraction": round(red_fraction, 4),
        "white_fraction": round(white_fraction, 4),
        "roi": [left, top, right, bottom],
    }


def locate_douyin_heart(image: Image.Image) -> dict[str, Any]:
    """Locate the current page's heart and reject implausible red-content blobs."""
    detection = legacy.detect_douyin_heart(image)
    result: dict[str, Any] = {
        "state": detection.state,
        "trusted": False,
        "center_px": list(detection.center) if detection.center else None,
        "center": (
            list(camera_point_to_normalized(image, detection.center))
            if detection.center
            else None
        ),
        "bbox": list(detection.bbox) if detection.bbox else None,
        "area": detection.area,
    }
    if not detection.center or not detection.bbox:
        return result

    width, height = image.size
    center_x, center_y = detection.center
    left, top, right, bottom = detection.bbox
    box_width = right - left
    box_height = bottom - top
    on_action_rail = (
        width * 0.78 <= center_x <= width * 0.94
        and height * 0.39 <= center_y <= height * 0.59
    )
    if detection.state == "unliked":
        plausible_shape = 22 <= box_width <= 58 and 20 <= box_height <= 55
    elif detection.state == "liked":
        # A true red heart is compact. Wide/tall red video content previously
        # caused false "liked" decisions, so it must not be accepted here.
        aspect = box_width / max(1, box_height)
        plausible_shape = (
            22 <= box_width <= 48
            and 20 <= box_height <= 52
            and 0.65 <= aspect <= 1.45
            and 300 <= detection.area <= 1900
        )
    else:
        plausible_shape = False
    result["trusted"] = bool(on_action_rail and plausible_shape)
    result["box_size"] = [box_width, box_height]
    return result


def douyin_heart_became_liked(
    before: dict[str, Any] | None,
    after: dict[str, Any],
) -> bool:
    """Verify a like from both the current color and the before/after gain."""
    if after.get("state") == "liked":
        return True
    if not before:
        return False
    before_fraction = float(before.get("red_fraction") or 0.0)
    after_fraction = float(after.get("red_fraction") or 0.0)
    before_count = int(before.get("red_count") or 0)
    after_count = int(after.get("red_count") or 0)
    return (
        after_fraction >= 0.18
        and after_fraction - before_fraction >= 0.10
        and after_count - before_count >= 250
    )


@dataclass(frozen=True)
class VisionDecision:
    screen_type: str
    action: str
    confidence: float
    reason: str
    page_title: str | None = None
    target: str = ""
    coordinate: tuple[int, int] | None = None
    text: str | None = None
    pinyin: str | None = None
    observed_input_text: str | None = None
    input_is_empty: bool | None = None
    sent_message_visible: bool | None = None
    delete_count: int | None = None
    keyboard_layout: dict[str, Any] | None = None
    wait_seconds: float = 1.0
    success: bool = False
    plan_step_id: str | None = None
    checkpoint_met: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.coordinate is not None:
            payload["coordinate"] = list(self.coordinate)
        return payload


class VisionProvider(Protocol):
    def status(self) -> dict[str, Any]: ...

    def parse_goal(self, text: str) -> dict[str, Any]: ...

    def decide(
        self,
        *,
        goal: str,
        frames: list[Image.Image],
        history: list[dict[str, Any]],
        allowed_texts: list[str],
        task_mode: str,
        expected_result: str | None,
    ) -> VisionDecision: ...


def _extract_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise VisionAgentError("模型没有返回 JSON 对象。")
        try:
            value = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise VisionAgentError(f"模型返回的 JSON 无法解析：{exc}") from exc
    if not isinstance(value, dict):
        raise VisionAgentError("模型返回值必须是 JSON 对象。")
    return value


def _validate_coordinate(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value)
    ):
        raise VisionAgentError("coordinate 必须是两个数字组成的数组。")
    x, y = (int(round(float(value[0]))), int(round(float(value[1]))))
    if not (0 <= x <= 1000 and 0 <= y <= 1000):
        raise VisionAgentError("模型坐标超出 0～1000 的安全范围。")
    return x, y


def _validate_keyboard_layout(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise VisionAgentError("keyboard_layout 必须是对象。")
    layout_type = value.get("type")
    if layout_type not in {"qwerty", "generic", "numeric_grid"}:
        raise VisionAgentError(
            "keyboard_layout.type 必须是 qwerty、generic 或 numeric_grid。"
        )
    raw_anchors = value.get("anchors")
    if not isinstance(raw_anchors, dict):
        raise VisionAgentError("keyboard_layout 缺少 anchors。")
    anchors: dict[str, list[int]] = {}
    if layout_type == "qwerty":
        required = ("q", "p", "a", "l", "z", "m", "backspace")
    elif layout_type == "numeric_grid":
        required = ("1", "3", "7", "9", "backspace")
    else:
        required = ("backspace",)
    for key in required:
        point = _validate_coordinate(raw_anchors.get(key))
        if point is None:
            raise VisionAgentError(f"keyboard_layout 缺少 {key} 锚点。")
        anchors[key] = [point[0], point[1]]
    return {"type": layout_type, "anchors": anchors}


def numeric_grid_key_coordinate(
    keyboard_layout: dict[str, Any],
    text: str,
    frame_size: tuple[int, int] | None = None,
) -> tuple[int, int]:
    """Calculate a numeric keypad key center from validated visual anchors.

    Qwen only identifies the four grid corners plus backspace.  The actual key
    center is calculated locally so a slightly low visual coordinate cannot
    land on the boundary between two rows.  On the observed Douyin keypad the
    backspace key shares the first digit row, making its Y coordinate the most
    reliable top-row reference.
    """

    if keyboard_layout.get("type") != "numeric_grid" or text not in "0123456789":
        raise VisionAgentError("数字输入缺少可用的 numeric_grid 键盘布局。")
    anchors = keyboard_layout.get("anchors") or {}
    try:
        one = tuple(int(value) for value in anchors["1"])
        three = tuple(int(value) for value in anchors["3"])
        seven = tuple(int(value) for value in anchors["7"])
        nine = tuple(int(value) for value in anchors["9"])
        backspace = tuple(int(value) for value in anchors["backspace"])
    except (KeyError, TypeError, ValueError) as exc:
        raise VisionAgentError("numeric_grid 键盘锚点不完整。") from exc

    left_x = round((one[0] + seven[0]) / 2)
    right_x = round((three[0] + nine[0]) / 2)
    top_y = backspace[1]
    bottom_y = round((seven[1] + nine[1]) / 2)
    x_step = (right_x - left_x) / 2
    y_step = (bottom_y - top_y) / 2
    if not (
        500 <= top_y <= 850
        and 650 <= bottom_y <= 940
        and 120 <= right_x - left_x <= 650
        and 35 <= y_step <= 140
        and abs(one[1] - three[1]) <= 80
        and abs(seven[1] - nine[1]) <= 80
        and abs(one[0] - seven[0]) <= 100
        and abs(three[0] - nine[0]) <= 100
        and abs(round((one[1] + three[1]) / 2) - top_y) <= 90
    ):
        raise VisionAgentError("numeric_grid 键盘锚点几何关系异常，拒绝猜坐标。")

    # The installed Douyin/Android numeric keyboard is a calibrated local
    # profile.  Qwen is only trusted to classify that a numeric grid is
    # visible; its returned coordinates may be in a resized-image coordinate
    # system.  Once the anchor geometry above proves the layout class, key
    # centers are calculated from the original camera frame locally.
    if frame_size is not None:
        width, height = frame_size
        aspect = width / height if height else 0.0
        if not (
            480 <= width <= 620
            and 900 <= height <= 1100
            and 0.48 <= aspect <= 0.58
        ):
            raise VisionAgentError(
                "当前画面尺寸不匹配已标定的数字键盘配置，拒绝猜坐标。"
            )
        column_ratios = (0.304, 0.500, 0.685)
        row_ratios = (0.665, 0.736, 0.802, 0.867)
        if text == "0":
            column, row = 1, 3
        else:
            number = int(text)
            column = (number - 1) % 3
            row = (number - 1) // 3
        return (
            int(round(width * column_ratios[column])),
            int(round(height * row_ratios[row])),
        )

    if text == "0":
        column, row = 1, 3
    else:
        number = int(text)
        column = (number - 1) % 3
        row = (number - 1) // 3
    return (
        int(round(left_x + column * x_step)),
        int(round(top_y + row * y_step)),
    )


DOUYIN_COMMENT_EMPTY_PLACEHOLDERS = (
    "爱评论的人，运气不会差",
    "爱评论的人,运气不会差",
    "说点什么",
    "留下你的精彩评论",
)


def is_douyin_comment_placeholder_text(value: str | None) -> bool:
    if not isinstance(value, str):
        return False
    normalized = re.sub(r"\s+", "", value)
    return any(
        re.sub(r"\s+", "", placeholder) == normalized
        for placeholder in DOUYIN_COMMENT_EMPTY_PLACEHOLDERS
    )


def decision_shows_empty_douyin_comment(decision: VisionDecision) -> bool:
    if decision.input_is_empty is True:
        return True
    evidence = "".join(
        value
        for value in (
            decision.target,
            decision.reason,
            decision.observed_input_text,
        )
        if isinstance(value, str)
    )
    return any(placeholder in evidence for placeholder in DOUYIN_COMMENT_EMPTY_PLACEHOLDERS)


def validate_decision(payload: dict[str, Any]) -> VisionDecision:
    action = str(payload.get("action", "")).strip()
    if action not in ALLOWED_ACTIONS:
        raise VisionAgentError(f"模型返回了未授权动作：{action or '(empty)'}")
    screen_type = str(payload.get("screen_type", "unknown")).strip()
    if screen_type not in ALLOWED_SCREEN_TYPES:
        screen_type = "unknown"
    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError) as exc:
        raise VisionAgentError("confidence 必须是 0～1 的数字。") from exc
    if not 0.0 <= confidence <= 1.0:
        raise VisionAgentError("confidence 超出 0～1。")

    # Models sometimes fill optional JSON fields even when an action does not
    # use them. Ignore those irrelevant fields instead of failing a safe task;
    # the action whitelist and the allowed_texts membership check still decide
    # what may actually reach the controller.
    coordinate = (
        _validate_coordinate(payload.get("coordinate"))
        if action in {"tap", "type_symbol"}
        else None
    )
    if action in {"tap", "type_symbol"} and coordinate is None:
        raise VisionAgentError(f"{action} 动作缺少 coordinate。")

    text = (
        payload.get("text")
        if action in {"type_symbol", "type_text", "type_pinyin"}
        else None
    )
    if action in {"type_symbol", "type_text", "type_pinyin"}:
        if text is not None and not isinstance(text, str):
            raise VisionAgentError("text 必须是字符串。")
        if not text:
            raise VisionAgentError(f"{action} 动作缺少 text。")
    if action == "type_symbol" and (
        not isinstance(text, str)
        or len(text) != 1
        or not text.isascii()
        or text.isspace()
        or not text.isprintable()
    ):
        raise VisionAgentError(
            "type_symbol 只允许点击一个画面中清晰可见的 ASCII 数字或符号键。"
        )
    if action == "type_text" and isinstance(text, str) and any(
        char.isascii()
        and char.isprintable()
        and not char.isalnum()
        and not char.isspace()
        for char in text
    ):
        raise VisionAgentError(
            "标点不能使用 type_text；必须切换键盘页并使用 type_symbol。"
        )
    pinyin = payload.get("pinyin") if action == "type_pinyin" else None
    if action == "type_pinyin":
        if not isinstance(pinyin, str) or not re.fullmatch(r"[a-z]{1,30}", pinyin):
            raise VisionAgentError(
                "type_pinyin 动作的 pinyin 必须是1～30个小写英文字母。"
            )
    observed_input_text = payload.get("observed_input_text")
    # Some vision-model responses serialize an inapplicable/empty observation
    # as "" instead of JSON null.  Treat only an actually empty string as
    # absent; non-empty text must still pass the strict visible-text checks
    # below.  This is especially important immediately after clear_text, when
    # the controller has verified an empty input and the next action is typing.
    if isinstance(observed_input_text, str) and not observed_input_text.strip():
        observed_input_text = None
    placeholder_was_misclassified = (
        action == "clear_text"
        and is_douyin_comment_placeholder_text(observed_input_text)
    )
    if placeholder_was_misclassified:
        # Douyin draws its empty-field hint where entered text would appear.
        # Convert this exact known placeholder into a non-mutating empty-field
        # observation before delete-count validation can issue physical taps.
        action = "wait"
        observed_input_text = None
    if observed_input_text is not None and (
        not isinstance(observed_input_text, str)
        or not observed_input_text
        or len(observed_input_text) > 100
        or "\n" in observed_input_text
        or "\r" in observed_input_text
    ):
        raise VisionAgentError(
            "observed_input_text 必须是逐字看清的1～100个可见字符。"
        )
    raw_delete_count = payload.get("delete_count") if action == "clear_text" else None
    delete_count: int | None = None
    if action == "clear_text":
        if (
            not isinstance(observed_input_text, str)
            or not observed_input_text
            or len(observed_input_text) > 100
            or "\n" in observed_input_text
            or "\r" in observed_input_text
        ):
            raise VisionAgentError(
                "clear_text 必须准确返回底部输入框内1～100个可见字符；"
                "为空、过长、换行或无法读清时必须 stop。"
            )
        if isinstance(raw_delete_count, bool) or not isinstance(raw_delete_count, int):
            raise VisionAgentError("clear_text 必须提供整数 delete_count。")
        expected_delete_count = exact_backspace_count(observed_input_text)
        if raw_delete_count != expected_delete_count:
            raise VisionAgentError(
                "clear_text 的 delete_count 与实际可编辑字符数不一致，"
                "已停止以避免误删。"
            )
        delete_count = raw_delete_count
    keyboard_layout = (
        _validate_keyboard_layout(payload.get("keyboard_layout"))
        if action in {"type_symbol", "type_pinyin", "clear_text"}
        else None
    )
    if action in {"type_pinyin", "clear_text"} and keyboard_layout is None:
        raise VisionAgentError(
            f"{action} 动作必须提供经过画面识别的 keyboard_layout。"
        )
    if action == "type_pinyin" and keyboard_layout.get("type") != "qwerty":
        raise VisionAgentError("type_pinyin 只允许使用 QWERTY keyboard_layout。")
    if (
        action == "type_symbol"
        and isinstance(text, str)
        and text.isdigit()
        and (
            keyboard_layout is None
            or keyboard_layout.get("type") != "numeric_grid"
        )
    ):
        raise VisionAgentError(
            "数字 type_symbol 必须提供 numeric_grid，并标出1、3、7、9和退格键中心。"
        )
    input_is_empty = (
        True if placeholder_was_misclassified else payload.get("input_is_empty")
    )
    if input_is_empty is not None and not isinstance(input_is_empty, bool):
        raise VisionAgentError("input_is_empty 必须是 true、false 或 null。")
    sent_message_visible = payload.get("sent_message_visible")
    if sent_message_visible is not None and not isinstance(
        sent_message_visible, bool
    ):
        raise VisionAgentError("sent_message_visible 必须是 true、false 或 null。")
    raw_wait_seconds = payload.get("wait_seconds", 1.0)
    if raw_wait_seconds is None:
        raw_wait_seconds = 1.0
    try:
        wait_seconds = float(raw_wait_seconds)
    except (TypeError, ValueError) as exc:
        raise VisionAgentError("wait_seconds 必须是数字。") from exc
    wait_seconds = max(0.5, min(wait_seconds, 5.0))

    raw_page_title = payload.get("page_title")
    if raw_page_title is not None and not isinstance(raw_page_title, str):
        raise VisionAgentError("page_title 必须是字符串或 null。")

    return VisionDecision(
        screen_type=screen_type,
        action=action,
        confidence=confidence,
        reason=(
            "抖音评论框显示空输入占位提示，按空框处理，禁止退格。"
            if placeholder_was_misclassified
            else str(payload.get("reason", "")).strip()[:300]
        ),
        page_title=(
            raw_page_title.strip()[:120]
            if raw_page_title is not None
            else None
        ),
        target=(
            "评论输入框为空"
            if placeholder_was_misclassified
            else str(payload.get("target", "")).strip()[:120]
        ),
        coordinate=coordinate,
        text=text,
        pinyin=pinyin,
        observed_input_text=observed_input_text,
        input_is_empty=input_is_empty,
        sent_message_visible=sent_message_visible,
        delete_count=delete_count,
        keyboard_layout=keyboard_layout,
        wait_seconds=wait_seconds,
        success=bool(payload.get("success", False)),
        plan_step_id=(
            str(payload.get("plan_step_id")).strip()[:40]
            if payload.get("plan_step_id") is not None
            else None
        ),
        checkpoint_met=bool(payload.get("checkpoint_met", False)),
    )


def _image_data_url(image: Image.Image) -> str:
    # Limit upload size and tokens while keeping mobile UI text readable.
    result = image.convert("RGB")
    if result.width > 720:
        height = int(round(result.height * 720 / result.width))
        result = result.resize((720, height), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    result.save(buffer, format="JPEG", quality=82, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _has_only_valid_inline_jpeg_images(messages: list[dict[str, Any]]) -> bool:
    """Return true only when every visual input is a valid inline JPEG.

    DashScope has occasionally returned its internal ``InvalidParameter`` URL
    error for an otherwise valid data URL.  Retrying that read-only request is
    safe only after the exact bytes have been validated locally; malformed or
    remote URLs must continue to fail closed without this exception.
    """

    found = False
    prefix = "data:image/jpeg;base64,"
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            found = True
            image_url = part.get("image_url")
            url = image_url.get("url") if isinstance(image_url, dict) else None
            if not isinstance(url, str) or not url.startswith(prefix):
                return False
            try:
                payload = base64.b64decode(url[len(prefix) :], validate=True)
            except (ValueError, binascii.Error):
                return False
            if len(payload) < 4 or not payload.startswith(b"\xff\xd8"):
                return False
            if not payload.endswith(b"\xff\xd9"):
                return False
    return found


def _is_retryable_dashscope_inline_url_rejection(
    response: httpx.Response,
    messages: list[dict[str, Any]],
) -> bool:
    if response.status_code != 400 or not _has_only_valid_inline_jpeg_images(messages):
        return False
    detail = response.text.lower()
    return (
        "internalerror.algo.invalidparameter" in detail
        and "provided url does not appear to be valid" in detail
    )


class DashScopeVisionProvider:
    """Small OpenAI-compatible adapter for Alibaba Model Studio."""

    TRANSIENT_HTTP_STATUS_CODES = {408, 429, 500, 502, 503, 504}

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        model_config: VisionModelConfig | None = None,
        enable_thinking: bool = False,
        timeout: float = 45.0,
        max_attempts: int = 3,
        retry_base_delay: float = 0.8,
    ) -> None:
        if model_config is not None and (model is not None or base_url is not None):
            raise ValueError("model_config 不能与 model/base_url 同时传入。")
        self.api_key = api_key if api_key is not None else os.getenv("DASHSCOPE_API_KEY", "")
        self.model_config = model_config or load_vision_model_config(
            model=model,
            base_url=base_url,
            enable_thinking=enable_thinking,
        )
        self.model = self.model_config.model
        self.base_url = self.model_config.base_url
        self.timeout = timeout
        self.max_attempts = max(1, int(max_attempts))
        self.retry_base_delay = max(0.0, float(retry_base_delay))
        self.last_usage: dict[str, Any] = {}
        self.usage_totals = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self.successful_call_count = 0
        self.last_request_id = ""
        self.last_network_attempts = 0
        self.last_finish_reason = ""
        self.last_response_model = ""

    @property
    def configured(self) -> bool:
        return bool(self.api_key.strip())

    def status(self) -> dict[str, Any]:
        return {
            "model_config_version": self.model_config.config_version,
            "provider": self.model_config.provider,
            "model": self.model,
            "thinking_enabled": self.model_config.enable_thinking,
            "coordinate_scale": self.model_config.coordinate_scale,
            "configured": self.configured,
            "base_url": self.base_url,
            "last_usage": self.last_usage,
            "usage_totals": dict(self.usage_totals),
            "successful_call_count": self.successful_call_count,
            "last_request_id": self.last_request_id,
            "last_network_attempts": self.last_network_attempts,
            "last_finish_reason": self.last_finish_reason,
            "response_model": self.last_response_model,
            "error": None if self.configured else "未配置 DASHSCOPE_API_KEY",
        }

    def _chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
        *,
        timeout: float | None = None,
        max_attempts: int | None = None,
    ) -> str:
        if not self.configured:
            raise VisionAgentError(
                "千问视觉尚未配置：请先设置 DASHSCOPE_API_KEY。"
            )
        self.last_usage = {}
        self.last_request_id = ""
        self.last_network_attempts = 0
        self.last_finish_reason = ""
        self.last_response_model = ""
        effective_timeout = self.timeout if timeout is None else max(1.0, float(timeout))
        effective_attempts = (
            self.max_attempts
            if max_attempts is None
            else max(1, int(max_attempts))
        )
        last_error: Exception | None = None
        for attempt in range(1, effective_attempts + 1):
            self.last_network_attempts = attempt
            try:
                response = httpx.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": 0.0,
                        "max_tokens": max_tokens,
                        **self.model_config.request_options(),
                    },
                    timeout=effective_timeout,
                )
                response.raise_for_status()
                payload = response.json()
                break
            except httpx.HTTPStatusError as exc:
                last_error = exc
                status_code = exc.response.status_code
                retryable_inline_rejection = (
                    attempt < effective_attempts
                    and _is_retryable_dashscope_inline_url_rejection(
                        exc.response,
                        messages,
                    )
                )
                if (
                    status_code not in self.TRANSIENT_HTTP_STATUS_CODES
                    and not retryable_inline_rejection
                ) or (
                    attempt >= effective_attempts
                ):
                    detail = exc.response.text[:500]
                    raise VisionAgentError(
                        f"千问视觉请求失败（HTTP {status_code}）：{detail}"
                    ) from exc
            except httpx.TimeoutException as exc:
                last_error = exc
                if attempt >= effective_attempts:
                    raise VisionAgentError(
                        f"千问视觉请求连续{attempt}次超时，未执行本轮动作。"
                    ) from exc
            except httpx.TransportError as exc:
                last_error = exc
                if attempt >= effective_attempts:
                    raise VisionAgentError(
                        f"千问视觉连接连续{attempt}次中断：{exc}"
                    ) from exc
            except ValueError as exc:
                raise VisionAgentError(
                    f"千问视觉响应不是有效 JSON：{exc}"
                ) from exc

            if self.retry_base_delay > 0:
                time.sleep(self.retry_base_delay * (2 ** (attempt - 1)))
        else:  # Defensive guard; each terminal failure above already raises.
            raise VisionAgentError(f"千问视觉连接失败：{last_error}")

        raw_usage = payload.get("usage") or {}
        self.last_usage = raw_usage if isinstance(raw_usage, dict) else {}
        normalized_usage: dict[str, int] = {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = self.last_usage.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                normalized_usage[key] = value
        if "total_tokens" not in normalized_usage:
            component_keys = ("prompt_tokens", "completion_tokens")
            if all(key in normalized_usage for key in component_keys):
                normalized_usage["total_tokens"] = sum(
                    normalized_usage[key] for key in component_keys
                )
        for key, value in normalized_usage.items():
            self.usage_totals[key] += value
        self.successful_call_count += 1
        self.last_request_id = str(payload.get("id") or "")
        self.last_response_model = str(payload.get("model") or "")
        try:
            choice = payload["choices"][0]
            self.last_finish_reason = str(choice.get("finish_reason") or "")
            content = choice["message"]["content"]
        except (AttributeError, KeyError, IndexError, TypeError) as exc:
            raise VisionAgentError("千问视觉响应缺少 message.content。") from exc
        if not isinstance(content, str) or not content.strip():
            raise VisionAgentError("千问视觉返回了空内容。")
        return content

    def parse_goal(self, text: str) -> dict[str, Any]:
        if goal_is_forbidden(text):
            return {
                "understood": False,
                "app_id": None,
                "operation": None,
                "params": {},
                "summary": "",
                "message": "该目标包含支付、账号安全、删除或批量关注等禁用操作。",
                "needs_confirmation": True,
                "provider": "local_safety_filter",
            }
        interaction_count = requested_social_interaction_count(text)
        if interaction_count > MAX_SOCIAL_INTERACTIONS_PER_TASK:
            return {
                "understood": False,
                "app_id": None,
                "operation": None,
                "params": {},
                "summary": "",
                "message": (
                    "单个任务最多允许"
                    f"{MAX_SOCIAL_INTERACTIONS_PER_TASK}次点赞或评论；"
                    "请拆成多个任务。"
                ),
                "needs_confirmation": True,
                "provider": "local_safety_filter",
            }
        prompt = f"""
你是手机机械臂任务入口。判断用户目标能否仅使用以下本地白名单动作完成：
tap（点屏幕）、type_symbol（点击画面中真实可见的单个标点键）、
type_text（输入用户提供的 ASCII 字母或数字原文）、
type_pinyin（用拼音键盘输入用户提供的中文原文）、swipe_up（上划）、
android_home、android_back、wait、finish、stop。

禁止接受：支付、下单、转账、删除数据、修改账号安全设置、绕过验证、
批量关注、任意系统命令。允许点赞和评论，包括用户明确指定1～50次的连续任务；
评论内容必须逐字来自用户原文。一次任务最多操作一个 App 和一个明确目标。

还必须生成冻结执行计划 execution_plan。它只能描述高层意图，不能包含坐标、
命令或模型临场发挥的动作。intent 只允许：open_app、open_target、focus_input、
clear_input、enter_text、submit、like_current、comment_current、next_item、
verify_result、navigate_back、navigate_home。每一步必须给出按顺序的 step_N、
简短 label、同一个 app_id、画面验收 checkpoint。app_id 必须是稳定的英文标识：
微信用wechat，抖音用douyin，设置用settings，安卓系统导航用android；禁止填中文。
enter_text/comment_current 用
text_ref 引用 allowed_texts 的下标，不能复制或改写文字。点赞/评论 count 必须与
用户原文一致；其他步骤 count 固定为1。用户明确不要发送时不得生成 submit。

用户原文：{text}

只输出一个 JSON 对象：
{{
  "understood": true或false,
  "summary": "给用户确认的简短中文摘要",
  "task_mode": "用户要求检查、验证、测试结果时为test，否则为operate",
  "expected_result": "test模式下需要从画面验证的结果，否则为null",
  "allowed_texts": ["按输入顺序列出所有要输入的原文片段；必须逐字来自用户原文；无需输入时为空数组"],
  "execution_plan": [
    {{
      "id": "step_1",
      "intent": "允许的高层意图之一",
      "label": "给用户看的步骤名称",
      "app_id": "单一App标识",
      "target": "目标控件或页面名称，无则null",
      "text_ref": "引用allowed_texts的整数下标，无则null",
      "count": 1,
      "checkpoint": "进入下一步前必须从画面确认的状态"
    }}
  ],
  "message": "不接受或需补充时的原因"
}}
""".strip()
        payload = _extract_json_object(
            self._chat(
                [
                    {
                        "role": "system",
                        "content": "你是严格、保守、只输出JSON的任务解析器。",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=900,
            )
        )
        understood = bool(payload.get("understood", False))
        summary = str(payload.get("summary", "")).strip()[:200]
        raw_allowed_texts = payload.get("allowed_texts")
        if raw_allowed_texts is None:
            # Accept the first-version response shape during a rolling upgrade.
            legacy_allowed_text = payload.get("allowed_text")
            raw_allowed_texts = (
                [legacy_allowed_text] if legacy_allowed_text is not None else []
            )
        if not isinstance(raw_allowed_texts, list):
            raise VisionAgentError("allowed_texts 必须是数组。")
        allowed_texts: list[str] = []
        for value in raw_allowed_texts:
            if not isinstance(value, str) or not value or len(value) > 100:
                raise VisionAgentError("待输入文字必须是1～100字的字符串。")
            if value not in text:
                raise VisionAgentError(
                    "模型提取的待输入文字不在用户原文中，已拒绝创建任务。"
                )
            if value not in allowed_texts:
                allowed_texts.append(value)
        if len(allowed_texts) > MAX_ALLOWED_TEXTS:
            raise VisionAgentError(
                f"单个任务最多允许{MAX_ALLOWED_TEXTS}段待输入文字。"
            )
        execution_plan = validate_execution_plan(
            payload.get("execution_plan"),
            goal=text,
            allowed_texts=allowed_texts,
        )
        if understood and not execution_plan:
            raise VisionAgentError("模型理解了任务但没有生成冻结执行计划。")

        task_mode = str(payload.get("task_mode", "operate")).strip().lower()
        if task_mode not in {"operate", "test"}:
            task_mode = "operate"
        expected_result_value = payload.get("expected_result")
        expected_result = (
            str(expected_result_value).strip()[:300]
            if expected_result_value is not None
            else None
        )
        if task_mode == "test" and not expected_result:
            expected_result = summary
        if task_mode == "operate":
            expected_result = None
        if understood and not summary:
            raise VisionAgentError("模型理解了任务但没有给出确认摘要。")
        message_value = payload.get("message")
        message = (
            message_value.strip()
            if isinstance(message_value, str)
            else ""
        )
        return {
            "understood": understood,
            "app_id": "agent" if understood else None,
            "operation": "agent.execute_goal" if understood else None,
            "params": (
                {
                    "goal": text,
                    "allowed_texts": allowed_texts,
                    "execution_plan": execution_plan,
                    "task_mode": task_mode,
                    "expected_result": expected_result,
                }
                if understood
                else {}
            ),
            "summary": summary,
            "message": message
            or ("已生成视觉 Agent 任务草稿。" if understood else "请明确操作目标。"),
            "needs_confirmation": True,
            "provider": self.model,
        }

    def decide(
        self,
        *,
        goal: str,
        frames: list[Image.Image],
        history: list[dict[str, Any]],
        allowed_texts: list[str],
        task_mode: str,
        expected_result: str | None,
    ) -> VisionDecision:
        if len(frames) < 4:
            raise VisionAgentError("视觉决策至少需要4帧观察。")
        recent_history = history[-8:]
        history_text = json.dumps(recent_history, ensure_ascii=False)
        controller_phase = next(
            (
                str(item.get("controller_phase"))
                for item in reversed(recent_history)
                if item.get("controller_phase")
            ),
            "",
        )
        controller_plan = next(
            (
                item.get("controller_plan")
                for item in reversed(recent_history)
                if isinstance(item.get("controller_plan"), dict)
            ),
            None,
        )
        phase_contracts = {
            "verify_empty_after_clear": (
                "当前是清空后的复核阶段。只允许识别底部输入框是否为空："
                "空则返回wait并填写input_is_empty=true；仍有真实文字则按准确字符数"
                "clear_text；看不清则wait或stop。禁止输入、发送和点击其他控件。"
            ),
            "awaiting_exact_pinyin_candidate": (
                "当前是拼音与候选词核对阶段。只允许tap准确候选词、clear_text、"
                "wait或stop；绝对禁止再次type_pinyin。"
            ),
            "verify_input_after_candidate": (
                "当前是候选词写入后的输入框复核阶段。只允许核对输入框后点击发送、"
                "clear_text、wait、finish或stop；绝对禁止再次输入或点击候选词。"
            ),
            "verify_message_after_send": (
                "当前是发送结果复核阶段。必须填写sent_message_visible和"
                "input_is_empty。两者都为true时返回finish且success=true；否则只允许"
                "wait、stop，或在历史明确允许补点且输入框仍逐字保留原文时tap同一"
                "发送按钮。绝对禁止type_pinyin、type_text、type_symbol、clear_text、"
                "滑动和导航。"
            ),
        }
        phase_contract = phase_contracts.get(controller_phase, "")
        prompt = f"""
你是通过物理机械臂操作真实手机的视觉控制器。四张图片按时间先后展示同一手机页面。
目标：{goal}
任务模式：{task_mode}
需要从画面验证的结果：{json.dumps(expected_result, ensure_ascii=False)}
允许输入的文字白名单：{json.dumps(allowed_texts, ensure_ascii=False)}
最近动作历史：{history_text}

当前控制器硬阶段：{controller_phase or "无"}
{phase_contract}
若存在控制器硬阶段，上述阶段契约优先于一般目标和动作说明；不得回到此前阶段。

当前冻结执行计划：{json.dumps(controller_plan, ensure_ascii=False)}
若存在冻结计划，只能处理 current_step。不得提前点击后续步骤的控件。
当当前画面已经满足 current_step.checkpoint 时，不要在同一轮继续下一个动作：
只能返回 wait（最后一步可返回finish），plan_step_id必须等于当前step id，
checkpoint_met=true。控制器会用本地规则复核证据后才推进下一步。
若尚未满足，则 checkpoint_met=false，并继续当前步骤允许的一个最小动作。
特别是 enter_text 后还有 submit 时，输入框逐字正确后先报告checkpoint完成，
禁止直接点击发送；下一轮进入submit步骤后才允许点击发送。

每次只能返回一个最小动作。屏幕坐标使用 0～1000 相对坐标：
左上角[0,0]，右下角[1000,1000]。不要输出像素坐标。

动作白名单：
- tap：仅在目标控件清晰、页面已稳定且置信度足够时使用，必须给coordinate。
  若打开应用图标或进入聊天会话的首次物理点击漏点，且连续清晰画面确认同一目标、
  同一位置仍未变化，可原位补点一次；补点后仍无变化必须stop。此规则不适用于发送、
  删除、确认、点赞、评论、关注、支付等动作。
- type_text：仅用于 ASCII 字母、数字和空格；输入框必须已聚焦、键盘可见，且 text 完全等于文字白名单中的一项。
  数字优先使用本地卖家输入动作，输入后必须重新观察输入框逐字确认；标点不能使用 type_text。
- type_symbol：仅用于一个 ASCII 标点。必须先从画面判断当前键盘页：
  若当前是 QWERTY 字母页，先用 tap 点击画面中真实可见的“123”“?123”“符”或等价数字/符号切换键，
  Q/W/E/R/T/Y/U/I/O/P 等大字母键帽角落的小数字或符号只是长按副标，不是当前可直接点击的独立键，
  即使副标看起来与目标字符相同也绝对禁止 type_symbol；
  下一轮重新观察；若目标字符仍不在当前页，只能点击画面中清晰可见且名称明确的数字/符号分类键，
  再次观察。只有在当前画面真实看见与 text 完全一致的键帽时，才返回 type_symbol，
  coordinate 必须是该键帽中心，text 必须是文字白名单中的单个字符。不可根据常见键盘布局猜坐标。
  点击后必须重新观察输入框并逐字确认；不一致时按真实字符数 clear_text，不能发送。
- type_pinyin：仅用于输入中文。必须从当前四帧确认画面是完整、稳定的标准 QWERTY 拼音键盘，text 必须完全等于文字白名单中的一项；
  pinyin 填该中文的连续小写无声调拼音，例如“你好”填“nihao”。
  每次 type_pinyin 必须同时返回 keyboard_layout：准确标出 q、p、a、l、z、m 六个字母键中心及 backspace 退格键中心。
  只识别这7个锚点；其余字母由本地代码计算，不要逐字返回坐标。若任何锚点被遮挡、键盘是九宫格/手写/悬浮布局、
  键盘正在移动、三行不是 QWERTYUIOP/ASDFGHJKL/ZXCVBNM，必须 stop，不能猜。
  本动作只点击拼音字母，不会选择候选词。
  下一步必须重新观察拼音输入区和候选栏。选择候选词的 tap 必须在 observed_input_text
  中逐字返回当前可见拼音；忽略输入法自动插入的空格或撇号后，它必须与刚输入的 pinyin 完全一致。
  如果少字母、多字母或无法逐字看清，绝对不能点击候选词，必须按实际可见字符数 clear_text 或 stop。
  只有拼音正确且候选栏里确实出现与 text 完全一致的中文候选词时，才允许点击一次；之后再次观察输入框逐字确认。
  候选词点击完成后进入“输入框验证阶段”：只检查聊天页底部、候选栏上方的输入框。
  候选栏可能仍然保留同一个词，这不代表候选词尚未点击；此阶段绝对不能再次点击候选词。
  如果输入框逐字等于 text：目标明确要求发送时，只能点击一次清晰可见的“发送”按钮；
  目标明确要求不要发送或只要求输入时立即 finish。输入框文字错误则 clear_text；画面不清则 wait 或 stop。
  点击发送后必须再次观察：只有看见新的本人发送气泡、气泡文字逐字等于 text、且输入框已经清空，才能 finish。
  如果输入框仍逐字保留 text 且同一个发送按钮仍在，可按控制器的补点规则补点一次；其他情况只能 wait 或 stop。
- clear_text：仅当键盘完整稳定、退格键清晰可见且输入框内已有错误文字时使用。它会按准确次数点击退格键；
  纯拼音组合中输入法自动显示的空格或撇号不占退格次数，例如n'nihao必须delete_count=6；其他文字仍逐字符计数。
  QWERTY键盘返回type=qwerty及q、p、a、l、z、m、backspace七个锚点；数字/符号等其他键盘返回
  type=generic且只标出当前画面真实可见的backspace锚点，不能沿用或猜测旧键盘坐标；
  只判断聊天页底部输入框内的文字，聊天记录中的气泡、时间和昵称都不属于输入框内容；
  必须把底部输入框内逐字看见的内容原样填入 observed_input_text，并把它的准确字符数填入 delete_count；
  本地程序只会按 delete_count 点击退格。两者不一致、文字被遮挡、包含换行、超过20个字符或不能准确计数时必须 stop；
  不要用 tap 猜测退格键坐标，不要输入新文字，必须在下一步先确认输入框已经清空。
  clear_text后的复核轮次，最后一张放大图只裁出“语音按钮与发送按钮之间的实际编辑框行”；
  它不包含编辑框上方仍可能保留旧词的输入法候选/联想栏。判断输入框文字或是否为空时
  必须只以这张窄ROI为准，完整画面中的聊天气泡、昵称、时间以及候选/联想词都不能作为输入框内容。
  如果确认输入框为空，只能返回wait、input_is_empty=true、target写“底部输入框为空”；
  不能在同一步直接重新输入。
  抖音评论框显示“爱评论的人，运气不会差”“说点什么”等灰色占位提示时，
  表示输入框为空，不是待删除的真实文字；此时必须返回input_is_empty=true，禁止clear_text。
- swipe_up / swipe_down / swipe_left / swipe_right：仅当目标明确要求浏览/寻找且页面允许该方向滑动。
- 连续点赞或评论：必须逐条处理。每条普通视频都要先稳定识别，视觉确认点赞变红或
  评论已经发布后才算完成当前一条。评论任务还必须先点击评论面板右上角或旁边清晰可见的
  X/关闭按钮，并在下一轮确认评论面板和键盘都已经消失，回到普通视频页，随后才能上划。
  评论面板或键盘仍在时严禁 swipe_up，否则滑动只会发生在评论列表中。最近动作历史里的
  controller_metrics 会给出上划次数和点击尝试次数，但点击尝试不等于成功，
  仍必须以当前画面复核。当前处理序号为 page_advances+1。达到用户指定数量后
  才能finish，不能提前结束。已点赞的视频视为该条已经完成，不得取消点赞。
  推荐流直播预览应直接上划跳过且不计数；全屏直播用返回键退出后再继续；
  广告、风险弹窗或无法分类的页面必须stop。
- android_home / android_back：仅用于安全导航。
- wait：页面加载或动画中。
- finish：画面已证明目标完成；test模式还必须从画面证明验收结果成立。
- stop：直播、广告、未知页、支付/下单/转账/账号安全页、风险弹窗、目标不明确、
  页面在四帧间发生结构变化、或无法安全确定下一步。连续任务中的直播按上一条规则
  安全跳过，不对直播执行点赞或评论。

不要猜隐藏控件，不要点击验证码，不要处理支付，不要自行改写待输入文字。
只输出一个 JSON 对象：
{{
  "screen_type": "android_home|app_page|chat_list|chat|video|live|dialog|keyboard|loading|unknown",
  "action": "tap|type_symbol|type_text|type_pinyin|clear_text|swipe_up|swipe_down|swipe_left|swipe_right|android_home|android_back|wait|finish|stop",
  "confidence": 0到1,
  "reason": "简短中文判断",
  "page_title": "逐字填写画面顶部当前页面标题；标题不可见或看不清时为null，禁止根据目标猜测",
  "target": "控件名称或页面状态",
  "coordinate": [x,y]或null,
  "text": "type_symbol填写单个目标字符；type_text或type_pinyin填写用户原文；否则null",
  "pinyin": "仅type_pinyin填写连续小写无声调拼音，否则null",
  "observed_input_text": "clear_text时填写底部输入框原文；选择拼音候选词的tap时填写当前可见拼音；发送补点tap时填写仍留在输入框的原文；其他动作null",
  "input_is_empty": "clear_text后的复核轮次，以及发送后验证轮次填写true或false；否则null",
  "sent_message_visible": "仅在发送后验证轮次填写true或false；必须真实看见新的本人消息气泡才为true，否则null",
  "delete_count": 仅clear_text填写observed_input_text的准确字符数，否则null,
  "keyboard_layout": {{"type":"qwerty","anchors":{{"q":[x,y],"p":[x,y],"a":[x,y],"l":[x,y],"z":[x,y],"m":[x,y],"backspace":[x,y]}}}}；数字type_symbol使用{{"type":"numeric_grid","anchors":{{"1":[x,y],"3":[x,y],"7":[x,y],"9":[x,y],"backspace":[x,y]}}}}；clear_text在非QWERTY键盘使用{{"type":"generic","anchors":{{"backspace":[x,y]}}}}；其他动作null,
  "wait_seconds": 0.5到5,
  "success": true或false,
  "plan_step_id": "存在冻结计划时填写当前step id，否则null",
  "checkpoint_met": "当前画面已满足当前step checkpoint时为true，否则false"
}}
""".strip()
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        timeline_frames = frames[:4]
        content.extend(
            {
                "type": "image_url",
                "image_url": {"url": _image_data_url(frame)},
            }
            for frame in timeline_frames
        )
        if len(frames) > 4:
            content.append(
                {
                    "type": "text",
                    "text": (
                        "最后一张是同一时刻聊天页底部实际编辑框行的窄幅放大图，"
                        "不包含其上方的输入法候选/联想栏。必须只以这张图逐字核对"
                        "编辑框文字以及是否为空；完整画面上方残留的候选词不能算作"
                        "输入框内容，不能根据目标文字猜测；是否出现新的本人消息气泡"
                        "仍以之前四张完整画面判断。"
                    ),
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _image_data_url(frames[-1])},
                }
            )
        messages = [
            {
                "role": "system",
                "content": "你是安全优先、每轮只执行一个动作、只输出JSON的手机视觉Agent。",
            },
            {"role": "user", "content": content},
        ]
        raw = self._chat(messages, max_tokens=650)
        try:
            return validate_decision(_extract_json_object(raw))
        except VisionAgentError as exc:
            # A missing tap coordinate is a malformed model response, not an
            # authorization to guess or reuse another coordinate.  Give the
            # model the same frozen observation one format-only correction
            # opportunity.  The repaired response still passes the complete
            # action/coordinate/confidence validator before any controller
            # action can run.
            if "动作缺少 coordinate" not in str(exc):
                raise
            repair_messages = [
                *messages,
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": (
                        "上一响应不合法：tap/type_symbol动作缺少coordinate。"
                        "请基于完全相同的图片和当前冻结步骤重新输出一个完整JSON。"
                        "如果目标控件清晰可见，coordinate必须填写该控件中心的"
                        "0～1000相对坐标；如果不能安全定位，只能返回wait或stop，"
                        "不得猜测、不得推进后续步骤。"
                    ),
                },
            ]
            repaired_raw = self._chat(repair_messages, max_tokens=650)
            return validate_decision(_extract_json_object(repaired_raw))


class ScriptedVisionProvider:
    """Deterministic provider used only by local tests."""

    def __init__(
        self,
        decisions: list[VisionDecision],
        parsed: dict[str, Any] | None = None,
    ) -> None:
        self.decisions = list(decisions)
        self.parsed = parsed

    def status(self) -> dict[str, Any]:
        return {
            "provider": "scripted_test",
            "model": "mock-vision-model",
            "configured": True,
            "last_usage": {},
            "last_request_id": "",
            "error": None,
        }

    def parse_goal(self, text: str) -> dict[str, Any]:
        if self.parsed is not None:
            return dict(self.parsed)
        return {
            "understood": True,
            "app_id": "agent",
            "operation": "agent.execute_goal",
            "params": {
                "goal": text,
                "allowed_texts": [],
                "task_mode": "operate",
                "expected_result": None,
            },
            "summary": text,
            "message": "mock",
            "needs_confirmation": True,
            "provider": "mock-vision-model",
        }

    def decide(
        self,
        *,
        goal: str,
        frames: list[Image.Image],
        history: list[dict[str, Any]],
        allowed_texts: list[str],
        task_mode: str,
        expected_result: str | None,
    ) -> VisionDecision:
        del goal, frames, history, allowed_texts, task_mode, expected_result
        if not self.decisions:
            raise VisionAgentError("模拟决策已用尽。")
        return self.decisions.pop(0)


class VisionAgentRunner:
    def __init__(
        self,
        controller: Any,
        provider: VisionProvider,
        *,
        min_confidence: float = 0.72,
        max_steps: int = 15,
        observation_frames: int = 4,
        observation_seconds: float = 1.5,
    ) -> None:
        self.controller = controller
        self.provider = provider
        self.min_confidence = min_confidence
        self.max_steps = max_steps
        self.observation_frames = max(4, observation_frames)
        self.observation_seconds = max(1.5, observation_seconds)

    def status(self) -> dict[str, Any]:
        value = dict(self.provider.status())
        value.update(
            {
                "min_confidence": self.min_confidence,
                "max_steps": self.max_steps,
                "observation_frames": self.observation_frames,
                "observation_seconds": self.observation_seconds,
            }
        )
        return value

    def _observe(self) -> list[Image.Image]:
        frames: list[Image.Image] = []
        interval = self.observation_seconds / (self.observation_frames - 1)
        # A browser preview may be reading the same seller window. Even with a
        # serialized capture lock, Windows can occasionally expose one
        # transient truncated frame while the seller UI repaints. Discard it,
        # wait, and collect a complete observation set; never send it to Qwen.
        deadline = time.monotonic() + self.observation_seconds + 5.0
        while len(frames) < self.observation_frames and time.monotonic() < deadline:
            self.controller._checkpoint()
            frame = self.controller.vision_capture()
            if frame.width < 400 or frame.height < 700:
                frames.clear()
                self.controller._sleep(0.5)
                continue
            frames.append(frame)
            if len(frames) < self.observation_frames:
                self.controller._sleep(interval)
        if len(frames) != self.observation_frames:
            raise VisionAgentError(
                "摄像头连续返回残缺画面，等待重采后仍未恢复，"
                "拒绝交给模型决策。"
            )
        return frames

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        goal = str(params.get("goal", "")).strip()
        if not goal or len(goal) > 500:
            raise VisionAgentError("视觉 Agent 目标长度必须在1～500字之间。")
        if goal_is_forbidden(goal):
            raise VisionAgentError("目标命中本地禁用操作规则，拒绝执行。")
        interaction_count = requested_social_interaction_count(goal)
        if interaction_count > MAX_SOCIAL_INTERACTIONS_PER_TASK:
            raise VisionAgentError(
                "单个任务最多允许"
                f"{MAX_SOCIAL_INTERACTIONS_PER_TASK}次点赞或评论。"
            )

        raw_allowed_texts = params.get("allowed_texts")
        if raw_allowed_texts is None:
            legacy_allowed_text = params.get("allowed_text")
            raw_allowed_texts = (
                [legacy_allowed_text] if legacy_allowed_text is not None else []
            )
        if not isinstance(raw_allowed_texts, list):
            raise VisionAgentError("允许输入文字白名单必须是数组。")
        allowed_texts: list[str] = []
        for value in raw_allowed_texts:
            if (
                not isinstance(value, str)
                or not value
                or len(value) > 100
                or value not in goal
            ):
                raise VisionAgentError("允许输入的文字必须逐字来自用户目标。")
            if value not in allowed_texts:
                allowed_texts.append(value)
        if len(allowed_texts) > MAX_ALLOWED_TEXTS:
            raise VisionAgentError(
                f"单个任务最多允许{MAX_ALLOWED_TEXTS}段待输入文字。"
            )
        execution_plan = validate_execution_plan(
            params.get("execution_plan"),
            goal=goal,
            allowed_texts=allowed_texts,
        )
        raw_input_sessions = params.get("input_sessions", [])
        if not isinstance(raw_input_sessions, list):
            raise VisionAgentError("input_sessions 必须是数组。")
        try:
            input_recovery = InputRecoveryCoordinator(
                raw_input_sessions,
                execution_plan,
            )
        except ValueError as exc:
            raise VisionAgentError(str(exc)) from exc

        task_mode = str(params.get("task_mode", "operate")).strip().lower()
        if task_mode not in {"operate", "test"}:
            raise VisionAgentError("task_mode 必须是 operate 或 test。")
        expected_result_value = params.get("expected_result")
        expected_result = (
            str(expected_result_value).strip()
            if expected_result_value is not None
            else None
        )
        if expected_result and len(expected_result) > 300:
            raise VisionAgentError("验收结果说明不能超过300个字符。")
        if task_mode == "test" and not expected_result:
            raise VisionAgentError("测试任务必须提供可从画面验证的预期结果。")

        status = self.provider.status()
        if not status.get("configured"):
            raise VisionAgentError(str(status.get("error") or "视觉模型未配置。"))

        run_dir = self.controller._new_run_dir("agent.execute_goal")
        history: list[dict[str, Any]] = []
        plan_cursor = 0
        evidence: list[str] = []
        previous_observation: Image.Image | None = None
        pinyin_candidate_pending: str | None = None
        pinyin_candidate_expected: str | None = None
        pinyin_keyboard_layout: dict[str, Any] | None = None
        pinyin_recovery_attempts = 0
        pinyin_input_verification: str | None = None
        pinyin_candidate_tap_coordinate: tuple[int, int] | None = None
        pinyin_candidate_tap_before: Image.Image | None = None
        pinyin_candidate_retry_used = False
        pinyin_send_verification: str | None = None
        pinyin_send_retry_used = False
        pinyin_send_observations = 0
        pinyin_send_before: Image.Image | None = None
        pinyin_send_post_empty: Image.Image | None = None
        pinyin_send_empty_observations = 0
        input_clear_verification: dict[str, Any] | None = None
        input_recovery_step_id: str | None = None
        input_recovery_restart_cursor: int | None = None
        navigation_retries_used: set[tuple[str, str]] = set()
        batch_plan_step = next(
            (
                item
                for item in execution_plan
                if item.get("intent") == "interact_batch"
            ),
            None,
        )
        batch_like_enabled = (
            batch_plan_step.get("like") is True
            if batch_plan_step
            else "点赞" in goal
        )
        batch_comment_enabled = (
            batch_plan_step.get("comment") is True
            if batch_plan_step
            else "评论" in goal
        )
        continuous_like_mode = interaction_count > 0 and batch_like_enabled
        continuous_comment_mode = interaction_count > 0 and batch_comment_enabled
        batch_comment_text = None
        batch_comment_segments: list[str] = []
        if batch_plan_step and continuous_comment_mode:
            batch_text_ref = batch_plan_step.get("text_ref")
            batch_segment_refs = batch_plan_step.get("comment_text_refs", [])
            if isinstance(batch_text_ref, int):
                batch_comment_text = allowed_texts[batch_text_ref]
            if isinstance(batch_segment_refs, list):
                batch_comment_segments = [
                    allowed_texts[ref]
                    for ref in batch_segment_refs
                    if isinstance(ref, int) and 0 <= ref < len(allowed_texts)
                ]
        batch_comment_state = (
            InputAttemptState(
                target_text=batch_comment_text,
                segments=list(batch_comment_segments),
            )
            if batch_comment_text and batch_comment_segments
            else None
        )
        batch_comment_segment_pending: str | None = None
        batch_comment_recovery_pending = False
        combined_page_like_done = False
        max_page_checks = (
            int(batch_plan_step.get("max_pages"))
            if batch_plan_step
            else min(15, interaction_count + 5)
        )
        comment_symbol_target = (
            allowed_texts[0]
            if (
                continuous_comment_mode
                and len(allowed_texts) == 1
                and len(allowed_texts[0]) == 1
                and allowed_texts[0].isascii()
                and allowed_texts[0].isprintable()
                and not allowed_texts[0].isspace()
                and not allowed_texts[0].isalpha()
            )
            else None
        )
        like_pending_verification = False
        like_page_tap_attempts = 0
        like_before_tap_metrics: dict[str, Any] | None = None
        like_tap_center_px: tuple[int, int] | None = None
        comment_input_pending: str | None = None
        comment_send_pending: str | None = None
        comment_close_verification = False
        symbol_keyboard_pending: str | None = None
        symbol_navigation_taps = 0
        comment_clear_verification: str | None = None
        comment_numeric_keyboard_ready = False
        comment_clear_keyboard_ready = False
        interaction_metrics = {
            "requested": interaction_count,
            "max_page_checks": max_page_checks,
            "pages_seen": 1 if interaction_count else 0,
            "page_advances": 0,
            "like_tap_attempts": 0,
            "completed_pages": 0,
            "verified_likes": 0,
            "skipped_already_liked": 0,
            "comment_submit_attempts": 0,
        }

        def controller_metrics_snapshot() -> dict[str, Any]:
            snapshot = dict(interaction_metrics)
            snapshot["input_recovery"] = input_recovery.metrics()
            if batch_comment_state is not None:
                snapshot["batch_comment_input"] = {
                    "segment_count": len(batch_comment_state.segments),
                    "completed_segments": len(
                        batch_comment_state.completed_segments
                    ),
                    "retypes_used": batch_comment_state.retypes_used,
                    "max_retypes": batch_comment_state.max_retypes,
                    "awaiting_empty_confirmation": (
                        batch_comment_state.awaiting_empty_confirmation
                    ),
                }
            return snapshot
        no_send_requested = any(
            token in goal
            for token in (
                "不要发送",
                "不发送",
                "别发送",
                "禁止发送",
                "不能发送",
                "无需发送",
                "只输入",
                "仅输入",
                "不要点击发送",
                "不点击发送",
                "别点击发送",
                "禁止点击发送",
                "不要按发送",
                "不按发送",
            )
        )
        send_requested = (
            any(
                token in goal
                for token in ("发送", "发给", "发消息", "发一条", "评论")
            )
            and not no_send_requested
        )
        clear_only_goal = (
            not allowed_texts
            and not send_requested
            and any(token in goal for token in ("清空", "删除", "退格"))
        )
        # A frozen plan may contain several deterministic input segments and
        # each segment needs observe -> act -> verify.  The old fixed limit of
        # 15 could stop a correct long-text workflow before its final verify.
        # Keep a hard ceiling, but size the budget from the confirmed plan and
        # requested social batch instead of allowing the model to run forever.
        step_limit = max(
            self.max_steps,
            min(600, len(execution_plan) * 6 + 10) if execution_plan else 0,
            min(240, interaction_count * 20 + 20)
            if interaction_count > 0
            else 0,
        )
        report_path = run_dir / "report.json"
        started_at = time.time()
        try:
            with self.controller.operation_lock:
                self.controller.clear_stop()
                for step in range(1, step_limit + 1):
                    self.controller._checkpoint()
                    send_phase_active_at_observe = (
                        pinyin_send_verification is not None
                    )
                    frames = self._observe()
                    before_path = run_dir / f"{step:02d}_observe.jpg"
                    self.controller._save_frame(
                        frames[-1],
                        before_path,
                        f"qwen observation step {step}",
                    )
                    evidence.append(str(before_path))
                    decision_history = list(history)
                    if execution_plan:
                        active_plan_step = execution_plan[plan_cursor]
                        decision_history.append(
                            {
                                "controller_plan": {
                                    "current_step": execution_plan[plan_cursor],
                                    "current_index": plan_cursor,
                                    "total_steps": len(execution_plan),
                                    "remaining_step_ids": [
                                        item["id"]
                                        for item in execution_plan[plan_cursor:]
                                    ],
                                },
                                "requirement": (
                                    "冻结计划由用户确认。一次只能处理current_step；"
                                    "不得提前执行后续步骤。画面已经满足checkpoint时，"
                                    "本轮只能返回wait或finish，并填写当前plan_step_id和"
                                    "checkpoint_met=true，让控制器先推进计划。"
                                ),
                            }
                        )
                        if active_plan_step.get("intent") == "enter_text":
                            decision_history.append(
                                {
                                    "controller_input_session": {
                                        "plan_step_id": active_plan_step["id"],
                                        "expected_input": active_plan_step.get(
                                            "expected_input"
                                        ),
                                        "requirement": (
                                            "只读取当前页面真实输入框ROI，observed_input_text"
                                            "必须返回输入框内从第一个字符到最后一个字符的"
                                            "完整可见文字，不能只返回刚输入的一段。若与"
                                            "expected_input不一致，只允许按完整可见文字的"
                                            "准确可编辑字符数clear_text；确认空框后控制器"
                                            "会回到本输入会话第一段重新输入。"
                                        ),
                                    }
                                }
                            )
                    if interaction_count:
                        decision_history.append(
                            {
                                "controller_metrics": controller_metrics_snapshot(),
                                "requirement": (
                                    "page_advances是已经离开的页面数，当前页序号为"
                                    "page_advances+1。tap_attempts只是点击尝试次数，"
                                    "不代表视觉验证成功。必须逐页从真实画面确认结果。"
                                ),
                            }
                        )
                    if input_clear_verification:
                        decision_history.append(
                            {
                                "controller_phase": "verify_empty_after_clear",
                                "cleared_text": input_clear_verification["text"],
                                "clear_only_goal": clear_only_goal,
                                "requirement": (
                                    "退格动作已经完成。最后一张放大图只包含底部输入区，"
                                    "必须以该图判断输入框，不得把上方聊天气泡当作输入。"
                                    "若输入框为空，返回wait、input_is_empty=true；若仍有"
                                    "真实可见字符，按当前准确字符数再次clear_text；看不清"
                                    "则wait或stop。控制器确认空框前禁止输入、发送、滑动"
                                    "或点击其他控件。"
                                ),
                            }
                        )
                    current_input_prefix = ""
                    if execution_plan:
                        active_step = execution_plan[plan_cursor]
                        if active_step.get("intent") == "enter_text":
                            active_ref = active_step.get("text_ref")
                            expected_value = str(
                                active_step.get("expected_input") or ""
                            )
                            if (
                                isinstance(active_ref, int)
                                and 0 <= active_ref < len(allowed_texts)
                                and expected_value.endswith(allowed_texts[active_ref])
                            ):
                                current_input_prefix = expected_value[
                                    : -len(allowed_texts[active_ref])
                                ]
                    if batch_comment_state is not None:
                        current_input_prefix = batch_comment_state.expected_prefix
                    if pinyin_candidate_pending:
                        decision_history.append(
                            {
                                "controller_phase": "awaiting_exact_pinyin_candidate",
                                "text": pinyin_candidate_pending,
                                "expected_visible_pinyin": pinyin_candidate_expected,
                                "requirement": (
                                    "先逐字读取输入框ROI的完整文字，并在"
                                    "observed_input_text返回原样文字。已完成前缀必须"
                                    f"逐字等于{current_input_prefix!r}，其后的当前可见"
                                    "拼音忽略空格和撇号后必须与"
                                    "expected_visible_pinyin完全一致；"
                                    "否则必须clear_text或stop，禁止点击候选词。"
                                    "拼音正确时只允许点击一次与原文完全一致的候选词；"
                                    "点击后必须进入输入框验证阶段。"
                                ),
                            }
                        )
                    elif pinyin_input_verification:
                        decision_history.append(
                            {
                                "controller_phase": "verify_input_after_candidate",
                                "text": pinyin_input_verification,
                                "requirement": (
                                    "候选词已经点击。只检查聊天页底部、候选栏上方的"
                                    "输入框；候选栏仍显示同词不代表需要再次点击。"
                                    + (
                                        "本任务明确要求发送。输入框逐字正确时，只允许"
                                        "点击一次清晰可见的发送按钮；错误则clear_text，"
                                        "不清楚则wait或stop。禁止点击候选词或其他控件。"
                                        if send_requested
                                        else
                                        "本任务不允许发送。输入框逐字正确则finish，"
                                        "错误则clear_text，不清楚则wait或stop；禁止tap。"
                                    )
                                ),
                            }
                        )
                    elif pinyin_send_verification:
                        decision_history.append(
                            {
                                "controller_phase": "verify_message_after_send",
                                "text": pinyin_send_verification,
                                "completed_observations": pinyin_send_observations,
                                "requirement": (
                                    "发送按钮已经点击一次。只观察是否出现新的本人发送"
                                    "气泡，并确认输入框已经清空；两项都满足时必须返回"
                                    "sent_message_visible=true、input_is_empty=true才finish。"
                                    "如果连续清晰画面仍显示底部输入框逐字保留完全相同的"
                                    "文字、同一个发送按钮仍可见，说明物理触控可能漏点；"
                                    "此时只允许对同一发送按钮补点一次，并且必须在"
                                    "observed_input_text中逐字返回当前输入框文字。"
                                    "除此之外只能wait或stop，禁止点击其他控件。"
                                    + (
                                        "补点机会已经使用，禁止任何tap，只能wait、"
                                        "finish或stop。"
                                        if pinyin_send_retry_used
                                        else
                                        "补点机会尚未使用。"
                                    )
                                ),
                            }
                        )
                    if comment_clear_verification:
                        decision_history.append(
                            {
                                "controller_phase": "verify_comment_input_empty",
                                "text": comment_clear_verification,
                                "requirement": (
                                    "刚才已经按视觉确认的字符数点击退格。现在只能检查"
                                    "评论编辑框本身：若输入框确实完全为空，返回wait、"
                                    "input_is_empty=true、target=评论输入框为空；"
                                    "若仍有可见字符，按当前真实字符数再次clear_text；"
                                    "看不清则wait或stop。确认空框之前禁止输入、发送、"
                                    "关闭评论区或上划。"
                                ),
                            }
                        )
                    elif comment_input_pending:
                        decision_history.append(
                            {
                                "controller_phase": "verify_comment_input",
                                "text": comment_input_pending,
                                "requirement": (
                                    "只检查评论编辑框内的可见文字，必须逐字返回"
                                    "observed_input_text。只有它与text完全一致时才允许"
                                    "点击红色发送按钮；若不同，必须按真实字符数"
                                    "clear_text；看不清则wait或stop。禁止上划。"
                                ),
                            }
                        )
                    elif (
                        continuous_comment_mode
                        and batch_comment_state is not None
                        and batch_comment_state.completed_segments
                        != batch_comment_state.segments
                    ):
                        next_index = len(batch_comment_state.completed_segments)
                        next_segment = batch_comment_state.segments[next_index]
                        decision_history.append(
                            {
                                "controller_phase": "enter_next_comment_segment",
                                "text": next_segment,
                                "expected_input_prefix": batch_comment_state.expected_prefix,
                                "segment_index": next_index + 1,
                                "segment_count": len(batch_comment_state.segments),
                                "requirement": (
                                    "先确认当前已在评论编辑框并且键盘完整清晰。"
                                    "只允许输入本段text，不允许一次输入完整评论或点击发送。"
                                    "中文使用type_pinyin；英文数字使用type_text；标点先"
                                    "切换到真实符号键盘后使用type_symbol。每段完成后必须"
                                    "逐字返回整个评论输入框文字，控制器确认完整前缀后才会"
                                    "放行下一段。任何错字、漏字或多字都必须clear_text。"
                                ),
                            }
                        )
                    elif symbol_keyboard_pending:
                        decision_history.append(
                            {
                                "controller_phase": "locate_exact_symbol_key",
                                "text": symbol_keyboard_pending,
                                "requirement": (
                                    (
                                        "当前仍处于首次数字/符号键盘导航阶段。"
                                        "如果画面还显示Q、W、E等QWERTY大写字母键，"
                                        "键帽角落的小数字/符号只是长按副标，绝不是可直接"
                                        "点击的目标键；本轮绝对禁止type_symbol。必须先tap"
                                        "画面底部清晰可见的123、?123、数字或符号切换键，"
                                        "然后重新观察。"
                                        if symbol_navigation_taps == 0
                                        else
                                        (
                                            "错误字符已经删除，且数字键盘页面没有改变。"
                                            "现在必须基于当前新画面重新定位目标数字键，"
                                            "不能沿用删除前的旧坐标。"
                                            if comment_numeric_keyboard_ready
                                            else
                                            "已经实际点击过至少一次数字/符号键盘切换键。"
                                        )
                                        +
                                        "先确认当前画面不再是QWERTY字母主页；只有真实看见"
                                        "与text完全一致、以大号字符作为主标签的独立键帽时，"
                                        "才允许type_symbol并点击键帽中心。若字符不在当前页，"
                                        "只允许tap清晰可见的数字/符号分类切换键后重新观察。"
                                    )
                                    + "严禁把Q/W/E等字母键上的角标当成数字键，"
                                    "严禁长按字母键、猜固定坐标、发送或上划。"
                                ),
                            }
                        )
                    elif comment_send_pending:
                        decision_history.append(
                            {
                                "controller_phase": "verify_comment_and_close_panel",
                                "text": comment_send_pending,
                                "requirement": (
                                    "发送按钮已经点击。先在评论列表中确认出现一条本人"
                                    "刚发布、文字逐字等于text的评论；确认后必须点击"
                                    "评论面板右上角或旁边清晰可见的X/关闭按钮，并在"
                                    "observed_input_text逐字返回刚发布的评论。"
                                    "未确认完整评论时只能wait或stop。禁止上划。"
                                ),
                            }
                        )
                    elif comment_close_verification:
                        decision_history.append(
                            {
                                "controller_phase": "verify_comment_panel_closed",
                                "requirement": (
                                    "刚才已经点击评论面板的X/关闭按钮。必须确认评论"
                                    "面板和键盘都已消失、当前是普通视频页。若仍看到"
                                    "评论列表、评论输入框或键盘，只能wait或stop；"
                                    "确认关闭后由本地状态机决定完成或上划。"
                                ),
                            }
                        )
                    decision_frames = frames
                    if pinyin_candidate_pending:
                        # Candidate selection needs the composition and candidate
                        # strip. It deliberately uses a different crop from the
                        # actual chat input field.
                        zoom = enlarged_vertical_roi(
                            frames[-1],
                            top_ratio=WECHAT_CANDIDATE_ROI_TOP,
                            bottom_ratio=WECHAT_CANDIDATE_ROI_BOTTOM,
                        )
                        decision_frames = [*frames, zoom]
                    elif (
                        input_clear_verification
                        or pinyin_input_verification
                        or pinyin_send_verification
                        or (
                            execution_plan
                            and 0 <= plan_cursor < len(execution_plan)
                            and execution_plan[plan_cursor].get("app_id") == "wechat"
                            and execution_plan[plan_cursor].get("intent")
                            == "verify_result"
                        )
                    ):
                        # Only the actual WeChat bottom input row is enlarged.
                        # Candidate words below it and chat bubbles above it must
                        # never be interpreted as input-field contents.
                        zoom = enlarged_vertical_roi(
                            frames[-1],
                            top_ratio=WECHAT_INPUT_ROI_TOP,
                            bottom_ratio=WECHAT_INPUT_ROI_BOTTOM,
                        )
                        decision_frames = [*frames, zoom]
                    elif symbol_keyboard_pending:
                        source = frames[-1]
                        top = max(0, int(round(source.height * 0.47)))
                        zoom = source.crop((0, top, source.width, source.height))
                        zoom = zoom.resize(
                            (zoom.width * 2, zoom.height * 2),
                            Image.Resampling.LANCZOS,
                        )
                        decision_frames = [*frames, zoom]
                    elif comment_clear_verification or comment_input_pending:
                        source = frames[-1]
                        top = max(0, int(round(source.height * 0.22)))
                        bottom = min(
                            source.height,
                            max(top + 1, int(round(source.height * 0.58))),
                        )
                        zoom = source.crop((0, top, source.width, bottom))
                        zoom = zoom.resize(
                            (zoom.width * 2, zoom.height * 2),
                            Image.Resampling.LANCZOS,
                        )
                        decision_frames = [*frames, zoom]
                    try:
                        decision = self.provider.decide(
                            goal=goal,
                            frames=decision_frames,
                            history=decision_history,
                            allowed_texts=allowed_texts,
                            task_mode=task_mode,
                            expected_result=expected_result,
                        )
                    except VisionAgentError as exc:
                        # The model is not allowed to invent a second candidate
                        # coordinate.  If the first, fully validated candidate
                        # tap produced almost no change in the fixed bottom
                        # input ROI and the model then emits a malformed tap,
                        # the controller may re-use that exact coordinate once.
                        # This is a physical missed-tap recovery, not a relaxed
                        # visual decision rule.
                        can_retry_candidate = (
                            pinyin_input_verification is not None
                            and pinyin_candidate_tap_coordinate is not None
                            and pinyin_candidate_tap_before is not None
                            and not pinyin_candidate_retry_used
                            and "coordinate" in str(exc)
                        )
                        candidate_roi_delta = float("inf")
                        if can_retry_candidate:
                            before_roi = enlarged_vertical_roi(
                                pinyin_candidate_tap_before,
                                top_ratio=WECHAT_CANDIDATE_ROI_TOP,
                                bottom_ratio=WECHAT_CANDIDATE_ROI_BOTTOM,
                                scale=1,
                            )
                            current_roi = enlarged_vertical_roi(
                                frames[-1],
                                top_ratio=WECHAT_CANDIDATE_ROI_TOP,
                                bottom_ratio=WECHAT_CANDIDATE_ROI_BOTTOM,
                                scale=1,
                            )
                            candidate_roi_delta = frame_mean_delta(
                                before_roi,
                                current_roi,
                            )
                        if (
                            can_retry_candidate
                            and candidate_roi_delta
                            < PINYIN_CANDIDATE_MISSED_TAP_ROI_DELTA
                        ):
                            retry_coordinate = pinyin_candidate_tap_coordinate
                            item = {
                                "step": step,
                                "screen_type": "chat",
                                "action": "tap",
                                "confidence": 1.0,
                                "reason": (
                                    "控制器检测到候选区画面未变化，复用首次已验证"
                                    "坐标补点一次"
                                ),
                                "target": (
                                    f"候选词{pinyin_input_verification}（控制器补点）"
                                ),
                                "coordinate": list(retry_coordinate),
                                "success": True,
                                "controller_owned": True,
                                "roi_delta": round(candidate_roi_delta, 4),
                            }
                            history.append(item)
                            self._write_json(
                                run_dir / f"{step:02d}_decision.json",
                                item,
                            )
                            self.controller.vision_tap_relative(
                                retry_coordinate[0],
                                retry_coordinate[1],
                            )
                            pinyin_candidate_retry_used = True
                            pinyin_candidate_tap_before = frames[-1].copy()
                            self.controller._sleep(3.0)
                            previous_observation = frames[-1].copy()
                            continue
                        raise
                    if (
                        continuous_like_mode
                        and decision.screen_type == "video"
                        and not (
                            continuous_comment_mode and combined_page_like_done
                        )
                    ):
                        located_heart = locate_douyin_heart(frames[-1])
                        center_value = located_heart.get("center_px")
                        current_center_px = (
                            (int(center_value[0]), int(center_value[1]))
                            if isinstance(center_value, list)
                            and len(center_value) == 2
                            else None
                        )
                        measurement_center = (
                            like_tap_center_px
                            if like_pending_verification
                            else current_center_px
                        )
                        heart = (
                            classify_douyin_heart_roi(
                                frames[-1],
                                measurement_center,
                            )
                            if measurement_center
                            else {
                                "state": "unknown",
                                "center_px": None,
                                "center": None,
                            }
                        )
                        heart["detector"] = located_heart
                        interaction_metrics["last_heart"] = heart
                        if like_pending_verification:
                            heart_transition = douyin_heart_became_liked(
                                like_before_tap_metrics,
                                heart,
                            )
                            interaction_metrics["last_heart_transition"] = (
                                heart_transition
                            )
                            if heart_transition:
                                interaction_metrics["verified_likes"] += 1
                                like_pending_verification = False
                                like_page_tap_attempts = 0
                                like_before_tap_metrics = None
                                like_tap_center_px = None
                                if continuous_comment_mode:
                                    combined_page_like_done = True
                                    decision = VisionDecision(
                                        screen_type="video",
                                        action="wait",
                                        confidence=1.0,
                                        reason=(
                                            "本地红心检测已确认当前页点赞成功；"
                                            "同页还需完成评论，禁止提前上划。"
                                        ),
                                        target="当前视频评论入口",
                                        wait_seconds=0.8,
                                    )
                                else:
                                    interaction_metrics["completed_pages"] += 1
                                if (
                                    not continuous_comment_mode
                                    and
                                    interaction_metrics["completed_pages"]
                                    >= interaction_count
                                ):
                                    decision = VisionDecision(
                                        screen_type="video",
                                        action="finish",
                                        confidence=1.0,
                                        reason=(
                                            "本地红心检测已确认当前页点赞成功，"
                                            "请求页数已经处理完成。"
                                        ),
                                        target="连续点赞任务完成",
                                        success=True,
                                    )
                                elif not continuous_comment_mode:
                                    decision = VisionDecision(
                                        screen_type="video",
                                        action="swipe_up",
                                        confidence=1.0,
                                        reason=(
                                            "本地红心检测已确认点赞成功，"
                                            "直接上划进入下一页。"
                                        ),
                                        target=(
                                            "下一条视频（已处理"
                                            f"{interaction_metrics['completed_pages']}页）"
                                        ),
                                     )
                            elif (
                                1 <= like_page_tap_attempts < 3
                                and like_tap_center_px is not None
                                and current_center_px is not None
                                and located_heart.get("trusted")
                                and located_heart.get("state") == "unliked"
                                and abs(
                                    current_center_px[0] - like_tap_center_px[0]
                                )
                                <= 10
                                and abs(
                                    current_center_px[1] - like_tap_center_px[1]
                                )
                                <= 10
                                and located_heart.get("center") is not None
                            ):
                                decision = VisionDecision(
                                    screen_type="video",
                                    action="tap",
                                    confidence=1.0,
                                    reason=(
                                        "上一次物理触碰后，同一位置仍是白心；"
                                        "重新动态定位的中心与上次相差不超过10像素，"
                                        "允许受控补点（每页最多点击三次）。"
                                    ),
                                    target="动态定位的点赞按钮（受控补点）",
                                    coordinate=tuple(located_heart["center"]),
                                )
                            else:
                                decision = VisionDecision(
                                    screen_type="video",
                                    action="stop",
                                    confidence=1.0,
                                    reason=(
                                        "点赞后未能在刚才点击的小区域确认由白变红，"
                                        "且不满足受控补点条件或补点后仍失败，立即停止；"
                                        "禁止继续盲点，以免误开评论或取消点赞。"
                                    ),
                                    target="点赞状态无法安全确认",
                                )
                        elif (
                            located_heart.get("trusted")
                            and located_heart.get("state") == "liked"
                        ):
                            interaction_metrics["skipped_already_liked"] += 1
                            if continuous_comment_mode:
                                combined_page_like_done = True
                                decision = VisionDecision(
                                    screen_type="video",
                                    action="wait",
                                    confidence=1.0,
                                    reason=(
                                        "本地检测到当前页已经是红心，不重复点赞；"
                                        "同页继续完成评论。"
                                    ),
                                    target="当前视频评论入口",
                                    wait_seconds=0.8,
                                )
                            else:
                                interaction_metrics["completed_pages"] += 1
                            if (
                                not continuous_comment_mode
                                and
                                interaction_metrics["completed_pages"]
                                >= interaction_count
                            ):
                                decision = VisionDecision(
                                    screen_type="video",
                                    action="finish",
                                    confidence=1.0,
                                    reason=(
                                        "当前页已经是红心，按已处理页面计数；"
                                        "请求页数已经完成。"
                                    ),
                                    target="连续点赞任务完成",
                                    success=True,
                                )
                            elif not continuous_comment_mode:
                                decision = VisionDecision(
                                    screen_type="video",
                                    action="swipe_up",
                                    confidence=1.0,
                                    reason=(
                                        "本地检测到当前页已经点赞，"
                                        "不重复点击，直接上划下一页。"
                                    ),
                                    target=(
                                        "下一条视频（已处理"
                                        f"{interaction_metrics['completed_pages']}页）"
                                    ),
                                )
                        elif (
                            located_heart.get("trusted")
                            and located_heart.get("state") == "unliked"
                            and located_heart.get("center") is not None
                        ):
                            decision = VisionDecision(
                                screen_type="video",
                                action="tap",
                                confidence=1.0,
                                reason=(
                                    "本地动态定位确认当前为白心，"
                                    "只点击一次并在下一轮验证红心。"
                                ),
                                target="动态定位的点赞按钮",
                                coordinate=tuple(located_heart["center"]),
                            )
                        else:
                            decision = VisionDecision(
                                screen_type="video",
                                action="stop",
                                confidence=1.0,
                                reason=(
                                    "未能在当前布局中可靠定位一个形状可信的爱心，"
                                    "停止且不使用固定坐标猜测。"
                                ),
                                target="点赞状态未知",
                            )
                    if continuous_comment_mode and comment_clear_verification:
                        if (
                            decision.action == "wait"
                            and decision_shows_empty_douyin_comment(decision)
                        ):
                            if (
                                batch_comment_recovery_pending
                                and batch_comment_state is not None
                            ):
                                try:
                                    batch_comment_state.confirm_empty(True)
                                except ValueError as exc:
                                    raise VisionAgentError(str(exc)) from exc
                                batch_comment_recovery_pending = False
                                batch_comment_segment_pending = None
                                comment_clear_keyboard_ready = False
                                comment_numeric_keyboard_ready = False
                                comment_clear_verification = None
                                symbol_keyboard_pending = None
                                symbol_navigation_taps = 0
                                decision = VisionDecision(
                                    screen_type=decision.screen_type,
                                    action="wait",
                                    confidence=decision.confidence,
                                    reason=(
                                        "控制器已确认评论输入框完全为空；"
                                        "下一轮从评论第一个分段重新输入。"
                                    ),
                                    target="评论输入框为空",
                                    input_is_empty=True,
                                    wait_seconds=max(0.8, decision.wait_seconds),
                                )
                            else:
                            # The recovery path is intentionally split across
                            # observations: delete first, prove the editor is
                            # empty, then relocalize the symbol. Backspace does
                            # not change the keyboard page, so preserve a
                            # previously verified numeric grid instead of
                            # needlessly pressing 123 again.
                                symbol_keyboard_pending = comment_clear_verification
                                symbol_navigation_taps = (
                                    1 if comment_clear_keyboard_ready else 0
                                )
                                comment_numeric_keyboard_ready = (
                                    comment_clear_keyboard_ready
                                )
                                comment_clear_keyboard_ready = False
                                comment_clear_verification = None
                        elif decision.action == "clear_text":
                            pass
                        elif decision.action not in {"wait", "stop"}:
                            raise VisionAgentError(
                                f"第{step}步退格后尚未确认评论输入框为空，"
                                f"禁止执行 {decision.action}。"
                            )
                    if continuous_comment_mode and symbol_keyboard_pending:
                        if decision.action == "type_symbol":
                            x, y = decision.coordinate or (-1, -1)
                            if (
                                symbol_navigation_taps < 1
                                or
                                decision.text != symbol_keyboard_pending
                                or y < 500
                            ):
                                raise VisionAgentError(
                                    f"第{step}步尚未实际切换到数字/符号键盘，或未在"
                                    f"当前键盘页可靠定位真实的“{symbol_keyboard_pending}”"
                                    "主键帽；QWERTY字母键角标不能点击，已拒绝操作。"
                                )
                        elif decision.action == "tap":
                            x, y = decision.coordinate or (-1, -1)
                            target_text = decision.target or ""
                            if not (
                                any(
                                    token in target_text
                                    for token in (
                                        "123",
                                        "?123",
                                        "数字",
                                        "符号",
                                        "常用",
                                        "英文",
                                    )
                                )
                                and y >= 500
                                and symbol_navigation_taps < 3
                            ):
                                raise VisionAgentError(
                                    f"第{step}步正在寻找目标字符"
                                    f"“{symbol_keyboard_pending}”，只允许切换"
                                    "明确可见的数字/符号键盘分类。"
                                )
                        elif decision.action not in {"wait", "stop"}:
                            raise VisionAgentError(
                                f"第{step}步尚未找到真实的"
                                f"“{symbol_keyboard_pending}”键，禁止执行"
                                f" {decision.action}。"
                            )
                    elif continuous_comment_mode and comment_input_pending:
                        if (
                            decision.action in {"wait", "stop"}
                            and comment_numeric_keyboard_ready
                            and isinstance(decision.observed_input_text, str)
                            and 1 <= len(decision.observed_input_text) <= 100
                            and decision.observed_input_text
                            != comment_input_pending
                            and decision.input_is_empty is False
                        ):
                            # This keyboard page was already verified locally.
                            # Delete exactly the visible wrong character count;
                            # do not let the model guess another recovery action.
                            decision = VisionDecision(
                                screen_type="keyboard",
                                action="clear_text",
                                confidence=max(
                                    self.min_confidence,
                                    decision.confidence,
                                ),
                                reason=(
                                    "控制器已确认当前为已标定数字/符号键盘，"
                                    "按视觉逐字识别到的实际错误字符数清空。"
                                ),
                                target="已标定数字/符号键盘退格键",
                                observed_input_text=decision.observed_input_text,
                                delete_count=len(decision.observed_input_text),
                                keyboard_layout={
                                    "type": "generic",
                                    "anchors": {"backspace": [862, 844]},
                                },
                            )
                        if decision.action == "tap":
                            x, y = decision.coordinate or (-1, -1)
                            if not (
                                any(
                                    token in (decision.target or "")
                                    for token in ("发送", "发布")
                                )
                                and decision.observed_input_text
                                == comment_input_pending
                                and 700 <= x <= 980
                                and 250 <= y <= 750
                            ):
                                raise VisionAgentError(
                                    f"第{step}步评论输入后未逐字确认完整内容，"
                                    "或目标不是右侧发送按钮，已拒绝点击。"
                                )
                        elif decision.action not in {"clear_text", "wait", "stop"}:
                            raise VisionAgentError(
                                f"第{step}步评论输入尚未核对完成，禁止执行"
                                f" {decision.action}。"
                            )
                    elif continuous_comment_mode and comment_send_pending:
                        if decision.action == "tap":
                            x, y = decision.coordinate or (-1, -1)
                            target_text = decision.target or ""
                            if not (
                                decision.observed_input_text
                                == comment_send_pending
                                and any(
                                    token in target_text
                                    for token in ("关闭", "X", "x", "评论面板")
                                )
                                and 740 <= x <= 990
                                and 120 <= y <= 560
                            ):
                                raise VisionAgentError(
                                    f"第{step}步发送评论后，尚未同时确认完整评论"
                                    "并定位评论面板关闭按钮，已拒绝点击。"
                                )
                        elif decision.action not in {"wait", "stop"}:
                            raise VisionAgentError(
                                f"第{step}步评论面板尚未关闭，禁止执行"
                                f" {decision.action}。"
                            )
                    elif continuous_comment_mode and comment_close_verification:
                        closed_text = f"{decision.target} {decision.reason}"
                        panel_confirmed_closed = (
                            decision.screen_type == "video"
                            and any(
                                token in closed_text
                                for token in ("评论面板已关闭", "评论区已关闭", "评论面板消失")
                            )
                            and decision.action in {"swipe_up", "finish"}
                        )
                        if panel_confirmed_closed:
                            if continuous_like_mode and not combined_page_like_done:
                                raise VisionAgentError(
                                    f"第{step}步评论面板虽已关闭，但当前页点赞尚未"
                                    "得到本地红心验证，禁止把该页计为完成。"
                                )
                            completed = interaction_metrics["completed_pages"] + 1
                            interaction_metrics["completed_pages"] = completed
                            if completed >= interaction_count:
                                decision = VisionDecision(
                                    screen_type="video",
                                    action="finish",
                                    confidence=1.0,
                                    reason=(
                                        "评论内容已核对、评论面板已关闭，"
                                        "请求的视频数量已经处理完成。"
                                    ),
                                    target="连续评论任务完成",
                                    success=True,
                                )
                            else:
                                decision = VisionDecision(
                                    screen_type="video",
                                    action="swipe_up",
                                    confidence=1.0,
                                    reason=(
                                        "评论内容已核对且评论面板已关闭，"
                                        "现在才允许上划下一条视频。"
                                    ),
                                    target=(
                                        "下一条视频（已处理"
                                        f"{completed}页）"
                                    ),
                                )
                        elif decision.action not in {"wait", "stop"}:
                            raise VisionAgentError(
                                f"第{step}步未确认评论面板和键盘均已消失，"
                                f"禁止执行 {decision.action}。"
                            )

                    # Text entry is a controller-owned state machine.  The
                    # model may classify the current pixels, but it cannot skip
                    # phases or choose an action that belongs to another phase.
                    if input_clear_verification:
                        if decision.action == "clear_text":
                            pass
                        elif (
                            decision.action in {"wait", "finish"}
                            and decision.input_is_empty is True
                        ):
                            input_clear_verification = None
                            if input_recovery_step_id is not None:
                                try:
                                    restart_cursor = input_recovery.confirm_empty(
                                        input_recovery_step_id
                                    )
                                except ValueError as exc:
                                    raise VisionAgentError(str(exc)) from exc
                                plan_cursor = (
                                    input_recovery_restart_cursor
                                    if input_recovery_restart_cursor is not None
                                    else restart_cursor
                                )
                                input_recovery_step_id = None
                                input_recovery_restart_cursor = None
                                pinyin_candidate_pending = None
                                pinyin_candidate_expected = None
                                pinyin_keyboard_layout = None
                                pinyin_input_verification = None
                                decision = VisionDecision(
                                    screen_type=decision.screen_type,
                                    action="wait",
                                    confidence=decision.confidence,
                                    reason=(
                                        "控制器已确认本次尝试输入的全部文字均已清空；"
                                        "冻结计划退回该输入会话第一段，下一轮从第一个"
                                        "字符重新输入。"
                                    ),
                                    target=execution_plan[plan_cursor]["label"],
                                    input_is_empty=True,
                                    wait_seconds=max(0.8, decision.wait_seconds),
                                    plan_step_id=execution_plan[plan_cursor]["id"],
                                )
                            elif clear_only_goal:
                                decision = VisionDecision(
                                    screen_type=decision.screen_type,
                                    action="finish",
                                    confidence=decision.confidence,
                                    reason=(
                                        "本地清空阶段确认底部输入框为空，"
                                        "清空专用任务完成。"
                                    ),
                                    target="底部输入框为空",
                                    input_is_empty=True,
                                    success=True,
                                )
                            elif decision.action == "finish":
                                decision = VisionDecision(
                                    screen_type=decision.screen_type,
                                    action="wait",
                                    confidence=decision.confidence,
                                    reason=(
                                        "本地清空阶段已确认输入框为空；"
                                        "下一轮才允许重新输入。"
                                    ),
                                    target="底部输入框为空",
                                    input_is_empty=True,
                                    wait_seconds=0.5,
                                )
                        elif decision.action not in {"wait", "stop"}:
                            raise VisionAgentError(
                                f"第{step}步退格后尚未确认底部输入框为空，"
                                f"禁止执行 {decision.action}。"
                            )

                    if (
                        batch_comment_state is not None
                        and batch_comment_segment_pending is not None
                        and not batch_comment_recovery_pending
                    ):
                        expected_comment_input = (
                            batch_comment_state.expected_prefix
                            + batch_comment_segment_pending
                        )
                        observed_comment_input = decision.observed_input_text or ""
                        if observed_comment_input == expected_comment_input:
                            if not batch_comment_state.accept_segment(
                                batch_comment_segment_pending,
                                observed_comment_input,
                            ):
                                raise VisionAgentError(
                                    "评论分段虽然通过画面读取，但控制器前缀校验失败。"
                                )
                            batch_comment_segment_pending = None
                            pinyin_input_verification = None
                            pinyin_candidate_pending = None
                            pinyin_candidate_expected = None
                            if (
                                batch_comment_state.completed_segments
                                == batch_comment_state.segments
                            ):
                                comment_input_pending = batch_comment_state.target_text
                            else:
                                decision = VisionDecision(
                                    screen_type=decision.screen_type,
                                    action="wait",
                                    confidence=decision.confidence,
                                    reason=(
                                        "控制器已逐字确认当前评论分段和完整前缀；"
                                        "下一轮才允许输入下一分段。"
                                    ),
                                    target="评论下一分段",
                                    observed_input_text=observed_comment_input,
                                    input_is_empty=False,
                                    wait_seconds=max(0.5, decision.wait_seconds),
                                )
                        elif decision.action == "clear_text":
                            expected_delete_count = editable_character_count(
                                observed_comment_input
                            )
                            if (
                                expected_delete_count < 1
                                or decision.delete_count != expected_delete_count
                            ):
                                raise VisionAgentError(
                                    "评论输入错误，但模型退格次数与输入框实际全文字符数"
                                    "不一致，已停止。"
                                )
                        elif decision.action not in {"wait", "stop"}:
                            raise VisionAgentError(
                                "评论当前分段尚未逐字验证，禁止发送或执行其他动作。"
                            )

                    if pinyin_candidate_pending and decision.action not in {
                        "tap",
                        "clear_text",
                        "wait",
                        "stop",
                    }:
                        raise VisionAgentError(
                            f"第{step}步正在核对拼音和候选词，"
                            f"禁止执行 {decision.action}。"
                        )

                    if pinyin_input_verification:
                        allowed_input_actions = (
                            {"tap", "clear_text", "wait", "stop"}
                            if send_requested
                            else {"finish", "clear_text", "wait", "stop"}
                        )
                        if decision.action not in allowed_input_actions:
                            raise VisionAgentError(
                                f"第{step}步正在核对候选词写入后的底部输入框，"
                                f"禁止执行 {decision.action}。"
                            )

                    if pinyin_send_verification:
                        if (
                            decision.action in {"wait", "finish"}
                            and decision.input_is_empty is True
                            and decision.sent_message_visible is not True
                            and pinyin_send_before is not None
                        ):
                            transition = sent_message_transition_metrics(
                                pinyin_send_before,
                                frames[-1],
                            )
                            if transition["transition_visible"]:
                                pinyin_send_empty_observations += 1
                                if pinyin_send_post_empty is None:
                                    pinyin_send_post_empty = frames[-1].copy()
                                    decision = VisionDecision(
                                        screen_type=decision.screen_type,
                                        action="wait",
                                        confidence=decision.confidence,
                                        reason=(
                                            "控制器已检测到发送前后输入区和消息区同时变化，"
                                            "再采集一轮稳定空输入框画面后完成"
                                        ),
                                        target="发送结果稳定复核",
                                        input_is_empty=True,
                                        wait_seconds=max(0.8, decision.wait_seconds),
                                    )
                                else:
                                    stable_delta = frame_mean_delta(
                                        pinyin_send_post_empty,
                                        frames[-1],
                                    )
                                    if stable_delta <= SEND_POST_STABLE_MAX_DELTA:
                                        decision = VisionDecision(
                                            screen_type=decision.screen_type,
                                            action="finish",
                                            confidence=max(
                                                self.min_confidence,
                                                decision.confidence,
                                            ),
                                            reason=(
                                                "控制器比较发送前后画面：底部输入框已清空、"
                                                "右侧消息区出现新增变化，且发送后画面再次观察"
                                                "保持稳定，确认本次发送完成"
                                            ),
                                            target=(
                                                f"新的本人消息气泡：{pinyin_send_verification}"
                                            ),
                                            input_is_empty=True,
                                            sent_message_visible=True,
                                            success=True,
                                        )
                        if decision.action not in {"tap", "wait", "finish", "stop"}:
                            raise VisionAgentError(
                                f"第{step}步正在验证发送结果，"
                                f"禁止执行 {decision.action}。"
                            )
                        if decision.action == "tap" and pinyin_send_observations < 1:
                            decision = VisionDecision(
                                screen_type=decision.screen_type,
                                action="wait",
                                confidence=decision.confidence,
                                reason=(
                                    "发送后的首次观察禁止立即补点；"
                                    "先等待下一组稳定画面。"
                                ),
                                target="发送后首次观察",
                                wait_seconds=max(1.0, decision.wait_seconds),
                            )
                        if decision.action == "finish" and not (
                            decision.success
                            and decision.sent_message_visible is True
                            and decision.input_is_empty is True
                        ):
                            raise VisionAgentError(
                                f"第{step}步尚未同时确认新的本人消息气泡和空输入框，"
                                "禁止结束发送任务。"
                            )
                    completed_plan_step: dict[str, Any] | None = None
                    fast_forwarded_plan_steps: list[dict[str, Any]] = []
                    if execution_plan:
                        current_plan_step = execution_plan[plan_cursor]
                        current_step_id = current_plan_step["id"]
                        resume_cursor = infer_safe_plan_resume_cursor(
                            execution_plan,
                            plan_cursor,
                            decision,
                            allowed_texts,
                        )
                        if resume_cursor > plan_cursor:
                            proposed_action = decision.action
                            fast_forwarded_plan_steps = [
                                dict(item)
                                for item in execution_plan[plan_cursor:resume_cursor]
                            ]
                            plan_cursor = resume_cursor
                            current_plan_step = execution_plan[plan_cursor]
                            current_step_id = current_plan_step["id"]
                            decision = replace(
                                decision,
                                reason=(
                                    "控制器根据同一清晰画面确认App已恢复到更深页面；"
                                    f"安全快进至 {current_step_id}。原判断："
                                    f"{decision.reason}"
                                ),
                                plan_step_id=current_step_id,
                            )
                            resumed_action_intent = {
                                "clear_text": "clear_input",
                                "type_text": "enter_text",
                                "type_pinyin": "enter_text",
                            }.get(proposed_action)
                            if (
                                resumed_action_intent is not None
                                and current_plan_step["intent"]
                                != resumed_action_intent
                            ):
                                decision = replace(
                                    decision,
                                    action="wait",
                                    reason=(
                                        "控制器已恢复到后续命名页面，但必须先单独"
                                        f"完成 {current_step_id} 的聚焦/空输入框验证；"
                                        f"本轮不执行提前提出的 {proposed_action}。"
                                    ),
                                    target=current_plan_step["label"],
                                    coordinate=None,
                                    text=None,
                                    pinyin=None,
                                    wait_seconds=max(0.8, decision.wait_seconds),
                                    success=False,
                                    checkpoint_met=False,
                                )
                        if decision.plan_step_id not in {None, current_step_id}:
                            raise VisionAgentError(
                                f"第{step}步模型试图处理 {decision.plan_step_id}，"
                                f"但冻结计划当前只允许 {current_step_id}。"
                            )
                        allowed_plan_actions = PLAN_INTENT_ACTIONS[
                            current_plan_step["intent"]
                        ]
                        checkpoint_completion_action = (
                            decision.checkpoint_met
                            and decision.action in {"wait", "finish"}
                        )
                        if (
                            decision.action not in allowed_plan_actions
                            and not checkpoint_completion_action
                        ):
                            history.append(
                                {
                                    "step": step,
                                    **decision.to_dict(),
                                    "blocked_by": "frozen_plan_action_gate",
                                    "allowed_plan_actions": sorted(
                                        allowed_plan_actions
                                    ),
                                }
                            )
                            raise VisionAgentError(
                                f"第{step}步冻结计划当前意图为"
                                f" {current_plan_step['intent']}，禁止执行"
                                f" {decision.action}。"
                            )
                        if (
                            current_plan_step["intent"] == "enter_text"
                            and decision.action == "tap"
                            and any(
                                token in (decision.target or "")
                                for token in ("发送", "发布", "确认")
                            )
                        ):
                            raise VisionAgentError(
                                f"第{step}步仍处于 enter_text，必须先逐字确认输入框"
                                "并推进冻结计划，禁止提前点击发送。"
                            )
                        if decision.checkpoint_met:
                            if decision.plan_step_id != current_step_id:
                                raise VisionAgentError(
                                    f"第{step}步声明checkpoint完成但没有返回当前"
                                    f" plan_step_id={current_step_id}。"
                                )
                            if decision.action not in {"wait", "finish"}:
                                raise VisionAgentError(
                                    f"第{step}步完成计划checkpoint时不能同时执行"
                                    f" {decision.action}。"
                                )
                            if not plan_checkpoint_is_proven(
                                current_plan_step,
                                decision,
                                allowed_texts,
                            ):
                                raise VisionAgentError(
                                    f"第{step}步冻结计划checkpoint缺少本地要求的"
                                    "画面证据，拒绝推进。"
                                )
                            if current_plan_step["intent"] == "enter_text":
                                text_ref = current_plan_step.get("text_ref")
                                segment = (
                                    allowed_texts[text_ref]
                                    if isinstance(text_ref, int)
                                    and 0 <= text_ref < len(allowed_texts)
                                    else ""
                                )
                                state = input_recovery.state_for_step(
                                    current_step_id
                                )
                                if state is not None and not state.accept_segment(
                                    segment,
                                    decision.observed_input_text or "",
                                ):
                                    raise VisionAgentError(
                                        f"第{step}步输入会话内部前缀校验失败，"
                                        "拒绝推进冻结计划。"
                                    )
                                pinyin_candidate_pending = None
                                pinyin_candidate_expected = None
                                pinyin_keyboard_layout = None
                                pinyin_input_verification = None
                                pinyin_candidate_tap_coordinate = None
                                pinyin_candidate_tap_before = None
                                pinyin_candidate_retry_used = False
                            completed_plan_step = dict(current_plan_step)
                            if plan_cursor < len(execution_plan) - 1:
                                plan_cursor += 1
                                decision = VisionDecision(
                                    screen_type=decision.screen_type,
                                    action="wait",
                                    confidence=decision.confidence,
                                    reason=(
                                        f"控制器确认 {current_step_id} 的checkpoint，"
                                        f"冻结计划推进到 {execution_plan[plan_cursor]['id']}"
                                    ),
                                    target=execution_plan[plan_cursor]["label"],
                                    wait_seconds=max(0.5, decision.wait_seconds),
                                    plan_step_id=execution_plan[plan_cursor]["id"],
                                )
                            else:
                                decision = VisionDecision(
                                    screen_type=decision.screen_type,
                                    action="finish",
                                    confidence=decision.confidence,
                                    reason=(
                                        f"控制器确认冻结计划最后一步 {current_step_id}"
                                    ),
                                    target=current_plan_step["label"],
                                    success=True,
                                    plan_step_id=current_step_id,
                                    checkpoint_met=True,
                                )
                        elif (
                            decision.action == "finish"
                            and plan_cursor < len(execution_plan) - 1
                        ):
                            raise VisionAgentError(
                                f"第{step}步冻结计划仍有未执行步骤，禁止提前finish。"
                            )
                    item = {"step": step, **decision.to_dict()}
                    if fast_forwarded_plan_steps:
                        item["fast_forwarded_plan_steps"] = fast_forwarded_plan_steps
                        item["plan_cursor_after_fast_forward"] = plan_cursor
                    if completed_plan_step is not None:
                        item["completed_plan_step"] = completed_plan_step
                        item["plan_cursor_after"] = plan_cursor
                    if continuous_like_mode and decision.screen_type == "video":
                        item["local_heart"] = interaction_metrics.get("last_heart")
                    history.append(item)
                    self._write_json(run_dir / f"{step:02d}_decision.json", item)

                    unchanged_repeat = (
                        previous_observation is not None
                        and len(history) >= 2
                        and not pinyin_send_verification
                        and decision.action
                        not in {"wait", "finish", "stop"}
                        and not (
                            continuous_like_mode
                            and like_pending_verification
                            and decision.action == "tap"
                            and 1 <= like_page_tap_attempts < 3
                        )
                        and repeated_action_matches(history[-2], item)
                        and frame_mean_delta(previous_observation, frames[-1])
                        < UNCHANGED_FRAME_MEAN_DELTA
                    )
                    if unchanged_repeat:
                        navigation_retry_key = safe_navigation_retry_key(item)
                        if (
                            navigation_retry_key is not None
                            and navigation_retry_key not in navigation_retries_used
                        ):
                            navigation_retries_used.add(navigation_retry_key)
                        else:
                            raise VisionAgentError(
                                f"第{step}步连续请求相同动作 {decision.action}，"
                                "但画面几乎未变化，已停止以避免重复操作。"
                            )

                    if decision.confidence < self.min_confidence:
                        raise VisionAgentError(
                            f"第{step}步模型置信度 {decision.confidence:.2f} "
                            f"低于安全阈值 {self.min_confidence:.2f}，已停止。"
                        )
                    if decision.screen_type == "unknown" and decision.action not in {
                        "wait",
                        "stop",
                    }:
                        raise VisionAgentError(
                            f"第{step}步页面未知，却请求 {decision.action}，已拒绝。"
                        )
                    if decision.screen_type == "live" and decision.action not in {
                        "wait",
                        "android_back",
                        "swipe_up",
                        "stop",
                    }:
                        raise VisionAgentError(
                            f"第{step}步识别为直播，却请求 {decision.action}，已拒绝。"
                        )
                    if pinyin_candidate_pending and decision.action == "tap":
                        observed = decision.observed_input_text or ""
                        expected = pinyin_candidate_expected or ""
                        target_matches = pinyin_candidate_pending in (
                            decision.target or ""
                        )
                        observed_has_prefix = observed.startswith(
                            current_input_prefix
                        )
                        observed_suffix = (
                            observed[len(current_input_prefix) :]
                            if observed_has_prefix
                            else observed
                        )
                        observed_canonical = canonical_visible_pinyin(
                            observed_suffix
                        )
                        expected_canonical = canonical_visible_pinyin(expected)
                        if (
                            not observed_has_prefix
                            or observed_canonical != expected_canonical
                        ):
                            current_plan_step_id = (
                                execution_plan[plan_cursor]["id"]
                                if execution_plan
                                else ""
                            )
                            has_recovery_session = bool(
                                execution_plan
                                and input_recovery.state_for_step(
                                    current_plan_step_id
                                )
                                is not None
                            )
                            can_recover = (
                                (has_recovery_session or pinyin_recovery_attempts < 2)
                                and pinyin_keyboard_layout is not None
                                and bool(observed)
                                and 1 <= editable_character_count(observed) <= 100
                                and (
                                    not execution_plan
                                    or input_recovery.state_for_step(
                                        current_plan_step_id
                                    )
                                    is not None
                                )
                            )
                            if not can_recover:
                                raise VisionAgentError(
                                    f"第{step}步拼音逐字核对不一致，且无法安全自动"
                                    "清空恢复，已拒绝点击候选词。"
                                )
                            if not has_recovery_session:
                                pinyin_recovery_attempts += 1
                            delete_count = editable_character_count(observed)
                            if execution_plan:
                                try:
                                    (
                                        delete_count,
                                        input_recovery_restart_cursor,
                                    ) = input_recovery.begin_recovery(
                                        current_plan_step_id,
                                        observed,
                                    )
                                    input_recovery_step_id = current_plan_step_id
                                except ValueError as exc:
                                    raise VisionAgentError(str(exc)) from exc
                            decision = VisionDecision(
                                screen_type=decision.screen_type,
                                action="clear_text",
                                confidence=decision.confidence,
                                reason=(
                                    "控制器逐字核对发现可见拼音与预期不一致；"
                                    "按实际可编辑字母数精确退格，随后必须确认空框、"
                                    "重新定位键盘再输入。"
                                ),
                                target="当前QWERTY键盘退格键",
                                observed_input_text=observed,
                                delete_count=delete_count,
                                keyboard_layout=pinyin_keyboard_layout,
                            )
                        elif not target_matches:
                            raise VisionAgentError(
                                f"第{step}步拼音正确，但目标不是与原文完全一致的"
                                "候选词，已拒绝点击。"
                            )
                    if pinyin_send_verification and decision.action == "tap":
                        x, y = decision.coordinate or (-1, -1)
                        is_safe_send_retry = (
                            not pinyin_send_retry_used
                            and send_requested
                            and "发送" in (decision.target or "")
                            and decision.observed_input_text
                            == pinyin_send_verification
                            and 700 <= x <= 980
                            and 350 <= y <= 750
                        )
                        if not is_safe_send_retry:
                            raise VisionAgentError(
                                f"第{step}步已点击发送按钮，只有在输入框仍逐字保留"
                                "同一文字时才允许对同一发送按钮补点一次；"
                                "当前请求不满足安全补点条件，已停止。"
                            )
                    if pinyin_input_verification and decision.action == "tap":
                        x, y = decision.coordinate or (-1, -1)
                        is_safe_send_tap = (
                            send_requested
                            and "发送" in (decision.target or "")
                            and 700 <= x <= 980
                            and 350 <= y <= 750
                        )
                        if not is_safe_send_tap:
                            raise VisionAgentError(
                                f"第{step}步已进入候选词选择后的输入框验证阶段，"
                                "只允许在任务明确要求发送时点击右侧发送按钮；"
                                "候选词或其他控件均禁止再次点击。"
                            )

                    if decision.action == "stop":
                        raise VisionAgentError(
                            f"千问视觉主动停止：{decision.reason or decision.target}"
                        )
                    if decision.action == "finish":
                        if not decision.success:
                            raise VisionAgentError("模型返回 finish 但未确认 success。")
                        if pinyin_input_verification and send_requested:
                            raise VisionAgentError(
                                "输入框文字虽已确认，但任务要求的发送动作尚未执行。"
                            )
                        outcome = "passed" if task_mode == "test" else "completed"
                        report = {
                            "schema_version": 1,
                            "outcome": outcome,
                            "task_mode": task_mode,
                            "goal": goal,
                            "expected_result": expected_result,
                            "execution_plan": execution_plan,
                            "plan_cursor": plan_cursor,
                            "duration_seconds": round(time.time() - started_at, 3),
                            "steps": history,
                            "evidence": evidence,
                            "controller_metrics": controller_metrics_snapshot(),
                            "agent": self.provider.status(),
                            "error": None,
                        }
                        self._write_json(report_path, report)
                        return {
                            "changed": True,
                            "agent": self.provider.status(),
                            "goal": goal,
                            "task_mode": task_mode,
                            "expected_result": expected_result,
                            "execution_plan": execution_plan,
                            "plan_cursor": plan_cursor,
                            "outcome": outcome,
                            "steps": history,
                            "evidence": evidence,
                            "controller_metrics": controller_metrics_snapshot(),
                            "run_dir": str(run_dir),
                            "report": str(report_path),
                        }
                    if decision.action == "tap":
                        assert decision.coordinate is not None
                        if pinyin_candidate_pending:
                            pinyin_candidate_tap_coordinate = (
                                int(decision.coordinate[0]),
                                int(decision.coordinate[1]),
                            )
                            pinyin_candidate_tap_before = frames[-1].copy()
                            pinyin_candidate_retry_used = False
                        if pinyin_input_verification:
                            pinyin_send_before = frames[-1].copy()
                            pinyin_send_post_empty = None
                            pinyin_send_empty_observations = 0
                        self.controller.vision_tap_relative(
                            decision.coordinate[0],
                            decision.coordinate[1],
                        )
                        target_text = decision.target or ""
                        is_symbol_navigation_tap = (
                            continuous_comment_mode
                            and comment_symbol_target is not None
                            and decision.coordinate[1] >= 500
                            and any(
                                token in target_text
                                for token in (
                                    "123",
                                    "?123",
                                    "数字",
                                    "符号",
                                    "常用",
                                    "英文",
                                )
                            )
                        )
                        if is_symbol_navigation_tap:
                            symbol_keyboard_pending = comment_symbol_target
                            symbol_navigation_taps += 1
                            comment_numeric_keyboard_ready = False
                        is_comment_symbol_input_focus_tap = (
                            continuous_comment_mode
                            and comment_symbol_target is not None
                            and not comment_symbol_target.isdigit()
                            and any(
                                token in target_text
                                for token in (
                                    "评论输入框",
                                    "评论编辑框",
                                    "发表评论输入框",
                                    "输入评论",
                                    "说说你的感受",
                                )
                            )
                        )
                        if is_comment_symbol_input_focus_tap:
                            # Once the editor has been focused, enter a strict
                            # two-stage state: first navigate away from QWERTY,
                            # then locate the large primary symbol key. This
                            # prevents Q/W/E corner legends from being mistaken
                            # for directly clickable digits.
                            symbol_keyboard_pending = comment_symbol_target
                            symbol_navigation_taps = 0
                        if any(token in target_text for token in ("爱心", "点赞")):
                            interaction_metrics["like_tap_attempts"] += 1
                            if continuous_like_mode:
                                like_page_tap_attempts += 1
                                like_pending_verification = True
                                like_before_tap_metrics = dict(
                                    interaction_metrics.get("last_heart") or {}
                                )
                                center_value = like_before_tap_metrics.get(
                                    "center_px"
                                )
                                like_tap_center_px = (
                                    (
                                        int(center_value[0]),
                                        int(center_value[1]),
                                    )
                                    if isinstance(center_value, list)
                                    and len(center_value) == 2
                                    else None
                                )
                        if (
                            "评论" in goal
                            and any(token in target_text for token in ("发送", "发布"))
                        ):
                            interaction_metrics["comment_submit_attempts"] += 1
                        is_comment_send_tap = (
                            continuous_comment_mode
                            and any(token in target_text for token in ("发送", "发布"))
                            and (
                                comment_input_pending is not None
                                or (
                                    batch_comment_state is None
                                    and pinyin_input_verification is not None
                                )
                            )
                        )
                        is_comment_close_tap = (
                            continuous_comment_mode
                            and comment_send_pending is not None
                            and any(
                                token in target_text
                                for token in ("关闭", "X", "x", "评论面板")
                            )
                        )
                        if is_comment_send_tap:
                            comment_send_pending = (
                                comment_input_pending or pinyin_input_verification
                            )
                            comment_input_pending = None
                            pinyin_input_verification = None
                            pinyin_send_verification = None
                        elif is_comment_close_tap:
                            comment_send_pending = None
                            comment_close_verification = True
                        elif pinyin_candidate_pending:
                            selected_segment = pinyin_candidate_pending
                            pinyin_input_verification = selected_segment
                            if batch_comment_state is not None:
                                batch_comment_segment_pending = selected_segment
                            pinyin_candidate_pending = None
                            pinyin_candidate_expected = None
                        elif pinyin_input_verification:
                            pinyin_send_verification = pinyin_input_verification
                            pinyin_input_verification = None
                            pinyin_send_retry_used = False
                            pinyin_send_observations = 0
                        elif pinyin_send_verification:
                            pinyin_send_retry_used = True
                    elif decision.action == "type_text":
                        if not decision.text or decision.text not in allowed_texts:
                            raise VisionAgentError(
                                "模型请求输入的内容与用户确认文字白名单不完全一致，已拒绝。"
                            )
                        if any(
                            char.isascii()
                            and char.isprintable()
                            and not char.isalnum()
                            and not char.isspace()
                            for char in decision.text
                        ):
                            raise VisionAgentError(
                                "标点不能使用 type_text；必须先切换键盘页，"
                                "再用 type_symbol 点击画面中真实可见的目标键。"
                            )
                        if batch_comment_state is not None:
                            next_index = len(batch_comment_state.completed_segments)
                            if (
                                next_index >= len(batch_comment_state.segments)
                                or decision.text
                                != batch_comment_state.segments[next_index]
                            ):
                                raise VisionAgentError(
                                    "批量评论只能按冻结计划输入当前精确分段。"
                                )
                        self.controller.vision_type_text(decision.text)
                        if batch_comment_state is not None:
                            batch_comment_segment_pending = decision.text
                            pinyin_input_verification = decision.text
                        elif continuous_comment_mode:
                            comment_input_pending = decision.text
                    elif decision.action == "type_symbol":
                        if (
                            not decision.text
                            or decision.text not in allowed_texts
                            or decision.coordinate is None
                            or symbol_keyboard_pending != decision.text
                            or symbol_navigation_taps < 1
                        ):
                            raise VisionAgentError(
                                "type_symbol 的字符未获用户确认，或尚未完成"
                                "至少一次真实的数字/符号键盘切换，已拒绝。"
                            )
                        if batch_comment_state is not None:
                            next_index = len(batch_comment_state.completed_segments)
                            if (
                                next_index >= len(batch_comment_state.segments)
                                or decision.text
                                != batch_comment_state.segments[next_index]
                            ):
                                raise VisionAgentError(
                                    "批量评论只能按冻结计划输入当前精确标点分段。"
                                )
                        self.controller.vision_tap_relative(
                            *(
                                numeric_grid_key_coordinate(
                                    decision.keyboard_layout,
                                    decision.text,
                                    frames[-1].size,
                                )
                                if (
                                    decision.text.isdigit()
                                    and decision.keyboard_layout is not None
                                    and decision.keyboard_layout.get("type")
                                    == "numeric_grid"
                                )
                                else decision.coordinate
                            ),
                        )
                        if batch_comment_state is not None:
                            batch_comment_segment_pending = decision.text
                            pinyin_input_verification = decision.text
                        else:
                            comment_input_pending = decision.text
                        comment_numeric_keyboard_ready = bool(
                            decision.text.isdigit()
                            and decision.keyboard_layout is not None
                            and decision.keyboard_layout.get("type")
                            == "numeric_grid"
                        )
                        symbol_keyboard_pending = None
                        symbol_navigation_taps = 0
                    elif decision.action == "type_pinyin":
                        if (
                            not decision.text
                            or decision.text not in allowed_texts
                            or not decision.pinyin
                        ):
                            raise VisionAgentError(
                                "模型请求拼音输入的内容不在用户确认文字白名单中，已拒绝。"
                            )
                        if batch_comment_state is not None:
                            next_index = len(batch_comment_state.completed_segments)
                            if (
                                next_index >= len(batch_comment_state.segments)
                                or decision.text
                                != batch_comment_state.segments[next_index]
                            ):
                                raise VisionAgentError(
                                    "批量评论只能按冻结计划输入当前精确中文分段。"
                                )
                        self.controller.vision_type_pinyin(
                            decision.text,
                            decision.pinyin,
                            decision.keyboard_layout,
                        )
                        pinyin_candidate_pending = decision.text
                        pinyin_candidate_expected = decision.pinyin
                        pinyin_keyboard_layout = decision.keyboard_layout
                        pinyin_input_verification = None
                        pinyin_candidate_tap_coordinate = None
                        pinyin_candidate_tap_before = None
                        pinyin_candidate_retry_used = False
                        pinyin_send_verification = None
                        pinyin_send_retry_used = False
                        pinyin_send_observations = 0
                        pinyin_send_before = None
                        pinyin_send_post_empty = None
                        pinyin_send_empty_observations = 0
                    elif decision.action == "clear_text":
                        # For a frozen structured workflow, every correction is
                        # a full-session retry.  Record the exact visible text,
                        # enforce the per-field two-retry ceiling, then return
                        # to that field's first segment only after a later frame
                        # proves the ROI is empty.
                        if execution_plan and input_recovery_step_id is None:
                            active_step_id = execution_plan[plan_cursor]["id"]
                            recovery_state = input_recovery.state_for_step(
                                active_step_id
                            )
                            if recovery_state is not None:
                                observed_text = decision.observed_input_text or ""
                                try:
                                    (
                                        expected_delete_count,
                                        input_recovery_restart_cursor,
                                    ) = input_recovery.begin_recovery(
                                        active_step_id,
                                        observed_text,
                                    )
                                except ValueError as exc:
                                    raise VisionAgentError(str(exc)) from exc
                                if decision.delete_count != expected_delete_count:
                                    raise VisionAgentError(
                                        "模型退格次数与控制器计算的完整输入字符数"
                                        "不一致，已停止。"
                                    )
                                input_recovery_step_id = active_step_id
                        if (
                            continuous_comment_mode
                            and batch_comment_state is not None
                        ):
                            observed_text = decision.observed_input_text or ""
                            try:
                                expected_delete_count = (
                                    editable_character_count(observed_text)
                                    if batch_comment_state.awaiting_empty_confirmation
                                    else batch_comment_state.begin_full_retype(
                                        observed_text
                                    )
                                )
                            except ValueError as exc:
                                raise VisionAgentError(str(exc)) from exc
                            if decision.delete_count != expected_delete_count:
                                raise VisionAgentError(
                                    "评论退格次数与控制器读取的本次输入全文字符数"
                                    "不一致，已停止。"
                                )
                            batch_comment_recovery_pending = True
                            comment_clear_verification = (
                                batch_comment_state.target_text
                            )
                        comment_clear_keyboard_ready = (
                            comment_numeric_keyboard_ready
                        )
                        self.controller.vision_clear_text(
                            decision.keyboard_layout,
                            decision.delete_count,
                        )
                        pinyin_candidate_pending = None
                        pinyin_candidate_expected = None
                        pinyin_keyboard_layout = None
                        pinyin_input_verification = None
                        pinyin_candidate_tap_coordinate = None
                        pinyin_candidate_tap_before = None
                        pinyin_candidate_retry_used = False
                        pinyin_send_verification = None
                        pinyin_send_retry_used = False
                        pinyin_send_observations = 0
                        pinyin_send_before = None
                        pinyin_send_post_empty = None
                        pinyin_send_empty_observations = 0
                        comment_input_pending = None
                        symbol_keyboard_pending = None
                        symbol_navigation_taps = 0
                        if (
                            continuous_comment_mode
                            and batch_comment_state is not None
                        ):
                            comment_clear_verification = (
                                batch_comment_state.target_text
                            )
                        elif continuous_comment_mode and comment_symbol_target:
                            comment_clear_verification = comment_symbol_target
                        elif not continuous_comment_mode:
                            input_clear_verification = {
                                "text": decision.observed_input_text or "",
                                "delete_count": decision.delete_count or 0,
                            }
                    elif decision.action == "swipe_up":
                        if (
                            interaction_count
                            and interaction_metrics["pages_seen"]
                            >= max_page_checks
                            and interaction_metrics["completed_pages"]
                            < interaction_count
                        ):
                            raise VisionAgentError(
                                "已达到本批最多检查页面数，仍未补足目标成功数；"
                                "保存当前证据并停止。"
                            )
                        self.controller.vision_swipe_up()
                        if interaction_count:
                            interaction_metrics["page_advances"] += 1
                            interaction_metrics["pages_seen"] += 1
                        if continuous_like_mode:
                            like_pending_verification = False
                            like_page_tap_attempts = 0
                            like_before_tap_metrics = None
                            like_tap_center_px = None
                            combined_page_like_done = False
                        if continuous_comment_mode:
                            comment_close_verification = False
                            symbol_keyboard_pending = None
                            symbol_navigation_taps = 0
                            comment_numeric_keyboard_ready = False
                            comment_clear_keyboard_ready = False
                            if batch_comment_text and batch_comment_segments:
                                batch_comment_state = InputAttemptState(
                                    target_text=batch_comment_text,
                                    segments=list(batch_comment_segments),
                                )
                                batch_comment_segment_pending = None
                                batch_comment_recovery_pending = False
                    elif decision.action == "swipe_down":
                        self.controller.vision_swipe_down()
                    elif decision.action == "swipe_left":
                        self.controller.vision_swipe_left()
                    elif decision.action == "swipe_right":
                        self.controller.vision_swipe_right()
                    elif decision.action == "android_home":
                        self.controller.vision_android_home()
                    elif decision.action == "android_back":
                        self.controller.vision_android_back()
                    elif decision.action == "wait":
                        pass
                    else:  # validate_decision already guards this.
                        raise VisionAgentError(f"未实现动作：{decision.action}")
                    if (
                        send_phase_active_at_observe
                        and pinyin_send_verification is not None
                    ):
                        pinyin_send_observations += 1
                    settle_seconds = decision.wait_seconds
                    if decision.action in {"clear_text", "type_pinyin"}:
                        # Repeated physical taps can leave the phone/camera
                        # vibrating after the actuator call has returned.
                        settle_seconds = max(settle_seconds, 3.0)
                    elif decision.action == "type_symbol":
                        settle_seconds = max(settle_seconds, 2.0)
                    elif comment_close_verification and decision.action == "tap":
                        settle_seconds = max(settle_seconds, 1.5)
                    self.controller._sleep(settle_seconds)
                    previous_observation = frames[-1].copy()

            raise VisionAgentError(
                f"达到最大步骤数 {step_limit} 仍未确认目标完成，已停止。"
            )
        except Exception as exc:
            report = {
                "schema_version": 1,
                "outcome": "failed",
                "task_mode": task_mode,
                "goal": goal,
                "expected_result": expected_result,
                "execution_plan": execution_plan,
                "plan_cursor": plan_cursor,
                "duration_seconds": round(time.time() - started_at, 3),
                "steps": history,
                "evidence": evidence,
                "controller_metrics": controller_metrics_snapshot(),
                "agent": self.provider.status(),
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }
            self._write_json(report_path, report)
            try:
                setattr(exc, "report_path", str(report_path))
            except (AttributeError, TypeError):
                pass
            raise
