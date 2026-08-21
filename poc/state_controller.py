from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

from text_input_utils import editable_character_count, split_input_segments
from observation_images import (
    ObservationRoi,
    build_overview,
    build_roi,
    map_roi_bounds_to_full,
    map_roi_point_to_full,
    measure_local_stability,
)
from vision_agent import (
    VisionAgentError,
    _extract_json_object,
)
from target_locator import LocalTargetResolver, TargetResolution
from douyin_page_signals import detect_douyin_local_signals


STATE_CONTROLLER_PROTOCOL_VERSION = "2026-08-10-douyin-evidence-first-v24"


PAGE_STATES = {
    "android_home",
    "wechat_home",
    "wechat_search",
    "wechat_search_results",
    "wechat_chat",
    "wechat_chat_keyboard",
    "wechat_plus_menu",
    "wechat_album",
    "douyin_home",
    "douyin_search",
    "douyin_search_results",
    "douyin_video",
    "douyin_comments",
    "douyin_live",
    "douyin_live_preview",
    "douyin_live_room",
    "douyin_ad",
    "unknown",
}

# Perception reports the durable page separately from temporary UI layers.
# ``state`` remains the controller-facing effective state for compatibility,
# while ``base_state`` and ``overlays`` preserve what is actually on screen.
BASE_PAGE_STATES = {
    "android_home",
    "wechat_home",
    "wechat_search",
    "wechat_search_results",
    "wechat_chat",
    "wechat_album",
    "douyin_home",
    "douyin_search",
    "douyin_search_results",
    "douyin_video",
    "douyin_live",
    "douyin_live_preview",
    "douyin_live_room",
    "douyin_ad",
    "unknown",
}
OVERLAY_STATES = {
    "keyboard",
    "wechat_plus_menu",
    "douyin_comments",
    "permission_dialog",
    "system_dialog",
    "app_dialog",
    "loading",
    "unknown_overlay",
}
BLOCKING_OVERLAYS = {
    "permission_dialog",
    "system_dialog",
    "app_dialog",
    "loading",
    "unknown_overlay",
}
LEGACY_STATE_COMPONENTS = {
    "wechat_chat_keyboard": ("wechat_chat", {"keyboard"}),
    "wechat_plus_menu": ("wechat_chat", {"wechat_plus_menu"}),
    "douyin_comments": ("douyin_video", {"douyin_comments"}),
}

HEART_STATES = {"liked", "unliked", "not_visible", "unknown"}
INPUT_SCOPES = {"active_input", "page_text", "unverified"}

# A text value is editable evidence only when its own box is located in the
# input area expected for the current page. These are deliberately wider than
# a particular phone theme, but exclude chat bubbles, result cards and titles.
INPUT_FIELD_REGIONS = {
    "wechat_search": (0, 0, 1000, 350),
    "wechat_chat": (0, 620, 920, 1000),
    "wechat_chat_keyboard": (0, 330, 920, 820),
    "douyin_search": (0, 0, 1000, 350),
    "douyin_comments": (0, 430, 940, 1000),
}

TARGET_NAMES = {
    "wechat_icon",
    "douyin_icon",
    "open_search",
    "clear_search",
    "submit_search",
    "search_input",
    "exact_chat",
    "chat_input",
    "exact_candidate",
    "send",
    "plus",
    "album",
    "selected_image",
    "album_send",
    "video_tab",
    "heart",
    "comments",
    "comment_input",
    "comment_send",
    "close_comments",
    "close_overlay",
    "symbol_exact",
}

# A target name is meaningful only inside the page state where that control is
# expected.  Filtering before coordinate validation prevents an unrelated,
# hallucinated control from invalidating the whole observation while keeping
# required controls strict in their own state.
STATE_TARGET_NAMES = {
    "android_home": {"wechat_icon", "douyin_icon"},
    "wechat_home": {"open_search"},
    "wechat_search": {"search_input", "exact_chat", "exact_candidate", "symbol_exact"},
    "wechat_search_results": {"exact_chat"},
    "wechat_chat": {"chat_input", "send", "plus"},
    "wechat_chat_keyboard": {"chat_input", "exact_candidate", "send", "plus", "symbol_exact"},
    "wechat_plus_menu": {"album"},
    "wechat_album": {"selected_image", "album_send"},
    "douyin_home": {"open_search"},
    "douyin_search": {"search_input", "exact_candidate", "submit_search", "symbol_exact"},
    "douyin_search_results": {"clear_search", "video_tab"},
    "douyin_video": {"open_search", "heart", "comments"},
    "douyin_comments": {"comment_input", "exact_candidate", "comment_send", "close_comments", "symbol_exact"},
    "douyin_live": set(),
    "douyin_live_preview": set(),
    "douyin_live_room": set(),
    "douyin_ad": set(),
    "unknown": set(),
}


@dataclass(frozen=True)
class PageObservation:
    """Pure perception result. It intentionally contains no proposed action."""

    state: str
    confidence: float
    stable: bool
    reason: str
    base_state: str = "unknown"
    overlays: tuple[str, ...] = ()
    blocking_overlays: tuple[str, ...] = ()
    page_title: str | None = None
    input_text: str | None = None
    input_is_empty: bool | None = None
    input_scope: str = "unverified"
    input_focused: bool | None = None
    input_bounds: tuple[int, int, int, int] | None = None
    composition_text: str | None = None
    candidate_text: str | None = None
    exact_match_count: int | None = None
    keyboard_visible: bool = False
    keyboard_layout: dict[str, Any] | None = None
    heart_state: str = "not_visible"
    selection_count: int | None = None
    sent_message_visible: bool | None = None
    new_image_visible: bool | None = None
    comment_sent_visible: bool | None = None
    page_fingerprint: str = ""
    targets: dict[str, tuple[int, int]] = field(default_factory=dict)
    target_bounds: dict[str, tuple[int, int, int, int]] = field(default_factory=dict)
    visible_texts: list[str] = field(default_factory=list)
    live_evidence: list[str] = field(default_factory=list)
    live_evidence_bounds: dict[str, tuple[int, int, int, int]] = field(
        default_factory=dict
    )
    ad_evidence: list[str] = field(default_factory=list)
    search_query_text: str | None = None
    search_query_verified: bool | None = None
    search_results_relevant: bool | None = None
    search_result_evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["overlays"] = list(self.overlays)
        value["blocking_overlays"] = list(self.blocking_overlays)
        value["targets"] = {
            name: list(point) for name, point in self.targets.items()
        }
        value["target_bounds"] = {
            name: list(bounds) for name, bounds in self.target_bounds.items()
        }
        value["live_evidence_bounds"] = {
            text: list(bounds)
            for text, bounds in self.live_evidence_bounds.items()
        }
        if self.input_bounds is not None:
            value["input_bounds"] = list(self.input_bounds)
        return value


@dataclass(frozen=True)
class ControllerAction:
    kind: str
    reason: str
    target: str = ""
    coordinate: tuple[int, int] | None = None
    text: str | None = None
    pinyin: str | None = None
    delete_count: int | None = None
    keyboard_layout: dict[str, Any] | None = None
    wait_seconds: float = 1.0
    transition: str = ""

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        if self.coordinate is not None:
            value["coordinate"] = list(self.coordinate)
        return value


class PageObserver(Protocol):
    def status(self) -> dict[str, Any]: ...

    def observe(
        self,
        *,
        operation: str,
        params: dict[str, Any],
        frames: list[Image.Image],
        controller_context: dict[str, Any],
    ) -> PageObservation: ...


def _coordinate(value: Any, name: str) -> tuple[int, int]:
    # Normalize equivalent model serializations at the perception boundary.
    # The controller still receives one strict tuple, never raw model data.
    if isinstance(value, dict) and set(value) == {"x", "y"}:
        value = [value["x"], value["y"]]
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value)
    ):
        raise VisionAgentError(f"视觉状态中的 {name} 坐标无效。")
    point = (int(round(float(value[0]))), int(round(float(value[1]))))
    if not all(0 <= item <= 1000 for item in point):
        raise VisionAgentError(f"视觉状态中的 {name} 坐标超出0～1000。")
    return point


def _bounds(value: Any, name: str) -> tuple[int, int, int, int]:
    if isinstance(value, dict) and set(value) == {"left", "top", "right", "bottom"}:
        value = [value["left"], value["top"], value["right"], value["bottom"]]
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value)
    ):
        raise VisionAgentError(f"视觉状态中的 {name} 边界框无效。")
    bounds = tuple(int(round(float(item))) for item in value)
    left, top, right, bottom = bounds
    if not all(0 <= item <= 1000 for item in bounds) or right <= left or bottom <= top:
        raise VisionAgentError(f"视觉状态中的 {name} 边界框超出0～1000或没有面积。")
    return bounds


def _target_coordinate(
    value: Any,
    name: str,
) -> tuple[tuple[int, int], tuple[int, int, int, int] | None]:
    """Accept a point or the model's equivalent tight target box.

    Visible page targets use this boundary normalization, and the inferred box
    is retained for the existing region/box safety checks instead of being
    discarded.
    """

    is_box = (
        isinstance(value, list)
        and len(value) == 4
    ) or (
        isinstance(value, dict)
        and set(value) == {"left", "top", "right", "bottom"}
    )
    if not is_box:
        return _coordinate(value, name), None
    bounds = _bounds(value, name)
    left, top, right, bottom = bounds
    return ((left + right) // 2, (top + bottom) // 2), bounds


def _keyboard_anchor_coordinate(value: Any, name: str) -> tuple[int, int]:
    """Normalize a keyboard key center without accepting a broad region.

    Qwen may serialize a visible key as either ``[x, y]`` or a tight
    ``[left, top, right, bottom]`` box.  A small key box is equivalent to its
    center; a large/ambiguous box is rejected.  Full QWERTY row geometry is
    still validated later by ``qwerty_keyboard_config_from_anchors`` before
    any physical key is pressed.
    """

    point, bounds = _target_coordinate(value, name)
    if bounds is None:
        return point
    left, top, right, bottom = bounds
    if right - left > 120 or bottom - top > 100:
        raise VisionAgentError(f"视觉状态中的 {name} 按键边界框过大，拒绝执行。")
    return point


def _box_inside_region(
    bounds: tuple[int, int, int, int],
    region: tuple[int, int, int, int],
) -> bool:
    left, top, right, bottom = bounds
    r_left, r_top, r_right, r_bottom = region
    return left >= r_left and top >= r_top and right <= r_right and bottom <= r_bottom


def _same_input_field(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> bool:
    """Allow camera jitter, but reject switching to another editable field."""

    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    iou = intersection / union if union else 0.0
    first_center = ((first[0] + first[2]) / 2, (first[1] + first[3]) / 2)
    second_center = ((second[0] + second[2]) / 2, (second[1] + second[3]) / 2)
    center_delta = max(
        abs(first_center[0] - second_center[0]),
        abs(first_center[1] - second_center[1]),
    )
    return iou >= 0.25 and center_delta <= 100


_DOUYIN_LIVE_PREVIEW_MARKERS = (
    "点击进入直播间",
    "进入直播间",
    "直播中",
    "本地视觉:直播中徽标",
)
_DOUYIN_LIVE_ROOM_MARKERS = (
    "说点什么",
    "在线人数",
    "小时榜",
    "粉丝团",
    "送礼",
    "礼物",
    "观众",
    "本地视觉:直播间右上角关闭键",
)
_DOUYIN_AD_MARKERS = (
    "跳过广告",
    "立即下载",
    "立即安装",
    "了解详情",
    "打开应用",
    "广告",
)


def _clean_evidence_list(value: Any, *, limit: int = 12) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value[:limit]:
        text = str(item).strip()[:160]
        if text and text not in result:
            result.append(text)
    return result


def _contains_marker(values: list[str], markers: tuple[str, ...]) -> list[str]:
    hits: list[str] = []
    for value in values:
        compact = re.sub(r"\s+", "", value)
        if any(re.sub(r"\s+", "", marker) in compact for marker in markers):
            hits.append(value)
    return hits


def _clean_evidence_bounds(
    value: Any,
    evidence: list[str],
) -> dict[str, tuple[int, int, int, int]]:
    """Keep only normalized boxes tied to an explicitly transcribed string."""

    if not isinstance(value, dict):
        return {}
    allowed = set(evidence)
    result: dict[str, tuple[int, int, int, int]] = {}
    for raw_text, raw_box in list(value.items())[:12]:
        text = str(raw_text).strip()[:160]
        if not text or text not in allowed:
            continue
        try:
            box = _bounds(raw_box, f"live_evidence_bounds.{text}")
        except VisionAgentError:
            continue
        left, top, right, bottom = box
        # A text label must occupy a local region, not the whole screen or a
        # near-zero point invented to satisfy the schema.
        area = (right - left) * (bottom - top)
        if 120 <= area <= 180000:
            result[text] = box
    return result


def _normalize_douyin_page_type(
    base_state: str,
    legacy_state: str,
    visible_texts: list[str],
    live_evidence: list[str],
    live_evidence_bounds: dict[str, tuple[int, int, int, int]],
    ad_evidence: list[str],
) -> tuple[str, list[str], list[str], str | None]:
    """Resolve feed type from explicit evidence, never a free-form reason."""

    if not (base_state.startswith("douyin_") or legacy_state.startswith("douyin_")):
        return base_state, live_evidence, ad_evidence, None

    # Model-transcribed live words are admissible only when the model also
    # locates the exact text. Local detectors are carried as separate trusted
    # markers and must agree with the bounded model evidence.
    bounded_live_values = [
        text for text in live_evidence if text in live_evidence_bounds
    ]
    local_preview = "本地视觉:直播中徽标" in live_evidence
    local_room = "本地视觉:直播间右上角关闭键" in live_evidence
    all_ad_values = [*visible_texts, *ad_evidence]
    preview_hits = _contains_marker(
        bounded_live_values,
        _DOUYIN_LIVE_PREVIEW_MARKERS,
    )
    room_hits = _contains_marker(
        bounded_live_values,
        _DOUYIN_LIVE_ROOM_MARKERS,
    )
    ad_hits = _contains_marker(all_ad_values, _DOUYIN_AD_MARKERS)
    normalized_live = list(dict.fromkeys([*live_evidence, *preview_hits, *room_hits]))
    normalized_ad = list(dict.fromkeys([*ad_evidence, *ad_hits]))

    # A live-shopping page may show products, prices and sales counts. Those
    # elements are not ad evidence, and explicit live evidence always wins.
    if preview_hits and local_preview:
        return (
            "douyin_live_preview",
            normalized_live,
            normalized_ad,
            "检测到明确直播预览证据",
        )
    if room_hits and local_room:
        return (
            "douyin_live_room",
            normalized_live,
            normalized_ad,
            "检测到明确直播间证据",
        )
    if ad_hits:
        return (
            "douyin_ad",
            normalized_live,
            normalized_ad,
            "检测到明确广告文字证据",
        )
    if base_state in {
        "douyin_live",
        "douyin_live_preview",
        "douyin_live_room",
        "douyin_ad",
    }:
        return (
            "unknown",
            normalized_live,
            normalized_ad,
            "模型给出直播或广告结论，但缺少带边界框且与本地视觉一致的证据",
        )
    return base_state, normalized_live, normalized_ad, None


def parse_page_observation(payload: dict[str, Any]) -> PageObservation:
    # Perception is deliberately not allowed to smuggle an action back into
    # the controller.  This is the architectural boundary that the old runner
    # did not have.
    forbidden = {"action", "next_action", "plan", "step", "tap", "swipe"}
    if forbidden.intersection(payload):
        raise VisionAgentError("页面观察结果包含动作或计划字段，已拒绝。")

    legacy_state = str(payload.get("state", "unknown")).strip()
    if legacy_state not in PAGE_STATES:
        legacy_state = "unknown"
    implied_base, implied_overlays = LEGACY_STATE_COMPONENTS.get(
        legacy_state,
        (legacy_state if legacy_state in BASE_PAGE_STATES else "unknown", set()),
    )
    raw_base_state = payload.get("base_state")
    base_state = (
        str(raw_base_state).strip()
        if raw_base_state is not None
        else implied_base
    )
    if base_state not in BASE_PAGE_STATES:
        base_state = "unknown"
    if legacy_state in LEGACY_STATE_COMPONENTS and raw_base_state is not None:
        if base_state != implied_base:
            raise VisionAgentError("页面观察结果中的主页面与复合页面状态矛盾。")

    overlays: set[str] = set(implied_overlays)
    raw_overlays = payload.get("overlays")
    if raw_overlays is not None:
        if not isinstance(raw_overlays, list):
            raise VisionAgentError("页面观察结果 overlays 必须是数组。")
        for item in raw_overlays[:8]:
            name = str(item).strip()
            overlays.add(name if name in OVERLAY_STATES else "unknown_overlay")
    if payload.get("keyboard_visible") is True:
        overlays.add("keyboard")

    if "wechat_plus_menu" in overlays and base_state != "wechat_chat":
        raise VisionAgentError("微信加号菜单弹层只能覆盖在微信聊天页上。")
    visible_texts_value = payload.get("visible_texts") or []
    visible_texts = (
        [str(item)[:120] for item in visible_texts_value[:30]]
        if isinstance(visible_texts_value, list)
        else []
    )
    live_evidence = _clean_evidence_list(payload.get("live_evidence"))
    live_evidence_bounds = _clean_evidence_bounds(
        payload.get("live_evidence_bounds"),
        live_evidence,
    )
    ad_evidence = _clean_evidence_list(payload.get("ad_evidence"))
    base_state, live_evidence, ad_evidence, normalization_reason = (
        _normalize_douyin_page_type(
            base_state,
            legacy_state,
            visible_texts,
            live_evidence,
            live_evidence_bounds,
            ad_evidence,
        )
    )
    if "douyin_comments" in overlays and base_state != "douyin_video":
        # Explicit, transcribed live/ad markers are stronger evidence than an
        # unsupported overlay label. Qwen may confuse the feed's bottom text
        # region with a comments sheet; do not let that hallucinated layer
        # invalidate a page type independently proven by visible text/local CV.
        if base_state in {
            "douyin_live_preview",
            "douyin_live_room",
            "douyin_ad",
        } and normalization_reason:
            overlays.discard("douyin_comments")
            normalization_reason += "；已忽略缺少独立证据的评论弹层标签"
        else:
            raise VisionAgentError("抖音评论弹层与直播/广告页面证据矛盾。")

    # Build one controller-facing effective state without losing the layer
    # information. Expected overlays take precedence over the base page.
    if base_state == "wechat_chat" and "wechat_plus_menu" in overlays:
        state = "wechat_plus_menu"
    elif base_state == "wechat_chat" and "keyboard" in overlays:
        state = "wechat_chat_keyboard"
    elif base_state == "douyin_video" and "douyin_comments" in overlays:
        state = "douyin_comments"
    else:
        state = base_state
    # Douyin's search results use a characteristic top tab row. Qwen has
    # repeatedly called that grid a "home feed" because the true query field
    # is partially hidden by the seller overlay. Normalize this high-signal UI
    # invariant before filtering state-specific targets.
    search_tabs = {"综合", "视频", "用户", "商品", "直播"}
    visible_search_tabs = search_tabs.intersection(
        {item.strip() for item in visible_texts}
    )
    if state in {"douyin_home", "douyin_video"} and len(visible_search_tabs) >= 3:
        state = "douyin_search_results"
        base_state = "douyin_search_results"
    confidence_value = payload.get("confidence", 0.0)
    if isinstance(confidence_value, bool) or not isinstance(confidence_value, (int, float)):
        confidence_value = 0.0
    confidence = max(0.0, min(1.0, float(confidence_value)))

    raw_targets = payload.get("targets") or {}
    if not isinstance(raw_targets, dict):
        raise VisionAgentError("页面观察结果 targets 必须是对象。")
    targets: dict[str, tuple[int, int]] = {}
    inferred_target_bounds: dict[str, tuple[int, int, int, int]] = {}
    state_targets = STATE_TARGET_NAMES.get(state, set())
    for raw_name, raw_point in raw_targets.items():
        name = str(raw_name).strip()[:60]
        # The observation schema permits omitted/non-visible controls. Some
        # models serialize those optional values as JSON null instead of
        # omitting the key. Treat both forms identically at the boundary.
        if raw_point is None:
            continue
        image_target = bool(
            state == "wechat_album"
            and re.fullmatch(r"image_[1-9]|image_1[0-9]|image_20", name)
        )
        recovery_target = bool(
            name == "close_overlay"
            and (bool(overlays.intersection(BLOCKING_OVERLAYS)) or state == "unknown")
        )
        if (name in TARGET_NAMES and name in state_targets) or image_target or recovery_target:
            point, inferred_bounds = _target_coordinate(raw_point, name)
            targets[name] = point
            if inferred_bounds is not None:
                inferred_target_bounds[name] = inferred_bounds

    raw_target_bounds = payload.get("target_bounds") or {}
    if not isinstance(raw_target_bounds, dict):
        raise VisionAgentError("页面观察结果 target_bounds 必须是对象。")
    target_bounds: dict[str, tuple[int, int, int, int]] = dict(inferred_target_bounds)
    for raw_name, raw_bounds in raw_target_bounds.items():
        name = str(raw_name).strip()[:60]
        if raw_bounds is None or name not in targets:
            continue
        target_bounds[name] = _bounds(raw_bounds, f"target_bounds.{name}")

    raw_keyboard = payload.get("keyboard_layout")
    keyboard_layout = None
    if isinstance(raw_keyboard, dict):
        layout_type = str(raw_keyboard.get("type", "unknown")).strip()
        anchors_value = raw_keyboard.get("anchors")
        if not isinstance(anchors_value, dict):
            # Accept the equivalent flat form emitted by Qwen while keeping a
            # single strict nested representation inside the controller.
            anchors_value = {
                name: raw_keyboard[name]
                for name in ("q", "p", "a", "l", "z", "m", "backspace")
                if name in raw_keyboard and raw_keyboard[name] is not None
            }
        anchors: dict[str, list[int]] = {}
        if isinstance(anchors_value, dict):
            for name, point in anchors_value.items():
                anchors[str(name)] = list(
                    _keyboard_anchor_coordinate(point, f"keyboard.{name}")
                )
        keyboard_layout = {"type": layout_type, "anchors": anchors}

    def optional_text(name: str, limit: int = 200) -> str | None:
        value = payload.get(name)
        if value is None:
            return None
        return str(value)[:limit]

    def optional_bool(name: str) -> bool | None:
        value = payload.get(name)
        return value if isinstance(value, bool) else None

    def optional_int(name: str) -> int | None:
        value = payload.get(name)
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    heart_state = str(payload.get("heart_state", "not_visible")).strip()
    if heart_state not in HEART_STATES:
        heart_state = "unknown"
    input_scope = str(payload.get("input_scope", "unverified")).strip()
    if input_scope not in INPUT_SCOPES:
        input_scope = "unverified"
    search_evidence_value = payload.get("search_result_evidence") or []
    search_result_evidence = (
        [str(item).strip()[:160] for item in search_evidence_value[:8] if str(item).strip()]
        if isinstance(search_evidence_value, list)
        else []
    )

    observation = PageObservation(
        state=state,
        confidence=confidence,
        stable=payload.get("stable") is True,
        reason=(
            (
                str(payload.get("reason", "")).strip()
                + (f"；{normalization_reason}" if normalization_reason else "")
            ).strip("；")[:500]
        ),
        base_state=base_state,
        overlays=tuple(sorted(overlays)),
        blocking_overlays=tuple(sorted(overlays.intersection(BLOCKING_OVERLAYS))),
        page_title=optional_text("page_title", 120),
        input_text=optional_text("input_text"),
        input_is_empty=optional_bool("input_is_empty"),
        input_scope=input_scope,
        input_focused=optional_bool("input_focused"),
        input_bounds=(
            _bounds(payload.get("input_bounds"), "input_bounds")
            if payload.get("input_bounds") is not None
            else None
        ),
        composition_text=optional_text("composition_text"),
        candidate_text=optional_text("candidate_text"),
        exact_match_count=optional_int("exact_match_count"),
        keyboard_visible=payload.get("keyboard_visible") is True,
        keyboard_layout=keyboard_layout,
        heart_state=heart_state,
        selection_count=optional_int("selection_count"),
        sent_message_visible=optional_bool("sent_message_visible"),
        new_image_visible=optional_bool("new_image_visible"),
        comment_sent_visible=optional_bool("comment_sent_visible"),
        page_fingerprint=str(payload.get("page_fingerprint", ""))[:100],
        targets=targets,
        target_bounds=target_bounds,
        visible_texts=visible_texts,
        live_evidence=live_evidence,
        live_evidence_bounds=live_evidence_bounds,
        ad_evidence=ad_evidence,
        search_query_text=optional_text("search_query_text", 120),
        search_query_verified=optional_bool("search_query_verified"),
        search_results_relevant=optional_bool("search_results_relevant"),
        search_result_evidence=search_result_evidence,
    )
    if observation.input_scope != "active_input" and (
        observation.input_text is not None or observation.input_is_empty is not None
    ):
        # Preserve the diagnostic value, but make the scope explicit. The
        # controller will never use page/history text as editable field text.
        pass
    if observation.input_scope == "active_input":
        if observation.input_focused is not True or observation.input_bounds is None:
            raise VisionAgentError("当前输入框缺少焦点或自身边界证据。")
        allowed_input_region = INPUT_FIELD_REGIONS.get(observation.state)
        if allowed_input_region is None or not _box_inside_region(
            observation.input_bounds,
            allowed_input_region,
        ):
            # Geometry is a stronger signal than model-provided semantics.
            # A label, suggestion row or chat bubble outside the page's
            # editable ROI must never become field text.  Discard that field
            # evidence and let the state controller decide whether the current
            # workflow has an independent causal input path.  Ordinary visible
            # input workflows still fail closed because they require an
            # ``active_input`` observation before typing, clearing or sending.
            observation = replace(
                observation,
                input_text=None,
                input_is_empty=None,
                input_scope="unverified",
                input_focused=None,
                input_bounds=None,
            )
    if observation.candidate_text and "exact_candidate" not in observation.targets:
        raise VisionAgentError(
            "页面观察结果自相矛盾：识别到精确候选词但没有提供 exact_candidate 坐标。"
        )
    if "exact_candidate" in observation.targets and not observation.candidate_text:
        raise VisionAgentError(
            "页面观察结果自相矛盾：提供了 exact_candidate 坐标但没有候选词文字。"
        )
    return observation


class DashScopePageObserver:
    """Narrow adapter: Qwen may describe the page, never choose an action."""

    def __init__(self, provider: Any) -> None:
        self.provider = provider
        self.last_raw_response = ""
        self.last_observation_diagnostics: dict[str, Any] = {}

    def status(self) -> dict[str, Any]:
        value = dict(self.provider.status())
        value["execution_architecture"] = "page_state_graph_v1"
        value["model_role"] = "observation_only"
        value["visual_input_strategy"] = (
            "medium_res_overview_high_res_roi_local_crosscheck"
        )
        return value

    @staticmethod
    def _attach_local_douyin_signals(
        payload: dict[str, Any],
        frame: Image.Image,
        *,
        operation: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not operation.startswith("douyin."):
            return payload, {}
        signals = detect_douyin_local_signals(frame)
        merged = dict(payload)
        evidence = _clean_evidence_list(merged.get("live_evidence"))
        if signals.live_preview_badge is not None:
            evidence.append("本地视觉:直播中徽标")
        if signals.live_room_close is not None:
            evidence.append("本地视觉:直播间右上角关闭键")
        merged["live_evidence"] = list(dict.fromkeys(evidence))
        return merged, signals.to_dict()

    @staticmethod
    def _needs_roi(
        observation: PageObservation,
        *,
        operation: str,
        params: dict[str, Any],
    ) -> list[ObservationRoi]:
        """Request detail only when the overview lacks evidence needed now."""

        state = observation.state
        targets = observation.targets
        bounds = observation.target_bounds
        requests: list[ObservationRoi] = []

        def add(name: str, box: tuple[int, int, int, int], purpose: str) -> None:
            if all(item.name != name for item in requests):
                requests.append(ObservationRoi(name, box, purpose))

        if state == "unknown":
            if observation.blocking_overlays:
                if "close_overlay" not in targets:
                    add(
                        "overlay_close",
                        (520, 0, 1000, 480),
                        "只寻找当前最上层弹层真实可见的关闭叉键",
                    )
                return requests[:2]
            if operation.startswith("douyin."):
                add(
                    "douyin_page_evidence",
                    (0, 80, 820, 940),
                    "中性辨别当前抖音页面类型，抄写实际可见的类型证据",
                )
                add(
                    "page_state_right",
                    (560, 80, 1000, 920),
                    "中性辨别右侧实际可见的控件、文字和颜色",
                )
            elif operation.startswith("wechat."):
                add(
                    "page_state_top",
                    (0, 0, 1000, 400),
                    "确认微信首页、搜索结果或聊天标题特征",
                )
                add(
                    "page_state_bottom",
                    (0, 520, 1000, 1000),
                    "确认微信底部导航、聊天输入框或键盘特征",
                )
            return requests[:2]

        if state == "android_home":
            target = "wechat_icon" if operation.startswith("wechat.") else "douyin_icon"
            if target not in targets or target not in bounds:
                add("app_grid", (0, 30, 1000, 930), f"识别{target}图标和自身边界")
        elif state == "wechat_home":
            if "open_search" not in targets:
                add("top_bar", (0, 0, 1000, 320), "识别微信顶部搜索入口")
        elif state in {"wechat_search", "wechat_search_results"}:
            if observation.candidate_text and "exact_candidate" in targets:
                return requests
            if "exact_chat" not in targets or observation.exact_match_count is None:
                add("search_results", (0, 0, 1000, 850), "读取搜索框和完全匹配的聊天结果")
            if observation.keyboard_visible and observation.keyboard_layout is None:
                add("input_keyboard", (0, 350, 1000, 1000), "读取输入框、候选栏和键盘布局")
        elif state in {"wechat_chat", "wechat_chat_keyboard"}:
            if not observation.page_title:
                add("top_bar", (0, 0, 1000, 260), "读取当前微信聊天标题")
            if operation == "wechat.send_text" and (
                observation.input_scope != "active_input"
                or "chat_input" not in targets
                or (state == "wechat_chat_keyboard" and observation.keyboard_layout is None)
            ):
                add("input_keyboard", (0, 330, 1000, 1000), "读取输入框、候选栏、发送键和键盘")
            if operation == "wechat.send_album_image" and "plus" not in targets:
                add("chat_controls", (0, 560, 1000, 1000), "识别聊天页加号控件")
        elif state == "wechat_plus_menu":
            if "album" not in targets:
                add("plus_menu", (0, 420, 1000, 1000), "识别微信加号菜单中的相册")
        elif state == "wechat_album":
            image_index = int(params.get("image_index") or 1)
            image_target = f"image_{image_index}"
            if image_target not in targets or "album_send" not in targets:
                add("album_grid", (0, 30, 1000, 950), "读取相册网格、目标图片和发送按钮")
        elif state in {"douyin_home", "douyin_video"}:
            if operation == "douyin.search" and "open_search" not in targets:
                add("top_bar", (500, 0, 1000, 260), "识别抖音顶部搜索入口")
            if operation in {"douyin.like_current", "douyin.batch_interact"}:
                # The overview is for page classification, not a final color
                # verdict. Always re-read the heart from the high-resolution
                # right rail so a confident low-detail guess cannot suppress
                # the verification crop.
                add(
                    "right_actions",
                    (580, 60, 1000, 880),
                    "独立复核爱心实际颜色和爱心自身边界，不沿用全景结论",
                )
            if operation in {"douyin.comment_current", "douyin.batch_interact"} and (
                "comments" not in targets
            ):
                add("right_actions", (580, 60, 1000, 880), "识别爱心和评论按钮")
        elif state.startswith("douyin_live") or state == "douyin_ad":
            add(
                "douyin_page_evidence",
                (0, 80, 820, 940),
                "中性复核当前抖音页面类型，抄写实际可见的类型证据",
            )
            add(
                "page_state_right",
                (560, 30, 1000, 920),
                "中性复核右侧实际可见的控件、文字和颜色",
            )
        elif state == "douyin_search":
            if observation.candidate_text and "exact_candidate" in targets:
                return requests
            if observation.input_scope != "active_input" or "submit_search" not in targets:
                add("input_keyboard", (0, 0, 1000, 1000), "读取抖音搜索框、候选栏、键盘和提交键")
        elif state == "douyin_search_results":
            if (
                observation.search_query_verified is not True
                or "video_tab" not in targets
            ):
                add("search_header", (0, 0, 1000, 380), "读取搜索词和视频标签")
        elif state == "douyin_comments":
            if (
                observation.input_scope != "active_input"
                or "comment_input" not in targets
                or (observation.keyboard_visible and observation.keyboard_layout is None)
            ):
                add("comment_input", (0, 360, 1000, 1000), "读取评论输入框、候选栏、键盘和发送键")

        if observation.blocking_overlays and "close_overlay" not in targets:
            add("overlay_close", (520, 0, 1000, 480), "只寻找当前最上层弹层真实可见的关闭叉键")
        return requests[:2]

    @staticmethod
    def _merge_roi_patch(
        payload: dict[str, Any],
        patch: dict[str, Any],
        roi: ObservationRoi,
    ) -> dict[str, Any]:
        forbidden = {"action", "next_action", "plan", "step", "tap", "swipe"}
        if forbidden.intersection(patch):
            raise VisionAgentError("ROI观察结果包含动作或计划字段，已拒绝。")
        merged = dict(payload)
        if str(merged.get("base_state") or merged.get("state") or "unknown") == "unknown":
            refined_state = str(patch.get("base_state") or "unknown").strip()
            if refined_state in BASE_PAGE_STATES and refined_state != "unknown":
                merged["base_state"] = refined_state
                merged.pop("state", None)
            if isinstance(patch.get("overlays"), list):
                merged["overlays"] = patch["overlays"]
            if patch.get("reason"):
                merged["reason"] = str(patch["reason"])
        local_targets = patch.get("targets") or {}
        local_bounds = patch.get("target_bounds") or {}
        if not isinstance(local_targets, dict) or not isinstance(local_bounds, dict):
            raise VisionAgentError("ROI观察结果的 targets/target_bounds 格式错误。")
        targets = dict(merged.get("targets") or {})
        target_bounds = dict(merged.get("target_bounds") or {})
        for name, value in local_targets.items():
            if value is None:
                continue
            target_name = str(name).strip()[:60]
            image_target = bool(
                re.fullmatch(r"image_[1-9]|image_1[0-9]|image_20", target_name)
            )
            # ROI responses are supplemental observations.  Ignore protocol-
            # external target aliases before validating their coordinates,
            # exactly as the main observation boundary does.  A stray key such
            # as ``like`` must not invalidate an otherwise useful heart-state
            # observation, and it must not be silently reinterpreted as a
            # different clickable control.
            if target_name not in TARGET_NAMES and not image_target:
                continue
            point, _ = _target_coordinate(value, target_name)
            targets[target_name] = list(map_roi_point_to_full(point, roi.bounds))
        for name, value in local_bounds.items():
            if value is None or str(name) not in targets:
                continue
            box = _bounds(value, f"roi.{roi.name}.target_bounds.{name}")
            target_bounds[str(name)] = list(map_roi_bounds_to_full(box, roi.bounds))
        merged["targets"] = targets
        merged["target_bounds"] = target_bounds

        if patch.get("input_bounds") is not None:
            box = _bounds(patch["input_bounds"], f"roi.{roi.name}.input_bounds")
            merged["input_bounds"] = list(map_roi_bounds_to_full(box, roi.bounds))

        keyboard = patch.get("keyboard_layout")
        if isinstance(keyboard, dict):
            anchors = keyboard.get("anchors")
            if not isinstance(anchors, dict):
                anchors = {
                    name: keyboard[name]
                    for name in ("q", "p", "a", "l", "z", "m", "backspace")
                    if keyboard.get(name) is not None
                }
            mapped: dict[str, list[int]] = {}
            for name, point in anchors.items():
                local, _ = _target_coordinate(point, f"keyboard.{name}")
                mapped[str(name)] = list(map_roi_point_to_full(local, roi.bounds))
            merged["keyboard_layout"] = {
                "type": str(keyboard.get("type") or "unknown"),
                "anchors": mapped,
            }

        direct_fields = {
            "page_title",
            "input_text",
            "input_is_empty",
            "input_scope",
            "input_focused",
            "composition_text",
            "candidate_text",
            "exact_match_count",
            "keyboard_visible",
            "heart_state",
            "selection_count",
            "sent_message_visible",
            "new_image_visible",
            "comment_sent_visible",
            "search_query_text",
            "search_query_verified",
            "search_results_relevant",
            "search_result_evidence",
        }
        for name in direct_fields:
            if name in patch and patch[name] is not None:
                merged[name] = patch[name]
        if isinstance(patch.get("visible_texts"), list):
            seen = list(merged.get("visible_texts") or [])
            for item in patch["visible_texts"]:
                text = str(item).strip()
                if text and text not in seen:
                    seen.append(text)
            merged["visible_texts"] = seen[:30]
        for evidence_name in ("live_evidence", "ad_evidence"):
            if isinstance(patch.get(evidence_name), list):
                seen = _clean_evidence_list(merged.get(evidence_name))
                for item in patch[evidence_name]:
                    text = str(item).strip()[:160]
                    if text and text not in seen:
                        seen.append(text)
                merged[evidence_name] = seen[:12]
        if isinstance(patch.get("live_evidence_bounds"), dict):
            mapped_evidence_bounds = dict(merged.get("live_evidence_bounds") or {})
            accepted_evidence = set(_clean_evidence_list(merged.get("live_evidence")))
            for raw_text, raw_box in list(patch["live_evidence_bounds"].items())[:12]:
                text = str(raw_text).strip()[:160]
                if not text or text not in accepted_evidence:
                    continue
                try:
                    box = _bounds(
                        raw_box,
                        f"roi.{roi.name}.live_evidence_bounds.{text}",
                    )
                except VisionAgentError:
                    continue
                mapped_evidence_bounds[text] = list(
                    map_roi_bounds_to_full(box, roi.bounds)
                )
            merged["live_evidence_bounds"] = mapped_evidence_bounds
        detail_confidence = patch.get("confidence")
        if isinstance(detail_confidence, (int, float)) and not isinstance(detail_confidence, bool):
            merged["confidence"] = min(
                float(merged.get("confidence") or 0.0),
                float(detail_confidence),
            )
        return merged

    def observe(
        self,
        *,
        operation: str,
        params: dict[str, Any],
        frames: list[Image.Image],
        controller_context: dict[str, Any],
    ) -> PageObservation:
        if len(frames) < 4:
            raise VisionAgentError("页面状态判断至少需要4帧。")
        local_stability = measure_local_stability(frames)
        if not local_stability.stable:
            self.last_observation_diagnostics = {
                "strategy": "medium_res_overview_high_res_roi_local_crosscheck",
                "local_stability": local_stability.to_dict(),
                "model_calls": 0,
                "images": [],
            }
            raise VisionAgentError(
                f"本地多帧稳定性检查未通过：{local_stability.reason}；禁止调用模型后继续动作。"
            )
        overview = build_overview(frames[-1])
        expected = {
            "operation": operation,
            "chat_name": params.get("chat_name"),
            "text": params.get("text"),
            "keyword": params.get("keyword"),
            "comment_text": params.get("comment_text"),
            "image_index": params.get("image_index"),
        }
        prompt = f"""
你是手机页面观察器，不是操作 Agent。图片是当前真实手机的一张中等清晰度全景。
控制器已在本地检查4帧稳定性；你不得根据单张图片猜测动作或覆盖本地稳定性结论。
你只能报告页面状态、屏幕文字和控件中心，绝对不能建议动作、步骤或计划。

当前任务只用于提供需要逐字比较的文本：
{json.dumps(expected, ensure_ascii=False)}
控制器只提供已经执行过的事实，不是请你规划：
{json.dumps(controller_context, ensure_ascii=False)}

        base_state 只报告不受临时弹层影响的主页面，只能取：
        {json.dumps(sorted(BASE_PAGE_STATES), ensure_ascii=False)}
        overlays 单独报告覆盖在主页面之上的临时层，只能取：
        {json.dumps(sorted(OVERLAY_STATES), ensure_ascii=False)}
        键盘可见时必须包含 keyboard；微信加号菜单包含 wechat_plus_menu；抖音评论面板包含
        douyin_comments。权限弹窗、系统弹窗、App自身弹窗、加载遮罩和无法识别的遮罩分别使用
        permission_dialog、system_dialog、app_dialog、loading、unknown_overlay。
        loading 只有在画面中真实清晰可见加载圆环、进度指示，或“加载中/正在加载/请稍候”等
        明确加载文字时才能填写。视频画面半透明、局部模糊、灰色圆角按钮、直播预览按钮、
        暂停画面或亮度变化都不是 loading 证据，禁止据此填写 loading。
        坐标统一使用0～1000相对坐标。

        targets 只填写主页面与当前弹层中真实清晰可见的控件中心，其他页面的控件一律不能填写。
所有 targets 中的控件还应在 target_bounds 中给出可点击矩形 [left,top,right,bottom]；
边界框必须包住该控件中心，只框控件自身，不得框整行、整块面板或相邻控件。无法确认边界时省略该控件，禁止猜测。
打开搜索入口、清除旧查询和提交搜索必须使用不同名称：open_search 只表示顶部放大镜/搜索入口；
clear_search 只表示旧搜索结果页查询栏中、提交按钮左侧的圆形 X；submit_search 只表示
输入关键词后的提交按钮（包括键盘右下角蓝色“搜索”键）。在 douyin_search_results 中，
右侧可见的“搜索”是 submit_search，不是 open_search。
名称使用：
wechat_icon, douyin_icon, open_search, clear_search, submit_search, search_input, exact_chat, chat_input,
exact_candidate, send, plus, album, selected_image, album_send, video_tab,
heart, comments, comment_input, comment_send, close_comments, symbol_exact。
阻塞弹层或未知页面如果有真实清晰可见、能够关闭当前最上层界面的叉键，只把这个叉键命名为
close_overlay；广告正文、图片内容、键盘退格、评论区关闭键均不能冒充 close_overlay。
相册目标图片另用 image_1、image_2……命名。

只读取当前页面“当前获得焦点的可编辑输入框”文字。聊天气泡、昵称、标题、历史记录、
搜索建议、占位提示和结果正文均不能算 input_text。只有文字确实位于当前焦点输入框内部时，
input_scope 才填 active_input；若识别到的同名文字来自页面正文/历史记录，填 page_text；
输入框被遮挡或无法确认时填 unverified，且 input_text 和 input_is_empty 必须填 null。
当前焦点输入框确认为空时 input_text=""、input_is_empty=true、input_scope="active_input"。
input_scope="active_input" 时还必须同时填写 input_focused=true 和输入框自身的矩形
input_bounds:[left,top,right,bottom]；不得框住聊天气泡、候选栏、整行面板或键盘。
无法确认焦点或边界时 input_focused/input_bounds 填 null，并将 input_scope 填 unverified。
拼音尚在组合态时 composition_text 填实际拼音，candidate_text 只填与控制器目标
逐字完全一致的候选词。candidate_text 非null时必须同时给出该候选词中心坐标
targets.exact_candidate；看不到完全一致候选就把两者都省略/填null，禁止只填其中一个。
如果是键盘，keyboard_layout.type 填 qwerty 或 generic；完整QWERTY时标注
q,p,a,l,z,m,backspace 六个中心。不要根据常见布局猜坐标。
微信聊天必须把顶部标题写入 page_title；搜索结果完全匹配数量写 exact_match_count。
        抖音爱心只按实际颜色报告 liked/unliked/unknown。页面类型必须按可见证据区分：
        - 中央有“点击进入直播间”或画面有“直播中”徽标时是 douyin_live_preview；
        - 已进入直播间，能看到在线人数、说点什么、礼物/粉丝团等直播控件时是 douyin_live_room；
        - 只有明确看到“广告”“跳过广告”“立即下载”“立即安装”或“了解详情”等广告标记，
          且没有任何直播证据时，才是 douyin_ad；
        - 商品、价格、销量、购物卡片和“讲解中”本身都不能作为广告证据；
        - 标准右侧爱心/评论/收藏/分享操作栏存在，且没有直播或广告证据时，才是 douyin_video。
        把画面中逐字可见的直播证据写入 live_evidence，并在 live_evidence_bounds 中按
        同一文字给出只包住该文字的矩形；没有可靠文字边界框就不得写入直播证据。
        把逐字可见的广告证据写入 ad_evidence；不得把你的推测或 reason 内容写入证据数组。
        直播证据与广告外观同时出现时，
        直播证据优先。无法取得明确证据时填 unknown，不得猜。
        旧状态 douyin_live 只为兼容历史数据，新观察不得使用。
        page_fingerprint 用作者名和画面主标题等可见信息组成短字符串。
抖音页面分类必须遵守：顶部出现“综合/视频/用户/商品/直播”横向标签并显示多宫格结果时，
必须是 douyin_search_results，绝不能填 douyin_home；douyin_home/douyin_video 只用于普通单视频
推荐流或首页，不得用于搜索结果网格。
抖音搜索结果页需要单独核验目标：search_query_text 只能抄写顶部结果页搜索框内部文字，
只有该搜索框完整清晰可见时 search_query_verified=true，否则为null/false；不得从历史记录、
搜索建议或结果正文复制。search_results_relevant 只有在至少两条清晰可见的独立结果直接
支持当前 keyword 时才为true，并把原画面中支持判断的短文字逐条写入
search_result_evidence；无关、证据不足或仅凭页面类型时必须为false/null。

只返回一个 JSON 对象，严禁出现 action、next_action、plan、step、tap、swipe 字段：
{{
          "base_state":"unknown", "overlays":[],
          "confidence":0.0, "stable":false, "reason":"",
  "page_title":null, "input_text":null, "input_is_empty":null, "input_scope":"unverified",
  "input_focused":null, "input_bounds":null,
  "composition_text":null, "candidate_text":null,
  "exact_match_count":null, "keyboard_visible":false,
  "keyboard_layout":null, "heart_state":"not_visible",
  "selection_count":null, "sent_message_visible":null,
  "new_image_visible":null, "comment_sent_visible":null,
  "page_fingerprint":"", "targets":{{}}, "target_bounds":{{}}, "visible_texts":[],
  "live_evidence":[], "live_evidence_bounds":{{}}, "ad_evidence":[],
  "search_query_text":null, "search_query_verified":null,
  "search_results_relevant":null, "search_result_evidence":[]
}}
"""
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        content.append(
            {"type": "image_url", "image_url": {"url": overview.data_url}}
        )
        messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
        raw = self.provider._chat(messages, max_tokens=1800)
        model_calls = 1
        self.last_raw_response = raw
        local_douyin_signals: dict[str, Any] = {}
        try:
            payload = _extract_json_object(raw)
            payload, local_douyin_signals = self._attach_local_douyin_signals(
                payload,
                frames[-1],
                operation=operation,
            )
            observation = parse_page_observation(payload)

        except VisionAgentError as exc:
            # A candidate label without its center (or vice versa) is a schema
            # contradiction, not a reason for the controller to guess a tap.
            # Give the observer exactly one chance to repair the JSON while
            # keeping the same four frames and observation-only boundary.
            if "exact_candidate" not in str(exc):
                raise
            repair = (
                "你上一个观察 JSON 自相矛盾。若 candidate_text 非null，必须在 targets 中"
                "同时给出该候选词中心 exact_candidate:[x,y]；若无法确认坐标，则两者都填null。"
                "只重新输出完整观察 JSON，不得建议或执行动作。"
            )
            repaired_raw = self.provider._chat(
                [
                    *messages,
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": repair},
                ],
                max_tokens=1800,
            )
            model_calls += 1
            self.last_raw_response = repaired_raw
            payload = _extract_json_object(repaired_raw)
            payload, local_douyin_signals = self._attach_local_douyin_signals(
                payload,
                frames[-1],
                operation=operation,
            )
            observation = parse_page_observation(payload)

        # The parsed state may deliberately downgrade an unsupported model
        # claim (for example, hallucinated live text) to unknown. Feed that
        # normalized result into ROI refinement so detail inspection can
        # correct the page type instead of being anchored to the raw claim.
        if operation.startswith("douyin."):
            payload["base_state"] = observation.base_state
            payload["overlays"] = list(observation.overlays)
            payload.pop("state", None)

        diagnostics: dict[str, Any] = {
            "strategy": "medium_res_overview_high_res_roi_local_crosscheck",
            "local_stability": local_stability.to_dict(),
            "model_calls": model_calls,
            "images": [overview.metadata()],
            "requested_rois": [],
            "local_douyin_signals": local_douyin_signals,
        }
        roi_requests = self._needs_roi(
            observation,
            operation=operation,
            params=params,
        )
        for roi in roi_requests:
            encoded = build_roi(frames[-1], roi)
            detail_prompt = f"""
你是手机页面观察器。图片仅是原始手机画面的局部ROI，不是完整页面。
全景模型给出的暂定候选状态：{observation.state}。这个候选可能错误，不是已确认事实，
不得为了迎合它而寻找画面中不存在的文字或控件。
ROI名称：{roi.name}
ROI用途：{roi.purpose}
ROI在原完整画面中的范围：{list(roi.bounds)}（完整画面坐标为0到1000）。

只报告这个ROI中实际清晰可见的证据，不得建议动作。
直播证据必须逐字抄入 live_evidence，并在 live_evidence_bounds 中按同一文字给出
只包住该文字的ROI相对矩形；没有可靠边界框就不得填写直播证据。
广告证据逐字抄入 ad_evidence；看不清就留空，绝不能根据页面风格猜字。
商品、价格、销量和购物卡片本身不是广告证据，不得写入 ad_evidence。
只有当暂定候选状态为 unknown 时，才允许根据这个ROI填写 base_state 和 overlays
以补充页面分类；否则 base_state 留空。无论暂定状态是什么，都必须独立如实报告证据。
所有 targets、target_bounds、input_bounds 和 keyboard_layout 锚点都必须使用
“当前ROI自身”的0到1000相对坐标；控制器会映射回完整原图。看不清就省略，禁止猜测。
只允许返回以下JSON字段；没有证据的字段填null或省略：
{{
  "base_state":null, "overlays":null, "confidence":0.0, "reason":null,
  "page_title":null,
  "input_text":null, "input_is_empty":null, "input_scope":"unverified",
  "input_focused":null, "input_bounds":null,
  "composition_text":null, "candidate_text":null,
  "exact_match_count":null, "keyboard_visible":null, "keyboard_layout":null,
  "heart_state":null, "selection_count":null,
  "sent_message_visible":null, "new_image_visible":null,
  "comment_sent_visible":null, "targets":{{}}, "target_bounds":{{}},
  "visible_texts":[], "live_evidence":[], "live_evidence_bounds":{{}}, "ad_evidence":[],
  "search_query_text":null,
  "search_query_verified":null, "search_results_relevant":null,
  "search_result_evidence":[]
}}
严禁返回action、next_action、plan、step、tap、swipe。
"""
            detail_raw = self.provider._chat(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": detail_prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": encoded.data_url},
                            },
                        ],
                    }
                ],
                max_tokens=900,
            )
            self.last_raw_response = detail_raw
            patch = _extract_json_object(detail_raw)
            payload = self._merge_roi_patch(payload, patch, roi)
            observation = parse_page_observation(payload)
            if operation.startswith("douyin."):
                payload["base_state"] = observation.base_state
                payload["overlays"] = list(observation.overlays)
                payload.pop("state", None)
            diagnostics["model_calls"] += 1
            diagnostics["images"].append(encoded.metadata())
            diagnostics["requested_rois"].append(
                {"name": roi.name, "bounds": list(roi.bounds), "purpose": roi.purpose}
            )

        observation = replace(
            observation,
            stable=True,
            reason=(
                f"{observation.reason}；本地稳定性：{local_stability.reason}"
                if observation.reason
                else f"本地稳定性：{local_stability.reason}"
            ),
        )
        self.last_observation_diagnostics = diagnostics
        return observation


def _is_chinese(text: str) -> bool:
    return bool(text) and all("\u3400" <= char <= "\u9fff" for char in text)


def _pinyin(text: str) -> str:
    try:
        from pypinyin import Style, lazy_pinyin
    except ImportError as exc:
        raise VisionAgentError("缺少本地拼音组件 pypinyin，请重新启动网页控制台安装依赖。") from exc
    value = "".join(lazy_pinyin(text, style=Style.NORMAL, errors="strict"))
    value = re.sub(r"[^a-z]", "", value.lower())
    if not value:
        raise VisionAgentError(f"无法在本地生成拼音：{text}")
    return value


class StateGraphController:
    """The only component allowed to select the next physical action."""

    def __init__(self, *, min_confidence: float = 0.72) -> None:
        self.min_confidence = min_confidence

    @staticmethod
    def _tap(
        obs: PageObservation,
        name: str,
        reason: str,
        *,
        expected_text: str | None = None,
        transition: str = "",
    ) -> ControllerAction:
        point = obs.targets.get(name)
        if point is None:
            point = LocalTargetResolver.coarse_coordinate(obs.state, name)
        if point is None:
            return ControllerAction("fail", f"画面中没有可靠定位 {name}：{reason}")
        return ControllerAction(
            "tap",
            reason,
            target=name,
            coordinate=point,
            text=expected_text,
            transition=transition,
        )

    @staticmethod
    def _field_key(operation: str, obs: PageObservation) -> str:
        if obs.state == "wechat_search":
            return "wechat_chat_search"
        if obs.state in {"wechat_chat", "wechat_chat_keyboard"}:
            return "wechat_message"
        if obs.state == "douyin_search":
            return "douyin_search"
        if obs.state == "douyin_comments":
            return "douyin_comment"
        return f"{operation}:{obs.state}"

    def _text_action(
        self,
        obs: PageObservation,
        target_text: str,
        context: dict[str, Any],
    ) -> ControllerAction | None:
        field_key = self._field_key(str(context["operation"]), obs)
        fields = context.setdefault("fields", {})
        field = fields.setdefault(
            field_key,
            {
                "started": False,
                "retypes": 0,
                "awaiting_empty": False,
                "candidate_pending": False,
            },
        )
        actual = obs.input_text
        input_verified = (
            obs.input_scope == "active_input"
            and obs.input_focused is True
            and obs.input_bounds is not None
        )
        if input_verified:
            previous_bounds = field.get("input_bounds")
            if previous_bounds is not None and not _same_input_field(
                tuple(previous_bounds),
                obs.input_bounds,
            ):
                return ControllerAction(
                    "fail",
                    "输入过程中焦点切换到了另一个输入框，已停止。",
                )
            field["input_bounds"] = list(obs.input_bounds)
        # During an IME composition the final field value is intentionally not
        # committed yet, so input_text may be null. Handle that distinct state
        # first and only allow the exact target candidate selected by vision.
        if obs.composition_text:
            if not input_verified or not obs.keyboard_visible:
                return ControllerAction(
                    "fail",
                    "拼音组合态缺少当前输入框焦点、边界或键盘证据。",
                )
            expected_segment = str(field.get("pending_segment") or "")
            if field.get("candidate_pending"):
                expected_text = str(field.get("candidate_expected_text") or "")
                if not input_verified or actual != expected_text:
                    return ControllerAction(
                        "fail",
                        "候选词点击后输入框没有显示预期文字，禁止重复点击候选词。",
                    )
                field.update(
                    {
                        "candidate_pending": False,
                        "candidate_expected_text": "",
                        "pending_segment": None,
                        "confirmed_text": actual,
                    }
                )
                return None
            if (
                expected_segment
                and obs.candidate_text == expected_segment
                and "exact_candidate" in obs.targets
            ):
                confirmed_text = str(field.get("confirmed_text") or "")
                field["candidate_pending"] = True
                field["candidate_expected_text"] = confirmed_text + expected_segment
                return self._tap(
                    obs,
                    "exact_candidate",
                    f"选择逐字一致候选词 {expected_segment}",
                    expected_text=expected_segment,
                )
            return ControllerAction("fail", "拼音组合态未看到逐字完全一致的候选词。")

        if not input_verified or actual is None or obs.input_is_empty is None:
            return ControllerAction("fail", "无法只从当前输入框ROI确认文字，拒绝猜测。")

        if field["awaiting_empty"]:
            if obs.input_is_empty is not True or actual != "":
                return ControllerAction("fail", "退格后输入框未被确认完全为空。")
            field.update(
                {
                    "awaiting_empty": False,
                    "started": False,
                    "confirmed_text": "",
                    "pending_segment": None,
                    "candidate_pending": False,
                    "candidate_expected_text": "",
                }
            )

        if field.get("candidate_pending"):
            expected_text = str(field.get("candidate_expected_text") or "")
            if actual != expected_text:
                return ControllerAction(
                    "fail",
                    "候选词点击后输入框没有显示预期文字，禁止继续或重复点击。",
                )
            field.update(
                {
                    "candidate_pending": False,
                    "candidate_expected_text": "",
                    "pending_segment": None,
                }
            )

        # A task may only continue from text that it causally produced.  Text
        # already present when the task first observes the field belongs to
        # the user or a previous task, even when it happens to equal (or be a
        # prefix of) the new target.  Never append to it or submit it.
        if actual and not field["started"]:
            return ControllerAction(
                "fail",
                "输入框原本存在文字且不是由本任务输入，禁止自动删除、接管、补写或提交。",
            )

        if actual == target_text:
            return None

        if actual and not target_text.startswith(actual):
            if not field["started"]:
                return ControllerAction("fail", "输入框原本存在非本任务文字，禁止自动删除。")
            if field["retypes"] >= 2:
                return ControllerAction("fail", "完整清空重输已达到2次上限。")
            delete_count = editable_character_count(actual)
            if delete_count < 1:
                return ControllerAction("fail", "无法确定要退格的准确字符数。")
            if not obs.keyboard_visible or not obs.keyboard_layout:
                return ControllerAction("fail", "退格前未确认当前输入框键盘及删除键。")
            field["retypes"] += 1
            field["awaiting_empty"] = True
            field["pending_segment"] = None
            field["candidate_pending"] = False
            field["candidate_expected_text"] = ""
            return ControllerAction(
                "clear_text",
                f"输入不一致，准确退格{delete_count}次并从头重输。",
                delete_count=delete_count,
                keyboard_layout=obs.keyboard_layout,
            )

        remaining = target_text[len(actual) :]
        if not remaining:
            return None
        # Segment the complete confirmed target once conceptually, then locate
        # the next unfinished part. Calling the public normalizer on
        # ``remaining`` would strip a legitimate internal leading space.
        consumed = len(actual)
        segment = ""
        for full_segment in split_input_segments(target_text):
            if consumed >= len(full_segment):
                consumed -= len(full_segment)
                continue
            segment = full_segment[consumed:]
            break
        if not segment:
            return ControllerAction("fail", "无法从已验证前缀确定下一输入分段。")
        field["started"] = True
        field["confirmed_text"] = actual
        field["pending_segment"] = segment
        if _is_chinese(segment):
            if not obs.keyboard_visible or not obs.keyboard_layout:
                return ControllerAction("fail", "需要输入中文，但完整键盘尚未被确认。")
            return ControllerAction(
                "type_pinyin",
                f"本地确定性输入拼音：{segment}",
                text=segment,
                pinyin=_pinyin(segment),
                keyboard_layout=obs.keyboard_layout,
            )
        if all(char.isascii() and (char.isalnum() or char == " ") for char in segment):
            if not obs.keyboard_visible:
                return ControllerAction("fail", "输入ASCII前未确认当前输入框键盘可见。")
            return ControllerAction("type_text", f"本地输入ASCII分段：{segment}", text=segment)
        if len(segment) == 1 and obs.candidate_text == segment and "symbol_exact" in obs.targets:
            if not obs.keyboard_visible:
                return ControllerAction("fail", "输入符号前未确认当前输入框键盘可见。")
            return self._tap(obs, "symbol_exact", f"点击画面真实可见的符号 {segment}")
        return ControllerAction("fail", f"当前键盘没有可靠定位待输入符号：{segment}")

    def _douyin_search_text_action(
        self,
        obs: PageObservation,
        target_text: str,
        context: dict[str, Any],
    ) -> ControllerAction | None:
        """Enter a search query without trusting history/suggestion text.

        The seller overlay hides Douyin's top search field on this machine.
        Therefore the controller permits one narrow causal proof chain only:
        this task opened the search page, locally typed the exact pinyin, saw
        and tapped the exact Chinese candidate, then observed composition end.
        Result relevance is verified separately before the task may finish.
        """
        fields = context.setdefault("fields", {})
        field = fields.setdefault(
            "douyin_search",
            {
                "started": False,
                "retypes": 0,
                "awaiting_empty": False,
                "candidate_pending": False,
                "causal_input_confirmed": False,
            },
        )

        # If the actual active field is visible, use the stricter generic
        # engine and its exact ROI comparison.
        if obs.input_scope == "active_input":
            return self._text_action(obs, target_text, context)

        # Text elsewhere on the page (especially history records) is never
        # accepted as editable input evidence.
        if field.get("causal_input_confirmed"):
            return None

        if field.get("candidate_pending"):
            if obs.composition_text:
                return ControllerAction(
                    "fail",
                    "点击候选词后拼音组合仍存在，禁止重复点击或提交搜索。",
                )
            if not obs.keyboard_visible:
                return ControllerAction("fail", "候选词点击后键盘状态无法确认。")
            field.update(
                {
                    "candidate_pending": False,
                    "candidate_expected_text": "",
                    "pending_segment": None,
                    "causal_input_confirmed": True,
                }
            )
            return None

        if obs.composition_text:
            expected_segment = str(field.get("pending_segment") or "")
            if (
                expected_segment
                and obs.candidate_text == expected_segment
                and "exact_candidate" in obs.targets
            ):
                field["candidate_pending"] = True
                field["candidate_expected_text"] = expected_segment
                return self._tap(
                    obs,
                    "exact_candidate",
                    f"选择逐字一致搜索候选词 {expected_segment}",
                    expected_text=expected_segment,
                )
            return ControllerAction("fail", "搜索拼音组合态未看到逐字完全一致候选词。")

        if not context.get("search_open_requested"):
            return ControllerAction("fail", "搜索页不是由本任务打开，禁止接管未知输入框。")
        if not obs.keyboard_visible or not obs.keyboard_layout:
            if "search_input" in obs.targets:
                return self._tap(obs, "search_input", "聚焦抖音搜索框")
            return ControllerAction("fail", "搜索输入框被遮挡且键盘未确认，禁止猜测。")
        if field.get("started"):
            # Local ASCII typing has no candidate selection evidence. It may
            # finish only when the active field becomes visually readable.
            return ControllerAction("fail", "搜索输入框不可见，无法验证本地输入结果。")

        field["started"] = True
        field["confirmed_text"] = ""
        field["pending_segment"] = target_text
        if target_text and all(_is_chinese(char) for char in target_text):
            return ControllerAction(
                "type_pinyin",
                f"本地确定性输入搜索拼音：{target_text}",
                text=target_text,
                pinyin=_pinyin(target_text),
                keyboard_layout=obs.keyboard_layout,
            )
        return ControllerAction(
            "fail",
            "搜索框被遮挡时只允许通过精确中文候选词建立输入证据。",
        )

    @staticmethod
    def _search_result_matches(obs: PageObservation, keyword: str) -> bool:
        expected = "".join(keyword.split()).casefold()
        query = "".join(str(obs.search_query_text or "").split()).casefold()
        if obs.search_query_verified is True and query == expected:
            return True
        evidence = [item for item in obs.search_result_evidence if item.strip()]
        if obs.search_results_relevant is not True or len(evidence) < 2:
            return False
        # At least one independent visible result must contain the exact
        # target. This deterministic gate prevents generic page semantics or
        # an unrelated F1 result page from becoming a false success.
        return any(expected in "".join(item.split()).casefold() for item in evidence)

    def _recovery_action(
        self,
        obs: PageObservation,
        context: dict[str, Any],
    ) -> ControllerAction:
        recovered = int(context.get("recovery_actions", 0))
        if recovered >= 3:
            return ControllerAction(
                "fail",
                "未知页面恢复已达到3个已验证动作上限，停止避免循环。",
            )
        if "close_overlay" in obs.targets:
            return self._tap(
                obs,
                "close_overlay",
                "先关闭当前最上层可见叉键，再重新判断页面",
                transition="overlay_close_requested",
            )
        return ControllerAction(
            "back",
            "未发现可靠叉键，按一次已校准安卓返回键后重新判断页面",
            transition="unknown_back_requested",
        )

    def _known_blocking_overlay_action(
        self,
        obs: PageObservation,
        context: dict[str, Any],
    ) -> ControllerAction:
        """Handle overlays without turning a recognized page into unknown navigation.

        A known base page is valuable state evidence.  Android Back is therefore
        reserved for a genuinely unknown page: using it for a loading mask can
        navigate away from a valid result page and create a submit/back loop.
        """
        if "close_overlay" in obs.targets:
            context.pop("known_overlay_waits", None)
            context.pop("known_overlay_key", None)
            return self._tap(
                obs,
                "close_overlay",
                "已知页面上存在阻塞弹层，先关闭最上层可靠叉键，再重新判断页面",
                transition="overlay_close_requested",
            )

        blocking = set(obs.blocking_overlays)
        if blocking.issubset({"loading"}):
            overlay_key = f"{obs.base_state}|{','.join(obs.blocking_overlays)}"
            if context.get("known_overlay_key") != overlay_key:
                context["known_overlay_key"] = overlay_key
                context["known_overlay_waits"] = 0
            waits = int(context.get("known_overlay_waits", 0))
            if waits < 2:
                context["known_overlay_waits"] = waits + 1
                return ControllerAction(
                    "wait",
                    f"已识别{obs.base_state}，仅等待加载层消失（{waits + 1}/2），禁止按返回键。",
                    wait_seconds=1.5,
                )
            return ControllerAction(
                "fail",
                f"已识别{obs.base_state}，但加载层连续两轮未消失；保留当前页面并安全停止。",
            )

        return ControllerAction(
            "fail",
            f"已识别{obs.base_state}，但存在无可靠叉键的阻塞层"
            f"（{','.join(obs.blocking_overlays)}）；禁止用返回键破坏已知页面。",
        )

    def next_action(
        self,
        observation: PageObservation,
        context: dict[str, Any],
    ) -> ControllerAction:
        obs = observation
        operation = str(context["operation"])
        params = context["params"]
        if not obs.stable:
            return ControllerAction("fail", "四帧画面不稳定，拒绝继续。")
        if obs.confidence < self.min_confidence:
            return ControllerAction(
                "fail",
                f"页面状态置信度{obs.confidence:.2f}不足{self.min_confidence:.2f}。",
            )
        if obs.state == "unknown":
            return self._recovery_action(obs, context)
        if obs.blocking_overlays:
            return self._known_blocking_overlay_action(obs, context)

        # A recognized non-blocking page proves that recovery has returned to
        # the normal state graph. The next future incident gets a fresh cap.
        context.pop("recovery_actions", None)
        context.pop("known_overlay_waits", None)
        context.pop("known_overlay_key", None)

        if obs.state == "android_home":
            icon = "wechat_icon" if operation.startswith("wechat.") else "douyin_icon"
            return self._tap(
                obs,
                icon,
                f"从桌面打开{icon}",
                transition="app_launch_requested",
            )

        if operation.startswith("wechat."):
            return self._wechat_action(obs, context)
        if operation.startswith("douyin."):
            return self._douyin_action(obs, context)
        return ControllerAction("fail", f"状态图不支持操作：{operation}")

    @staticmethod
    def validate_finish(
        observation: PageObservation,
        context: dict[str, Any],
    ) -> None:
        """Second safety gate: terminal states require controller-owned facts."""
        operation = str(context["operation"])
        params = context["params"]
        if operation == "wechat.send_text":
            if not context.get("send_clicked"):
                raise VisionAgentError("微信文字任务没有发送动作证据，禁止成功。")
            if observation.input_is_empty is not True or observation.sent_message_visible is not True:
                raise VisionAgentError("微信文字任务缺少发送后的双重视觉证据。")
            return
        if operation == "wechat.send_album_image":
            if not context.get("image_sent") or observation.new_image_visible is not True:
                raise VisionAgentError("微信图片任务缺少发送动作或新图片证据。")
            return
        if operation == "douyin.search":
            expected_keyword = str(params.get("keyword") or "")
            if (
                not context.get("search_submitted")
                or not context.get("search_results_verified")
                or context.get("verified_search_keyword") != expected_keyword
                or observation.state != "douyin_search_results"
            ):
                raise VisionAgentError("抖音搜索没有完整的输入、提交与目标结果证据，禁止成功。")
            return
        if operation == "douyin.batch_interact":
            target_count = int(params["target_count"])
            completed = list(context.get("completed_fingerprints") or [])
            if len(set(completed)) < target_count:
                raise VisionAgentError("抖音批量任务缺少足够的唯一视频完成证据。")
            if context.get("pending_swipe_from") or context.get("comments_close_requested"):
                raise VisionAgentError("抖音批量任务仍有未验证的页面切换，禁止成功。")
            return
        raise VisionAgentError(f"没有定义成功不变量：{operation}")

    def _wechat_action(self, obs: PageObservation, context: dict[str, Any]) -> ControllerAction:
        operation = str(context["operation"])
        params = context["params"]
        chat_name = str(params["chat_name"])
        if obs.state == "wechat_home":
            return self._tap(obs, "open_search", "打开微信搜索")
        if obs.state == "wechat_search":
            text_action = self._text_action(obs, chat_name, context)
            if text_action is not None:
                if text_action.kind == "fail" and obs.input_is_empty is True and "search_input" in obs.targets:
                    return self._tap(obs, "search_input", "先聚焦微信搜索框")
                return text_action
            if obs.exact_match_count != 1:
                return ControllerAction("fail", "聊天搜索结果不是唯一完全匹配。")
            return self._tap(obs, "exact_chat", f"打开唯一匹配聊天：{chat_name}")
        if obs.state == "wechat_search_results":
            if obs.exact_match_count != 1:
                return ControllerAction("fail", "聊天搜索结果不是唯一完全匹配。")
            return self._tap(obs, "exact_chat", f"打开唯一匹配聊天：{chat_name}")
        if obs.state in {"wechat_chat", "wechat_chat_keyboard"}:
            if obs.page_title != chat_name:
                return ControllerAction("fail", f"微信聊天标题不是目标：{chat_name}")
            if operation == "wechat.send_album_image":
                if context.get("image_sent"):
                    if obs.new_image_visible is True:
                        return ControllerAction("finish", "已在目标聊天看到新的图片消息。")
                    return ControllerAction("fail", "发送后没有验证到新的图片消息。")
                return self._tap(obs, "plus", "打开微信加号菜单")

            target_text = str(params["text"])
            if context.get("send_clicked"):
                if obs.input_is_empty is True and obs.sent_message_visible is True:
                    return ControllerAction("finish", "输入框已清空且新消息逐字可见。")
                return ControllerAction("fail", "发送后未同时验证输入框清空和新消息。")
            if not obs.keyboard_visible:
                return self._tap(obs, "chat_input", "聚焦聊天页底部输入框")
            text_action = self._text_action(obs, target_text, context)
            if text_action is not None:
                return text_action
            return self._tap(obs, "send", "输入框逐字正确，点击发送")
        if obs.state == "wechat_plus_menu" and operation == "wechat.send_album_image":
            return self._tap(obs, "album", "打开相册")
        if obs.state == "wechat_album" and operation == "wechat.send_album_image":
            image_index = int(params["image_index"])
            if obs.selection_count == 1:
                return self._tap(obs, "album_send", "确认只选中一张图片并发送")
            if obs.selection_count not in {0, None}:
                return ControllerAction("fail", "相册选择计数不是0或1，拒绝继续。")
            return self._tap(obs, f"image_{image_index}", f"选择最近第{image_index}张图片")
        return ControllerAction("fail", f"微信状态图不接受当前页面：{obs.state}")

    def _douyin_action(self, obs: PageObservation, context: dict[str, Any]) -> ControllerAction:
        operation = str(context["operation"])
        params = context["params"]
        keyword = params.get("keyword")
        search_required = operation == "douyin.search" or bool(keyword)
        search_flow_done = bool(context.get("search_flow_done"))

        # A search page is task-owned either after the controller explicitly
        # opened it, or after this same task has already started its guarded
        # input transaction.  The latter matters when execution is resumed
        # between typing pinyin and selecting/verifying the exact candidate.
        # Merely seeing a search page is never sufficient ownership evidence.
        search_field = dict((context.get("fields") or {}).get("douyin_search") or {})
        search_input_owned = bool(
            context.get("search_open_requested")
            or search_field.get("started")
            or search_field.get("pending_segment")
            or search_field.get("candidate_pending")
            or search_field.get("causal_input_confirmed")
        )

        # Search is a controller-owned phase. A normal video page can never be
        # treated as a successful search merely because it appeared after the
        # app icon was tapped.
        if (
            search_required
            and not search_flow_done
            and context.get("video_results_requested")
            and obs.state == "douyin_video"
        ):
            context["search_flow_done"] = True
            search_flow_done = True
        if search_required and not search_flow_done and obs.state in {
            "douyin_home",
            "douyin_video",
        }:
            if context.get("search_submitted"):
                return ControllerAction("fail", "搜索已提交但未进入可验证的搜索结果页。")
            return self._tap(
                obs,
                "open_search",
                "打开抖音搜索",
                transition="search_open_requested",
            )
        if search_required and not search_flow_done and obs.state == "douyin_search":
            if not search_input_owned:
                if (
                    context.get("app_opened_by_task")
                    and not context.get("startup_normalize_back_requested")
                ):
                    return ControllerAction(
                        "back",
                        "本任务启动抖音后恢复到旧搜索页；先用一次已校准返回键归一化入口，再重新观察。",
                        transition="startup_normalize_back_requested",
                    )
                return ControllerAction(
                    "fail",
                    "搜索页不是由本任务打开，或启动入口归一化后仍停在旧搜索页；禁止接管未知输入框。",
                )
            target = str(keyword or params.get("keyword") or "")
            text_action = self._douyin_search_text_action(obs, target, context)
            if text_action is not None:
                if text_action.kind == "fail" and obs.input_is_empty is True and "search_input" in obs.targets:
                    return self._tap(obs, "search_input", "聚焦抖音搜索框")
                return text_action
            return self._tap(
                obs,
                "submit_search",
                "搜索词逐字正确，提交搜索",
                transition="search_submitted",
            )
        if search_required and not search_flow_done and obs.state == "douyin_search_results":
            if not context.get("search_submitted"):
                # Do not require the physical stylus to hit Douyin's tiny
                # query-clear icon.  Seller-side centre calibration is not a
                # multi-point screen calibration, so small controls near an
                # edge are not reliable enough for a state-recovery action.
                # Android Back has a dedicated calibrated coordinate and is
                # reversible; after it we observe again and continue from the
                # actual page that appears.
                if context.get("search_recovery_back_requested"):
                    return ControllerAction(
                        "fail",
                        "已执行一次安全返回，但页面仍停留在旧搜索结果；禁止重复操作。",
                    )
                return ControllerAction(
                    "back",
                    "旧搜索结果不属于本任务，使用已校准的Android返回键退出后重新观察。",
                    transition="search_recovery_back_requested",
                )
            target = str(keyword or params.get("keyword") or "")
            if not self._search_result_matches(obs, target):
                return ControllerAction(
                    "fail",
                    f"搜索结果页缺少与目标“{target}”一致的查询框或结果证据。",
                )
            context["search_results_verified"] = True
            context["verified_search_keyword"] = target
            if operation == "douyin.search":
                return ControllerAction("finish", f"抖音搜索结果已验证匹配：{target}")
            return self._tap(
                obs,
                "video_tab",
                "进入视频结果流",
                transition="video_results_requested",
            )
        if obs.state.startswith("douyin_live") or obs.state == "douyin_ad":
            if search_required and not search_flow_done:
                return ControllerAction("back", "搜索阶段误入直播或广告，返回后重新观察。")
            context["pages_checked"] = int(context.get("pages_checked", 0)) + 1
            if context["pages_checked"] > int(params["target_count"]) + 5:
                return ControllerAction("fail", "已达到目标数+5的页面检查上限。")
            return ControllerAction(
                "swipe_up",
                "直播或广告不执行账号动作，直接跳过。",
                transition="page_change_requested",
            )
        if obs.state == "douyin_comments":
            if context.get("comments_close_requested"):
                waits = int(context.get("comments_close_waits", 0))
                if waits >= 1:
                    return ControllerAction("fail", "关闭评论区后页面仍未恢复视频流。")
                context["comments_close_waits"] = waits + 1
                return ControllerAction("wait", "等待评论区关闭动画完成。", wait_seconds=1.5)
            comment_text = str(params.get("comment_text") or "")
            if context.get("comment_submitted"):
                if obs.comment_sent_visible is not True:
                    return ControllerAction("fail", "评论发送后没有验证到新评论。")
                return self._tap(
                    obs,
                    "close_comments",
                    "评论已验证，关闭评论区",
                    transition="comments_close_requested",
                )
            if not obs.keyboard_visible:
                return self._tap(obs, "comment_input", "聚焦评论输入框")
            text_action = self._text_action(obs, comment_text, context)
            if text_action is not None:
                return text_action
            return self._tap(
                obs,
                "comment_send",
                "评论逐字正确，点击发送",
                transition="comment_submitted",
            )
        if obs.state in {"douyin_home", "douyin_video"}:
            if search_required and not search_flow_done:
                return ControllerAction("fail", "搜索阶段尚未完成，禁止在普通视频页执行或结束。")
            target_count = int(params["target_count"])
            completed_fingerprints = context.setdefault("completed_fingerprints", [])
            completed = len(completed_fingerprints)
            context["completed"] = completed
            if completed >= target_count:
                return ControllerAction("finish", f"已完成{completed}个目标视频。")

            fingerprint = obs.page_fingerprint.strip()
            if not fingerprint:
                return ControllerAction("fail", "无法生成当前视频稳定指纹，拒绝重复或跨页操作。")
            pending_swipe_from = str(context.get("pending_swipe_from") or "")
            if pending_swipe_from:
                if fingerprint == pending_swipe_from:
                    waits = int(context.get("page_change_waits", 0))
                    if waits >= 1:
                        return ControllerAction("fail", "上划后视频指纹未变化，禁止重复计数或重复操作。")
                    context["page_change_waits"] = waits + 1
                    return ControllerAction("wait", "等待上划后的新视频稳定。", wait_seconds=1.5)
                context.pop("pending_swipe_from", None)
                context["page_change_waits"] = 0

            if context.get("comments_close_requested"):
                context["comments_close_requested"] = False
                context["comments_close_waits"] = 0
                context["comment_submitted"] = False
                context["comment_done"] = True

            if context.get("active_video") != fingerprint:
                context["active_video"] = fingerprint
                context["pages_checked"] = int(context.get("pages_checked", 0)) + 1
                if context["pages_checked"] > target_count + 5:
                    return ControllerAction("fail", "已达到目标数+5的页面检查上限。")
                context["like_done"] = not bool(params.get("like"))
                context["like_pending"] = False
                context["comment_done"] = not bool(params.get("comment"))
                context["comment_submitted"] = False

            if not context["like_done"]:
                if obs.heart_state == "liked":
                    context["like_done"] = True
                elif obs.heart_state == "unliked":
                    if context.get("like_pending"):
                        return ControllerAction("fail", "点赞后爱心仍明确为未点赞，拒绝重复点击。")
                    return self._tap(
                        obs,
                        "heart",
                        "爱心明确为未点赞，执行一次点赞",
                        transition="like_requested",
                    )
                else:
                    return ControllerAction("fail", "无法可靠判断当前爱心状态。")
            if not context["comment_done"]:
                return self._tap(obs, "comments", "打开当前视频评论区")

            if fingerprint not in completed_fingerprints:
                completed_fingerprints.append(fingerprint)
            context["completed"] = len(completed_fingerprints)
            if context["completed"] >= target_count:
                return ControllerAction("finish", f"已完成{context['completed']}个目标视频。")
            return ControllerAction(
                "swipe_up",
                "当前视频动作已验证，进入下一条。",
                transition="page_change_requested",
            )
        return ControllerAction("fail", f"抖音状态图不接受当前页面：{obs.state}")

    @staticmethod
    def _observation_signature(observation: PageObservation) -> dict[str, Any]:
        return {
            "state": observation.state,
            "base_state": observation.base_state,
            "overlays": list(observation.overlays),
            "page_fingerprint": observation.page_fingerprint,
            "input_text": observation.input_text,
            "composition_text": observation.composition_text,
            "input_scope": observation.input_scope,
            "selection_count": observation.selection_count,
            "heart_state": observation.heart_state,
        }

    @classmethod
    def apply_action_result(
        cls,
        action: ControllerAction,
        context: dict[str, Any],
        observation: PageObservation,
    ) -> None:
        """Record an issued action, never an unverified success fact."""

        if action.kind == "wait":
            return
        context["pending_action"] = {
            "kind": action.kind,
            "target": action.target,
            "transition": action.transition,
            "text": action.text,
            "attempts": 0,
            "before": cls._observation_signature(observation),
        }

    @staticmethod
    def _commit_confirmed_action(pending: dict[str, Any], context: dict[str, Any]) -> None:
        transition = str(pending.get("transition") or "")
        target = str(pending.get("target") or "")
        if transition == "app_launch_requested":
            context["app_opened_by_task"] = True
        elif transition == "startup_normalize_back_requested":
            context["startup_normalize_back_requested"] = True
        elif transition == "search_open_requested":
            context["search_open_requested"] = True
        elif transition == "search_recovery_back_requested":
            context["search_recovery_back_requested"] = True
        elif transition == "search_submitted":
            context["search_submitted"] = True
        elif transition == "video_results_requested":
            context["video_results_requested"] = True
        elif transition == "page_change_requested":
            context["pending_swipe_from"] = str(context.get("active_video") or "")
            context["page_change_waits"] = 0
        elif transition == "comments_close_requested":
            context["comments_close_requested"] = True
            context["comments_close_waits"] = 0
        elif transition in {"overlay_close_requested", "unknown_back_requested"}:
            context["recovery_actions"] = int(context.get("recovery_actions", 0)) + 1
        if pending.get("kind") == "tap":
            if target == "send":
                context["send_clicked"] = True
            elif target == "album_send":
                context["image_sent"] = True
            elif target == "heart":
                context["like_pending"] = True
            elif target == "comment_send":
                context["comment_submitted"] = True

    @staticmethod
    def _pending_action_confirmed(
        pending: dict[str, Any],
        observation: PageObservation,
    ) -> bool:
        before = dict(pending.get("before") or {})
        kind = str(pending.get("kind") or "")
        target = str(pending.get("target") or "")
        state_changed = observation.state != before.get("state")
        overlays_changed = list(observation.overlays) != list(before.get("overlays") or [])
        fingerprint_changed = bool(
            observation.page_fingerprint
            and observation.page_fingerprint != before.get("page_fingerprint")
        )
        input_changed = (
            observation.input_scope == "active_input"
            and observation.input_focused is True
            and observation.input_bounds is not None
            and observation.input_text != before.get("input_text")
        )
        composition_changed = bool(
            observation.composition_text
            and observation.composition_text != before.get("composition_text")
        )

        if kind == "clear_text":
            return (
                observation.input_scope == "active_input"
                and observation.input_focused is True
                and observation.input_bounds is not None
                and observation.input_is_empty is True
                and observation.input_text == ""
            )
        if kind in {"type_text", "type_pinyin"}:
            return input_changed or composition_changed
        if kind == "swipe_up":
            return fingerprint_changed
        if kind in {"back", "home"}:
            return state_changed or overlays_changed or fingerprint_changed
        if kind != "tap":
            return False

        if target in {"chat_input", "search_input", "comment_input"}:
            return (
                observation.keyboard_visible
                and observation.input_scope == "active_input"
                and observation.input_focused is True
                and observation.input_bounds is not None
            )
        if target == "exact_candidate":
            return input_changed or (
                bool(before.get("composition_text"))
                and not observation.composition_text
            )
        if target == "symbol_exact":
            return input_changed
        if target == "send":
            return observation.input_is_empty is True and observation.sent_message_visible is True
        if target == "album_send":
            return observation.new_image_visible is True
        if target == "heart":
            return observation.heart_state == "liked"
        if target == "comment_send":
            return observation.comment_sent_visible is True
        if target == "close_comments":
            return observation.state != "douyin_comments" and "douyin_comments" not in observation.overlays
        if target == "plus":
            return observation.state == "wechat_plus_menu"
        if target == "album":
            return observation.state == "wechat_album"
        if target.startswith("image_") or target == "selected_image":
            return observation.selection_count == 1
        if target == "exact_chat":
            return observation.state in {"wechat_chat", "wechat_chat_keyboard"}
        if target == "open_search":
            return observation.state in {"wechat_search", "douyin_search"}
        if target == "submit_search":
            return observation.state == "douyin_search_results"
        if target == "video_tab":
            return observation.state in {"douyin_home", "douyin_video"}
        if target == "comments":
            return observation.state == "douyin_comments"
        if target == "wechat_icon":
            return observation.state.startswith("wechat_")
        if target == "douyin_icon":
            return observation.state.startswith("douyin_")
        return state_changed or overlays_changed or fingerprint_changed

    @classmethod
    def confirm_action_result(
        cls,
        observation: PageObservation,
        context: dict[str, Any],
    ) -> ControllerAction | None:
        pending = context.get("pending_action")
        if not isinstance(pending, dict):
            return None
        if cls._pending_action_confirmed(pending, observation):
            context.pop("pending_action", None)
            cls._commit_confirmed_action(pending, context)
            return None
        attempts = int(pending.get("attempts", 0))
        if attempts < 1:
            pending["attempts"] = attempts + 1
            return ControllerAction(
                "wait",
                f"动作 {pending.get('kind')}:{pending.get('target') or '-'} 尚未观察到结果，等待一次。",
                wait_seconds=1.5,
            )
        return ControllerAction(
            "fail",
            f"动作 {pending.get('kind')}:{pending.get('target') or '-'} 连续两次未达到预期结果。",
        )


class StateGraphRunner:
    def __init__(
        self,
        robot: Any,
        observer: PageObserver,
        *,
        min_confidence: float = 0.72,
        max_actions: int = 160,
        observation_frames: int = 4,
        observation_seconds: float = 1.5,
    ) -> None:
        self.robot = robot
        self.observer = observer
        self.controller = StateGraphController(min_confidence=min_confidence)
        self.max_actions = max_actions
        self.observation_frames = max(4, observation_frames)
        self.observation_seconds = max(1.5, observation_seconds)
        self.target_resolver = LocalTargetResolver()

    def status(self) -> dict[str, Any]:
        value = dict(self.observer.status())
        value.update(
            {
                "min_confidence": self.controller.min_confidence,
                "max_actions": self.max_actions,
                "observation_frames": self.observation_frames,
                "observation_seconds": self.observation_seconds,
            }
        )
        return value

    def _observe_frames(self) -> list[Image.Image]:
        frames: list[Image.Image] = []
        interval = self.observation_seconds / (self.observation_frames - 1)
        deadline = time.monotonic() + self.observation_seconds + 5.0
        while len(frames) < self.observation_frames and time.monotonic() < deadline:
            self.robot._checkpoint()
            frame = self.robot.vision_capture()
            if frame.width < 400 or frame.height < 700:
                frames.clear()
                self.robot._sleep(0.5)
                continue
            frames.append(frame)
            if len(frames) < self.observation_frames:
                self.robot._sleep(interval)
        if len(frames) != self.observation_frames:
            raise VisionAgentError("摄像头连续返回残缺画面，拒绝状态判断。")
        return frames

    @staticmethod
    def _write_json(path: Path, value: dict[str, Any]) -> None:
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _bump(mapping: dict[str, Any], key: str, amount: int = 1) -> None:
        mapping[key] = int(mapping.get(key, 0)) + amount

    @classmethod
    def _reliability_metrics(
        cls,
        context: dict[str, Any],
        *,
        started_at: float,
        finished_at: float,
        success: bool,
        failure_stage: str | None = None,
    ) -> dict[str, Any]:
        raw = dict(context.get("metrics") or {})
        issued = int(raw.get("actions_issued", 0))
        confirmed = int(raw.get("actions_confirmed", 0))
        recovery_issued = int(raw.get("recovery_actions_issued", 0))
        recovery_confirmed = int(raw.get("recovery_actions_confirmed", 0))
        return {
            "success": bool(success),
            "duration_seconds": round(max(0.0, finished_at - started_at), 3),
            "failure_stage": failure_stage,
            "actions_issued": issued,
            "actions_confirmed": confirmed,
            "action_confirmation_rate": round(confirmed / issued, 4) if issued else 1.0,
            "confirmation_waits": int(raw.get("confirmation_waits", 0)),
            "confirmation_failures": int(raw.get("confirmation_failures", 0)),
            "action_kinds": dict(raw.get("action_kinds") or {}),
            "tap_targets": dict(raw.get("tap_targets") or {}),
            "target_resolution_methods": dict(raw.get("target_resolution_methods") or {}),
            "recovery_actions_issued": recovery_issued,
            "recovery_actions_confirmed": recovery_confirmed,
            "recovery_confirmation_rate": (
                round(recovery_confirmed / recovery_issued, 4)
                if recovery_issued
                else 1.0
            ),
            "unknown_observations": int(raw.get("unknown_observations", 0)),
            "blocking_overlay_observations": int(
                raw.get("blocking_overlay_observations", 0)
            ),
            "states_seen": dict(raw.get("states_seen") or {}),
            "input_retypes": sum(
                int(field.get("retypes", 0))
                for field in dict(context.get("fields") or {}).values()
                if isinstance(field, dict)
            ),
        }

    def _execute_action(
        self,
        action: ControllerAction,
        resolution: TargetResolution | None = None,
    ) -> dict[str, Any]:
        command_point: tuple[int, int] | None = None
        if action.kind == "tap":
            assert action.coordinate is not None
            coordinate = (
                resolution.resolved_coordinate
                if resolution is not None
                else action.coordinate
            )
            command_point = self.robot.vision_tap_relative(*coordinate)
        elif action.kind == "type_text":
            assert action.text is not None
            self.robot.vision_type_text(action.text)
        elif action.kind == "type_pinyin":
            assert action.text is not None and action.pinyin is not None
            self.robot.vision_type_pinyin(
                action.text,
                action.pinyin,
                action.keyboard_layout,
            )
        elif action.kind == "clear_text":
            self.robot.vision_clear_text(action.keyboard_layout, action.delete_count)
        elif action.kind == "swipe_up":
            self.robot.vision_swipe_up()
        elif action.kind == "back":
            command_point = self.robot.vision_android_back()
        elif action.kind == "home":
            command_point = self.robot.vision_android_home()
        elif action.kind == "wait":
            pass
        else:
            raise VisionAgentError(f"控制器不能执行动作：{action.kind}")
        return {
            "command_point_px": list(command_point) if command_point is not None else None,
            "physical_click_requested": command_point is not None,
        }

    def execute(self, operation: str, params: dict[str, Any]) -> dict[str, Any]:
        if operation not in {
            "wechat.send_text",
            "wechat.send_album_image",
            "douyin.search",
            "douyin.batch_interact",
        }:
            raise VisionAgentError(f"状态图尚未接入操作：{operation}")
        status = self.observer.status()
        if not status.get("configured"):
            raise VisionAgentError(str(status.get("error") or "视觉模型未配置。"))

        source_params = dict(params.get("source_params") or params)
        run_dir = self.robot._new_run_dir(f"state_graph.{operation}")
        report_path = run_dir / "report.json"
        context: dict[str, Any] = {
            "architecture": "page_state_graph_v1",
            "operation": operation,
            "params": source_params,
            "completed": 0,
            "observations": 0,
            "fields": {},
            "metrics": {
                "actions_issued": 0,
                "actions_confirmed": 0,
                "confirmation_waits": 0,
                "confirmation_failures": 0,
                "action_kinds": {},
                "tap_targets": {},
                "target_resolution_methods": {},
                "recovery_actions_issued": 0,
                "recovery_actions_confirmed": 0,
                "unknown_observations": 0,
                "blocking_overlay_observations": 0,
                "states_seen": {},
            },
        }
        log: list[dict[str, Any]] = []
        evidence: list[str] = []
        started = time.time()
        current_stage = "initialization"
        try:
            with self.robot.operation_lock:
                self.robot.clear_stop()
                for index in range(1, self.max_actions + 1):
                    current_stage = "capture"
                    frames = self._observe_frames()
                    context["observations"] = index
                    image_path = run_dir / f"{index:03d}_observe.jpg"
                    self.robot._save_frame(frames[-1], image_path, f"state graph observe {index}")
                    evidence.append(str(image_path))
                    current_stage = "perception"
                    observation = self.observer.observe(
                        operation=operation,
                        params=source_params,
                        frames=frames,
                        controller_context={
                            "completed": context.get("completed", 0),
                            "send_clicked": context.get("send_clicked", False),
                            "image_sent": context.get("image_sent", False),
                            "like_done": context.get("like_done"),
                            "comment_done": context.get("comment_done"),
                            "comment_submitted": context.get("comment_submitted", False),
                            "search_open_requested": context.get("search_open_requested", False),
                            "search_submitted": context.get("search_submitted", False),
                            "verified_search_keyword": context.get("verified_search_keyword"),
                            "fields": context.get("fields", {}),
                        },
                    )
                    metrics = context["metrics"]
                    self._bump(metrics["states_seen"], observation.state)
                    if observation.state == "unknown" or observation.base_state == "unknown":
                        metrics["unknown_observations"] += 1
                    if observation.blocking_overlays:
                        metrics["blocking_overlay_observations"] += 1

                    pending_before = context.get("pending_action")
                    current_stage = "postcondition_confirmation"
                    confirmation_action = self.controller.confirm_action_result(
                        observation,
                        context,
                    )
                    if isinstance(pending_before, dict):
                        if confirmation_action is None and "pending_action" not in context:
                            metrics["actions_confirmed"] += 1
                            if str(pending_before.get("transition") or "") in {
                                "overlay_close_requested",
                                "unknown_back_requested",
                            }:
                                metrics["recovery_actions_confirmed"] += 1
                        elif confirmation_action is not None and confirmation_action.kind == "wait":
                            metrics["confirmation_waits"] += 1
                        elif confirmation_action is not None and confirmation_action.kind == "fail":
                            metrics["confirmation_failures"] += 1

                    if confirmation_action is not None:
                        action = confirmation_action
                    else:
                        current_stage = "controller_decision"
                        action = self.controller.next_action(observation, context)
                    entry = {
                        "index": index,
                        "observation": observation.to_dict(),
                        "controller_action": action.to_dict(),
                    }
                    log.append(entry)
                    if action.kind == "finish":
                        current_stage = "finish_validation"
                        self._write_json(run_dir / f"{index:03d}_decision.json", entry)
                        self.controller.validate_finish(observation, context)
                        finished = time.time()
                        reliability = self._reliability_metrics(
                            context,
                            started_at=started,
                            finished_at=finished,
                            success=True,
                        )
                        result = {
                            "success": True,
                            "architecture": "page_state_graph_v1",
                            "operation": operation,
                            "actions": len(log) - 1,
                            "observations": len(log),
                            "completed": context.get("completed", 0),
                            "evidence": evidence,
                            "report": str(report_path),
                            "reliability_metrics": reliability,
                        }
                        self._write_json(
                            report_path,
                            {**result, "started_at": started, "finished_at": finished, "log": log},
                        )
                        return result
                    if action.kind == "fail":
                        self._write_json(run_dir / f"{index:03d}_decision.json", entry)
                        raise VisionAgentError(action.reason)
                    resolution: TargetResolution | None = None
                    if action.kind == "tap":
                        current_stage = "target_resolution"
                        assert action.coordinate is not None
                        resolution_params = dict(source_params)
                        if action.target == "exact_candidate" and action.text:
                            # This private value is emitted only after the
                            # controller has matched the current IME segment
                            # against the observed exact candidate.  The
                            # locator may use it for a guarded semantic/box
                            # fallback when camera moire defeats Windows OCR.
                            resolution_params["_expected_candidate_text"] = action.text
                        resolution = self.target_resolver.resolve(
                            frames=frames,
                            observation=observation,
                            target=action.target,
                            proposed=action.coordinate,
                            params=resolution_params,
                        )
                        entry["target_resolution"] = resolution.to_dict()
                        self._bump(
                            metrics["target_resolution_methods"],
                            resolution.method,
                        )
                    self._write_json(run_dir / f"{index:03d}_decision.json", entry)
                    current_stage = "action_execution"
                    execution = self._execute_action(action, resolution)
                    entry["execution"] = execution
                    # Persist the point actually delivered to main.exe. This
                    # lets reports distinguish perception errors from physical
                    # contact failures without inferring from a later frame.
                    self._write_json(run_dir / f"{index:03d}_decision.json", entry)
                    self.controller.apply_action_result(action, context, observation)
                    if action.kind != "wait":
                        metrics["actions_issued"] += 1
                        self._bump(metrics["action_kinds"], action.kind)
                        if action.kind == "tap":
                            self._bump(metrics["tap_targets"], action.target or "-")
                        if action.transition in {
                            "overlay_close_requested",
                            "unknown_back_requested",
                        }:
                            metrics["recovery_actions_issued"] += 1
                    current_stage = "settle"
                    settle = action.wait_seconds
                    if action.kind in {"type_text", "type_pinyin", "clear_text"}:
                        settle = max(settle, 3.0)
                    elif action.kind in {"tap", "swipe_up", "back", "home"}:
                        settle = max(settle, 1.2)
                    self.robot._sleep(settle)
            raise VisionAgentError("状态图动作数达到上限，已停止。")
        except Exception as exc:
            raw_observation = getattr(self.observer, "last_raw_response", "")
            finished = time.time()
            reliability = self._reliability_metrics(
                context,
                started_at=started,
                finished_at=finished,
                success=False,
                failure_stage=current_stage,
            )
            self._write_json(
                report_path,
                {
                    "success": False,
                    "architecture": "page_state_graph_v1",
                    "operation": operation,
                    "error": str(exc),
                    "started_at": started,
                    "finished_at": finished,
                    "failure_stage": current_stage,
                    "reliability_metrics": reliability,
                    "context": context,
                    "evidence": evidence,
                    "log": log,
                    "raw_observation": raw_observation,
                },
            )
            if isinstance(exc, VisionAgentError):
                exc.report_path = str(report_path)
                raise
            wrapped = VisionAgentError(f"状态图执行失败：{type(exc).__name__}: {exc}")
            wrapped.report_path = str(report_path)
            raise wrapped from exc


class ScriptedPageObserver:
    """Offline test observer; never calls Qwen or hardware."""

    def __init__(self, observations: list[PageObservation]) -> None:
        self.observations = list(observations)

    def status(self) -> dict[str, Any]:
        return {
            "configured": True,
            "execution_architecture": "page_state_graph_v1",
            "model_role": "observation_only",
        }

    def observe(self, **_: Any) -> PageObservation:
        if not self.observations:
            raise VisionAgentError("模拟页面观察已用尽。")
        return self.observations.pop(0)
