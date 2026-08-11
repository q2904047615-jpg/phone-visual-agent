from __future__ import annotations

import math
import re
import statistics
from dataclasses import asdict, dataclass
from typing import Any, Callable

from PIL import Image

import robot_gui_poc as legacy
from ocr_runtime import OcrMatch, find_text as find_ocr_text, recognize as recognize_ocr
from vision_agent import VisionAgentError


@dataclass(frozen=True)
class OcrTargetSpec:
    text: str
    region: tuple[float, float, float, float]
    required_samples: int = 2
    max_spread_px: int = 18
    # OCR locates glyphs, while a reliable touch should land inside the
    # control's hit box. Express the safe-point adjustment in text heights so
    # it scales with preview resolution instead of using a global pixel offset.
    safe_offset_text_heights: tuple[float, float] = (0.0, 0.0)
    # Text buttons are clicked at their OCR glyph-derived safe point. Icon
    # labels are different: the glyph confirms identity, but the clickable
    # body is the independently validated model box above/around that label.
    click_geometry: str = "ocr_center"


@dataclass(frozen=True)
class TargetResolution:
    target: str
    method: str
    proposed_coordinate: tuple[int, int]
    resolved_coordinate: tuple[int, int]
    pixel_center: tuple[int, int]
    samples: int
    spread_px: float
    detail: str
    verified_state: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["proposed_coordinate"] = list(self.proposed_coordinate)
        value["resolved_coordinate"] = list(self.resolved_coordinate)
        value["pixel_center"] = list(self.pixel_center)
        return value


# These are semantic guards, not calibrated click points. They only describe
# where a control is allowed to exist on the phone page. A target outside its
# page-specific region is rejected before the arm can move.
TARGET_REGIONS: dict[tuple[str, str], tuple[float, float, float, float]] = {
    ("android_home", "wechat_icon"): (0.02, 0.05, 0.98, 0.92),
    ("android_home", "douyin_icon"): (0.02, 0.05, 0.98, 0.92),
    ("wechat_home", "open_search"): (0.55, 0.00, 1.00, 0.20),
    ("wechat_search", "search_input"): (0.05, 0.00, 0.95, 0.20),
    ("wechat_search", "exact_chat"): (0.02, 0.08, 0.98, 0.80),
    ("wechat_search_results", "exact_chat"): (0.02, 0.08, 0.98, 0.85),
    ("wechat_chat", "chat_input"): (0.02, 0.72, 0.86, 0.98),
    ("wechat_chat", "send"): (0.65, 0.72, 1.00, 0.98),
    ("wechat_chat", "plus"): (0.78, 0.72, 1.00, 0.98),
    ("wechat_chat_keyboard", "chat_input"): (0.02, 0.40, 0.86, 0.72),
    ("wechat_chat_keyboard", "send"): (0.65, 0.40, 1.00, 0.75),
    ("wechat_chat_keyboard", "plus"): (0.78, 0.40, 1.00, 0.75),
    ("wechat_plus_menu", "album"): (0.00, 0.45, 1.00, 1.00),
    ("wechat_album", "album_send"): (0.55, 0.00, 1.00, 1.00),
    ("wechat_album", "selected_image"): (0.02, 0.08, 0.98, 0.92),
    ("douyin_home", "open_search"): (0.55, 0.00, 1.00, 0.18),
    ("douyin_video", "open_search"): (0.55, 0.00, 1.00, 0.18),
    ("douyin_search_results", "open_search"): (0.70, 0.00, 0.98, 0.10),
    # The old-query clear icon is visible immediately to the left of the
    # submit-search label. Resolve it from that label instead of guessing a
    # point inside the field hidden by the seller PX/MM overlay.
    ("douyin_search_results", "clear_search"): (0.68, 0.00, 0.77, 0.08),
    ("douyin_search_results", "video_tab"): (0.05, 0.02, 0.55, 0.18),
    ("douyin_search", "search_input"): (0.02, 0.00, 0.92, 0.22),
    ("douyin_search", "submit_search"): (0.55, 0.00, 1.00, 1.00),
    ("douyin_video", "heart"): (0.65, 0.12, 1.00, 0.78),
    ("douyin_video", "comments"): (0.65, 0.18, 1.00, 0.88),
    ("douyin_comments", "comment_input"): (0.00, 0.55, 0.90, 1.00),
    ("douyin_comments", "comment_send"): (0.55, 0.45, 1.00, 1.00),
    ("douyin_comments", "close_comments"): (0.55, 0.00, 1.00, 0.55),
}


# Page-owned controls that cannot be visually re-found because the seller
# preview draws an opaque overlay over them.  These are normalized page-layout
# anchors, not model guesses.  They are still allowed only after the page state
# itself has passed the multi-frame visual gate.
FIXED_PAGE_TARGETS: dict[tuple[str, str], tuple[int, int]] = {}


STATIC_OCR_TARGETS: dict[tuple[str, str], OcrTargetSpec] = {
    ("android_home", "wechat_icon"): OcrTargetSpec(
        "微信", (0.02, 0.05, 0.98, 0.92), click_geometry="model_box_center"
    ),
    ("android_home", "douyin_icon"): OcrTargetSpec(
        "抖音", (0.02, 0.05, 0.98, 0.92), click_geometry="model_box_center"
    ),
    ("douyin_search_results", "open_search"): OcrTargetSpec(
        "搜索",
        (0.70, 0.00, 0.98, 0.10),
        safe_offset_text_heights=(0.0, 0.75),
    ),
    ("douyin_search_results", "clear_search"): OcrTargetSpec(
        "搜索",
        (0.65, 0.00, 0.98, 0.10),
        # Current Douyin places the clear X about 3.2 label-heights to the
        # left of the submit label. This is remeasured on every frame.
        safe_offset_text_heights=(-3.2, 0.0),
    ),
    ("douyin_search_results", "video_tab"): OcrTargetSpec("视频", (0.05, 0.02, 0.55, 0.18)),
    ("douyin_search", "submit_search"): OcrTargetSpec("搜索", (0.55, 0.00, 1.00, 1.00)),
    ("wechat_chat_keyboard", "send"): OcrTargetSpec("发送", (0.65, 0.40, 1.00, 0.75)),
    ("wechat_plus_menu", "album"): OcrTargetSpec("相册", (0.00, 0.45, 1.00, 1.00)),
    ("wechat_album", "album_send"): OcrTargetSpec("发送", (0.55, 0.00, 1.00, 1.00)),
    ("douyin_comments", "comment_send"): OcrTargetSpec("发送", (0.55, 0.45, 1.00, 1.00)),
}


def _inside(
    point: tuple[int, int],
    region: tuple[float, float, float, float],
) -> bool:
    x, y = point
    left, top, right, bottom = region
    return left * 1000 <= x <= right * 1000 and top * 1000 <= y <= bottom * 1000


def _target_region(state: str, target: str) -> tuple[float, float, float, float] | None:
    if target == "close_overlay":
        # Recovery is permitted only for a genuine top/right close control.
        # The model must still provide the control's own tight bounds.
        return (0.55, 0.00, 1.00, 0.45)
    region = TARGET_REGIONS.get((state, target))
    if region is not None:
        return region
    if state == "wechat_album" and (
        target == "selected_image"
        or bool(re.fullmatch(r"image_[1-9]|image_1[0-9]|image_20", target))
    ):
        return (0.02, 0.08, 0.98, 0.92)
    if target in {"exact_candidate", "symbol_exact"} and state in {
        "wechat_search",
        "wechat_chat_keyboard",
        "douyin_search",
        "douyin_comments",
    }:
        # Candidate words and symbol keys are dynamic controls rather than
        # static page buttons.  They may be clicked only while a verified
        # keyboard-bearing input state owns the page, and only inside the
        # candidate/keyboard band that is independently searched by OCR.
        return (0.00, 0.45, 1.00, 0.90)
    return None


def _box_inside_region(
    bounds: tuple[int, int, int, int],
    region: tuple[float, float, float, float],
) -> bool:
    left, top, right, bottom = bounds
    return _inside((left, top), region) and _inside((right, bottom), region)


def _pixel_to_relative(point: tuple[int, int], frame: Image.Image) -> tuple[int, int]:
    return (
        int(round(point[0] * 1000 / max(1, frame.width - 1))),
        int(round(point[1] * 1000 / max(1, frame.height - 1))),
    )


def _relative_to_pixel(point: tuple[int, int], frame: Image.Image) -> tuple[int, int]:
    return (
        int(round(point[0] * (frame.width - 1) / 1000)),
        int(round(point[1] * (frame.height - 1) / 1000)),
    )


def _median_center(centers: list[tuple[int, int]]) -> tuple[int, int]:
    return (
        int(round(statistics.median(point[0] for point in centers))),
        int(round(statistics.median(point[1] for point in centers))),
    )


def _spread(centers: list[tuple[int, int]], center: tuple[int, int]) -> float:
    return max(
        (math.hypot(point[0] - center[0], point[1] - center[1]) for point in centers),
        default=0.0,
    )


class LocalTargetResolver:
    """Turn a semantic target into a verified physical click point.

    Qwen's point is only a coarse hint. Text controls are re-found with local
    Windows OCR on multiple frames; the Douyin heart is re-found with the
    deterministic local color/shape detector. Unmapped controls keep the model
    point only after a page-specific region guard.
    """

    def __init__(
        self,
        *,
        ocr_recognizer: Callable[..., dict[str, Any]] = recognize_ocr,
        heart_detector: Callable[[Image.Image], Any] = legacy.detect_douyin_heart,
    ) -> None:
        self.ocr_recognizer = ocr_recognizer
        self.heart_detector = heart_detector

    @staticmethod
    def fixed_coordinate(state: str, target: str) -> tuple[int, int] | None:
        return FIXED_PAGE_TARGETS.get((state, target))

    @staticmethod
    def coarse_coordinate(state: str, target: str) -> tuple[int, int] | None:
        """Return a non-executable hint for a locally resolvable target.

        The controller may use this only to start local OCR resolution when
        the model omitted a point.  The hint is never sent to the robot.
        """
        fixed = FIXED_PAGE_TARGETS.get((state, target))
        if fixed is not None:
            return fixed
        spec = STATIC_OCR_TARGETS.get((state, target))
        if spec is None:
            return None
        left, top, right, bottom = spec.region
        return (
            int(round((left + right) * 500)),
            int(round((top + bottom) * 500)),
        )

    @staticmethod
    def _ocr_spec(
        state: str,
        target: str,
        observation: Any,
        params: dict[str, Any],
    ) -> OcrTargetSpec | None:
        static = STATIC_OCR_TARGETS.get((state, target))
        if static is not None:
            return static
        if target == "exact_chat":
            text = str(params.get("chat_name") or "").strip()
            if text:
                return OcrTargetSpec(text, (0.02, 0.08, 0.98, 0.85))
        if target in {"exact_candidate", "symbol_exact"}:
            text = str(getattr(observation, "candidate_text", None) or "").strip()
            if text:
                return OcrTargetSpec(text, (0.00, 0.45, 1.00, 0.90))
        return None

    @staticmethod
    def _crop_region(
        frame: Image.Image,
        region: tuple[float, float, float, float],
    ) -> tuple[Image.Image, int, int]:
        left, top, right, bottom = region
        x0 = max(0, min(frame.width - 1, int(math.floor(left * frame.width))))
        y0 = max(0, min(frame.height - 1, int(math.floor(top * frame.height))))
        x1 = max(x0 + 1, min(frame.width, int(math.ceil(right * frame.width))))
        y1 = max(y0 + 1, min(frame.height, int(math.ceil(bottom * frame.height))))
        return frame.crop((x0, y0, x1, y1)), x0, y0

    def _resolve_ocr(
        self,
        *,
        frames: list[Image.Image],
        proposed: tuple[int, int],
        target: str,
        spec: OcrTargetSpec,
        model_bounds: tuple[int, int, int, int] | None = None,
        model_safe_center: tuple[int, int] | None = None,
    ) -> TargetResolution:
        centers: list[tuple[int, int]] = []
        frames_with_any_match = 0
        expected = _relative_to_pixel(proposed, frames[-1])
        for frame in frames:
            crop, offset_x, offset_y = self._crop_region(frame, spec.region)
            payload = self.ocr_recognizer(crop, "zh-Hans-CN", scale=3.0)
            matches: list[OcrMatch] = find_ocr_text(payload, spec.text)
            if not matches:
                continue
            translated = []
            for match in matches:
                offset_height_x, offset_height_y = spec.safe_offset_text_heights
                translated.append(
                    (
                        match.center[0]
                        + offset_x
                        + int(round(match.height * offset_height_x)),
                        match.center[1]
                        + offset_y
                        + int(round(match.height * offset_height_y)),
                    )
                )
            frames_with_any_match += 1
            if model_bounds is not None:
                # Duplicate labels are common (for example, a page-level
                # search button and the keyboard search key).  OCR may refine
                # a model target only when both sources refer to the same
                # spatial control.  A distant exact-text match must never
                # override the model's tight clickable bounds.
                left, top, right, bottom = model_bounds
                px_left, px_top = _relative_to_pixel((left, top), frame)
                px_right, px_bottom = _relative_to_pixel((right, bottom), frame)
                box_width = max(1, px_right - px_left)
                box_height = max(1, px_bottom - px_top)
                margin = max(8, int(round(max(box_width, box_height) * 0.25)))
                translated = [
                    point
                    for point in translated
                    if px_left - margin <= point[0] <= px_right + margin
                    and px_top - margin <= point[1] <= px_bottom + margin
                ]
                if not translated:
                    continue
            centers.append(
                min(
                    translated,
                    key=lambda point: math.hypot(
                        point[0] - expected[0], point[1] - expected[1]
                    ),
                )
            )
        if len(centers) < spec.required_samples:
            if (
                model_safe_center is not None
                and frames_with_any_match >= spec.required_samples
            ):
                pixel = _relative_to_pixel(model_safe_center, frames[-1])
                return TargetResolution(
                    target=target,
                    method="qwen_box_ocr_spatial_fallback",
                    proposed_coordinate=proposed,
                    resolved_coordinate=model_safe_center,
                    pixel_center=pixel,
                    samples=frames_with_any_match,
                    spread_px=0.0,
                    detail=(
                        f"OCR多帧确认页面存在文字“{spec.text}”，但仅识别到边界框外的同名控件；"
                        "忽略远处OCR结果并使用已通过页面区域校验的模型边界框中心"
                    ),
                )
            raise VisionAgentError(
                f"本地OCR只在{len(centers)}/{len(frames)}帧识别到“{spec.text}”，"
                f"未达到{spec.required_samples}帧一致性要求；禁止点击{target}。"
            )
        center = _median_center(centers)
        spread = _spread(centers, center)
        if spread > spec.max_spread_px:
            raise VisionAgentError(
                f"本地OCR定位“{spec.text}”的多帧偏差为{spread:.1f}px，画面或控件不稳定；"
                f"禁止点击{target}。"
            )
        if spec.click_geometry == "model_box_center":
            if model_safe_center is None:
                raise VisionAgentError(
                    f"OCR已确认图标标签“{spec.text}”，但模型没有提供可点击图标边界框；"
                    f"禁止点击{target}。"
                )
            pixel = _relative_to_pixel(model_safe_center, frames[-1])
            return TargetResolution(
                target=target,
                method="qwen_box_ocr_identity_fusion",
                proposed_coordinate=proposed,
                resolved_coordinate=model_safe_center,
                pixel_center=pixel,
                samples=len(centers),
                spread_px=round(spread, 2),
                detail=(
                    f"OCR多帧确认图标身份：{spec.text}；"
                    "点击已通过页面区域校验的图标边界框中心，不点击下方文字标签"
                ),
            )
        resolved = _pixel_to_relative(center, frames[-1])
        if not _inside(resolved, spec.region):
            raise VisionAgentError(f"本地OCR定位结果越出{target}允许区域，禁止点击。")
        return TargetResolution(
            target=target,
            method="windows_ocr_multiframe",
            proposed_coordinate=proposed,
            resolved_coordinate=resolved,
            pixel_center=center,
            samples=len(centers),
            spread_px=round(spread, 2),
            detail=(
                f"OCR精确文字：{spec.text}；"
                f"安全内点偏置：{spec.safe_offset_text_heights}x文字高度"
            ),
        )

    @staticmethod
    def _candidate_semantic_box_fallback(
        *,
        observation: Any,
        target: str,
        params: dict[str, Any],
        proposed: tuple[int, int],
        model_box: tuple[tuple[int, int, int, int], tuple[int, int]] | None,
        frames: list[Image.Image],
    ) -> TargetResolution | None:
        if target != "exact_candidate" or model_box is None:
            return None
        expected = str(params.get("_expected_candidate_text") or "").strip()
        candidate = str(getattr(observation, "candidate_text", None) or "").strip()
        visible = [str(item).strip() for item in (getattr(observation, "visible_texts", None) or [])]
        confidence = float(getattr(observation, "confidence", 0.0) or 0.0)
        if (
            not expected
            or candidate != expected
            or expected not in visible
            or getattr(observation, "stable", False) is not True
            or confidence < 0.72
        ):
            return None
        _bounds, safe_center = model_box
        pixel = _relative_to_pixel(safe_center, frames[-1])
        return TargetResolution(
            target=target,
            method="qwen_multiframe_exact_text_box_guarded",
            proposed_coordinate=proposed,
            resolved_coordinate=safe_center,
            pixel_center=pixel,
            samples=len(frames),
            spread_px=0.0,
            detail=(
                f"控制器目标、四帧候选文字和可见文字逐字一致：{expected}；"
                "Windows OCR受相机摩尔纹影响未能精修，使用已通过候选栏区域校验的紧边界框中心"
            ),
        )

    @staticmethod
    def _validated_model_box(
        *,
        state: str,
        target: str,
        proposed: tuple[int, int],
        observation: Any,
    ) -> tuple[tuple[int, int, int, int], tuple[int, int]] | None:
        bounds_map = getattr(observation, "target_bounds", {}) or {}
        bounds = bounds_map.get(target)
        if bounds is None:
            return None
        allowed_region = _target_region(state, target)
        if allowed_region is None:
            raise VisionAgentError(
                f"{state} 的 {target} 尚无页面专属点击区域，禁止采用模型坐标。"
            )
        left, top, right, bottom = tuple(int(item) for item in bounds)
        if not (left <= proposed[0] <= right and top <= proposed[1] <= bottom):
            raise VisionAgentError(
                f"模型给出的{target}中心不在其边界框内，禁止点击。"
            )
        if not _box_inside_region((left, top, right, bottom), allowed_region):
            raise VisionAgentError(
                f"模型给出的{target}边界框越出{state}允许区域，禁止点击。"
            )
        width = right - left
        height = bottom - top
        if width < 8 or height < 8 or width > 500 or height > 350:
            raise VisionAgentError(
                f"模型给出的{target}边界框尺寸异常({width}x{height})，禁止点击。"
            )
        return (left, top, right, bottom), ((left + right) // 2, (top + bottom) // 2)

    def _resolve_heart(
        self,
        *,
        frames: list[Image.Image],
        proposed: tuple[int, int],
        target: str,
    ) -> TargetResolution:
        centers: list[tuple[int, int]] = []
        states: list[str] = []
        for frame in frames:
            detection = self.heart_detector(frame)
            if detection.center and detection.state in {"liked", "unliked"}:
                centers.append(tuple(detection.center))
                states.append(str(detection.state))
        if len(centers) < 2:
            raise VisionAgentError(
                f"本地爱心检测只在{len(centers)}/{len(frames)}帧稳定定位，禁止点击。"
            )
        center = _median_center(centers)
        spread = _spread(centers, center)
        if spread > 18:
            raise VisionAgentError(
                f"本地爱心定位多帧偏差为{spread:.1f}px，禁止点击。"
            )
        resolved = _pixel_to_relative(center, frames[-1])
        region = TARGET_REGIONS[("douyin_video", "heart")]
        if not _inside(resolved, region):
            raise VisionAgentError("本地爱心定位越出右侧爱心区域，禁止点击。")
        state = max(set(states), key=states.count)
        return TargetResolution(
            target=target,
            method="local_heart_multiframe",
            proposed_coordinate=proposed,
            resolved_coordinate=resolved,
            pixel_center=center,
            samples=len(centers),
            spread_px=round(spread, 2),
            detail=f"本地爱心状态：{state}",
            verified_state=state,
        )

    def resolve(
        self,
        *,
        frames: list[Image.Image],
        observation: Any,
        target: str,
        proposed: tuple[int, int],
        params: dict[str, Any],
    ) -> TargetResolution:
        if len(frames) < 2:
            raise VisionAgentError("精确落点校正至少需要2帧画面。")
        state = str(observation.state)
        fixed = self.fixed_coordinate(state, target)
        if fixed is not None:
            region = TARGET_REGIONS[(state, target)]
            if not _inside(fixed, region):
                raise VisionAgentError(f"固定页面内点越出{target}允许区域，禁止点击。")
            pixel = _relative_to_pixel(fixed, frames[-1])
            return TargetResolution(
                target=target,
                method="deterministic_obscured_hitbox",
                proposed_coordinate=proposed,
                resolved_coordinate=fixed,
                pixel_center=pixel,
                samples=len(frames),
                spread_px=0.0,
                detail="卖家PX/MM浮层遮挡视觉目标；使用页面状态专属安全内点",
            )
        spec = self._ocr_spec(state, target, observation, params)
        if spec is not None:
            model_box = self._validated_model_box(
                state=state,
                target=target,
                proposed=proposed,
                observation=observation,
            )
            try:
                return self._resolve_ocr(
                    frames=frames,
                    proposed=proposed,
                    target=target,
                    spec=spec,
                    model_bounds=model_box[0] if model_box else None,
                    model_safe_center=model_box[1] if model_box else None,
                )
            except VisionAgentError:
                fallback = self._candidate_semantic_box_fallback(
                    observation=observation,
                    target=target,
                    params=params,
                    proposed=proposed,
                    model_box=model_box,
                    frames=frames,
                )
                if fallback is not None:
                    return fallback
                raise
        if state == "douyin_video" and target == "heart":
            return self._resolve_heart(
                frames=frames,
                proposed=proposed,
                target=target,
            )

        # Only targets without a local resolver are allowed to retain the
        # model point. Locally resolved targets must not be rejected because
        # Qwen's coarse hint was imprecise; their final point is independently
        # validated inside _resolve_ocr/_resolve_heart.
        allowed_region = _target_region(state, target)
        if allowed_region is None:
            raise VisionAgentError(
                f"{state} 的 {target} 尚无页面专属点击区域，禁止采用模型坐标。"
            )
        if not _inside(proposed, allowed_region):
            raise VisionAgentError(
                f"模型给出的{target}坐标不在{state}允许区域内，禁止点击。"
            )

        model_box = self._validated_model_box(
            state=state,
            target=target,
            proposed=proposed,
            observation=observation,
        )
        if model_box is None:
            raise VisionAgentError(
                f"模型只给出{target}中心但没有可点击边界框，禁止点击。"
            )
        _, safe_center = model_box

        pixel = _relative_to_pixel(safe_center, frames[-1])
        return TargetResolution(
            target=target,
            method="qwen_box_region_guarded",
            proposed_coordinate=proposed,
            resolved_coordinate=safe_center,
            pixel_center=pixel,
            samples=len(frames),
            spread_px=0.0,
            detail="模型中心与可点击边界框相互校验，并点击边界框安全中心",
        )
