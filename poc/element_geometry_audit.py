from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

from PIL import Image


ELEMENT_GEOMETRY_AUDIT_PROTOCOL_VERSION = (
    "2026-08-16-element-geometry-audit-v2"
)
MIN_GEOMETRY_AUDIT_CONFIDENCE = 0.92
DEFAULT_ROI_MIN_SPAN = 0.60
DEFAULT_ROI_TARGET_SCALE = 1.50
LITERAL_ROI_MIN_WIDTH_SPAN = 0.60
LITERAL_ROI_MIN_HEIGHT_SPAN = 0.20
LITERAL_ROI_WIDTH_TARGET_SCALE = 1.50
LITERAL_ROI_HEIGHT_TARGET_SCALE = 3.00
DEFAULT_INTERNAL_EDGE_MARGIN = 20.0
MAX_GEOMETRY_MATCHES = 8

ALLOWED_VISUAL_ROLES = frozenset(
    {
        "button",
        "icon",
        "input",
        "tab",
        "toggle",
        "list_item",
        "keyboard_key",
        "image",
        "container",
        "text",
    }
)


class ElementGeometryAuditError(RuntimeError):
    """The local geometry audit did not establish one safe target box."""


class _DuplicateJSONKeyError(ValueError):
    pass


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKeyError(key)
        result[key] = value
    return result


def _finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _validated_local_bounds(value: Any) -> tuple[float, float, float, float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ElementGeometryAuditError("geometry match bounds 必须包含4个数值。")
    if not all(_finite_number(part) for part in value):
        raise ElementGeometryAuditError("geometry match bounds 含非有限数值。")
    left, top, right, bottom = (float(part) for part in value)
    if not (
        0.0 <= left < right <= 1000.0
        and 0.0 <= top < bottom <= 1000.0
    ):
        raise ElementGeometryAuditError(
            "geometry match bounds 超出 crop-local 0..1000 坐标。"
        )
    return left, top, right, bottom


def _validated_full_bounds(value: Any) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ElementGeometryAuditError("粗候选 bounds 必须包含4个数值。")
    if not all(_finite_number(part) for part in value):
        raise ElementGeometryAuditError("粗候选 bounds 含非有限数值。")
    left, top, right, bottom = (float(part) for part in value)
    if not (
        0.0 <= left < right <= 1.0
        and 0.0 <= top < bottom <= 1.0
    ):
        raise ElementGeometryAuditError("粗候选 bounds 超出 full-frame 0..1 坐标。")
    return left, top, right, bottom


@dataclass(frozen=True)
class CropTransform:
    """One exact pixel crop and its only allowed crop-local coordinate system."""

    full_size: tuple[int, int]
    pixel_bounds: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        width, height = self.full_size
        left, top, right, bottom = self.pixel_bounds
        if (
            isinstance(width, bool)
            or isinstance(height, bool)
            or not isinstance(width, int)
            or not isinstance(height, int)
            or width <= 0
            or height <= 0
        ):
            raise ElementGeometryAuditError("full frame 尺寸无效。")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in self.pixel_bounds):
            raise ElementGeometryAuditError("pixel_bounds 必须是整数像素。")
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            raise ElementGeometryAuditError("pixel_bounds 超出 full frame。")

    @property
    def crop_size(self) -> tuple[int, int]:
        left, top, right, bottom = self.pixel_bounds
        return right - left, bottom - top

    def crop(self, image: Image.Image) -> Image.Image:
        if image.size != self.full_size:
            raise ElementGeometryAuditError("待裁剪画面尺寸与 CropTransform 不一致。")
        return image.crop(self.pixel_bounds)

    def map_bounds_to_full(
        self,
        local_bounds: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        """Map crop-local 0..1000 bounds through the exact pixel rectangle."""

        local = _validated_local_bounds(list(local_bounds))
        full_width, full_height = self.full_size
        left_px, top_px, right_px, bottom_px = self.pixel_bounds
        crop_width = right_px - left_px
        crop_height = bottom_px - top_px
        left, top, right, bottom = local
        mapped = (
            (left_px + left * crop_width / 1000.0) / full_width,
            (top_px + top * crop_height / 1000.0) / full_height,
            (left_px + right * crop_width / 1000.0) / full_width,
            (top_px + bottom * crop_height / 1000.0) / full_height,
        )
        if not (
            0.0 <= mapped[0] < mapped[2] <= 1.0
            and 0.0 <= mapped[1] < mapped[3] <= 1.0
        ):
            raise ElementGeometryAuditError("映射后的 bounds 超出 full frame。")
        return mapped

    def internal_edges_touched(
        self,
        local_bounds: tuple[float, float, float, float],
        *,
        margin: float = DEFAULT_INTERNAL_EDGE_MARGIN,
    ) -> tuple[str, ...]:
        if not _finite_number(margin) or not 0.0 <= float(margin) < 500.0:
            raise ElementGeometryAuditError("内部边缘 margin 无效。")
        left, top, right, bottom = _validated_local_bounds(list(local_bounds))
        full_width, full_height = self.full_size
        crop_left, crop_top, crop_right, crop_bottom = self.pixel_bounds
        threshold = float(margin)
        touched: list[str] = []
        if crop_left > 0 and left < threshold:
            touched.append("left")
        if crop_top > 0 and top < threshold:
            touched.append("top")
        if crop_right < full_width and right > 1000.0 - threshold:
            touched.append("right")
        if crop_bottom < full_height and bottom > 1000.0 - threshold:
            touched.append("bottom")
        return tuple(touched)


def _build_candidate_crop_transform(
    full_size: tuple[int, int],
    rough_bounds: tuple[float, float, float, float],
    *,
    minimum_width_span: float,
    minimum_height_span: float,
    width_target_scale: float,
    height_target_scale: float,
) -> CropTransform:
    """Build one candidate-relative ROI without App or pixel patches."""

    width, height = full_size
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, int)
        or not isinstance(height, int)
        or width <= 0
        or height <= 0
    ):
        raise ElementGeometryAuditError("full frame 尺寸无效。")
    if (
        not _finite_number(minimum_width_span)
        or not 0.0 < float(minimum_width_span) <= 1.0
        or not _finite_number(minimum_height_span)
        or not 0.0 < float(minimum_height_span) <= 1.0
        or not _finite_number(width_target_scale)
        or float(width_target_scale) < 1.0
        or not _finite_number(height_target_scale)
        or float(height_target_scale) < 1.0
    ):
        raise ElementGeometryAuditError("ROI 比例参数无效。")

    left, top, right, bottom = _validated_full_bounds(rough_bounds)
    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0
    roi_width = min(
        1.0,
        max(
            float(minimum_width_span),
            (right - left) * float(width_target_scale),
        ),
    )
    roi_height = min(
        1.0,
        max(
            float(minimum_height_span),
            (bottom - top) * float(height_target_scale),
        ),
    )

    roi_left = min(max(0.0, center_x - roi_width / 2.0), 1.0 - roi_width)
    roi_top = min(max(0.0, center_y - roi_height / 2.0), 1.0 - roi_height)
    roi_right = roi_left + roi_width
    roi_bottom = roi_top + roi_height

    pixel_left = max(0, min(width - 1, round(roi_left * width)))
    pixel_top = max(0, min(height - 1, round(roi_top * height)))
    pixel_right = max(pixel_left + 1, min(width, round(roi_right * width)))
    pixel_bottom = max(pixel_top + 1, min(height, round(roi_bottom * height)))
    return CropTransform(
        full_size=full_size,
        pixel_bounds=(pixel_left, pixel_top, pixel_right, pixel_bottom),
    )


def build_candidate_crop_transform(
    full_size: tuple[int, int],
    rough_bounds: tuple[float, float, float, float],
    *,
    minimum_span: float = DEFAULT_ROI_MIN_SPAN,
    target_scale: float = DEFAULT_ROI_TARGET_SCALE,
) -> CropTransform:
    """Build the broad ROI used for non-literal and structural controls."""

    return _build_candidate_crop_transform(
        full_size,
        rough_bounds,
        minimum_width_span=minimum_span,
        minimum_height_span=minimum_span,
        width_target_scale=target_scale,
        height_target_scale=target_scale,
    )


def build_literal_candidate_crop_transform(
    full_size: tuple[int, int],
    rough_bounds: tuple[float, float, float, float],
) -> CropTransform:
    """Build a wide, shallow ROI that isolates one literal text selector row."""

    return _build_candidate_crop_transform(
        full_size,
        rough_bounds,
        minimum_width_span=LITERAL_ROI_MIN_WIDTH_SPAN,
        minimum_height_span=LITERAL_ROI_MIN_HEIGHT_SPAN,
        width_target_scale=LITERAL_ROI_WIDTH_TARGET_SCALE,
        height_target_scale=LITERAL_ROI_HEIGHT_TARGET_SCALE,
    )


@dataclass(frozen=True)
class GeometryAuditMatch:
    match_id: str
    literal_label: str
    visual_role: str
    bounds: tuple[float, float, float, float]
    confidence: float
    fully_visible: bool
    whole_control: bool
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class ElementGeometryAuditPayload:
    source_ref: str
    crop_clear: bool
    enumeration_complete: bool
    matches: tuple[GeometryAuditMatch, ...]


@dataclass(frozen=True)
class AuditedElementGeometry:
    source_ref: str
    literal_label: str
    visual_role: str
    local_bounds: tuple[float, float, float, float]
    full_bounds: tuple[float, float, float, float]
    confidence: float
    evidence: tuple[str, ...]
    transform: CropTransform


_SOURCE_REF_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{7,127}$")
_MATCH_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")
MAX_GEOMETRY_EVIDENCE_CHARS = 200
_FORBIDDEN_EVIDENCE = re.compile(
    r"(?:coordinates?|coords?|bounds?\s*(?:[=:]|\[|\(|-?\d)|"
    r"\bx\s*[=:]|\by\s*[=:]|"
    r"\b(?:tap|click|press|swipe|drag|execute|suggest)\b|"
    r"点击|滑动|拖动|按下|坐标|执行|建议)",
    re.IGNORECASE,
)


def _evidence_contains_control_info(text: str, *, literal_label: str) -> bool:
    """Reject control language outside an exact quoted/visible UI label."""

    remainder = str(text or "")
    label = str(literal_label or "").strip()
    if label:
        remainder = remainder.replace(label, "")
    return _FORBIDDEN_EVIDENCE.search(remainder) is not None


def parse_element_geometry_audit(
    raw: str,
    *,
    visible_literal_labels: tuple[str, ...] = (),
) -> ElementGeometryAuditPayload:
    text = str(raw or "").strip()
    if not text or text.startswith("```"):
        raise ElementGeometryAuditError("geometry audit 必须只返回一个JSON对象。")
    try:
        value = json.loads(text, object_pairs_hook=_reject_duplicate_pairs)
    except _DuplicateJSONKeyError as exc:
        raise ElementGeometryAuditError(
            f"geometry audit 包含重复JSON字段：{exc}"
        ) from exc
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ElementGeometryAuditError("geometry audit JSON无法解析。") from exc
    if not isinstance(value, dict):
        raise ElementGeometryAuditError("geometry audit 顶层必须是对象。")
    required = {
        "protocol_version",
        "source_ref",
        "crop_clear",
        "enumeration_complete",
        "matches",
    }
    if set(value) != required:
        raise ElementGeometryAuditError("geometry audit 顶层字段不完整或包含额外字段。")
    if value["protocol_version"] != ELEMENT_GEOMETRY_AUDIT_PROTOCOL_VERSION:
        raise ElementGeometryAuditError("geometry audit 协议版本不匹配。")
    source_ref = value["source_ref"]
    if not isinstance(source_ref, str) or not _SOURCE_REF_PATTERN.fullmatch(source_ref):
        raise ElementGeometryAuditError("geometry audit source_ref 无效。")
    if not isinstance(value["crop_clear"], bool) or not isinstance(
        value["enumeration_complete"], bool
    ):
        raise ElementGeometryAuditError("geometry audit 完整性字段必须是布尔值。")
    raw_matches = value["matches"]
    if not isinstance(raw_matches, list) or len(raw_matches) > MAX_GEOMETRY_MATCHES:
        raise ElementGeometryAuditError("geometry audit matches 数量或格式无效。")

    allowed_scene_labels = tuple(
        dict.fromkeys(
            item.strip()
            for item in tuple(visible_literal_labels or ())
            if isinstance(item, str) and item.strip()
        )
    )
    if (
        len(allowed_scene_labels) > 8
        or any(len(item) > 200 for item in allowed_scene_labels)
    ):
        raise ElementGeometryAuditError("visible_literal_labels 无效。")
    matches: list[GeometryAuditMatch] = []
    seen_ids: set[str] = set()
    match_fields = {
        "match_id",
        "literal_label",
        "visual_role",
        "bounds",
        "confidence",
        "fully_visible",
        "whole_control",
        "evidence",
    }
    for item in raw_matches:
        if not isinstance(item, dict) or set(item) != match_fields:
            raise ElementGeometryAuditError(
                "geometry audit match 字段不完整或包含额外字段。"
            )
        match_id = item["match_id"]
        if (
            not isinstance(match_id, str)
            or not _MATCH_ID_PATTERN.fullmatch(match_id)
            or match_id in seen_ids
        ):
            raise ElementGeometryAuditError("geometry audit match_id 无效或重复。")
        seen_ids.add(match_id)
        role = item["visual_role"]
        if not isinstance(role, str) or role not in ALLOWED_VISUAL_ROLES:
            raise ElementGeometryAuditError("geometry audit visual_role 无效。")
        label = item["literal_label"]
        if (
            not isinstance(label, str)
            or len(label.strip()) > 200
            or (not label.strip() and role != "input")
        ):
            raise ElementGeometryAuditError("geometry audit literal_label 无效。")
        bounds = _validated_local_bounds(item["bounds"])
        confidence = item["confidence"]
        if not _finite_number(confidence) or not 0.0 <= float(confidence) <= 1.0:
            raise ElementGeometryAuditError("geometry audit confidence 无效。")
        if not isinstance(item["fully_visible"], bool) or not isinstance(
            item["whole_control"], bool
        ):
            raise ElementGeometryAuditError(
                "geometry audit 可见性字段必须是布尔值。"
            )
        raw_evidence = item["evidence"]
        if (
            not isinstance(raw_evidence, list)
            or not 1 <= len(raw_evidence) <= 2
            or any(not isinstance(part, str) for part in raw_evidence)
        ):
            raise ElementGeometryAuditError("geometry audit evidence 格式无效。")
        evidence = tuple(part.strip() for part in raw_evidence)
        evidence_without_literal_labels = []
        allowed_labels = tuple(
            dict.fromkeys((label.strip(), *allowed_scene_labels))
        )
        for part in evidence:
            remainder = part
            for visible_label in sorted(allowed_labels, key=len, reverse=True):
                remainder = remainder.replace(visible_label, "")
            evidence_without_literal_labels.append(remainder)
        if any(
            not part
            or len(part) > MAX_GEOMETRY_EVIDENCE_CHARS
            or _evidence_contains_control_info(remainder, literal_label="")
            for part, remainder in zip(evidence, evidence_without_literal_labels)
        ):
            raise ElementGeometryAuditError(
                "geometry audit evidence 为空、过长或包含控制信息。"
            )
        matches.append(
            GeometryAuditMatch(
                match_id=match_id,
                literal_label=label.strip(),
                visual_role=role,
                bounds=bounds,
                confidence=float(confidence),
                fully_visible=item["fully_visible"],
                whole_control=item["whole_control"],
                evidence=evidence,
            )
        )
    return ElementGeometryAuditPayload(
        source_ref=source_ref,
        crop_clear=value["crop_clear"],
        enumeration_complete=value["enumeration_complete"],
        matches=tuple(matches),
    )


def select_unique_audited_geometry(
    raw: str,
    *,
    expected_source_ref: str,
    expected_label: str,
    expected_role: str,
    transform: CropTransform,
    visible_literal_labels: tuple[str, ...] = (),
    minimum_confidence: float = MIN_GEOMETRY_AUDIT_CONFIDENCE,
    internal_edge_margin: float = DEFAULT_INTERNAL_EDGE_MARGIN,
) -> AuditedElementGeometry:
    """Return one mapped target or fail; the rough box is never a fallback."""

    if not isinstance(expected_source_ref, str) or not _SOURCE_REF_PATTERN.fullmatch(
        expected_source_ref
    ):
        raise ElementGeometryAuditError("expected_source_ref 无效。")
    label = str(expected_label or "").strip()
    if expected_role not in ALLOWED_VISUAL_ROLES:
        raise ElementGeometryAuditError("expected_role 无效。")
    if not label and expected_role != "input":
        raise ElementGeometryAuditError(
            "geometry audit 仅对 role=input 允许空逐字标签。"
        )
    if not _finite_number(minimum_confidence) or not 0.0 <= float(
        minimum_confidence
    ) <= 1.0:
        raise ElementGeometryAuditError("minimum_confidence 无效。")

    payload = parse_element_geometry_audit(
        raw,
        visible_literal_labels=visible_literal_labels,
    )
    if payload.source_ref != expected_source_ref:
        raise ElementGeometryAuditError("geometry audit source_ref 与请求不一致。")
    if payload.crop_clear is not True or payload.enumeration_complete is not True:
        raise ElementGeometryAuditError("crop 不清晰或匹配枚举不完整。")
    if len(payload.matches) != 1:
        raise ElementGeometryAuditError("crop 内没有建立唯一 label+role 匹配。")
    match = payload.matches[0]
    if match.literal_label != label or match.visual_role != expected_role:
        raise ElementGeometryAuditError("geometry match 没有逐字复用 label+role。")
    if match.confidence < float(minimum_confidence):
        raise ElementGeometryAuditError("geometry match confidence 不足。")
    if match.fully_visible is not True or match.whole_control is not True:
        raise ElementGeometryAuditError("geometry match 未证明完整可见的整个控件。")
    touched = transform.internal_edges_touched(
        match.bounds,
        margin=internal_edge_margin,
    )
    if touched:
        raise ElementGeometryAuditError(
            "geometry match 触及内部 crop 边缘：" + ",".join(touched)
        )
    full_bounds = transform.map_bounds_to_full(match.bounds)
    return AuditedElementGeometry(
        source_ref=payload.source_ref,
        literal_label=match.literal_label,
        visual_role=match.visual_role,
        local_bounds=match.bounds,
        full_bounds=full_bounds,
        confidence=match.confidence,
        evidence=match.evidence,
        transform=transform,
    )


def element_geometry_audit_prompt(
    *,
    source_ref: str,
    literal_label: str,
    visual_role: str,
    visible_evidence: str,
    visible_literal_labels: tuple[str, ...] = (),
) -> str:
    """Describe one-image, one-coordinate-space read-only localization."""

    if not _SOURCE_REF_PATTERN.fullmatch(str(source_ref or "")):
        raise ElementGeometryAuditError("source_ref 无效。")
    label = str(literal_label or "").strip()
    evidence = str(visible_evidence or "").strip()
    if visual_role not in ALLOWED_VISUAL_ROLES:
        raise ElementGeometryAuditError("visual_role 无效。")
    if len(label) > 200 or (not label and visual_role != "input"):
        raise ElementGeometryAuditError("literal_label 无效。")
    allowed_labels = tuple(
        dict.fromkeys(
            item.strip()
            for item in (label, *tuple(visible_literal_labels or ()))
            if isinstance(item, str) and item.strip()
        )
    )
    if (
        len(allowed_labels) > 8
        or any(len(item) > 200 for item in allowed_labels)
    ):
        raise ElementGeometryAuditError("visible_literal_labels 无效。")
    evidence_without_literal_labels = evidence
    for visible_label in sorted(allowed_labels, key=len, reverse=True):
        evidence_without_literal_labels = evidence_without_literal_labels.replace(
            visible_label,
            "",
        )
    if (
        not evidence
        or len(evidence) > MAX_GEOMETRY_EVIDENCE_CHARS
        or _evidence_contains_control_info(
            evidence_without_literal_labels,
            literal_label="",
        )
    ):
        raise ElementGeometryAuditError("visible_evidence 无效。")
    schema = {
        "protocol_version": ELEMENT_GEOMETRY_AUDIT_PROTOCOL_VERSION,
        "source_ref": source_ref,
        "crop_clear": True,
        "enumeration_complete": True,
        "matches": [
            {
                "match_id": "match-1",
                "literal_label": label,
                "visual_role": visual_role,
                "bounds": [0, 0, 1000, 1000],
                "confidence": 0.0,
                "fully_visible": True,
                "whole_control": True,
                "evidence": ["literal visible fact"],
            }
        ],
    }
    return (
        "You are a read-only, app-independent element geometry auditor. "
        "You receive exactly one image: one pixel crop. It is the only image and "
        "the only coordinate space. Its left/top is 0 and right/bottom is 1000. "
        "Never infer or return full-frame coordinates, an action, a plan, or a goal. "
        "Enumerate every occurrence inside this crop that exactly matches the supplied "
        "literal label and visual role. Only when visual_role=input and literal_label is "
        "empty, enumerate every whole visible input control matching the supplied visual "
        "evidence and return literal_label as an empty string. Tight bounds must contain "
        "the whole control, "
        "not merely a broad row or neighboring control. If the crop is unclear, an "
        "occurrence is clipped, or enumeration cannot be completed, report those facts "
        "without guessing. The local controller will map accepted crop-local bounds.\n"
        f"source_ref={source_ref}\n"
        f"literal_label={json.dumps(label, ensure_ascii=False)}\n"
        f"visual_role={visual_role}\n"
        f"visible_literal_labels={json.dumps(allowed_labels, ensure_ascii=False)}\n"
        f"visible_evidence={json.dumps(evidence, ensure_ascii=False)}\n"
        "Return exactly one JSON object with no duplicate or extra fields and no Markdown:\n"
        + json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
    )
