"""Application boundary for typed-input lineage image matching and storage."""

from __future__ import annotations

from agent.domain.validation import NormalizedBounds, reject_if
from pathlib import Path
from typing import Any, Callable, Literal, Protocol

from PIL import Image, ImageOps

from agent.domain.input_value_lineage import (
    SURFACE_DESCRIPTOR_HEIGHT,
    SURFACE_DESCRIPTOR_WIDTH,
    InputValueLineageError,
    TypedInputLineage,
    _valid_bounds,
)


SURFACE_DESCRIPTOR_MAX_MEAN_DISTANCE = 18.0
VerifiedInputActionType = Literal['literal', 'text', 'newline']


class TypedInputLineageStorePort(Protocol):
    """Persist and mint the single authoritative lineage for one device."""

    ttl_seconds: float
    clock: Callable[[], float]

    def write(self, record: TypedInputLineage) -> Path: ...

    def discard(self, device_id: str) -> None: ...

    def load(self, device_id: str) -> TypedInputLineage | None: ...

    def record_verified_action(self, *, action_type: VerifiedInputActionType, device_id: str,
        resolved_action: dict[str, Any], before_scene: dict[str, Any], after_scene: dict[str, Any],
        after_frames: tuple[Image.Image, ...], hardware_receipt: dict[str, Any] | None=None,
        source: str | None=None) -> TypedInputLineage: ...


def describe_input_surface(frame: Image.Image, bounds: NormalizedBounds) -> str:
    reject_if(not isinstance(frame, Image.Image), InputValueLineageError("输入表面描述缺少真实图像帧。"))
    valid = _valid_bounds(bounds)
    reject_if(valid is None or frame.width < 2 or frame.height < 2, InputValueLineageError("输入表面描述的图像或 bounds 无效。"))
    left, top, right, bottom = valid
    left = max(0.0, left - 0.04)
    top = max(0.0, top - 0.035)
    right = min(1.0, right + 0.04)
    bottom = min(1.0, bottom + 0.035)
    pixel_box = (round(left * frame.width), round(top * frame.height), round(right * frame.width),
        round(bottom * frame.height))
    reject_if(pixel_box[0] >= pixel_box[2] or pixel_box[1] >= pixel_box[3], InputValueLineageError("输入表面描述的局部区域为空。"))
    gray = frame.convert("L").crop(pixel_box)
    normalized = ImageOps.autocontrast(gray, cutoff=1).resize((SURFACE_DESCRIPTOR_WIDTH, SURFACE_DESCRIPTOR_HEIGHT),
        Image.Resampling.LANCZOS)
    return normalized.tobytes().hex()


def build_surface_descriptors(frames: Any, bounds: NormalizedBounds) -> tuple[str, ...]:
    reject_if(not isinstance(frames, (list, tuple)) or len(frames) != 4, InputValueLineageError("输入表面连续性必须绑定动作后四帧。"))
    descriptors = tuple(describe_input_surface(frame, bounds) for frame in frames)
    reject_if(not descriptors, InputValueLineageError("输入表面连续性没有可用的局部描述。"))
    return descriptors


def surface_descriptors_match(descriptors: tuple[str, ...], *, frame: Image.Image | None,
    bounds: NormalizedBounds) -> bool:
    if frame is None or not descriptors:
        return False
    try:
        current = bytes.fromhex(describe_input_surface(frame, bounds))
    except (InputValueLineageError, ValueError):
        return False
    for descriptor in descriptors:
        try:
            prior = bytes.fromhex(descriptor)
        except ValueError:
            continue
        if len(prior) != len(current):
            continue
        mean_distance = sum((abs(first - second) for first, second in zip(prior, current))) / len(current)
        if mean_distance <= SURFACE_DESCRIPTOR_MAX_MEAN_DISTANCE:
            return True
    return False


def lineage_matches_visual(record: TypedInputLineage, *, current_frame: Image.Image | None=None,
    **context: Any) -> bool:
    return record.matches_visual(**context, surface_matches=surface_descriptors_match(record.surface_descriptors,
        frame=current_frame, bounds=record.input_bounds))


def lineage_matches_persisted_surface_cue(record: TypedInputLineage, *, input_bounds: NormalizedBounds | None,
    current_frame: Image.Image | None, **context: Any) -> bool:
    surface_matches = bool(input_bounds is not None and surface_descriptors_match(record.surface_descriptors,
        frame=current_frame, bounds=input_bounds))
    return record.matches_persisted_surface_cue(**context, input_bounds=input_bounds, surface_matches=surface_matches)


def lineage_matches_trailing_newline_cue(record: TypedInputLineage, *, input_bounds: NormalizedBounds | None,
    current_frame: Image.Image | None=None, **context: Any) -> bool:
    surface_matches = bool(input_bounds is not None and surface_descriptors_match(record.surface_descriptors,
        frame=current_frame, bounds=input_bounds))
    return record.matches_trailing_newline_cue(**context, input_bounds=input_bounds, surface_matches=surface_matches)
