from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from io import BytesIO
from typing import Iterable

from PIL import Image, ImageChops, ImageFilter, ImageStat


@dataclass(frozen=True)
class EncodedObservationImage:
    """One bounded image sent to the observation model.

    ``bounds`` always refers to the original full camera frame in normalized
    0..1000 coordinates.  The model may therefore report ROI-local geometry;
    the controller maps it back to the original frame before any target can
    reach the physical action adapter.
    """

    role: str
    bounds: tuple[int, int, int, int]
    data_url: str
    width: int
    height: int
    jpeg_bytes: int

    def metadata(self) -> dict[str, object]:
        return {
            "role": self.role,
            "bounds": list(self.bounds),
            "width": self.width,
            "height": self.height,
            "jpeg_bytes": self.jpeg_bytes,
        }


@dataclass(frozen=True)
class LocalFrameStability:
    stable: bool
    mean_delta: float
    max_delta: float
    frame_count: int
    threshold: float
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "stable": self.stable,
            "mean_delta": round(self.mean_delta, 3),
            "max_delta": round(self.max_delta, 3),
            "frame_count": self.frame_count,
            "threshold": self.threshold,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ObservationRoi:
    name: str
    bounds: tuple[int, int, int, int]
    purpose: str


def _normalized_crop(
    image: Image.Image,
    bounds: tuple[int, int, int, int],
) -> Image.Image:
    left, top, right, bottom = bounds
    if not (0 <= left < right <= 1000 and 0 <= top < bottom <= 1000):
        raise ValueError(f"ROI越界：{bounds}")
    x0 = max(0, min(image.width - 1, round(left * image.width / 1000)))
    y0 = max(0, min(image.height - 1, round(top * image.height / 1000)))
    x1 = max(x0 + 1, min(image.width, round(right * image.width / 1000)))
    y1 = max(y0 + 1, min(image.height, round(bottom * image.height / 1000)))
    return image.crop((x0, y0, x1, y1))


def _encode_bounded_jpeg(
    image: Image.Image,
    *,
    role: str,
    bounds: tuple[int, int, int, int],
    width_candidates: Iterable[int],
    max_bytes: int,
) -> EncodedObservationImage:
    source = image.convert("RGB")
    qualities = (70, 60, 52, 44, 36, 30)
    last: tuple[Image.Image, bytes] | None = None
    for requested_width in width_candidates:
        width = max(48, min(source.width, int(requested_width)))
        height = max(1, round(source.height * width / source.width))
        resized = (
            source
            if (width, height) == source.size
            else source.resize((width, height), Image.Resampling.LANCZOS)
        )
        for quality in qualities:
            buffer = BytesIO()
            resized.save(buffer, format="JPEG", quality=quality, optimize=True)
            payload = buffer.getvalue()
            last = (resized, payload)
            if len(payload) <= max_bytes:
                encoded = base64.b64encode(payload).decode("ascii")
                return EncodedObservationImage(
                    role=role,
                    bounds=bounds,
                    data_url=f"data:image/jpeg;base64,{encoded}",
                    width=resized.width,
                    height=resized.height,
                    jpeg_bytes=len(payload),
                )
    assert last is not None
    resized, payload = last
    if len(payload) > max_bytes:
        raise ValueError(
            f"{role}压缩后仍为{len(payload)}字节，超过{max_bytes}字节安全预算。"
        )
    encoded = base64.b64encode(payload).decode("ascii")
    return EncodedObservationImage(
        role=role,
        bounds=bounds,
        data_url=f"data:image/jpeg;base64,{encoded}",
        width=resized.width,
        height=resized.height,
        jpeg_bytes=len(payload),
    )


def build_overview(
    image: Image.Image,
    *,
    max_bytes: int = 28000,
) -> EncodedObservationImage:
    """Build a readable full-page image used for page classification.

    The former 160-pixel overview erased small Douyin labels and icon colors.
    320 pixels still keeps the request small while preserving roughly four
    times as many pixels for the observer.
    """

    return _encode_bounded_jpeg(
        image,
        role="overview",
        bounds=(0, 0, 1000, 1000),
        width_candidates=(320, 288, 256, 224, 192),
        max_bytes=max_bytes,
    )


def build_roi(
    image: Image.Image,
    roi: ObservationRoi,
    *,
    max_bytes: int = 24000,
) -> EncodedObservationImage:
    """Build a crop with more pixel density than the full-page overview."""

    crop = _normalized_crop(image, roi.bounds)
    key_douyin_rois = {
        "douyin_page_evidence",
        "page_state_right",
        "right_actions",
    }
    width_candidates = (
        (400, 360, 320, 288, 256, 224)
        if roi.name in key_douyin_rois
        else (320, 288, 256, 224, 192, 160)
    )
    return _encode_bounded_jpeg(
        crop,
        role=roi.name,
        bounds=roi.bounds,
        width_candidates=width_candidates,
        max_bytes=max_bytes,
    )


def map_roi_point_to_full(
    point: tuple[int, int],
    bounds: tuple[int, int, int, int],
) -> tuple[int, int]:
    x, y = point
    left, top, right, bottom = bounds
    return (
        round(left + x * (right - left) / 1000),
        round(top + y * (bottom - top) / 1000),
    )


def map_roi_bounds_to_full(
    box: tuple[int, int, int, int],
    bounds: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    left, top = map_roi_point_to_full((box[0], box[1]), bounds)
    right, bottom = map_roi_point_to_full((box[2], box[3]), bounds)
    return left, top, right, bottom


def _static_band_sheet(image: Image.Image) -> Image.Image:
    """Use UI-heavy outer bands; avoid most moving video content."""

    gray = image.convert("L").resize((96, 160), Image.Resampling.BILINEAR)
    top = gray.crop((0, 0, 96, 26))
    bottom = gray.crop((0, 130, 96, 160))
    left = gray.crop((0, 26, 14, 130)).resize((20, 104))
    right = gray.crop((76, 26, 96, 130)).resize((20, 104))
    sheet = Image.new("L", (136, 104), 0)
    sheet.paste(top.resize((96, 26)), (20, 0))
    sheet.paste(bottom.resize((96, 30)), (20, 74))
    sheet.paste(left, (0, 0))
    sheet.paste(right, (116, 0))
    return sheet


def measure_local_stability(
    frames: list[Image.Image],
    *,
    threshold: float | None = None,
) -> LocalFrameStability:
    """Measure camera/UI stability locally; no frame leaves the machine."""

    if len(frames) < 2:
        raise ValueError("本地稳定性判断至少需要2帧。")
    sizes = {frame.size for frame in frames}
    if len(sizes) != 1:
        return LocalFrameStability(
            stable=False,
            mean_delta=float("inf"),
            max_delta=float("inf"),
            frame_count=len(frames),
            threshold=float(threshold or 0.0),
            reason="连续画面尺寸发生变化",
        )
    limit = float(
        threshold
        if threshold is not None
        else os.environ.get("ROBOT_LOCAL_FRAME_DELTA_MAX", "38.0")
    )
    sheets = [_static_band_sheet(frame) for frame in frames]
    deltas: list[float] = []
    for first, second in zip(sheets, sheets[1:]):
        value = ImageStat.Stat(ImageChops.difference(first, second)).mean[0]
        deltas.append(float(value))
    mean_delta = sum(deltas) / len(deltas)
    max_delta = max(deltas)
    stable = max_delta <= limit
    return LocalFrameStability(
        stable=stable,
        mean_delta=mean_delta,
        max_delta=max_delta,
        frame_count=len(frames),
        threshold=limit,
        reason=(
            "外圈静态UI多帧一致"
            if stable
            else f"外圈静态UI变化{max_delta:.1f}超过阈值{limit:.1f}"
        ),
    )


def measure_frame_sharpness(image: Image.Image) -> float:
    """Return a local, content-agnostic sharpness score for frame selection.

    Stability and sharpness serve different purposes: the outer UI bands prove
    that the page/camera has not moved, while this score chooses the clearest
    full frame from that already-stable group.  It never changes page state or
    relaxes the stability gate.
    """

    source = image.convert("L")
    max_width = 360
    if source.width > max_width:
        height = max(1, round(source.height * max_width / source.width))
        source = source.resize((max_width, height), Image.Resampling.LANCZOS)
    blurred = source.filter(ImageFilter.GaussianBlur(radius=1.0))
    high_frequency = ImageChops.difference(source, blurred)
    return float(ImageStat.Stat(high_frequency).rms[0])
