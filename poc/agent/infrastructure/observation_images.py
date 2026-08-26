from __future__ import annotations

import hashlib
import os
from typing import Iterable

from PIL import Image, ImageChops, ImageFilter, ImageStat

from agent.domain.visual_evidence import LocalFrameStability, VisualObstruction


def local_frame_fingerprint(frame: Image.Image) -> str:
    compact = frame.convert("L").resize((64, 96), Image.Resampling.BILINEAR)
    return hashlib.sha256(compact.tobytes()).hexdigest()[:20]


def _ratio_bounds(
    left: int,
    top: int,
    right: int,
    bottom: int,
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    return (
        max(0, min(1000, round(left * 1000 / width))),
        max(0, min(1000, round(top * 1000 / height))),
        max(0, min(1000, round(right * 1000 / width))),
        max(0, min(1000, round(bottom * 1000 / height))),
    )


def detect_top_edge_opaque_bands(image: Image.Image) -> tuple[VisualObstruction, ...]:
    """Detect shallow partial-width dark bands attached to the image top.

    Seller previews can paint an opaque coordinate readout over the camera
    surface.  Its pixel size changes with window/DPI scaling, so detection is
    based on relative geometry and contrast instead of a fixed crop or app
    coordinate.  Full-width status bars, narrow phone letterboxing, and deep
    dark panels are deliberately excluded.
    """

    source = image.convert("L")
    analysis_width = min(270, source.width)
    if analysis_width < 80 or source.height < 120:
        return ()
    analysis_height = max(1, round(source.height * analysis_width / source.width))
    gray = source.resize((analysis_width, analysis_height), Image.Resampling.BILINEAR)
    pixels = gray.load()
    dark_limit = 64
    probe_bottom = max(8, min(analysis_height // 7, round(analysis_height * 0.12)))
    edge_skip = max(2, round(analysis_width * 0.03))
    usable_width = max(1, analysis_width - 2 * edge_skip)

    row_dark: list[float] = []
    for y in range(probe_bottom):
        dark = sum(
            1
            for x in range(edge_skip, analysis_width - edge_skip)
            if pixels[x, y] <= dark_limit
        )
        row_dark.append(dark / usable_width)

    attach_limit = max(2, round(analysis_height * 0.02))
    start = next(
        (index for index, ratio in enumerate(row_dark[: attach_limit + 1]) if ratio >= 0.22),
        None,
    )
    if start is None:
        return ()

    low_run = 0
    end = start
    for y in range(start, probe_bottom):
        if row_dark[y] >= 0.18:
            end = y + 1
            low_run = 0
        else:
            low_run += 1
            if low_run >= 3:
                break
    band_height = end - start
    if band_height < max(3, round(analysis_height * 0.012)):
        return ()

    column_dark: list[float] = []
    for x in range(analysis_width):
        dark = sum(1 for y in range(start, end) if pixels[x, y] <= dark_limit)
        column_dark.append(dark / band_height)

    active = [ratio >= 0.68 for ratio in column_dark]
    bridge = max(1, round(analysis_width * 0.015))
    index = 0
    while index < analysis_width:
        if active[index]:
            index += 1
            continue
        gap_start = index
        while index < analysis_width and not active[index]:
            index += 1
        if (
            gap_start > 0
            and index < analysis_width
            and index - gap_start <= bridge
        ):
            for gap_index in range(gap_start, index):
                active[gap_index] = True

    runs: list[tuple[int, int]] = []
    index = 0
    while index < analysis_width:
        if not active[index]:
            index += 1
            continue
        run_start = index
        while index < analysis_width and active[index]:
            index += 1
        runs.append((run_start, index))

    results: list[VisualObstruction] = []
    for run_start, run_end in runs:
        run_width = run_end - run_start
        width_ratio = run_width / analysis_width
        if not 0.14 <= width_ratio <= 0.90:
            continue
        band_dark = sum(column_dark[run_start:run_end]) / run_width
        below_start = end
        below_end = min(analysis_height, end + max(4, band_height * 2))
        if below_end <= below_start:
            continue
        below_dark = sum(
            1
            for y in range(below_start, below_end)
            for x in range(run_start, run_end)
            if pixels[x, y] <= dark_limit
        ) / (run_width * (below_end - below_start))
        if band_dark < 0.74 or below_dark >= band_dark * 0.55:
            continue
        bounds = _ratio_bounds(
            run_start,
            0,
            run_end,
            min(analysis_height, end + max(1, round(band_height * 0.08))),
            width=analysis_width,
            height=analysis_height,
        )
        results.append(
            VisualObstruction(
                kind="top_edge_opaque_band",
                bounds=bounds,
                reason="顶部存在浅层、非全宽且与下方画面不连续的不透明暗色区域",
            )
        )
    return tuple(results)


def _bounds_iou(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / max(1, first_area + second_area - intersection)


def consensus_top_edge_obstructions(
    frames: Iterable[Image.Image],
) -> tuple[VisualObstruction, ...]:
    """Return only top-edge obstructions repeated across the stable frame tail."""

    frame_list = list(frames)
    if not frame_list:
        return ()
    detections = [detect_top_edge_opaque_bands(frame) for frame in frame_list]
    required = max(2, (len(frame_list) + 1) // 2) if len(frame_list) > 1 else 1
    accepted: list[VisualObstruction] = []
    for candidate in (item for frame in detections for item in frame):
        if any(_bounds_iou(candidate.bounds, item.bounds) >= 0.60 for item in accepted):
            continue
        matches: list[VisualObstruction] = []
        for frame_detections in detections:
            match = max(
                frame_detections,
                key=lambda item: _bounds_iou(candidate.bounds, item.bounds),
                default=None,
            )
            if match is not None and _bounds_iou(candidate.bounds, match.bounds) >= 0.60:
                matches.append(match)
        if len(matches) < required:
            continue
        coordinates = tuple(
            sorted(item.bounds[index] for item in matches)[len(matches) // 2]
            for index in range(4)
        )
        accepted.append(
            VisualObstruction(
                kind=candidate.kind,
                bounds=coordinates,
                reason=candidate.reason,
            )
        )
    return tuple(accepted)


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
    allow_leading_outlier: bool = False,
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

    # A read-only camera observation can include one leading frame from the
    # previous UI state even though the newest three frames have converged.
    # Callers must opt into ignoring that leading sample.  Action execution and
    # post-action verification keep the stricter full-window default.
    required_pairs = min(2, len(deltas)) if allow_leading_outlier else len(deltas)
    evaluated_deltas = deltas[-required_pairs:]
    mean_delta = sum(evaluated_deltas) / len(evaluated_deltas)
    max_delta = max(evaluated_deltas)
    stable = max_delta <= limit
    return LocalFrameStability(
        stable=stable,
        mean_delta=mean_delta,
        max_delta=max_delta,
        frame_count=len(frames),
        threshold=limit,
        reason=(
            (
                f"末尾{required_pairs + 1}帧外圈静态UI一致"
                if allow_leading_outlier
                else "完整采样窗口外圈静态UI一致"
            )
            if stable
            else (
                (
                    f"末尾{required_pairs + 1}帧外圈静态UI变化"
                    if allow_leading_outlier
                    else "完整采样窗口外圈静态UI变化"
                )
                + f"{max_delta:.1f}超过阈值{limit:.1f}"
            )
        ),
    )


def measure_static_band_identity_delta(
    reference_frames: list[Image.Image] | tuple[Image.Image, ...],
    candidate_frames: list[Image.Image] | tuple[Image.Image, ...],
) -> float:
    """Compare camera framing while tolerating changes in central App content.

    The outer-band descriptor deliberately excludes most of the editable or
    scrollable center.  Each candidate is matched to its closest fresh
    pre-action reference, then the median candidate distance is returned so a
    single transient frame cannot establish camera-return authority.
    """

    references = tuple(reference_frames)
    candidates = tuple(candidate_frames)
    if not references or not candidates:
        raise ValueError("取景身份比较需要动作前后真实帧。")
    sizes = {frame.size for frame in references + candidates}
    if len(sizes) != 1:
        raise ValueError("取景身份比较的动作前后画面尺寸不一致。")
    reference_sheets = tuple(_static_band_sheet(frame) for frame in references)
    candidate_sheets = tuple(_static_band_sheet(frame) for frame in candidates)
    nearest_deltas = sorted(
        min(
            float(
                ImageStat.Stat(
                    ImageChops.difference(candidate, reference)
                ).mean[0]
            )
            for reference in reference_sheets
        )
        for candidate in candidate_sheets
    )
    middle = len(nearest_deltas) // 2
    if len(nearest_deltas) % 2:
        return nearest_deltas[middle]
    return (nearest_deltas[middle - 1] + nearest_deltas[middle]) / 2.0


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
