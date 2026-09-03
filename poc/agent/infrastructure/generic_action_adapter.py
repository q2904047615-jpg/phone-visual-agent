"""Infrastructure adapter for one observed, verified device action."""

from __future__ import annotations

from agent.domain.validation import NormalizedBounds, NormalizedPoint, canonical_digest, dataclass_wire, reject_if
import math
import statistics
import time
import re
import uuid
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageChops, ImageFilter, ImageStat

from agent.domain import DeviceActionRequest, DeviceExecutionError, DeviceExecutor
from agent.domain.confirmation_authority import ConfirmationAuthority
from agent.domain.text_transport import text_digest
from agent.application.text_transport import TrustedTextTransportPort
from agent.infrastructure import RobotDeviceExecutor
from agent.domain.generic_goal import GenericIntentDraft
from agent.infrastructure.generic_scene_observer import SingleStepGenericSceneObserver
from agent.application.action_adapter import GenericActionAdapterError
from agent.domain.action_capabilities import build_device_capability_snapshot
from agent.infrastructure.windows_ocr_runtime import find_text, recognize as recognize_ocr
from agent.infrastructure.observation_images import (
    measure_frame_sharpness,
    measure_local_stability,
    measure_material_visual_transition,
    measure_static_band_identity_delta,
)
from agent.infrastructure.orientation_safety import (
    OrientationCredential,
    OrientationFrameMismatchError,
    OrientationSafetyError,
    _mint_locally_verified_qwerty_credential,
    _mint_single_step_scene_credential,
    frame_fingerprint,
    validate_device_id,
)
from agent.infrastructure.qwen_runtime_errors import classify_qwen_error
from agent.infrastructure.model_failure_diagnostics import model_failure_payload, persist_model_failure_payload
from agent.domain.semantic_action import SemanticAction
from agent.domain.ui_scene import UIElement, UIScene, UISceneError
from agent.domain.universal_action_controller import (
    LOCAL_POINT_GROUNDING_SOURCE,
    MAX_LOCAL_GROUNDING_BOX_GAP,
    LocalPointGrounding,
    ResolvedSemanticAction,
    UniversalActionController,
    UniversalActionError,
)
from agent.infrastructure.robot_controller import WorkflowNotReady, qwerty_keyboard_config_from_anchors


QWEN_FAILURE_DIAGNOSTIC_VERSION = "2026-08-17-qwen-failure-diagnostic-v1"

def stable_qwerty_ocr_anchors(frames: tuple[Image.Image, ...] | list[Image.Image], anchors: dict[str, Any], *,
    ocr_recognizer: Any=recognize_ocr) -> dict[str, list[int]] | None:
    """Refine a QWERTY contract with stable local OCR row geometry."""

    frame_list = list(frames)[-3:]
    required = {"q", "p", "a", "l", "z", "m", "backspace"}
    if len(frame_list) != 3 or not isinstance(anchors, dict):
        return None
    try:
        original = {key: [round(float(value[0])), round(float(value[1]))] for key,
            value in anchors.items() if isinstance(value, (list, tuple)) and len(value) == 2}
        if set(original) != required or any((not 0 <= point[0] <= 1000 for point in original.values())):
            return None
        row_keys = (("q", "p"), ("a", "l"), ("z", "m", "backspace"))
        model_rows = tuple(statistics.mean(original[key][1] for key in keys) for keys in row_keys)
        trusted_model_y = bool(all((0 <= point[1] <= 1000 for point in original.values()))
            and model_rows[0] < model_rows[1] < model_rows[2] and (min(model_rows[1] - model_rows[0],
            model_rows[2] - model_rows[1]) >= 35))
        horizontal_probe = {key: list(value) for key, value in original.items()}
        for (y, keys) in zip((650, 750, 850), row_keys):
            for key in keys:
                horizontal_probe[key][1] = y
        qwerty_keyboard_config_from_anchors(horizontal_probe)

        alphabets = ("qwertyuiop", "asdfghjkl", "zxcvbnm")

        def clusters(hits: list[tuple[str, float | None, float]], height: int) -> list[tuple[float, int, list]]:
            groups: list[list[tuple[str, float | None, float]]] = []
            tolerance = max(8.0, height * 0.025)
            for hit in sorted(hits, key=lambda item: item[2]):
                if not groups or abs(hit[2] - statistics.median((item[2] for item in groups[-1]))) > tolerance:
                    groups.append([hit])
                else:
                    groups[-1].append(hit)
            return [(float(statistics.median((item[2] for item in group))), len({item[0] for item in group}),
                group) for group in groups if len({item[0] for item in group}) >= 2]

        def horizontal_fit(group: list[tuple[str, float | None, float]], alphabet: str, width: int):
            centers: dict[str, list[float]] = {}
            for (label, center_x, _) in group:
                if center_x is not None and math.isfinite(center_x):
                    centers.setdefault(label, []).append(1000.0 * center_x / width)
            points = [(alphabet.index(label), statistics.median(values)) for label, values in centers.items()]
            if width <= 0 or len(points) < 3:
                return None
            mean_index = statistics.mean(index for index, _ in points)
            mean_x = statistics.mean(x for _, x in points)
            denominator = sum((index - mean_index) ** 2 for index, _ in points)
            if denominator <= 0:
                return None
            pitch = sum((index - mean_index) * (x - mean_x) for index, x in points) / denominator
            first_x = mean_x - pitch * mean_index
            residual = max(abs(x - (first_x + pitch * index)) for index, x in points)
            return (first_x, pitch) if 45 <= pitch <= 130 and 0 <= first_x and (first_x + pitch * (len(alphabet) - 1) <=
                1000) and (residual <= 25) else None

        def combined_fit(top_group: list, bottom_group: list, width: int):
            top_fit = horizontal_fit(top_group, alphabets[0], width)
            bottom_fit = horizontal_fit(bottom_group, alphabets[2], width)
            count = int(top_fit is not None) + int(bottom_fit is not None)
            if any((item[1] is not None for item in (*top_group, *bottom_group))) and (not count):
                return None
            if top_fit and bottom_fit:
                top_first, top_pitch = top_fit
                bottom_first, bottom_pitch = bottom_fit
                if abs(top_pitch - bottom_pitch) > 18 or abs(bottom_first - (top_first + 1.5 * top_pitch)) > 60:
                    return None
                return (count, statistics.mean((top_first, bottom_first - 1.5 * bottom_pitch)),
                    statistics.mean((top_pitch, bottom_pitch)))
            if top_fit:
                return count, *top_fit
            if bottom_fit:
                first, pitch = bottom_fit
                return count, first - 1.5 * pitch, pitch
            return count, None, None

        per_frame_rows: list[tuple[float, float, float | None, float | None]] = []
        for frame in frame_list:
            hits = {alphabet: [] for alphabet in alphabets}
            payload = ocr_recognizer(frame.convert("RGB"), "zh-Hans-CN", scale=3.0)
            words = (word for line in payload.get("lines") or [] for word in line.get("words") or [])
            for word in words:
                label = str(word.get("text") or "").strip().casefold()
                if len(label) != 1 or not label.isascii() or (not label.isalpha()):
                    continue
                left, width = word.get("left"), word.get("width")
                has_x = all(isinstance(part, (int, float)) and not isinstance(part, bool) for part in (left, width))
                hit = (label, float(left) + float(width) / 2.0 if has_x else None, float(word.get('top',
                    0)) + float(word.get('height', 0)) / 2.0)
                for alphabet in alphabets:
                    if label in alphabet:
                        hits[alphabet].append(hit)
                        break
            row_clusters = [clusters(hits[alphabet], frame.height) for alphabet in alphabets]
            candidates = []
            expected_top = frame.height * model_rows[0] / 1000.0
            expected_bottom = frame.height * model_rows[2] / 1000.0
            for (top_y, top_count, top_group) in row_clusters[0]:
                for (bottom_y, bottom_count, bottom_group) in row_clusters[2]:
                    if not frame.height * 0.08 <= bottom_y - top_y <= frame.height * 0.22:
                        continue
                    midpoint = (top_y + bottom_y) / 2.0
                    middle_count = max((count for center, count, _
                        in row_clusters[1] if abs(center - midpoint) <= frame.height * 0.04), default=0)
                    fitted = combined_fit(top_group, bottom_group, frame.width)
                    if fitted is None:
                        continue
                    fit_count, first_x, pitch = fitted
                    distance = abs(top_y - expected_top) + abs(bottom_y - expected_bottom) if trusted_model_y else 0.0
                    candidates.append((fit_count, min(top_count, bottom_count), top_count + middle_count + bottom_count,
                        -distance, top_y, bottom_y, first_x if first_x is not None else math.nan,
                        pitch if pitch is not None else math.nan))
            if candidates:
                *_, top_y, bottom_y, first_x, pitch = max(candidates)
                per_frame_rows.append((top_y, bottom_y, first_x if math.isfinite(first_x) else None,
                    pitch if math.isfinite(pitch) else None))

        if (len(per_frame_rows) < 2 or any((max((row[index] for row in per_frame_rows)) - min((row[index] for row
            in per_frame_rows)) > 10 for index in (0, 1)))):
            return None
        height = frame_list[-1].height
        top_y, bottom_y = (round(1000 * statistics.median((row[index] for row in per_frame_rows)) / height) for index
            in (0, 1))
        middle_y, span = round((top_y + bottom_y) / 2.0), bottom_y - top_y
        if (span <= 0 or (trusted_model_y and any((abs(actual - original[key][1]) > span * 1.5 for key, actual in (('q',
            top_y), ('a', middle_y), ('z', bottom_y)))))):
            return None

        snapped = {key: list(value) for key, value in original.items()}
        horizontal = [(float(first), float(pitch)) for _, _, first,
            pitch in per_frame_rows if first is not None and pitch is not None]
        if len(horizontal) >= 2:
            first_values, pitch_values = zip(*horizontal)
            if max(first_values) - min(first_values) > 15 or max(pitch_values) - min(pitch_values) > 6:
                return None
            q_x, pitch = round(statistics.median(first_values)), statistics.median(pitch_values)
            offsets = {"q": 0, "p": 9, "a": .5, "l": 8.5, "z": 1.5, "m": 7.5, "backspace": 9}
            local_x = {key: round(q_x + offset * pitch) for key, offset in offsets.items()}
            if any((not 0 <= value <= 1000 for value in local_x.values())):
                return None
            for (key, x) in local_x.items():
                snapped[key][0] = x
        for (y, keys) in zip((top_y, middle_y, bottom_y), row_keys):
            for key in keys:
                snapped[key][1] = y
        qwerty_keyboard_config_from_anchors(snapped)
        return snapped
    except Exception:
        return None


class LocalPointGroundingAmbiguousError(RuntimeError):
    """Positive local evidence found multiple equally plausible surfaces."""


@dataclass(frozen=True)
class _SurfaceCandidate:
    bounds: tuple[int, int, int, int]
    point: tuple[int, int]
    area: int
    interior_depth: int
    score: float


def _target_for_local_point_grounding(scene: UIScene, action: SemanticAction) -> UIElement | None:
    if action.action not in {'tap_semantic', 'dismiss_overlay', 'double_tap', 'long_press'}:
        return None
    element_id = str(action.params.get("element_id") or "").strip()
    try:
        element = scene.get_element(element_id)
    except UISceneError:
        return None
    if (element.role not in {'button', 'icon', 'text', 'tab', 'toggle', 'image', 'list_item'}
        or element.element_id.startswith('local_audited_')
        or element.meaning == 'application_text_input'
        or element.meaning.startswith(('input_', 'ime_', 'switch_keyboard_'))
        or str(action.params.get('label') or '') != element.label
        or str(action.params.get('target') or '') != element.meaning
        or str(action.params.get('role') or '') != element.role):
        return None
    return element


def _grounding_from_geometry(*, scene: UIScene, element: UIElement,
    grounded_bounds: NormalizedBounds, grounded_point: NormalizedPoint,
    matched_frames: int, inspected_frames: int) -> LocalPointGrounding:
    grounding = LocalPointGrounding(source=LOCAL_POINT_GROUNDING_SOURCE, scene_fingerprint=scene.fingerprint,
        element_id=element.element_id, label=element.label,
        model_bounds=tuple(float(value) for value in element.bounds),
        proposed_point=tuple(float(value) for value in element.center),
        grounded_bounds=tuple(float(value) for value in grounded_bounds),
        grounded_point=tuple(float(value) for value in grounded_point),
        matched_frames=matched_frames, inspected_frames=inspected_frames)
    grounding.validate_for(scene, element)
    return grounding


def _stable_exact_text_grounding(frames: list[Image.Image], scene: UIScene, element: UIElement,
    *, ocr_recognizer: Any) -> LocalPointGrounding | None:
    """Return stable unique exact-label geometry without owning fallback policy."""

    label = element.label.strip()
    compact_label = re.sub(r"[\s\u3000]+", "", label)
    if len(compact_label) < 2 or len(compact_label) > 64:
        return None

    def unique_match_geometry(frame: Image.Image) -> tuple[NormalizedBounds, NormalizedPoint] | None:
        payload = ocr_recognizer(frame.convert('RGB'), 'zh-Hans-CN', scale=1.5)
        matches = find_text(payload, label)
        if len(matches) != 1 or frame.width <= 0 or frame.height <= 0:
            return None
        match = matches[0]
        bounds = (float(match.left) / frame.width, float(match.top) / frame.height,
            float(match.left + match.width) / frame.width, float(match.top + match.height) / frame.height)
        left, top, right, bottom = bounds
        if not (0.0 <= left < right <= 1.0 and 0.0 <= top < bottom <= 1.0):
            return None
        return bounds, ((left + right) / 2.0, (top + bottom) / 2.0)

    inspected_frames = 1
    newest = unique_match_geometry(frames[-1])
    if newest is None:
        return None
    proposed_point = element.center
    if math.dist(proposed_point, newest[1]) <= 0.025:
        return None

    stable_matches = [newest]
    for frame in reversed(frames[:-1]):
        inspected_frames += 1
        candidate = unique_match_geometry(frame)
        if candidate is None or math.dist(candidate[1], newest[1]) > 0.015:
            continue
        if any(abs(candidate[0][index + 2] - candidate[0][index]
            - (newest[0][index + 2] - newest[0][index])) > 0.03 for index in (0, 1)):
            continue
        stable_matches.append(candidate)
        if len(stable_matches) >= 2:
            break
    if len(stable_matches) < 2:
        return None

    grounded_bounds = tuple(float(statistics.median(match[0][index] for match in stable_matches))
        for index in range(4))
    grounded_point = ((grounded_bounds[0] + grounded_bounds[2]) / 2.0,
        (grounded_bounds[1] + grounded_bounds[3]) / 2.0)
    if math.dist(proposed_point, grounded_point) <= 0.025:
        return None
    return _grounding_from_geometry(scene=scene, element=element, grounded_bounds=grounded_bounds,
        grounded_point=grounded_point, matched_frames=len(stable_matches), inspected_frames=inspected_frames)


def _box_intersection(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> int:
    return max(0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0, min(first[3], second[3]) - max(first[1], second[1]))


def _box_iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    intersection = _box_intersection(first, second)
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - intersection
    return float(intersection) / union if union > 0 else 0.0


def _deepest_component_point(component: np.ndarray, *, offset_x: int,
    offset_y: int) -> tuple[tuple[int, int], int]:
    """Choose a maximum-depth point after surrounding the component with background."""

    # A connected component is cropped to its own bounds.  Eroding that tight crop
    # lets Pillow replicate foreground edge pixels outside the image and can leave a
    # false "deepest" point on the component boundary.  An explicit background pad
    # makes every visible edge participate in the distance-to-boundary calculation.
    padded = np.pad(np.asarray(component, dtype=bool), 1, mode='constant', constant_values=False)
    mask = Image.fromarray(np.where(padded, 255, 0).astype(np.uint8), mode='L')
    deepest = np.argwhere(padded)
    interior_depth = 0
    for _ in range(max(1, min(component.shape) // 2 + 1)):
        eroded = mask.filter(ImageFilter.MinFilter(3))
        coordinates = np.argwhere(np.asarray(eroded, dtype=np.uint8) > 0)
        if coordinates.size == 0:
            break
        mask = eroded
        deepest = coordinates
        interior_depth += 1
    center_y = float(np.median(deepest[:, 0]))
    center_x = float(np.median(deepest[:, 1]))
    nearest = min(deepest, key=lambda value: (float(value[0]) - center_y) ** 2
        + (float(value[1]) - center_x) ** 2)
    return ((offset_x + int(nearest[1]) - 1, offset_y + int(nearest[0]) - 1),
        interior_depth)


def _surface_candidates(frame: Image.Image, model_bounds: NormalizedBounds) -> list[_SurfaceCandidate]:
    """Detect generic filled control surfaces near one already-selected model box."""

    rgb = frame.convert('RGB')
    width, height = rgb.size
    if width <= 0 or height <= 0:
        return []
    model = (round(model_bounds[0] * width), round(model_bounds[1] * height),
        round(model_bounds[2] * width), round(model_bounds[3] * height))
    model_width = max(1, model[2] - model[0])
    model_height = max(1, model[3] - model[1])
    margin_x = round(min(width * 0.05, model_width * 0.50))
    margin_y = round(min(height * 0.055, model_height * 1.20))
    crop_box = (max(0, model[0] - margin_x), max(0, model[1] - margin_y),
        min(width, model[2] + margin_x), min(height, model[3] + margin_y))
    if crop_box[2] - crop_box[0] < 3 or crop_box[3] - crop_box[1] < 3:
        return []
    pixels = np.asarray(rgb.crop(crop_box).filter(ImageFilter.GaussianBlur(2)), dtype=np.int16)
    crop_height, crop_width = pixels.shape[:2]
    seed_points: set[tuple[int, int]] = set()
    for x_fraction in (0.10, 0.30, 0.50, 0.70, 0.90):
        for y_fraction in (0.12, 0.32, 0.52, 0.72):
            seed_x = max(0, min(crop_width - 1,
                round(model[0] + model_width * x_fraction) - crop_box[0]))
            seed_y = max(0, min(crop_height - 1,
                round(model[1] + model_height * y_fraction) - crop_box[1]))
            seed_points.add((seed_x, seed_y))
    # A model box can be slightly beyond a visible control instead of merely
    # overlapping it.  Probe the already-bounded neighborhood as well; later
    # geometry still requires one stable component within the domain gap.
    for x_fraction in (0.08, 0.22, 0.38, 0.55, 0.72, 0.88):
        for y_fraction in (0.08, 0.20, 0.34, 0.50, 0.68, 0.85):
            seed_points.add((max(0, min(crop_width - 1, round((crop_width - 1) * x_fraction))),
                max(0, min(crop_height - 1, round((crop_height - 1) * y_fraction)))))
    seeds = [(seed_x, seed_y, pixels[seed_y, seed_x])
        for seed_x, seed_y in sorted(seed_points)]

    raw_candidates: list[_SurfaceCandidate] = []
    model_area = model_width * model_height
    model_diagonal = max(1.0, math.hypot(model_width, model_height))
    model_center = ((model[0] + model[2]) / 2.0, (model[1] + model[3]) / 2.0)
    mask_cache: dict[tuple[int, int, int], np.ndarray] = {}
    component_claims: dict[tuple[int, int, int], np.ndarray] = {}
    for seed_x, seed_y, seed_color in seeds:
        color_key = tuple(int(value) // 8 for value in seed_color)
        mask_array = mask_cache.get(color_key)
        if mask_array is None:
            delta = np.abs(pixels - seed_color)
            squared_delta = delta.astype(np.int32) ** 2
            raw_mask = np.logical_and(np.max(delta, axis=2) <= 48,
                np.sum(squared_delta, axis=2) <= 58 * 58)
            mask = Image.fromarray(np.where(raw_mask, 255, 0).astype(np.uint8), mode='L')
            mask = mask.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(5))
            mask_array = np.asarray(mask, dtype=np.uint8) > 0
            mask_cache[color_key] = mask_array
        claimed = component_claims.setdefault(color_key, np.zeros((crop_height, crop_width), dtype=bool))
        if not mask_array[seed_y, seed_x] or claimed[seed_y, seed_x]:
            continue
        queue: deque[tuple[int, int]] = deque(((seed_x, seed_y),))
        visited = np.zeros((crop_height, crop_width), dtype=bool)
        visited[seed_y, seed_x] = True
        points: list[tuple[int, int]] = []
        while queue:
            x, y = queue.popleft()
            points.append((x, y))
            for next_x, next_y in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if (0 <= next_x < crop_width and 0 <= next_y < crop_height
                    and mask_array[next_y, next_x] and not visited[next_y, next_x]):
                    visited[next_y, next_x] = True
                    queue.append((next_x, next_y))
        claimed |= visited
        if not points:
            continue
        xs, ys = zip(*points)
        local_bounds = (min(xs), min(ys), max(xs) + 1, max(ys) + 1)
        bounds = (local_bounds[0] + crop_box[0], local_bounds[1] + crop_box[1],
            local_bounds[2] + crop_box[0], local_bounds[3] + crop_box[1])
        area = len(points)
        box_width, box_height = bounds[2] - bounds[0], bounds[3] - bounds[1]
        fill_ratio = area / max(1, box_width * box_height)
        touched_edges = sum((local_bounds[0] <= 1, local_bounds[1] <= 1,
            local_bounds[2] >= crop_width - 1, local_bounds[3] >= crop_height - 1))
        overlap = _box_intersection(bounds, model)
        overlap_ratio = overlap / max(1, min(box_width * box_height, model_area))
        horizontal_gap = max(0, bounds[0] - model[2], model[0] - bounds[2]) / width
        vertical_gap = max(0, bounds[1] - model[3], model[1] - bounds[3]) / height
        normalized_gap = math.hypot(horizontal_gap, vertical_gap)
        if (area < max(20, round(model_area * 0.10)) or area > model_area * 8
            or box_width < model_width * 0.25 or box_height < model_height * 0.22
            or fill_ratio < 0.28 or touched_edges >= 2
            or (overlap_ratio < 0.12 and normalized_gap > MAX_LOCAL_GROUNDING_BOX_GAP)):
            continue
        component = visited[local_bounds[1]:local_bounds[3], local_bounds[0]:local_bounds[2]]
        point, interior_depth = _deepest_component_point(
            component, offset_x=bounds[0], offset_y=bounds[1])
        required_interior_depth = max(2, round(min(box_width, box_height) * 0.08))
        if interior_depth < required_interior_depth:
            continue
        center_distance = math.dist(point, model_center) / model_diagonal
        area_ratio = max(1e-6, area / model_area)
        proximity_score = max(0.0, 1.0 - normalized_gap / MAX_LOCAL_GROUNDING_BOX_GAP)
        depth_score = min(1.0, interior_depth / max(1.0, min(box_width, box_height) * 0.25))
        score = (overlap_ratio * 3.0 + proximity_score * 1.5 + depth_score * 0.5
            - center_distance - abs(math.log(area_ratio)) * 0.20)
        raw_candidates.append(_SurfaceCandidate(bounds=bounds, point=point, area=area,
            interior_depth=interior_depth, score=score))

    deduplicated: list[_SurfaceCandidate] = []
    for candidate in sorted(raw_candidates, key=lambda item: item.score, reverse=True):
        if any(_box_iou(candidate.bounds, known.bounds) >= 0.72 for known in deduplicated):
            continue
        deduplicated.append(candidate)
    return deduplicated


def _stable_surface_grounding(frames: list[Image.Image], scene: UIScene,
    element: UIElement) -> LocalPointGrounding | None:
    if len({frame.size for frame in frames}) != 1:
        return None
    per_frame = [_surface_candidates(frame, element.bounds) for frame in reversed(frames)]
    newest = per_frame[0]
    if not newest:
        return None
    if len(newest) > 1 and newest[0].score - newest[1].score < 0.35:
        raise LocalPointGroundingAmbiguousError("当前目标附近存在多个同等可信的控件表面，拒绝猜测落点。")
    selected = newest[0]
    stable = [selected]
    inspected_frames = 1
    for candidates in per_frame[1:]:
        inspected_frames += 1
        matches = [candidate for candidate in candidates
            if _box_iou(candidate.bounds, selected.bounds) >= 0.60]
        if len(matches) > 1 and matches[0].score - matches[1].score < 0.35:
            raise LocalPointGroundingAmbiguousError("当前目标附近的控件表面跨帧不唯一，拒绝猜测落点。")
        if matches:
            stable.append(matches[0])
        if len(stable) >= 2:
            break
    if len(stable) < 2:
        return None

    width, height = frames[-1].size
    if width <= 0 or height <= 0:
        return None
    grounded_bounds = tuple(float(statistics.median(candidate.bounds[index] for candidate in stable))
        / (width if index in (0, 2) else height) for index in range(4))
    grounded_point = (float(statistics.median(candidate.point[0] for candidate in stable)) / width,
        float(statistics.median(candidate.point[1] for candidate in stable)) / height)
    if math.dist(element.center, grounded_point) <= 0.012:
        return None
    if math.dist(element.center, grounded_point) > 0.12:
        return None
    return _grounding_from_geometry(scene=scene, element=element, grounded_bounds=grounded_bounds,
        grounded_point=grounded_point, matched_frames=len(stable), inspected_frames=inspected_frames)


def stable_visual_point_grounding(frames: tuple[Image.Image, ...] | list[Image.Image], scene: UIScene,
    action: SemanticAction, *, ocr_recognizer: Any=recognize_ocr) -> LocalPointGrounding | None:
    """Refine one Qwen-selected point from stable OCR or generic surface geometry."""

    frame_list = list(frames)[-3:]
    if not frame_list or not scene.fingerprint:
        return None
    element = _target_for_local_point_grounding(scene, action)
    if element is None:
        return None

    try:
        text_grounding = _stable_exact_text_grounding(frame_list, scene, element,
            ocr_recognizer=ocr_recognizer)
    except Exception:
        text_grounding = None
    if text_grounding is not None:
        return text_grounding
    return _stable_surface_grounding(frame_list, scene, element)


def _persist_qwen_failure_diagnostic(*, evidence_dir: Path | None, prefix: str, raw_response: str, error: Exception,
    diagnostics: dict[str, Any] | None=None) -> Path | None:
    """Persist bounded redacted model output without changing failure policy."""

    raw = str(raw_response or "")
    if evidence_dir is None or not raw:
        return None
    details = diagnostics if isinstance(diagnostics, dict) else {}
    payload = model_failure_payload(
        artifact_version=QWEN_FAILURE_DIAGNOSTIC_VERSION,
        raw=raw,
        failed_stage=str(details.get('failed_stage') or 'unknown'),
        error_type=str(details.get('error_type') or classify_qwen_error(error, raw_response=raw)),
        error_message=str(error),
    )
    return persist_model_failure_payload(
        payload,
        evidence_dir=evidence_dir,
        prefix=prefix,
        default_prefix='observation',
        suffix='qwen_failure',
    )


def persist_observer_failure_diagnostic(observer: Any, *, evidence_dir: Path | None, prefix: str,
    error: Exception) -> tuple[str, ...]:
    diagnostics = getattr(observer, "last_diagnostics", {})
    raw_response = getattr(observer, "last_raw_response", "")
    try:
        path = _persist_qwen_failure_diagnostic(evidence_dir=evidence_dir, prefix=prefix, raw_response=raw_response,
            error=error, diagnostics=diagnostics if isinstance(diagnostics, dict) else None)
    except Exception as diagnostic_error:
        if isinstance(diagnostics, dict):
            diagnostics['diagnostic_persistence_error'] = type(diagnostic_error).__name__
        return ()
    return (str(path),) if path is not None else ()


@dataclass(frozen=True)
class GenericActionExecutionResult:
    requested_action: SemanticAction
    rebound_action: SemanticAction
    resolved_action: ResolvedSemanticAction
    before_scene: UIScene
    after_scene: UIScene
    planned_scene_fingerprint: str
    confirmation_frame_identity_verified: bool
    confirmation_frame_delta: float | None
    physical_actions: int
    after_model_decision: dict[str, Any] = field(repr=False, compare=False)
    primary_input_confirmation_reused: bool = False
    action_outcome: str = "matched"
    verification_errors: tuple[str, ...] = ()
    robot_result: Any = None
    hardware_receipt: dict[str, Any] | None = None
    execution_metadata: dict[str, Any] = field(default_factory=dict)
    evidence: tuple[str, ...] = ()
    after_frames: tuple[Image.Image, ...] = field(default_factory=tuple, repr=False, compare=False)
    after_frame_paths: tuple[str, ...] = ()
    observation_errors: tuple[str, ...] = ()
    controller_transition_evidence: tuple[str, ...] = ()
    before_frames: tuple[Image.Image, ...] = field(default_factory=tuple, repr=False, compare=False)
    before_frame_paths: tuple[str, ...] = ()
    orientation_credential: OrientationCredential | None = None

    def __post_init__(self) -> None:
        # Freeze sampling lists because a completed result is a live promotion source.
        object.__setattr__(self, "after_frames", tuple(self.after_frames))
        object.__setattr__(self, "before_frames", tuple(self.before_frames))
        reject_if(not isinstance(self.after_model_decision, Mapping),
            ValueError("动作结果缺少同一动作后截图的 Qwen decision。"))
        object.__setattr__(self, "after_model_decision", dict(self.after_model_decision))
        object.__setattr__(self, 'controller_transition_evidence', tuple(self.controller_transition_evidence))
        object.__setattr__(self, 'execution_metadata', dict(self.execution_metadata))

    def to_dict(self) -> dict[str, Any]:
        value = dataclass_wire(self, omit=('after_frames', 'before_frames', 'after_model_decision'))
        value.update(after_frame_count=len(self.after_frames), before_frame_count=len(self.before_frames),
            robot_result=self.robot_result)
        return value


class GenericSingleActionAdapter:
    """The only generic bridge from a verified scene to one robot action."""

    PHYSICAL_KINDS = frozenset({'tap_semantic', 'dismiss_overlay', 'double_tap', 'scroll', 'swipe_element', 'reveal_system_navigation',
        'back', 'home', 'open_recent_apps', 'input_verified_text', 'press_enter', 'clear_verified_text', 'long_press',
        'drag'})
    DEVICE_ACTION_KINDS = PHYSICAL_KINDS | {'launch_app'}

    @staticmethod
    def _post_action_goal(goal: GenericIntentDraft, *, authority: ConfirmationAuthority | None,
        resolved: ResolvedSemanticAction, physical_actions: int) -> GenericIntentDraft:
        """Project one scoped execution fact into the immediate post-action Qwen call."""

        if authority is None or physical_actions != 1:
            return goal
        entities = dict(goal.entities)
        active = entities.get('active_subgoal_visual_context')
        if not isinstance(active, dict) or str(active.get('subgoal_id') or '') != authority.subgoal_id:
            return goal
        current_receipt = active.get('transition_receipt')
        if not isinstance(current_receipt, dict):
            return goal
        active = dict(active)
        active['transition_receipt'] = {
            'state': 'executed',
            'subgoal_id': authority.subgoal_id,
            'required_operation': str(current_receipt.get('required_operation') or ''),
            'executed_operation': resolved.kind,
            'effect_ids': sorted(authority.effect_ids),
        }
        entities['active_subgoal_visual_context'] = active
        projected = replace(goal, entities=entities)
        projected.validate()
        return projected
    GEOMETRY_BOUND_KINDS = frozenset({'tap_semantic', 'dismiss_overlay', 'double_tap', 'input_verified_text',
        'press_enter', 'clear_verified_text', 'long_press', 'drag'})
    INDEPENDENT_GEOMETRY_AUDIT_KINDS = GEOMETRY_BOUND_KINDS
    LOCAL_INPUT_AUXILIARY_MEANINGS = frozenset({'ime_exact_candidate', 'input_exact_literal_key',
        'input_exact_enter_key', 'switch_keyboard_layout', 'switch_keyboard_case', 'switch_keyboard_input_mode'})

    def _local_qwerty_orientation_credential(self, *, requested: SemanticAction, scene: UIScene,
        frames: list[Image.Image]) -> OrientationCredential | None:
        if (requested.params.get('text_transport') == 'adb_keyboard'
            or requested.action not in {'tap_semantic', 'press_enter', 'input_verified_text',
            'clear_verified_text'} or not callable(self.qwerty_row_snapper)):
            return None
        element_id = str(requested.params.get("element_id") or "").strip()
        try:
            target = scene.get_element(element_id)
        except UISceneError:
            return None
        input_id = element_id if target.meaning == 'application_text_input' else str(target.states.get(
            'input_element_id') or '').strip()
        try:
            input_element = scene.get_element(input_id)
        except UISceneError:
            return None
        geometry = input_element.states.get("keyboard_geometry")
        if (input_element.meaning != 'application_text_input' or input_element.states.get('focused') is not True
            or (not isinstance(geometry, dict)) or (geometry.get('type') != 'qwerty')
            or (geometry.get('source') != 'input_structure_audit')):
            return None
        snapped = self.qwerty_row_snapper(frames, geometry.get("anchors"))
        if not isinstance(snapped, dict):
            return None
        try:
            qwerty_keyboard_config_from_anchors(snapped)
            return _mint_locally_verified_qwerty_credential(device_id=self.device_id,
                scene_fingerprint=scene.fingerprint, frame=frames[-1], anchors=snapped)
        except (OrientationSafetyError, WorkflowNotReady, TypeError, ValueError):
            return None

    def _single_step_scene_orientation_credential(self, *, scene: UIScene,
        frames: list[Image.Image]) -> OrientationCredential:
        """Mint a local frame binding without consuming optional Qwen metadata."""

        reject_if(not frames, OrientationSafetyError("单步方向绑定缺少当前稳定帧。"))
        return _mint_single_step_scene_credential(device_id=self.device_id, scene_fingerprint=scene.fingerprint,
            frame=frames[-1].convert('RGB'))

    def supported_action_kinds(self) -> frozenset[str]:
        """Return only actions backed by callable methods on this device."""

        capability_provider = getattr(self.robot, "hardware_capabilities", None)
        declared = capability_provider() if callable(capability_provider) else {}
        if not isinstance(declared, dict):
            declared = {}
        methods = {'tap_semantic': 'vision_tap_relative', 'dismiss_overlay': 'vision_dismiss_overlay_relative',
            'double_tap': 'vision_double_tap_relative', 'reveal_system_navigation': 'vision_reveal_system_navigation',
            'back': 'vision_android_back', 'home': 'vision_android_home',
            'open_recent_apps': 'vision_android_recent_apps', 'input_verified_text': 'vision_type_text_with_layout',
            'long_press': 'vision_long_press_relative', 'drag': 'vision_drag_relative'}
        supported = {'wait_for_change'} if bool(declared.get('wait_for_change', True)) else set()
        supported.update((action for action, method in methods.items() if bool(declared.get(action,
            True)) and callable(getattr(self.robot, method, None))))
        if (bool(declared.get('swipe', True)) and any((callable(getattr(self.robot, f'vision_swipe_{direction}',
            None)) for direction in ('up', 'down', 'left', 'right')))):
            supported.update({"scroll", "swipe_element"})
        if 'tap_semantic' in supported:
            supported.add("press_enter")
        if bool(declared.get('input_verified_text', True)) and callable(getattr(self.robot, 'vision_clear_text', None)):
            supported.add("clear_verified_text")
        if self.text_transport is not None:
            profile = self.text_transport.profile
            profile.validate()
            if profile.enabled and 'append_text' in profile.capabilities:
                supported.add('input_verified_text')
            else:
                supported.discard('input_verified_text')
            if profile.enabled and 'clear_text' in profile.capabilities:
                supported.add('clear_verified_text')
            else:
                supported.discard('clear_verified_text')
        if self.app_launcher is not None and bool(getattr(self.app_launcher, 'enabled', False)):
            supported.add('launch_app')
        return frozenset(supported)

    def text_transport_profile(self) -> Any | None:
        return self.text_transport.profile if self.text_transport is not None else None

    def resolve_app_launch_target(self, app_id: str, app_name: str) -> Any | None:
        resolver = getattr(self.app_launcher, 'resolve', None)
        return resolver(app_id, app_name) if callable(resolver) else None

    def capability_snapshot(self) -> Any:
        provider = getattr(self.robot, "hardware_capability_profile", None)
        raw_profile = provider() if callable(provider) else None
        return build_device_capability_snapshot(device_id=str(getattr(self.robot, 'device_id', '') or 'unknown-device'),
            supported_actions=self.supported_action_kinds(), raw_profile=raw_profile)

    def __init__(self, *, capture: Callable[[], Image.Image], observer: SingleStepGenericSceneObserver, robot: Any,
        device_executor: DeviceExecutor | None=None, app_launcher: Any=None,
        text_transport: TrustedTextTransportPort | None=None,
        controller: UniversalActionController | None=None,
        frame_interval: float=0.37, post_action_settle: float=1.5, post_action_timeout: float | None=None,
        post_action_continuous_timeout: float | None=None,
        post_action_min_relative_sharpness: float=0.8, post_action_min_reference_sharpness: float=2.0,
        post_action_phone_view_delta_max: float=45.0, confirmation_frame_delta_max: float=6.0,
        qwerty_row_snapper: Callable[[tuple[Image.Image, ...] | list[Image.Image], dict[str, Any]], dict[str,
        Any] | None] | None=None, point_grounder: Callable[[tuple[Image.Image, ...] | list[Image.Image], UIScene,
        SemanticAction], LocalPointGrounding | None] | None=None, require_local_qwerty_row_snap: bool=False,
        device_id: str) -> None:
        self.capture = capture
        self.observer = observer
        self.robot = robot
        self.app_launcher = app_launcher
        self.text_transport = text_transport
        if text_transport is not None:
            text_transport.profile.validate()
            reject_if(text_transport.profile.device_id != device_id,
                ValueError('ADB Keyboard profile 与 adapter device_id 不一致。'))
        self.device_executor = device_executor or RobotDeviceExecutor(robot, app_launcher=app_launcher,
            text_transport=text_transport)
        self.controller = controller or UniversalActionController()
        self.frame_interval = max(0.0, float(frame_interval))
        self.post_action_settle = max(0.0, float(post_action_settle))
        self.post_action_timeout = max(0.0, 10.0 if post_action_timeout is None else float(post_action_timeout))
        self.post_action_continuous_timeout = max(self.post_action_timeout,
            45.0 if post_action_continuous_timeout is None and post_action_timeout
            is None else self.post_action_timeout if post_action_continuous_timeout
            is None else float(post_action_continuous_timeout))
        self.post_action_min_relative_sharpness = max(0.0, min(1.0, float(post_action_min_relative_sharpness)))
        self.post_action_min_reference_sharpness = max(0.0, float(post_action_min_reference_sharpness))
        self.post_action_phone_view_delta_max = max(0.0, float(post_action_phone_view_delta_max))
        self.confirmation_frame_delta_max = max(0.0, float(confirmation_frame_delta_max))
        self.qwerty_row_snapper = qwerty_row_snapper
        self.point_grounder = point_grounder
        self.require_local_qwerty_row_snap = bool(require_local_qwerty_row_snap)
        try:
            self.device_id = validate_device_id(device_id)
        except OrientationSafetyError as exc:
            raise ValueError(str(exc)) from exc

    @staticmethod
    def _confirmation_frame_delta(planned_frames: tuple[Image.Image, ...] | list[Image.Image],
        fresh_frames: list[Image.Image]) -> float:
        reject_if(not planned_frames or not fresh_frames, GenericActionAdapterError("确认前缺少本地真实帧，不能验证画面身份。"))
        planned_sizes = {frame.size for frame in planned_frames}
        fresh_sizes = {frame.size for frame in fresh_frames}
        reject_if(len(planned_sizes) != 1 or len(fresh_sizes) != 1 or planned_sizes != fresh_sizes, GenericActionAdapterError("确认前真实画面尺寸发生变化。"))

        def compact(frame: Image.Image) -> Image.Image:
            return frame.convert("L").resize((96, 160), Image.Resampling.BILINEAR)

        planned = [compact(frame) for frame in planned_frames]
        fresh = [compact(frame) for frame in fresh_frames]
        return min((float(ImageStat.Stat(ImageChops.difference(first,
            second)).mean[0]) for first in planned for second in fresh))

    def _capture_frame(self) -> Image.Image:
        frame = self.capture().convert("RGB")
        reject_if(frame.width < 400 or frame.height < 700, GenericActionAdapterError("摄像头返回残缺画面，停止单步动作。"))
        return frame

    def _capture_frame_burst(self) -> list[Image.Image]:
        frames: list[Image.Image] = []
        for index in range(4):
            frames.append(self._capture_frame())
            if index < 3 and self.frame_interval:
                time.sleep(self.frame_interval)
        return frames

    def _capture_confirmation_frames(self, *, evidence_dir: Path | None, prefix: str) -> tuple[list[Image.Image],
        tuple[str, ...]]:
        """Capture four fresh local frames without re-interpreting the scene."""

        frames = self._capture_frame_burst()
        stability = measure_local_stability(frames)
        paths = self._save_frames(frames, evidence_dir, prefix)
        reject_if(not stability.stable, GenericActionAdapterError(f'确认前本地多帧稳定性检查未通过：{stability.reason}', evidence=paths))
        return frames, paths

    def _capture_scene_once(self, goal: GenericIntentDraft, *, evidence_dir: Path | None=None,
        prefix: str, available_action_kinds: frozenset[str] | None=None
        ) -> tuple[UIScene, list[Image.Image], tuple[str, ...], dict[str, Any]]:
        frames = self._capture_frame_burst()
        paths = self._save_frames(frames, evidence_dir, prefix)
        try:
            scene, model_decision = self._observe_scene(frames, goal.to_dict(),
                available_action_kinds=available_action_kinds)
        except RuntimeError as exc:
            diagnostic_paths = persist_observer_failure_diagnostic(self.observer, evidence_dir=evidence_dir,
                prefix=prefix, error=exc)
            raise GenericActionAdapterError(f'通用页面观察失败：{exc}', evidence=paths + diagnostic_paths) from exc
        return scene, frames, paths, model_decision

    def _observe_scene(self, frames: list[Image.Image] | tuple[Image.Image, ...], goal_context: dict[str, Any], *,
        available_action_kinds: frozenset[str] | None=None
        ) -> tuple[UIScene, dict[str, Any]]:
        kwargs: dict[str, Any] = {'frames': list(frames), 'goal_context': goal_context}
        if getattr(self.observer, 'supports_runtime_action_contract', False) is True:
            supported = self.supported_action_kinds()
            scoped = supported if available_action_kinds is None else frozenset(available_action_kinds)
            reject_if(not scoped or scoped - supported,
                GenericActionAdapterError('当前子目标动作集合为空或超出设备能力。'))
            kwargs['available_action_kinds'] = scoped
        observe_with_decision = getattr(self.observer, "observe_with_decision", None)
        reject_if(not callable(observe_with_decision),
            GenericActionAdapterError("当前观察器不支持同一截图响应中的 scene + decision 合同。"))
        scene, model_decision = observe_with_decision(**kwargs)
        reject_if(not isinstance(model_decision, Mapping),
            GenericActionAdapterError("当前观察缺少同一截图响应中的 Qwen decision。"))
        return scene, dict(model_decision)

    def capture_scene(
        self,
        goal: GenericIntentDraft,
        *,
        evidence_dir: Path | None,
        prefix: str,
        available_action_kinds: frozenset[str] | None=None,
    ) -> tuple[UIScene, list[Image.Image], tuple[str, ...], dict[str, Any]]:
        # One step permits one Qwen request; stability sampling never triggers model resampling.
        try:
            return self._capture_scene_once(goal, evidence_dir=evidence_dir, prefix=f'{prefix}_attempt_1',
                available_action_kinds=available_action_kinds)
        except GenericActionAdapterError as exc:
            error = f"第1轮动作前观察失败：{exc}"
            raise GenericActionAdapterError('动作前通用页面观察失败：' + error, evidence=tuple(exc.evidence),
                observation_errors=(error,)) from exc

    @staticmethod
    def _requires_post_action_relative_clarity(resolved: ResolvedSemanticAction) -> bool:
        """Use relative clarity only when an action preserves a comparable input surface."""

        return resolved.kind in {'input_verified_text', 'press_enter', 'clear_verified_text'}

    @staticmethod
    def _requires_post_action_phone_view_identity(resolved: ResolvedSemanticAction) -> bool:
        return resolved.kind in {'clear_verified_text', 'double_tap', 'drag', 'input_verified_text', 'long_press'}

    def _post_action_timeout_for(self, resolved: ResolvedSemanticAction) -> float:
        if self._requires_post_action_phone_view_identity(resolved):
            return self.post_action_continuous_timeout
        return self.post_action_timeout

    def _capture_stable_post_action_frames(self, *, deadline: float, evidence_dir: Path | None, prefix: str,
        clarity_reference_frames: tuple[Image.Image, ...]=(), require_relative_clarity: bool=False,
        require_phone_view_identity: bool=False) -> tuple[list[Image.Image], tuple[str, ...]]:
        """Wait locally for four stable and, when required, relatively clear frames."""

        frames: list[Image.Image] = []
        last_stability = None
        reference_sharpness = statistics.median((measure_frame_sharpness(frame) for frame
            in clarity_reference_frames)) if (require_relative_clarity
            or require_phone_view_identity) and clarity_reference_frames else None
        clarity_is_comparable = bool(require_relative_clarity and reference_sharpness is not None
            and (reference_sharpness >= self.post_action_min_reference_sharpness))
        phone_view_is_comparable = bool(require_phone_view_identity and reference_sharpness is not None
            and (reference_sharpness >= self.post_action_min_reference_sharpness))
        last_candidate_sharpness: float | None = None
        last_relative_sharpness: float | None = None
        last_phone_view_delta: float | None = None
        while True:
            frames.append(self._capture_frame())
            if len(frames) > 4:
                frames.pop(0)
            if len(frames) == 4:
                last_stability = measure_local_stability(frames)
                if last_stability.stable:
                    clarity_accepted = True
                    if clarity_is_comparable:
                        last_candidate_sharpness = statistics.median(measure_frame_sharpness(frame) for frame in frames)
                        last_relative_sharpness = last_candidate_sharpness / reference_sharpness
                        clarity_accepted = last_relative_sharpness >= self.post_action_min_relative_sharpness
                    phone_view_accepted = True
                    if phone_view_is_comparable:
                        last_phone_view_delta = measure_static_band_identity_delta(clarity_reference_frames, frames)
                        phone_view_accepted = last_phone_view_delta <= self.post_action_phone_view_delta_max
                    if clarity_accepted and phone_view_accepted:
                        return list(frames), self._save_frames(frames, evidence_dir, prefix)
            if time.monotonic() >= deadline:
                paths = self._save_frames(frames, evidence_dir, f"{prefix}_timeout")
                if (last_stability is not None and last_stability.stable and (last_phone_view_delta is not None)
                    and (last_phone_view_delta > self.post_action_phone_view_delta_max)):
                    raise GenericActionAdapterError(
                        "动作后画面已稳定但相机尚未回到手机取景："
                        f"取景差异{last_phone_view_delta:.1f}，要求最多{self.post_action_phone_view_delta_max:.1f}",
                        evidence=paths,
                    )
                if last_stability is not None and last_stability.stable and (last_relative_sharpness is not None):
                    raise GenericActionAdapterError(
                        "动作后画面在限定时间内虽已稳定但仍不够清晰："
                        f"参考清晰度{reference_sharpness:.3f}，候选清晰度{last_candidate_sharpness:.3f}，"
                        f"相对值{last_relative_sharpness:.3f}，要求至少{self.post_action_min_relative_sharpness:.3f}",
                        evidence=paths,
                    )
                reason = last_stability.reason if last_stability is not None else "未能采集到连续4帧"
                raise GenericActionAdapterError(f"动作后画面在限定时间内没有稳定：{reason}", evidence=paths)
            if self.frame_interval:
                time.sleep(min(self.frame_interval, max(0.0, deadline - time.monotonic())))

    def _observe_stable_post_action_scene(self, goal: GenericIntentDraft, *, before: UIScene,
        before_frames: tuple[Image.Image, ...], resolved: ResolvedSemanticAction,
        evidence_dir: Path | None, evidence_prefix: str,
        available_action_kinds: frozenset[str] | None=None) -> tuple[UIScene, tuple[Image.Image, ...], tuple[str, ...], tuple[str, ...], tuple[str,
        ...], tuple[str, ...], dict[str, Any]]:
        action_timeout = self._post_action_timeout_for(resolved)
        if self.post_action_settle:
            time.sleep(min(self.post_action_settle, action_timeout))

        prefix = f"{evidence_prefix}_after_attempt_1"
        try:
            frames, paths = self._capture_stable_post_action_frames(deadline=time.monotonic() + action_timeout,
                evidence_dir=evidence_dir, prefix=prefix, clarity_reference_frames=before_frames,
                require_relative_clarity=self._requires_post_action_relative_clarity(resolved),
                require_phone_view_identity=self._requires_post_action_phone_view_identity(resolved))
        except GenericActionAdapterError as exc:
            raise GenericActionAdapterError(f'动作后画面采集失败：{exc}', evidence=tuple(exc.evidence)) from exc
        all_paths = paths
        try:
            after, model_decision = self._observe_scene(frames, goal.to_dict(),
                available_action_kinds=available_action_kinds)
        except RuntimeError as exc:
            all_paths += persist_observer_failure_diagnostic(self.observer, evidence_dir=evidence_dir, prefix=prefix,
                error=exc)
            observation_errors = (f"第1轮动作后观察失败：{exc}",)
            raise GenericActionAdapterError('通用页面观察失败：' + observation_errors[0], evidence=all_paths,
                observation_errors=observation_errors) from exc

        try:
            controller_evidence = self.controller.verify_after_action(resolved, before, after)
        except UniversalActionError as exc:
            raise GenericActionAdapterError(f"必要动作后硬校验失败：{exc}", evidence=all_paths,
                verification_errors=(str(exc),)) from exc
        return (after, tuple(frames), paths, all_paths, (), controller_evidence, model_decision)

    def _prepare_keyboard_geometry(self, resolved: ResolvedSemanticAction, scene: UIScene, frames: tuple[Image.Image,
        ...] | list[Image.Image]) -> dict[str, Any] | None:
        if resolved.kind not in {'input_verified_text', 'clear_verified_text'}:
            return None
        reject_if(resolved.kind == 'input_verified_text' and (not resolved.text), GenericActionAdapterError("输入动作缺少已校验文字。"))
        try:
            input_element = scene.get_element(str(resolved.target_element_id or ''))
        except UISceneError as exc:
            raise GenericActionAdapterError(f"当前文字输入缺少可信输入框：{exc}") from exc
        if resolved.text_transport == 'adb_keyboard':
            typed_field_id = str(input_element.states.get('input_field_id') or '').strip()
            reject_if(typed_field_id in {'', 'unknown'} or typed_field_id != resolved.input_field_id,
                GenericActionAdapterError("ADB Keyboard 动作缺少当前 typed input_field_id。"))
            return None
        geometry = input_element.states.get("keyboard_geometry")
        allowed_types = {"input_verified_text": {"qwerty"}, "clear_verified_text": {"qwerty", "generic"}}
        reject_if(
            not isinstance(geometry, dict) or geometry.get('source') != 'input_structure_audit'
            or geometry.get('type') not in allowed_types[resolved.kind],
            GenericActionAdapterError("当前文字动作缺少本轮输入结构审计签发的键盘几何；拒绝使用静态配置。"),
        )
        prepared = dict(geometry)
        if geometry.get('type') == 'qwerty' and self.require_local_qwerty_row_snap:
            reject_if(not callable(self.qwerty_row_snapper), GenericActionAdapterError("真机文字输入缺少本地 QWERTY 行中心复核器。"))
            snapped = self.qwerty_row_snapper(frames, geometry.get("anchors"))
            reject_if(not isinstance(snapped, dict), GenericActionAdapterError("本地 OCR 未能稳定确认 QWERTY 三行中心，拒绝按模型粗坐标输入。"))
            prepared.update(anchors=snapped, row_snap_source="stable_local_ocr")
        if prepared.get('type') == 'qwerty':
            try:
                qwerty_keyboard_config_from_anchors(prepared.get("anchors"))
            except WorkflowNotReady as exc:
                raise GenericActionAdapterError(f"当前 QWERTY 几何未通过动作前本地复核：{exc}") from exc
        else:
            backspace = (prepared.get("anchors") or {}).get("backspace")
            reject_if(
                not isinstance(backspace, list) or len(backspace) != 2 or any((isinstance(part,
                bool) or not isinstance(part, (int, float)) or (not 0 <= float(part) <= 1000) for part in backspace)),
                GenericActionAdapterError("非 QWERTY 清空缺少本轮完整可见退格键中心。"),
            )
        validator = getattr(self.robot, "validate_verified_text", None)
        if resolved.kind == 'input_verified_text' and callable(validator):
            try:
                validator(resolved.input_fragment, dict(input_element.states), target_text=resolved.text,
                    input_method=resolved.input_method, pinyin=resolved.input_pinyin)
            except (UISceneError, ValueError, RuntimeError) as exc:
                raise GenericActionAdapterError(f"当前文字输入不满足设备已验证配置：{exc}") from exc
        reject_if(resolved.kind == 'clear_verified_text' and resolved.delete_count is None, GenericActionAdapterError("清空动作缺少已验证退格次数。"))
        return prepared

    def _arm_physical_execution(self, requested: SemanticAction, resolved: ResolvedSemanticAction, scene: UIScene,
        frames: tuple[Image.Image, ...] | list[Image.Image], paths: tuple[str,
        ...]) -> tuple[OrientationCredential | None, Callable[[], Any] | None]:
        if resolved.text_transport == 'adb_keyboard' and resolved.kind in {'input_verified_text',
            'clear_verified_text'}:
            return None, None
        clear = getattr(self.robot, "clear_physical_execution_authorization", None)
        if resolved.kind not in self.PHYSICAL_KINDS:
            return None, clear if callable(clear) else None
        arm = getattr(self.robot, "arm_physical_execution", None)
        reject_if(not callable(arm) or not callable(clear), GenericActionAdapterError("机械臂控制器未提供共享物理执行门禁，拒绝动作。", evidence=paths))
        clear()
        try:
            credential = self._local_qwerty_orientation_credential(requested=requested, scene=scene, frames=frames)
            diagnostic_flag = "local_qwerty_rows_verified"
            if credential is None:
                credential = self._single_step_scene_orientation_credential(scene=scene, frames=frames)
                diagnostic_flag = "local_frame_binding_verified"
            selected = len(frames) - 1
            try:
                self.observer.last_orientation_audit_diagnostics = {'audit_source': credential.source, 'model_calls': 0,
                    diagnostic_flag: True, 'selected_frame_index': selected, 'scene_fingerprint': scene.fingerprint}
            except Exception:
                pass
            if 0 <= selected < len(paths):
                with Image.open(paths[selected]) as persisted:
                    credential = replace(credential, evidence_frame_fingerprint=frame_fingerprint(persisted.convert(
                        'RGB')))
            hardware_action = {'clear_verified_text': 'input_verified_text',
                'press_enter': 'tap_semantic', 'scroll': 'swipe',
                'swipe_element': 'swipe'}.get(resolved.kind, resolved.kind)
            credential.assert_authorizes(device_id=self.device_id, scene_fingerprint=scene.fingerprint,
                frame_size=credential.frame_size, action=hardware_action)
            arm(credential, action=hardware_action, scene_fingerprint=scene.fingerprint)
            return credential, clear
        except (OrientationSafetyError, RuntimeError, ValueError) as exc:
            clear()
            raise GenericActionAdapterError(f'动作前单步画面方向凭据校验失败：{exc}', evidence=paths) from exc

    def execute(self, *, requested_action: SemanticAction, planned_scene: UIScene, goal: GenericIntentDraft,
        confirmed: bool, evidence_dir: Path | None=None, planned_frames: tuple[Image.Image,
        ...] | list[Image.Image]=(), action_authority: ConfirmationAuthority | None=None,
        available_action_kinds: frozenset[str] | None=None,
        post_action_available_action_kinds: frozenset[str] | None=None
        ) -> GenericActionExecutionResult:
        reject_if(confirmed is not True, GenericActionAdapterError("必须明确确认当前这一个语义动作。"))
        safe_node = re.sub(r"[^a-zA-Z0-9_-]+", "_", requested_action.node_id)[:48]
        evidence_prefix = f"{safe_node or 'action'}_{uuid.uuid4().hex}"
        local_frame_identity_verified = False
        confirmation_frame_delta: float | None = None
        reject_if(not planned_frames, GenericActionAdapterError(
            "执行动作必须携带产生该 Qwen 动作的当前截图帧。"))
        before_frames, before_paths = self._capture_confirmation_frames(evidence_dir=evidence_dir, prefix=f'{
            evidence_prefix}_before')
        frame_delta = self._confirmation_frame_delta(planned_frames, before_frames)
        confirmation_frame_delta = frame_delta
        if frame_delta > self.confirmation_frame_delta_max:
            raise GenericActionAdapterError(
                "确认时本地真实画面已变化："
                f"差异{frame_delta:.2f}超过阈值{self.confirmation_frame_delta_max:.2f}",
                evidence=before_paths,
            )
        local_frame_identity_verified = True
        before = planned_scene
        # Qwen's action is immutable after selection.  Local code may validate
        # device/scope/geometry, but it may not rewrite its semantic fields.
        rebound = requested_action
        local_point_grounding: LocalPointGrounding | None = None
        if callable(self.point_grounder):
            try:
                local_point_grounding = self.point_grounder(before_frames, before, rebound)
            except LocalPointGroundingAmbiguousError as exc:
                raise GenericActionAdapterError(f'确认前本地视觉落点不唯一：{exc}', evidence=before_paths) from exc
            except Exception:
                # No positive local geometry evidence keeps Qwen's canonical point.
                local_point_grounding = None
        try:
            resolved = self.controller.resolve_one(rebound, before, confirmed=True,
                local_point_grounding=local_point_grounding)
        except UniversalActionError as exc:
            raise GenericActionAdapterError(f'确认前控制器拒绝动作：{exc}', evidence=before_paths) from exc

        reject_if(
            resolved.kind not in self.DEVICE_ACTION_KINDS and resolved.kind != 'wait_for_change',
            GenericActionAdapterError(f'当前通用硬件适配器尚未开放：{resolved.kind}', evidence=before_paths),
        )

        prepared_keyboard_geometry = self._prepare_keyboard_geometry(resolved, before, before_frames)
        orientation_credential, clear_authorization = self._arm_physical_execution(requested_action, resolved, before,
            before_frames, before_paths)

        physical_actions = 0
        robot_result: Any = None
        hardware_receipt: dict[str, Any] | None = None
        execution_metadata: dict[str, Any] = {}

        def executor_point(point: NormalizedPoint | None) -> tuple[int, int] | None:
            if point is None:
                return None
            return (max(0, min(1000, round(point[0] * 1000))), max(0, min(1000, round(point[1] * 1000))))

        text_scope = None
        if resolved.text_transport == 'adb_keyboard':
            transport = self.text_transport
            reject_if(transport is None or action_authority is None,
                GenericActionAdapterError("ADB Keyboard 动作缺少 transport 或已消费的一次性 authority。",
                evidence=before_paths))
            assert transport is not None and action_authority is not None
            reject_if(action_authority.device_id != self.device_id
                or action_authority.fingerprint != before.fingerprint
                or action_authority.decision_node_id != requested_action.node_id
                or action_authority.action_digest != canonical_digest(requested_action.to_dict()),
                GenericActionAdapterError("ADB Keyboard 动作 authority 与当前设备、画面或 canonical 动作不一致。",
                evidence=before_paths))
            prior = resolved.prior_input_value
            fragment = resolved.input_fragment if resolved.kind == 'input_verified_text' else ''
            expected = resolved.expected_input_value
            reject_if(not isinstance(prior, str) or not isinstance(fragment, str)
                or not isinstance(expected, str) or not resolved.input_field_id,
                GenericActionAdapterError("ADB Keyboard 动作缺少精确 typed 文字事务。", evidence=before_paths))
            try:
                text_scope = transport.mint_action_scope(session_id=action_authority.session_id,
                    task_id=action_authority.task_id, revision=action_authority.revision,
                    action_id=action_authority.decision_node_id, input_field_id=resolved.input_field_id,
                    observation_fingerprint=before.fingerprint, prior_text_digest=text_digest(prior),
                    fragment_text_digest=text_digest(fragment), expected_text_digest=text_digest(expected))
            except (RuntimeError, ValueError) as exc:
                raise GenericActionAdapterError(f"ADB Keyboard 单动作 scope 签发失败：{exc}",
                    evidence=before_paths) from exc

        execution_request = DeviceActionRequest(kind=resolved.kind, point=executor_point(resolved.normalized_point),
            end_point=executor_point(resolved.normalized_end_point), direction=resolved.direction,
            hold_seconds=resolved.hold_seconds, input_fragment=resolved.input_fragment,
            input_method=resolved.input_method, input_pinyin=resolved.input_pinyin,
            text_transport=resolved.text_transport, text_scope=text_scope,
            keyboard_geometry=prepared_keyboard_geometry, delete_count=resolved.delete_count, wait_seconds=max(0.5,
            self.post_action_settle) if resolved.kind == 'wait_for_change' else None, launch_ref=resolved.launch_ref)
        try:
            execution_result = self.device_executor.execute(execution_request)
            physical_actions = execution_result.physical_actions
            robot_result = execution_result.transport_result
            hardware_receipt = execution_result.hardware_receipt
            execution_metadata = dict(execution_result.metadata)
        except DeviceExecutionError as exc:
            raise GenericActionAdapterError(f'设备执行器拒绝动作：{exc}', physical_actions=exc.physical_actions,
                evidence=before_paths, execution_metadata=getattr(exc, 'metadata', {})) from exc
        except OrientationSafetyError as exc:
            gate_evidence = before_paths
            if isinstance(exc, OrientationFrameMismatchError):
                gate_evidence += self._save_frames([exc.actual_frame], evidence_dir, f'{
                    evidence_prefix}_physical_gate_actual')
            raise GenericActionAdapterError(f'共享物理执行门在控制端原语前拒绝动作：{exc}', physical_actions=0,
                evidence=gate_evidence) from exc
        except Exception as exc:
            raise GenericActionAdapterError(f'设备单步动作调用失败：{exc}', physical_actions=physical_actions,
                evidence=before_paths) from exc
        finally:
            if callable(clear_authorization):
                clear_authorization()

        try:
            (
                after,
                after_frames,
                after_frame_paths,
                all_after_paths,
                observation_errors,
                controller_transition_evidence,
                after_model_decision,
            ) = self._observe_stable_post_action_scene(
                self._post_action_goal(goal, authority=action_authority, resolved=resolved,
                    physical_actions=physical_actions),
                before=before,
                before_frames=tuple(before_frames),
                resolved=resolved,
                evidence_dir=evidence_dir,
                evidence_prefix=evidence_prefix,
                available_action_kinds=(post_action_available_action_kinds
                    if post_action_available_action_kinds is not None else available_action_kinds),
            )
        except (GenericActionAdapterError, UniversalActionError) as exc:
            evidence = before_paths + tuple(getattr(exc, "evidence", ()))
            raise GenericActionAdapterError(f'单步动作后验证失败：{exc}', physical_actions=physical_actions, evidence=evidence,
                observation_errors=tuple(getattr(exc, 'observation_errors', ())), verification_errors=tuple(getattr(exc,
                'verification_errors', ())), execution_metadata=execution_metadata) from exc
        if (action_authority is not None and action_authority.effect_ids
            and str(after_model_decision.get('status') or '').strip() == 'finish'):
            transition = measure_material_visual_transition(before_frames, after_frames)
            execution_metadata = {**execution_metadata, 'effect_visual_transition': transition}
            reject_if(not transition['material'], GenericActionAdapterError(
                '外部效果动作后的真实帧没有可归因的新变化；动作前已有画面不能作为本次 finish 证据，'
                '本次已执行1次且不会自动重复。',
                physical_actions=physical_actions,
                evidence=before_paths + all_after_paths,
                verification_errors=('外部效果缺少动作前后真实帧变化',),
                execution_metadata=execution_metadata,
            ))
        return GenericActionExecutionResult(requested_action=requested_action, rebound_action=rebound,
            resolved_action=resolved, before_scene=before, after_scene=after,
            planned_scene_fingerprint=planned_scene.fingerprint,
            confirmation_frame_identity_verified=local_frame_identity_verified,
            confirmation_frame_delta=confirmation_frame_delta, physical_actions=physical_actions,
            primary_input_confirmation_reused=False,
            action_outcome='matched', verification_errors=(),
            robot_result=robot_result, hardware_receipt=hardware_receipt, execution_metadata=execution_metadata,
            evidence=before_paths + all_after_paths,
            after_frames=after_frames, after_frame_paths=after_frame_paths, observation_errors=observation_errors,
            controller_transition_evidence=controller_transition_evidence,
            after_model_decision=after_model_decision, before_frames=before_frames,
            before_frame_paths=before_paths, orientation_credential=orientation_credential)


    @staticmethod
    def _save_frames(frames: list[Image.Image], evidence_dir: Path | None, prefix: str) -> tuple[str, ...]:
        if evidence_dir is None:
            return ()
        evidence_dir.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []
        for (index, frame) in enumerate(frames, start=1):
            path = evidence_dir / f"{prefix}_{index}.jpg"
            frame.save(path, format="JPEG", quality=92)
            paths.append(str(path))
        return tuple(paths)
