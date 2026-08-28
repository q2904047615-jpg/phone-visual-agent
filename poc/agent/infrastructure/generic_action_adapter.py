"""Infrastructure adapter for one observed, verified device action."""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import time
import re
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageChops, ImageStat

from agent.domain.canonical_action_protocol import CanonicalActionProtocolError
from agent.domain import DeviceActionRequest, DeviceExecutionError, DeviceExecutor
from agent.infrastructure import RobotDeviceExecutor
from agent.domain.generic_goal import GenericIntentDraft
from agent.infrastructure.generic_scene_observer import SingleStepGenericSceneObserver
from agent.application.action_adapter import GenericActionAdapterError
from agent.domain.post_action_observation import (
    POST_ACTION_VISUAL_CONTEXT_VERSION,
    POST_NAVIGATION_RESULT_COMPLETION_CONDITIONS,
    POST_NAVIGATION_RESULT_OBJECTIVE,
    POST_NAVIGATION_RESULT_OBSERVATION_PHASE,
    PostActionVisualContext,
)
from agent.application.input_value_lineage import TypedInputLineageStorePort
from agent.domain.action_capabilities import build_device_capability_snapshot
from agent.domain.input_value_lineage import (
    InputValueLineageError,
    TypedInputLineage,
    build_pending_chinese_preedit_lineage,
    build_pending_ime_candidate_lineage,
    build_pending_input_state_lineage,
    build_pending_literal_lineage,
    build_pending_newline_lineage,
    build_pending_text_lineage,
    input_app_identity_compatible,
    input_screen_identity_compatible,
)
from agent.domain.trusted_observation import bounds_overlap
from agent.infrastructure.windows_ocr_runtime import find_text, recognize as recognize_ocr
from agent.infrastructure.observation_images import (
    measure_frame_sharpness,
    measure_local_stability,
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
from agent.domain.vision_model import VisionAgentError
from agent.domain.semantic_action import SemanticAction
from agent.domain.ui_scene import UIElement, UIScene, UISceneError
from agent.domain.universal_action_controller import (
    LOCAL_POINT_GROUNDING_SOURCE,
    LocalPointGrounding,
    ResolvedSemanticAction,
    UniversalActionController,
    UniversalActionError,
)
from agent.infrastructure.robot_controller import WorkflowNotReady, qwerty_keyboard_config_from_anchors


QWEN_FAILURE_DIAGNOSTIC_VERSION = "2026-08-17-qwen-failure-diagnostic-v1"
MAX_REDACTED_QWEN_RESPONSE_CHARS = 16000
_IMAGE_DATA_URL_RE = re.compile('data:image/[^;\\s\\"\']+;base64,[A-Za-z0-9+/=_-]+', re.IGNORECASE)
_SECRET_FIELD_RE = re.compile(
    r"(?P<prefix>[\"']?(?:authorization|api[_-]?key|access[_-]?token|"
    r"refresh[_-]?token|token|secret|password)[\"']?\s*[:=]\s*)"
    r"(?P<quote>[\"'])(?P<value>.*?)(?P=quote)",
    re.IGNORECASE,
)
_UNQUOTED_SECRET_FIELD_RE = re.compile(
    r"(?P<prefix>[\"']?(?:authorization|api[_-]?key|access[_-]?token|"
    r"refresh[_-]?token|token|secret|password)[\"']?\s*[:=]\s*)"
    r"(?![\"'])(?P<value>[^,}\]\s]+)",
    re.IGNORECASE,
)
_BEARER_RE = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]+", re.IGNORECASE)
_URL_SECRET_RE = re.compile(
    r"(?P<prefix>[?&](?:api[_-]?key|access[_-]?token|token|secret|password)=)"
    r"[^&#\s\"']+",
    re.IGNORECASE,
)
_URL_USERINFO_RE = re.compile(r"(https?://)[^/@\s:]+:[^/@\s]+@", re.IGNORECASE)
_OPENAI_STYLE_SECRET_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}")

_POST_NAVIGATION_RESULT_KINDS = frozenset({'tap_semantic', 'double_tap', 'swipe', 'back', 'home', 'open_recent_apps'})
_POST_NAVIGATION_ALLOWED_EFFECT_KEYS = frozenset(
    {
        "scene_changed",
        "content_changed",
        "current_video_changed",
        "description",
        "app_id",
        "screen_id",
    }
)
_VISUAL_FOCUS_KEYS = frozenset(
    {
        "subgoal_id",
        "objective",
        "constraints",
        "completion_conditions",
        "execution_class",
        "goal_entities",
    }
)


def _first_valid_lineage(
    builders: tuple[Callable[..., TypedInputLineage], ...],
    **values: Any,
) -> TypedInputLineage | None:
    for builder in builders:
        try:
            return builder(**values)
        except (InputValueLineageError, TypeError, ValueError):
            continue
    return None


def _sanitized_visual_focus(value: Any) -> dict[str, Any] | None:
    """Accept only the bridge-owned shallow visual-focus shape."""

    if not isinstance(value, dict) or set(value) != _VISUAL_FOCUS_KEYS:
        return None
    if (
        not str(value.get("subgoal_id") or "").strip()
        or not str(value.get("objective") or "").strip()
        or not isinstance(value.get("constraints"), list)
        or not isinstance(value.get("completion_conditions"), list)
        or not isinstance(value.get("goal_entities"), dict)
    ):
        return None
    focus = dict(value)
    goal_entities = dict(focus["goal_entities"])
    # Observation phases are minted only after a physical action by this
    # module.  A model, user payload or stale goal draft cannot pre-authorize
    # an action-result observation mode.
    goal_entities.pop("observation_phase", None)
    focus["goal_entities"] = goal_entities
    return focus


def _post_action_observation_context(
    goal: GenericIntentDraft,
    resolved: ResolvedSemanticAction,
    *,
    physical_action_executed: bool = False,
) -> dict[str, Any]:
    """Build an action-result context for the actual active subgoal only."""

    context = goal.to_dict()
    entities = context.get("entities")
    if not isinstance(entities, dict):
        return context
    focus = _sanitized_visual_focus(entities.get('active_subgoal_visual_context'))
    sanitized_entities = {
        key: value
        for key, value in entities.items()
        if not (
            isinstance(key, str)
            and key.endswith("_subgoal_visual_context")
            and key != "active_subgoal_visual_context"
        )
    }
    if focus is not None:
        sanitized_entities["active_subgoal_visual_context"] = focus
    else:
        sanitized_entities.pop("active_subgoal_visual_context", None)
    context = dict(context)
    context["entities"] = sanitized_entities
    entities = sanitized_entities

    expected = resolved.expected_effect
    changed_result = (
        isinstance(expected, dict)
        and (
            expected.get("scene_changed") is True
            or expected.get("content_changed") is True
        )
    )
    if (
        not physical_action_executed
        or focus is None
        or str(focus.get("execution_class") or "").strip() != "navigate"
        or resolved.kind not in _POST_NAVIGATION_RESULT_KINDS
        or not isinstance(expected, dict)
        or not changed_result
        or "element_state" in expected
        or set(expected) - _POST_NAVIGATION_ALLOWED_EFFECT_KEYS
    ):
        return context

    goal_entities = focus.get("goal_entities")
    if not isinstance(goal_entities, dict):
        return context
    result_entities = dict(goal_entities)
    result_entities.pop("target_ui_label", None)
    result_entities['observation_phase'] = POST_NAVIGATION_RESULT_OBSERVATION_PHASE
    result_focus = dict(focus)
    result_focus["objective"] = POST_NAVIGATION_RESULT_OBJECTIVE
    result_focus['completion_conditions'] = list(POST_NAVIGATION_RESULT_COMPLETION_CONDITIONS)
    result_focus["goal_entities"] = result_entities
    result_context = dict(context)
    result_context["entities"] = dict(entities)
    result_context["entities"]["active_subgoal_visual_context"] = result_focus
    return result_context


def _post_action_visual_context( resolved: ResolvedSemanticAction, ) -> PostActionVisualContext | None:
    """Project one locally validated canonical transition into Qwen context.

    The adapter calls this only after the device executor reports one physical
    action.  The summary intentionally carries neither coordinates nor a
    success verdict; the new pixels and Controller still own verification.
    Legacy/non-formal test actions have no typed transition and therefore do
    not mint cross-step context.
    """

    raw_expectations = resolved.formal_transition.get("expectations")
    if not isinstance(raw_expectations, list) or not raw_expectations:
        return None
    if not all(isinstance(item, dict) for item in raw_expectations):
        raise GenericActionAdapterError("canonical动作的typed后置条件结构无效。")
    payload = {
        "protocol_version": POST_ACTION_VISUAL_CONTEXT_VERSION,
        "execution_state": "physical_action_executed",
        "outcome": "pending_visual_verification",
        "canonical_action_kind": resolved.kind,
        "expected_postconditions": [dict(item) for item in raw_expectations],
    }
    try:
        return PostActionVisualContext.from_dict(payload)
    except (CanonicalActionProtocolError, VisionAgentError, TypeError, ValueError) as exc:
        raise GenericActionAdapterError(f'canonical动作不能建立typed动作后视觉上下文：{exc}') from exc


def stable_qwerty_ocr_anchors(
    frames: tuple[Image.Image, ...] | list[Image.Image],
    anchors: dict[str, Any],
    *,
    ocr_recognizer: Any = recognize_ocr,
) -> dict[str, list[int]] | None:
    """Refine a QWERTY contract with stable local OCR row geometry."""

    frame_list = list(frames)[-3:]
    required = {"q", "p", "a", "l", "z", "m", "backspace"}
    if len(frame_list) != 3 or not isinstance(anchors, dict):
        return None
    try:
        original = {
            key: [round(float(value[0])), round(float(value[1]))]
            for key, value in anchors.items()
            if isinstance(value, (list, tuple)) and len(value) == 2
        }
        if set(original) != required or any(not 0 <= point[0] <= 1000 for point in original.values()):
            return None
        row_keys = (("q", "p"), ("a", "l"), ("z", "m", "backspace"))
        model_rows = tuple(statistics.mean(original[key][1] for key in keys) for keys in row_keys)
        trusted_model_y = bool(
            all(0 <= point[1] <= 1000 for point in original.values())
            and model_rows[0] < model_rows[1] < model_rows[2]
            and min(model_rows[1] - model_rows[0], model_rows[2] - model_rows[1]) >= 35
        )
        horizontal_probe = {key: list(value) for key, value in original.items()}
        for y, keys in zip((650, 750, 850), row_keys):
            for key in keys:
                horizontal_probe[key][1] = y
        qwerty_keyboard_config_from_anchors(horizontal_probe)

        alphabets = ("qwertyuiop", "asdfghjkl", "zxcvbnm")

        def clusters(hits: list[tuple[str, float | None, float]], height: int) -> list[tuple[float, int, list]]:
            groups: list[list[tuple[str, float | None, float]]] = []
            tolerance = max(8.0, height * 0.025)
            for hit in sorted(hits, key=lambda item: item[2]):
                if not groups or abs(hit[2] - statistics.median(item[2] for item in groups[-1])) > tolerance:
                    groups.append([hit])
                else:
                    groups[-1].append(hit)
            return [
                (float(statistics.median(item[2] for item in group)), len({item[0] for item in group}), group)
                for group in groups
                if len({item[0] for item in group}) >= 2
            ]

        def horizontal_fit(group: list[tuple[str, float | None, float]], alphabet: str, width: int):
            centers: dict[str, list[float]] = {}
            for label, center_x, _ in group:
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
            return (first_x, pitch) if (
                45 <= pitch <= 130
                and 0 <= first_x
                and first_x + pitch * (len(alphabet) - 1) <= 1000
                and residual <= 25
            ) else None

        def combined_fit(top_group: list, bottom_group: list, width: int):
            top_fit = horizontal_fit(top_group, alphabets[0], width)
            bottom_fit = horizontal_fit(bottom_group, alphabets[2], width)
            count = int(top_fit is not None) + int(bottom_fit is not None)
            if any(item[1] is not None for item in (*top_group, *bottom_group)) and not count:
                return None
            if top_fit and bottom_fit:
                top_first, top_pitch = top_fit
                bottom_first, bottom_pitch = bottom_fit
                if abs(top_pitch - bottom_pitch) > 18 or abs(bottom_first - (top_first + 1.5 * top_pitch)) > 60:
                    return None
                return count, statistics.mean((top_first, bottom_first - 1.5 * bottom_pitch)), statistics.mean(
                    (top_pitch, bottom_pitch)
                )
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
                if len(label) != 1 or not label.isascii() or not label.isalpha():
                    continue
                left, width = word.get("left"), word.get("width")
                has_x = all(isinstance(part, (int, float)) and not isinstance(part, bool) for part in (left, width))
                hit = (
                    label,
                    float(left) + float(width) / 2.0 if has_x else None,
                    float(word.get("top", 0)) + float(word.get("height", 0)) / 2.0,
                )
                for alphabet in alphabets:
                    if label in alphabet:
                        hits[alphabet].append(hit)
                        break
            row_clusters = [clusters(hits[alphabet], frame.height) for alphabet in alphabets]
            candidates = []
            expected_top = frame.height * model_rows[0] / 1000.0
            expected_bottom = frame.height * model_rows[2] / 1000.0
            for top_y, top_count, top_group in row_clusters[0]:
                for bottom_y, bottom_count, bottom_group in row_clusters[2]:
                    if not frame.height * 0.08 <= bottom_y - top_y <= frame.height * 0.22:
                        continue
                    midpoint = (top_y + bottom_y) / 2.0
                    middle_count = max(
                        (count for center, count, _ in row_clusters[1] if abs(center - midpoint) <= frame.height * 0.04),
                        default=0,
                    )
                    fitted = combined_fit(top_group, bottom_group, frame.width)
                    if fitted is None:
                        continue
                    fit_count, first_x, pitch = fitted
                    distance = abs(top_y - expected_top) + abs(bottom_y - expected_bottom) if trusted_model_y else 0.0
                    candidates.append(
                        (
                            fit_count,
                            min(top_count, bottom_count),
                            top_count + middle_count + bottom_count,
                            -distance,
                            top_y,
                            bottom_y,
                            first_x if first_x is not None else math.nan,
                            pitch if pitch is not None else math.nan,
                        )
                    )
            if candidates:
                *_, top_y, bottom_y, first_x, pitch = max(candidates)
                per_frame_rows.append(
                    (top_y, bottom_y, first_x if math.isfinite(first_x) else None, pitch if math.isfinite(pitch) else None)
                )

        if len(per_frame_rows) < 2 or any(
            max(row[index] for row in per_frame_rows) - min(row[index] for row in per_frame_rows) > 10
            for index in (0, 1)
        ):
            return None
        height = frame_list[-1].height
        top_y, bottom_y = (
            round(1000 * statistics.median(row[index] for row in per_frame_rows) / height) for index in (0, 1)
        )
        middle_y, span = round((top_y + bottom_y) / 2.0), bottom_y - top_y
        if span <= 0 or (
            trusted_model_y
            and any(
                abs(actual - original[key][1]) > span * 1.5
                for key, actual in (("q", top_y), ("a", middle_y), ("z", bottom_y))
            )
        ):
            return None

        snapped = {key: list(value) for key, value in original.items()}
        horizontal = [
            (float(first), float(pitch)) for _, _, first, pitch in per_frame_rows if first is not None and pitch is not None
        ]
        if len(horizontal) >= 2:
            first_values, pitch_values = zip(*horizontal)
            if max(first_values) - min(first_values) > 15 or max(pitch_values) - min(pitch_values) > 6:
                return None
            q_x, pitch = round(statistics.median(first_values)), statistics.median(pitch_values)
            offsets = {"q": 0, "p": 9, "a": .5, "l": 8.5, "z": 1.5, "m": 7.5, "backspace": 9}
            local_x = {key: round(q_x + offset * pitch) for key, offset in offsets.items()}
            if any(not 0 <= value <= 1000 for value in local_x.values()):
                return None
            for key, x in local_x.items():
                snapped[key][0] = x
        for y, keys in zip((top_y, middle_y, bottom_y), row_keys):
            for key in keys:
                snapped[key][1] = y
        qwerty_keyboard_config_from_anchors(snapped)
        return snapped
    except Exception:
        return None


def stable_text_ocr_grounding(
    frames: tuple[Image.Image, ...] | list[Image.Image],
    scene: UIScene,
    action: SemanticAction,
    *,
    ocr_recognizer: Any = recognize_ocr,
) -> LocalPointGrounding | None:
    """Refine one coarse point with a stable exact-label OCR match.

    The canonical action and its target are already fixed before this helper
    runs.  OCR may only refine that target's point; it cannot add candidates,
    change the action, or make an ambiguous label executable.  Any unavailable,
    duplicate, unstable, or distant OCR result simply keeps the existing model
    center.
    """

    if action.action not in { "tap_semantic", "dismiss_overlay", "double_tap", "long_press", }:
        return None
    frame_list = list(frames)[-3:]
    if not frame_list or not scene.fingerprint:
        return None
    element_id = str(action.params.get("element_id") or "").strip()
    try:
        element = scene.get_element(element_id)
    except UISceneError:
        return None
    if (
        element.role
        not in {"button", "icon", "text", "tab", "toggle", "image", "list_item"}
        or element.element_id.startswith("local_audited_")
        or element.meaning == "application_text_input"
        or element.meaning.startswith(("input_", "ime_", "switch_keyboard_"))
        or str(action.params.get("label") or "") != element.label
        or str(action.params.get("target") or "") != element.meaning
        or str(action.params.get("role") or "") != element.role
    ):
        return None
    label = element.label.strip()
    compact_label = re.sub(r"[\s\u3000]+", "", label)
    if len(compact_label) < 2 or len(compact_label) > 64:
        return None

    def unique_match_geometry(
        frame: Image.Image,
    ) -> tuple[
        tuple[float, float, float, float],
        tuple[float, float],
    ] | None:
        payload = ocr_recognizer(frame.convert('RGB'), 'zh-Hans-CN', scale=1.5)
        matches = find_text(payload, label)
        if len(matches) != 1 or frame.width <= 0 or frame.height <= 0:
            return None
        match = matches[0]
        bounds = (
            float(match.left) / frame.width,
            float(match.top) / frame.height,
            float(match.left + match.width) / frame.width,
            float(match.top + match.height) / frame.height,
        )
        left, top, right, bottom = bounds
        if not (0.0 <= left < right <= 1.0 and 0.0 <= top < bottom <= 1.0):
            return None
        return bounds, ((left + right) / 2.0, (top + bottom) / 2.0)

    try:
        inspected_frames = 1
        newest = unique_match_geometry(frame_list[-1])
        if newest is None:
            return None
        proposed_point = element.center
        # If Qwen's center already agrees with the exact visible label, keep
        # the original point and avoid spending a second local OCR call.
        if math.dist(proposed_point, newest[1]) <= 0.025:
            return None

        stable_matches = [newest]
        for frame in reversed(frame_list[:-1]):
            inspected_frames += 1
            candidate = unique_match_geometry(frame)
            if candidate is None:
                continue
            if math.dist(candidate[1], newest[1]) > 0.015:
                continue
            if any(
                abs(
                    (candidate[0][index + 2] - candidate[0][index])
                    - (newest[0][index + 2] - newest[0][index])
                )
                > 0.03
                for index in (0, 1)
            ):
                continue
            stable_matches.append(candidate)
            if len(stable_matches) >= 2:
                break
        if len(stable_matches) < 2:
            return None

        grounded_bounds = tuple(
            float(statistics.median(match[0][index] for match in stable_matches))
            for index in range(4)
        )
        grounded_point = (
            (grounded_bounds[0] + grounded_bounds[2]) / 2.0,
            (grounded_bounds[1] + grounded_bounds[3]) / 2.0,
        )
        if ( math.dist(proposed_point, grounded_point) <= 0.025 ):
            return None
        grounding = LocalPointGrounding(
            source=LOCAL_POINT_GROUNDING_SOURCE,
            scene_fingerprint=scene.fingerprint,
            element_id=element.element_id,
            label=element.label,
            model_bounds=tuple(float(value) for value in element.bounds),
            proposed_point=tuple(float(value) for value in proposed_point),
            grounded_bounds=grounded_bounds,
            grounded_point=grounded_point,
            matched_frames=len(stable_matches),
            inspected_frames=inspected_frames,
        )
        grounding.validate_for(scene, element)
        return grounding
    except Exception:
        return None


def _redact_qwen_failure_response(raw: str) -> str:
    redacted = _IMAGE_DATA_URL_RE.sub("[REDACTED_IMAGE_DATA_URL]", str(raw or ""))
    redacted = _SECRET_FIELD_RE.sub(
        lambda match: (
            f"{match.group('prefix')}{match.group('quote')}"
            f"[REDACTED_SECRET]{match.group('quote')}"
        ),
        redacted,
    )
    redacted = _UNQUOTED_SECRET_FIELD_RE.sub(lambda match: f'{match.group('prefix')}[REDACTED_SECRET]', redacted)
    redacted = _BEARER_RE.sub("Bearer [REDACTED_SECRET]", redacted)
    redacted = _URL_SECRET_RE.sub(lambda match: f'{match.group('prefix')}[REDACTED_SECRET]', redacted)
    redacted = _URL_USERINFO_RE.sub(r"\1[REDACTED_CREDENTIALS]@", redacted)
    return _OPENAI_STYLE_SECRET_RE.sub("[REDACTED_SECRET]", redacted)


def _persist_qwen_failure_diagnostic(
    *,
    evidence_dir: Path | None,
    prefix: str,
    raw_response: str,
    error: Exception,
    diagnostics: dict[str, Any] | None = None,
) -> Path | None:
    """Persist bounded redacted model output without changing failure policy."""

    raw = str(raw_response or "")
    if evidence_dir is None or not raw:
        return None
    output_dir = Path(evidence_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_prefix = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(prefix or ""))[:96]
    target = output_dir / f"{safe_prefix or 'observation'}_qwen_failure.json"
    redacted = _redact_qwen_failure_response(raw)
    bounded = redacted[:MAX_REDACTED_QWEN_RESPONSE_CHARS]
    public_error = _redact_qwen_failure_response(str(error))[:1000]
    details = diagnostics if isinstance(diagnostics, dict) else {}
    payload = {
        "artifact_version": QWEN_FAILURE_DIAGNOSTIC_VERSION,
        "failed_stage": str(details.get("failed_stage") or "unknown")[:120],
        "error_type": str(
            details.get("error_type")
            or classify_qwen_error(error, raw_response=raw)
        )[:120],
        "error_message": public_error,
        "raw_response_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "raw_response_length": len(raw),
        "redacted_response_truncated": len(redacted) > len(bounded),
        "redacted_raw_response": bounded,
    }
    encoded = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    temporary = output_dir / f".{target.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_bytes(encoded)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def persist_observer_failure_diagnostic(
    observer: Any,
    *,
    evidence_dir: Path | None,
    prefix: str,
    error: Exception,
) -> tuple[str, ...]:
    diagnostics = getattr(observer, "last_diagnostics", {})
    raw_response = getattr(observer, "last_raw_response", "")
    try:
        path = _persist_qwen_failure_diagnostic(
            evidence_dir=evidence_dir,
            prefix=prefix,
            raw_response=raw_response,
            error=error,
            diagnostics=diagnostics if isinstance(diagnostics, dict) else None,
        )
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
    primary_input_confirmation_reused: bool = False
    action_outcome: str = "matched"
    verification_errors: tuple[str, ...] = ()
    robot_result: Any = None
    hardware_receipt: dict[str, Any] | None = None
    evidence: tuple[str, ...] = ()
    after_frames: tuple[Image.Image, ...] = field(default_factory=tuple, repr=False, compare=False)
    after_frame_paths: tuple[str, ...] = ()
    observation_errors: tuple[str, ...] = ()
    controller_transition_evidence: tuple[str, ...] = ()
    before_frames: tuple[Image.Image, ...] = field(default_factory=tuple, repr=False, compare=False)
    before_frame_paths: tuple[str, ...] = ()
    orientation_credential: OrientationCredential | None = None

    def __post_init__(self) -> None:
        # Capture helpers intentionally build mutable lists while sampling.  The
        # completed execution result is a live promotion source, so freeze both
        # frame collections at the result boundary instead of exposing a
        # list-shaped object that the promotion validator must reject.
        object.__setattr__(self, "after_frames", tuple(self.after_frames))
        object.__setattr__(self, "before_frames", tuple(self.before_frames))
        object.__setattr__(self, 'controller_transition_evidence', tuple(self.controller_transition_evidence))

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_action": self.requested_action.to_dict(),
            "rebound_action": self.rebound_action.to_dict(),
            "resolved_action": self.resolved_action.to_dict(),
            "before_scene": self.before_scene.to_dict(),
            "after_scene": self.after_scene.to_dict(),
            "planned_scene_fingerprint": self.planned_scene_fingerprint,
            "confirmation_frame_identity_verified": (
                self.confirmation_frame_identity_verified
            ),
            "confirmation_frame_delta": self.confirmation_frame_delta,
            "physical_actions": self.physical_actions,
            "primary_input_confirmation_reused": (
                self.primary_input_confirmation_reused
            ),
            "action_outcome": self.action_outcome,
            "verification_errors": list(self.verification_errors),
            "robot_result": self.robot_result,
            "hardware_receipt": (
                dict(self.hardware_receipt)
                if self.hardware_receipt is not None
                else None
            ),
            "evidence": list(self.evidence),
            "after_frame_count": len(self.after_frames),
            "after_frame_paths": list(self.after_frame_paths),
            "observation_errors": list(self.observation_errors),
            "controller_transition_evidence": list(
                self.controller_transition_evidence
            ),
            "before_frame_count": len(self.before_frames),
            "before_frame_paths": list(self.before_frame_paths),
            "orientation_credential": (
                self.orientation_credential.to_dict()
                if self.orientation_credential is not None
                else None
            ),
        }


class GenericSingleActionAdapter:
    """The only generic bridge from a verified scene to one robot action."""

    PHYSICAL_KINDS = frozenset(
        {
            "tap_semantic",
            "dismiss_overlay",
            "double_tap",
            "swipe",
            "reveal_system_navigation",
            "back",
            "home",
            "open_recent_apps",
            "input_verified_text",
            "press_enter",
            "clear_verified_text",
            "long_press",
            "drag",
        }
    )
    GEOMETRY_BOUND_KINDS = frozenset(
        {
            "tap_semantic",
            "dismiss_overlay",
            "double_tap",
            "input_verified_text",
            "press_enter",
            "clear_verified_text",
            "long_press",
            "drag",
        }
    )
    INDEPENDENT_GEOMETRY_AUDIT_KINDS = frozenset(
        {
            "tap_semantic",
            "dismiss_overlay",
            "double_tap",
            "input_verified_text",
            "press_enter",
            "clear_verified_text",
            "long_press",
            "drag",
        }
    )
    LOCAL_INPUT_AUXILIARY_MEANINGS = frozenset(
        {
            "ime_exact_candidate",
            "input_exact_literal_key",
            "input_exact_enter_key",
            "switch_keyboard_layout",
            "switch_keyboard_case",
            "switch_keyboard_input_mode",
        }
    )

    @classmethod
    def _primary_input_confirmation_reusable( cls, requested: SemanticAction, scene: UIScene, ) -> bool:
        """Return whether one strict input audit may survive local recapture.

        The caller has already compared the planned and fresh four-frame sets.
        This predicate only accepts the exact canonical goal element minted by
        the dedicated input audit; ordinary compact controls never qualify.
        """

        if requested.action not in {
            "tap_semantic",
            "press_enter",
            "input_verified_text",
            "clear_verified_text",
        } or not str(requested.params.get("formal_candidate_id") or "").strip():
            return False
        element_id = str(requested.params.get("element_id") or "").strip()
        try:
            element = scene.get_element(element_id)
            unique = scene.unique_trusted_goal_element()
        except UISceneError:
            return False
        if unique is None or unique.element_id != element_id:
            return False
        states = element.states
        if (
            not element_id.startswith("local_audited_")
            or states.get("primary_input_geometry_verified") is not True
            or states.get("geometry_audit_source") != "input_structure_audit"
            or states.get("goal_relevant") is not True
            or states.get("fully_visible") is not True
            or not any(str(item).strip() for item in element.evidence)
            or requested.params.get("target") != element.meaning
            or requested.params.get("role") != element.role
            or requested.params.get("label") != element.label
            or requested.params.get("states") != states
        ):
            return False
        if requested.action in {"input_verified_text", "clear_verified_text"}:
            return bool(
                element.role == "input"
                and element.meaning == "application_text_input"
                and str(states.get("input_field_id") or "").strip()
                not in {"", "unknown"}
            )
        if requested.action == "press_enter":
            return bool(
                element.meaning == "input_exact_enter_key"
                and states.get("input_enter_key") is True
                and states.get("key_action") == "newline"
            )
        return bool(
            element.meaning == "application_text_input"
            or element.meaning in cls.LOCAL_INPUT_AUXILIARY_MEANINGS
            or element.meaning == "input_next_field_key"
        )

    def _local_qwerty_orientation_credential(
        self,
        *,
        requested: SemanticAction,
        scene: UIScene,
        frames: list[Image.Image],
    ) -> OrientationCredential | None:
        if requested.action not in {
            "tap_semantic",
            "press_enter",
            "input_verified_text",
            "clear_verified_text",
        } or not callable(self.qwerty_row_snapper):
            return None
        element_id = str(requested.params.get("element_id") or "").strip()
        try:
            target = scene.get_element(element_id)
        except UISceneError:
            return None
        input_id = (
            element_id
            if target.meaning == "application_text_input"
            else str(target.states.get("input_element_id") or "").strip()
        )
        try:
            input_element = scene.get_element(input_id)
        except UISceneError:
            return None
        geometry = input_element.states.get("keyboard_geometry")
        if (
            input_element.meaning != "application_text_input"
            or input_element.states.get("focused") is not True
            or not isinstance(geometry, dict)
            or geometry.get("type") != "qwerty"
            or geometry.get("source") != "input_structure_audit"
        ):
            return None
        snapped = self.qwerty_row_snapper(frames, geometry.get("anchors"))
        if not isinstance(snapped, dict):
            return None
        try:
            qwerty_keyboard_config_from_anchors(snapped)
            return _mint_locally_verified_qwerty_credential(
                device_id=self.device_id,
                scene_fingerprint=scene.fingerprint,
                frame=frames[-1],
                anchors=snapped,
            )
        except (OrientationSafetyError, WorkflowNotReady, TypeError, ValueError):
            return None

    def _single_step_scene_orientation_credential(
        self,
        *,
        scene: UIScene,
        frames: list[Image.Image],
    ) -> OrientationCredential:
        """Mint locally from the direction facts in the sole step response."""

        if not frames:
            raise OrientationSafetyError("单步方向绑定缺少当前稳定帧。")
        alignment = scene.camera_alignment
        return _mint_single_step_scene_credential(
            device_id=self.device_id,
            scene_fingerprint=scene.fingerprint,
            frame=frames[-1].convert("RGB"),
            camera_layout_orientation_value=(
                alignment.camera_layout_orientation
            ),
            phone_content_rotation=alignment.phone_content_rotation,
            confidence=float(alignment.confidence),
            evidence=tuple(alignment.evidence),
        )

    def supported_action_kinds(self) -> frozenset[str]:
        """Return only actions backed by callable methods on this device."""

        capability_provider = getattr(self.robot, "hardware_capabilities", None)
        declared = capability_provider() if callable(capability_provider) else {}
        if not isinstance(declared, dict):
            declared = {}

        def available(action: str, method_name: str) -> bool:
            return callable(getattr(self.robot, method_name, None)) and bool(declared.get(action, True))

        supported = {'wait_for_change'} if bool(declared.get('wait_for_change', True)) else set()
        if available("tap_semantic", "vision_tap_relative"):
            supported.add("tap_semantic")
        if available("dismiss_overlay", "vision_dismiss_overlay_relative"):
            supported.add("dismiss_overlay")
        if available("double_tap", "vision_double_tap_relative"):
            supported.add("double_tap")
        if bool(declared.get("swipe", True)) and any(
            callable(getattr(self.robot, f"vision_swipe_{direction}", None))
            for direction in ("up", "down", "left", "right")
        ):
            supported.add("swipe")
        if available( "reveal_system_navigation", "vision_reveal_system_navigation", ):
            supported.add("reveal_system_navigation")
        if available("back", "vision_android_back"):
            supported.add("back")
        if available("home", "vision_android_home"):
            supported.add("home")
        if available("open_recent_apps", "vision_android_recent_apps"):
            supported.add("open_recent_apps")
        if available("input_verified_text", "vision_type_text_with_layout"):
            supported.add("input_verified_text")
        if available("tap_semantic", "vision_tap_relative"):
            supported.add("press_enter")
        if (
            bool(declared.get("input_verified_text", True))
            and callable(getattr(self.robot, "vision_clear_text", None))
        ):
            supported.add("clear_verified_text")
        if available("long_press", "vision_long_press_relative"):
            supported.add("long_press")
        if available("drag", "vision_drag_relative"):
            supported.add("drag")
        return frozenset(supported)

    def capability_snapshot(self) -> Any:
        provider = getattr(self.robot, "hardware_capability_profile", None)
        raw_profile = provider() if callable(provider) else None
        return build_device_capability_snapshot(
            device_id=str(getattr(self.robot, "device_id", "") or "unknown-device"),
            supported_actions=self.supported_action_kinds(),
            raw_profile=raw_profile,
        )

    def capability_gap( self, requested_action: str, *, required_parameters: tuple[str, ...] = (), ) -> Any:
        return self.capability_snapshot().gap(requested_action, required_parameters=required_parameters)

    def __init__(
        self,
        *,
        capture: Callable[[], Image.Image],
        observer: SingleStepGenericSceneObserver,
        robot: Any,
        device_executor: DeviceExecutor | None = None,
        controller: UniversalActionController | None = None,
        frame_interval: float = 0.37,
        post_action_settle: float = 1.5,
        post_action_timeout: float | None = None,
        post_action_continuous_timeout: float | None = None,
        post_action_max_observations: int = 2,
        post_action_min_relative_sharpness: float = 0.80,
        post_action_min_reference_sharpness: float = 2.0,
        post_action_phone_view_delta_max: float = 45.0,
        confirmation_frame_delta_max: float = 6.0,
        qwerty_row_snapper: Callable[
            [tuple[Image.Image, ...] | list[Image.Image], dict[str, Any]],
            dict[str, Any] | None,
        ]
        | None = None,
        text_point_grounder: Callable[
            [
                tuple[Image.Image, ...] | list[Image.Image],
                UIScene,
                SemanticAction,
            ],
            LocalPointGrounding | None,
        ]
        | None = None,
        require_local_qwerty_row_snap: bool = False,
        device_id: str,
        input_lineage_store: TypedInputLineageStorePort | None = None,
    ) -> None:
        self.capture = capture
        self.observer = observer
        self.robot = robot
        self.device_executor = device_executor or RobotDeviceExecutor(robot)
        self.controller = controller or UniversalActionController()
        self.frame_interval = max(0.0, float(frame_interval))
        self.post_action_settle = max(0.0, float(post_action_settle))
        self.post_action_timeout = max(0.0, 10.0 if post_action_timeout is None else float(post_action_timeout))
        self.post_action_continuous_timeout = max(
            self.post_action_timeout,
            (
                45.0
                if post_action_continuous_timeout is None
                and post_action_timeout is None
                else self.post_action_timeout
                if post_action_continuous_timeout is None
                else float(post_action_continuous_timeout)
            ),
        )
        self.post_action_max_observations = min(2, max(1, int(post_action_max_observations)))
        self.post_action_min_relative_sharpness = max(0.0, min(1.0, float(post_action_min_relative_sharpness)))
        self.post_action_min_reference_sharpness = max(0.0, float(post_action_min_reference_sharpness))
        self.post_action_phone_view_delta_max = max(0.0, float(post_action_phone_view_delta_max))
        self.confirmation_frame_delta_max = max(0.0, float(confirmation_frame_delta_max))
        self.qwerty_row_snapper = qwerty_row_snapper
        self.text_point_grounder = text_point_grounder
        self.require_local_qwerty_row_snap = bool(require_local_qwerty_row_snap)
        self.input_lineage_store = input_lineage_store
        try:
            self.device_id = validate_device_id(device_id)
        except OrientationSafetyError as exc:
            raise ValueError(str(exc)) from exc

    @staticmethod
    def _confirmation_frame_delta(
        planned_frames: tuple[Image.Image, ...] | list[Image.Image],
        fresh_frames: list[Image.Image],
    ) -> float:
        if not planned_frames or not fresh_frames:
            raise GenericActionAdapterError("确认前缺少本地真实帧，不能验证画面身份。")
        planned_sizes = {frame.size for frame in planned_frames}
        fresh_sizes = {frame.size for frame in fresh_frames}
        if len(planned_sizes) != 1 or len(fresh_sizes) != 1 or planned_sizes != fresh_sizes:
            raise GenericActionAdapterError("确认前真实画面尺寸发生变化。")

        def compact(frame: Image.Image) -> Image.Image:
            return frame.convert("L").resize((96, 160), Image.Resampling.BILINEAR)

        planned = [compact(frame) for frame in planned_frames]
        fresh = [compact(frame) for frame in fresh_frames]
        return min(
            float(ImageStat.Stat(ImageChops.difference(first, second)).mean[0])
            for first in planned
            for second in fresh
        )

    def _capture_frame(self) -> Image.Image:
        frame = self.capture().convert("RGB")
        if frame.width < 400 or frame.height < 700:
            raise GenericActionAdapterError("摄像头返回残缺画面，停止单步动作。")
        return frame

    def _capture_confirmation_frames(
        self,
        *,
        evidence_dir: Path | None,
        prefix: str,
    ) -> tuple[list[Image.Image], tuple[str, ...]]:
        """Capture four fresh local frames without re-interpreting the scene."""

        frames: list[Image.Image] = []
        for index in range(4):
            frames.append(self._capture_frame())
            if index < 3 and self.frame_interval:
                time.sleep(self.frame_interval)
        stability = measure_local_stability(frames)
        paths = self._save_frames(frames, evidence_dir, prefix)
        if not stability.stable:
            raise GenericActionAdapterError(f'确认前本地多帧稳定性检查未通过：{stability.reason}', evidence=paths)
        return frames, paths

    def _capture_scene_once(
        self,
        goal: GenericIntentDraft,
        *,
        evidence_dir: Path | None = None,
        prefix: str,
    ) -> tuple[UIScene, list[Image.Image], tuple[str, ...]]:
        frames: list[Image.Image] = []
        for index in range(4):
            frames.append(self._capture_frame())
            if index < 3 and self.frame_interval:
                time.sleep(self.frame_interval)
        paths = self._save_frames(frames, evidence_dir, prefix)
        try:
            scene = self._observe_scene(frames, goal.to_dict())
        except RuntimeError as exc:
            diagnostic_paths = persist_observer_failure_diagnostic(
                self.observer,
                evidence_dir=evidence_dir,
                prefix=prefix,
                error=exc,
            )
            raise GenericActionAdapterError(f'通用页面观察失败：{exc}', evidence=paths + diagnostic_paths) from exc
        return scene, frames, paths

    def _observe_scene(
        self,
        frames: list[Image.Image] | tuple[Image.Image, ...],
        goal_context: dict[str, Any],
        *,
        input_lineage_override: TypedInputLineage | None = None,
        post_action_context: PostActionVisualContext | None = None,
    ) -> UIScene:
        kwargs: dict[str, Any] = {'frames': list(frames), 'goal_context': goal_context}
        if getattr(self.observer, "input_lineage_store", None) is not None:
            kwargs["device_id"] = self.device_id
            kwargs["input_lineage_override"] = input_lineage_override
        if (
            post_action_context is not None
            and getattr(
                self.observer,
                "supports_post_action_visual_context",
                False,
            )
            is True
        ):
            kwargs["post_action_context"] = post_action_context
        return self.observer.observe(**kwargs)

    def capture_scene(
        self,
        goal: GenericIntentDraft,
        *,
        evidence_dir: Path | None,
        prefix: str,
    ) -> tuple[UIScene, list[Image.Image], tuple[str, ...]]:
        # One fresh closed-loop step may make at most one Qwen request.  Local
        # frame collection can wait for stability, but an invalid observation
        # is returned immediately instead of resampling the model.
        try:
            return self._capture_scene_once(goal, evidence_dir=evidence_dir, prefix=f'{prefix}_attempt_1')
        except GenericActionAdapterError as exc:
            error = f"第1轮动作前观察失败：{exc}"
            raise GenericActionAdapterError(
                "动作前通用页面观察失败：" + error,
                evidence=tuple(exc.evidence),
                observation_errors=(error,),
            ) from exc

    @staticmethod
    def _requires_post_action_relative_clarity( resolved: ResolvedSemanticAction, ) -> bool:
        """Return whether the action preserves a comparable input surface.

        Text mutations keep the keyboard/page structure in place, so their
        post-action frame sharpness can be compared with the fresh pre-action
        frames.  Focusing an input is different: opening the soft keyboard
        replaces a large part of the frame and changes the score even when the
        new view is plainly readable.  Focus still passes the ordinary stable
        frame and trusted-observation clarity gates; it must not use this
        content-sensitive *relative* comparison.
        """

        return resolved.kind in {'input_verified_text', 'press_enter', 'clear_verified_text'}

    @staticmethod
    def _requires_post_action_phone_view_identity( resolved: ResolvedSemanticAction, ) -> bool:
        return resolved.kind in {'clear_verified_text', 'double_tap', 'drag', 'input_verified_text', 'long_press'}

    def _post_action_timeout_for( self, resolved: ResolvedSemanticAction, ) -> float:
        if self._requires_post_action_phone_view_identity(resolved):
            return self.post_action_continuous_timeout
        return self.post_action_timeout

    def _capture_stable_post_action_frames(
        self,
        *,
        deadline: float,
        evidence_dir: Path | None,
        prefix: str,
        clarity_reference_frames: tuple[Image.Image, ...] = (),
        require_relative_clarity: bool = False,
        require_phone_view_identity: bool = False,
    ) -> tuple[list[Image.Image], tuple[str, ...]]:
        """Wait locally for four stable and, when required, relatively clear frames."""

        frames: list[Image.Image] = []
        last_stability = None
        reference_sharpness = (
            statistics.median(measure_frame_sharpness(frame) for frame in clarity_reference_frames)
            if (require_relative_clarity or require_phone_view_identity) and clarity_reference_frames
            else None
        )
        clarity_is_comparable = bool(
            require_relative_clarity
            and reference_sharpness is not None
            and reference_sharpness >= self.post_action_min_reference_sharpness
        )
        phone_view_is_comparable = bool(
            require_phone_view_identity
            and reference_sharpness is not None
            and reference_sharpness >= self.post_action_min_reference_sharpness
        )
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
                if (
                    last_stability is not None
                    and last_stability.stable
                    and last_phone_view_delta is not None
                    and last_phone_view_delta > self.post_action_phone_view_delta_max
                ):
                    raise GenericActionAdapterError(
                        "动作后画面已稳定但相机尚未回到手机取景："
                        f"取景差异{last_phone_view_delta:.1f}，要求最多{self.post_action_phone_view_delta_max:.1f}",
                        evidence=paths,
                    )
                if last_stability is not None and last_stability.stable and last_relative_sharpness is not None:
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

    def _observe_stable_post_action_scene(
        self,
        goal: GenericIntentDraft,
        *,
        before: UIScene,
        before_frames: tuple[Image.Image, ...],
        resolved: ResolvedSemanticAction,
        input_lineage_override: TypedInputLineage | None,
        evidence_dir: Path | None,
        evidence_prefix: str,
    ) -> tuple[
        UIScene,
        tuple[Image.Image, ...],
        tuple[str, ...],
        tuple[str, ...],
        tuple[str, ...],
        tuple[str, ...],
    ]:
        action_timeout = self._post_action_timeout_for(resolved)
        if self.post_action_settle:
            time.sleep(min(self.post_action_settle, action_timeout))

        observation_context = _post_action_observation_context(goal, resolved, physical_action_executed=True)
        post_action_visual_context = _post_action_visual_context(resolved)
        prefix = f"{evidence_prefix}_after_attempt_1"
        try:
            frames, paths = self._capture_stable_post_action_frames(
                deadline=time.monotonic() + action_timeout,
                evidence_dir=evidence_dir,
                prefix=prefix,
                clarity_reference_frames=before_frames,
                require_relative_clarity=self._requires_post_action_relative_clarity(resolved),
                require_phone_view_identity=self._requires_post_action_phone_view_identity(resolved),
            )
        except GenericActionAdapterError as exc:
            raise GenericActionAdapterError(f'动作后画面采集失败：{exc}', evidence=tuple(exc.evidence)) from exc
        all_paths = paths
        try:
            after = self._observe_scene(
                frames,
                observation_context,
                input_lineage_override=input_lineage_override,
                post_action_context=post_action_visual_context,
            )
        except RuntimeError as exc:
            all_paths += persist_observer_failure_diagnostic(
                self.observer, evidence_dir=evidence_dir, prefix=prefix, error=exc
            )
            observation_errors = (f"第1轮动作后观察失败：{exc}",)
            raise GenericActionAdapterError(
                "通用页面观察失败：" + observation_errors[0],
                evidence=all_paths,
                observation_errors=observation_errors,
            ) from exc

        after = self._reconcile_literal_key_visual_wrap(resolved, before, after)
        after = self._reconcile_verified_text_horizontal_suffix(resolved, before, after)
        try:
            self.controller.verify_after_action(resolved, before, after)
            verification_errors: tuple[str, ...] = ()
        except UniversalActionError as exc:
            verification_errors = (f"第1轮动作结果不匹配：{exc}",)
        return (after, tuple(frames), paths, all_paths, (), verification_errors)

    @staticmethod
    def _reconcile_literal_key_visual_wrap(
        resolved: ResolvedSemanticAction,
        before: UIScene,
        after: UIScene,
    ) -> UIScene:
        """Remove presentation-only line wraps under one exact key receipt."""

        if resolved.kind != "tap_semantic" or not resolved.target_element_id:
            return after
        try:
            key = before.get_element(resolved.target_element_id)
        except UISceneError:
            return after
        if key.meaning != "input_exact_literal_key":
            return after
        states = key.states
        prior = states.get("prior_input_value")
        key_value = states.get("key_value")
        expected = states.get("expected_input_value")
        input_id = str(states.get("input_element_id") or "").strip()
        expected_state = resolved.expected_effect.get("element_state")
        expected_states = expected_state.get("states") if isinstance(expected_state, dict) else None
        if (
            not isinstance(prior, str)
            or not isinstance(key_value, str)
            or len(key_value) != 1
            or key_value in {"\r", "\n"}
            or "\r" in prior
            or "\n" in prior
            or expected != prior + key_value
            or not isinstance(expected, str)
            or "\r" in expected
            or "\n" in expected
            or expected_states != {"value": expected}
            or not input_id
        ):
            return after
        candidates = tuple(
            element
            for element in after.elements
            if element.element_id == input_id
            and element.role == "input"
            and element.meaning == "application_text_input"
            and float(element.confidence) >= 0.9
            and element.states.get("fully_visible") is True
            and element.states.get("focused") is True
        )
        if len(candidates) != 1:
            return after
        candidate = candidates[0]
        observed = candidate.states.get("value")
        if (
            not isinstance(observed, str)
            or not ({"\r", "\n"} & set(observed))
            or observed.count("\r") + observed.count("\n") > 3
            or observed.replace("\r", "").replace("\n", "") != expected
            or not any(observed in str(item) for item in candidate.evidence)
        ):
            return after
        replacement = replace(
            candidate,
            label=expected if candidate.label == observed else candidate.label,
            states={**candidate.states, "value": expected},
            evidence=candidate.evidence + ("本地逐键回执确认该换行为控件视觉软折行",),
        )
        reconciled = replace(
            after,
            elements=tuple(replacement if element.element_id == input_id else element for element in after.elements),
        )
        reconciled.validate()
        return reconciled

    @staticmethod
    def _reconcile_verified_text_horizontal_suffix(
        resolved: ResolvedSemanticAction,
        before: UIScene,
        after: UIScene,
    ) -> UIScene:
        """Recover a clipped single-line suffix only from an exact typed transaction."""

        if (
            resolved.kind != "input_verified_text"
            or resolved.input_method != "direct_latin"
            or not resolved.formal_candidate_id
            or not resolved.target_element_id
        ):
            return after
        prior = resolved.prior_input_value
        fragment = resolved.input_fragment
        expected = resolved.expected_input_value
        expected_state = resolved.expected_effect.get("element_state")
        expected_states = expected_state.get("states") if isinstance(expected_state, dict) else None
        if (
            not isinstance(prior, str)
            or not prior
            or not isinstance(fragment, str)
            or not fragment
            or not isinstance(expected, str)
            or expected != prior + fragment
            or any(marker in expected for marker in ("\r", "\n"))
            or expected_states != {"value": expected}
        ):
            return after
        try:
            before_input = before.get_element(resolved.target_element_id)
        except UISceneError:
            return after
        before_states = before_input.states
        typed_field_id = str(before_states.get("input_field_id") or "").strip()
        if (
            before_input.role != "input"
            or before_input.meaning != "application_text_input"
            or before_states.get("focused") is not True
            or before_states.get("input_multiline") is not False
            or before_states.get("value") != prior
            or before_states.get("keyboard_layout") != "qwerty"
            or before_states.get("keyboard_input_mode") != "direct_latin"
            or not typed_field_id
            or typed_field_id == "unknown"
        ):
            return after
        candidates = tuple(
            element
            for element in after.elements
            if element.element_id == resolved.target_element_id
            and element.role == "input"
            and element.meaning == "application_text_input"
            and float(element.confidence) >= 0.9
            and element.states.get("fully_visible") is True
            and element.states.get("focused") is True
            and element.states.get("input_multiline") is False
            and str(element.states.get("input_field_id") or "").strip() == typed_field_id
            and element.states.get("keyboard_layout") == "qwerty"
            and element.states.get("keyboard_input_mode") == "direct_latin"
        )
        if len(candidates) != 1:
            return after
        candidate = candidates[0]
        observed = candidate.states.get("value")
        visible_prior_suffix = observed[: -len(fragment)] if isinstance(observed, str) and len(observed) > len(fragment) else ""
        required_overlap = min(4, len(prior))
        if (
            not isinstance(observed, str)
            or not observed
            or observed == expected
            or not expected.endswith(observed)
            or not observed.endswith(fragment)
            or len(visible_prior_suffix) < required_overlap
            or not prior.endswith(visible_prior_suffix)
            or not any(observed in str(item) for item in candidate.evidence)
            or not input_app_identity_compatible(before.foreground_app_id, after.foreground_app_id)
            or not input_screen_identity_compatible(before.screen_id, after.screen_id)
        ):
            return after
        replacement = replace(
            candidate,
            states={
                **candidate.states,
                "visible_value_suffix": observed,
                "value_visibility": "horizontal_suffix",
                "value": expected,
            },
            evidence=candidate.evidence + (
                "本地精确分段交易确认完整值：" + expected,
                "画面仅显示横向滚动尾段：" + observed,
            ),
        )
        reconciled = replace(
            after,
            elements=tuple(
                replacement if element.element_id == resolved.target_element_id else element for element in after.elements
            ),
        )
        reconciled.validate()
        return reconciled

    def _prepare_keyboard_geometry(
        self,
        resolved: ResolvedSemanticAction,
        scene: UIScene,
        frames: tuple[Image.Image, ...] | list[Image.Image],
    ) -> dict[str, Any] | None:
        if resolved.kind not in {"input_verified_text", "clear_verified_text"}:
            return None
        if resolved.kind == "input_verified_text" and not resolved.text:
            raise GenericActionAdapterError("输入动作缺少已校验文字。")
        try:
            input_element = scene.get_element(
                str(resolved.target_element_id or ""), min_confidence=self.controller.min_confidence
            )
        except UISceneError as exc:
            raise GenericActionAdapterError(f"当前文字输入缺少可信输入框：{exc}") from exc
        geometry = input_element.states.get("keyboard_geometry")
        allowed_types = {"input_verified_text": {"qwerty"}, "clear_verified_text": {"qwerty", "generic"}}
        if (
            not isinstance(geometry, dict)
            or geometry.get("source") != "input_structure_audit"
            or geometry.get("type") not in allowed_types[resolved.kind]
        ):
            raise GenericActionAdapterError("当前文字动作缺少本轮输入结构审计签发的键盘几何；拒绝使用静态配置。")
        prepared = dict(geometry)
        if geometry.get("type") == "qwerty" and self.require_local_qwerty_row_snap:
            if not callable(self.qwerty_row_snapper):
                raise GenericActionAdapterError("真机文字输入缺少本地 QWERTY 行中心复核器。")
            snapped = self.qwerty_row_snapper(frames, geometry.get("anchors"))
            if not isinstance(snapped, dict):
                raise GenericActionAdapterError("本地 OCR 未能稳定确认 QWERTY 三行中心，拒绝按模型粗坐标输入。")
            prepared.update(anchors=snapped, row_snap_source="stable_local_ocr")
        if prepared.get("type") == "qwerty":
            try:
                qwerty_keyboard_config_from_anchors(prepared.get("anchors"))
            except WorkflowNotReady as exc:
                raise GenericActionAdapterError(f"当前 QWERTY 几何未通过动作前本地复核：{exc}") from exc
        else:
            backspace = (prepared.get("anchors") or {}).get("backspace")
            if (
                not isinstance(backspace, list)
                or len(backspace) != 2
                or any(
                    isinstance(part, bool) or not isinstance(part, (int, float)) or not 0 <= float(part) <= 1000
                    for part in backspace
                )
            ):
                raise GenericActionAdapterError("非 QWERTY 清空缺少本轮完整可见退格键中心。")
        validator = getattr(self.robot, "validate_verified_text", None)
        if resolved.kind == "input_verified_text" and callable(validator):
            try:
                validator(
                    resolved.input_fragment,
                    dict(input_element.states),
                    target_text=resolved.text,
                    input_method=resolved.input_method,
                    pinyin=resolved.input_pinyin,
                )
            except (UISceneError, ValueError, RuntimeError) as exc:
                raise GenericActionAdapterError(f"当前文字输入不满足设备已验证配置：{exc}") from exc
        if resolved.kind == "clear_verified_text" and resolved.delete_count is None:
            raise GenericActionAdapterError("清空动作缺少已验证退格次数。")
        return prepared

    def _arm_physical_execution(
        self,
        requested: SemanticAction,
        resolved: ResolvedSemanticAction,
        scene: UIScene,
        frames: tuple[Image.Image, ...] | list[Image.Image],
        paths: tuple[str, ...],
    ) -> tuple[OrientationCredential | None, Callable[[], Any] | None]:
        clear = getattr(self.robot, "clear_physical_execution_authorization", None)
        if resolved.kind not in self.PHYSICAL_KINDS:
            return None, clear if callable(clear) else None
        arm = getattr(self.robot, "arm_physical_execution", None)
        if not callable(arm) or not callable(clear):
            raise GenericActionAdapterError("机械臂控制器未提供共享物理执行门禁，拒绝动作。", evidence=paths)
        clear()
        try:
            credential = self._local_qwerty_orientation_credential(requested=requested, scene=scene, frames=frames)
            diagnostic_flag = "local_qwerty_rows_verified"
            if credential is None:
                credential = self._single_step_scene_orientation_credential(scene=scene, frames=frames)
                diagnostic_flag = "single_step_scene_reused"
            selected = len(frames) - 1
            try:
                self.observer.last_orientation_audit_diagnostics = {
                    "audit_source": credential.source,
                    "model_calls": 0,
                    diagnostic_flag: True,
                    "selected_frame_index": selected,
                    "scene_fingerprint": scene.fingerprint,
                }
            except Exception:
                pass
            if 0 <= selected < len(paths):
                with Image.open(paths[selected]) as persisted:
                    credential = replace(
                        credential, evidence_frame_fingerprint=frame_fingerprint(persisted.convert("RGB"))
                    )
            hardware_action = {
                "clear_verified_text": "input_verified_text",
                "press_enter": "tap_semantic",
            }.get(resolved.kind, resolved.kind)
            credential.assert_authorizes(
                device_id=self.device_id,
                scene_fingerprint=scene.fingerprint,
                frame_size=credential.frame_size,
                action=hardware_action,
            )
            arm(credential, action=hardware_action, scene_fingerprint=scene.fingerprint)
            return credential, clear
        except (OrientationSafetyError, RuntimeError, ValueError) as exc:
            clear()
            raise GenericActionAdapterError(f'动作前单步画面方向凭据校验失败：{exc}', evidence=paths) from exc

    @staticmethod
    def _pending_input_lineage(
        resolved: ResolvedSemanticAction,
        device_id: str,
        before: UIScene,
        hardware_receipt: dict[str, Any] | None,
    ) -> TypedInputLineage | None:
        values = {"device_id": device_id, "resolved_action": resolved.to_dict(), "before_scene": before.to_dict()}
        if hardware_receipt is not None:
            return _first_valid_lineage(
                (
                    build_pending_newline_lineage,
                    build_pending_literal_lineage,
                    build_pending_ime_candidate_lineage,
                    build_pending_input_state_lineage,
                ),
                **values,
                hardware_receipt=hardware_receipt,
            )
        if resolved.kind == "input_verified_text":
            return _first_valid_lineage((build_pending_text_lineage, build_pending_chinese_preedit_lineage), **values)
        return None

    def _record_verified_input_lineage(
        self,
        resolved: ResolvedSemanticAction,
        before: UIScene,
        after: UIScene,
        after_frames: tuple[Image.Image, ...],
        hardware_receipt: dict[str, Any] | None,
    ) -> None:
        store = self.input_lineage_store
        if store is None:
            return
        values = {
            "device_id": self.device_id,
            "resolved_action": resolved.to_dict(),
            "before_scene": before.to_dict(),
            "after_scene": after.to_dict(),
            "after_frames": after_frames,
        }
        try:
            if resolved.kind == "input_verified_text":
                store.record_verified_text_action(**values)
            elif resolved.kind == "clear_verified_text":
                store.discard(self.device_id)
            elif resolved.kind == "press_enter" and hardware_receipt is not None:
                store.record_verified_newline_action(**values, hardware_receipt=hardware_receipt)
            elif hardware_receipt is not None:
                store.record_verified_literal_action(**values, hardware_receipt=hardware_receipt)
        except (InputValueLineageError, OSError, TypeError, ValueError):
            pass

    def execute(
        self,
        *,
        requested_action: SemanticAction,
        planned_scene: UIScene,
        goal: GenericIntentDraft,
        confirmed: bool,
        evidence_dir: Path | None = None,
        planned_frames: tuple[Image.Image, ...] | list[Image.Image] = (),
    ) -> GenericActionExecutionResult:
        if confirmed is not True:
            raise GenericActionAdapterError("必须明确确认当前这一个语义动作。")
        safe_node = re.sub(r"[^a-zA-Z0-9_-]+", "_", requested_action.node_id)[:48]
        evidence_prefix = f"{safe_node or 'action'}_{uuid.uuid4().hex}"
        local_frame_identity_verified = False
        primary_input_confirmation_reused = False
        confirmation_frame_delta: float | None = None
        if planned_frames:
            before_frames, before_paths = self._capture_confirmation_frames(
                evidence_dir=evidence_dir,
                prefix=f"{evidence_prefix}_before",
            )
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
            primary_input_confirmation_reused = (
                self._primary_input_confirmation_reusable(
                    requested_action,
                    planned_scene,
                )
            )
        else:
            before, before_frames, before_paths = self.capture_scene(
                goal,
                evidence_dir=evidence_dir,
                prefix=f"{evidence_prefix}_before",
            )
        try:
            rebind_planned_scene = planned_scene
            # The fresh four-frame window has already proved the pixels are the
            # same as the fingerprint-bound planning scene.  Re-reading those
            # identical pixels with Qwen cannot add independent authority and
            # formerly spent up to three extra calls (scene + two crops).
            # Rebind the immutable canonical candidate against that same scene;
            # bounds, uniqueness and controller reachability remain local gates.
            rebound = self._rebind_action(
                requested_action,
                rebind_planned_scene,
                before,
                local_frame_identity_verified=local_frame_identity_verified,
                # A verified text-input action never executes at the input
                # element's model-drawn center.  Once the original and fresh
                # four-frame sets prove the pixels are unchanged, bind the
                # unique focused input by its exact semantic/state identity
                # and execute only from the fresh input-structure keyboard
                # geometry below.  Tap, clear, long-press and drag retain the
                # strict planned/fresh geometry-overlap gate.
                require_geometry_overlap=not (
                    local_frame_identity_verified
                    and requested_action.action == "input_verified_text"
                ),
            )
        except GenericActionAdapterError as exc:
            raise GenericActionAdapterError(
                str(exc),
                evidence=before_paths + tuple(getattr(exc, "evidence", ())),
            ) from exc
        except RuntimeError as exc:
            raise GenericActionAdapterError(f'确认前独立目标几何审计失败：{exc}', evidence=before_paths) from exc
        local_point_grounding: LocalPointGrounding | None = None
        if callable(self.text_point_grounder):
            try:
                local_point_grounding = self.text_point_grounder(before_frames, before, rebound)
            except Exception:
                # OCR is a precision aid, not another product boundary.  If it
                # is unavailable or inconclusive, retain the canonical target's
                # existing point instead of blocking an otherwise legal action.
                local_point_grounding = None
        try:
            resolved = self.controller.resolve_one(
                rebound,
                before,
                confirmed=True,
                local_point_grounding=local_point_grounding,
            )
        except UniversalActionError as exc:
            raise GenericActionAdapterError(f'确认前控制器拒绝动作：{exc}', evidence=before_paths) from exc

        if resolved.kind not in self.PHYSICAL_KINDS and resolved.kind != "wait_for_change":
            raise GenericActionAdapterError(f'当前通用硬件适配器尚未开放：{resolved.kind}', evidence=before_paths)

        prepared_keyboard_geometry = self._prepare_keyboard_geometry(resolved, before, before_frames)
        orientation_credential, clear_authorization = self._arm_physical_execution(
            requested_action, resolved, before, before_frames, before_paths
        )

        physical_actions = 0
        robot_result: Any = None
        hardware_receipt: dict[str, Any] | None = None

        def executor_point( point: tuple[float, float] | None, ) -> tuple[int, int] | None:
            if point is None:
                return None
            return (max(0, min(1000, round(point[0] * 1000))), max(0, min(1000, round(point[1] * 1000))))

        execution_request = DeviceActionRequest(
            kind=resolved.kind,
            point=executor_point(resolved.normalized_point),
            end_point=executor_point(resolved.normalized_end_point),
            direction=resolved.direction,
            hold_seconds=resolved.hold_seconds,
            input_fragment=resolved.input_fragment,
            input_method=resolved.input_method,
            input_pinyin=resolved.input_pinyin,
            keyboard_geometry=prepared_keyboard_geometry,
            delete_count=resolved.delete_count,
            wait_seconds=(
                max(0.5, self.post_action_settle)
                if resolved.kind == "wait_for_change"
                else None
            ),
        )
        try:
            execution_result = self.device_executor.execute(execution_request)
            physical_actions = execution_result.physical_actions
            robot_result = execution_result.transport_result
            hardware_receipt = execution_result.hardware_receipt
        except DeviceExecutionError as exc:
            raise GenericActionAdapterError(
                f"设备执行器拒绝动作：{exc}",
                physical_actions=exc.physical_actions,
                evidence=before_paths,
            ) from exc
        except OrientationSafetyError as exc:
            gate_evidence = before_paths
            if isinstance(exc, OrientationFrameMismatchError):
                gate_evidence += self._save_frames(
                    [exc.actual_frame],
                    evidence_dir,
                    f"{evidence_prefix}_physical_gate_actual",
                )
            raise GenericActionAdapterError(
                f"共享物理执行门在控制端原语前拒绝动作：{exc}",
                physical_actions=0,
                evidence=gate_evidence,
            ) from exc
        except Exception as exc:
            raise GenericActionAdapterError(
                f"机械臂单步动作调用失败：{exc}",
                physical_actions=physical_actions,
                evidence=before_paths,
            ) from exc
        finally:
            if callable(clear_authorization):
                clear_authorization()

        pending_input_lineage = self._pending_input_lineage(resolved, self.device_id, before, hardware_receipt)

        try:
            (
                after,
                after_frames,
                after_frame_paths,
                all_after_paths,
                observation_errors,
                verification_errors,
            ) = self._observe_stable_post_action_scene(
                goal,
                before=before,
                before_frames=tuple(before_frames),
                resolved=resolved,
                input_lineage_override=pending_input_lineage,
                evidence_dir=evidence_dir,
                evidence_prefix=evidence_prefix,
            )
        except (GenericActionAdapterError, UniversalActionError) as exc:
            evidence = before_paths + tuple(getattr(exc, "evidence", ()))
            raise GenericActionAdapterError(
                f"单步动作后验证失败：{exc}",
                physical_actions=physical_actions,
                evidence=evidence,
                observation_errors=tuple(getattr(exc, "observation_errors", ())),
                verification_errors=tuple(getattr(exc, "verification_errors", ())),
            ) from exc

        controller_transition_evidence: tuple[str, ...] = ()
        if not verification_errors:
            try:
                controller_transition_evidence = self.controller.transition_evidence_from_verified_action(
                    resolved, before, after
                )
            except UniversalActionError as exc:
                verification_errors = verification_errors + (f'控制器转换证据复核失败：{exc}',)

        if not verification_errors:
            self._record_verified_input_lineage(resolved, before, after, after_frames, hardware_receipt)

        return GenericActionExecutionResult(
            requested_action=requested_action,
            rebound_action=rebound,
            resolved_action=resolved,
            before_scene=before,
            after_scene=after,
            planned_scene_fingerprint=planned_scene.fingerprint,
            confirmation_frame_identity_verified=local_frame_identity_verified,
            confirmation_frame_delta=confirmation_frame_delta,
            physical_actions=physical_actions,
            primary_input_confirmation_reused=primary_input_confirmation_reused,
            action_outcome="mismatched" if verification_errors else "matched",
            verification_errors=verification_errors,
            robot_result=robot_result,
            hardware_receipt=hardware_receipt,
            evidence=before_paths + all_after_paths,
            after_frames=after_frames,
            after_frame_paths=after_frame_paths,
            observation_errors=observation_errors,
            controller_transition_evidence=controller_transition_evidence,
            before_frames=before_frames,
            before_frame_paths=before_paths,
            orientation_credential=orientation_credential,
        )

    def _rebind_action(
        self,
        requested: SemanticAction,
        planned_scene: UIScene,
        fresh_scene: UIScene,
        *,
        local_frame_identity_verified: bool = False,
        require_geometry_overlap: bool = True,
    ) -> SemanticAction:
        planned_app = planned_scene.foreground_app_id
        fresh_app = fresh_scene.foreground_app_id
        if (
            not local_frame_identity_verified
            and planned_app != "unknown"
            and fresh_app != "unknown"
            and planned_app != fresh_app
        ):
            raise GenericActionAdapterError(f"确认时前台 App 已变化：{planned_app} -> {fresh_app}")
        if (
            not local_frame_identity_verified
            and planned_scene.screen_id != "unknown"
            and planned_scene.screen_id != fresh_scene.screen_id
        ):
            raise GenericActionAdapterError(f"确认时页面已变化：{planned_scene.screen_id} -> {fresh_scene.screen_id}")
        single_element_actions = {
            "tap_semantic",
            "dismiss_overlay",
            "input_verified_text",
            "press_enter",
            "clear_verified_text",
            "double_tap",
            "long_press",
        }
        if requested.action not in single_element_actions | {"drag"}:
            return requested

        def stable_rebind_states(states: dict[str, Any]) -> dict[str, Any]:
            return {
                key: value
                for key, value in states.items()
                if key not in {"goal_relevant", "keyboard_geometry"}
                and not (
                    key == "keyboard_case_mode"
                    and str(value or "").strip().casefold() == "unknown"
                )
            }

        def rebind_element(prefix: str = "") -> UIElement:
            original_id = str(requested.params.get(f"{prefix}element_id") or "").strip()
            try:
                original = planned_scene.get_element(original_id)
            except UISceneError as exc:
                raise GenericActionAdapterError(f"原始场景目标无效：{exc}") from exc
            matches = fresh_scene.find_elements(
                label=original.label or None,
                role=original.role,
                states=stable_rebind_states(dict(original.states)),
            )
            selector_roles = {"button", "tab", "list_item", "text"}
            if not matches and requested.action in {"tap_semantic", "dismiss_overlay", "double_tap"} and prefix == "":
                stable_states = stable_rebind_states(dict(original.states))
                if original.label and original.role in selector_roles:
                    aliases = tuple(
                        element
                        for element in fresh_scene.find_elements(label=original.label, states=stable_states)
                        if element.role in selector_roles
                    )
                elif not original.label and original.role in {"icon", "button"}:
                    aliases = tuple(
                        element
                        for element in fresh_scene.find_elements(meaning=original.meaning, states=stable_states)
                        if element.role in {"icon", "button"} and not element.label
                    )
                else:
                    aliases = ()
                if len(aliases) == 1:
                    matches = aliases
            if len(matches) != 1:
                raise GenericActionAdapterError(f'确认时目标语义不再严格唯一：{original.meaning}，匹配{len(matches)}个')
            current = matches[0]
            original_stable_states = stable_rebind_states(dict(original.states))
            current_stable_states = stable_rebind_states(dict(current.states))
            if str(original.states.get("keyboard_case_mode") or "").strip().casefold() == "unknown":
                current_stable_states.pop("keyboard_case_mode", None)
            states_match = current_stable_states == original_stable_states
            if (
                not states_match
                and "fully_visible" not in original_stable_states
                and current_stable_states.get("fully_visible") is True
            ):
                states_match = {
                    key: value
                    for key, value in current_stable_states.items()
                    if key != "fully_visible"
                } == original_stable_states
            if current.label != original.label or not states_match:
                raise GenericActionAdapterError('确认时目标标签或状态已经变化，旧确认失效。')
            overlap = bounds_overlap(original.bounds, current.bounds)
            widths = (original.bounds[2] - original.bounds[0], current.bounds[2] - current.bounds[0])
            heights = (original.bounds[3] - original.bounds[1], current.bounds[3] - current.bounds[1])
            center_delta = tuple(abs(left - right) for left, right in zip(original.center, current.center))
            tight_loose_same_target = (
                overlap["intersection_over_smaller"] >= 0.50
                and center_delta[0] <= max(0.03, 0.25 * max(widths))
                and center_delta[1] <= max(0.02, 0.50 * max(heights))
            )
            if ( require_geometry_overlap and overlap["iou"] < 0.60 and not tight_loose_same_target ):
                raise GenericActionAdapterError(
                    "确认时目标区域已明显移动，旧确认失效："
                    f"iou={overlap['iou']:.3f}, smaller_coverage={overlap['intersection_over_smaller']:.3f}, "
                    f"center_delta=({center_delta[0]:.3f},{center_delta[1]:.3f})。"
                )
            return current

        prefixes = ("source_", "destination_") if requested.action == "drag" else ("",)
        params = dict(requested.params)
        for prefix in prefixes:
            current = rebind_element(prefix)
            params.update(
                {
                    f"{prefix}element_id": current.element_id,
                    f"{prefix}target": current.meaning,
                    f"{prefix}role": current.role,
                    f"{prefix}label": current.label,
                    f"{prefix}states": dict(current.states),
                }
            )
        return SemanticAction(node_id=requested.node_id, action=requested.action, params=params)

    @staticmethod
    def _save_frames( frames: list[Image.Image], evidence_dir: Path | None, prefix: str, ) -> tuple[str, ...]:
        if evidence_dir is None:
            return ()
        evidence_dir.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []
        for index, frame in enumerate(frames, start=1):
            path = evidence_dir / f"{prefix}_{index}.jpg"
            frame.save(path, format="JPEG", quality=92)
            paths.append(str(path))
        return tuple(paths)
